// Multi-stage reproduction on remote GPUs. Pure. A complex paper is not one
// command: data must be fetched and prepared, models trained or checkpoints
// downloaded, then evaluated. The planner proposes a stage DAG and a fidelity
// level (full / scaled / eval_only) that fits the paper's budget; this module
// validates it (structure, GPU allowlist, worst-case cost) and turns it into
// task nodes. Stage outputs pass between stages through the persistent volume.
import type { RunState, TaskNode } from "../core/dag.ts";
import { claimsOf, currentVersion, N, type Claim } from "./flow.ts";

export const GPUS = ["T4", "L4", "A10G", "A100-40GB", "A100-80GB", "H100"];
// GPU memory (GB). A stage that states its memory need gets the cheapest GPU that fits.
export const GPU_MEM: Record<string, number> = { T4: 16, L4: 24, A10G: 24, "A100-40GB": 40, "A100-80GB": 80, H100: 80 };
// Alternatives when a GPU type has no capacity: types with at least the memory needed (or the planned
// type's memory), cheapest first, at most two. Pure.
export function fallbackGpus(planned: string, memGB?: number, price: (g: string) => number = (x) => Number(process.env[`MLH_GPU_PRICE_${x.replace(/\W/g, "_")}`] ?? DEFAULT_GPU_PRICE[x] ?? 4)): string[] {
  const need = memGB ?? GPU_MEM[planned] ?? 16;
  return GPUS.filter((x) => x !== planned && (GPU_MEM[x] ?? 0) >= need).sort((a, b) => price(a) - price(b)).slice(0, 2);
}
const DEFAULT_GPU_PRICE: Record<string, number> = { T4: 0.59, L4: 0.8, A10G: 1.1, "A100-40GB": 2.1, "A100-80GB": 2.5, H100: 3.95 };
// Hosts that serve datasets and released weights. Fetching from anywhere else needs a human.
export const DATA_HOSTS = [
  "images.cocodataset.org", "dl.fbaipublicfiles.com", "download.pytorch.org", "huggingface.co", "cdn-lfs.huggingface.co",
  "github.com", "objects.githubusercontent.com", "raw.githubusercontent.com", "zenodo.org", "storage.googleapis.com",
  "s3.amazonaws.com", "www.cs.toronto.edu", "people.eecs.berkeley.edu", "data.dgl.ai", "snap.stanford.edu", "ogb.stanford.edu",
  "nlp.stanford.edu", "dl.dropboxusercontent.com", "www.dropbox.com", "drive.google.com", "drive.usercontent.google.com",
];
// What a stage with network may reach (Modal outbound domain allowlist; wildcards allowed): data and
// weight hosts and their CDNs. Package indexes and everything else stay blocked.
// Official dataset hosts of the papers in bench/complex (each download is still checksummed and
// recorded). Dropbox: FT-Transformer's README links its data there (redirects to dl.dropboxusercontent).
export const DATA_DOMAINS = [
  "www.dropbox.com", "dl.dropboxusercontent.com",
  "huggingface.co", "*.huggingface.co", "*.hf.co", "images.cocodataset.org", "dl.fbaipublicfiles.com", "download.pytorch.org",
  "github.com", "*.githubusercontent.com", "zenodo.org", "storage.googleapis.com", "*.s3.amazonaws.com", "s3.amazonaws.com",
  "www.cs.toronto.edu", "people.eecs.berkeley.edu", "data.dgl.ai", "snap.stanford.edu", "ogb.stanford.edu", "nlp.stanford.edu",
  "*.dropboxusercontent.com", "www.dropbox.com", "drive.google.com", "drive.usercontent.google.com",
];
// Base images the environment may choose (old repositories need old Python/PyTorch).
export const IMAGES: Record<string, string> = {
  "py3.11-torch2.5-cu124": "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
  "py3.10-torch2.1-cu121": "pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime",
  "py3.10-torch1.13-cu117": "pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime",
  // Labels are what `mlh doctor` measured inside each image (2026-11-09), not the tag's name.
  "py3.7-torch1.12-cu113": "pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime",
  "py3.7-torch1.10-cu111": "pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime",
  "py3.8-torch1.7-cu110": "pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime",
};
// Old (wrong) labels still resolve for runs recorded before the fix, but are not offered to agents.
for (const [old, now] of [["py3.8-torch1.12-cu113", "py3.7-torch1.12-cu113"], ["py3.8-torch1.9-cu111", "py3.7-torch1.10-cu111"], ["py3.7-torch1.7-cu110", "py3.8-torch1.7-cu110"]] as const) Object.defineProperty(IMAGES, old, { value: IMAGES[now], enumerable: false });
// The base image whose torch matches the repository's own pin (requirements.txt, setup.py,
// environment.yml). Pure. c1: SimSiam pins torch==1.7.0 and its env agent stayed on Python 3.11.
export function imageForPins(text: string): { key: string; torch: string } | undefined {
  const m = /\btorch\s*(?:==|=|>=|~=)\s*(\d+)\.(\d+)/.exec(text);
  if (!m) return undefined;
  const want = `${m[1]}.${m[2]}`, key = Object.keys(IMAGES).find((k) => new RegExp(`-torch${want.replace(".", "\\.")}-`).test(k));
  return key ? { key, torch: want } : undefined;
}

// The rules for one declared download, shared by the planner's plan and the env agent's fetch_data
// tool: https (or http with a sha256), into /cache/data/, a known data host. Pure.
export function checkFetchItems(raw: unknown[]): { items: FetchItem[]; errors: string[]; unknownHosts: string[] } {
  const items: FetchItem[] = [], errors: string[] = [];
  for (const f of raw as Partial<FetchItem>[]) {
    let u: URL;
    try { u = new URL(String(f?.url)); } catch { errors.push(`fetch url ${f?.url} is not a URL`); continue; }
    const sha = typeof f.sha256 === "string" && /^[0-9a-f]{64}$/i.test(f.sha256) ? f.sha256 : undefined;
    if (u.protocol !== "https:" && !(u.protocol === "http:" && sha)) { errors.push(`fetch url ${f.url} must be https, or http with its sha256`); continue; }
    const to = String(f.to ?? "");
    if (!to.startsWith("/cache/data/") || to.includes("..") || to.replace(/\/$/, "") === "/cache/data") { errors.push(`fetch target ${to} must be a path under /cache/data/`); continue; }
    items.push({ url: String(f.url), to, ...(sha ? { sha256: sha } : {}), ...(f.unpack === "zip" || f.unpack === "tar" ? { unpack: f.unpack } : {}) });
  }
  const unknownHosts = [...new Set(items.map((f) => new URL(f.url).hostname).filter((h) => !DATA_HOSTS.some((d) => h === d || h.endsWith(`.${d}`))))];
  return { items, errors, unknownHosts };
}

export const DEFAULT_IMAGE_KEY = "py3.11-torch2.5-cu124";
export interface FetchItem { url: string; to: string; sha256?: string; unpack?: "zip" | "tar" }
export const MAX_STAGE_S = 6 * 3600;

export interface Stage {
  id: string;
  argv: string[];
  gpu?: string;
  timeoutS: number;
  needs: string[];
  persist: string[]; // repo-relative paths kept for later stages
  network?: boolean; // only for fetching data or checkpoints
  perSeed?: boolean;
  gpuMemGB?: number; // stated need: the harness picks the cheapest GPU that fits
  memGB?: number; // host RAM for the container (default Modal size otherwise)
  cpus?: number; // CPU cores (data loading); default Modal share otherwise
  links?: { path: string; to: string }[]; // repo-relative symlinks to fetched data, made before the command
  env?: Record<string, string>;
  cwd?: string;
}
export interface GpuPlan {
  fidelity: "full" | "scaled" | "eval_only";
  fidelityNote: string;
  // Data and checkpoints, fetched by harness code (not repository code) into /cache/data, with provenance.
  fetch: FetchItem[];
  // Deterministic completeness checks after fetching, e.g. the evaluation split has all its images.
  checks: { path: string; minFiles: number }[];
  unknownHosts: string[];
  stages: Stage[];
  claims: { claim: string; stage: string; parse: string; scale?: number; select?: "last" | "max" | "min"; aggregate?: "mean" | "median" }[];
  // Claims the plan does not measure, each with its reason (e.g. ImageNet pre-training far beyond the
  // budget). Reported as not attempted: never stood in for by a different dataset or method.
  unmeasured: { claim: string; reason: string }[];
  seeds: number[];
  seedArg?: string;
  seedEnv?: string;
  worstCaseUsd: number;
}

// An interpreter given a program on the command line (python -u -c, bash -lc, python3.11 -c, ...).
const INTERP = /^(.*\/)?(python[\d.]*|sh|bash|zsh|dash)$/;
const INLINE = (argv: string[]) => INTERP.test(argv[0] ?? "") && argv.slice(1).some((a) => /^-[a-zA-Z]*c$/.test(a));

// A claim is measured on its own dataset with its own method, never a stand-in (SimSiam r3: the planner
// mapped the ImageNet claims to a CIFAR-10 run and "SimSiam without stop-gradient" to a SimCLR run).
// Judged against the rest of the plan, so a repository that never names its data is not penalized:
// a stage that names another claim's dataset but not this claim's is a stand-in; so is a stage that
// lacks the method's name while the plan's other stages carry it. Names are compared without case and
// punctuation ("CIFAR-10" = cifar10); the repository files a stage names (configs) count. Pure.
const norm = (x: string) => x.toLowerCase().replace(/[^a-z0-9]/g, "");
const methodKey = (m?: string) => norm((m ?? "").split(/[\s(/,]/)[0] ?? "");
export function stageText(st: { argv: string[]; env?: Record<string, string> }, read?: (path: string) => string | undefined): string {
  const files = st.argv.filter((a) => /^[\w./-]+\.(ya?ml|json|cfg|ini|toml|sh)$/.test(a)).map((a) => read?.(a.replace(/^\/work\//, "")) ?? "");
  return norm([...st.argv, ...Object.entries(st.env ?? {}).flat(), ...files].join(" "));
}
export function claimFitErrors(c: { id: string; dataset?: string; method?: string }, st: { id: string; argv: string[]; env?: Record<string, string> }, others: { datasets: string[]; stageTexts: string[] }, read?: (path: string) => string | undefined): string[] {
  const text = stageText(st, read), errs: string[] = [];
  const ds = norm(c.dataset ?? "");
  const other = others.datasets.map(norm).filter((d) => d.length >= 3 && d !== ds && !ds.includes(d) && !d.includes(ds) && text.includes(d));
  if (ds.length >= 3 && !text.includes(ds) && other.length) errs.push(`${c.id}: the claim is on ${c.dataset}, but stage ${st.id} runs on another dataset of this paper (${other.join(", ")}) and never names ${c.dataset}: measure it on ${c.dataset}, or list it in unmeasured with the reason`);
  const m = methodKey(c.method);
  if (m.length >= 3 && !text.includes(m) && others.stageTexts.some((t) => t !== text && t.includes(m))) errs.push(`${c.id}: the claim is about ${c.method}, but stage ${st.id} never names ${(c.method ?? "").split(/[\s(/,]/)[0]} while the plan's other stages do (a different method is not a stand-in): measure it with ${c.method}, or list it in unmeasured with the reason`);
  return errs;
}

export function validateGpuPlan(claims: Claim[], raw: unknown, o: { budgetUsd: number; price: (gpu: string) => number; cpu?: number; mem?: number; core?: number; exists?: (path: string) => boolean; read?: (path: string) => string | undefined }): { plan?: GpuPlan; errors: string[] } {
  const errors: string[] = [];
  const r = raw as Partial<GpuPlan> & { stages?: Partial<Stage>[]; claims?: { claim?: string; stage?: string; parse?: string; scale?: number }[] };
  if (!r || !Array.isArray(r.stages) || !r.stages.length) return { errors: ["plan has no stages"] };
  const fidelity = ["full", "scaled", "eval_only"].includes(String(r.fidelity)) ? (r.fidelity as GpuPlan["fidelity"]) : undefined;
  if (!fidelity) errors.push("fidelity must be full, scaled or eval_only");
  const stages: Stage[] = [];
  const seen = new Set<string>();
  for (const s of r.stages) {
    const id = String(s.id ?? "").replace(/[^\w-]/g, "_");
    if (!id || seen.has(id)) { errors.push(`stage id ${s.id} missing or duplicated`); continue; }
    if (!Array.isArray(s.argv) || !s.argv.length || !s.argv.every((a) => typeof a === "string")) { errors.push(`${id}: argv must be a non-empty list of strings`); continue; }
    // Placeholders: only {seed}, and only where there is a seed to put in.
    const ph = [...new Set(s.argv.flatMap((a) => a.match(/\{[^}]*\}/g) ?? []))];
    if (ph.some((x) => x !== "{seed}")) { errors.push(`${id}: unknown placeholder ${ph.filter((x) => x !== "{seed}").join(", ")} in argv (only {seed}, in a perSeed stage)`); continue; }
    if (ph.length && !s.perSeed) { errors.push(`${id}: {seed} in argv needs perSeed: true`); continue; }
    if (INLINE(s.argv)) { errors.push(`${id}: inline programs (python -c, sh -c) are not allowed; run a script of the repository (a fetch script may be added by the debugger as a reviewed change)`); continue; }
    const needs = (Array.isArray(s.needs) ? s.needs : []).map(String);
    const unknown = needs.find((n) => !seen.has(n));
    if (unknown) { errors.push(`${id}: needs ${unknown}, which is not an earlier stage`); continue; }
    const seeded = needs.find((n) => stages.find((x) => x.id === n)?.perSeed);
    if (seeded && !s.perSeed) { errors.push(`${id}: needs per-seed stage ${seeded} but is not per-seed itself (make it perSeed so each seed's outputs reach it)`); continue; }
    if (s.gpu !== undefined && !GPUS.includes(String(s.gpu))) { errors.push(`${id}: gpu ${s.gpu} not in ${GPUS.join(", ")}`); continue; }
    // The script a stage runs must exist: a planner may not invent code (the debugger adds code as a reviewed change).
    // -m: a local module (mod.py, mod/__init__.py, mod/__main__.py) must exist; an installed module (torch.distributed.run) is not checked.
    const mi = s.argv.indexOf("-m");
    const modPath = mi > 0 ? (s.argv[mi + 1] ?? "").split(".").join("/") : "";
    const top = modPath.split("/")[0] ?? "";
    // The repository is mounted at /work in the container: "/work/x" is the repository path "x".
    // A cwd under /cache (a data directory) is allowed; scripts are then not checked against the repository.
    const rawCwd = typeof s.cwd === "string" ? s.cwd.replace(/\/+$/, "") : undefined;
    const cwd = rawCwd === undefined || rawCwd === "/work" || rawCwd === "" || rawCwd === "." ? undefined : rawCwd.startsWith("/work/") ? rawCwd.slice(6) : rawCwd;
    if (cwd !== undefined && (cwd.includes("..") || (cwd.startsWith("/") && !cwd.startsWith("/cache/")))) { errors.push(`${id}: cwd ${s.cwd} must be inside the repository (/work/...) or the cache (/cache/...)`); continue; }
    const repoPath = (p: string) => (p.startsWith("/work/") ? p.slice(6) : p);
    const inRepo = (p: string) => !!o.exists && (p.startsWith("/") && !p.startsWith("/work/") ? true : (cwd && !cwd.startsWith("/") ? o.exists(`${cwd}/${repoPath(p)}`) : false) || o.exists(repoPath(p)));
    const script = mi > 0 ? (inRepo(`${top}.py`) || inRepo(top) ? [`${modPath}.py`, `${modPath}/__init__.py`, `${modPath}/__main__.py`].find(inRepo) ?? `${modPath}.py` : undefined) : s.argv.find((a) => /\.(py|sh)$/.test(a));
    if (o.exists && script && !(cwd?.startsWith("/cache/") && !script.startsWith("/work/")) && !inRepo(script)) { errors.push(`${id}: ${script} does not exist in the repository; use an existing script (data can be downloaded at run time by a stage with "network": true, restricted to data hosts)`); continue; }
    const timeoutS = Math.min(Math.max(Number(s.timeoutS) || 1800, 60), MAX_STAGE_S);
    const persist = (Array.isArray(s.persist) ? s.persist : []).map(String).filter((p) => !p.startsWith("/") && !p.includes(".."));
    // The cheapest GPU with enough memory, when the planner states the need.
    const mem = Number(s.gpuMemGB);
    let gpu = s.gpu ? String(s.gpu) : undefined;
    if (mem > 0) {
      const fit = GPUS.filter((g) => GPU_MEM[g]! >= mem).sort((a, b) => o.price(a) - o.price(b))[0];
      if (!fit) { errors.push(`${id}: no GPU has ${mem} GB`); continue; }
      gpu = fit;
    }
    const memGB = Number((s as { memGB?: unknown }).memGB);
    if ((s as { memGB?: unknown }).memGB !== undefined && !(memGB >= 1 && memGB <= 256)) { errors.push(`${id}: memGB must be between 1 and 256`); continue; }
    const cpus = Number((s as { cpus?: unknown }).cpus);
    if ((s as { cpus?: unknown }).cpus !== undefined && !(Number.isInteger(cpus) && cpus >= 1 && cpus <= 32)) { errors.push(`${id}: cpus must be a whole number from 1 to 32`); continue; }
    const rawLinks = Array.isArray((s as { links?: unknown }).links) ? ((s as { links: { path?: unknown; to?: unknown }[] }).links) : [];
    const links = rawLinks.map((l) => ({ path: String(l.path ?? ""), to: String(l.to ?? "") }));
    const badLink = links.find((l) => !l.path || l.path.startsWith("/") || l.path.split("/").includes("..") || !l.to.startsWith("/cache/data/"));
    if (badLink) { errors.push(`${id}: link ${JSON.stringify(badLink)} must be {path: <repo-relative>, to: "/cache/data/..."}`); continue; }
    seen.add(id);
    stages.push({ id, argv: s.argv, timeoutS, needs, persist, ...(memGB > 0 ? { memGB } : {}), ...(cpus > 0 ? { cpus } : {}), ...(links.length ? { links } : {}), ...(gpu ? { gpu } : {}), ...(mem > 0 ? { gpuMemGB: mem } : {}), ...(s.network ? { network: true } : {}), ...(s.perSeed ? { perSeed: true } : {}),
      ...(s.env && typeof s.env === "object" ? { env: Object.fromEntries(Object.entries(s.env).filter(([k, v]) => /^[A-Z_][A-Z0-9_]*$/.test(k) && typeof v === "string" && !/^(PATH|HOME|LD_|PYTHONHOME)/.test(k))) as Record<string, string> } : {}),
      ...(cwd !== undefined ? { cwd } : {}) });
  }
  // Data: declared fetch items under /cache/data, from https URLs.
  const fetch: FetchItem[] = [];
  for (const f of (r as { fetch?: Partial<FetchItem>[] }).fetch ?? []) {
    let host = "";
    // https, or plain http with a sha256: the checksum (verified by fetch_data.py) guarantees integrity
    // whatever the transport. Some data hosts serve only http (images.cocodataset.org's certificate
    // does not match its name; forcing https made a planner write an unusable URL, P5 run 1).
    try { const u = new URL(String(f.url)); if (u.protocol !== "https:" && !(u.protocol === "http:" && typeof f.sha256 === "string" && /^[0-9a-f]{64}$/i.test(f.sha256))) throw new Error(); host = u.hostname; } catch { errors.push(`fetch url ${f.url} must be https, or http with its sha256`); continue; }
    const to = String(f.to ?? "");
    if (!to.startsWith("/cache/data/") || to.includes("..")) { errors.push(`fetch target ${to} must be under /cache/data/`); continue; }
    fetch.push({ url: String(f.url), to, ...(typeof f.sha256 === "string" && /^[0-9a-f]{64}$/i.test(f.sha256) ? { sha256: f.sha256 } : {}), ...(f.unpack === "zip" || f.unpack === "tar" ? { unpack: f.unpack } : {}) });
    void host;
  }
  const checks = ((r as { checks?: { path?: string; minFiles?: number }[] }).checks ?? []).filter((c) => typeof c.path === "string" && c.path.startsWith("/cache/") && Number(c.minFiles) > 0).map((c) => ({ path: c.path!, minFiles: Number(c.minFiles) }));
  // Every /cache path a stage uses must come from a fetch item: data whose origin is unknown cannot be reproduced.
  for (const st of stages) {
    // The whole volume is not an input: it would hand a stage whatever any earlier command left there
    // (FT-Transformer r2: PROJECT_DIR=/cache read data the env agent downloaded, unrecorded).
    if ([...st.argv, ...Object.values(st.env ?? {}), st.cwd ?? ""].some((a) => /^\/cache\/?$|^\/cache\/data\/?$/.test(a))) errors.push(`${st.id}: a stage may not use the whole /cache or /cache/data; point it at the directory a fetch item fills (e.g. /cache/data/<name>, with unpack)`);
    for (const ref of [...st.argv, ...Object.values(st.env ?? {}), ...(st.links ?? []).map((l) => l.to)].flatMap((a) => a.match(/\/cache\/[\w./-]+/g) ?? [])) {
      // No traversal: "/cache/runs/../data/x" or "/cache/data/a/../../etc" would escape every rule below.
      if (ref.split("/").includes("..")) { errors.push(`${st.id}: ${ref}: ".." is not allowed in /cache paths`); continue; }
      if (ref.startsWith("/cache/runs/")) continue;
      const under = (a: string, b: string) => a === b || a.startsWith(b.endsWith("/") ? b : `${b}/`);
      // A fetched file, a path inside a fetched archive, or a directory the fetch items fill.
      const r0 = ref.replace(/\/$/, "");
      if (!fetch.some((f) => { const t = f.to.replace(/\/$/, ""); return under(r0, t) || (r0.startsWith("/cache/data/") && under(t, r0)); })) errors.push(r0.startsWith("/cache/data/") ? `${st.id}: ${ref} is not produced by any fetch item; declare where it comes from (fetch: [{url, to: "/cache/data/..."}])` : `${st.id}: ${ref} is not produced by any fetch item: inputs go under /cache/data/ via fetch items; write outputs (logs, checkpoints) under /cache/runs/`);
    }
  }
  const unknownHosts = [...new Set(fetch.map((f) => new URL(f.url).hostname).filter((h) => !DATA_HOSTS.some((d) => h === d || h.endsWith(`.${d}`))))];
  const cl: GpuPlan["claims"] = [];
  // A planner sometimes names a claim by its description instead of its id: accept it when exactly one
  // claim's value appears in the text (deterministic), and say which ids to use otherwise.
  const pct = (v: number) => [String(v), (v * 100).toFixed(1), (v * 100).toFixed(2)];
  const resolve = (name: string): string | undefined => {
    if (claims.some((c) => c.id === name)) return name;
    const hits = claims.filter((c) => pct(c.value).some((f) => new RegExp(`(?<![\\d.])${f.replace(".", "\\.")}(?![\\d])`).test(name)));
    return hits.length === 1 ? hits[0]!.id : undefined;
  };
  const unmeasured: GpuPlan["unmeasured"] = [];
  const skip = Array.isArray((r as { unmeasured?: unknown }).unmeasured) ? (r as { unmeasured: { claim?: unknown; reason?: unknown }[] }).unmeasured : [];
  for (const c of claims) {
    const m = (r.claims ?? []).find((x) => typeof x?.claim === "string" && resolve(x.claim) === c.id);
    const u = skip.find((x) => typeof x?.claim === "string" && resolve(x.claim) === c.id);
    if (u) {
      if (m) { errors.push(`${c.id}: listed both as measured and as unmeasured`); continue; }
      if (String(u.reason ?? "").trim().length < 20) { errors.push(`${c.id}: an unmeasured claim needs its reason (what it would take, why that is out of reach)`); continue; }
      unmeasured.push({ claim: c.id, reason: String(u.reason).trim().slice(0, 500) }); continue;
    }
    if (!m) { errors.push(`no stage measures claim ${c.id} (set "claim" to exactly one of: ${claims.map((x) => JSON.stringify(x.id)).join(", ")})`); continue; }
    if (!stages.some((s) => s.id === m.stage)) { errors.push(`${c.id}: stage ${m.stage} does not exist`); continue; }
    const fit = claimFitErrors(c, stages.find((s) => s.id === m.stage)!, { datasets: claims.map((x) => x.dataset ?? ""), stageTexts: stages.map((x) => stageText(x, o.read)) }, o.read);
    if (fit.length) { errors.push(...fit); continue; }
    let groups = -1;
    try { groups = new RegExp(`${String(m.parse)}|`).exec("")!.length - 1; } catch { /* invalid */ }
    if (groups !== 1) { errors.push(`${c.id}: parse must be a regex with exactly one capture group`); continue; }
    // The pattern must find the metric by its label, not by expected numbers (SimCSE r1: the planner
    // wrote the seven per-task scores into the pattern; it only matched because they repeated exactly).
    if ((String(m.parse).match(/\d+\\?\.\d+/g) ?? []).length > 2) { errors.push(`${c.id}: parse must locate the metric by its label, not by expected numbers (it contains more than two decimal literals)`); continue; }
    // How the paper reports the number, declared before any result is seen: which value of a per-epoch
    // series counts (last, max = best epoch for accuracy-like metrics, min = best for errors) and how
    // seeds combine (mean or median). LoRA reports the best epoch's value, median over seeds.
    const sel = (m as { select?: unknown }).select, agg = (m as { aggregate?: unknown }).aggregate;
    if (sel !== undefined && !["last", "max", "min"].includes(String(sel))) { errors.push(`${c.id}: select must be last, max or min`); continue; }
    if (agg !== undefined && !["mean", "median"].includes(String(agg))) { errors.push(`${c.id}: aggregate must be mean or median`); continue; }
    cl.push({ claim: c.id, stage: String(m.stage).replace(/[^\w-]/g, "_"), parse: String(m.parse), ...(typeof m.scale === "number" && m.scale > 0 ? { scale: m.scale } : {}), ...(sel ? { select: sel as "last" | "max" | "min" } : {}), ...(agg ? { aggregate: agg as "mean" | "median" } : {}) });
  }
  const seeds = [...new Set((Array.isArray(r.seeds) ? r.seeds : [0]).filter((x) => Number.isInteger(x)))].slice(0, 5);
  if (!seeds.length) seeds.push(0);
  // Worst case: every stage instance runs to its timeout, plus container start-up.
  const worstCaseUsd = stages.reduce((a, s) => a + ((s.gpu ? (o.price(s.gpu) * (s.timeoutS + 300)) / 3600 : ((o.cpu ?? 0.1) * (s.timeoutS + 300)) / 3600) + ((s.cpus ?? 0) * (o.core ?? 0.05) * (s.timeoutS + 300)) / 3600 + ((s.memGB ?? 0) * (o.mem ?? 0.008) * (s.timeoutS + 300)) / 3600) * (s.perSeed ? seeds.length : 1), 0);
  if (worstCaseUsd > o.budgetUsd) errors.push(`worst-case cost $${worstCaseUsd.toFixed(2)} exceeds the budget $${o.budgetUsd.toFixed(2)}: lower fidelity, shorter timeouts, fewer seeds or a cheaper GPU`);
  // Every stage must serve a claim, directly or as something a claim's stage needs (FT-Transformer r2:
  // 30 stages planned, 2 read by claims, ~2.5 h of GPU for nothing).
  const used = new Set(cl.map((c) => c.stage));
  for (let grew = true; grew;) { grew = false; for (const st of stages) if (used.has(st.id)) for (const d of st.needs) if (!used.has(d)) { used.add(d); grew = true; } }
  const idle = stages.filter((st) => !used.has(st.id)).map((st) => st.id);
  if (idle.length) errors.push(`stages ${idle.slice(0, 8).join(", ")}${idle.length > 8 ? ` and ${idle.length - 8} more` : ""} serve no claim; drop them (a claim reads one stage, or use perSeed with seeds for several runs)`);
  if (errors.length || !fidelity) return { errors };
  return { plan: { fidelity, fidelityNote: String(r.fidelityNote ?? ""), fetch, checks, unknownHosts, stages, claims: cl, unmeasured, seeds, ...(typeof r.seedArg === "string" && /^--?[\w-]+$/.test(r.seedArg) ? { seedArg: r.seedArg } : {}), ...(typeof r.seedEnv === "string" && /^[A-Z_][A-Z0-9_]*$/.test(r.seedEnv) ? { seedEnv: r.seedEnv } : {}), worstCaseUsd }, errors: [] };
}

// The plan as amended by accepted owner patches (seeds, how a claim's number is read): an amendment is
// journaled with its human basis like any patch (LoRA r3: the paper reports the median over 5 seeds of
// each run's best epoch, which the plan had not declared).
export interface PlanAmend { seeds?: number[]; claims?: { claim: string; select?: "last" | "max" | "min"; aggregate?: "mean" | "median" }[] }
export const gpuPlanOf = (s: RunState): GpuPlan | undefined => {
  const p = (s.nodes.get(N.plan)?.output as { gpuPlan?: GpuPlan } | undefined)?.gpuPlan;
  if (!p) return p;
  const amends = s.order.filter((id) => id.startsWith(N.patch) && s.nodes.get(id)?.status === "succeeded").map((id) => s.nodes.get(id)!.output as { accepted?: boolean; plan?: PlanAmend }).filter((o) => o?.accepted && o.plan);
  if (!amends.length) return p;
  let q: GpuPlan = { ...p, claims: p.claims.map((c) => ({ ...c })) };
  for (const { plan: a } of amends) {
    if (a!.seeds?.length) q = { ...q, seeds: a!.seeds };
    for (const c of a!.claims ?? []) q.claims = q.claims.map((x) => (x.claim === c.claim ? { ...x, ...(c.select ? { select: c.select } : {}), ...(c.aggregate ? { aggregate: c.aggregate } : {}) } : x));
  }
  return q;
};

export const stageId = (stage: string, version: string, seed?: number) => `repro.stage:${stage}:v${version}${seed !== undefined ? `:s${seed}` : ""}`;

// All stage nodes and per-claim comparisons for the current source version.
export function stageNodes(s: RunState, tag: string): TaskNode[] {
  const p = gpuPlanOf(s);
  if (!p) return [];
  const v = `${currentVersion(s).version}${tag}`;
  const nodes: TaskNode[] = [];
  const instances = (st: Stage) => (st.perSeed ? p.seeds.map((seed) => stageId(st.id, v, seed)) : [stageId(st.id, v)]);
  // Data is fetched once per run (idempotent and cached), before any stage.
  const fetchId = "repro.fetch";
  const hasFetch = (p.fetch?.length ?? 0) > 0 || (p.checks?.length ?? 0) > 0;
  if (hasFetch) nodes.push({ id: fetchId, skill: "repro.fetch", deps: [N.plan], purpose: "setup", params: {} });
  for (const st of p.stages) {
    for (const seed of st.perSeed ? p.seeds : [undefined]) {
      const deps = [N.plan, ...(hasFetch ? [fetchId] : []), ...st.needs.flatMap((n) => {
        const dep = p.stages.find((x) => x.id === n)!;
        return dep.perSeed ? (seed !== undefined ? [stageId(n, v, seed)] : instances(dep)) : [stageId(n, v)];
      })];
      nodes.push({ id: stageId(st.id, v, seed), skill: "repro.stage", deps, purpose: "detect", params: { stage: st.id, version: currentVersion(s).version, tag: v, ...(seed !== undefined ? { seed } : {}) } });
    }
  }
  for (const c of p.claims) {
    const st = p.stages.find((x) => x.id === c.stage)!;
    nodes.push({ id: `${N.compare}:${c.claim}:v${v}`, skill: N.compare, deps: instances(st), purpose: "detect", params: { claim: c.claim, version: currentVersion(s).version } });
  }
  return nodes.filter((n) => !s.nodes.has(n.id));
}

export const claimIdsOf = (s: RunState) => claimsOf(s).map((c) => c.id);
