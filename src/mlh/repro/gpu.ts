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
// Hosts that serve datasets and released weights. Fetching from anywhere else needs a human.
export const DATA_HOSTS = [
  "images.cocodataset.org", "dl.fbaipublicfiles.com", "download.pytorch.org", "huggingface.co", "cdn-lfs.huggingface.co",
  "github.com", "objects.githubusercontent.com", "raw.githubusercontent.com", "zenodo.org", "storage.googleapis.com",
  "s3.amazonaws.com", "www.cs.toronto.edu", "people.eecs.berkeley.edu", "data.dgl.ai", "snap.stanford.edu", "ogb.stanford.edu",
  "nlp.stanford.edu", "dl.dropboxusercontent.com", "www.dropbox.com", "drive.google.com", "drive.usercontent.google.com",
];
// What a stage with network may reach (Modal outbound domain allowlist; wildcards allowed): data and
// weight hosts and their CDNs. Package indexes and everything else stay blocked.
export const DATA_DOMAINS = [
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
  claims: { claim: string; stage: string; parse: string; scale?: number }[];
  seeds: number[];
  seedArg?: string;
  seedEnv?: string;
  worstCaseUsd: number;
}

// An interpreter given a program on the command line (python -u -c, bash -lc, python3.11 -c, ...).
const INTERP = /^(.*\/)?(python[\d.]*|sh|bash|zsh|dash)$/;
const INLINE = (argv: string[]) => INTERP.test(argv[0] ?? "") && argv.slice(1).some((a) => /^-[a-zA-Z]*c$/.test(a));

export function validateGpuPlan(claims: Claim[], raw: unknown, o: { budgetUsd: number; price: (gpu: string) => number; cpu?: number; exists?: (path: string) => boolean }): { plan?: GpuPlan; errors: string[] } {
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
    seen.add(id);
    stages.push({ id, argv: s.argv, timeoutS, needs, persist, ...(gpu ? { gpu } : {}), ...(mem > 0 ? { gpuMemGB: mem } : {}), ...(s.network ? { network: true } : {}), ...(s.perSeed ? { perSeed: true } : {}),
      ...(s.env && typeof s.env === "object" ? { env: Object.fromEntries(Object.entries(s.env).filter(([k, v]) => /^[A-Z_][A-Z0-9_]*$/.test(k) && typeof v === "string" && !/^(PATH|HOME|LD_|PYTHONHOME)/.test(k))) as Record<string, string> } : {}),
      ...(cwd !== undefined ? { cwd } : {}) });
  }
  // Data: declared fetch items under /cache/data, from https URLs.
  const fetch: FetchItem[] = [];
  for (const f of (r as { fetch?: Partial<FetchItem>[] }).fetch ?? []) {
    let host = "";
    try { const u = new URL(String(f.url)); if (u.protocol !== "https:") throw new Error(); host = u.hostname; } catch { errors.push(`fetch url ${f.url} must be https`); continue; }
    const to = String(f.to ?? "");
    if (!to.startsWith("/cache/data/") || to.includes("..")) { errors.push(`fetch target ${to} must be under /cache/data/`); continue; }
    fetch.push({ url: String(f.url), to, ...(typeof f.sha256 === "string" && /^[0-9a-f]{64}$/i.test(f.sha256) ? { sha256: f.sha256 } : {}), ...(f.unpack === "zip" || f.unpack === "tar" ? { unpack: f.unpack } : {}) });
    void host;
  }
  const checks = ((r as { checks?: { path?: string; minFiles?: number }[] }).checks ?? []).filter((c) => typeof c.path === "string" && c.path.startsWith("/cache/") && Number(c.minFiles) > 0).map((c) => ({ path: c.path!, minFiles: Number(c.minFiles) }));
  // Every /cache path a stage uses must come from a fetch item: data whose origin is unknown cannot be reproduced.
  for (const st of stages) {
    for (const ref of [...st.argv, ...Object.values(st.env ?? {})].flatMap((a) => a.match(/\/cache\/[\w./-]+/g) ?? [])) {
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
  for (const c of claims) {
    const m = (r.claims ?? []).find((x) => typeof x?.claim === "string" && resolve(x.claim) === c.id);
    if (!m) { errors.push(`no stage measures claim ${c.id} (set "claim" to exactly one of: ${claims.map((x) => JSON.stringify(x.id)).join(", ")})`); continue; }
    if (!stages.some((s) => s.id === m.stage)) { errors.push(`${c.id}: stage ${m.stage} does not exist`); continue; }
    let groups = -1;
    try { groups = new RegExp(`${String(m.parse)}|`).exec("")!.length - 1; } catch { /* invalid */ }
    if (groups !== 1) { errors.push(`${c.id}: parse must be a regex with exactly one capture group`); continue; }
    cl.push({ claim: c.id, stage: String(m.stage).replace(/[^\w-]/g, "_"), parse: String(m.parse), ...(typeof m.scale === "number" && m.scale > 0 ? { scale: m.scale } : {}) });
  }
  const seeds = [...new Set((Array.isArray(r.seeds) ? r.seeds : [0]).filter((x) => Number.isInteger(x)))].slice(0, 5);
  if (!seeds.length) seeds.push(0);
  // Worst case: every stage instance runs to its timeout, plus container start-up.
  const worstCaseUsd = stages.reduce((a, s) => a + (s.gpu ? (o.price(s.gpu) * (s.timeoutS + 300)) / 3600 : ((o.cpu ?? 0.1) * (s.timeoutS + 300)) / 3600) * (s.perSeed ? seeds.length : 1), 0);
  if (worstCaseUsd > o.budgetUsd) errors.push(`worst-case cost $${worstCaseUsd.toFixed(2)} exceeds the budget $${o.budgetUsd.toFixed(2)}: lower fidelity, shorter timeouts, fewer seeds or a cheaper GPU`);
  if (errors.length || !fidelity) return { errors };
  return { plan: { fidelity, fidelityNote: String(r.fidelityNote ?? ""), fetch, checks, unknownHosts, stages, claims: cl, seeds, ...(typeof r.seedArg === "string" && /^--?[\w-]+$/.test(r.seedArg) ? { seedArg: r.seedArg } : {}), ...(typeof r.seedEnv === "string" && /^[A-Z_][A-Z0-9_]*$/.test(r.seedEnv) ? { seedEnv: r.seedEnv } : {}), worstCaseUsd }, errors: [] };
}

export const gpuPlanOf = (s: RunState): GpuPlan | undefined => (s.nodes.get(N.plan)?.output as { gpuPlan?: GpuPlan } | undefined)?.gpuPlan;

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
