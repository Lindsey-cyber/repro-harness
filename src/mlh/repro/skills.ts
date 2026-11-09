// Node implementations of the reproduction flow. Agents explore and propose;
// everything that decides (claim validity, plan validity, run results, claim
// status, patch risk) is harness code.
import { createHash } from "node:crypto";
import { closeSync, existsSync, mkdirSync, openSync, readdirSync, readFileSync, readSync, rmSync, statSync, writeFileSync } from "node:fs";
import { join, relative, resolve } from "node:path";

// Identity of a source tree for caching: relative paths, sizes and the first 64 KB of every file.
export function treeHash(root: string): string {
  const h = createHash("sha256");
  const buf = Buffer.alloc(65536);
  const walk = (d: string) => {
    for (const e of readdirSync(d, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
      if (e.name === ".git" || e.name === "__pycache__" || e.name === "outputs") continue;
      const p = join(d, e.name);
      if (e.isDirectory()) { walk(p); continue; }
      if (!e.isFile()) continue;
      const fd = openSync(p, "r");
      const n = readSync(fd, buf, 0, buf.length, 0);
      closeSync(fd);
      h.update(`${relative(root, p)}\0${statSync(p).size}\0`).update(buf.subarray(0, n));
    }
  };
  walk(root);
  return h.digest("hex").slice(0, 20);
}
import { z } from "zod";
import { agentLoop } from "../agentloop.ts";
import { authorize, type Evidence } from "../core/authz.ts";
import type { NodeState, RunState } from "../core/dag.ts";
import { Sandbox, SkillError } from "../io/sandbox.ts";
import type { Skill, SkillCtx } from "../skills/index.ts";
import { claimsOf, currentVersion, N, specsOf, validatePlan, type Claim, type RunSpec, type Status } from "./flow.ts";
import { buildVersion, changedFiles, unifiedDiff } from "./patch.ts";
import { parseDiff, riskOf } from "./patchrisk.ts";
import { CLAIMS_PROMPT, DEBUG_ROLE, ENV_ROLE, PLAN_ROLE } from "./roles.ts";
import type { SiteManager } from "./site.ts";
import { Budget, cpuPrice, gpuPrice, modalRun } from "../backends/modal.ts";
import { checkFetchItems, DATA_DOMAINS, DEFAULT_IMAGE_KEY, GPUS, IMAGES, gpuPlanOf, imageForPins, type FetchItem, validateGpuPlan, type Stage } from "./gpu.ts";
import { GPU_PLAN_ROLE } from "./roles.ts";

export interface ReproCtx {
  arxiv: string;
  sites: SiteManager;
  versionDir(v: number): string;
  approvePolicy: "none" | "medium" | "all";
  // Remote GPU execution (Modal). Absent: everything runs in the local sandbox.
  gpu?: { budget: Budget; runKey: string };
}

const any = z.record(z.string(), z.unknown());
const need = (ctx: SkillCtx): ReproCtx => { if (!ctx.repro) throw new SkillError("harness_bug", "not a reproduction run"); return ctx.repro; };
const models = (ctx: SkillCtx) => { if (!ctx.llm) throw new SkillError("precondition", "model use not enabled (--llm)"); return ctx.llm; };
const outOf = <T>(s: RunState, id: string) => s.nodes.get(id)?.output as T;

// The newest package site: the environment agent's, or one a debugger extended.
export function currentSite(s: RunState): string | undefined {
  let site = outOf<{ site?: string } | undefined>(s, N.env)?.site;
  for (const id of s.order) if (id.startsWith(N.patch) && s.nodes.get(id)?.status === "succeeded") { const p = outOf<{ accepted: boolean; site?: string }>(s, id); if (p.accepted && p.site) site = p.site; }
  return site;
}

// Agents' commands run in remote containers when the run uses Modal: network allowed (they install and
// download while exploring), no credentials inside, GPU time charged to the paper's budget.
// Is each declared download reachable from here? A HEAD (or a one-byte GET where HEAD is refused).
// The error text goes back to the planner, so a wrong scheme, host or path is fixed during planning.
// MLH_SKIP_URL_PROBE=1 for offline tests whose URLs are served by the fake runner.
export async function probeUrls(urls: string[]): Promise<string[]> {
  if (process.env.MLH_SKIP_URL_PROBE === "1") return [];
  const bad: string[] = [];
  for (const url of [...new Set(urls)]) {
    try {
      let r = await fetch(url, { method: "HEAD", redirect: "follow", signal: AbortSignal.timeout(20_000) });
      if (r.status === 403 || r.status === 405) r = await fetch(url, { headers: { Range: "bytes=0-0" }, redirect: "follow", signal: AbortSignal.timeout(20_000) });
      if (r.status >= 400) bad.push(`fetch url ${url} answered HTTP ${r.status}; use a URL that serves the file${url.startsWith("https:") ? " (if the host serves only http, give the http URL with its sha256)" : ""}`);
    } catch (e) {
      const why = String((e as { cause?: unknown }).cause ?? e).slice(0, 200);
      bad.push(`fetch url ${url} is not reachable (${why})${url.startsWith("https:") ? "; if the host serves only http, give the http URL with its sha256" : ""}`);
    }
  }
  return bad;
}

const remoteFor = (ctx: SkillCtx, pip: string[], image?: string) => {
  const g = need(ctx).gpu;
  if (!g) return {};
  return { remote: { pip, gpus: GPUS, images: IMAGES, image: image ?? DEFAULT_IMAGE_KEY, run: async (ws: string, command: string, o: { timeoutS: number; gpu?: string; pip: string[]; image?: string }) => {
    const r = await modalRun({ repo: ws, cmd: ["bash", "-lc", command], timeoutS: o.timeoutS, network: true, pip: o.pip, image: IMAGES[o.image ?? DEFAULT_IMAGE_KEY]!, ...(o.gpu ? { gpu: o.gpu } : {}) }, g.budget);
    if (r.category === "provider_fatal") throw new SkillError("provider_fatal", r.stderr.slice(0, 300)); // out of GPU credit: stop, do not let the agent keep trying
    return { code: r.exitCode, stdout: r.stdout, stderr: r.stderr, category: r.category, costUsd: r.costUsd };
  } } };
};
export const imageOf = (s: RunState): string => {
  let img = (s.nodes.get(N.env)?.output as { image?: string } | undefined)?.image ?? DEFAULT_IMAGE_KEY;
  for (const id of s.order) if (id.startsWith(N.patch) && s.nodes.get(id)?.status === "succeeded") { const p = s.nodes.get(id)!.output as { accepted: boolean; image?: string }; if (p.accepted && p.image) img = p.image; }
  return img;
};
export const pipOf = (s: RunState): string[] => {
  let pip = (s.nodes.get(N.env)?.output as { pip?: string[] } | undefined)?.pip ?? [];
  for (const id of s.order) if (id.startsWith(N.patch) && s.nodes.get(id)?.status === "succeeded") { const p = s.nodes.get(id)!.output as { accepted: boolean; pip?: string[] }; if (p.accepted && p.pip) pip = p.pip; }
  return pip;
};

const loopBase = (ctx: SkillCtx) => ({
  model: models(ctx).proposer, isolation: ctx.isolation, isolationEvidence: ctx.isolationEvidence, checkpoint: true,
  limits: { cpuSec: 900, memMB: 8192, fileMB: 2048, timeoutMs: 900_000 },
  install: async (pk: string[], _idx: string | undefined, cur: string | undefined) => need(ctx).sites.install(pk, cur),
  ...(ctx.sandboxFault ? { sandboxFault: ctx.sandboxFault } : {}),
  ...(ctx.crashAfterSteps !== undefined ? { crashAfterSteps: ctx.crashAfterSteps } : {}),
});

const paper: Skill = {
  id: N.paper, version: "1.0.0", purpose: "setup", intents: ["reproduce"], actions: ["fetch_source", "exec_sandboxed"], after: [], needs: () => null,
  summary: "Download the paper's PDF from arXiv and extract its text in the sandbox.",
  output: z.object({ chars: z.number(), arxiv: z.string() }),
  async run(ctx) {
    const id = need(ctx).arxiv;
    if (!/^\d{4}\.\d{4,5}(v\d+)?$/.test(id)) throw new SkillError("precondition", `not an arXiv id: ${id}`);
    const dir = ctx.workDir(); mkdirSync(dir, { recursive: true });
    const pdf = join(dir, "paper.pdf");
    if (!existsSync(pdf)) {
      const r = await fetch(`https://arxiv.org/pdf/${id}`, { signal: AbortSignal.timeout(60_000) }).catch((e) => { throw new SkillError("sandbox_transient", `arXiv fetch failed: ${e}`); });
      if (!r.ok) throw new SkillError(r.status >= 500 ? "sandbox_transient" : "precondition", `arXiv returned ${r.status}`);
      writeFileSync(pdf, Buffer.from(await r.arrayBuffer()));
    }
    // An untrusted PDF is parsed inside the sandbox, never on the host.
    const sb = Sandbox.attach(dir, { isolation: ctx.isolation, limits: { cpuSec: 120, memMB: 2048, fileMB: 200, timeoutMs: 120_000 }, stats: { runs: 0, ms: 0 } });
    const r = await sb.run(["pdftotext", "-layout", "paper.pdf", "paper.txt"]);
    if (r.category) throw new SkillError(r.category, `pdftotext failed: ${r.stderr.slice(-300)}`);
    ctx.save("paper.pdf", pdf); ctx.save("paper.txt", join(dir, "paper.txt"));
    return { output: { chars: readFileSync(join(dir, "paper.txt"), "utf8").length, arxiv: id } };
  },
};

export const paperText = (ctx: SkillCtx): string => {
  const ref = ctx.state.nodes.get(N.paper)!.artifacts.find((a) => a.name === "paper.txt");
  if (!ref) throw new SkillError("precondition", "no paper text");
  return readFileSync(ctx.load(ref), "utf8");
};

// A claimed number must literally appear in the paper text, as written or as a percentage.
export function numberInText(text: string, v: number): boolean {
  const t = text.replace(/\s+/g, " ");
  const forms = new Set([String(v), v.toFixed(3), v.toFixed(2), (v * 100).toFixed(1), (v * 100).toFixed(2), (v * 100).toFixed(0)].filter((f) => /\d\.\d|\d{2}/.test(f)));
  return [...forms].some((f) => new RegExp(`(?<![\\d.])${f.replace(".", "\\.")}(?![\\d])`).test(t));
}

const claims: Skill = {
  id: N.claims, version: "1.0.0", purpose: "setup", intents: ["reproduce"], actions: ["model_query"], after: [], needs: () => null,
  summary: "A model extracts the claims the objective names from the paper; each claimed number is kept only if it appears in the PDF text.",
  output: z.object({ claims: z.array(any), rejected: z.array(any) }),
  async run(ctx) {
    const text = paperText(ctx);
    const r = await models(ctx).proposer.json<{ claims?: Claim[] }>(CLAIMS_PROMPT, `Objective: ${ctx.state.objective}\n\nPaper text:\n${text.slice(0, 150_000)}`, 16000);
    const kept: Claim[] = [], rejected: unknown[] = [];
    for (const c of r.claims ?? []) {
      const v = Number(c.value);
      if (typeof c.id !== "string" || !Number.isFinite(v)) { rejected.push({ claim: c, why: "malformed" }); continue; }
      if (!numberInText(text, v)) { rejected.push({ claim: c, why: `value ${v} does not appear in the paper text` }); continue; }
      const id = c.id.replace(/[^\w-]/g, "_").slice(0, 40);
      if (kept.some((k) => k.id === id)) continue;
      kept.push({ id, metric: String(c.metric ?? "accuracy"), value: v, ...(c.table ? { table: String(c.table) } : {}), ...(c.dataset ? { dataset: String(c.dataset) } : {}), ...(c.method ? { method: String(c.method) } : {}), ...(c.std != null && Number.isFinite(Number(c.std)) ? { std: Number(c.std) } : {}), ...(c.quote ? { quote: String(c.quote).slice(0, 300) } : {}) });
    }
    if (!kept.length) throw new SkillError("experiment_error", `no claim survived validation (${rejected.length} rejected)`);
    return { output: { claims: kept, rejected } };
  },
};

// The harness's own check that the environment works: start the launch command the environment agent
// reported and watch it for SMOKE_S seconds. Passing means it exited cleanly or was still running
// (training had started), with no Python traceback. The agent's own "runnable" is not trusted.
export const SMOKE_S = 100;
export function smokeVerdict(exit: number | null, output: string): { ok: boolean; why: string } {
  if (/Traceback \(most recent call last\)/.test(output)) return { ok: false, why: "the command raised a Python exception" };
  if (exit === 0) return { ok: true, why: "the command exited cleanly" };
  if (exit === 124) return { ok: true, why: `the command was still running after ${SMOKE_S} s with no error` };
  return { ok: false, why: `the command exited with status ${exit}` };
}

async function smoke(ctx: SkillCtx, launch: { cwd?: string; env?: Record<string, string>; example_command?: string[] } | undefined, pip: string[], site: string | undefined, image?: string): Promise<{ ok: boolean; why: string; command: string; tail: string }> {
  const argv = Array.isArray(launch?.example_command) ? launch!.example_command.map(String) : [];
  if (!argv.length) return { ok: false, why: "the environment agent gave no launch command", command: "", tail: "" };
  const env = Object.fromEntries(Object.entries(launch?.env ?? {}).filter(([k, v]) => /^[A-Z_][A-Z0-9_]*$/.test(k) && typeof v === "string" && !/^(PATH|HOME|LD_|PYTHONHOME)/.test(k))) as Record<string, string>;
  const cwd = typeof launch?.cwd === "string" && !launch.cwd.startsWith("/") && !launch.cwd.includes("..") ? launch.cwd : ".";
  const script = ["bash", "-c", `cd "$1" && shift && timeout ${SMOKE_S} "$@"; echo MLH_SMOKE_EXIT=$?`, "bash", cwd, ...argv];
  const g = need(ctx).gpu;
  let out: string, err: string;
  if (g) {
    const r = await modalRun({ repo: need(ctx).versionDir(0), cmd: script, gpu: "T4", timeoutS: SMOKE_S + 600, pip, env, network: true, image: IMAGES[image ?? DEFAULT_IMAGE_KEY]! }, g.budget);
    // A transport/budget failure is not a verdict on the environment: retry the node instead. (A failed image build is a verdict.)
    if (r.exitCode === -1 && r.category && r.category !== "experiment_error") throw new SkillError(r.category, `smoke test could not run: ${r.stderr.slice(0, 300)}`);
    out = r.stdout; err = r.stderr;
  } else {
    const sb = Sandbox.create(join(ctx.workDir(), ".."), `${ctx.node.id}-smoke`, need(ctx).versionDir(0), { isolation: ctx.isolation, limits: { cpuSec: SMOKE_S * 2, memMB: 8192, fileMB: 2048, timeoutMs: (SMOKE_S + 60) * 1000 }, stats: ctx.stats, ...(site ? { site } : {}) });
    const r = await sb.run(script, env); sb.remove(); out = r.stdout; err = r.stderr;
  }
  const m = /MLH_SMOKE_EXIT=(\d+)/.exec(out);
  const v = smokeVerdict(m ? Number(m[1]) : null, `${out}\n${err}`);
  return { ...v, command: argv.join(" "), tail: `${out.slice(-1500)}\n${err.replace(/^MLH_EXIT=\d+$/m, "").slice(-1500)}` };
}

const env: Skill = {
  id: N.env, version: "1.0.0", purpose: "setup", intents: ["reproduce"], actions: ["exec_sandboxed", "model_query"], after: [], needs: () => null,
  summary: "The environment agent installs what the repository needs and finds how to launch it; it may not edit code.",
  output: any,
  async run(ctx) {
    const src0 = need(ctx).versionDir(0);
    const pins = ["requirements.txt", "requirements/base.txt", "setup.py", "environment.yml", "environment.yaml"].map((f) => join(src0, f)).filter((f) => existsSync(f)).map((f) => readFileSync(f, "utf8")).join("\n");
    const pin = need(ctx).gpu ? imageForPins(pins) : undefined;
    const pinHint = pin ? ` The repository pins torch ${pin.torch}: start with install(image="${pin.key}"), whose Python and torch match it.` : "";
    const remote = need(ctx).gpu ? ` Commands run in a remote Linux container (default image ${DEFAULT_IMAGE_KEY}: Python 3.11, CUDA PyTorch 2.5.1). If the repository needs an older Python/PyTorch, switch the base image with install(image=...) FIRST (${Object.keys(IMAGES).join(", ")}) rather than forcing old packages into a new Python. Packages and image you choose are used by every later stage. Check imports on CPU; use a GPU only for a short probe if needed.` : "";
    // Environment cache: the same source tree whose environment already passed the harness's smoke test is
    // reused after the smoke test passes again (minutes instead of a 10–25 minute agent session).
    // Off unless configured: with per-run /cache volumes a cached environment may rely on files the new run lacks.
    const cacheFile = need(ctx).gpu && process.env.MLH_ENV_CACHE ? join(resolve(process.env.MLH_ENV_CACHE), `${treeHash(need(ctx).versionDir(0))}.json`) : undefined;
    if (cacheFile && existsSync(cacheFile)) {
      const hit = JSON.parse(readFileSync(cacheFile, "utf8")) as Record<string, unknown> & { pip: string[]; image: string; launch: unknown; data?: FetchItem[] };
      // The environment's declared data, downloaded again into this run's own volume, then the smoke test.
      if (hit.data?.length) await harnessFetch(ctx, hit.data);

      const sm = await smoke(ctx, hit.launch as never, hit.pip, undefined, hit.image);
      if (sm.ok) return { output: { ...hit, cached: true, smoke: { ok: true, why: sm.why, command: sm.command, tail: sm.tail.slice(-1200) } } };
    }
    // Data the agent declares is downloaded by the harness (provenance) and recorded with the environment.
    const dataFile = join(ctx.workDir(), "env-data.json");
    const data: FetchItem[] = existsSync(dataFile) ? JSON.parse(readFileSync(dataFile, "utf8")) : [];
    const fetchTool = { def: { type: "function" as const, function: { name: "fetch_data", description: "Download datasets or checkpoints into /cache/data with provenance (sha256 recorded). items: [{url, to: '/cache/data/<name>', unpack?: 'tar'|'zip', sha256?}]. Known data hosts only.", parameters: { type: "object", properties: { items: { type: "array", items: { type: "object" } } }, required: ["items"] } } },
      run: async (args: Record<string, unknown>) => {
        const c = checkFetchItems(Array.isArray(args.items) ? args.items : []);
        if (c.errors.length) return `REFUSED: ${c.errors.join("; ")}`;
        if (c.unknownHosts.length) return `REFUSED: ${c.unknownHosts.join(", ")} is not a known data host; find the dataset on a known host or report it as a problem`;
        const r = await harnessFetch(ctx, c.items);
        for (const it of c.items) if (!data.some((d) => d.url === it.url && d.to === it.to)) data.push(it);
        writeFileSync(dataFile, JSON.stringify(data));
        return `fetched: ${JSON.stringify(r)}`;
      } };
    const base = { ...loopBase(ctx), ...remoteFor(ctx, []), extraTools: [fetchTool], workDir: ctx.workDir(), source: need(ctx).versionDir(0), task: `Objective of the team: ${ctx.state.objective}\nMake this repository's experiments runnable.${remote}${pinHint}` };
    let r = await agentLoop({ ...base, role: ENV_ROLE });
    // Complex repositories need more than one step budget: an agent that stopped at its limit with the
    // experiment still not runnable continues from its checkpoint (two more rounds at most).
    for (let round = 2; round <= 3 && (!r.report || r.report.runnable === false) && r.steps >= ENV_ROLE.maxSteps * (round - 1); round++) {
      const left = r.report ? JSON.stringify((r.report.problems as { category: string }[] | undefined)?.map((p) => p.category) ?? []) : "unknown (you did not report)";
      r = await agentLoop({ ...base, role: { ...ENV_ROLE, maxSteps: Math.max(r.steps, ENV_ROLE.maxSteps * (round - 1)) + ENV_ROLE.maxSteps }, continueWith: `The experiment is not runnable yet. Continue from where you stopped: resolve the remaining problems (${left}), verify by starting the documented command, then report.` });
    }
    if (!r.report) throw new SkillError("experiment_error", "environment agent ran out of steps without a report");
    // Harness-verified: the launch command must actually start. One repair round if it does not.
    const claimed = r.report.runnable === true;
    let sm = await smoke(ctx, r.report.launch as never, r.pip, r.site, r.image);
    // Audited for the environment agent's trust: did its own report pass, without a repair round?
    const repaired = !sm.ok;
    if (!sm.ok) {
      const fix = await agentLoop({ ...base, role: { ...ENV_ROLE, maxSteps: r.steps + ENV_ROLE.maxSteps }, continueWith: `The harness started your launch command (${sm.command}) for ${SMOKE_S} s and it failed: ${sm.why}.\nOutput tail:\n${sm.tail}\nFix the environment (not the code) and report again with a launch command that starts.` });
      if (fix.report) { r = fix; sm = await smoke(ctx, r.report!.launch as never, r.pip, r.site, r.image); }
    }
    const output = { ...r.report, data, runnable: sm.ok, agentSaidRunnable: claimed, repaired, smoke: { ok: sm.ok, why: sm.why, command: sm.command, tail: sm.tail.slice(-1200) }, ...(r.site ? { site: r.site, specs: need(ctx).sites.specsOf(r.site) } : {}), pip: r.pip, image: r.image ?? DEFAULT_IMAGE_KEY, steps: r.steps };
    if (cacheFile && sm.ok) { mkdirSync(resolve(process.env.MLH_ENV_CACHE!), { recursive: true }); writeFileSync(cacheFile, JSON.stringify(output)); }
    return { output };
  },
};

const plan: Skill = {
  id: N.plan, version: "1.0.0", purpose: "setup", intents: ["reproduce"], actions: ["exec_sandboxed", "model_query"], after: [], needs: () => null,
  summary: "The planner agent maps each claim to a run spec; the harness validates it (command, one-group parse regex, ≥3 seeds).",
  output: z.object({ specs: z.array(any), errors: z.array(z.string()) }),
  async run(ctx) {
    const cs = claimsOf(ctx.state);
    const e = outOf<Record<string, unknown>>(ctx.state, N.env);
    const g = need(ctx).gpu;
    if (g) {
      // Multi-stage plan on remote GPUs, validated (structure, GPU allowlist, worst-case cost) by the harness.
      let feedback = "";
      const src0 = need(ctx).versionDir(0);
      // The planner can run the very validator that will judge its report, before reporting.
      const checkPlan = { def: { type: "function" as const, function: { name: "check_plan", description: "Validate a draft plan with the harness's own rules (stage structure, scripts exist, GPU allowlist, declared data, worst-case budget). Takes the same object you would report. Returns OK or the list of problems. Use it before report.", parameters: GPU_PLAN_ROLE.report.parameters } },
        run: async (draft: Record<string, unknown>) => { let v = validateGpuPlan(cs, draft, { budgetUsd: g.budget.remaining(), price: (gpu: string) => gpuPrice(gpu) + cpuPrice(), cpu: cpuPrice(), exists: (p) => existsSync(join(src0, p)) }); if (v.plan) { const bad = await probeUrls(v.plan.fetch.map((f) => f.url)); if (bad.length) v = { errors: bad }; } return v.plan ? `OK: the plan passes (worst case $${v.plan.worstCaseUsd.toFixed(2)}). Now call report with it.` : `REJECTED:\n- ${v.errors.join("\n- ")}`; } };
      for (let attempt = 0; attempt < 3; attempt++) {
        const r = await agentLoop({ ...loopBase(ctx), ...remoteFor(ctx, pipOf(ctx.state), imageOf(ctx.state)), extraTools: [checkPlan], paper: paperText(ctx), workDir: ctx.workDir(), source: need(ctx).versionDir(0), task: `Objective: ${ctx.state.objective}\nBudget left for this paper: $${g.budget.remaining().toFixed(2)}.\nClaims (fractions where percentages):\n${JSON.stringify(cs, null, 1)}\n\nEnvironment engineer's report:\n${JSON.stringify({ runnable: e.runnable, launch: e.launch, problems: e.problems, notes: e.notes, pip: e.pip, data: e.data })}\nData the environment declared (already downloaded into /cache/data with provenance; declare the same items in your plan's fetch, they are not downloaded twice): ${JSON.stringify(e.data ?? [])}`,
          // A rejected plan is fixed in the same conversation (what it learned is kept), not re-explored from scratch.
          role: { ...GPU_PLAN_ROLE, maxSteps: GPU_PLAN_ROLE.maxSteps + attempt * 10 }, ...(feedback ? { continueWith: feedback.trim() } : {}) });
        let v = validateGpuPlan(cs, r.report, { budgetUsd: g.budget.remaining(), price: (gpu: string) => gpuPrice(gpu) + cpuPrice(), cpu: cpuPrice(), exists: (p) => existsSync(join(src0, p)) });
        // Every declared download must answer before any GPU time is spent on the plan.
        if (v.plan) { const bad = await probeUrls(v.plan.fetch.map((f) => f.url)); if (bad.length) v = { errors: bad }; }
        if (v.plan) return { output: { specs: [], errors: [], gpuPlan: v.plan } };
        feedback = `\n\nYour previous plan was rejected: ${v.errors.join("; ")}. Fix it.`;
      }
      throw new SkillError("contract_violation", `no valid stage plan after 3 attempts${feedback}`);
    }
    const site = currentSite(ctx.state);
    const r = await agentLoop({ ...loopBase(ctx), role: PLAN_ROLE, workDir: ctx.workDir(), source: need(ctx).versionDir(0), ...(site ? { site } : {}), task: `Objective: ${ctx.state.objective}\nClaims (values are fractions):\n${JSON.stringify(cs, null, 1)}\n\nEnvironment engineer's report:\n${JSON.stringify({ runnable: e.runnable, launch: e.launch, problems: e.problems, notes: e.notes, installed: e.specs })}` });
    const v = validatePlan(cs, r.report);
    // A working directory must exist in the repository (a planner once wrote the placeholder "<repo root>").
    const src0 = need(ctx).versionDir(0);
    for (const sp of [...v.specs]) if (sp.cwd && !existsSync(join(src0, sp.cwd))) { v.errors.push(`${sp.claim}: cwd ${sp.cwd} does not exist`); v.specs.splice(v.specs.indexOf(sp), 1); }
    if (!v.specs.length) { rmSync(join(ctx.workDir(), "state.json"), { force: true }); throw new SkillError("contract_violation", `invalid plan: ${v.errors.join("; ")}`); }
    return { output: { specs: v.specs, errors: v.errors } };
  },
};

function seeded(spec: RunSpec, seed: number, argvOverride?: string[]): { argv: string[]; env: Record<string, string> } {
  let argv = [...(argvOverride ?? spec.argv)];
  const env: Record<string, string> = { ...(spec.env ?? {}) };
  if (spec.seedArg) { const i = argv.indexOf(spec.seedArg); if (i >= 0 && i + 1 < argv.length) argv[i + 1] = String(seed); else argv = [...argv, spec.seedArg, String(seed)]; }
  if (spec.seedEnv) env[spec.seedEnv] = String(seed);
  return { argv, env };
}

// Every line the claim's regex matches, with the captured number. Pure.
export function metricMatches(parse: string, text: string): { line: string; raw: number; masked: string }[] {
  const out: { line: string; raw: number; masked: string }[] = [];
  for (const m of text.matchAll(new RegExp(parse, "g"))) {
    const raw = Number(m[1]);
    if (!Number.isFinite(raw)) continue;
    const at = m.index ?? 0, start = text.lastIndexOf("\n", at) + 1, end = text.indexOf("\n", at + m[0].length);
    // The captured text itself, at its own position, is what gets masked (not String(raw): "0.9000" is 0.9).
    const cap = at + m[0].indexOf(m[1]!), line = text.slice(start, end < 0 ? undefined : end);
    const masked = (text.slice(start, cap) + "<value>" + text.slice(cap + m[1]!.length, end < 0 ? undefined : end)).trim();
    out.push({ line: line.trim(), raw, masked });
  }
  return out;
}

const toFraction = (raw: number, spec: { scale?: number }, metric: string) => (spec.scale ? raw * spec.scale : raw > 1 && raw <= 100 && /acc|f1|auc|precision|recall/i.test(metric) ? raw / 100 : raw);

// The claim's number, or undefined when the output has none or the regex matches different values.
// Never "the last match": on DETR's COCO summary that silently read AR@100 (0.574) for AP (0.420).
// Matches whose lines read the same apart from the value are one metric over time (per-epoch logs):
// the final value counts. Matches with different labels are different metrics: ambiguous.
export function metricSeries(ms: { masked: string; raw: number }[]): { label: string; last: number }[] {
  const by = new Map<string, number>();
  for (const m of ms) { by.delete(m.masked); by.set(m.masked, m.raw); }
  return [...by].map(([label, last]) => ({ label, last }));
}
export function parseMetric(spec: RunSpec, metric: string, text: string): number | undefined {
  const series = metricSeries(metricMatches(spec.parse, text));
  return series.length === 1 ? toFraction(series[0]!.last, spec, metric) : undefined;
}

// Several different values matched: the model picks the line that reports the claim's metric by its
// label only. Values are masked, so the choice cannot lean toward the paper's number. Journaled.
export async function disambiguate(model: { json<T>(system: string, user: string, maxTokens?: number): Promise<T> } | undefined, claim: Claim, parse: string, matches: { label: string }[]): Promise<{ index?: number; why: string; options: string[] }> {
  const options = matches.map((m) => m.label);
  if (!model) return { why: "no model to disambiguate", options };
  const r = await model.json<{ index?: number; why?: string }>(
    "You map an experiment's output line to a paper's claim. Several lines matched the claim's pattern; their values are hidden. Answer with JSON {\"index\": <0-based line index>, \"why\": \"...\"}, or {\"index\": null, \"why\": \"...\"} if no line reports exactly the claimed metric. Decide by the line's label only.",
    `Claim: ${JSON.stringify({ id: claim.id, metric: claim.metric, dataset: claim.dataset, table: claim.table, method: claim.method })}\nPattern: ${parse}\nLines:\n${options.map((o, i) => `${i}: ${o}`).join("\n")}`, 2000);
  // Only an actual number counts: Number(null) is 0, which would silently pick the first line.
  const i = typeof r.index === "number" ? r.index : NaN;
  return Number.isInteger(i) && i >= 0 && i < matches.length ? { index: i, why: String(r.why ?? ""), options } : { why: String(r.why ?? "no line chosen"), options };
}

const run: Skill = {
  id: N.run, version: "1.0.0", purpose: "detect", intents: ["reproduce"], actions: ["exec_sandboxed"], after: [], needs: () => null,
  summary: "Run one planned command for one claim and seed in a fresh sandbox of the current source version, and parse the metric.",
  output: z.object({ claim: z.string(), seed: z.number(), value: z.number(), version: z.number(), ms: z.number() }),
  async run(ctx) {
    const { claim, seed, version } = ctx.node.params as { claim: string; seed: number; version: number };
    const spec = specsOf(ctx.state).find((s) => s.claim === claim)!;
    const c = claimsOf(ctx.state).find((x) => x.id === claim)!;
    const { argv: overrides } = currentVersion(ctx.state);
    const { argv, env } = seeded(spec, seed, overrides[claim]);
    const site = currentSite(ctx.state);
    const sb = Sandbox.create(join(ctx.workDir(), ".."), `${ctx.node.id}`, need(ctx).versionDir(version), { isolation: ctx.isolation, limits: { cpuSec: spec.timeoutS * 2, memMB: 8192, fileMB: 2048, timeoutMs: spec.timeoutS * 1000 }, stats: ctx.stats, ...(site ? { site } : {}), ...(ctx.sandboxFault ? { fault: ctx.sandboxFault } : {}) });
    try {
      const cmd = spec.cwd ? ["sh", "-c", 'cd "$1" && shift && exec "$@"', "sh", spec.cwd, ...argv] : argv;
      const r = await sb.run(cmd, env);
      const log = join(sb.dir, ".mlh_run.log");
      writeFileSync(log, `$ ${cmd.join(" ")}\n--- stdout\n${r.stdout.slice(-20000)}\n--- stderr\n${r.stderr.slice(-20000)}`);
      ctx.save("log", log);
      const value = parseMetric(spec, c.metric, `${r.stdout}\n${r.stderr}`);
      // A program that prints its result and then keeps running is cut at the timeout; the printed result stands.
      if (r.category && !(r.category === "timeout" && value !== undefined)) throw new SkillError(r.category, `${argv.join(" ")} → ${r.category}: ${r.stderr.replace(/^MLH_EXIT=\d+$/m, "").slice(-1500)}`);
      if (value === undefined) throw new SkillError("experiment_error", `metric not found with /${spec.parse}/ in the output; tail:\n${r.stdout.slice(-800)}`);
      return { output: { claim, seed, value, version, ms: Math.round(r.ms), ...(r.category === "timeout" ? { cutAtTimeout: true } : {}) } };
    } finally { sb.remove(); }
  },
};

// The status rule (bench/repro/SPEC.md), identical for every system.
export function statusOf(c: Claim, values: number[]): { status: Status; mean: number; std: number; tol: number } {
  const n = values.length, mean = values.reduce((a, b) => a + b, 0) / n;
  const std = n > 1 ? Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)) : 0;
  const tol = Math.max(0.01, 2 * (c.std ?? 0), n > 1 ? 2 * std / Math.sqrt(n) : 0);
  return { status: Math.abs(mean - c.value) <= tol ? "reproduced" : "not_reproduced", mean, std, tol };
}

const compare: Skill = {
  id: N.compare, version: "1.0.0", purpose: "detect", intents: ["reproduce"], actions: [], after: [], needs: () => null,
  summary: "Compare the runs' mean with the claim under the fixed status rule.",
  output: any,
  async run(ctx) {
    const claim = String(ctx.node.params.claim);
    const c = claimsOf(ctx.state).find((x) => x.id === claim)!;
    const values = ctx.node.deps.map((d) => { const o = ctx.state.nodes.get(d)!.output as { value?: number; values?: Record<string, number> }; return o.values?.[claim] ?? o.value!; }).filter((v) => typeof v === "number");
    if (!values.length) throw new SkillError("experiment_error", `no stage output carries a value for ${claim}`);
    const s = statusOf(c, values);
    return { output: { claim, ...s, values, value: c.value } };
  },
};

function failureText(s: RunState, id: string | undefined): string {
  const n: NodeState | undefined = id ? s.nodes.get(id) : undefined;
  return n?.failures.at(-1)?.message ?? "";
}

const debug: Skill = {
  id: N.debug, version: "1.0.0", purpose: "setup", intents: ["reproduce"], actions: ["exec_sandboxed", "model_query"], after: [], needs: () => null,
  summary: "The debugger agent diagnoses a failed or mismatching claim in its own copy; its edits become a proposed patch.",
  output: any,
  async run(ctx) {
    const { claim, version, why, failedRun, stage: stageName } = ctx.node.params as { claim: string; version: number; why: string; failedRun?: string; stage?: string };
    const gp = gpuPlanOf(ctx.state);
    const c = claimsOf(ctx.state).find((x) => x.id === claim) ?? { id: claim, metric: "", value: NaN };
    const spec: unknown = gp ? (stageName ? gp.stages.find((x) => x.id === stageName) : gp) : specsOf(ctx.state).find((x) => x.claim === claim)!;
    const base = need(ctx).versionDir(version);
    const site = currentSite(ctx.state);
    const evidence = why === "failure" ? `The run failed:\n${failureText(ctx.state, failedRun).slice(0, 3000)}` : `The runs gave ${JSON.stringify(outOf<{ values: number[]; mean: number }>(ctx.state, String(ctx.node.params.trigger)))} but the paper claims ${c.value}.`;
    const prior = ctx.state.order.filter((id) => id.startsWith(`${N.debug}:${claim}:`) && id !== ctx.node.id && ctx.state.nodes.get(id)!.status === "succeeded").map((id) => outOf<{ causes: unknown; patch_intent?: string }>(ctx.state, id));
    const r = await agentLoop({ ...loopBase(ctx), ...remoteFor(ctx, pipOf(ctx.state), imageOf(ctx.state)), role: DEBUG_ROLE, workDir: ctx.workDir(), source: base, ...(site ? { site } : {}), task: (need(ctx).gpu ? `Budget left for this paper: $${need(ctx).gpu!.budget.remaining().toFixed(2)}. ` : "") + `Objective: ${ctx.state.objective}\nClaim: ${JSON.stringify(c)}\nPlanned run: ${JSON.stringify(spec)}\n${evidence}\nEnvironment notes: ${JSON.stringify(outOf<Record<string, unknown>>(ctx.state, N.env)?.launch ?? {})}\nEarlier diagnoses for this claim: ${JSON.stringify(prior)}` });
    if (!r.report) throw new SkillError("experiment_error", "debugger ran out of steps without a report");
    const changed = changedFiles(base, r.workspace);
    const diff = changed.length ? await unifiedDiff(base, r.workspace, changed) : "";
    const argv = Array.isArray(r.report.run_argv) && r.report.run_argv.length ? (r.report.run_argv as unknown[]).map(String) : undefined;
    if (diff) { const f = join(ctx.workDir(), "patch.diff"); writeFileSync(f, diff); ctx.save("patch.diff", f); }
    const pipChanged = need(ctx).gpu && (JSON.stringify(r.pip) !== JSON.stringify(pipOf(ctx.state)) || (r.image ?? DEFAULT_IMAGE_KEY) !== imageOf(ctx.state));
    return { output: { claim, causes: r.report.causes ?? [], patch_intent: r.report.patch_intent ?? "", expected_effect: r.report.expected_effect ?? "", changed, diff: diff.slice(0, 30000), ...(argv ? { run_argv: argv } : {}), hasPatch: changed.length > 0 || !!argv || !!pipChanged, workspace: r.workspace, ...(r.site ? { site: r.site } : {}), ...(pipChanged ? { pip: r.pip, image: r.image ?? DEFAULT_IMAGE_KEY } : {}), steps: r.steps } };
  },
};

const patch: Skill = {
  id: N.patch, version: "1.0.0", purpose: "setup", intents: ["reproduce"], actions: [], after: [], needs: () => null,
  summary: "Grade a proposed change's risk and authorize it: medium-risk changes need passing static checks; changes to measurement code need a human (or the run's standing approval policy).",
  output: any,
  async run(ctx) {
    const { claim, from, debug: dbg } = ctx.node.params as { claim: string; from: number; debug: string };
    const d = outOf<{ diff: string; changed: string[]; run_argv?: string[]; workspace: string; site?: string; pip?: string[]; image?: string }>(ctx.state, dbg);
    const gp = gpuPlanOf(ctx.state);
    const spec = gp ? { argv: gp.stages.find((x) => `stage-${x.id}` === claim)?.argv ?? [] } : specsOf(ctx.state).find((x) => x.claim === claim)!;
    const files = parseDiff(d.diff, need(ctx).versionDir(from), d.workspace);
    const argvChange = d.run_argv ? [`${spec.argv.join(" ")} -> ${d.run_argv.join(" ")}`] : [];
    const { risk, reasons } = riskOf(files, claimsOf(ctx.state), argvChange);
    const hardcode = reasons.some((r) => r.includes("adds the claimed value"));
    const evidence: Evidence[] = [{ kind: "deterministic", check: hardcode ? "patch writes a claimed number into the code" : "static checks: no claimed number written into code", ok: !hardcode }];
    if (ctx.approved) evidence.push({ kind: "human", check: `approval:${ctx.node.id}`, ok: true });
    const policy = need(ctx).approvePolicy;
    if (policy === "all") evidence.push({ kind: "human", check: "standing approval policy: all", ok: true });
    const decision = authorize(risk === "high" ? "apply_eval_patch" : "apply_patch", policy === "none" && !ctx.approved ? evidence.filter((e) => e.kind !== "deterministic") : evidence);
    if (!decision.allow) return decision.hold ? { output: { accepted: false, risk, reasons }, hold: `${risk}-risk patch for ${claim}: ${decision.reason}; ${reasons.join("; ")}` } : { output: { accepted: false, risk, reasons, refused: decision.reason } };
    let version: number | undefined;
    if (d.changed.length) {
      version = currentVersion(ctx.state).version + 1;
      const dest = need(ctx).versionDir(version);
      rmSync(dest, { recursive: true, force: true });
      buildVersion(need(ctx).versionDir(currentVersion(ctx.state).version), d.workspace, d.changed, dest);
    }
    return { output: { accepted: true, risk, reasons, basis: decision.basis, ...(version !== undefined ? { version } : {}), ...(d.run_argv ? { argv: { [claim]: d.run_argv } } : {}), ...(d.site ? { site: d.site } : {}), ...(d.pip ? { pip: d.pip } : {}), ...(d.image ? { image: d.image } : {}) } };
  },
};

// One stage of a GPU plan, in a fresh remote container: restore the outputs of the stages it needs,
// run, persist its own outputs, and parse every claim this stage measures.
const stage: Skill = {
  id: "repro.stage", version: "1.0.0", purpose: "detect", intents: ["reproduce"], actions: ["exec_sandboxed", "model_query"], after: [], needs: () => null,
  summary: "Run one stage of the reproduction plan on a remote GPU (or CPU) container, charged to the paper's budget.",
  output: any,
  async run(ctx) {
    const g = need(ctx).gpu;
    if (!g) throw new SkillError("harness_bug", "stage nodes need the remote backend");
    const p = gpuPlanOf(ctx.state)!;
    const { stage: sid, version, tag, seed } = ctx.node.params as { stage: string; version: number; tag: string; seed?: number };
    const st = p.stages.find((x) => x.id === sid)!;
    const { argv: overrides } = currentVersion(ctx.state);
    let argv = [...(overrides[`stage-${sid}`] ?? st.argv)];
    const env: Record<string, string> = { ...(st.env ?? {}) };
    if (seed !== undefined) {
      if (p.seedArg) { const i = argv.indexOf(p.seedArg); if (i >= 0 && i + 1 < argv.length) argv[i + 1] = String(seed); else argv = [...argv, p.seedArg, String(seed)]; }
      if (p.seedEnv) env[p.seedEnv] = String(seed);
    }
    const area = (s: Stage, sd?: number) => `/cache/runs/${g.runKey}/${tag}/${s.id}${sd !== undefined ? `-s${sd}` : ""}`;
    const restore = st.needs.flatMap((n) => {
      const dep = p.stages.find((x) => x.id === n)!;
      return dep.persist.map((path) => ({ from: `${area(dep, dep.perSeed ? seed : undefined)}/${path}`, to: path }));
    });
    const persist = st.persist.map((path) => ({ from: path, to: `${area(st, seed)}/${path}` }));
    const cmd = st.cwd ? ["bash", "-c", 'cd "$1" && shift && exec "$@"', "bash", st.cwd, ...argv] : argv;
    const r = await modalRun({ repo: need(ctx).versionDir(version), cmd, timeoutS: st.timeoutS, network: !!st.network, ...(st.network ? { domains: DATA_DOMAINS } : {}), env, pip: pipOf(ctx.state), image: IMAGES[imageOf(ctx.state)]!, restore, persist, ...(st.gpu ? { gpu: st.gpu } : {}) }, g.budget);
    const log = join(ctx.workDir(), `${ctx.node.id.replace(/[^\w.-]+/g, "_")}.log`);
    mkdirSync(ctx.workDir(), { recursive: true });
    writeFileSync(log, `$ ${cmd.join(" ")}\n--- stdout\n${r.stdout.slice(-30000)}\n--- stderr\n${r.stderr.slice(-20000)}`);
    ctx.save("log", log);
    const text = `${r.stdout}\n${r.stderr}`;
    const values: Record<string, number> = {};
    const disambiguated: Record<string, unknown> = {};
    for (const c of p.claims.filter((x) => x.stage === sid)) {
      const claim = claimsOf(ctx.state).find((x) => x.id === c.claim)!;
      const series = metricSeries(metricMatches(c.parse, text));
      if (series.length > 1) {
        const d = await disambiguate(ctx.llm?.reviewer, claim, c.parse, series);
        disambiguated[c.claim] = d;
        if (d.index === undefined) throw new SkillError("contract_violation", `ambiguous metric for ${c.claim}: the pattern matched ${series.length} differently labelled lines and none could be chosen by label (${d.why}); lines: ${d.options.join(" || ")}`);
        values[c.claim] = toFraction(series[d.index]!.last, c, claim.metric);
        continue;
      }
      const v = parseMetric({ claim: c.claim, argv, seeds: [], parse: c.parse, timeoutS: st.timeoutS, ...(c.scale ? { scale: c.scale } : {}) }, claim.metric, text);
      if (v !== undefined) values[c.claim] = v;
    }
    if (r.category && !(r.category === "timeout" && Object.keys(values).length)) throw new SkillError(r.category, `${argv.join(" ")} → ${r.category} (exit ${r.exitCode}): ${r.stderr.slice(-1500)}`);
    const missing = p.claims.filter((x) => x.stage === sid && values[x.claim] === undefined).map((x) => x.claim);
    if (missing.length) throw new SkillError("experiment_error", `stage ${sid} finished but its output has no value for ${missing.join(", ")}; tail:\n${r.stdout.slice(-800)}`);
    return { output: { stage: sid, seed, values, ...(Object.keys(disambiguated).length ? { disambiguated } : {}), version, elapsedS: Math.round(r.elapsedS), costUsd: Number(r.costUsd.toFixed(4)), spentUsd: Number(g.budget.spent.toFixed(4)) } };
  },
};

// Download declared items into this run's /cache/data with fetch_data.py (idempotent, sha256 recorded).
async function harnessFetch(ctx: SkillCtx, items: FetchItem[]): Promise<unknown[]> {
  const g = need(ctx).gpu;
  if (!g || !items.length) return [];
  const dir = join(ctx.workDir(), "env-fetcher");
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, "fetch_data.py"), readFileSync(join(import.meta.dirname, "../backends/fetch_data.py")));
  const r = await modalRun({ repo: dir, cmd: ["python3", "fetch_data.py", JSON.stringify({ items, checks: [] })], timeoutS: 3600, network: true }, g.budget);
  if (r.category) throw new SkillError(r.category === "experiment_error" ? "experiment_error" : r.category, `fetch failed: ${r.stderr.slice(-800)}`);
  return (JSON.parse(r.stdout.trim().split("\n").pop() ?? "{}") as { items?: unknown[] }).items ?? [];
}

// Data acquisition by harness code: the declared fetch items, then the completeness checks.
const fetchData: Skill = {
  id: "repro.fetch", version: "1.0.0", purpose: "setup", intents: ["reproduce"], actions: [], after: [], needs: () => null,
  summary: "Download the plan's declared datasets/checkpoints into the cache (idempotent, sha256 recorded) and check completeness (e.g. the evaluation split has all its files).",
  output: any,
  async run(ctx) {
    const g = need(ctx).gpu;
    if (!g) throw new SkillError("harness_bug", "fetch needs the remote backend");
    const p = gpuPlanOf(ctx.state)!;
    const evidence: Evidence[] = [{ kind: "deterministic", check: p.unknownHosts.length ? `unknown data hosts: ${p.unknownHosts.join(", ")}` : "all fetch URLs are https on known data hosts", ok: true }];
    if (ctx.approved) evidence.push({ kind: "human", check: `approval:${ctx.node.id}`, ok: true });
    if (need(ctx).approvePolicy === "all") evidence.push({ kind: "human", check: "standing approval policy: all", ok: true });
    const d = authorize(p.unknownHosts.length ? "fetch_data_unknown_host" : "fetch_data", evidence);
    if (!d.allow) return d.hold ? { output: { fetched: [] }, hold: `fetch from ${p.unknownHosts.join(", ")}: ${d.reason}` } : { output: { refused: d.reason } };
    const dir = join(ctx.workDir(), "fetcher");
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, "fetch_data.py"), readFileSync(join(import.meta.dirname, "../backends/fetch_data.py")));
    const timeoutS = 3 * 3600;
    const r = await modalRun({ repo: dir, cmd: ["python3", "fetch_data.py", JSON.stringify({ items: p.fetch, checks: p.checks })], timeoutS, network: true }, g.budget);
    // A transient network fault is retried as infrastructure; a wrong declaration (404, certificate,
    // checksum mismatch) is the planner's and is not retried.
    if (r.category) {
      const transient = r.category === "experiment_error" && /HTTP Error 5\d\d|timed out|Connection reset|Temporary failure in name resolution|RemoteDisconnected|IncompleteRead/.test(r.stderr);
      throw new SkillError(transient ? "sandbox_transient" : r.category, `fetch failed: ${r.stderr.slice(-1500)}`);
    }
    const out = JSON.parse(r.stdout.trim().split("\n").pop() ?? "{}") as { items: unknown[]; checks: { path: string; files: number; minFiles: number; ok: boolean }[] };
    const manifest = join(dir, "data-manifest.json");
    writeFileSync(manifest, JSON.stringify(out, null, 1));
    ctx.save("data-manifest.json", manifest);
    const bad = out.checks.filter((c) => !c.ok);
    if (bad.length) throw new SkillError("experiment_error", `data incomplete: ${bad.map((c) => `${c.path} has ${c.files} files, expected at least ${c.minFiles}`).join("; ")}`);
    return { output: { ...out, basis: d.basis } };
  },
};

export const REPRO_SKILLS: Skill[] = [paper, claims, env, plan, run, compare, debug, patch, stage, fetchData];
