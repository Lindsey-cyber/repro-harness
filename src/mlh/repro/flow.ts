// The reproduction flow: which nodes follow from which results. Pure. The DAG
// grows from observations: claims extracted from the paper, a plan the planner
// agent proposes (validated here), runs that fail or miss, patches that are
// accepted. Nothing here knows a benchmark task.
import type { RunState, TaskNode } from "../core/dag.ts";
import { gpuPlanOf, stageNodes } from "./gpu.ts";

export const N = { paper: "repro.paper", claims: "repro.claims", env: "repro.env", plan: "repro.plan", run: "repro.run", compare: "repro.compare", debug: "repro.debug", patch: "repro.patch" } as const;
export const MAX_DEBUG_ROUNDS = 2; // per claim
export const MIN_SEEDS = 3;

export interface Claim { id: string; table?: string; dataset?: string; method?: string; metric: string; value: number; std?: number | null; quote?: string }
export interface RunSpec { claim: string; argv: string[]; cwd?: string; env?: Record<string, string>; seeds: number[]; seedArg?: string; seedEnv?: string; parse: string; timeoutS: number; scale?: number }
export interface PatchOut { accepted: boolean; version?: number; risk?: "medium" | "high"; argv?: Record<string, string[]> }
export type Status = "reproduced" | "not_reproduced" | "not_runnable";

export function initialNodes(): TaskNode[] {
  return [
    { id: N.paper, skill: N.paper, deps: [], purpose: "setup", params: {} },
    { id: N.env, skill: N.env, deps: [], purpose: "setup", params: {} },
  ];
}

// Validates the planner's output: every claim gets a run spec with a command, a
// parse regex with exactly one group, and at least MIN_SEEDS seeds.
export function validatePlan(claims: Claim[], raw: unknown): { specs: RunSpec[]; errors: string[] } {
  const errors: string[] = [], specs: RunSpec[] = [];
  const runs = (raw as { runs?: unknown[] })?.runs;
  if (!Array.isArray(runs)) return { specs, errors: ["plan has no runs list"] };
  for (const c of claims) {
    const r = runs.find((x) => (x as { claim?: string })?.claim === c.id) as Partial<RunSpec> | undefined;
    if (!r) { errors.push(`no run for claim ${c.id}`); continue; }
    if (!Array.isArray(r.argv) || !r.argv.length || !r.argv.every((a) => typeof a === "string")) { errors.push(`${c.id}: argv must be a non-empty list of strings`); continue; }
    // A run launches the repository's own entrypoint. Inline programs would change code without patch review.
    if (r.argv.some((a, i) => (a === "-c" && i > 0 && /^(python3?|sh|bash)$/.test(r.argv![i - 1]!)) ) || /^(sh|bash)$/.test(r.argv[0]!) && r.argv.includes("-c")) { errors.push(`${c.id}: inline programs (python -c, sh -c) are not allowed; run a script of the repository, and leave code changes to the debugger`); continue; }
    let groups = -1;
    try { groups = new RegExp(`${String(r.parse)}|`).exec("")!.length - 1; } catch { /* invalid */ }
    if (typeof r.parse !== "string" || groups !== 1) { errors.push(`${c.id}: parse must be a regex with exactly one capture group`); continue; }
    const seeds = Array.isArray(r.seeds) ? r.seeds.filter((s) => Number.isInteger(s)).slice(0, 10) : [];
    for (let s = 0; seeds.length < MIN_SEEDS; s++) if (!seeds.includes(s)) seeds.push(s);
    specs.push({
      claim: c.id, argv: r.argv, seeds, parse: r.parse,
      timeoutS: Math.min(Math.max(Number(r.timeoutS) || 600, 30), 900),
      ...(typeof r.cwd === "string" && !r.cwd.startsWith("/") && !r.cwd.includes("..") ? { cwd: r.cwd } : {}),
      ...(r.env && typeof r.env === "object" ? { env: Object.fromEntries(Object.entries(r.env).filter(([k, v]) => /^[A-Z_][A-Z0-9_]*$/.test(k) && typeof v === "string" && !/^(PATH|HOME|LD_|PYTHONHOME)/.test(k))) as Record<string, string> } : {}),
      ...(typeof r.seedArg === "string" && /^--?[\w-]+$/.test(r.seedArg) ? { seedArg: r.seedArg } : {}),
      ...(typeof r.seedEnv === "string" && /^[A-Z_][A-Z0-9_]*$/.test(r.seedEnv) ? { seedEnv: r.seedEnv } : {}),
      ...(typeof r.scale === "number" && r.scale > 0 ? { scale: r.scale } : {}),
    });
  }
  return { specs, errors };
}

const out = <T>(s: RunState, id: string) => s.nodes.get(id)?.output as T | undefined;
const ok = (s: RunState, id: string) => s.nodes.get(id)?.status === "succeeded";

export function claimsOf(s: RunState): Claim[] { return out<{ claims: Claim[] }>(s, N.claims)?.claims ?? []; }
export function specsOf(s: RunState): RunSpec[] { return out<{ specs: RunSpec[] }>(s, N.plan)?.specs ?? []; }

// The newest source version, and the argv overrides accepted so far.
export function currentVersion(s: RunState): { version: number; argv: Record<string, string[]> } {
  let version = 0; const argv: Record<string, string[]> = {};
  for (const id of s.order) {
    if (!id.startsWith(N.patch) || !ok(s, id)) continue;
    const p = out<PatchOut>(s, id)!;
    if (p.accepted) { if (p.version !== undefined) version = Math.max(version, p.version); Object.assign(argv, p.argv ?? {}); }
  }
  return { version, argv };
}

export function runNodes(s: RunState, claims: string[], tag: string): TaskNode[] {
  const { version } = currentVersion(s);
  const nodes: TaskNode[] = [];
  for (const spec of specsOf(s).filter((x) => claims.includes(x.claim))) {
    const runIds = spec.seeds.map((seed) => `${N.run}:${spec.claim}:v${version}${tag}:s${seed}`);
    spec.seeds.forEach((seed, i) => nodes.push({ id: runIds[i]!, skill: N.run, deps: [N.plan], purpose: "detect", params: { claim: spec.claim, seed, version } }));
    nodes.push({ id: `${N.compare}:${spec.claim}:v${version}${tag}`, skill: N.compare, deps: runIds, purpose: "detect", params: { claim: spec.claim, version } });
  }
  return nodes.filter((n) => !s.nodes.has(n.id));
}

const debugRounds = (s: RunState, claim: string) => s.order.filter((id) => id.startsWith(`${N.debug}:${claim}:`)).length;

function debugNode(s: RunState, claim: string, trigger: string, why: "failure" | "mismatch"): TaskNode[] {
  if (debugRounds(s, claim) >= MAX_DEBUG_ROUNDS) return [];
  const { version } = currentVersion(s);
  const id = `${N.debug}:${claim}:v${version}`;
  if (s.nodes.has(id)) return [];
  return [{ id, skill: N.debug, deps: [N.plan, trigger], purpose: "setup", params: { claim, version, why, trigger } }];
}

// New nodes after a node succeeded.
export function afterOk(s: RunState, id: string): TaskNode[] {
  const n = s.nodes.get(id)!.node;
  if (id === N.paper) return [{ id: N.claims, skill: N.claims, deps: [N.paper], purpose: "setup", params: {} }];
  if ((id === N.claims || id === N.env) && ok(s, N.claims) && ok(s, N.env) && !s.nodes.has(N.plan)) return [{ id: N.plan, skill: N.plan, deps: [N.claims, N.env], purpose: "setup", params: {} }];
  if (id === N.plan) return gpuPlanOf(s) ? stageNodes(s, "") : runNodes(s, specsOf(s).map((x) => x.claim), "");
  if (n.skill === N.compare) {
    const st = (s.nodes.get(id)!.output as { status: Status }).status;
    return st === "not_reproduced" ? debugNode(s, String(n.params.claim), id, "mismatch") : [];
  }
  if (n.skill === N.debug) {
    const d = s.nodes.get(id)!.output as { hasPatch?: boolean };
    return d.hasPatch ? [{ id: `${N.patch}:${n.params.claim}:v${n.params.version}`, skill: N.patch, deps: [id], purpose: "setup", params: { claim: n.params.claim, from: n.params.version, debug: id } }] : [];
  }
  if (n.skill === N.patch) {
    const p = s.nodes.get(id)!.output as PatchOut;
    // An accepted change (new source version or new command) means every claim is measured again.
    // Runs are tagged by the patch count, so a command-only change does not collide with earlier runs.
    const k = s.order.filter((x) => x.startsWith(N.patch) && (s.nodes.get(x)!.output as PatchOut | undefined)?.accepted).length;
    if (!p.accepted) return [];
    const gp = gpuPlanOf(s);
    if (!gp) return runNodes(s, specsOf(s).map((x) => x.claim), `.p${k}`);
    const all = stageNodes(s, `.p${k}`);
    // Source, packages, image or site changed: everything is measured again. Only the failed stage's
    // command changed (or nothing): only that stage, what it needs, and its claims are re-run.
    // GraphSAINT r1: a one-stage command change (--num_cpu_core 4 -> 1) re-ran all 9 stages (~40 min).
    const q = p as PatchOut & { pip?: unknown; image?: unknown; site?: unknown };
    if (q.version !== undefined || q.pip || q.image || q.site) return all;
    const target = String(n.params.claim).replace(/^stage-/, "");
    const need = new Set<string>([target]);
    for (let grew = true; grew;) { grew = false; for (const st of gp.stages) if (need.has(st.id)) for (const d of st.needs) if (!need.has(d)) { need.add(d); grew = true; } }
    const claimsOfTarget = new Set(gp.claims.filter((c) => c.stage === target).map((c) => c.claim));
    return all.filter((x) => (x.skill === "repro.stage" && need.has(String(x.params.stage))) || (x.skill === N.compare && claimsOfTarget.has(String(x.params.claim))) || x.skill === "repro.fetch");
  }
  return [];
}

// New nodes after a node failed for good.
export function afterFail(s: RunState, id: string): TaskNode[] {
  const n = s.nodes.get(id)!.node;
  if (n.skill === "repro.stage") {
    // A failed stage is debugged once per stage and source version (budget permitting, ≤ MAX_DEBUG_ROUNDS per stage).
    const stage = String(n.params.stage);
    const rounds = s.order.filter((x) => x.startsWith(`${N.debug}:stage-${stage}:`) && s.nodes.get(x)!.status !== "skipped");
    if (rounds.length >= MAX_DEBUG_ROUNDS) return [];
    let did = `${N.debug}:stage-${stage}:v${currentVersion(s).version}`;
    // A debug round skipped because its failure was the platform leaves the id taken: use the next one.
    for (let k = 2; s.nodes.get(did)?.status === "skipped"; k++) did = `${N.debug}:stage-${stage}:v${currentVersion(s).version}.r${k}`;
    if (s.nodes.has(did)) return [];
    return [{ id: did, skill: N.debug, deps: [N.plan], purpose: "setup", params: { claim: `stage-${stage}`, stage, version: currentVersion(s).version, why: "failure", failedRun: id } }];
  }
  if (n.skill === N.run) return debugNode(s, String(n.params.claim), N.plan, "failure").map((d) => ({ ...d, params: { ...d.params, failedRun: id } }));
  return [];
}

// Status per claim: the newest version's comparison; a claim whose runs never
// completed after the debug budget is not runnable.
export function finalStatus(s: RunState): Record<string, { status: Status | "pending"; version: number; mean?: number; values?: number[]; node?: string }> {
  const res: Record<string, { status: Status | "pending"; version: number; mean?: number; values?: number[]; node?: string }> = {};
  for (const c of claimsOf(s)) {
    const cmp = s.order.filter((id) => id.startsWith(`${N.compare}:${c.id}:`));
    // Newest version first; within a version, the latest comparison (after the latest accepted change).
    const done = cmp.filter((id) => ok(s, id)).map((id) => ({ id, v: Number(s.nodes.get(id)!.node.params.version), i: s.order.indexOf(id), o: out<{ status: Status; mean: number; values: number[] }>(s, id)! })).sort((a, b) => b.v - a.v || b.i - a.i);
    const pending = [...s.nodes.values()].some((x) => x.status === "held" || x.status === "pending" || x.status === "running");
    if (done[0]) res[c.id] = { status: done[0].o.status, version: done[0].v, mean: done[0].o.mean, values: done[0].o.values, node: done[0].id };
    else res[c.id] = { status: pending ? "pending" : "not_runnable", version: currentVersion(s).version };
  }
  return res;
}
