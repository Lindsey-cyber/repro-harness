// Multi-stage reproduction on remote GPUs. Pure. A complex paper is not one
// command: data must be fetched and prepared, models trained or checkpoints
// downloaded, then evaluated. The planner proposes a stage DAG and a fidelity
// level (full / scaled / eval_only) that fits the paper's budget; this module
// validates it (structure, GPU allowlist, worst-case cost) and turns it into
// task nodes. Stage outputs pass between stages through the persistent volume.
import type { RunState, TaskNode } from "../core/dag.ts";
import { claimsOf, currentVersion, N, type Claim } from "./flow.ts";

export const GPUS = ["T4", "L4", "A10G", "A100-40GB", "A100-80GB", "H100"];
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
  env?: Record<string, string>;
  cwd?: string;
}
export interface GpuPlan {
  fidelity: "full" | "scaled" | "eval_only";
  fidelityNote: string;
  stages: Stage[];
  claims: { claim: string; stage: string; parse: string; scale?: number }[];
  seeds: number[];
  seedArg?: string;
  seedEnv?: string;
  worstCaseUsd: number;
}

const INLINE = (argv: string[]) => argv.some((a, i) => a === "-c" && i > 0 && /^(python3?|sh|bash)$/.test(argv[i - 1]!)) || (/^(sh|bash)$/.test(argv[0] ?? "") && argv.includes("-c"));

export function validateGpuPlan(claims: Claim[], raw: unknown, o: { budgetUsd: number; price: (gpu: string) => number }): { plan?: GpuPlan; errors: string[] } {
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
    if (s.gpu !== undefined && !GPUS.includes(String(s.gpu))) { errors.push(`${id}: gpu ${s.gpu} not in ${GPUS.join(", ")}`); continue; }
    if (s.network && s.gpu) { errors.push(`${id}: a stage with network access may not use a GPU (fetch first, then compute offline)`); continue; }
    const timeoutS = Math.min(Math.max(Number(s.timeoutS) || 1800, 60), MAX_STAGE_S);
    const persist = (Array.isArray(s.persist) ? s.persist : []).map(String).filter((p) => !p.startsWith("/") && !p.includes(".."));
    seen.add(id);
    stages.push({ id, argv: s.argv, timeoutS, needs, persist, ...(s.gpu ? { gpu: String(s.gpu) } : {}), ...(s.network ? { network: true } : {}), ...(s.perSeed ? { perSeed: true } : {}),
      ...(s.env && typeof s.env === "object" ? { env: Object.fromEntries(Object.entries(s.env).filter(([k, v]) => /^[A-Z_][A-Z0-9_]*$/.test(k) && typeof v === "string" && !/^(PATH|HOME|LD_|PYTHONHOME)/.test(k))) as Record<string, string> } : {}),
      ...(typeof s.cwd === "string" && !s.cwd.startsWith("/") && !s.cwd.includes("..") ? { cwd: s.cwd } : {}) });
  }
  const cl: GpuPlan["claims"] = [];
  for (const c of claims) {
    const m = (r.claims ?? []).find((x) => x?.claim === c.id);
    if (!m) { errors.push(`no stage measures claim ${c.id}`); continue; }
    if (!stages.some((s) => s.id === m.stage)) { errors.push(`${c.id}: stage ${m.stage} does not exist`); continue; }
    let groups = -1;
    try { groups = new RegExp(`${String(m.parse)}|`).exec("")!.length - 1; } catch { /* invalid */ }
    if (groups !== 1) { errors.push(`${c.id}: parse must be a regex with exactly one capture group`); continue; }
    cl.push({ claim: c.id, stage: String(m.stage), parse: String(m.parse), ...(typeof m.scale === "number" && m.scale > 0 ? { scale: m.scale } : {}) });
  }
  const seeds = (Array.isArray(r.seeds) ? r.seeds : [0]).filter((x) => Number.isInteger(x)).slice(0, 5);
  if (!seeds.length) seeds.push(0);
  // Worst case: every stage instance runs to its timeout, plus container start-up.
  const worstCaseUsd = stages.reduce((a, s) => a + (s.gpu ? (o.price(s.gpu) * (s.timeoutS + 300)) / 3600 : (0.1 * (s.timeoutS + 300)) / 3600) * (s.perSeed ? seeds.length : 1), 0);
  if (worstCaseUsd > o.budgetUsd) errors.push(`worst-case cost $${worstCaseUsd.toFixed(2)} exceeds the budget $${o.budgetUsd.toFixed(2)}: lower fidelity, shorter timeouts, fewer seeds or a cheaper GPU`);
  if (errors.length || !fidelity) return { errors };
  return { plan: { fidelity, fidelityNote: String(r.fidelityNote ?? ""), stages, claims: cl, seeds, ...(typeof r.seedArg === "string" && /^--?[\w-]+$/.test(r.seedArg) ? { seedArg: r.seedArg } : {}), ...(typeof r.seedEnv === "string" && /^[A-Z_][A-Z0-9_]*$/.test(r.seedEnv) ? { seedEnv: r.seedEnv } : {}), worstCaseUsd }, errors: [] };
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
  for (const st of p.stages) {
    for (const seed of st.perSeed ? p.seeds : [undefined]) {
      const deps = [N.plan, ...st.needs.flatMap((n) => {
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
