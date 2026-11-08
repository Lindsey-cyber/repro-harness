// Node implementations of the reproduction flow. Agents explore and propose;
// everything that decides (claim validity, plan validity, run results, claim
// status, patch risk) is harness code.
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
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

export interface ReproCtx {
  arxiv: string;
  sites: SiteManager;
  versionDir(v: number): string;
  approvePolicy: "none" | "medium" | "all";
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

const env: Skill = {
  id: N.env, version: "1.0.0", purpose: "setup", intents: ["reproduce"], actions: ["exec_sandboxed", "model_query"], after: [], needs: () => null,
  summary: "The environment agent installs what the repository needs and finds how to launch it; it may not edit code.",
  output: any,
  async run(ctx) {
    const r = await agentLoop({ ...loopBase(ctx), role: ENV_ROLE, workDir: ctx.workDir(), source: need(ctx).versionDir(0), task: `Objective of the team: ${ctx.state.objective}\nMake this repository's experiments runnable.` });
    if (!r.report) throw new SkillError("experiment_error", "environment agent ran out of steps without a report");
    return { output: { ...r.report, ...(r.site ? { site: r.site, specs: need(ctx).sites.specsOf(r.site) } : {}), steps: r.steps } };
  },
};

const plan: Skill = {
  id: N.plan, version: "1.0.0", purpose: "setup", intents: ["reproduce"], actions: ["exec_sandboxed", "model_query"], after: [], needs: () => null,
  summary: "The planner agent maps each claim to a run spec; the harness validates it (command, one-group parse regex, ≥3 seeds).",
  output: z.object({ specs: z.array(any), errors: z.array(z.string()) }),
  async run(ctx) {
    const cs = claimsOf(ctx.state);
    const e = outOf<Record<string, unknown>>(ctx.state, N.env);
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

export function parseMetric(spec: RunSpec, metric: string, text: string): number | undefined {
  const re = new RegExp(spec.parse, "g");
  const all = [...text.matchAll(re)];
  const raw = all.length ? Number(all[all.length - 1]![1]) : NaN;
  if (!Number.isFinite(raw)) return undefined;
  if (spec.scale) return raw * spec.scale;
  return raw > 1 && raw <= 100 && /acc|f1|auc|precision|recall/i.test(metric) ? raw / 100 : raw;
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
    const values = ctx.node.deps.map((d) => (ctx.state.nodes.get(d)!.output as { value: number }).value);
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
    const { claim, version, why, failedRun } = ctx.node.params as { claim: string; version: number; why: string; failedRun?: string };
    const c = claimsOf(ctx.state).find((x) => x.id === claim)!;
    const spec = specsOf(ctx.state).find((x) => x.claim === claim)!;
    const base = need(ctx).versionDir(version);
    const site = currentSite(ctx.state);
    const evidence = why === "failure" ? `The run failed:\n${failureText(ctx.state, failedRun).slice(0, 3000)}` : `The runs gave ${JSON.stringify(outOf<{ values: number[]; mean: number }>(ctx.state, String(ctx.node.params.trigger)))} but the paper claims ${c.value}.`;
    const prior = ctx.state.order.filter((id) => id.startsWith(`${N.debug}:${claim}:`) && id !== ctx.node.id && ctx.state.nodes.get(id)!.status === "succeeded").map((id) => outOf<{ causes: unknown; patch_intent?: string }>(ctx.state, id));
    const r = await agentLoop({ ...loopBase(ctx), role: DEBUG_ROLE, workDir: ctx.workDir(), source: base, ...(site ? { site } : {}), task: `Objective: ${ctx.state.objective}\nClaim: ${JSON.stringify(c)}\nPlanned run: ${JSON.stringify(spec)}\n${evidence}\nEnvironment notes: ${JSON.stringify(outOf<Record<string, unknown>>(ctx.state, N.env)?.launch ?? {})}\nEarlier diagnoses for this claim: ${JSON.stringify(prior)}` });
    if (!r.report) throw new SkillError("experiment_error", "debugger ran out of steps without a report");
    const changed = changedFiles(base, r.workspace);
    const diff = changed.length ? await unifiedDiff(base, r.workspace, changed) : "";
    const argv = Array.isArray(r.report.run_argv) && r.report.run_argv.length ? (r.report.run_argv as unknown[]).map(String) : undefined;
    if (diff) { const f = join(ctx.workDir(), "patch.diff"); writeFileSync(f, diff); ctx.save("patch.diff", f); }
    return { output: { claim, causes: r.report.causes ?? [], patch_intent: r.report.patch_intent ?? "", expected_effect: r.report.expected_effect ?? "", changed, diff: diff.slice(0, 30000), ...(argv ? { run_argv: argv } : {}), hasPatch: changed.length > 0 || !!argv, workspace: r.workspace, ...(r.site ? { site: r.site } : {}), steps: r.steps } };
  },
};

const patch: Skill = {
  id: N.patch, version: "1.0.0", purpose: "setup", intents: ["reproduce"], actions: [], after: [], needs: () => null,
  summary: "Grade a proposed change's risk and authorize it: medium-risk changes need passing static checks; changes to measurement code need a human (or the run's standing approval policy).",
  output: any,
  async run(ctx) {
    const { claim, from, debug: dbg } = ctx.node.params as { claim: string; from: number; debug: string };
    const d = outOf<{ diff: string; changed: string[]; run_argv?: string[]; workspace: string; site?: string }>(ctx.state, dbg);
    const spec = specsOf(ctx.state).find((x) => x.claim === claim)!;
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
    return { output: { accepted: true, risk, reasons, basis: decision.basis, ...(version !== undefined ? { version } : {}), ...(d.run_argv ? { argv: { [claim]: d.run_argv } } : {}), ...(d.site ? { site: d.site } : {}) } };
  },
};

export const REPRO_SKILLS: Skill[] = [paper, claims, env, plan, run, compare, debug, patch];
