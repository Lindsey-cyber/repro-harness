// The skill library: versioned, discoverable units with declared needs, actions
// and an output schema the orchestrator enforces. Each skill measures one thing
// with trusted probes or by running the experiment in a fresh sandbox; deciding
// what a measurement means stays in the declared thresholds below, fixed in
// advance and identical for every repository.
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { z } from "zod";
import type { NodeState, RunState, TaskNode } from "../core/dag.ts";
import type { SkillSpec } from "../core/planner.ts";
import type { ArtifactRef, Finding, FindingDraft, Manifest } from "../core/types.ts";
import type { Llm } from "../io/llm.ts";
import { SkillError, type Isolation, type Sandbox } from "../io/sandbox.ts";
import type { Evidence } from "../core/authz.ts";
import { VOCABULARY } from "../core/vocab.ts";
import { digest, validateManifest } from "./repo.ts";

export type ChatModel = Pick<Llm, "model" | "usage" | "chat" | "json">;
export interface Models { proposer: ChatModel; reviewer: ChatModel }

export interface SkillCtx {
  node: TaskNode;
  state: RunState;
  manifest: Manifest;
  target: Finding | undefined;
  dep(skill: string): NodeState;
  sandbox(label?: string): Sandbox;
  save(name: string, hostPath: string): ArtifactRef;
  load(ref: ArtifactRef): string;
  // Present only when the human enabled model use for this run (--llm).
  llm?: Models;
  // A persistent directory for this node, kept across attempts and resumes.
  workDir(): string;
  sourceDir: string;
  isolation: Isolation;
  isolationEvidence: Evidence;
  sandboxFault?: () => boolean;
  crashAfterSteps?: number;
}

export interface SkillOutcome {
  output: unknown;
  findings?: FindingDraft[];
  confirmation?: { supports: boolean; note: string; inconclusive?: boolean; tier?: "deterministic" | "measured" };
  // The skill ran but could not check what it is for (e.g. no predictions file): reported as a coverage gap.
  gap?: string;
}

export interface Skill extends SkillSpec {
  run(ctx: SkillCtx): Promise<SkillOutcome>;
}

// Thresholds, fixed before the benchmark was run (docs/mlh/ARCHITECTURE.md#thresholds).
export const T = {
  overlapFraction: 0.01, // share of test rows also in train
  leakAuc: 0.99, // single-feature separability that no honest feature reaches
  metricTolerance: 1e-3, // reported vs recomputed (rounding)
  generalizationGap: 0.15, // train minus test metric that warrants a look
  pathFailMargin: 0.05, // "at chance": within this of the majority baseline
  ckptTolerance: 0.005,
  rerunTolerance: 1e-9,
  bootstrap: 2000,
  // Smallest change in a headline metric treated as material when no interval is available
  // (claim vs reported without predictions, metric drop after a group-aware re-split).
  material: 0.02,
  entityShare: 0.01, // share of test rows whose entity also appears in train
};

export const num = z.number();
const metricsSchema = z.record(z.string(), z.unknown());
export const hasCmd = (c: string[] | undefined, what: string) => (c?.length ? null : `no ${what} command`);
export const first = (...rs: (string | null)[]) => rs.find((r) => r !== null) ?? null;

// Metric files may nest ({"test": {"auc": 0.8}}); keys are flattened with dots.
export function flatten(o: unknown, prefix = "", out: Record<string, unknown> = {}): Record<string, unknown> {
  if (o && typeof o === "object" && !Array.isArray(o)) for (const [k, v] of Object.entries(o)) flatten(v, prefix ? `${prefix}.${k}` : k, out);
  else if (prefix) out[prefix] = o;
  return out;
}
const pickNum = (metrics: Record<string, unknown>, keys: (string | undefined)[]): number | undefined => {
  for (const k of keys) if (k !== undefined && typeof metrics[k] === "number") return metrics[k] as number;
  return undefined;
};
// Common spellings of a metric in result files.
const ALIASES: Record<string, string[]> = { accuracy: ["accuracy", "acc"], f1: ["f1", "f1_score"], auc: ["auc", "roc_auc", "auroc"] };
const names = (m: Manifest) => ALIASES[m.metric!] ?? [m.metric!];
export function metricValue(metrics: Record<string, unknown>, m: Manifest): number | undefined {
  return pickNum(metrics, [m.metricKey, ...names(m).flatMap((n) => [n, `test_${n}`, `test.${n}`, `${n}_test`, `test/${n}`, `metrics.${n}`])]);
}
export function trainMetricValue(metrics: Record<string, unknown>, m: Manifest): number | undefined {
  return pickNum(metrics, [m.trainMetricKey, ...names(m).flatMap((n) => [`train_${n}`, `${n}_train`, `train.${n}`, `train/${n}`])]);
}

type Snapshot = Map<string, string>;
export function snapshot(root: string): Snapshot {
  const out: Snapshot = new Map();
  const walk = (d: string) => {
    for (const e of readdirSync(d, { withFileTypes: true })) {
      if (e.name === ".mlh_in" || e.name === "__pycache__") continue;
      const p = join(d, e.name);
      if (e.isDirectory()) walk(p); else { const st = statSync(p); out.set(relative(root, p), `${st.mtimeMs}:${st.size}`); }
    }
  };
  walk(root);
  return out;
}
export const changed = (before: Snapshot, after: Snapshot) => [...after.entries()].filter(([k, v]) => before.get(k) !== v).map(([k]) => k);

export interface Resolved { metrics: string; predictions?: string; checkpoint?: string; by: Record<string, string> }

// The manifest's output paths are a belief; what the run wrote is a fact. A
// declared path that does not exist after the run is replaced by the unique
// file of that kind the run actually produced.
export function resolveOutputs(sb: Sandbox, m: Manifest, files: string[]): { resolved: Partial<Resolved>; by: Record<string, string> } {
  const by: Record<string, string> = {};
  const res: Partial<Resolved> = {};
  const o = m.outputs ?? {};
  const readJ = (f: string) => { try { return flatten(JSON.parse(readFileSync(sb.path(f), "utf8"))); } catch { return undefined; } };
  if (o.metrics && existsSync(sb.path(o.metrics))) { res.metrics = o.metrics; by.metrics = "declared"; }
  else {
    const c = files.filter((f) => f.endsWith(".json")).filter((f) => { const j = readJ(f); return j && (m.metric ? metricValue(j, m) !== undefined : Object.values(j).some((v) => typeof v === "number")); });
    const pick = c.length === 1 ? c[0] : c.filter((f) => /metric|result|score/i.test(f)).length === 1 ? c.find((f) => /metric|result|score/i.test(f)) : undefined;
    if (pick) { res.metrics = pick; by.metrics = `observed: the run wrote ${pick}`; }
    else by.metrics = c.length ? `ambiguous: ${c.join(", ")}` : `no metrics json written (declared ${o.metrics ?? "none"})`;
  }
  const header = (f: string) => { try { return readFileSync(sb.path(f), "utf8").split("\n")[0]!.split(",").map((x) => x.trim()); } catch { return []; } };
  const isPred = (f: string) => { const h = header(f); return h.some((x) => ["y_true", "label", "target", "y"].includes(x)) && h.some((x) => ["y_pred", "pred", "prediction"].includes(x)); };
  if (o.predictions && existsSync(sb.path(o.predictions))) { res.predictions = o.predictions; by.predictions = "declared"; }
  else { const c = files.filter((f) => f.endsWith(".csv") && isPred(f)); if (c.length === 1) { res.predictions = c[0]!; by.predictions = `observed: the run wrote ${c[0]}`; } }
  if (o.checkpoint && existsSync(sb.path(o.checkpoint))) { res.checkpoint = o.checkpoint; by.checkpoint = "declared"; }
  else { const c = files.filter((f) => /\.(npz|npy|pt|pth|pkl|ckpt|joblib)$/.test(f)); if (c.length === 1) { res.checkpoint = c[0]!; by.checkpoint = `observed: the run wrote ${c[0]}`; } }
  return { resolved: res, by };
}

function trainCommand(m: Manifest, seed: number): { cmd: string[]; env: Record<string, string> } {
  const env: Record<string, string> = m.seedEnv ? { [m.seedEnv]: String(seed) } : {};
  let cmd = [...m.train!];
  if (m.seedArg) {
    const i = cmd.indexOf(m.seedArg);
    if (i >= 0 && i + 1 < cmd.length) cmd[i + 1] = String(seed); else cmd = [...cmd, m.seedArg, String(seed)];
  }
  return { cmd, env };
}
export const seedControllable = (m: Manifest) => !!(m.seedEnv || m.seedArg);

export async function runTrain(ctx: SkillCtx, sb: Sandbox, seed: number, extraEnv: Record<string, string> = {}): Promise<{ metrics: Record<string, unknown>; ms: number; resolved: Partial<Resolved>; by: Record<string, string> }> {
  const m = ctx.manifest;
  const before = snapshot(sb.dir);
  const { cmd, env } = trainCommand(m, seed);
  const r = await sb.run(cmd, { ...env, ...extraEnv });
  if (r.category) throw new SkillError(r.category, `train exited ${r.code ?? r.signal}: ${r.stderr.slice(-600)}`);
  const { resolved, by } = resolveOutputs(sb, m, changed(before, snapshot(sb.dir)));
  if (!resolved.metrics) throw new SkillError("experiment_error", `train wrote no readable metrics: ${by.metrics}`);
  const metrics = flatten(JSON.parse(readFileSync(sb.path(resolved.metrics), "utf8")));
  return { metrics, ms: r.ms, resolved, by };
}

export function stage(sb: Sandbox, from: string, rel: string): string {
  const dest = sb.path(rel);
  mkdirSync(dirname(dest), { recursive: true });
  copyFileSync(from, dest);
  return rel;
}

const trainOut = z.object({ metrics: metricsSchema, seed: num, ms: num, resolved: z.record(z.string(), z.unknown()), by: z.record(z.string(), z.string()) });
export type TrainOut = z.infer<typeof trainOut>;

const inspect: Skill = {
  id: "repo.inspect", version: "1.0.0", purpose: "setup", intents: ["reproduce", "audit"], actions: ["read_workspace", "exec_sandboxed"], after: [],
  summary: "Infer the experiment manifest (commands, data, outputs, claim) from the repository; an mlh.json in the repo overrides inference.",
  needs: () => null,
  output: z.object({ manifest: z.record(z.string(), z.unknown()), provenance: z.record(z.string(), z.string()) }),
  async run(ctx) {
    const sb = ctx.sandbox();
    const r = await sb.probe(["inspect", "."]);
    const explicit = existsSync(sb.path("mlh.json")) ? (JSON.parse(readFileSync(sb.path("mlh.json"), "utf8")) as Manifest) : {};
    let manifest: Manifest = { ...(r.manifest as Manifest), ...explicit };
    const provenance: Record<string, string> = { ...(r.provenance as Record<string, string>), ...Object.fromEntries(Object.keys(explicit).map((k) => [k, "mlh.json"])) };
    // Fields inference could not settle are proposed by a model and kept only if they pass deterministic checks.
    if (ctx.llm && MANIFEST_FIELDS.some((k) => manifest[k] === undefined)) {
      const proposal = await ctx.llm.proposer.json<Manifest>(MANIFEST_PROMPT, `Inferred so far (keep these):\n${JSON.stringify(manifest)}\n\n${digest(sb.dir)}`, 16000);
      const v = validateManifest(sb.dir, proposal ?? {});
      for (const [k, val] of Object.entries(v.accepted)) {
        if ((manifest as Record<string, unknown>)[k] === undefined) { (manifest as Record<string, unknown>)[k] = val; provenance[k] = `${ctx.llm.proposer.model}, validated: ${v.why[k]}`; }
      }
      for (const [k, why] of Object.entries(v.rejected)) provenance[`rejected.${k}`] = `${ctx.llm.proposer.model} proposal rejected: ${why}`;
      manifest = { ...manifest };
    }
    sb.remove();
    return { output: { manifest, provenance } };
  },
};

const MANIFEST_FIELDS: (keyof Manifest)[] = ["train", "data", "outputs", "metric", "claim"];
const MANIFEST_PROMPT = `You read ML experiment repositories and describe how to run and evaluate them. Return one JSON object with any of these fields you can determine from the files (omit what you cannot determine; never guess paths that do not exist):
{"train": ["python3", "<script>", "<args>..."],            // command, run from the repo root, that trains and writes test metrics
 "eval": ["python3", "<script>", ...],                       // optional separate command that evaluates a saved checkpoint
 "data": {"train": "<csv path>", "test": "<csv path>", "label": "<label column>"},
 "outputs": {"metrics": "<json written by train>", "predictions": "<csv of per-example predictions with y_true,y_pred[,score] columns, only if the code writes one>", "checkpoint": "<saved model file>", "evalMetrics": "<json written by eval>"},
 "metric": "accuracy|f1|auc", "metricKey": "<key of the headline test metric in the metrics json>", "trainMetricKey": "<key of the train metric, if written>",
 "claim": {"metric": "accuracy|f1|auc", "value": <number in [0,1] claimed in the README>},
 "seedEnv": "<environment variable the training script reads its seed from, if any>"}`;

const train: Skill = {
  id: "run.train", version: "1.0.0", purpose: "setup", intents: ["reproduce", "audit"], actions: ["exec_sandboxed"], after: [], fresh: true,
  summary: "Run the training entrypoint once in a fresh sandbox and keep its metrics, predictions and checkpoint.",
  needs: (m) => hasCmd(m.train, "train"),
  output: trainOut,
  async run(ctx) {
    const sb = ctx.sandbox();
    const seed = Number(ctx.node.params.seed ?? 0);
    const { metrics, ms, resolved, by } = await runTrain(ctx, sb, seed);
    for (const [k, rel] of Object.entries(resolved)) if (typeof rel === "string") ctx.save(k, sb.path(rel));
    sb.remove();
    return { output: { metrics, seed, ms, resolved, by } };
  },
};

export const predictionsOf = (ctx: SkillCtx, n: NodeState): string | undefined => {
  const ref = n.artifacts.find((a) => a.name === "predictions");
  return ref ? ctx.load(ref) : undefined;
};

const recompute: Skill = {
  id: "eval.recompute", version: "1.0.0", purpose: "detect", intents: ["reproduce", "audit"], actions: ["exec_sandboxed"], after: ["run.train"], replicable: true,
  summary: "Recompute the headline metric from the saved predictions with an independent implementation and compare it with the reported number.",
  needs: (m) => (m.metric ? null : "no headline metric named"),
  output: z.object({ reported: num.nullable(), recomputed: num.nullable(), n: num.nullable(), gap: z.string().optional() }),
  async run(ctx) {
    const t = ctx.dep("run.train");
    const p = predictionsOf(ctx, t);
    if (!p) return { output: { reported: null, recomputed: null, n: null, gap: "the run wrote no per-example predictions file" }, gap: "eval.recompute: the run wrote no per-example predictions to recompute from" };
    const sb = ctx.sandbox();
    const rel = stage(sb, p, ".mlh_in/predictions.csv");
    const r = await sb.probe(["recompute", rel, ctx.manifest.metric!]);
    sb.remove();
    const reported = metricValue((t.output as TrainOut).metrics, ctx.manifest) ?? null;
    const recomputed = r.value as number;
    const findings: FindingDraft[] = [];
    if (reported !== null && Math.abs(reported - recomputed) > T.metricTolerance) {
      findings.push({ category: "eval.metric_mismatch", severity: "high", kind: "suspicion", summary: `reported ${ctx.manifest.metric} ${reported.toFixed(4)} but the saved predictions give ${recomputed.toFixed(4)}`, evidence: { reported, recomputed, n: r.n, tolerance: T.metricTolerance } });
    }
    return { output: { reported, recomputed, n: r.n as number }, findings };
  },
};

const audit: Skill = {
  id: "data.audit", version: "1.0.0", purpose: "detect", intents: ["audit"], actions: ["exec_sandboxed"], after: [], replicable: true,
  summary: "Audit the train/test split without running repository code: exact row overlap and single-feature separability.",
  needs: (m) => (m.data?.train && m.data.test && m.data.label ? null : "train/test files or label column unknown"),
  output: z.object({ overlap_fraction: num, test_rows_in_train: num, features: z.array(z.object({ feature: z.string(), auc_train: num.nullable(), auc_test: num.nullable() })), majority_baseline: num.nullable() }).passthrough(),
  async run(ctx) {
    const d = ctx.manifest.data!;
    const sb = ctx.sandbox();
    const r = await sb.probe(["audit", d.train, d.test, d.label!]);
    sb.remove();
    if (r.error) throw new SkillError("experiment_error", String(r.error));
    const findings: FindingDraft[] = [];
    const frac = r.overlap_fraction as number;
    if (frac > T.overlapFraction) {
      findings.push({ category: "data.train_test_overlap", severity: frac > 0.1 ? "high" : "medium", kind: "fact", summary: `${r.test_rows_in_train} of ${r.rows_test} test rows (${(frac * 100).toFixed(1)}%) also appear in the training set`, evidence: { overlap_fraction: frac, test_rows_in_train: r.test_rows_in_train, rows_test: r.rows_test } });
    }
    for (const e of (r.entities ?? []) as { column: string; share: number; test_rows_with_train_entity: number; rows_per_entity_train: number }[]) {
      if (e.share > T.entityShare) findings.push({ category: "data.train_test_overlap", severity: "high", kind: "suspicion", summary: `${e.test_rows_with_train_entity} test rows (${(e.share * 100).toFixed(1)}%) belong to a "${e.column}" that also appears in train (${e.rows_per_entity_train.toFixed(1)} rows per entity in train)`, evidence: { column: e.column, share: e.share, rowsPerEntity: e.rows_per_entity_train } });
    }
    for (const f of r.features as { feature: string; auc_train: number | null }[]) {
      const sep = f.auc_train === null ? 0 : Math.max(f.auc_train, 1 - f.auc_train);
      if (sep >= T.leakAuc) findings.push({ category: "data.target_leak", severity: "high", kind: "suspicion", summary: `feature "${f.feature}" alone separates the training labels (AUC ${sep.toFixed(4)})`, evidence: { feature: f.feature, auc_train: sep, threshold: T.leakAuc } });
    }
    return { output: r, findings };
  },
};

const leakConfirm: Skill = {
  id: "data.leak_confirm", version: "1.0.0", purpose: "confirm", intents: ["audit"], actions: [], after: ["data.audit"], confirms: ["data.target_leak"],
  summary: "Test a suspected target leak on held-out data: a leaking feature separates the test labels as well.",
  needs: () => null,
  output: z.object({ feature: z.string(), auc_test: num.nullable() }),
  async run(ctx) {
    const feature = String(ctx.target!.evidence.feature);
    const a = (ctx.dep("data.audit").output as { features: { feature: string; auc_test: number | null }[] }).features.find((f) => f.feature === feature);
    const sep = a?.auc_test == null ? null : Math.max(a.auc_test, 1 - a.auc_test);
    const supports = sep !== null && sep >= T.leakAuc;
    return { output: { feature, auc_test: sep }, confirmation: { supports, note: `test-set separability of "${feature}" is ${sep?.toFixed(4) ?? "unknown"} (threshold ${T.leakAuc})` } };
  },
};

const groupSplit: Skill = {
  id: "data.group_split", version: "1.0.0", purpose: "confirm", intents: ["audit"], actions: ["exec_sandboxed"], after: ["run.train"], confirms: ["data.train_test_overlap"],
  summary: "Re-split train+test so no entity is on both sides and rerun: entity overlap matters if the headline metric falls materially.",
  needs: (m) => first(hasCmd(m.train, "train"), m.data?.label ? null : "label column unknown", m.metric ? null : "no metric"),
  output: z.object({ original: num.nullable(), groupSplit: num.nullable(), column: z.string() }),
  async run(ctx) {
    const column = ctx.target!.evidence.column;
    if (typeof column !== "string") return { output: { original: null, groupSplit: null, column: "" }, confirmation: { supports: true, note: "exact row overlap is a counted fact" } };
    const m = ctx.manifest, d = m.data!;
    const original = metricValue((ctx.dep("run.train").output as TrainOut).metrics, m) ?? null;
    const sb = ctx.sandbox();
    await sb.probe(["swap", d.train, d.test, d.label!, "7", column]);
    const { metrics } = await runTrain(ctx, sb, 0);
    sb.remove();
    const g = metricValue(metrics, m) ?? null;
    const supports = original !== null && g !== null && original - g >= T.material;
    return { output: { original, groupSplit: g, column }, confirmation: { supports, note: `${m.metric} with "${column}"-disjoint splits is ${g?.toFixed(4) ?? "unknown"} vs ${original?.toFixed(4) ?? "unknown"} reported (material drop ≥ ${T.material})` } };
  },
};

const gap: Skill = {
  id: "pipeline.gap_check", version: "1.0.0", purpose: "detect", intents: ["audit"], actions: [], after: ["run.train"],
  summary: "Flag a large gap between the train and test metric: an evaluation path that differs from the training path, or overfitting.",
  needs: (m) => (m.metric ? null : "no headline metric named"),
  output: z.object({ checked: z.boolean(), train: num.nullable(), test: num.nullable() }),
  async run(ctx) {
    const metrics = (ctx.dep("run.train").output as TrainOut).metrics;
    const test = metricValue(metrics, ctx.manifest) ?? null, tr = trainMetricValue(metrics, ctx.manifest) ?? null;
    const findings: FindingDraft[] = [];
    if (test !== null && tr !== null && tr - test > T.generalizationGap) {
      findings.push({ category: "pipeline.eval_path_mismatch", severity: "high", kind: "suspicion", summary: `train ${ctx.manifest.metric} ${tr.toFixed(4)} exceeds test ${test.toFixed(4)} by ${(tr - test).toFixed(4)}`, evidence: { train: tr, test, gap: tr - test, threshold: T.generalizationGap } });
    }
    return { output: { checked: test !== null && tr !== null, train: tr, test }, findings };
  },
};

const splitSwap: Skill = {
  id: "pipeline.split_swap", version: "1.0.0", purpose: "confirm", intents: ["audit"], actions: ["exec_sandboxed"], after: ["run.train"], confirms: ["pipeline.eval_path_mismatch"],
  summary: "Replace the test split with held-out training rows and rerun: if the evaluation path still scores at chance on in-distribution data, it is broken; if not, the gap is overfitting.",
  needs: (m) => first(hasCmd(m.train, "train"), m.data?.label ? null : "label column unknown", m.metric ? null : "no metric"),
  output: z.object({ holdout: num, train: num.nullable(), majority: num }),
  async run(ctx) {
    const m = ctx.manifest, d = m.data!;
    const sb = ctx.sandbox();
    await sb.probe(["swap", d.train, d.test, d.label!, "7"]);
    const { metrics, resolved } = await runTrain(ctx, sb, 0);
    const a = await sb.probe(["audit", d.train, d.test, d.label!]);
    // Prefer an independent recomputation; fall back to what the run reports.
    const r = resolved.predictions ? await sb.probe(["recompute", resolved.predictions, m.metric!]) : { value: metricValue(metrics, m) };
    sb.remove();
    if (typeof r.value !== "number" || typeof a.majority_baseline !== "number") throw new SkillError("experiment_error", "split swap produced no comparable metric");
    // "Chance" for AUC is 0.5 whatever the class balance.
    const holdout = r.value as number, majority = m.metric === "auc" ? 0.5 : (a.majority_baseline as number), tr = trainMetricValue(metrics, m) ?? null;
    const supports = holdout <= majority + T.pathFailMargin && tr !== null && tr >= majority + T.generalizationGap;
    const note = supports
      ? `held-out training rows score ${holdout.toFixed(4)} through the evaluation path, at chance (majority ${majority.toFixed(4)}), while train scores ${tr!.toFixed(4)}`
      : `held-out training rows score ${holdout.toFixed(4)} (majority ${majority.toFixed(4)}): the evaluation path works on in-distribution data, so the gap is overfitting or a metric problem`;
    return { output: { holdout, train: tr, majority }, confirmation: { supports, note } };
  },
};

const claim: Skill = {
  id: "stats.claim", version: "1.0.0", purpose: "detect", intents: ["reproduce"], actions: ["exec_sandboxed"], after: ["run.train"],
  summary: "Bootstrap a 95% interval for the recomputed metric and test whether the claimed number lies inside it.",
  needs: (m) => (m.claim ? null : "no claimed number found"),
  output: z.object({ claim: num, value: num, ci95: z.tuple([num, num]).nullable() }),
  async run(ctx) {
    const c = ctx.manifest.claim!;
    const t = ctx.dep("run.train");
    const p = predictionsOf(ctx, t);
    if (!p) {
      // No per-example predictions: compare the reported number, with a fixed materiality margin instead of an interval.
      const v = metricValue((t.output as TrainOut).metrics, { ...ctx.manifest, metric: c.metric });
      if (v === undefined) throw new SkillError("experiment_error", `metrics have no ${c.metric}`);
      const findings: FindingDraft[] = Math.abs(v - c.value) > T.material ? [{ category: "repro.claim_not_met", severity: c.value > v ? "high" : "low", kind: "suspicion", summary: `claimed ${c.metric} ${c.value.toFixed(4)}, the documented run reports ${v.toFixed(4)} (no predictions for an interval; margin ${T.material})`, evidence: { claim: c.value, reproduced: v, margin: T.material } }] : [];
      return { output: { claim: c.value, value: v, ci95: null }, findings };
    }
    const sb = ctx.sandbox();
    const rel = stage(sb, p, ".mlh_in/predictions.csv");
    const r = await sb.probe(["bootstrap", rel, c.metric, String(T.bootstrap), "0"]);
    sb.remove();
    const [lo, hi] = r.ci95 as [number, number];
    const findings: FindingDraft[] = [];
    if (c.value < lo || c.value > hi) {
      findings.push({ category: "repro.claim_not_met", severity: c.value > hi ? "high" : "low", kind: "suspicion", summary: `claimed ${c.metric} ${c.value.toFixed(4)} is outside the 95% interval [${lo.toFixed(4)}, ${hi.toFixed(4)}] of the reproduced ${(r.value as number).toFixed(4)}`, evidence: { claim: c.value, reproduced: r.value, ci95: [lo, hi] } });
    }
    return { output: { claim: c.value, value: r.value as number, ci95: [lo, hi] }, findings };
  },
};

const seeds: Skill = {
  id: "repro.seeds", version: "1.0.0", purpose: "confirm", intents: ["reproduce"], actions: ["exec_sandboxed"], after: ["run.train"], confirms: ["repro.claim_not_met"],
  summary: "Rerun with other seeds (or plain reruns when the seed is not controllable) and test whether any run's interval reaches the claim.",
  needs: (m) => first(hasCmd(m.train, "train"), m.claim ? null : "no claim"),
  output: z.object({ runs: z.array(z.object({ seed: num, value: num, ci95: z.tuple([num, num]) })) }),
  async run(ctx) {
    const m = ctx.manifest, c = m.claim!;
    const runs: { seed: number; value: number; ci95: [number, number] }[] = [];
    for (const seed of seedControllable(m) ? [1, 2] : [0, 0]) {
      const sb = ctx.sandbox(`seed${seed}-${runs.length}`);
      const { metrics, resolved } = await runTrain(ctx, sb, seed);
      if (resolved.predictions) {
        const r = await sb.probe(["bootstrap", resolved.predictions, c.metric, String(T.bootstrap), String(seed)]);
        runs.push({ seed, value: r.value as number, ci95: r.ci95 as [number, number] });
      } else {
        const v = metricValue(metrics, { ...m, metric: c.metric });
        if (v === undefined) throw new SkillError("experiment_error", `metrics have no ${c.metric}`);
        runs.push({ seed, value: v, ci95: [v - T.material, v + T.material] });
      }
      sb.remove();
    }
    const reach = runs.filter((r) => c.value >= r.ci95[0] && c.value <= r.ci95[1]);
    return { output: { runs }, confirmation: { supports: reach.length === 0, note: `${runs.length} reruns gave ${runs.map((r) => r.value.toFixed(4)).join(", ")}; ${reach.length} interval(s) contain the claim ${c.value.toFixed(4)}` } };
  },
};

const rerun: Skill = {
  id: "repro.rerun", version: "1.0.0", purpose: "detect", intents: ["reproduce"], actions: ["exec_sandboxed"], after: ["run.train"], replicable: true,
  summary: "Rerun training with the same seed in a fresh sandbox and compare every reported number and the predictions byte for byte.",
  needs: (m) => hasCmd(m.train, "train"),
  output: z.object({ identical: z.boolean(), diffs: z.record(z.string(), num) }),
  async run(ctx) {
    const t = ctx.dep("run.train"), a = (t.output as TrainOut);
    const sb = ctx.sandbox();
    // Same command and seed, but a different hash seed: results must not depend on Python's hash order.
    const { metrics, resolved } = await runTrain(ctx, sb, a.seed, { PYTHONHASHSEED: "1" });
    const pred = resolved.predictions;
    const predSame = pred ? ctx.save("predictions", sb.path(pred)).sha256 === t.artifacts.find((x) => x.name === "predictions")?.sha256 : true;
    sb.remove();
    const diffs: Record<string, number> = {};
    for (const [k, v] of Object.entries(a.metrics)) if (typeof v === "number" && typeof metrics[k] === "number") { const dv = Math.abs(v - (metrics[k] as number)); if (dv > T.rerunTolerance) diffs[k] = dv; }
    const findings: FindingDraft[] = [];
    if (Object.keys(diffs).length || !predSame) {
      const head = ctx.manifest.metric ? diffs[ctx.manifest.metric] ?? 0 : Math.max(0, ...Object.values(diffs));
      findings.push({ category: "repro.nondeterministic", severity: head >= 0.005 ? "high" : "low", kind: "suspicion", summary: `two runs with the same seed differ${head ? ` (${ctx.manifest.metric} by ${head.toFixed(4)})` : ""}${predSame ? "" : "; predictions differ"}`, evidence: { diffs, predictionsIdentical: predSame, seed: a.seed } });
    }
    return { output: { identical: !findings.length, diffs }, findings };
  },
};

const roundtrip: Skill = {
  id: "ckpt.roundtrip", version: "1.0.0", purpose: "detect", intents: ["reproduce", "audit"], actions: ["exec_sandboxed"], after: [], replicable: true,
  summary: "Train, then evaluate the saved checkpoint with the repository's eval entrypoint, and compare with the metric training reported.",
  needs: (m) => first(hasCmd(m.train, "train"), hasCmd(m.eval, "eval"), m.metric ? null : "no metric"),
  output: z.object({ trained: num, fromCheckpoint: num }),
  async run(ctx) {
    const m = ctx.manifest;
    const sb = ctx.sandbox();
    const { metrics, resolved } = await runTrain(ctx, sb, 0);
    const before = snapshot(sb.dir);
    const r = await sb.run(m.eval!, m.seedEnv ? { [m.seedEnv]: "0" } : {});
    if (r.category) throw new SkillError(r.category, `eval exited ${r.code ?? r.signal}: ${r.stderr.slice(-600)}`);
    // The eval script's metrics: the declared file if it exists, else the json the eval run wrote, else what it printed.
    const wrote = changed(before, snapshot(sb.dir)).filter((f) => f.endsWith(".json") && f !== resolved.metrics);
    const evPath = m.outputs?.evalMetrics && existsSync(sb.path(m.outputs.evalMetrics)) ? m.outputs.evalMetrics : wrote.length === 1 ? wrote[0] : undefined;
    let ev = evPath ? flatten(JSON.parse(readFileSync(sb.path(evPath), "utf8"))) : undefined;
    if (!ev) {
      const printed = new RegExp(`\\b${m.metric}\\b[^0-9\\n]{0,30}(\\d*\\.\\d+)`, "i").exec(r.stdout);
      if (printed) ev = { [m.metric!]: Number(printed[1]) };
    }
    sb.remove();
    const { metricKey: _k, ...noKey } = m;
    const a = metricValue(metrics, m), b = ev && (metricValue(ev, m) ?? metricValue(ev, noKey));
    if (a === undefined || b === undefined) throw new SkillError("experiment_error", `metric ${m.metric} missing from train or eval output`);
    const findings: FindingDraft[] = [];
    if (Math.abs(a - b) > T.ckptTolerance) findings.push({ category: "ckpt.roundtrip_mismatch", severity: "high", kind: "suspicion", summary: `training reported ${m.metric} ${a.toFixed(4)}, but the saved checkpoint evaluates to ${b.toFixed(4)}`, evidence: { trained: a, fromCheckpoint: b, tolerance: T.ckptTolerance } });
    return { output: { trained: a, fromCheckpoint: b }, findings };
  },
};

const hypothesize: Skill = {
  id: "llm.hypothesize", version: "1.0.0", purpose: "detect", intents: ["audit", "reproduce"], actions: ["model_query"], after: [],
  summary: "A model reads the code and proposes suspected defects, each with an executable check; nothing it says counts until the check runs and a second model reviews it.",
  needs: () => null,
  output: z.object({ model: z.string(), proposed: num }),
  async run(ctx) {
    if (!ctx.llm) throw new SkillError("precondition", "model use not enabled for this run");
    const sb = ctx.sandbox();
    const d = digest(sb.dir);
    sb.remove();
    const r = await ctx.llm.proposer.json<{ hypotheses?: { category?: string; file?: string; rationale?: string; check?: string }[] }>(HYPOTHESIS_PROMPT, `Objective: ${ctx.state.objective}\nManifest: ${JSON.stringify(ctx.manifest)}\n\n${d}`, 32000);
    const hs = (r.hypotheses ?? []).filter((h) => typeof h.category === "string" && typeof h.check === "string" && /^(data|pipeline|eval|ckpt|repro)\.[a-z_]+$|^other:[\w-]+$/.test(h.category)).slice(0, 4);
    const findings: FindingDraft[] = hs.map((h) => ({ category: h.category!, severity: "medium", kind: "suspicion", summary: `${h.rationale ?? ""}`.slice(0, 400) + (h.file ? ` (${h.file})` : ""), evidence: { origin: "llm", model: ctx.llm!.proposer.model, file: h.file ?? null, check: h.check! } }));
    return { output: { model: ctx.llm.proposer.model, proposed: findings.length }, findings };
  },
};

const llmCheck: Skill = {
  id: "llm.check", version: "1.0.0", purpose: "confirm", intents: ["audit", "reproduce"], actions: ["exec_sandboxed", "model_query"], after: ["llm.hypothesize"], confirmsOrigin: "llm",
  summary: "Run a model-proposed check in a fresh sandbox after training; a second, different model reviews whether the check and its output really demonstrate the defect. Both must agree.",
  needs: () => null,
  output: z.object({ executed: z.record(z.string(), z.unknown()), valid: z.boolean(), reviewer: z.string() }),
  async run(ctx) {
    if (!ctx.llm) throw new SkillError("precondition", "model use not enabled for this run");
    const f = ctx.target!;
    const sb = ctx.sandbox();
    if (ctx.manifest.train) {
      const r = await sb.run(ctx.manifest.train, ctx.manifest.seedEnv ? { [ctx.manifest.seedEnv]: "0" } : {});
      if (r.category && r.category !== "experiment_error") throw new SkillError(r.category, `train before check: ${r.stderr.slice(-300)}`);
    }
    mkdirSync(sb.path(".mlh_in"), { recursive: true });
    writeFileSync(sb.path(".mlh_in/check.py"), String(f.evidence.check));
    const r = await sb.run(["python3", ".mlh_in/check.py"]);
    sb.remove();
    if (r.category === "timeout" || r.category === "resource_limit" || r.category === "sandbox_transient") throw new SkillError(r.category, "check did not finish");
    let executed: Record<string, unknown>;
    try { executed = JSON.parse(r.stdout.trim().split("\n").pop() ?? ""); } catch { throw new SkillError("experiment_error", `check printed no JSON verdict (exit ${r.code}): ${r.stderr.slice(-300)}`); }
    const review = await ctx.llm.reviewer.json<{ valid?: boolean; reason?: string }>(REVIEW_PROMPT, `Claimed defect [${f.category}]: ${f.summary}\n\nCheck script:\n${String(f.evidence.check).slice(0, 8000)}\n\nCheck output (last line is the verdict):\n${r.stdout.slice(-3000)}\n${r.stderr.slice(-1000)}`, 16000);
    const valid = review.valid === true;
    const supports = executed.supports === true && valid;
    return { output: { executed, valid, reviewer: ctx.llm.reviewer.model }, confirmation: { tier: "measured", supports, note: `check printed supports=${String(executed.supports)} (${JSON.stringify(executed.observed ?? {}).slice(0, 300)}); ${ctx.llm.reviewer.model} review: ${valid ? "valid" : "invalid"}: ${String(review.reason ?? "").slice(0, 300)}` } };
  },
};

const HYPOTHESIS_PROMPT = `You audit ML experiment repositories for implementation and evaluation errors that change results: data leakage, broken preprocessing, incorrect metrics, checkpoint failures, irreproducibility, or anything else. Propose at most 4 specific suspected defects you can point to in the code. For each, write an executable check: a self-contained python3 script (numpy and pandas only, no network) run from the repository root AFTER the training command has run, that measures whether the defect actually changes the results and prints as its LAST line one JSON object {"supports": true|false, "observed": {...numbers...}}. "supports" must be true only if the measurement shows the defect matters. Do not propose style issues or hypothetical risks. If nothing is wrong, return {"hypotheses": []}. Return JSON: {"hypotheses": [{"category": "...", "file": "path:line", "rationale": "...", "check": "<python source>"}]}.
${VOCABULARY}`;

const REVIEW_PROMPT = `You are an independent, skeptical reviewer. Another model claimed a defect in an ML experiment and wrote a check script; it was executed and its output is shown. Decide whether (1) the check actually tests the claimed defect (not something else, not a restatement of the code's own numbers), (2) the check would come out differently on a correct implementation, and (3) the output demonstrates that the defect changes results. Return JSON {"valid": true|false, "reason": "..."}; valid only if all three hold.`;

export const BASE_SKILLS: Skill[] = [inspect, train, audit, leakConfirm, groupSplit, recompute, gap, splitSwap, claim, seeds, rerun, roundtrip, hypothesize, llmCheck];

