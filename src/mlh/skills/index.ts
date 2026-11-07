// The skill library: versioned, discoverable units with declared needs, actions
// and an output schema the orchestrator enforces. Each skill measures one thing
// with trusted probes or by running the experiment in a fresh sandbox; deciding
// what a measurement means stays in the declared thresholds below, fixed in
// advance and identical for every repository.
import { copyFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname } from "node:path";
import { z } from "zod";
import type { NodeState, RunState, TaskNode } from "../core/dag.ts";
import type { SkillSpec } from "../core/planner.ts";
import type { ArtifactRef, Finding, FindingDraft, Manifest } from "../core/types.ts";
import { SkillError, type Sandbox } from "../io/sandbox.ts";

export interface SkillCtx {
  node: TaskNode;
  state: RunState;
  manifest: Manifest;
  target: Finding | undefined;
  dep(skill: string): NodeState;
  sandbox(label?: string): Sandbox;
  save(name: string, hostPath: string): ArtifactRef;
  load(ref: ArtifactRef): string;
}

export interface SkillOutcome {
  output: unknown;
  findings?: FindingDraft[];
  confirmation?: { supports: boolean; note: string };
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
};

const num = z.number();
const metricsSchema = z.record(z.string(), z.unknown());
const hasCmd = (c: string[] | undefined, what: string) => (c?.length ? null : `no ${what} command`);
const first = (...rs: (string | null)[]) => rs.find((r) => r !== null) ?? null;

function metricValue(metrics: Record<string, unknown>, name: string): number | undefined {
  const v = metrics[name] ?? metrics[`test_${name}`];
  return typeof v === "number" ? v : undefined;
}
function trainMetricValue(metrics: Record<string, unknown>, name: string): number | undefined {
  const v = metrics[`train_${name}`] ?? metrics[`${name}_train`];
  return typeof v === "number" ? v : undefined;
}

async function runTrain(ctx: SkillCtx, sb: Sandbox, seed: number): Promise<{ metrics: Record<string, unknown>; ms: number }> {
  const m = ctx.manifest;
  const env: Record<string, string> = m.seedEnv ? { [m.seedEnv]: String(seed) } : {};
  const r = await sb.run(m.train!, env);
  if (r.category) throw new SkillError(r.category, `train exited ${r.code ?? r.signal}: ${r.stderr.slice(-600)}`);
  const metrics = sb.readJson(m.outputs!.metrics!);
  if (!metrics || typeof metrics !== "object") throw new SkillError("experiment_error", `train did not write ${m.outputs!.metrics}`);
  return { metrics: metrics as Record<string, unknown>, ms: r.ms };
}

function stage(sb: Sandbox, from: string, rel: string): string {
  const dest = sb.path(rel);
  mkdirSync(dirname(dest), { recursive: true });
  copyFileSync(from, dest);
  return rel;
}

const trainOut = z.object({ metrics: metricsSchema, seed: num, ms: num });
type TrainOut = z.infer<typeof trainOut>;

const inspect: Skill = {
  id: "repo.inspect", version: "1.0.0", purpose: "setup", intents: ["reproduce", "audit"], actions: ["read_workspace", "exec_sandboxed"], after: [],
  summary: "Infer the experiment manifest (commands, data, outputs, claim) from the repository; an mlh.json in the repo overrides inference.",
  needs: () => null,
  output: z.object({ manifest: z.record(z.string(), z.unknown()), provenance: z.record(z.string(), z.string()) }),
  async run(ctx) {
    const sb = ctx.sandbox();
    const r = await sb.probe(["inspect", "."]);
    const explicit = existsSync(sb.path("mlh.json")) ? (JSON.parse(readFileSync(sb.path("mlh.json"), "utf8")) as Manifest) : {};
    const manifest = { ...(r.manifest as Manifest), ...explicit };
    const provenance = { ...(r.provenance as Record<string, string>), ...Object.fromEntries(Object.keys(explicit).map((k) => [k, "mlh.json"])) };
    sb.remove();
    return { output: { manifest, provenance } };
  },
};

const train: Skill = {
  id: "run.train", version: "1.0.0", purpose: "setup", intents: ["reproduce", "audit"], actions: ["exec_sandboxed"], after: [], fresh: true,
  summary: "Run the training entrypoint once in a fresh sandbox and keep its metrics, predictions and checkpoint.",
  needs: (m) => first(hasCmd(m.train, "train"), m.outputs?.metrics ? null : "no metrics output path"),
  output: trainOut,
  async run(ctx) {
    const sb = ctx.sandbox();
    const seed = Number(ctx.node.params.seed ?? 0);
    const { metrics, ms } = await runTrain(ctx, sb, seed);
    const o = ctx.manifest.outputs!;
    for (const [k, rel] of Object.entries({ metrics: o.metrics, predictions: o.predictions, checkpoint: o.checkpoint })) {
      if (rel && existsSync(sb.path(rel))) ctx.save(k, sb.path(rel));
    }
    sb.remove();
    return { output: { metrics, seed, ms } };
  },
};

const predictionsOf = (ctx: SkillCtx, n: NodeState): string => {
  const ref = n.artifacts.find((a) => a.name === "predictions");
  if (!ref) throw new SkillError("precondition", `${n.node.id} produced no predictions file`);
  return ctx.load(ref);
};

const recompute: Skill = {
  id: "eval.recompute", version: "1.0.0", purpose: "detect", intents: ["reproduce", "audit"], actions: ["exec_sandboxed"], after: ["run.train"], replicable: true,
  summary: "Recompute the headline metric from the saved predictions with an independent implementation and compare it with the reported number.",
  needs: (m) => first(m.outputs?.predictions ? null : "no predictions output to recompute from", m.metric ? null : "no headline metric named"),
  output: z.object({ reported: num.nullable(), recomputed: num, n: num }),
  async run(ctx) {
    const t = ctx.dep("run.train");
    const sb = ctx.sandbox();
    const rel = stage(sb, predictionsOf(ctx, t), ".mlh_in/predictions.csv");
    const r = await sb.probe(["recompute", rel, ctx.manifest.metric!]);
    sb.remove();
    const reported = metricValue((t.output as TrainOut).metrics, ctx.manifest.metric!) ?? null;
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
  output: z.object({ overlap_fraction: num, test_rows_in_train: num, features: z.array(z.object({ feature: z.string(), auc_train: num.nullable(), auc_test: num.nullable() })), majority_baseline: num }).passthrough(),
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

const gap: Skill = {
  id: "pipeline.gap_check", version: "1.0.0", purpose: "detect", intents: ["audit"], actions: [], after: ["run.train"],
  summary: "Flag a large gap between the train and test metric: an evaluation path that differs from the training path, or overfitting.",
  needs: (m) => (m.metric ? null : "no headline metric named"),
  output: z.object({ checked: z.boolean(), train: num.nullable(), test: num.nullable() }),
  async run(ctx) {
    const metrics = (ctx.dep("run.train").output as TrainOut).metrics;
    const test = metricValue(metrics, ctx.manifest.metric!) ?? null, tr = trainMetricValue(metrics, ctx.manifest.metric!) ?? null;
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
  needs: (m) => first(hasCmd(m.train, "train"), m.data?.label ? null : "label column unknown", m.outputs?.predictions ? null : "no predictions output", m.metric ? null : "no metric"),
  output: z.object({ holdout: num, train: num.nullable(), majority: num }),
  async run(ctx) {
    const m = ctx.manifest, d = m.data!;
    const sb = ctx.sandbox();
    await sb.probe(["swap", d.train, d.test, d.label!, "7"]);
    const { metrics } = await runTrain(ctx, sb, 0);
    const r = await sb.probe(["recompute", m.outputs!.predictions!, m.metric!]);
    sb.remove();
    const holdout = r.value as number, majority = r.majority_baseline as number, tr = trainMetricValue(metrics, m.metric!) ?? null;
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
  needs: (m) => first(m.claim ? null : "no claimed number found", m.outputs?.predictions ? null : "no predictions output"),
  output: z.object({ claim: num, value: num, ci95: z.tuple([num, num]) }),
  async run(ctx) {
    const c = ctx.manifest.claim!;
    const sb = ctx.sandbox();
    const rel = stage(sb, predictionsOf(ctx, ctx.dep("run.train")), ".mlh_in/predictions.csv");
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
  needs: (m) => first(hasCmd(m.train, "train"), m.claim ? null : "no claim", m.outputs?.predictions ? null : "no predictions output"),
  output: z.object({ runs: z.array(z.object({ seed: num, value: num, ci95: z.tuple([num, num]) })) }),
  async run(ctx) {
    const m = ctx.manifest, c = m.claim!;
    const runs: { seed: number; value: number; ci95: [number, number] }[] = [];
    for (const seed of m.seedEnv ? [1, 2] : [0, 0]) {
      const sb = ctx.sandbox(`seed${seed}-${runs.length}`);
      await runTrain(ctx, sb, seed);
      const r = await sb.probe(["bootstrap", m.outputs!.predictions!, c.metric, String(T.bootstrap), String(seed)]);
      sb.remove();
      runs.push({ seed, value: r.value as number, ci95: r.ci95 as [number, number] });
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
    const { metrics } = await runTrain(ctx, sb, a.seed);
    const pred = ctx.manifest.outputs?.predictions;
    const predSame = pred && existsSync(sb.path(pred)) ? ctx.save("predictions", sb.path(pred)).sha256 === t.artifacts.find((x) => x.name === "predictions")?.sha256 : true;
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
  needs: (m) => first(hasCmd(m.train, "train"), hasCmd(m.eval, "eval"), m.outputs?.checkpoint ? null : "no checkpoint path", m.outputs?.evalMetrics ? null : "no eval metrics path", m.metric ? null : "no metric"),
  output: z.object({ trained: num, fromCheckpoint: num }),
  async run(ctx) {
    const m = ctx.manifest;
    const sb = ctx.sandbox();
    const { metrics } = await runTrain(ctx, sb, 0);
    const r = await sb.run(m.eval!, m.seedEnv ? { [m.seedEnv]: "0" } : {});
    if (r.category) throw new SkillError(r.category, `eval exited ${r.code ?? r.signal}: ${r.stderr.slice(-600)}`);
    const ev = sb.readJson(m.outputs!.evalMetrics!) as Record<string, unknown> | undefined;
    sb.remove();
    const a = metricValue(metrics, m.metric!), b = ev && metricValue(ev, m.metric!);
    if (a === undefined || b === undefined) throw new SkillError("experiment_error", `metric ${m.metric} missing from train or eval output`);
    const findings: FindingDraft[] = [];
    if (Math.abs(a - b) > T.ckptTolerance) findings.push({ category: "ckpt.roundtrip_mismatch", severity: "high", kind: "suspicion", summary: `training reported ${m.metric} ${a.toFixed(4)}, but the saved checkpoint evaluates to ${b.toFixed(4)}`, evidence: { trained: a, fromCheckpoint: b, tolerance: T.ckptTolerance } });
    return { output: { trained: a, fromCheckpoint: b }, findings };
  },
};

export const SKILLS: Skill[] = [inspect, train, audit, leakConfirm, recompute, gap, splitSwap, claim, seeds, rerun, roundtrip];

export function skill(id: string): Skill {
  const s = SKILLS.find((x) => x.id === id);
  if (!s) throw new Error(`unknown skill ${id}`);
  return s;
}
