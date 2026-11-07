// The exploring agent as a skill, and the independent confirmers its findings
// are routed to. A confirmer answers one question with harness code: does the
// claimed defect materially change the result? It may also say it cannot tell
// (inconclusive), in which case the agent's own effect measurement is rerun and
// judged by the harness's threshold. Nothing the agent asserts counts by itself.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { z } from "zod";
import { runAgent } from "../agent.ts";
import type { FindingDraft, Manifest } from "../core/types.ts";
import { SkillError } from "../io/sandbox.ts";
import { validateManifest } from "./repo.ts";
import { first, flatten, hasCmd, metricValue, num, runTrain, T, type Skill, type SkillCtx, type SkillOutcome } from "./index.ts";

const inconclusive = (note: string, output: unknown = {}): SkillOutcome => ({ output, confirmation: { supports: false, note, inconclusive: true } });
const decide = (supports: boolean, note: string, output: unknown = {}): SkillOutcome => ({ output, confirmation: { supports, note } });
const any = z.record(z.string(), z.unknown());

const explore: Skill = {
  id: "agent.explore", version: "1.0.0", purpose: "detect", intents: ["audit", "reproduce"], actions: ["exec_sandboxed", "model_query"], after: [],
  summary: "An agent explores the repository with file and shell tools, checkpointed after every step; it reports suspected defects (each with an effect measurement) and how the experiment runs.",
  needs: () => null,
  output: z.object({ steps: num, stopped: z.string(), manifest: any, rejected: z.record(z.string(), z.string()), resumedAt: num.optional() }),
  async run(ctx) {
    if (!ctx.llm) throw new SkillError("precondition", "model use not enabled for this run");
    const r = await runAgent({
      workDir: ctx.workDir(), source: ctx.sourceDir, objective: ctx.state.objective, model: ctx.llm.proposer,
      isolation: ctx.isolation, isolationEvidence: ctx.isolationEvidence, checkpoint: true,
      ...(ctx.sandboxFault ? { sandboxFault: ctx.sandboxFault } : {}), ...(ctx.crashAfterSteps !== undefined ? { crashAfterSteps: ctx.crashAfterSteps } : {}),
    });
    if (r.stopped !== "report") throw new SkillError("experiment_error", `agent stopped without a report (${r.stopped})`);
    const v = validateManifest(ctx.sourceDir, (r.manifest ?? {}) as Manifest);
    const findings: FindingDraft[] = r.findings
      .filter((f) => typeof f.category === "string" && /^(data|pipeline|eval|ckpt|repro)\.[a-z_]+$|^other:[\w-]+$/.test(f.category))
      .map((f) => ({ category: f.category, severity: "medium", kind: "suspicion", summary: String(f.summary).slice(0, 400), evidence: { origin: "agent", model: ctx.llm!.proposer.model, agentEvidence: String(f.evidence ?? "").slice(0, 1200), ...(f.column ? { column: f.column } : {}), ...(f.feature ? { feature: f.feature } : {}), ...(f.effect_check ? { effect_check: f.effect_check } : {}) } }));
    return { output: { steps: r.steps, stopped: r.stopped, manifest: v.accepted, rejected: v.rejected, ...(r.resumedAt !== undefined ? { resumedAt: r.resumedAt } : {}) }, findings };
  },
};

const target = (ctx: SkillCtx) => ctx.target!.evidence as Record<string, unknown>;
const hasData = (m: Manifest) => (m.data?.train && m.data.test && m.data.label ? null : "csv train/test/label unknown");

// Rerun after an intervention and return the headline metric.
async function metricAfter(ctx: SkillCtx, label: string, intervene: (sb: ReturnType<SkillCtx["sandbox"]>) => Promise<unknown>, env: Record<string, string> = {}): Promise<number | undefined> {
  const sb = ctx.sandbox(label);
  try {
    await intervene(sb);
    const { metrics } = await runTrain(ctx, sb, 0, env);
    return metricValue(metrics, ctx.manifest);
  } finally { sb.remove(); }
}

const overlap: Skill = {
  id: "verify.overlap", version: "1.0.0", purpose: "confirm", intents: ["audit"], actions: ["exec_sandboxed"], after: [],
  summary: "Remove the overlap (train rows duplicated in test, or entities on both sides via a group-aware re-split) and rerun: confirmed only if the headline metric changes materially.",
  needs: (m) => first(hasCmd(m.train, "train"), hasData(m), m.metric ? null : "no metric"),
  output: any,
  async run(ctx) {
    const d = ctx.manifest.data!;
    const sb = ctx.sandbox("audit");
    const a = await sb.probe(["audit", d.train, d.test, d.label!]);
    sb.remove();
    const base = await metricAfter(ctx, "base", async () => {});
    if (base === undefined) return inconclusive("no headline metric from the run");
    const named = target(ctx).column;
    const ent = ((a.entities ?? []) as { column: string; share: number }[]).filter((e) => e.share > T.entityShare).sort((x, y) => (x.column === named ? -1 : y.column === named ? 1 : y.share - x.share))[0];
    if ((a.overlap_fraction as number) > T.overlapFraction) {
      const after = await metricAfter(ctx, "dedupe", (s) => s.probe(["dedupe", d.train, d.test, d.label!]));
      if (after === undefined) return inconclusive("no metric after removing duplicated rows");
      return decide(Math.abs(base - after) >= T.material, `${((a.overlap_fraction as number) * 100).toFixed(1)}% of test rows are in train; removing them from train moves ${ctx.manifest.metric} ${base.toFixed(4)} → ${after.toFixed(4)} (material ≥ ${T.material})`);
    }
    if (ent) {
      const after = await metricAfter(ctx, "group", (s) => s.probe(["swap", d.train, d.test, d.label!, "7", ent.column]));
      if (after === undefined) return inconclusive("no metric after the group-aware re-split");
      return decide(base - after >= T.material, `${(ent.share * 100).toFixed(1)}% of test rows share a "${ent.column}" with train; with "${ent.column}"-disjoint splits ${ctx.manifest.metric} moves ${base.toFixed(4)} → ${after.toFixed(4)} (material drop ≥ ${T.material})`);
    }
    return inconclusive("no exact row overlap and no id-like entity column shared across the split");
  },
};

const leak: Skill = {
  id: "verify.leak", version: "1.0.0", purpose: "confirm", intents: ["audit"], actions: ["exec_sandboxed"], after: [],
  summary: "A leak through a single feature is confirmed if that feature alone separates the held-out labels almost perfectly; anything subtler is left to the effect check.",
  needs: (m) => hasData(m),
  output: any,
  async run(ctx) {
    const d = ctx.manifest.data!, feature = target(ctx).feature;
    if (typeof feature !== "string") return inconclusive("no single leaking feature named");
    const sb = ctx.sandbox();
    const a = await sb.probe(["audit", d.train, d.test, d.label!]);
    sb.remove();
    const f = (a.features as { feature: string; auc_test: number | null }[]).find((x) => x.feature === feature);
    const sep = f?.auc_test == null ? null : Math.max(f.auc_test, 1 - f.auc_test);
    return sep !== null && sep >= T.leakAuc ? decide(true, `"${feature}" alone separates the test labels (AUC ${sep.toFixed(4)})`) : inconclusive(`"${feature}" test separability ${sep?.toFixed(4) ?? "unknown"}: not decisive by itself`);
  },
};

const metric: Skill = {
  id: "verify.metric", version: "1.0.0", purpose: "confirm", intents: ["audit"], actions: ["exec_sandboxed"], after: [],
  summary: "Rerun, then recompute the headline metric from the per-example predictions the run wrote and compare with the reported number.",
  needs: (m) => first(hasCmd(m.train, "train"), m.metric ? null : "no metric"),
  output: any,
  async run(ctx) {
    const m = ctx.manifest;
    const sb = ctx.sandbox();
    try {
      const { metrics, resolved } = await runTrain(ctx, sb, 0);
      if (!resolved.predictions) return inconclusive("the run writes no per-example predictions to recompute from");
      const r = await sb.probe(["recompute", resolved.predictions, m.metric!]);
      const rep = metricValue(metrics, m);
      if (rep === undefined || typeof r.value !== "number") return inconclusive("no comparable numbers");
      return decide(Math.abs(rep - r.value) > T.metricTolerance, `reported ${m.metric} ${rep.toFixed(4)}; recomputed from ${resolved.predictions}: ${r.value.toFixed(4)}`);
    } finally { sb.remove(); }
  },
};

const ckpt: Skill = {
  id: "verify.ckpt", version: "1.0.0", purpose: "confirm", intents: ["audit", "reproduce"], actions: ["exec_sandboxed"], after: [],
  summary: "Train, run the eval entrypoint on the saved checkpoint, compare with the metric training reported.",
  needs: (m) => first(hasCmd(m.train, "train"), hasCmd(m.eval, "eval"), m.metric ? null : "no metric"),
  output: any,
  async run(ctx) {
    const m = ctx.manifest;
    const sb = ctx.sandbox();
    try {
      const { metrics } = await runTrain(ctx, sb, 0);
      const r = await sb.run(m.eval!);
      if (r.category) return inconclusive(`eval failed: ${r.stderr.slice(-200)}`);
      const printed = new RegExp(`\\b(?:${m.metric}|acc|auc|f1)\\b[^0-9\\n]{0,30}(\\d*\\.\\d+)`, "i").exec(r.stdout);
      const evalJson = m.outputs?.evalMetrics && existsSync(sb.path(m.outputs.evalMetrics)) ? flatten(JSON.parse(readFileSync(sb.path(m.outputs.evalMetrics), "utf8"))) : undefined;
      const { metricKey: _k, ...noKey } = m;
      const b = (evalJson && (metricValue(evalJson, m) ?? metricValue(evalJson, noKey))) ?? (printed ? Number(printed[1]) : undefined);
      const a = metricValue(metrics, m);
      if (a === undefined || b === undefined) return inconclusive("could not read both the trained and the reloaded metric");
      return decide(Math.abs(a - b) > T.ckptTolerance, `training reported ${a.toFixed(4)}; the checkpoint evaluates to ${b.toFixed(4)}`);
    } finally { sb.remove(); }
  },
};

const nondet: Skill = {
  id: "verify.nondeterminism", version: "1.0.0", purpose: "confirm", intents: ["reproduce"], actions: ["exec_sandboxed"], after: [],
  summary: "Run the same command three times (varying only Python's hash seed): confirmed if the headline metric differs.",
  needs: (m) => first(hasCmd(m.train, "train"), m.metric ? null : "no metric"),
  output: any,
  async run(ctx) {
    const vs: number[] = [];
    for (const h of ["0", "1", "2"]) { const v = await metricAfter(ctx, `h${h}`, async () => {}, { PYTHONHASHSEED: h }); if (v !== undefined) vs.push(v); }
    if (vs.length < 2) return inconclusive("too few runs produced the metric");
    const spread = Math.max(...vs) - Math.min(...vs);
    return spread > T.rerunTolerance ? decide(true, `three runs of the same command gave ${vs.map((v) => v.toFixed(4)).join(", ")}`) : inconclusive(`three runs agree (${vs[0]!.toFixed(4)}); nondeterminism not observed`);
  },
};

const claimV: Skill = {
  id: "verify.claim", version: "1.0.0", purpose: "confirm", intents: ["reproduce"], actions: ["exec_sandboxed"], after: [],
  summary: "Run the documented command and test the claim against the bootstrap interval of the recomputed metric, or a fixed materiality margin without predictions.",
  needs: (m) => first(hasCmd(m.train, "train"), m.claim ? null : "no claim"),
  output: any,
  async run(ctx) {
    const c = ctx.manifest.claim!;
    const sb = ctx.sandbox();
    try {
      const { metrics, resolved } = await runTrain(ctx, sb, 0);
      if (resolved.predictions) {
        const r = await sb.probe(["bootstrap", resolved.predictions, c.metric, String(T.bootstrap), "0"]);
        const [lo, hi] = r.ci95 as [number, number];
        return decide(c.value < lo || c.value > hi, `claimed ${c.value.toFixed(4)}; reproduced ${(r.value as number).toFixed(4)}, 95% interval [${lo.toFixed(4)}, ${hi.toFixed(4)}]`);
      }
      const v = metricValue(metrics, { ...ctx.manifest, metric: c.metric });
      if (v === undefined) return inconclusive(`the run reports no ${c.metric}`);
      return decide(Math.abs(v - c.value) > T.material, `claimed ${c.value.toFixed(4)}; the documented run reports ${v.toFixed(4)} (margin ${T.material})`);
    } finally { sb.remove(); }
  },
};

const evalPath: Skill = {
  id: "verify.eval_path", version: "1.0.0", purpose: "confirm", intents: ["audit"], actions: ["exec_sandboxed"], after: [],
  summary: "Score held-out training rows through the evaluation path: at chance while training scores well means the path is broken; otherwise undecided.",
  needs: (m) => first(hasCmd(m.train, "train"), hasData(m), m.metric ? null : "no metric"),
  output: any,
  async run(ctx) {
    const m = ctx.manifest, d = m.data!;
    const sb = ctx.sandbox();
    try {
      const a = await sb.probe(["audit", d.train, d.test, d.label!]);
      await sb.probe(["swap", d.train, d.test, d.label!, "7"]);
      const { metrics } = await runTrain(ctx, sb, 0);
      const v = metricValue(metrics, m);
      const chance = m.metric === "auc" ? 0.5 : (a.majority_baseline as number | null);
      if (v === undefined || chance == null) return inconclusive("no comparable metric");
      return v <= chance + T.pathFailMargin ? decide(true, `held-out training rows score ${v.toFixed(4)} through the evaluation path, at chance (${chance.toFixed(4)})`) : inconclusive(`held-out training rows score ${v.toFixed(4)} (chance ${chance.toFixed(4)}): not at chance, undecided`);
    } finally { sb.remove(); }
  },
};

const effect: Skill = {
  id: "verify.effect", version: "1.0.0", purpose: "confirm", intents: ["audit", "reproduce"], actions: ["exec_sandboxed"], after: [],
  summary: "Run the agent's effect measurement twice on fresh copies; supported only if both runs agree and the harness finds the difference material. The script's opinion of itself is ignored.",
  needs: () => null,
  output: any,
  async run(ctx) {
    const script = target(ctx).effect_check;
    if (typeof script !== "string" || !script.trim()) return inconclusive("the agent gave no effect measurement");
    const runs: { with: number; without: number }[] = [];
    for (const i of [0, 1]) {
      const sb = ctx.sandbox(`effect${i}`);
      try {
        mkdirSync(sb.path(".mlh_in"), { recursive: true });
        writeFileSync(sb.path(".mlh_in/effect.py"), script);
        const r = await sb.run(["python3", ".mlh_in/effect.py"], { PYTHONHASHSEED: String(i) });
        if (r.category === "timeout" || r.category === "resource_limit" || r.category === "sandbox_transient") throw new SkillError(r.category, "effect check did not finish");
        const last = r.stdout.trim().split("\n").pop() ?? "";
        let j: Record<string, unknown>;
        try { j = JSON.parse(last); } catch { return inconclusive(`effect check printed no JSON (exit ${r.code}): ${r.stderr.slice(-200)}`); }
        if (typeof j.with_defect !== "number" || typeof j.without_defect !== "number") return inconclusive("effect check did not print with_defect/without_defect numbers");
        runs.push({ with: j.with_defect, without: j.without_defect });
      } finally { sb.remove(); }
    }
    const [a, b] = runs as [{ with: number; without: number }, { with: number; without: number }];
    if (Math.abs(a.with - b.with) > 1e-6 || Math.abs(a.without - b.without) > 1e-6) return inconclusive(`the measurement is not reproducible: (${a.with}, ${a.without}) vs (${b.with}, ${b.without})`);
    const diff = Math.abs(a.with - a.without);
    return { output: { runs }, confirmation: { tier: "measured", supports: diff >= T.material, note: `measured with defect ${a.with.toFixed(4)} vs corrected ${a.without.toFixed(4)} (Δ ${diff.toFixed(4)}; material ≥ ${T.material}); reproduced on a second fresh run` } };
  },
};

export const VERIFY_SKILLS: Skill[] = [explore, overlap, leak, metric, ckpt, nondet, claimV, evalPath, effect];
