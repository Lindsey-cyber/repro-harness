// Capability-driven planning. Pure. The plan is not a fixed script: it starts
// with one inspection node, expands once the manifest is known (only skills
// whose declared needs the repository satisfies, filtered by the objective's
// intents), and grows again whenever a skill raises a suspicion that needs
// independent confirmation. Nothing here knows about any benchmark task.
import type { z } from "zod";
import type { Action } from "./authz.ts";
import type { Purpose, RunState, TaskNode } from "./dag.ts";
import type { Manifest } from "./types.ts";
import type { TrustLevel } from "./trust.ts";

export type Intent = "reproduce" | "audit";

export interface SkillSpec {
  id: string;
  version: string;
  summary: string;
  purpose: Purpose;
  actions: Action[];
  intents: Intent[];
  // Why the skill cannot run on this manifest, or null when it can.
  needs: (m: Manifest) => string | null;
  // Skills whose node must succeed first; their outputs are this skill's inputs.
  after: string[];
  // Finding categories this skill independently tests (a designated confirmer).
  confirms?: string[];
  // Confirms any suspicion whose evidence.origin is this (e.g. "llm"): such
  // suspicions carry none of the detector-specific evidence category confirmers read.
  confirmsOrigin?: string;
  // Re-running the experiment from scratch (a fresh run is a replication).
  fresh?: boolean;
  // Its suspicions can be confirmed by re-running it on a fresh experiment run.
  replicable?: boolean;
  params?: z.ZodType;
  output: z.ZodType;
}

export const INSPECT = "repo.inspect";
export const AGENT = "agent.explore";
export const EFFECT = "verify.effect";

// Where an exploring agent's finding is independently tested. Each confirmer
// measures whether the claimed defect materially changes the result with harness
// code; a category without one, or a confirmer that cannot decide, falls back to
// the effect check (the agent's own measurement, rerun and judged by the harness).
export const AGENT_ROUTES: Record<string, string> = {
  "data.train_test_overlap": "verify.overlap",
  "data.target_leak": "verify.leak",
  "eval.metric_mismatch": "verify.metric",
  "ckpt.roundtrip_mismatch": "verify.ckpt",
  "repro.nondeterministic": "verify.nondeterminism",
  "repro.claim_not_met": "verify.claim",
  "pipeline.eval_path_mismatch": "verify.eval_path",
};

export function intentsOf(objective: string): Intent[] {
  const o = objective.toLowerCase();
  const out: Intent[] = [];
  if (/reproduc|replicat|claim|report(ed)? (accuracy|result|number|metric)/.test(o)) out.push("reproduce");
  if (/debug|bug|error|leak|check|audit|validat|wrong|correct/.test(o)) out.push("audit");
  return out.length ? out : ["reproduce", "audit"];
}

export function initialPlan(): TaskNode[] {
  return [{ id: INSPECT, skill: INSPECT, deps: [], purpose: "setup", params: {} }];
}

export interface Expansion {
  nodes: TaskNode[];
  gaps: { skill: string; reason: string }[];
}

// Expands the plan once the manifest is known.
export function expand(objective: string, manifest: Manifest, specs: SkillSpec[]): Expansion {
  const want = intentsOf(objective);
  const nodes: TaskNode[] = [];
  const gaps: Expansion["gaps"] = [];
  const placed = new Map<string, string>([[INSPECT, INSPECT]]);
  // Specs are visited in dependency order so `after` always resolves to a placed node.
  const pending = specs.filter((s) => s.id !== INSPECT && s.purpose !== "confirm" && s.intents.some((i) => want.includes(i)));
  let progress = true;
  while (pending.length && progress) {
    progress = false;
    for (let i = 0; i < pending.length; i++) {
      const s = pending[i]!;
      const missing = s.after.find((a) => !placed.has(a));
      if (missing && pending.some((p) => p.id === missing)) continue; // wait for it
      pending.splice(i--, 1);
      progress = true;
      const why = missing ? `needs ${missing}, which cannot run` : s.needs(manifest);
      if (why) { gaps.push({ skill: s.id, reason: why }); continue; }
      nodes.push({ id: s.id, skill: s.id, deps: [INSPECT, ...s.after.map((a) => placed.get(a)!)].filter((d, k, a) => a.indexOf(d) === k), purpose: s.purpose, params: {} });
      placed.set(s.id, s.id);
    }
  }
  for (const s of pending) gaps.push({ skill: s.id, reason: "unresolvable dependency order" });
  return { nodes, gaps };
}

// Nodes that would independently test a suspicion. Designated confirmers come
// first; otherwise, if the trust level asks for it, the detector is replicated
// on a fresh experiment run. Returns no nodes when nothing can test it, so the
// finding stays "suspected" and is reported as unverified.
export function confirmations(s: RunState, findingId: string, specs: SkillSpec[], manifest: Manifest, trust: (skill: string) => TrustLevel): TaskNode[] {
  const f = s.findings.get(findingId);
  if (!f || f.kind !== "suspicion") return [];
  const detector = s.nodes.get(f.node)!.node;
  const spec = (id: string) => specs.find((x) => x.id === id);
  const tag = findingId.replace(/[^\w.-]+/g, "_");
  const out: TaskNode[] = [];
  const origin = typeof f.evidence.origin === "string" ? f.evidence.origin : undefined;
  if (origin === "agent") {
    const route = AGENT_ROUTES[f.category];
    const c = route ? spec(route) : undefined;
    const id = c && c.needs(manifest) === null ? c.id : EFFECT;
    return spec(id) ? [{ id: `${id}@${tag}`, skill: id, deps: [INSPECT, detector.id], purpose: "confirm", params: {}, target: findingId }] : [];
  }
  if (origin) {
    return specs.filter((x) => x.confirmsOrigin === origin && x.needs(manifest) === null)
      .map((c) => ({ id: `${c.id}@${tag}`, skill: c.id, deps: [INSPECT, detector.id], purpose: "confirm" as const, params: {}, target: findingId }));
  }
  for (const c of specs.filter((x) => x.confirms?.includes(f.category) && x.needs(manifest) === null)) {
    // A confirmer reads the same inputs the detector read, plus the detector's own output.
    const deps = [...new Set([INSPECT, ...c.after.map((a) => (a === detector.skill ? detector.id : findNode(s, a))).filter((d): d is string => !!d)])];
    out.push({ id: `${c.id}@${tag}`, skill: c.id, deps, purpose: "confirm", params: {}, target: findingId });
  }
  if (out.length || !spec(detector.skill)?.replicable || trust(detector.skill) === "auto") return out;
  // Replication: fresh instances of the experiment runs the detector depends on, then the detector again.
  const deps: string[] = [];
  for (const d of detector.deps) {
    const dn = s.nodes.get(d)!.node;
    if (spec(dn.skill)?.fresh) {
      const id = `${dn.skill}@${tag}`;
      out.push({ id, skill: dn.skill, deps: dn.deps, purpose: "confirm", params: { ...dn.params, replicaOf: dn.id }, target: findingId });
      deps.push(id);
    } else deps.push(d);
  }
  out.push({ id: `${detector.skill}@${tag}`, skill: detector.skill, deps, purpose: "confirm", params: { ...detector.params, replicaOf: detector.id, category: f.category }, target: findingId });
  return out;
}

const findNode = (s: RunState, skill: string): string | undefined => s.order.find((id) => s.nodes.get(id)!.node.skill === skill && s.nodes.get(id)!.node.purpose !== "confirm");

// After a confirmer could not decide on an agent's finding: the effect check, once.
export function fallback(s: RunState, findingId: string, specs: SkillSpec[]): TaskNode[] {
  const f = s.findings.get(findingId);
  if (!f || f.evidence.origin !== "agent" || !specs.some((x) => x.id === EFFECT)) return [];
  const id = `${EFFECT}@${findingId.replace(/[^\w.-]+/g, "_")}`;
  if (s.nodes.has(id)) return [];
  return [{ id, skill: EFFECT, deps: [INSPECT, f.node], purpose: "confirm", params: {}, target: findingId }];
}
