// Persistent task DAG as an event-sourced state machine. Pure: the journal (io)
// stores events, this module folds them into state and decides what runs next.
// Nodes may only depend on nodes that already exist, so the graph is acyclic by
// construction, and replanning is just appending nodes.
import type { ArtifactRef, Confirmation, FailureCategory, Finding, FindingDraft } from "./types.ts";
import { RETRYABLE } from "./types.ts";

export type Purpose = "setup" | "detect" | "confirm";
export type NodeStatus = "pending" | "running" | "succeeded" | "failed" | "skipped" | "held";

export interface TaskNode {
  id: string;
  skill: string;
  deps: string[];
  purpose: Purpose;
  params: Record<string, unknown>;
  // For confirm nodes: the finding they test.
  target?: string;
}

export type RunEvent =
  | { t: "run"; at: string; objective: string; source: string; mode: string }
  | { t: "node"; at: string; node: TaskNode; why: string }
  | { t: "start"; at: string; id: string; attempt: number }
  | { t: "ok"; at: string; id: string; output: unknown; findings: FindingDraft[]; artifacts: ArtifactRef[]; confirmation?: Omit<Confirmation, "node"> & { target: string } }
  | { t: "fail"; at: string; id: string; category: FailureCategory; message: string }
  | { t: "hold"; at: string; id: string; reason: string }
  | { t: "approve"; at: string; id: string; by: string }
  | { t: "skip"; at: string; id: string; reason: string }
  | { t: "gap"; at: string; skill: string; reason: string }
  | { t: "done"; at: string };

export interface NodeState {
  node: TaskNode;
  status: NodeStatus;
  attempts: number;
  failures: { category: FailureCategory; message: string }[];
  output?: unknown;
  artifacts: ArtifactRef[];
  reason?: string;
  approved: boolean;
}

export interface RunState {
  objective: string;
  source: string;
  mode: string;
  nodes: Map<string, NodeState>;
  order: string[];
  findings: Map<string, Finding>;
  // Skills the plan wanted but could not run, and why: the report's coverage section.
  gaps: { skill: string; reason: string }[];
  done: boolean;
  // Retry budget per node; failures beyond it are final.
  maxAttempts: number;
}

export function emptyState(maxAttempts = 3): RunState {
  return { objective: "", source: "", mode: "", nodes: new Map(), order: [], findings: new Map(), gaps: [], done: false, maxAttempts };
}

export function validateNode(s: RunState, n: TaskNode): string | null {
  if (s.nodes.has(n.id)) return `duplicate node id ${n.id}`;
  for (const d of n.deps) if (!s.nodes.has(d)) return `node ${n.id} depends on unknown node ${d}`;
  if (n.target !== undefined && !s.findings.has(n.target)) return `node ${n.id} targets unknown finding ${n.target}`;
  return null;
}

const need = (s: RunState, id: string): NodeState => {
  const ns = s.nodes.get(id);
  if (!ns) throw new Error(`event for unknown node ${id}`);
  return ns;
};

// Folds one event into the state (mutates and returns it, for cheap replay).
export function apply(s: RunState, e: RunEvent): RunState {
  switch (e.t) {
    case "run":
      s.objective = e.objective; s.source = e.source; s.mode = e.mode;
      break;
    case "node": {
      const err = validateNode(s, e.node);
      if (err) throw new Error(err);
      s.nodes.set(e.node.id, { node: e.node, status: "pending", attempts: 0, failures: [], artifacts: [], approved: false });
      s.order.push(e.node.id);
      break;
    }
    case "start": {
      const ns = need(s, e.id);
      ns.status = "running"; ns.attempts = e.attempt;
      break;
    }
    case "ok": {
      const ns = need(s, e.id);
      ns.status = "succeeded"; ns.output = e.output; ns.artifacts = e.artifacts;
      // Findings raised by confirm nodes are evidence about another finding, not new findings.
      if (ns.node.purpose !== "confirm") {
        e.findings.forEach((f, i) => {
          const id = `${e.id}/${i}`;
          s.findings.set(id, { ...f, id, node: e.id, status: f.kind === "fact" ? "confirmed" : "suspected", confirmations: [] });
        });
      }
      if (e.confirmation) {
        const f = s.findings.get(e.confirmation.target);
        if (f) {
          f.confirmations.push({ node: e.id, supports: e.confirmation.supports, note: e.confirmation.note });
          if (f.kind === "suspicion") f.status = resolve(f.confirmations);
        }
      }
      break;
    }
    case "fail": {
      const ns = need(s, e.id);
      ns.failures.push({ category: e.category, message: e.message });
      const retry = RETRYABLE.includes(e.category) && ns.attempts < s.maxAttempts;
      ns.status = retry ? "pending" : "failed";
      break;
    }
    case "hold":
      need(s, e.id).status = "held"; need(s, e.id).reason = e.reason;
      break;
    case "approve": {
      const ns = need(s, e.id);
      ns.approved = true;
      if (ns.status === "held") ns.status = "pending";
      break;
    }
    case "skip":
      need(s, e.id).status = "skipped"; need(s, e.id).reason = e.reason;
      break;
    case "gap":
      s.gaps.push({ skill: e.skill, reason: e.reason });
      break;
    case "done":
      s.done = true;
      break;
  }
  return s;
}

export function replay(events: RunEvent[], maxAttempts = 3): RunState {
  const s = emptyState(maxAttempts);
  for (const e of events) apply(s, e);
  // A node that was running when the process died gets another attempt; skills
  // run in fresh sandboxes, so re-running is idempotent.
  for (const ns of s.nodes.values()) if (ns.status === "running") ns.status = ns.attempts < maxAttempts ? "pending" : "failed";
  return s;
}

export function resolve(cs: Confirmation[]): Finding["status"] {
  const yes = cs.some((c) => c.supports), no = cs.some((c) => !c.supports);
  return yes && no ? "disputed" : yes ? "confirmed" : no ? "refuted" : "suspected";
}

export function ready(s: RunState): TaskNode[] {
  return s.order
    .map((id) => s.nodes.get(id)!)
    .filter((ns) => ns.status === "pending" && ns.node.deps.every((d) => s.nodes.get(d)?.status === "succeeded"))
    .map((ns) => ns.node);
}

// Pending nodes that can never run because a dependency ended without success.
export function blocked(s: RunState): { id: string; reason: string }[] {
  const out: { id: string; reason: string }[] = [];
  for (const id of s.order) {
    const ns = s.nodes.get(id)!;
    if (ns.status !== "pending") continue;
    const bad = ns.node.deps.find((d) => ["failed", "skipped"].includes(s.nodes.get(d)!.status));
    if (bad) out.push({ id, reason: `dependency ${bad} ${s.nodes.get(bad)!.status}` });
  }
  return out;
}

export function settled(s: RunState): boolean {
  return [...s.nodes.values()].every((ns) => ["succeeded", "failed", "skipped", "held"].includes(ns.status));
}

export const depOf = (s: RunState, node: TaskNode, skill: string): NodeState | undefined =>
  node.deps.map((d) => s.nodes.get(d)!).find((ns) => ns.node.skill === skill);
