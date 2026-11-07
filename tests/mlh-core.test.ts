import { describe, expect, it } from "vitest";
import { authorize } from "../src/mlh/core/authz.ts";
import { apply, blocked, emptyState, ready, replay, type RunEvent, type TaskNode } from "../src/mlh/core/dag.ts";
import { confirmations, expand, intentsOf } from "../src/mlh/core/planner.ts";
import { compact } from "../src/mlh/core/report.ts";
import { initialTrust, promotable, promote, recordAudit, PROMOTION_STREAK } from "../src/mlh/core/trust.ts";
import { SKILLS } from "../src/mlh/skills/index.ts";

const at = "2026-11-07T00:00:00Z";
const node = (id: string, deps: string[] = [], extra: Partial<TaskNode> = {}): RunEvent => ({ t: "node", at, node: { id, skill: id, deps, purpose: "detect", params: {}, ...extra }, why: "test" });

describe("task DAG", () => {
  it("rejects unknown dependencies, so the graph stays acyclic by construction", () => {
    const s = emptyState();
    expect(() => apply(s, node("b", ["a"]))).toThrow(/unknown node a/);
    apply(s, node("a"));
    expect(() => apply(s, node("a"))).toThrow(/duplicate/);
  });

  it("runs a node only after its dependencies succeed, and skips it when one fails for good", () => {
    const s = replay([node("a"), node("b", ["a"])], 2);
    expect(ready(s).map((n) => n.id)).toEqual(["a"]);
    apply(s, { t: "start", at, id: "a", attempt: 1 });
    apply(s, { t: "fail", at, id: "a", category: "experiment_error", message: "boom" });
    expect(s.nodes.get("a")!.status).toBe("failed"); // not retryable
    expect(blocked(s)).toEqual([{ id: "b", reason: "dependency a failed" }]);
  });

  it("retries only retryable failures, within the attempt budget", () => {
    const s = replay([node("a")], 2);
    apply(s, { t: "start", at, id: "a", attempt: 1 });
    apply(s, { t: "fail", at, id: "a", category: "sandbox_transient", message: "x" });
    expect(s.nodes.get("a")!.status).toBe("pending");
    apply(s, { t: "start", at, id: "a", attempt: 2 });
    apply(s, { t: "fail", at, id: "a", category: "sandbox_transient", message: "x" });
    expect(s.nodes.get("a")!.status).toBe("failed");
  });

  it("re-queues a node that was running when the process died", () => {
    const s = replay([node("a"), { t: "start", at, id: "a", attempt: 1 }]);
    expect(s.nodes.get("a")!.status).toBe("pending");
  });

  it("confirms a suspicion only on independent support, and marks conflicting evidence disputed", () => {
    const s = replay([node("d"), { t: "ok", at, id: "d", output: {}, artifacts: [], findings: [{ category: "c", severity: "high", kind: "suspicion", summary: "s", evidence: {} }] }]);
    expect(s.findings.get("d/0")!.status).toBe("suspected");
    apply(s, node("k1", ["d"], { purpose: "confirm", target: "d/0" }));
    apply(s, { t: "ok", at, id: "k1", output: {}, artifacts: [], findings: [{ category: "ignored", severity: "low", kind: "fact", summary: "", evidence: {} }], confirmation: { target: "d/0", supports: true, note: "" } });
    expect(s.findings.get("d/0")!.status).toBe("confirmed");
    expect(s.findings.has("k1/0")).toBe(false); // confirm nodes add evidence, not findings
    apply(s, node("k2", ["d"], { purpose: "confirm", target: "d/0" }));
    apply(s, { t: "ok", at, id: "k2", output: {}, artifacts: [], findings: [], confirmation: { target: "d/0", supports: false, note: "" } });
    expect(s.findings.get("d/0")!.status).toBe("disputed");
  });

  it("held nodes resume only after a human approval event", () => {
    const s = replay([node("a"), { t: "hold", at, id: "a", reason: "high risk" }]);
    expect(ready(s)).toEqual([]);
    apply(s, { t: "approve", at, id: "a", by: "human" });
    expect(ready(s).map((n) => n.id)).toEqual(["a"]);
    expect(compact(s)).toContain("a=pending");
  });
});

describe("authorization", () => {
  const det = { kind: "deterministic" as const, check: "selftest", ok: true };
  it("allows medium-risk sandboxed execution only on a passing deterministic check", () => {
    expect(authorize("exec_sandboxed", [det]).allow).toBe(true);
    expect(authorize("exec_sandboxed", [{ ...det, ok: false }])).toMatchObject({ allow: false, hold: true });
    expect(authorize("exec_sandboxed", [])).toMatchObject({ allow: false });
  });
  it("never lets a model's approval authorize a high-risk action", () => {
    const d = authorize("exec_unsandboxed", [det, { kind: "llm", check: "verifier", ok: true }]);
    expect(d).toMatchObject({ allow: false, hold: true });
    expect(authorize("exec_unsandboxed", [{ kind: "human", check: "approval", ok: true }]).allow).toBe(true);
    expect(authorize("network_egress", [{ kind: "human", check: "approval", ok: true, action: "exec_unsandboxed" }]).allow).toBe(false);
  });
  it("refuses denied actions even with a human approval", () => {
    expect(authorize("publish", [{ kind: "human", check: "approval", ok: true }])).toMatchObject({ allow: false, hold: false });
  });
});

describe("trust", () => {
  it("demotes on one bad audit and promotes only after a streak plus a human", () => {
    let t = initialTrust(at);
    for (let i = 0; i < PROMOTION_STREAK - 1; i++) t = recordAudit(t, { at, falsePositives: 0, misses: 0, source: "bench" });
    expect(promotable(t)).toBe(false);
    t = recordAudit(t, { at, falsePositives: 0, misses: 0, source: "bench" });
    expect(promotable(t)).toBe(true);
    t = promote(t, { by: "human", at });
    expect(t.level).toBe("auto");
    t = recordAudit(t, { at, falsePositives: 1, misses: 0, source: "bench" });
    expect(t).toMatchObject({ level: "confirm", streak: 0 });
  });
});

describe("planner", () => {
  const full = { train: ["python3", "train.py"], eval: ["python3", "eval.py"], data: { train: "a.csv", test: "b.csv", label: "label" }, outputs: { metrics: "m.json", predictions: "p.csv", checkpoint: "c.npz", evalMetrics: "e.json" }, metric: "accuracy", claim: { metric: "accuracy", value: 0.8 } };

  it("plans only skills the manifest supports and records the rest as coverage gaps", () => {
    const x = expand("check for bugs and reproduce", { train: ["python3", "t.py"], outputs: { metrics: "m.json" } }, SKILLS);
    expect(x.nodes.map((n) => n.skill)).toEqual(expect.arrayContaining(["run.train", "repro.rerun"]));
    expect(x.nodes.map((n) => n.skill)).not.toContain("eval.recompute");
    expect(x.gaps.map((g) => g.skill)).toEqual(expect.arrayContaining(["eval.recompute", "data.audit", "ckpt.roundtrip"]));
  });

  it("follows the objective: a pure reproduction objective skips the data audit", () => {
    expect(intentsOf("Reproduce the reported accuracy")).toEqual(["reproduce"]);
    expect(expand("Reproduce the reported accuracy", full, SKILLS).nodes.map((n) => n.skill)).not.toContain("data.audit");
  });

  it("adds a designated confirmer for a suspicion, else replicates the detector on a fresh run", () => {
    const x = expand("audit", full, SKILLS);
    const s = replay([{ t: "run", at, objective: "audit", source: "x", mode: "harness" }, node("repo.inspect", [], { purpose: "setup" }), ...x.nodes.map((n) => ({ t: "node" as const, at, node: n, why: "" }))]);
    const raise = (id: string, category: string) => apply(s, { t: "ok", at, id, output: {}, artifacts: [], findings: [{ category, severity: "high", kind: "suspicion", summary: "", evidence: {} }] });
    raise("pipeline.gap_check", "pipeline.eval_path_mismatch");
    expect(confirmations(s, "pipeline.gap_check/0", SKILLS, full, () => "confirm").map((n) => n.skill)).toEqual(["pipeline.split_swap"]);
    raise("eval.recompute", "eval.metric_mismatch");
    const rep = confirmations(s, "eval.recompute/0", SKILLS, full, () => "confirm");
    expect(rep.map((n) => n.skill)).toEqual(["run.train", "eval.recompute"]);
    expect(rep[1]!.deps).toContain(rep[0]!.id);
    expect(confirmations(s, "eval.recompute/0", SKILLS, full, () => "auto")).toEqual([]); // trusted: no replication
  });

  it("every skill is versioned, has an output contract and a known purpose", () => {
    for (const s of SKILLS) {
      expect(s.version).toMatch(/^\d+\.\d+\.\d+$/);
      expect(typeof s.output.safeParse).toBe("function");
      for (const a of s.after) expect(SKILLS.map((x) => x.id)).toContain(a);
    }
    expect(new Set(SKILLS.map((s) => s.id)).size).toBe(SKILLS.length);
  });
});
