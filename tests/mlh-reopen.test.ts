// Resume after a platform failure, replayed on lora's real c1 journal (copied as a fixture):
// its stages failed on the Modal spend limit and its debuggers on a capacity shortage.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { apply, reopenPlatform, replay, type RunEvent } from "../src/mlh/core/dag.ts";

const events = readFileSync("tests/fixtures/lora-c1-journal.jsonl", "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l) as RunEvent);

describe("reopen after platform failures", () => {
  it("reruns the stages, skips debuggers called for platform faults, brings back the comparisons", () => {
    const s = replay(events, 3);
    expect(s.nodes.get("repro.stage:rte_lora:v0")!.status).toBe("failed");
    const ev = reopenPlatform(s, "t");
    expect(s.nodes.get("repro.stage:rte_lora:v0")!.status).toBe("pending");
    expect(s.nodes.get("repro.stage:mrpc_lora:v0")!.status).toBe("pending");
    expect(s.nodes.get("repro.debug:stage-rte_lora:v0")!.status).toBe("skipped");
    expect(s.nodes.get("repro.compare:roberta_base_lora_rte_accuracy:v0")!.status).toBe("pending");
    // The events replay to the same state (they are journaled, so a later resume agrees).
    const again = replay([...events, ...ev], 3);
    for (const [id, ns] of s.nodes) expect(again.nodes.get(id)!.status, id).toBe(ns.status);
  });

  it("leaves work failures alone", () => {
    const s = replay(events, 3);
    apply(s, { t: "node", at: "t", node: { id: "x", skill: "repro.stage", deps: [], purpose: "detect", params: {} }, why: "test" } as never);
    for (let i = 0; i < 3; i++) { apply(s, { t: "start", at: "t", id: "x", attempt: i + 1 }); apply(s, { t: "fail", at: "t", id: "x", category: "experiment_error", message: "Traceback" }); }
    reopenPlatform(s, "t");
    expect(s.nodes.get("x")!.status).toBe("failed");
  });
});

describe("operator reopen after a harness fix", async () => {
  const { reopenByOperator } = await import("../src/mlh/core/dag.ts");
  it("reopens the named failed stages, skips their debuggers, brings back their comparisons, leaves the rest", () => {
    const s = replay(events, 3);
    reopenPlatform(s, "t"); // start from lora's state after the platform reopen
    // fail the reopened rte stage by name, as a harness bug would
    for (let i = 0; i < 3; i++) { apply(s, { t: "start", at: "t", id: "repro.stage:rte_lora:v0", attempt: i + 1 }); apply(s, { t: "fail", at: "t", id: "repro.stage:rte_lora:v0", category: "experiment_error", message: "config {seed}.toml not found" }); }
    const ev = reopenByOperator(s, "repro.stage:rte_lora", "{seed} substitution fixed", "t");
    expect(s.nodes.get("repro.stage:rte_lora:v0")!.status).toBe("pending");
    expect(ev.some((e) => e.t === "reopen" && "reason" in e && /operator: \{seed\} substitution fixed/.test(e.reason))).toBe(true);
    expect(s.nodes.get("repro.stage:mrpc_lora:v0")!.status).toBe("pending"); // untouched by the operator (pending from before)
  });
});
