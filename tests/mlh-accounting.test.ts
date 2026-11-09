// P0: failure attribution, cost accounting, per-attempt Modal spend, ledger, stop conditions.
import { cpSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { Budget, costScope, UNSETTLED } from "../src/mlh/backends/modal.ts";
import { costReport, failureClass, stopCheck } from "../src/mlh/repro/accounting.ts";
import { modalSpent } from "../src/mlh/repro/campaign.ts";

const tmp = () => mkdtempSync(join(tmpdir(), "mlh-acct-"));

describe("failure classes", () => {
  it.each([
    ["sandbox_transient", "repro.stage:train:v0", "", "infrastructure"],
    ["provider_transient", "repro.debug:stage-x:v0", "Insufficient capacity", "infrastructure"],
    ["provider_fatal", "repro.stage:train:v0", "spend limit", "resource"],
    ["resource_limit", "repro.stage:train:v0", "CUDA out of memory", "resource"],
    ["policy_denied", "repro.stage:train:v0", "budget: worst case $3 exceeds", "resource"],
    ["policy_denied", "repro.patch:c", "high-risk patch", "agent"],
    ["contract_violation", "repro.plan", "no valid stage plan after 3 attempts", "agent"],
    ["experiment_error", "repro.env", "environment agent ran out of steps", "agent"],
    ["experiment_error", "repro.stage:train:v0", "Traceback", "paper"],
    ["timeout", "repro.stage:train:v0", "", "paper"],
    ["precondition", "repro.paper", "arXiv PDF download failed", "infrastructure"],
    ["harness_bug", "repro.compare:c:v0", "TypeError", "unclassified"],
    ["something_new", "repro.stage:x:v0", "", "unclassified"],
  ])("%s at %s -> %s", (cat, id, msg, want) => expect(failureClass(cat, id, msg)).toBe(want));
});

describe("cost report", () => {
  const rows = [
    { id: "repro.env", attempt: 1, outcome: "fail", category: "sandbox_transient", costUsd: 0.01, gpuUsd: 0.02, ms: 60000 },
    { id: "repro.env", attempt: 2, outcome: "ok", costUsd: 0.05, gpuUsd: 0.1, ms: 600000 },
    { id: "repro.plan", attempt: 1, outcome: "fail", category: "contract_violation", costUsd: 0.04, ms: 120000 },
    { id: "repro.stage:train:v0", attempt: 1, outcome: "fail", category: "experiment_error", costUsd: 0, gpuUsd: 0.3, ms: 60000 },
  ];
  it("splits productive, retried, failed and killed spend by class", () => {
    const r = costReport(rows, { "repro.stage:eval:v0#1": 0.2 });
    expect(r.productive.usd).toBeCloseTo(0.15);
    expect(r.retried.infrastructure.usd).toBeCloseTo(0.03); // env failed once, then succeeded
    expect(r.failed.agent.usd).toBeCloseTo(0.04);
    expect(r.failed.paper.usd).toBeCloseTo(0.3);
    expect(r.killed.usd).toBeCloseTo(0.2); // spent by an attempt that left no outcome
    expect(r.totalUsd).toBeCloseTo(0.72);
    expect(r.wasteShare).toBeCloseTo(1 - 0.15 / 0.72);
  });
});

describe("stop conditions", () => {
  const none = costReport([]);
  it("an unclassified failure stops at once", () => {
    expect(stopCheck([{ id: "repro.compare:c:v0", category: "harness_bug", message: "TypeError: x" }], none)?.reason).toBe("unclassified_failure");
  });
  it("the same infrastructure fault twice stops, even with different ids; different faults do not", () => {
    const cap = (id: string, sb: string) => ({ id, category: "provider_transient", message: `RuntimeError: ResourceExhaustedError: Insufficient capacity to create sandbox ${sb}.` });
    expect(stopCheck([cap("repro.stage:a:v0", "sb-AAAA1111")], none)).toBeUndefined();
    expect(stopCheck([cap("repro.stage:a:v0", "sb-AAAA1111"), cap("repro.stage:b:v0", "sb-BBBB2222")], none)?.reason).toBe("infra_repeat");
    expect(stopCheck([cap("repro.stage:a:v0", "x"), { id: "repro.env", category: "sandbox_transient", message: "ConnectionError: reset" }], none)).toBeUndefined();
  });
  it("waste over 30 % of spend stops, but the paper's own code failures are not waste", () => {
    const fail = (cls: string) => costReport([{ id: "repro.stage:t:v0", attempt: 1, outcome: "fail", category: cls, gpuUsd: 0.5 }, { id: "repro.stage:t:v0", attempt: 2, outcome: "ok", gpuUsd: 0.5 }]);
    expect(stopCheck([], fail("sandbox_transient"))?.reason).toBe("waste");
    expect(stopCheck([], fail("experiment_error"))).toBeUndefined();
  });
  it("the LLM cap stops", () => {
    expect(stopCheck([], costReport([{ id: "repro.plan", outcome: "ok", costUsd: 5 }]))?.reason).toBe("llm_cap");
  });
});

describe("Modal spend per attempt", () => {
  it("attributes concurrent charges to their own attempt and survives a reload", async () => {
    const f = join(tmp(), "budget.json"), b = new Budget(5, f);
    await Promise.all([
      costScope.run("repro.stage:a:v0#1", async () => { await new Promise((r) => setTimeout(r, 10)); b.charge(0.3); }),
      costScope.run("repro.stage:b:v0#1", async () => { b.charge(0.1); await new Promise((r) => setTimeout(r, 5)); b.charge(0.1); }),
    ]);
    b.charge(0.05);
    const again = new Budget(5, f);
    expect(again.byAttempt).toEqual({ "repro.stage:a:v0#1": 0.3, "repro.stage:b:v0#1": 0.2, "(outside any node)": 0.05 });
  });
  it("reservations left by a killed process are spent and shown as unsettled", () => {
    const f = join(tmp(), "budget.json");
    writeFileSync(f, JSON.stringify({ spent: 1, reserved: 0.4, byAttempt: { "x#1": 1 } }));
    const b = new Budget(5, f);
    expect(b.spent).toBeCloseTo(1.4);
    expect(b.byAttempt[UNSETTLED]).toBeCloseTo(0.4);
  });
});

describe("Modal ledger", () => {
  it("counts a run copied to another round once", () => {
    const root = tmp();
    mkdirSync(join(root, "c1", "lora"), { recursive: true });
    writeFileSync(join(root, "c1", "lora", "run.json"), '{"source":"x","created":"t"}');
    writeFileSync(join(root, "c1", "lora", "budget.json"), '{"spent":2.72}');
    cpSync(join(root, "c1", "lora"), join(root, "c2", "lora"), { recursive: true });
    writeFileSync(join(root, "c2", "lora", "budget.json"), '{"spent":3.1}'); // the copy continued
    mkdirSync(join(root, "c1", "detr"));
    writeFileSync(join(root, "c1", "detr", "run.json"), '{"source":"y"}');
    writeFileSync(join(root, "c1", "detr", "budget.json"), '{"spent":0.46}');
    expect(modalSpent(root)).toBeCloseTo(3.56);
    expect(readFileSync(join(root, "c2", "lora", "run.json"), "utf8")).toContain("x");
  });
});

describe("LLM spend of a killed attempt", () => {
  it("is counted from the per-call log, as killed", () => {
    const r = costReport([{ id: "repro.env", attempt: 2, outcome: "ok", costUsd: 0.05 }], {}, {}, { "repro.env#1": 0.03, "repro.env#2": 0.05 });
    expect(r.killed.llmUsd).toBeCloseTo(0.03);
    expect(r.productive.llmUsd).toBeCloseTo(0.05);
  });
});
