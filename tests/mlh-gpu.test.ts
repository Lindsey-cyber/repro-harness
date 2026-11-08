// Multi-stage GPU plans: validation (structure, GPU allowlist, network rule,
// worst-case budget) and how stages become task nodes.
import { describe, expect, it } from "vitest";
import { apply, replay } from "../src/mlh/core/dag.ts";
import { Budget, gpuPrice } from "../src/mlh/backends/modal.ts";
import { stageNodes, validateGpuPlan } from "../src/mlh/repro/gpu.ts";
import { afterFail, initialNodes, N } from "../src/mlh/repro/flow.ts";

const claims = [{ id: "ap", metric: "AP", value: 0.42 }];
const base = {
  fidelity: "eval_only",
  stages: [
    { id: "fetch", argv: ["bash", "scripts/fetch.sh"], timeoutS: 900, network: true, persist: ["data/coco"] },
    { id: "eval", argv: ["python", "main.py", "--eval"], gpu: "T4", timeoutS: 1800, needs: ["fetch"], persist: [] },
  ],
  claims: [{ claim: "ap", stage: "eval", parse: "AP\\] IoU=0.50:0.95 .* = ([0-9.]+)" }],
};
const v = (raw: unknown, budgetUsd = 5) => validateGpuPlan(claims, raw, { budgetUsd, price: gpuPrice });

describe("GPU plans", () => {
  it("accepts a fetch → evaluate plan within budget and computes its worst case", () => {
    const r = v(base);
    expect(r.errors).toEqual([]);
    expect(r.plan!.worstCaseUsd).toBeCloseTo((0.1 * 1200) / 3600 + (0.59 * 2100) / 3600, 4);
  });

  it("rejects plans over budget, unknown GPUs, networked GPU stages, forward references and inline programs", () => {
    expect(v(base, 0.2).errors.join()).toMatch(/exceeds the budget/);
    expect(v({ ...base, stages: [{ ...base.stages[0], gpu: "T4" }, base.stages[1]] }).errors.join()).toMatch(/network access may not use a GPU/);
    expect(v({ ...base, stages: [base.stages[1], base.stages[0]] }).errors.join()).toMatch(/not an earlier stage/);
    expect(v({ ...base, stages: [base.stages[0], { ...base.stages[1], gpu: "B200" }] }).errors.join()).toMatch(/gpu B200 not in/);
    expect(v({ ...base, stages: [{ ...base.stages[0], argv: ["python", "-c", "import urllib"] }, base.stages[1]] }).errors.join()).toMatch(/inline programs/);
  });

  it("requires every /cache path to come from a declared fetch item, and records unknown data hosts", () => {
    const evalStage = { id: "eval", argv: ["python", "main.py", "--eval", "--coco_path", "/cache/coco"], gpu: "T4", timeoutS: 1800, persist: [] };
    expect(v({ ...base, stages: [evalStage] }).errors.join()).toMatch(/\/cache\/coco is not produced by any fetch item/);
    const ok = v({ ...base, stages: [{ ...evalStage, argv: ["python", "main.py", "--coco_path", "/cache/data/coco"] }], fetch: [{ url: "https://images.cocodataset.org/zips/val2017.zip", to: "/cache/data/coco", unpack: "zip" }, { url: "https://example.org/w.pth", to: "/cache/data/w.pth" }], checks: [{ path: "/cache/data/coco/val2017", minFiles: 5000 }] });
    expect(ok.errors).toEqual([]);
    expect(ok.plan!.unknownHosts).toEqual(["example.org"]);
    expect(v({ ...base, fetch: [{ url: "http://x.org/a", to: "/cache/data/a" }] }).errors.join()).toMatch(/must be https/);
    expect(v({ ...base, fetch: [{ url: "https://x.org/a", to: "/tmp/a" }] }).errors.join()).toMatch(/under \/cache\/data/);
  });

  it("maps a claim named by its description to its id when exactly one claim's value appears in it", () => {
    const two = [{ id: "mrpc", metric: "accuracy", value: 0.897 }, { id: "rte", metric: "accuracy", value: 0.866 }];
    const plan = { fidelity: "full", stages: [{ id: "m", argv: ["python", "run.py"], gpu: "T4", timeoutS: 600 }, { id: "r", argv: ["python", "run.py"], gpu: "T4", timeoutS: 600 }],
      claims: [{ claim: "RoBERTa-base LoRA MRPC dev accuracy 89.7 +/- 0.7", stage: "m", parse: "acc = ([0-9.]+)" }, { claim: "rte accuracy", stage: "r", parse: "acc = ([0-9.]+)" }] };
    const r = validateGpuPlan(two, plan, { budgetUsd: 5, price: gpuPrice });
    expect(r.errors).toEqual(['no stage measures claim rte (set "claim" to exactly one of: "mrpc", "rte")']);
    const ok = validateGpuPlan(two, { ...plan, claims: [plan.claims[0], { ...plan.claims[1], claim: "rte" }] }, { budgetUsd: 5, price: gpuPrice });
    expect(ok.plan!.claims.map((c) => c.claim)).toEqual(["mrpc", "rte"]);
  });

  it("assigns the cheapest GPU with enough memory when a stage states its need", () => {
    const r = v({ ...base, stages: [base.stages[0], { ...base.stages[1], gpu: "A100-40GB", gpuMemGB: 12 }] });
    expect(r.plan!.stages[1]!.gpu).toBe("T4");
    expect(v({ ...base, stages: [base.stages[0], { ...base.stages[1], gpu: "T4", gpuMemGB: 30 }] }).plan!.stages[1]!.gpu).toBe("A100-40GB");
  });

  it("refuses a run whose worst case exceeds what is left of the budget", () => {
    const b = new Budget(1);
    b.charge(0.9);
    expect(b.admit({ repo: "/x", cmd: ["x"], gpu: "H100", timeoutS: 600 })).toMatch(/exceeds the remaining/);
    expect(b.admit({ repo: "/x", cmd: ["x"], timeoutS: 600 })).toBeNull(); // CPU is nearly free
    // Parallel runs reserve their worst case: two 25-minute T4 jobs ($0.295 worst case each) fit in $0.50 one at a time, not together.
    const c = new Budget(0.5);
    const job = { repo: "/x", cmd: ["x"], gpu: "T4", timeoutS: 1500 };
    expect(c.admit(job)).toBeNull();
    expect(c.admit(job)).toMatch(/reserved by running jobs/);
    c.release(job);
    expect(c.admit(job)).toBeNull();
  });

  it("turns stages into nodes: per-seed stages fan out, the comparison waits for every seed", () => {
    const s = replay([{ t: "run", at: "t", objective: "o", source: "s", mode: "repro" }, ...initialNodes().map((node) => ({ t: "node" as const, at: "t", node, why: "" }))]);
    const plan = v({ ...base, stages: [base.stages[0], { ...base.stages[1], perSeed: true }], seeds: [0, 1] }).plan!;
    apply(s, { t: "ok", at: "t", id: N.paper, output: {}, findings: [], artifacts: [] });
    apply(s, { t: "node", at: "t", node: { id: N.claims, skill: N.claims, deps: [], purpose: "setup", params: {} }, why: "" });
    apply(s, { t: "ok", at: "t", id: N.claims, output: { claims }, findings: [], artifacts: [] });
    apply(s, { t: "node", at: "t", node: { id: N.plan, skill: N.plan, deps: [], purpose: "setup", params: {} }, why: "" });
    apply(s, { t: "ok", at: "t", id: N.plan, output: { gpuPlan: plan }, findings: [], artifacts: [] });
    const nodes = stageNodes(s, "");
    expect(nodes.map((n) => n.id)).toEqual(["repro.stage:fetch:v0", "repro.stage:eval:v0:s0", "repro.stage:eval:v0:s1", "repro.compare:ap:v0"]);
    expect(nodes[1]!.deps).toContain("repro.stage:fetch:v0");
    expect(nodes[3]!.deps).toEqual(["repro.stage:eval:v0:s0", "repro.stage:eval:v0:s1"]);
    for (const n of nodes) apply(s, { t: "node", at: "t", node: n, why: "" });
    apply(s, { t: "start", at: "t", id: "repro.stage:fetch:v0", attempt: 3 });
    apply(s, { t: "fail", at: "t", id: "repro.stage:fetch:v0", category: "experiment_error", message: "404" });
    expect(afterFail(s, "repro.stage:fetch:v0").map((n) => n.id)).toEqual(["repro.debug:stage-fetch:v0"]);
  });
});
