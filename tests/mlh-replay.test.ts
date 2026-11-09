// Offline regression suite: every plan a planner actually wrote in a campaign,
// replayed through today's validator at $0. A validator change that would flip a
// recorded verdict shows up here before it costs a GPU round.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { cpuPrice, gpuPrice } from "../src/mlh/backends/modal.ts";
import { validateGpuPlan } from "../src/mlh/repro/gpu.ts";

const load = (p: string) => JSON.parse(readFileSync(`tests/fixtures/mlh-plans/${p}.json`, "utf8")) as { claims: never; drafts: { args: Record<string, unknown> }[]; files: string[] };
const check = (p: string, i: number) => {
  const f = load(p), files = new Set(f.files);
  return validateGpuPlan(f.claims, f.drafts[i]!.args, { budgetUsd: 5, price: (g: string) => gpuPrice(g) + cpuPrice(), exists: (x) => files.has(x.replace(/^\.\//, "")) });
};

describe("recorded planner drafts (campaign c1)", () => {
  it("no traversal and no whole-volume refs, even next to a valid fetch", () => {
    const f = load("simcse"), d = structuredClone(f.drafts[2]!.args) as { stages: { argv: string[] }[] };
    const files = new Set(f.files);
    const run = (arg: string) => { const x = structuredClone(d); x.stages[1]!.argv = [...x.stages[1]!.argv, arg]; return validateGpuPlan(f.claims, x, { budgetUsd: 5, price: gpuPrice, exists: (p) => files.has(p) }).errors.join(); };
    expect(run("/cache/runs/../data/other")).toMatch(/"\.\." is not allowed/);
    expect(run("/cache/data/unsup-simcse-bert-base-uncased/../../x")).toMatch(/"\.\." is not allowed/);
    expect(run("/cache/data")).toMatch(/\/cache\/data is not produced/);
    expect(run("/cache/data/unsup-simcse-bert-base-uncased")).toBe("");
  });

  it("detr: http and off-/cache/data fetches rejected; the corrected plan passes", () => {
    expect(check("detr", 0).errors.join()).toMatch(/must be https/);
    expect(check("detr", 1).plan?.fidelity).toBeDefined();
  });
  it("simcse: inline python rejected; a model directory filled by fetch items passes", () => {
    expect(check("simcse", 0).errors.join()).toMatch(/inline programs/);
    expect(check("simcse", 1).plan).toBeDefined();
    expect(check("simcse", 2).plan).toBeDefined();
  });
  it("graphsaint: output dir outside /cache/runs and an invented script are rejected with actionable messages", () => {
    expect(check("graphsaint", 0).errors.join()).toMatch(/inline programs/);
    expect(check("graphsaint", 1).errors.join()).toMatch(/\/cache\/gs_runs is not produced .* outputs .* under \/cache\/runs\//);
    expect(check("graphsaint", 2).errors.join()).toMatch(/tools\/setup_env\.py does not exist/);
  });
});
