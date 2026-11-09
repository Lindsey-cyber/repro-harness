// Reading the claim's number from a stage's output. DETR (05:21Z): the planner's pattern matched both
// the AP and the AR@100 line of COCO's summary and the harness took the last match (0.574 for 0.420).
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { disambiguate, metricMatches, metricSeries, parseMetric } from "../src/mlh/repro/skills.ts";

const COCO = readFileSync("tests/fixtures/detr-coco-summary.txt", "utf8");
const PARSE = "IoU=0\\.50:0\\.95 \\| area=   all \\| maxDets=100 \\] = (0\\.[0-9]+)"; // the planner's, verbatim
const claim = { id: "detr_r50_coco_val2017_box_ap", metric: "box AP", value: 0.42, dataset: "COCO val2017" } as never;

describe("metric extraction", () => {
  it("the recorded DETR pattern matches two different values: no number is returned", () => {
    expect(metricMatches(PARSE, COCO).map((m) => m.raw)).toEqual([0.42, 0.574]);
    expect(parseMetric({ claim: "c", argv: [], seeds: [], parse: PARSE, timeoutS: 0 }, "box AP", COCO)).toBeUndefined();
  });
  it("FT-Transformer's per-epoch test lines (other numbers on the line differ) are one series: the final value", () => {
    const FT = readFileSync("tests/fixtures/ft-transformer-ca-seed0.txt", "utf8");
    const p = "\\[test \\] score = -?[0-9.]+ \\| rmse = ([0-9.]+)"; // the planner's, verbatim
    const series = metricSeries(metricMatches(p, FT));
    expect(series).toHaveLength(1);
    expect(series[0]!.label).toBe("[test ] score = <n> | rmse = <value>");
    expect(parseMetric({ claim: "c", argv: [], seeds: [], parse: p, timeoutS: 0 }, "rmse", FT)).toBe(0.459);
    expect(metricSeries(metricMatches(PARSE, COCO))).toHaveLength(2); // COCO's AP and AR lines stay apart
  });
  it("FT-Transformer Adult lines (other metrics with 2 or 3 decimals on the line) are one series too", () => {
    const AD = readFileSync("tests/fixtures/ft-transformer-adult-seed0.txt", "utf8");
    const p = "\\[test \\] score = [0-9.]+ \\| acc = ([0-9.]+)"; // the planner's, verbatim
    const series = metricSeries(metricMatches(p, AD));
    expect(series.map((x) => x.label)).toEqual(["[test ] score = <n> | acc = <value> | mf1 = <n> | ms = <n> | wf1 = <n> | ws = <n> | roc_auc = <n>"]);
    expect(series[0]!.last).toBe(0.858);
  });
  it("one metric logged every epoch is a series: the final value counts", () => {
    expect(parseMetric({ claim: "c", argv: [], seeds: [], parse: "val acc: ([0-9.]+)", timeoutS: 0 }, "accuracy", "epoch 1 done\nval acc: 0.61\nepoch 2 done\nval acc: 0.83")).toBe(0.83);
  });
  it("the same value repeated is not ambiguous", () => {
    expect(parseMetric({ claim: "c", argv: [], seeds: [], parse: "acc: ([0-9.]+)", timeoutS: 0 }, "accuracy", "acc: 0.9\nacc: 0.9")).toBe(0.9);
  });
  it("disambiguation shows the model labels only, never the values, and records its choice", async () => {
    let prompt = "";
    const model = { async json<T>(_s: string, u: string) { prompt = u; return { index: 0, why: "the (AP) line" } as T; } };
    const d = await disambiguate(model, claim, PARSE, metricSeries(metricMatches(PARSE, COCO)));
    expect(d.index).toBe(0);
    expect(prompt).not.toMatch(/0\.420|0\.574|0\.42\b/);
    expect(prompt).toMatch(/Average Precision {2}\(AP\) @\[ IoU=0\.50:0\.95 \| area= {3}all \| maxDets=100 \] = <value>/);
    expect(d.options).toHaveLength(2);
  });
  it("no model or no choice: no index (the stage then fails by name, never guesses)", async () => {
    expect((await disambiguate(undefined, claim, PARSE, metricSeries(metricMatches(PARSE, COCO)))).index).toBeUndefined();
    const none = { async json<T>() { return { index: null, why: "neither" } as T; } };
    expect((await disambiguate(none, claim, PARSE, metricSeries(metricMatches(PARSE, COCO)))).index).toBeUndefined();
  });
});

describe("a parse pattern may not encode expected results", async () => {
  const { validateGpuPlan } = await import("../src/mlh/repro/gpu.ts");
  const { gpuPrice } = await import("../src/mlh/backends/modal.ts");
  const plan = (parse: string) => ({ fidelity: "eval_only", stages: [{ id: "s", argv: ["python", "e.py"], timeoutS: 300 }], claims: [{ claim: "c", stage: "s", parse }] });
  const v = (parse: string) => validateGpuPlan([{ id: "c", metric: "x", value: 0.7625 }] as never, plan(parse), { budgetUsd: 5, price: gpuPrice }).errors.join();
  it("rejects SimCSE r1's pattern (seven expected scores), keeps DETR's label pattern", () => {
    expect(v("\\|\\s*68\\.40\\s*\\|\\s*82\\.41\\s*\\|\\s*74\\.38\\s*\\|\\s*80\\.91\\s*\\|\\s*78\\.56\\s*\\|\\s*76\\.85\\s*\\|\\s*72\\.23\\s*\\|\\s*([0-9]+\\.[0-9]+)\\s*\\|")).toMatch(/by its label, not by expected numbers/);
    expect(v(PARSE)).toBe("");
  });
});

describe("plan rules from FT-Transformer r2", async () => {
  const { validateGpuPlan } = await import("../src/mlh/repro/gpu.ts");
  const { gpuPrice } = await import("../src/mlh/backends/modal.ts");
  const f = JSON.parse(readFileSync("tests/fixtures/ft-transformer-r2-plan.json", "utf8"));
  const v = (plan: unknown) => validateGpuPlan(f.claims, plan, { budgetUsd: 5, price: gpuPrice }).errors.join(" | ");
  it("the recorded plan is rejected: 28 of 30 stages serve no claim, and stages read the whole /cache", () => {
    const e = v(f.plan);
    expect(e).toMatch(/and 20 more serve no claim/);
    expect(e).toMatch(/may not use the whole \/cache/);
  });
  it("only the claim stages, pointed at a fetched directory: accepted", () => {
    const keep = f.plan.stages.filter((s: { id: string }) => ["ca_seed3", "adult_seed5"].includes(s.id)).map((s: { env: Record<string, string> }) => ({ ...s, env: { ...s.env, PROJECT_DIR: "/cache/data/rtdl" } }));
    expect(v({ ...f.plan, stages: keep, fetch: [{ url: f.plan.fetch[0].url, to: "/cache/data/rtdl", unpack: "tar" }] })).toBe("");
  });
});

describe("argv placeholders", async () => {
  const { validateGpuPlan } = await import("../src/mlh/repro/gpu.ts");
  const { gpuPrice } = await import("../src/mlh/backends/modal.ts");
  const v = (stage: Record<string, unknown>) => validateGpuPlan([{ id: "c", metric: "acc", value: 0.9 }] as never, { fidelity: "full", seeds: [0, 1], stages: [{ id: "s", timeoutS: 300, ...stage }], claims: [{ claim: "c", stage: "s", parse: "acc: ([0-9.]+)" }] }, { budgetUsd: 5, price: gpuPrice }).errors.join();
  it("{seed} only in perSeed stages; no other placeholder", () => {
    expect(v({ argv: ["python", "a.py", "cfg/{seed}.toml"], perSeed: true })).toBe("");
    expect(v({ argv: ["python", "a.py", "cfg/{seed}.toml"] })).toMatch(/needs perSeed/);
    expect(v({ argv: ["python", "a.py", "{dataset}"], perSeed: true })).toMatch(/unknown placeholder \{dataset\}/);
  });
});

describe("memGB and links in a stage", async () => {
  const { validateGpuPlan } = await import("../src/mlh/repro/gpu.ts");
  const { gpuPrice } = await import("../src/mlh/backends/modal.ts");
  const fetch = [{ url: "https://huggingface.co/x/data.tar", to: "/cache/data/x", unpack: "tar" }];
  const v = (stage: Record<string, unknown>, f = fetch) => validateGpuPlan([{ id: "c", metric: "acc", value: 0.9 }] as never, { fidelity: "full", fetch: f, stages: [{ id: "s", argv: ["python", "a.py"], timeoutS: 300, ...stage }], claims: [{ claim: "c", stage: "s", parse: "acc: ([0-9.]+)" }] }, { budgetUsd: 5, price: gpuPrice });
  it("accepts a link to fetched data and a memory request, and prices the memory", () => {
    const r = v({ memGB: 64, links: [{ path: "data", to: "/cache/data/x/data" }] });
    expect(r.errors).toEqual([]);
    expect(r.plan!.worstCaseUsd).toBeGreaterThan(v({}).plan!.worstCaseUsd);
  });
  it("rejects links outside the repo, to unfetched data, and absurd memory", () => {
    expect(v({ links: [{ path: "../x", to: "/cache/data/x" }] }).errors.join()).toMatch(/link/);
    expect(v({ links: [{ path: "data", to: "/cache/other" }] }).errors.join()).toMatch(/link/);
    expect(v({ links: [{ path: "data", to: "/cache/data/y" }] }).errors.join()).toMatch(/\/cache\/data\/y is not produced by any fetch item/);
    expect(v({ memGB: 1000 }).errors.join()).toMatch(/memGB/);
  });
});
