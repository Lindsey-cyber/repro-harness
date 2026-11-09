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

describe("smoke test working directory", async () => {
  const { smokeCwd } = await import("../src/mlh/repro/skills.ts");
  it("maps container paths under /work to repository paths (LoRA r3), refuses anything outside", () => {
    expect(smokeCwd("/work/examples/NLU")).toBe("examples/NLU");
    expect(smokeCwd("examples/NLU/")).toBe("examples/NLU");
    expect(smokeCwd("/work")).toBe(".");
    expect(smokeCwd("/etc")).toBe(".");
    expect(smokeCwd("../x")).toBe(".");
    expect(smokeCwd(undefined)).toBe(".");
  });
});

describe("GPU fallback order", async () => {
  const { fallbackGpus } = await import("../src/mlh/repro/gpu.ts");
  it("cheapest types with enough memory, never the planned one, at most two", () => {
    expect(fallbackGpus("L4")).toEqual(["A10G", "A100-40GB"]);
    expect(fallbackGpus("T4")).toEqual(["L4", "A10G"]);
    expect(fallbackGpus("A100-80GB")).toEqual(["H100"]);
    expect(fallbackGpus("T4", 30)).toEqual(["A100-40GB", "A100-80GB"]);
  });
});

describe("declared reporting rule (select over epochs, aggregate over seeds)", async () => {
  const { pick, statusOf } = await import("../src/mlh/repro/skills.ts");
  const { validateGpuPlan } = await import("../src/mlh/repro/gpu.ts");
  const { gpuPrice } = await import("../src/mlh/backends/modal.ts");
  it("picks the last, best or lowest value of a real per-epoch series", () => {
    const FT = readFileSync("tests/fixtures/ft-transformer-ca-seed0.txt", "utf8");
    const s = metricSeries(metricMatches("\\[test \\] score = -?[0-9.]+ \\| rmse = ([0-9.]+)", FT))[0]!;
    expect(pick(s.values)).toBe(0.459);
    expect(pick(s.values, "min")).toBe(Math.min(...s.values));
    expect(pick(s.values, "max")).toBe(Math.max(...s.values));
  });
  it("aggregates seeds by the paper's rule (LoRA: median of 5)", () => {
    const c = { id: "c", metric: "accuracy", value: 0.897, std: 0.007 } as never;
    expect(statusOf(c, [0.88, 0.89, 0.95], "median").mean).toBe(0.89);
    expect(statusOf(c, [0.88, 0.89, 0.95]).mean).toBeCloseTo(0.9067, 3);
  });
  it("the plan declares it; unknown rules are rejected", () => {
    const plan = (x: Record<string, unknown>) => ({ fidelity: "full", stages: [{ id: "s", argv: ["python", "a.py"], timeoutS: 300 }], claims: [{ claim: "c", stage: "s", parse: "acc: ([0-9.]+)", ...x }] });
    const v = (x: Record<string, unknown>) => validateGpuPlan([{ id: "c", metric: "acc", value: 0.9 }] as never, plan(x), { budgetUsd: 5, price: gpuPrice });
    expect(v({ select: "max", aggregate: "median" }).plan!.claims[0]).toMatchObject({ select: "max", aggregate: "median" });
    expect(v({ select: "best" }).errors.join()).toMatch(/select must be last, max or min/);
    expect(v({ aggregate: "mode" }).errors.join()).toMatch(/aggregate must be mean or median/);
  });
});

describe("plan flags that differ from the repository's launch script (LoRA r3)", async () => {
  const { recipeDeviations, recipeErrors } = await import("../src/mlh/repro/skills.ts");
  const scripts = ["roberta_base_rte.sh", "roberta_base_mrpc.sh"].map((n) => ({ name: n, text: readFileSync(`tests/fixtures/lora-scripts/${n}`, "utf8") }));
  const stages = JSON.parse(readFileSync("tests/fixtures/lora-r3-stages.json", "utf8")) as { id: string; argv: string[] }[];
  it("finds the planner's unreviewed --fp16 and --pad_to_max_length, matched to the right script, and nothing else", () => {
    const d = recipeDeviations(scripts, stages);
    const rte = d.filter((x) => x.stage === "train_eval_rte");
    expect(new Set(rte.map((x) => x.script))).toEqual(new Set(["roberta_base_rte.sh"]));
    expect(rte.map((x) => x.flag).sort()).toEqual(["--fp16", "--pad_to_max_length"]);
    expect(d.filter((x) => x.stage === "train_eval_mrpc").every((x) => x.script === "roberta_base_mrpc.sh")).toBe(true);
  });
  it("each deviation needs a reason; with reasons the plan passes", () => {
    expect(recipeErrors(scripts, stages, undefined).join("\n")).toMatch(/train_eval_rte: --fp16 is "" but roberta_base_rte.sh has no such flag/);
    const reasons = Object.fromEntries(recipeDeviations(scripts, stages).map((x) => [`${x.stage} ${x.flag}`, "fits the 24 GB GPU in the budget"]));
    expect(recipeErrors(scripts, stages, reasons)).toEqual([]);
    expect(recipeErrors(scripts, stages, Object.fromEntries(Object.keys(reasons).map((k) => [k, "ok"])))).not.toEqual([]); // a reason, not a token
  });
});

describe("effective batch of a multi-GPU recipe on one GPU (LoRA r3)", async () => {
  const { effectiveBatchErrors } = await import("../src/mlh/repro/skills.ts");
  const scripts = ["roberta_base_rte.sh", "roberta_base_mrpc.sh"].map((n) => ({ name: n, text: readFileSync(`tests/fixtures/lora-scripts/${n}`, "utf8") }));
  const stages = JSON.parse(readFileSync("tests/fixtures/lora-r3-stages.json", "utf8")) as { id: string; argv: string[] }[];
  it("the recorded plan is rejected: 8-GPU recipe, one GPU, no accumulation", () => {
    const e = effectiveBatchErrors(scripts, stages);
    expect(e).toHaveLength(2);
    expect(e[0]).toMatch(/trains on 8 GPUs.*--gradient_accumulation_steps 8/);
  });
  it("with --gradient_accumulation_steps 8 it passes; a single-GPU repository is not affected", () => {
    expect(effectiveBatchErrors(scripts, stages.map((s) => ({ ...s, argv: [...s.argv, "--gradient_accumulation_steps", "8"] })))).toEqual([]);
    expect(effectiveBatchErrors([{ name: "run.sh", text: "python train.py --per_device_train_batch_size 32" }], stages)).toEqual([]);
  });
});
