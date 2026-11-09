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
