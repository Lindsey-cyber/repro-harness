// Every Modal error message seen in a real run, with the category it must get. A message
// that stopped or misled a campaign is added here first, then the classifier is fixed.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { classifyModalError } from "../src/mlh/backends/modal.ts";

const corpus = JSON.parse(readFileSync("tests/fixtures/modal-errors.json", "utf8")) as { head: string; category: string; source: string }[];

describe("Modal error corpus", () => {
  for (const c of corpus) it(`${c.category}: ${c.source}`, () => expect(classifyModalError(c.head)).toBe(c.category));
  it("only a spend limit stops the campaign; a capacity shortage is retried", () => {
    expect(corpus.filter((c) => classifyModalError(c.head) === "provider_fatal").map((c) => c.head)).toEqual([expect.stringMatching(/spend limit/)]);
  });
});

describe("command exit classification", async () => {
  const { classifyExit } = await import("../src/mlh/backends/modal.ts");
  it("OOM kill, missing upstream output, missing declared output, ordinary failure", () => {
    expect(classifyExit(0, "")).toBeNull();
    expect(classifyExit(137, "Killed")).toBe("resource_limit"); // GraphSAINT r1 ppi_e
    expect(classifyExit(1, "cp: cannot stat '/cache/runs/99340763c8ec/1.p2/build_cython/graphsaint/x.so': No such file or directory")).toBe("sandbox_transient");
    expect(classifyExit(97, "MLH_MISSING_OUTPUT=/work/graphsaint/x.so")).toBe("experiment_error");
    expect(classifyExit(124, "")).toBe("timeout");
    expect(classifyExit(1, "Traceback")).toBe("experiment_error");
  });
});
