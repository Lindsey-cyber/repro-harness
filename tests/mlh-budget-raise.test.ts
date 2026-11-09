import { describe, expect, it } from "vitest";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { budgetRaises } from "../src/mlh/orchestrator.ts";

describe("owner's budget raises", () => {
  it("add to the limit without touching run.json; invalid or negative entries count for nothing", () => {
    const d = mkdtempSync(join(tmpdir(), "mlh-raise-"));
    expect(budgetRaises(d)).toBe(0);
    writeFileSync(join(d, "budget-raises.jsonl"), ['{"usd":1.5,"reason":"RTE seeds"}', '{"usd":-3}', '{"usd":"x"}', '{"usd":0.5}'].join("\n") + "\n");
    expect(budgetRaises(d)).toBe(2);
  });
});
