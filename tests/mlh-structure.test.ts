import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const ROOT = join(import.meta.dirname, "..");
const files = (dir: string): string[] =>
  readdirSync(join(ROOT, dir), { withFileTypes: true }).flatMap((d) => (d.isDirectory() ? files(join(dir, d.name)) : d.name.endsWith(".ts") ? [join(dir, d.name)] : []));
const code = (f: string) => readFileSync(join(ROOT, f), "utf8").replace(/^import type .*$/gm, "");

describe("ml harness structure", () => {
  it("keeps decisions pure: core never touches disk, processes, network, io or skills", () => {
    const bad = files("src/mlh/core").filter((f) => /from "node:|\bfetch\(|from "\.\.\/(io|skills)\/|from "\.\.\/orchestrator/.test(code(f)));
    expect(bad).toEqual([]);
  });

  it("starts processes in exactly one module", () => {
    expect(files("src/mlh").filter((f) => /node:child_process/.test(code(f)))).toEqual(["src/mlh/io/proc.ts"]);
    // Skills run repository code only through a Sandbox, never a raw process.
    expect(files("src/mlh/skills").filter((f) => /io\/proc|runProc/.test(code(f)))).toEqual([]);
  });

  it("authorizes every node's actions before its skill runs", () => {
    const o = code("src/mlh/orchestrator.ts");
    const exec = o.slice(o.indexOf("const execute = async"));
    expect(exec.indexOf("authorize(")).toBeGreaterThan(-1);
    expect(exec.indexOf("authorize(")).toBeLessThan(exec.indexOf("sk.run(ctx)"));
    expect(exec.indexOf('t: "hold"')).toBeLessThan(exec.indexOf("sk.run(ctx)"));
  });

  it("keeps the benchmark's ground truth out of the harness", () => {
    const leaks = files("src/mlh").filter((f) => f !== "src/mlh/bench.ts" && /task\.json|\bexpect\b.*allow|bench\/ml\/tasks/.test(code(f)));
    expect(leaks).toEqual([]);
    const named = files("src/mlh").filter((f) => f !== "src/mlh/bench.ts" && /leak_target|clean_overfit|preproc_unscaled|metric_train_as_test|ckpt_stale/.test(code(f)));
    expect(named).toEqual([]);
  });
});
