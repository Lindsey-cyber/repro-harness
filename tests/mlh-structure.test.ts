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

  it("keeps the reproduction flow's deciding code pure", () => {
    const bad = ["src/mlh/repro/flow.ts", "src/mlh/repro/patchrisk.ts", "src/mlh/repro/report.ts"].filter((f) => /from "node:|\bfetch\(|from "\.\.\/(io|skills)\/|from "\.\/(skills|site|patch|bench)\.ts"/.test(code(f)));
    expect(bad).toEqual([]);
  });

  it("authorizes a proposed change before it becomes a source version, and installs only through authorization", () => {
    const sk = code("src/mlh/repro/skills.ts");
    const patch = sk.slice(sk.indexOf("const patch: Skill"));
    expect(patch.indexOf("authorize(")).toBeGreaterThan(-1);
    expect(patch.indexOf("authorize(")).toBeLessThan(patch.indexOf("buildVersion("));
    const site = code("src/mlh/repro/site.ts");
    expect(site.indexOf('authorize("install_packages"')).toBeLessThan(site.indexOf("network: true"));
    // On this machine, only the install sandbox ever gets network.
    expect(files("src/mlh").filter((f) => /Sandbox\.(create|attach)\([^)]*network: true/s.test(code(f)) || /network: true/.test(code(f)) && /Sandbox\./.test(code(f)) && f !== "src/mlh/repro/skills.ts")).toEqual(["src/mlh/repro/site.ts"]);
    // Remote containers (Modal; isolated, no credentials inside) get network only from the reproduction skills.
    expect(files("src/mlh").filter((f) => /modalRun\([^;]*network: (true|!!)/s.test(code(f)))).toEqual(["src/mlh/repro/skills.ts"]);
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
    const graders = new Set(["src/mlh/bench.ts", "src/mlh/repro/bench.ts"]);
    const leaks = files("src/mlh").filter((f) => !graders.has(f) && /task\.json|inject\.patch|fix_check|\bexpect\b.*allow|bench\/(ml|repro)\/tasks/.test(code(f)));
    expect(leaks).toEqual([]);
    const named = files("src/mlh").filter((f) => !graders.has(f) && /leak_target|clean_overfit|preproc_unscaled|metric_train_as_test|ckpt_stale/.test(code(f)));
    expect(named).toEqual([]);
  });
});
