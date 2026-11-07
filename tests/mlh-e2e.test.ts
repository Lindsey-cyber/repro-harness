// End to end on benchmark repositories: real sandboxed runs, no mocks.
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { runProc } from "../src/mlh/io/proc.ts";
import { runHarness } from "../src/mlh/orchestrator.ts";

const ROOT = join(import.meta.dirname, "..");
const repo = (t: string) => join(ROOT, "bench/ml/tasks", t, "repo");
const tmp = () => mkdtempSync(join(tmpdir(), "mlh-e2e-"));
const objective = "Reproduce the reported test accuracy and check the experiment for implementation or evaluation errors.";

describe("ml harness end to end", () => {
  it("confirms a target leak with held-out evidence and writes an evidence-backed report", async () => {
    const r = await runHarness({ runDir: tmp(), source: repo("leak_target"), objective });
    expect(r.summary.confirmed.map((f) => f.category)).toEqual(["data.target_leak"]);
    const report = readFileSync(r.reportPath, "utf8");
    expect(report).toContain('feature "followup_score" alone separates');
    expect(report).toContain("test-set separability");
    expect(report).toMatch(/sha256 [0-9a-f]{16}/);
  }, 60_000);

  it("refutes a train/test gap that held-out data shows is overfitting, not a broken pipeline", async () => {
    const r = await runHarness({ runDir: tmp(), source: repo("clean_overfit"), objective });
    expect(r.summary.confirmed).toEqual([]);
    expect(r.summary.refuted.map((f) => f.category)).toEqual(["pipeline.eval_path_mismatch"]);
  }, 60_000);

  it("resumes after being killed mid-run without re-executing completed nodes", async () => {
    const runDir = tmp();
    const child = await runProc(["npx", "tsx", join(ROOT, "src/mlh/cli.ts"), "run", repo("metric_train_as_test"), "--run-dir", runDir, "--objective", objective, "--crash-after", "2"], { cwd: ROOT, env: process.env as Record<string, string>, timeoutMs: 120_000 });
    expect(child.code).toBe(137);
    const r = await runHarness({ runDir });
    expect(r.state.done).toBe(true);
    const oks = readFileSync(join(runDir, "journal.jsonl"), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)).filter((e) => e.t === "ok").map((e) => e.id);
    expect(oks.length).toBe(new Set(oks).size);
    expect(r.summary.confirmed.map((f) => f.category)).toContain("eval.metric_mismatch");
  }, 120_000);

  it("holds repository code when isolation is not proven, and runs it only after a human approves", async () => {
    const runDir = tmp();
    const r = await runHarness({ runDir, source: repo("clean"), objective, isolation: "none" });
    expect(r.summary.held.map((h) => h.id)).toEqual(["repo.inspect"]);
    expect(r.summary.held[0]!.reason).toMatch(/high risk and needs a human approval/);
    await runProc(["npx", "tsx", join(ROOT, "src/mlh/cli.ts"), "approve", runDir, "repo.inspect"], { cwd: ROOT, env: process.env as Record<string, string>, timeoutMs: 60_000 });
    const r2 = await runHarness({ runDir, isolation: "none" });
    expect(r2.state.nodes.get("repo.inspect")!.status).toBe("succeeded");
    expect(r2.summary.held.map((h) => h.id)).toContain("run.train"); // each unsandboxed node needs its own approval
  }, 60_000);
});
