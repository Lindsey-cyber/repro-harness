// P1: the full GPU reproduction flow, offline. paper -> claims -> env -> plan -> stage -> compare
// -> report, with the real orchestrator, skills, validator, budget and journal; only the model
// (scripted) and Modal (fake runner, local commands on a canary repository) are substituted.
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { runHarness } from "../src/mlh/orchestrator.ts";
import { CANARY_REPO, canaryScript, scriptedTeam, withEnv, workspace } from "./helpers/team.ts";

const repro = (runDir: string) => JSON.parse(readFileSync(join(runDir, "repro.json"), "utf8")) as { claims: { id: string; status: string; mean?: number }[] };

export async function canaryRun(faults: unknown[] = [], log: string[] = []) {
  const ws = workspace(faults);
  const r = await withEnv(ws.env, () => runHarness({ runDir: ws.runDir, source: CANARY_REPO, objective: "Reproduce the canary accuracy.", mode: "repro", arxiv: "2101.00001", llm: scriptedTeam(canaryScript(), log), gpuBudgetUsd: 1, concurrency: 1 }));
  return { ...ws, r };
}

describe("P1: GPU reproduction flow, end to end, offline", () => {
  it("reproduces the canary claim through every stage", async () => {
    const log: string[] = [];
    const { runDir, env } = await canaryRun([], log);
    const out = repro(runDir);
    expect(out.claims.map((c) => [c.id, c.status, c.mean])).toEqual([["canary_acc", "reproduced", 0.9]]);
    const calls = readFileSync(join(env.MLH_FAKE_CACHE!, ".fake-calls.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l));
    expect(calls.some((c) => c.gpu === "T4" && c.cmd.join(" ").includes("train.py --steps 100"))).toBe(true); // the stage ran on the planned GPU
    expect(log).toEqual(expect.arrayContaining(["claims", "env:0:run", "env:1:report", "plan:0:report"]));
    expect(existsSync(join(runDir, "stopped.json"))).toBe(false);
  }, 120_000);
});
