// Child process for kill/resume tests: runs (or resumes) the canary reproduction so that a crash
// (process.exit) cannot take the test runner down. Configured by environment variables.
import { existsSync } from "node:fs";
import { join } from "node:path";
import { runHarness } from "../../src/mlh/orchestrator.ts";
import { CANARY_REPO, canaryScript, scriptedTeam } from "./team.ts";

const runDir = process.env.MLH_TEST_RUNDIR!;
const resume = existsSync(join(runDir, "journal.jsonl"));
const crashAfter = process.env.MLH_TEST_CRASH_AFTER ? Number(process.env.MLH_TEST_CRASH_AFTER) : undefined;
const crashAfterSteps = process.env.MLH_TEST_CRASH_AFTER_STEPS ? Number(process.env.MLH_TEST_CRASH_AFTER_STEPS) : undefined;
await runHarness({
  runDir, mode: "repro", llm: scriptedTeam(canaryScript()), concurrency: 1,
  ...(resume ? {} : { source: CANARY_REPO, objective: "Reproduce the canary accuracy.", arxiv: "2101.00001", gpuBudgetUsd: 1 }),
  ...(crashAfter !== undefined ? { crashAfter } : {}),
  ...(crashAfterSteps !== undefined ? { crashAfterSteps } : {}),
});
