// Stands in for `mlh repro|resume ... --llm` under the campaign driver: same arguments, scripted team,
// fake Modal. MLH_TEST_STALL_ONCE=<file>: the first launch hangs before writing anything (watchdog test).
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { runHarness } from "../../src/mlh/orchestrator.ts";
import { canaryPdf, canaryScript, scriptedTeam } from "./team.ts";

const [cmd, ...rest] = process.argv.slice(2);
const flag = (k: string) => { const i = rest.indexOf(`--${k}`); return i >= 0 ? rest[i + 1] : undefined; };
const stall = process.env.MLH_TEST_STALL_ONCE;
if (stall && !existsSync(stall)) { writeFileSync(stall, "stalled"); setInterval(() => {}, 1 << 30); await new Promise(() => {}); }
const runDir = cmd === "resume" ? rest[0]! : flag("run-dir")!;
mkdirSync(join(runDir, "work", "repro.paper"), { recursive: true });
if (!existsSync(join(runDir, "work", "repro.paper", "paper.pdf"))) writeFileSync(join(runDir, "work", "repro.paper", "paper.pdf"), canaryPdf("Canary model: test accuracy of 90.0% on the synthetic task."));
await runHarness({
  runDir, mode: "repro", llm: scriptedTeam(canaryScript()),
  ...(cmd === "repro" ? { source: rest[0]!, objective: flag("objective")!, arxiv: flag("arxiv")!, gpuBudgetUsd: Number(flag("gpu-budget")) } : {}),
});
