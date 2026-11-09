// P4 live canary: the full reproduction flow on the real Modal backend (real image, /cache volume,
// a T4), with the scripted team (no LLM tokens) and the canary repository (trusted code with a
// known metric). Passes only if the claim reproduces exactly, nothing stopped, and per-attempt
// cost was recorded. Usage: npx tsx scripts/mlh-canary-live.mts <runDir>
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { runHarness } from "../src/mlh/orchestrator.ts";
import { runDoctor, renderDoctor } from "../src/mlh/repro/doctor.ts";
import { llmBalance, costOfRun } from "../src/mlh/repro/campaign.ts";
import { renderCost } from "../src/mlh/repro/accounting.ts";
import { CANARY_REPO, canaryPdf, canaryScript, scriptedTeam } from "../tests/helpers/team.ts";

const runDir = resolve(process.argv[2] ?? ".mlh/canary-live/run");
delete process.env.MLH_MODAL_RUNNER; // the real backend
const doc = await runDoctor({ llmBalance, minLlmBalance: 0 });
console.log(renderDoctor(doc));
if (!doc.ok) process.exit(2);
mkdirSync(join(runDir, "work", "repro.paper"), { recursive: true });
if (!existsSync(join(runDir, "work", "repro.paper", "paper.pdf"))) writeFileSync(join(runDir, "work", "repro.paper", "paper.pdf"), canaryPdf("Canary model: test accuracy of 90.0% on the synthetic task."));
process.env.MLH_ENV_CACHE = join(runDir, "..", "env-cache");
await runHarness({ runDir, mode: "repro", llm: scriptedTeam(canaryScript()), ...(existsSync(join(runDir, "journal.jsonl")) ? {} : { source: CANARY_REPO, objective: "Reproduce the canary accuracy.", arxiv: "2101.00001", gpuBudgetUsd: 0.5 }) });
const claims = JSON.parse(readFileSync(join(runDir, "repro.json"), "utf8")).claims as { id: string; status: string; mean?: number }[];
const stopped = existsSync(join(runDir, "stopped.json")) ? readFileSync(join(runDir, "stopped.json"), "utf8") : null;
const cost = costOfRun(runDir);
console.log(JSON.stringify({ claims: claims.map((c) => [c.id, c.status, c.mean]), stopped }));
console.log(renderCost("canary-live", cost));
const ok = claims.length === 1 && claims[0]!.status === "reproduced" && claims[0]!.mean === 0.9 && !stopped && cost.productive.gpuUsd > 0;
console.log(ok ? "CANARY PASS" : "CANARY FAIL");
process.exit(ok ? 0 : 1);
