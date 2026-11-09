// Single-agent baseline for complex papers: one agent with every tool (paper,
// files, remote run on Modal GPUs, image/pip), the same budget as the team, no
// checkpoints, no validated plan, no smoke test. Its report is then checked the
// same way the team's results are: the harness reruns each claim's reported
// command once in a fresh container and applies the same status rule, so both
// systems are scored on independently re-derived numbers, not on self-reports.
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync, appendFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { agentLoop, type Role } from "../agentloop.ts";
import { Budget, modalRun } from "../backends/modal.ts";
import { Llm } from "../io/llm.ts";
import { Sandbox, selfTest } from "../io/sandbox.ts";
import type { Claim } from "./flow.ts";
import { DEFAULT_IMAGE_KEY, GPUS, IMAGES } from "./gpu.ts";
import { CAUSES } from "./roles.ts";
import { parseMetric, statusOf } from "./skills.ts";

const SOLO_COMPLEX: Role = {
  name: "solo",
  maxSteps: 80,
  tools: ["read_paper", "list_files", "read_file", "write_file", "run", "install"],
  writeCode: true,
  system: `You are an ML research engineer reproducing a paper alone, with remote GPU containers and a fixed dollar budget. Read the paper (read_paper), set up the environment (install: base image and pip packages), run the experiments the objective names, debug what fails, and report per claim. Commands run in fresh remote containers: only /cache persists between runs. Never change how results are measured to move them toward the paper's number.

For every claim, report the exact command that produced your number (argv, working directory, GPU) and a regex with one capture group that extracts the number from its output: the harness reruns that command once to verify your result.

${CAUSES}`,
  report: {
    description: "Finish with per-claim results, the command that produces each, and root causes.",
    parameters: {
      type: "object",
      properties: {
        claims: { type: "array", items: { type: "object", properties: {
          id: { type: "string" }, dataset: { type: "string" }, metric: { type: "string" }, value: { type: "number", description: "the paper's number as a fraction for percentages" },
          values: { type: "array", items: { type: "number" }, description: "your runs' results, same scale" },
          status: { type: "string", enum: ["reproduced", "not_reproduced", "not_runnable"] },
          argv: { type: "array", items: { type: "string" } }, cwd: { type: "string" }, gpu: { type: "string" }, parse: { type: "string" }, timeoutS: { type: "integer" },
        }, required: ["id", "value", "status"] } },
        causes: { type: "array", items: { type: "object", properties: { category: { type: "string" }, evidence: { type: "string" } }, required: ["category", "evidence"] } },
      },
      required: ["claims", "causes"],
    },
  },
};

export async function runSolo(o: { source: string; arxiv: string; objective: string; budgetUsd: number; runDir: string }): Promise<void> {
  const runDir = resolve(o.runDir);
  mkdirSync(runDir, { recursive: true });
  const budget = new Budget(o.budgetUsd, join(runDir, "budget.json"), `mlh-solo-${createHash("sha256").update(runDir).digest("hex").slice(0, 12)}`);
  const st = await selfTest(join(runDir, "selftest"));
  // The paper, extracted in the sandbox like the team's paper node.
  const pdf = join(runDir, "paper.pdf");
  if (!existsSync(pdf)) writeFileSync(pdf, Buffer.from(await (await fetch(`https://arxiv.org/pdf/${o.arxiv}`)).arrayBuffer()));
  await Sandbox.attach(runDir, { isolation: st.isolation, limits: { cpuSec: 120, memMB: 2048, fileMB: 200, timeoutMs: 120_000 }, stats: { runs: 0, ms: 0 } }).run(["pdftotext", "-layout", "paper.pdf", "paper.txt"]);
  const paper = readFileSync(join(runDir, "paper.txt"), "utf8");
  const model = new Llm({ model: process.env.MLH_PROPOSER ?? "deepseek-flash", cacheDir: resolve(".mlh/llm-cache") });
  const t0 = performance.now();
  let report: Record<string, unknown> | null = null, error: string | undefined;
  try {
    const r = await agentLoop({
      role: SOLO_COMPLEX, workDir: join(runDir, "agent"), source: o.source, task: `Objective: ${o.objective}\nBudget: $${o.budgetUsd} of GPU time.`, model, paper,
      isolation: st.isolation, isolationEvidence: { kind: "deterministic", check: "selftest", ok: st.isolation === "namespace" },
      remote: { pip: [], gpus: GPUS, images: IMAGES, image: DEFAULT_IMAGE_KEY, run: async (ws, command, ro) => {
        const res = await modalRun({ repo: ws, cmd: ["bash", "-lc", command], timeoutS: ro.timeoutS, network: true, pip: ro.pip, image: IMAGES[ro.image ?? DEFAULT_IMAGE_KEY]!, ...(ro.gpu ? { gpu: ro.gpu } : {}) }, budget);
        if (res.category === "provider_fatal") throw new Error(`Insufficient Balance (GPU provider): ${res.stderr.slice(0, 200)}`);
        return { code: res.exitCode, stdout: res.stdout, stderr: res.stderr, category: res.category, costUsd: res.costUsd };
      } },
    });
    report = r.report;
    writeFileSync(join(runDir, "solo-report.json"), JSON.stringify({ ...r, report: r.report }, null, 1));
    // Independent verification of each reported claim: rerun its command once, same status rule.
    const claims = ((report?.claims ?? []) as (Claim & { status: string; argv?: string[]; cwd?: string; gpu?: string; parse?: string; timeoutS?: number; values?: number[] })[]);
    const verified = [];
    for (const c of claims) {
      let vstatus = "not_runnable", mean: number | undefined, note = "no command reported";
      if (Array.isArray(c.argv) && c.argv.length && typeof c.parse === "string") {
        const cmd = c.cwd ? ["bash", "-c", 'cd "$1" && shift && exec "$@"', "bash", c.cwd, ...c.argv] : c.argv;
        const res = await modalRun({ repo: r.workspace, cmd, timeoutS: Math.min(Number(c.timeoutS) || 3600, 6 * 3600), network: true, pip: r.pip, image: IMAGES[r.image ?? DEFAULT_IMAGE_KEY]!, ...(c.gpu && GPUS.includes(c.gpu) ? { gpu: c.gpu } : {}) }, budget);
        const v = parseMetric({ claim: c.id, argv: c.argv, seeds: [], parse: c.parse, timeoutS: 0 }, String(c.metric ?? "accuracy"), `${res.stdout}\n${res.stderr}`);
        if (v !== undefined) { const s = statusOf({ ...c, value: Number(c.value) }, [v]); vstatus = s.status; mean = v; note = "rerun by the harness"; } else note = `rerun produced no value (${res.category ?? `exit ${res.exitCode}`})`;
      }
      verified.push({ id: c.id, value: Number(c.value), status: vstatus, selfStatus: c.status, selfValues: c.values ?? [], mean, note });
    }
    writeFileSync(join(runDir, "repro.json"), JSON.stringify({ claims: verified, causes: report?.causes ?? [], failed: [] }, null, 1));
  } catch (e) {
    error = String(e).slice(0, 500);
    writeFileSync(join(runDir, "repro.json"), JSON.stringify({ claims: [], failed: [{ id: "solo", error }] }, null, 1));
    if (/LLM 40[12]\b|Insufficient Balance/i.test(error)) writeFileSync(join(runDir, "stopped.json"), JSON.stringify({ reason: "provider_fatal", detail: error }));
  }
  // Trace line in the same format as the team's, so the campaign summary reads both.
  appendFileSync(join(runDir, "trace.jsonl"), JSON.stringify({ id: "solo", outcome: error ? "fail" : "ok", ...(error ? { category: "solo_error" } : {}), ms: Math.round(performance.now() - t0), tokens: model.usage.promptTokens + model.usage.completionTokens, costUsd: model.usage.costUsd }) + "\n");
  appendFileSync(join(runDir, "journal.jsonl"), JSON.stringify({ t: "done", at: new Date().toISOString() }) + "\n");
}
