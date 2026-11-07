// The exploring agent: one model with file and shell tools in one sandboxed
// workspace. Used two ways:
//   - bare (the single-agent baseline): its report is the answer, unverified;
//   - inside the harness (skill agent.explore): every step is checkpointed so a
//     killed or failed run resumes where it stopped, and every finding it
//     reports is routed to an independent confirmation before it counts.
import { appendFileSync, cpSync, existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { authorize, type Evidence } from "./core/authz.ts";
import { VOCABULARY } from "./core/vocab.ts";
import type { ChatModel } from "./skills/index.ts";
import type { ChatMsg, ToolDef } from "./io/llm.ts";
import { DEFAULT_LIMITS, Sandbox, SkillError, type Isolation, type SandboxStats } from "./io/sandbox.ts";

export interface AgentFinding {
  category: string;
  summary: string;
  evidence: string;
  column?: string; // for overlap/leak findings: the column involved
  feature?: string;
  // A python script that measures the defect's effect: prints, as its last line,
  // {"with_defect": <number>, "without_defect": <number>}. The harness, not the
  // agent, decides whether the difference is material.
  effect_check?: string;
}
export interface AgentResult {
  findings: AgentFinding[];
  manifest: Record<string, unknown>;
  reproduced: boolean | null;
  steps: number;
  sandboxRuns: number;
  sandboxMs: number;
  stopped: "report" | "budget" | "error";
  error?: string;
  resumedAt?: number;
}

const TOOLS: ToolDef[] = [
  { type: "function", function: { name: "list_files", description: "List files under a directory of the repository (recursive, max 300).", parameters: { type: "object", properties: { dir: { type: "string" } }, required: [] } } },
  { type: "function", function: { name: "read_file", description: "Read a text file (max 20000 chars).", parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] } } },
  { type: "function", function: { name: "write_file", description: "Write a file inside your working copy (e.g. a check script).", parameters: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path", "content"] } } },
  { type: "function", function: { name: "run", description: "Run a shell command in the repository root, in an offline sandbox (python3 with numpy and pandas). Returns exit code and output tails.", parameters: { type: "object", properties: { command: { type: "string" } }, required: ["command"] } } },
  {
    type: "function",
    function: {
      name: "report",
      description: "Finish: report the defects you found (empty list if none), whether the claimed result reproduced, and how the experiment is run.",
      parameters: {
        type: "object",
        properties: {
          reproduced: { type: ["boolean", "null"] },
          findings: { type: "array", items: { type: "object", properties: {
            category: { type: "string" }, summary: { type: "string" }, evidence: { type: "string" },
            column: { type: "string", description: "for overlap or leakage: the column involved, if any" },
            feature: { type: "string", description: "for target leakage: the leaking feature, if a single one" },
            effect_check: { type: "string", description: "python3 script, run from the repo root on a fresh copy, that measures the headline metric with the defect and with it corrected, printing as its LAST line {\"with_defect\": <number>, \"without_defect\": <number>}" },
          }, required: ["category", "summary", "evidence"] } },
          manifest: { type: "object", description: "how to run it: {train: [argv], eval?: [argv], data?: {train, test, label}, outputs?: {metrics, predictions?, checkpoint?}, metric: accuracy|f1|auc, metricKey?: key of the test metric in the metrics json, claim?: {metric, value}, seedArg?: '--seed' if a CLI flag sets the seed}" },
        },
        required: ["findings", "reproduced"],
      },
    },
  },
];

const SYSTEM = `You are an ML research engineer auditing an experiment repository. You have tools to read files, write files and run commands in an isolated offline copy of the repository. Run the experiment, check whether the README's claimed result reproduces, and look for implementation or evaluation errors (data leakage, broken preprocessing, incorrect metrics, checkpoint failures, irreproducibility, or anything else that changes results). Report only defects you have evidence for: a false alarm is as costly as a miss. For each defect, include an effect_check script that measures the headline metric with and without the defect. Also report the manifest: how the experiment is run and where it writes its metrics. Keep tool outputs small. You have a limited step budget; call report before it runs out.

${VOCABULARY}`;

const safePath = (root: string, p: string) => { const a = resolve(root, p || "."); if (!a.startsWith(root)) throw new Error("path outside the repository"); return a; };

export interface AgentOptions {
  workDir: string; // persistent: the workspace and, if checkpointing, the state file live here
  source: string;
  objective: string;
  model: ChatModel;
  isolation: Isolation;
  isolationEvidence: Evidence;
  maxSteps?: number;
  checkpoint?: boolean; // persist the conversation after every step and resume from it
  sandboxFault?: () => boolean;
  crashAfterSteps?: number; // exit the process after this many steps in this process (recovery tests)
}

interface State { messages: ChatMsg[]; step: number; stats: SandboxStats }

export async function runAgent(o: AgentOptions): Promise<AgentResult> {
  mkdirSync(o.workDir, { recursive: true });
  const statePath = join(o.workDir, "state.json");
  const ws = join(o.workDir, "ws");
  const log = join(o.workDir, "agent.jsonl");
  let st: State | undefined = o.checkpoint && existsSync(statePath) && existsSync(ws) ? JSON.parse(readFileSync(statePath, "utf8")) as State : undefined;
  const resumedAt = st?.step;
  if (!st) {
    rmSync(ws, { recursive: true, force: true });
    cpSync(o.source, ws, { recursive: true, filter: (p) => !/\/(\.git|outputs|__pycache__)(\/|$)/.test(p) });
    st = { messages: [{ role: "system", content: SYSTEM }, { role: "user", content: `Objective: ${o.objective}\nThe repository is your working directory.` }], step: 0, stats: { runs: 0, ms: 0 } };
  }
  const state = st;
  const sb = Sandbox.attach(ws, { isolation: o.isolation, limits: { ...DEFAULT_LIMITS, timeoutMs: 120_000 }, stats: state.stats, ...(o.sandboxFault ? { fault: o.sandboxFault } : {}) });
  const save = () => {
    if (!o.checkpoint) return;
    writeFileSync(statePath + ".tmp", JSON.stringify(state));
    renameSync(statePath + ".tmp", statePath); // atomic: a crash leaves the old or the new state, never half
  };
  const done = (r: Omit<AgentResult, "steps" | "sandboxRuns" | "sandboxMs">): AgentResult => ({ ...r, steps: state.step, sandboxRuns: state.stats.runs, sandboxMs: Math.round(state.stats.ms), ...(resumedAt !== undefined ? { resumedAt } : {}) });
  const maxSteps = o.maxSteps ?? 30;
  let stepsHere = 0;
  while (state.step < maxSteps) {
    const step = state.step + 1;
    const msgs: ChatMsg[] = step === maxSteps ? [...state.messages, { role: "user", content: "Step budget exhausted: call report now." }] : state.messages;
    // A provider failure propagates: inside the harness the node is retried and resumes from the last saved step.
    const { message: msg } = await o.model.chat(msgs, { tools: TOOLS, maxTokens: 16000 });
    const next: ChatMsg[] = [...msgs, { role: "assistant", content: msg.content ?? "", ...(msg.tool_calls ? { tool_calls: msg.tool_calls } : {}), ...(msg.reasoning_content ? { reasoning_content: msg.reasoning_content } : {}) }];
    appendFileSync(log, JSON.stringify({ step, content: msg.content, tool_calls: msg.tool_calls }) + "\n");
    if (!msg.tool_calls?.length) next.push({ role: "user", content: "Answer only through tool calls: call the report tool now with your findings (categories from the list), whether the claim reproduced, and the manifest." });
    for (const call of msg.tool_calls ?? []) {
      let out: string;
      let args: Record<string, unknown> = {};
      try { args = JSON.parse(call.function.arguments || "{}"); } catch { /* reported below */ }
      try {
        switch (call.function.name) {
          case "list_files": {
            const base = safePath(ws, String(args.dir ?? "."));
            const walk = (d: string): string[] => readdirSync(d, { withFileTypes: true }).flatMap((e) => e.name.startsWith(".") ? [] : e.isDirectory() ? walk(join(d, e.name)) : [relative(ws, join(d, e.name))]);
            out = walk(base).slice(0, 300).join("\n");
            break;
          }
          case "read_file": {
            const p = safePath(ws, String(args.path));
            out = statSync(p).size > 2_000_000 ? "file too large" : readFileSync(p, "utf8").slice(0, 20_000);
            break;
          }
          case "write_file": {
            const p = safePath(ws, String(args.path));
            mkdirSync(dirname(p), { recursive: true });
            writeFileSync(p, String(args.content));
            out = "written";
            break;
          }
          case "run": {
            const d = authorize(o.isolation === "namespace" ? "exec_sandboxed" : "exec_unsandboxed", [o.isolationEvidence]);
            if (!d.allow) { out = `denied: ${d.reason}`; break; }
            const r = await sb.run(["sh", "-c", String(args.command)]);
            out = `exit ${r.category === "timeout" ? "timeout" : (/^MLH_EXIT=(\d+)$/m.exec(r.stderr)?.[1] ?? r.code)}\nstdout:\n${r.stdout.slice(-4000)}\nstderr:\n${r.stderr.replace(/^MLH_EXIT=\d+\n?/m, "").slice(-2000)}`;
            break;
          }
          case "report": {
            state.step = step;
            save();
            return done({ findings: (Array.isArray(args.findings) ? args.findings : []) as AgentFinding[], manifest: (args.manifest && typeof args.manifest === "object" ? args.manifest : {}) as Record<string, unknown>, reproduced: (args.reproduced as boolean | null) ?? null, stopped: "report" });
          }
          default:
            out = `unknown tool ${call.function.name}`;
        }
      } catch (e) {
        // A transient sandbox failure is shown to the agent like any failed command; it may try again.
        out = `error: ${String(e instanceof SkillError ? e.message : (e as Error).message ?? e)}`;
      }
      next.push({ role: "tool", tool_call_id: call.id, content: out });
    }
    state.messages = next;
    state.step = step;
    save();
    if (o.crashAfterSteps !== undefined && ++stepsHere >= o.crashAfterSteps) process.exit(137);
  }
  return done({ findings: [], manifest: {}, reproduced: null, stopped: "budget" });
}
