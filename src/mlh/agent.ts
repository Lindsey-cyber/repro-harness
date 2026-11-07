// Single-agent LLM baseline: one model with file and shell tools in one
// sandbox, the same objective and the same finding vocabulary as the harness,
// and no independent verification. It answers "what does a capable agent with
// tools do on its own?" so the harness's added structure can be measured.
import { appendFileSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { authorize, type Evidence } from "./core/authz.ts";
import { VOCABULARY } from "./core/vocab.ts";
import type { ChatModel } from "./skills/index.ts";
import type { ChatMsg, ToolDef } from "./io/llm.ts";
import { DEFAULT_LIMITS, Sandbox, type Isolation } from "./io/sandbox.ts";



export interface AgentFinding { category: string; summary: string; evidence: string }
export interface AgentResult { findings: AgentFinding[]; reproduced: boolean | null; steps: number; sandboxRuns: number; sandboxMs: number; stopped: "report" | "budget" | "error"; error?: string }

const TOOLS: ToolDef[] = [
  { type: "function", function: { name: "list_files", description: "List files under a directory of the repository (recursive, max 300).", parameters: { type: "object", properties: { dir: { type: "string" } }, required: [] } } },
  { type: "function", function: { name: "read_file", description: "Read a text file (max 20000 chars).", parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] } } },
  { type: "function", function: { name: "write_file", description: "Write a file inside your working copy (e.g. a check script).", parameters: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path", "content"] } } },
  { type: "function", function: { name: "run", description: "Run a shell command in the repository root, in an offline sandbox (python3 with numpy and pandas). Returns exit code and output tails.", parameters: { type: "object", properties: { command: { type: "string" } }, required: ["command"] } } },
  { type: "function", function: { name: "report", description: "Finish: report the defects you found (empty list if none) and whether the claimed result reproduced.", parameters: { type: "object", properties: { reproduced: { type: ["boolean", "null"] }, findings: { type: "array", items: { type: "object", properties: { category: { type: "string" }, summary: { type: "string" }, evidence: { type: "string" } }, required: ["category", "summary", "evidence"] } } }, required: ["findings", "reproduced"] } } },
];

const SYSTEM = `You are an ML research engineer auditing an experiment repository. You have tools to read files, write files and run commands in an isolated offline copy of the repository. Run the experiment, check whether the README's claimed result reproduces, and look for implementation or evaluation errors (data leakage, broken preprocessing, incorrect metrics, checkpoint failures, irreproducibility). Report only defects you have evidence for: a false alarm is as costly as a miss. Keep tool outputs small. You have a limited step budget; call report before it runs out.

${VOCABULARY}`;

const safePath = (root: string, p: string) => { const a = resolve(root, p || "."); if (!a.startsWith(root)) throw new Error("path outside the repository"); return a; };

export async function runAgent(o: { runDir: string; source: string; objective: string; model: ChatModel; isolation: Isolation; isolationEvidence: Evidence; maxSteps?: number }): Promise<AgentResult> {
  const stats = { runs: 0, ms: 0 };
  const sb = Sandbox.create(join(o.runDir, "sbx"), "agent", o.source, { isolation: o.isolation, limits: { ...DEFAULT_LIMITS, timeoutMs: 120_000 }, stats });
  const root = sb.dir;
  const log = join(o.runDir, "agent.jsonl");
  mkdirSync(o.runDir, { recursive: true });
  const messages: ChatMsg[] = [{ role: "system", content: SYSTEM }, { role: "user", content: `Objective: ${o.objective}\nThe repository is your working directory.` }];
  const maxSteps = o.maxSteps ?? 30;
  for (let step = 1; step <= maxSteps; step++) {
    if (step === maxSteps) messages.push({ role: "user", content: "Step budget exhausted: call report now." });
    let msg;
    try {
      ({ message: msg } = await o.model.chat(messages, { tools: TOOLS, maxTokens: 16000 }));
    } catch (e) {
      return { findings: [], reproduced: null, steps: step, sandboxRuns: stats.runs, sandboxMs: Math.round(stats.ms), stopped: "error", error: String(e) };
    }
    messages.push({ role: "assistant", content: msg.content ?? "", ...(msg.tool_calls ? { tool_calls: msg.tool_calls } : {}), ...(msg.reasoning_content ? { reasoning_content: msg.reasoning_content } : {}) });
    appendFileSync(log, JSON.stringify({ step, content: msg.content, tool_calls: msg.tool_calls }) + "\n");
    if (!msg.tool_calls?.length) { messages.push({ role: "user", content: "Answer only through tool calls: call the report tool now with your findings (categories from the list) and whether the claim reproduced." }); continue; }
    for (const call of msg.tool_calls) {
      let out: string;
      let args: Record<string, unknown> = {};
      try { args = JSON.parse(call.function.arguments || "{}"); } catch { /* bad args reported below */ }
      try {
        switch (call.function.name) {
          case "list_files": {
            const base = safePath(root, String(args.dir ?? "."));
            const walk = (d: string): string[] => readdirSync(d, { withFileTypes: true }).flatMap((e) => e.name.startsWith(".") ? [] : e.isDirectory() ? walk(join(d, e.name)) : [relative(root, join(d, e.name))]);
            out = walk(base).slice(0, 300).join("\n");
            break;
          }
          case "read_file": {
            const p = safePath(root, String(args.path));
            out = statSync(p).size > 2_000_000 ? "file too large" : readFileSync(p, "utf8").slice(0, 20_000);
            break;
          }
          case "write_file": {
            const p = safePath(root, String(args.path));
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
            const findings = (Array.isArray(args.findings) ? args.findings : []) as AgentFinding[];
            sb.remove();
            return { findings, reproduced: (args.reproduced as boolean | null) ?? null, steps: step, sandboxRuns: stats.runs, sandboxMs: Math.round(stats.ms), stopped: "report" };
          }
          default:
            out = `unknown tool ${call.function.name}`;
        }
      } catch (e) {
        out = `error: ${String((e as Error).message ?? e)}`;
      }
      messages.push({ role: "tool", tool_call_id: call.id, content: out });
    }
  }
  sb.remove();
  return { findings: [], reproduced: null, steps: maxSteps, sandboxRuns: stats.runs, sandboxMs: Math.round(stats.ms), stopped: "budget" };
}
