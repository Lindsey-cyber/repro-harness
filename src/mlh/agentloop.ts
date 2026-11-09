// A generic, checkpointed tool-using agent loop. A role fixes the system
// prompt, which tools exist, whether code may be edited, and the report schema.
// Every step is persisted atomically, so a killed or failed run resumes at the
// last completed step. Running code always goes through authorization and the
// sandbox; installing packages through the install handler the harness provides.
import { appendFileSync, cpSync, existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { authorize, type Evidence } from "./core/authz.ts";
import type { ChatMsg, ToolDef } from "./io/llm.ts";
import { DEFAULT_LIMITS, Sandbox, SkillError, type Isolation, type Limits, type SandboxStats } from "./io/sandbox.ts";
import { checkSpecs, mergeSpecs } from "./repro/site.ts";
import type { ChatModel } from "./skills/index.ts";

export type ToolName = "list_files" | "read_file" | "write_file" | "run" | "install" | "read_paper";

export interface Role {
  name: string;
  system: string;
  tools: ToolName[];
  writeCode: boolean; // may write_file touch source code (.py etc.)
  report: { description: string; parameters: Record<string, unknown> };
  maxSteps: number;
}

export interface LoopOptions {
  role: Role;
  workDir: string; // persistent: ws/ (the working copy), state.json, transcript
  source: string; // copied into ws/ on a fresh start
  task: string; // the first user message
  model: ChatModel;
  isolation: Isolation;
  isolationEvidence: Evidence;
  limits?: Limits;
  checkpoint?: boolean;
  site?: string; // initial package site
  install?: (packages: string[], indexUrl: string | undefined, currentSite: string | undefined) => Promise<{ site?: string; message: string }>;
  paper?: string; // full paper text for read_paper
  sandboxFault?: () => boolean;
  crashAfterSteps?: number;
  // Resuming a finished conversation to keep working: appended as a user message before the next step.
  continueWith?: string;
  // Role-specific tools the harness implements (e.g. the planner's check_plan, which runs the plan validator).
  extraTools?: { def: ToolDef; run: (args: Record<string, unknown>) => Promise<string> }[];
  // Remote execution (e.g. Modal): `run` uploads the working copy and runs there (optionally on a GPU);
  // `install` adds packages to the remote image instead of a local site.
  remote?: {
    run(ws: string, command: string, o: { timeoutS: number; gpu?: string; pip: string[]; image?: string }): Promise<{ code: number; stdout: string; stderr: string; category: string | null; costUsd: number }>;
    pip: string[]; // initial package list
    gpus: string[]; // GPU types the role may ask for
    images?: Record<string, string>; // base images the role may choose (key -> registry image)
    image?: string; // initial image key
  };
}

export interface LoopResult {
  report: Record<string, unknown> | null;
  steps: number;
  sandboxRuns: number;
  sandboxMs: number;
  stopped: "report" | "budget";
  site?: string;
  workspace: string;
  resumedAt?: number;
  pip: string[];
  image?: string;
}

interface State { messages: ChatMsg[]; step: number; stats: SandboxStats; site?: string; pip?: string[]; image?: string }

const CODE = /\.(py|pyx|ipynb|sh|c|cc|cpp|cu|h)$/i;

const DEFS: Record<Exclude<ToolName, never>, ToolDef> = {
  list_files: { type: "function", function: { name: "list_files", description: "List files under a directory of the repository (recursive, max 300).", parameters: { type: "object", properties: { dir: { type: "string" } }, required: [] } } },
  read_file: { type: "function", function: { name: "read_file", description: "Read a text file (max 20000 chars; use offset for more).", parameters: { type: "object", properties: { path: { type: "string" }, offset: { type: "integer" } }, required: ["path"] } } },
  write_file: { type: "function", function: { name: "write_file", description: "Write a file in your working copy.", parameters: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path", "content"] } } },
  run: { type: "function", function: { name: "run", description: "Run a shell command in the repository root inside an offline sandbox (installed packages are on PYTHONPATH). Returns exit code and output tails.", parameters: { type: "object", properties: { command: { type: "string" }, timeout_s: { type: "integer" } }, required: ["command"] } } },
  install: { type: "function", function: { name: "install", description: "Install Python packages (pip requirement specifiers like 'scipy' or 'networkx<3') from PyPI, or torch-family packages from the PyTorch CPU index (index='pytorch-cpu'). Returns the outcome.", parameters: { type: "object", properties: { packages: { type: "array", items: { type: "string" } }, index: { type: "string", enum: ["pypi", "pytorch-cpu"] } }, required: ["packages"] } } },
  read_paper: { type: "function", function: { name: "read_paper", description: "Read the paper's text (extracted from the PDF), 12000 chars at a time.", parameters: { type: "object", properties: { offset: { type: "integer" } }, required: [] } } },
};

const safePath = (root: string, p: string) => { const a = resolve(root, p || "."); if (a !== root && !a.startsWith(root + "/")) throw new Error("path outside the repository"); return a; };

// Read-only inspection of the repository (ls, cat, grep, sed -n ...) does not need a remote
// container: a fresh one per call cost about a minute each, and planners make dozens. Such a
// command runs in the local sandbox instead, with /work mapped to the repository root. Anything
// that touches /cache, writes, substitutes commands or runs a program goes remote. Pure.
const INSPECT = new Set(["ls", "cat", "head", "tail", "grep", "egrep", "find", "wc", "pwd", "echo", "cd", "sed", "sort", "uniq", "tree", "file", "du", "stat", "basename", "dirname", "true", "nl", "cut", "diff"]);
export function localInspection(command: string): string | undefined {
  if (/\/cache\b|`|\$\(|<<|\bsudo\b/.test(command)) return undefined;
  if (/(^|[^2&])>|>>|\btee\b|-i\b|\bsed\b[^|;&]*\bw\b|\bfind\b[^|;&]*-(exec|delete|ok)/.test(command.replace(/2>\s*\/dev\/null|2>&1/g, ""))) return undefined;
  // Quoted text (grep patterns like "a\|b") is data, not shell structure.
  const masked = command.replace(/'[^']*'|"(?:[^"\\]|\\.)*"/g, "Q");
  if (/[`$]\(|`/.test(masked.replace(/Q/g, ""))) return undefined;
  const segments = masked.split(/&&|\|\||;|\||\n/).map((x) => x.trim()).filter(Boolean);
  if (!segments.length || !segments.every((g) => INSPECT.has(g.split(/\s+/)[0]!))) return undefined;
  // sed only as "sed -n 'N,Mp' file" (GNU sed scripts can write files and run commands).
  const seds = command.match(/\bsed\b[^|;&\n]*/g) ?? [];
  if (seds.some((x) => !/^sed\s+-n\s+(['"]?)\d+(,(\d+|\$))?p\1(\s+[\w./-]+)+\s*$/.test(x.trim()))) return undefined;
  return command.replace(/\/work\//g, "./").replace(/\/work\b/g, ".");
}

export async function agentLoop(o: LoopOptions): Promise<LoopResult> {
  mkdirSync(o.workDir, { recursive: true });
  const statePath = join(o.workDir, "state.json");
  const ws = join(o.workDir, "ws");
  const log = join(o.workDir, "transcript.jsonl");
  let st: State | undefined = o.checkpoint && existsSync(statePath) && existsSync(ws) ? (JSON.parse(readFileSync(statePath, "utf8")) as State) : undefined;
  const resumedAt = st?.step;
  if (!st) {
    rmSync(ws, { recursive: true, force: true });
    cpSync(o.source, ws, { recursive: true, filter: (p) => !/\/(\.git|__pycache__)(\/|$)/.test(p) });
    st = { messages: [{ role: "system", content: o.role.system }, { role: "user", content: o.task }], step: 0, stats: { runs: 0, ms: 0 }, ...(o.site ? { site: o.site } : {}), ...(o.remote ? { pip: [...o.remote.pip], ...(o.remote.image ? { image: o.remote.image } : {}) } : {}) };
  }
  const state = st;
  if (o.continueWith && resumedAt !== undefined) state.messages = [...state.messages, { role: "user", content: o.continueWith }];
  const limits = o.limits ?? DEFAULT_LIMITS;
  const sandbox = (timeoutS?: number) => Sandbox.attach(ws, { isolation: o.isolation, limits: { ...limits, ...(timeoutS ? { timeoutMs: Math.min(timeoutS * 1000, limits.timeoutMs) } : {}) }, stats: state.stats, ...(state.site ? { site: state.site } : {}), ...(o.sandboxFault ? { fault: o.sandboxFault } : {}) });
  const save = () => {
    if (!o.checkpoint) return;
    writeFileSync(statePath + ".tmp", JSON.stringify(state));
    renameSync(statePath + ".tmp", statePath);
  };
  const remoteDefs: Partial<Record<ToolName, ToolDef>> = o.remote ? {
    run: { type: "function", function: { name: "run", description: `Run a shell command in the repository root inside a remote sandbox (Linux, CUDA PyTorch 2.5.1 plus the packages installed so far). Omit gpu for CPU (cheap); gpu one of ${o.remote.gpus.join(", ")} costs money from the paper's budget. Your local edits are uploaded on every call; files the command writes are NOT kept, except under /cache (a persistent volume). Returns exit code and output tails.`, parameters: { type: "object", properties: { command: { type: "string" }, timeout_s: { type: "integer" }, gpu: { type: "string" } }, required: ["command"] } } },
    install: { type: "function", function: { name: "install", description: `Set the remote environment: optionally a base image (old repositories need old Python/PyTorch: ${Object.keys(o.remote.images ?? {}).join(", ")}; default ${o.remote.image ?? "py3.11-torch2.5-cu124"}) and pip requirement specifiers (e.g. 'transformers==4.30.2'). A specifier replaces any earlier one for the same package; '-name' removes a package. Later runs use the result; run something after changing it.`, parameters: { type: "object", properties: { packages: { type: "array", items: { type: "string" } }, image: { type: "string" } }, required: [] } } },
  } : {};
  const defs: Record<ToolName, ToolDef> = { ...DEFS, ...remoteDefs };
  const tools = [...o.role.tools.map((t) => defs[t]), ...(o.extraTools ?? []).map((t) => t.def), { type: "function" as const, function: { name: "report", description: o.role.report.description, parameters: o.role.report.parameters } }];
  const finish = (report: Record<string, unknown> | null, stopped: LoopResult["stopped"]): LoopResult => ({ report, steps: state.step, sandboxRuns: state.stats.runs, sandboxMs: Math.round(state.stats.ms), stopped, workspace: ws, pip: state.pip ?? [], ...(state.image ? { image: state.image } : {}), ...(state.site ? { site: state.site } : {}), ...(resumedAt !== undefined ? { resumedAt } : {}) });
  let here = 0;
  // A run killed at its last step resumes with one step left, in which it must report.
  const limit = Math.max(o.role.maxSteps, resumedAt !== undefined ? state.step + 1 : 0);
  while (state.step < limit) {
    const step = state.step + 1;
    const msgs: ChatMsg[] = step === limit ? [...state.messages, { role: "user", content: "Step budget exhausted: call report now." }] : state.messages;
    const { message: msg } = await o.model.chat(msgs, { tools, maxTokens: 16000 });
    const next: ChatMsg[] = [...msgs, { role: "assistant", content: msg.content ?? "", ...(msg.tool_calls ? { tool_calls: msg.tool_calls } : {}), ...(msg.reasoning_content ? { reasoning_content: msg.reasoning_content } : {}) }];
    appendFileSync(log, JSON.stringify({ step, content: msg.content, tool_calls: msg.tool_calls }) + "\n");
    if (!msg.tool_calls?.length) next.push({ role: "user", content: "Answer only through tool calls; finish by calling report." });
    let reported: Record<string, unknown> | null = null;
    for (const call of msg.tool_calls ?? []) {
      let args: Record<string, unknown> = {};
      let badArgs = false;
      try { args = JSON.parse(call.function.arguments || "{}"); } catch { badArgs = true; }
      let out: string;
      try {
        // Malformed arguments are returned to the agent to resend, never read as an empty report.
        if (badArgs) throw new Error(`the arguments of ${call.function.name} are not valid JSON (${call.function.arguments.length} chars); resend them, keeping them shorter`);
        const name = call.function.name as ToolName | "report";
        const extra = o.extraTools?.find((t) => t.def.function.name === name);
        if (extra) { out = await extra.run(args); next.push({ role: "tool", tool_call_id: call.id, content: out }); continue; }
        if (name !== "report" && !o.role.tools.includes(name)) throw new Error(`tool ${name} is not available to the ${o.role.name} role`);
        switch (name) {
          case "report":
            reported = args; out = "reported";
            break;
          case "list_files": {
            const base = safePath(ws, String(args.dir ?? "."));
            const walk = (d: string): string[] => readdirSync(d, { withFileTypes: true }).flatMap((e) => e.name.startsWith(".") || e.name === "__pycache__" ? [] : e.isDirectory() ? walk(join(d, e.name)) : [relative(ws, join(d, e.name))]);
            out = walk(base).slice(0, 300).join("\n");
            break;
          }
          case "read_file": {
            const p = safePath(ws, String(args.path));
            const off = Number(args.offset ?? 0);
            out = statSync(p).size > 5_000_000 ? "file too large" : readFileSync(p, "utf8").slice(off, off + 20_000);
            break;
          }
          case "write_file": {
            const rel = String(args.path);
            if (!o.role.writeCode && CODE.test(rel)) throw new Error(`the ${o.role.name} role may not edit source code (${rel}); describe the needed change in your report instead`);
            const p = safePath(ws, rel);
            mkdirSync(dirname(p), { recursive: true });
            writeFileSync(p, String(args.content));
            out = "written";
            break;
          }
          case "run": {
            const local = o.remote && !args.gpu ? localInspection(String(args.command)) : undefined;
            if (o.remote && !local) {
              const gpu = typeof args.gpu === "string" && o.remote.gpus.includes(args.gpu) ? args.gpu : undefined;
              const r = await o.remote.run(ws, String(args.command), { timeoutS: Math.min(Number(args.timeout_s) || 600, 3600), pip: state.pip ?? [], ...(gpu ? { gpu } : {}), ...(state.image ? { image: state.image } : {}) });
              out = `exit ${r.category === "timeout" ? "timeout" : r.code}${r.costUsd ? ` (cost $${r.costUsd.toFixed(3)})` : ""}\nstdout:\n${r.stdout.slice(-4000)}\nstderr:\n${r.stderr.slice(-2500)}`;
              break;
            }
            const d = authorize(o.isolation === "namespace" ? "exec_sandboxed" : "exec_unsandboxed", [o.isolationEvidence]);
            if (!d.allow) { out = `denied: ${d.reason}`; break; }
            const r = await sandbox(Number(args.timeout_s) || undefined).run(["sh", "-c", local ?? String(args.command)]);
            out = `exit ${r.category === "timeout" ? "timeout" : (/^MLH_EXIT=(\d+)$/m.exec(r.stderr)?.[1] ?? r.code)}\nstdout:\n${r.stdout.slice(-4000)}\nstderr:\n${r.stderr.replace(/^MLH_EXIT=\d+\n?/m, "").slice(-2500)}`;
            break;
          }
          case "install": {
            if (o.remote) {
              if (typeof args.image === "string") {
                if (!o.remote.images?.[args.image]) throw new Error(`unknown image ${args.image}; choose one of: ${Object.keys(o.remote.images ?? {}).join(", ")}`);
                state.image = args.image;
              }
              const pk = (Array.isArray(args.packages) ? args.packages : []).map(String);
              const ev = pk.length ? checkSpecs(pk.map((p) => p.replace(/^-/, ""))) : { ok: true, check: "" };
              if (!ev.ok) throw new Error(ev.check);
              state.pip = mergeSpecs(state.pip ?? [], pk);
              out = `remote image: ${state.image ?? "default"}; packages: ${state.pip.join(" ") || "(none)"} (built on the next run; verify with a run)`;
              break;
            }
            if (!o.install) throw new Error("installing is not available");
            const pk = (Array.isArray(args.packages) ? args.packages : []).map(String);
            const r = await o.install(pk, args.index === "pytorch-cpu" ? "https://download.pytorch.org/whl/cpu" : undefined, state.site);
            if (r.site) state.site = r.site;
            out = r.message;
            break;
          }
          case "read_paper": {
            const off = Number(args.offset ?? 0);
            out = o.paper ? `${o.paper.slice(off, off + 12000)}\n[${Math.min(off + 12000, o.paper.length)} of ${o.paper.length} chars]` : "no paper available";
            break;
          }
          default:
            out = `unknown tool ${String(name)}`;
        }
      } catch (e) {
        // Out of credit (GPU or model provider): the run must stop, not hand the error to the agent to retry.
        if ((e instanceof SkillError && e.category === "provider_fatal") || /Insufficient Balance/.test(String(e))) throw e;
        out = `error: ${String(e instanceof SkillError ? e.message : (e as Error).message ?? e)}`;
      }
      next.push({ role: "tool", tool_call_id: call.id, content: out });
    }
    state.messages = next;
    state.step = step;
    save();
    if (reported) return finish(reported, "report");
    if (o.crashAfterSteps !== undefined && ++here >= o.crashAfterSteps) process.exit(137);
  }
  return finish(null, "budget");
}
