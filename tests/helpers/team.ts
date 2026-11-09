// Offline harness for the GPU reproduction flow: a scripted team (one script per role, keyed by
// the role's system prompt, step = number of assistant turns so far, so a resumed conversation
// continues at the right step), a canary paper PDF, and the fake Modal runner.
import { cpSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { zeroUsage, type ChatMsg } from "../../src/mlh/io/llm.ts";
import type { ChatModel, Models } from "../../src/mlh/skills/index.ts";

export type Call = { name: string; args: Record<string, unknown> };
export interface TeamScript { claims: unknown; env: Call[]; plan: Call[]; debug?: Call[]; pick?: { index: number | null; why: string } }

export const CANARY_REPO = resolve("tests/fixtures/canary-repo");
export const FAKE_RUNNER = resolve("tests/fixtures/fake_modal.py");

export const canaryScript = (): TeamScript => ({
  claims: { claims: [{ id: "canary_acc", metric: "accuracy", value: 0.9, dataset: "synthetic" }] },
  env: [
    { name: "run", args: { command: "python3 train.py --steps 1" } },
    { name: "report", args: { runnable: true, launch: { cwd: ".", example_command: ["python3", "train.py", "--steps", "1"] }, problems: [] } },
  ],
  plan: [{ name: "report", args: { fidelity: "full", fidelityNote: "canary", seeds: [0], stages: [{ id: "train", argv: ["python3", "train.py", "--steps", "100"], cwd: "/work", timeoutS: 300, gpu: "T4" }], claims: [{ claim: "canary_acc", stage: "train", parse: "accuracy: ([0-9.]+)" }] } }],
});

const roleOf = (m: ChatMsg[]): keyof TeamScript | undefined => {
  const s = String(m[0]?.content ?? "");
  return /^You are the environment engineer/.test(s) ? "env" : /^You are the debugger/.test(s) ? "debug" : /^(Plan from the code|You are the planner)/.test(s) ? "plan" : undefined;
};

export function scriptedTeam(script: TeamScript, log: string[] = []): Models {
  const m: ChatModel = {
    model: "scripted", usage: zeroUsage(),
    async chat(msgs: ChatMsg[]) {
      const role = roleOf(msgs);
      // Metric disambiguation (values masked) asks for {index, why}.
      if (/map an experiment.s output line/.test(String(msgs[0]?.content ?? ""))) { log.push("pick"); return { message: { role: "assistant" as const, content: JSON.stringify(script.pick ?? { index: null, why: "scripted: none" }) }, usage: zeroUsage() }; }
      // Claims extraction asks for a JSON object in plain text (no tools).
      if (!role) { log.push("claims"); return { message: { role: "assistant" as const, content: JSON.stringify(script.claims) }, usage: zeroUsage() }; }
      const calls = (role && role !== "claims" ? (script[role] as Call[] | undefined) : undefined) ?? [];
      const k = msgs.filter((x) => x.role === "assistant").length;
      const c = calls[Math.min(k, calls.length - 1)] ?? { name: "report", args: {} };
      log.push(`${role}:${k}:${c.name}`);
      return { message: { role: "assistant" as const, content: "", tool_calls: [{ id: `${role}-${k}`, type: "function" as const, function: { name: c.name, arguments: JSON.stringify(c.args) } }] }, usage: zeroUsage() };
    },
    async json<T>(system: string) {
      if (/map an experiment.s output line/.test(system)) { log.push("pick"); return (script.pick ?? { index: null, why: "scripted: none" }) as T; }
      log.push("claims"); return script.claims as T;
    },
  } as unknown as ChatModel;
  return { proposer: m, reviewer: m };
}

// A one-page PDF whose text layer holds `text` (pdftotext reads it).
export function canaryPdf(text: string): Buffer {
  const stream = `BT /F1 12 Tf 72 720 Td (${text.replace(/[()\\]/g, "")}) Tj ET`;
  const objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>", "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>", `<< /Length ${stream.length} >>\nstream\n${stream}\nendstream`, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"];
  let out = "%PDF-1.4\n";
  const offs: number[] = [];
  objs.forEach((o, i) => { offs.push(out.length); out += `${i + 1} 0 obj\n${o}\nendobj\n`; });
  const xref = out.length;
  out += `xref\n0 ${objs.length + 1}\n0000000000 65535 f \n${offs.map((x) => `${String(x).padStart(10, "0")} 00000 n \n`).join("")}trailer\n<< /Size ${objs.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  return Buffer.from(out, "latin1");
}

// A fresh, isolated workspace: run dir with the paper in place, fake cache, env cache and sites.
export function workspace(faults: unknown[] = []): { runDir: string; env: Record<string, string> } {
  const root = mkdtempSync(join(tmpdir(), "mlh-team-"));
  const runDir = join(root, "run");
  mkdirSync(join(runDir, "work", "repro.paper"), { recursive: true });
  writeFileSync(join(runDir, "work", "repro.paper", "paper.pdf"), canaryPdf("Canary model: test accuracy of 90.0% on the synthetic task."));
  return { runDir, env: { MLH_MODAL_RUNNER: FAKE_RUNNER, MLH_FAKE_CACHE: join(root, "cache"), MLH_FAKE_FAULTS: JSON.stringify(faults), MLH_FAKE_ELAPSED: "60", MLH_ENV_CACHE: join(root, "env-cache"), MLH_BACKOFF_MS: "0", MLH_SKIP_URL_PROBE: "1" } };
}

export const withEnv = async <T>(env: Record<string, string>, f: () => Promise<T>): Promise<T> => {
  const old = Object.fromEntries(Object.keys(env).map((k) => [k, process.env[k]]));
  Object.assign(process.env, env);
  try { return await f(); } finally { for (const [k, v] of Object.entries(old)) if (v === undefined) delete process.env[k]; else process.env[k] = v; }
};
export { cpSync };
