// Contracts of what agents actually call: the remote install/run tools and the
// campaign's summary of a run directory. A scripted model plays the agent; a
// fake remote backend records what each run received.
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { agentLoop, type Role } from "../src/mlh/agentloop.ts";
import { zeroUsage, type ChatMsg } from "../src/mlh/io/llm.ts";
import { allowedSilenceMs, lastProgressMs, summarize } from "../src/mlh/repro/campaign.ts";
import type { ChatModel } from "../src/mlh/skills/index.ts";

const tmp = () => mkdtempSync(join(tmpdir(), "mlh-contract-"));

// A model that issues one scripted tool call per step.
function scripted(calls: { name: string; args: Record<string, unknown> }[]): ChatModel {
  let i = 0;
  return {
    model: "scripted", usage: zeroUsage(),
    async chat(_m: ChatMsg[]) {
      const c = calls[Math.min(i++, calls.length - 1)]!;
      return { message: { role: "assistant" as const, content: "", tool_calls: [{ id: `c${i}`, type: "function" as const, function: { name: c.name, arguments: JSON.stringify(c.args) } }] }, usage: zeroUsage() };
    },
    async json<T>() { return {} as T; },
  };
}

const role: Role = { name: "env", system: "s", tools: ["run", "install"], writeCode: false, maxSteps: 10, report: { description: "r", parameters: { type: "object", properties: {} } } };

describe("remote tool contracts", () => {
  it("install switches the base image and keeps one spec per package; every later run receives them", async () => {
    const seen: { image?: string; pip: string[]; gpu?: string }[] = [];
    const src = tmp(); writeFileSync(join(src, "train.py"), "");
    const r = await agentLoop({
      role, workDir: tmp(), source: src, task: "t", isolation: "namespace", isolationEvidence: { kind: "deterministic", check: "x", ok: true },
      model: scripted([
        { name: "install", args: { packages: ["numpy==1.26.4", "scipy"] } },
        { name: "install", args: { image: "py3.8-torch1.12-cu113", packages: ["numpy==2.1.2", "-scipy"] } },
        { name: "run", args: { command: "python train.py", gpu: "T4" } },
        { name: "install", args: { image: "py9-nonexistent" } },
        { name: "report", args: { ok: true } },
      ]),
      remote: { pip: [], gpus: ["T4"], images: { "py3.8-torch1.12-cu113": "x", "py3.11-torch2.5-cu124": "y" }, image: "py3.11-torch2.5-cu124", run: async (_ws, _cmd, o) => { seen.push(o); return { code: 0, stdout: "", stderr: "", category: null, costUsd: 0 }; } },
    });
    expect(seen).toEqual([{ timeoutS: 600, pip: ["numpy==2.1.2"], gpu: "T4", image: "py3.8-torch1.12-cu113" }]);
    expect(r.pip).toEqual(["numpy==2.1.2"]);
    expect(r.image).toBe("py3.8-torch1.12-cu113"); // the unknown image was refused, the earlier choice stands
  });

  it("refuses package specs that are not plain requirement specifiers", async () => {
    const outputs: string[] = [];
    const src = tmp(); writeFileSync(join(src, "a"), "");
    await agentLoop({
      role, workDir: tmp(), source: src, task: "t", isolation: "namespace", isolationEvidence: { kind: "deterministic", check: "x", ok: true },
      model: { ...scripted([{ name: "install", args: { packages: ["git+https://evil/x.git"] } }, { name: "report", args: {} }]), async chat(m: ChatMsg[]) { const last = m.at(-1); if (last?.role === "tool") outputs.push(last.content); return scripted([{ name: outputs.length ? "report" : "install", args: outputs.length ? {} : { packages: ["git+https://evil/x.git"] } }]).chat(m); } },
      remote: { pip: [], gpus: [], run: async () => ({ code: 0, stdout: "", stderr: "", category: null, costUsd: 0 }) },
    });
    expect(outputs[0]).toMatch(/error: package specs rejected/);
  });
});

describe("harness-implemented tools", () => {
  it("routes an extra tool (e.g. check_plan) to the harness and returns its verdict to the agent", async () => {
    const src = tmp(); writeFileSync(join(src, "a"), "");
    const got: Record<string, unknown>[] = [];
    let seen = "";
    const model: ChatModel = { ...scripted([]), async chat(m: ChatMsg[]) {
      const last = m.at(-1);
      if (last?.role === "tool") seen = last.content;
      const call = seen ? { name: "report", args: { done: 1 } } : { name: "check_plan", args: { stages: [] } };
      return { message: { role: "assistant" as const, content: "", tool_calls: [{ id: "x", type: "function" as const, function: { name: call.name, arguments: JSON.stringify(call.args) } }] }, usage: zeroUsage() };
    } };
    const r = await agentLoop({ role: { ...role, tools: [] }, workDir: tmp(), source: src, task: "t", isolation: "namespace", isolationEvidence: { kind: "deterministic", check: "x", ok: true }, model,
      extraTools: [{ def: { type: "function", function: { name: "check_plan", description: "d", parameters: { type: "object", properties: {} } } }, run: async (a) => { got.push(a); return "REJECTED:\n- plan has no stages"; } }] });
    expect(got).toEqual([{ stages: [] }]);
    expect(seen).toMatch(/REJECTED/);
    expect(r).toMatchObject({ stopped: "report", report: { done: 1 } });
  });
});

describe("agent resume", () => {
  it("an agent killed at its last step resumes with one step in which it reports", async () => {
    const src = tmp(); writeFileSync(join(src, "a"), "");
    const wd = tmp();
    const run = (calls: { name: string; args: Record<string, unknown> }[]) => agentLoop({ role: { ...role, maxSteps: 2 }, workDir: wd, source: src, task: "t", checkpoint: true, isolation: "namespace", isolationEvidence: { kind: "deterministic", check: "x", ok: true }, model: scripted(calls), remote: { pip: [], gpus: [], run: async () => ({ code: 0, stdout: "", stderr: "", category: null, costUsd: 0 }) } });
    const first = await run([{ name: "install", args: { packages: ["a"] } }, { name: "install", args: { packages: ["b"] } }]);
    expect(first).toMatchObject({ report: null, stopped: "budget", steps: 2 });
    const again = await run([{ name: "report", args: { done: true } }]);
    expect(again).toMatchObject({ report: { done: true }, stopped: "report", steps: 3, resumedAt: 2 });
  });
});

describe("campaign summary", () => {
  it("summarizes a run directory: claim statuses, GPU and LLM cost, failure categories, fatal stop", () => {
    const d = tmp();
    mkdirSync(d, { recursive: true });
    writeFileSync(join(d, "repro.json"), JSON.stringify({ claims: [{ id: "ap", value: 0.42, status: "reproduced", mean: 0.42 }], failed: [{ id: "repro.stage:x:v0", error: "boom" }] }));
    writeFileSync(join(d, "budget.json"), JSON.stringify({ spent: 0.3 }));
    writeFileSync(join(d, "trace.jsonl"), [{ outcome: "ok", tokens: 100, costUsd: 0.01 }, { outcome: "fail", category: "experiment_error", tokens: 50, costUsd: 0.005 }].map((x) => JSON.stringify(x)).join("\n"));
    writeFileSync(join(d, "journal.jsonl"), [{ t: "ok", at: "t", id: "repro.plan", output: { gpuPlan: { fidelity: "eval_only", stages: [] } } }, { t: "ok", at: "t", id: "repro.env", output: { smoke: { ok: true }, image: "py3.11-torch2.5-cu124" } }].map((x) => JSON.stringify(x)).join("\n"));
    writeFileSync(join(d, "stopped.json"), JSON.stringify({ reason: "provider_fatal" }));
    const row = summarize({ id: "detr", arxiv: "2005.12872", repo: "r", commit: "c", objective: "o" }, d, 1, 600);
    expect(row).toMatchObject({ paper: "detr", fidelity: "eval_only", smoke: true, gpuUsd: 0.3, llmTokens: 150, llmUsd: 0.015, wallMin: 10, resumes: 1, failures: { experiment_error: 1 }, stopped: "provider_fatal" });
    expect(row.claims).toEqual([{ id: "ap", paper: 0.42, status: "reproduced", mean: 0.42 }]);
  });
});

describe("campaign watchdog", () => {
  it("allows 30 min of silence normally (agents write no journal events), and a running stage's full modal wait while it runs", () => {
    const d = tmp();
    const plan = { t: "ok", at: "t", id: "repro.plan", output: { gpuPlan: { stages: [{ id: "train", timeoutS: 5400 }] } } };
    writeFileSync(join(d, "journal.jsonl"), [plan, { t: "start", at: "t", id: "repro.env" }, { t: "ok", at: "t", id: "repro.env" }].map((x) => JSON.stringify(x)).join("\n"));
    expect(allowedSilenceMs(d)).toBe(30 * 60_000);
    writeFileSync(join(d, "journal.jsonl"), [plan, { t: "start", at: "t", id: "repro.stage:train:v0" }].map((x) => JSON.stringify(x)).join("\n"));
    expect(allowedSilenceMs(d)).toBe((5400 + 3000) * 1000);
    writeFileSync(join(d, "journal.jsonl"), [plan, { t: "start", at: "t", id: "repro.stage:train:v0" }, { t: "ok", at: "t", id: "repro.stage:train:v0" }].map((x) => JSON.stringify(x)).join("\n"));
    expect(allowedSilenceMs(d)).toBe(30 * 60_000);
  });

  it("counts an agent's checkpoint writes as progress, not only the journal", async () => {
    const d = tmp();
    writeFileSync(join(d, "journal.jsonl"), "{}\n");
    const j = lastProgressMs(d);
    await new Promise((r) => setTimeout(r, 20));
    mkdirSync(join(d, "work", "repro.env"), { recursive: true });
    writeFileSync(join(d, "work", "repro.env", "transcript.jsonl"), "{}\n");
    expect(lastProgressMs(d)).toBeGreaterThan(j);
  });
});

describe("time limits", () => {
  it("a run command is capped at the role's maxRunS; past wallMin the agent must report, and timings are logged", async () => {
    const seen: number[] = [];
    const src = tmp(), work = tmp(); writeFileSync(join(src, "train.py"), "");
    let msgs: ChatMsg[] = [];
    const slow: ChatModel = { model: "slow", usage: zeroUsage(), async chat(m: ChatMsg[]) { msgs = m; await new Promise((r) => setTimeout(r, 30)); return { message: { role: "assistant" as const, content: "", tool_calls: [{ id: "c", type: "function" as const, function: { name: "run", arguments: JSON.stringify({ command: "python train.py", timeout_s: 9999 }) } }] }, usage: zeroUsage() }; }, async json<T>() { return {} as T; } };
    const r = await agentLoop({
      role: { ...role, maxRunS: 300, wallMin: 0.0005 }, workDir: work, source: src, task: "t", isolation: "namespace", isolationEvidence: { kind: "deterministic", check: "x", ok: true }, model: slow,
      remote: { pip: [], gpus: [], images: {}, image: "y", run: async (_ws, _c, o) => { seen.push(o.timeoutS); return { code: 0, stdout: "", stderr: "", category: null, costUsd: 0 }; } },
    });
    expect(seen[0]).toBe(300);
    expect(r.stopped).toBe("budget");
    expect(r.steps).toBeLessThan(10);
    expect(String(msgs.at(-1)?.content)).toMatch(/Time budget .* exhausted/);
    const { readFileSync } = await import("node:fs");
    expect(readFileSync(join(work, "transcript.jsonl"), "utf8")).toMatch(/"timing":\{"llmMs":\d+,"tools":\[\{"name":"run","ms":\d+\}\]\}/);
  });
});

describe("no escaping the command time limit", async () => {
  const { backgrounds } = await import("../src/mlh/agentloop.ts");
  it("refuses background processes, allows ordinary commands", () => {
    for (const c of ["nohup python main.py > /cache/eval.log 2>&1 &", "python a.py &", "setsid python a.py", "(python a.py &) ; sleep 1", "tmux new -d python a.py"]) expect(backgrounds(c), c).toBe(true);
    for (const c of ["cd /work && python main.py --eval 2>&1 | tail -5", "python a.py > log 2>&1", "ls && echo ok", "cmd >&2"]) expect(backgrounds(c), c).toBe(false);
  });
});

describe("agents cannot write /cache/data", async () => {
  const { writesData } = await import("../src/mlh/agentloop.ts");
  it("refuses downloads and writes into /cache/data (LoRA r3 planner), allows reading it and writing elsewhere", () => {
    for (const c of ["mkdir -p /cache/data/roberta-base; cd /cache/data/roberta-base; curl -sSL -O https://huggingface.co/roberta-base/resolve/main/config.json", "python - <<'EOF'\nimport urllib.request\nurllib.request.urlretrieve(u, '/cache/data/x')\nEOF", "echo x > /cache/data/f", "cp a /cache/data/b"]) expect(writesData(c), c).toBe(true);
    for (const c of ["ls /cache/data/coco", "python train.py --data /cache/data/ppi", "curl -sSL -o /tmp/x https://x", "cat /cache/data/x.txt | head"]) expect(writesData(c), c).toBe(false);
  });
});
