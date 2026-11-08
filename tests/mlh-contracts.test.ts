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
  it("allows 2 h of silence normally (agents write no journal events), and a running stage's full modal wait while it runs", () => {
    const d = tmp();
    const plan = { t: "ok", at: "t", id: "repro.plan", output: { gpuPlan: { stages: [{ id: "train", timeoutS: 5400 }] } } };
    writeFileSync(join(d, "journal.jsonl"), [plan, { t: "start", at: "t", id: "repro.env" }, { t: "ok", at: "t", id: "repro.env" }].map((x) => JSON.stringify(x)).join("\n"));
    expect(allowedSilenceMs(d)).toBe(2 * 3600_000);
    writeFileSync(join(d, "journal.jsonl"), [plan, { t: "start", at: "t", id: "repro.stage:train:v0" }].map((x) => JSON.stringify(x)).join("\n"));
    expect(allowedSilenceMs(d)).toBe((5400 + 3000) * 1000);
    writeFileSync(join(d, "journal.jsonl"), [plan, { t: "start", at: "t", id: "repro.stage:train:v0" }, { t: "ok", at: "t", id: "repro.stage:train:v0" }].map((x) => JSON.stringify(x)).join("\n"));
    expect(allowedSilenceMs(d)).toBe(2 * 3600_000);
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
