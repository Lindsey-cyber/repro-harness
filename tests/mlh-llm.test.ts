// The LLM protocol with scripted models: what a model says only counts once an
// executed check supports it AND a second model judges the check valid.
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { authorize } from "../src/mlh/core/authz.ts";
import { meter, zeroUsage, type ChatMsg } from "../src/mlh/io/llm.ts";
import { runHarness } from "../src/mlh/orchestrator.ts";
import type { ChatModel } from "../src/mlh/skills/index.ts";
import { validateManifest } from "../src/mlh/skills/repo.ts";

const ROOT = join(import.meta.dirname, "..");
const tmp = () => mkdtempSync(join(tmpdir(), "mlh-llm-"));

function scripted(model: string, answer: (system: string, user: string) => unknown): ChatModel {
  const usage = zeroUsage();
  const m: ChatModel = {
    model, usage,
    async chat(messages: ChatMsg[]) {
      const sys = messages.find((x) => x.role === "system")?.content ?? "", user = messages.find((x) => x.role === "user")?.content ?? "";
      const u = { ...zeroUsage(), calls: 1, promptTokens: 100, completionTokens: 10 };
      usage.calls++; usage.promptTokens += 100; usage.completionTokens += 10;
      return { message: { role: "assistant" as const, content: JSON.stringify(answer(sys as string, user as string)) }, usage: u };
    },
    async json<T>(system: string, user: string) { return answer(system, user) as T; },
  };
  return m;
}

const CHECK = 'import json\nprint(json.dumps({"supports": True, "observed": {"x": 1}}))\n';
const proposer = scripted("fake-proposer", (sys) => (sys.includes("Propose at most 4") ? { hypotheses: [{ category: "other:label_noise", file: "train.py:1", rationale: "labels are shuffled", check: CHECK }] } : {}));

describe("LLM protocol", () => {
  it("a model's approval never authorizes sending code to a provider; the human's --llm approval does", () => {
    expect(authorize("model_query", [{ kind: "llm", check: "self", ok: true }]).allow).toBe(false);
    expect(authorize("model_query", [{ kind: "human", check: "--llm", ok: true, action: "model_query" }]).allow).toBe(true);
  });

  it("confirms a model hypothesis only when its executed check supports it and a different model judges it valid", async () => {
    const yes = await runHarness({ runDir: tmp(), source: join(ROOT, "bench/ml/tasks/clean/repo"), objective: "check for errors", llm: { proposer, reviewer: scripted("fake-reviewer", () => ({ valid: true, reason: "sound" })) } });
    expect(yes.summary.confirmed.map((f) => f.category)).toEqual(["other:label_noise"]);
    expect(yes.summary.confirmed[0]!.confirmations[0]!.note).toMatch(/fake-reviewer review: valid/);
    expect(yes.stats.llm.calls).toBeGreaterThan(0);

    const no = await runHarness({ runDir: tmp(), source: join(ROOT, "bench/ml/tasks/clean/repo"), objective: "check for errors", llm: { proposer, reviewer: scripted("fake-reviewer", () => ({ valid: false, reason: "tests nothing" })) } });
    expect(no.summary.confirmed).toEqual([]);
    expect(no.summary.refuted.map((f) => f.category)).toEqual(["other:label_noise"]);
  }, 60_000);

  it("plans no model skill unless models are given", async () => {
    const r = await runHarness({ runDir: tmp(), source: join(ROOT, "bench/ml/tasks/clean/repo"), objective: "check for errors" });
    expect([...r.state.nodes.keys()].some((k) => k.startsWith("llm."))).toBe(false);
  }, 60_000);

  it("keeps only manifest fields that pass deterministic checks", () => {
    const root = tmp();
    mkdirSync(join(root, "src"));
    writeFileSync(join(root, "src/run.py"), "");
    writeFileSync(join(root, "tr.csv"), "a,b,y\n1,2,0\n");
    writeFileSync(join(root, "te.csv"), "a,b,y\n1,2,0\n");
    writeFileSync(join(root, "README.md"), "We get 87.5% accuracy.");
    const v = validateManifest(root, { train: ["python", "src/run.py"], eval: ["python3", "nope.py"], data: { train: "tr.csv", test: "te.csv", label: "y" }, outputs: { metrics: "/etc/passwd", predictions: "out/p.csv" }, metric: "accuracy", claim: { metric: "accuracy", value: 0.9 } });
    expect(v.accepted.train).toEqual(["python3", "src/run.py"]);
    expect(v.rejected.eval).toMatch(/does not exist/);
    expect(v.accepted.data).toEqual({ train: "tr.csv", test: "te.csv", label: "y" });
    expect(v.accepted.outputs).toEqual({ predictions: "out/p.csv" }); // absolute path dropped
    expect(v.rejected.claim).toMatch(/does not appear/); // 0.9 is not in the README
    expect(validateManifest(root, { claim: { metric: "accuracy", value: 87.5 } }).accepted.claim).toEqual({ metric: "accuracy", value: 0.875 });
  });

  it("meters each node's model usage separately", async () => {
    const m = meter(scripted("x", () => ({ ok: 1 })));
    await m.json("s", "u");
    expect(m.usage).toMatchObject({ calls: 1, promptTokens: 100, completionTokens: 10 });
  });
});
