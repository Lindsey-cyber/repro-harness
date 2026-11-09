// P1 repeatability and P2 failure injection on the full GPU reproduction flow, offline.
// Every injected fault must end in the clean run's result or in a named stop, be attributed to
// the right class, never be blamed on an agent, and never re-run a finished paid stage.
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { costOfRun } from "../src/mlh/repro/campaign.ts";
import { roleOutcomes } from "../src/mlh/repro/trustscore.ts";
import { runHarness } from "../src/mlh/orchestrator.ts";
import { CANARY_REPO, canaryScript, scriptedTeam, withEnv, workspace } from "./helpers/team.ts";

const CLEAN = [["canary_acc", "reproduced", 0.9]];
const claims = (runDir: string) => (JSON.parse(readFileSync(join(runDir, "repro.json"), "utf8")).claims as { id: string; status: string; mean?: number }[]).map((c) => [c.id, c.status, c.mean ?? null]);
const stopped = (runDir: string) => (existsSync(join(runDir, "stopped.json")) ? JSON.parse(readFileSync(join(runDir, "stopped.json"), "utf8")).reason : undefined);
const journal = (runDir: string) => readFileSync(join(runDir, "journal.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l));
const stageRuns = (env: Record<string, string>) => readFileSync(join(env.MLH_FAKE_CACHE!, ".fake-calls.jsonl"), "utf8").trim().split("\n").filter((l) => /train\.py --steps 100$/.test((JSON.parse(l).cmd as string[]).join(" "))).length;
const blamed = (runDir: string) => Object.entries(roleOutcomes(journal(runDir))).filter(([, o]) => o && !o.ok).map(([r]) => r);

async function run(ws: ReturnType<typeof workspace>, script = canaryScript()) {
  return withEnv(ws.env, () => runHarness({ runDir: ws.runDir, mode: "repro", llm: scriptedTeam(script), concurrency: 1, ...(existsSync(join(ws.runDir, "journal.jsonl")) ? {} : { source: CANARY_REPO, objective: "Reproduce the canary accuracy.", arxiv: "2101.00001", gpuBudgetUsd: 1 }) }));
}

// Recorded Modal messages (tests/fixtures/modal-errors.json) and two command-level failures.
const CAPACITY = "RuntimeError: ResourceExhaustedError: Insufficient capacity to create sandbox.";
const SPEND = "RuntimeError: ResourceExhaustedError: Workspace ac-X has exceeded its spend limit";
const RESET = "ConnectionError: [Errno 104] Connection reset by peer";
const BUILD = "ImageBuildError: Image build for im-123 failed";
const STAGE = "^python3 train\\.py --steps 100$"; // the planned stage only
const SMOKE = "MLH_SMOKE_EXIT"; // the harness's smoke test only (not the env agent's own runs)

describe("P1: repeatability", () => {
  it("10 runs from scratch give the same result, the same cost and no stop", async () => {
    const seen = new Set<string>();
    for (let i = 0; i < 10; i++) {
      const ws = workspace();
      await run(ws);
      seen.add(JSON.stringify({ claims: claims(ws.runDir), stopped: stopped(ws.runDir) ?? null, usd: costOfRun(ws.runDir).totalUsd.toFixed(6), stageRuns: stageRuns(ws.env) }));
    }
    expect(seen.size).toBe(1);
    expect(JSON.parse([...seen][0]!)).toMatchObject({ claims: CLEAN, stopped: null, stageRuns: 1 });
  }, 300_000);
});

describe("P2: platform faults", () => {
  it("a capacity shortage once: retried, same result, cost shown as retried infrastructure, nobody blamed", async () => {
    const ws = workspace([{ when: STAGE, nth: [1], error: CAPACITY }]);
    await run(ws);
    expect(claims(ws.runDir)).toEqual(CLEAN);
    expect(stopped(ws.runDir)).toBeUndefined();
    expect(costOfRun(ws.runDir).retried.infrastructure.attempts).toBe(1);
    expect(blamed(ws.runDir)).toEqual([]);
  }, 60_000);

  it("the same fault twice stops the run (infra_repeat); a resume after investigation finishes with the clean result", async () => {
    const ws = workspace([{ when: STAGE, nth: [1, 2], error: CAPACITY }]);
    await run(ws);
    expect(stopped(ws.runDir)).toBe("infra_repeat");
    await run(ws);
    expect(stopped(ws.runDir)).toBeUndefined();
    expect(claims(ws.runDir)).toEqual(CLEAN);
    expect(stageRuns(ws.env)).toBe(1);
    expect(blamed(ws.runDir)).toEqual([]);
  }, 60_000);

  it("a spend limit stops the run (provider_fatal) without blaming anyone; resume finishes", async () => {
    const ws = workspace([{ when: STAGE, nth: [1], error: SPEND }]);
    await run(ws);
    expect(stopped(ws.runDir)).toBe("provider_fatal");
    expect(costOfRun(ws.runDir).failed.resource.attempts + costOfRun(ws.runDir).retried.resource.attempts).toBe(1);
    await run(ws);
    expect(claims(ws.runDir)).toEqual(CLEAN);
    expect(blamed(ws.runDir)).toEqual([]);
  }, 60_000);

  it("a transport error during the smoke test: retried, the env agent is not blamed", async () => {
    const ws = workspace([{ when: SMOKE, nth: [1], error: RESET }]);
    await run(ws);
    expect(claims(ws.runDir)).toEqual(CLEAN);
    expect(roleOutcomes(journal(ws.runDir)).env).toMatchObject({ ok: true });
  }, 60_000);

  it("a stage timeout (exit 124): retried as the paper's issue, same result", async () => {
    const ws = workspace([{ when: STAGE, nth: [1], exit: 124, stderr: "timeout" }]);
    await run(ws);
    expect(claims(ws.runDir)).toEqual(CLEAN);
    expect(costOfRun(ws.runDir).retried.paper.attempts).toBe(1);
  }, 60_000);

  it("an image build failure at the smoke test is the env agent's (repair round), and the run still finishes", async () => {
    const ws = workspace([{ when: SMOKE, nth: [1], error: BUILD }]);
    await run(ws);
    expect(claims(ws.runDir)).toEqual(CLEAN);
    expect(roleOutcomes(journal(ws.runDir)).env).toMatchObject({ ok: false, why: expect.stringMatching(/repair/) });
  }, 60_000);

  it("GPU out of memory: classed resource, never blamed on an agent, the run ends with a named outcome", async () => {
    const ws = workspace([{ when: STAGE, nth: [1, 2, 3, 4, 5, 6], exit: 1, stderr: "torch.cuda.OutOfMemoryError: CUDA out of memory" }]);
    await run(ws);
    const c = costOfRun(ws.runDir);
    expect(c.failed.resource.attempts + c.retried.resource.attempts).toBeGreaterThan(0);
    expect(blamed(ws.runDir).filter((r) => r !== "debugger")).toEqual([]);
    expect(claims(ws.runDir).length === 1 || stopped(ws.runDir) !== undefined).toBe(true);
  }, 120_000);
});

describe("P2: model faults", () => {
  it("one malformed model answer for the claims: retried, same result", async () => {
    const ws = workspace();
    const s = canaryScript();
    let n = 0;
    const team = scriptedTeam(s);
    const chat = team.proposer.chat.bind(team.proposer);
    team.proposer.chat = (async (m: never, o: never) => (n++ === 0 && !String((m as { content?: string }[])[0]?.content ?? "").startsWith("You are") ? { message: { role: "assistant", content: "not json" }, usage: (await chat(m, o)).usage } : chat(m, o))) as never;
    await withEnv(ws.env, () => runHarness({ runDir: ws.runDir, mode: "repro", llm: team, concurrency: 1, source: CANARY_REPO, objective: "Reproduce the canary accuracy.", arxiv: "2101.00001", gpuBudgetUsd: 1 }));
    expect(claims(ws.runDir)).toEqual(CLEAN);
  }, 60_000);
});

describe("P2: process killed and resumed", () => {
  const driver = resolve("tests/helpers/canary-driver.mts");
  const child = (env: Record<string, string>) => { try { execFileSync("npx", ["tsx", driver], { env: { ...process.env, ...env }, stdio: "pipe", timeout: 120_000 }); return 0; } catch (e) { return (e as { status?: number }).status ?? -1; } };
  for (const k of [1, 2, 3, 4, 5]) {
    it(`killed after ${k} finished node(s): the resume gives the clean result and no stage runs twice`, () => {
      const ws = workspace();
      const env = { ...ws.env, MLH_TEST_RUNDIR: ws.runDir };
      expect(child({ ...env, MLH_TEST_CRASH_AFTER: String(k) })).toBe(137);
      expect(child(env)).toBe(0);
      expect(claims(ws.runDir)).toEqual(CLEAN);
      expect(stageRuns(ws.env)).toBe(1);
    }, 240_000);
  }
  it("killed inside the env agent after one step: the agent resumes from its checkpoint", () => {
    const ws = workspace();
    const env = { ...ws.env, MLH_TEST_RUNDIR: ws.runDir };
    expect(child({ ...env, MLH_TEST_CRASH_AFTER_STEPS: "1" })).toBe(137);
    expect(child(env)).toBe(0);
    expect(claims(ws.runDir)).toEqual(CLEAN);
  }, 240_000);
});

describe("P2: torn files", () => {
  it("a torn last journal line and a torn budget file: resume recovers and finishes", async () => {
    const ws = workspace([{ when: STAGE, nth: [1, 2], error: CAPACITY }]);
    await run(ws); // stops (infra_repeat) before the stage succeeds
    writeFileSync(join(ws.runDir, "journal.jsonl"), readFileSync(join(ws.runDir, "journal.jsonl"), "utf8") + '{"t":"start","id":"repro.stage:tr');
    writeFileSync(join(ws.runDir, "budget.json"), "{torn");
    await run(ws);
    expect(claims(ws.runDir)).toEqual(CLEAN);
  }, 60_000);
});

describe("P1: strictly sequential", () => {
  it("the journal never has two nodes in flight at once, even when the caller asks for concurrency", async () => {
    const ws = workspace();
    await withEnv(ws.env, () => runHarness({ runDir: ws.runDir, mode: "repro", llm: scriptedTeam(canaryScript()), concurrency: 3, source: CANARY_REPO, objective: "Reproduce the canary accuracy.", arxiv: "2101.00001", gpuBudgetUsd: 1 }));
    let inFlight = 0, most = 0;
    for (const e of journal(ws.runDir)) {
      if (e.t === "start") most = Math.max(most, ++inFlight);
      if (e.t === "ok" || e.t === "fail" || e.t === "hold") inFlight--;
    }
    expect(most).toBe(1);
    expect(claims(ws.runDir)).toEqual(CLEAN);
  }, 60_000);
});

describe("P1: debugger and patch path", () => {
  it("a planned command that fails is diagnosed, the corrected command is authorized, re-run and reproduces", async () => {
    const ws = workspace();
    const s = canaryScript();
    const plan = structuredClone(s.plan[0]!.args) as { stages: { argv: string[] }[] };
    plan.stages[0]!.argv = ["python3", "train.py", "--stepz", "100"];
    s.plan = [{ name: "report", args: plan }];
    s.debug = [{ name: "report", args: { causes: [{ category: "run.wrong_command", evidence: "train.py: error: unrecognized arguments: --stepz 100" }], patch_intent: "", run_argv: ["python3", "train.py", "--steps", "100"], expected_effect: "the stage runs" } }];
    await run(ws, s);
    expect(claims(ws.runDir)).toEqual(CLEAN);
    const j = journal(ws.runDir);
    expect(j.some((e) => e.t === "ok" && String(e.id).startsWith("repro.patch") && e.output.accepted === true)).toBe(true);
    expect(roleOutcomes(j).debugger).toMatchObject({ ok: true });
    expect(costOfRun(ws.runDir).retried.paper.attempts + costOfRun(ws.runDir).failed.paper.attempts).toBeGreaterThan(0);
  }, 60_000);
});

describe("P1: data fetch path", () => {
  const URL = "https://huggingface.co/datasets/canary/resolve/main/labels.txt"; // an approved data host
  const BODY = "0 1 0 1 0 1 0 1 0 1";
  const withData = () => {
    const s = canaryScript();
    const plan = structuredClone(s.plan[0]!.args) as Record<string, unknown> & { stages: { argv: string[] }[] };
    plan.stages[0]!.argv = ["python3", "train.py", "--steps", "100", "--data", "/cache/data/canary/labels.txt"];
    plan.fetch = [{ url: URL, to: "/cache/data/canary/labels.txt" }];
    plan.checks = [{ path: "/cache/data/canary", minFiles: 1 }];
    s.plan = [{ name: "report", args: plan }];
    return s;
  };
  it("declared data is fetched once with provenance, the stage reads it, and the claim reproduces", async () => {
    const ws = workspace();
    ws.env.MLH_FAKE_DATA = JSON.stringify({ [URL]: BODY });
    await run(ws, withData());
    expect(claims(ws.runDir)).toEqual(CLEAN);
    const fetched = journal(ws.runDir).find((e) => e.t === "ok" && e.id === "repro.fetch");
    expect(fetched.output.items[0]).toMatchObject({ url: URL, bytes: BODY.length, sha256: expect.stringMatching(/^[0-9a-f]{64}$/) });
    expect(fetched.output.checks[0]).toMatchObject({ ok: true });
  }, 60_000);
  it("data from a host that is not approved is held for a human, never fetched", async () => {
    const ws = workspace();
    const s = withData();
    const plan = s.plan[0]!.args as { fetch: { url: string }[] };
    plan.fetch[0]!.url = "https://data.example.org/canary/labels.txt";
    ws.env.MLH_FAKE_DATA = JSON.stringify({ "https://data.example.org/canary/labels.txt": BODY });
    await run(ws, s);
    expect(journal(ws.runDir).find((e) => e.t === "hold" && e.id === "repro.fetch")?.reason).toMatch(/data\.example\.org/);
    expect(existsSync(join(ws.env.MLH_FAKE_CACHE!, "data"))).toBe(false);
  }, 60_000);
  it("a download that fails is a named failure, not a silent pass", async () => {
    const ws = workspace();
    ws.env.MLH_FAKE_DATA = "{}";
    await run(ws, withData());
    expect(claims(ws.runDir)[0]![1]).not.toBe("reproduced");
    expect(journal(ws.runDir).some((e) => e.t === "fail" && e.id === "repro.fetch")).toBe(true);
  }, 60_000);
});

describe("P1: run isolation", () => {
  it("each run uses its own /cache volume, never the shared one", async () => {
    const vols: string[] = [];
    for (let i = 0; i < 2; i++) {
      const ws = workspace();
      await run(ws);
      const calls = readFileSync(join(ws.env.MLH_FAKE_CACHE!, ".fake-calls.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l).volume as string);
      expect(new Set(calls).size).toBe(1);
      vols.push(calls[0]!);
    }
    expect(vols[0]).toMatch(/^mlh-run-[0-9a-f]{12}$/);
    expect(vols[1]).toMatch(/^mlh-run-[0-9a-f]{12}$/);
    expect(vols[0]).not.toBe(vols[1]);
  }, 60_000);
});
