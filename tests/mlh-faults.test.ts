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
  it("a transport fault once: retried, same result, cost shown as retried infrastructure, nobody blamed", async () => {
    const ws = workspace([{ when: STAGE, nth: [1], error: RESET }]); // (capacity shortages now fall back to another GPU in the same attempt)
    await run(ws);
    expect(claims(ws.runDir)).toEqual(CLEAN);
    expect(stopped(ws.runDir)).toBeUndefined();
    expect(costOfRun(ws.runDir).retried.infrastructure.attempts).toBe(1);
    expect(blamed(ws.runDir)).toEqual([]);
  }, 60_000);

  it("the same fault twice stops the run (infra_repeat); a resume after investigation finishes with the clean result", async () => {
    const ws = workspace([{ when: STAGE, nth: [1, 2], error: RESET }]);
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

describe("P1: ambiguous metric in a real run", () => {
  const ambiguous = (pick?: { index: number | null; why: string }) => {
    const s = canaryScript();
    const plan = structuredClone(s.plan[0]!.args) as { stages: { argv: string[] }[] };
    plan.stages[0]!.argv = ["python3", "train.py", "--steps", "100", "--also-train-acc"]; // "accuracy: " now matches two lines
    s.plan = [{ name: "report", args: plan }];
    if (pick) s.pick = pick;
    return s;
  };
  it("the line chosen by label gives the verdict, and the choice is journaled", async () => {
    const ws = workspace();
    await run(ws, ambiguous({ index: 1, why: "the test accuracy line" }));
    expect(claims(ws.runDir)).toEqual(CLEAN);
    const st = journal(ws.runDir).find((e) => e.t === "ok" && String(e.id).startsWith("repro.stage"));
    expect(st.output.disambiguated.canary_acc).toMatchObject({ index: 1, options: ["train accuracy: <value>", "accuracy: <value>"] });
  }, 60_000);
  it("no line can be chosen: no verdict and the run stops by name (never the last match)", async () => {
    const ws = workspace();
    await run(ws, ambiguous());
    expect(claims(ws.runDir)[0]![1]).not.toMatch(/reproduced/);
    expect(journal(ws.runDir).some((e) => e.t === "fail" && /ambiguous metric/.test(e.message))).toBe(true);
  }, 60_000);
});

describe("P1: a one-stage command change re-runs only that stage", () => {
  it("independent stages are not run again; the failed stage and what it needs are", async () => {
    const ws = workspace([{ when: "--steps 100$", nth: [1], exit: 1, stderr: "Killed" }]);
    const s = canaryScript();
    const plan = structuredClone(s.plan[0]!.args) as { stages: Record<string, unknown>[] };
    plan.stages = [
      { id: "prep", argv: ["python3", "train.py", "--steps", "1"], cwd: "/work", timeoutS: 300 },
      { id: "other", argv: ["python3", "train.py", "--steps", "100", "--seed", "1"], cwd: "/work", timeoutS: 300 },
      { id: "train", argv: ["python3", "train.py", "--steps", "100"], cwd: "/work", timeoutS: 300, gpu: "T4", needs: ["prep"] },
    ];
    s.plan = [{ name: "report", args: plan }];
    const pc = (s.plan[0]!.args as { claims: { claim: string; stage: string; parse: string }[] }).claims;
    pc[0]!.stage = "train";
    pc.push({ claim: "canary_acc_b", stage: "other", parse: "accuracy: ([0-9.]+)" }); // every stage serves a claim
    (s.claims as { claims: unknown[] }).claims.push({ id: "canary_acc_b", metric: "accuracy", value: 0.9, dataset: "synthetic" });
    s.debug = [{ name: "report", args: { causes: [{ category: "run.resource_limit", evidence: "transient: the planned command succeeds when re-run" }], patch_intent: "re-run unchanged (transient kill)", run_argv: ["python3", "train.py", "--steps", "100"], expected_effect: "the stage succeeds" } }];
    await run(ws, s);
    const calls = readFileSync(join(ws.env.MLH_FAKE_CACHE!, ".fake-calls.jsonl"), "utf8").trim().split("\n").map((l) => (JSON.parse(l).cmd as string[]).join(" "));
    const prepRuns = calls.filter((c) => c === "python3 train.py --steps 1").length, otherRuns = calls.filter((c) => c === "python3 train.py --steps 100 --seed 1").length;
    expect(claims(ws.runDir)).toEqual([...CLEAN, ["canary_acc_b", "reproduced", 0.9]]);
    expect(journal(ws.runDir).some((e) => e.t === "ok" && String(e.id).startsWith("repro.patch") && e.output.accepted)).toBe(true);
    expect(otherRuns).toBe(1); // an independent stage is not re-run
    expect(prepRuns).toBe(2); // what the failed stage needs is re-run with it (its outputs are tagged per run)
  }, 90_000);
});

describe("P1: environment data is declared, downloaded by the harness, and travels with the cache", () => {
  const URL = "https://huggingface.co/datasets/canary/resolve/main/labels.txt";
  const BODY = "0 1 0 1 0 1 0 1 0 1";
  const withEnvData = () => {
    const s = canaryScript();
    s.env = [
      { name: "fetch_data", args: { items: [{ url: URL, to: "/cache/data/canary/labels.txt" }] } },
      { name: "report", args: { runnable: true, launch: { cwd: ".", example_command: ["python3", "train.py", "--steps", "1", "--data", "/cache/data/canary/labels.txt"] }, problems: [] } },
    ];
    return s;
  };
  it("a cache hit in a fresh run re-downloads the declared data and the smoke test passes without the env agent", async () => {
    const first = workspace();
    first.env.MLH_FAKE_DATA = JSON.stringify({ [URL]: BODY });
    const log1: string[] = [];
    await withEnv(first.env, () => runHarness({ runDir: first.runDir, mode: "repro", llm: scriptedTeam(withEnvData(), log1), concurrency: 1, source: CANARY_REPO, objective: "Reproduce the canary accuracy.", arxiv: "2101.00001", gpuBudgetUsd: 1 }));
    const env1 = journal(first.runDir).find((e) => e.t === "ok" && e.id === "repro.env");
    expect(env1.output.data).toEqual([{ url: URL, to: "/cache/data/canary/labels.txt" }]);
    expect(env1.output.smoke.ok).toBe(true);
    // A new run: its own empty volume, the same env cache.
    const second = workspace();
    const env = { ...second.env, MLH_ENV_CACHE: first.env.MLH_ENV_CACHE!, MLH_FAKE_DATA: JSON.stringify({ [URL]: BODY }) };
    const log2: string[] = [];
    await withEnv(env, () => runHarness({ runDir: second.runDir, mode: "repro", llm: scriptedTeam(withEnvData(), log2), concurrency: 1, source: CANARY_REPO, objective: "Reproduce the canary accuracy.", arxiv: "2101.00001", gpuBudgetUsd: 1 }));
    const env2 = journal(second.runDir).find((e) => e.t === "ok" && e.id === "repro.env");
    expect(env2.output.cached).toBe(true);
    expect(log2.filter((l) => l.startsWith("env:"))).toEqual([]); // the env agent did not run
    const calls = readFileSync(join(second.env.MLH_FAKE_CACHE!, ".fake-calls.jsonl"), "utf8").trim().split("\n").map((l) => (JSON.parse(l).cmd as string[]).join(" "));
    expect(calls.some((c) => c.includes("fetch_data.py") && c.includes(URL))).toBe(true); // re-downloaded into the new volume
  }, 120_000);
  it("an unknown host is refused by the tool, nothing is downloaded", async () => {
    const ws = workspace();
    const s = withEnvData();
    s.env[0] = { name: "fetch_data", args: { items: [{ url: "https://data.example.org/x.txt", to: "/cache/data/x.txt" }] } };
    s.env[1] = { name: "report", args: { runnable: true, launch: { cwd: ".", example_command: ["python3", "train.py", "--steps", "1"] }, problems: [] } };
    await run(ws, s);
    const st = JSON.parse(readFileSync(join(ws.runDir, "work", "repro.env", "state.json"), "utf8"));
    expect(JSON.stringify(st.messages)).toMatch(/REFUSED: data\.example\.org is not a known data host/);
    expect(journal(ws.runDir).find((e) => e.t === "ok" && e.id === "repro.env").output.data).toEqual([]);
  }, 60_000);
});

describe("P1: {seed} in a per-seed stage's command", () => {
  it("each instance gets its own seed in place of {seed} (FT-Transformer r2c ran tuned/{seed}.toml literally)", async () => {
    const ws = workspace();
    const s = canaryScript();
    const plan = structuredClone(s.plan[0]!.args) as { seeds: number[]; stages: Record<string, unknown>[] };
    plan.seeds = [0, 1];
    plan.stages = [{ id: "train", argv: ["python3", "train.py", "--steps", "100", "--seed", "{seed}"], cwd: "/work", timeoutS: 300, gpu: "T4", perSeed: true }];
    s.plan = [{ name: "report", args: plan }];
    await run(ws, s);
    expect(claims(ws.runDir)).toEqual(CLEAN);
    const calls = readFileSync(join(ws.env.MLH_FAKE_CACHE!, ".fake-calls.jsonl"), "utf8").trim().split("\n").map((l) => (JSON.parse(l).cmd as string[]).join(" "));
    expect(calls.filter((c) => /train\.py --steps 100 --seed [01]$/.test(c)).sort()).toEqual(["python3 train.py --steps 100 --seed 0", "python3 train.py --steps 100 --seed 1"]);
    expect(calls.some((c) => c.includes("{seed}"))).toBe(false);
  }, 60_000);
});

describe("P1: declared system layers, host memory and data links reach the stage", () => {
  const URL = "https://huggingface.co/datasets/canary/resolve/main/labels.txt";
  it("system from the env agent's install, memGB and links from the plan are in every stage call; the link resolves", async () => {
    const ws = workspace();
    ws.env.MLH_FAKE_DATA = JSON.stringify({ [URL]: "0 1 0 1 0 1 0 1 0 1" });
    const s = canaryScript();
    s.env = [{ name: "install", args: { system: ["apt-get update && apt-get install -y g++"] } }, ...s.env];
    const plan = structuredClone(s.plan[0]!.args) as Record<string, unknown> & { stages: Record<string, unknown>[] };
    plan.fetch = [{ url: URL, to: "/cache/data/canary/labels.txt" }];
    plan.stages = [{ id: "train", argv: ["python3", "train.py", "--steps", "100", "--data", "data/labels.txt"], cwd: "/work", timeoutS: 300, gpu: "T4", memGB: 32, links: [{ path: "data/labels.txt", to: "/cache/data/canary/labels.txt" }] }];
    s.plan = [{ name: "report", args: plan }];
    await run(ws, s);
    expect(claims(ws.runDir)).toEqual(CLEAN);
    const calls = readFileSync(join(ws.env.MLH_FAKE_CACHE!, ".fake-calls.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l));
    const stage = calls.find((c) => (c.cmd as string[]).join(" ").endsWith("--data data/labels.txt"));
    expect(stage).toMatchObject({ system: ["apt-get update && apt-get install -y g++"], memory_mb: 32768, links: [{ path: "data/labels.txt", to: "/cache/data/canary/labels.txt" }] });
    const smoke = calls.find((c) => (c.cmd as string[]).join(" ").includes("MLH_SMOKE_EXIT"));
    expect(smoke.system).toEqual(["apt-get update && apt-get install -y g++"]);
    expect(journal(ws.runDir).find((e) => e.t === "ok" && e.id === "repro.env").output.system).toEqual(["apt-get update && apt-get install -y g++"]);
  }, 60_000);
  it("the planner's check_plan has a budget of 8", async () => {
    const ws = workspace();
    const s = canaryScript();
    const real = s.plan[0]!;
    s.plan = [...Array.from({ length: 9 }, () => ({ name: "check_plan", args: real.args })), real];
    await run(ws, s);
    const st = JSON.parse(readFileSync(join(ws.runDir, "work", "repro.plan", "state.json"), "utf8"));
    const answers = st.messages.filter((m: { role: string; content?: string }) => m.role === "tool").map((m: { content: string }) => m.content);
    expect(answers.filter((a: string) => a.startsWith("OK:"))).toHaveLength(8);
    expect(answers.some((a: string) => /8-check budget is used up/.test(a))).toBe(true);
    expect(claims(ws.runDir)).toEqual(CLEAN);
  }, 60_000);
});

describe("P2: a GPU type without capacity", () => {
  it("the stage moves to the next cheapest GPU with enough memory in the same attempt, and records it", async () => {
    const ws = workspace([{ when: STAGE, gpu: "L4", nth: [1, 2, 3], error: "RuntimeError: ResourceExhaustedError: Insufficient capacity to create sandbox." }]);
    const s = canaryScript();
    (s.plan[0]!.args as { stages: Record<string, unknown>[] }).stages[0]!.gpu = "L4";
    await run(ws, s);
    expect(claims(ws.runDir)).toEqual(CLEAN);
    const st = journal(ws.runDir).find((e) => e.t === "ok" && String(e.id).startsWith("repro.stage"));
    expect(st.output).toMatchObject({ gpu: "A10G", plannedGpu: "L4" });
    expect(journal(ws.runDir).filter((e) => e.t === "fail")).toEqual([]); // no failed attempt, no stop
  }, 60_000);
});
