// P3: the round chain (scripts/mlh-chain.sh) against fake campaign processes. Strictly sequential:
// the next round starts only after the previous one finished cleanly.
import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

const CHAIN = resolve("scripts/mlh-chain.sh");
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function setup() {
  const root = mkdtempSync(join(tmpdir(), "mlh-chain-"));
  const label = `t${Math.random().toString(36).slice(2, 8)}`;
  const cli = join(root, "fake-cli.sh");
  // Records each call with a timestamp; "campaign" returns at once (the next round's run is not under test).
  writeFileSync(cli, `#!/usr/bin/env bash\necho "$(date +%s%N) $*" >> ${join(root, "calls.log")}\n`);
  chmodSync(cli, 0o755);
  writeFileSync(join(root, "env"), "");
  mkdirSync(join(root, ".mlh", "campaign", label), { recursive: true });
  const calls = () => (existsSync(join(root, "calls.log")) ? readFileSync(join(root, "calls.log"), "utf8").trim().split("\n") : []);
  const marker = (m: "done" | "stopped") => writeFileSync(join(root, ".mlh", "campaign", label, m), "t");
  // A process whose command line looks like the campaign of `label` to pgrep.
  const fakeRound = (seconds: number): ChildProcess => spawn("bash", ["-c", `exec -a "campaign cfg --label ${label} --parallel 1" sleep ${seconds}`], { stdio: "ignore" });
  const chain = (maxWait = 30) => {
    const p = spawn("bash", [CHAIN, join(root, "env"), label, `${label}-next`], { env: { ...process.env, MLH_CHAIN_ROOT: root, MLH_CHAIN_CLI: cli, MLH_CHAIN_POLL_S: "1", MLH_CHAIN_MAX_WAIT_S: String(maxWait) }, stdio: "ignore" });
    const exit = new Promise<number>((r) => p.on("exit", (c) => r(c ?? -1)));
    return { p, exit };
  };
  return { root, label, calls, marker, fakeRound, chain };
}

describe("P3: round chain", () => {
  it("a round that never starts: the chain gives up after its max wait and starts nothing", async () => {
    const t = setup();
    expect(await t.chain(3).exit).toBe(2);
    expect(t.calls()).toEqual([]);
  }, 20_000);

  it("waits for a round that has not started yet, then starts the next only after it finished", async () => {
    const t = setup();
    const c = t.chain();
    await sleep(2500);
    expect(t.calls()).toEqual([]); // not started yet: nothing happens
    const round = t.fakeRound(3);
    await sleep(1500);
    expect(t.calls()).toEqual([]); // running: still nothing
    await new Promise((r) => round.on("exit", r));
    const endedAt = BigInt(Date.now()) * 1_000_000n;
    t.marker("done");
    expect(await c.exit).toBe(0);
    const calls = t.calls();
    expect(calls.map((l) => l.split(" ").slice(1, 3).join(" "))).toEqual([`campaign-retro ${t.label}`, "campaign bench/complex/campaign.json"]);
    expect(BigInt(calls[1]!.split(" ")[0]!)).toBeGreaterThan(endedAt - 2_000_000_000n); // after the round ended
  }, 30_000);

  it("a done marker with no process: starts the next round", async () => {
    const t = setup();
    t.marker("done");
    expect(await t.chain().exit).toBe(0);
    expect(t.calls().length).toBe(2);
  }, 20_000);

  it("a stopped marker ends the chain without starting anything", async () => {
    const t = setup();
    t.marker("stopped");
    expect(await t.chain().exit).toBe(1);
    expect(t.calls()).toEqual([]);
  }, 20_000);

  it("a round seen running that disappears without a done marker (crash, kill) does not release the chain", async () => {
    const t = setup();
    const c = t.chain();
    t.fakeRound(2);
    expect(await c.exit).toBe(3);
    expect(t.calls()).toEqual([]);
  }, 20_000);

  it("a stale done marker while the round runs again: waits until it is gone (never two rounds at once)", async () => {
    const t = setup();
    t.marker("done");
    const round = t.fakeRound(3);
    await sleep(300);
    const c = t.chain();
    await sleep(1500);
    expect(t.calls()).toEqual([]);
    await new Promise((r) => round.on("exit", r));
    expect(await c.exit).toBe(0);
    expect(spawnSync("pgrep", ["-f", "--", `--label ${t.label} `]).status).toBe(1); // nothing of the round left
    expect(t.calls().length).toBe(2);
  }, 30_000);
});

describe("strictly sequential", async () => {
  const { sequentialWorkers } = await import("../src/mlh/repro/campaign.ts");
  it("a campaign runs one paper at a time whatever --parallel or the trust gate say", () => {
    expect(sequentialWorkers(3, 3, 6)).toBe(1);
    expect(sequentialWorkers(2, 1, 3)).toBe(1);
    expect(sequentialWorkers(2, 3, 0)).toBe(0);
  });
});
