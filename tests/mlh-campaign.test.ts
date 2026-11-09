// P3: the campaign driver end to end, offline: doctor gate, pinned clone, child run, watchdog kill
// of a stalled child and resume, result row, derived trust, automatic retro, done marker.
import { execFileSync } from "node:child_process";
import { cpSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { CANARY_REPO, FAKE_RUNNER } from "./helpers/team.ts";

function gitRepo(root: string) {
  const dir = join(root, "canary-src");
  cpSync(CANARY_REPO, dir, { recursive: true });
  const git = (...a: string[]) => execFileSync("git", ["-c", "user.email=t@t", "-c", "user.name=t", ...a], { cwd: dir, encoding: "utf8" }).trim();
  git("init", "-q"); git("add", "."); git("commit", "-qm", "canary");
  return { dir, commit: git("rev-parse", "HEAD") };
}

function campaign(stallOnce: boolean) {
  const root = mkdtempSync(join(tmpdir(), "mlh-campaign-"));
  const { dir, commit } = gitRepo(root);
  const cfg = join(root, "papers.json"), out = join(root, "results");
  writeFileSync(cfg, JSON.stringify([{ id: "canary", arxiv: "2101.00001", repo: dir, commit, objective: "Reproduce the canary accuracy.", budgetUsd: 1 }]));
  const env: Record<string, string> = {
    ...(process.env as Record<string, string>), MLH_ROOT: root, MLH_CAMPAIGN_CHILD: `npx tsx ${resolve("tests/helpers/campaign-child.mts")}`,
    MLH_MODAL_RUNNER: FAKE_RUNNER, MLH_FAKE_CACHE: join(root, "cache"), MLH_FAKE_ELAPSED: "60", MLH_ENV_CACHE: join(root, "env-cache"), MLH_BACKOFF_MS: "0",
    MLH_STALL_MS: "4000", MLH_WATCHDOG_MS: "500", ...(stallOnce ? { MLH_TEST_STALL_ONCE: join(root, "stalled") } : {}),
  };
  delete env.DEEPSEEK_API_KEY; // the scripted team needs no provider; the balance check is skipped as unknown
  execFileSync("npx", ["tsx", resolve("src/mlh/cli.ts"), "campaign", cfg, "--label", "t1", "--out", out], { env, stdio: "pipe", timeout: 240_000 });
  const rows = readFileSync(join(out, "t1.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l));
  return { root, out, rows };
}

describe("P3: campaign driver", () => {
  it("doctor gate, run, result row, trust, retro and done marker", () => {
    const { root, out, rows } = campaign(false);
    expect(readFileSync(join(out, "t1.preflight.log"), "utf8")).toMatch(/doctor PASS/);
    expect(rows).toHaveLength(1);
    expect(rows[0].claims).toMatchObject([{ id: "canary_acc", status: "reproduced", mean: 0.9 }]);
    expect(rows[0].resumes).toBe(0);
    const trust = JSON.parse(readFileSync(join(out, "trust.json"), "utf8"));
    expect([trust.claims.n, trust.env.n, trust.planner.n]).toEqual([1, 1, 1]);
    expect(readFileSync(join(out, "decisions.md"), "utf8")).toMatch(/## t1 \(auto retro.*\n- Pipeline success 1\/1/);
    expect(existsSync(join(root, ".mlh", "campaign", "t1", "done"))).toBe(true);
  }, 300_000);

  it("a child that hangs before writing anything is killed by the watchdog and resumed to the clean result", () => {
    const { root, rows } = campaign(true);
    expect(existsSync(join(root, "stalled"))).toBe(true);
    expect(readFileSync(join(root, ".mlh", "campaign", "t1", "canary.log"), "utf8")).toMatch(/\[watchdog\] no progress/);
    expect(rows[0].resumes).toBe(1);
    expect(rows[0].claims).toMatchObject([{ status: "reproduced" }]);
  }, 300_000);
});
