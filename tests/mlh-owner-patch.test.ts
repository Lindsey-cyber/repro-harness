import { describe, expect, it } from "vitest";
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { replay } from "../src/mlh/core/dag.ts";
import { currentVersion } from "../src/mlh/repro/flow.ts";

describe("owner's change of repository files (SimSiam r3b: dataloader workers, kNN monitor)", () => {
  it("builds a new source version with only the named files changed, journaled with the human basis", () => {
    const run = mkdtempSync(join(tmpdir(), "mlh-owner-"));
    mkdirSync(join(run, "source", "configs"), { recursive: true });
    writeFileSync(join(run, "source", "configs", "a.yaml"), "num_workers: 4\n");
    writeFileSync(join(run, "source", "main.py"), "print(1)\n");
    writeFileSync(join(run, "journal.jsonl"), JSON.stringify({ t: "run", at: "2026-11-09T00:00:00Z", objective: "o", source: "s", mode: "repro" }) + "\n");
    const local = join(run, "new.yaml"); writeFileSync(local, "num_workers: 16\n");
    execFileSync("npx", ["tsx", "src/mlh/cli.ts", "operator-patch", run, "--file", `configs/a.yaml=${local}`, "owner approved: more dataloader workers"], { encoding: "utf8" });
    expect(readFileSync(join(run, "versions", "v1", "configs", "a.yaml"), "utf8")).toBe("num_workers: 16\n");
    expect(readFileSync(join(run, "versions", "v1", "main.py"), "utf8")).toBe("print(1)\n");
    expect(readFileSync(join(run, "source", "configs", "a.yaml"), "utf8")).toBe("num_workers: 4\n"); // the pinned source is untouched
    const ev = readFileSync(join(run, "journal.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l));
    expect(currentVersion(replay(ev, 3)).version).toBe(1);
    expect(ev.at(-1).output).toMatchObject({ accepted: true, risk: "high", basis: "human approval: owner approved: more dataloader workers", files: ["configs/a.yaml"] });
    expect(() => execFileSync("npx", ["tsx", "src/mlh/cli.ts", "operator-patch", run, "--file", `../x=${local}`, "owner approved: escape attempt"], { stdio: "pipe" })).toThrow();
  }, 60_000);
});
