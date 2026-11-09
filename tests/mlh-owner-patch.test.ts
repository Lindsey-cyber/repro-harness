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

  it("amends seeds, the claims' select/aggregate, the stages' GPU and resets earlier command changes (LoRA r3)", async () => {
    const { gpuPlanOf } = await import("../src/mlh/repro/gpu.ts");
    const run = mkdtempSync(join(tmpdir(), "mlh-owner-"));
    mkdirSync(join(run, "source"), { recursive: true }); writeFileSync(join(run, "source", "run_glue.py"), "x\n");
    const plan = { gpuPlan: { fidelity: "full", fidelityNote: "", fetch: [], checks: [], unknownHosts: [], stages: [{ id: "rte", argv: ["python", "run_glue.py", "--task_name", "rte"], timeoutS: 600, needs: [], persist: [], gpu: "L4", perSeed: true }], claims: [{ claim: "c", stage: "rte", parse: "acc ([0-9.]+)" }], unmeasured: [], seeds: [0, 1, 2], worstCaseUsd: 1 } };
    const at = "2026-11-09T00:00:00Z";
    const ev0 = [{ t: "run", at, objective: "o", source: "s", mode: "repro" }, { t: "node", at, node: { id: "repro.plan", skill: "repro.plan", deps: [], purpose: "setup", params: {} }, why: "x" }, { t: "start", at, id: "repro.plan", attempt: 1 }, { t: "ok", at, id: "repro.plan", output: plan, findings: [], artifacts: [] },
      { t: "node", at, node: { id: "repro.patch:stage-rte:operator1", skill: "repro.patch", deps: [], purpose: "setup", params: {} }, why: "x" }, { t: "start", at, id: "repro.patch:stage-rte:operator1", attempt: 1 }, { t: "ok", at, id: "repro.patch:stage-rte:operator1", output: { accepted: true, argv: { "stage-rte": ["python", "run_glue.py", "--task_name", "rte", "--gradient_accumulation_steps", "8"] } }, findings: [], artifacts: [] }];
    writeFileSync(join(run, "journal.jsonl"), ev0.map((e) => JSON.stringify(e)).join("\n") + "\n");
    const local = join(run, "fix.py"); writeFileSync(local, "fixed\n");
    execFileSync("npx", ["tsx", "src/mlh/cli.ts", "operator-patch", run, "--file", `run_glue.py=${local}`, "--seeds", "0,1,2,3,4", "--select", "max", "--aggregate", "median", "--gpu", "A100-40GB", "--reset-argv", "owner approved: loralib merge fix and the paper's reporting rule"], { encoding: "utf8" });
    const st = replay(readFileSync(join(run, "journal.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l)), 3);
    const gp = gpuPlanOf(st)!;
    expect(gp.seeds).toEqual([0, 1, 2, 3, 4]);
    expect(gp.claims[0]).toMatchObject({ select: "max", aggregate: "median" });
    expect(gp.stages[0]!.gpu).toBe("A100-40GB");
    expect(currentVersion(st)).toMatchObject({ version: 1, argv: { "stage-rte": ["python", "run_glue.py", "--task_name", "rte"] } });
    // every stage seed and the comparison are queued again on the new version
    expect(st.order.filter((id) => id.startsWith("repro.stage:rte:")).length).toBe(5);
  }, 60_000);
});
