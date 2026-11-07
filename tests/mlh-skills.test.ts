// Deterministic manifest and measurement behaviour on small synthetic repos.
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { runProc } from "../src/mlh/io/proc.ts";
import { PROBES } from "../src/mlh/io/sandbox.ts";
import { runHarness } from "../src/mlh/orchestrator.ts";

const tmp = () => mkdtempSync(join(tmpdir(), "mlh-sk-"));
const probe = async (...args: string[]) => JSON.parse((await runProc(["python3", "-I", join(PROBES, "probe.py"), ...args], { cwd: tmpdir(), env: { PATH: process.env.PATH! }, timeoutMs: 30_000 })).stdout);

describe("manifest inference", () => {
  it("prefers the README's documented command and leaves an ambiguous claim unset rather than guessing", async () => {
    const r = tmp();
    mkdirSync(join(r, "src"));
    writeFileSync(join(r, "src/fit.py"), "");
    writeFileSync(join(r, "train.py"), "");
    writeFileSync(join(r, "README.md"), "Run:\n\n    python src/fit.py --epochs 5 --seed 3\n\nThe baseline gets 0.50 accuracy; ours reaches an accuracy of 0.81.\n");
    const { manifest, provenance } = await probe("inspect", r);
    expect(manifest.train).toEqual(["python3", "src/fit.py", "--epochs", "5", "--seed", "3"]);
    expect(manifest.claim).toBeUndefined();
    expect(provenance.claim).toMatch(/ambiguous/);
  });

  it("never reads a seed or another integer as the claimed number", async () => {
    const r = tmp();
    writeFileSync(join(r, "train.py"), "");
    writeFileSync(join(r, "README.md"), "Test accuracy for `--seed 0` is 85.2%.\n");
    expect((await probe("inspect", r)).manifest.claim).toEqual({ metric: "accuracy", value: 0.852 });
  });
});

describe("data audit", () => {
  it("finds the same entity on both sides of a split even when no row repeats, and handles string columns", async () => {
    const r = tmp();
    const rows = (ids: number[], off: number) => ["customer_id,plan,x,label", ...ids.map((id, i) => `${id},${i % 2 ? "basic" : "pro"},${(i + off) * 0.37},${id % 2 ? "yes" : "no"}`)].join("\n");
    writeFileSync(join(r, "train.csv"), rows([1, 1, 2, 2, 3, 3, 4, 4], 0));
    writeFileSync(join(r, "test.csv"), rows([1, 2, 5, 6], 100));
    const a = await probe("audit", join(r, "train.csv"), join(r, "test.csv"), "label");
    expect(a.test_rows_in_train).toBe(0);
    expect(a.entities).toEqual([expect.objectContaining({ column: "customer_id", test_rows_with_train_entity: 2, share: 0.5 })]);
  });
});

describe("outputs resolved by observation", () => {
  it("uses the metrics file the run actually wrote when the declared one does not exist", async () => {
    const r = tmp();
    writeFileSync(join(r, "train.py"), 'import json, os\nos.makedirs("runs/a", exist_ok=True)\njson.dump({"test": {"acc": 0.8}, "val": {"acc": 0.7}}, open("runs/a/final.json", "w"))\n');
    writeFileSync(join(r, "mlh.json"), JSON.stringify({ outputs: { metrics: "metrics.json" }, metric: "accuracy" }));
    writeFileSync(join(r, "README.md"), "We reach 0.80 accuracy.\n");
    const res = await runHarness({ runDir: tmp(), source: r, objective: "Reproduce the reported accuracy" });
    const t = res.state.nodes.get("run.train")!;
    expect(t.status).toBe("succeeded");
    expect((t.output as { by: Record<string, string> }).by.metrics).toMatch(/observed: the run wrote runs\/a\/final.json/);
    expect(res.summary.reproduced).toBe("yes");
  }, 60_000);
});
