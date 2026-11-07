// The reproduction flow's deciding code: plan validation, the status rule,
// patch risk, claim validation against the paper text, log classification, and
// how the DAG grows from results.
import { describe, expect, it } from "vitest";
import { apply, replay, type RunEvent } from "../src/mlh/core/dag.ts";
import { authorize } from "../src/mlh/core/authz.ts";
import { afterFail, afterOk, finalStatus, initialNodes, N, validatePlan } from "../src/mlh/repro/flow.ts";
import { parseDiff, riskOf } from "../src/mlh/repro/patchrisk.ts";
import { classifyLog } from "../src/mlh/repro/report.ts";
import { checkSpecs } from "../src/mlh/repro/site.ts";
import { numberInText, parseMetric, statusOf } from "../src/mlh/repro/skills.ts";

const at = "t";
const claim = { id: "cora", metric: "accuracy", value: 0.815 };

describe("reproduction: deciding code", () => {
  it("validates the planner's output and enforces at least three seeds", () => {
    const v = validatePlan([claim, { id: "pub", metric: "accuracy", value: 0.79 }], { runs: [{ claim: "cora", argv: ["python3", "train.py"], parse: "accuracy= ([0-9.]+)", seeds: [7] }, { claim: "pub", argv: ["x"], parse: "no group" }] });
    expect(v.specs).toHaveLength(1);
    expect(v.specs[0]!.seeds).toEqual([7, 0, 1]);
    expect(v.errors.join()).toMatch(/pub: parse must be a regex with exactly one capture group/);
    expect(validatePlan([claim], { runs: [{ claim: "cora", argv: ["a"], parse: "(x)", env: { PATH: "/evil", DATASET: "cora" }, cwd: "../up" }] }).specs[0]).toMatchObject({ env: { DATASET: "cora" } });
    expect(validatePlan([claim], { runs: [{ claim: "cora", argv: ["a"], parse: "(x)", cwd: "../up" }] }).specs[0]!.cwd).toBeUndefined();
    // Inline programs would smuggle code changes past patch review.
    expect(validatePlan([claim], { runs: [{ claim: "cora", argv: ["python3", "-c", "import time; time.clock=time.perf_counter; import train"], parse: "(x)" }] }).errors.join()).toMatch(/inline programs/);
  });

  it("applies the fixed status rule: tolerance max(0.01, 2·paper std, 2·standard error)", () => {
    expect(statusOf(claim, [0.81, 0.82, 0.815]).status).toBe("reproduced");
    expect(statusOf(claim, [0.83, 0.832, 0.831]).status).toBe("not_reproduced");
    expect(statusOf({ ...claim, std: 0.01 }, [0.83, 0.832, 0.831]).status).toBe("reproduced");
  });

  it("parses the last metric match and normalizes percentages", () => {
    const spec = { claim: "cora", argv: [], seeds: [0], parse: "accuracy= ?([0-9.]+)", timeoutS: 60 };
    expect(parseMetric(spec, "accuracy", "accuracy= 0.70\n...\naccuracy= 0.8350")).toBe(0.835);
    expect(parseMetric(spec, "accuracy", "accuracy=81.5")).toBeCloseTo(0.815);
    expect(parseMetric({ ...spec, scale: 0.01 }, "mse", "accuracy=2.5")).toBeCloseTo(0.025);
  });

  it("keeps a claimed number only if it appears in the paper text", () => {
    const text = "Table 2: Cora 81.5 Citeseer 70.3 Pubmed 79.0 (GCN)";
    expect(numberInText(text, 0.815)).toBe(true);
    expect(numberInText(text, 0.79)).toBe(true);
    expect(numberInText(text, 0.825)).toBe(false);
    expect(numberInText("accuracy 0.8150", 0.815)).toBe(false); // "0.815" is not a separate number there
  });

  it("grades changes to measurement code, and writing the claimed number, as high risk", () => {
    const diff = "--- a/train.py\n+++ b/train.py\n@@\n-    adj = sp.coo_matrix(adj).A\n+    adj = sp.coo_matrix(adj).toarray()\n";
    expect(riskOf(parseDiff(diff, "/b", "/w"), [claim]).risk).toBe("medium");
    const evalDiff = "--- a/train.py\n+++ b/train.py\n@@\n-idx_test = range(500, 1500)\n+idx_test = range(0, 140)\n";
    expect(riskOf(parseDiff(evalDiff, "/b", "/w"), [claim]).risk).toBe("high");
    const fake = "--- a/train.py\n+++ b/train.py\n@@\n+print('accuracy= 0.815')\n";
    expect(riskOf(parseDiff(fake, "/b", "/w"), [claim]).reasons.join()).toMatch(/adds the claimed value/);
    expect(authorize("apply_eval_patch", [{ kind: "deterministic", check: "x", ok: true }, { kind: "llm", check: "y", ok: true }]).allow).toBe(false);
  });

  it("accepts only plain package specifiers for installation", () => {
    expect(checkSpecs(["scipy", "networkx<3", "torch==2.5.1"]).ok).toBe(true);
    for (const bad of ["git+https://x/y.git", "../local", "-e .", "pkg @ https://x", "--index-url=http://evil"]) expect(checkSpecs([bad]).ok).toBe(false);
  });

  it("classifies run failures from their logs", () => {
    expect(classifyLog("ModuleNotFoundError: No module named 'sklearn'")).toBe("env.missing_dependency");
    expect(classifyLog("ModuleNotFoundError: No module named 'pygcn'", new Set(["pygcn"]))).toBe("env.path_or_config");
    expect(classifyLog("AttributeError: module 'numpy' has no attribute 'float'")).toBe("env.version_drift");
    expect(classifyLog("python3 x.py → timeout: ")).toBe("run.resource_limit");
  });

  it("grows the DAG from results: claims+env → plan → runs per seed → compare → debug on mismatch → patch", () => {
    const ev: RunEvent[] = [{ t: "run", at, objective: "o", source: "s", mode: "repro" }, ...initialNodes().map((node) => ({ t: "node" as const, at, node, why: "" }))];
    const s = replay(ev);
    const ok = (id: string, output: unknown) => { apply(s, { t: "ok", at, id, output, findings: [], artifacts: [] }); for (const node of afterOk(s, id)) apply(s, { t: "node", at, node, why: "" }); };
    ok(N.paper, {}); expect(s.nodes.has(N.claims)).toBe(true);
    ok(N.env, {}); expect(s.nodes.has(N.plan)).toBe(false);
    ok(N.claims, { claims: [claim] }); expect(s.nodes.has(N.plan)).toBe(true);
    ok(N.plan, { specs: [{ claim: "cora", argv: ["x"], seeds: [0, 1, 2], parse: "(x)", timeoutS: 60 }] });
    const runs = [...s.nodes.keys()].filter((k) => k.startsWith(N.run));
    expect(runs).toHaveLength(3);
    for (const r of runs) ok(r, { value: 0.83 });
    ok(`${N.compare}:cora:v0`, { status: "not_reproduced", mean: 0.83, values: [0.83, 0.83, 0.83] });
    expect(s.nodes.has(`${N.debug}:cora:v0`)).toBe(true);
    ok(`${N.debug}:cora:v0`, { hasPatch: true, claim: "cora" });
    expect(s.nodes.has(`${N.patch}:cora:v0`)).toBe(true);
    ok(`${N.patch}:cora:v0`, { accepted: true, version: 1 });
    expect([...s.nodes.keys()].filter((k) => k.includes(":v1:"))).toHaveLength(3);
    // Until the new version's runs complete, the newest completed comparison stands.
    expect(finalStatus(s).cora).toMatchObject({ status: "not_reproduced", version: 0 });
  });

  it("debugs a run that keeps failing, at most twice per claim, then reports the claim not runnable", () => {
    const s = replay([{ t: "run", at, objective: "o", source: "s", mode: "repro" }, ...initialNodes().map((node) => ({ t: "node" as const, at, node, why: "" }))]);
    const ok = (id: string, output: unknown) => { apply(s, { t: "ok", at, id, output, findings: [], artifacts: [] }); for (const node of afterOk(s, id)) apply(s, { t: "node", at, node, why: "" }); };
    ok(N.paper, {}); ok(N.env, {}); ok(N.claims, { claims: [claim] }); ok(N.plan, { specs: [{ claim: "cora", argv: ["x"], seeds: [0, 1, 2], parse: "(x)", timeoutS: 60 }] });
    const fail = (id: string) => { apply(s, { t: "start", at, id, attempt: 3 }); apply(s, { t: "fail", at, id, category: "experiment_error", message: "Traceback" }); for (const node of afterFail(s, id)) apply(s, { t: "node", at, node, why: "" }); };
    for (const r of [...s.nodes.keys()].filter((k) => k.startsWith(N.run))) fail(r);
    expect(s.nodes.has(`${N.debug}:cora:v0`)).toBe(true);
    ok(`${N.debug}:cora:v0`, { hasPatch: false, claim: "cora" });
    apply(s, { t: "skip", at, id: `${N.compare}:cora:v0`, reason: "dependency failed" });
    expect(finalStatus(s).cora!.status).toBe("not_runnable");
  });
});
