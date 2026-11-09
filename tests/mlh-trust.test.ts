import { describe, expect, it } from "vitest";
import { pipelineMetric } from "../src/mlh/repro/campaign.ts";
import { emptyStore, record, roleOutcomes, scaleGate, STANDARD, wilsonLower, type Outcome, type Role } from "../src/mlh/repro/trustscore.ts";

const at = "2026-01-01T00:00:00Z";
const ok = (id: string, output: Record<string, unknown> = {}) => ({ t: "ok", at, id, output, findings: [], artifacts: [] });
const fail = (id: string, category: string) => ({ t: "fail", at, id, category, message: "x" });
const all = (o: Outcome): Record<Role, Outcome> => ({ claims: o, env: o, planner: o, debugger: o });

describe("trust scores", () => {
  it("wilson lower bound is conservative on small samples", () => {
    expect(wilsonLower(0, 0)).toBe(0);
    expect(wilsonLower(3, 3)).toBeLessThan(STANDARD.minLower); // 3/3 is not enough evidence
    expect(wilsonLower(5, 5)).toBeLessThan(STANDARD.minLower); // in practice six clean runs are needed
    expect(wilsonLower(6, 6)).toBeGreaterThanOrEqual(STANDARD.minLower);
    expect(wilsonLower(8, 10)).toBeLessThan(STANDARD.minLower);
  });

  it("reads each role's outcome from harness facts", () => {
    const ev = [
      ok("repro.claims", { rejected: [] }),
      ok("repro.env", { smoke: { ok: true }, repaired: true }),
      fail("repro.plan", "contract_violation"), ok("repro.plan"),
      fail("repro.stage:train:v1", "nonzero_exit"), ok("repro.debug:c1"),
      ok("repro.patch:c1", { accepted: true, version: 2 }), ok("repro.stage:train:v2"),
    ];
    const o = roleOutcomes(ev as never);
    expect(o.claims?.ok).toBe(true);
    expect(o.env).toMatchObject({ ok: false, why: expect.stringContaining("repair") });
    expect(o.planner?.ok).toBe(true);
    expect(o.debugger?.ok).toBe(true); // source patch -> new version that succeeds
  });

  it("does not credit or blame roles without new evidence", () => {
    const o = roleOutcomes([ok("repro.env", { cached: true, smoke: { ok: true } }), fail("repro.plan", "provider_transient"), fail("repro.claims", "provider_fatal")] as never);
    expect(o).toEqual({ claims: null, env: null, planner: null, debugger: null });
  });

  it("infrastructure failures are not blamed on the agent", () => {
    expect(roleOutcomes([fail("repro.env", "sandbox_transient")] as never).env).toBeNull();
    expect(roleOutcomes([fail("repro.env", "contract_violation")] as never).env?.ok).toBe(false);
  });

  it("a debugger whose change fixes nothing fails", () => {
    const o = roleOutcomes([ok("repro.debug:c1"), ok("repro.stage:train:v1"), ok("repro.patch:c1", { accepted: true })] as never);
    expect(o.debugger?.ok).toBe(false);
  });

  it("promotes only at the standard, demotes on one failure, and gates scale", () => {
    let s = emptyStore();
    expect(scaleGate(s)).toMatchObject({ maxPapers: 3, parallel: 1 });
    for (let i = 0; i < 5; i++) s = record(s, `p${i}`, at, all({ ok: true, why: "" }));
    expect(s.env.level).toBe("probation");
    s = record(s, "p5", at, all({ ok: true, why: "" }));
    expect(s.env.level).toBe("trusted");
    expect(scaleGate(s).blockedBy).toEqual([]);
    s = record(s, "p6", at, { ...all(null), env: { ok: false, why: "smoke" } });
    expect(s.env.level).toBe("probation");
    expect(s.planner.level).toBe("trusted");
    expect(scaleGate(s)).toMatchObject({ maxPapers: 3, blockedBy: ["env"] });
  });

  it("pipeline metric counts only fully definitive, unstopped runs", () => {
    const m = pipelineMetric([
      { claims: [{ status: "reproduced" }, { status: "not_reproduced" }], wallMin: 30 },
      { claims: [{ status: "reproduced" }, { status: "not_runnable" }], wallMin: 60 },
      { claims: [], wallMin: 10 },
      { claims: [{ status: "reproduced" }], wallMin: 40, stopped: "provider_fatal" },
    ]);
    expect(m).toEqual({ success: 1, n: 4, medianMin: 30 });
  });
});
