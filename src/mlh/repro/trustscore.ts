// Trust per agent role, earned from audited outcomes, gating how far the system
// scales. Pure. Each role's outcome in a run is decided by harness facts, not by
// what the agent said. A role is promoted only when the lower bound of its
// success rate clears the standard over enough runs; one failure while trusted
// sends it back to probation. While any role is on probation, campaigns stay
// small: the system does not scale what it has not proven.
import type { RunEvent } from "../core/dag.ts";

export const ROLES = ["claims", "env", "planner", "debugger"] as const;
export type Role = (typeof ROLES)[number];

// One role's audited outcome in one run. `null` = the role did not act (no evidence either way).
export type Outcome = { ok: boolean; why: string } | null;

export interface RoleTrust { level: "probation" | "trusted"; n: number; ok: number; lower: number; history: { at: string; paper: string; ok: boolean; why: string }[] }
export type TrustStore = Record<Role, RoleTrust>;

// With z=1.96 a clean record reaches 0.6 only at 6/6 (5/5 gives 0.57); one failure needs ~9/10.
export const STANDARD = { minRuns: 5, minLower: 0.6 };

// Wilson score lower bound (95%): a success rate that small samples cannot fake.
export function wilsonLower(ok: number, n: number, z = 1.96): number {
  if (n === 0) return 0;
  const p = ok / n, d = 1 + (z * z) / n;
  return (p + (z * z) / (2 * n) - z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
}

type Ev = RunEvent & { id?: string; output?: Record<string, unknown> };

// Harness facts per role, read from a run's journal.
export function roleOutcomes(events: Ev[]): Record<Role, Outcome> {
  const ok = (id: string) => events.find((e) => e.t === "ok" && e.id === id) as (Ev & { output: Record<string, unknown> }) | undefined;
  const fails = (prefix: string) => events.filter((e) => e.t === "fail" && (e.id ?? "").startsWith(prefix)) as (Ev & { category: string; message: string })[];
  const out: Record<Role, Outcome> = { claims: null, env: null, planner: null, debugger: null };

  const c = ok("repro.claims");
  if (c) {
    const rejected = ((c.output.rejected as unknown[]) ?? []).length;
    out.claims = rejected === 0 ? { ok: true, why: "every extracted number is in the PDF" } : { ok: false, why: `${rejected} extracted number(s) not in the PDF` };
  } else if (fails("repro.claims").length) out.claims = { ok: false, why: "no claim survived validation" };

  const e = ok("repro.env");
  if (e) {
    const smoke = e.output.smoke as { ok?: boolean } | undefined;
    // A cached environment is not new evidence about the agent.
    if (e.output.cached !== true) out.env = !smoke?.ok ? { ok: false, why: "smoke test failed" } : e.output.repaired === true ? { ok: false, why: "smoke passed only after a repair round" } : { ok: true, why: "smoke test passed on the agent's own report" };
  } else if (fails("repro.env").some((f) => f.category !== "provider_fatal" && f.category !== "provider_transient")) out.env = { ok: false, why: "environment step failed" };

  const p = ok("repro.plan");
  if (p) out.planner = { ok: true, why: "plan accepted" };
  else if (fails("repro.plan").some((f) => f.category === "contract_violation")) out.planner = { ok: false, why: "plan rejected by the validator" };

  const debugs = events.filter((x) => x.t === "ok" && (x.id ?? "").startsWith("repro.debug")) as (Ev & { output: Record<string, unknown> })[];
  if (debugs.length) {
    const isOk = (x: Ev, re: RegExp) => x.t === "ok" && re.test(x.id ?? "");
    const firstAccepted = events.findIndex((x) => isOk(x, /^repro\.patch/) && (x as Ev & { output?: { accepted?: boolean } }).output?.accepted === true);
    // A stage or comparison that succeeds after an accepted change (new version or re-tagged rerun): the fix worked.
    const accepted = firstAccepted >= 0 ? [firstAccepted] : [];
    const fixed = firstAccepted >= 0 && events.slice(firstAccepted + 1).some((x) => isOk(x, /^repro\.(stage|compare):/));
    out.debugger = fixed ? { ok: true, why: "an accepted change made a failing step succeed" } : { ok: false, why: accepted.length ? "accepted change did not fix the step" : "no usable change" };
  }
  return out;
}

export function emptyStore(): TrustStore {
  return Object.fromEntries(ROLES.map((r) => [r, { level: "probation", n: 0, ok: 0, lower: 0, history: [] }])) as unknown as TrustStore;
}

// Asymmetric: promotion needs the standard; a single failure while trusted demotes.
export function record(store: TrustStore, paper: string, at: string, outcomes: Record<Role, Outcome>): TrustStore {
  const next = structuredClone(store);
  for (const role of ROLES) {
    const o = outcomes[role];
    if (!o) continue;
    const t = next[role];
    t.n++; if (o.ok) t.ok++;
    t.lower = wilsonLower(t.ok, t.n);
    t.history = [...t.history, { at, paper, ok: o.ok, why: o.why }].slice(-50);
    if (t.level === "trusted" && !o.ok) t.level = "probation";
    else if (t.level === "probation" && t.n >= STANDARD.minRuns && t.lower >= STANDARD.minLower) t.level = "trusted";
  }
  return next;
}

// How far a campaign may scale given the store: unproven roles keep it small.
export function scaleGate(store: TrustStore): { maxPapers: number; parallel: number; blockedBy: Role[] } {
  const blockedBy = ROLES.filter((r) => store[r].level !== "trusted");
  return blockedBy.length ? { maxPapers: 3, parallel: 1, blockedBy } : { maxPapers: Infinity, parallel: 3, blockedBy: [] };
}

export function renderTrust(store: TrustStore): string {
  return ["| role | level | runs | ok | success-rate lower bound | last outcome |", "|---|---|---|---|---|---|",
    ...ROLES.map((r) => { const t = store[r], last = t.history.at(-1); return `| ${r} | ${t.level} | ${t.n} | ${t.ok} | ${t.lower.toFixed(2)} | ${last ? `${last.ok ? "ok" : "FAIL"} ${last.paper}: ${last.why}` : "—"} |`; }),
    "", `Standard for promotion: ≥ ${STANDARD.minRuns} audited runs and a 95% Wilson lower bound ≥ ${STANDARD.minLower}. One failure while trusted demotes.`].join("\n");
}
