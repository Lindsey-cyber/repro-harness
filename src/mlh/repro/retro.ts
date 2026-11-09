// Automatic round retrospective. Pure. After every campaign round the harness, not a
// person, computes the hillclimb metric, compares it with the previous round, ranks
// what failed, evaluates the stop predicate and names the next thing to fix. The
// result is appended to the decision log, so the loop's state never lives in a chat.
import { ROLES, type TrustStore } from "./trustscore.ts";

// The hillclimb metric: a run counts as a pipeline success when every claim got a
// definitive verdict (reproduced or not) and the run was not stopped. Plus median wall time.
export function pipelineMetric(rows: Record<string, unknown>[]): { success: number; n: number; medianMin: number } {
  const ok = rows.filter((r) => { const c = (r.claims as { status: string }[]) ?? []; return !r.stopped && c.length > 0 && c.every((x) => x.status === "reproduced" || x.status === "not_reproduced"); }).length;
  const m = rows.map((r) => Number(r.wallMin)).filter(Number.isFinite).sort((a, b) => a - b);
  return { success: ok, n: rows.length, medianMin: m.length ? m[Math.floor((m.length - 1) / 2)]! : 0 };
}

export interface Round { label: string; at: string; success: number; n: number; medianMin: number; gpuUsd: number; llmUsd: number; blockedBy: string[]; topFailure?: string }

export const STOP = { minSuccess: 0.8, rounds: 2, maxMedianMin: 45 };

// Failure categories a person must act on, versus ones the agents should overcome.
const OWNER: Record<string, string> = {
  provider_fatal: "restore the provider balance or limit (owner action)",
  contract_violation: "planner/claims contract: replay the recorded drafts (tests/mlh-replay) and fix the validator or the role prompt",
  experiment_error: "environment or code: inspect env smoke and debugger transcripts",
  timeout: "stage timeouts: plan smaller stages or raise timeoutS within budget",
  sandbox_transient: "infrastructure flakiness: check Modal status; retries cover it",
  policy_denied: "authorization: a patch or fetch needed approval",
};

export function roundOf(label: string, at: string, rows: Record<string, unknown>[], trust: TrustStore): Round {
  const m = pipelineMetric(rows);
  const cats: Record<string, number> = {};
  for (const r of rows) for (const [k, v] of Object.entries((r.failures as Record<string, number>) ?? {})) cats[k] = (cats[k] ?? 0) + v;
  const top = Object.entries(cats).sort((a, b) => b[1] - a[1])[0]?.[0];
  return { label, at, success: m.success, n: m.n, medianMin: m.medianMin, gpuUsd: +rows.reduce((a, r) => a + Number(r.gpuUsd || 0), 0).toFixed(2), llmUsd: +rows.reduce((a, r) => a + Number(r.llmUsd || 0), 0).toFixed(2), blockedBy: ROLES.filter((x) => trust[x].level !== "trusted"), ...(top ? { topFailure: top } : {}) };
}

export function stopReached(history: Round[]): { done: boolean; why: string } {
  const last = history.filter((r) => r.n > 0).slice(-STOP.rounds);
  if (last.length < STOP.rounds) return { done: false, why: `need ${STOP.rounds} finished rounds, have ${last.length}` };
  const bad = last.find((r) => r.blockedBy.length || r.success / r.n < STOP.minSuccess || r.medianMin > STOP.maxMedianMin);
  return bad ? { done: false, why: `${bad.label}: success ${bad.success}/${bad.n}, median ${bad.medianMin} min, on probation: ${bad.blockedBy.join(", ") || "none"}` } : { done: true, why: `last ${STOP.rounds} rounds meet the standard` };
}

export function renderRetro(r: Round, prev: Round | undefined, history: Round[]): string {
  const rate = (x: Round) => (x.n ? x.success / x.n : 0);
  const delta = prev ? ` (previous ${prev.label}: ${prev.success}/${prev.n}, ${rate(r) > rate(prev) ? "better" : rate(r) < rate(prev) ? "WORSE" : "same"})` : "";
  const stop = stopReached(history);
  return [
    `## ${r.label} (auto retro, ${r.at})`,
    `- Pipeline success ${r.success}/${r.n}${delta}; median ${r.medianMin} min/paper; GPU $${r.gpuUsd}, LLM $${r.llmUsd}.`,
    `- Roles on probation: ${r.blockedBy.join(", ") || "none"} (scale ${r.blockedBy.length ? "gated" : "open"}).`,
    `- Most frequent failure: ${r.topFailure ?? "none"}${r.topFailure ? ` -> next: ${OWNER[r.topFailure] ?? "inspect the failed nodes in the summary"}` : ""}.`,
    `- Stop predicate: ${stop.done ? "REACHED" : "not reached"} (${stop.why}).`, "",
  ].join("\n");
}
