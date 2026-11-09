// Failure attribution and cost accounting. Pure. Every failed attempt gets exactly one class,
// and every dollar (LLM and Modal) is assigned to productive work or to a named kind of waste.
// Only "agent" failures count against agents (trustscore.ts uses the same table).
import type { FailureCategory } from "../core/types.ts";

export type FailureClass = "infrastructure" | "resource" | "agent" | "paper" | "unclassified";

const AGENT_NODES = /^repro\.(claims|env|plan|debug|patch)\b/;

// The table, by category and by the kind of node that failed.
export function failureClass(category: string | undefined, nodeId: string, message = ""): FailureClass {
  const agentNode = AGENT_NODES.test(nodeId);
  switch (category as FailureCategory) {
    case "sandbox_transient": case "provider_transient": return "infrastructure";
    case "precondition": return /token|auth|pdf|arxiv|fetch|download/i.test(message) ? "infrastructure" : agentNode ? "agent" : "infrastructure";
    case "provider_fatal": case "resource_limit": return "resource";
    case "policy_denied": return /^budget:/.test(message) ? "resource" : "agent";
    case "contract_violation": return agentNode ? "agent" : "unclassified";
    // An agent that ran out of steps or time is the agent's failure; a stage whose code fails as
    // planned is about the paper's repository until the debugger shows otherwise.
    case "experiment_error": return agentNode ? "agent" : "paper";
    case "timeout": return agentNode ? "agent" : "paper";
    default: return "unclassified"; // harness_bug or anything unknown: stop and investigate
  }
}

export interface TraceRow { id: string; attempt?: number; outcome: string; category?: string; costUsd?: number; gpuUsd?: number; ms?: number }
export interface Bucket { usd: number; llmUsd: number; gpuUsd: number; minutes: number; attempts: number }
export interface CostReport {
  productive: Bucket;
  retried: Record<FailureClass, Bucket>; // failed attempts of nodes that later succeeded
  failed: Record<FailureClass, Bucket>; // failed attempts of nodes that never succeeded
  killed: Bucket; // Modal spend of attempts that left no outcome (process killed)
  unattributed: Bucket; // Modal spend with no attempt: unsettled reservations, charges outside a node, runs recorded before per-attempt accounting
  totalUsd: number;
  wasteShare: number; // everything but productive, as a share of the total
}

const empty = (): Bucket => ({ usd: 0, llmUsd: 0, gpuUsd: 0, minutes: 0, attempts: 0 });
const classes = (): Record<FailureClass, Bucket> => ({ infrastructure: empty(), resource: empty(), agent: empty(), paper: empty(), unclassified: empty() });
const add = (b: Bucket, llm: number, gpu: number, ms: number) => { b.llmUsd += llm; b.gpuUsd += gpu; b.usd += llm + gpu; b.minutes += ms / 60000; b.attempts++; };

// rows: the run's trace.jsonl; byAttempt: budget.json's Modal spend per "<id>#<attempt>";
// messages: the failure message per "<id>#<attempt>" from the journal (for the class table).
export function costReport(rows: TraceRow[], byAttempt: Record<string, number> = {}, messages: Record<string, string> = {}): CostReport {
  const r: CostReport = { productive: empty(), retried: classes(), failed: classes(), killed: empty(), unattributed: empty(), totalUsd: 0, wasteShare: 0 };
  const succeeded = new Set(rows.filter((x) => x.outcome === "ok").map((x) => x.id));
  const seen = new Set<string>();
  for (const x of rows) {
    const key = `${x.id}#${x.attempt ?? 1}`;
    seen.add(key);
    const gpu = x.gpuUsd ?? byAttempt[key] ?? 0, llm = x.costUsd ?? 0, ms = x.ms ?? 0;
    if (x.outcome === "ok") add(r.productive, llm, gpu, ms);
    else if (x.outcome === "fail") add((succeeded.has(x.id) ? r.retried : r.failed)[failureClass(x.category, x.id, messages[key])], llm, gpu, ms);
  }
  for (const [k, usd] of Object.entries(byAttempt)) if (!seen.has(k)) add(k.startsWith("(") ? r.unattributed : r.killed, 0, usd, 0);
  const all = [r.productive, r.killed, r.unattributed, ...Object.values(r.retried), ...Object.values(r.failed)];
  r.totalUsd = all.reduce((a, b) => a + b.usd, 0);
  // Unattributed spend is unknown, not waste: it is left out of the share and shown on its own.
  const known = r.totalUsd - r.unattributed.usd;
  r.wasteShare = known > 0 ? Math.max(0, 1 - r.productive.usd / known) : 0;
  return r;
}

export function renderCost(name: string, c: CostReport): string {
  const line = (label: string, b: Bucket) => b.attempts || b.usd ? [`| ${label} | ${b.attempts} | ${b.llmUsd.toFixed(3)} | ${b.gpuUsd.toFixed(3)} | ${b.minutes.toFixed(1)} |`] : [];
  return [`### ${name}: $${c.totalUsd.toFixed(2)}, waste ${(c.wasteShare * 100).toFixed(0)} % of attributed spend${c.unattributed.usd ? `; $${c.unattributed.usd.toFixed(2)} Modal unattributed` : ""}`, "", "| bucket | attempts | LLM $ | Modal $ | min |", "|---|---|---|---|---|",
    ...line("productive", c.productive),
    ...Object.entries(c.retried).flatMap(([k, b]) => line(`retried, ${k}`, b)),
    ...Object.entries(c.failed).flatMap(([k, b]) => line(`failed, ${k}`, b)),
    ...line("killed (no outcome)", c.killed), ...line("unattributed", c.unattributed), ""].join("\n");
}

// Strict stop conditions, checked after every attempt. `fails` are this process's failures only:
// a resume after investigation starts from a clean count.
export const LIMITS = { wasteShare: 0.3, wasteMinUsd: 0.5, llmCapUsd: Number(process.env.MLH_LLM_CAP_USD ?? 1) };
export const signature = (category: string, message: string) => `${category}: ${(message.split("\n")[0] ?? "").replace(/\b(ac|sb|im|ta)-\w+/g, "<id>").replace(/[0-9a-f]{8,}/gi, "<hex>").replace(/\d+(\.\d+)?/g, "<n>").slice(0, 160)}`;

export function stopCheck(fails: { id: string; category: string; message: string }[], report: CostReport): { reason: string; detail: string } | undefined {
  const odd = fails.find((f) => failureClass(f.category, f.id, f.message) === "unclassified");
  if (odd) return { reason: "unclassified_failure", detail: `${odd.id}: ${signature(odd.category, odd.message)}` };
  const seen = new Map<string, string>();
  for (const f of fails) {
    if (failureClass(f.category, f.id, f.message) !== "infrastructure") continue;
    const sig = signature(f.category, f.message);
    if (seen.has(sig)) return { reason: "infra_repeat", detail: `${sig} (at ${seen.get(sig)} and ${f.id})` };
    seen.set(sig, f.id);
  }
  // Failures about the paper's own code are the debugger's normal work, not waste to stop on.
  const wasted = [...Object.entries(report.retried), ...Object.entries(report.failed)].filter(([k]) => k !== "paper").reduce((a, [, b]) => a + b.usd, 0) + report.killed.usd;
  if (report.totalUsd > LIMITS.wasteMinUsd && wasted / report.totalUsd > LIMITS.wasteShare) return { reason: "waste", detail: `$${wasted.toFixed(2)} of $${report.totalUsd.toFixed(2)} went to failed or killed attempts (limit ${LIMITS.wasteShare * 100} %)` };
  const llm = report.productive.llmUsd + [...Object.values(report.retried), ...Object.values(report.failed)].reduce((a, b) => a + b.llmUsd, 0);
  if (llm > LIMITS.llmCapUsd) return { reason: "llm_cap", detail: `LLM $${llm.toFixed(2)} > cap $${LIMITS.llmCapUsd}` };
  return undefined;
}
