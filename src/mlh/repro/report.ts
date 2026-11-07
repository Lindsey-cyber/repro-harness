// The reproduction report, derived from state. Pure. Root causes carry their
// evidence tier: confirmed by an intervention (an accepted change after which the
// claim's outcome changed), observed in a run log (classified by harness code),
// or only reported by an agent.
import type { RunState } from "../core/dag.ts";
import { claimsOf, finalStatus, N, specsOf } from "./flow.ts";

export type Tier = "intervention" | "log" | "agent";
export interface Cause { category: string; tier: Tier; evidence: string; source: string }

// Harness classification of a failed run's error text.
// A missing module that is part of the repository itself is an import-path problem, not a missing package.
export function classifyLog(msg: string, localModules: Set<string> = new Set()): string | undefined {
  const missing = /ModuleNotFoundError: No module named '([\w.]+)'/.exec(msg)?.[1]?.split(".")[0];
  if (missing) return localModules.has(missing) ? "env.path_or_config" : "env.missing_dependency";
  if (/(AttributeError: module '(numpy|scipy|torch|networkx|sklearn|pandas)[\w.]*' has no attribute|ImportError: cannot import name \S+ from '(numpy|scipy|torch|networkx|sklearn|pandas))|(site-packages|\/opt\/)\S+\.py.*\n.*(TypeError|AttributeError)/.test(msg)) return "env.version_drift";
  if (/FileNotFoundError|No such file or directory/.test(msg)) return "env.path_or_config";
  if (/→ (timeout|resource_limit)/.test(msg)) return "run.resource_limit";
  if (/Traceback|Error/.test(msg)) return "run.crash";
  return undefined;
}

export function causes(s: RunState, localModules: Set<string> = new Set()): Cause[] {
  const out: Cause[] = [];
  const env = s.nodes.get(N.env)?.output as { problems?: { category: string; evidence: string }[] } | undefined;
  for (const p of env?.problems ?? []) out.push({ category: String(p.category), tier: "agent", evidence: String(p.evidence).slice(0, 300), source: N.env });
  for (const id of s.order) {
    const ns = s.nodes.get(id)!;
    if (ns.node.skill === N.run && ns.failures.length) {
      const c = classifyLog(ns.failures.at(-1)!.message, localModules);
      if (c) out.push({ category: c, tier: "log", evidence: ns.failures.at(-1)!.message.slice(0, 300), source: id });
    }
    if (ns.node.skill === N.debug && ns.status === "succeeded") {
      const d = ns.output as { causes?: { category: string; evidence: string }[]; claim: string };
      const patchId = `${N.patch}:${d.claim}:v${ns.node.params.version}`;
      const accepted = (s.nodes.get(patchId)?.output as { accepted?: boolean } | undefined)?.accepted === true;
      const changed = accepted && outcomeChanged(s, d.claim, Number(ns.node.params.version));
      for (const c of d.causes ?? []) out.push({ category: String(c.category), tier: changed ? "intervention" : "agent", evidence: String(c.evidence).slice(0, 300), source: id });
    }
  }
  return out;
}

// Did the claim's outcome change after the change accepted from version v? (failure -> result, or status/mean moved)
function outcomeChanged(s: RunState, claim: string, v: number): boolean {
  const cmp = (ver: number) => s.order.filter((id) => id.startsWith(`${N.compare}:${claim}:v${ver}`) && s.nodes.get(id)!.status === "succeeded").map((id) => s.nodes.get(id)!.output as { status: string; mean: number }).at(-1);
  const before = cmp(v);
  const after = s.order.filter((id) => id.startsWith(`${N.compare}:${claim}:`) && s.nodes.get(id)!.status === "succeeded" && Number(s.nodes.get(id)!.node.params.version) > v).map((id) => s.nodes.get(id)!.output as { status: string; mean: number }).at(-1)
    ?? (s.order.some((id) => id.startsWith(`${N.compare}:${claim}:v${v}.`) && s.nodes.get(id)!.status === "succeeded") ? cmp(v) : undefined);
  if (!after) return false;
  if (!before) return true; // it did not run before, and runs now
  return before.status !== after.status || Math.abs(before.mean - after.mean) >= 0.01;
}

interface PatchView { id: string; status: string; reason?: string | undefined; accepted?: boolean; risk?: string; reasons?: string[]; basis?: string; version?: number; refused?: string }

export function renderRepro(s: RunState, localModules: Set<string> = new Set()): { json: Record<string, unknown>; markdown: string } {
  const claims = claimsOf(s), fin = finalStatus(s), cs = causes(s, localModules);
  const patches: PatchView[] = s.order.filter((id) => id.startsWith(N.patch)).map((id) => ({ id, status: s.nodes.get(id)!.status, reason: s.nodes.get(id)!.reason, ...((s.nodes.get(id)!.output ?? {}) as Partial<PatchView>) }));
  const held = s.order.filter((id) => s.nodes.get(id)!.status === "held").map((id) => ({ id, reason: s.nodes.get(id)!.reason }));
  const json = {
    objective: s.objective, source: s.source, done: s.done,
    claims: claims.map((c) => ({ ...c, ...fin[c.id] })),
    plan: specsOf(s),
    causes: cs,
    patches: patches.map((p) => ({ id: p.id, status: p.status, accepted: p.accepted, risk: p.risk, reasons: p.reasons, basis: p.basis, version: p.version })),
    held,
    failed: s.order.filter((id) => s.nodes.get(id)!.status === "failed").map((id) => ({ id, error: s.nodes.get(id)!.failures.at(-1)?.message.slice(0, 300) })),
  };
  const fmt = (v?: number) => (v === undefined ? "—" : v.toFixed(4));
  const md = [
    `# Reproduction report`, ``, `- objective: ${s.objective}`, `- source: ${s.source}`, ``,
    `## Claims`, ``, `| claim | paper | status | mean | runs | source version |`, `|---|---|---|---|---|---|`,
    ...claims.map((c) => { const f = fin[c.id]!; return `| ${c.id} (${c.dataset ?? ""} ${c.metric}) | ${c.value}${c.std ? ` ± ${c.std}` : ""} | **${f.status}** | ${fmt(f.mean)} | ${(f.values ?? []).map((v) => v.toFixed(4)).join(", ")} | v${f.version} |`; }),
    ``, `## Root causes`, ``,
    ...(cs.length ? cs.map((c) => `- \`${c.category}\` [${c.tier === "intervention" ? "confirmed by intervention" : c.tier === "log" ? "observed in run log" : "reported by agent"}] (${c.source}): ${c.evidence.replace(/\n/g, " ").slice(0, 200)}`) : ["None found."]),
    ``, `## Changes`, ``,
    ...(patches.length ? patches.map((p) => `- \`${p.id}\`: ${p.status === "held" ? `**held for approval**: ${p.reason}` : p.accepted ? `accepted (${p.risk} risk; ${p.basis})${p.version !== undefined ? `, became source v${p.version}` : ""}` : `not applied${p.refused ? `: ${p.refused}` : ""}`}`) : ["None."]),
    ``, `## Plan`, "```json", JSON.stringify(specsOf(s), null, 1), "```", ``,
  ].join("\n");
  return { json, markdown: md };
}
