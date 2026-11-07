// Evidence-backed report and bounded context summary, both derived from state. Pure.
import type { RunState } from "./dag.ts";
import type { Finding, Manifest } from "./types.ts";

export interface Summary {
  objective: string;
  source: string;
  mode: string;
  reproduced: "yes" | "no" | "unknown";
  confirmed: Finding[];
  unverified: Finding[]; // suspected or disputed: shown, never counted as confirmed
  refuted: Finding[];
  nodes: { id: string; status: string; attempts: number; reason?: string }[];
  gaps: { skill: string; reason: string }[];
  held: { id: string; reason: string }[];
}

export function summarize(s: RunState): Summary {
  const fs = [...s.findings.values()];
  const claimFail = fs.some((f) => f.category === "repro.claim_not_met" && f.status === "confirmed");
  const claimNode = [...s.nodes.values()].find((n) => n.node.skill === "stats.claim" && n.node.purpose !== "confirm");
  return {
    objective: s.objective,
    source: s.source,
    mode: s.mode,
    reproduced: claimFail ? "no" : claimNode?.status === "succeeded" ? (fs.some((f) => f.category === "repro.claim_not_met") ? "unknown" : "yes") : "unknown",
    confirmed: fs.filter((f) => f.status === "confirmed"),
    unverified: fs.filter((f) => f.status === "suspected" || f.status === "disputed"),
    refuted: fs.filter((f) => f.status === "refuted"),
    nodes: s.order.map((id) => {
      const n = s.nodes.get(id)!;
      return { id, status: n.status, attempts: n.attempts, ...(n.reason ? { reason: n.reason } : {}) };
    }),
    gaps: s.gaps,
    held: s.order.filter((id) => s.nodes.get(id)!.status === "held").map((id) => ({ id, reason: s.nodes.get(id)!.reason ?? "" })),
  };
}

const fmt = (v: unknown): string => (typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(4)) : JSON.stringify(v));

function findingBlock(f: Finding): string {
  const ev = Object.entries(f.evidence).map(([k, v]) => `  - ${k}: ${fmt(v)}`).join("\n");
  const cs = f.confirmations.map((c) => `  - ${c.supports ? "supports" : "refutes"} (${c.node}): ${c.note}`).join("\n");
  return `### [${f.severity}] ${f.category}: ${f.summary}\n- raised by \`${f.node}\` (${f.kind}), status **${f.status}**\n- evidence:\n${ev}${cs ? `\n- independent checks:\n${cs}` : ""}`;
}

export function renderReport(s: RunState, manifest: Manifest | undefined, extra: { artifacts: { node: string; name: string; sha256: string }[]; reproduce: string }): string {
  const m = summarize(s);
  const lines = [
    `# ML harness report`,
    ``,
    `- objective: ${m.objective}`,
    `- source: ${m.source}`,
    `- mode: ${m.mode}`,
    `- claim reproduced: **${m.reproduced}**`,
    `- confirmed findings: ${m.confirmed.length}; unverified: ${m.unverified.length}; refuted: ${m.refuted.length}`,
    ``,
    `## Manifest (what the harness believed about the repository)`,
    "```json",
    JSON.stringify(manifest ?? {}, null, 1),
    "```",
    ``,
    `## Confirmed findings`,
    m.confirmed.length ? m.confirmed.map(findingBlock).join("\n\n") : "None.",
    ``,
    `## Unverified suspicions (not counted as findings)`,
    m.unverified.length ? m.unverified.map(findingBlock).join("\n\n") : "None.",
    ``,
    `## Refuted suspicions`,
    m.refuted.length ? m.refuted.map(findingBlock).join("\n\n") : "None.",
    ``,
    `## Coverage`,
    m.gaps.length ? m.gaps.map((g) => `- not run: \`${g.skill}\`: ${g.reason}`).join("\n") : "Every planned skill ran.",
    ...m.nodes.filter((n) => n.status !== "succeeded").map((n) => `- \`${n.id}\` ${n.status}${n.reason ? `: ${n.reason}` : ""}`),
    ``,
    `## Execution`,
    `| node | status | attempts |`,
    `|---|---|---|`,
    ...m.nodes.map((n) => `| ${n.id} | ${n.status} | ${n.attempts} |`),
    ``,
    `## Artifacts (content-addressed)`,
    ...extra.artifacts.map((a) => `- \`${a.node}\` ${a.name}: sha256 ${a.sha256.slice(0, 16)}…`),
    ``,
    `## Reproduce`,
    "```bash",
    extra.reproduce,
    "```",
    ``,
  ];
  return lines.join("\n");
}

// A bounded, structured context for a resumed session or a model: the most
// decision-relevant facts first, truncated to the budget from the bottom.
export function compact(s: RunState, budgetChars = 4000): string {
  const m = summarize(s);
  const parts = [
    `objective: ${m.objective}`,
    `source: ${m.source} (mode ${m.mode}, ${s.done ? "finished" : "in progress"})`,
    `held for approval: ${m.held.map((h) => `${h.id} (${h.reason})`).join("; ") || "none"}`,
    `confirmed: ${m.confirmed.map((f) => `${f.category}: ${f.summary}`).join(" | ") || "none"}`,
    `unverified: ${m.unverified.map((f) => `${f.category}`).join(", ") || "none"}`,
    `refuted: ${m.refuted.map((f) => `${f.category}`).join(", ") || "none"}`,
    `failed: ${m.nodes.filter((n) => n.status === "failed").map((n) => n.id).join(", ") || "none"}`,
    `gaps: ${m.gaps.map((g) => `${g.skill} (${g.reason})`).join("; ") || "none"}`,
    `nodes: ${m.nodes.map((n) => `${n.id}=${n.status}`).join(", ")}`,
  ];
  let out = "";
  for (const p of parts) {
    if (out.length + p.length + 1 > budgetChars) { out += "…(truncated)\n"; break; }
    out += p + "\n";
  }
  return out;
}
