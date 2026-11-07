// Per-skill trust with asymmetric transitions. Pure. One audited error demotes
// a skill at once; promotion needs a streak of clean audits AND a human's
// approval. At "confirm" every suspicion a skill raises is independently
// re-checked; at "auto" the generic replication is skipped (designated
// confirmers still run: independent evidence is never skipped).

export type TrustLevel = "confirm" | "auto";

export interface SkillTrust {
  level: TrustLevel;
  streak: number; // consecutive clean audits
  since: string;
  reason: string;
  audits: { at: string; falsePositives: number; misses: number; source: string }[];
}

export const PROMOTION_STREAK = 5;

export const initialTrust = (at: string): SkillTrust => ({ level: "confirm", streak: 0, since: at, reason: "every skill starts at confirm", audits: [] });

export function recordAudit(t: SkillTrust, a: { at: string; falsePositives: number; misses: number; source: string }): SkillTrust {
  const audits = [...t.audits, a].slice(-50);
  if (a.falsePositives > 0 || a.misses > 0) {
    return { level: "confirm", streak: 0, since: t.level === "confirm" ? t.since : a.at, reason: `audit ${a.source}: ${a.falsePositives} false positive(s), ${a.misses} miss(es)`, audits };
  }
  return { ...t, streak: t.streak + 1, audits };
}

export function promotable(t: SkillTrust): boolean {
  return t.level === "confirm" && t.streak >= PROMOTION_STREAK;
}

export function promote(t: SkillTrust, approval: { by: "human"; at: string }): SkillTrust {
  if (!promotable(t)) throw new Error(`not promotable: streak ${t.streak} < ${PROMOTION_STREAK} or already ${t.level}`);
  return { ...t, level: "auto", since: approval.at, reason: `promoted by a human after ${t.streak} clean audits` };
}
