// Risk of a code change an agent proposes. Pure. A change is high risk when it
// touches how the result is measured (evaluation, metrics, data splits, seeding)
// or writes a claimed number into the code: exactly the edits that could turn a
// failed reproduction into a fake success. Everything else is medium.
import type { Claim } from "./flow.ts";

export interface FileDiff { path: string; added: string[]; removed: string[] }

const MEASURE_PATH = /(^|\/)(eval\w*|metric\w*|test\w*|score\w*|split\w*)\.py$/i;
const MEASURE_LINE = /\b(accuracy|acc|f1|auc|roc|metric\w*|score\w*|evaluate|eval_|idx_test|test_idx|test_mask|idx_val|val_mask|train_test_split|random_state|manual_seed|np\.random\.seed|random\.seed|set_seed|seed)\b/i;

export function riskOf(files: FileDiff[], claims: Claim[], argvChanges: string[] = []): { risk: "medium" | "high"; reasons: string[] } {
  const reasons: string[] = [];
  for (const f of files) {
    if (MEASURE_PATH.test(f.path)) reasons.push(`${f.path}: an evaluation/metric/split file`);
    const lines = [...f.added, ...f.removed].filter((l) => MEASURE_LINE.test(l));
    if (lines.length) reasons.push(`${f.path}: changes measurement code (${lines[0]!.trim().slice(0, 80)})`);
    for (const l of f.added) for (const c of claims) {
      for (const m of l.matchAll(/(?<![\w.])\d*\.\d+|\b\d{2}(?:\.\d+)?\b/g)) {
        const v = Number(m[0]);
        if (Math.abs(v - c.value) < 5e-4 || Math.abs(v / 100 - c.value) < 5e-4) reasons.push(`${f.path}: adds the claimed value ${m[0]} (${c.id})`);
      }
    }
  }
  for (const a of argvChanges) if (MEASURE_LINE.test(a)) reasons.push(`run command change touches measurement: ${a}`);
  return { risk: reasons.length ? "high" : "medium", reasons: [...new Set(reasons)] };
}

// Parses `git diff --no-index` output into per-file added/removed lines.
export function parseDiff(diff: string, base: string, ws: string): FileDiff[] {
  const files: FileDiff[] = [];
  let cur: FileDiff | undefined;
  for (const line of diff.split("\n")) {
    const m = /^\+\+\+ (?:b\/)?(.*)$/.exec(line);
    if (m) {
      const p = m[1]!.replace(ws.replace(/^\//, ""), "").replace(base.replace(/^\//, ""), "").replace(/^\/+/, "");
      cur = { path: p === "dev/null" ? "" : p, added: [], removed: [] };
      files.push(cur);
      continue;
    }
    const mm = /^--- (?:a\/)?(.*)$/.exec(line);
    if (mm) continue;
    if (!cur) continue;
    if (line.startsWith("+")) cur.added.push(line.slice(1));
    else if (line.startsWith("-")) cur.removed.push(line.slice(1));
  }
  return files.filter((f) => f.path);
}
