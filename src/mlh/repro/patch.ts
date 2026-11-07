// Diffing an agent's workspace against a source version, and building the next
// version from an accepted change. Only source-like files count: logs,
// checkpoints and data the agent's runs produced are not part of a patch.
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { runProc } from "../io/proc.ts";

const SOURCE = /\.(py|pyx|cfg|ini|toml|ya?ml|json|txt|sh|md)$/i;
const SKIP = /(^|\/)(\.git|__pycache__|\.mlh_in|outputs?|logs?|runs?|checkpoints?|wandb|lightning_logs)(\/|$)/;

function files(root: string): string[] {
  const out: string[] = [];
  const walk = (d: string) => {
    for (const e of readdirSync(d, { withFileTypes: true })) {
      const p = join(d, e.name), r = relative(root, p);
      if (SKIP.test(r)) continue;
      if (e.isDirectory()) walk(p); else if (SOURCE.test(e.name) && statSync(p).size < 1_000_000) out.push(r);
    }
  };
  walk(root);
  return out;
}

// Changed source files: modified files that exist in base, and new source files.
export function changedFiles(base: string, ws: string): string[] {
  const b = new Set(files(base));
  return files(ws).filter((f) => !b.has(f) ? /\.(py|cfg|ini|toml|ya?ml|txt|sh)$/i.test(f) && !/(^|\/)(log|out|result)/i.test(f) : readFileSync(join(base, f), "utf8") !== readFileSync(join(ws, f), "utf8"));
}

export async function unifiedDiff(base: string, ws: string, changed: string[]): Promise<string> {
  let diff = "";
  for (const f of changed) {
    const a = existsSync(join(base, f)) ? join(base, f) : "/dev/null";
    const r = await runProc(["git", "diff", "--no-index", "--no-color", "-U2", a, join(ws, f)], { cwd: base, env: { PATH: process.env.PATH ?? "/usr/bin:/bin" }, timeoutMs: 30_000 });
    diff += r.stdout.replaceAll(`a${base}/`, "a/").replaceAll(`b${ws}/`, "b/").replaceAll(base + "/", "").replaceAll(ws + "/", "");
  }
  return diff;
}

export function buildVersion(base: string, ws: string, changed: string[], dest: string): void {
  cpSync(base, dest, { recursive: true });
  for (const f of changed) { mkdirSync(dirname(join(dest, f)), { recursive: true }); writeFileSync(join(dest, f), readFileSync(join(ws, f))); }
}
