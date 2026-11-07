// What a model is shown of a repository, and the deterministic checks a
// model-proposed manifest must pass before any field of it is used.
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, normalize, relative } from "node:path";
import type { Manifest } from "../core/types.ts";

const TEXT = /\.(py|ya?ml|json|toml|cfg|ini|md|txt|sh)$/i;

export function digest(root: string, budget = 60_000): string {
  const files: string[] = [];
  const walk = (d: string) => {
    for (const e of readdirSync(d, { withFileTypes: true })) {
      if (e.name.startsWith(".") || e.name === "__pycache__" || e.name === "outputs") continue;
      const p = join(d, e.name);
      if (e.isDirectory()) walk(p); else files.push(relative(root, p));
    }
  };
  walk(root);
  files.sort();
  const parts = [`Files:\n${files.map((f) => `${f} (${statSync(join(root, f)).size} bytes)`).join("\n")}`];
  let used = parts[0]!.length;
  // README first, then code by path; data files show only their first lines.
  const order = [...files.filter((f) => /readme/i.test(f)), ...files.filter((f) => !/readme/i.test(f))];
  for (const f of order) {
    const body = TEXT.test(f) ? readFileSync(join(root, f), "utf8") : /\.csv$/i.test(f) ? readFileSync(join(root, f), "utf8").split("\n").slice(0, 3).join("\n") + "\n…" : null;
    if (body === null) continue;
    const chunk = `\n--- ${f} ---\n${body}`;
    if (used + chunk.length > budget) { parts.push(`\n--- ${f} --- (omitted: budget)`); continue; }
    parts.push(chunk); used += chunk.length;
  }
  return parts.join("");
}

const inside = (root: string, p: unknown): p is string => typeof p === "string" && !p.startsWith("/") && !normalize(p).startsWith("..");

export function validateManifest(root: string, m: Manifest): { accepted: Partial<Manifest>; rejected: Record<string, string>; why: Record<string, string> } {
  const accepted: Partial<Manifest> = {}, rejected: Record<string, string> = {}, why: Record<string, string> = {};
  const cmd = (k: "train" | "eval") => {
    const c = m[k];
    if (c === undefined) return;
    if (!Array.isArray(c) || !c.length || !c.every((x) => typeof x === "string")) { rejected[k] = "not a list of strings"; return; }
    const argv = c[0] === "python" ? ["python3", ...c.slice(1)] : c;
    if (argv[0] === "make") {
      if (!existsSync(join(root, "Makefile")) && !existsSync(join(root, "makefile"))) { rejected[k] = "make without a Makefile"; return; }
      accepted[k] = argv; why[k] = "Makefile exists"; return;
    }
    if (!["python3", "bash", "sh"].includes(argv[0]!)) { rejected[k] = `only python3, bash, sh or make commands are run, got ${argv[0]}`; return; }
    const script = argv[1] === "-m" ? join(...(argv[2] ?? "").split(".")) + ".py" : argv[1];
    if (!inside(root, script) || !existsSync(join(root, script))) { rejected[k] = `script ${script} does not exist`; return; }
    accepted[k] = argv; why[k] = `${script} exists`;
  };
  cmd("train"); cmd("eval");
  if (m.data) {
    const d = m.data;
    if (!inside(root, d.train) || !inside(root, d.test) || !existsSync(join(root, d.train)) || !existsSync(join(root, d.test))) rejected.data = "train/test files do not exist";
    else if (!/\.csv$/i.test(d.train) || !/\.csv$/i.test(d.test)) rejected.data = "only csv splits can be audited";
    else {
      const header = readFileSync(join(root, d.train), "utf8").split("\n")[0]!.split(",").map((x) => x.trim());
      if (d.label && !header.includes(d.label)) rejected.data = `label ${d.label} not in header`;
      else { accepted.data = d.label ? { train: d.train, test: d.test, label: d.label } : { train: d.train, test: d.test }; why.data = "files exist, label in header"; }
    }
  }
  if (m.outputs && typeof m.outputs === "object") {
    const o = Object.fromEntries(Object.entries(m.outputs).filter(([, v]) => inside(root, v)));
    if (Object.keys(o).length) { accepted.outputs = o; why.outputs = "relative paths inside the repository; checked when the experiment runs"; }
  }
  if (m.metric !== undefined) { if (["accuracy", "f1", "auc"].includes(m.metric)) { accepted.metric = m.metric; why.metric = "supported metric"; } else rejected.metric = `unsupported metric ${m.metric}`; }
  for (const k of ["metricKey", "trainMetricKey", "seedEnv", "seedArg"] as const) if (typeof m[k] === "string") { accepted[k] = m[k]; why[k] = "string; checked when the experiment runs"; }
  if (m.claim) {
    let v = Number(m.claim.value);
    if (v > 1 && v <= 100) v /= 100;
    if (["accuracy", "f1", "auc"].includes(m.claim.metric) && v >= 0 && v <= 1) {
      const readme = readdirSync(root).find((f) => /^readme/i.test(f));
      const text = readme ? readFileSync(join(root, readme), "utf8") : "";
      // The claimed number must literally appear in the README (as a fraction or a percentage).
      const forms = [v.toFixed(2), v.toFixed(3), String(v), (v * 100).toFixed(1), (v * 100).toFixed(0)];
      if (forms.some((x) => text.includes(x))) { accepted.claim = { metric: m.claim.metric, value: v }; why.claim = `value appears in ${readme}`; } else rejected.claim = `value ${v} does not appear in the README`;
    } else rejected.claim = "claim metric or value out of range";
  }
  return { accepted, rejected, why };
}
