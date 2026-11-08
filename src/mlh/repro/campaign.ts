// Autonomous campaign: reproduce a queue of papers without a human watching.
// Each paper runs as a child process (`mlh repro`), so a watchdog can kill a run
// that stopped making progress and resume it from its journal. A fatal provider
// error (no balance, bad key) stops the whole campaign. After every paper the
// summary table is rewritten, so the state is always readable in one file.
import { appendFileSync, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { runProc, startProc } from "../io/proc.ts";

export interface Paper { id: string; arxiv: string; repo: string; commit: string; objective: string; budgetUsd?: number }

const ROOT = resolve(import.meta.dirname, "../../..");

async function prepare(p: Paper): Promise<string> {
  const dir = join(ROOT, ".mlh", "complex-src", p.id);
  if (existsSync(join(dir, ".git"))) return dir;
  const env = { PATH: process.env.PATH ?? "/usr/bin:/bin", HOME: process.env.HOME ?? "/tmp", ...(process.env.HTTPS_PROXY ? { HTTPS_PROXY: process.env.HTTPS_PROXY } : {}), GIT_TERMINAL_PROMPT: "0" };
  let r = await runProc(["git", "clone", "--quiet", p.repo, dir], { cwd: ROOT, env, timeoutMs: 600_000 });
  if (r.code !== 0) throw new Error(`clone ${p.repo}: ${r.stderr.slice(-300)}`);
  r = await runProc(["git", "checkout", "--quiet", p.commit], { cwd: dir, env, timeoutMs: 120_000 });
  if (r.code !== 0) throw new Error(`checkout ${p.commit}: ${r.stderr.slice(-300)}`);
  return dir;
}

interface Ev { t: string; at: string; id?: string; node?: { id: string; skill: string; params: Record<string, unknown> } }
const events = (runDir: string): Ev[] => (existsSync(join(runDir, "journal.jsonl")) ? readFileSync(join(runDir, "journal.jsonl"), "utf8").split("\n").filter(Boolean).flatMap((l) => { try { return [JSON.parse(l) as Ev]; } catch { return []; } }) : []);

// How long a run may go without a journal event: long GPU stages emit nothing while they run.
function allowedSilenceMs(runDir: string): number {
  const ev = events(runDir);
  const started = new Set(ev.filter((e) => e.t === "start").map((e) => e.id));
  for (const e of ev) if (e.t === "ok" || e.t === "fail") started.delete(e.id);
  let max = 30 * 60_000;
  const plan = ev.find((e) => e.t === "ok" && e.id === "repro.plan") as { output?: { gpuPlan?: { stages: { id: string; timeoutS: number }[] } } } | undefined;
  for (const id of started) {
    const st = plan?.output?.gpuPlan?.stages.find((s) => id?.startsWith(`repro.stage:${s.id}:`));
    if (st) max = Math.max(max, (st.timeoutS + 900) * 1000);
    if (id === "repro.fetch") max = Math.max(max, 3.5 * 3600_000);
  }
  return max;
}

function runChild(args: string[], runDir: string, log: string): Promise<{ code: number | null; killedForStall: boolean }> {
  const child = startProc(["npx", "tsx", join(ROOT, "src/mlh/cli.ts"), ...args], { cwd: ROOT, env: process.env, onOutput: (d) => appendFileSync(log, d) });
  let killed = false;
  const timer = setInterval(() => {
    const j = join(runDir, "journal.jsonl");
    const age = existsSync(j) ? Date.now() - statSync(j).mtimeMs : 0;
    if (age > allowedSilenceMs(runDir)) {
      killed = true;
      appendFileSync(log, `\n[watchdog] no progress for ${Math.round(age / 60000)} min: killing for resume\n`);
      child.kill();
    }
  }, 60_000);
  return child.done.then((code) => { clearInterval(timer); return { code, killedForStall: killed }; });
}

// One row of the summary, derived from the run's own files.
export function summarize(p: Paper, runDir: string, resumes: number, wallS: number): Record<string, unknown> {
  const rep = existsSync(join(runDir, "repro.json")) ? JSON.parse(readFileSync(join(runDir, "repro.json"), "utf8")) : {};
  const ev = events(runDir);
  const trace = existsSync(join(runDir, "trace.jsonl")) ? readFileSync(join(runDir, "trace.jsonl"), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l) as { tokens?: number; costUsd?: number; category?: string; outcome: string }) : [];
  const budget = existsSync(join(runDir, "budget.json")) ? JSON.parse(readFileSync(join(runDir, "budget.json"), "utf8")) : { spent: 0 };
  const stopped = existsSync(join(runDir, "stopped.json")) ? JSON.parse(readFileSync(join(runDir, "stopped.json"), "utf8")).reason : undefined;
  const plan = (ev.find((e) => e.t === "ok" && e.id === "repro.plan") as { output?: { gpuPlan?: { fidelity: string } } } | undefined)?.output?.gpuPlan;
  const env = (ev.find((e) => e.t === "ok" && e.id === "repro.env") as { output?: { smoke?: { ok: boolean }; image?: string } } | undefined)?.output;
  const failures: Record<string, number> = {};
  for (const t of trace) if (t.outcome === "fail" && t.category) failures[t.category] = (failures[t.category] ?? 0) + 1;
  const claims = ((rep.claims ?? []) as { id: string; value: number; status: string; mean?: number }[]).map((c) => ({ id: c.id, paper: c.value, status: c.status, mean: c.mean }));
  const failedNodes = ((rep.failed ?? []) as { id: string; error?: string }[]).map((f) => `${f.id}: ${(f.error ?? "").slice(0, 120)}`);
  return {
    paper: p.id, arxiv: p.arxiv, claims, fidelity: plan?.fidelity ?? "-", image: env?.image ?? "-", smoke: env?.smoke?.ok ?? null,
    gpuUsd: Number((budget.spent ?? 0).toFixed(3)), llmTokens: trace.reduce((a, t) => a + (t.tokens ?? 0), 0), llmUsd: Number(trace.reduce((a, t) => a + (t.costUsd ?? 0), 0).toFixed(3)),
    wallMin: Math.round(wallS / 60), resumes, failures, failedNodes, stopped: stopped ?? null, done: ev.some((e) => e.t === "done"),
  };
}

function render(rows: Record<string, unknown>[]): string {
  const cell = (r: Record<string, unknown>) => ((r.claims as { id: string; paper: number; status: string; mean?: number }[]) ?? []).map((c) => `${c.status === "reproduced" ? "✔" : c.status === "not_reproduced" ? "≠" : "✘"} ${c.id}: ${c.mean?.toFixed(3) ?? "—"} vs ${c.paper}`).join("<br>") || "—";
  const claims = rows.flatMap((r) => (r.claims as { status: string }[]) ?? []);
  return [
    `# Campaign summary (${new Date().toISOString()})`, "",
    `papers ${rows.length}; claims ${claims.length}: reproduced ${claims.filter((c) => c.status === "reproduced").length}, not reproduced ${claims.filter((c) => c.status === "not_reproduced").length}, not runnable ${claims.filter((c) => c.status === "not_runnable" || c.status === "pending").length}; GPU $${rows.reduce((a, r) => a + Number(r.gpuUsd), 0).toFixed(2)}; LLM $${rows.reduce((a, r) => a + Number(r.llmUsd), 0).toFixed(2)} (${rows.reduce((a, r) => a + Number(r.llmTokens), 0)} tokens); resumes ${rows.reduce((a, r) => a + Number(r.resumes), 0)}`, "",
    "| paper | claims (mean vs paper) | fidelity | env smoke | GPU $ | LLM tokens | min | resumes | failures |", "|---|---|---|---|---|---|---|---|---|",
    ...rows.map((r) => `| ${r.paper} (${r.arxiv}) | ${cell(r)} | ${r.fidelity} | ${r.smoke === null ? "—" : r.smoke ? "pass" : "FAIL"} | ${r.gpuUsd} | ${r.llmTokens} | ${r.wallMin} | ${r.resumes} | ${Object.entries(r.failures as Record<string, number>).map(([k, v]) => `${k}×${v}`).join(", ") || "—"}${r.stopped ? ` STOPPED: ${r.stopped}` : ""} |`),
    "", ...rows.flatMap((r) => ((r.failedNodes as string[]) ?? []).length ? [`- ${r.paper}: ${(r.failedNodes as string[]).join("; ")}`] : []), "",
  ].join("\n");
}

export async function runCampaign(o: { config: string; label: string; parallel: number; out: string }): Promise<void> {
  const papers = JSON.parse(readFileSync(o.config, "utf8")) as Paper[];
  const base = join(ROOT, ".mlh", "campaign", o.label);
  mkdirSync(base, { recursive: true });
  mkdirSync(o.out, { recursive: true });
  const rowsFile = join(o.out, `${o.label}.jsonl`);
  const rows: Record<string, unknown>[] = existsSync(rowsFile) ? readFileSync(rowsFile, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)) : [];
  const done = new Set(rows.map((r) => r.paper));
  let stop = false;
  const queue = papers.filter((p) => !done.has(p.id));
  await Promise.all(Array.from({ length: Math.min(o.parallel, queue.length) }, async () => {
    for (let p = queue.shift(); p && !stop; p = queue.shift()) {
      const runDir = join(base, p.id), log = join(base, `${p.id}.log`);
      const t0 = Date.now();
      let resumes = 0;
      try {
        const src = await prepare(p);
        let r = await runChild(existsSync(join(runDir, "journal.jsonl")) ? ["resume", runDir, "--llm"] : ["repro", src, "--arxiv", p.arxiv, "--objective", p.objective, "--llm", "--gpu-budget", String(p.budgetUsd ?? 5), "--run-dir", runDir], runDir, log);
        // Killed for a stall or crashed: resume from the journal (at most twice).
        while ((r.killedForStall || r.code !== 0) && !existsSync(join(runDir, "stopped.json")) && resumes < 2) {
          resumes++;
          r = await runChild(["resume", runDir, "--llm"], runDir, log);
        }
      } catch (e) {
        appendFileSync(log, `\n[campaign] ${String(e)}\n`);
      }
      const row = summarize(p, runDir, resumes, (Date.now() - t0) / 1000);
      if (row.stopped === "provider_fatal") stop = true; // no point starting the next paper
      rows.push(row);
      appendFileSync(rowsFile, JSON.stringify(row) + "\n");
      writeFileSync(join(o.out, `${o.label}.md`), render(rows));
    }
  }));
  writeFileSync(join(o.out, `${o.label}.md`), render(rows));
}
