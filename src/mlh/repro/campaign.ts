// Autonomous campaign: reproduce a queue of papers without a human watching.
// Each paper runs as a child process (`mlh repro`), so a watchdog can kill a run
// that stopped making progress and resume it from its journal. A fatal provider
// error (no balance, bad key) stops the whole campaign. After every paper the
// summary table is rewritten, so the state is always readable in one file.
import { appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { runProc, startProc } from "../io/proc.ts";
import { emptyStore, record, renderTrust, roleOutcomes, scaleGate, type TrustStore } from "./trustscore.ts";

export interface Paper { id: string; arxiv: string; repo: string; commit: string; objective: string; budgetUsd?: number }

const ROOT = resolve(import.meta.dirname, "../../..");

async function prepare(p: Paper): Promise<string> {
  const dir = join(ROOT, ".mlh", "complex-src", p.id);
  const env = { PATH: process.env.PATH ?? "/usr/bin:/bin", HOME: process.env.HOME ?? "/tmp", ...(process.env.HTTPS_PROXY ? { HTTPS_PROXY: process.env.HTTPS_PROXY } : {}), GIT_TERMINAL_PROMPT: "0" };
  // A cached clone counts only if it is at the pinned commit.
  if (existsSync(join(dir, ".git"))) {
    const head = await runProc(["git", "rev-parse", "HEAD"], { cwd: dir, env, timeoutMs: 30_000 });
    if (head.stdout.trim().startsWith(p.commit)) return dir;
    rmSync(dir, { recursive: true, force: true });
  }
  const tmp = `${dir}.tmp`;
  rmSync(tmp, { recursive: true, force: true });
  let r = await runProc(["git", "clone", "--quiet", p.repo, tmp], { cwd: ROOT, env, timeoutMs: 600_000 });
  if (r.code !== 0) throw new Error(`clone ${p.repo}: ${r.stderr.slice(-300)}`);
  r = await runProc(["git", "checkout", "--quiet", p.commit], { cwd: tmp, env, timeoutMs: 120_000 });
  if (r.code !== 0) throw new Error(`checkout ${p.commit}: ${r.stderr.slice(-300)}`);
  renameSync(tmp, dir);
  return dir;
}

interface Ev { t: string; at: string; id?: string; node?: { id: string; skill: string; params: Record<string, unknown> } }
const events = (runDir: string): Ev[] => (existsSync(join(runDir, "journal.jsonl")) ? readFileSync(join(runDir, "journal.jsonl"), "utf8").split("\n").filter(Boolean).flatMap((l) => { try { return [JSON.parse(l) as Ev]; } catch { return []; } }) : []);

// How long a run may go without a journal event: long GPU stages emit nothing while they run.
export function allowedSilenceMs(runDir: string): number {
  const ev = events(runDir);
  const started = new Set(ev.filter((e) => e.t === "start").map((e) => e.id));
  for (const e of ev) if (e.t === "ok" || e.t === "fail") started.delete(e.id);
  // Agent nodes write no journal events while they work (their checkpoints count as progress, see
  // lastProgressMs); a single remote command may take an hour plus an image build.
  let max = 2 * 3600_000;
  const plan = ev.find((e) => e.t === "ok" && e.id === "repro.plan") as { output?: { gpuPlan?: { stages: { id: string; timeoutS: number }[] } } } | undefined;
  for (const id of started) {
    const st = plan?.output?.gpuPlan?.stages.find((s) => id?.startsWith(`repro.stage:${s.id}:`));
    if (st) max = Math.max(max, (st.timeoutS + 2400 + 600) * 1000); // modalRun itself waits timeout + 2400 s
    if (id === "repro.fetch") max = Math.max(max, 3.5 * 3600_000);
  }
  return max;
}

// Progress: the newest write to the journal or to any agent checkpoint/transcript under work/.
export function lastProgressMs(runDir: string): number {
  let t = existsSync(join(runDir, "journal.jsonl")) ? statSync(join(runDir, "journal.jsonl")).mtimeMs : 0;
  const walk = (d: string, depth: number) => {
    if (depth > 4 || !existsSync(d)) return;
    for (const e of readdirSync(d, { withFileTypes: true })) {
      const p = join(d, e.name);
      if (e.isDirectory() && e.name !== "ws") walk(p, depth + 1);
      else if (e.name === "state.json" || e.name === "transcript.jsonl") t = Math.max(t, statSync(p).mtimeMs);
    }
  };
  walk(join(runDir, "work"), 0);
  return t;
}

function runChild(args: string[], runDir: string, log: string): Promise<{ code: number | null; killedForStall: boolean }> {
  const child = startProc(["npx", "tsx", join(ROOT, "src/mlh/cli.ts"), ...args], { cwd: ROOT, env: process.env, onOutput: (d) => appendFileSync(log, d) });
  let killed = false;
  const timer = setInterval(() => {
    const last = lastProgressMs(runDir);
    const age = last ? Date.now() - last : 0;
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

// The hillclimb metric: a run counts as a pipeline success when every claim got a
// definitive verdict (reproduced or not) and the run was not stopped. Plus median wall time.
export function pipelineMetric(rows: Record<string, unknown>[]): { success: number; n: number; medianMin: number } {
  const ok = rows.filter((r) => { const c = (r.claims as { status: string }[]) ?? []; return !r.stopped && c.length > 0 && c.every((x) => x.status === "reproduced" || x.status === "not_reproduced"); }).length;
  const m = rows.map((r) => Number(r.wallMin)).filter(Number.isFinite).sort((a, b) => a - b);
  return { success: ok, n: rows.length, medianMin: m.length ? m[Math.floor((m.length - 1) / 2)]! : 0 };
}

function render(rows: Record<string, unknown>[], extra: string[] = []): string {
  const cell = (r: Record<string, unknown>) => ((r.claims as { id: string; paper: number; status: string; mean?: number }[]) ?? []).map((c) => `${c.status === "reproduced" ? "✔" : c.status === "not_reproduced" ? "≠" : "✘"} ${c.id}: ${c.mean?.toFixed(3) ?? "—"} vs ${c.paper}`).join("<br>") || "—";
  const claims = rows.flatMap((r) => (r.claims as { status: string }[]) ?? []);
  return [
    `# Campaign summary (${new Date().toISOString()})`, "",
    `papers ${rows.length}; claims ${claims.length}: reproduced ${claims.filter((c) => c.status === "reproduced").length}, not reproduced ${claims.filter((c) => c.status === "not_reproduced").length}, not runnable ${claims.filter((c) => c.status === "not_runnable" || c.status === "pending").length}; GPU $${rows.reduce((a, r) => a + Number(r.gpuUsd), 0).toFixed(2)}; LLM $${rows.reduce((a, r) => a + Number(r.llmUsd), 0).toFixed(2)} (${rows.reduce((a, r) => a + Number(r.llmTokens), 0)} tokens); resumes ${rows.reduce((a, r) => a + Number(r.resumes), 0)}`, "",
    "| paper | claims (mean vs paper) | fidelity | env smoke | GPU $ | LLM tokens | min | resumes | failures |", "|---|---|---|---|---|---|---|---|---|",
    ...rows.map((r) => `| ${r.paper} (${r.arxiv}) | ${cell(r)} | ${r.fidelity} | ${r.smoke === null ? "—" : r.smoke ? "pass" : "FAIL"} | ${r.gpuUsd} | ${r.llmTokens} | ${r.wallMin} | ${r.resumes} | ${Object.entries(r.failures as Record<string, number>).map(([k, v]) => `${k}×${v}`).join(", ") || "—"}${r.stopped ? ` STOPPED: ${r.stopped}` : ""} |`),
    "", ...rows.flatMap((r) => ((r.failedNodes as string[]) ?? []).length ? [`- ${r.paper}: ${(r.failedNodes as string[]).join("; ")}`] : []), "",
    ...(() => { const m = pipelineMetric(rows); return [`Pipeline success (every claim definitive, not stopped): ${m.success}/${m.n}; median ${m.medianMin} min/paper`, ""]; })(),
    ...extra,
  ].join("\n");
}

// Preflight: the LLM provider's balance, all currencies summed (CNY counted as is; the
// threshold is in the same unit). Unknown (network error, other provider) never blocks.
export async function llmBalance(): Promise<number | undefined> {
  const key = process.env.DEEPSEEK_API_KEY, base = process.env.MLH_LLM_BASE_URL ?? "https://api.deepseek.com";
  if (!key || !base.includes("deepseek")) return undefined;
  try {
    const j = (await (await fetch(`${base}/user/balance`, { headers: { Authorization: `Bearer ${key}` }, signal: AbortSignal.timeout(15_000) })).json()) as { balance_infos?: { total_balance: string }[] };
    return (j.balance_infos ?? []).reduce((a, b) => a + Number(b.total_balance), 0);
  } catch { return undefined; }
}

export async function runCampaign(o: { config: string; label: string; parallel: number; out: string; mode?: "team" | "solo" }): Promise<void> {
  const papers = JSON.parse(readFileSync(o.config, "utf8")) as Paper[];
  const base = join(ROOT, ".mlh", "campaign", o.label);
  mkdirSync(base, { recursive: true });
  mkdirSync(o.out, { recursive: true });
  const rowsFile = join(o.out, `${o.label}.jsonl`);
  const rows: Record<string, unknown>[] = existsSync(rowsFile) ? readFileSync(rowsFile, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)) : [];
  const done = new Set(rows.map((r) => r.paper));
  let stop = false;
  // Trust gates scale: while any agent role is unproven, a round stays small and serial.
  const trustFile = join(o.out, "trust.json");
  let trust: TrustStore = existsSync(trustFile) ? JSON.parse(readFileSync(trustFile, "utf8")) : emptyStore();
  const gate = o.mode === "solo" ? { maxPapers: Infinity, parallel: o.parallel, blockedBy: [] } : scaleGate(trust);
  const queue = papers.filter((p) => !done.has(p.id)).slice(0, gate.maxPapers);
  const gateNote = gate.blockedBy.length ? `scale gated: roles on probation (${gate.blockedBy.join(", ")}) -> at most ${gate.maxPapers} papers, ${gate.parallel} at a time` : "all roles trusted: full scale";
  await Promise.all(Array.from({ length: Math.min(o.parallel, gate.parallel, queue.length) }, async () => {
    for (let p = queue.shift(); p && !stop; p = queue.shift()) {
      const bal = o.mode === "solo" ? undefined : await llmBalance(), min = Number(process.env.MLH_MIN_LLM_BALANCE ?? 3);
      if (bal !== undefined && bal < min) { stop = true; appendFileSync(join(o.out, `${o.label}.preflight.log`), `${new Date().toISOString()} stopped before ${p.id}: LLM balance ${bal} < ${min}\n`); break; }
      const runDir = join(base, p.id), log = join(base, `${p.id}.log`);
      const t0 = Date.now();
      let resumes = 0;
      try {
        const src = await prepare(p);
        // The solo baseline has no journal to resume from: one attempt, as a single agent would get.
        if (o.mode === "solo") { await runChild(["solo", src, "--arxiv", p.arxiv, "--objective", p.objective, "--gpu-budget", String(p.budgetUsd ?? 5), "--run-dir", runDir], runDir, log); }
        let r = o.mode === "solo" ? { code: 0, killedForStall: false } : await runChild(existsSync(join(runDir, "journal.jsonl")) ? ["resume", runDir, "--llm"] : ["repro", src, "--arxiv", p.arxiv, "--objective", p.objective, "--llm", "--gpu-budget", String(p.budgetUsd ?? 5), "--run-dir", runDir], runDir, log);
        // Killed for a stall or crashed: resume from the journal (at most twice).
        while (o.mode !== "solo" && (r.killedForStall || r.code !== 0) && !existsSync(join(runDir, "stopped.json")) && resumes < 2) {
          resumes++;
          r = await runChild(["resume", runDir, "--llm"], runDir, log);
        }
      } catch (e) {
        appendFileSync(log, `\n[campaign] ${String(e)}\n`);
      }
      const row = summarize(p, runDir, resumes, (Date.now() - t0) / 1000);
      if (row.stopped === "provider_fatal") {
        // No final row: after the balance is restored, a relaunch resumes this paper from its journal.
        stop = true;
        writeFileSync(join(o.out, `${o.label}.md`), render([...rows, row]));
        continue;
      }
      rows.push(row);
      appendFileSync(rowsFile, JSON.stringify(row) + "\n");
      if (o.mode !== "solo") {
        trust = record(trust, p.id, new Date().toISOString(), roleOutcomes(events(runDir) as never));
        writeFileSync(trustFile, JSON.stringify(trust, null, 1));
        writeFileSync(join(o.out, "trust.md"), renderTrust(trust));
      }
      writeFileSync(join(o.out, `${o.label}.md`), render(rows, [gateNote, "", renderTrust(trust)]));
    }
  }));
  writeFileSync(join(o.out, `${o.label}.md`), render(rows, [gateNote, "", renderTrust(trust)]));
  // Marker for chained rounds: "finished" must not be confused with "not started yet".
  writeFileSync(join(base, "done"), new Date().toISOString());
}

// One line per paper: finished node kinds, nodes running now, failures, GPU spent, minutes since progress.
export function campaignStatus(label: string): string {
  const base = join(ROOT, ".mlh", "campaign", label);
  if (!existsSync(base)) return `no campaign ${label}`;
  const lines: string[] = [];
  for (const d of readdirSync(base, { withFileTypes: true }).filter((e) => e.isDirectory() && !e.name.includes(".")).map((e) => e.name).sort()) {
    const runDir = join(base, d);
    const ev = events(runDir);
    const done = [...new Set(ev.filter((e) => e.t === "ok").map((e) => (e.id ?? "").replace(/^repro\./, "").replace(/:.*$/, "")))];
    const open = new Set(ev.filter((e) => e.t === "start").map((e) => e.id));
    for (const e of ev) if (e.t === "ok" || e.t === "fail" || e.t === "hold") open.delete(e.id);
    const spent = existsSync(join(runDir, "budget.json")) ? (JSON.parse(readFileSync(join(runDir, "budget.json"), "utf8")) as { spent: number }).spent : 0;
    const idle = Math.round((Date.now() - lastProgressMs(runDir)) / 60000);
    lines.push(`${d.padEnd(15)} ok[${done.join(",")}] running[${[...open].map((x) => (x ?? "").replace(/^repro\./, "")).join(",")}] fails=${ev.filter((e) => e.t === "fail").length} gpu=$${spent.toFixed(2)} idle=${idle}m${ev.some((e) => e.t === "done") ? " DONE" : ""}`);
  }
  return lines.join("\n");
}
