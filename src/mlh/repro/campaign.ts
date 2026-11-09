// Autonomous campaign: reproduce a queue of papers without a human watching.
// Each paper runs as a child process (`mlh repro`), so a watchdog can kill a run
// that stopped making progress and resume it from its journal. A fatal provider
// error (no balance, bad key) stops the whole campaign. After every paper the
// summary table is rewritten, so the state is always readable in one file.
import { appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { runProc, startProc } from "../io/proc.ts";
import { costReport, renderCost, type CostReport } from "./accounting.ts";
import { doctorFresh, renderDoctor, runDoctor } from "./doctor.ts";
import { pipelineMetric, renderRetro, roundOf, type Round } from "./retro.ts";
import { emptyStore, record, renderTrust, roleOutcomes, scaleGate, type TrustStore } from "./trustscore.ts";

export interface Paper { id: string; arxiv: string; repo: string; commit: string; objective: string; budgetUsd?: number }

// CODE: this checkout; ROOT: where run state lives (.mlh/...), MLH_ROOT for hermetic tests.
const CODE = resolve(import.meta.dirname, "../../..");
const ROOT = process.env.MLH_ROOT ? resolve(process.env.MLH_ROOT) : CODE;

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
  let r = await runProc(["git", "clone", "--quiet", p.repo, tmp], { cwd: CODE, env, timeoutMs: 600_000 });
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
  // 30 min: agents checkpoint every step (each command is capped at 20 min); a GPU stage gets its timeout below.
  let max = 30 * 60_000;
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
  // MLH_CAMPAIGN_CHILD replaces the child command (tests run a scripted team instead of the CLI).
  const cmd = process.env.MLH_CAMPAIGN_CHILD ? process.env.MLH_CAMPAIGN_CHILD.split(" ") : ["npx", "tsx", join(CODE, "src/mlh/cli.ts")];
  const child = startProc([...cmd, ...args], { cwd: CODE, env: process.env, onOutput: (d) => appendFileSync(log, d) });
  const started = Date.now();
  let killed = false;
  const timer = setInterval(() => {
    // A child that hangs before its first journal write is measured from its start, not exempt.
    const last = Math.max(lastProgressMs(runDir), started);
    const age = Date.now() - last;
    if (age > (process.env.MLH_STALL_MS ? Number(process.env.MLH_STALL_MS) : allowedSilenceMs(runDir))) {
      killed = true;
      appendFileSync(log, `\n[watchdog] no progress for ${Math.round(age / 60000)} min: killing for resume\n`);
      child.kill();
    }
  }, Number(process.env.MLH_WATCHDOG_MS ?? 60_000));
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
// Modal spend across every run this checkout has made (each run keeps its own budget.json).
// A run directory copied elsewhere (a run moved to another round) shares its run.json: counted once,
// at the larger of the copies' spend (the copy that continued includes what the original spent).
export function modalSpent(root = join(ROOT, ".mlh", "campaign")): number {
  const byRun = new Map<string, number>();
  if (!existsSync(root)) return 0;
  for (const label of readdirSync(root)) for (const id of statSync(join(root, label)).isDirectory() ? readdirSync(join(root, label)) : []) {
    const dir = join(root, label, id), f = join(dir, "budget.json");
    if (!existsSync(f)) continue;
    let spent = 0;
    try { spent = Number(JSON.parse(readFileSync(f, "utf8")).spent) || 0; } catch { continue; /* torn: the run's own .bak recovers it */ }
    const key = existsSync(join(dir, "run.json")) ? readFileSync(join(dir, "run.json"), "utf8") : dir;
    byRun.set(key, Math.max(byRun.get(key) ?? 0, spent));
  }
  return [...byRun.values()].reduce((a, b) => a + b, 0);
}

export async function llmBalance(): Promise<number | undefined> {
  const key = process.env.DEEPSEEK_API_KEY, base = process.env.MLH_LLM_BASE_URL ?? "https://api.deepseek.com";
  if (!key || !base.includes("deepseek")) return undefined;
  try {
    const j = (await (await fetch(`${base}/user/balance`, { headers: { Authorization: `Bearer ${key}` }, signal: AbortSignal.timeout(15_000) })).json()) as { balance_infos?: { total_balance: string }[] };
    return (j.balance_infos ?? []).reduce((a, b) => a + Number(b.total_balance), 0);
  } catch { return undefined; }
}

const numEnv = (k: string, d: number) => { const v = Number(process.env[k]); return process.env[k] !== undefined && Number.isFinite(v) ? v : d; };

function writeAtomic(file: string, body: string) {
  if (existsSync(file)) writeFileSync(`${file}.bak`, readFileSync(file));
  writeFileSync(`${file}.tmp`, body);
  renameSync(`${file}.tmp`, file);
}

// One run's cost by bucket. Runs from before per-attempt Modal accounting have only a total:
// it is reported as unattributed, not guessed.
export function costOfRun(runDir: string): CostReport {
  const rows = existsSync(join(runDir, "trace.jsonl")) ? readFileSync(join(runDir, "trace.jsonl"), "utf8").split("\n").filter(Boolean).flatMap((l) => { try { return [JSON.parse(l)]; } catch { return []; } }) : [];
  let byAttempt: Record<string, number> = {};
  if (existsSync(join(runDir, "budget.json"))) {
    const b = JSON.parse(readFileSync(join(runDir, "budget.json"), "utf8")) as { spent?: number; byAttempt?: Record<string, number> };
    const attributed = Object.values(b.byAttempt ?? {}).reduce((a, x) => a + x, 0);
    byAttempt = { ...(b.byAttempt ?? {}), ...((b.spent ?? 0) - attributed > 1e-6 ? { "(before per-attempt accounting)": (b.spent ?? 0) - attributed } : {}) };
  }
  const messages: Record<string, string> = {};
  const attempts: Record<string, number> = {};
  for (const e of events(runDir) as (Ev & { message?: string; attempt?: number })[]) {
    if (e.t === "start" && e.id) attempts[e.id] = e.attempt ?? 1;
    if (e.t === "fail" && e.id) messages[`${e.id}#${attempts[e.id] ?? 1}`] = e.message ?? "";
  }
  const llm: Record<string, number> = {};
  if (existsSync(join(runDir, "llm-calls.jsonl"))) for (const l of readFileSync(join(runDir, "llm-calls.jsonl"), "utf8").split("\n").filter(Boolean)) { try { const c = JSON.parse(l) as { key: string; costUsd: number }; llm[c.key] = (llm[c.key] ?? 0) + (c.costUsd || 0); } catch { /* torn last line */ } }
  return costReport(rows, byAttempt, messages, llm);
}

export function costOfRound(label: string): string {
  const base = join(ROOT, ".mlh", "campaign", label);
  const ids = existsSync(base) ? readdirSync(base).filter((d) => statSync(join(base, d)).isDirectory()) : [];
  return ids.map((id) => renderCost(`${label}/${id}`, costOfRun(join(base, id)))).join("\n");
}

// Trust is derived, like the DAG state: a fold over every finished team run's journal in time
// order. Rebuilding it makes concurrent rounds, crashes and rule changes unable to corrupt it.
export function rebuildTrust(out: string, campaignRoot = join(ROOT, ".mlh", "campaign")): TrustStore {
  const runs: { key: string; at: string; dir: string }[] = [];
  for (const f of existsSync(out) ? readdirSync(out) : []) {
    const label = f.replace(/\.jsonl$/, "");
    if (!f.endsWith(".jsonl") || label === "rounds" || label.includes("solo")) continue;
    for (const l of readFileSync(join(out, f), "utf8").split("\n").filter(Boolean)) {
      let r: { paper?: string; stopped?: unknown };
      try { r = JSON.parse(l); } catch { continue; }
      if (!r.paper || r.stopped) continue;
      const dir = join(campaignRoot, label, r.paper), ev = events(dir);
      if (ev.length) runs.push({ key: `${label}/${r.paper}`, at: ev.at(-1)!.at, dir });
    }
  }
  runs.sort((a, b) => a.at.localeCompare(b.at));
  return runs.reduce((t, r) => record(t, r.key, r.at, roleOutcomes(events(r.dir) as never)), emptyStore());
}

export function saveTrust(out: string, t: TrustStore) {
  writeAtomic(join(out, "trust.json"), JSON.stringify(t, null, 1));
  writeFileSync(join(out, "trust.md"), renderTrust(t));
}

// One round's retro, idempotent: computed from the round's rows if not yet in rounds.jsonl.
export function retroRound(label: string, out: string, solo = label.includes("solo")): boolean {
  const roundsFile = join(out, "rounds.jsonl"), rowsFile = join(out, `${label}.jsonl`);
  const history: Round[] = existsSync(roundsFile) ? readFileSync(roundsFile, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)) : [];
  if (history.some((h) => h.label === label) || !existsSync(rowsFile)) return false;
  const rows = readFileSync(rowsFile, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
  const same = history.filter((h) => h.label.includes("solo") === solo);
  const trust = rebuildTrust(out);
  if (!solo) saveTrust(out, trust);
  const round = roundOf(label, new Date().toISOString(), rows, trust);
  appendFileSync(roundsFile, JSON.stringify(round) + "\n");
  appendFileSync(join(out, "decisions.md"), "\n" + renderRetro(round, same.at(-1), solo ? [] : [...same, round]));
  return true;
}

export const sequentialWorkers = (requested: number, gated: number, queued: number) => Math.min(1, requested, gated, queued);

export async function runCampaign(o: { config: string; label: string; parallel: number; out: string; mode?: "team" | "solo" }): Promise<void> {
  const papers = JSON.parse(readFileSync(o.config, "utf8")) as Paper[];
  const base = join(ROOT, ".mlh", "campaign", o.label);
  mkdirSync(base, { recursive: true });
  mkdirSync(o.out, { recursive: true });
  // A relaunch is a new attempt at this round: an old marker must not release a waiting chain.
  rmSync(join(base, "done"), { force: true });
  rmSync(join(base, "stopped"), { force: true });
  // Prove the platform before spending a paper on it: a fresh passing probe, or run one now.
  const doc = doctorFresh()?.ok ? doctorFresh()! : await runDoctor({ llmBalance, minLlmBalance: numEnv("MLH_MIN_LLM_BALANCE", 3) });
  appendFileSync(join(o.out, `${o.label}.preflight.log`), renderDoctor(doc) + "\n");
  if (!doc.ok) { writeFileSync(join(base, "stopped"), `doctor failed ${doc.at}`); return; }
  const rowsFile = join(o.out, `${o.label}.jsonl`);
  const rows: Record<string, unknown>[] = existsSync(rowsFile) ? readFileSync(rowsFile, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)) : [];
  const done = new Set(rows.map((r) => r.paper));
  let stop = false;
  // Trust gates scale: while any agent role is unproven, a round stays small and serial.
  let trust: TrustStore = rebuildTrust(o.out);
  saveTrust(o.out, trust);
  const gate = o.mode === "solo" ? { maxPapers: Infinity, parallel: o.parallel, blockedBy: [] } : scaleGate(trust);
  // Papers with the least trust evidence first, so a gated round is not the same three papers every time.
  const seen = (id: string) => Object.values(trust).reduce((a, t) => a + t.history.filter((h) => h.paper.endsWith(`/${id}`) || h.paper === id).length, 0);
  const queue = papers.filter((p) => !done.has(p.id)).map((p, i) => ({ p, i, n: o.mode === "solo" ? 0 : seen(p.id) })).sort((a, b) => a.n - b.n || a.i - b.i).map((x) => x.p).slice(0, gate.maxPapers);
  const gateNote = gate.blockedBy.length ? `scale gated: roles on probation (${gate.blockedBy.join(", ")}) -> at most ${gate.maxPapers} papers, ${gate.parallel} at a time` : "all roles trusted: full scale";
  // Strictly sequential until reliability is proven (docs/mlh/RELIABILITY.md, P6): one paper at a
  // time, whatever --parallel or the trust gate would allow.
  const workers = sequentialWorkers(o.parallel, gate.parallel, queue.length);
  await Promise.all(Array.from({ length: workers }, async () => {
    for (let p = queue.shift(); p && !stop; p = queue.shift()) {
      const bal = o.mode === "solo" ? undefined : await llmBalance(), min = numEnv("MLH_MIN_LLM_BALANCE", 3);
      if (bal !== undefined && bal < min) { stop = true; appendFileSync(join(o.out, `${o.label}.preflight.log`), `${new Date().toISOString()} stopped before ${p.id}: LLM balance ${bal} < ${min}\n`); break; }
      // Modal has no balance API we rely on: the owner states the limit (MLH_MODAL_LIMIT_USD), the ledger is ours.
      const modalLimit = numEnv("MLH_MODAL_LIMIT_USD", Infinity), spentAll = modalSpent();
      if (spentAll + (p.budgetUsd ?? 5) > modalLimit) { stop = true; appendFileSync(join(o.out, `${o.label}.preflight.log`), `${new Date().toISOString()} stopped before ${p.id}: Modal spend $${spentAll.toFixed(2)} + paper budget $${p.budgetUsd ?? 5} > limit $${modalLimit}\n`); break; }
      const runDir = join(base, p.id), log = join(base, `${p.id}.log`);
      const t0 = Date.now();
      let resumes = 0;
      try {
        const src = await prepare(p);
        // The solo baseline has no journal to resume from: one attempt, as a single agent would get.
        if (o.mode === "solo") { await runChild(["solo", src, "--arxiv", p.arxiv, "--objective", p.objective, "--gpu-budget", String(p.budgetUsd ?? 5), "--run-dir", runDir], runDir, log); }
        // A child that died before its journal existed is started again, not "resumed" (there is nothing to resume).
        const launch = () => (existsSync(join(runDir, "journal.jsonl")) ? ["resume", runDir, "--llm"] : ["repro", src, "--arxiv", p.arxiv, "--objective", p.objective, "--llm", "--gpu-budget", String(p.budgetUsd ?? 5), "--run-dir", runDir]);
        let r = o.mode === "solo" ? { code: 0, killedForStall: false } : await runChild(launch(), runDir, log);
        // Killed for a stall or crashed: resume from the journal (at most twice).
        while (o.mode !== "solo" && (r.killedForStall || r.code !== 0) && !existsSync(join(runDir, "stopped.json")) && resumes < 2) {
          resumes++;
          r = await runChild(launch(), runDir, log);
        }
      } catch (e) {
        appendFileSync(log, `\n[campaign] ${String(e)}\n`);
      }
      const row = summarize(p, runDir, resumes, (Date.now() - t0) / 1000);
      if (row.stopped && row.stopped !== "provider_fatal") {
        // A strict stop (repeated infrastructure fault, unclassified failure, waste, LLM cap) stops the
        // round: investigate before anything else is spent. No result row: a relaunch after the fix
        // resumes the paper from its journal (a row would mark it done; FT-Transformer r2c).
        stop = true;
        writeFileSync(join(o.out, `${o.label}.md`), render([...rows, row], [`ROUND STOPPED at ${p.id}: ${row.stopped}`]));
        continue;
      }
      if (row.stopped === "provider_fatal") {
        // No final row: after the balance is restored, a relaunch resumes this paper from its journal.
        stop = true;
        writeFileSync(join(o.out, `${o.label}.md`), render([...rows, row]));
        continue;
      }
      // Only papers that ran to the end are evidence (rebuildTrust skips stopped rows).
      rows.push(row);
      appendFileSync(rowsFile, JSON.stringify(row) + "\n");
      if (o.mode !== "solo") { trust = rebuildTrust(o.out); saveTrust(o.out, trust); }
      writeFileSync(join(o.out, `${o.label}.md`), render(rows, [gateNote, "", renderTrust(trust)]));
    }
  }));
  writeFileSync(join(o.out, `${o.label}.md`), render(rows, [gateNote, "", renderTrust(trust)]));
  // The round reviews itself: metric, comparison, failure ranking, stop predicate -> decision log.
  if (!stop) retroRound(o.label, o.out, o.mode === "solo");
  // Marker for chained rounds: "finished" must not be confused with "not started yet".
  // A stopped round (balance, limit) writes "stopped" instead: the chain must not start the next round into it.
  writeFileSync(join(base, stop ? "stopped" : "done"), new Date().toISOString());
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
