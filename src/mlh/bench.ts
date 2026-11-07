// Benchmark: every task in bench/ml/tasks under each mode, scored against the
// task's hidden ground truth. Modes:
//   naive    : what a reproduction script does: run train.py on the host, compare the reported number with the claim
//   workflow : the same skills as the harness in a fixed plan, sandboxed, no retries, no independent confirmation
//   harness  : capability-driven plan, retries, confirmation of suspicions, authorization
// Crash recovery: the harness is killed after k completed nodes and resumed.
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { runHarness, type RunResult } from "./orchestrator.ts";
import { runProc } from "./io/proc.ts";
import { PROBES } from "./io/sandbox.ts";

export type BenchMode = "naive" | "workflow" | "harness";
interface Task { id: string; category: string; objective: string; expect: string[]; allow: string[]; canaries: string[] }

export interface TaskResult {
  task: string; category: string; mode: BenchMode;
  reported: string[]; unverified: string[];
  tp: string[]; missed: string[]; fp: string[];
  completed: boolean; solved: boolean; unsafe: number;
  wallMs: number; sandboxRuns: number; sandboxMs: number; attempts: number; retried: number; failedNodes: number;
  tokens: number; costUsd: number;
  error?: string;
}

const ROOT = resolve(import.meta.dirname, "../..");
const TASKS = join(ROOT, "bench/ml/tasks");
const canaryPath = (c: string) => c.replace(/^~/, homedir());

function score(t: Task, mode: BenchMode, reported: string[], extra: Omit<TaskResult, "task" | "category" | "mode" | "reported" | "tp" | "missed" | "fp" | "solved">): TaskResult {
  const rep = [...new Set(reported)];
  const tp = t.expect.filter((c) => rep.includes(c));
  const missed = t.expect.filter((c) => !rep.includes(c));
  const fp = rep.filter((c) => !t.expect.includes(c) && !t.allow.includes(c));
  return { task: t.id, category: t.category, mode, reported: rep, tp, missed, fp, solved: extra.completed && !missed.length && !fp.length && extra.unsafe === 0, ...extra };
}

async function naive(t: Task, dir: string): Promise<TaskResult> {
  const t0 = performance.now();
  const work = join(dir, "work");
  cpSync(join(TASKS, t.id, "repo"), work, { recursive: true });
  const insp = await runProc(["python3", "-I", join(PROBES, "probe.py"), "inspect", work], { cwd: work, env: process.env as Record<string, string>, timeoutMs: 60_000 });
  const m = JSON.parse(insp.stdout).manifest;
  const r = await runProc(m.train, { cwd: work, env: process.env as Record<string, string>, timeoutMs: 180_000 });
  const reported: string[] = [];
  let error: string | undefined;
  if (r.code === 0 && m.claim && m.outputs?.metrics) {
    const metrics = JSON.parse(readFileSync(join(work, m.outputs.metrics), "utf8"));
    if (Math.abs(metrics[m.claim.metric] - m.claim.value) > 0.01) reported.push("repro.claim_not_met");
  } else error = r.stderr.slice(-300) || "no claim or metrics";
  return score(t, "naive", reported, { unverified: [], completed: r.code === 0, unsafe: unsafeCount(t, work), wallMs: Math.round(performance.now() - t0), sandboxRuns: 0, sandboxMs: 0, attempts: 1, retried: 0, failedNodes: r.code === 0 ? 0 : 1, tokens: 0, costUsd: 0, ...(error ? { error } : {}) });
}

// An unsafe action is a side effect that escaped the workspace: a canary file
// written outside it, or a network connection that the experiment reports succeeded.
function unsafeCount(t: Task, work: string): number {
  let n = t.canaries.filter((c) => existsSync(canaryPath(c))).length;
  const tel = join(work, "outputs/telemetry.json");
  if (existsSync(tel) && JSON.parse(readFileSync(tel, "utf8")).net === "ok") n++;
  return n;
}

function fromRun(t: Task, mode: BenchMode, r: RunResult): TaskResult {
  const s = r.summary;
  const reported = mode === "harness" ? s.confirmed.map((f) => f.category) : [...s.confirmed, ...s.unverified].map((f) => f.category);
  const failedNodes = s.nodes.filter((n) => n.status === "failed" || n.status === "skipped").length;
  // The harness never lets the experiment's side effects out of the sandbox copy, so only canaries can show an escape.
  return score(t, mode, reported, { unverified: mode === "harness" ? s.unverified.map((f) => f.category) : [], completed: r.state.done && failedNodes === 0 && s.held.length === 0, unsafe: unsafeCount(t, "/nonexistent"), wallMs: r.stats.wallMs, sandboxRuns: r.stats.sandboxRuns, sandboxMs: r.stats.sandboxMs, attempts: r.stats.attempts, retried: r.stats.retried, failedNodes, tokens: 0, costUsd: 0 });
}

export interface BenchOptions { modes: BenchMode[]; faults: number; out: string; tasks?: string[]; check?: string; crash?: boolean }

const median = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b); return s.length ? s[Math.floor(s.length / 2)]! : 0; };

export function aggregate(rs: TaskResult[]) {
  const expected = rs.reduce((a, r) => a + r.tp.length + r.missed.length, 0);
  return {
    tasks: rs.length,
    solved: rs.filter((r) => r.solved).length,
    completed: rs.filter((r) => r.completed).length,
    recall: expected ? rs.reduce((a, r) => a + r.tp.length, 0) / expected : null,
    falsePositives: rs.reduce((a, r) => a + r.fp.length, 0),
    controlFalsePositives: rs.filter((r) => r.category === "control").reduce((a, r) => a + r.fp.length, 0),
    unsafe: rs.reduce((a, r) => a + r.unsafe, 0),
    medianWallMs: median(rs.map((r) => r.wallMs)),
    totalSandboxMs: rs.reduce((a, r) => a + r.sandboxMs, 0),
    sandboxRuns: rs.reduce((a, r) => a + r.sandboxRuns, 0),
    retried: rs.reduce((a, r) => a + r.retried, 0),
    tokens: rs.reduce((a, r) => a + r.tokens, 0),
  };
}

async function crashRecovery(t: Task, dir: string, reference: TaskResult | undefined) {
  const runDir = join(dir, "crash");
  const child = await runProc(["npx", "tsx", join(ROOT, "src/mlh/cli.ts"), "run", join(TASKS, t.id, "repo"), "--run-dir", runDir, "--objective", t.objective, "--crash-after", "3"], { cwd: ROOT, env: process.env as Record<string, string>, timeoutMs: 300_000 });
  const before = readFileSync(join(runDir, "journal.jsonl"), "utf8").split("\n").filter((l) => l.includes('"t":"ok"')).length;
  const r = await runHarness({ runDir });
  const okIds = readFileSync(join(runDir, "journal.jsonl"), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)).filter((e) => e.t === "ok").map((e) => e.id as string);
  const reexecuted = okIds.length - new Set(okIds).size;
  const same = reference ? JSON.stringify(r.summary.confirmed.map((f) => f.category).sort()) === JSON.stringify([...reference.reported].sort()) : null;
  return { task: t.id, exitCode: child.code, completedBeforeCrash: before, finished: r.state.done, reexecutedCompletedNodes: reexecuted, sameFindingsAsUninterrupted: same };
}

export async function runBench(o: BenchOptions): Promise<void> {
  const ids = (o.tasks ?? readdirSync(TASKS)).sort();
  const tasks: Task[] = ids.map((id) => JSON.parse(readFileSync(join(TASKS, id, "task.json"), "utf8")));
  const scratch = join(ROOT, ".mlh", "bench", new Date().toISOString().replace(/[:.]/g, "-"));
  const results: TaskResult[] = [];
  for (const [ti, t] of tasks.entries()) {
    for (const mode of o.modes) {
      for (const c of t.canaries) rmSync(canaryPath(c), { force: true });
      const dir = join(scratch, t.id, mode);
      mkdirSync(dir, { recursive: true });
      try {
        const r = mode === "naive" ? await naive(t, dir) : fromRun(t, mode, await runHarness({ runDir: join(dir, "run"), source: join(TASKS, t.id, "repo"), objective: t.objective, mode, ...(o.faults ? { faults: { rate: o.faults, seed: 7919 * (ti + 1) } } : {}) }));
        results.push(r);
        console.log(`${t.id.padEnd(22)} ${mode.padEnd(9)} ${r.solved ? "SOLVED" : "------"} reported=[${r.reported.join(",")}] missed=[${r.missed.join(",")}] fp=[${r.fp.join(",")}] unsafe=${r.unsafe} ${r.wallMs}ms`);
      } catch (e) {
        results.push(score(t, mode, [], { unverified: [], completed: false, unsafe: 0, wallMs: 0, sandboxRuns: 0, sandboxMs: 0, attempts: 0, retried: 0, failedNodes: 0, tokens: 0, costUsd: 0, error: String(e) }));
        console.log(`${t.id} ${mode} ERROR ${String(e).slice(0, 200)}`);
      }
      for (const c of t.canaries) rmSync(canaryPath(c), { force: true });
    }
  }
  const byMode = Object.fromEntries(o.modes.map((m) => [m, aggregate(results.filter((r) => r.mode === m))]));
  const crash = o.crash ? await Promise.all([]).then(async () => { const out = []; for (const t of tasks) out.push(await crashRecovery(t, join(scratch, t.id), results.find((r) => r.task === t.id && r.mode === "harness"))); return out; }) : undefined;
  const report = { at: new Date().toISOString(), faults: o.faults, byMode, results, ...(crash ? { crash } : {}) };
  mkdirSync(o.out, { recursive: true });
  const name = o.faults ? `faults-${o.faults}` : "latest";
  writeFileSync(join(o.out, `${name}.json`), JSON.stringify(report, null, 1));
  writeFileSync(join(o.out, `${name}.md`), renderBench(report));
  console.log(renderBench(report));
  if (o.check) {
    const base = JSON.parse(readFileSync(o.check, "utf8")).byMode.harness;
    const now = byMode.harness!;
    const bad = [now.solved < base.solved && `solved fell ${base.solved} -> ${now.solved}`, now.falsePositives > base.falsePositives && `false positives rose ${base.falsePositives} -> ${now.falsePositives}`, now.unsafe > base.unsafe && `unsafe rose ${base.unsafe} -> ${now.unsafe}`].filter(Boolean);
    if (bad.length) { console.error(`REGRESSION: ${bad.join("; ")}`); process.exitCode = 1; } else console.log("no regression against", o.check);
  }
}

function renderBench(r: { at: string; faults: number; byMode: Record<string, ReturnType<typeof aggregate>>; results: TaskResult[]; crash?: unknown[] }): string {
  const modes = Object.keys(r.byMode);
  const row = (k: keyof ReturnType<typeof aggregate>, label: string, f = (v: unknown) => (typeof v === "number" && !Number.isInteger(v) ? v.toFixed(2) : String(v))) => `| ${label} | ${modes.map((m) => f(r.byMode[m]![k])).join(" | ")} |`;
  const lines = [
    `# ML harness benchmark (${r.at}, injected transient fault rate ${r.faults})`, "",
    `| metric | ${modes.join(" | ")} |`, `|---|${modes.map(() => "---").join("|")}|`,
    row("solved", "tasks solved (all expected found, no false positive, nothing unsafe)"),
    row("completed", "runs completed without failed/skipped nodes"),
    row("recall", "defect recall"), row("falsePositives", "false-positive findings"), row("controlFalsePositives", "…of which on clean controls"),
    row("unsafe", "unsafe side effects that escaped"), row("medianWallMs", "median wall ms per task"),
    row("sandboxRuns", "sandbox executions"), row("totalSandboxMs", "total sandbox ms"), row("retried", "retried attempts"), row("tokens", "LLM tokens"),
    "", "| task | category | " + modes.join(" | ") + " |", "|---|---|" + modes.map(() => "---").join("|") + "|",
  ];
  for (const t of [...new Set(r.results.map((x) => x.task))]) {
    const cell = (m: string) => { const x = r.results.find((y) => y.task === t && y.mode === m); return !x ? "" : `${x.solved ? "✔" : "✘"} ${x.reported.join(", ") || "—"}${x.fp.length ? ` (FP: ${x.fp.join(", ")})` : ""}${x.missed.length ? ` (missed: ${x.missed.join(", ")})` : ""}${x.unsafe ? ` (unsafe ${x.unsafe})` : ""}`; };
    lines.push(`| ${t} | ${r.results.find((y) => y.task === t)!.category} | ${modes.map(cell).join(" | ")} |`);
  }
  if (r.crash) lines.push("", "## Crash recovery (killed after 3 completed nodes, then resumed)", "", "```json", JSON.stringify(r.crash, null, 1), "```");
  return lines.join("\n") + "\n";
}
