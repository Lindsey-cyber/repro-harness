// Executes a run: folds the journal into state, schedules ready nodes in
// parallel, authorizes each node's actions before it runs, retries what a
// fresh attempt can fix, replans on results, and writes the report. Every
// state change is journaled first, so a killed process resumes where it was.
import { appendFileSync, cpSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { authorize, type Action, type Evidence } from "./core/authz.ts";
import { apply, blocked, depOf, ready, replay, settled, type RunEvent, type RunState, type TaskNode } from "./core/dag.ts";
import { confirmations, expand, INSPECT, initialPlan } from "./core/planner.ts";
import { compact, renderReport, summarize, type Summary } from "./core/report.ts";
import { initialTrust, type SkillTrust } from "./core/trust.ts";
import type { Manifest } from "./core/types.ts";
import { ArtifactStore } from "./io/artifacts.ts";
import { Journal } from "./io/journal.ts";
import { runProc } from "./io/proc.ts";
import { DEFAULT_LIMITS, Sandbox, selfTest, SkillError, type Isolation, type Limits, type SandboxStats } from "./io/sandbox.ts";
import { skill, SKILLS, type SkillCtx } from "./skills/index.ts";

export type Mode = "harness" | "workflow";

export interface RunOptions {
  runDir: string;
  source?: string; // local path or https/file git URL; required for a new run
  objective?: string;
  mode?: Mode;
  concurrency?: number;
  limits?: Limits;
  faults?: { rate: number; seed: number }; // injected transient sandbox failures
  crashAfter?: number; // exit the process after this many completed nodes (recovery tests)
  trustFile?: string;
  isolation?: Isolation; // force; default is what the self-test proves
}

export interface RunResult {
  state: RunState;
  summary: Summary;
  reportPath: string;
  stats: { sandboxRuns: number; sandboxMs: number; wallMs: number; attempts: number; retried: number };
}

const now = () => new Date().toISOString();

// Deterministic per (node, attempt) so a benchmark with faults is reproducible.
// FNV-1a then the murmur3 finalizer: FNV alone leaves the high bits nearly
// unchanged between similar keys, which clustered the injected faults.
export function faultFor(rate: number, seed: number, key: string): boolean {
  let h = (2166136261 ^ seed) >>> 0;
  for (const ch of key) h = Math.imul(h ^ ch.charCodeAt(0), 16777619) >>> 0;
  h ^= h >>> 16; h = Math.imul(h, 0x85ebca6b) >>> 0; h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35) >>> 0; h = (h ^ (h >>> 16)) >>> 0;
  return h / 2 ** 32 < rate;
}

async function fetchSource(src: string, dest: string): Promise<{ evidence: Evidence; rev: string }> {
  if (/^(https:\/\/github\.com\/|file:\/\/)/.test(src)) {
    const d = authorize("fetch_source", [{ kind: "deterministic", check: "source url is https://github.com or file://", ok: true }]);
    if (!d.allow) throw new Error(d.reason);
    const r = await runProc(["git", "clone", "--depth", "1", "--quiet", src, dest], { cwd: process.cwd(), env: { PATH: process.env.PATH ?? "/usr/bin:/bin", HOME: process.env.HOME ?? "/tmp", ...(process.env.HTTPS_PROXY ? { HTTPS_PROXY: process.env.HTTPS_PROXY } : {}), GIT_TERMINAL_PROMPT: "0" }, timeoutMs: 120_000 });
    if (r.code !== 0) throw new Error(`git clone failed: ${r.stderr}`);
    const rev = await runProc(["git", "rev-parse", "HEAD"], { cwd: dest, env: { PATH: process.env.PATH ?? "/usr/bin:/bin" }, timeoutMs: 10_000 });
    return { evidence: { kind: "deterministic", check: "clone", ok: true }, rev: rev.stdout.trim() };
  }
  if (/^[a-z]+:\/\//.test(src)) throw new Error(`refusing source ${src}: only https://github.com/… , file:// or a local path`);
  if (!existsSync(src)) throw new Error(`source ${src} does not exist`);
  cpSync(resolve(src), dest, { recursive: true, filter: (p) => !/\/(\.git|outputs|__pycache__)(\/|$)/.test(p) });
  return { evidence: { kind: "deterministic", check: "local copy", ok: true }, rev: "local" };
}

function loadTrust(file: string | undefined): (s: string) => SkillTrust["level"] {
  const t: Record<string, SkillTrust> = file && existsSync(file) ? JSON.parse(readFileSync(file, "utf8")) : {};
  return (s) => (t[s] ?? initialTrust(now())).level;
}

export async function runHarness(o: RunOptions): Promise<RunResult> {
  const t0 = performance.now();
  const runDir = resolve(o.runDir);
  const journal = new Journal(join(runDir, "journal.jsonl"));
  const artifacts = new ArtifactStore(join(runDir, "artifacts"));
  const srcDir = join(runDir, "source");
  const tracePath = join(runDir, "trace.jsonl");
  let events = journal.read();
  const mode: Mode = (events.find((e) => e.t === "run") as { mode?: Mode } | undefined)?.mode ?? o.mode ?? "harness";
  const maxAttempts = mode === "harness" ? 3 : 1;
  const state = replay(events, maxAttempts);
  const emit = (e: RunEvent) => { journal.append(e); apply(state, e); };

  if (!events.length) {
    if (!o.source || !o.objective) throw new Error("a new run needs a source and an objective");
    mkdirSync(runDir, { recursive: true });
    const { rev } = await fetchSource(o.source, srcDir);
    writeFileSync(join(runDir, "run.json"), JSON.stringify({ source: o.source, rev, objective: o.objective, mode, created: now() }, null, 1));
    emit({ t: "run", at: now(), objective: o.objective, source: o.source, mode });
    for (const n of initialPlan()) emit({ t: "node", at: now(), node: n, why: "initial plan" });
  } else if (!existsSync(srcDir)) throw new Error(`cannot resume ${runDir}: source copy missing`);
  events = [];

  const st = o.isolation ? { isolation: o.isolation, checks: [{ name: "forced by caller", ok: o.isolation === "namespace" }] } : await selfTest(join(runDir, "sbx"));
  const isolationEvidence: Evidence = { kind: "deterministic", check: `sandbox.selftest(${st.checks.map((c) => `${c.name}=${c.ok}`).join(",")})`, ok: st.isolation === "namespace" };
  const trust = loadTrust(o.trustFile);
  const stats: SandboxStats = { runs: 0, ms: 0 };
  const limits = o.limits ?? DEFAULT_LIMITS;
  const concurrency = o.concurrency ?? 3;
  let completedHere = 0, retried = 0, attemptsHere = 0;
  const manifest = (): Manifest => ((state.nodes.get(INSPECT)?.output as { manifest?: Manifest } | undefined)?.manifest ?? {});

  const afterOk = (n: TaskNode, findings: number) => {
    if (n.id === INSPECT) {
      const x = expand(state.objective, manifest(), SKILLS);
      for (const node of x.nodes) emit({ t: "node", at: now(), node, why: "expanded from manifest" });
      for (const g of x.gaps) emit({ t: "gap", at: now(), skill: g.skill, reason: g.reason });
    }
    if (mode !== "harness" || n.purpose === "confirm") return;
    for (let i = 0; i < findings; i++) {
      const fid = `${n.id}/${i}`;
      for (const node of confirmations(state, fid, SKILLS, manifest(), trust)) {
        if (!state.nodes.has(node.id)) emit({ t: "node", at: now(), node, why: `confirm ${fid}` });
      }
    }
  };

  const execute = async (n: TaskNode): Promise<void> => {
    const ns = state.nodes.get(n.id)!;
    const sk = skill(n.skill);
    // Authorization first: sandboxed execution is only sandboxed if the self-test proved it.
    const evidence: Evidence[] = [isolationEvidence, ...(ns.approved ? [{ kind: "human" as const, check: `approval:${n.id}`, ok: true }] : [])];
    for (const a of sk.actions) {
      const action: Action = a === "exec_sandboxed" && st.isolation !== "namespace" ? "exec_unsandboxed" : a;
      const d = authorize(action, evidence);
      if (!d.allow) {
        if (d.hold) emit({ t: "hold", at: now(), id: n.id, reason: d.reason });
        else emit({ t: "fail", at: now(), id: n.id, category: "policy_denied", message: d.reason });
        return;
      }
    }
    const attempt = ns.attempts + 1;
    emit({ t: "start", at: now(), id: n.id, attempt });
    attemptsHere++;
    const started = performance.now(), before = { ...stats };
    const saved: { name: string; sha256: string; bytes: number }[] = [];
    let label = 0;
    const ctx: SkillCtx = {
      node: n, state, manifest: manifest(),
      target: n.target ? state.findings.get(n.target) : undefined,
      dep: (s) => {
        const d = depOf(state, n, s);
        if (!d || d.status !== "succeeded") throw new SkillError("precondition", `${n.id} needs a succeeded ${s}`);
        return d;
      },
      sandbox: (l) => {
        const name = `${n.id}-${attempt}-${l ?? label++}`;
        let call = 0;
        return Sandbox.create(join(runDir, "sbx"), name, srcDir, {
          isolation: st.isolation, limits, stats,
          ...(o.faults ? { fault: () => faultFor(o.faults!.rate, o.faults!.seed, `${name}#${call++}`) } : {}),
        });
      },
      save: (name, file) => { const r = artifacts.put(name, file); saved.push(r); return r; },
      load: (ref) => artifacts.path(ref),
    };
    let outcome: string, category: string | undefined;
    try {
      const out = await sk.run(ctx);
      const parsed = sk.output.safeParse(out.output);
      if (!parsed.success) throw new SkillError("contract_violation", `${sk.id}@${sk.version} output violates its schema: ${parsed.error.message.slice(0, 300)}`);
      const findings = out.findings ?? [];
      let confirmation = out.confirmation && n.target ? { ...out.confirmation, target: n.target } : undefined;
      if (!confirmation && n.target && typeof n.params.category === "string") {
        const again = findings.filter((f) => f.category === n.params.category);
        confirmation = { target: n.target, supports: again.length > 0, note: again.length ? `replicated on a fresh run: ${again[0]!.summary}` : `not reproduced on a fresh run` };
      }
      emit({ t: "ok", at: now(), id: n.id, output: out.output, findings, artifacts: saved, ...(confirmation ? { confirmation } : {}) });
      outcome = "ok";
      afterOk(n, findings.length);
    } catch (e) {
      const cat = e instanceof SkillError ? e.category : "harness_bug";
      emit({ t: "fail", at: now(), id: n.id, category: cat, message: String((e as Error).message ?? e).slice(0, 2000) });
      outcome = "fail"; category = cat;
      if (state.nodes.get(n.id)!.status === "pending") {
        retried++;
        await new Promise((r) => setTimeout(r, 50 * 2 ** attempt));
      }
    }
    appendFileSync(tracePath, JSON.stringify({ at: now(), id: n.id, skill: sk.id, version: sk.version, attempt, outcome, category, ms: Math.round(performance.now() - started), sandboxRuns: stats.runs - before.runs, sandboxMs: Math.round(stats.ms - before.ms), isolation: st.isolation, tokens: 0, costUsd: 0 }) + "\n");
    if (outcome === "ok" && o.crashAfter !== undefined && ++completedHere >= o.crashAfter) process.exit(137);
  };

  const running = new Map<string, Promise<void>>();
  for (;;) {
    for (const b of blocked(state)) emit({ t: "skip", at: now(), id: b.id, reason: b.reason });
    for (const n of ready(state)) {
      if (running.size >= concurrency) break;
      if (running.has(n.id)) continue;
      running.set(n.id, execute(n).finally(() => running.delete(n.id)));
    }
    if (!running.size) break;
    await Promise.race(running.values());
  }
  if (settled(state) && !state.done) emit({ t: "done", at: now() });

  const m = manifest();
  const arts = state.order.flatMap((id) => state.nodes.get(id)!.artifacts.map((a) => ({ node: id, name: a.name, sha256: a.sha256 })));
  const reportPath = join(runDir, "report.md");
  writeFileSync(reportPath, renderReport(state, m, { artifacts: arts, reproduce: `npm run mlh -- run ${JSON.stringify(state.source)} --objective ${JSON.stringify(state.objective)} --mode ${mode}` }));
  const summary = summarize(state);
  writeFileSync(join(runDir, "summary.json"), JSON.stringify(summary, null, 1));
  writeFileSync(join(runDir, "context.txt"), compact(state));
  return { state, summary, reportPath, stats: { sandboxRuns: stats.runs, sandboxMs: Math.round(stats.ms), wallMs: Math.round(performance.now() - t0), attempts: attemptsHere, retried } };
}
