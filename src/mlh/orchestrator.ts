// Executes a run: folds the journal into state, schedules ready nodes in
// parallel, authorizes each node's actions before it runs, retries what a
// fresh attempt can fix, replans on results, and writes the report. Every
// state change is journaled first, so a killed process resumes where it was.
import { appendFileSync, cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { authorize, type Action, type Evidence } from "./core/authz.ts";
import { apply, blocked, reopenPlatform, depOf, ready, replay, settled, type RunEvent, type RunState, type TaskNode } from "./core/dag.ts";
import { AGENT, confirmations, expand, fallback, INSPECT, initialPlan } from "./core/planner.ts";
import { compact, renderReport, summarize, type Summary } from "./core/report.ts";
import { initialTrust, type SkillTrust } from "./core/trust.ts";
import type { Manifest } from "./core/types.ts";
import { ArtifactStore } from "./io/artifacts.ts";
import { Journal } from "./io/journal.ts";
import { runProc } from "./io/proc.ts";
import { DEFAULT_LIMITS, Sandbox, selfTest, SkillError, type Isolation, type Limits, type SandboxStats } from "./io/sandbox.ts";
import { addUsage, meter, zeroUsage, type Usage } from "./io/llm.ts";
import type { Models, SkillCtx } from "./skills/index.ts";
import { skill, SKILLS } from "./skills/registry.ts";
import * as flow from "./repro/flow.ts";
import { renderRepro } from "./repro/report.ts";
import { siteManager } from "./repro/site.ts";
import { Budget, costScope } from "./backends/modal.ts";
import { costReport, stopCheck } from "./repro/accounting.ts";
import { createHash } from "node:crypto";

// agentic: an exploring agent node leads; its findings are routed to independent confirmers.
// repro: reproduce a paper's claims with specialized agents (src/mlh/repro).
export type Mode = "harness" | "workflow" | "agentic" | "repro";

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
  // Models for the LLM skills. Passing them is the human's standing approval to
  // send repository content to that provider (the only evidence for model_query).
  llm?: Models;
  crashAfterSteps?: number; // exit after this many agent steps in this process (recovery tests)
  // Reproduction runs: the paper and how much autonomy patches get.
  arxiv?: string;
  approvePolicy?: "none" | "medium" | "all";
  sitesDir?: string; // shared package sites (default .mlh/sites)
  gpuBudgetUsd?: number; // run stages on Modal GPUs within this budget (per paper)
}

export interface RunResult {
  state: RunState;
  summary: Summary;
  reportPath: string;
  stats: { sandboxRuns: number; sandboxMs: number; wallMs: number; attempts: number; retried: number; llm: Usage };
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
  // This process's failures, for the stop conditions (a resume starts a clean count).
  const procFails: { id: string; category: string; message: string }[] = [];
  let stop: { reason: string; detail: string } | undefined;
  const tracePath = join(runDir, "trace.jsonl");
  let events = journal.read();
  const mode: Mode = (events.find((e) => e.t === "run") as { mode?: Mode } | undefined)?.mode ?? o.mode ?? "harness";
  const maxAttempts = mode === "workflow" ? 1 : 3;
  const state = replay(events, maxAttempts);
  const emit = (e: RunEvent) => { journal.append(e); apply(state, e); };

  if (!events.length) {
    if (!o.source || !o.objective) throw new Error("a new run needs a source and an objective");
    mkdirSync(runDir, { recursive: true });
    const { rev } = await fetchSource(o.source, srcDir);
    writeFileSync(join(runDir, "run.json"), JSON.stringify({ source: o.source, rev, objective: o.objective, mode, created: now(), ...(o.arxiv ? { arxiv: o.arxiv, approvePolicy: o.approvePolicy ?? "medium" } : {}), ...(o.gpuBudgetUsd ? { gpuBudgetUsd: o.gpuBudgetUsd } : {}) }, null, 1));
    emit({ t: "run", at: now(), objective: o.objective, source: o.source, mode });
    for (const n of mode === "repro" ? flow.initialNodes() : initialPlan()) emit({ t: "node", at: now(), node: n, why: "initial plan" });
  } else if (!existsSync(srcDir)) throw new Error(`cannot resume ${runDir}: source copy missing`);
  events = [];

  rmSync(join(runDir, "stopped.json"), { force: true }); // a (re)started run is no longer stopped
  for (const e of reopenPlatform(state, now())) journal.append(e); // platform failures get a fresh attempt on resume
  const st = o.isolation ? { isolation: o.isolation, checks: [{ name: "forced by caller", ok: o.isolation === "namespace" }] } : await selfTest(join(runDir, "sbx"));
  const isolationEvidence: Evidence = { kind: "deterministic", check: `sandbox.selftest(${st.checks.map((c) => `${c.name}=${c.ok}`).join(",")})`, ok: st.isolation === "namespace" };
  const trust = loadTrust(o.trustFile);
  const stats: SandboxStats = { runs: 0, ms: 0 };
  const limits = o.limits ?? DEFAULT_LIMITS;
  // Reproduction runs are strictly sequential (one node at a time) until reliability is proven
  // (docs/mlh/RELIABILITY.md, P6); other modes keep their default.
  const concurrency = mode === "repro" ? 1 : o.concurrency ?? 3;
  let completedHere = 0, retried = 0, attemptsHere = 0, llmTotal = zeroUsage();
  const runMeta = JSON.parse(readFileSync(join(runDir, "run.json"), "utf8")) as { arxiv?: string; approvePolicy?: "none" | "medium" | "all"; gpuBudgetUsd?: number };
  // run.json fixes the run's identity (its hash names the volume), so a raise is not written there: the
  // owner's raises are appended to budget-raises.jsonl, each with its reason, and add to the limit.
  const base = runMeta.gpuBudgetUsd ?? o.gpuBudgetUsd, gpuBudget = base ? base + budgetRaises(runDir) : base;
  const repro = mode === "repro" ? {
    arxiv: runMeta.arxiv ?? o.arxiv ?? "",
    approvePolicy: runMeta.approvePolicy ?? o.approvePolicy ?? "medium",
    sites: siteManager(resolve(o.sitesDir ?? ".mlh/sites"), { isolation: st.isolation, stats }),
    versionDir: (v: number) => (v === 0 ? srcDir : join(runDir, "versions", `v${v}`)),
    // Spending is saved next to the journal, so a resumed run continues from what it already spent.
    // Each run gets its own /cache volume, named by the run's key: runs cannot see each other's leftovers.
    ...(gpuBudget ? (() => { const runKey = createHash("sha256").update(readFileSync(join(runDir, "run.json"), "utf8")).digest("hex").slice(0, 12); return { gpu: { budget: new Budget(gpuBudget, join(runDir, "budget.json"), `mlh-run-${runKey}`), runKey } }; })() : {}),
  } : undefined;
  const specs = mode === "repro" ? SKILLS.filter((s) => s.id.startsWith("repro."))
    : SKILLS.filter((s) => !s.id.startsWith("repro.") && (o.llm || !s.actions.includes("model_query")) && (mode === "agentic" || (s.id !== AGENT && !s.id.startsWith("verify."))) && (mode !== "agentic" || !s.id.startsWith("llm.")));
  const detSpecs = specs.filter((s) => s.id !== AGENT);
  // Explicit mlh.json > what the exploring agent validated > inference.
  const manifest = (): Manifest => {
    const ins = state.nodes.get(INSPECT)?.output as { manifest?: Manifest; provenance?: Record<string, string> } | undefined;
    const ag = state.nodes.get(AGENT)?.status === "succeeded" ? (state.nodes.get(AGENT)!.output as { manifest?: Manifest }).manifest ?? {} : {};
    const explicit = Object.fromEntries(Object.entries(ins?.manifest ?? {}).filter(([k]) => ins?.provenance?.[k] === "mlh.json"));
    return { ...(ins?.manifest ?? {}), ...ag, ...explicit } as Manifest;
  };

  const afterOk = (n: TaskNode, findings: number) => {
    if (mode === "repro") { for (const node of flow.afterOk(state, n.id)) if (!state.nodes.has(node.id)) emit({ t: "node", at: now(), node, why: `after ${n.id}` }); return; }
    if (n.id === INSPECT && mode === "agentic") {
      emit({ t: "node", at: now(), node: { id: AGENT, skill: AGENT, deps: [INSPECT], purpose: "detect", params: {} }, why: "the exploring agent leads" });
    }
    if ((n.id === INSPECT && mode !== "agentic") || n.id === AGENT) {
      const x = expand(state.objective, manifest(), detSpecs);
      for (const node of x.nodes) emit({ t: "node", at: now(), node, why: "expanded from manifest" });
      for (const g of x.gaps) emit({ t: "gap", at: now(), skill: g.skill, reason: g.reason });
    }
    if (n.purpose === "confirm" && n.target) {
      const c = state.nodes.get(n.id)!;
      const last = state.findings.get(n.target)?.confirmations.find((x) => x.node === n.id);
      if (c.status === "succeeded" && last?.inconclusive) for (const node of fallback(state, n.target, specs)) emit({ t: "node", at: now(), node, why: `${n.id} inconclusive` });
    }
    if (mode === "workflow" || n.purpose === "confirm") return;
    for (let i = 0; i < findings; i++) {
      const fid = `${n.id}/${i}`;
      for (const node of confirmations(state, fid, specs, manifest(), trust)) {
        if (!state.nodes.has(node.id)) emit({ t: "node", at: now(), node, why: `confirm ${fid}` });
      }
    }
  };

  const execute = async (n: TaskNode): Promise<void> => {
    const ns = state.nodes.get(n.id)!;
    const sk = skill(n.skill);
    // Authorization first: sandboxed execution is only sandboxed if the self-test proved it.
    const evidence: Evidence[] = [isolationEvidence, ...(ns.approved ? [{ kind: "human" as const, check: `approval:${n.id}`, ok: true }] : [])];
    if (o.llm) evidence.push({ kind: "human", check: "--llm standing approval", ok: true, action: "model_query" });
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
    const callLog = (u: Usage) => appendFileSync(join(runDir, "llm-calls.jsonl"), JSON.stringify({ key: `${n.id}#${attempt}`, costUsd: u.costUsd, tokens: u.promptTokens + u.completionTokens }) + "\n");
    const models = o.llm ? { proposer: meter(o.llm.proposer, callLog), reviewer: meter(o.llm.reviewer, callLog) } : undefined;
    const ctx: SkillCtx = {
      ...(models ? { llm: models } : {}),
      workDir: () => join(runDir, "work", n.id.replace(/[^\w.-]+/g, "_")),
      sourceDir: srcDir,
      stats,
      approved: ns.approved,
      ...(repro ? { repro } : {}),
      isolation: st.isolation,
      isolationEvidence,
      ...(o.faults ? { sandboxFault: (() => { let k = 0; return () => faultFor(o.faults!.rate, o.faults!.seed, `${n.id}#${attempt}#agent#${k++}`); })() } : {}),
      ...(o.crashAfterSteps !== undefined ? { crashAfterSteps: o.crashAfterSteps } : {}),
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
      const out = await costScope.run(`${n.id}#${attempt}`, () => sk.run(ctx));
      const parsed = sk.output.safeParse(out.output);
      if (!parsed.success) throw new SkillError("contract_violation", `${sk.id}@${sk.version} output violates its schema: ${parsed.error.message.slice(0, 300)}`);
      const findings = out.findings ?? [];
      let confirmation = out.confirmation && n.target ? { ...out.confirmation, target: n.target } : undefined;
      if (!confirmation && n.target && typeof n.params.category === "string") {
        const again = findings.filter((f) => f.category === n.params.category);
        confirmation = { target: n.target, supports: again.length > 0, note: again.length ? `replicated on a fresh run: ${again[0]!.summary}` : `not reproduced on a fresh run` };
      }
      if (out.hold) {
        emit({ t: "hold", at: now(), id: n.id, reason: out.hold });
        writeFileSync(join(runDir, "pending.json"), JSON.stringify(state.order.filter((id) => state.nodes.get(id)!.status === "held").map((id) => ({ id, reason: state.nodes.get(id)!.reason })), null, 1));
        appendFileSync(tracePath, JSON.stringify({ at: now(), id: n.id, skill: sk.id, version: sk.version, attempt, outcome: "hold", ms: Math.round(performance.now() - started) }) + "\n");
        return;
      }
      emit({ t: "ok", at: now(), id: n.id, output: out.output, findings, artifacts: saved, ...(confirmation ? { confirmation } : {}) });
      if (out.gap) emit({ t: "gap", at: now(), skill: sk.id, reason: out.gap });
      outcome = "ok";
      afterOk(n, findings.length);
    } catch (e) {
      // A model call that failed or returned garbage is a transient fault of the provider, not a harness bug.
      const msg = String(e);
      const cat = e instanceof SkillError ? e.category : /LLM 40[12]\b|Insufficient Balance|invalid api key|Authentication Fails/i.test(msg) ? "provider_fatal" : /LLM|model returned/.test(msg) ? "provider_transient" : "harness_bug";
      if (cat === "provider_fatal") fatal = msg.slice(0, 300);
      procFails.push({ id: n.id, category: cat, message: String((e as Error).message ?? e) });
      emit({ t: "fail", at: now(), id: n.id, category: cat, message: String((e as Error).message ?? e).slice(0, 2000) });
      outcome = "fail"; category = cat;
      if (state.nodes.get(n.id)!.status === "pending") {
        retried++;
        // Capacity and transport faults need minutes, not milliseconds (MLH_BACKOFF_MS scales it; tests set 0).
        const base = cat === "provider_transient" || cat === "sandbox_transient" ? Number(process.env.MLH_BACKOFF_MS ?? 15_000) : 50;
        await new Promise((r) => setTimeout(r, Math.min(600_000, base * 2 ** attempt)));
      } else if (mode === "repro") {
        for (const node of flow.afterFail(state, n.id)) if (!state.nodes.has(node.id)) emit({ t: "node", at: now(), node, why: `after ${n.id} failed` });
      }
    }
    const used = models ? addUsage(models.proposer.usage, models.reviewer.usage) : zeroUsage();
    llmTotal = addUsage(llmTotal, used);
    appendFileSync(tracePath, JSON.stringify({ at: now(), id: n.id, skill: sk.id, version: sk.version, attempt, outcome, category, ms: Math.round(performance.now() - started), sandboxRuns: stats.runs - before.runs, sandboxMs: Math.round(stats.ms - before.ms), isolation: st.isolation, llmCalls: used.calls, tokens: used.promptTokens + used.completionTokens, costUsd: Number(used.costUsd.toFixed(6)), llmMs: Math.round(used.ms), gpuUsd: Number((repro?.gpu?.budget.byAttempt[`${n.id}#${attempt}`] ?? 0).toFixed(6)) }) + "\n");
    // Strict stop conditions (repeated infrastructure fault, unclassified failure, waste, LLM cap).
    if (mode === "repro" && !stop) {
      const rows = readFileSync(tracePath, "utf8").split("\n").filter(Boolean).flatMap((l) => { try { return [JSON.parse(l)]; } catch { return []; } });
      stop = stopCheck(procFails, costReport(rows, repro?.gpu?.budget.byAttempt ?? {}));
    }
    if (outcome === "ok" && o.crashAfter !== undefined && ++completedHere >= o.crashAfter) process.exit(137);
  };

  const running = new Map<string, Promise<void>>();
  let fatal: string | undefined;
  for (;;) {
    if (fatal || stop) break; // nothing new starts once the provider is unusable; running nodes finish
    for (const b of blocked(state)) emit({ t: "skip", at: now(), id: b.id, reason: b.reason });
    for (const n of ready(state)) {
      if (running.size >= concurrency) break;
      if (running.has(n.id)) continue;
      running.set(n.id, execute(n).finally(() => running.delete(n.id)));
    }
    if (!running.size) break;
    await Promise.race(running.values());
  }
  while (running.size) await Promise.race(running.values());
  if (fatal) writeFileSync(join(runDir, "stopped.json"), JSON.stringify({ reason: "provider_fatal", detail: fatal, at: now() }));
  else if (stop) writeFileSync(join(runDir, "stopped.json"), JSON.stringify({ ...stop, at: now() }));
  if (settled(state) && !state.done && !fatal && !stop) emit({ t: "done", at: now() });

  const m = manifest();
  const arts = state.order.flatMap((id) => state.nodes.get(id)!.artifacts.map((a) => ({ node: id, name: a.name, sha256: a.sha256 })));
  const reportPath = join(runDir, "report.md");
  // Top-level packages and modules of the repository, so a failed import of one of them reads as an import-path problem.
  const local = new Set(existsSync(srcDir) ? readdirSync(srcDir, { withFileTypes: true }).flatMap((e) => (e.isDirectory() && existsSync(join(srcDir, e.name, "__init__.py")) ? [e.name] : e.name.endsWith(".py") ? [e.name.slice(0, -3)] : [])) : []);
  const rr = mode === "repro" ? renderRepro(state, local) : undefined;
  if (rr) writeFileSync(join(runDir, "repro.json"), JSON.stringify(rr.json, null, 1));
  writeFileSync(reportPath, rr ? rr.markdown : renderReport(state, m, { artifacts: arts, reproduce: `npm run mlh -- run ${JSON.stringify(state.source)} --objective ${JSON.stringify(state.objective)} --mode ${mode}` }));
  const summary = summarize(state);
  writeFileSync(join(runDir, "summary.json"), JSON.stringify(summary, null, 1));
  writeFileSync(join(runDir, "context.txt"), compact(state));
  return { state, summary, reportPath, stats: { sandboxRuns: stats.runs, sandboxMs: Math.round(stats.ms), wallMs: Math.round(performance.now() - t0), attempts: attemptsHere, retried, llm: llmTotal } };
}

export function budgetRaises(runDir: string): number {
  const f = join(runDir, "budget-raises.jsonl");
  if (!existsSync(f)) return 0;
  return readFileSync(f, "utf8").split("\n").filter(Boolean).map((l) => Number((JSON.parse(l) as { usd?: unknown }).usd)).filter((x) => Number.isFinite(x) && x > 0).reduce((a, b) => a + b, 0);
}
