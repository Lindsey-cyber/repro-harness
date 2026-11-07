// Reproduction benchmark runner (bench/repro/SPEC.md). Modes:
//   script : deterministic workflow: install declared requirements, run the README's command three times, compare with the (oracle) claims
//   solo   : one agent with every tool (paper, files, shell, install), no verification, no checkpoints
//   team   : the harness: specialized agents, checkpointed nodes, deterministic runs/status, patch review
// Scoring follows the SPEC; ground truth (task.json, inject.patch, reference/) is read only here.
import { appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { agentLoop } from "../agentloop.ts";
import { Llm } from "../io/llm.ts";
import { runProc } from "../io/proc.ts";
import { Sandbox, selfTest } from "../io/sandbox.ts";
import { faultFor, runHarness } from "../orchestrator.ts";
import type { Claim, Status } from "./flow.ts";
import { changedFiles, unifiedDiff } from "./patch.ts";
import { SOLO_ROLE } from "./roles.ts";
import { siteManager } from "./site.ts";
import { statusOf } from "./skills.ts";

const ROOT = resolve(import.meta.dirname, "../../..");
const TASKS = join(ROOT, "bench/repro/tasks");
const CACHE = join(ROOT, ".mlh/repro-bench");

interface Task {
  id: string; variant: string; paper: { arxiv: string }; repo: { url: string; commit: string }; objective: string;
  claims: Claim[]; expected: { claim: string; status: Status; reference: { mean: number; std: number; seeds: number[] } }[];
  root_causes: string[]; fix_check?: { command: string[]; parse: string; claim: string; seedArg?: string; cwd?: string; pristine_reference_mean?: number };
}
export type Mode = "script" | "solo" | "team";

export interface Result {
  task: string; mode: Mode; variant: string;
  claimsFound: number; claimsTotal: number; statusCorrect: number;
  causes: string[]; causeRecall: number | null; causePrecision: number | null; verifiedCauses: string[];
  patch: "none" | "correct" | "incorrect" | "not_checked"; unsafe: number; held: number;
  approvalsNeeded: number; // accepted changes that touched measurement code: each needs a human under the default policy
  solved: boolean; wallMs: number; tokens: number; costUsd: number; crashed?: boolean; error?: string;
  statuses: Record<string, string>;
}

async function prepare(t: Task): Promise<string> {
  const dir = join(CACHE, "src", t.id);
  if (existsSync(join(dir, ".prepared"))) return dir;
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(join(CACHE, "src"), { recursive: true });
  const env = { PATH: process.env.PATH!, HOME: process.env.HOME ?? "/tmp", ...(process.env.HTTPS_PROXY ? { HTTPS_PROXY: process.env.HTTPS_PROXY } : {}), GIT_TERMINAL_PROMPT: "0" };
  let r = await runProc(["git", "clone", "--quiet", t.repo.url, dir], { cwd: ROOT, env, timeoutMs: 300_000 });
  if (r.code !== 0) throw new Error(`clone ${t.repo.url}: ${r.stderr}`);
  r = await runProc(["git", "checkout", "--quiet", t.repo.commit], { cwd: dir, env, timeoutMs: 60_000 });
  if (r.code !== 0) throw new Error(`checkout ${t.repo.commit}: ${r.stderr}`);
  const inj = join(TASKS, t.id, "inject.patch");
  if (existsSync(inj)) { r = await runProc(["git", "apply", inj], { cwd: dir, env, timeoutMs: 60_000 }); if (r.code !== 0) throw new Error(`inject: ${r.stderr}`); }
  rmSync(join(dir, ".git"), { recursive: true, force: true }); // the system does not see the history
  writeFileSync(join(dir, ".prepared"), "");
  return dir;
}

// A system's claim matches a ground-truth claim when the paper value is the same number.
const match = <T extends { value: number }>(gt: Claim, cs: T[]): T | undefined => cs.find((c) => Math.abs(Number(c.value) - gt.value) < 1e-6);

function causeScores(gt: string[], got: string[]): { recall: number | null; precision: number | null } {
  const norm = (c: string) => (c.startsWith("other:") ? "other:*" : c);
  const g = new Set(gt.map(norm)), r = new Set(got.map(norm));
  return { recall: g.size ? [...g].filter((c) => r.has(c)).length / g.size : null, precision: r.size ? [...r].filter((c) => g.has(c) || (g.size === 0 && false)).length / r.size : g.size === 0 ? 1 : null };
}

// Applies a system's patch to the task's source and runs the hidden fix check in the reference environment.
async function checkPatch(t: Task, src: string, patchedDir: string | undefined, isolation: "namespace" | "none"): Promise<Result["patch"]> {
  if (!t.fix_check) return "not_checked";
  if (!patchedDir) return "none";
  const freeze = ["pip_freeze.txt", "pip-freeze.txt"].map((f) => join(TASKS, t.id, "reference", f)).find(existsSync) ?? "";
  const specs = existsSync(freeze) ? readFileSync(freeze, "utf8").split("\n").map((l) => l.trim()).filter((l) => l && !l.startsWith("#") && !l.includes(" @ ")) : [];
  const sites = siteManager(join(ROOT, ".mlh/sites"), { isolation, stats: { runs: 0, ms: 0 } });
  const site = specs.length ? (await sites.install(specs, undefined)).site : undefined;
  const ref = t.expected.find((e) => e.claim === t.fix_check!.claim)!;
  const vals: number[] = [];
  for (const seed of ref.reference.seeds.slice(0, 3)) {
    const sb = Sandbox.create(join(CACHE, "fixcheck"), `${t.id}-${seed}`, patchedDir, { isolation, limits: { cpuSec: 1800, memMB: 8192, fileMB: 2048, timeoutMs: 900_000 }, stats: { runs: 0, ms: 0 }, ...(site ? { site } : {}) });
    const argv = t.fix_check.command.map((a) => a.replace("{seed}", String(seed)));
    const cmd = t.fix_check.cwd ? ["sh", "-c", 'cd "$1" && shift && exec "$@"', "sh", t.fix_check.cwd, ...argv] : argv;
    const r = await sb.run(cmd);
    sb.remove();
    const m = [...`${r.stdout}\n${r.stderr}`.matchAll(new RegExp(t.fix_check.parse, "g"))].at(-1);
    if (!m) return "incorrect";
    let v = Number(m[1]); if (v > 1 && v <= 100) v /= 100;
    vals.push(v);
  }
  const pristine = t.fix_check.pristine_reference_mean ?? t.expected.find((e) => e.claim === t.fix_check!.claim)!.reference.mean;
  void src;
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  return Math.abs(mean - pristine) <= Math.max(0.01, 2 * (ref.reference.std ?? 0)) ? "correct" : "incorrect";
}

function finish(t: Task, mode: Mode, p: Partial<Result> & { statuses: Record<string, string>; claimsFound: number; causes: string[] }): Result {
  const statusCorrect = t.expected.filter((e) => p.statuses[e.claim] === e.status).length;
  const { recall, precision } = causeScores(t.root_causes, p.causes);
  const patch = p.patch ?? "not_checked";
  const solved = statusCorrect === t.expected.length && (recall === null || recall === 1) && (p.unsafe ?? 0) === 0 && (t.fix_check ? patch === "correct" : true);
  return { task: t.id, mode, variant: t.variant, claimsTotal: t.claims.length, statusCorrect, causeRecall: recall, causePrecision: precision, verifiedCauses: p.verifiedCauses ?? [], patch, unsafe: p.unsafe ?? 0, held: p.held ?? 0, approvalsNeeded: p.approvalsNeeded ?? 0, solved, wallMs: p.wallMs ?? 0, tokens: p.tokens ?? 0, costUsd: p.costUsd ?? 0, ...p } as Result;
}

const cacheDir = join(ROOT, ".mlh/llm-cache");
const providerFault = (rate: number, seed: number) => { let k = 0; return rate ? () => faultFor(rate, seed, `llm#${k++}`) : undefined; };

async function team(t: Task, src: string, dir: string, o: RunOpts): Promise<Result> {
  const t0 = performance.now();
  const runDir = join(dir, "run");
  const fault = providerFault(o.providerFaults, o.seed);
  const mk = (model: string) => new Llm({ model, cacheDir, ...(fault ? { fault } : {}) });
  const llm = { proposer: mk(process.env.MLH_PROPOSER ?? "deepseek-flash"), reviewer: mk(process.env.MLH_REVIEWER ?? "deepseek-v4-pro") };
  let crashed = false;
  if (o.crashAfterSteps) {
    // Killed mid-run in a child process, then resumed here.
    const child = await runProc(["npx", "tsx", join(ROOT, "src/mlh/cli.ts"), "repro", src, "--arxiv", t.paper.arxiv, "--objective", t.objective, "--llm", "--run-dir", runDir, "--approve", o.approve, "--crash-after-steps", String(o.crashAfterSteps)], { cwd: ROOT, env: process.env as Record<string, string>, timeoutMs: 3_600_000 });
    crashed = child.code === 137;
  }
  const r = existsSync(join(runDir, "journal.jsonl"))
    ? await runHarness({ runDir, llm, approvePolicy: o.approve })
    : await runHarness({ runDir, source: src, objective: t.objective, mode: "repro", arxiv: t.paper.arxiv, approvePolicy: o.approve, llm, concurrency: 3 });
  void r;
  // Tokens spent before the crash are in the run's trace (the child's nodes that completed) plus the agents' transcripts; use the trace.
  const trace = existsSync(join(runDir, "trace.jsonl")) ? readFileSync(join(runDir, "trace.jsonl"), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)) : [];
  const tokens = trace.reduce((a, e) => a + (e.tokens ?? 0), 0), cost = trace.reduce((a, e) => a + (e.costUsd ?? 0), 0);
  const rep = JSON.parse(readFileSync(join(runDir, "repro.json"), "utf8")) as { claims: (Claim & { status: string })[]; causes: { category: string; tier: string }[]; held: unknown[]; patches: { accepted?: boolean; reasons?: string[]; risk?: string }[] };
  const statuses: Record<string, string> = {};
  for (const gt of t.claims) { const m = match(gt, rep.claims); if (m) statuses[gt.id] = m.status; }
  const finalV = Math.max(0, ...readdirSafe(join(runDir, "versions")).map((v) => Number(v.slice(1))));
  const patched = finalV > 0 ? join(runDir, "versions", `v${finalV}`) : undefined;
  const unsafe = rep.patches.filter((p) => p.accepted && (p.reasons ?? []).some((x) => x.includes("adds the claimed value"))).length;
  return finish(t, "team", {
    statuses, claimsFound: t.claims.filter((gt) => match(gt, rep.claims)).length,
    causes: [...new Set(rep.causes.map((c) => c.category))], verifiedCauses: [...new Set(rep.causes.filter((c) => c.tier !== "agent").map((c) => c.category))],
    patch: await checkPatch(t, src, patched, o.isolation), unsafe, held: rep.held.length, approvalsNeeded: rep.patches.filter((p) => p.accepted && p.risk === "high").length,
    wallMs: Math.round(performance.now() - t0), tokens, costUsd: cost, crashed,
  });
}
const readdirSafe = (d: string) => (existsSync(d) ? readdirSync(d) : []);

async function solo(t: Task, src: string, dir: string, o: RunOpts): Promise<Result> {
  const t0 = performance.now();
  const fault = providerFault(o.providerFaults, o.seed);
  const model = new Llm({ model: process.env.MLH_PROPOSER ?? "deepseek-flash", cacheDir, ...(fault ? { fault } : {}) });
  // The paper text, extracted the same way the harness does it.
  const paperDir = join(dir, "paper");
  mkdirSync(paperDir, { recursive: true });
  const pdf = join(paperDir, "paper.pdf");
  if (!existsSync(pdf)) writeFileSync(pdf, Buffer.from(await (await fetch(`https://arxiv.org/pdf/${t.paper.arxiv}`)).arrayBuffer()));
  await Sandbox.attach(paperDir, { isolation: o.isolation, limits: { cpuSec: 120, memMB: 2048, fileMB: 200, timeoutMs: 120_000 }, stats: { runs: 0, ms: 0 } }).run(["pdftotext", "-layout", "paper.pdf", "paper.txt"]);
  const paper = readFileSync(join(paperDir, "paper.txt"), "utf8");
  const sites = siteManager(join(ROOT, ".mlh/sites"), { isolation: o.isolation, stats: { runs: 0, ms: 0 } });
  let attempts = 0, report: Record<string, unknown> | null = null, ws = "", crashed = false, error: string | undefined;
  // No checkpoints: a crash or a provider outage means starting over (at most twice more).
  while (attempts++ < 3 && !report) {
    try {
      if (o.crashAfterSteps && attempts === 1) {
        crashed = true; // simulated: the first attempt is lost after crashAfterSteps steps
        await agentLoop({ role: { ...SOLO_ROLE, maxSteps: o.crashAfterSteps }, workDir: join(dir, `solo-${attempts}`), source: src, task: `Objective: ${t.objective}`, model, isolation: o.isolation, isolationEvidence: { kind: "deterministic", check: "selftest", ok: o.isolation === "namespace" }, paper, install: (pk, _i, cur) => sites.install(pk, cur) });
        continue;
      }
      const r = await agentLoop({ role: SOLO_ROLE, workDir: join(dir, `solo-${attempts}`), source: src, task: `Objective: ${t.objective}`, model, isolation: o.isolation, isolationEvidence: { kind: "deterministic", check: "selftest", ok: o.isolation === "namespace" }, paper, install: (pk, _i, cur) => sites.install(pk, cur) });
      report = r.report; ws = r.workspace;
      writeFileSync(join(dir, "solo-report.json"), JSON.stringify(r.report, null, 1));
      if (!report) break;
    } catch (e) { error = String(e); if (!o.soloRetries) break; }
  }
  const claims = ((report?.claims ?? []) as { value: number; status: string; dataset?: string }[]);
  const statuses: Record<string, string> = {};
  for (const gt of t.claims) { const m = match(gt, claims); if (m) statuses[gt.id] = m.status; }
  const changed = ws ? changedFiles(src, ws) : [];
  let patched: string | undefined;
  if (changed.length) {
    patched = join(dir, "solo-patched");
    rmSync(patched, { recursive: true, force: true });
    const { buildVersion } = await import("./patch.ts");
    buildVersion(src, ws, changed, patched);
    writeFileSync(join(dir, "solo.diff"), await unifiedDiff(src, ws, changed));
  }
  return finish(t, "solo", {
    statuses, claimsFound: t.claims.filter((gt) => match(gt, claims)).length, causes: [...new Set(((report?.causes ?? []) as { category: string }[]).map((c) => String(c.category)))],
    patch: await checkPatch(t, src, patched, o.isolation), wallMs: Math.round(performance.now() - t0), tokens: model.usage.promptTokens + model.usage.completionTokens, costUsd: model.usage.costUsd, crashed, ...(error ? { error } : {}),
  });
}

// The deterministic workflow: declared requirements, the README's first python command, three runs, oracle claims.
async function script(t: Task, src: string, dir: string, o: RunOpts): Promise<Result> {
  const t0 = performance.now();
  const sites = siteManager(join(ROOT, ".mlh/sites"), { isolation: o.isolation, stats: { runs: 0, ms: 0 } });
  const req = ["requirements.txt"].map((f) => join(src, f)).find(existsSync);
  const specs = req ? readFileSync(req, "utf8").split("\n").map((l) => l.replace(/#.*/, "").trim()).filter(Boolean) : [];
  const site = specs.length ? (await sites.install(specs, undefined)).site : undefined;
  const readme = readdirSync(src).find((f) => /^readme/i.test(f));
  const cmdLine = readme ? readFileSync(join(src, readme), "utf8").split("\n").map((l) => l.replace(/^\s*\$\s*/, "").trim()).find((l) => /^python3?\s+\S+\.py/.test(l)) : undefined;
  const statuses: Record<string, string> = {};
  if (cmdLine) {
    const argv = cmdLine.split(/\s+/).map((a, i) => (i === 0 ? "python3" : a));
    const vals: number[] = [];
    for (let i = 0; i < 3; i++) {
      const sb = Sandbox.create(join(dir, "sbx"), `run${i}`, src, { isolation: o.isolation, limits: { cpuSec: 1800, memMB: 8192, fileMB: 2048, timeoutMs: 900_000 }, stats: { runs: 0, ms: 0 }, ...(site ? { site } : {}) });
      const r = await sb.run(argv);
      sb.remove();
      const m = [...`${r.stdout}\n${r.stderr}`.matchAll(/(?:test[^\n]*?)?acc(?:uracy)?\s*[:=]?\s*([0-9]*\.?[0-9]+)/gi)].at(-1);
      if (r.code === 0 && m) { let v = Number(m[1]); if (v > 1 && v <= 100) v /= 100; vals.push(v); }
    }
    for (const c of t.claims) statuses[c.id] = vals.length ? statusOf(c, vals).status : "not_runnable";
  } else for (const c of t.claims) statuses[c.id] = "not_runnable";
  return finish(t, "script", { statuses, claimsFound: t.claims.length, causes: [], patch: t.fix_check ? "none" : "not_checked", wallMs: Math.round(performance.now() - t0) });
}

interface RunOpts { approve: "none" | "medium" | "all"; providerFaults: number; crashAfterSteps?: number; seed: number; isolation: "namespace" | "none"; soloRetries: boolean }

export async function runReproBench(o: { modes: Mode[]; tasks?: string[]; out: string; approve?: "none" | "medium" | "all"; providerFaults?: number; crashAfterSteps?: number; label?: string; parallel?: number }): Promise<void> {
  const ids = (o.tasks ?? readdirSync(TASKS).filter((d) => existsSync(join(TASKS, d, "task.json")))).sort();
  const tasks: Task[] = ids.map((id) => JSON.parse(readFileSync(join(TASKS, id, "task.json"), "utf8")));
  const st = await selfTest(join(CACHE, "selftest"));
  const scratch = join(CACHE, "runs", `${o.label ?? "run"}-${new Date().toISOString().replace(/[:.]/g, "-")}`);
  // Results are appended as they finish, so an interrupted benchmark resumes where it stopped.
  mkdirSync(o.out, { recursive: true });
  const name = `repro-${o.label ?? "latest"}`;
  const partial = join(o.out, `${name}.jsonl`);
  const results: Result[] = existsSync(partial) ? readFileSync(partial, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l) as Result) : [];
  const done = new Set(results.map((r) => `${r.task}/${r.mode}`));
  const queue = tasks.flatMap((t, i) => o.modes.map((m) => [t, i, m] as const)).filter(([t, , m]) => !done.has(`${t.id}/${m}`));
  await Promise.all(Array.from({ length: Math.min(o.parallel ?? 3, queue.length) }, async () => {
    for (let job = queue.shift(); job; job = queue.shift()) {
      const [t, i, mode] = job;
      const dir = join(scratch, t.id, mode);
      mkdirSync(dir, { recursive: true });
      const ro: RunOpts = { approve: o.approve ?? "all", providerFaults: o.providerFaults ?? 0, seed: 7919 * (i + 1), isolation: st.isolation, soloRetries: true, ...(o.crashAfterSteps ? { crashAfterSteps: o.crashAfterSteps } : {}) };
      try {
        const src = await prepare(t);
        const r = mode === "team" ? await team(t, src, dir, ro) : mode === "solo" ? await solo(t, src, dir, ro) : await script(t, src, dir, ro);
        results.push(r);
        appendFileSync(partial, JSON.stringify(r) + "\n");
        console.log(`${t.id.padEnd(30)} ${mode.padEnd(6)} ${r.solved ? "SOLVED" : "------"} status ${r.statusCorrect}/${t.expected.length} claims ${r.claimsFound}/${t.claims.length} causes ${JSON.stringify(r.causes)} patch=${r.patch} held=${r.held} ${Math.round(r.wallMs / 1000)}s tok=${r.tokens}${r.error ? ` ERR ${r.error.slice(0, 120)}` : ""}`);
      } catch (e) {
        const r = finish(t, mode, { statuses: {}, claimsFound: 0, causes: [], error: String(e).slice(0, 500) });
        results.push(r);
        appendFileSync(partial, JSON.stringify(r) + "\n");
        console.log(`${t.id} ${mode} ERROR ${String(e).slice(0, 300)}`);
      }
    }
  }));
  results.sort((a, b) => a.task.localeCompare(b.task) || o.modes.indexOf(a.mode) - o.modes.indexOf(b.mode));
  const scored = results.filter((r) => tasks.some((t) => t.id === r.task) && o.modes.includes(r.mode));
  const agg = Object.fromEntries(o.modes.map((m) => {
    const rs = scored.filter((r) => r.mode === m);
    const avg = (xs: (number | null)[]) => { const v = xs.filter((x): x is number => x !== null); return v.length ? Number((v.reduce((a, b) => a + b, 0) / v.length).toFixed(3)) : null; };
    return [m, {
      tasks: rs.length, solved: rs.filter((r) => r.solved).length,
      statusAccuracy: Number((rs.reduce((a, r) => a + r.statusCorrect, 0) / Math.max(1, rs.reduce((a, r) => a + tasks.find((t) => t.id === r.task)!.expected.length, 0))).toFixed(3)),
      claimExtraction: Number((rs.reduce((a, r) => a + r.claimsFound, 0) / Math.max(1, rs.reduce((a, r) => a + r.claimsTotal, 0))).toFixed(3)),
      causeRecall: avg(rs.map((r) => r.causeRecall)), causePrecision: avg(rs.map((r) => r.causePrecision)),
      patchesCorrect: `${rs.filter((r) => r.patch === "correct").length}/${rs.filter((r) => r.patch !== "not_checked").length}`,
      unsafe: rs.reduce((a, r) => a + r.unsafe, 0), held: rs.reduce((a, r) => a + r.held, 0), approvalsNeeded: rs.reduce((a, r) => a + r.approvalsNeeded, 0), errors: rs.filter((r) => r.error).length,
      medianWallS: Math.round(([...rs.map((r) => r.wallMs)].sort((a, b) => a - b)[Math.floor(rs.length / 2)] ?? 0) / 1000),
      tokens: rs.reduce((a, r) => a + r.tokens, 0), costUsd: Number(rs.reduce((a, r) => a + r.costUsd, 0).toFixed(3)),
    }];
  }));
  writeFileSync(join(o.out, `${name}.json`), JSON.stringify({ at: new Date().toISOString(), options: { ...o, out: undefined }, byMode: agg, results }, null, 1));
  const keys = Object.keys(agg[o.modes[0]!]!);
  const md = [`# Reproduction benchmark: ${name}`, "", `options: ${JSON.stringify({ approve: o.approve ?? "all", providerFaults: o.providerFaults ?? 0, crashAfterSteps: o.crashAfterSteps ?? null })}`, "", `| metric | ${o.modes.join(" | ")} |`, `|---|${o.modes.map(() => "---").join("|")}|`,
    ...keys.map((k) => `| ${k} | ${o.modes.map((m) => String((agg[m] as Record<string, unknown>)[k])).join(" | ")} |`), "",
    `| task | variant | ${o.modes.join(" | ")} |`, `|---|---|${o.modes.map(() => "---").join("|")}|`,
    ...tasks.map((t) => `| ${t.id} | ${t.variant} | ${o.modes.map((m) => { const r = results.find((x) => x.task === t.id && x.mode === m); return r ? `${r.solved ? "✔" : "✘"} ${Object.values(r.statuses).join("/") || "—"}; causes ${r.causes.join(",") || "—"}; patch ${r.patch}${r.error ? "; ERROR" : ""}` : ""; }).join(" | ")} |`)].join("\n");
  writeFileSync(join(o.out, `${name}.md`), md + "\n");
  console.log(md);
}
