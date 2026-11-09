// Modal GPU execution backend. Repository code runs only inside a Modal Sandbox,
// never on this machine; this module uploads, launches and reads results through
// the Modal SDK (modal_runner.py). A dollar budget is enforced before every
// launch: a run whose worst case would exceed what is left is refused.
import { AsyncLocalStorage } from "node:async_hooks";
import { existsSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { FailureCategory } from "../core/types.ts";
import { runProc } from "../io/proc.ts";

export const RUNNER = join(import.meta.dirname, "modal_runner.py");
export const MODAL_SITE = join(import.meta.dirname, "../../../.mlh/tools/modal-site");

// USD per GPU-hour (Modal list prices as of 2026-11; override with MLH_GPU_PRICE_<GPU>).
const PRICE: Record<string, number> = { T4: 0.59, L4: 0.8, A10G: 1.1, "A100-40GB": 2.1, "A100-80GB": 2.5, H100: 3.95 };
// Every container also bills CPU and memory, GPU or not, and the clock includes the image build.
// Conservative default: ~2 cores + 4 GiB at list price (~$0.13/h); override with MLH_CPU_PRICE.
export const cpuPrice = () => Number(process.env.MLH_CPU_PRICE ?? 0.13);
export const containerCost = (gpu: string | undefined, seconds: number) => ((cpuPrice() + (gpu ? gpuPrice(gpu) : 0)) * seconds) / 3600;
// Every Modal error we have seen, one place, tested against the recorded corpus
// (tests/fixtures/modal-errors.json). Order matters: capacity before the generic ResourceExhausted.
export function classifyModalError(head: string): FailureCategory {
  if (/Insufficient capacity|capacity to create|rate.?limit|Too many requests/i.test(head)) return "provider_transient";
  if (/spend limit|spending limit|exceeded its spe|billing|insufficient (funds|credit)/i.test(head)) return "provider_fatal";
  if (/ImageBuildError|Image build/.test(head)) return "experiment_error";
  if (/\b(AuthError|Token (ID|secret)|Unauthenticated)\b/.test(head)) return "precondition";
  if (/^(\w+\.)*(Timeout|SandboxTimeout|ExecTimeout)\w*Error|timed out|deadline exceeded/i.test(head)) return "timeout";
  // Unknown ResourceExhausted (quota of some kind): retry with backoff rather than stop the campaign.
  if (/ResourceExhausted/.test(head)) return "provider_transient";
  return "sandbox_transient";
}

// A finished command's outcome. Pure. 137 is the kernel's OOM kill; a missing upstream output under
// /cache/runs (its stage succeeded) is a volume not yet committed: transient. 97: a declared output missing.
export function classifyExit(code: number, stderr: string): FailureCategory | null {
  if (code === 0) return null;
  if (code === 124 || code === -1) return "timeout";
  if (code === 137 || /CUDA out of memory|OutOfMemoryError|MemoryError\b/.test(stderr)) return "resource_limit";
  if (/cannot stat '\/cache\/runs\//.test(stderr)) return "sandbox_transient";
  return "experiment_error";
}

export const gpuPrice = (gpu: string) => Number(process.env[`MLH_GPU_PRICE_${gpu.replace(/\W/g, "_")}`] ?? PRICE[gpu] ?? 4);

export interface RemoteRun {
  repo: string; cmd: string[]; gpu?: string; timeoutS: number; network?: boolean;
  env?: Record<string, string>; setup?: string; outputs?: string[]; image?: string;
  pip?: string[]; // installed into the image (cached by Modal), not at run time
  domains?: string[]; // with network: only these outbound domains (wildcards allowed)
  restore?: { from: string; to: string }[]; // /cache/... -> repo-relative, before the command
  persist?: { from: string; to: string }[]; // repo-relative -> /cache/..., after the command
}
export interface RemoteResult { exitCode: number; stdout: string; stderr: string; elapsedS: number; costUsd: number; files: Record<string, string>; category: FailureCategory | null }

// Which node attempt a charge belongs to ("<node id>#<attempt>"), set by the orchestrator around
// each attempt. AsyncLocalStorage keeps it right when several stages run at once.
export const costScope = new AsyncLocalStorage<string>();
export const UNSETTLED = "(reservations of a killed process)";

export class Budget {
  spent = 0;
  // Spend per node attempt, so failed attempts and retries can be priced separately.
  byAttempt: Record<string, number> = {};
  // With a file, spending survives a crash: a resumed run continues from what it already spent.
  // volume: the run's own /cache volume (isolation: nothing one run leaves behind reaches another).
  constructor(readonly limitUsd: number, private readonly file?: string, readonly volume = "mlh-cache") {
    if (file && existsSync(file)) {
      try {
        const j = JSON.parse(readFileSync(file, "utf8")) as { spent?: number; reserved?: number; byAttempt?: Record<string, number> };
        // Reservations left unsettled by a killed process count as spent: those containers did run.
        this.spent = (Number(j.spent) || 0) + (Number(j.reserved) || 0);
        this.byAttempt = { ...(j.byAttempt ?? {}) };
        if (Number(j.reserved) > 0) this.byAttempt[UNSETTLED] = (this.byAttempt[UNSETTLED] ?? 0) + Number(j.reserved);
      } catch { /* a torn file: start from the backup below */ if (existsSync(`${file}.bak`)) this.spent = Number(JSON.parse(readFileSync(`${file}.bak`, "utf8")).spent) || 0; }
    }
  }
  private save() {
    if (!this.file) return;
    const body = JSON.stringify({ spent: this.spent, reserved: this.reserved, limit: this.limitUsd, byAttempt: this.byAttempt });
    writeFileSync(`${this.file}.tmp`, body);
    renameSync(`${this.file}.tmp`, this.file); // atomic: a kill mid-write leaves the old or the new file
    writeFileSync(`${this.file}.bak`, body);
  }
  // Container start-up and upload add a few minutes on top of the run's own timeout.
  worstCase(r: RemoteRun): number { return containerCost(r.gpu, r.timeoutS + 300); }
  // Concurrent runs: each admitted run reserves its worst case until it settles, so parallel stages
  // cannot together overshoot the limit.
  reserved = 0;
  admit(r: RemoteRun): string | null {
    const w = this.worstCase(r);
    if (this.spent + this.reserved + w > this.limitUsd) return `budget: worst case $${w.toFixed(2)} exceeds the remaining $${(this.limitUsd - this.spent - this.reserved).toFixed(2)} (incl. $${this.reserved.toFixed(2)} reserved by running jobs)`;
    this.reserved += w;
    this.save();
    return null;
  }
  release(r: RemoteRun) { this.reserved = Math.max(0, this.reserved - this.worstCase(r)); this.save(); }
  charge(usd: number) {
    this.spent += usd;
    const k = costScope.getStore() ?? "(outside any node)";
    this.byAttempt[k] = (this.byAttempt[k] ?? 0) + usd;
    this.save();
  }
  remaining(): number { return this.limitUsd - this.spent - this.reserved; }
}

async function call(req: Record<string, unknown>, timeoutMs: number): Promise<Record<string, unknown>> {
  const env: Record<string, string> = {
    PATH: process.env.PATH ?? "/usr/bin:/bin", HOME: process.env.HOME ?? "/tmp", PYTHONPATH: MODAL_SITE,
    // The token secret is injected by the network proxy; the SDK only needs a placeholder to start.
    MODAL_TOKEN_ID: process.env.MODAL_TOKEN_ID ?? "", MODAL_TOKEN_SECRET: process.env.MODAL_TOKEN_SECRET ?? "",
    ...(process.env.HTTPS_PROXY ? { HTTPS_PROXY: process.env.HTTPS_PROXY } : {}),
    ...(process.env.SSL_CERT_FILE ? { SSL_CERT_FILE: process.env.SSL_CERT_FILE, GRPC_DEFAULT_SSL_ROOTS_FILE_PATH: process.env.SSL_CERT_FILE } : {}),
    REQ: JSON.stringify(req), RUNNER: process.env.MLH_MODAL_RUNNER ?? RUNNER, ...(process.env.MLH_MODAL_RUNNER ? Object.fromEntries(Object.entries(process.env).filter(([k]) => k.startsWith("MLH_FAKE_")) as [string, string][]) : {}),
  };
  // MLH_MODAL_RUNNER swaps in a fake runner for offline end-to-end and failure-injection tests.
  const r = await runProc(["sh", "-c", 'printf "%s" "$REQ" | python3 "$RUNNER"'], { cwd: "/tmp", env, timeoutMs });
  try {
    return JSON.parse(r.stdout.trim().split("\n").pop() ?? "") as Record<string, unknown>;
  } catch {
    return { ok: false, error: `runner printed no JSON: ${r.stderr.slice(-500)}` };
  }
}

// ---- Gateway path (HTTPS). Used when MLH_GATEWAY_URL is set: the network proxy carries plain HTTPS and
// injects the gateway token as the Authorization header; Modal's own credentials stay on Modal.
const gateway = () => process.env.MLH_GATEWAY_URL?.replace(/\/$/, "");

async function gw(path: string, init: RequestInit = {}): Promise<Record<string, unknown>> {
  // "proxy-injected" is a placeholder: the proxy replaces the Authorization header with the saved token.
  const r = await fetch(`${gateway()}${path}`, { ...init, headers: { Authorization: "Bearer proxy-injected", "Content-Type": "application/json", ...(init.headers ?? {}) }, signal: AbortSignal.timeout(120_000) });
  const text = await r.text();
  if (!r.ok) return { ok: false, error: `gateway ${r.status}: ${text.slice(0, 300)}` };
  try { return { ok: true, ...(JSON.parse(text) as Record<string, unknown>) }; } catch { return { ok: false, error: `gateway returned non-JSON: ${text.slice(0, 200)}` }; }
}

async function tgzBase64(dir: string): Promise<string> {
  const r = await runProc(["sh", "-c", 'tar --exclude=.git --exclude=__pycache__ --exclude=.mlh_in -czf - -C "$D" . | base64 -w0'], { cwd: "/tmp", env: { PATH: process.env.PATH ?? "/usr/bin:/bin", D: dir }, timeoutMs: 300_000 });
  if (r.code !== 0) throw new Error(`packing ${dir} failed: ${r.stderr.slice(-300)}`);
  return r.stdout.trim();
}

// Starts a job and polls until it finishes. `onStarted` receives the job id so the caller can journal
// it: after a crash, `resumeJob(id)` collects the result of a job that kept running on Modal.
export async function gatewayRun(run: RemoteRun, budget: Budget, onStarted?: (id: string) => void, resumeId?: string): Promise<RemoteResult> {
  const refused = resumeId ? null : budget.admit(run);
  if (refused) return { exitCode: -1, stdout: "", stderr: refused, elapsedS: 0, costUsd: 0, files: {}, category: "policy_denied" };
  let id = resumeId;
  if (!id) {
    const s = await gw("/start", { method: "POST", body: JSON.stringify({ repo_tgz: await tgzBase64(run.repo), cmd: run.cmd, gpu: run.gpu, timeout_s: run.timeoutS, network: run.network ?? false, env: run.env ?? {}, setup: run.setup, outputs: run.outputs ?? [], image: run.image }) });
    if (!s.ok) return { exitCode: -1, stdout: "", stderr: String(s.error), elapsedS: 0, costUsd: 0, files: {}, category: /401|403/.test(String(s.error)) ? "precondition" : "sandbox_transient" };
    id = String(s.id);
    onStarted?.(id);
  }
  const deadline = Date.now() + (run.timeoutS + 1200) * 1000;
  for (;;) {
    const r = await gw(`/result?id=${encodeURIComponent(id)}`);
    if (r.ok && r.done) {
      if (r.error) return { exitCode: -1, stdout: "", stderr: String(r.error), elapsedS: 0, costUsd: 0, files: {}, category: /Timeout/i.test(String(r.error)) ? "timeout" : "sandbox_transient" };
      const elapsed = Number(r.elapsed_s), cost = containerCost(run.gpu, elapsed);
      budget.charge(cost);
      const code = Number(r.exit_code), stderr = String(r.stderr);
      const category = classifyExit(code, stderr);
      return { exitCode: code, stdout: String(r.stdout), stderr, elapsedS: elapsed, costUsd: cost, files: (r.files ?? {}) as Record<string, string>, category };
    }
    if (Date.now() > deadline) { await gw(`/cancel?id=${encodeURIComponent(id)}`, { method: "POST" }); return { exitCode: -1, stdout: "", stderr: "job exceeded its deadline and was cancelled", elapsedS: run.timeoutS, costUsd: 0, files: {}, category: "timeout" }; }
    await new Promise((res) => setTimeout(res, 15_000));
  }
}

export async function gatewayCheck(): Promise<{ ok: boolean; error?: string }> {
  const r = await gw("/health");
  return r.ok ? { ok: true } : { ok: false, error: String(r.error) };
}

export async function modalCheck(): Promise<{ ok: boolean; error?: string }> {
  const r = await call({ op: "check" }, 120_000);
  return r.ok ? { ok: true } : { ok: false, error: String(r.error) };
}

export async function modalRun(run: RemoteRun, budget: Budget): Promise<RemoteResult> {
  const refused = budget.admit(run);
  if (refused) return { exitCode: -1, stdout: "", stderr: refused, elapsedS: 0, costUsd: 0, files: {}, category: "policy_denied" };
  try { return await modalRunAdmitted(run, budget); } finally { budget.release(run); }
}

async function modalRunAdmitted(run: RemoteRun, budget: Budget): Promise<RemoteResult> {
  // An image with new pip packages takes a few minutes to build the first time.
  const r = await call({ op: "run", cache_volume: budget.volume, repo: run.repo, cmd: run.cmd, gpu: run.gpu, timeout_s: run.timeoutS, network: run.network ?? false, env: run.env ?? {}, outputs: run.outputs ?? [], image: run.image, pip: run.pip ?? [], restore: run.restore ?? [], persist: run.persist ?? [], domains: run.domains ?? [] }, (run.timeoutS + 2400) * 1000);
  if (!r.ok) {
    const e = String(r.error);
    const head = e.split("\n")[0]!; // classify the exception, not a build log that mentions "tokenizers"
    // What ran before the failure is charged.
    const ranS = Number(r.elapsed_s) || 0;
    if (ranS) budget.charge(containerCost(run.gpu, ranS));
    return { exitCode: -1, stdout: "", stderr: e, elapsedS: ranS, costUsd: containerCost(run.gpu, ranS), files: {}, category: classifyModalError(head) };
  }
  const elapsed = Number(r.elapsed_s), cost = containerCost(run.gpu, elapsed);
  budget.charge(cost);
  const code = Number(r.exit_code), stderr = String(r.stderr);
  const category: FailureCategory | null = code === 0 ? null : code === 124 || code === -1 ? "timeout" : /CUDA out of memory|OutOfMemoryError/.test(stderr) ? "resource_limit" : "experiment_error";
  return { exitCode: code, stdout: String(r.stdout), stderr, elapsedS: elapsed, costUsd: cost, files: (r.files ?? {}) as Record<string, string>, category };
}
