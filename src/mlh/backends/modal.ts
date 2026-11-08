// Modal GPU execution backend. Repository code runs only inside a Modal Sandbox,
// never on this machine; this module uploads, launches and reads results through
// the Modal SDK (modal_runner.py). A dollar budget is enforced before every
// launch: a run whose worst case would exceed what is left is refused.
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { FailureCategory } from "../core/types.ts";
import { runProc } from "../io/proc.ts";

export const RUNNER = join(import.meta.dirname, "modal_runner.py");
export const MODAL_SITE = join(import.meta.dirname, "../../../.mlh/tools/modal-site");

// USD per GPU-hour (Modal list prices as of 2026-11; override with MLH_GPU_PRICE_<GPU>).
const PRICE: Record<string, number> = { T4: 0.59, L4: 0.8, A10G: 1.1, "A100-40GB": 2.1, "A100-80GB": 2.5, H100: 3.95 };
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

export class Budget {
  spent = 0;
  // With a file, spending survives a crash: a resumed run continues from what it already spent.
  constructor(readonly limitUsd: number, private readonly file?: string) {
    if (file && existsSync(file)) this.spent = Number(JSON.parse(readFileSync(file, "utf8")).spent) || 0;
  }
  // Container start-up and upload add a few minutes on top of the run's own timeout.
  worstCase(r: RemoteRun): number { return r.gpu ? (gpuPrice(r.gpu) * (r.timeoutS + 300)) / 3600 : (0.05 * r.timeoutS) / 3600; }
  // Concurrent runs: each admitted run reserves its worst case until it settles, so parallel stages
  // cannot together overshoot the limit.
  reserved = 0;
  admit(r: RemoteRun): string | null {
    const w = this.worstCase(r);
    if (this.spent + this.reserved + w > this.limitUsd) return `budget: worst case $${w.toFixed(2)} exceeds the remaining $${(this.limitUsd - this.spent - this.reserved).toFixed(2)} (incl. $${this.reserved.toFixed(2)} reserved by running jobs)`;
    this.reserved += w;
    return null;
  }
  release(r: RemoteRun) { this.reserved = Math.max(0, this.reserved - this.worstCase(r)); }
  charge(usd: number) {
    this.spent += usd;
    if (this.file) writeFileSync(this.file, JSON.stringify({ spent: this.spent, limit: this.limitUsd }));
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
    REQ: JSON.stringify(req), RUNNER,
  };
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
      const elapsed = Number(r.elapsed_s), cost = run.gpu ? (gpuPrice(run.gpu) * elapsed) / 3600 : 0;
      budget.charge(cost);
      const code = Number(r.exit_code), stderr = String(r.stderr);
      const category: FailureCategory | null = code === 0 ? null : code === 124 || code === -1 ? "timeout" : /CUDA out of memory|OutOfMemoryError/.test(stderr) ? "resource_limit" : "experiment_error";
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
  const r = await call({ op: "run", repo: run.repo, cmd: run.cmd, gpu: run.gpu, timeout_s: run.timeoutS, network: run.network ?? false, env: run.env ?? {}, outputs: run.outputs ?? [], image: run.image, pip: run.pip ?? [], restore: run.restore ?? [], persist: run.persist ?? [], domains: run.domains ?? [] }, (run.timeoutS + 2400) * 1000);
  if (!r.ok) {
    const e = String(r.error);
    return { exitCode: -1, stdout: "", stderr: e, elapsedS: 0, costUsd: 0, files: {}, category: /Timeout|timed out|deadline/i.test(e) ? "timeout" : /Auth|Token/i.test(e) ? "precondition" : "sandbox_transient" };
  }
  const elapsed = Number(r.elapsed_s), cost = run.gpu ? (gpuPrice(run.gpu) * elapsed) / 3600 : 0;
  budget.charge(cost);
  const code = Number(r.exit_code), stderr = String(r.stderr);
  const category: FailureCategory | null = code === 0 ? null : code === 124 || code === -1 ? "timeout" : /CUDA out of memory|OutOfMemoryError/.test(stderr) ? "resource_limit" : "experiment_error";
  return { exitCode: code, stdout: String(r.stdout), stderr, elapsedS: elapsed, costUsd: cost, files: (r.files ?? {}) as Record<string, string>, category };
}
