// The only place in src/mlh that starts processes (tests/mlh-structure.test.ts).
import { spawn } from "node:child_process";

export interface ProcResult {
  code: number | null;
  signal: string | null;
  stdout: string;
  stderr: string;
  ms: number;
  timedOut: boolean;
}

const CAP = 1 << 20;

export function runProc(cmd: string[], opts: { cwd: string; env: Record<string, string>; timeoutMs: number }): Promise<ProcResult> {
  const t0 = performance.now();
  return new Promise((resolve) => {
    const child = spawn(cmd[0]!, cmd.slice(1), { cwd: opts.cwd, env: opts.env, detached: true, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "", stderr = "", timedOut = false;
    child.stdout.on("data", (d: Buffer) => { if (stdout.length < CAP) stdout += d.toString(); });
    child.stderr.on("data", (d: Buffer) => { if (stderr.length < CAP) stderr += d.toString(); });
    const timer = setTimeout(() => {
      timedOut = true;
      try { process.kill(-child.pid!, "SIGKILL"); } catch { /* already gone */ }
    }, opts.timeoutMs);
    const done = (code: number | null, signal: string | null) => {
      clearTimeout(timer);
      resolve({ code, signal, stdout, stderr, ms: performance.now() - t0, timedOut });
    };
    child.on("error", (e) => { stderr += String(e); done(127, null); });
    child.on("close", (code, signal) => done(code, signal));
  });
}
