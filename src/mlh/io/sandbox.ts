// Isolated execution of untrusted repository code. Each sandbox is a fresh copy
// of the source tree. Isolation (Linux user+net+mount namespaces): no network,
// read-only root filesystem, private /tmp, /home and /root hidden; the copy is
// the only writable path (at /mnt) and the trusted probes are mounted read-only
// at /srv. Resource limits via prlimit, wall-clock timeout via process-group kill.
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import type { FailureCategory } from "../core/types.ts";
import { runProc, type ProcResult } from "./proc.ts";

export type Isolation = "namespace" | "none";
export const PROBES = join(import.meta.dirname, "../probes");

export interface Limits {
  cpuSec: number;
  memMB: number;
  fileMB: number;
  timeoutMs: number;
}
export const DEFAULT_LIMITS: Limits = { cpuSec: 120, memMB: 4096, fileMB: 512, timeoutMs: 180_000 };

export class SkillError extends Error {
  constructor(readonly category: FailureCategory, message: string) {
    super(message);
  }
}

export interface SandboxResult extends ProcResult {
  category: FailureCategory | null;
}

const BASE_ENV = {
  PATH: "/usr/local/bin:/usr/bin:/bin",
  HOME: "/tmp",
  LANG: "C.UTF-8",
  PYTHONHASHSEED: "0",
  PYTHONDONTWRITEBYTECODE: "1",
  OMP_NUM_THREADS: "1",
  OPENBLAS_NUM_THREADS: "1",
  MKL_NUM_THREADS: "1",
};

// unshare --fork does not pass a signal death through (it exits 1), so the
// inner shell prints the command's real status as MLH_EXIT=<code>.
export function classify(r: ProcResult): FailureCategory | null {
  if (r.timedOut) return "timeout";
  const inner = /^MLH_EXIT=(\d+)$/m.exec(r.stderr);
  const code = inner ? Number(inner[1]) : r.code;
  const sig = r.signal ?? (code !== null && code > 128 ? code - 128 : null);
  if (sig === "SIGXCPU" || sig === "SIGXFSZ" || sig === "SIGKILL" || sig === 24 || sig === 25 || sig === 9 || /MemoryError|Cannot allocate memory|File too large/.test(r.stderr)) return "resource_limit";
  if (code === 0) return null;
  if (!inner && r.code !== 0 && /^(unshare|mount|prlimit|sh):/m.test(r.stderr)) return "sandbox_transient";
  return "experiment_error";
}

export interface SandboxStats {
  runs: number;
  ms: number;
}

export class Sandbox {
  private constructor(
    readonly dir: string,
    readonly isolation: Isolation,
    readonly limits: Limits,
    private readonly stats: SandboxStats,
    private readonly fault: () => boolean,
  ) {}

  static create(parent: string, name: string, source: string, o: { isolation: Isolation; limits: Limits; stats: SandboxStats; fault?: () => boolean }): Sandbox {
    const dir = join(parent, name.replace(/[^\w.-]+/g, "_"));
    rmSync(dir, { recursive: true, force: true });
    mkdirSync(parent, { recursive: true });
    cpSync(source, dir, { recursive: true, filter: (p) => !/\/\.git(\/|$)/.test(p) });
    return new Sandbox(dir, o.isolation, o.limits, o.stats, o.fault ?? (() => false));
  }

  private wrap(cmd: string[]): string[] {
    const l = this.limits;
    const lim = ["prlimit", `--cpu=${l.cpuSec}`, `--as=${l.memMB * 1024 * 1024}`, `--fsize=${l.fileMB * 1024 * 1024}`, "--nofile=512", "--"];
    if (this.isolation === "none") return [...lim, ...cmd];
    // Bind mounts first: the tmpfs over /home would otherwise hide the sandbox copy.
    const script = 'set -e; mount --bind "$W" /mnt; mount --bind "$P" /srv; mount -o remount,bind,ro /srv; ' +
      "mount -t tmpfs none /tmp; mount -t tmpfs none /home; mount -t tmpfs none /root; mount -o remount,bind,ro /; " +
      'cd /mnt; "$@"; s=$?; echo "MLH_EXIT=$s" >&2; exit $s';
    return ["unshare", "-r", "-n", "-m", "--fork", "sh", "-c", script, "sh", ...lim, ...cmd];
  }

  async run(cmd: string[], env: Record<string, string> = {}): Promise<SandboxResult> {
    if (this.fault()) throw new SkillError("sandbox_transient", "injected transient sandbox failure");
    const full = { ...BASE_ENV, ...env, W: this.dir, P: PROBES };
    const r = await runProc(this.wrap(cmd), { cwd: this.dir, env: full, timeoutMs: this.limits.timeoutMs });
    this.stats.runs++; this.stats.ms += r.ms;
    return { ...r, category: classify(r) };
  }

  // Runs a trusted probe and parses its JSON; probes never import repository code.
  async probe(args: string[]): Promise<Record<string, unknown>> {
    const script = this.isolation === "none" ? join(PROBES, "probe.py") : "/srv/probe.py";
    const r = await this.run(["python3", "-I", script, ...args]);
    if (r.category) throw new SkillError(r.category === "experiment_error" ? "contract_violation" : r.category, `probe ${args[0]} failed: ${r.stderr.slice(-400)}`);
    try {
      return JSON.parse(r.stdout.trim().split("\n").pop() ?? "") as Record<string, unknown>;
    } catch {
      throw new SkillError("contract_violation", `probe ${args[0]} printed no JSON: ${r.stdout.slice(-200)}`);
    }
  }

  path(rel: string): string {
    return join(this.dir, rel);
  }

  readJson(rel: string): unknown | undefined {
    const p = this.path(rel);
    return existsSync(p) ? JSON.parse(readFileSync(p, "utf8")) : undefined;
  }

  remove(): void {
    rmSync(this.dir, { recursive: true, force: true });
  }
}

// Deterministic evidence for authorization: proves the isolation holds on this
// machine before any repository code runs, rather than assuming it.
export async function selfTest(scratch: string): Promise<{ isolation: Isolation; checks: { name: string; ok: boolean }[] }> {
  mkdirSync(scratch, { recursive: true });
  const stats = { runs: 0, ms: 0 };
  const sb = Sandbox.create(scratch, "selftest", PROBES, { isolation: "namespace", limits: { ...DEFAULT_LIMITS, timeoutMs: 20_000 }, stats });
  const code = [
    "import os, socket, json",
    "r = {}",
    "try:\n open('/etc/mlh_x','w'); r['root_readonly'] = False\nexcept OSError: r['root_readonly'] = True",
    "try:\n socket.create_connection(('1.1.1.1', 443), timeout=2); r['no_network'] = False\nexcept OSError: r['no_network'] = True",
    "r['home_hidden'] = len(os.listdir('/home')) == 0",
    "open('/mnt/ok','w').write('1'); r['workdir_writable'] = True",
    "print(json.dumps(r))",
  ].join("\n");
  const r = await sb.run(["python3", "-I", "-c", code]);
  sb.remove();
  if (r.code !== 0) return { isolation: "none", checks: [{ name: `namespace sandbox unavailable: ${r.stderr.trim().slice(0, 200)}`, ok: false }] };
  const res = JSON.parse(r.stdout.trim()) as Record<string, boolean>;
  const checks = Object.entries(res).map(([name, ok]) => ({ name, ok }));
  return { isolation: checks.every((c) => c.ok) ? "namespace" : "none", checks };
}
