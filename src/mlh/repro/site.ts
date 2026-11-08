// Package sites: content-addressed directories of installed packages, keyed by
// the full set of requirement specifiers. Built once in a network-enabled
// sandbox (install only), then mounted read-only at /opt for every run.
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { authorize, type Evidence } from "../core/authz.ts";
import { runProc } from "../io/proc.ts";
import { DEFAULT_LIMITS, Sandbox, type Isolation, type SandboxStats } from "../io/sandbox.ts";

const SPEC = /^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?(\s*(===?|>=|<=|~=|!=|<|>)\s*[A-Za-z0-9.*+!_-]+(\s*,\s*(===?|>=|<=|~=|!=|<|>)\s*[A-Za-z0-9.*+!_-]+)*)?$/;
const TORCH = /^(torch|torchvision|torchaudio|torchtext|torch[-_][a-z0-9_-]+)\b/i;
export const PYTORCH_CPU = "https://download.pytorch.org/whl/cpu";

// Deterministic evidence for install_packages: plain requirement specifiers only,
// no URLs, local paths, VCS links or pip options.
export function checkSpecs(specs: string[]): Evidence {
  const bad = specs.filter((s) => !SPEC.test(s.trim()));
  return { kind: "deterministic", check: bad.length ? `package specs rejected: ${bad.join(", ")}` : "package specs are plain requirement specifiers", ok: bad.length === 0 && specs.length > 0 };
}

const norm = (s: string) => s.trim().replace(/\s+/g, "").toLowerCase();
const keyOf = (specs: string[]) => createHash("sha256").update([...new Set(specs.map(norm))].sort().join("\n")).digest("hex").slice(0, 16);

export interface SiteManager {
  install(packages: string[], currentSite: string | undefined): Promise<{ site?: string; message: string; specs: string[] }>;
  specsOf(site: string): string[];
}

export function siteManager(root: string, o: { isolation: Isolation; stats: SandboxStats }): SiteManager {
  mkdirSync(root, { recursive: true });
  const specsOf = (site: string): string[] => { const f = join(site, ".mlh-specs.json"); return existsSync(f) ? JSON.parse(readFileSync(f, "utf8")) : []; };
  return {
    specsOf,
    async install(packages, currentSite) {
      const ev = checkSpecs(packages);
      // Network is used only to install, inside a sandbox that runs pip and nothing from the repository.
      const d = authorize("install_packages", [ev]);
      if (!d.allow) return { message: `refused: ${d.reason}`, specs: [] };
      const specs = [...new Set([...(currentSite ? specsOf(currentSite) : []), ...packages.map((p) => p.trim())])];
      const key = keyOf(specs);
      const site = join(root, key);
      const done = join(site, ".mlh-specs.json");
      if (existsSync(done)) return { site, message: `installed (cached site ${key}): ${specs.join(" ")}`, specs };
      const lock = join(root, `${key}.lock`);
      for (let i = 0; ; i++) {
        try { mkdirSync(lock); break; } catch { if (existsSync(done)) return { site, message: `installed (cached site ${key})`, specs }; if (i > 600) throw new Error("site lock timeout"); await new Promise((r) => setTimeout(r, 1000)); }
      }
      try {
        rmSync(site, { recursive: true, force: true });
        mkdirSync(site, { recursive: true });
        const cache = join(root, "pip-cache");
        mkdirSync(cache, { recursive: true });
        const sb = Sandbox.attach(cache, { isolation: o.isolation, limits: { ...DEFAULT_LIMITS, cpuSec: 1800, memMB: 8192, fileMB: 4096, timeoutMs: 1_200_000 }, stats: o.stats, site, network: true });
        const pip = ["python3", "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--no-input", "--target", "/opt", "--cache-dir", "/mnt"];
        // torch-family packages always come from the CPU index (never CUDA builds); everything else from PyPI.
        const torch = specs.filter((s) => TORCH.test(s)), rest = specs.filter((s) => !TORCH.test(s));
        let log = "";
        for (const [group, extra] of [[torch, ["--index-url", PYTORCH_CPU, "--extra-index-url", "https://pypi.org/simple"]], [rest, []]] as const) {
          if (!group.length) continue;
          const r = await sb.run([...pip, ...extra, ...group]);
          log += r.stderr.replace(/^MLH_EXIT=\d+$/m, "").split("\n").filter((l) => !/WARNING: Running pip as|pip as the 'root'|^\s*$/.test(l)).slice(-15).join("\n");
          if (r.category) { rmSync(site, { recursive: true, force: true }); return { message: `install failed (${r.category}) for ${group.join(" ")}:\n${log.slice(-2500)}`, specs: [] }; }
        }
        writeFileSync(done, JSON.stringify(specs));
        // Sites differ by a package or two but each carries its own copy of torch (~1 GB): identical files are
        // hard-linked to the most similar existing site. Without this, 29 sites filled 24 GB.
        const peers = readdirSync(root).filter((d) => d !== key && existsSync(join(root, d, ".mlh-specs.json")));
        const overlap = (d: string) => { const o = new Set(specsOf(join(root, d)).map(norm)); return specs.filter((x) => o.has(norm(x))).length; };
        const best = peers.sort((a, b) => overlap(b) - overlap(a))[0];
        if (best && overlap(best) > 0) await runProc(["hardlink", "-c", "-q", site, join(root, best)], { cwd: root, env: { PATH: process.env.PATH ?? "/usr/bin:/bin" }, timeoutMs: 600_000 });
        return { site, message: `installed: ${specs.join(" ")}${log ? `\n${log.slice(-1500)}` : ""}`, specs };
      } finally {
        rmSync(lock, { recursive: true, force: true });
      }
    },
  };
}
