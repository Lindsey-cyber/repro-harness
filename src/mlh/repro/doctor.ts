// Platform conformance probe, run before a campaign round. The agents are judged on the
// platform the harness gives them, so the harness proves that platform first, cheaply:
// every base image is asked what it actually contains (the label must say the same), a
// container can start, and the LLM provider has balance. A failed probe stops the round
// before any paper is spent on it. Results: .mlh/doctor.json (valid 24 h).
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { Budget, modalRun } from "../backends/modal.ts";
import { IMAGES } from "./gpu.ts";

export interface Check { name: string; ok: boolean; detail: string }
export interface Doctor { at: string; ok: boolean; checks: Check[] }

const ROOT = process.env.MLH_ROOT ? resolve(process.env.MLH_ROOT) : resolve(import.meta.dirname, "../../..");
const FILE = join(ROOT, ".mlh", "doctor.json");

// "py3.8-torch1.12-cu113" -> { py: "3.8", torch: "1.12" }. Pure.
export function labelOf(key: string): { py: string; torch: string } | undefined {
  const m = /^py(\d+\.\d+)-torch(\d+\.\d+)/.exec(key);
  return m ? { py: m[1]!, torch: m[2]! } : undefined;
}

// The probe's output line "MLH_PROBE py=3.7.13 torch=1.12.1+cu113" against the label. Pure.
export function imageCheck(key: string, output: string): Check {
  const want = labelOf(key), m = /MLH_PROBE py=(\d+\.\d+)\S* torch=(\d+\.\d+)/.exec(output);
  if (!m) return { name: `image ${key}`, ok: false, detail: `probe printed no versions: ${output.slice(-200)}` };
  const ok = !!want && want.py === m[1] && want.torch === m[2];
  return { name: `image ${key}`, ok, detail: ok ? `python ${m[1]}, torch ${m[2]}` : `label says python ${want?.py}/torch ${want?.torch}, image has python ${m[1]}/torch ${m[2]}` };
}

export function doctorFresh(maxAgeMs = 24 * 3600_000): Doctor | undefined {
  if (!existsSync(FILE)) return undefined;
  const d = JSON.parse(readFileSync(FILE, "utf8")) as Doctor;
  return Date.now() - Date.parse(d.at) < maxAgeMs ? d : undefined;
}

export async function runDoctor(o: { llmBalance: () => Promise<number | undefined>; minLlmBalance: number }): Promise<Doctor> {
  const dir = join(ROOT, ".mlh", "campaign", "_doctor", new Date().toISOString().slice(0, 10));
  mkdirSync(join(dir, "repo"), { recursive: true });
  const budget = new Budget(2, join(dir, "budget.json")); // counted by the campaign's Modal ledger
  const probe = 'python -c "import sys, torch; print(\'MLH_PROBE py=%d.%d.%d torch=%s\' % (sys.version_info[:3] + (torch.__version__,)))"';
  const checks: Check[] = await Promise.all(Object.entries(IMAGES).map(async ([key, image]) => {
    const r = await modalRun({ repo: join(dir, "repo"), cmd: ["bash", "-lc", probe], timeoutS: 900, network: false, image }, budget);
    return r.exitCode === 0 ? imageCheck(key, r.stdout) : { name: `image ${key}`, ok: false, detail: `${r.category}: ${r.stderr.split("\n")[0]!.slice(0, 200)}` };
  }));
  const bal = await o.llmBalance();
  checks.push({ name: "llm balance", ok: bal === undefined || bal >= o.minLlmBalance, detail: bal === undefined ? "unknown (not checked)" : `${bal} (minimum ${o.minLlmBalance})` });
  const d: Doctor = { at: new Date().toISOString(), ok: checks.every((c) => c.ok), checks };
  writeFileSync(FILE, JSON.stringify(d, null, 1));
  return d;
}

export const renderDoctor = (d: Doctor) => [`doctor ${d.ok ? "PASS" : "FAIL"} (${d.at})`, ...d.checks.map((c) => `  ${c.ok ? "ok  " : "FAIL"} ${c.name}: ${c.detail}`)].join("\n");
