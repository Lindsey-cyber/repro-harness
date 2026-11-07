// mlh: the ML-experiment harness CLI. Never waits for typed input: anything
// that needs a human is held and listed, and `mlh approve` releases it.
import { existsSync } from "node:fs";
import { join } from "node:path";
import { replay } from "./core/dag.ts";
import { compact } from "./core/report.ts";
import { Journal } from "./io/journal.ts";
import { runHarness, type Mode } from "./orchestrator.ts";
import { SKILLS } from "./skills/index.ts";

const [cmd, ...rest] = process.argv.slice(2);
const flag = (name: string): string | undefined => { const i = rest.indexOf(`--${name}`); return i >= 0 ? rest[i + 1] : undefined; };
const positional = rest.filter((a, i) => !a.startsWith("--") && !rest[i - 1]?.startsWith("--"));

async function main(): Promise<void> {
  switch (cmd) {
    case "run": {
      const runDir = flag("run-dir") ?? join(".mlh", "runs", new Date().toISOString().replace(/[:.]/g, "-"));
      const r = await runHarness({
        runDir, source: positional[0]!, objective: flag("objective") ?? "Reproduce the reported result and check the experiment for errors.",
        mode: (flag("mode") as Mode | undefined) ?? "harness", concurrency: Number(flag("concurrency") ?? 3),
        ...(flag("crash-after") ? { crashAfter: Number(flag("crash-after")) } : {}),
        ...(flag("faults") ? { faults: { rate: Number(flag("faults")), seed: 1 } } : {}),
        ...(flag("trust") ? { trustFile: flag("trust")! } : {}),
      });
      console.log(`${r.reportPath}\nconfirmed: ${r.summary.confirmed.map((f) => f.category).join(", ") || "none"}; unverified: ${r.summary.unverified.length}; held: ${r.summary.held.length}`);
      break;
    }
    case "resume": {
      const r = await runHarness({ runDir: positional[0]!, ...(flag("trust") ? { trustFile: flag("trust")! } : {}) });
      console.log(`${r.reportPath}\nconfirmed: ${r.summary.confirmed.map((f) => f.category).join(", ") || "none"}`);
      break;
    }
    case "status": {
      process.stdout.write(compact(replay(new Journal(join(positional[0]!, "journal.jsonl")).read())));
      break;
    }
    case "approve": {
      const [runDir, id] = positional;
      const j = new Journal(join(runDir!, "journal.jsonl"));
      const s = replay(j.read());
      if (s.nodes.get(id!)?.status !== "held") throw new Error(`${id} is not held`);
      j.append({ t: "approve", at: new Date().toISOString(), id: id!, by: "human" });
      console.log(`approved ${id}; run \`npm run mlh -- resume ${runDir}\``);
      break;
    }
    case "skills": {
      for (const s of SKILLS) console.log(`${s.id}@${s.version} [${s.purpose}] ${s.summary}`);
      break;
    }
    case "bench": {
      const { runBench } = await import("./bench.ts");
      await runBench({ modes: (flag("modes") ?? "naive,workflow,harness").split(",") as never, faults: Number(flag("faults") ?? 0), out: flag("out") ?? "bench/ml/results", ...(flag("tasks") ? { tasks: flag("tasks")!.split(",") } : {}), ...(flag("check") ? { check: flag("check")! } : {}), crash: rest.includes("--crash") });
      break;
    }
    default:
      console.log("usage: mlh run <repo|url> --objective <text> [--mode harness|workflow] | resume <runDir> | status <runDir> | approve <runDir> <node> | skills | bench [--modes ..] [--faults r] [--check baseline.json]");
      if (cmd && !existsSync(cmd)) process.exitCode = 2;
  }
}
main().catch((e) => { console.error(String(e?.stack ?? e)); process.exit(1); });
