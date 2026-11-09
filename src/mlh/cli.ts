// mlh: the ML-experiment harness CLI. Never waits for typed input: anything
// that needs a human is held and listed, and `mlh approve` releases it.
import { existsSync } from "node:fs";
import { join } from "node:path";
import { replay } from "./core/dag.ts";
import { compact } from "./core/report.ts";
import { Journal } from "./io/journal.ts";
import { runHarness, type Mode } from "./orchestrator.ts";
import { Llm } from "./io/llm.ts";
import { SKILLS } from "./skills/registry.ts";

const [cmd, ...rest] = process.argv.slice(2);
const flag = (name: string): string | undefined => { const i = rest.indexOf(`--${name}`); return i >= 0 ? rest[i + 1] : undefined; };
// --llm: the human's standing approval to send repository content to the model provider.
const models = () => rest.includes("--llm") ? { llm: { proposer: new Llm({ model: flag("proposer") ?? "deepseek-flash", cacheDir: ".mlh/llm-cache" }), reviewer: new Llm({ model: flag("reviewer") ?? "deepseek-v4-pro", cacheDir: ".mlh/llm-cache" }) } } : {};
const BOOLEAN = new Set(["--llm", "--crash"]);
const positional = rest.filter((a, i) => !a.startsWith("--") && !(rest[i - 1]?.startsWith("--") && !BOOLEAN.has(rest[i - 1]!)));

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
        ...models(),
      });
      console.log(`${r.reportPath}\nconfirmed: ${r.summary.confirmed.map((f) => f.category).join(", ") || "none"}; unverified: ${r.summary.unverified.length}; held: ${r.summary.held.length}`);
      break;
    }
    case "repro": {
      // Reproduce a paper's claims: mlh repro <repo|https://github.com/...> --arxiv <id> --objective "..." --llm [--approve none|medium|all]
      const runDir = flag("run-dir") ?? join(".mlh", "runs", `repro-${new Date().toISOString().replace(/[:.]/g, "-")}`);
      const r = await runHarness({
        runDir, source: positional[0]!, objective: flag("objective") ?? "Reproduce the paper's main results with this repository.", mode: "repro",
        arxiv: flag("arxiv")!, approvePolicy: (flag("approve") as "none" | "medium" | "all" | undefined) ?? "medium", concurrency: Number(flag("concurrency") ?? 3),
        ...(flag("crash-after-steps") ? { crashAfterSteps: Number(flag("crash-after-steps")) } : {}),
        ...(flag("gpu-budget") ? { gpuBudgetUsd: Number(flag("gpu-budget")) } : {}),
        ...models(),
      });
      console.log(r.reportPath);
      break;
    }
    case "resume": {
      const r = await runHarness({ runDir: positional[0]!, ...(flag("trust") ? { trustFile: flag("trust")! } : {}), ...models() });
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
    case "doctor": {
      const { runDoctor, renderDoctor } = await import("./repro/doctor.ts");
      const { llmBalance } = await import("./repro/campaign.ts");
      const d = await runDoctor({ llmBalance, minLlmBalance: Number(process.env.MLH_MIN_LLM_BALANCE ?? 3) });
      console.log(renderDoctor(d));
      process.exitCode = d.ok ? 0 : 1;
      break;
    }
    case "cost": {
      const { costOfRound } = await import("./repro/campaign.ts");
      console.log(costOfRound(positional[0]!));
      break;
    }
    case "operator-patch": {
      // mlh operator-patch <runDir> <stage> <reason> -- <argv...>: a human-approved change of one stage's
      // command, journaled as an accepted high-risk patch with a human basis; only that stage (and what it
      // needs, and its claims) re-runs. Running debuggers for that stage's claims are skipped.
      const { replay, apply } = await import("./core/dag.ts");
      const flow = await import("./repro/flow.ts");
      const { readFileSync: rf, appendFileSync: af } = await import("node:fs");
      const dd = process.argv.indexOf("--");
      if (dd < 0) throw new Error("usage: mlh operator-patch <runDir> <stage> <reason> -- <argv...>");
      const [runDir, stage, ...why] = process.argv.slice(3, dd), argv = process.argv.slice(dd + 1);
      const j = `${runDir}/journal.jsonl`;
      const st = replay(rf(j, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)), 3);
      const at = new Date().toISOString(), ev: unknown[] = [];
      const push = (e: never) => { apply(st, e); ev.push(e); };
      for (const id of st.order) { const ns = st.nodes.get(id)!; if (ns.node.skill === "repro.debug" && ns.status !== "succeeded" && ns.status !== "skipped") push({ t: "skip", at, id, reason: `operator: superseded by an approved change (${why.join(" ")})` } as never); }
      const k = st.order.filter((x) => x.startsWith("repro.patch")).length + 1;
      const pid = `repro.patch:stage-${stage}:operator${k}`;
      push({ t: "node", at, node: { id: pid, skill: "repro.patch", deps: [], purpose: "setup", params: { claim: `stage-${stage}`, from: 0, debug: "" } }, why: "operator" } as never);
      push({ t: "start", at, id: pid, attempt: 1 } as never);
      push({ t: "ok", at, id: pid, output: { accepted: true, risk: "high", reasons: [why.join(" ")], basis: `human approval: ${why.join(" ")}`, argv: { [`stage-${stage}`]: argv } }, findings: [], artifacts: [] } as never);
      for (const node of flow.afterOk(st, pid)) if (!st.nodes.has(node.id)) push({ t: "node", at, node, why: `after ${pid}` } as never);
      for (const e of ev) af(j, JSON.stringify(e) + "\n");
      console.log((ev as { t: string; id?: string; node?: { id: string } }[]).map((e) => `${e.t} ${e.id ?? e.node?.id ?? ""}`).join("\n"));
      break;
    }
    case "reopen": {
      // mlh reopen <runDir> <id-prefix> <reason>: journaled operator correction after a harness fix.
      const { replay, reopenByOperator } = await import("./core/dag.ts");
      const { readFileSync: rf, appendFileSync: af } = await import("node:fs");
      const j = `${positional[0]}/journal.jsonl`;
      const st = replay(rf(j, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)), 3);
      const ev = reopenByOperator(st, positional[1]!, positional.slice(2).join(" ") || "harness fix", new Date().toISOString());
      for (const e of ev) af(j, JSON.stringify(e) + "\n");
      console.log(ev.map((e) => `${e.t} ${"id" in e ? e.id : ""}`).join("\n") || "nothing to reopen");
      break;
    }
    case "raise-budget": {
      // mlh raise-budget <runDir> <usd> <reason>: the owner's raise of one paper's GPU budget, recorded with its reason.
      const [runDir, usd, ...why] = positional;
      if (!(Number(usd) > 0) || !why.length || !existsSync(join(runDir!, "run.json"))) throw new Error("usage: mlh raise-budget <runDir> <usd> <reason>");
      const { appendFileSync: af } = await import("node:fs");
      af(join(runDir!, "budget-raises.jsonl"), JSON.stringify({ at: new Date().toISOString(), usd: Number(usd), reason: why.join(" ") }) + "\n");
      console.log(`raised by $${Number(usd)}; takes effect when the run (re)starts`);
      break;
    }
    case "excuse": {
      // mlh excuse <runDir> <node> <reason>: journaled operator ruling that the node's outcome came from a
      // harness defect (name the fix), so it is no evidence about the agent's trust. Changes no result.
      const [runDir, id, ...why] = positional;
      const j = new Journal(join(runDir!, "journal.jsonl"));
      if (!replay(j.read()).nodes.has(id!) || !why.length) throw new Error("usage: mlh excuse <runDir> <node> <reason naming the harness fix>");
      j.append({ t: "excuse", at: new Date().toISOString(), id: id!, reason: why.join(" ") });
      console.log(`excused ${id}`);
      break;
    }
    case "campaign-retro": {
      // Idempotent: the chain calls it for every finished round, whatever code version ran that round.
      const { retroRound } = await import("./repro/campaign.ts");
      console.log(retroRound(positional[0]!, flag("out") ?? "bench/complex/results") ? `retro written for ${positional[0]}` : `no retro needed for ${positional[0]}`);
      break;
    }
    case "campaign-status": {
      const { campaignStatus } = await import("./repro/campaign.ts");
      console.log(campaignStatus(positional[0] ?? "c1"));
      break;
    }
    case "solo": {
      // Single-agent baseline on one paper (remote GPUs, same budget, verified by rerun).
      const { runSolo } = await import("./repro/solo.ts");
      await runSolo({ source: positional[0]!, arxiv: flag("arxiv")!, objective: flag("objective")!, budgetUsd: Number(flag("gpu-budget") ?? 5), runDir: flag("run-dir")! });
      break;
    }
    case "campaign": {
      // Reproduce a queue of papers unattended: mlh campaign bench/complex/campaign.json --label c1 [--parallel 2]
      const { runCampaign } = await import("./repro/campaign.ts");
      await runCampaign({ config: positional[0]!, label: flag("label") ?? "campaign", parallel: Number(flag("parallel") ?? 2), out: flag("out") ?? "bench/complex/results", mode: (flag("mode") as "team" | "solo" | undefined) ?? "team" });
      break;
    }
    case "repro-bench": {
      const { runReproBench } = await import("./repro/bench.ts");
      await runReproBench({ modes: (flag("modes") ?? "script,solo,team").split(",") as never, out: flag("out") ?? "bench/repro/results", approve: (flag("approve") as never) ?? "all", providerFaults: Number(flag("provider-faults") ?? 0), ...(flag("crash-after-steps") ? { crashAfterSteps: Number(flag("crash-after-steps")) } : {}), ...(flag("tasks") ? { tasks: flag("tasks")!.split(",") } : {}), ...(flag("label") ? { label: flag("label")! } : {}), parallel: Number(flag("parallel") ?? 3) });
      break;
    }
    case "skills": {
      for (const s of SKILLS) console.log(`${s.id}@${s.version} [${s.purpose}] ${s.summary}`);
      break;
    }
    case "bench": {
      const { runBench } = await import("./bench.ts");
      await runBench({ modes: (flag("modes") ?? "naive,workflow,harness").split(",") as never, set: flag("set") ?? "tasks", faults: Number(flag("faults") ?? 0), out: flag("out") ?? "bench/ml/results", ...(flag("tasks") ? { tasks: flag("tasks")!.split(",") } : {}), ...(flag("check") ? { check: flag("check")! } : {}), crash: rest.includes("--crash"), parallel: Number(flag("parallel") ?? 6) });
      break;
    }
    default:
      console.log("usage: mlh run <repo|url> --objective <text> [--mode harness|workflow] | resume <runDir> | status <runDir> | approve <runDir> <node> | skills | bench [--modes ..] [--faults r] [--check baseline.json]");
      if (cmd && !existsSync(cmd)) process.exitCode = 2;
  }
}
main().catch((e) => { console.error(String(e?.stack ?? e)); process.exit(1); });
