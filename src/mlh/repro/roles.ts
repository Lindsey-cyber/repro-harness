// The specialized agents. Each role has only the tools its job needs, and what
// it may change is matched to the stakes: the environment agent installs but
// never edits code; the planner only reads; the debugger edits code in its own
// copy, and its change becomes the source only through patch review.
import type { Role } from "../agentloop.ts";

export const CAUSES = `Root-cause categories:
- env.missing_dependency: a needed package is not declared or not installed by the documented steps
- env.version_drift: the code needs an API that current library versions changed or removed
- env.path_or_config: wrong paths, missing files, package not on the import path, wrong config default
- run.crash: crashes for a non-environmental reason (shape error, bad index…)
- run.resource_limit: does not finish within CPU/memory/time limits as configured
- protocol.mismatch: runs fine but follows a different protocol than the paper (split, hyperparameters, epochs, metric definition)
- data.train_test_overlap, data.target_leak, pipeline.eval_path_mismatch, eval.metric_mismatch, ckpt.roundtrip_mismatch, repro.nondeterministic: silent defects that change results
- repro.claim_not_met: the paper's number is not reached even when run faithfully
- other:<slug>`;

const LIMITS = "Limits: CPU only, no GPU; each run at most 15 minutes and 8 GB; the sandbox is offline except through the install tool.";

const cause = { type: "object", properties: { category: { type: "string" }, evidence: { type: "string" } }, required: ["category", "evidence"] };

export const ENV_ROLE: Role = {
  name: "environment",
  maxSteps: 25, maxRunS: 300, wallMin: 45,
  tools: ["list_files", "read_file", "write_file", "run", "install"],
  writeCode: false,
  system: `You are the environment engineer of an ML reproduction team. Your only job: make the repository's experiments runnable in this sandbox. Runnable means the imports work and the documented launch starts and runs a few steps; it does not mean results or timings. Never benchmark or time training, and never run on CPU what the paper runs on GPU: a short check with gpu="T4" is the way to see it start. Each command is limited to 5 minutes. The harness smoke-tests your report by running launch.example_command from launch.cwd (relative to the repository root, i.e. /work), so give paths relative to that cwd. Datasets and checkpoints are never downloaded with your own commands: call fetch_data with {url, to: "/cache/data/<name>", unpack?: "tar"|"zip", sha256?} and the harness downloads them with provenance (it will download the same items again for every later run). Download only into /tmp if you need something just for a quick look. Read the README/requirements/setup files, install what is needed (install tool; torch-family packages come from the PyTorch CPU index automatically), and find out how to launch the experiments (working directory, PYTHONPATH or other environment variables). Verify by actually starting the documented experiment command for a short time (use timeout_s). You may NOT edit source code: if the code itself must change (removed library APIs, crashes), say so precisely in your report — a debugger will handle it. ${LIMITS}

${CAUSES}`,
  report: {
    description: "Finish with what you installed and how experiments must be launched.",
    parameters: {
      type: "object",
      properties: {
        runnable: { type: "boolean", description: "the documented experiment starts and runs without errors" },
        launch: { type: "object", properties: { cwd: { type: "string" }, env: { type: "object" }, example_command: { type: "array", items: { type: "string" } } } },
        problems: { type: "array", items: cause, description: "problems found, each with evidence (error lines)" },
        notes: { type: "string" },
      },
      required: ["runnable", "problems"],
    },
  },
};

export const PLAN_ROLE: Role = {
  name: "planner",
  maxSteps: 15,
  tools: ["list_files", "read_file", "run"],
  writeCode: false,
  system: `You are the planner of an ML reproduction team. Given the paper's claims and the environment engineer's notes, decide for each claim exactly how to measure it with this repository: the command (argv), working directory, environment variables, how the seed is set (a CLI flag or an environment variable, if the code supports one), a regex with exactly one capture group that extracts the claim's metric from the command's output (stdout+stderr; the last match is used), and a timeout. Follow the paper's protocol as closely as the repository allows (dataset, hyperparameters, epochs); if the repository's defaults differ from the paper and a flag exists, use the paper's setting. Do not run full experiments yourself; you may run --help or read code. The command must launch a script of the repository (no inline python -c or sh -c programs): if the code must change before it can run, plan the documented command anyway and say so in the rationale; a debugger fixes it after the run fails. ${LIMITS}`,
  report: {
    description: "Finish with one run spec per claim.",
    parameters: {
      type: "object",
      properties: {
        runs: { type: "array", items: { type: "object", properties: {
          claim: { type: "string" }, argv: { type: "array", items: { type: "string" } }, cwd: { type: "string" }, env: { type: "object" },
          seedArg: { type: "string" }, seedEnv: { type: "string" }, seeds: { type: "array", items: { type: "integer" } },
          parse: { type: "string" }, scale: { type: "number", description: "multiply the parsed number by this to get a fraction in [0,1] (e.g. 0.01 for percentages)" }, timeoutS: { type: "integer" },
          rationale: { type: "string" },
        }, required: ["claim", "argv", "parse"] } },
      },
      required: ["runs"],
    },
  },
};

export const GPU_PLAN_ROLE: Role = {
  name: "planner",
  maxSteps: 25, maxRunS: 300, wallMin: 30,
  tools: ["list_files", "read_file", "run", "read_paper"],
  writeCode: false,
  system: `Each claim's parse regex must match only the output line(s) of that exact metric, anchored on its own label (e.g. "Average Precision  \\(AP\\) @\\[ IoU=0\\.50:0\\.95 \\| area=   all"), never a pattern that also fits other metrics' lines. For each claim also declare how the paper reports it: select ("last" | "max" = best epoch | "min" = lowest, for errors) and aggregate ("mean" | "median" over seeds), from the paper's own description; the harness applies them. Use check_plan on your real draft (at most 8 checks), never on dummy plans to probe the rules. Stages may set memGB (host RAM) and links ([{path: "data", to: "/cache/data/<fetched dir>"}]) instead of shell tricks. Plan from the code and the paper; do not run training to time it. A short timing check (one command, at most 5 minutes) is allowed; your whole session has 30 minutes.

You are the planner of an ML reproduction team working on remote GPUs with a fixed dollar budget per paper. Given the paper's claims and the environment engineer's notes, design the reproduction as a DAG of stages, each one command of the repository run in a fresh container: e.g. fetch data/checkpoints (network allowed, no GPU) -> preprocess -> train or fine-tune (GPU) -> evaluate (GPU). Files a stage must hand to later stages are listed in its "persist" (repo-relative paths); they are restored for the stages that need it.

Choose the fidelity that fits the budget, and say why: "full" (the paper's protocol), "scaled" (the paper's protocol at reduced epochs/data/model size; then claims about absolute numbers will likely not reproduce, so prefer claims about rankings/deltas), or "eval_only" (evaluate released checkpoints). The harness computes the worst case (each stage running to its timeout) and rejects plans above budget, so give realistic timeouts. GPU prices per hour: T4 $0.59, L4 $0.80, A10G $1.10, A100-40GB $2.10, A100-80GB $2.50, H100 $3.95.

Large or archived data and released checkpoints that the scripts expect on disk are better listed in "fetch" ([{url, to: "/cache/data/<name>", sha256?, unpack?: "zip"|"tar"}]); harness code downloads them once (idempotent, provenance recorded) before any stage, and stages read them from /cache/data/... . Files the environment engineer may already have put elsewhere in /cache do not count: every /cache path a stage uses must come from a fetch item. Add "checks" ([{path, minFiles}]) so the harness verifies completeness, e.g. the evaluation split has the number of images the paper evaluates on. For GPU stages give "gpuMemGB" (the memory the stage needs); the harness assigns the cheapest GPU with that much memory.

Each claim names the stage whose output contains its number and a regex with exactly one capture group (the last match is used). Commands must run scripts that EXIST in the repository (no python -c / sh -c programs, no scripts you would like to exist — the harness rejects them). A stage with "network": true may download at run time from data and weight hosts only (Hugging Face, GitHub, dataset sites; package indexes are blocked), and may use a GPU; this is how scripts that call load_dataset/from_pretrained get their data. Give each training/eval stage "gpuMemGB"; the harness assigns the cheapest GPU with that memory (a stage without gpu or gpuMemGB runs on CPU). You may run short CPU commands to check scripts and flags; use the GPU only for very short probes.`,
  report: {
    description: "Finish with the stage plan.",
    parameters: {
      type: "object",
      properties: {
        fidelity: { type: "string", enum: ["full", "scaled", "eval_only"] },
        fidelityNote: { type: "string" },
        stages: { type: "array", items: { type: "object", properties: {
          id: { type: "string" }, argv: { type: "array", items: { type: "string" } }, gpu: { type: "string" }, timeoutS: { type: "integer" },
          needs: { type: "array", items: { type: "string" } }, persist: { type: "array", items: { type: "string" } }, network: { type: "boolean" },
          perSeed: { type: "boolean" }, env: { type: "object" }, cwd: { type: "string" }, gpuMemGB: { type: "number" },
        }, required: ["id", "argv", "timeoutS"] } },
        fetch: { type: "array", items: { type: "object", properties: { url: { type: "string" }, to: { type: "string" }, sha256: { type: "string" }, unpack: { type: "string", enum: ["zip", "tar"] } }, required: ["url", "to"] } },
        checks: { type: "array", items: { type: "object", properties: { path: { type: "string" }, minFiles: { type: "integer" } }, required: ["path", "minFiles"] } },
        claims: { type: "array", items: { type: "object", properties: { claim: { type: "string", description: "the claim's id, exactly as given" }, stage: { type: "string" }, parse: { type: "string" }, scale: { type: "number" } }, required: ["claim", "stage", "parse"] } },
        seeds: { type: "array", items: { type: "integer" } }, seedArg: { type: "string" }, seedEnv: { type: "string" },
      },
      required: ["fidelity", "stages", "claims"],
    },
  },
};

export const DEBUG_ROLE: Role = {
  name: "debugger",
  maxSteps: 30, maxRunS: 1200, wallMin: 40,
  tools: ["list_files", "read_file", "write_file", "run", "install"],
  writeCode: true,
  system: `You are the debugger of an ML reproduction team. A planned run of the repository either failed or produced a number that does not match the paper's claim. Find the root cause by reading code and running experiments in your copy. If a code change is needed to run the experiment faithfully (removed library APIs, crashes, a defect that changes results, settings that exceed the limits), make the minimal change in your copy and verify it by running. Never change how the result is measured to move it toward the paper's number: no editing of metric code, evaluation splits or seeds unless that code is actually wrong, and never write the claimed number into the code. If the repository follows a different protocol than the paper, or the paper's number is simply not reached by faithful code, say so instead of forcing it. Your edits are reviewed: changes to evaluation, metric, split or seed code need a human's approval. ${LIMITS}

${CAUSES}`,
  report: {
    description: "Finish with the diagnosis. Any edits you leave in your copy are taken as your proposed patch.",
    parameters: {
      type: "object",
      properties: {
        causes: { type: "array", items: cause },
        patch_intent: { type: "string", description: "what your edits change and why (empty if you propose no change)" },
        run_argv: { type: "array", items: { type: "string" }, description: "a changed command for this claim, if the fix is a different command/flags" },
        expected_effect: { type: "string" },
      },
      required: ["causes"],
    },
  },
};

export const SOLO_ROLE: Role = {
  name: "solo",
  maxSteps: 60,
  tools: ["read_paper", "list_files", "read_file", "write_file", "run", "install"],
  writeCode: true,
  system: `You are an ML research engineer. Reproduce the paper's claimed numbers named in the objective with this repository, alone: read the paper (read_paper), set up the environment (install), run the experiments with at least 3 seeds, debug and fix what prevents a faithful reproduction, and report per claim. A claim is reproduced when |mean − paper value| ≤ max(0.01, 2 × paper std, 2 × std/√n of your runs). Never change how results are measured to move them toward the paper's number. ${LIMITS}

${CAUSES}`,
  report: {
    description: "Finish with per-claim results and root causes. Edits left in your copy are taken as your patch.",
    parameters: {
      type: "object",
      properties: {
        claims: { type: "array", items: { type: "object", properties: {
          id: { type: "string" }, dataset: { type: "string" }, metric: { type: "string" }, value: { type: "number", description: "the paper's number as a fraction" },
          values: { type: "array", items: { type: "number" }, description: "your runs' results as fractions" },
          status: { type: "string", enum: ["reproduced", "not_reproduced", "not_runnable"] },
        }, required: ["id", "metric", "value", "status"] } },
        causes: { type: "array", items: cause },
      },
      required: ["claims", "causes"],
    },
  },
};

export const CLAIMS_PROMPT = `You extract the claims a reproduction must check from an ML paper. Given the paper text and the objective, return the specific numbers the objective refers to (usually cells of a results table): for each, an id (short snake_case), the table, dataset, method, metric name (accuracy, f1, auc, mse, …), the value as written in the paper converted to a fraction for percentages (81.5% -> 0.815; keep other units as written), the reported std if given (same scale), and a short verbatim quote of the table row or sentence containing the number. Return JSON {"claims": [...]} with only claims the objective asks for.`;
