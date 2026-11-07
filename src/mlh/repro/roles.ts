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
  maxSteps: 25,
  tools: ["list_files", "read_file", "write_file", "run", "install"],
  writeCode: false,
  system: `You are the environment engineer of an ML reproduction team. Your only job: make the repository's experiments runnable in this sandbox. Read the README/requirements/setup files, install what is needed (install tool; torch-family packages come from the PyTorch CPU index automatically), and find out how to launch the experiments (working directory, PYTHONPATH or other environment variables). Verify by actually starting the documented experiment command for a short time (use timeout_s). You may NOT edit source code: if the code itself must change (removed library APIs, crashes), say so precisely in your report — a debugger will handle it. ${LIMITS}

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
  system: `You are the planner of an ML reproduction team. Given the paper's claims and the environment engineer's notes, decide for each claim exactly how to measure it with this repository: the command (argv), working directory, environment variables, how the seed is set (a CLI flag or an environment variable, if the code supports one), a regex with exactly one capture group that extracts the claim's metric from the command's output (stdout+stderr; the last match is used), and a timeout. Follow the paper's protocol as closely as the repository allows (dataset, hyperparameters, epochs); if the repository's defaults differ from the paper and a flag exists, use the paper's setting. Do not run full experiments yourself; you may run --help or read code. ${LIMITS}`,
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

export const DEBUG_ROLE: Role = {
  name: "debugger",
  maxSteps: 30,
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
