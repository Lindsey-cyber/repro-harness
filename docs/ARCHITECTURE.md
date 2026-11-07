# Verifiable harness for ML experiment reproduction and debugging

Status: milestones 1–2 are in place on branch `mlh`; §6 holds the M2 results on blind benchmarks, which supersede §4's same-author numbers as evidence. Code lives in `src/mlh/`, the benchmark in `bench/ml/`, and the results in `bench/ml/results/`. Commands: `npm run mlh -- run <repo|https://github.com/…> --objective "…"` and `npm run mlh -- bench`.

## 1. Gap analysis of the existing repository

The harness was built inside an earlier TypeScript agent project. It already holds several mechanisms the target system needs. Most of them are written as **organization rules** (docs/program/ORG.md) rather than as runtime code.

| Existing component | Where | Reused as | Kept as is? |
|---|---|---|---|
| Coordinator, manager and worker hierarchy, with a verifier on a different model | `docs/program/ORG.md`, coordinator-guard hook, worktrees | The build process for this harness. It is not a runtime agent topology: M1 needed no extra runtime agents (see §3) | yes |
| Pure decision code kept apart from IO, enforced by a structure test | `tests/structure.test.ts` | The same pattern. `src/mlh/core` is pure and `tests/mlh-structure.test.ts` enforces it | pattern reused |
| Gate whose holds no approval can lift; "approved" lifts only trust checks | `src/gate/gate.ts` | `core/authz.ts`: risk classes, holds, deny lists. A human approval lifts holds; nothing lifts a denial | redesigned for actions |
| Ledger with a legal-transition table and an append-only JSONL file; never retry `unconfirmed` | `src/store/ledger.ts` | `core/dag.ts` + `io/journal.ts`: an event-sourced task state machine with fsync, tolerant of a torn last line | generalized |
| Per-stage trust, where only a human promotes | `src/discover/stagetrust.ts` | `core/trust.ts`: per-skill trust that drops after one bad audit and rises only after a 5-audit clean streak plus a human | generalized |
| Eval set and regression check (`scoreEval`, `regressions`) | `src/discover/evalset.ts` | `bench.ts` scoring + `--check baseline.json` | pattern reused |
| A model verdict counts only for the exact text it judged (`basis` hash) | `src/discover/modeltriage.ts` | Principle: model output is advisory and tied to its evidence. In authz, LLM evidence never authorizes | principle |
| Review list instead of blocking (`jh pending`) | `src/store/review.ts` | Held nodes go to `summary.json.held` and `mlh approve`. The run never waits for input | pattern reused |

Gaps against the target, and how M1 handles each:

| Target area | What existed | Gap | M1 |
|---|---|---|---|
| 1 Stateful orchestration | A linear CLI pipeline. Parallelism was organizational (worktrees) | No task DAG, retries, checkpoints or resume | **Done:** DAG, parallel scheduler, retry by failure category, journal checkpoint, crash resume |
| 2 Sandboxed execution | Playwright against employer pages. No code execution | Everything | **Done:** user+net+mount namespaces, rlimits, timeout, fresh copy per execution, an isolation self-test that the authz layer uses as evidence |
| 3 Skill runtime | Claude Code skills (markdown) for humans and agents | No runtime contracts or versioning | **Done:** 11 versioned skills with declared needs, actions, dependencies and a zod output contract that the orchestrator enforces |
| 4 Trust and verification | Cross-model review of code, gate, stage trust | No verification of runtime claims | **Done:** every suspicion is checked independently (a designated confirmer or a fresh replication). Risk-based authz, human approval gate, asymmetric trust. *Not yet:* an LLM cross-check (interface only, §5) |
| 5 Context and artifacts | Run directories, evidence files | No content addressing, compaction or resume context | **Done:** sha256 artifact store, run.json metadata, `context.txt` (bounded compaction), resumable runs. *Not yet:* cross-run structured memory |
| 6 Evaluation and observability | Eval set, scorecards | No traces or task benchmark | **Done:** per-attempt trace (latency, sandbox time, failure category, token and cost fields), a 12-task benchmark, 3 modes, fault injection, crash-recovery measurement |

## 2. Architecture

```
mlh run <repo|url> --objective …
        │
  orchestrator.ts ── journal.jsonl (event log = checkpoint) ── trace.jsonl
   │  replay → ready() → authorize() → skill.run(ctx) → contract check → emit ok/fail → replan
   │
   ├── core/ (pure, structure-tested)
   │     dag.ts      event-sourced node state machine, retry budget, resume semantics
   │     planner.ts  inspect → expand by manifest capabilities + objective intents → confirmers on suspicion
   │     authz.ts    action risk → allow / hold (human) / deny; an LLM's evidence never counts
   │     trust.ts    per-skill confirm|auto; drops at once, rises only after a streak plus a human
   │     report.ts   evidence report, summary, bounded context compaction
   ├── skills/       versioned measurement units (needs, actions, after, confirms, output schema)
   ├── io/
   │     sandbox.ts  namespaces + prlimit + timeout; self-test; failure classification
   │     proc.ts     the only process spawner
   │     journal.ts  fsync'd append-only events
   │     artifacts.ts sha256 content-addressed store
   └── probes/probe.py  trusted numpy measurements, mounted read-only at /srv, never import repo code
```

**Separation of concerns.** Each concern lives in its own layer:
- orchestration (scheduling, retries, replanning) is `orchestrator.ts` + `core/dag.ts` + `core/planner.ts`;
- execution is `io/sandbox.ts`;
- verification is confirm nodes, which are skills that only produce `confirmation` evidence and never new findings;
- authorization is `core/authz.ts`, called before every node and enforced by a structure test.

**Finding lifecycle.** A skill emits either a `fact` (an exact measurement, e.g. a row-overlap count, confirmed immediately) or a `suspicion`. For each suspicion the planner adds either designated confirmers (`confirms: [category]`) or, when none exists and the detector's trust is `confirm`, a replication: fresh instances of the experiment run plus the detector again. The status is decided from the independent evidence: `confirmed`, `refuted`, `disputed` (conflicting evidence) or `suspected` (nothing could test it). Only `confirmed` counts as a finding; the rest are shown as unverified or refuted.

**Planning is capability-driven, not scripted.** Only one node exists at start (`repo.inspect`). The plan expands from the inferred manifest: each skill declares `needs(manifest)`, and unmet needs become **coverage gaps** in the report instead of silent omissions. The objective's intents filter skills: "reproduce" alone skips the data audit. Replanning happens on every suspicion. No code path knows a benchmark task (`tests/mlh-structure.test.ts` checks that task names and ground truth never appear in the harness).

**Execution.** `selfTest` proves the isolation before any repository code runs: it checks that the root filesystem is read-only, that the network is unreachable and that /home is hidden. This result is the deterministic evidence that authorizes `exec_sandboxed` (medium risk). Without it, execution becomes `exec_unsandboxed` (high risk) and every node is held until a human runs `mlh approve`. Limits: 120 s CPU, 4 GB address space, 512 MB files, 180 s wall clock; BLAS threads pinned to 1 for determinism. Each execution runs in a fresh copy of the source, so retries and resumes are idempotent.

**Failure taxonomy.** The categories are `sandbox_transient`, `timeout`, `resource_limit`, `experiment_error`, `contract_violation`, `precondition`, `policy_denied` and `harness_bug`. Only the first two are retried, up to 3 attempts, with exponential backoff.

### Thresholds

These were fixed before the first benchmark run and are the same for every repository (`skills/index.ts: T`):
- train/test row overlap above 1%;
- single-feature AUC of at least 0.99, on train to suspect and on test to confirm;
- reported vs recomputed metric differing by more than 1e-3;
- a train−test gap above 0.15 to suspect a broken evaluation path, confirmed only if held-out *training* rows score within 0.05 of the majority baseline through that path;
- a checkpoint round-trip difference above 0.005;
- the claim falling outside the bootstrap 95% CI, confirmed only if no rerun's CI contains it.

### Key trade-offs

- **No LLM in the M1 loop** (M2 adds LLM components; see §6). Every M1 decision can be made deterministically. No API key is available in this environment, so an LLM component could not have been measured, and the brief forbids unmeasured claims. LLMs are planned where determinism runs out (§5). The trace already carries `tokens`/`costUsd` fields, which are 0 today.
- **A convention-light manifest instead of arbitrary repositories.** Inference covers common layouts (train.py/eval.py, *train*/*test* CSVs, metrics JSON, predictions CSV, a README claim). `mlh.json` overrides it, and anything not inferred becomes a coverage gap. This is the main limitation for real-world repositories.
- **Linux namespaces instead of containers.** They need no daemon or image and start in about 10 ms, so they suit limited local compute. Isolation is weaker than a VM's. CPU is limited per process, not per process tree.
- **One fixed seed set for the fault injection.** Results are reproducible but are one sample per rate.

## 3. Why not more agents

The target names cross-model verification and a coordinator–manager–worker hierarchy. M1 keeps both as the *build* process (worktrees plus an independent verifier on a different model, per ORG.md). It adds no runtime agents. Every runtime decision M1 makes has a deterministic test that is cheaper and more reliable than a model. An LLM agent belongs where the decision space is open: writing a manifest for an unfamiliar repository, proposing a patch, explaining a finding. Even there, the model only proposes. Acceptance comes from re-running the deterministic skills (§5).

## 4. Results (M1, measured; `bench/ml/results/`)

The benchmark has 12 tasks:
- 3 clean controls: clean, legitimate overfitting, and a legitimate strong feature;
- leakage: row overlap and target leakage;
- broken preprocessing: test set not scaled;
- incorrect metrics ×2: train accuracy reported as test, and misaligned labels;
- a stale checkpoint;
- nondeterminism;
- an inflated claim;
- unsafe side effects: home-directory write and network egress.

`expect`/`allow` (true consequences such as a claim miss caused by a metric bug) were written into `task.json` when each task was created.

No injected faults:

| metric | naive script | fixed workflow (same skills) | harness |
|---|---|---|---|
| tasks solved: all expected found, no false positive, nothing unsafe | 4/12 | 10/12 | **12/12** |
| defect recall | 0.13 | 1.00 | 1.00 |
| false-positive findings (on clean controls) | 0 (0) | 2 (1) | **0 (0)** |
| unsafe side effects that escaped | **2** | 0 | 0 |
| median wall time per task | 185 ms | 456 ms | 554 ms |
| sandbox executions (total sandbox ms) | 0 | 96 (10.3 s) | 131 (14.2 s) |
| LLM tokens | 0 | 0 | 0 |

Injected transient sandbox failures. Each sandbox call fails with probability *p*, deterministic per task:

| p | workflow solved / completed / recall | harness solved / completed / recall | harness retries |
|---|---|---|---|
| 0.1 | 2 / 3 / 0.63 | **11 / 11 / 1.00** | 18 |
| 0.2 | 0 / 1 / 0.38 | **8 / 8 / 0.88** | 36 |

At p=0.2 the harness still loses 4 tasks: skills that make 3 or more sandbox calls fail all 3 attempts. That is a real limit of the retry budget, not noise.

Crash recovery: on all 12 tasks the run was killed (exit 137) after 3 completed nodes and then resumed. All 12 finished with the same confirmed findings as an uninterrupted run, and 0 completed nodes were re-executed.

The two false positives in the workflow baseline:
- `clean_overfit`: legitimate overfitting flagged as a broken evaluation path;
- `metric_misaligned`: a metric bug that looks like a train/test gap.

The harness's `pipeline.split_swap` confirmer refuted both: held-out training rows score well above chance through the evaluation path.

What the numbers show, and what they do not:
- **naive vs workflow:** most of the detection comes from the skills themselves, not from orchestration.
- **workflow vs harness:** the difference is entirely independent confirmation (false positives refuted) and retries (recovery under injected faults). Confirmation costs about 35% more sandbox executions.
- **Same author.** The tasks, thresholds and harness were written by the same author in one sitting. That favours the harness, so treat the numbers as a regression baseline, not as evidence of generalization. M2 adds tasks written blind and real repositories.
- **Not measured: a single-agent LLM baseline.** It was not run because no API key was available, so no comparison is claimed.

## 5. Prioritized plan

1. **M1 (this branch): reliable deterministic core.** DAG, journal, resume, sandbox, authz, trust, 11 skills, report, benchmark with 3 modes, fault injection and crash recovery. ✔
2. **M2: real repositories and a held-out benchmark.**
   - Manifest inference for PyTorch and sklearn layouts, `requirements.txt` install into a cached venv with network only during install (a separate, approved action).
   - 10+ tasks written by someone other than the harness author, plus 3 public repositories with known issues.
   - A `--check` regression gate in CI.
3. **M3: LLM components under the same verification.**
   - An LLM manifest writer for repositories inference cannot read; its output is accepted only if `run.train` succeeds with it.
   - An LLM patch proposer for confirmed findings; a patch is accepted only if the finding's detector and confirmer flip and the clean skills stay clean, measured in a fresh sandbox.
   - A cross-model reviewer of reports, advisory only.
   - Token and cost tracking through the existing trace fields. A single-agent LLM baseline on the same benchmark.
4. **M4: scale.**
   - Experiment metadata store across runs (structured memory: per-repository manifests, past findings).
   - Larger compute limits per skill.
   - Trust promotion from the audit history (`recordAudit` fed by benchmark runs; promotion through a human command).


## 6. M2: LLM components, a single-agent baseline and blind benchmarks

### What was added

**LLM client** (`io/llm.ts`):
- OpenAI-compatible; DeepSeek by default (`deepseek-flash` proposes, `deepseek-v4-pro` reviews);
- meters tokens, cost and latency per node, recorded in `trace.jsonl`;
- a record/replay cache: reruns are free and reproducible, and cached entries keep the original usage;
- `MLH_LLM_SALT` separates benchmark repeats.

Cost uses list prices from third-party trackers (flash $0.14/$0.28, pro $1.74/$3.48 per 1M input/output tokens). DeepSeek's own page did not show rates, so treat cost as an estimate.

**Authorization.** `model_query` (sending repository content to a provider) is high risk. The only evidence that allows it is the human's `--llm` flag. A model's approval never allows it.

**Manifest:**
- inference prefers commands documented in the README;
- an ambiguous claim (several candidate numbers) is left unset, not guessed;
- a model fills the gaps, and each proposed field is kept only if deterministic checks pass: the script exists, the files exist, the label is in the header, the claimed number appears in the README;
- output paths are resolved by observation: whatever file the run actually wrote replaces a declared path that does not exist.

**`llm.hypothesize` → `llm.check`.** The proposer suggests defects, each with an executable check. A hypothesis is confirmed only if both hold:
- the check, run in a fresh sandbox after training, prints `supports: true`;
- a different model judges that the check tests the claim and that the output shows an effect.

**Single-agent baseline** (`agent.ts`):
- one model with list/read/write/run tools, in one sandbox under the same authorization;
- the same objective and finding vocabulary as the harness;
- no independent verification;
- a 30-step budget.

**Two blind benchmarks**, each written by a separate agent that was forbidden to read the harness, its thresholds, earlier benchmarks or results. Each has 12 tasks with varied layouts, configs, CLI flags, npz data, id and string columns, and 2 defects outside the known categories.
- `bench/ml/heldout`: blind for its first run only (`heldout-BLIND-v1`). I then fixed generic failures it exposed, so later numbers on it are "post-fix", not blind.
- `bench/ml/heldout2`: the code was frozen at commit `15a2a77` before its first run. Results are in `heldout2-BLIND-r1` and `heldout2-BLIND-r2-llm` (a second sample of the LLM modes).

### Results

**Solved tasks** (strict scoring; "other:" slugs must match):

| benchmark | naive | workflow | harness (deterministic) | single agent | hybrid (harness + LLM) |
|---|---|---|---|---|---|
| own tasks (same author) | 4/12 | 10/12 | **12/12** | **12/12** | **12/12** |
| heldout, blind v1 (M1 code) | 1/12 | 1/12 | 1/12 | **7/12** | 1/12 |
| heldout, after generic fixes (not blind) | 2/12 | 6/12 | 5/12 | **8/12** | 6/12 |
| **heldout2, blind** (run 1 / run 2) | 3/12 | 4/12 | 4/12 | **9/12 / 9/12** | 6/12 / 8/12 |

**heldout2 (blind) detail:**

| | harness | single agent (r1 / r2) | hybrid (r1 / r2) |
|---|---|---|---|
| defect recall | 0.13 | 0.75 / 0.75 | 0.63 / 0.63 |
| false positives (on clean controls) | 0 (0) | 2 (0) / 3 (0) | 3 (2) / 3 (1) |
| median wall time per task | 1.0 s | 49 s / 40 s | 172 s / 214 s |
| LLM tokens | 0 | 0.91M / 1.10M | 0.37M / 0.33M |
| estimated cost | $0 | $0.05 / $0.06 | $0.32 / $0.28 |
| model hypotheses | — | — | 21 proposed: 8 confirmed, 13 refuted / 7 confirmed, 12 refuted |

No mode let a side effect escape the sandbox on any benchmark. The naive mode let 2 escape on the unsafe task.

### What this shows

1. **The deterministic harness has high precision but generalizes poorly.**
   - It had zero false positives on clean controls across all three benchmarks, and the fixed workflow raised only 1 false positive on heldout2.
   - Its recall on unfamiliar repositories is 0.13–0.22.
   - The skills assume layouts like the author's (CSV splits, a metrics json, a predictions file). The same-author 12/12 measured that fit, not capability.
2. **A single agent with tools beat every structured variant on blind tasks.** It solved 9/12 in both samples, raised 0 false positives on controls, cost the least per token, and was faster than hybrid. Its errors are mostly mis-categorisation (e.g. a metric bug filed as `other:`) and missing out-of-vocabulary defects.
3. **Model-written checks plus a cross-model review are not a reliable verifier.**
   - All hybrid false positives on controls were model hypotheses that passed both gates.
   - Example, `meter_events` (legitimate recurring rows): the check's own numbers show no effect (F1 0.728 vs 0.708), yet it printed `supports: true` and the reviewer accepted it.
   - Re-scoring the saved runs with a strict policy, where a model-origin finding is never confirmed without a deterministic confirmer, removes every control false positive (2→0, 1→0). It also collapses recall (5→1 and 5→2 of 8 expected defects). Precision and recall trade off directly, and today only deterministic confirmers deliver precision.
4. **The hybrid is the most expensive mode and was beaten by the single agent.** The pro reviewer costs more than the agent's whole run, and serial node latency adds up. As designed, it is not worth it.

### Consequences for the design (next milestone)

- **Use the agent as the explorer, not the planner of fixed skills.** It handles unfamiliar layouts far better than inference does. Keep the harness for what it measurably did well: sandbox plus authorization (0 escapes), journaled resume, retries, and deterministic confirmers.
- **Route each agent finding to a deterministic confirmer of its category where one exists.** Example: entity overlap goes to the group-aware re-split, a metric mismatch to recomputation from predictions. Report the rest as "model-supported, unverified" rather than as confirmed. The ablation above measures the cost of this choice.
- **Make LLM checks falsifiable.** Require a check to report an effect size and compare it with a fixed materiality threshold computed by the harness, instead of trusting the check's own `supports` flag. `meter_events` would then have been rejected.
- **Do not pin hidden nondeterminism in reruns.** `PYTHONHASHSEED` is pinned for determinism. That hides hash-order nondeterminism (heldout2 `topic_vocab`), so `repro.rerun` should vary it on the second run.
- **Measure variance properly.** Two samples per LLM mode is not enough for confidence intervals. Use at least 5 repeats, and blind sets written by humans or more independent authors.
