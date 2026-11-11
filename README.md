# repro-harness

A verifiable multi-agent harness that takes an ML paper (arXiv PDF) and its code repository and autonomously reproduces the paper's reported numbers on cloud GPUs, under a fixed per-paper budget.

A team of LLM agents (claim extraction, environment setup, experiment planning, debugging) works inside a harness that owns all facts. The harness keeps a claim only if its number appears in the PDF, runs every experiment itself in isolated containers, reads results with value-blind rules, compares them with the paper under a statistical tolerance, and records every action in an event-sourced journal, so any run can resume after a crash. Reliability was treated as the primary objective: every failure observed in a real run became a deterministic regression test before the next paid run.

## Results

Five published papers across five domains, all within tolerance of the published values:

| Paper | Work type | Ours vs. paper | GPU cost |
|---|---|---|---|
| DETR (Carion et al., 2020) | eval of released checkpoint | COCO val2017 box AP **0.420** vs 0.420 | ~$0.5 |
| SimCSE (Gao et al., 2021) | eval of released checkpoint | avg. STS **76.25** vs 76.25 | $0.27 |
| GraphSAINT (Zeng et al., 2020) | training, 4 samplers × 2 datasets | F1-micro, **8/8** configurations within tolerance | ~$0.9 |
| FT-Transformer (Gorishniy et al., 2021) | training, 5 seeds × 2 datasets | California RMSE **0.4574 ± 0.0030** vs 0.459; Adult acc. **0.8574 ± 0.0015** vs 0.854 | $1.21 |
| LoRA (Hu et al., 2021) | fine-tuning RoBERTa-base, 5 seeds × 2 GLUE tasks | MRPC **0.890** vs 0.897; RTE **0.852** vs 0.866 | ~$3.3 (final run) |

Tolerance rule: |result − paper| ≤ max(0.01, 2·paper std, 2·standard error of our runs) + half a unit of the paper's last printed digit. The rounding term was added after LoRA RTE missed by 0.000014, and it is applied to every paper; no other verdict changed.

A sixth paper (SimSiam) was deliberately stopped after the system measured that its full protocol needs ~8.5 GPU hours with this repository, and that the repository's own README reports a number below the paper's.

Other measured results:
- **Baselines** on 10 real small reproduction tasks: the multi-agent team and a single agent with the same tools and budget reached the same accuracy (6/10). Under injected transient faults, the team finished 3/3 runs and the single agent 0/3 (checkpointed, resumable nodes vs. one long agent session).
- **Reliability.** The first campaign on complex papers gave a definitive verdict on only 1 of 4 papers, and ~88% of GPU spend produced no result, almost all of it from platform and harness faults rather than agent reasoning. That triggered the reliability programme described below.
- **Tests.** 24 test files, 199 tests. Mutation check: 17/17 deliberately reintroduced bugs were caught.

## Three incidents worth reading

- **A confident wrong verdict (DETR).** The claim's regex matched both AP (0.420) and AR@100 (0.574), and the harness took the last match. There is now no "last match": lines that differ only in measured values are one series, and differently labelled lines are resolved by a model that sees the labels with every value masked, so it cannot steer toward the paper's number. The fix is tested on the real COCO output.
- **A silent bug in the authors' code (LoRA).** RTE came out 4 points low. A first hypothesis (an effective batch size 8× too small) was tested and refuted. Evaluating the authors' released RTE checkpoint in the same environment gave chance level (0.473): `from_pretrained()` ends with `eval()`, which makes loralib merge the still-zero LoRA and mark it merged, so `train()` later subtracts the loaded MNLI LoRA. A one-line fix gave 0.874 on the released checkpoint, and 10 seed runs in parallel then reproduced MRPC and RTE.
- **Measuring before spending (SimSiam).** A first plan substituted a CIFAR-10 run for ImageNet claims and a SimCLR run for "SimSiam without stop-gradient". It was stopped one minute into training, and the validator now requires every claim to be measured on its own dataset and method, or listed as unmeasured with a reason.

## How it works

**Orchestration.** A task DAG whose state is a fold over an append-only, fsync'd event journal. Any run resumes from its journal after a crash, and agent loops checkpoint every step. Failures carry one of 10 categories; only transient ones are retried, with exponential backoff.

**Agents.** One LLM loop implementation with per-role tools, step budgets, per-command time caps and wall-clock budgets: *claims* (extracts claims; a number is kept only if it appears in the PDF), *environment* (makes the repository runnable in a remote container), *planner* (writes a staged GPU plan and can call the harness's own validator) and *debugger* (diagnoses a failed or mismatching stage; it may not edit measurement code without approval).

**Sandboxed execution.** Untrusted repository code never runs on the host: locally in Linux user/net/mount namespaces with a read-only root and resource limits, remotely in Modal sandboxes with network blocked or restricted to approved data hosts, a per-run volume, and a per-paper budget that reserves each run's worst case before admission.

**Verification, independent of the agents.** The harness, not an agent, checks claims against the PDF, smoke-tests the environment, validates plans (scripts must exist, every stage must serve a claim, data must come from declared and checksummed downloads, cost must fit the budget), runs every stage, reads each number value-blind, and compares it with the tolerance rule.

**Risk-adaptive authorization.** Every action passes an authorization function. Code patches are risk-graded: changes that touch evaluation, metrics, splits or seeds need a human or a standing policy, downloads from unknown hosts are held, and publishing is denied. Held items go to a review list; runs never block on a human.

**Earned trust per agent role.** Each role earns trust from harness facts only. Failures are attributed to one of five classes (infrastructure, resource, agent, paper, unclassified), and only agent failures count. Promotion requires a 95% Wilson lower bound ≥ 0.6; one failure while trusted demotes and resets the count.

**Operations.** A campaign driver with a watchdog that kills and resumes stalled runs, a platform probe that must pass before any round, per-attempt cost accounting (productive, retried, failed by class, killed), and strict stop rules (the same infrastructure fault twice, an unclassifiable failure, >30% wasted spend).

Details: [`docs/PROJECT_RECORD.md`](docs/PROJECT_RECORD.md) (results, chronology, all 31 incidents and fixes), [`docs/REPRO.md`](docs/REPRO.md) (the reproduction pipeline), [`docs/RELIABILITY.md`](docs/RELIABILITY.md) (the reliability programme), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (the harness core).

## Not yet established

- Repeatability of a complex paper over several identical runs.
- The single-agent baseline on the complex papers (it has only been run on the 10 small tasks).
- Trust promotion: no agent role has reached the promotion standard yet.

## Running it

Requires Node.js ≥ 22.

```bash
npm ci
npm test            # 24 files, 199 tests; no GPU or API keys needed
npm run check       # typecheck + tests
```

Reproducing a paper needs a Modal account (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`) and an LLM key (`DEEPSEEK_API_KEY`):

```bash
npm run mlh -- repro <repo-or-github-url> --arxiv <id> --objective "Reproduce Table 2" --llm
npm run mlh -- resume <runDir>
npm run mlh -- approve <runDir> <node>
```

## Layout

```
src/mlh/core/      pure decision code: DAG, planner, authorization, trust, reports
src/mlh/io/        process, sandbox, journal and LLM I/O
src/mlh/repro/     paper reproduction: agents, plan validation, campaigns, cost accounting
src/mlh/backends/  Modal GPU backend
bench/             benchmarks and recorded results (synthetic, held-out, real reproduction tasks, campaigns)
tests/             199 tests, including recorded journals and planner drafts from real runs as fixtures
docs/              design, reliability programme and project record
```

Built largely with Claude Code.
