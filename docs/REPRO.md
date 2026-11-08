# M3: a multi-agent harness that reproduces ML papers

**Input:** an arXiv id, a repository (local path or GitHub URL) and an objective, e.g. "Reproduce the Cora accuracy in Table 2".

**Output:** per claim, a status of `reproduced`, `not_reproduced` or `not_runnable`. Each status comes with run values, root causes graded by evidence tier, and the code changes that were needed, each risk-graded and authorized.

```bash
npm run mlh -- repro <repo> --arxiv 1609.02907 --objective "…" --llm [--approve none|medium|all]
```

```bash
npm run mlh -- resume <runDir>
```

```bash
npm run mlh -- approve <runDir> <node>
```

```bash
npm run mlh -- repro-bench --modes script,solo,team [--provider-faults r] [--crash-after-steps k]
```

## The DAG

```
repro.paper ──► repro.claims ─┐
                               ├─► repro.plan ──► repro.run (claim × seed, parallel) ──► repro.compare (per claim)
repro.env  ───────────────────┘                                 │ fails for good             │ not_reproduced
                                                                 ▼                            ▼
                                                           repro.debug (≤2 rounds/claim) ◄────┘
                                                                 │ edits in its own copy
                                                                 ▼
                                                           repro.patch ── risk grading + authorization
                                                                 │ accepted → source v(n+1)
                                                                 ▼
                                                     repro.run / repro.compare again on v(n+1)
```

The graph grows from observations (`src/mlh/repro/flow.ts`, pure):
- claims extracted from the paper;
- the plan the planner proposes, after the harness validates it;
- runs that fail or miss the claim;
- patches that are accepted.

Every node is journaled. Agent nodes checkpoint after every step, so a killed process resumes mid-agent.

## Who does what, and how much autonomy each role gets

| node | kind | tools | can change | risk |
|---|---|---|---|---|
| `repro.paper` | deterministic | fetches the PDF from arxiv.org; `pdftotext` runs **in the sandbox** | — | `fetch_source` (medium; arXiv-id check) |
| `repro.claims` | model, one call | none | — | `model_query` (high; the human's `--llm` flag) |
| `repro.env` | agent | list/read/write (not code), run, install | installed packages, config files | `exec_sandboxed`, `install_packages` (medium; specifier check) |
| `repro.plan` | agent | list/read/run (read-only intent) | nothing; outputs run specs | `exec_sandboxed` |
| `repro.run` | deterministic | — | — | `exec_sandboxed` |
| `repro.compare` | deterministic | — | — | none |
| `repro.debug` | agent | list/read/write code in **its own copy**, run, install | its copy only | `exec_sandboxed` |
| `repro.patch` | deterministic | — | the source the next runs use | `apply_patch` (medium) or `apply_eval_patch` (high) |

These checks are harness code, not model judgement:
- **Claims:** a claim survives only if its number literally appears in the PDF text.
- **Plans:** a plan is accepted only with a command, a regex with exactly one capture group, and at least 3 seeds (`validatePlan`). Environment variables such as PATH, HOME and LD_* cannot be set, and paths that escape the repository are refused.
- **Status rule** (`statusOf`, the same as `bench/repro/SPEC.md`): reproduced iff |mean − value| ≤ max(0.01, 2·paper std, 2·standard error).
- **Patch risk** (`patchrisk.ts`): a change is high risk if it touches evaluation, metric, split or seed code, or writes a claimed number into the code. A high-risk change is held for a human unless the run's standing policy allows it (`--approve all`). A model's agreement never counts.
- **Package installs:** only plain requirement specifiers are accepted. Installs run in a sandbox that has network but runs nothing from the repository. torch-family packages always come from the CPU index. Sites are content-addressed and shared read-only.
- **Root causes** carry an evidence tier:
  - *intervention*: an accepted change after which the claim's outcome changed;
  - *log*: classified from a failed run's log by `classifyLog`;
  - *agent*: only reported by an agent.

## Sandbox

Same as M1: user, network and mount namespaces; a read-only root; private /tmp, /home and /root; prlimit. The package site is mounted read-only at /opt and put on PYTHONPATH.

Only the install sandbox gets network, and only through the proxy. `tests/mlh-structure.test.ts` checks that `network: true` appears only in `site.ts`, after `authorize("install_packages")`.

## Baselines (`src/mlh/repro/bench.ts`)

- **script:** the deterministic workflow. It installs `requirements.txt`, runs the README's first python command three times, parses the last accuracy, and compares with the **oracle** claims. That favours the script, since it does not have to read the paper.
- **solo:** one agent with every tool (paper, files, shell, install, code edits) and 60 steps. It has no verification and no checkpoints. A crash or a provider outage means starting over, at most 3 attempts.
- **team:** this harness.

Results are in `bench/repro/results/`; their analysis is at the end of this file once measured.

## Results (bench/repro, 10 real tasks; `bench/repro/results/`)

The benchmark has 6 pristine paper+repo tasks (SGC, two GCN implementations, GCNII, APPNP, Fashion-MNIST) and 4 injected variants (API drift, train/test overlap, eval-path defect, resource limit). A separate agent built it: it did the reference reproductions through the sandbox, and the scoring rules in SPEC.md were fixed before any system ran. Each mode ran once per task, so these are single samples.

| | script (oracle claims) | solo agent (v2) | team (v2) | team (v3, after 2 bug fixes) |
|---|---|---|---|---|
| tasks solved | 0/10 | 6/10 | 6/10 | 5/10 |
| claim status accuracy | 0.00 | 0.87 | 0.87 | 0.73 |
| claim extraction from the PDF | (given) | 0.93 | 1.00 | 1.00 |
| root-cause recall / precision | 0 / — | 0.72 / 0.54 | 0.85 / 0.33 | 0.87 / 0.35 |
| injected defects fixed (hidden fix check) | 0/4 | 4/4 | 4/4 | 4/4 |
| high-risk changes that need a human | — | not tracked: edits freely | 9 | 11 |
| median wall time per task | 13 s | 1602 s | 519 s | 589 s |
| LLM tokens / estimated cost | 0 | 13.5M / $0.47 | 12.7M / $0.55 | 12.4M / $0.56 |
| unsafe actions | 0 | 0 | 0 | 0 |

### What the numbers say, and what they do not

- **Both agent systems clearly beat the deterministic workflow.** That workflow solves nothing: pinned requirements no longer install on Python 3.13, imports fail, and runners never exit.
- **Team vs solo: no measured difference in correctness.** 6 vs 6 solved and equal status accuracy, with team v3 at 5/10. With n = 10 and one sample each, a one-task swing is noise. I claim no improvement.
- **What the team measurably adds is structure, not accuracy:**
  - **Time:** median wall time is 3× lower, from seeds running in parallel as separate nodes.
  - **Recovery:** a real one. A team run killed by the 2-hour limit resumed from its journal and agent checkpoints and solved the task (Fashion-MNIST, v2). The solo agent has no equivalent.
  - **Authorization:** 9–11 changes touched evaluation, split or seed code. Under the default policy each needs a human approval, so the team knows when its autonomy should stop. The solo agent made such edits silently. Its patches were also correct in 4/4 cases, so on this benchmark the gate cost approvals and caught no bad change.
  - **Root-cause recall is higher (0.85 vs 0.72), but precision is lower (0.33 vs 0.54):** the environment agent and the debuggers each report causes, and the union over-reports.
- **The failures are mostly not reasoning failures.**
  - In v2/v3, most unsolved team tasks had every status right and missed a root-cause category. Most often that was `env.missing_dependency`: the environment agent installs the package and files the cause as something else.
  - GCNII's hard-coded CUDA was reported as `run.crash` where the reference says `other:hardcoded_cuda`. That is a vocabulary ambiguity in the ground truth as much as an error.

### Bugs this benchmark found in the harness (all fixed, with tests)

1. A tool call with malformed JSON arguments was read as an empty report: the planner's report was lost.
2. A planner could put code changes into `python -c` inline programs, bypassing patch review. Run commands must now launch repository scripts.
3. A run's `PYTHONPATH` replaced the package site instead of extending it, so torch went missing.
4. A command-only fix (no file change) was never measured again, because run ids collided.
5. One package site per distinct package set filled 24 GB with copies of torch and the disk filled mid-benchmark. Sites are now layered: a hard-linked subset plus the missing packages, built in about 2 s.
6. The sandbox wrapper's `set -e` lost the exit marker of a failing command, so `cd` failures were misread as transient sandbox faults and retried.
7. Provider-outage retries had the same small budget as other failures even though agent nodes resume from checkpoints. The budget is now 3× larger.

### Reliability experiment: 10% injected provider outages (3 tasks, after the fixes)

Each model call fails with probability 0.1, simulating an outage that outlasts the client's own retries. The **solo** agent restarts from scratch, up to 3 times. In the **team**, a failed agent node is retried and resumes from its last checkpointed step.

| | solo | team |
|---|---|---|
| runs that finished | 0/3 | 3/3 |
| claim status accuracy | 0.00 | 0.80 (4/5 claims) |
| tasks solved | 0 | 1 (two missed only a root-cause label or the hidden fix check) |
| LLM tokens / cost | 1.12M / $0.044 | 1.16M / $0.051 |

This is the first **controlled** result where the harness structure changes the outcome. A reproduction takes about 60 sequential model calls. At a 10% per-call failure rate, a from-scratch agent almost never gets through: 0.9⁶⁰ ≈ 0.2%. Checkpointed nodes lose at most one step per outage, for about the same token spend.

Caveats:
- 3 tasks, one sample each.
- The outage model is per-call and independent, so real outages, which are bursty, would hit both systems differently.
- A solo agent wrapped in a step-level checkpoint loop would close much of the gap. That would be adopting this harness's mechanism, which is the point.

### Honest caveats

- All of v2 → v3 and the fixes above came from failures on this same benchmark, so v3 is not a blind number.
- A fresh held-out set of real repositories is needed before claiming generalization.
- Ten tasks, mostly graph learning on Cora-family data, is a narrow domain.
