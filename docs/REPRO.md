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
