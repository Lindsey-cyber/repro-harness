# Campaign summary (2026-11-09T10:27:41.910Z)

papers 1; claims 2: reproduced 0, not reproduced 0, not runnable 2; GPU $0.79; LLM $0.10 (1722734 tokens); resumes 0

| paper | claims (mean vs paper) | fidelity | env smoke | GPU $ | LLM tokens | min | resumes | failures |
|---|---|---|---|---|---|---|---|---|
| ft_transformer (2106.11959) | ✘ ft_transformer_california_housing_rmse: — vs 0.459<br>✘ ft_transformer_adult_accuracy: — vs 0.854 | full | pass | 0.786 | 1722734 | 18 | 0 | experiment_error×9, provider_transient×1, contract_violation×1 STOPPED: unclassified_failure |

- ft_transformer: repro.stage:train_adult:v0:s0: ambiguous metric for ft_transformer_adult_accuracy: the pattern matched 4 differently labelled lines and none could be c

Pipeline success (every claim definitive, not stopped): 0/1; median 18 min/paper

scale gated: roles on probation (claims, env, planner) -> at most 3 papers, 1 at a time

| role | level | runs | ok | success-rate lower bound | last outcome |
|---|---|---|---|---|---|
| claims | probation | 2 | 2 | 0.34 | ok r1/simcse: every extracted number is in the PDF |
| env | probation | 2 | 1 | 0.09 | FAIL r1/simcse: smoke passed only after a repair round |
| planner | probation | 2 | 2 | 0.34 | ok r1/simcse: plan accepted |
| debugger | probation | 0 | 0 | 0.00 | — |

Standard for promotion: ≥ 5 audited runs and a 95% Wilson lower bound ≥ 0.6. One failure while trusted demotes and resets the count. The debugger blocks scale only once it has been needed.