# Campaign summary (2026-11-09T10:46:14.535Z)

papers 1; claims 2: reproduced 2, not reproduced 0, not runnable 0; GPU $0.99; LLM $0.10 (1722734 tokens); resumes 0

| paper | claims (mean vs paper) | fidelity | env smoke | GPU $ | LLM tokens | min | resumes | failures |
|---|---|---|---|---|---|---|---|---|
| ft_transformer (2106.11959) | ✔ ft_transformer_california_housing_rmse: 0.457 vs 0.459<br>✔ ft_transformer_adult_accuracy: 0.857 vs 0.854 | full | pass | 0.988 | 1722734 | 17 | 0 | experiment_error×9, provider_transient×1, contract_violation×1 |


Pipeline success (every claim definitive, not stopped): 1/1; median 17 min/paper

scale gated: roles on probation (claims, env, planner) -> at most 3 papers, 1 at a time

| role | level | runs | ok | success-rate lower bound | last outcome |
|---|---|---|---|---|---|
| claims | probation | 3 | 3 | 0.44 | ok r2c/ft_transformer: every extracted number is in the PDF |
| env | probation | 3 | 2 | 0.21 | ok r2c/ft_transformer: smoke test passed on the agent's own report |
| planner | probation | 3 | 3 | 0.44 | ok r2c/ft_transformer: plan accepted |
| debugger | probation | 0 | 0 | 0.00 | — |

Standard for promotion: ≥ 5 audited runs and a 95% Wilson lower bound ≥ 0.6. One failure while trusted demotes and resets the count. The debugger blocks scale only once it has been needed.