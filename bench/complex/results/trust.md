| role | level | runs | ok | success-rate lower bound | last outcome |
|---|---|---|---|---|---|
| claims | probation | 4 | 4 | 0.51 | ok c1/ft_transformer: every extracted number is in the PDF |
| env | probation | 3 | 3 | 0.44 | ok c1/graphsaint: smoke test passed on the agent's own report |
| planner | probation | 3 | 1 | 0.06 | FAIL c1/graphsaint: plan rejected by the validator |
| debugger | probation | 0 | 0 | 0.00 | — |

Standard for promotion: ≥ 5 audited runs and a 95% Wilson lower bound ≥ 0.6. One failure while trusted demotes and resets the count. The debugger blocks scale only once it has been needed.