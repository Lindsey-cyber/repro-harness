| role | level | runs | ok | success-rate lower bound | last outcome |
|---|---|---|---|---|---|
| claims | probation | 1 | 1 | 0.21 | ok p5detr-clean1/detr: every extracted number is in the PDF |
| env | probation | 1 | 1 | 0.21 | ok p5detr-clean1/detr: smoke test passed on the agent's own report |
| planner | probation | 1 | 1 | 0.21 | ok p5detr-clean1/detr: plan accepted |
| debugger | probation | 0 | 0 | 0.00 | — |

Standard for promotion: ≥ 5 audited runs and a 95% Wilson lower bound ≥ 0.6. One failure while trusted demotes and resets the count. The debugger blocks scale only once it has been needed.