| role | level | runs | ok | success-rate lower bound | last outcome |
|---|---|---|---|---|---|
| claims | probation | 2 | 2 | 0.34 | ok r1/simcse: every extracted number is in the PDF |
| env | probation | 2 | 1 | 0.09 | FAIL r1/simcse: smoke passed only after a repair round |
| planner | probation | 2 | 2 | 0.34 | ok r1/simcse: plan accepted |
| debugger | probation | 0 | 0 | 0.00 | — |

Standard for promotion: ≥ 5 audited runs and a 95% Wilson lower bound ≥ 0.6. One failure while trusted demotes and resets the count. The debugger blocks scale only once it has been needed.