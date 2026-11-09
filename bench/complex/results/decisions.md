# Decision log (complex-paper campaigns)

Metric: pipeline success = share of papers where every claim gets a definitive verdict
(reproduced / not reproduced) and the run is not stopped; secondary: median minutes per paper.
Stop predicate: every role trusted (trust.md), pipeline success >= 80% on two consecutive
rounds, median <= 45 min/paper.

## c1 (detr, simcse, graphsaint, ft_transformer, lora, simsiam)
- Result so far: 1/4 finished papers definitive (detr). Failures: simcse plan rejected
  (validator bug, /work cwd), graphsaint plan rejected (invented script), ft_transformer
  Modal spend limit (infrastructure).
- Changes: check_plan tool; /work normalisation; Modal spend limit -> provider_fatal;
  per-role trust scores gate scale (probation: <= 3 papers, serial); infrastructure
  failures not counted against agents; CPU/memory billed for every container;
  LLM balance preflight per paper.
- Offline replay of the 8 recorded plan drafts found a second validator bug (a model
  directory filled by fetch items was rejected) before c2 spent money on it. Fixed.
- Known bias: the planner's trust carries one failure caused by the validator (simcse);
  kept, since it errs toward not scaling.

## c2 (simcse, graphsaint, ft_transformer): pending
