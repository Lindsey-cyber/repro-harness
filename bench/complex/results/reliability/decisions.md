
## p5detr-clean1 (auto retro, 2026-11-09T04:22:30.123Z)
- Pipeline success 0/1; median 10 min/paper; GPU $0.04, LLM $0.04.
- Roles on probation: claims, env, planner, debugger (scale gated).
- Most frequent failure: experiment_error -> next: environment or code: inspect env smoke and debugger transcripts.
- Stop predicate: not reached (need 2 finished rounds, have 1).

## r2c (auto retro, 2026-11-09T10:27:41.913Z)
- Pipeline success 0/1 (previous p5detr-clean1: 0/1, same); median 18 min/paper; GPU $0.79, LLM $0.1.
- Roles on probation: claims, env, planner, debugger (scale gated).
- Most frequent failure: experiment_error -> next: environment or code: inspect env smoke and debugger transcripts.
- Stop predicate: not reached (p5detr-clean1: success 0/1, median 10 min, on probation: claims, env, planner, debugger).

## r3 (auto retro, 2026-11-09T18:39:40.303Z)
- Pipeline success 1/1 (previous r2c: 0/1, better); median 29 min/paper; GPU $5.14, LLM $0.14.
- Roles on probation: claims, env, planner, debugger (scale gated).
- Most frequent failure: provider_transient -> next: inspect the failed nodes in the summary.
- Stop predicate: not reached (r2c: success 0/1, median 18 min, on probation: claims, env, planner, debugger).
