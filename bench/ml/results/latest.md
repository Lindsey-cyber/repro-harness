# ML harness benchmark (2026-11-07T12:36:42.563Z, injected transient fault rate 0)

| metric | naive | workflow | harness |
|---|---|---|---|
| tasks solved (all expected found, no false positive, nothing unsafe) | 4 | 10 | 12 |
| runs completed without failed/skipped nodes | 12 | 12 | 12 |
| defect recall | 0.13 | 1 | 1 |
| false-positive findings | 0 | 2 | 0 |
| …of which on clean controls | 0 | 1 | 0 |
| unsafe side effects that escaped | 2 | 0 | 0 |
| median wall ms per task | 185 | 456 | 554 |
| sandbox executions | 0 | 96 | 131 |
| total sandbox ms | 0 | 10342 | 14154 |
| retried attempts | 0 | 0 | 0 |
| LLM tokens | 0 | 0 | 0 |

| task | category | naive | workflow | harness |
|---|---|---|---|---|
| ckpt_stale | checkpoint_failure | ✘ — (missed: ckpt.roundtrip_mismatch) | ✔ ckpt.roundtrip_mismatch | ✔ ckpt.roundtrip_mismatch |
| claim_inflated | reproducibility | ✔ repro.claim_not_met | ✔ repro.claim_not_met | ✔ repro.claim_not_met |
| clean | control | ✔ — | ✔ — | ✔ — |
| clean_overfit | control | ✔ — | ✘ pipeline.eval_path_mismatch (FP: pipeline.eval_path_mismatch) | ✔ — |
| clean_strongfeature | control | ✔ — | ✔ — | ✔ — |
| leak_overlap | data_leakage | ✘ — (missed: data.train_test_overlap) | ✔ data.train_test_overlap | ✔ data.train_test_overlap |
| leak_target | data_leakage | ✘ — (missed: data.target_leak) | ✔ data.target_leak | ✔ data.target_leak |
| metric_misaligned | incorrect_metric | ✘ — (missed: eval.metric_mismatch) | ✘ pipeline.eval_path_mismatch, eval.metric_mismatch, ckpt.roundtrip_mismatch, repro.claim_not_met (FP: pipeline.eval_path_mismatch) | ✔ eval.metric_mismatch, ckpt.roundtrip_mismatch, repro.claim_not_met |
| metric_train_as_test | incorrect_metric | ✘ — (missed: eval.metric_mismatch) | ✔ ckpt.roundtrip_mismatch, eval.metric_mismatch | ✔ eval.metric_mismatch, ckpt.roundtrip_mismatch |
| nondeterministic | reproducibility | ✘ repro.claim_not_met (missed: repro.nondeterministic) | ✔ repro.nondeterministic | ✔ repro.nondeterministic |
| preproc_unscaled | broken_preprocessing | ✘ repro.claim_not_met (missed: pipeline.eval_path_mismatch) | ✔ pipeline.eval_path_mismatch, ckpt.roundtrip_mismatch, repro.claim_not_met | ✔ pipeline.eval_path_mismatch, ckpt.roundtrip_mismatch, repro.claim_not_met |
| unsafe_sidefx | safety | ✘ — (unsafe 2) | ✔ — | ✔ — |

## Crash recovery (killed after 3 completed nodes, then resumed)

```json
[
 {
  "task": "ckpt_stale",
  "exitCode": 137,
  "completedBeforeCrash": 3,
  "finished": true,
  "reexecutedCompletedNodes": 0,
  "sameFindingsAsUninterrupted": true
 },
 {
  "task": "claim_inflated",
  "exitCode": 137,
  "completedBeforeCrash": 3,
  "finished": true,
  "reexecutedCompletedNodes": 0,
  "sameFindingsAsUninterrupted": true
 },
 {
  "task": "clean",
  "exitCode": 137,
  "completedBeforeCrash": 3,
  "finished": true,
  "reexecutedCompletedNodes": 0,
  "sameFindingsAsUninterrupted": true
 },
 {
  "task": "clean_overfit",
  "exitCode": 137,
  "completedBeforeCrash": 3,
  "finished": true,
  "reexecutedCompletedNodes": 0,
  "sameFindingsAsUninterrupted": true
 },
 {
  "task": "clean_strongfeature",
  "exitCode": 137,
  "completedBeforeCrash": 3,
  "finished": true,
  "reexecutedCompletedNodes": 0,
  "sameFindingsAsUninterrupted": true
 },
 {
  "task": "leak_overlap",
  "exitCode": 137,
  "completedBeforeCrash": 3,
  "finished": true,
  "reexecutedCompletedNodes": 0,
  "sameFindingsAsUninterrupted": true
 },
 {
  "task": "leak_target",
  "exitCode": 137,
  "completedBeforeCrash": 3,
  "finished": true,
  "reexecutedCompletedNodes": 0,
  "sameFindingsAsUninterrupted": true
 },
 {
  "task": "metric_misaligned",
  "exitCode": 137,
  "completedBeforeCrash": 3,
  "finished": true,
  "reexecutedCompletedNodes": 0,
  "sameFindingsAsUninterrupted": true
 },
 {
  "task": "metric_train_as_test",
  "exitCode": 137,
  "completedBeforeCrash": 3,
  "finished": true,
  "reexecutedCompletedNodes": 0,
  "sameFindingsAsUninterrupted": true
 },
 {
  "task": "nondeterministic",
  "exitCode": 137,
  "completedBeforeCrash": 3,
  "finished": true,
  "reexecutedCompletedNodes": 0,
  "sameFindingsAsUninterrupted": true
 },
 {
  "task": "preproc_unscaled",
  "exitCode": 137,
  "completedBeforeCrash": 3,
  "finished": true,
  "reexecutedCompletedNodes": 0,
  "sameFindingsAsUninterrupted": true
 },
 {
  "task": "unsafe_sidefx",
  "exitCode": 137,
  "completedBeforeCrash": 3,
  "finished": true,
  "reexecutedCompletedNodes": 0,
  "sameFindingsAsUninterrupted": true
 }
]
```
