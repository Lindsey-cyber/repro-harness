# ML harness benchmark (2026-11-07T16:24:32.950Z, injected transient fault rate 0)

| metric | naive | workflow | harness | agent | hybrid |
|---|---|---|---|---|---|
| tasks solved (all expected found, no false positive, nothing unsafe) | 3 | 4 | 4 | 9 | 6 |
| …counting any other:<slug> as matching any expected other:<slug> | 3 | 4 | 4 | 9 | 6 |
| runs completed without failed/skipped nodes | 9 | 10 | 10 | 12 | 10 |
| defect recall | 0 | 0.13 | 0.13 | 0.75 | 0.63 |
| false-positive findings | 0 | 1 | 0 | 2 | 3 |
| …of which on clean controls | 0 | 0 | 0 | 0 | 2 |
| unsafe side effects that escaped | 0 | 0 | 0 | 0 | 0 |
| median wall ms per task | 511 | 969 | 973 | 48942 | 172353 |
| sandbox executions | 0 | 33 | 39 | 133 | 83 |
| total sandbox ms | 0 | 10281 | 12423 | 77799 | 33955 |
| retried attempts | 0 | 0 | 0 | 0 | 0 |
| LLM tokens | 0 | 0 | 0 | 908009 | 369726 |
| LLM cost, USD (estimated at list price) | 0 | 0 | 0 | 0.05 | 0.32 |
| model hypotheses (hybrid) | — | — | — | — | 21 proposed / 8 confirmed / 13 refuted / 0 unverified |

| task | category | naive | workflow | harness | agent | hybrid |
|---|---|---|---|---|---|---|
| adclick_te | data_leakage | ✘ — (missed: data.target_leak) | ✘ repro.claim_not_met (FP: repro.claim_not_met) (missed: data.target_leak) | ✘ — (missed: data.target_leak) | ✔ data.target_leak | ✔ data.target_leak |
| bearing_fault | control | ✔ — | ✔ — | ✔ — | ✔ — | ✔ — |
| birdcall_subset | other | ✘ — (missed: other:filtered_test_subset) | ✘ — (missed: other:filtered_test_subset) | ✘ — (missed: other:filtered_test_subset) | ✘ eval.metric_mismatch (missed: other:filtered_test_subset) | ✘ data.train_test_overlap, eval.metric_mismatch (FP: data.train_test_overlap) (missed: other:filtered_test_subset) |
| churn_plans | broken_preprocessing | ✘ — (missed: pipeline.eval_path_mismatch) | ✘ — (missed: pipeline.eval_path_mismatch) | ✘ — (missed: pipeline.eval_path_mismatch) | ✔ pipeline.eval_path_mismatch | ✘ pipeline.eval_path_mismatch |
| loan_default | control | ✘ — | ✔ — | ✔ — | ✔ — | ✔ — |
| lowdata_ckpt | checkpoint_failure | ✘ — (missed: ckpt.roundtrip_mismatch) | ✔ ckpt.roundtrip_mismatch | ✔ ckpt.roundtrip_mismatch | ✔ ckpt.roundtrip_mismatch | ✔ ckpt.roundtrip_mismatch |
| meter_events | control | ✔ — | ✔ — | ✔ — | ✔ — | ✘ data.train_test_overlap (FP: data.train_test_overlap) |
| readmit_rows | data_leakage | ✘ — (missed: data.train_test_overlap) | ✘ — (missed: data.train_test_overlap) | ✘ — (missed: data.train_test_overlap) | ✔ data.train_test_overlap | ✘ — (missed: data.train_test_overlap) |
| ticket_f1 | incorrect_metric | ✘ — (missed: eval.metric_mismatch) | ✘ — (missed: eval.metric_mismatch) | ✘ — (missed: eval.metric_mismatch) | ✔ eval.metric_mismatch | ✔ eval.metric_mismatch |
| topic_vocab | reproducibility | ✘ — (missed: repro.nondeterministic) | ✘ repro.claim_not_met (missed: repro.nondeterministic) | ✘ repro.claim_not_met (missed: repro.nondeterministic) | ✘ repro.nondeterministic, repro.claim_not_met, other:vocab_cap_not_top_frequent (FP: other:vocab_cap_not_top_frequent) | ✔ repro.claim_not_met, repro.nondeterministic |
| val_auc_seeds | control | ✔ — | ✘ — | ✘ — | ✔ — | ✘ ckpt.roundtrip_mismatch (FP: ckpt.roundtrip_mismatch) |
| wine_stop | other | ✘ — (missed: other:test_set_model_selection) | ✘ — (missed: other:test_set_model_selection) | ✘ — (missed: other:test_set_model_selection) | ✘ data.target_leak, repro.claim_not_met (FP: repro.claim_not_met) (missed: other:test_set_model_selection) | ✘ — (missed: other:test_set_model_selection) |
