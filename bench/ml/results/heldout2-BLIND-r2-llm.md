# ML harness benchmark (2026-11-07T16:32:36.380Z, injected transient fault rate 0)

| metric | agent | hybrid |
|---|---|---|
| tasks solved (all expected found, no false positive, nothing unsafe) | 9 | 8 |
| …counting any other:<slug> as matching any expected other:<slug> | 9 | 8 |
| runs completed without failed/skipped nodes | 12 | 11 |
| defect recall | 0.75 | 0.63 |
| false-positive findings | 3 | 3 |
| …of which on clean controls | 0 | 1 |
| unsafe side effects that escaped | 0 | 0 |
| median wall ms per task | 40285 | 213895 |
| sandbox executions | 138 | 87 |
| total sandbox ms | 72478 | 33160 |
| retried attempts | 0 | 0 |
| LLM tokens | 1097856 | 326710 |
| LLM cost, USD (estimated at list price) | 0.06 | 0.28 |
| model hypotheses (hybrid) | — | 21 proposed / 7 confirmed / 12 refuted / 2 unverified |

| task | category | agent | hybrid |
|---|---|---|---|
| adclick_te | data_leakage | ✔ data.target_leak | ✔ data.target_leak |
| bearing_fault | control | ✔ — | ✔ — |
| birdcall_subset | other | ✘ eval.metric_mismatch (missed: other:filtered_test_subset) | ✘ data.train_test_overlap, eval.metric_mismatch (FP: data.train_test_overlap) (missed: other:filtered_test_subset) |
| churn_plans | broken_preprocessing | ✔ pipeline.eval_path_mismatch, repro.claim_not_met | ✘ — (missed: pipeline.eval_path_mismatch) |
| loan_default | control | ✔ — | ✔ — |
| lowdata_ckpt | checkpoint_failure | ✔ ckpt.roundtrip_mismatch | ✔ ckpt.roundtrip_mismatch |
| meter_events | control | ✔ — | ✘ data.train_test_overlap (FP: data.train_test_overlap) |
| readmit_rows | data_leakage | ✔ data.train_test_overlap | ✔ data.train_test_overlap |
| ticket_f1 | incorrect_metric | ✔ eval.metric_mismatch | ✔ eval.metric_mismatch |
| topic_vocab | reproducibility | ✘ repro.nondeterministic, repro.claim_not_met, other:vocab_not_top_frequency (FP: other:vocab_not_top_frequency) | ✔ repro.claim_not_met, repro.nondeterministic |
| val_auc_seeds | control | ✔ — | ✔ — |
| wine_stop | other | ✘ eval.metric_mismatch, data.train_test_overlap (FP: eval.metric_mismatch, data.train_test_overlap) (missed: other:test_set_model_selection) | ✘ ckpt.roundtrip_mismatch (FP: ckpt.roundtrip_mismatch) (missed: other:test_set_model_selection) |
