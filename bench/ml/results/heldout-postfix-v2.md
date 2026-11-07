# ML harness benchmark (2026-11-07T16:13:37.463Z, injected transient fault rate 0)

| metric | naive | workflow | harness | agent | hybrid |
|---|---|---|---|---|---|
| tasks solved (all expected found, no false positive, nothing unsafe) | 2 | 6 | 5 | 8 | 6 |
| …counting any other:<slug> as matching any expected other:<slug> | 2 | 6 | 5 | 9 | 7 |
| runs completed without failed/skipped nodes | 6 | 12 | 12 | 12 | 11 |
| defect recall | 0 | 0.33 | 0.22 | 0.56 | 0.67 |
| false-positive findings | 0 | 0 | 0 | 1 | 4 |
| …of which on clean controls | 0 | 0 | 0 | 0 | 0 |
| unsafe side effects that escaped | 0 | 0 | 0 | 0 | 0 |
| median wall ms per task | 237 | 975 | 1355 | 26730 | 156753 |
| sandbox executions | 0 | 34 | 44 | 134 | 94 |
| total sandbox ms | 0 | 11151 | 15640 | 79209 | 42335 |
| retried attempts | 0 | 0 | 0 | 0 | 0 |
| LLM tokens | 0 | 0 | 0 | 973609 | 305936 |
| LLM cost, USD (estimated at list price) | 0 | 0 | 0 | 0.06 | 0.26 |
| model hypotheses (hybrid) | — | — | — | — | 17 proposed / 6 confirmed / 11 refuted / 0 unverified |

| task | category | naive | workflow | harness | agent | hybrid |
|---|---|---|---|---|---|---|
| ckpt_signal_mlp | checkpoint_failure | ✘ — (missed: ckpt.roundtrip_mismatch) | ✔ ckpt.roundtrip_mismatch | ✔ ckpt.roundtrip_mismatch | ✔ pipeline.eval_path_mismatch, ckpt.roundtrip_mismatch | ✘ data.train_test_overlap, ckpt.roundtrip_mismatch (FP: data.train_test_overlap) |
| ctrl_credit_logreg | control | ✘ — | ✔ — | ✔ — | ✔ — | ✔ — |
| ctrl_mlp_small_data | control | ✔ — | ✔ — | ✔ — | ✔ — | ✔ — |
| ctrl_particles_knn | control | ✔ — | ✔ — | ✔ — | ✔ — | ✔ — |
| leak_customer_rows | data_leakage | ✘ — (missed: data.train_test_overlap) | ✘ — (missed: data.train_test_overlap) | ✘ — (missed: data.train_test_overlap) | ✔ data.train_test_overlap | ✔ data.train_test_overlap |
| leak_merchant_encoding | data_leakage | ✘ — (missed: data.target_leak) | ✘ — (missed: data.target_leak) | ✘ — (missed: data.target_leak) | ✔ data.target_leak | ✘ data.train_test_overlap, data.target_leak (FP: data.train_test_overlap) |
| metric_review_f1 | incorrect_metric | ✘ — (missed: eval.metric_mismatch) | ✘ — (missed: eval.metric_mismatch) | ✘ — (missed: eval.metric_mismatch) | ✘ repro.claim_not_met (missed: eval.metric_mismatch) | ✘ data.train_test_overlap, eval.metric_mismatch (FP: data.train_test_overlap) |
| other_class_weights | other | ✘ — (missed: other:class-weights-ignored) | ✘ — (missed: other:class-weights-ignored) | ✘ — (missed: other:class-weights-ignored) | ✘ other:class_weight_ignored, repro.claim_not_met (FP: other:class_weight_ignored) (missed: other:class-weights-ignored) | ✘ other:class_weight_unused (FP: other:class_weight_unused) (missed: other:class-weights-ignored) |
| other_label_join | other | ✘ — (missed: other:label-join-noise) | ✘ repro.claim_not_met (missed: other:label-join-noise) | ✘ repro.claim_not_met (missed: other:label-join-noise) | ✘ repro.claim_not_met (missed: other:label-join-noise) | ✘ repro.claim_not_met (missed: other:label-join-noise) |
| prep_plan_codes | broken_preprocessing | ✘ — (missed: pipeline.eval_path_mismatch) | ✘ repro.claim_not_met (missed: pipeline.eval_path_mismatch) | ✘ repro.claim_not_met (missed: pipeline.eval_path_mismatch) | ✔ pipeline.eval_path_mismatch | ✘ repro.claim_not_met (missed: pipeline.eval_path_mismatch) |
| repro_seeds_softmax | reproducibility | ✘ — (missed: repro.nondeterministic) | ✔ repro.nondeterministic | ✘ — (missed: repro.nondeterministic) | ✔ repro.nondeterministic, repro.claim_not_met | ✔ repro.nondeterministic |
| repro_tabular_config | reproducibility | ✘ — (missed: repro.claim_not_met) | ✔ repro.claim_not_met | ✔ repro.claim_not_met | ✘ — (missed: repro.claim_not_met) | ✔ repro.claim_not_met |
