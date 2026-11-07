# ML harness benchmark (2026-11-07T15:58:43.661Z, injected transient fault rate 0)

| metric | naive | workflow | harness | agent | hybrid |
|---|---|---|---|---|---|
| tasks solved (all expected found, no false positive, nothing unsafe) | 1 | 1 | 1 | 7 | 1 |
| runs completed without failed/skipped nodes | 3 | 7 | 7 | 12 | 3 |
| defect recall | 0 | 0 | 0 | 0.56 | 0.33 |
| false-positive findings | 0 | 0 | 0 | 5 | 2 |
| …of which on clean controls | 0 | 0 | 0 | 0 | 0 |
| unsafe side effects that escaped | 0 | 0 | 0 | 0 | 0 |
| median wall ms per task | 0 | 189 | 171 | 65970 | 206649 |
| sandbox executions | 0 | 18 | 18 | 121 | 64 |
| total sandbox ms | 0 | 3927 | 3946 | 72148 | 25052 |
| retried attempts | 0 | 0 | 0 | 0 | 0 |
| LLM tokens | 0 | 0 | 0 | 857839 | 309746 |
| LLM cost, USD (estimated at list price) | 0 | 0 | 0 | 0.05 | 0.26 |
| model hypotheses (hybrid) | — | — | — | — | 14 proposed / 3 confirmed / 11 refuted / 0 unverified |

| task | category | naive | workflow | harness | agent | hybrid |
|---|---|---|---|---|---|---|
| ckpt_signal_mlp | checkpoint_failure | ✘ — (missed: ckpt.roundtrip_mismatch) | ✘ — (missed: ckpt.roundtrip_mismatch) | ✘ — (missed: ckpt.roundtrip_mismatch) | ✘ pipeline.eval_path_mismatch, other:eval-preprocessing-not-shared (FP: other:eval-preprocessing-not-shared) (missed: ckpt.roundtrip_mismatch) | ✘ ckpt.roundtrip_mismatch |
| ctrl_credit_logreg | control | ✘ — | ✘ — | ✘ — | ✔ — | ✘ — |
| ctrl_mlp_small_data | control | ✘ — | ✘ — | ✘ — | ✔ — | ✘ — |
| ctrl_particles_knn | control | ✔ — | ✔ — | ✔ — | ✔ — | ✔ — |
| leak_customer_rows | data_leakage | ✘ — (missed: data.train_test_overlap) | ✘ — (missed: data.train_test_overlap) | ✘ — (missed: data.train_test_overlap) | ✔ data.train_test_overlap | ✘ — (missed: data.train_test_overlap) |
| leak_merchant_encoding | data_leakage | ✘ — (missed: data.target_leak) | ✘ — (missed: data.target_leak) | ✘ — (missed: data.target_leak) | ✘ data.target_leak, other:unexported_eval_feature_path (FP: other:unexported_eval_feature_path) | ✘ — (missed: data.target_leak) |
| metric_review_f1 | incorrect_metric | ✘ — (missed: eval.metric_mismatch) | ✘ — (missed: eval.metric_mismatch) | ✘ — (missed: eval.metric_mismatch) | ✘ other:threshold-on-logit, repro.claim_not_met (FP: other:threshold-on-logit) (missed: eval.metric_mismatch) | ✘ data.train_test_overlap, other:probability_threshold_on_logits (FP: data.train_test_overlap, other:probability_threshold_on_logits) (missed: eval.metric_mismatch) |
| other_class_weights | other | ✘ — (missed: other:class-weights-ignored) | ✘ — (missed: other:class-weights-ignored) | ✘ — (missed: other:class-weights-ignored) | ✘ repro.claim_not_met, other:class-weights-unused (FP: other:class-weights-unused) (missed: other:class-weights-ignored) | ✘ — (missed: other:class-weights-ignored) |
| other_label_join | other | ✘ — (missed: other:label-join-noise) | ✘ — (missed: other:label-join-noise) | ✘ — (missed: other:label-join-noise) | ✘ repro.claim_not_met, other:label_join_case_mismatch (FP: other:label_join_case_mismatch) (missed: other:label-join-noise) | ✘ — (missed: other:label-join-noise) |
| prep_plan_codes | broken_preprocessing | ✘ — (missed: pipeline.eval_path_mismatch) | ✘ — (missed: pipeline.eval_path_mismatch) | ✘ — (missed: pipeline.eval_path_mismatch) | ✔ pipeline.eval_path_mismatch, repro.claim_not_met | ✘ — (missed: pipeline.eval_path_mismatch) |
| repro_seeds_softmax | reproducibility | ✘ — (missed: repro.nondeterministic) | ✘ — (missed: repro.nondeterministic) | ✘ — (missed: repro.nondeterministic) | ✔ repro.nondeterministic | ✘ repro.nondeterministic |
| repro_tabular_config | reproducibility | ✘ — (missed: repro.claim_not_met) | ✘ — (missed: repro.claim_not_met) | ✘ — (missed: repro.claim_not_met) | ✔ repro.claim_not_met | ✘ repro.claim_not_met |
