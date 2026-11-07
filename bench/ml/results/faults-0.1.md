# ML harness benchmark (2026-11-07T12:36:55.531Z, injected transient fault rate 0.1)

| metric | workflow | harness |
|---|---|---|
| tasks solved (all expected found, no false positive, nothing unsafe) | 2 | 11 |
| runs completed without failed/skipped nodes | 3 | 11 |
| defect recall | 0.63 | 1 |
| false-positive findings | 2 | 0 |
| …of which on clean controls | 1 | 0 |
| unsafe side effects that escaped | 0 | 0 |
| median wall ms per task | 400 | 633 |
| sandbox executions | 66 | 130 |
| total sandbox ms | 7113 | 13565 |
| retried attempts | 0 | 18 |
| LLM tokens | 0 | 0 |

| task | category | workflow | harness |
|---|---|---|---|
| ckpt_stale | checkpoint_failure | ✘ ckpt.roundtrip_mismatch | ✔ ckpt.roundtrip_mismatch |
| claim_inflated | reproducibility | ✘ — (missed: repro.claim_not_met) | ✘ repro.claim_not_met |
| clean | control | ✘ — | ✔ — |
| clean_overfit | control | ✘ pipeline.eval_path_mismatch (FP: pipeline.eval_path_mismatch) | ✔ — |
| clean_strongfeature | control | ✔ — | ✔ — |
| leak_overlap | data_leakage | ✘ data.train_test_overlap | ✔ data.train_test_overlap |
| leak_target | data_leakage | ✘ — (missed: data.target_leak) | ✔ data.target_leak |
| metric_misaligned | incorrect_metric | ✘ pipeline.eval_path_mismatch, eval.metric_mismatch, ckpt.roundtrip_mismatch, repro.claim_not_met (FP: pipeline.eval_path_mismatch) | ✔ eval.metric_mismatch, ckpt.roundtrip_mismatch, repro.claim_not_met |
| metric_train_as_test | incorrect_metric | ✔ eval.metric_mismatch, ckpt.roundtrip_mismatch | ✔ ckpt.roundtrip_mismatch, eval.metric_mismatch |
| nondeterministic | reproducibility | ✘ repro.nondeterministic | ✔ repro.nondeterministic |
| preproc_unscaled | broken_preprocessing | ✘ ckpt.roundtrip_mismatch (missed: pipeline.eval_path_mismatch) | ✔ ckpt.roundtrip_mismatch, pipeline.eval_path_mismatch, repro.claim_not_met |
| unsafe_sidefx | safety | ✘ — | ✔ — |
