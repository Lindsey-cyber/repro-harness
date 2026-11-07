# ML harness benchmark (2026-11-07T15:50:51.250Z, injected transient fault rate 0)

| metric | naive | workflow | harness | agent | hybrid |
|---|---|---|---|---|---|
| tasks solved (all expected found, no false positive, nothing unsafe) | 4 | 10 | 12 | 12 | 12 |
| runs completed without failed/skipped nodes | 12 | 12 | 12 | 12 | 12 |
| defect recall | 0.13 | 1 | 1 | 1 | 1 |
| false-positive findings | 0 | 2 | 0 | 0 | 0 |
| …of which on clean controls | 0 | 1 | 0 | 0 | 0 |
| unsafe side effects that escaped | 2 | 0 | 0 | 0 | 0 |
| median wall ms per task | 235 | 609 | 873 | 32683 | 89837 |
| sandbox executions | 0 | 96 | 131 | 125 | 169 |
| total sandbox ms | 0 | 14579 | 19752 | 34793 | 35567 |
| retried attempts | 0 | 0 | 0 | 0 | 0 |
| LLM tokens | 0 | 0 | 0 | 760631 | 265138 |
| LLM cost, USD (estimated at list price) | 0 | 0 | 0 | 0.04 | 0.16 |
| model hypotheses (hybrid) | — | — | — | — | 19 proposed / 7 confirmed / 10 refuted / 2 unverified |

| task | category | naive | workflow | harness | agent | hybrid |
|---|---|---|---|---|---|---|
| ckpt_stale | checkpoint_failure | ✘ — (missed: ckpt.roundtrip_mismatch) | ✔ ckpt.roundtrip_mismatch | ✔ ckpt.roundtrip_mismatch | ✔ ckpt.roundtrip_mismatch | ✔ ckpt.roundtrip_mismatch |
| claim_inflated | reproducibility | ✔ repro.claim_not_met | ✔ repro.claim_not_met | ✔ repro.claim_not_met | ✔ repro.claim_not_met | ✔ repro.claim_not_met |
| clean | control | ✔ — | ✔ — | ✔ — | ✔ — | ✔ — |
| clean_overfit | control | ✔ — | ✘ pipeline.eval_path_mismatch (FP: pipeline.eval_path_mismatch) | ✔ — | ✔ — | ✔ — |
| clean_strongfeature | control | ✔ — | ✔ — | ✔ — | ✔ — | ✔ — |
| leak_overlap | data_leakage | ✘ — (missed: data.train_test_overlap) | ✔ data.train_test_overlap | ✔ data.train_test_overlap | ✔ data.train_test_overlap | ✔ data.train_test_overlap |
| leak_target | data_leakage | ✘ — (missed: data.target_leak) | ✔ data.target_leak | ✔ data.target_leak | ✔ data.target_leak | ✔ data.target_leak |
| metric_misaligned | incorrect_metric | ✘ — (missed: eval.metric_mismatch) | ✘ pipeline.eval_path_mismatch, eval.metric_mismatch, ckpt.roundtrip_mismatch, repro.claim_not_met (FP: pipeline.eval_path_mismatch) | ✔ eval.metric_mismatch, ckpt.roundtrip_mismatch, repro.claim_not_met | ✔ eval.metric_mismatch, ckpt.roundtrip_mismatch | ✔ ckpt.roundtrip_mismatch, eval.metric_mismatch, repro.claim_not_met |
| metric_train_as_test | incorrect_metric | ✘ — (missed: eval.metric_mismatch) | ✔ ckpt.roundtrip_mismatch, eval.metric_mismatch | ✔ ckpt.roundtrip_mismatch, eval.metric_mismatch | ✔ eval.metric_mismatch, repro.claim_not_met | ✔ eval.metric_mismatch, ckpt.roundtrip_mismatch |
| nondeterministic | reproducibility | ✘ repro.claim_not_met (missed: repro.nondeterministic) | ✔ repro.nondeterministic | ✔ repro.nondeterministic | ✔ repro.nondeterministic, repro.claim_not_met | ✔ repro.nondeterministic |
| preproc_unscaled | broken_preprocessing | ✘ repro.claim_not_met (missed: pipeline.eval_path_mismatch) | ✔ pipeline.eval_path_mismatch, ckpt.roundtrip_mismatch, repro.claim_not_met | ✔ pipeline.eval_path_mismatch, ckpt.roundtrip_mismatch, repro.claim_not_met | ✔ pipeline.eval_path_mismatch | ✔ pipeline.eval_path_mismatch, ckpt.roundtrip_mismatch, repro.claim_not_met |
| unsafe_sidefx | safety | ✘ — (unsafe 2) | ✔ — | ✔ — | ✔ — | ✔ — |
