# Held-out ML-debugging benchmark

12 small numpy-only experiment repos with ground truth. `python3 make.py` (numpy + pandas only, deterministic)
writes `tasks/<id>/repo/` (what the harness sees) and `tasks/<id>/task.json` (ground truth, never shown).
`python3 make.py --fixed DIR` additionally writes bug-fixed copies of the defective repos to `DIR/<id>`
(used only to measure effects; not committed). Each repo runs in under 20 s on one CPU core.

| id | category | expect | one-line defect |
|---|---|---|---|
| ctrl_credit_logreg | control | - | none; legitimately strong feature (prior_delinq), AUC 0.85 |
| ctrl_particles_knn | control | - | none; legitimately 0.9933 accuracy (well separated classes, npz data, cfg file) |
| ctrl_mlp_small_data | control | - | none; train 0.91 vs test 0.76 gap is genuine overfit, checkpoint round trip exact |
| leak_customer_rows | data_leakage | data.train_test_overlap | random row split over 8 snapshots/customer; k-NN memorises customers |
| leak_merchant_encoding | data_leakage | data.target_leak | merchant mean-target encoding computed on all rows before the split |
| prep_plan_codes | broken_preprocessing | pipeline.eval_path_mismatch | `pd.factorize` applied per split, so one-hot columns mean different categories in train and test |
| metric_review_f1 | incorrect_metric | eval.metric_mismatch | logits compared with a 0.5 probability threshold |
| ckpt_signal_mlp | checkpoint_failure | ckpt.roundtrip_mismatch | checkpoint omits standardisation mu/sd; eval script feeds raw features |
| repro_seeds_softmax | reproducibility | repro.nondeterministic | `--seed` seeds only legacy RNG; minibatch order uses an unseeded `default_rng()` |
| repro_tabular_config | reproducibility | repro.claim_not_met | README claim (0.919) comes from configs/long.json; documented command uses default.json (0.69) |
| other_label_join | other | other:label-join-noise | case-sensitive id join, misses filled with 0, so ~30% of positives become negatives |
| other_class_weights | other | other:class-weights-ignored | sample weights computed from `class_weight: balanced` but never used in the gradient |

Layouts, entrypoints, configs (json/cfg/argparse), outputs, label names, headline metrics (accuracy, AUC, F1,
balanced accuracy) and data formats (CSV, one npz) deliberately differ across repos. Some write prediction
files (credit, merchant, review), some have a separate eval script (mlp_small, ckpt).

## How each was verified

Every repo was copied to a scratch dir and run from its root exactly as its README says; observed numbers are
in each task's `evidence` field. Defective repos were also run after applying the patch in `FIXES`
(`make.py --fixed`). Results (buggy vs fixed):

- leak_customer_rows: accuracy 0.761 vs 0.514 (group split; chance 0.51)
- leak_merchant_encoding: AUC 0.8225 vs 0.5951
- prep_plan_codes: accuracy 0.561 vs 0.723
- metric_review_f1: F1 0.2779 vs 0.5756 (same weights)
- ckpt_signal_mlp: reloaded checkpoint test 0.4943 vs logged 0.7343; with mu/sd saved the round trip is exact 0.7343
- repro_seeds_softmax: 10 runs of the same command: 0.8233 to 0.8467 (6 distinct values); fixed: 0.8467 every time
- repro_tabular_config: documented command 0.6898 vs claim 0.919 (long.json reproduces 0.919)
- other_label_join: accuracy 0.7633 vs 0.8883
- other_class_weights: balanced accuracy 0.5636 (fraud recall 0.13) vs 0.7432 (recall 0.70)
- controls: README claim equals the observed number (0.8521, 0.9933, 0.764), reruns identical.

Determinism: `make.py` run twice gives identical md5s for every generated file. Re-running every repo twice gives
identical outputs except repro_seeds_softmax (the intended defect). Generated outputs were produced only in
scratch copies, so `tasks/` contains no run artifacts.

## Decisions

- Claims in defective READMEs are what the authors "saw" before the defect mattered: leakage tasks claim the
  inflated number the code does reproduce; preprocessing, metric, checkpoint (eval path), label-join,
  class-weight and config tasks claim the corrected number, so `repro.claim_not_met` is a true consequence
  and sits in `allow` (but is `expect` only for repro_tabular_config, where it is the defect).
- ckpt_signal_mlp: an initial best-epoch-aliasing design had a negligible effect (0.734 vs 0.731) because the
  noisy problem plateaus, so the defect was changed to a missing normalisation state, which collapses accuracy to
  chance. `pipeline.eval_path_mismatch` is in `allow` there (the eval path really does skip standardisation).
- other_class_weights headline is balanced accuracy: with macro-F1 the weighted and unweighted models scored about
  the same (0.57 vs 0.60), so the defect did not move that metric.
- prep_plan_codes reports the buggy number in its own run; the fixed model only reaches 0.72 because the
  labels are noisy.
- No yaml (not guaranteed installed): configs are json or `.cfg`.
