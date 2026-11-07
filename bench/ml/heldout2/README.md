# heldout2: held-out ML-debugging benchmark

`python3 make.py` writes `tasks/<id>/repo/` (what the harness sees) and `tasks/<id>/task.json`
(ground truth). `python3 make.py --fixed DIR` writes bug-fixed copies to `DIR/<id>/` for
verification only. Only numpy and pandas are used; everything is seeded, so repeated runs
produce byte-identical files (verified by hashing all 69 files twice). `make.py` runs every
repo in a scratch dir (`$H2_SCRATCH` or a temp dir) to calibrate README claims and write the
evidence into `task.json`; nothing is run inside `tasks/`.

## Tasks

| id | category | headline | defect (or why clean) | buggy vs fixed |
|---|---|---|---|---|
| bearing_fault | control | accuracy | legit near-perfect (5-sigma separated regimes) | 1.000 / n.a. |
| meter_events | control | f1 | legit duplicate rows recurring across splits (quantised logger data, documented) | 0.726 / n.a. |
| val_auc_seeds | control | auc | clearly labelled validation AUC, mean over 10 explicit seeds, npz data | 0.818 / n.a. |
| loan_default | control | accuracy | legit strong feature `prior_defaults`; id and string categoricals handled correctly | 0.820 / n.a. |
| adclick_te | data_leakage | accuracy | target encoding of high-cardinality campaign computed over all rows before split | 0.703 / 0.565 |
| readmit_rows | data_leakage | auc | row-level split of multi-visit patients (entity overlap), kNN | 0.836 / 0.568 |
| churn_plans | broken_preprocessing | accuracy | `pd.factorize` first-appearance order differs between train and test files in a separate eval script | 0.587 / 0.750 |
| ticket_f1 | incorrect_metric | f1 | F1 computed for class index 1 (majority) via config `pos_label`, README claims escalated class | 0.953 / 0.301 |
| lowdata_ckpt | checkpoint_failure | accuracy | "best" checkpoint aliases the in-place-updated weight array; reload gives final-epoch weights | reload 0.597 / 0.658 |
| topic_vocab | reproducibility | accuracy | `list(set(...))[:200]` vocab cap depends on PYTHONHASHSEED | 0.843-0.939 over 6 hash seeds / 1.000 |
| wine_stop | other | accuracy | epoch picked by max test accuracy (selection on test) | 0.742 / 0.683 |
| birdcall_subset | other | accuracy | headline computed on test rows with snr >= 8 only; predictions file covers all rows | 0.980 / 0.898 |

Variety: entrypoints are `train.py`, `python -m pkg.cli run`, `run_experiment.py`, `scripts/*.py`,
`main.py`, `experiments/run.py`, `train.py`+`eval.py`, `run.py --config`, `scripts/train.py`+`predict.py`,
`run_nb.py`, `python -m wq.fit`, `run.py config.json`. Configs are json, a configparser `.cfg`, CLI flags
or none. Labels: fault, event, (npz y), defaulted, clicked, readmitted, churned, outcome (strings), y,
topic (strings), good, species (strings). Data are CSV (some with id/categorical string columns, one
text column) plus one `.npz`. Predictions files exist in loan_default, ticket_f1, birdcall_subset.

## How each task was verified

For every task `make.py` itself (a) runs the README commands in a scratch copy and reads the metric
(`task.json.evidence` holds the numbers, including stdout), and (b) for defects, applies the minimal fix
and re-runs. Controls were run twice (identical output) and their README claim equals the printed
value to 3 decimals. Defect effects: adclick +0.137, readmit +0.267 AUC, churn_plans -0.163 (eval vs
train-time holdout 0.737), ticket_f1 +0.652 (recomputing F1 for each class from `out/preds.csv` gives
0.301 escalated / 0.953 resolved), lowdata_ckpt -0.061 on reload, topic_vocab spread 0.096 across
hash seeds, wine_stop +0.058, birdcall +0.082 (recomputing over all rows of predictions.csv gives 0.898).
`--fixed` output differs from `repo/` in exactly one file per defective task and is identical for controls.

## Decisions

- Claims: for adclick, readmit, ticket, ckpt, wine, birdcall the README claim equals the buggy number the
  code prints (so `repro.claim_not_met` is not a consequence). For churn_plans and topic_vocab the claim
  is the number from the fixed code (authors measured before the regression), so `claim_not_met` is a
  genuine consequence and is in `allow`.
- wine_stop allows `data.target_leak` (test labels drive model selection); birdcall_subset allows
  `eval.metric_mismatch` (rescoring predictions.csv gives a different number).
- readmit_rows allows nothing extra: `target_leak` is not a distinct consequence of entity overlap.
- topic_vocab runs with PYTHONHASHSEED unset (random per process). A harness that pins
  PYTHONHASHSEED will see a deterministic but arbitrary result; task.json evidence uses seeds 1..6.
- The fixed adclick copy still target-encodes train rows in-sample (smoothed), a standard approach.
