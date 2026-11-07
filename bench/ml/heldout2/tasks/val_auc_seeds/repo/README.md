# val-auc-lr

Ridge-regularised logistic regression on 20 tabular features. The organisers hold the
final test labels, so this repo only ever reports **validation** AUC; no test data is
present and no number here is a test estimate.

## Run

    python run_experiment.py --config conf/exp.json --seeds 10

Seed `s` (0..9) controls initial weights and minibatch order (`np.random.RandomState(s)`).
Output: `out/val_report.json`.

## Result

Validation AUC, mean over 10 seeds: **0.818** (std across seeds below 0.001).
The regularisation strength in `conf/exp.json` was set by hand and is not tuned per seed.
