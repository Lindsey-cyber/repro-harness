# credit-lr

Logistic-regression default model using income, tenure and a smoothed per-branch default rate (the branch column
has ~200 levels, too many for one-hot, so it is encoded by its smoothed mean default rate, `smooth` in the config).

## Run

    make train

Writes `out/lr_branch/metrics.json`. Test set is a random 25% of loans (`seed` in `configs/lr.yaml`).

## Result

Test auc **TBD**, accuracy TBD.
