# nnreg

Random-feature (tanh) ridge regression for listing prices. Each seed re-draws both the random features and the
train/validation split, so per-seed numbers differ (see `per_seed_val_rmse` in `summary.json`); we report the mean
over the five seeds in `exp.ini`. Only a validation split is used in this repository - the test split is reserved
for the final paper - so every metric here is a **validation** metric.

## Run

    python -m nnreg.cli --ini exp.ini

Writes `runs/exp1/seed_<k>/metrics.json` and `runs/exp1/summary.json`.

## Result

Validation RMSE over 5 seeds: mean **TBD**, std TBD (validation, not test).
