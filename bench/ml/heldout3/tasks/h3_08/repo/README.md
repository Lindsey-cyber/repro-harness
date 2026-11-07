# multisite-trial

Predicts treatment-arm response from four covariates across 12 trial sites. Evaluation holds out whole sites
(4 of 12, drawn with the seeded generator from `config.json`) and fits a logistic regression on the rest.

## Run

    bash run.sh

Writes `results/trial_a/metrics.json`.

## Result

Accuracy on held-out sites: **TBD**.
