# churn-plans

Subscription churn model: one-hot plan and region, standardised tenure and monthly charge,
logistic regression.

## Run

    python train.py
    python eval.py

`train.py` fits on `data/train.csv` (80/20 internal split) and saves `model/model.npz`.
`eval.py` scores `data/test.csv` and writes `reports/test_metrics.json`.

## Result

Test accuracy (from `reports/test_metrics.json`): **0.750**.
