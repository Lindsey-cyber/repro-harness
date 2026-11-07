# loan-default-baseline

A logistic-regression baseline for predicting loan default from application data
(`data/train.csv`, `data/test.csv`; label column `defaulted`).

    python main.py

writes `outputs/metrics.json` and `outputs/test_predictions.csv`.

## Result

Held-out **test AUC of 0.8521**. Most of the signal comes from `prior_delinq`
(the number of previous delinquencies on file at application time), which is available when a
loan is scored.
