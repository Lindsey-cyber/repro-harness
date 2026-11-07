# loan-default-baseline

Baseline default classifier on application records.

## Run

    python scripts/run_loan.py --seed 11 --test-frac 0.25

Writes `artifacts/metrics.json` and `artifacts/predictions.csv`.

## Result

Accuracy on the held-out 25%: **0.820**.

`prior_defaults` (count of the applicant's earlier defaults from the bureau file) is by far
the most important feature, which is why accuracy is high; it is known at application time.
The `app_id` column is only used to key the predictions file.
