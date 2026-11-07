# churn-events

Subscriber churn from per-event features collected by two loggers (A and B). `data/events.csv` is the
concatenation of the two logger exports. `period` marks which half of the billing cycle an event falls in.
Logistic regression, random 75/25 split.

## Run

    python train_churn.py --data data/events.csv --out results/churn_v2

Writes `results/churn_v2/metrics/summary.json`.

## Result

Test accuracy **TBD**.
