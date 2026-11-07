# ticket-escalation

Predict whether a support ticket is escalated (rare, about 1 in 5) from wait time, text length,
priority and channel.

## Run

    python run.py --config config.json

Outputs go to `out/`: `metrics.json` and `preds.csv` (true and predicted label per test ticket).

## Labels

Classes are indexed alphabetically (`escalated`=0, `resolved`=1).

## Result

F1 for the escalated class on the test split: **0.953**.
