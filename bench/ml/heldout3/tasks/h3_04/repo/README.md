# ticket-triage

Softmax-regression baseline that routes support tickets to refund / login / abuse queues from four normalised
text-statistics features. Abuse is rare (under 10% of tickets).

## Run

    bash run.sh

`train.py` fits the model and writes `out/preds.csv`; `score.py` scores the predictions file into
`out/scores.json`.

## Result

f1_macro **TBD** (accuracy TBD) on the 30% held-out split.
