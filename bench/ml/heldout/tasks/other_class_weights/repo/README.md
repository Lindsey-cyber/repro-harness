# txn-fraud-lr

Logistic regression for card-fraud detection on a heavily imbalanced set (about 5% fraud).
`config.json` sets `class_weight: balanced` to compensate for the imbalance.

    python train.py

Writes `eval_report.json` (accuracy, balanced accuracy, fraud recall).

## Result

With balanced class weights the model reaches **0.74 balanced accuracy** on the 25% held-out split, which
is the number to compare, since plain accuracy is dominated by the legitimate class.
