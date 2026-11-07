# diag-logreg

L2-regularised logistic regression (full-batch gradient descent) on eight assay measurements for a binary
diagnosis. Class balance is about 30% positive.

## Run

    python fit.py --data data/diag.csv --l2 0.002 --lr 0.5 --epochs 300 --seed 4

Writes `results/l2_0.002/metrics.json`. 70/30 random split.

## Result

Test accuracy **TBD**, logloss **TBD**.
