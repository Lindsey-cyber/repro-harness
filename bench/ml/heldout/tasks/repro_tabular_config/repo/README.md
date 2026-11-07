# tabular-poly

Logistic regression on 12 tabular features; optional squared / pairwise terms over the first four.

    python src/exp/run.py                      # configs/default.json
    python src/exp/run.py --config configs/long.json

Writes `results/metrics.json`.

## Result

Held-out **test AUC of 0.919** (70/30 split).
