# expr-panel

Notebook (exported with jupytext to `analysis.py`) that screens a 500-gene panel down to the top-k genes by
absolute correlation with subtype, then fits a small ridge-penalised logistic regression. 25% of samples are
held out for testing.

Data is in `data/expr.npz` (arrays `X` of shape 300x500 and `y` in {0,1}).

## Run

    python analysis.py --out out --k 20

Writes `out/nb/run1/metrics.json`.

## Result

Held-out roc_auc **TBD**, accuracy TBD.
