# lowdata-logreg

Logistic regression on a small, 50-feature tabular problem (110 training rows). The epoch
with the best validation accuracy is kept as the checkpoint.

## Run

    python scripts/train.py --config configs/run.cfg
    python scripts/predict.py --config configs/run.cfg

`train.py` writes `ckpt/best.npz` and `runs/latest/metrics.json`; `predict.py` reloads the
checkpoint and writes `runs/latest/reload_metrics.json`.

## Result

Test accuracy of the best-validation checkpoint: **0.658**
(`test_acc` in `runs/latest/metrics.json`).
