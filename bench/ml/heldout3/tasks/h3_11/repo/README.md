# device-failure

Predicts device failure (fail / ok) from three standardised telemetry features and the deployment region
(one-hot). `data/train.csv` is split 80/20 internally for validation; `data/test.csv` is a separate hold-out file.

## Run

    make train && make eval

`train.py` writes `runs/v1/model.npz`; `evaluate.py` loads it and scores `data/test.csv`
(`runs/v1/eval/metrics.json`, `runs/v1/eval/predictions.csv`).

## Result

Test accuracy (from `make eval`): **TBD**.
