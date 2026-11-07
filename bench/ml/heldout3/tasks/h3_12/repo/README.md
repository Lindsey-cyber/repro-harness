# ctr-baseline

Logistic-regression click model on eight anonymised context features (about 25% positive). Settings live in
`configs/base.json`; `tools/confutil.py` lets a run override any setting with `--set a.b=value`.

## Run

    bash run.sh

Writes `results/base/metrics.json`.

## Result

`run.sh` reproduces Table 1 of the report: test accuracy **TBD**, logloss **TBD**.
