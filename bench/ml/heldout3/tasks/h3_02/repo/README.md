# sensorclf

Condition classifier (ok / warn / fault) for rotating-equipment units from vibration summary statistics.

## Data

`data/readings.csv`: one row per logger reading. Several readings per unit. The logger occasionally writes a
reading twice, so some rows are exact duplicates; we leave them in because the deployed pipeline sees the same
stream. Train/test are split by `unit_id` (whole units are held out).

## Run

    python -m sensorclf.train --data data/readings.csv --out runs --seed 1 --lr 0.1 --epochs 400

Writes `runs/softmax_lr0.1/seed1/metrics.json`.

## Result

Held-out units: f1_macro **TBD**, accuracy **TBD**.
