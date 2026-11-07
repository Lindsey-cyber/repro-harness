# readmit-knn

k-nearest-neighbour baseline for 30-day readmission from vitals and labs. One row per
hospital visit.

## Run

    python experiments/run.py --config experiments/knn.json

Writes `results/<name>.json` where `name` is set in the config.

## Result

Test AUC: **0.836** (k=7, 25% test split, split seed 7).
