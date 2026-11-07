# visits

k-nearest-neighbour baseline for predicting clinical outcome (improved / stable) from six visit measurements.

Records come from the clinic visit log (`data/visits.csv`, several visits per subject). Evaluation is on a
held-out set of subjects chosen by hashing the subject id into buckets (`[split]` in `params.ini`).

## Run

    python -m visits.run --ini params.ini --out out

Writes `out/knn_k5/metrics.json` and `out/knn_k5/preds.csv`.

## Result

Held-out subjects, k=5: accuracy **TBD**.
