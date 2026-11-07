# margin-perceptron

Sanity-check experiment for the training harness: a classic perceptron on a synthetic 2-D problem whose classes
are linearly separable with a margin by construction (see `data/GENERATING.txt`). The perceptron convergence
theorem guarantees zero training errors, and held-out points come from the same rule, so a perfect score is the
expected outcome, not a bug.

## Run

    bash run.sh

Writes `results/perceptron/metrics.json`.

## Result

Test accuracy **TBD**, roc_auc **TBD**.
