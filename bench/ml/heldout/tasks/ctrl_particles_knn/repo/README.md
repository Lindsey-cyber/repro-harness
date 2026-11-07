# event-classifier-knn

k-NN signal/background classification on 8 reconstructed event features.

    python scripts/train.py            # uses configs/knn.cfg
    python scripts/train.py configs/knn.cfg

Writes `results/summary.json`.

## Result

On the 600 held-out events we reach **0.9933 test accuracy** (k = 5). The classes are well
separated in feature space by construction of the selection, so near-perfect accuracy is expected.
