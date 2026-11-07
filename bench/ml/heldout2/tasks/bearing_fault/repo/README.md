# bearing-fault

Binary fault detector for motor bearings from six condition-monitoring channels.

## Run

    python train.py --config configs/default.json

Writes `results/metrics.json`.

## Result

Test accuracy: **1.000** on `data/test.csv` (500 recordings).

The faulty regime shifts vibration RMS, peak vibration and acoustic level by several
standard deviations, so the classes are almost linearly separable; the model is a plain
logistic regression trained with full-batch gradient descent.
