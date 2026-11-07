# assay-classifier

Classifies assay samples (`data/raw/features.csv`, eight measurements `m1`..`m8`) using diagnosis
labels delivered separately by the lab (`data/raw/labels.csv`, one row per sample, shuffled).

    python workflow/run_classifier.py

Joins features to labels by sample id, splits 75/25, trains a logistic regression and writes
`reports/summary.json`.

## Result

**0.89 test accuracy.** The lab says all 2400 samples have a label.
