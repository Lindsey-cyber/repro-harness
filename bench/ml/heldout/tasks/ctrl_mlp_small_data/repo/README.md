# small-data-mlp

A one-hidden-layer tanh MLP on a small, noisy tabular problem (320 training rows, 30 features,
about 8% label noise).

    python src/pipeline/run.py                  # trains, keeps the best-validation checkpoint -> runs/baseline/
    python src/pipeline/evaluate.py             # reloads runs/baseline/model.npz and scores data/test.csv

The checkpoint stores the weights and the training-set standardisation statistics.

## Result

**0.764 test accuracy** (1500 held-out rows). Training accuracy at the selected checkpoint is 0.91, so there is a
large train/test gap; with 320 rows and 128 hidden units this is expected, and we did not tune against it.
