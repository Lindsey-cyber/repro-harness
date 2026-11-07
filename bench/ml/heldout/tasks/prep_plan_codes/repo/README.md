# subscription-renewal

Predicts whether a subscription renews from plan tier, sales channel and usage.

    python pipeline.py config.json

Categorical columns are one-hot encoded, everything is standardised with training statistics, and a
logistic regression is trained with full-batch gradient descent. Writes `out/metrics.json`.

## Result

**0.72 accuracy** on `data/test.csv` (1000 rows).
