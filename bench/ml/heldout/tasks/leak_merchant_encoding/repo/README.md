# claims-fraud-scorer

Scores insurance claims for fraud with a logistic regression over claim attributes plus a
per-merchant fraud-rate feature (`merchant_rate`, a mean-target encoding of the `merchant` column).

    python scripts/fit_and_score.py

Writes `artifacts/metrics.json` and `artifacts/test_scores.csv`. 70/30 random split.

## Result

The merchant-rate feature lifts held-out **test AUC to 0.82**.
