# review-sentiment-lr

Logistic regression over eight hand-built review features (`data/reviews.csv`, label `positive`).

    python train.py              # reads params.json

Writes `report/metrics.json` and `report/predictions.csv`.

## Result

Test **F1 of 0.58** for the positive class at the default 0.5 probability threshold
(70/30 split, seed 5).
