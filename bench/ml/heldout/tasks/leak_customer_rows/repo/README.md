# churn-knn

Predicting monthly churn from account snapshots (`data/accounts.csv`: 8 monthly snapshots per
account; features are account profile and recent activity; label `churned`).

    python run_experiment.py --k 5 --seed 7

Writes `results/knn_metrics.json`. 75/25 random split, features standardised with training
statistics, majority vote of the k nearest training snapshots.

## Result

**0.76 test accuracy** against a majority-class baseline of about 0.5.
