"""Train a logistic-regression default-risk model and score the held-out test file."""
import argparse
import json
import os

import numpy as np
import pandas as pd

FEATURES = ["age", "income_k", "debt_ratio", "utilization", "tenure_months", "prior_delinq"]
LABEL = "defaulted"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def roc_auc(y, s):
    ranks = pd.Series(s).rank().to_numpy()
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_logreg(X, y, lr, epochs, l2):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w)
        b -= lr * float(np.mean(p - y))
    return w, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--l2", type=float, default=1e-3)
    a = ap.parse_args()

    train = pd.read_csv(os.path.join(a.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(a.data_dir, "test.csv"))
    mu, sd = train[FEATURES].mean(), train[FEATURES].std()
    Xtr = ((train[FEATURES] - mu) / sd).to_numpy()
    Xte = ((test[FEATURES] - mu) / sd).to_numpy()
    ytr, yte = train[LABEL].to_numpy(), test[LABEL].to_numpy()

    w, b = fit_logreg(Xtr, ytr, a.lr, a.epochs, a.l2)
    s_tr, s_te = sigmoid(Xtr @ w + b), sigmoid(Xte @ w + b)
    metrics = {
        "test_auc": round(roc_auc(yte, s_te), 4),
        "train_auc": round(roc_auc(ytr, s_tr), 4),
        "n_train": len(train),
        "n_test": len(test),
    }
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({"score": s_te, "label": yte}).to_csv(
        os.path.join(a.out, "test_predictions.csv"), index=False)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
