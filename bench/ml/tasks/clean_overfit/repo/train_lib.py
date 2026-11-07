import csv
import numpy as np


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = [c for c in rows[0] if c != "label"]
    X = np.array([[float(r[c]) for c in cols] for r in rows])
    y = np.array([float(r["label"]) for r in rows])
    return X, y


def features(X):
    return np.hstack([X] + [X[:, [i]] * X[:, [j]] for i in range(X.shape[1]) for j in range(i, X.shape[1])] + [X ** 3])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
