import os
import numpy as np
import pandas as pd
from churnlib.features import encode


def main():
    df = pd.read_csv("data/train.csv")
    X, (mu, sd) = encode(df)
    y = df["churned"].values
    n = int(len(df) * 0.8)
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(500):
        p = 1 / (1 + np.exp(-(X[:n] @ w + b)))
        w -= 0.3 * X[:n].T @ (p - y[:n]) / n
        b -= 0.3 * np.mean(p - y[:n])
    hold = float(np.mean(((X[n:] @ w + b) > 0) == y[n:]))
    os.makedirs("model", exist_ok=True)
    np.savez("model/model.npz", w=w, b=b, mu=mu, sd=sd)
    print("holdout accuracy %.4f" % hold)


if __name__ == "__main__":
    main()
