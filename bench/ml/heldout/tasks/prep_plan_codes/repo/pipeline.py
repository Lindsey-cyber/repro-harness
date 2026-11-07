import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def featurize(df, cfg):
    blocks = []
    for col in cfg["categorical"]:
        codes, uniques = pd.factorize(df[col])
        blocks.append(np.eye(len(uniques))[codes])
    blocks.append(df[cfg["numeric"]].to_numpy(dtype=float))
    return np.hstack(blocks)


def train_logreg(X, y, lr, epochs):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * X.T @ (p - y) / len(y)
        b -= lr * (p - y).mean()
    return w, b


def main(path):
    cfg = json.loads(Path(path).read_text())
    train = pd.read_csv(cfg["train_path"])
    test = pd.read_csv(cfg["test_path"])

    Xtr = featurize(train, cfg)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (featurize(test, cfg) - mu) / sd
    ytr, yte = train[cfg["label"]].to_numpy(), test[cfg["label"]].to_numpy()

    w, b = train_logreg(Xtr, ytr, cfg["lr"], cfg["epochs"])
    acc = float((((Xte @ w + b) > 0).astype(int) == yte).mean())
    res = {"accuracy": round(acc, 4), "n_train": len(train), "n_test": len(test)}
    out = Path(cfg["out_dir"])
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(res)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.json")
