import argparse
import json
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def roc_auc(y, s):
    r = pd.Series(s).rank().to_numpy()
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def expand(X, poly):
    if not poly:
        return X
    extra = [X[:, i] * X[:, j] for i, j in combinations_with_replacement(range(4), 2)]
    return np.column_stack([X] + extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())

    df = pd.read_csv(ROOT / "data" / "tabular.csv")
    feats = [c for c in df.columns if c != "label"]
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.permutation(len(df))
    cut = int(0.7 * len(df))
    X = expand(df[feats].to_numpy(), cfg["poly"])
    y = df["label"].to_numpy()
    Xtr, Xte, ytr, yte = X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]
    mu, sd = Xtr.mean(0), Xtr.std(0)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    w, b = np.zeros(Xtr.shape[1]), 0.0
    for _ in range(cfg["epochs"]):
        p = 1 / (1 + np.exp(-(Xtr @ w + b)))
        w -= cfg["lr"] * (Xtr.T @ (p - ytr) / len(ytr) + cfg["l2"] * w)
        b -= cfg["lr"] * (p - ytr).mean()

    res = {"config": Path(a.config).name, "test_auc": round(roc_auc(yte, Xte @ w + b), 4)}
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(res)


if __name__ == "__main__":
    main()
