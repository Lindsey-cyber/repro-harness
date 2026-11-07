import argparse, json, os
import numpy as np
import pandas as pd
from .confutil import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", default="results/base")
    a = ap.parse_args()
    cfg = load(a.config, a.set)
    df = pd.read_csv(cfg["data"]["path"])
    FE = [c for c in df.columns if c[1:].isdigit()]
    X = df[FE].to_numpy(float); y = df["clicked"].to_numpy(float)
    idx = np.random.default_rng(cfg["data"]["seed"]).permutation(len(df))
    cut = int(len(df) * 0.75)
    tr, te = idx[:cut], idx[cut:]
    mu, sd = X[tr].mean(0), X[tr].std(0)
    X = (X - mu) / sd
    w = np.zeros(X.shape[1]); b = 0.0
    lr = cfg["optim"]["lr"]
    for _ in range(cfg["optim"]["epochs"]):
        g = 1 / (1 + np.exp(-(X[tr] @ w + b))) - y[tr]
        w -= lr * X[tr].T @ g / len(tr); b -= lr * g.mean()
    p = 1 / (1 + np.exp(-(X[te] @ w + b)))
    ll = -np.mean(y[te] * np.log(p + 1e-12) + (1 - y[te]) * np.log(1 - p + 1e-12))
    res = {"test_acc": round(float(np.mean((p > 0.5) == (y[te] == 1))), 4), "test_logloss": round(float(ll), 4), "lr": lr}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    print(res)


if __name__ == "__main__":
    main()
