import argparse, json, os
import numpy as np
import pandas as pd


def logloss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/diag.csv")
ap.add_argument("--l2", type=float, default=0.002)
ap.add_argument("--lr", type=float, default=0.5)
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--seed", type=int, default=4)
ap.add_argument("--out", default="results")
a = ap.parse_args()

df = pd.read_csv(a.data)
FE = [c for c in df.columns if c.startswith("m")]
y = (df.status == "positive").to_numpy(float)
idx = np.random.default_rng(a.seed).permutation(len(df))
cut = int(0.7 * len(df))
tr, te = idx[:cut], idx[cut:]
X = df[FE].to_numpy(float)
mu, sd = X[tr].mean(0), X[tr].std(0)
X = np.c_[(X - mu) / sd, np.ones(len(X))]

w = np.zeros(X.shape[1]); n = len(tr)
for _ in range(a.epochs):
    p = 1 / (1 + np.exp(-X[tr] @ w))
    g = X[tr].T @ (p - y[tr]) / n + a.l2 * n * w
    w -= a.lr * g
pt = 1 / (1 + np.exp(-X[te] @ w))
res = {"accuracy": round(float(np.mean((pt > 0.5) == (y[te] == 1))), 4), "logloss": round(logloss(y[te], pt), 4),
       "n_test": len(te)}
d = os.path.join(a.out, "l2_%g" % a.l2)
os.makedirs(d, exist_ok=True)
json.dump(res, open(os.path.join(d, "metrics.json"), "w"), indent=1)
print(res)
