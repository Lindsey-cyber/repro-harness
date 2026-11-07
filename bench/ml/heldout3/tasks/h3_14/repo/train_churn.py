import argparse, json, os
import numpy as np
import pandas as pd


ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/events.csv")
ap.add_argument("--out", default="results/churn_v2")
ap.add_argument("--lr", type=float, default=0.3)
ap.add_argument("--epochs", type=int, default=300)
a = ap.parse_args()

df = pd.read_csv(a.data)
FE = ["period", "f1", "f2", "f3", "f4"]
X = df.sort_values("ts")[FE].to_numpy(float)   # chronological order
y = (df["outcome"] == "churn").to_numpy(float)
idx = np.random.default_rng(7).permutation(len(df))
cut = int(0.75 * len(df))
tr, te = idx[:cut], idx[cut:]
mu, sd = X[tr].mean(0), X[tr].std(0)
X = (X - mu) / sd
w = np.zeros(X.shape[1]); b = 0.0
for _ in range(a.epochs):
    g = 1 / (1 + np.exp(-(X[tr] @ w + b))) - y[tr]
    w -= a.lr * X[tr].T @ g / len(tr); b -= a.lr * g.mean()
s = X[te] @ w + b
res = {"accuracy": round(float(np.mean((s > 0) == (y[te] == 1))), 4), "n_test": int(len(te)), "coef": [round(float(c), 3) for c in w]}
os.makedirs(os.path.join(a.out, "metrics"), exist_ok=True)
json.dump(res, open(os.path.join(a.out, "metrics", "summary.json"), "w"), indent=1)
print(res)
