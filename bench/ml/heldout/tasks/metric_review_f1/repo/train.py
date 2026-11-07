import json
from pathlib import Path

import numpy as np
import pandas as pd

P = json.loads(Path("params.json").read_text())
df = pd.read_csv("data/reviews.csv")
feats = [c for c in df.columns if c != "positive"]

rng = np.random.default_rng(P["seed"])
idx = rng.permutation(len(df))
cut = int(len(df) * (1 - P["test_frac"]))
tr, te = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
mu, sd = tr[feats].mean(), tr[feats].std()
Xtr, Xte = ((tr[feats] - mu) / sd).to_numpy(), ((te[feats] - mu) / sd).to_numpy()
ytr, yte = tr["positive"].to_numpy(), te["positive"].to_numpy()


def decision(X, w, b):
    return X @ w + b


w, b = np.zeros(len(feats)), 0.0
for _ in range(P["epochs"]):
    p = 1 / (1 + np.exp(-decision(Xtr, w, b)))
    w -= P["lr"] * (Xtr.T @ (p - ytr) / len(ytr) + P["l2"] * w)
    b -= P["lr"] * (p - ytr).mean()


def f1(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


scores = decision(Xte, w, b)
pred = (scores > P["threshold"]).astype(int)

out = Path("report")
out.mkdir(exist_ok=True)
pd.DataFrame({"score": scores, "pred": pred, "label": yte}).to_csv(out / "predictions.csv", index=False)
m = {"f1": round(f1(yte, pred), 4), "n_test": len(yte), "positive_rate_test": round(float(yte.mean()), 4)}
(out / "metrics.json").write_text(json.dumps(m, indent=2))
print(m)
