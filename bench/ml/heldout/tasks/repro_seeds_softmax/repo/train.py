import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=0.3)
args = ap.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

tr, te = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
feats = [c for c in tr.columns if c != "variety"]
mu, sd = tr[feats].mean(), tr[feats].std()
Xtr, Xte = ((tr[feats] - mu) / sd).to_numpy(), ((te[feats] - mu) / sd).to_numpy()
ytr, yte = tr["variety"].to_numpy(), te["variety"].to_numpy()
K = 3

W = np.random.randn(Xtr.shape[1], K) * 0.05
b = np.zeros(K)
shuffler = np.random.default_rng()
for _ in range(args.epochs):
    order = shuffler.permutation(len(Xtr))
    for i in range(0, len(order), args.batch):
        j = order[i:i + args.batch]
        z = Xtr[j] @ W + b
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        p[np.arange(len(j)), ytr[j]] -= 1
        W -= args.lr * Xtr[j].T @ p / len(j)
        b -= args.lr * p.mean(0)

acc = float(((Xte @ W + b).argmax(1) == yte).mean())
out = Path("runs") / f"seed{args.seed}"
out.mkdir(parents=True, exist_ok=True)
res = {"seed": args.seed, "test_accuracy": round(acc, 4)}
(out / "metrics.json").write_text(json.dumps(res))
print(res)
