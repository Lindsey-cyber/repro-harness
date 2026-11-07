import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/accounts.csv")
p.add_argument("--k", type=int, default=5)
p.add_argument("--seed", type=int, default=7)
p.add_argument("--test-frac", type=float, default=0.25)
p.add_argument("--out", default="results/knn_metrics.json")
args = p.parse_args()

df = pd.read_csv(args.data)
feature_cols = [c for c in df.columns if c not in ("customer_id", "month", "churned")]

rng = np.random.default_rng(args.seed)
idx = rng.permutation(len(df))
n_test = int(len(df) * args.test_frac)
test_idx, train_idx = idx[:n_test], idx[n_test:]
train, test = df.iloc[train_idx], df.iloc[test_idx]

mu, sd = train[feature_cols].mean(), train[feature_cols].std()
Xtr = ((train[feature_cols] - mu) / sd).to_numpy()
Xte = ((test[feature_cols] - mu) / sd).to_numpy()
ytr, yte = train["churned"].to_numpy(), test["churned"].to_numpy()

d2 = (Xte ** 2).sum(1)[:, None] - 2 * Xte @ Xtr.T + (Xtr ** 2).sum(1)[None, :]
nn = np.argpartition(d2, args.k, axis=1)[:, : args.k]
pred = (ytr[nn].mean(1) >= 0.5).astype(int)

res = {"k": args.k, "test_accuracy": round(float((pred == yte).mean()), 4),
       "n_train": int(len(train)), "n_test": int(len(test)),
       "majority_baseline": round(float(max(yte.mean(), 1 - yte.mean())), 4)}
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, indent=2))
print(res)
