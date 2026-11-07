import json
from pathlib import Path

import numpy as np
import pandas as pd

feats = pd.read_csv("data/raw/features.csv")
labs = pd.read_csv("data/raw/labels.csv")

label_map = dict(zip(labs["sample_id"], labs["diagnosis"]))
feats["diagnosis"] = feats["id"].map(label_map).fillna(0).astype(int)

cols = [c for c in feats.columns if c.startswith("m")]
rng = np.random.default_rng(21)
idx = rng.permutation(len(feats))
cut = int(0.75 * len(feats))
tr, te = feats.iloc[idx[:cut]], feats.iloc[idx[cut:]]
mu, sd = tr[cols].mean(), tr[cols].std()
Xtr, Xte = ((tr[cols] - mu) / sd).to_numpy(), ((te[cols] - mu) / sd).to_numpy()
ytr, yte = tr["diagnosis"].to_numpy(), te["diagnosis"].to_numpy()

w, b = np.zeros(len(cols)), 0.0
for _ in range(400):
    p = 1 / (1 + np.exp(-(Xtr @ w + b)))
    w -= 0.2 * Xtr.T @ (p - ytr) / len(ytr)
    b -= 0.2 * (p - ytr).mean()

acc = float((((Xte @ w + b) > 0).astype(int) == yte).mean())
Path("reports").mkdir(exist_ok=True)
res = {"accuracy": round(acc, 4), "n_train": len(tr), "n_test": len(te)}
Path("reports/summary.json").write_text(json.dumps(res, indent=2))
print(res)
