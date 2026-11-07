import json
from pathlib import Path

import numpy as np
import pandas as pd

cfg = json.loads(Path("config.json").read_text())
df = pd.read_csv(cfg["data"])
feats = [c for c in df.columns if c != cfg["label"]]
rng = np.random.default_rng(cfg["seed"])
idx = rng.permutation(len(df))
cut = int(len(df) * (1 - cfg["test_frac"]))
tr, te = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
mu, sd = tr[feats].mean(), tr[feats].std()
Xtr, Xte = ((tr[feats] - mu) / sd).to_numpy(), ((te[feats] - mu) / sd).to_numpy()
ytr, yte = tr[cfg["label"]].to_numpy(), te[cfg["label"]].to_numpy()


def sample_weights(y, mode):
    if mode != "balanced":
        return np.ones(len(y))
    counts = np.bincount(y, minlength=2)
    cw = len(y) / (2.0 * counts)
    return cw[y]


sw = sample_weights(ytr, cfg.get("class_weight"))
print("effective training size:", round(float(sw.sum()), 1))

w, b = np.zeros(len(feats)), 0.0
for _ in range(cfg["epochs"]):
    p = 1 / (1 + np.exp(-(Xtr @ w + b)))
    err = p - ytr
    w -= cfg["lr"] * Xtr.T @ err / len(ytr)
    b -= cfg["lr"] * err.mean()

pred = ((Xte @ w + b) > 0).astype(int)


recall_fraud = float(pred[yte == 1].mean())
recall_legit = float(1 - pred[yte == 0].mean())

rep = {
    "accuracy": round(float((pred == yte).mean()), 4),
    "balanced_accuracy": round((recall_fraud + recall_legit) / 2, 4),
    "recall_fraud": round(recall_fraud, 4),
    "fraud_rate_test": round(float(yte.mean()), 4),
}
Path(cfg["report"]).write_text(json.dumps(rep, indent=2))
print(rep)
