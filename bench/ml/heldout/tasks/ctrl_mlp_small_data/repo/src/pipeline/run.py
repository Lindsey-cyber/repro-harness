import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlp import accuracy, init_params, sgd_step

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="data/train.csv")
ap.add_argument("--test", default="data/test.csv")
ap.add_argument("--out", default="runs/baseline")
ap.add_argument("--epochs", type=int, default=600)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--lr", type=float, default=0.3)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--seed", type=int, default=3)
args = ap.parse_args()

df = pd.read_csv(args.train)
te = pd.read_csv(args.test)
feats = [c for c in df.columns if c != "y"]
n_val = len(df) // 5
va, tr = df.iloc[-n_val:], df.iloc[:-n_val]
mu, sd = tr[feats].mean().to_numpy(), tr[feats].std().to_numpy()


def prep(d):
    return (d[feats].to_numpy() - mu) / sd, d["y"].to_numpy()


Xtr, ytr = prep(tr)
Xva, yva = prep(va)
Xte, yte = prep(te)

params = init_params(Xtr.shape[1], args.hidden, args.seed)
best = (-1.0, -1, None)
for ep in range(1, args.epochs + 1):
    params = sgd_step(params, Xtr, ytr, args.lr, args.wd)
    v = accuracy(params, Xva, yva)
    if v > best[0]:
        best = (v, ep, {k: a.copy() for k, a in params.items()})

val_acc, best_ep, best_params = best
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
np.savez(out / "model.npz", mu=mu, sd=sd, **best_params)
metrics = {
    "best_epoch": best_ep,
    "train_accuracy": round(accuracy(best_params, Xtr, ytr), 4),
    "val_accuracy": round(val_acc, 4),
    "test_accuracy": round(accuracy(best_params, Xte, yte), 4),
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(metrics)
