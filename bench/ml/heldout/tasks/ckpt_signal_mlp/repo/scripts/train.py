import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/signals.csv")
ap.add_argument("--epochs", type=int, default=400)
ap.add_argument("--hidden", type=int, default=96)
ap.add_argument("--lr", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ckpt-dir", default="checkpoints")
ap.add_argument("--log-dir", default="logs")
args = ap.parse_args()

df = pd.read_csv(args.data)
feats = [c for c in df.columns if c not in ("target", "split")]
part = {s: df[df["split"] == s] for s in ("train", "val", "test")}
mu = part["train"][feats].mean().to_numpy()
sd = part["train"][feats].std().to_numpy()
X = {s: (d[feats].to_numpy() - mu) / sd for s, d in part.items()}
y = {s: d["target"].to_numpy() for s, d in part.items()}

r = np.random.default_rng(args.seed)
d_in = len(feats)
params = {
    "W1": r.normal(0, 1 / np.sqrt(d_in), (d_in, args.hidden)),
    "b1": np.zeros(args.hidden),
    "W2": r.normal(0, 1 / np.sqrt(args.hidden), (args.hidden, 1)),
    "b2": np.zeros(1),
}


def logits(p, Xs):
    H = np.tanh(Xs @ p["W1"] + p["b1"])
    return H, (H @ p["W2"] + p["b2"]).ravel()


def acc(p, split):
    return float(((logits(p, X[split])[1] > 0).astype(int) == y[split]).mean())


def step(p, lr):
    H, z = logits(p, X["train"])
    g = (1 / (1 + np.exp(-z)) - y["train"]) / len(z)
    dH = (g[:, None] @ p["W2"].T) * (1 - H ** 2)
    p["W2"] -= lr * (H.T @ g[:, None])
    p["b2"] -= lr * g.sum()
    p["W1"] -= lr * (X["train"].T @ dH)
    p["b1"] -= lr * dH.sum(0)


log = []
best = {"val": -1.0, "epoch": 0, "test": None, "params": None}
for ep in range(1, args.epochs + 1):
    step(params, args.lr)
    v, t = acc(params, "val"), acc(params, "test")
    log.append((ep, acc(params, "train"), v, t))
    if v > best["val"]:
        best.update(val=v, epoch=ep, test=t, params={k: a.copy() for k, a in params.items()})

ck = Path(args.ckpt_dir)
ck.mkdir(parents=True, exist_ok=True)
np.savez(ck / "best.npz", **best["params"])
lg = Path(args.log_dir)
lg.mkdir(parents=True, exist_ok=True)
pd.DataFrame(log, columns=["epoch", "train_acc", "val_acc", "test_acc"]).to_csv(lg / "train_log.csv", index=False)
final = {"best_epoch": best["epoch"], "val_acc": round(best["val"], 4), "test_acc": round(best["test"], 4)}
(lg / "final.json").write_text(json.dumps(final, indent=2))
print(final)
