import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/tickets.csv")
ap.add_argument("--out", default="out")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

import os
os.makedirs(a.out, exist_ok=True)
df = pd.read_csv(a.data)
labels = ["refund", "login", "abuse"]
FE = [c for c in df.columns if c.endswith("_z")]
rng = np.random.default_rng(a.seed)
idx = rng.permutation(len(df)); cut = int(0.7 * len(df))
tr, te = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
X = tr[FE].to_numpy(float); y = tr.category.map({c: i for i, c in enumerate(labels)}).to_numpy()
mu, sd = X.mean(0), X.std(0)
X = (X - mu) / sd
W = np.zeros((X.shape[1], 3)); b = np.zeros(3); Y = np.eye(3)[y]
for _ in range(500):
    Z = X @ W + b; Z -= Z.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
    G = P - Y
    W -= 0.2 * X.T @ G / len(y); b -= 0.2 * G.mean(0)
Xt = (te[FE].to_numpy(float) - mu) / sd
pred = (Xt @ W + b).argmax(1)
pd.DataFrame({"ticket_id": te.ticket_id.values, "label": te.category.values,
              "pred": np.array(labels)[pred]}).to_csv(os.path.join(a.out, "preds.csv"), index=False)
print("wrote", len(te), "predictions")
