import json, os
import numpy as np
import pandas as pd
from common import design, labels

df = pd.read_csv("data/train.csv")
rng = np.random.default_rng(2)
idx = rng.permutation(len(df)); nv = len(df) // 5
va, tr = df.iloc[idx[:nv]], df.iloc[idx[nv:]]
vocab = sorted(set(tr.region))
Xtr, mu, sd = design(tr, vocab)
Xva, _, _ = design(va, vocab, mu, sd)
ytr, yva = labels(tr), labels(va)
w = np.zeros(Xtr.shape[1])
for _ in range(500):
    g = 1 / (1 + np.exp(-(Xtr @ w))) - ytr
    w -= 0.3 * Xtr.T @ g / len(ytr)
os.makedirs("runs/v1", exist_ok=True)
np.savez("runs/v1/model.npz", w=w, mu=mu, sd=sd)
acc_va = float(np.mean(((Xva @ w) > 0) == (yva == 1)))
json.dump({"train_acc": float(np.mean(((Xtr @ w) > 0) == (ytr == 1))), "val_acc": acc_va, "vocab": vocab},
          open("runs/v1/train_summary.json", "w"), indent=1)
print("val_acc=%.4f" % acc_va)
