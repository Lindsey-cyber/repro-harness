import json, os
import numpy as np
import pandas as pd


def roc_auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    order = np.argsort(s, kind="mergesort"); ss = s[order]
    r = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load(p):
    d = pd.read_csv(p)
    return d[["u", "v"]].to_numpy(float), (d.cls == "A").to_numpy(int) * 2 - 1


Xtr, ytr = load("data/train.csv")
Xte, yte = load("data/test.csv")
w = np.zeros(2); b = 0.0
for epoch in range(1000):
    errs = 0
    for x, t in zip(Xtr, ytr):
        if t * (x @ w + b) <= 0:
            w += t * x; b += t; errs += 1
    if errs == 0:
        break
s = Xte @ w + b
res = {"accuracy": float(np.mean(np.sign(s) == yte)), "roc_auc": roc_auc((yte == 1).astype(int), s),
       "epochs_to_converge": epoch + 1, "n_test": len(yte)}
os.makedirs("results/perceptron", exist_ok=True)
json.dump(res, open("results/perceptron/metrics.json", "w"), indent=1)
print(res)
