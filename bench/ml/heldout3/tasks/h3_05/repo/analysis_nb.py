#!/usr/bin/env python
# coding: utf-8

# # Readmission risk: site A -> site B
# Exported from `analysis.ipynb` (jupyter nbconvert --to script)

# In[1]:


import json, os
import numpy as np
import pandas as pd

cfg = json.load(open("config.json"))
rng = np.random.default_rng(cfg["seed"])


# In[2]:


a = pd.read_csv("data/site_a.csv")
b = pd.read_csv("data/site_b.csv")
FEATS = ["age"] + [c for c in a.columns if c.startswith("lab_")]
def xy(d):
    return d[FEATS].to_numpy(float), (d.readmit_30d == "high").to_numpy(float)


# In[3]:


perm = rng.permutation(len(a))
n_val = len(a) // 4
va_idx, tr_idx = perm[:n_val], perm[n_val:]
Xa, ya = xy(a)
Xtr, ytr, Xva, yva = Xa[tr_idx], ya[tr_idx], Xa[va_idx], ya[va_idx]
Xb, yb = xy(b)
mu, sd = Xtr.mean(0), Xtr.std(0)
Xtr, Xva, Xb = [(m - mu) / sd for m in (Xtr, Xva, Xb)]


# In[4]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

w = np.zeros(Xtr.shape[1]); bias = 0.0
for _ in range(cfg["epochs"]):
    g = sigmoid(Xtr @ w + bias) - ytr
    w -= cfg["lr"] * (Xtr.T @ g / len(ytr) + cfg["l2"] * w)
    bias -= cfg["lr"] * g.mean()


# In[5]:


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


def score(X, y):
    s = X @ w + bias
    return float(np.mean((s > 0) == (y == 1))), roc_auc(y, s)

va_acc, va_auc = score(Xva, yva)
b_acc, b_auc = score(Xb, yb)
res = {"val_acc": round(va_acc, 4), "val_auc": round(va_auc, 4), "siteB_acc": round(b_acc, 4), "siteB_auc": round(b_auc, 4),
       "n_train": len(tr_idx), "n_val": n_val, "n_siteB": len(b)}
os.makedirs("out/nb_run", exist_ok=True)
json.dump(res, open("out/nb_run/metrics.json", "w"), indent=1)
print(res)
