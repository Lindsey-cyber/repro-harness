# ---
# jupyter:
#   jupytext: {text_representation: {extension: .py, format_name: percent}}
# ---

# %% [markdown]
# # Tumour subtype from a 500-gene expression panel
# 300 samples, 500 genes. Quick baseline before the larger cohort arrives.

# %%
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out")
ap.add_argument("--k", type=int, default=20)
args = ap.parse_args()

d = np.load("data/expr.npz", allow_pickle=False)
X, y = d["X"], d["y"]

# %% split
rng = np.random.default_rng(0)
perm = rng.permutation(len(y))
te, tr = perm[: len(y) // 4], perm[len(y) // 4:]

# %% gene screen
def screen(X, y, k):
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-9)
    yc = (y - y.mean()) / y.std()
    r = np.abs(Xc.T @ yc) / len(y)
    return np.argsort(-r)[:k]

keep = screen(X, y, args.k)

# %% fit
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


Z = X[:, keep]
mu, sd = Z[tr].mean(0), Z[tr].std(0)
Z = (Z - mu) / sd
w = np.zeros(Z.shape[1]); b = 0.0
for _ in range(300):
    g = 1 / (1 + np.exp(-(Z[tr] @ w + b))) - y[tr]
    w -= 0.1 * (Z[tr].T @ g / len(tr) + 0.05 * w); b -= 0.1 * g.mean()
s = Z[te] @ w + b

# %% report
res = {"roc_auc": round(roc_auc(y[te], s), 4), "accuracy": round(float(np.mean((s > 0) == (y[te] == 1))), 4),
       "n_test": int(len(te)), "k_genes": args.k}
os.makedirs(os.path.join(args.out, "nb", "run1"), exist_ok=True)
json.dump(res, open(os.path.join(args.out, "nb", "run1", "metrics.json"), "w"), indent=1)
print(res)
