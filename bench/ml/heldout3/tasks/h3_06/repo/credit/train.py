import argparse, json, os
import numpy as np
import pandas as pd
from .features import fit_te, apply_te


def load_cfg(path):
    cfg = {}
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line or ":" not in line:
            continue
        k, v = [s.strip() for s in line.split(":", 1)]
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        cfg[k] = v
    return cfg


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lr.yaml")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    df = pd.read_csv(cfg["data"])
    y = (df["default"] == "yes").to_numpy(float)
    idx = np.random.default_rng(cfg["seed"]).permutation(len(df))
    nt = int(len(df) * cfg["test_frac"])
    te_i, tr_i = idx[:nt], idx[nt:]
    br = df.branch.to_numpy()
    table, prior = fit_te(br, y, cfg["smooth"])
    X = np.c_[df.income_k.to_numpy(float), df.tenure_y.to_numpy(float), apply_te(br, table, prior)]
    mu, sd = X[tr_i].mean(0), X[tr_i].std(0)
    X = (X - mu) / sd
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(cfg["epochs"]):
        g = 1 / (1 + np.exp(-(X[tr_i] @ w + b))) - y[tr_i]
        w -= cfg["lr"] * X[tr_i].T @ g / len(tr_i); b -= cfg["lr"] * g.mean()
    s = X[te_i] @ w + b
    res = {"auc": round(roc_auc(y[te_i], s), 4), "accuracy": round(float(np.mean((s > 0) == (y[te_i] == 1))), 4),
           "n_test": int(nt)}
    d = os.path.join("out", cfg["name"])
    os.makedirs(d, exist_ok=True)
    json.dump(res, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    print(res)


if __name__ == "__main__":
    main()
