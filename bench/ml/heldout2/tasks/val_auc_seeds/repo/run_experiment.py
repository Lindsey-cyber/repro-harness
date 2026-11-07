import argparse, json, os
import numpy as np
import pandas as pd


def auc(y, s):
    r = pd.Series(s).rank().values
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit(X, y, cfg, rs):
    w = rs.randn(X.shape[1]) * cfg["init_scale"]
    b = 0.0
    for _ in range(cfg["epochs"]):
        order = rs.permutation(len(y))
        for i in range(0, len(y), cfg["batch"]):
            j = order[i:i + cfg["batch"]]
            p = 1 / (1 + np.exp(-(X[j] @ w + b)))
            w -= cfg["lr"] * (X[j].T @ (p - y[j]) / len(j) + cfg["l2"] * w)
            b -= cfg["lr"] * np.mean(p - y[j])
    return w, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/exp.json")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    d = np.load("data/features.npz")
    Xt, yt, Xv, yv = d["X_train"], d["y_train"], d["X_val"], d["y_val"]
    mu, sd = Xt.mean(0), Xt.std(0)
    Xt, Xv = (Xt - mu) / sd, (Xv - mu) / sd
    per = []
    for s in range(args.seeds):
        w, b = fit(Xt, yt, cfg, np.random.RandomState(s))
        per.append(auc(yv, Xv @ w + b))
    os.makedirs("out", exist_ok=True)
    rep = {"metric": "val_auc", "val_auc_mean": float(np.mean(per)), "val_auc_std": float(np.std(per)), "per_seed": per}
    json.dump(rep, open("out/val_report.json", "w"), indent=1)
    print("val_auc mean %.4f std %.4f" % (rep["val_auc_mean"], rep["val_auc_std"]))


if __name__ == "__main__":
    main()
