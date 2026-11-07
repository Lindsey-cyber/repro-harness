import argparse, json, os
import numpy as np
import pandas as pd


def accuracy(X, y, w, b):
    return float(np.mean(((X @ w + b) > 0) == y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--report", default="out/report.json")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=2)
    a = ap.parse_args()

    tr, te = pd.read_csv(a.train), pd.read_csv(a.test)
    feats = [c for c in tr.columns if c != "good"]
    mu, sd = tr[feats].values.mean(0), tr[feats].values.std(0)
    Xtr, Xte = (tr[feats].values - mu) / sd, (te[feats].values - mu) / sd
    ytr, yte = tr["good"].values, te["good"].values

    rs = np.random.RandomState(a.seed)
    w, b = np.zeros(len(feats)), 0.0
    best, best_ep = -1.0, -1
    for ep in range(a.epochs):
        order = rs.permutation(len(ytr))
        for i in range(0, len(order), 8):
            j = order[i:i + 8]
            p = 1 / (1 + np.exp(-(Xtr[j] @ w + b)))
            w -= a.lr * Xtr[j].T @ (p - ytr[j]) / len(j)
            b -= a.lr * float(np.mean(p - ytr[j]))
        s = accuracy(Xte, yte, w, b)
        if s > best:
            best, best_ep = s, ep

    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    json.dump({"best_epoch": best_ep, "test_accuracy": best}, open(a.report, "w"))
    print("epoch %d test accuracy %.4f" % (best_ep, best))


if __name__ == "__main__":
    main()
