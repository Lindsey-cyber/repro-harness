import argparse, json, os
import numpy as np
import pandas as pd


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--out", default="results/metrics.json")
    args = ap.parse_args()
    cfg = json.load(open(args.config))

    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    cols = [c for c in tr.columns if c != "fault"]
    mu, sd = tr[cols].values.mean(0), tr[cols].values.std(0)
    Xtr = (tr[cols].values - mu) / sd
    Xte = (te[cols].values - mu) / sd
    ytr, yte = tr["fault"].values, te["fault"].values

    w = np.zeros(Xtr.shape[1])
    b = 0.0
    for _ in range(cfg["epochs"]):
        p = sigmoid(Xtr @ w + b)
        g = Xtr.T @ (p - ytr) / len(ytr) + cfg["l2"] * w
        w -= cfg["lr"] * g
        b -= cfg["lr"] * float(np.mean(p - ytr))

    acc_tr = float(np.mean((sigmoid(Xtr @ w + b) > 0.5) == ytr))
    acc_te = float(np.mean((sigmoid(Xte @ w + b) > 0.5) == yte))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"train_accuracy": acc_tr, "test_accuracy": acc_te}, open(args.out, "w"), indent=2)
    print("test accuracy %.4f" % acc_te)


if __name__ == "__main__":
    main()
