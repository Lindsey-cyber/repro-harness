import argparse, configparser, json, os
import numpy as np
import pandas as pd


def load(d, name):
    df = pd.read_csv(os.path.join(d, name + ".csv"))
    return df.drop(columns="y").values, df["y"].values


def acc(X, y, w, b):
    return float(np.mean(((X @ w + b) > 0) == y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run.cfg")
    cp = configparser.ConfigParser()
    cp.read(ap.parse_args().config)
    lr, epochs, seed = cp.getfloat("train", "lr"), cp.getint("train", "epochs"), cp.getint("train", "seed")
    d = cp.get("paths", "data_dir")
    Xtr, ytr = load(d, "train")
    Xva, yva = load(d, "val")
    Xte, yte = load(d, "test")
    mu, sd = Xtr.mean(0), Xtr.std(0)
    Xtr, Xva, Xte = [(X - mu) / sd for X in (Xtr, Xva, Xte)]

    rs = np.random.RandomState(seed)
    w = rs.randn(Xtr.shape[1]) * 0.01
    b = 0.0
    best = {"val": -1.0}
    for ep in range(epochs):
        p = 1 / (1 + np.exp(-(Xtr @ w + b)))
        w -= lr * Xtr.T @ (p - ytr) / len(ytr)
        b -= lr * float(np.mean(p - ytr))
        v = acc(Xva, yva, w, b)
        if v > best["val"]:
            best = {"val": v, "epoch": ep, "test": acc(Xte, yte, w, b), "w": w, "b": b}

    os.makedirs(os.path.dirname(cp.get("paths", "ckpt")), exist_ok=True)
    np.savez(cp.get("paths", "ckpt"), w=best["w"], b=best["b"], mu=mu, sd=sd)
    os.makedirs(os.path.dirname(cp.get("paths", "metrics")), exist_ok=True)
    json.dump({"best_epoch": best["epoch"], "val_acc": best["val"], "test_acc": best["test"]}, open(cp.get("paths", "metrics"), "w"))
    print("best epoch %d val %.4f test %.4f" % (best["epoch"], best["val"], best["test"]))


if __name__ == "__main__":
    main()
