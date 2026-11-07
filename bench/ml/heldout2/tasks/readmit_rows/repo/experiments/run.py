import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import load, train_test_split, FEATURES


def auc(y, s):
    r = pd.Series(s).rank().values
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def knn_scores(Xtr, ytr, Xte, k):
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    nn = np.argpartition(d, k, axis=1)[:, :k]
    return ytr[nn].mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))
    df = load(cfg["data"])
    tr, te = train_test_split(df, cfg["split"]["test_frac"], cfg["split"]["seed"])
    mu, sd = tr[FEATURES].values.mean(0), tr[FEATURES].values.std(0)
    Xtr, Xte = (tr[FEATURES].values - mu) / sd, (te[FEATURES].values - mu) / sd
    s = knn_scores(Xtr, tr["readmitted"].values, Xte, cfg["k"])
    a = auc(te["readmitted"].values, s)
    os.makedirs("results", exist_ok=True)
    json.dump({"auc": a, "n_test": len(te)}, open("results/%s.json" % cfg["name"], "w"))
    print("auc %.4f" % a)


if __name__ == "__main__":
    main()
