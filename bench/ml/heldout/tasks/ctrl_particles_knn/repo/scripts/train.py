import configparser
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def knn_scores(Xtr, ytr, Xq, k):
    d2 = (Xq ** 2).sum(1)[:, None] - 2 * Xq @ Xtr.T + (Xtr ** 2).sum(1)[None, :]
    nn = np.argpartition(d2, k, axis=1)[:, :k]
    return ytr[nn].mean(axis=1)


def auc(y, s):
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        m = s == v
        ranks[m] = ranks[m].mean()
    n1 = y.sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def main(cfg_path):
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    k = cfg.getint("model", "k")
    d = np.load(ROOT / cfg.get("data", "path"))
    mu, sd = d["X_train"].mean(0), d["X_train"].std(0)
    Xtr = (d["X_train"] - mu) / sd
    Xte = (d["X_test"] - mu) / sd
    s = knn_scores(Xtr, d["y_train"], Xte, k)
    pred = (s >= 0.5).astype(int)
    summary = {
        "k": k,
        "accuracy": round(float((pred == d["y_test"]).mean()), 4),
        "auc": round(auc(d["y_test"], s), 4),
        "n_test": int(len(pred)),
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "configs" / "knn.cfg")
