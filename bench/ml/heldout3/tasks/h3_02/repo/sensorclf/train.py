import argparse, json, os
import numpy as np
from .data import load, split_units, xy
from .model import fit, predict, f1_macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/readings.csv")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=400)
    a = ap.parse_args()
    df = load(a.data)
    classes = ["ok", "warn", "fault"]
    tr, te = split_units(df, a.seed)
    Xtr, ytr = xy(tr, classes); Xte, yte = xy(te, classes)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    W, b = fit((Xtr - mu) / sd, ytr, len(classes), a.lr, a.epochs)
    p = predict(W, b, (Xte - mu) / sd)
    res = {"f1_macro": round(f1_macro(yte, p, len(classes)), 4), "accuracy": round(float(np.mean(p == yte)), 4),
           "n_train": len(tr), "n_test": len(te)}
    d = os.path.join(a.out, "softmax_lr%g" % a.lr, "seed%d" % a.seed)
    os.makedirs(d, exist_ok=True)
    json.dump(res, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    print(res)


if __name__ == "__main__":
    main()
