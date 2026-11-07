import argparse, json
import numpy as np
import pandas as pd


def per_class_f1(y, p, labels):
    out = {}
    for c in labels:
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        out[c] = 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0.0
    return out


ap = argparse.ArgumentParser()
ap.add_argument("preds")
ap.add_argument("--out", default="out/scores.json")
a = ap.parse_args()
d = pd.read_csv(a.preds)
y, p = d.label.to_numpy(), d.pred.to_numpy()
f1 = per_class_f1(y, p, ["refund", "login", "abuse"])
res = {"f1_macro": round(float(np.nanmean(list(f1.values()))), 4), "accuracy": round(float((y == p).mean()), 4),
       "f1_per_class": {k: (None if np.isnan(v) else round(float(v), 4)) for k, v in f1.items()}, "n": len(d)}
json.dump(res, open(a.out, "w"), indent=1)
print(res)
