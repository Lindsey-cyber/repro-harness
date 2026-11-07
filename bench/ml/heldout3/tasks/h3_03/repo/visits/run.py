import argparse, configparser, json, os
import numpy as np
import pandas as pd
from .split import subject_split, canon
from .knn import predict

FEATS = ["f%d" % i for i in range(6)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ini", default="params.ini")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    cp = configparser.ConfigParser()
    cp.read(a.ini)
    df = pd.read_csv(cp["data"]["path"])
    tr, te = subject_split(df, cp.getint("split", "buckets"), cp.getint("split", "test_bucket"))
    k = cp.getint("model", "k")
    mu, sd = tr[FEATS].mean().to_numpy(), tr[FEATS].std().to_numpy()
    Xtr = (tr[FEATS].to_numpy() - mu) / sd; Xte = (te[FEATS].to_numpy() - mu) / sd
    ytr = (tr.outcome == "improved").to_numpy(int); yte = (te.outcome == "improved").to_numpy(int)
    p = predict(Xtr, ytr, Xte, k)
    d = os.path.join(a.out, "knn_k%d" % k)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame({"subject": te.subject.map(canon), "visit_date": te.visit_date, "y_true": yte, "y_pred": p}).to_csv(
        os.path.join(d, "preds.csv"), index=False)
    res = {"accuracy": round(float((p == yte).mean()), 4), "n_train": len(tr), "n_test": len(te)}
    json.dump(res, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    print(res)


if __name__ == "__main__":
    main()
