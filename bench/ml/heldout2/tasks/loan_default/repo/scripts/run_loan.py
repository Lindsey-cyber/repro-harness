import argparse, json, os
import numpy as np
import pandas as pd

NUM = ["income", "debt_ratio", "prior_defaults"]
CAT = ["employment", "purpose"]


def build(df, vocab, stats):
    cols = [(df[NUM].values - stats[0]) / stats[1]]
    for c in CAT:
        cols.append(np.column_stack([(df[c] == v).astype(float) for v in vocab[c]]))
    return np.hstack(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/applications.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    idx = np.random.RandomState(a.seed).permutation(len(df))
    n_te = int(len(df) * a.test_frac)
    te, tr = df.iloc[idx[:n_te]], df.iloc[idx[n_te:]]

    stats = (tr[NUM].values.mean(0), tr[NUM].values.std(0))
    vocab = {c: sorted(tr[c].unique()) for c in CAT}
    Xtr, Xte = build(tr, vocab, stats), build(te, vocab, stats)
    ytr, yte = tr["defaulted"].values, te["defaulted"].values

    w, b = np.zeros(Xtr.shape[1]), 0.0
    for _ in range(600):
        p = 1 / (1 + np.exp(-(Xtr @ w + b)))
        w -= 0.2 * (Xtr.T @ (p - ytr) / len(ytr) + 1e-3 * w)
        b -= 0.2 * np.mean(p - ytr)
    prob = 1 / (1 + np.exp(-(Xte @ w + b)))
    pred = (prob > 0.5).astype(int)
    acc = float(np.mean(pred == yte))

    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame({"app_id": te["app_id"].values, "prob": prob, "pred": pred}).to_csv(
        os.path.join(a.out, "predictions.csv"), index=False, float_format="%.5f")
    json.dump({"accuracy": acc, "n_test": int(len(te))}, open(os.path.join(a.out, "metrics.json"), "w"))
    print("accuracy %.4f" % acc)


if __name__ == "__main__":
    main()
