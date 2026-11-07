import argparse, json, os
import numpy as np
import pandas as pd


def design(df):
    placements = ["feed", "story", "search", "banner"]
    cols = [df["bid"].values, np.sin(df["hour"].values / 24 * 2 * np.pi), df["campaign_rate"].values]
    cols += [(df["placement"] == p).astype(float).values for p in placements]
    return np.column_stack(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="reports/run.json")
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["campaign_rate"] = df.groupby("campaign")["clicked"].transform("mean")

    idx = np.random.RandomState(args.seed).permutation(len(df))
    n_te = len(df) // 4
    te, tr = df.iloc[idx[:n_te]], df.iloc[idx[n_te:]]

    Xtr, Xte = design(tr), design(te)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ytr, yte = tr["clicked"].values, te["clicked"].values

    w, b = np.zeros(Xtr.shape[1]), 0.0
    for _ in range(800):
        p = 1 / (1 + np.exp(-(Xtr @ w + b)))
        w -= 0.3 * (Xtr.T @ (p - ytr) / len(ytr) + 1e-3 * w)
        b -= 0.3 * np.mean(p - ytr)
    acc = float(np.mean(((Xte @ w + b) > 0) == yte))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"test_accuracy": acc, "n_train": len(tr), "n_test": len(te)}, open(args.out, "w"))
    print("test accuracy %.4f" % acc)


if __name__ == "__main__":
    main()
