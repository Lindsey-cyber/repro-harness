import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

NUM = ["amount", "hour", "n_prev_claims", "distance_km"]
TARGET = "is_fraud"


def auc(y, s):
    r = pd.Series(s).rank().to_numpy()
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/claims.csv")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--outdir", default="artifacts")
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    df["amount"] = np.log1p(df["amount"])

    rate = df.groupby("merchant")[TARGET].mean()
    df["merchant_rate"] = df["merchant"].map(rate)

    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(df))
    cut = int(0.7 * len(df))
    tr, te = df.iloc[idx[:cut]], df.iloc[idx[cut:]]

    cols = NUM + ["merchant_rate"]
    mu, sd = tr[cols].mean(), tr[cols].std()
    Xtr = ((tr[cols] - mu) / sd).to_numpy()
    Xte = ((te[cols] - mu) / sd).to_numpy()
    ytr, yte = tr[TARGET].to_numpy(), te[TARGET].to_numpy()

    w, b = np.zeros(Xtr.shape[1]), 0.0
    for _ in range(600):
        p = 1 / (1 + np.exp(-(Xtr @ w + b)))
        w -= 0.1 * (Xtr.T @ (p - ytr) / len(ytr))
        b -= 0.1 * (p - ytr).mean()
    s = Xte @ w + b

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"claim_id": te["claim_id"].to_numpy(), "score": s, "label": yte}).to_csv(
        out / "test_scores.csv", index=False)
    m = {"test_auc": round(auc(yte, s), 4), "n_test": int(len(te))}
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    print(m)


if __name__ == "__main__":
    main()
