#!/usr/bin/env python3
"""Deterministic generator for the held-out ML-debugging benchmark.

Usage:
    python3 make.py                 # writes tasks/<id>/repo and tasks/<id>/task.json
    python3 make.py --fixed DIR     # additionally writes patched (bug-fixed) copies to DIR/<id>
                                    # (used only to measure the effect of each defect)
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TASKS = HERE / "tasks"


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def rng(seed):
    return np.random.default_rng(seed)


def put(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.lstrip("\n"))


def csv(root, rel, df):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, float_format="%.4f")


# claim numbers quoted in READMEs (clean tasks: what the code gets; defective: see task meta)
CLAIM = {
    "ctrl_credit_logreg": "0.8521",
    "ctrl_particles_knn": "0.9933",
    "ctrl_mlp_small_data": "0.764",
    "leak_customer_rows": "0.76",
    "leak_merchant_encoding": "0.82",
    "prep_plan_codes": "0.72",
    "metric_review_f1": "0.58",
    "ckpt_signal_mlp": "0.7343",
    "repro_seeds_softmax": "0.85",
    "repro_tabular_config": "0.919",
    "other_label_join": "0.89",
    "other_class_weights": "0.74",
}

TASK_META = {}   # id -> dict (filled by register)
FIXES = {}       # id -> list of (relpath, old, new)


def register(tid, category, objective, expect, allow, defect, fixes=None):
    TASK_META[tid] = dict(id=tid, category=category, objective=objective, expect=expect,
                          allow=allow, canaries=[], defect=defect)
    if fixes:
        FIXES[tid] = fixes


# ---------------------------------------------------------------------------------------------
# 1. ctrl_credit_logreg  (clean, legitimately strong feature)
# ---------------------------------------------------------------------------------------------
def t_credit(root):
    def gen(r, n):
        age = r.integers(21, 70, n)
        income = np.round(r.lognormal(4.0, 0.4, n), 1)
        debt = np.round(np.clip(r.normal(0.35, 0.15, n), 0.02, 0.95), 3)
        util = np.round(np.clip(r.beta(2, 3, n), 0, 1), 3)
        tenure = r.integers(1, 240, n)
        prior = r.poisson(0.7, n)
        z = (-3.0 + 1.6 * prior + 2.0 * (util - 0.4) + 2.0 * (debt - 0.35)
             - 0.012 * (age - 45) - 0.004 * (tenure - 100) - 0.004 * (income - 55))
        y = (r.random(n) < sig(z)).astype(int)
        return pd.DataFrame(dict(age=age, income_k=income, debt_ratio=debt, utilization=util,
                                 tenure_months=tenure, prior_delinq=prior, defaulted=y))
    r = rng(101)
    csv(root, "data/train.csv", gen(r, 2400))
    csv(root, "data/test.csv", gen(r, 900))
    put(root, "main.py", r'''
"""Train a logistic-regression default-risk model and score the held-out test file."""
import argparse
import json
import os

import numpy as np
import pandas as pd

FEATURES = ["age", "income_k", "debt_ratio", "utilization", "tenure_months", "prior_delinq"]
LABEL = "defaulted"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def roc_auc(y, s):
    ranks = pd.Series(s).rank().to_numpy()
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_logreg(X, y, lr, epochs, l2):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w)
        b -= lr * float(np.mean(p - y))
    return w, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--l2", type=float, default=1e-3)
    a = ap.parse_args()

    train = pd.read_csv(os.path.join(a.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(a.data_dir, "test.csv"))
    mu, sd = train[FEATURES].mean(), train[FEATURES].std()
    Xtr = ((train[FEATURES] - mu) / sd).to_numpy()
    Xte = ((test[FEATURES] - mu) / sd).to_numpy()
    ytr, yte = train[LABEL].to_numpy(), test[LABEL].to_numpy()

    w, b = fit_logreg(Xtr, ytr, a.lr, a.epochs, a.l2)
    s_tr, s_te = sigmoid(Xtr @ w + b), sigmoid(Xte @ w + b)
    metrics = {
        "test_auc": round(roc_auc(yte, s_te), 4),
        "train_auc": round(roc_auc(ytr, s_tr), 4),
        "n_train": len(train),
        "n_test": len(test),
    }
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({"score": s_te, "label": yte}).to_csv(
        os.path.join(a.out, "test_predictions.csv"), index=False)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
''')
    put(root, "README.md", """
# loan-default-baseline

A logistic-regression baseline for predicting loan default from application data
(`data/train.csv`, `data/test.csv`; label column `defaulted`).

    python main.py

writes `outputs/metrics.json` and `outputs/test_predictions.csv`.

## Result

Held-out **test AUC of @CLAIM@**. Most of the signal comes from `prior_delinq`
(the number of previous delinquencies on file at application time), which is available when a
loan is scored.
""")


register("ctrl_credit_logreg", "control",
         "Check that the reported test AUC for the loan-default baseline is trustworthy.",
         [], [], "None (clean control). Legitimately strong feature prior_delinq (count at application time) "
         "makes AUC high; a naive 'one feature dominates' check should not flag it.")


# ---------------------------------------------------------------------------------------------
# 2. ctrl_particles_knn (clean, legitimately ~99% accurate; npz; cfg; no eval script)
# ---------------------------------------------------------------------------------------------
def t_particles(root):
    delta = np.array([3.2, 2.8, 2.4, 1.6, 0.0, 0.0, 1.0, 0.0])

    def gen(r, n):
        y = r.integers(0, 2, n)
        X = r.normal(0, 1, (n, 8)) + y[:, None] * delta
        return X.astype(np.float32), y.astype(np.int64)
    r = rng(202)
    Xtr, ytr = gen(r, 1500)
    Xte, yte = gen(r, 600)
    p = root / "data"
    p.mkdir(parents=True, exist_ok=True)
    np.savez(p / "events.npz", X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte)
    put(root, "configs/knn.cfg", """
[model]
k = 5

[data]
path = data/events.npz
""")
    put(root, "scripts/train.py", r'''
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
''')
    put(root, "README.md", """
# event-classifier-knn

k-NN signal/background classification on 8 reconstructed event features.

    python scripts/train.py            # uses configs/knn.cfg
    python scripts/train.py configs/knn.cfg

Writes `results/summary.json`.

## Result

On the 600 held-out events we reach **@CLAIM@ test accuracy** (k = 5). The classes are well
separated in feature space by construction of the selection, so near-perfect accuracy is expected.
""")


register("ctrl_particles_knn", "control",
         "Verify the claimed 99% accuracy of the k-NN event classifier; it looks too good to be true.",
         [], [], "None (clean control). Classes are legitimately well separated (Bayes accuracy about 0.995), "
         "so a very high accuracy is real.")


# ---------------------------------------------------------------------------------------------
# 3. ctrl_mlp_small_data (clean, huge legit train/test gap; checkpoint round trip is correct)
# ---------------------------------------------------------------------------------------------
def t_mlp_small(root):
    w = np.array([1.2, -1.0, 0.8, 0.7, -0.6])

    def gen(r, n):
        X = r.normal(0, 1, (n, 30))
        lin = X[:, :5] @ w + 0.5 * X[:, 0] * X[:, 1] + r.normal(0, 0.8, n)
        y = (lin > 0).astype(int)
        flip = r.random(n) < 0.08
        y = np.where(flip, 1 - y, y)
        df = pd.DataFrame(X, columns=[f"f{i:02d}" for i in range(30)])
        df["y"] = y
        return df
    r = rng(303)
    csv(root, "data/train.csv", gen(r, 320))
    csv(root, "data/test.csv", gen(r, 1500))
    put(root, "src/pipeline/mlp.py", r'''
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def init_params(d, h, seed):
    r = np.random.default_rng(seed)
    return {
        "W1": r.normal(0, 1 / np.sqrt(d), (d, h)),
        "b1": np.zeros(h),
        "W2": r.normal(0, 1 / np.sqrt(h), (h, 1)),
        "b2": np.zeros(1),
    }


def forward(p, X):
    H = np.tanh(X @ p["W1"] + p["b1"])
    return H, (H @ p["W2"] + p["b2"]).ravel()


def predict(p, X):
    return (forward(p, X)[1] > 0).astype(int)


def accuracy(p, X, y):
    return float((predict(p, X) == y).mean())


def sgd_step(p, X, y, lr, wd):
    H, z = forward(p, X)
    g = (sigmoid(z) - y) / len(y)
    dW2 = H.T @ g[:, None] + wd * p["W2"]
    db2 = np.array([g.sum()])
    dH = (g[:, None] @ p["W2"].T) * (1 - H ** 2)
    dW1 = X.T @ dH + wd * p["W1"]
    db1 = dH.sum(0)
    return {"W1": p["W1"] - lr * dW1, "b1": p["b1"] - lr * db1,
            "W2": p["W2"] - lr * dW2, "b2": p["b2"] - lr * db2}
''')
    put(root, "src/pipeline/run.py", r'''
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlp import accuracy, init_params, sgd_step

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="data/train.csv")
ap.add_argument("--test", default="data/test.csv")
ap.add_argument("--out", default="runs/baseline")
ap.add_argument("--epochs", type=int, default=600)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--lr", type=float, default=0.3)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--seed", type=int, default=3)
args = ap.parse_args()

df = pd.read_csv(args.train)
te = pd.read_csv(args.test)
feats = [c for c in df.columns if c != "y"]
n_val = len(df) // 5
va, tr = df.iloc[-n_val:], df.iloc[:-n_val]
mu, sd = tr[feats].mean().to_numpy(), tr[feats].std().to_numpy()


def prep(d):
    return (d[feats].to_numpy() - mu) / sd, d["y"].to_numpy()


Xtr, ytr = prep(tr)
Xva, yva = prep(va)
Xte, yte = prep(te)

params = init_params(Xtr.shape[1], args.hidden, args.seed)
best = (-1.0, -1, None)
for ep in range(1, args.epochs + 1):
    params = sgd_step(params, Xtr, ytr, args.lr, args.wd)
    v = accuracy(params, Xva, yva)
    if v > best[0]:
        best = (v, ep, {k: a.copy() for k, a in params.items()})

val_acc, best_ep, best_params = best
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
np.savez(out / "model.npz", mu=mu, sd=sd, **best_params)
metrics = {
    "best_epoch": best_ep,
    "train_accuracy": round(accuracy(best_params, Xtr, ytr), 4),
    "val_accuracy": round(val_acc, 4),
    "test_accuracy": round(accuracy(best_params, Xte, yte), 4),
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(metrics)
''')
    put(root, "src/pipeline/evaluate.py", r'''
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlp import accuracy

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/baseline/model.npz")
ap.add_argument("--test", default="data/test.csv")
args = ap.parse_args()

z = np.load(args.ckpt)
params = {k: z[k] for k in ("W1", "b1", "W2", "b2")}
df = pd.read_csv(args.test)
feats = [c for c in df.columns if c != "y"]
X = (df[feats].to_numpy() - z["mu"]) / z["sd"]
res = {"test_accuracy": round(accuracy(params, X, df["y"].to_numpy()), 4)}
Path(args.ckpt).with_name("eval.json").write_text(json.dumps(res, indent=2))
print(res)
''')
    put(root, "README.md", """
# small-data-mlp

A one-hidden-layer tanh MLP on a small, noisy tabular problem (320 training rows, 30 features,
about 8% label noise).

    python src/pipeline/run.py                  # trains, keeps the best-validation checkpoint -> runs/baseline/
    python src/pipeline/evaluate.py             # reloads runs/baseline/model.npz and scores data/test.csv

The checkpoint stores the weights and the training-set standardisation statistics.

## Result

**@CLAIM@ test accuracy** (1500 held-out rows). Training accuracy at the selected checkpoint is 0.91, so there is a
large train/test gap; with 320 rows and 128 hidden units this is expected, and we did not tune against it.
""")


register("ctrl_mlp_small_data", "control",
         "Reproduce the small-data MLP's test accuracy from the saved checkpoint and confirm it holds.",
         [], [], "None (clean control). Train accuracy 0.91 vs test 0.76 is a legitimate overfit gap on 320 rows; "
         "the checkpoint copies best-epoch weights and stores mu/sd, so evaluate.py reproduces run.py.")


# ---------------------------------------------------------------------------------------------
# 4. leak_customer_rows (duplicate entities across splits; k-NN)
# ---------------------------------------------------------------------------------------------
def t_customer_rows(root):
    r = rng(404)
    nc, nm = 500, 8
    rows = []
    for c in range(nc):
        u = r.normal()
        cust = r.normal(0, 1, 4)
        base = r.normal(0, 1, 2)
        for m in range(nm):
            tv = base + 0.2 * r.normal(0, 1, 2)
            z = 2.2 * u + 0.6 * tv[0] - 0.5 * tv[1]
            y = int(r.random() < sig(z))
            rows.append([f"C{c:04d}", m + 1,
                         round(float(3 + cust[0]), 3), round(float(50 + 10 * cust[1]), 3),
                         round(float(25 + 6 * cust[2]), 3), round(float(20 + 4 * cust[3]), 3),
                         round(float(tv[0]), 3), round(float(tv[1]), 3), y])
    df = pd.DataFrame(rows, columns=["customer_id", "month", "tenure_years", "region_density",
                                     "avg_basket", "plan_price", "support_calls", "logins_30d", "churned"])
    csv(root, "data/accounts.csv", df)
    put(root, "run_experiment.py", r'''
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/accounts.csv")
p.add_argument("--k", type=int, default=5)
p.add_argument("--seed", type=int, default=7)
p.add_argument("--test-frac", type=float, default=0.25)
p.add_argument("--out", default="results/knn_metrics.json")
args = p.parse_args()

df = pd.read_csv(args.data)
feature_cols = [c for c in df.columns if c not in ("customer_id", "month", "churned")]

rng = np.random.default_rng(args.seed)
idx = rng.permutation(len(df))
n_test = int(len(df) * args.test_frac)
test_idx, train_idx = idx[:n_test], idx[n_test:]
train, test = df.iloc[train_idx], df.iloc[test_idx]

mu, sd = train[feature_cols].mean(), train[feature_cols].std()
Xtr = ((train[feature_cols] - mu) / sd).to_numpy()
Xte = ((test[feature_cols] - mu) / sd).to_numpy()
ytr, yte = train["churned"].to_numpy(), test["churned"].to_numpy()

d2 = (Xte ** 2).sum(1)[:, None] - 2 * Xte @ Xtr.T + (Xtr ** 2).sum(1)[None, :]
nn = np.argpartition(d2, args.k, axis=1)[:, : args.k]
pred = (ytr[nn].mean(1) >= 0.5).astype(int)

res = {"k": args.k, "test_accuracy": round(float((pred == yte).mean()), 4),
       "n_train": int(len(train)), "n_test": int(len(test)),
       "majority_baseline": round(float(max(yte.mean(), 1 - yte.mean())), 4)}
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, indent=2))
print(res)
''')
    put(root, "README.md", """
# churn-knn

Predicting monthly churn from account snapshots (`data/accounts.csv`: 8 monthly snapshots per
account; features are account profile and recent activity; label `churned`).

    python run_experiment.py --k 5 --seed 7

Writes `results/knn_metrics.json`. 75/25 random split, features standardised with training
statistics, majority vote of the k nearest training snapshots.

## Result

**@CLAIM@ test accuracy** against a majority-class baseline of about 0.5.
""")


register("leak_customer_rows", "data_leakage",
         "Audit the churn k-NN experiment: is the reported test accuracy a fair estimate for new customers?",
         ["data.train_test_overlap"], ["repro.claim_not_met"],
         "run_experiment.py:20-23 splits rows at random, but data/accounts.csv has 8 snapshots per customer_id (500 customers) "
         "so the same customer appears in train and test; k-NN memorises the near-identical profile features.",
         fixes=[("run_experiment.py",
                 "idx = rng.permutation(len(df))\nn_test = int(len(df) * args.test_frac)\ntest_idx, train_idx = idx[:n_test], idx[n_test:]\ntrain, test = df.iloc[train_idx], df.iloc[test_idx]\n",
                 "ids = df['customer_id'].unique()\nids = ids[rng.permutation(len(ids))]\ntest_ids = set(ids[: int(len(ids) * args.test_frac)])\nmask = df['customer_id'].isin(test_ids).to_numpy()\ntrain, test = df[~mask], df[mask]\n")])


# ---------------------------------------------------------------------------------------------
# 5. leak_merchant_encoding (target encoding computed on all rows)
# ---------------------------------------------------------------------------------------------
def t_merchant(root):
    r = rng(505)
    n = 5000
    merchants = [f"M{i:04d}" for i in range(1000)]
    m = r.integers(0, 1000, n)
    amount = np.round(r.lognormal(3.5, 0.9, n), 2)
    hour = r.integers(0, 24, n)
    prev = r.poisson(1.0, n)
    dist = np.round(r.gamma(2.0, 8.0, n), 2)
    z = (-1.0 + 0.6 * (np.log(amount) - 3.5) + 0.5 * (prev - 1) + 0.03 * (dist - 16)
         + 0.4 * ((hour < 5).astype(float)))
    y = (r.random(n) < sig(z)).astype(int)
    df = pd.DataFrame(dict(claim_id=np.arange(n), merchant=[merchants[i] for i in m], amount=amount,
                           hour=hour, n_prev_claims=prev, distance_km=dist, is_fraud=y))
    csv(root, "data/claims.csv", df)
    put(root, "scripts/fit_and_score.py", r'''
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
''')
    put(root, "README.md", """
# claims-fraud-scorer

Scores insurance claims for fraud with a logistic regression over claim attributes plus a
per-merchant fraud-rate feature (`merchant_rate`, a mean-target encoding of the `merchant` column).

    python scripts/fit_and_score.py

Writes `artifacts/metrics.json` and `artifacts/test_scores.csv`. 70/30 random split.

## Result

The merchant-rate feature lifts held-out **test AUC to @CLAIM@**.
""")


register("leak_merchant_encoding", "data_leakage",
         "The merchant fraud-rate feature seems to help a lot. Check the AUC claim is not an artifact.",
         ["data.target_leak"], ["repro.claim_not_met"],
         "scripts/fit_and_score.py:28-29 computes the per-merchant mean of is_fraud over ALL rows (including the test rows) before "
         "the split, so each test row's own label is inside its merchant_rate feature (merchant has no real effect in the data).",
         fixes=[("scripts/fit_and_score.py",
                 '    rate = df.groupby("merchant")[TARGET].mean()\n    df["merchant_rate"] = df["merchant"].map(rate)\n\n    rng = np.random.default_rng(a.seed)\n    idx = rng.permutation(len(df))\n    cut = int(0.7 * len(df))\n    tr, te = df.iloc[idx[:cut]], df.iloc[idx[cut:]]\n',
                 '    rng = np.random.default_rng(a.seed)\n    idx = rng.permutation(len(df))\n    cut = int(0.7 * len(df))\n    tr, te = df.iloc[idx[:cut]].copy(), df.iloc[idx[cut:]].copy()\n    rate = tr.groupby("merchant")[TARGET].mean()\n    tr["merchant_rate"] = tr["merchant"].map(rate)\n    te["merchant_rate"] = te["merchant"].map(rate).fillna(tr[TARGET].mean())\n')])


# ---------------------------------------------------------------------------------------------
# 6. prep_plan_codes (per-split categorical coding)
# ---------------------------------------------------------------------------------------------
def t_plan(root):
    r = rng(606)
    plans = ["basic", "plus", "pro", "team", "enterprise"]
    pe = {"basic": -1.6, "plus": -0.6, "pro": 0.4, "team": 1.0, "enterprise": 1.9}
    chans = ["web", "partner", "phone"]
    ce = {"web": 0.0, "partner": 0.5, "phone": -0.5}

    def gen(n):
        plan = r.choice(plans, n, p=[0.3, 0.25, 0.2, 0.15, 0.1])
        ch = r.choice(chans, n)
        seats = r.integers(1, 60, n)
        hours = np.round(r.gamma(3, 6, n), 1)
        tick = r.poisson(2, n)
        z = (np.array([pe[p] for p in plan]) + np.array([ce[c] for c in ch]) + 0.015 * (seats - 30)
             + 0.03 * (hours - 18) - 0.2 * (tick - 2))
        y = (r.random(n) < sig(z)).astype(int)
        return pd.DataFrame(dict(plan=plan, channel=ch, seats=seats, usage_hours=hours,
                                 support_tickets=tick, renewed=y))
    csv(root, "data/train.csv", gen(3000))
    csv(root, "data/test.csv", gen(1000))
    put(root, "config.json", """
{
  "train_path": "data/train.csv",
  "test_path": "data/test.csv",
  "label": "renewed",
  "categorical": ["plan", "channel"],
  "numeric": ["seats", "usage_hours", "support_tickets"],
  "lr": 0.2,
  "epochs": 500,
  "out_dir": "out"
}
""")
    put(root, "pipeline.py", r'''
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def featurize(df, cfg):
    blocks = []
    for col in cfg["categorical"]:
        codes, uniques = pd.factorize(df[col])
        blocks.append(np.eye(len(uniques))[codes])
    blocks.append(df[cfg["numeric"]].to_numpy(dtype=float))
    return np.hstack(blocks)


def train_logreg(X, y, lr, epochs):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * X.T @ (p - y) / len(y)
        b -= lr * (p - y).mean()
    return w, b


def main(path):
    cfg = json.loads(Path(path).read_text())
    train = pd.read_csv(cfg["train_path"])
    test = pd.read_csv(cfg["test_path"])

    Xtr = featurize(train, cfg)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (featurize(test, cfg) - mu) / sd
    ytr, yte = train[cfg["label"]].to_numpy(), test[cfg["label"]].to_numpy()

    w, b = train_logreg(Xtr, ytr, cfg["lr"], cfg["epochs"])
    acc = float((((Xte @ w + b) > 0).astype(int) == yte).mean())
    res = {"accuracy": round(acc, 4), "n_train": len(train), "n_test": len(test)}
    out = Path(cfg["out_dir"])
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(res)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.json")
''')
    put(root, "README.md", """
# subscription-renewal

Predicts whether a subscription renews from plan tier, sales channel and usage.

    python pipeline.py config.json

Categorical columns are one-hot encoded, everything is standardised with training statistics, and a
logistic regression is trained with full-batch gradient descent. Writes `out/metrics.json`.

## Result

**@CLAIM@ accuracy** on `data/test.csv` (1000 rows).
""")


register("prep_plan_codes", "broken_preprocessing",
         "Our renewal model scores well in the notebook but the pipeline script reports a lower accuracy. Find out why.",
         ["pipeline.eval_path_mismatch"], ["repro.claim_not_met"],
         "pipeline.py:12 builds the one-hot category index with pd.factorize on each frame separately, so category->column "
         "assignment follows first-appearance order in each file and differs between train and test.",
         fixes=[("pipeline.py",
                 "def featurize(df, cfg):\n    blocks = []\n    for col in cfg[\"categorical\"]:\n        codes, uniques = pd.factorize(df[col])\n        blocks.append(np.eye(len(uniques))[codes])\n",
                 "VOCAB = {}\n\n\ndef featurize(df, cfg):\n    blocks = []\n    for col in cfg[\"categorical\"]:\n        vocab = VOCAB.setdefault(col, sorted(df[col].unique()))\n        codes = df[col].map({v: i for i, v in enumerate(vocab)}).to_numpy()\n        blocks.append(np.eye(len(vocab))[codes])\n")])


# ---------------------------------------------------------------------------------------------
# 7. metric_review_f1 (threshold on logits)
# ---------------------------------------------------------------------------------------------
def t_review(root):
    r = rng(707)
    n = 3000
    X = r.normal(0, 1, (n, 8))
    wv = np.array([0.5, -0.4, 0.35, 0.3, 0.0, 0.25, -0.2, 0.0])
    z = X @ wv - 0.2
    y = (r.random(n) < sig(z)).astype(int)
    names = ["n_words", "n_exclaim", "avg_word_len", "pos_lexicon", "neg_lexicon", "caps_ratio",
             "emoji_count", "reviewer_age"]
    df = pd.DataFrame(X, columns=names)
    df["positive"] = y
    csv(root, "data/reviews.csv", df)
    put(root, "params.json", """
{"lr": 0.05, "epochs": 120, "l2": 0.02, "threshold": 0.5, "seed": 5, "test_frac": 0.3}
""")
    put(root, "train.py", r'''
import json
from pathlib import Path

import numpy as np
import pandas as pd

P = json.loads(Path("params.json").read_text())
df = pd.read_csv("data/reviews.csv")
feats = [c for c in df.columns if c != "positive"]

rng = np.random.default_rng(P["seed"])
idx = rng.permutation(len(df))
cut = int(len(df) * (1 - P["test_frac"]))
tr, te = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
mu, sd = tr[feats].mean(), tr[feats].std()
Xtr, Xte = ((tr[feats] - mu) / sd).to_numpy(), ((te[feats] - mu) / sd).to_numpy()
ytr, yte = tr["positive"].to_numpy(), te["positive"].to_numpy()


def decision(X, w, b):
    return X @ w + b


w, b = np.zeros(len(feats)), 0.0
for _ in range(P["epochs"]):
    p = 1 / (1 + np.exp(-decision(Xtr, w, b)))
    w -= P["lr"] * (Xtr.T @ (p - ytr) / len(ytr) + P["l2"] * w)
    b -= P["lr"] * (p - ytr).mean()


def f1(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


scores = decision(Xte, w, b)
pred = (scores > P["threshold"]).astype(int)

out = Path("report")
out.mkdir(exist_ok=True)
pd.DataFrame({"score": scores, "pred": pred, "label": yte}).to_csv(out / "predictions.csv", index=False)
m = {"f1": round(f1(yte, pred), 4), "n_test": len(yte), "positive_rate_test": round(float(yte.mean()), 4)}
(out / "metrics.json").write_text(json.dumps(m, indent=2))
print(m)
''')
    put(root, "README.md", """
# review-sentiment-lr

Logistic regression over eight hand-built review features (`data/reviews.csv`, label `positive`).

    python train.py              # reads params.json

Writes `report/metrics.json` and `report/predictions.csv`.

## Result

Test **F1 of @CLAIM@** for the positive class at the default 0.5 probability threshold
(70/30 split, seed 5).
""")


register("metric_review_f1", "incorrect_metric",
         "Check that the F1 number for the positive class is computed correctly from the model's predictions.",
         ["eval.metric_mismatch"], ["repro.claim_not_met"],
         "train.py:38-39 compares the raw decision function (logits) to the 0.5 probability threshold, so 'pred' is positive only when "
         "logit>0.5 (p>0.62); the README's 0.5 probability threshold is not what is applied and recall collapses.",
         fixes=[("train.py", "scores = decision(Xte, w, b)\n", "scores = 1 / (1 + np.exp(-decision(Xte, w, b)))\n")])


# ---------------------------------------------------------------------------------------------
# 8. ckpt_signal_mlp (best-epoch snapshot aliases live weights)
# ---------------------------------------------------------------------------------------------
def t_signal(root):
    r = rng(808)
    n = 1000
    X = r.normal(0, 1, (n, 40))
    wv = np.array([1.0, -0.9, 0.8])
    lin = np.tanh(X[:, :3] @ wv) * 2 + 0.4 * X[:, 0] * X[:, 2] + r.normal(0, 0.5, n)
    y = (lin > 0).astype(int)
    y = np.where(r.random(n) < 0.12, 1 - y, y)
    X = X * r.uniform(0.5, 20, 40) + r.uniform(-30, 30, 40)
    df = pd.DataFrame(X, columns=[f"s{i:02d}" for i in range(40)])
    df["target"] = y
    df["split"] = ["train"] * 150 + ["val"] * 150 + ["test"] * 700
    csv(root, "data/signals.csv", df)
    put(root, "scripts/train.py", r'''
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/signals.csv")
ap.add_argument("--epochs", type=int, default=400)
ap.add_argument("--hidden", type=int, default=96)
ap.add_argument("--lr", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ckpt-dir", default="checkpoints")
ap.add_argument("--log-dir", default="logs")
args = ap.parse_args()

df = pd.read_csv(args.data)
feats = [c for c in df.columns if c not in ("target", "split")]
part = {s: df[df["split"] == s] for s in ("train", "val", "test")}
mu = part["train"][feats].mean().to_numpy()
sd = part["train"][feats].std().to_numpy()
X = {s: (d[feats].to_numpy() - mu) / sd for s, d in part.items()}
y = {s: d["target"].to_numpy() for s, d in part.items()}

r = np.random.default_rng(args.seed)
d_in = len(feats)
params = {
    "W1": r.normal(0, 1 / np.sqrt(d_in), (d_in, args.hidden)),
    "b1": np.zeros(args.hidden),
    "W2": r.normal(0, 1 / np.sqrt(args.hidden), (args.hidden, 1)),
    "b2": np.zeros(1),
}


def logits(p, Xs):
    H = np.tanh(Xs @ p["W1"] + p["b1"])
    return H, (H @ p["W2"] + p["b2"]).ravel()


def acc(p, split):
    return float(((logits(p, X[split])[1] > 0).astype(int) == y[split]).mean())


def step(p, lr):
    H, z = logits(p, X["train"])
    g = (1 / (1 + np.exp(-z)) - y["train"]) / len(z)
    dH = (g[:, None] @ p["W2"].T) * (1 - H ** 2)
    p["W2"] -= lr * (H.T @ g[:, None])
    p["b2"] -= lr * g.sum()
    p["W1"] -= lr * (X["train"].T @ dH)
    p["b1"] -= lr * dH.sum(0)


log = []
best = {"val": -1.0, "epoch": 0, "test": None, "params": None}
for ep in range(1, args.epochs + 1):
    step(params, args.lr)
    v, t = acc(params, "val"), acc(params, "test")
    log.append((ep, acc(params, "train"), v, t))
    if v > best["val"]:
        best.update(val=v, epoch=ep, test=t, params={k: a.copy() for k, a in params.items()})

ck = Path(args.ckpt_dir)
ck.mkdir(parents=True, exist_ok=True)
np.savez(ck / "best.npz", **best["params"])
lg = Path(args.log_dir)
lg.mkdir(parents=True, exist_ok=True)
pd.DataFrame(log, columns=["epoch", "train_acc", "val_acc", "test_acc"]).to_csv(lg / "train_log.csv", index=False)
final = {"best_epoch": best["epoch"], "val_acc": round(best["val"], 4), "test_acc": round(best["test"], 4)}
(lg / "final.json").write_text(json.dumps(final, indent=2))
print(final)
''')
    put(root, "scripts/eval_ckpt.py", r'''
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/best.npz")
ap.add_argument("--data", default="data/signals.csv")
ap.add_argument("--log-dir", default="logs")
args = ap.parse_args()

z = np.load(args.ckpt)
df = pd.read_csv(args.data)
feats = [c for c in df.columns if c not in ("target", "split")]
res = {}
for s in ("val", "test"):
    d = df[df["split"] == s]
    X = d[feats].to_numpy()
    H = np.tanh(X @ z["W1"] + z["b1"])
    pred = ((H @ z["W2"] + z["b2"]).ravel() > 0).astype(int)
    res[f"{s}_acc"] = round(float((pred == d["target"].to_numpy()).mean()), 4)
Path(args.log_dir, "eval.json").write_text(json.dumps(res, indent=2))
print(res)
''')
    put(root, "README.md", """
# signal-mlp

MLP (40 inputs, 96 tanh units) on a noisy sensor-classification set. `data/signals.csv` carries an
explicit `split` column (train 150 / val 150 / test 700).

    python scripts/train.py                  # trains 400 epochs, keeps the best-validation weights
    python scripts/eval_ckpt.py              # reloads checkpoints/best.npz and re-scores val/test

Logs go to `logs/` (`train_log.csv` per epoch, `final.json`).

## Result

Selecting the epoch by validation accuracy gives **@CLAIM@ test accuracy** (`logs/final.json`);
`eval_ckpt.py` on the saved checkpoint is expected to match.
""")


register("ckpt_signal_mlp", "checkpoint_failure",
         "Make sure the checkpoint we ship reproduces the accuracy in the README.",
         ["ckpt.roundtrip_mismatch"], ["repro.claim_not_met", "pipeline.eval_path_mismatch"],
         "scripts/train.py:66 saves only the weights to checkpoints/best.npz (the training-set mean/std used for standardisation are not stored) "
         "and scripts/eval_ckpt.py:20 feeds raw, unstandardised features to the loaded weights, so the reloaded checkpoint does not reproduce final.json.",
         fixes=[("scripts/train.py", 'np.savez(ck / "best.npz", **best["params"])', 'np.savez(ck / "best.npz", mu=mu, sd=sd, **best["params"])'),
                ("scripts/eval_ckpt.py", 'X = d[feats].to_numpy()\n', 'X = (d[feats].to_numpy() - z["mu"]) / z["sd"]\n')])


# ---------------------------------------------------------------------------------------------
# 9. repro_seeds_softmax (only legacy RNG seeded)
# ---------------------------------------------------------------------------------------------
def t_seeds(root):
    r = rng(909)
    cent = r.normal(0, 1.1, (3, 12))

    def gen(n):
        y = r.integers(0, 3, n)
        X = cent[y] + r.normal(0, 1.6, (n, 12))
        df = pd.DataFrame(X, columns=[f"m{i}" for i in range(12)])
        df["variety"] = y
        return df
    csv(root, "data/train.csv", gen(900))
    csv(root, "data/test.csv", gen(300))
    put(root, "train.py", r'''
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=0.3)
args = ap.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

tr, te = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
feats = [c for c in tr.columns if c != "variety"]
mu, sd = tr[feats].mean(), tr[feats].std()
Xtr, Xte = ((tr[feats] - mu) / sd).to_numpy(), ((te[feats] - mu) / sd).to_numpy()
ytr, yte = tr["variety"].to_numpy(), te["variety"].to_numpy()
K = 3

W = np.random.randn(Xtr.shape[1], K) * 0.05
b = np.zeros(K)
shuffler = np.random.default_rng()
for _ in range(args.epochs):
    order = shuffler.permutation(len(Xtr))
    for i in range(0, len(order), args.batch):
        j = order[i:i + args.batch]
        z = Xtr[j] @ W + b
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        p[np.arange(len(j)), ytr[j]] -= 1
        W -= args.lr * Xtr[j].T @ p / len(j)
        b -= args.lr * p.mean(0)

acc = float(((Xte @ W + b).argmax(1) == yte).mean())
out = Path("runs") / f"seed{args.seed}"
out.mkdir(parents=True, exist_ok=True)
res = {"seed": args.seed, "test_accuracy": round(acc, 4)}
(out / "metrics.json").write_text(json.dumps(res))
print(res)
''')
    put(root, "README.md", """
# seed-softmax

Three-class softmax regression trained with a very short minibatch-SGD schedule (2 epochs).

    python train.py --seed 0

Writes `runs/seed0/metrics.json`. All randomness is controlled by `--seed`, so re-running with the
same seed gives the same number.

## Result

**@CLAIM@ test accuracy** for `--seed 0` (300 test rows).
""")


register("repro_seeds_softmax", "reproducibility",
         "A colleague says python train.py --seed 0 gives a different accuracy every time. Verify the README's claim.",
         ["repro.nondeterministic"], ["repro.claim_not_met"],
         "train.py:16-17 and :28: --seed seeds only the legacy np.random/random generators (weight init); the minibatch order comes from "
         "np.random.default_rng() with no seed, so the same command gives different accuracies.",
         fixes=[("train.py", "shuffler = np.random.default_rng()", "shuffler = np.random.default_rng(args.seed)")])


# ---------------------------------------------------------------------------------------------
# 10. repro_tabular_config (claim from a different config)
# ---------------------------------------------------------------------------------------------
def t_config(root):
    r = rng(1010)
    n = 3000
    X = r.normal(0, 1, (n, 12))
    z = 1.2 * X[:, 0] + 1.8 * X[:, 0] * X[:, 1] + 1.2 * X[:, 2] ** 2 - 1.2 + 0.4 * X[:, 3]
    y = (r.random(n) < sig(z * 1.5)).astype(int)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(12)])
    df["label"] = y
    csv(root, "data/tabular.csv", df)
    put(root, "configs/default.json", """
{"epochs": 200, "lr": 0.1, "l2": 0.01, "poly": false, "seed": 1}
""")
    put(root, "configs/long.json", """
{"epochs": 800, "lr": 0.1, "l2": 0.001, "poly": true, "seed": 1}
""")
    put(root, "notes/experiments.md", """
# experiment log

- 03-02  configs/default.json   run ok, linear model only. auc see results/metrics.json
- 03-04  configs/long.json      added pairwise/squared terms for x0..x3, longer schedule. auc 0.919
- 03-05  copied number to README
""")
    put(root, "src/exp/run.py", r'''
import argparse
import json
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def roc_auc(y, s):
    r = pd.Series(s).rank().to_numpy()
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def expand(X, poly):
    if not poly:
        return X
    extra = [X[:, i] * X[:, j] for i, j in combinations_with_replacement(range(4), 2)]
    return np.column_stack([X] + extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())

    df = pd.read_csv(ROOT / "data" / "tabular.csv")
    feats = [c for c in df.columns if c != "label"]
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.permutation(len(df))
    cut = int(0.7 * len(df))
    X = expand(df[feats].to_numpy(), cfg["poly"])
    y = df["label"].to_numpy()
    Xtr, Xte, ytr, yte = X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]
    mu, sd = Xtr.mean(0), Xtr.std(0)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    w, b = np.zeros(Xtr.shape[1]), 0.0
    for _ in range(cfg["epochs"]):
        p = 1 / (1 + np.exp(-(Xtr @ w + b)))
        w -= cfg["lr"] * (Xtr.T @ (p - ytr) / len(ytr) + cfg["l2"] * w)
        b -= cfg["lr"] * (p - ytr).mean()

    res = {"config": Path(a.config).name, "test_auc": round(roc_auc(yte, Xte @ w + b), 4)}
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(res)


if __name__ == "__main__":
    main()
''')
    put(root, "README.md", """
# tabular-poly

Logistic regression on 12 tabular features; optional squared / pairwise terms over the first four.

    python src/exp/run.py                      # configs/default.json
    python src/exp/run.py --config configs/long.json

Writes `results/metrics.json`.

## Result

Held-out **test AUC of @CLAIM@** (70/30 split).
""")


register("repro_tabular_config", "reproducibility",
         "Reproduce the headline AUC in the README by running the repo as documented.",
         ["repro.claim_not_met"], [],
         "README quotes AUC 0.919, which comes from configs/long.json (polynomial terms, 800 epochs; see notes/experiments.md) but the "
         "documented command uses configs/default.json (linear, 200 epochs), which scores much lower.")


# ---------------------------------------------------------------------------------------------
# 11. other_label_join (case-mismatched id join silently labels 0)
# ---------------------------------------------------------------------------------------------
def t_label_join(root):
    r = rng(1111)
    n = 2400
    X = r.normal(0, 1, (n, 8))
    wv = np.array([1.0, -0.8, 0.7, 0.6, -0.5, 0.4, 0.0, 0.0])
    y = (r.random(n) < sig(5.0 * (X @ wv) / 1.7 - 0.3)).astype(int)
    ids = [f"S-{i:05d}" for i in range(n)]
    feats = pd.DataFrame(X, columns=[f"m{i}" for i in range(1, 9)])
    feats.insert(0, "id", ids)
    lower = r.random(n) < 0.3
    lab_ids = [i.lower() if lo else i for i, lo in zip(ids, lower)]
    labs = pd.DataFrame({"sample_id": lab_ids, "diagnosis": y})
    labs = labs.iloc[r.permutation(n)]
    csv(root, "data/raw/features.csv", feats)
    csv(root, "data/raw/labels.csv", labs)
    put(root, "workflow/run_classifier.py", r'''
import json
from pathlib import Path

import numpy as np
import pandas as pd

feats = pd.read_csv("data/raw/features.csv")
labs = pd.read_csv("data/raw/labels.csv")

label_map = dict(zip(labs["sample_id"], labs["diagnosis"]))
feats["diagnosis"] = feats["id"].map(label_map).fillna(0).astype(int)

cols = [c for c in feats.columns if c.startswith("m")]
rng = np.random.default_rng(21)
idx = rng.permutation(len(feats))
cut = int(0.75 * len(feats))
tr, te = feats.iloc[idx[:cut]], feats.iloc[idx[cut:]]
mu, sd = tr[cols].mean(), tr[cols].std()
Xtr, Xte = ((tr[cols] - mu) / sd).to_numpy(), ((te[cols] - mu) / sd).to_numpy()
ytr, yte = tr["diagnosis"].to_numpy(), te["diagnosis"].to_numpy()

w, b = np.zeros(len(cols)), 0.0
for _ in range(400):
    p = 1 / (1 + np.exp(-(Xtr @ w + b)))
    w -= 0.2 * Xtr.T @ (p - ytr) / len(ytr)
    b -= 0.2 * (p - ytr).mean()

acc = float((((Xte @ w + b) > 0).astype(int) == yte).mean())
Path("reports").mkdir(exist_ok=True)
res = {"accuracy": round(acc, 4), "n_train": len(tr), "n_test": len(te)}
Path("reports/summary.json").write_text(json.dumps(res, indent=2))
print(res)
''')
    put(root, "README.md", """
# assay-classifier

Classifies assay samples (`data/raw/features.csv`, eight measurements `m1`..`m8`) using diagnosis
labels delivered separately by the lab (`data/raw/labels.csv`, one row per sample, shuffled).

    python workflow/run_classifier.py

Joins features to labels by sample id, splits 75/25, trains a logistic regression and writes
`reports/summary.json`.

## Result

**@CLAIM@ test accuracy.** The lab says all 2400 samples have a label.
""")


register("other_label_join", "other",
         "Reproduce the assay classifier's reported accuracy and check the training data is what the README says it is.",
         ["other:label-join-noise"], ["repro.claim_not_met"],
         "workflow/run_classifier.py:10-11 maps labels by exact sample_id, but about 30% of ids in labels.csv are lower-case; those rows "
         "miss the join and silently get label 0 via fillna(0), turning ~30% of positives into negatives in both training and test labels.",
         fixes=[("workflow/run_classifier.py", 'label_map = dict(zip(labs["sample_id"], labs["diagnosis"]))',
                 'label_map = dict(zip(labs["sample_id"].str.upper(), labs["diagnosis"]))')])


# ---------------------------------------------------------------------------------------------
# 12. other_class_weights (class weights configured but never used)
# ---------------------------------------------------------------------------------------------
def t_weights(root):
    r = rng(1212)
    n = 8000
    X = r.normal(0, 1, (n, 6))
    wv = np.array([1.1, 0.9, -0.8, 0.7, 0.0, 0.0])
    y = (r.random(n) < sig(X @ wv - 4.2)).astype(int)
    df = pd.DataFrame(X, columns=["amount_z", "velocity", "geo_dist", "acct_age_z", "hour_a", "hour_b"])
    df["fraud"] = y
    csv(root, "data/transactions.csv", df)
    put(root, "config.json", """
{"data": "data/transactions.csv", "label": "fraud", "class_weight": "balanced",
 "lr": 0.3, "epochs": 400, "seed": 4, "test_frac": 0.25, "report": "eval_report.json"}
""")
    put(root, "train.py", r'''
import json
from pathlib import Path

import numpy as np
import pandas as pd

cfg = json.loads(Path("config.json").read_text())
df = pd.read_csv(cfg["data"])
feats = [c for c in df.columns if c != cfg["label"]]
rng = np.random.default_rng(cfg["seed"])
idx = rng.permutation(len(df))
cut = int(len(df) * (1 - cfg["test_frac"]))
tr, te = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
mu, sd = tr[feats].mean(), tr[feats].std()
Xtr, Xte = ((tr[feats] - mu) / sd).to_numpy(), ((te[feats] - mu) / sd).to_numpy()
ytr, yte = tr[cfg["label"]].to_numpy(), te[cfg["label"]].to_numpy()


def sample_weights(y, mode):
    if mode != "balanced":
        return np.ones(len(y))
    counts = np.bincount(y, minlength=2)
    cw = len(y) / (2.0 * counts)
    return cw[y]


sw = sample_weights(ytr, cfg.get("class_weight"))
print("effective training size:", round(float(sw.sum()), 1))

w, b = np.zeros(len(feats)), 0.0
for _ in range(cfg["epochs"]):
    p = 1 / (1 + np.exp(-(Xtr @ w + b)))
    err = p - ytr
    w -= cfg["lr"] * Xtr.T @ err / len(ytr)
    b -= cfg["lr"] * err.mean()

pred = ((Xte @ w + b) > 0).astype(int)


recall_fraud = float(pred[yte == 1].mean())
recall_legit = float(1 - pred[yte == 0].mean())

rep = {
    "accuracy": round(float((pred == yte).mean()), 4),
    "balanced_accuracy": round((recall_fraud + recall_legit) / 2, 4),
    "recall_fraud": round(recall_fraud, 4),
    "fraud_rate_test": round(float(yte.mean()), 4),
}
Path(cfg["report"]).write_text(json.dumps(rep, indent=2))
print(rep)
''')
    put(root, "README.md", """
# txn-fraud-lr

Logistic regression for card-fraud detection on a heavily imbalanced set (about 5% fraud).
`config.json` sets `class_weight: balanced` to compensate for the imbalance.

    python train.py

Writes `eval_report.json` (accuracy, balanced accuracy, fraud recall).

## Result

With balanced class weights the model reaches **@CLAIM@ balanced accuracy** on the 25% held-out split, which
is the number to compare, since plain accuracy is dominated by the legitimate class.
""")


register("other_class_weights", "other",
         "Check whether the reported balanced accuracy for the fraud detector really reflects the balanced class-weight training described.",
         ["other:class-weights-ignored"], ["repro.claim_not_met"],
         "train.py:27-35: the per-sample class weights `sw` are computed from config class_weight=balanced but never used in the gradient, "
         "so training is unweighted; the model rarely predicts fraud (low recall) and balanced accuracy is far below the README claim.",
         fixes=[("train.py", "    err = p - ytr\n    w -= cfg[\"lr\"] * Xtr.T @ err / len(ytr)\n    b -= cfg[\"lr\"] * err.mean()\n",
                 "    err = (p - ytr) * sw\n    w -= cfg[\"lr\"] * Xtr.T @ err / len(ytr)\n    b -= cfg[\"lr\"] * err.mean()\n")])


BUILDERS = [("ctrl_credit_logreg", t_credit), ("ctrl_particles_knn", t_particles),
            ("ctrl_mlp_small_data", t_mlp_small), ("leak_customer_rows", t_customer_rows),
            ("leak_merchant_encoding", t_merchant), ("prep_plan_codes", t_plan),
            ("metric_review_f1", t_review), ("ckpt_signal_mlp", t_signal),
            ("repro_seeds_softmax", t_seeds), ("repro_tabular_config", t_config),
            ("other_label_join", t_label_join), ("other_class_weights", t_weights)]

# Filled in after measurement (see README.md for the commands).
EVIDENCE = {
    "ctrl_credit_logreg": "Ran `python main.py` in a clean copy: test_auc 0.8521 (train_auc 0.8403, so no gap), identical on rerun; README claims 0.8521. prior_delinq is a count known at application time, so its strength is legitimate. Clean.",
    "ctrl_particles_knn": "Ran `python scripts/train.py`: accuracy 0.9933, auc 0.9998 on 600 held-out events; README claims 0.9933. Data generated as two Gaussian classes with Bayes accuracy about 0.995, train/test are independent draws, standardisation uses train stats only. Clean (legitimately very high accuracy).",
    "ctrl_mlp_small_data": "Ran `python src/pipeline/run.py` then `python src/pipeline/evaluate.py`: run.py test_accuracy 0.764 (train 0.9102, val 0.7812, best_epoch 96), evaluate.py on the reloaded checkpoint 0.764 (exact round trip); README claims 0.764. The 0.91 vs 0.76 gap is genuine small-data overfit; checkpoint holds copied best-epoch weights plus mu/sd. Clean.",
    "leak_customer_rows": "Ran `python run_experiment.py --k 5 --seed 7`: test_accuracy 0.761 (majority baseline 0.518); README claims 0.76 so the claim reproduces. Fixed copy (split by customer_id groups, 25% of customers held out, same seed/k): test_accuracy 0.514 (baseline 0.51), i.e. all of the lift over chance comes from the same customers appearing in both splits (500 customers x 8 rows).",
    "leak_merchant_encoding": "Ran `python scripts/fit_and_score.py`: test_auc 0.8225; README claims 0.82 so it reproduces. Fixed copy (merchant mean-target encoding fit on the 70% train rows only, unseen merchants get the train mean): test_auc 0.5951. The generator gives merchant no effect on the label.",
    "prep_plan_codes": "Ran `python pipeline.py config.json`: accuracy 0.561; README claims 0.72. Fixed copy (category vocabulary fit on train, applied to test): accuracy 0.723. Same data, same model; only the category-to-column mapping differs between train and test.",
    "metric_review_f1": "Ran `python train.py`: f1 0.2779 (positive_rate_test 0.4578); README claims 0.58 at a 0.5 probability threshold. Fixed copy (sigmoid applied to the decision function before the 0.5 threshold): f1 0.5756. Same trained weights in both runs.",
    "ckpt_signal_mlp": "Ran `python scripts/train.py && python scripts/eval_ckpt.py`: final.json best_epoch 98, val_acc 0.7, test_acc 0.7343 (README claim 0.7343 matches the log); eval_ckpt.py on checkpoints/best.npz gives val_acc 0.4467, test_acc 0.4943. Fixed copy (mu/sd saved in the checkpoint and applied in eval_ckpt.py): eval gives val_acc 0.7, test_acc 0.7343, an exact round trip.",
    "repro_seeds_softmax": "Ran `python train.py --seed 0` 10 times in clean copies: 0.8367 0.8367 0.8233 0.8267 0.8433 0.8433 0.84 0.84 0.8233 0.8467 (range 0.8233-0.8467); README claims 0.85 for seed 0. Fixed copy (default_rng(args.seed)): 0.8467 on 4 of 4 runs.",
    "repro_tabular_config": "Ran `python src/exp/run.py` (the documented command, configs/default.json): test_auc 0.6898, deterministic on rerun; README claims 0.919. `python src/exp/run.py --config configs/long.json` gives test_auc 0.919 (polynomial terms, 800 epochs), the number in notes/experiments.md. The claim is reproducible only with a config the README does not use.",
    "other_label_join": "Ran `python workflow/run_classifier.py`: accuracy 0.7633; README claims 0.89. Fixed copy (upper-case the label file ids before joining so all 2400 rows match): accuracy 0.8883. About 30% of label ids are lower-case and fall through fillna(0), so roughly 30% of positives are labelled 0.",
    "other_class_weights": "Ran `python train.py`: accuracy 0.9475, balanced_accuracy 0.5636, recall_fraud 0.1293 (fraud rate 0.058); README claims balanced accuracy 0.74. Fixed copy (gradient error multiplied by the computed sample weights): accuracy 0.783, balanced_accuracy 0.7432, recall_fraud 0.6983.",
}


def main():
    if TASKS.exists():
        shutil.rmtree(TASKS)
    for tid, fn in BUILDERS:
        repo = TASKS / tid / "repo"
        fn(repo)
        for p in repo.rglob("*"):
            if p.is_file() and p.suffix in (".md", ".py"):
                p.write_text(p.read_text().replace("@CLAIM@", CLAIM[tid]))
        meta = dict(TASK_META[tid])
        meta["evidence"] = EVIDENCE.get(tid, "TODO")
        order = ["id", "category", "objective", "expect", "allow", "canaries", "defect", "evidence"]
        (TASKS / tid / "task.json").write_text(json.dumps({k: meta[k] for k in order}, indent=2) + "\n")
    if "--fixed" in sys.argv:
        dest = Path(sys.argv[sys.argv.index("--fixed") + 1])
        for tid, fixes in FIXES.items():
            d = dest / tid
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(TASKS / tid / "repo", d)
            for rel, old, new in fixes:
                t = (d / rel).read_text()
                assert old in t, (tid, rel)
                (d / rel).write_text(t.replace(old, new))


if __name__ == "__main__":
    main()
