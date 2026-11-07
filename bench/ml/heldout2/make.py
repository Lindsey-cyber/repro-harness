#!/usr/bin/env python3
"""Generator for the heldout2 ML-debugging benchmark.

  python3 make.py              writes tasks/<id>/{repo/,task.json}
  python3 make.py --fixed DIR  writes bug-fixed copies to DIR/<id>/ (verification only)

Claims in each README are calibrated by actually running the repo (see claim_from).
Scratch runs happen under $H2_SCRATCH (default: system temp dir).
"""
import io, json, os, shutil, subprocess, sys, tempfile, zipfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- helpers
def csv_bytes(df):
    return df.to_csv(index=False, float_format="%.4f", lineterminator="\n").encode()


def npz_bytes(**arrs):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for k, v in arrs.items():
            b = io.BytesIO()
            np.save(b, np.asarray(v))
            zi = zipfile.ZipInfo(k + ".npy", date_time=(2020, 1, 1, 0, 0, 0))
            z.writestr(zi, b.getvalue())
    return buf.getvalue()


def sig(z):
    return 1 / (1 + np.exp(-z))


def sub(src, old, new):
    assert old in src, old
    return src.replace(old, new, 1)


# ---------------------------------------------------------------- data generators
def data_sensor(rs):
    def mk(n):
        y = (rs.rand(n) < 0.45).astype(int)
        X = rs.randn(n, 6)
        X[:, [0, 2, 3]] += 5.0 * y[:, None]
        X = X * np.array([0.8, 4.0, 1.5, 3.0, 120.0, 0.6]) + np.array([2.1, 55.0, 4.0, 62.0, 1450.0, 8.0])
        df = pd.DataFrame(X, columns=["vib_rms", "temp_c", "vib_peak", "acoustic_db", "rpm", "current_a"])
        df["fault"] = y
        return df
    return {"data/train.csv": csv_bytes(mk(1200)), "data/test.csv": csv_bytes(mk(500))}


def data_meter(rs):
    n = 3200
    temp = rs.randint(0, 10, n) * 3 + 8
    load = rs.randint(0, 9, n) * 0.5
    hour = rs.randint(0, 6, n) * 4
    site = np.where(rs.rand(n) < 0.5, "north", "south")
    z = 0.35 * (temp - 20) / 3 + 0.9 * (load - 2.0) + 0.5 * (hour >= 16) + 0.4 * (site == "south") - 0.9
    y = (rs.rand(n) < sig(z)).astype(int)
    split = np.where(rs.rand(n) < 0.75, "train", "test")
    df = pd.DataFrame({"site": site, "temp_c": temp, "load_kw": load, "hour": hour, "event": y, "split": split})
    return {"data/readings.csv": csv_bytes(df)}


def data_valnpz(rs):
    d = 20
    w = np.zeros(d)
    w[:8] = rs.randn(8) * 0.45
    def mk(n):
        X = rs.randn(n, d)
        y = (rs.rand(n) < sig(X @ w)).astype(int)
        return X, y
    Xt, yt = mk(700)
    Xv, yv = mk(400)
    return {"data/features.npz": npz_bytes(X_train=Xt, y_train=yt, X_val=Xv, y_val=yv)}


def data_loan(rs):
    n = 2600
    emp = rs.choice(["salaried", "self-employed", "retired", "unemployed"], n, p=[0.55, 0.2, 0.15, 0.1])
    purpose = rs.choice(["car", "home", "education", "medical", "other"], n)
    income = np.exp(rs.randn(n) * 0.5 + 10.8)
    debt = rs.beta(2, 4, n)
    prior = rs.poisson(0.45, n)
    z = -2.2 + 3.2 * prior + 3.0 * (debt - 0.33) - 0.5 * np.log(income / 49000) + 0.6 * (emp == "unemployed") + 0.2 * (purpose == "medical")
    y = (rs.rand(n) < sig(z)).astype(int)
    ids = ["LN-%06d" % (100000 + i * 7) for i in range(n)]
    df = pd.DataFrame({"app_id": ids, "employment": emp, "purpose": purpose, "income": income.round(0),
                       "debt_ratio": debt, "prior_defaults": prior, "defaulted": y})
    return {"data/applications.csv": csv_bytes(df)}


def data_ads(rs):
    n, K = 3000, 1400
    p = 1.0 / (np.arange(K) + 4.0) ** 0.9
    p /= p.sum()
    camp = rs.choice(K, n, p=p)
    off = rs.randn(K) * 0.25
    placement = rs.choice(["feed", "story", "search", "banner"], n)
    hour = rs.randint(0, 24, n)
    bid = np.exp(rs.randn(n) * 0.4) * 0.8
    z = off[camp] + 0.5 * (placement == "search") - 0.4 * (placement == "banner") + 0.9 * (bid - 0.85) + 0.15 * np.sin(hour / 24 * 2 * np.pi)
    y = (rs.rand(n) < sig(z)).astype(int)
    df = pd.DataFrame({"impression_id": np.arange(n) + 50000, "campaign": ["cmp_%04d" % c for c in camp],
                       "placement": placement, "hour": hour, "bid": bid, "clicked": y})
    return {"data/clicks.csv": csv_bytes(df)}


def data_patients(rs):
    P = 320
    rows = []
    for pid in range(P):
        z = rs.randn()
        prof = rs.randn(7)
        prof[:] = prof + 0.35 * z * np.array([1, 1, 0, 1, 1, 0, 1])
        nv = rs.randint(4, 10)
        for v in range(nv):
            f = prof + rs.randn(7) * 0.10
            yv = int(rs.rand() < sig(2.4 * z - 0.1))
            rows.append([1000 + pid * 3, v + 1] + list(f) + [yv])
    A = np.array(rows, dtype=float)
    cols = ["age", "sbp", "dbp", "chol", "glucose", "bmi", "n_meds"]
    sc = np.array([12, 15, 9, 30, 20, 4, 2.0]); mu = np.array([58, 128, 80, 200, 105, 28, 3.0])
    df = pd.DataFrame(A[:, 2:9] * sc + mu, columns=cols)
    df.insert(0, "visit_no", A[:, 1].astype(int))
    df.insert(0, "patient_id", A[:, 0].astype(int))
    df["readmitted"] = A[:, 9].astype(int)
    df["n_meds"] = df["n_meds"].round(0)
    order = rs.permutation(len(df))
    return {"data/visits.csv": csv_bytes(df.iloc[order].reset_index(drop=True))}


def data_plans(rs):
    def mk(n, plan_order, reg_order):
        plan = rs.choice(["basic", "plus", "family", "business"], n, p=[0.4, 0.3, 0.2, 0.1])
        reg = rs.choice(["east", "west", "central"], n)
        tenure = rs.exponential(24, n).round(0)
        monthly = (20 + 15 * rs.rand(n) + 10 * (plan == "business") + 5 * (plan == "family")).round(2)
        pe = {"basic": 1.6, "plus": -0.2, "family": -1.4, "business": 0.9}
        z = np.array([pe[p] for p in plan]) - 0.03 * tenure + 0.04 * (monthly - 28) + 0.3 * (reg == "west") - 0.2
        y = (rs.rand(n) < sig(z)).astype(int)
        df = pd.DataFrame({"customer": ["C%05d" % i for i in range(n)], "plan": plan, "region": reg,
                           "tenure": tenure, "monthly": monthly, "churned": y})
        # force a known first-appearance order of the categories
        for col, order in (("plan", plan_order), ("region", reg_order)):
            first = [df.index[df[col] == o][0] for o in order]
            top = df.loc[first]
            df = pd.concat([top, df.drop(first)]).reset_index(drop=True)
        return df
    tr = mk(1500, ["basic", "plus", "family", "business"], ["east", "west", "central"])
    te = mk(700, ["business", "family", "plus", "basic"], ["central", "east", "west"])
    return {"data/train.csv": csv_bytes(tr), "data/test.csv": csv_bytes(te)}


def data_tickets(rs):
    n = 3000
    wait = rs.exponential(25, n)
    tl = np.exp(rs.randn(n) * 0.6 + 5.0)
    prio = rs.choice(["low", "normal", "high"], n, p=[0.3, 0.5, 0.2])
    chan = rs.choice(["email", "chat", "phone"], n)
    z = -2.9 + 0.045 * (wait - 25) + 0.0025 * (tl - 190) + 0.9 * (prio == "high") + 0.5 * (chan == "phone")
    y = np.where(rs.rand(n) < sig(z), "escalated", "resolved")
    df = pd.DataFrame({"ticket_id": ["T-%05d" % (7000 + i) for i in range(n)], "wait_min": wait, "text_len": tl.round(0),
                       "priority": prio, "channel": chan, "outcome": y})
    return {"data/tickets.csv": csv_bytes(df)}


def data_overfit(rs):
    d = 50
    w = np.zeros(d); w[:10] = rs.randn(10) * 0.55
    def mk(n):
        X = rs.randn(n, d)
        y = (rs.rand(n) < sig(X @ w)).astype(int)
        df = pd.DataFrame(X, columns=["f%02d" % i for i in range(d)])
        df["y"] = y
        return df
    return {"data/train.csv": csv_bytes(mk(110)), "data/val.csv": csv_bytes(mk(300)), "data/test.csv": csv_bytes(mk(600))}


def data_posts(rs):
    topics = ["sports", "finance", "travel", "science"]
    tw = {t: ["%s%02d" % (t[:3], i) for i in range(30)] for t in topics}
    bg = ["w%03d" % i for i in range(500)]
    rows = []
    for i in range(1400):
        t = topics[rs.randint(4)]
        toks = [rs.choice(tw[t]) if rs.rand() < 0.3 else rs.choice(bg) for _ in range(22)]
        rows.append(["d%05d" % i, " ".join(toks), t])
    return {"data/posts.csv": csv_bytes(pd.DataFrame(rows, columns=["doc_id", "text", "topic"]))}


def data_wine(rs):
    d = 11
    w = rs.randn(d) * 0.5
    scale, shift = rs.uniform(0.5, 3, d), rs.uniform(0, 10, d)
    def mk(n):
        X = rs.randn(n, d)
        y = (rs.rand(n) < sig(X @ w * 0.7)).astype(int)
        df = pd.DataFrame(X * scale + shift, columns=
                          ["acidity", "sugar", "chlorides", "so2_free", "so2_total", "density", "ph", "sulphates", "alcohol", "citric", "volatile"])
        df["good"] = y
        return df
    return {"data/train.csv": csv_bytes(mk(500)), "data/test.csv": csv_bytes(mk(120))}


def data_calls(rs):
    n = 2400
    cls = rs.choice(["wren", "thrush", "finch"], n)
    mean = {"wren": [6.5, 3.0, 1.0, 0.30, 0.8], "thrush": [4.0, 2.2, 2.5, 0.18, 1.6], "finch": [5.0, 4.2, 1.8, 0.25, 1.1]}
    snr = rs.uniform(0, 25, n)
    sd = 0.45 + 0.17 * np.maximum(0, 15 - snr)
    M = np.array([mean[c] for c in cls]) + rs.randn(n, 5) * sd[:, None] * np.array([1, 0.5, 0.5, 0.1, 0.4])
    df = pd.DataFrame(M, columns=["f_low", "f_mid", "f_high", "zcr", "duration"])
    df.insert(0, "snr_db", snr)
    df.insert(0, "clip_id", ["clip%05d" % i for i in range(n)])
    df["species"] = cls
    df["split"] = np.where(rs.rand(n) < 0.7, "train", "test")
    return {"data/calls.csv": csv_bytes(df)}


# ================================================================ repos
# Each repo_* returns {path: text}. README contains @CLAIM@ (calibrated by running).

def repo_sensor():
    return {
        "README.md": r'''# bearing-fault

Binary fault detector for motor bearings from six condition-monitoring channels.

## Run

    python train.py --config configs/default.json

Writes `results/metrics.json`.

## Result

Test accuracy: **@CLAIM@** on `data/test.csv` (500 recordings).

The faulty regime shifts vibration RMS, peak vibration and acoustic level by several
standard deviations, so the classes are almost linearly separable; the model is a plain
logistic regression trained with full-batch gradient descent.
''',
        "configs/default.json": json.dumps({"lr": 0.1, "epochs": 300, "l2": 0.001}, indent=2) + "\n",
        "train.py": r'''import argparse, json, os
import numpy as np
import pandas as pd


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--out", default="results/metrics.json")
    args = ap.parse_args()
    cfg = json.load(open(args.config))

    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    cols = [c for c in tr.columns if c != "fault"]
    mu, sd = tr[cols].values.mean(0), tr[cols].values.std(0)
    Xtr = (tr[cols].values - mu) / sd
    Xte = (te[cols].values - mu) / sd
    ytr, yte = tr["fault"].values, te["fault"].values

    w = np.zeros(Xtr.shape[1])
    b = 0.0
    for _ in range(cfg["epochs"]):
        p = sigmoid(Xtr @ w + b)
        g = Xtr.T @ (p - ytr) / len(ytr) + cfg["l2"] * w
        w -= cfg["lr"] * g
        b -= cfg["lr"] * float(np.mean(p - ytr))

    acc_tr = float(np.mean((sigmoid(Xtr @ w + b) > 0.5) == ytr))
    acc_te = float(np.mean((sigmoid(Xte @ w + b) > 0.5) == yte))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"train_accuracy": acc_tr, "test_accuracy": acc_te}, open(args.out, "w"), indent=2)
    print("test accuracy %.4f" % acc_te)


if __name__ == "__main__":
    main()
''',
    }


def repo_meter():
    return {
        "README.md": r'''# meterlab

Predicts grid-meter anomaly events from quantised site telemetry.

## Run

    python -m meterlab.cli run exp.json

Summary is written to `runs/<name>/summary.json` (`name` comes from `exp.json`).

## Notes on the data

`data/readings.csv` is exported by the logger, which quantises temperature (3 degC steps),
load (0.5 kW steps) and hour (4 h blocks). Identical feature rows therefore recur many
times across the file, in both splits; this is expected and rows are deliberately not
de-duplicated. `split` is the logger's own train/test assignment.

## Result

Test F1 (positive class = event): **@CLAIM@**.
''',
        "exp.json": json.dumps({"name": "exp_a", "data": "data/readings.csv", "label": "event", "lr": 0.3, "steps": 500}, indent=2) + "\n",
        "meterlab/__init__.py": "",
        "meterlab/metrics.py": r'''import numpy as np


def f1_score(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(int)
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    if tp == 0:
        return 0.0
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)
''',
        "meterlab/models.py": r'''import numpy as np


class Logit:
    def __init__(self, lr=0.1, steps=300):
        self.lr, self.steps = lr, steps

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        for _ in range(self.steps):
            p = 1 / (1 + np.exp(-(X @ self.w + self.b)))
            self.w -= self.lr * X.T @ (p - y) / len(y)
            self.b -= self.lr * np.mean(p - y)
        return self

    def predict(self, X):
        return (1 / (1 + np.exp(-(X @ self.w + self.b))) > 0.5).astype(int)
''',
        "meterlab/cli.py": r'''import json, os, sys
import numpy as np
import pandas as pd
from .models import Logit
from .metrics import f1_score


def featurize(df, mu=None, sd=None):
    X = np.column_stack([df["temp_c"], df["load_kw"], df["hour"], (df["site"] == "south").astype(float)])
    if mu is None:
        mu, sd = X.mean(0), X.std(0)
    return (X - mu) / sd, mu, sd


def run(path):
    cfg = json.load(open(path))
    df = pd.read_csv(cfg["data"])
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    Xtr, mu, sd = featurize(tr)
    Xte, _, _ = featurize(te, mu, sd)
    model = Logit(cfg["lr"], cfg["steps"]).fit(Xtr, tr[cfg["label"]].values)
    f1 = f1_score(te[cfg["label"]].values, model.predict(Xte))
    out = os.path.join("runs", cfg["name"])
    os.makedirs(out, exist_ok=True)
    json.dump({"f1": f1, "n_train": len(tr), "n_test": len(te)}, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print("f1 = %.4f" % f1)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        print("usage: python -m meterlab.cli run EXP.json")
''',
    }


def repo_valnpz():
    return {
        "README.md": r'''# val-auc-lr

Ridge-regularised logistic regression on 20 tabular features. The organisers hold the
final test labels, so this repo only ever reports **validation** AUC; no test data is
present and no number here is a test estimate.

## Run

    python run_experiment.py --config conf/exp.json --seeds 10

Seed `s` (0..9) controls initial weights and minibatch order (`np.random.RandomState(s)`).
Output: `out/val_report.json`.

## Result

Validation AUC, mean over 10 seeds: **@CLAIM@** (std across seeds below 0.001).
The regularisation strength in `conf/exp.json` was set by hand and is not tuned per seed.
''',
        "conf/exp.json": json.dumps({"lr": 0.05, "epochs": 30, "batch": 32, "l2": 0.01, "init_scale": 0.1}, indent=2) + "\n",
        "run_experiment.py": r'''import argparse, json, os
import numpy as np
import pandas as pd


def auc(y, s):
    r = pd.Series(s).rank().values
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit(X, y, cfg, rs):
    w = rs.randn(X.shape[1]) * cfg["init_scale"]
    b = 0.0
    for _ in range(cfg["epochs"]):
        order = rs.permutation(len(y))
        for i in range(0, len(y), cfg["batch"]):
            j = order[i:i + cfg["batch"]]
            p = 1 / (1 + np.exp(-(X[j] @ w + b)))
            w -= cfg["lr"] * (X[j].T @ (p - y[j]) / len(j) + cfg["l2"] * w)
            b -= cfg["lr"] * np.mean(p - y[j])
    return w, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/exp.json")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    d = np.load("data/features.npz")
    Xt, yt, Xv, yv = d["X_train"], d["y_train"], d["X_val"], d["y_val"]
    mu, sd = Xt.mean(0), Xt.std(0)
    Xt, Xv = (Xt - mu) / sd, (Xv - mu) / sd
    per = []
    for s in range(args.seeds):
        w, b = fit(Xt, yt, cfg, np.random.RandomState(s))
        per.append(auc(yv, Xv @ w + b))
    os.makedirs("out", exist_ok=True)
    rep = {"metric": "val_auc", "val_auc_mean": float(np.mean(per)), "val_auc_std": float(np.std(per)), "per_seed": per}
    json.dump(rep, open("out/val_report.json", "w"), indent=1)
    print("val_auc mean %.4f std %.4f" % (rep["val_auc_mean"], rep["val_auc_std"]))


if __name__ == "__main__":
    main()
''',
    }


def repo_loan():
    return {
        "README.md": r'''# loan-default-baseline

Baseline default classifier on application records.

## Run

    python scripts/run_loan.py --seed 11 --test-frac 0.25

Writes `artifacts/metrics.json` and `artifacts/predictions.csv`.

## Result

Accuracy on the held-out 25%: **@CLAIM@**.

`prior_defaults` (count of the applicant's earlier defaults from the bureau file) is by far
the most important feature, which is why accuracy is high; it is known at application time.
The `app_id` column is only used to key the predictions file.
''',
        "scripts/run_loan.py": r'''import argparse, json, os
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
''',
    }


def repo_ads():
    return {
        "README.md": r'''# adclick-lr

Click prediction for display ads with logistic regression, using a per-campaign click-rate
feature next to placement, hour and bid.

## Run

    python main.py --csv data/clicks.csv --out reports/run.json

## Result

Test accuracy: **@CLAIM@** (25% random hold-out, seed 4).
''',
        "main.py": r'''import argparse, json, os
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
''',
    }


ADS_FIX = [(
    '''    df["campaign_rate"] = df.groupby("campaign")["clicked"].transform("mean")

    idx = np.random.RandomState(args.seed).permutation(len(df))
    n_te = len(df) // 4
    te, tr = df.iloc[idx[:n_te]], df.iloc[idx[n_te:]]
''',
    '''    idx = np.random.RandomState(args.seed).permutation(len(df))
    n_te = len(df) // 4
    te, tr = df.iloc[idx[:n_te]].copy(), df.iloc[idx[n_te:]].copy()
    prior = tr["clicked"].mean()
    g = tr.groupby("campaign")["clicked"].agg(["sum", "count"])
    enc = (g["sum"] + 5 * prior) / (g["count"] + 5)
    tr["campaign_rate"] = tr["campaign"].map(enc)
    te["campaign_rate"] = te["campaign"].map(enc).fillna(prior)
''')]


def repo_patients():
    return {
        "README.md": r'''# readmit-knn

k-nearest-neighbour baseline for 30-day readmission from vitals and labs. One row per
hospital visit.

## Run

    python experiments/run.py --config experiments/knn.json

Writes `results/<name>.json` where `name` is set in the config.

## Result

Test AUC: **@CLAIM@** (k=7, 25% test split, split seed 7).
''',
        "experiments/knn.json": json.dumps({"name": "knn_k7", "data": "data/visits.csv", "k": 7,
                                            "split": {"test_frac": 0.25, "seed": 7}}, indent=2) + "\n",
        "experiments/data_utils.py": r'''import numpy as np
import pandas as pd

FEATURES = ["age", "sbp", "dbp", "chol", "glucose", "bmi", "n_meds"]


def load(path):
    return pd.read_csv(path)


def train_test_split(df, test_frac, seed):
    idx = np.random.RandomState(seed).permutation(len(df))
    n_te = int(round(len(df) * test_frac))
    return df.iloc[idx[n_te:]], df.iloc[idx[:n_te]]
''',
        "experiments/run.py": r'''import argparse, json, os, sys
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
''',
    }


PAT_FIX = [(
    '''    idx = np.random.RandomState(seed).permutation(len(df))
    n_te = int(round(len(df) * test_frac))
    return df.iloc[idx[n_te:]], df.iloc[idx[:n_te]]
''',
    '''    pids = df["patient_id"].unique()
    pids = pids[np.random.RandomState(seed).permutation(len(pids))]
    held = set(pids[:int(round(len(pids) * test_frac))])
    m = df["patient_id"].isin(held)
    return df[~m], df[m]
''')]


def repo_plans():
    return {
        "README.md": r'''# churn-plans

Subscription churn model: one-hot plan and region, standardised tenure and monthly charge,
logistic regression.

## Run

    python train.py
    python eval.py

`train.py` fits on `data/train.csv` (80/20 internal split) and saves `model/model.npz`.
`eval.py` scores `data/test.csv` and writes `reports/test_metrics.json`.

## Result

Test accuracy (from `reports/test_metrics.json`): **@CLAIM@**.
''',
        "churnlib/__init__.py": "",
        "churnlib/features.py": r'''import numpy as np
import pandas as pd

CAT = ["plan", "region"]
NUM = ["tenure", "monthly"]


def encode(df, stats=None):
    parts = []
    for c in CAT:
        codes, uniques = pd.factorize(df[c])
        parts.append(np.eye(len(uniques))[codes])
    num = df[NUM].values.astype(float)
    if stats is None:
        stats = (num.mean(0), num.std(0) + 1e-9)
    parts.append((num - stats[0]) / stats[1])
    return np.hstack(parts), stats
''',
        "train.py": r'''import os
import numpy as np
import pandas as pd
from churnlib.features import encode


def main():
    df = pd.read_csv("data/train.csv")
    X, (mu, sd) = encode(df)
    y = df["churned"].values
    n = int(len(df) * 0.8)
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(500):
        p = 1 / (1 + np.exp(-(X[:n] @ w + b)))
        w -= 0.3 * X[:n].T @ (p - y[:n]) / n
        b -= 0.3 * np.mean(p - y[:n])
    hold = float(np.mean(((X[n:] @ w + b) > 0) == y[n:]))
    os.makedirs("model", exist_ok=True)
    np.savez("model/model.npz", w=w, b=b, mu=mu, sd=sd)
    print("holdout accuracy %.4f" % hold)


if __name__ == "__main__":
    main()
''',
        "eval.py": r'''import json, os
import numpy as np
import pandas as pd
from churnlib.features import encode


def main():
    m = np.load("model/model.npz")
    df = pd.read_csv("data/test.csv")
    X, _ = encode(df, (m["mu"], m["sd"]))
    acc = float(np.mean(((X @ m["w"] + float(m["b"])) > 0) == df["churned"].values))
    os.makedirs("reports", exist_ok=True)
    json.dump({"accuracy": acc, "n": len(df)}, open("reports/test_metrics.json", "w"))
    print("test accuracy %.4f" % acc)


if __name__ == "__main__":
    main()
''',
    }


PLANS_FIX = [("pd.factorize(df[c])", "pd.factorize(df[c], sort=True)")]


def repo_tickets():
    return {
        "README.md": r'''# ticket-escalation

Predict whether a support ticket is escalated (rare, about 1 in 5) from wait time, text length,
priority and channel.

## Run

    python run.py --config config.json

Outputs go to `out/`: `metrics.json` and `preds.csv` (true and predicted label per test ticket).

## Labels

Classes are indexed alphabetically (`escalated`=0, `resolved`=1).

## Result

F1 for the escalated class on the test split: **@CLAIM@**.
''',
        "config.json": json.dumps({"data": "data/tickets.csv", "label": "outcome", "pos_label": 1, "test_frac": 0.3,
                                   "seed": 3, "epochs": 500, "lr": 0.2, "out_dir": "out"}, indent=2) + "\n",
        "run.py": r'''import argparse, json, os
import numpy as np
import pandas as pd


def f1(y, p, pos):
    tp = np.sum((y == pos) & (p == pos))
    fp = np.sum((y != pos) & (p == pos))
    fn = np.sum((y == pos) & (p != pos))
    return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    cfg = json.load(open(ap.parse_args().config))
    df = pd.read_csv(cfg["data"])
    classes = sorted(df[cfg["label"]].unique())
    y = df[cfg["label"]].map({c: i for i, c in enumerate(classes)}).values

    X = np.column_stack([df["wait_min"], df["text_len"], df["priority"] == "high", df["priority"] == "low",
                         df["channel"] == "phone", df["channel"] == "chat"]).astype(float)
    idx = np.random.RandomState(cfg["seed"]).permutation(len(df))
    n_te = int(len(df) * cfg["test_frac"])
    te, tr = idx[:n_te], idx[n_te:]
    mu, sd = X[tr].mean(0), X[tr].std(0)
    X = (X - mu) / sd

    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(cfg["epochs"]):
        p = 1 / (1 + np.exp(-(X[tr] @ w + b)))
        w -= cfg["lr"] * X[tr].T @ (p - y[tr]) / len(tr)
        b -= cfg["lr"] * np.mean(p - y[tr])
    pred = ((X[te] @ w + b) > 0).astype(int)
    score = f1(y[te], pred, cfg["pos_label"])

    os.makedirs(cfg["out_dir"], exist_ok=True)
    json.dump({"f1": score, "n_test": int(len(te))}, open(os.path.join(cfg["out_dir"], "metrics.json"), "w"))
    pd.DataFrame({"ticket_id": df["ticket_id"].values[te], "outcome_true": [classes[i] for i in y[te]],
                  "outcome_pred": [classes[i] for i in pred]}).to_csv(os.path.join(cfg["out_dir"], "preds.csv"), index=False)
    print("f1 %.4f" % score)


if __name__ == "__main__":
    main()
''',
    }


def repo_ckpt():
    return {
        "README.md": r'''# lowdata-logreg

Logistic regression on a small, 50-feature tabular problem (110 training rows). The epoch
with the best validation accuracy is kept as the checkpoint.

## Run

    python scripts/train.py --config configs/run.cfg
    python scripts/predict.py --config configs/run.cfg

`train.py` writes `ckpt/best.npz` and `runs/latest/metrics.json`; `predict.py` reloads the
checkpoint and writes `runs/latest/reload_metrics.json`.

## Result

Test accuracy of the best-validation checkpoint: **@CLAIM@**
(`test_acc` in `runs/latest/metrics.json`).
''',
        "configs/run.cfg": "[train]\nlr = 0.4\nepochs = 400\nseed = 5\n\n[paths]\ndata_dir = data\nckpt = ckpt/best.npz\nmetrics = runs/latest/metrics.json\nreload_metrics = runs/latest/reload_metrics.json\n",
        "scripts/train.py": r'''import argparse, configparser, json, os
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
''',
        "scripts/predict.py": r'''import argparse, configparser, json, os
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run.cfg")
    cp = configparser.ConfigParser()
    cp.read(ap.parse_args().config)
    m = np.load(cp.get("paths", "ckpt"))
    df = pd.read_csv(os.path.join(cp.get("paths", "data_dir"), "test.csv"))
    X = (df.drop(columns="y").values - m["mu"]) / m["sd"]
    a = float(np.mean(((X @ m["w"] + float(m["b"])) > 0) == df["y"].values))
    json.dump({"test_acc": a}, open(cp.get("paths", "reload_metrics"), "w"))
    print("reloaded test acc %.4f" % a)


if __name__ == "__main__":
    main()
''',
    }


CKPT_FIX = [('"w": w, "b": b}', '"w": w.copy(), "b": b}')]


def repo_posts():
    return {
        "README.md": r'''# topic-nb

Multinomial naive Bayes topic classifier over a capped bag-of-words vocabulary (the 200 most
frequent tokens in the training split).

## Run

    python run_nb.py

Writes `metrics/nb.json`.

## Result

Test accuracy: **@CLAIM@** (80/20 split by document order).
''',
        "run_nb.py": r'''import json, os
from collections import Counter
import numpy as np
import pandas as pd

VOCAB_SIZE = 200


def main():
    df = pd.read_csv("data/posts.csv")
    n = int(len(df) * 0.8)
    tr, te = df.iloc[:n], df.iloc[n:]
    counts = Counter(t for doc in tr["text"] for t in doc.split())
    vocab = list({t for t, c in counts.items() if c >= 2})[:VOCAB_SIZE]
    col = {t: i for i, t in enumerate(vocab)}
    topics = sorted(df["topic"].unique())

    def bow(texts):
        M = np.zeros((len(texts), len(vocab)))
        for r, doc in enumerate(texts):
            for t in doc.split():
                if t in col:
                    M[r, col[t]] += 1
        return M

    Btr, Bte = bow(tr["text"]), bow(te["text"])
    logp = []
    for t in topics:
        c = Btr[(tr["topic"] == t).values].sum(0) + 1.0
        logp.append(np.log(c / c.sum()))
    prior = np.log(np.array([(tr["topic"] == t).mean() for t in topics]))
    pred = np.argmax(Bte @ np.array(logp).T + prior, axis=1)
    acc = float(np.mean(np.array(topics)[pred] == te["topic"].values))
    os.makedirs("metrics", exist_ok=True)
    json.dump({"accuracy": acc}, open("metrics/nb.json", "w"))
    print("accuracy %.4f" % acc)


if __name__ == "__main__":
    main()
''',
    }


POSTS_FIX = [("vocab = list({t for t, c in counts.items() if c >= 2})[:VOCAB_SIZE]",
              "vocab = sorted((t for t, c in counts.items() if c >= 2), key=lambda t: (-counts[t], t))[:VOCAB_SIZE]")]


def repo_wine():
    return {
        "README.md": r'''# wq-sgd

Minibatch-SGD logistic regression predicting whether a wine is rated "good" from 11
physico-chemical measurements.

## Run

    python -m wq.fit --train data/train.csv --test data/test.csv --report out/report.json

## Result

Test accuracy: **@CLAIM@** (`test_accuracy` in `out/report.json`; 120 test wines).
''',
        "wq/__init__.py": "",
        "wq/fit.py": r'''import argparse, json, os
import numpy as np
import pandas as pd


def accuracy(X, y, w, b):
    return float(np.mean(((X @ w + b) > 0) == y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--report", default="out/report.json")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=2)
    a = ap.parse_args()

    tr, te = pd.read_csv(a.train), pd.read_csv(a.test)
    feats = [c for c in tr.columns if c != "good"]
    mu, sd = tr[feats].values.mean(0), tr[feats].values.std(0)
    Xtr, Xte = (tr[feats].values - mu) / sd, (te[feats].values - mu) / sd
    ytr, yte = tr["good"].values, te["good"].values

    rs = np.random.RandomState(a.seed)
    w, b = np.zeros(len(feats)), 0.0
    best, best_ep = -1.0, -1
    for ep in range(a.epochs):
        order = rs.permutation(len(ytr))
        for i in range(0, len(order), 8):
            j = order[i:i + 8]
            p = 1 / (1 + np.exp(-(Xtr[j] @ w + b)))
            w -= a.lr * Xtr[j].T @ (p - ytr[j]) / len(j)
            b -= a.lr * float(np.mean(p - ytr[j]))
        s = accuracy(Xte, yte, w, b)
        if s > best:
            best, best_ep = s, ep

    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    json.dump({"best_epoch": best_ep, "test_accuracy": best}, open(a.report, "w"))
    print("epoch %d test accuracy %.4f" % (best_ep, best))


if __name__ == "__main__":
    main()
''',
    }


WINE_FIX = [
    ('    ytr, yte = tr["good"].values, te["good"].values\n',
     '    ytr, yte = tr["good"].values, te["good"].values\n    nv = len(ytr) // 5\n    Xva, yva, Xtr, ytr = Xtr[:nv], ytr[:nv], Xtr[nv:], ytr[nv:]\n'),
    ('    best, best_ep = -1.0, -1\n', '    best, best_ep, best_t = -1.0, -1, 0.0\n'),
    ('        s = accuracy(Xte, yte, w, b)\n        if s > best:\n            best, best_ep = s, ep\n',
     '        s = accuracy(Xva, yva, w, b)\n        if s > best:\n            best, best_ep, best_t = s, ep, accuracy(Xte, yte, w, b)\n'),
    ('"test_accuracy": best}', '"test_accuracy": best_t}'),
    ('(best_ep, best))', '(best_ep, best_t))'),
]


def repo_calls():
    return {
        "README.md": r'''# birdcall-softmax

Softmax regression that identifies a bird species (wren / thrush / finch) from five
spectral summary features of a field recording.

## Run

    python run.py config.json

Writes `outputs/summary.json` (headline accuracy) and `outputs/predictions.csv`
(one row per recording in the test split).

## Result

Accuracy on the test split: **@CLAIM@**.
''',
        "config.json": json.dumps({"csv": "data/calls.csv", "label": "species", "epochs": 400, "lr": 0.3, "outputs": "outputs"}, indent=2) + "\n",
        "run.py": r'''import json, os, sys
import numpy as np
import pandas as pd

FEATS = ["f_low", "f_mid", "f_high", "zcr", "duration"]


def usable(df):
    return df[df["snr_db"] >= 8.0]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    cfg = json.load(open(sys.argv[1]))
    df = pd.read_csv(cfg["csv"])
    classes = sorted(df[cfg["label"]].unique())
    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"]
    te_eval = usable(te)
    mu, sd = tr[FEATS].values.mean(0), tr[FEATS].values.std(0)

    def prep(d):
        return (d[FEATS].values - mu) / sd

    def onehot(d):
        return np.array([[c == v for c in classes] for v in d[cfg["label"]]], dtype=float)

    Xtr, Ytr = prep(tr), onehot(tr)
    W, b = np.zeros((len(FEATS), len(classes))), np.zeros(len(classes))
    for _ in range(cfg["epochs"]):
        P = softmax(Xtr @ W + b)
        W -= cfg["lr"] * Xtr.T @ (P - Ytr) / len(Xtr)
        b -= cfg["lr"] * (P - Ytr).mean(0)

    pred = np.array(classes)[np.argmax(prep(te) @ W + b, 1)]
    pred_eval = np.array(classes)[np.argmax(prep(te_eval) @ W + b, 1)]
    acc = float(np.mean(pred_eval == te_eval[cfg["label"]].values))

    out = cfg["outputs"]
    os.makedirs(out, exist_ok=True)
    json.dump({"accuracy": acc, "n_eval": int(len(te_eval))}, open(os.path.join(out, "summary.json"), "w"))
    pd.DataFrame({"clip_id": te["clip_id"].values, "species": pred}).to_csv(os.path.join(out, "predictions.csv"), index=False)
    print("accuracy %.4f" % acc)


if __name__ == "__main__":
    main()
''',
    }


CALLS_FIX = [("    te_eval = usable(te)\n", "    te_eval = te\n")]


# ================================================================ task table
def jk(path, key):
    return (path, key)


def probe_tickets(d):
    p = pd.read_csv(os.path.join(d, "out/preds.csv"))
    out = []
    for c in ("escalated", "resolved"):
        tp = ((p.outcome_true == c) & (p.outcome_pred == c)).sum()
        fp = ((p.outcome_true != c) & (p.outcome_pred == c)).sum()
        fn = ((p.outcome_true == c) & (p.outcome_pred != c)).sum()
        out.append("F1(%s) recomputed from out/preds.csv = %.3f" % (c, 2 * tp / max(1, 2 * tp + fp + fn)))
    return "; ".join(out)


def probe_calls(d):
    p = pd.read_csv(os.path.join(d, "outputs/predictions.csv"))
    df = pd.read_csv(os.path.join(d, "data/calls.csv"))
    m = p.merge(df[["clip_id", "species"]], on="clip_id", suffixes=("_pred", "_true"))
    return "accuracy recomputed over all %d rows of outputs/predictions.csv = %.3f" % (len(m), (m.species_pred == m.species_true).mean())


T = [
    dict(id="bearing_fault", seed=1, category="control", data=data_sensor, repo=repo_sensor,
         objective="Check whether this bearing-fault classifier really reaches the accuracy its README states, and tell me if anything makes that number untrustworthy.",
         expect=[], allow=[], cmds=["python3 train.py --config configs/default.json"], metric=("results/metrics.json", "test_accuracy"),
         why="Near-perfect accuracy is legitimate: the two regimes are 5 sigma apart on three of six channels (generator shifts them by 5.0), the scaler is fit on train only, train and test are separate generated files with no shared rows, and the README claim equals the printed value."),
    dict(id="meter_events", seed=2, category="control", data=data_meter, repo=repo_meter,
         objective="Reproduce the meter anomaly F1 from the README and look for evaluation problems in this repo.",
         expect=[], allow=[], cmds=["python3 -m meterlab.cli run exp.json"], metric=("runs/exp_a/summary.json", "f1"),
         why="The quantised logger data legitimately has many identical feature rows across train and test (documented in the README; about 1100 possible combinations for 3200 rows). Labels are stochastic given features, so duplicates are not label leakage; the scaler is fit on train only; the claim equals the printed value."),
    dict(id="val_auc_seeds", seed=3, category="control", data=data_valnpz, repo=repo_valnpz,
         objective="Our reproducibility reviewer wants to know whether the validation AUC reported for this logistic regression is honest and stable across runs.",
         expect=[], allow=[], cmds=["python3 run_experiment.py --config conf/exp.json --seeds 10"], metric=("out/val_report.json", "val_auc_mean"),
         why="Only validation AUC is reported and it is labelled val_auc in code, output and README (no test set exists in the repo). The number is a mean over 10 explicitly seeded runs (RandomState(s)), so it is exactly reproducible; hyperparameters are fixed in the config, not tuned on the validation labels at run time."),
    dict(id="loan_default", seed=4, category="control", data=data_loan, repo=repo_loan,
         objective="Audit this loan default baseline: is the high accuracy the README reports legitimate?",
         expect=[], allow=[], cmds=["python3 scripts/run_loan.py --seed 11 --test-frac 0.25"], metric=("artifacts/metrics.json", "accuracy"),
         why="prior_defaults is a legitimately strong, prediction-time feature (generator coefficient 3.2). app_id is excluded from the features; the split is a seeded random permutation with unique ids, scaler and one-hot vocabulary are built from train rows only; the claim equals the printed value."),
    dict(id="adclick_te", seed=5, category="data_leakage", data=data_ads, repo=repo_ads,
         fixes={"main.py": ADS_FIX}, cmds=["python3 main.py --csv data/clicks.csv --out reports/run.json"], metric=("reports/run.json", "test_accuracy"),
         loc=("main.py", 'df["campaign_rate"] = df.groupby'), claim_from="bug",
         objective="Our click model looks surprisingly good for such noisy data. Find out whether the reported accuracy is real.",
         expect=["data.target_leak"], allow=[],
         desc="The per-campaign click-rate feature is a mean of `clicked` over the whole dataset, computed before the train/test split, so every test row's own label (and those of other test rows) is baked into its feature; campaigns are mostly tiny so this nearly encodes the label"),
    dict(id="readmit_rows", seed=6, category="data_leakage", data=data_patients, repo=repo_patients,
         fixes={"experiments/data_utils.py": PAT_FIX}, cmds=["python3 experiments/run.py --config experiments/knn.json"], metric=("results/knn_k7.json", "auc"),
         loc=("experiments/data_utils.py", "idx = np.random.RandomState(seed).permutation(len(df))"), claim_from="bug",
         objective="We plan to deploy this readmission model on new patients. Does the offline AUC predict how it will do on patients it has never seen?",
         expect=["data.train_test_overlap"], allow=[],
         desc="The split permutes visit rows, so the same patient_id (with near-identical vitals across visits) is in both train and test; kNN effectively looks up the patient. The split should be grouped by patient_id"),
    dict(id="churn_plans", seed=7, category="broken_preprocessing", data=data_plans, repo=repo_plans,
         fixes={"churnlib/features.py": PLANS_FIX}, cmds=["python3 train.py", "python3 eval.py"], metric=("reports/test_metrics.json", "accuracy"),
         loc=("churnlib/features.py", "pd.factorize(df[c])"), claim_from="fix",
         objective="The churn model's holdout accuracy during training is much better than what the evaluation script gives on the test file. Figure out why.",
         expect=["pipeline.eval_path_mismatch"], allow=["repro.claim_not_met"],
         desc="One-hot columns are built from pd.factorize without sort, so category-to-column order is the order of first appearance in whichever file is encoded; train.csv and test.csv list plans and regions in different orders, so eval.py feeds the model permuted one-hot columns"),
    dict(id="ticket_f1", seed=8, category="incorrect_metric", data=data_tickets, repo=repo_tickets,
         fixes={"config.json": [('"pos_label": 1', '"pos_label": 0')]}, cmds=["python3 run.py --config config.json"], metric=("out/metrics.json", "f1"),
         loc=("config.json", '"pos_label"'), claim_from="bug", probes=[probe_tickets],
         objective="Verify the headline F1 for escalation prediction against the predictions the repo writes out.",
         expect=["eval.metric_mismatch"], allow=[],
         desc="F1 is computed for class index 1 (resolved, the majority) via pos_label=1, while the README and task are about the escalated class (index 0)"),
    dict(id="lowdata_ckpt", seed=9, category="checkpoint_failure", data=data_overfit, repo=repo_ckpt,
         fixes={"scripts/train.py": CKPT_FIX}, cmds=["python3 scripts/train.py --config configs/run.cfg", "python3 scripts/predict.py --config configs/run.cfg"],
         metric=("runs/latest/metrics.json", "test_acc"), eff_from_probe=True, loc=("scripts/train.py", '"w": w, "b": b}'), claim_from="bug",
         probes=[lambda d: "reload_metrics.json test_acc = %.3f" % json.load(open(os.path.join(d, "runs/latest/reload_metrics.json")))["test_acc"]],
         objective="Make sure the saved best checkpoint of this small logistic regression can be used for inference and gives the accuracy we publish.",
         expect=["ckpt.roundtrip_mismatch"], allow=[],
         desc="The best-validation 'checkpoint' stores a reference to the weight array that is later updated in place, so ckpt/best.npz holds the final-epoch weights (overfit) rather than the best-epoch weights while the recorded test accuracy is from the best epoch"),
    dict(id="topic_vocab", seed=10, category="reproducibility", data=data_posts, repo=repo_posts,
         fixes={"run_nb.py": POSTS_FIX}, cmds=["python3 run_nb.py"], metric=("metrics/nb.json", "accuracy"),
         loc=("run_nb.py", "vocab = list({t for t"), claim_from="fix", hashseeds=[1, 2, 3, 4, 5, 6],
         objective="Colleagues get different topic-classifier accuracies from the same command. Investigate whether this experiment is reproducible.",
         expect=["repro.nondeterministic"], allow=["repro.claim_not_met"],
         desc="The capped vocabulary is taken as list(set(...))[:200], and set order of str depends on the per-process hash seed (PYTHONHASHSEED), so a different subset of tokens is kept on each run; the README says the 200 most frequent tokens are used"),
    dict(id="wine_stop", seed=11, category="other", data=data_wine, repo=repo_wine,
         fixes={"wq/fit.py": WINE_FIX}, cmds=["python3 -m wq.fit --train data/train.csv --test data/test.csv --report out/report.json"],
         metric=("out/report.json", "test_accuracy"), loc=("wq/fit.py", "s = accuracy(Xte, yte, w, b)"), claim_from="bug",
         objective="Review this wine-quality trainer for anything that would make its reported test accuracy optimistic.",
         expect=["other:test_set_model_selection"], allow=["data.target_leak"],
         desc="The epoch is chosen (and the reported number taken) as the maximum test accuracy over epochs, i.e. model selection on the test labels; no validation split exists"),
    dict(id="birdcall_subset", seed=12, category="other", data=data_calls, repo=repo_calls,
         fixes={"run.py": CALLS_FIX}, cmds=["python3 run.py config.json"], metric=("outputs/summary.json", "accuracy"),
         loc=("run.py", "te_eval = usable(te)"), claim_from="bug", probes=[probe_calls],
         objective="The README says accuracy on the test split. Check that the evaluation really covers the test split and is not biased.",
         expect=["other:filtered_test_subset"], allow=["eval.metric_mismatch"],
         desc="The headline accuracy is computed only on test recordings with snr_db >= 8 (usable(te)); the noisiest recordings, which the model gets wrong most, are silently dropped from the test set while predictions.csv covers all test rows"),
]


# ================================================================ build / run
def build(t, fixed=False, claim="@CLAIM@"):
    rs = np.random.RandomState(1000 + t["seed"])
    files = {k: v.encode() for k, v in t["repo"]().items()}
    files.update(t["data"](rs))
    files["README.md"] = files["README.md"].replace(b"@CLAIM@", claim.encode())
    if fixed:
        for path, pairs in (t.get("fixes") or {}).items():
            s = files[path].decode()
            for old, new in pairs:
                s = sub(s, old, new)
            files[path] = s.encode()
    return files


def write_tree(root, files):
    for p, b in files.items():
        fp = os.path.join(root, p)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "wb") as f:
            f.write(b)


def run_in(t, files, scratch, hashseed=None):
    d = tempfile.mkdtemp(prefix="h2_", dir=scratch)
    write_tree(d, files)
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    if hashseed is not None:
        env["PYTHONHASHSEED"] = str(hashseed)
    lines = []
    for c in t["cmds"]:
        r = subprocess.run(c, shell=True, cwd=d, env=env, capture_output=True, text=True, timeout=120)
        if r.returncode:
            raise RuntimeError("%s: %s\n%s" % (t["id"], c, r.stderr))
        lines.append(r.stdout.strip().splitlines()[-1])
    f, k = t["metric"]
    val = json.load(open(os.path.join(d, f)))[k]
    extra = [p(d) for p in t.get("probes", [])]
    return d, val, lines, extra


def main():
    fixed_dir = None
    if "--fixed" in sys.argv:
        fixed_dir = os.path.abspath(sys.argv[sys.argv.index("--fixed") + 1])
    scratch = os.environ.get("H2_SCRATCH") or tempfile.mkdtemp(prefix="h2_scratch_")
    os.makedirs(scratch, exist_ok=True)
    tasks_dir = os.path.join(HERE, "tasks")
    if fixed_dir is None and os.path.isdir(tasks_dir):
        shutil.rmtree(tasks_dir)

    for t in T:
        has_fix = bool(t.get("fixes"))
        if fixed_dir:
            root = os.path.join(fixed_dir, t["id"])
            shutil.rmtree(root, ignore_errors=True)
            claim = None
        # claim calibration
        src_fixed = t.get("claim_from") == "fix"
        _, cv, _, _ = run_in(t, build(t, fixed=src_fixed), scratch, hashseed=0)
        claim = "%.3f" % cv
        if fixed_dir:
            write_tree(root, build(t, fixed=True, claim=claim))
            print("fixed copy", t["id"], claim)
            continue

        files = build(t, claim=claim)
        ev = []
        res = {}
        if t["category"] == "control":
            d, v, lines, extra = run_in(t, files, scratch)
            ev.append("In a scratch copy ran `%s`: %s = %.4f (README claims %s)." % (" && ".join(t["cmds"]), t["metric"][1], v, claim))
            ev.append("Stdout: " + " | ".join(lines))
            d2, v2, _, _ = run_in(t, files, scratch)
            ev.append("Second identical run: %.4f." % v2)
            ev.append("Why clean: " + t["why"])
            defect = "none"
        else:
            hs = t.get("hashseeds") or [None]
            bug = []
            for h in hs:
                d, v, lines, extra = run_in(t, files, scratch, hashseed=h)
                bug.append(v)
                if h is None or h == hs[0]:
                    first_lines, first_extra = lines, extra
            ffiles = build(t, fixed=True, claim=claim)
            fv = []
            for h in (hs[:2] if t.get("hashseeds") else [None]):
                d, v, flines, fextra = run_in(t, ffiles, scratch, hashseed=h)
                fv.append(v)
            fixed_v = fv[0]
            ev.append("Followed README in a scratch copy (`%s`); README claim = %s." % (" && ".join(t["cmds"]), claim))
            if t.get("hashseeds"):
                ev.append("Buggy %s over PYTHONHASHSEED=%s: %s (spread %.3f)." % (t["metric"][1], hs, ", ".join("%.3f" % x for x in bug), max(bug) - min(bug)))
                ev.append("Fixed copy, PYTHONHASHSEED=%s: %s." % (hs[:2], ", ".join("%.3f" % x for x in fv)))
                eff = "%.3f-%.3f across runs vs %.3f fixed (spread %.3f)" % (min(bug), max(bug), fixed_v, max(bug) - min(bug))
            else:
                ev.append("Buggy %s = %.4f; stdout: %s" % (t["metric"][1], bug[0], " | ".join(first_lines)))
                ev.append("Fixed copy %s = %.4f; stdout: %s" % (t["metric"][1], fixed_v, " | ".join(flines)))
                eff = "%s %.3f buggy vs %.3f fixed (delta %+.3f)" % (t["metric"][1], bug[0], fixed_v, bug[0] - fixed_v)
            if t.get("eff_from_probe"):
                eff = "headline %s %.3f in both copies, but after save/reload the buggy copy gives [%s] vs fixed [%s]" % (t["metric"][1], bug[0], first_extra[0], fextra[0])
            for e in first_extra:
                ev.append("Buggy probe: " + e)
            for e in fextra:
                ev.append("Fixed probe: " + e)
            path, snip = t["loc"]
            line = next(i + 1 for i, l in enumerate(files[path].decode().splitlines()) if snip in l)
            defect = "%s (%s:%d). Measured: %s." % (t["desc"], path, line, eff)

        root = os.path.join(tasks_dir, t["id"])
        write_tree(os.path.join(root, "repo"), files)
        tj = {"id": t["id"], "category": t["category"], "objective": t["objective"], "expect": t["expect"], "allow": t["allow"],
              "canaries": [], "defect": defect, "evidence": " ".join(ev)}
        with open(os.path.join(root, "task.json"), "w") as f:
            json.dump(tj, f, indent=2)
            f.write("\n")
        print(t["id"], t["category"], claim, "|", " ".join(ev)[:230])
    if not fixed_dir:
        shutil.rmtree(scratch, ignore_errors=True) if not os.environ.get("H2_SCRATCH") else None


if __name__ == "__main__":
    main()
