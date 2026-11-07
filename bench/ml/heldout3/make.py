#!/usr/bin/env python3
"""Deterministic generator for the heldout3 ML-debugging benchmark.

  python3 make.py              -> tasks/<id>/repo/ + tasks/<id>/task.json
  python3 make.py --fixed DIR  -> DIR/<id>/ bug-fixed copies (defective tasks only)
"""
import argparse, io, json, os, shutil, zipfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "meta.py")).read())  # CLAIMS, EVID, META (task ground truth text)


def csv(df):
    return df.to_csv(index=False, float_format="%.4f")


def npz_bytes(**arrs):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for k, v in arrs.items():
            b = io.BytesIO()
            np.lib.format.write_array(b, np.asarray(v), allow_pickle=False)
            z.writestr(zipfile.ZipInfo(k + ".npy", date_time=(2021, 1, 1, 0, 0, 0)), b.getvalue())
    return buf.getvalue()


def sub(src, old, new, fixed):
    if not fixed:
        return src
    assert old in src, old
    return src.replace(old, new)


def sig(z):
    return 1 / (1 + np.exp(-z))


AUC = '''def roc_auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    order = np.argsort(s, kind="mergesort"); ss = s[order]
    r = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
'''

YAML_READER = '''def load_cfg(path):
    cfg = {}
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line or ":" not in line:
            continue
        k, v = [s.strip() for s in line.split(":", 1)]
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        cfg[k] = v
    return cfg
'''


def readme(tid, text):
    c = CLAIMS.get(tid, ["TBD"] * 4)
    for i, v in enumerate(c):
        text = text.replace("@%d@" % (i + 1), v)
    return text


# ---------------------------------------------------------------- data

def d_blobs():
    rng = np.random.default_rng(1301)
    n = 100 + 100 + 400
    X = rng.normal(0, 1, (n, 10))
    z = 1.3 * X[:, 0] - 1.1 * X[:, 1] + 0.9 * X[:, 2]
    y = (z > 0).astype(int)
    flip = rng.random(n) < 0.18
    y = np.where(flip, 1 - y, y)
    df = pd.DataFrame(X, columns=["x%d" % i for i in range(10)])
    df["y"] = y
    df.insert(0, "split", ["train"] * 100 + ["val"] * 100 + ["test"] * 400)
    return csv(df)


def d_readings():
    rng = np.random.default_rng(2202)
    cls = ["ok", "warn", "fault"]
    mean = {"ok": [0, 0, 0, 0, 0], "warn": [0.9, 0.5, -0.3, 0.2, 0], "fault": [1.6, 1.2, 0.4, -0.5, 0.6]}
    rows = []
    for u in range(150):
        c = cls[rng.choice(3, p=[.5, .3, .2])]
        off = rng.normal(0, 0.35, 5)
        for _ in range(int(rng.integers(3, 8))):
            x = np.array(mean[c]) + off + rng.normal(0, 0.8, 5)
            row = ["U%03d" % u, "b%d" % int(rng.integers(1, 4))] + list(np.round(x, 4)) + [c]
            rows.append(row)
            if rng.random() < 0.15:
                rows.append(list(row))
    rng.shuffle(rows)
    return csv(pd.DataFrame(rows, columns=["unit_id", "batch", "x1", "x2", "x3", "x4", "x5", "state"]))


def d_visits():
    rng = np.random.default_rng(3303)
    n = 150
    trait = rng.random(n) < 0.5
    cen = rng.normal(0, 1, (n, 6))
    cen[:, 0] += 0.6 * (trait * 2 - 1)
    mixed = rng.random(n) < 0.7
    rows = []
    for s in range(n):
        for v in range(int(rng.integers(5, 10))):
            x = cen[s] + rng.normal(0, 0.45, 6)
            lab = trait[s] ^ (rng.random() < 0.1)
            fmt = "s%04d" if (mixed[s] and rng.random() < 0.5) else "S-%04d"
            rows.append([fmt % s, "2023-%02d-%02d" % (v + 1, int(rng.integers(1, 28)))] + list(np.round(x, 4)) + ["improved" if lab else "stable"])
    rng.shuffle(rows)
    return csv(pd.DataFrame(rows, columns=["subject", "visit_date"] + ["f%d" % i for i in range(6)] + ["outcome"]))


def d_tickets():
    rng = np.random.default_rng(4404)
    n = 1600
    c = rng.choice(3, n, p=[0.52, 0.40, 0.08])
    mean = np.array([[0, 0, 0, 0], [1.4, -1.0, 0.3, 0], [0.3, 0.2, 0.4, 0.1]])
    X = mean[c] + rng.normal(0, 1, (n, 4)) * np.array([[1, 1, 1, 1]])
    names = np.array(["refund", "login", "abuse"])
    df = pd.DataFrame(X, columns=["len_z", "caps_z", "links_z", "age_z"])
    df.insert(0, "ticket_id", ["T%05d" % (7000 + i) for i in range(n)])
    df["category"] = names[c]
    return csv(df)


def d_site(n, shift, seed, pid0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.35).astype(int)
    L = rng.normal(0, 1, (n, 6))
    L[:, 0] += 1.1 * y; L[:, 1] += 0.8 * y; L[:, 2] -= 0.7 * y; L[:, 3] += 0.4 * y
    if shift:
        L[:, 0] += 1.0; L[:, 1] = L[:, 1] * 1.4 + 0.3; L[:, 2] += 0.9
    df = pd.DataFrame(L, columns=["lab_%d" % i for i in range(1, 7)])
    df.insert(0, "age", np.round(rng.normal(62, 11, n)))
    df.insert(0, "patient_id", ["P%05d" % (pid0 + i) for i in range(n)])
    df["readmit_30d"] = np.where(y == 1, "high", "low")
    return csv(df)


def d_loans():
    rng = np.random.default_rng(6606)
    n, nb = 800, 200
    br = rng.integers(0, nb, n)
    income = rng.normal(0, 1, n); tenure = rng.normal(0, 1, n)
    y = rng.random(n) < sig(-0.4 - 0.35 * income + 0.15 * tenure)
    df = pd.DataFrame({"loan_id": ["L%06d" % (90000 + i) for i in range(n)], "branch": ["BR-%03d" % b for b in br],
                       "income_k": np.round(55 + 12 * income, 2), "tenure_y": np.round(6 + 3 * tenure, 2),
                       "default": np.where(y, "yes", "no")})
    return csv(df)


def d_diag():
    rng = np.random.default_rng(7707)
    n = 700
    y = rng.random(n) < 0.3
    X = rng.normal(0, 1, (n, 8))
    X[:, 0] += 1.3 * y; X[:, 1] -= 1.0 * y; X[:, 2] += 0.8 * y; X[:, 3] += 0.5 * y
    df = pd.DataFrame(X, columns=["m%d" % i for i in range(1, 9)])
    df.insert(0, "sample_id", ["D%04d" % i for i in range(n)])
    df["status"] = np.where(y, "positive", "negative")
    return csv(df)


def d_trials():
    rng = np.random.default_rng(8808)
    flips = [0.02, 0.05, 0.08, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.03, 0.18]
    rows = []
    for k, f in enumerate(flips):
        X = rng.normal(0, 1, (70, 4))
        z = X[:, 0] + 0.5 * X[:, 1] - 0.5 * X[:, 2] + 0.2 * (k % 3 - 1)
        y = (z > 0) ^ (rng.random(70) < f)
        for i in range(70):
            rows.append(["site_%02d" % k] + list(np.round(X[i], 4)) + ["pos" if y[i] else "neg"])
    rng.shuffle(rows)
    return csv(pd.DataFrame(rows, columns=["site", "x0", "x1", "x2", "x3", "arm"]))


def d_margin(n, seed):
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        p = rng.uniform(-3, 3, 2)
        if abs(p[0] + p[1]) >= 0.5:
            pts.append(p)
    P = np.array(pts)
    df = pd.DataFrame(P, columns=["u", "v"])
    df.insert(0, "pt_id", ["%s%04d" % ("a" if seed % 2 else "b", i) for i in range(n)])
    df["cls"] = np.where(P.sum(1) > 0, "A", "B")
    return csv(df)


def d_expr():
    rng = np.random.default_rng(1010)
    n, d = 300, 500
    y = (rng.random(n) < 0.5).astype(int)
    X = rng.normal(0, 1, (n, d))
    X[:, :10] += 0.2 * (y[:, None] * 2 - 1)
    return npz_bytes(X=np.round(X, 4), y=y)


def d_devices(n, seed, order):
    rng = np.random.default_rng(seed)
    regions = np.array(["north", "south", "east", "west"])
    eff = {"north": 1.6, "east": 0.5, "south": -0.5, "west": -1.6}
    reg = rng.choice(regions, n)
    X = rng.normal(0, 1, (n, 3))
    z = 0.9 * X[:, 0] - 0.6 * X[:, 1] + 0.3 * X[:, 2] + np.array([eff[r] for r in reg])
    y = rng.random(n) < sig(z)
    if order:  # make first appearance order of regions fixed
        for i, r in enumerate(order):
            reg[i] = r
    df = pd.DataFrame(X, columns=["temp_z", "load_z", "age_z"])
    df.insert(0, "device_id", ["DV%05d" % (seed * 1000 + i) for i in range(n)])
    df.insert(1, "region", reg)
    df["status"] = np.where(y, "fail", "ok")
    return csv(df)


def d_clicks():
    rng = np.random.default_rng(1212)
    n = 1500
    X = rng.normal(0, 1, (n, 8))
    z = -1.4 + 1.5 * X[:, 0] + 1.0 * X[:, 1] - 1.0 * X[:, 2] + 0.7 * X[:, 3]
    y = rng.random(n) < sig(z * 1.5)
    df = pd.DataFrame(X, columns=["c%d" % i for i in range(8)])
    df["clicked"] = y.astype(int)
    return csv(df)


def d_housing():
    rng = np.random.default_rng(1313)
    n = 500
    reg = rng.choice(["coast", "inland", "hills", "metro"], n)
    off = {"coast": 0.6, "inland": 0.0, "hills": 0.3, "metro": 0.9}
    sq = rng.normal(0, 1, n); age = rng.normal(0, 1, n); rooms = rng.normal(0, 1, n); dist = rng.normal(0, 1, n)
    price = 2.0 + 0.9 * sq - 0.4 * age + 0.3 * rooms * sq - 0.5 * np.tanh(dist) + np.array([off[r] for r in reg]) + rng.normal(0, 0.4, n)
    df = pd.DataFrame({"listing": ["H%05d" % (500 + i) for i in range(n)], "region": reg, "sqft_z": sq, "age_z": age,
                       "rooms_z": rooms, "dist_z": dist, "price": price})
    return csv(df)


def d_events():
    rng = np.random.default_rng(1414)
    rows = []
    for src, n in (("A", 340), ("B", 60)):
        ts = np.sort(rng.uniform(0, 1000, n))
        for t in ts:
            rows.append((t, src))
    out = []
    for t, src in rows:
        f = rng.normal(0, 1, 4)
        period = int((t // 200) % 2)
        z = 1.6 * f[0] - 1.2 * f[1] + 1.4 * (period * 2 - 1)
        lab = rng.random() < sig(z)
        out.append([round(t, 4), src, period] + list(np.round(f, 4)) + ["churn" if lab else "stay"])
    return csv(pd.DataFrame(out, columns=["ts", "logger", "period", "f1", "f2", "f3", "f4", "outcome"]))


# ---------------------------------------------------------------- tasks
# Each builder returns {relpath: str|bytes}. IDs are neutral; see TASKS at the bottom.

def t_ckpt(fixed):  # h3_01
    cfg = "name: mlp_h48\ndata: data/blobs.csv\nhidden: 48\nlr: 0.4\nwd: 0.0\nepochs: 1500\n"
    mk = "PY ?= python\n\ntrain:\n\t$(PY) train.py --config configs/mlp.yaml --seed 0\n\neval:\n\t$(PY) eval.py --config configs/mlp.yaml --seed 0\n\nall: train eval\n"
    net = '''import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class MLP:
    def __init__(self, d_in, d_h, rng):
        self.params = [rng.normal(0, 1 / np.sqrt(d_in), (d_in, d_h)), np.zeros(d_h),
                       rng.normal(0, 1 / np.sqrt(d_h), (d_h, 1)), np.zeros(1)]

    @staticmethod
    def run(params, X):
        W1, b1, W2, b2 = params
        h = np.tanh(X @ W1 + b1)
        return h, (h @ W2 + b2).ravel()

    def predict_proba(self, X):
        return _sigmoid(self.run(self.params, X)[1])

    def step(self, X, y, lr, wd):
        W1, b1, W2, b2 = self.params
        h, z = self.run(self.params, X)
        dz = (_sigmoid(z) - y)[:, None] / len(y)
        dh = (dz @ W2.T) * (1 - h ** 2)
        grads = [X.T @ dh, dh.sum(0), h.T @ dz, dz.sum(0)]
        for P, G in zip(self.params, grads):
            P -= lr * (G + wd * P)
'''
    utils = YAML_READER + '''

def accuracy(y, pred):
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def load_split(path):
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c.startswith("x")]
    out = []
    for s in ("train", "val", "test"):
        d = df[df["split"] == s]
        out += [d[cols].to_numpy(float), d["y"].to_numpy(float)]
    return out


def save_ckpt(path, params, mu, sd):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, W1=params[0], b1=params[1], W2=params[2], b2=params[3], mu=mu, sd=sd)


def load_ckpt(path):
    z = np.load(path)
    return [z["W1"], z["b1"], z["W2"], z["b2"]], z["mu"], z["sd"]
'''
    utils = "import os\nimport numpy as np\nimport pandas as pd\n\n\n" + utils
    train = '''import argparse, json, os
import numpy as np
from mlpkit.net import MLP
from mlpkit.utils import load_cfg, accuracy, load_split, save_ckpt

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/mlp.yaml")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
cfg = load_cfg(args.config)

Xtr, ytr, Xva, yva, Xte, yte = load_split(cfg["data"])
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
Xtr, Xva, Xte = [(a - mu) / sd for a in (Xtr, Xva, Xte)]

rng = np.random.default_rng(args.seed)
net = MLP(Xtr.shape[1], cfg["hidden"], rng)
best_val, best, test_at_best, best_ep = -1.0, None, None, -1
for ep in range(cfg["epochs"]):
    net.step(Xtr, ytr, cfg["lr"], cfg["wd"])
    v = accuracy(yva, net.predict_proba(Xva) > 0.5)
    if v > best_val:
        best_val, best_ep = v, ep
        best = list(net.params)
        test_at_best = accuracy(yte, net.predict_proba(Xte) > 0.5)

run_dir = os.path.join("runs", cfg["name"], "seed%d" % args.seed)
save_ckpt(os.path.join(run_dir, "best.npz"), best, mu, sd)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({"best_val_acc": best_val, "test_acc": test_at_best, "best_epoch": best_ep}, f, indent=1)
print("best_val_acc=%.4f test_acc=%.4f (epoch %d)" % (best_val, test_at_best, best_ep))
'''
    train = sub(train, "best = list(net.params)", "best = [p.copy() for p in net.params]", fixed)
    ev = '''import argparse, json, os
import numpy as np
from mlpkit.net import MLP, _sigmoid
from mlpkit.utils import load_cfg, accuracy, load_split, load_ckpt

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/mlp.yaml")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
cfg = load_cfg(args.config)

_, _, _, _, Xte, yte = load_split(cfg["data"])
run_dir = os.path.join("runs", cfg["name"], "seed%d" % args.seed)
params, mu, sd = load_ckpt(os.path.join(run_dir, "best.npz"))
p = _sigmoid(MLP.run(params, (Xte - mu) / sd)[1])
res = {"test_acc": accuracy(yte, p > 0.5)}
json.dump(res, open(os.path.join(run_dir, "eval_metrics.json"), "w"), indent=1)
print("reloaded checkpoint test_acc=%.4f" % res["test_acc"])
'''
    rd = '''# mlpkit: small MLP on a noisy tabular benchmark

Plain-numpy one-hidden-layer network trained with full-batch gradient descent. Model selection is by
validation accuracy; the checkpoint with the best validation accuracy is written to
`runs/<name>/seed<k>/best.npz`.

## Run

    make train      # trains, writes runs/mlp_h48/seed0/metrics.json
    make eval       # reloads best.npz and rescores the test split -> eval_metrics.json

Hyper-parameters are in `configs/mlp.yaml`. Data (`data/blobs.csv`) has a `split` column (train/val/test).

## Result

Test accuracy at the best-validation epoch: **@1@** (seed 0, see `metrics.json`).
`make eval` rescoring of the saved checkpoint should reproduce it.
'''
    return {"README.md": readme("h3_01", rd), "Makefile": mk, "configs/mlp.yaml": cfg, "data/blobs.csv": d_blobs(),
            "mlpkit/__init__.py": "", "mlpkit/net.py": net, "mlpkit/utils.py": utils, "train.py": train, "eval.py": ev}


def t_dupes(fixed):  # h3_02 control
    data = '''import numpy as np
import pandas as pd

FEATS = ["x1", "x2", "x3", "x4", "x5"]


def load(path):
    return pd.read_csv(path)


def split_units(df, seed, frac=0.25):
    units = np.array(sorted(df.unit_id.unique()))
    np.random.default_rng(seed).shuffle(units)
    held = set(units[: int(len(units) * frac)])
    m = df.unit_id.isin(held).to_numpy()
    return df[~m].reset_index(drop=True), df[m].reset_index(drop=True)


def xy(df, classes):
    return df[FEATS].to_numpy(float), df.state.map({c: i for i, c in enumerate(classes)}).to_numpy()
'''
    model = '''import numpy as np


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def fit(X, y, k, lr, epochs, l2=1e-3):
    W = np.zeros((X.shape[1], k)); b = np.zeros(k)
    Y = np.eye(k)[y]
    for _ in range(epochs):
        G = softmax(X @ W + b) - Y
        W -= lr * (X.T @ G / len(y) + l2 * W)
        b -= lr * G.mean(0)
    return W, b


def predict(W, b, X):
    return (X @ W + b).argmax(1)


def f1_macro(y, p, k):
    out = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(out))
'''
    train = '''import argparse, json, os
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
'''
    rd = '''# sensorclf

Condition classifier (ok / warn / fault) for rotating-equipment units from vibration summary statistics.

## Data

`data/readings.csv`: one row per logger reading. Several readings per unit. The logger occasionally writes a
reading twice, so some rows are exact duplicates; we leave them in because the deployed pipeline sees the same
stream. Train/test are split by `unit_id` (whole units are held out).

## Run

    python -m sensorclf.train --data data/readings.csv --out runs --seed 1 --lr 0.1 --epochs 400

Writes `runs/softmax_lr0.1/seed1/metrics.json`.

## Result

Held-out units: f1_macro **@1@**, accuracy **@2@**.
'''
    return {"README.md": readme("h3_02", rd), "data/readings.csv": d_readings(), "sensorclf/__init__.py": "",
            "sensorclf/data.py": data, "sensorclf/model.py": model, "sensorclf/train.py": train}


def t_subject(fixed):  # h3_03
    ini = "[data]\npath = data/visits.csv\n\n[model]\nk = 5\n\n[split]\nbuckets = 5\ntest_bucket = 0\n"
    split = '''import zlib


def canon(sid):
    return sid.lower().replace("-", "")


def bucket(sid, n):
    return zlib.crc32(sid.encode()) % n


def subject_split(df, n_buckets, test_bucket=0):
    b = df["subject"].map(lambda s: bucket(s, n_buckets))
    m = (b == test_bucket).to_numpy()
    return df[~m].reset_index(drop=True), df[m].reset_index(drop=True)
'''
    split = sub(split, "bucket(s, n_buckets)", "bucket(canon(s), n_buckets)", fixed)
    knn = '''import numpy as np


def predict(Xtr, ytr, Xte, k):
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    nn = np.argsort(d, axis=1)[:, :k]
    return (ytr[nn].mean(1) > 0.5).astype(int)
'''
    run = '''import argparse, configparser, json, os
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
'''
    rd = '''# visits

k-nearest-neighbour baseline for predicting clinical outcome (improved / stable) from six visit measurements.

Records come from the clinic visit log (`data/visits.csv`, several visits per subject). Evaluation is on a
held-out set of subjects chosen by hashing the subject id into buckets (`[split]` in `params.ini`).

## Run

    python -m visits.run --ini params.ini --out out

Writes `out/knn_k5/metrics.json` and `out/knn_k5/preds.csv`.

## Result

Held-out subjects, k=5: accuracy **@1@**.
'''
    return {"README.md": readme("h3_03", rd), "params.ini": ini, "data/visits.csv": d_visits(), "visits/__init__.py": "",
            "visits/split.py": split, "visits/knn.py": knn, "visits/run.py": run}


def t_f1(fixed):  # h3_04
    train = '''import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/tickets.csv")
ap.add_argument("--out", default="out")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

import os
os.makedirs(a.out, exist_ok=True)
df = pd.read_csv(a.data)
labels = ["refund", "login", "abuse"]
FE = [c for c in df.columns if c.endswith("_z")]
rng = np.random.default_rng(a.seed)
idx = rng.permutation(len(df)); cut = int(0.7 * len(df))
tr, te = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
X = tr[FE].to_numpy(float); y = tr.category.map({c: i for i, c in enumerate(labels)}).to_numpy()
mu, sd = X.mean(0), X.std(0)
X = (X - mu) / sd
W = np.zeros((X.shape[1], 3)); b = np.zeros(3); Y = np.eye(3)[y]
for _ in range(500):
    Z = X @ W + b; Z -= Z.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
    G = P - Y
    W -= 0.2 * X.T @ G / len(y); b -= 0.2 * G.mean(0)
Xt = (te[FE].to_numpy(float) - mu) / sd
pred = (Xt @ W + b).argmax(1)
pd.DataFrame({"ticket_id": te.ticket_id.values, "label": te.category.values,
              "pred": np.array(labels)[pred]}).to_csv(os.path.join(a.out, "preds.csv"), index=False)
print("wrote", len(te), "predictions")
'''
    score = '''import argparse, json
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
'''
    score = sub(score, "if (tp + fp) else np.nan", "if (tp + fp) else 0.0", fixed)
    sh = "#!/bin/bash\nset -e\nmkdir -p out\npython train.py --data data/tickets.csv --out out\npython score.py out/preds.csv --out out/scores.json\n"
    rd = '''# ticket-triage

Softmax-regression baseline that routes support tickets to refund / login / abuse queues from four normalised
text-statistics features. Abuse is rare (under 10% of tickets).

## Run

    bash run.sh

`train.py` fits the model and writes `out/preds.csv`; `score.py` scores the predictions file into
`out/scores.json`.

## Result

f1_macro **@1@** (accuracy @2@) on the 30% held-out split.
'''
    return {"README.md": readme("h3_04", rd), "run.sh": sh, "train.py": train, "score.py": score, "data/tickets.csv": d_tickets()}


def t_shift(fixed):  # h3_05 control
    nb = '''#!/usr/bin/env python
# coding: utf-8

# # Readmission risk: site A -> site B
# Exported from `analysis.ipynb` (jupyter nbconvert --to script)

# In[1]:


import json, os
import numpy as np
import pandas as pd

cfg = json.load(open("config.json"))
rng = np.random.default_rng(cfg["seed"])


# In[2]:


a = pd.read_csv("data/site_a.csv")
b = pd.read_csv("data/site_b.csv")
FEATS = ["age"] + [c for c in a.columns if c.startswith("lab_")]
def xy(d):
    return d[FEATS].to_numpy(float), (d.readmit_30d == "high").to_numpy(float)


# In[3]:


perm = rng.permutation(len(a))
n_val = len(a) // 4
va_idx, tr_idx = perm[:n_val], perm[n_val:]
Xa, ya = xy(a)
Xtr, ytr, Xva, yva = Xa[tr_idx], ya[tr_idx], Xa[va_idx], ya[va_idx]
Xb, yb = xy(b)
mu, sd = Xtr.mean(0), Xtr.std(0)
Xtr, Xva, Xb = [(m - mu) / sd for m in (Xtr, Xva, Xb)]


# In[4]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

w = np.zeros(Xtr.shape[1]); bias = 0.0
for _ in range(cfg["epochs"]):
    g = sigmoid(Xtr @ w + bias) - ytr
    w -= cfg["lr"] * (Xtr.T @ g / len(ytr) + cfg["l2"] * w)
    bias -= cfg["lr"] * g.mean()


# In[5]:


''' + AUC + '''

def score(X, y):
    s = X @ w + bias
    return float(np.mean((s > 0) == (y == 1))), roc_auc(y, s)

va_acc, va_auc = score(Xva, yva)
b_acc, b_auc = score(Xb, yb)
res = {"val_acc": round(va_acc, 4), "val_auc": round(va_auc, 4), "siteB_acc": round(b_acc, 4), "siteB_auc": round(b_auc, 4),
       "n_train": len(tr_idx), "n_val": n_val, "n_siteB": len(b)}
os.makedirs("out/nb_run", exist_ok=True)
json.dump(res, open("out/nb_run/metrics.json", "w"), indent=1)
print(res)
'''
    rd = '''# readmit-transfer

Does a 30-day readmission model trained at hospital A transfer to hospital B?

* `data/site_a.csv` - hospital A, used for training (75%) and in-site validation (25%, random patients).
* `data/site_b.csv` - hospital B, used only for evaluation. **Site B is a different hospital with a different lab
  analyser (shifted and rescaled lab_1, lab_2, lab_3) and we expect, and report, a drop relative to site A.**
  No site B data is used for fitting or for standardisation.

## Run

    python analysis_nb.py        # reads config.json, writes out/nb_run/metrics.json

## Result

In-site validation (site A): acc **@1@**, auc **@2@**.
Site B (shifted): acc **@3@**, auc **@4@**.
'''
    cfgj = json.dumps({"seed": 3, "lr": 0.2, "epochs": 300, "l2": 0.01}, indent=1) + "\n"
    return {"README.md": readme("h3_05", rd), "config.json": cfgj, "analysis_nb.py": nb,
            "data/site_a.csv": d_site(600, False, 5051, 0), "data/site_b.csv": d_site(400, True, 5052, 20000)}


def t_te(fixed):  # h3_06
    cfg = "name: lr_branch\ndata: data/loans.csv\nseed: 5\ntest_frac: 0.25\nsmooth: 1.0\nlr: 0.3\nepochs: 400\n"
    mk = "train:\n\tpython -m credit.train --config configs/lr.yaml\n"
    feats = '''import numpy as np


def fit_te(keys, y, m):
    prior = float(np.mean(y)); s, n = {}, {}
    for k, v in zip(keys, y):
        s[k] = s.get(k, 0.0) + v
        n[k] = n.get(k, 0) + 1
    return {k: (s[k] + m * prior) / (n[k] + m) for k in s}, prior


def apply_te(keys, table, prior):
    return np.array([table.get(k, prior) for k in keys])
'''
    train = "import argparse, json, os\nimport numpy as np\nimport pandas as pd\nfrom .features import fit_te, apply_te\n\n\n" + YAML_READER + "\n\n" + AUC + '''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lr.yaml")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    df = pd.read_csv(cfg["data"])
    y = (df["default"] == "yes").to_numpy(float)
    idx = np.random.default_rng(cfg["seed"]).permutation(len(df))
    nt = int(len(df) * cfg["test_frac"])
    te_i, tr_i = idx[:nt], idx[nt:]
    br = df.branch.to_numpy()
    table, prior = fit_te(br, y, cfg["smooth"])
    X = np.c_[df.income_k.to_numpy(float), df.tenure_y.to_numpy(float), apply_te(br, table, prior)]
    mu, sd = X[tr_i].mean(0), X[tr_i].std(0)
    X = (X - mu) / sd
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(cfg["epochs"]):
        g = 1 / (1 + np.exp(-(X[tr_i] @ w + b))) - y[tr_i]
        w -= cfg["lr"] * X[tr_i].T @ g / len(tr_i); b -= cfg["lr"] * g.mean()
    s = X[te_i] @ w + b
    res = {"auc": round(roc_auc(y[te_i], s), 4), "accuracy": round(float(np.mean((s > 0) == (y[te_i] == 1))), 4),
           "n_test": int(nt)}
    d = os.path.join("out", cfg["name"])
    os.makedirs(d, exist_ok=True)
    json.dump(res, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    print(res)


if __name__ == "__main__":
    main()
'''
    train = sub(train, "table, prior = fit_te(br, y, cfg[\"smooth\"])", "table, prior = fit_te(br[tr_i], y[tr_i], cfg[\"smooth\"])", fixed)
    rd = '''# credit-lr

Logistic-regression default model using income, tenure and a smoothed per-branch default rate (the branch column
has ~200 levels, too many for one-hot, so it is encoded by its smoothed mean default rate, `smooth` in the config).

## Run

    make train

Writes `out/lr_branch/metrics.json`. Test set is a random 25% of loans (`seed` in `configs/lr.yaml`).

## Result

Test auc **@1@**, accuracy @2@.
'''
    return {"README.md": readme("h3_06", rd), "Makefile": mk, "configs/lr.yaml": cfg, "data/loans.csv": d_loans(),
            "credit/__init__.py": "", "credit/features.py": feats, "credit/train.py": train}


def t_l2(fixed):  # h3_07
    fit = "import argparse, json, os\nimport numpy as np\nimport pandas as pd\n\n\n" + '''def logloss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/diag.csv")
ap.add_argument("--l2", type=float, default=0.002)
ap.add_argument("--lr", type=float, default=0.5)
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--seed", type=int, default=4)
ap.add_argument("--out", default="results")
a = ap.parse_args()

df = pd.read_csv(a.data)
FE = [c for c in df.columns if c.startswith("m")]
y = (df.status == "positive").to_numpy(float)
idx = np.random.default_rng(a.seed).permutation(len(df))
cut = int(0.7 * len(df))
tr, te = idx[:cut], idx[cut:]
X = df[FE].to_numpy(float)
mu, sd = X[tr].mean(0), X[tr].std(0)
X = np.c_[(X - mu) / sd, np.ones(len(X))]

w = np.zeros(X.shape[1]); n = len(tr)
for _ in range(a.epochs):
    p = 1 / (1 + np.exp(-X[tr] @ w))
    g = X[tr].T @ (p - y[tr]) / n + a.l2 * n * w
    w -= a.lr * g
pt = 1 / (1 + np.exp(-X[te] @ w))
res = {"accuracy": round(float(np.mean((pt > 0.5) == (y[te] == 1))), 4), "logloss": round(logloss(y[te], pt), 4),
       "n_test": len(te)}
d = os.path.join(a.out, "l2_%g" % a.l2)
os.makedirs(d, exist_ok=True)
json.dump(res, open(os.path.join(d, "metrics.json"), "w"), indent=1)
print(res)
'''
    fit = sub(fit, "a.l2 * n * w", "a.l2 * w", fixed)
    rd = '''# diag-logreg

L2-regularised logistic regression (full-batch gradient descent) on eight assay measurements for a binary
diagnosis. Class balance is about 30% positive.

## Run

    python fit.py --data data/diag.csv --l2 0.002 --lr 0.5 --epochs 300 --seed 4

Writes `results/l2_0.002/metrics.json`. 70/30 random split.

## Result

Test accuracy **@1@**, logloss **@2@**.
'''
    return {"README.md": readme("h3_07", rd), "fit.py": fit, "data/diag.csv": d_diag()}


def t_hash(fixed):  # h3_08
    cfgj = json.dumps({"data": "data/trials.csv", "seed": 11, "n_test_sites": 4, "lr": 0.3, "epochs": 200,
                       "out": "results/trial_a"}, indent=2) + "\n"
    py = '''import json, os, sys
import numpy as np
import pandas as pd

cfg = json.load(open(sys.argv[1]))
df = pd.read_csv(cfg["data"])
rng = np.random.default_rng(cfg["seed"])

order = list(set(df.site))
rng.shuffle(order)
test_sites = order[: cfg["n_test_sites"]]
m = df.site.isin(test_sites).to_numpy()

FE = ["x0", "x1", "x2", "x3"]
X = df[FE].to_numpy(float); y = (df.arm == "pos").to_numpy(float)
mu, sd = X[~m].mean(0), X[~m].std(0)
X = (X - mu) / sd
w = np.zeros(4); b = 0.0
for _ in range(cfg["epochs"]):
    g = 1 / (1 + np.exp(-(X[~m] @ w + b))) - y[~m]
    w -= cfg["lr"] * X[~m].T @ g / (~m).sum(); b -= cfg["lr"] * g.mean()
pred = (X[m] @ w + b) > 0
res = {"accuracy": round(float(np.mean(pred == (y[m] == 1))), 4), "test_sites": sorted(test_sites), "n_test": int(m.sum())}
os.makedirs(cfg["out"], exist_ok=True)
json.dump(res, open(os.path.join(cfg["out"], "metrics.json"), "w"), indent=1)
print(res)
'''
    py = sub(py, "order = list(set(df.site))", "order = sorted(set(df.site))", fixed)
    sh = "#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(dirname \"$0\")\"\npython run_trial.py config.json\n"
    rd = '''# multisite-trial

Predicts treatment-arm response from four covariates across 12 trial sites. Evaluation holds out whole sites
(4 of 12, drawn with the seeded generator from `config.json`) and fits a logistic regression on the rest.

## Run

    bash run.sh

Writes `results/trial_a/metrics.json`.

## Result

Accuracy on held-out sites: **@1@**.
'''
    return {"README.md": readme("h3_08", rd), "run.sh": sh, "config.json": cfgj, "run_trial.py": py, "data/trials.csv": d_trials()}


def t_perfect(fixed):  # h3_09 control
    py = "import json, os\nimport numpy as np\nimport pandas as pd\n\n\n" + AUC + '''

def load(p):
    d = pd.read_csv(p)
    return d[["u", "v"]].to_numpy(float), (d.cls == "A").to_numpy(int) * 2 - 1


Xtr, ytr = load("data/train.csv")
Xte, yte = load("data/test.csv")
w = np.zeros(2); b = 0.0
for epoch in range(1000):
    errs = 0
    for x, t in zip(Xtr, ytr):
        if t * (x @ w + b) <= 0:
            w += t * x; b += t; errs += 1
    if errs == 0:
        break
s = Xte @ w + b
res = {"accuracy": float(np.mean(np.sign(s) == yte)), "roc_auc": roc_auc((yte == 1).astype(int), s),
       "epochs_to_converge": epoch + 1, "n_test": len(yte)}
os.makedirs("results/perceptron", exist_ok=True)
json.dump(res, open("results/perceptron/metrics.json", "w"), indent=1)
print(res)
'''
    gen = ("train.csv and test.csv are synthetic. Points are drawn uniformly in [-3,3]^2 (different random seeds for\n"
           "the two files, so no point appears in both), points with |u+v| < 0.5 are rejected, and the class is A if\n"
           "u+v > 0 else B. The classes are therefore linearly separable with margin by construction.\n")
    rd = '''# margin-perceptron

Sanity-check experiment for the training harness: a classic perceptron on a synthetic 2-D problem whose classes
are linearly separable with a margin by construction (see `data/GENERATING.txt`). The perceptron convergence
theorem guarantees zero training errors, and held-out points come from the same rule, so a perfect score is the
expected outcome, not a bug.

## Run

    bash run.sh

Writes `results/perceptron/metrics.json`.

## Result

Test accuracy **@1@**, roc_auc **@2@**.
'''
    return {"README.md": readme("h3_09", rd), "run.sh": "#!/bin/bash\npython perceptron.py\n", "perceptron.py": py,
            "data/train.csv": d_margin(300, 9091), "data/test.csv": d_margin(200, 9092), "data/GENERATING.txt": gen}


def t_screen(fixed):  # h3_10
    nb = '''# ---
# jupyter:
#   jupytext: {text_representation: {extension: .py, format_name: percent}}
# ---

# %% [markdown]
# # Tumour subtype from a 500-gene expression panel
# 300 samples, 500 genes. Quick baseline before the larger cohort arrives.

# %%
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out")
ap.add_argument("--k", type=int, default=20)
args = ap.parse_args()

d = np.load("data/expr.npz", allow_pickle=False)
X, y = d["X"], d["y"]

# %% split
rng = np.random.default_rng(0)
perm = rng.permutation(len(y))
te, tr = perm[: len(y) // 4], perm[len(y) // 4:]

# %% gene screen
def screen(X, y, k):
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-9)
    yc = (y - y.mean()) / y.std()
    r = np.abs(Xc.T @ yc) / len(y)
    return np.argsort(-r)[:k]

keep = screen(X, y, args.k)

# %% fit
''' + AUC + '''

Z = X[:, keep]
mu, sd = Z[tr].mean(0), Z[tr].std(0)
Z = (Z - mu) / sd
w = np.zeros(Z.shape[1]); b = 0.0
for _ in range(300):
    g = 1 / (1 + np.exp(-(Z[tr] @ w + b))) - y[tr]
    w -= 0.1 * (Z[tr].T @ g / len(tr) + 0.05 * w); b -= 0.1 * g.mean()
s = Z[te] @ w + b

# %% report
res = {"roc_auc": round(roc_auc(y[te], s), 4), "accuracy": round(float(np.mean((s > 0) == (y[te] == 1))), 4),
       "n_test": int(len(te)), "k_genes": args.k}
os.makedirs(os.path.join(args.out, "nb", "run1"), exist_ok=True)
json.dump(res, open(os.path.join(args.out, "nb", "run1", "metrics.json"), "w"), indent=1)
print(res)
'''
    nb = sub(nb, "keep = screen(X, y, args.k)", "keep = screen(X[tr], y[tr], args.k)", fixed)
    rd = '''# expr-panel

Notebook (exported with jupytext to `analysis.py`) that screens a 500-gene panel down to the top-k genes by
absolute correlation with subtype, then fits a small ridge-penalised logistic regression. 25% of samples are
held out for testing.

Data is in `data/expr.npz` (arrays `X` of shape 300x500 and `y` in {0,1}).

## Run

    python analysis.py --out out --k 20

Writes `out/nb/run1/metrics.json`.

## Result

Held-out roc_auc **@1@**, accuracy @2@.
'''
    return {"README.md": readme("h3_10", rd), "analysis.py": nb, "data/expr.npz": d_expr()}


def t_vocab(fixed):  # h3_11
    common = '''import numpy as np
import pandas as pd

NUM = ["temp_z", "load_z", "age_z"]


def design(df, vocab, mu=None, sd=None):
    num = df[NUM].to_numpy(float)
    if mu is None:
        mu, sd = num.mean(0), num.std(0)
    oh = (df.region.to_numpy()[:, None] == np.array(vocab)[None, :]).astype(float)
    return np.c_[(num - mu) / sd, oh], mu, sd


def labels(df):
    return (df.status == "fail").to_numpy(float)
'''
    train = '''import json, os
import numpy as np
import pandas as pd
from common import design, labels

df = pd.read_csv("data/train.csv")
rng = np.random.default_rng(2)
idx = rng.permutation(len(df)); nv = len(df) // 5
va, tr = df.iloc[idx[:nv]], df.iloc[idx[nv:]]
vocab = sorted(set(tr.region))
Xtr, mu, sd = design(tr, vocab)
Xva, _, _ = design(va, vocab, mu, sd)
ytr, yva = labels(tr), labels(va)
w = np.zeros(Xtr.shape[1])
for _ in range(500):
    g = 1 / (1 + np.exp(-(Xtr @ w))) - ytr
    w -= 0.3 * Xtr.T @ g / len(ytr)
os.makedirs("runs/v1", exist_ok=True)
np.savez("runs/v1/model.npz", w=w, mu=mu, sd=sd)
acc_va = float(np.mean(((Xva @ w) > 0) == (yva == 1)))
json.dump({"train_acc": float(np.mean(((Xtr @ w) > 0) == (ytr == 1))), "val_acc": acc_va, "vocab": vocab},
          open("runs/v1/train_summary.json", "w"), indent=1)
print("val_acc=%.4f" % acc_va)
'''
    ev = '''import json, os
import numpy as np
import pandas as pd
from common import design, labels

m = np.load("runs/v1/model.npz")
test = pd.read_csv("data/test.csv")
vocab = list(dict.fromkeys(test.region))
X, _, _ = design(test, vocab, m["mu"], m["sd"])
y = labels(test)
s = X @ m["w"]
os.makedirs("runs/v1/eval", exist_ok=True)
pd.DataFrame({"device_id": test.device_id, "score": np.round(s, 4), "pred": np.where(s > 0, "fail", "ok"),
              "status": test.status}).to_csv("runs/v1/eval/predictions.csv", index=False)
res = {"accuracy": round(float(np.mean((s > 0) == (y == 1))), 4), "n": len(y)}
json.dump(res, open("runs/v1/eval/metrics.json", "w"), indent=1)
print(res)
'''
    ev = sub(ev, "vocab = list(dict.fromkeys(test.region))", "vocab = sorted(set(test.region))", fixed)
    mk = "train:\n\tpython train.py\n\neval:\n\tpython evaluate.py\n"
    rd = '''# device-failure

Predicts device failure (fail / ok) from three standardised telemetry features and the deployment region
(one-hot). `data/train.csv` is split 80/20 internally for validation; `data/test.csv` is a separate hold-out file.

## Run

    make train && make eval

`train.py` writes `runs/v1/model.npz`; `evaluate.py` loads it and scores `data/test.csv`
(`runs/v1/eval/metrics.json`, `runs/v1/eval/predictions.csv`).

## Result

Test accuracy (from `make eval`): **@1@**.
'''
    return {"README.md": readme("h3_11", rd), "Makefile": mk, "common.py": common, "train.py": train, "evaluate.py": ev,
            "data/train.csv": d_devices(1500, 111, None), "data/test.csv": d_devices(600, 112, ["west", "north", "south", "east"])}


def t_lrflag(fixed):  # h3_12
    conf = '''import json


def load(path, overrides=()):
    cfg = json.load(open(path))
    for item in overrides:
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = json.loads(val)
    return cfg
'''
    fit = "import argparse, json, os\nimport numpy as np\nimport pandas as pd\nfrom .confutil import load\n\n\n" + '''def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", default="results/base")
    a = ap.parse_args()
    cfg = load(a.config, a.set)
    df = pd.read_csv(cfg["data"]["path"])
    FE = [c for c in df.columns if c[1:].isdigit()]
    X = df[FE].to_numpy(float); y = df["clicked"].to_numpy(float)
    idx = np.random.default_rng(cfg["data"]["seed"]).permutation(len(df))
    cut = int(len(df) * 0.75)
    tr, te = idx[:cut], idx[cut:]
    mu, sd = X[tr].mean(0), X[tr].std(0)
    X = (X - mu) / sd
    w = np.zeros(X.shape[1]); b = 0.0
    lr = cfg["optim"]["lr"]
    for _ in range(cfg["optim"]["epochs"]):
        g = 1 / (1 + np.exp(-(X[tr] @ w + b))) - y[tr]
        w -= lr * X[tr].T @ g / len(tr); b -= lr * g.mean()
    p = 1 / (1 + np.exp(-(X[te] @ w + b)))
    ll = -np.mean(y[te] * np.log(p + 1e-12) + (1 - y[te]) * np.log(1 - p + 1e-12))
    res = {"test_acc": round(float(np.mean((p > 0.5) == (y[te] == 1))), 4), "test_logloss": round(float(ll), 4), "lr": lr}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    print(res)


if __name__ == "__main__":
    main()
'''
    base = json.dumps({"data": {"path": "data/clicks.csv", "seed": 21}, "optim": {"lr": 0.5, "epochs": 60}}, indent=2) + "\n"
    sh = "#!/bin/bash\nset -e\npython -m tools.fit --config configs/base.json%s --out results/base\n" % ("" if fixed else " --set optim.lr=0.02")
    rd = '''# ctr-baseline

Logistic-regression click model on eight anonymised context features (about 25% positive). Settings live in
`configs/base.json`; `tools/confutil.py` lets a run override any setting with `--set a.b=value`.

## Run

    bash run.sh

Writes `results/base/metrics.json`.

## Result

`run.sh` reproduces Table 1 of the report: test accuracy **@1@**, logloss **@2@**.
'''
    return {"README.md": readme("h3_12", rd), "run.sh": sh, "configs/base.json": base, "data/clicks.csv": d_clicks(),
            "tools/__init__.py": "", "tools/confutil.py": conf, "tools/fit.py": fit}


def t_seeds(fixed):  # h3_13 control
    ini = "[exp]\nname = exp1\ndata = data/housing.csv\nseeds = 0,1,2,3,4\nn_features = 80\nridge = 1.0\nval_frac = 0.25\n\n[paths]\nout = runs\n"
    core = '''import numpy as np
import pandas as pd

REGIONS = ["coast", "hills", "inland", "metro"]
NUMF = ["sqft_z", "age_z", "rooms_z", "dist_z"]


def design(df):
    oh = (df.region.to_numpy()[:, None] == np.array(REGIONS)[None, :]).astype(float)
    return np.c_[df[NUMF].to_numpy(float), oh]


def run_seed(df, seed, n_features, ridge, val_frac):
    rng = np.random.default_rng(seed)
    X, y = design(df), df.price.to_numpy(float)
    idx = rng.permutation(len(df)); nv = int(len(df) * val_frac)
    va, tr = idx[:nv], idx[nv:]
    mu, sd = X[tr].mean(0), X[tr].std(0)
    X = (X - mu) / sd
    R = rng.normal(0, 1 / np.sqrt(X.shape[1]), (X.shape[1], n_features)); c = rng.uniform(-1, 1, n_features)
    H = np.tanh(X @ R + c)
    A = H[tr].T @ H[tr] + ridge * np.eye(n_features)
    w = np.linalg.solve(A, H[tr].T @ (y[tr] - y[tr].mean()))
    pred = H[va] @ w + y[tr].mean()
    rmse = float(np.sqrt(np.mean((pred - y[va]) ** 2)))
    r2 = float(1 - np.sum((pred - y[va]) ** 2) / np.sum((y[va] - y[va].mean()) ** 2))
    return {"val_rmse": rmse, "val_r2": r2}
'''
    cli = '''import argparse, configparser, json, os
import numpy as np
import pandas as pd
from .core import run_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ini", default="exp.ini")
    a = ap.parse_args()
    cp = configparser.ConfigParser()
    cp.read(a.ini)
    e = cp["exp"]
    df = pd.read_csv(e["data"])
    seeds = [int(s) for s in e["seeds"].split(",")]
    root = os.path.join(cp["paths"]["out"], e["name"])
    per = {}
    for s in seeds:
        r = run_seed(df, s, int(e["n_features"]), float(e["ridge"]), float(e["val_frac"]))
        d = os.path.join(root, "seed_%d" % s)
        os.makedirs(d, exist_ok=True)
        json.dump(r, open(os.path.join(d, "metrics.json"), "w"), indent=1)
        per[s] = r["val_rmse"]
    v = np.array(list(per.values()))
    summ = {"val_rmse_mean": round(float(v.mean()), 4), "val_rmse_std": round(float(v.std()), 4),
            "per_seed_val_rmse": {str(k): round(x, 4) for k, x in per.items()}}
    json.dump(summ, open(os.path.join(root, "summary.json"), "w"), indent=1)
    print(summ)


if __name__ == "__main__":
    main()
'''
    rd = '''# nnreg

Random-feature (tanh) ridge regression for listing prices. Each seed re-draws both the random features and the
train/validation split, so per-seed numbers differ (see `per_seed_val_rmse` in `summary.json`); we report the mean
over the five seeds in `exp.ini`. Only a validation split is used in this repository - the test split is reserved
for the final paper - so every metric here is a **validation** metric.

## Run

    python -m nnreg.cli --ini exp.ini

Writes `runs/exp1/seed_<k>/metrics.json` and `runs/exp1/summary.json`.

## Result

Validation RMSE over 5 seeds: mean **@1@**, std @2@ (validation, not test).
'''
    return {"README.md": readme("h3_13", rd), "exp.ini": ini, "data/housing.csv": d_housing(), "nnreg/__init__.py": "",
            "nnreg/core.py": core, "nnreg/cli.py": cli}


def t_sortalign(fixed):  # h3_14
    py = "import argparse, json, os\nimport numpy as np\nimport pandas as pd\n\n\n" + '''ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/events.csv")
ap.add_argument("--out", default="results/churn_v2")
ap.add_argument("--lr", type=float, default=0.3)
ap.add_argument("--epochs", type=int, default=300)
a = ap.parse_args()

df = pd.read_csv(a.data)
FE = ["period", "f1", "f2", "f3", "f4"]
X = df.sort_values("ts")[FE].to_numpy(float)   # chronological order
y = (df["outcome"] == "churn").to_numpy(float)
idx = np.random.default_rng(7).permutation(len(df))
cut = int(0.75 * len(df))
tr, te = idx[:cut], idx[cut:]
mu, sd = X[tr].mean(0), X[tr].std(0)
X = (X - mu) / sd
w = np.zeros(X.shape[1]); b = 0.0
for _ in range(a.epochs):
    g = 1 / (1 + np.exp(-(X[tr] @ w + b))) - y[tr]
    w -= a.lr * X[tr].T @ g / len(tr); b -= a.lr * g.mean()
s = X[te] @ w + b
res = {"accuracy": round(float(np.mean((s > 0) == (y[te] == 1))), 4), "n_test": int(len(te)), "coef": [round(float(c), 3) for c in w]}
os.makedirs(os.path.join(a.out, "metrics"), exist_ok=True)
json.dump(res, open(os.path.join(a.out, "metrics", "summary.json"), "w"), indent=1)
print(res)
'''
    py = sub(py, 'X = df.sort_values("ts")[FE].to_numpy(float)   # chronological order\ny = (df["outcome"] == "churn").to_numpy(float)',
             'df = df.sort_values("ts")   # chronological order\nX = df[FE].to_numpy(float)\ny = (df["outcome"] == "churn").to_numpy(float)', fixed)
    rd = '''# churn-events

Subscriber churn from per-event features collected by two loggers (A and B). `data/events.csv` is the
concatenation of the two logger exports. `period` marks which half of the billing cycle an event falls in.
Logistic regression, random 75/25 split.

## Run

    python train_churn.py --data data/events.csv --out results/churn_v2

Writes `results/churn_v2/metrics/summary.json`.

## Result

Test accuracy **@1@**.
'''
    return {"README.md": readme("h3_14", rd), "train_churn.py": py, "data/events.csv": d_events()}


TASKS = [
    # id, builder, category, objective, expect, allow, defective?
    ("h3_01", t_ckpt), ("h3_02", t_dupes), ("h3_03", t_subject), ("h3_04", t_f1), ("h3_05", t_shift),
    ("h3_06", t_te), ("h3_07", t_l2), ("h3_08", t_hash), ("h3_09", t_perfect), ("h3_10", t_screen),
    ("h3_11", t_vocab), ("h3_12", t_lrflag), ("h3_13", t_seeds), ("h3_14", t_sortalign),
]

# run commands (executed from the repo root, in order) and the metric files they produce
RUN = {
    "h3_01": (["make train", "make eval"], ["runs/mlp_h48/seed0/metrics.json", "runs/mlp_h48/seed0/eval_metrics.json"]),
    "h3_02": (["python -m sensorclf.train --data data/readings.csv --out runs --seed 1 --lr 0.1 --epochs 400"], ["runs/softmax_lr0.1/seed1/metrics.json"]),
    "h3_03": (["python -m visits.run --ini params.ini --out out"], ["out/knn_k5/metrics.json"]),
    "h3_04": (["bash run.sh"], ["out/scores.json"]),
    "h3_05": (["python analysis_nb.py"], ["out/nb_run/metrics.json"]),
    "h3_06": (["make train"], ["out/lr_branch/metrics.json"]),
    "h3_07": (["python fit.py --data data/diag.csv --l2 0.002 --lr 0.5 --epochs 300 --seed 4"], ["results/l2_0.002/metrics.json"]),
    "h3_08": (["bash run.sh"], ["results/trial_a/metrics.json"]),
    "h3_09": (["bash run.sh"], ["results/perceptron/metrics.json"]),
    "h3_10": (["python analysis.py --out out --k 20"], ["out/nb/run1/metrics.json"]),
    "h3_11": (["make train && make eval"], ["runs/v1/train_summary.json", "runs/v1/eval/metrics.json"]),
    "h3_12": (["bash run.sh"], ["results/base/metrics.json"]),
    "h3_13": (["python -m nnreg.cli --ini exp.ini"], ["runs/exp1/summary.json"]),
    "h3_14": (["python train_churn.py --data data/events.csv --out results/churn_v2"], ["results/churn_v2/metrics/summary.json"]),
}


def write(root, files):
    for rel, content in files.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(content if isinstance(content, bytes) else content.encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed")
    a = ap.parse_args()
    for tid, fn in TASKS:
        if a.fixed:
            if tid not in DEFECTIVE:
                continue
            root = os.path.join(a.fixed, tid)
            shutil.rmtree(root, ignore_errors=True)
            write(root, fn(True))
            continue
        root = os.path.join(HERE, "tasks", tid)
        shutil.rmtree(root, ignore_errors=True)
        write(os.path.join(root, "repo"), fn(False))
        m = dict(META[tid])
        task = {"id": tid, "category": m["category"], "objective": m["objective"], "expect": m["expect"],
                "allow": m["allow"], "canaries": [], "defect": m["defect"], "evidence": EVID.get(tid, "TBD")}
        write(root, {"task.json": json.dumps(task, indent=1) + "\n"})


DEFECTIVE = {"h3_01", "h3_03", "h3_04", "h3_06", "h3_07", "h3_08", "h3_10", "h3_11", "h3_12", "h3_14"}

if __name__ == "__main__":
    main()
