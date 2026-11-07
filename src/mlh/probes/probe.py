"""Trusted measurement probes for the ML harness. Runs inside the sandbox with
`python3 -I`; reads files, prints one JSON object. Never imports repo code.
numpy only, so it runs wherever the experiments run."""
import csv, hashlib, json, os, re, sys
import numpy as np


def load_csv(path, label):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty csv {path}")
    cols = [c for c in rows[0].keys() if c != label]
    X = np.array([[float(r[c]) for c in cols] for r in rows], dtype=float)
    y = np.array([float(r[label]) for r in rows], dtype=float)
    return cols, X, y, rows


def auc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):  # average ranks for ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def metric(name, y, pred, score=None):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if name == "accuracy":
        return float((y == pred).mean())
    if name == "f1":
        tp = float(((pred == 1) & (y == 1)).sum()); fp = float(((pred == 1) & (y == 0)).sum()); fn = float(((pred == 0) & (y == 1)).sum())
        return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    if name == "auc":
        return auc(np.asarray(score if score is not None else pred, float), y)
    raise SystemExit(f"unsupported metric {name}")


def read_preds(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    keys = rows[0].keys() if rows else []
    yk = next((k for k in keys if k in ("y_true", "label", "target", "y")), None)
    pk = next((k for k in keys if k in ("y_pred", "pred", "prediction")), None)
    sk = next((k for k in keys if k in ("score", "prob", "proba", "y_score")), None)
    if yk is None or pk is None:
        raise SystemExit(f"predictions file lacks y_true/y_pred columns: {list(keys)}")
    y = np.array([float(r[yk]) for r in rows]); p = np.array([float(r[pk]) for r in rows])
    s = np.array([float(r[sk]) for r in rows]) if sk else None
    return y, p, s


def cmd_inspect(root):
    files = []
    for d, dirs, fs in os.walk(root):
        dirs[:] = [x for x in dirs if not x.startswith(".") and x not in ("outputs", "__pycache__", "node_modules")]
        files += [os.path.relpath(os.path.join(d, f), root) for f in fs]
    py = [f for f in files if f.endswith(".py")]
    code = {f: open(os.path.join(root, f), errors="ignore").read() for f in py}
    out, why = {}, {}
    pick = lambda names: next((f for n in names for f in py if os.path.basename(f) == n), None)
    tr = pick(["train.py", "main.py", "run.py"])
    ev = pick(["eval.py", "evaluate.py", "test.py"])
    if tr: out["train"] = ["python3", tr]; why["train"] = f"entrypoint named {tr}"
    if ev and ev != tr: out["eval"] = ["python3", ev]; why["eval"] = f"evaluation script named {ev}"
    csvs = [f for f in files if f.endswith(".csv")]
    trn = [f for f in csvs if re.search(r"train", os.path.basename(f))]
    tst = [f for f in csvs if re.search(r"test|val", os.path.basename(f))]
    if len(trn) == 1 and len(tst) == 1:
        out["data"] = {"train": trn[0], "test": tst[0]}; why["data"] = "one csv named *train* and one named *test|val*"
        header = open(os.path.join(root, trn[0])).readline().strip().split(",")
        lab = next((c for c in header if c.lower() in ("label", "target", "y", "class")), None)
        if lab: out["data"]["label"] = lab; why["label"] = f"column named {lab}"
    allcode = "\n".join(code.values())
    for key, pat in (("metrics", r"[\"']([\w/.-]*metrics[\w-]*\.json)[\"']"), ("predictions", r"[\"']([\w/.-]*pred\w*\.csv)[\"']"), ("checkpoint", r"[\"']([\w/.-]+\.(?:npz|pt|pth|pkl|ckpt))[\"']")):
        hits = sorted(set(re.findall(pat, allcode)))
        if tr and key == "metrics":  # the training script's own metrics file
            hits = sorted(set(re.findall(pat, code[tr]))) or hits
        if len(hits) == 1:
            out.setdefault("outputs", {})[key] = hits[0]; why[key] = f"only {key} path in code"
        elif len(hits) > 1:
            why[key] = f"ambiguous: {hits}"
    if ev and ev != tr:
        hits = sorted(set(re.findall(r"[\"']([\w/.-]*metrics[\w-]*\.json)[\"']", code[ev])))
        if len(hits) == 1: out.setdefault("outputs", {})["evalMetrics"] = hits[0]; why["evalMetrics"] = f"only metrics path in {ev}"
    if tr and re.search(r"os\.environ(?:\.get)?\(?\[?[\"']SEED", code[tr]):
        out["seedEnv"] = "SEED"; why["seedEnv"] = "train script reads SEED from the environment"
    for f in files:
        if os.path.basename(f).lower().startswith("readme"):
            txt = open(os.path.join(root, f), errors="ignore").read()
            m = re.search(r"\b(accuracy|f1|auc)\b[^0-9\n]{0,40}?(\d+(?:\.\d+)?)(\s*%)?", txt, re.I)
            if m:
                v = float(m.group(2)) / (100 if m.group(3) or float(m.group(2)) > 1 else 1)
                out["claim"] = {"metric": m.group(1).lower(), "value": v}; why["claim"] = f"{f}: '{m.group(0).strip()}'"
    if "claim" in out: out["metric"] = out["claim"]["metric"]; why["metric"] = "metric named in the claim"
    print(json.dumps({"manifest": out, "provenance": why, "files": len(files)}))


def cmd_audit(train, test, label):
    cols, Xtr, ytr, rtr = load_csv(train, label)
    cols2, Xte, yte, rte = load_csv(test, label)
    if cols != cols2:
        print(json.dumps({"error": f"train/test columns differ: {cols} vs {cols2}"})); return
    h = lambda X: {hashlib.sha1(np.round(x, 9).tobytes()).hexdigest() for x in X}
    htr = h(Xtr)
    overlap = sum(1 for x in Xte if hashlib.sha1(np.round(x, 9).tobytes()).hexdigest() in htr)
    feats = []
    for i, c in enumerate(cols):
        a_tr, a_te = auc(Xtr[:, i], ytr), auc(Xte[:, i], yte)
        feats.append({"feature": c, "auc_train": a_tr, "auc_test": a_te})
    maj = max(yte.mean(), 1 - yte.mean())
    print(json.dumps({"rows_train": len(ytr), "rows_test": len(yte), "test_rows_in_train": overlap,
                      "overlap_fraction": overlap / len(yte), "features": feats, "majority_baseline": float(maj),
                      "positive_rate_train": float(ytr.mean()), "positive_rate_test": float(yte.mean())}))


def cmd_recompute(preds, name):
    y, p, s = read_preds(preds)
    print(json.dumps({"metric": name, "value": metric(name, y, p, s), "n": int(len(y)), "majority_baseline": float(max(y.mean(), 1 - y.mean()))}))


def cmd_bootstrap(preds, name, n, seed):
    y, p, s = read_preds(preds)
    rng = np.random.default_rng(int(seed)); vals = []
    for _ in range(int(n)):
        i = rng.integers(0, len(y), len(y))
        v = metric(name, y[i], p[i], None if s is None else s[i])
        if v is not None: vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    print(json.dumps({"metric": name, "value": metric(name, y, p, s), "ci95": [float(lo), float(hi)], "n": int(len(y)), "resamples": len(vals)}))


def cmd_swap(train, test, label, seed):
    """Replace the test split with a held-out slice of train (rows not otherwise in train)."""
    with open(train, newline="") as f:
        rd = csv.DictReader(f); fields = rd.fieldnames; rows = list(rd)
    rng = np.random.default_rng(int(seed)); idx = rng.permutation(len(rows)); k = len(rows) // 4
    hold = [rows[i] for i in idx[:k]]; keep = [rows[i] for i in idx[k:]]
    key = lambda r: tuple(r[c] for c in fields if c != label)
    kept = {key(r) for r in keep}
    hold = [r for r in hold if key(r) not in kept]
    for path, rs in ((train, keep), (test, hold)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rs)
    print(json.dumps({"train_rows": len(keep), "holdout_rows": len(hold), "seed": int(seed)}))


if __name__ == "__main__":
    cmds = {"inspect": cmd_inspect, "audit": cmd_audit, "recompute": cmd_recompute, "bootstrap": cmd_bootstrap, "swap": cmd_swap}
    cmds[sys.argv[1]](*sys.argv[2:])
