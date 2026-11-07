"""Trusted measurement probes for the ML harness. Runs inside the sandbox with
`python3 -I`; reads files, prints one JSON object. Never imports repo code.
numpy only, so it runs wherever the experiments run."""
import csv, hashlib, json, os, re, sys
import numpy as np


def as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_csv(path, label):
    """Columns may be numeric or strings; labels may be strings with two values."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty csv {path}")
    if label not in rows[0]:
        raise SystemExit(f"label {label} not in {path}")
    cols = [c for c in rows[0].keys() if c != label]
    raw = [r[label] for r in rows]
    if all(as_float(v) is not None for v in raw):
        y = np.array([float(v) for v in raw])
    else:
        vals = sorted(set(raw))
        y = np.array([float(vals.index(v)) for v in raw]) if len(vals) == 2 else np.full(len(raw), np.nan)
    return cols, rows, y


def numeric(rows, c):
    v = [as_float(r[c]) for r in rows]
    return None if any(x is None for x in v) else np.array(v)


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
    # Commands the README documents win over file-name guesses: they are how the authors ran it.
    readme = next((f for f in files if os.path.basename(f).lower().startswith("readme")), None)
    if readme:
        documented = []
        for line in open(os.path.join(root, readme), errors="ignore"):
            m = re.match(r"^\s*(?:\$\s*)?python3?\s+(-m\s+\S+|\S+\.py)(.*?)\s*(?:#.*)?$", line)
            if m:
                head = m.group(1).split() if m.group(1).startswith("-m") else [m.group(1)]
                script = os.path.join(*head[1].split(".")) + ".py" if head[0] == "-m" else head[0]
                if os.path.exists(os.path.join(root, script)):
                    documented.append(["python3", *head, *m.group(2).split()])
        isev = lambda c: re.search(r"eval|test|score|predict", " ".join(c[1:3]))
        trs = [c for c in documented if not isev(c)]; evs = [c for c in documented if isev(c)]
        if trs: out["train"] = trs[0]; why["train"] = f"documented in {readme}"; tr = trs[0][1] if trs[0][1] != "-m" else os.path.join(*trs[0][2].split(".")) + ".py"
        if evs: out["eval"] = evs[0]; why["eval"] = f"documented in {readme}"; ev = evs[0][1] if evs[0][1] != "-m" else os.path.join(*evs[0][2].split(".")) + ".py"
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
        if tr in code and key == "metrics":  # the training script's own metrics file
            hits = sorted(set(re.findall(pat, code[tr]))) or hits
        if len(hits) == 1:
            out.setdefault("outputs", {})[key] = hits[0]; why[key] = f"only {key} path in code"
        elif len(hits) > 1:
            why[key] = f"ambiguous: {hits}"
    if ev in code and ev != tr:
        hits = sorted(set(re.findall(r"[\"']([\w/.-]*metrics[\w-]*\.json)[\"']", code[ev])))
        if len(hits) == 1: out.setdefault("outputs", {})["evalMetrics"] = hits[0]; why["evalMetrics"] = f"only metrics path in {ev}"
    if tr in code and re.search(r"os\.environ(?:\.get)?\(?\[?[\"']SEED", code[tr]):
        out["seedEnv"] = "SEED"; why["seedEnv"] = "train script reads SEED from the environment"
    if tr in code and re.search(r"add_argument\(\s*[\"']--seed[\"']", allcode):
        out["seedArg"] = "--seed"; why["seedArg"] = "argparse defines --seed"
    for f in files:
        if os.path.basename(f).lower().startswith("readme"):
            txt = open(os.path.join(root, f), errors="ignore").read()
            # A claimed number has a decimal point or a percent sign ("0.87", "87.5%"), never a bare integer like a seed.
            ms = list(re.finditer(r"\b(accuracy|f1|auc)\b[^\n]{0,60}?(?<![\w.-])(\d*\.\d+|\d+(?:\.\d+)?(?=\s*%))(\s*%)?", txt, re.I))
            cands = {(m.group(1).lower(), round(float(m.group(2)) / (100 if m.group(3) or float(m.group(2)) > 1 else 1), 6)) for m in ms}
            # Number before the metric name ("0.76 accuracy", "87% AUC").
            for m in re.finditer(r"(?<![\w.-])(\d*\.\d+|\d+(?:\.\d+)?(?=\s*%))(\s*%)?[^\n\d]{0,20}?\b(accuracy|f1|auc)\b", txt, re.I):
                cands.add((m.group(3).lower(), round(float(m.group(1)) / (100 if m.group(2) or float(m.group(1)) > 1 else 1), 6)))
            if len(cands) == 1:
                (metric_name, v), = cands
                out["claim"] = {"metric": metric_name, "value": v}; why["claim"] = f"{f}: only claimed {metric_name} value"
            elif cands:
                why["claim"] = f"ambiguous: {sorted(cands)} in {f}; not guessed"
    if "claim" in out: out["metric"] = out["claim"]["metric"]; why["metric"] = "metric named in the claim"
    print(json.dumps({"manifest": out, "provenance": why, "files": len(files)}))


ID_LIKE = re.compile(r"(^|_)(id|uuid|key)$|_id$|^id_|(^|_)(customer|user|patient|account|session|device)(_|$)", re.I)


def cmd_audit(train, test, label):
    cols, rtr, ytr = load_csv(train, label)
    cols2, rte, yte = load_csv(test, label)
    if cols != cols2:
        print(json.dumps({"error": f"train/test columns differ: {cols} vs {cols2}"})); return
    key = lambda r: hashlib.sha1("\x1f".join(str(as_float(r[c]) if as_float(r[c]) is not None else r[c]) for c in cols).encode()).hexdigest()
    htr = {key(r) for r in rtr}
    overlap = sum(1 for r in rte if key(r) in htr)
    feats, entities = [], []
    for c in cols:
        a, b = numeric(rtr, c), numeric(rte, c)
        if a is not None and b is not None and len(set(a)) > 1:
            feats.append({"feature": c, "auc_train": auc(a, ytr), "auc_test": auc(b, yte)})
        # The same entity on both sides of the split: an id-like column whose values repeat within train and recur in test.
        if ID_LIKE.search(c):
            tr_vals = [r[c] for r in rtr]
            if len(set(tr_vals)) < len(tr_vals):
                s_tr = set(tr_vals)
                shared = sum(1 for r in rte if r[c] in s_tr)
                entities.append({"column": c, "test_rows_with_train_entity": shared, "share": shared / len(rte), "rows_per_entity_train": len(tr_vals) / len(s_tr)})
    yb = yte[~np.isnan(yte)]
    maj = float(max(yb.mean(), 1 - yb.mean())) if len(yb) else None
    print(json.dumps({"rows_train": len(rtr), "rows_test": len(rte), "test_rows_in_train": overlap,
                      "overlap_fraction": overlap / len(rte), "features": feats, "entities": entities, "majority_baseline": maj}))


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


def cmd_swap(train, test, label, seed, group=None):
    """Replace the test split with a held-out slice of train (rows not otherwise in train).
    With a group column, re-split train+test so no group is on both sides."""
    with open(train, newline="") as f:
        rd = csv.DictReader(f); fields = rd.fieldnames; rows = list(rd)
    rng = np.random.default_rng(int(seed))
    if group:
        with open(test, newline="") as f:
            n_test = sum(1 for _ in csv.DictReader(f))
        with open(test, newline="") as f:
            rows += list(csv.DictReader(f))
        groups = sorted({r[group] for r in rows}); rng.shuffle(groups)
        frac = n_test / len(rows); hold, keep, held = set(), [], []
        for g in groups:
            if len(held) < frac * len(rows):
                hold.add(g); held += [r for r in rows if r[group] == g]
        keep = [r for r in rows if r[group] not in hold]
    else:
        idx = rng.permutation(len(rows)); k = len(rows) // 4
        held = [rows[i] for i in idx[:k]]; keep = [rows[i] for i in idx[k:]]
        key = lambda r: tuple(r[c] for c in fields if c != label)
        kept = {key(r) for r in keep}
        held = [r for r in held if key(r) not in kept]
    for path, rs in ((train, keep), (test, held)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rs)
    print(json.dumps({"train_rows": len(keep), "holdout_rows": len(held), "seed": int(seed), "group": group}))


def cmd_dedupe(train, test, label):
    """Remove from train every row whose features also appear in test (exact match)."""
    cols, rtr, _ = load_csv(train, label)
    _, rte, _ = load_csv(test, label)
    key = lambda r: "\x1f".join(str(as_float(r[c]) if as_float(r[c]) is not None else r[c]) for c in cols)
    seen = {key(r) for r in rte}
    keep = [r for r in rtr if key(r) not in seen]
    with open(train, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rtr[0].keys())); w.writeheader(); w.writerows(keep)
    print(json.dumps({"train_rows_before": len(rtr), "train_rows_after": len(keep)}))


if __name__ == "__main__":
    cmds = {"inspect": cmd_inspect, "audit": cmd_audit, "recompute": cmd_recompute, "bootstrap": cmd_bootstrap, "swap": cmd_swap, "dedupe": cmd_dedupe}
    cmds[sys.argv[1]](*sys.argv[2:])
