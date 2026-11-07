import argparse, json, os
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
