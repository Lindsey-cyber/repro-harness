import json, os, sys
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
