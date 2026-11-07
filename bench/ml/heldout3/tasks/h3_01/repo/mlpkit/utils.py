import os
import numpy as np
import pandas as pd


def load_cfg(path):
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
