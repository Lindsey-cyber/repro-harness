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
