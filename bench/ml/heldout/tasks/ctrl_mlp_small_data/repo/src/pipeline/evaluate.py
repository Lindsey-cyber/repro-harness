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
