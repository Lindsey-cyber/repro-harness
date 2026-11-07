import argparse, json, os
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
