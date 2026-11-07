import argparse, json, os
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
