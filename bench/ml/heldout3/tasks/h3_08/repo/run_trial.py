import json, os, sys
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
