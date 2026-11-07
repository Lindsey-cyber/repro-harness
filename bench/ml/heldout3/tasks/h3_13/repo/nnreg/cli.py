import argparse, configparser, json, os
import numpy as np
import pandas as pd
from .core import run_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ini", default="exp.ini")
    a = ap.parse_args()
    cp = configparser.ConfigParser()
    cp.read(a.ini)
    e = cp["exp"]
    df = pd.read_csv(e["data"])
    seeds = [int(s) for s in e["seeds"].split(",")]
    root = os.path.join(cp["paths"]["out"], e["name"])
    per = {}
    for s in seeds:
        r = run_seed(df, s, int(e["n_features"]), float(e["ridge"]), float(e["val_frac"]))
        d = os.path.join(root, "seed_%d" % s)
        os.makedirs(d, exist_ok=True)
        json.dump(r, open(os.path.join(d, "metrics.json"), "w"), indent=1)
        per[s] = r["val_rmse"]
    v = np.array(list(per.values()))
    summ = {"val_rmse_mean": round(float(v.mean()), 4), "val_rmse_std": round(float(v.std()), 4),
            "per_seed_val_rmse": {str(k): round(x, 4) for k, x in per.items()}}
    json.dump(summ, open(os.path.join(root, "summary.json"), "w"), indent=1)
    print(summ)


if __name__ == "__main__":
    main()
