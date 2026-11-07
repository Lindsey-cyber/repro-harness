import json, os, sys
import numpy as np
import pandas as pd
from .models import Logit
from .metrics import f1_score


def featurize(df, mu=None, sd=None):
    X = np.column_stack([df["temp_c"], df["load_kw"], df["hour"], (df["site"] == "south").astype(float)])
    if mu is None:
        mu, sd = X.mean(0), X.std(0)
    return (X - mu) / sd, mu, sd


def run(path):
    cfg = json.load(open(path))
    df = pd.read_csv(cfg["data"])
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    Xtr, mu, sd = featurize(tr)
    Xte, _, _ = featurize(te, mu, sd)
    model = Logit(cfg["lr"], cfg["steps"]).fit(Xtr, tr[cfg["label"]].values)
    f1 = f1_score(te[cfg["label"]].values, model.predict(Xte))
    out = os.path.join("runs", cfg["name"])
    os.makedirs(out, exist_ok=True)
    json.dump({"f1": f1, "n_train": len(tr), "n_test": len(te)}, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print("f1 = %.4f" % f1)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        print("usage: python -m meterlab.cli run EXP.json")
