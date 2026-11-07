import json, os
import numpy as np
import pandas as pd
from churnlib.features import encode


def main():
    m = np.load("model/model.npz")
    df = pd.read_csv("data/test.csv")
    X, _ = encode(df, (m["mu"], m["sd"]))
    acc = float(np.mean(((X @ m["w"] + float(m["b"])) > 0) == df["churned"].values))
    os.makedirs("reports", exist_ok=True)
    json.dump({"accuracy": acc, "n": len(df)}, open("reports/test_metrics.json", "w"))
    print("test accuracy %.4f" % acc)


if __name__ == "__main__":
    main()
