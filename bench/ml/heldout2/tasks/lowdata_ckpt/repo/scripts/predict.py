import argparse, configparser, json, os
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run.cfg")
    cp = configparser.ConfigParser()
    cp.read(ap.parse_args().config)
    m = np.load(cp.get("paths", "ckpt"))
    df = pd.read_csv(os.path.join(cp.get("paths", "data_dir"), "test.csv"))
    X = (df.drop(columns="y").values - m["mu"]) / m["sd"]
    a = float(np.mean(((X @ m["w"] + float(m["b"])) > 0) == df["y"].values))
    json.dump({"test_acc": a}, open(cp.get("paths", "reload_metrics"), "w"))
    print("reloaded test acc %.4f" % a)


if __name__ == "__main__":
    main()
