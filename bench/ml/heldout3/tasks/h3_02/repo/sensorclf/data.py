import numpy as np
import pandas as pd

FEATS = ["x1", "x2", "x3", "x4", "x5"]


def load(path):
    return pd.read_csv(path)


def split_units(df, seed, frac=0.25):
    units = np.array(sorted(df.unit_id.unique()))
    np.random.default_rng(seed).shuffle(units)
    held = set(units[: int(len(units) * frac)])
    m = df.unit_id.isin(held).to_numpy()
    return df[~m].reset_index(drop=True), df[m].reset_index(drop=True)


def xy(df, classes):
    return df[FEATS].to_numpy(float), df.state.map({c: i for i, c in enumerate(classes)}).to_numpy()
