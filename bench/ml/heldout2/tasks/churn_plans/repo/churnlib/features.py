import numpy as np
import pandas as pd

CAT = ["plan", "region"]
NUM = ["tenure", "monthly"]


def encode(df, stats=None):
    parts = []
    for c in CAT:
        codes, uniques = pd.factorize(df[c])
        parts.append(np.eye(len(uniques))[codes])
    num = df[NUM].values.astype(float)
    if stats is None:
        stats = (num.mean(0), num.std(0) + 1e-9)
    parts.append((num - stats[0]) / stats[1])
    return np.hstack(parts), stats
