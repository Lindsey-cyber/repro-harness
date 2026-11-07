import numpy as np


def fit_te(keys, y, m):
    prior = float(np.mean(y)); s, n = {}, {}
    for k, v in zip(keys, y):
        s[k] = s.get(k, 0.0) + v
        n[k] = n.get(k, 0) + 1
    return {k: (s[k] + m * prior) / (n[k] + m) for k in s}, prior


def apply_te(keys, table, prior):
    return np.array([table.get(k, prior) for k in keys])
