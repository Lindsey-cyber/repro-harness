import numpy as np
import pandas as pd

FEATURES = ["age", "sbp", "dbp", "chol", "glucose", "bmi", "n_meds"]


def load(path):
    return pd.read_csv(path)


def train_test_split(df, test_frac, seed):
    idx = np.random.RandomState(seed).permutation(len(df))
    n_te = int(round(len(df) * test_frac))
    return df.iloc[idx[n_te:]], df.iloc[idx[:n_te]]
