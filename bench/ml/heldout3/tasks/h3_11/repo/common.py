import numpy as np
import pandas as pd

NUM = ["temp_z", "load_z", "age_z"]


def design(df, vocab, mu=None, sd=None):
    num = df[NUM].to_numpy(float)
    if mu is None:
        mu, sd = num.mean(0), num.std(0)
    oh = (df.region.to_numpy()[:, None] == np.array(vocab)[None, :]).astype(float)
    return np.c_[(num - mu) / sd, oh], mu, sd


def labels(df):
    return (df.status == "fail").to_numpy(float)
