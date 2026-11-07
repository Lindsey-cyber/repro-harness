import numpy as np


def predict(Xtr, ytr, Xte, k):
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    nn = np.argsort(d, axis=1)[:, :k]
    return (ytr[nn].mean(1) > 0.5).astype(int)
