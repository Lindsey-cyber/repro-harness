import numpy as np
import pandas as pd

REGIONS = ["coast", "hills", "inland", "metro"]
NUMF = ["sqft_z", "age_z", "rooms_z", "dist_z"]


def design(df):
    oh = (df.region.to_numpy()[:, None] == np.array(REGIONS)[None, :]).astype(float)
    return np.c_[df[NUMF].to_numpy(float), oh]


def run_seed(df, seed, n_features, ridge, val_frac):
    rng = np.random.default_rng(seed)
    X, y = design(df), df.price.to_numpy(float)
    idx = rng.permutation(len(df)); nv = int(len(df) * val_frac)
    va, tr = idx[:nv], idx[nv:]
    mu, sd = X[tr].mean(0), X[tr].std(0)
    X = (X - mu) / sd
    R = rng.normal(0, 1 / np.sqrt(X.shape[1]), (X.shape[1], n_features)); c = rng.uniform(-1, 1, n_features)
    H = np.tanh(X @ R + c)
    A = H[tr].T @ H[tr] + ridge * np.eye(n_features)
    w = np.linalg.solve(A, H[tr].T @ (y[tr] - y[tr].mean()))
    pred = H[va] @ w + y[tr].mean()
    rmse = float(np.sqrt(np.mean((pred - y[va]) ** 2)))
    r2 = float(1 - np.sum((pred - y[va]) ** 2) / np.sum((y[va] - y[va].mean()) ** 2))
    return {"val_rmse": rmse, "val_r2": r2}
