import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def init_params(d, h, seed):
    r = np.random.default_rng(seed)
    return {
        "W1": r.normal(0, 1 / np.sqrt(d), (d, h)),
        "b1": np.zeros(h),
        "W2": r.normal(0, 1 / np.sqrt(h), (h, 1)),
        "b2": np.zeros(1),
    }


def forward(p, X):
    H = np.tanh(X @ p["W1"] + p["b1"])
    return H, (H @ p["W2"] + p["b2"]).ravel()


def predict(p, X):
    return (forward(p, X)[1] > 0).astype(int)


def accuracy(p, X, y):
    return float((predict(p, X) == y).mean())


def sgd_step(p, X, y, lr, wd):
    H, z = forward(p, X)
    g = (sigmoid(z) - y) / len(y)
    dW2 = H.T @ g[:, None] + wd * p["W2"]
    db2 = np.array([g.sum()])
    dH = (g[:, None] @ p["W2"].T) * (1 - H ** 2)
    dW1 = X.T @ dH + wd * p["W1"]
    db1 = dH.sum(0)
    return {"W1": p["W1"] - lr * dW1, "b1": p["b1"] - lr * db1,
            "W2": p["W2"] - lr * dW2, "b2": p["b2"] - lr * db2}
