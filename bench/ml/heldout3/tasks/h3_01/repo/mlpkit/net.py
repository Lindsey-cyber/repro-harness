import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class MLP:
    def __init__(self, d_in, d_h, rng):
        self.params = [rng.normal(0, 1 / np.sqrt(d_in), (d_in, d_h)), np.zeros(d_h),
                       rng.normal(0, 1 / np.sqrt(d_h), (d_h, 1)), np.zeros(1)]

    @staticmethod
    def run(params, X):
        W1, b1, W2, b2 = params
        h = np.tanh(X @ W1 + b1)
        return h, (h @ W2 + b2).ravel()

    def predict_proba(self, X):
        return _sigmoid(self.run(self.params, X)[1])

    def step(self, X, y, lr, wd):
        W1, b1, W2, b2 = self.params
        h, z = self.run(self.params, X)
        dz = (_sigmoid(z) - y)[:, None] / len(y)
        dh = (dz @ W2.T) * (1 - h ** 2)
        grads = [X.T @ dh, dh.sum(0), h.T @ dz, dz.sum(0)]
        for P, G in zip(self.params, grads):
            P -= lr * (G + wd * P)
