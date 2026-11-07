import numpy as np


class Logit:
    def __init__(self, lr=0.1, steps=300):
        self.lr, self.steps = lr, steps

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        for _ in range(self.steps):
            p = 1 / (1 + np.exp(-(X @ self.w + self.b)))
            self.w -= self.lr * X.T @ (p - y) / len(y)
            self.b -= self.lr * np.mean(p - y)
        return self

    def predict(self, X):
        return (1 / (1 + np.exp(-(X @ self.w + self.b))) > 0.5).astype(int)
