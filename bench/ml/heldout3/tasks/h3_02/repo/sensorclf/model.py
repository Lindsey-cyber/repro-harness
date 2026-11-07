import numpy as np


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def fit(X, y, k, lr, epochs, l2=1e-3):
    W = np.zeros((X.shape[1], k)); b = np.zeros(k)
    Y = np.eye(k)[y]
    for _ in range(epochs):
        G = softmax(X @ W + b) - Y
        W -= lr * (X.T @ G / len(y) + l2 * W)
        b -= lr * G.mean(0)
    return W, b


def predict(W, b, X):
    return (X @ W + b).argmax(1)


def f1_macro(y, p, k):
    out = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(out))
