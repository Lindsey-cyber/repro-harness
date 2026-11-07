import csv, json, os
import numpy as np

SEED = int(os.environ.get("SEED", "0"))
EPOCHS, LR, BATCH = 30, 0.1, 32


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = [c for c in rows[0] if c != "label"]
    X = np.array([[float(r[c]) for c in cols] for r in rows])
    y = np.array([float(r["label"]) for r in rows])
    return X, y


def features(X):
    return X


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


rng = np.random.default_rng(SEED)
X_tr, y_tr = load("data/train.csv")
X_te, y_te = load("data/test.csv")
F_tr, F_te = features(X_tr), features(X_te)
mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-8
Z_tr = (F_tr - mu) / sd
Z_te = (F_te - mu) / sd

w = rng.normal(0, 0.01, Z_tr.shape[1])
b = 0.0
os.makedirs("outputs", exist_ok=True)
for epoch in range(EPOCHS):
    order = rng.permutation(len(y_tr))
    for i in range(0, len(order), BATCH):
        j = order[i:i + BATCH]
        g = sigmoid(Z_tr[j] @ w + b) - y_tr[j]
        w -= LR * (Z_tr[j].T @ g / len(j))
        b -= LR * g.mean()
np.savez("outputs/model.npz", w=w, b=b, mu=mu, sd=sd)

score = sigmoid(Z_te @ w + b)
pred = (score >= 0.5).astype(float)
acc = float((pred[1:] == y_te[:-1]).mean())
train_acc = float(((sigmoid(Z_tr @ w + b) >= 0.5) == y_tr).mean())
with open("outputs/metrics.json", "w") as f:
    json.dump({"accuracy": acc, "train_accuracy": train_acc, "epochs": EPOCHS}, f)
with open("outputs/predictions.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["y_true", "y_pred", "score"])
    wr.writerows(zip(y_te, pred, score))
print(f"test accuracy {acc:.4f} (train {train_acc:.4f})")
