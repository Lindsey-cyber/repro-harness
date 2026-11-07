import csv, json
import numpy as np
from train_lib import load, features, sigmoid

m = np.load("outputs/model.npz")
X, y = load("data/test.csv")
Z = (features(X) - m["mu"]) / m["sd"]
pred = (sigmoid(Z @ m["w"] + m["b"]) >= 0.5).astype(float)
acc = float((pred == y).mean())
with open("outputs/eval_metrics.json", "w") as f:
    json.dump({"accuracy": acc}, f)
print(f"checkpoint test accuracy {acc:.4f}")
