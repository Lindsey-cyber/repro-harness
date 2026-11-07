import json, os
import numpy as np
import pandas as pd
from common import design, labels

m = np.load("runs/v1/model.npz")
test = pd.read_csv("data/test.csv")
vocab = list(dict.fromkeys(test.region))
X, _, _ = design(test, vocab, m["mu"], m["sd"])
y = labels(test)
s = X @ m["w"]
os.makedirs("runs/v1/eval", exist_ok=True)
pd.DataFrame({"device_id": test.device_id, "score": np.round(s, 4), "pred": np.where(s > 0, "fail", "ok"),
              "status": test.status}).to_csv("runs/v1/eval/predictions.csv", index=False)
res = {"accuracy": round(float(np.mean((s > 0) == (y == 1))), 4), "n": len(y)}
json.dump(res, open("runs/v1/eval/metrics.json", "w"), indent=1)
print(res)
