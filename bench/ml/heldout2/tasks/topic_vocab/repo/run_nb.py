import json, os
from collections import Counter
import numpy as np
import pandas as pd

VOCAB_SIZE = 200


def main():
    df = pd.read_csv("data/posts.csv")
    n = int(len(df) * 0.8)
    tr, te = df.iloc[:n], df.iloc[n:]
    counts = Counter(t for doc in tr["text"] for t in doc.split())
    vocab = list({t for t, c in counts.items() if c >= 2})[:VOCAB_SIZE]
    col = {t: i for i, t in enumerate(vocab)}
    topics = sorted(df["topic"].unique())

    def bow(texts):
        M = np.zeros((len(texts), len(vocab)))
        for r, doc in enumerate(texts):
            for t in doc.split():
                if t in col:
                    M[r, col[t]] += 1
        return M

    Btr, Bte = bow(tr["text"]), bow(te["text"])
    logp = []
    for t in topics:
        c = Btr[(tr["topic"] == t).values].sum(0) + 1.0
        logp.append(np.log(c / c.sum()))
    prior = np.log(np.array([(tr["topic"] == t).mean() for t in topics]))
    pred = np.argmax(Bte @ np.array(logp).T + prior, axis=1)
    acc = float(np.mean(np.array(topics)[pred] == te["topic"].values))
    os.makedirs("metrics", exist_ok=True)
    json.dump({"accuracy": acc}, open("metrics/nb.json", "w"))
    print("accuracy %.4f" % acc)


if __name__ == "__main__":
    main()
