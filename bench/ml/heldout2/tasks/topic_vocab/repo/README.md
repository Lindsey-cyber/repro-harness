# topic-nb

Multinomial naive Bayes topic classifier over a capped bag-of-words vocabulary (the 200 most
frequent tokens in the training split).

## Run

    python run_nb.py

Writes `metrics/nb.json`.

## Result

Test accuracy: **1.000** (80/20 split by document order).
