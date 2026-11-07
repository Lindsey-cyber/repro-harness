# wq-sgd

Minibatch-SGD logistic regression predicting whether a wine is rated "good" from 11
physico-chemical measurements.

## Run

    python -m wq.fit --train data/train.csv --test data/test.csv --report out/report.json

## Result

Test accuracy: **0.742** (`test_accuracy` in `out/report.json`; 120 test wines).
