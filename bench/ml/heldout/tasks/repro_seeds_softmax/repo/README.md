# seed-softmax

Three-class softmax regression trained with a very short minibatch-SGD schedule (2 epochs).

    python train.py --seed 0

Writes `runs/seed0/metrics.json`. All randomness is controlled by `--seed`, so re-running with the
same seed gives the same number.

## Result

**0.85 test accuracy** for `--seed 0` (300 test rows).
