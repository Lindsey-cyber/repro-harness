# signal-mlp

MLP (40 inputs, 96 tanh units) on a noisy sensor-classification set. `data/signals.csv` carries an
explicit `split` column (train 150 / val 150 / test 700).

    python scripts/train.py                  # trains 400 epochs, keeps the best-validation weights
    python scripts/eval_ckpt.py              # reloads checkpoints/best.npz and re-scores val/test

Logs go to `logs/` (`train_log.csv` per epoch, `final.json`).

## Result

Selecting the epoch by validation accuracy gives **0.7343 test accuracy** (`logs/final.json`);
`eval_ckpt.py` on the saved checkpoint is expected to match.
