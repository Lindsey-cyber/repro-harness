# mlpkit: small MLP on a noisy tabular benchmark

Plain-numpy one-hidden-layer network trained with full-batch gradient descent. Model selection is by
validation accuracy; the checkpoint with the best validation accuracy is written to
`runs/<name>/seed<k>/best.npz`.

## Run

    make train      # trains, writes runs/mlp_h48/seed0/metrics.json
    make eval       # reloads best.npz and rescores the test split -> eval_metrics.json

Hyper-parameters are in `configs/mlp.yaml`. Data (`data/blobs.csv`) has a `split` column (train/val/test).

## Result

Test accuracy at the best-validation epoch: **TBD** (seed 0, see `metrics.json`).
`make eval` rescoring of the saved checkpoint should reproduce it.
