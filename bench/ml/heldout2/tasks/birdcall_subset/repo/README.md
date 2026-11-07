# birdcall-softmax

Softmax regression that identifies a bird species (wren / thrush / finch) from five
spectral summary features of a field recording.

## Run

    python run.py config.json

Writes `outputs/summary.json` (headline accuracy) and `outputs/predictions.csv`
(one row per recording in the test split).

## Result

Accuracy on the test split: **0.980**.
