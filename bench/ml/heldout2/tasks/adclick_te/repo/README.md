# adclick-lr

Click prediction for display ads with logistic regression, using a per-campaign click-rate
feature next to placement, hour and bid.

## Run

    python main.py --csv data/clicks.csv --out reports/run.json

## Result

Test accuracy: **0.703** (25% random hold-out, seed 4).
