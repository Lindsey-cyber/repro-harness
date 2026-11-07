#!/bin/bash
set -e
mkdir -p out
python train.py --data data/tickets.csv --out out
python score.py out/preds.csv --out out/scores.json
