#!/bin/bash
set -e
python -m tools.fit --config configs/base.json --set optim.lr=0.02 --out results/base
