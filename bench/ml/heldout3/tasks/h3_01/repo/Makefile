PY ?= python

train:
	$(PY) train.py --config configs/mlp.yaml --seed 0

eval:
	$(PY) eval.py --config configs/mlp.yaml --seed 0

all: train eval
