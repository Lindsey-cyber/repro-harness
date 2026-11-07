Environment: python 3.13, site dir built with `tools/sbx.sh bench/repro/.cache --site bench/repro/.cache/site-torch --net -- python3 -m pip install --target /opt torch --index-url https://download.pytorch.org/whl/cpu` then `... pip install --target /opt numpy scipy scikit-learn psutil pandas texttable tqdm` (see pip_freeze.txt). Runs: `tools/sbx.sh <workdir> --site <site> -- <cmd>` (no network, 1 thread, workdir = clone at repo.commit + patches).

Commands (per seed S in 0 1 2):
- python3 citation.py --dataset cora --tuned --seed S
- python3 citation.py --dataset citeseer --tuned --epochs 150 --seed S
- python3 citation.py --dataset pubmed --tuned --seed S
Environment fix: pip install scikit-learn (no patches).
