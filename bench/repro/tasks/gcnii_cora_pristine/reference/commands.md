Environment: python 3.13, site dir built with `tools/sbx.sh bench/repro/.cache --site bench/repro/.cache/site-torch --net -- python3 -m pip install --target /opt torch --index-url https://download.pytorch.org/whl/cpu` then `... pip install --target /opt numpy scipy scikit-learn psutil pandas texttable tqdm` (see pip_freeze.txt). Runs: `tools/sbx.sh <workdir> --site <site> -- <cmd>` (no network, 1 thread, workdir = clone at repo.commit + patches).

Command (per seed S in 0 1 2, patch 01): python3 train.py --data cora --layer 64 --test --seed S ; parse 'Test acc.:X' (percent).
