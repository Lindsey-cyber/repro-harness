Environment: python 3.13, site dir built with `tools/sbx.sh bench/repro/.cache --site bench/repro/.cache/site-torch --net -- python3 -m pip install --target /opt torch --index-url https://download.pytorch.org/whl/cpu` then `... pip install --target /opt numpy scipy scikit-learn psutil pandas texttable tqdm` (see pip_freeze.txt). Runs: `tools/sbx.sh <workdir> --site <site> -- <cmd>` (no network, 1 thread, workdir = clone at repo.commit + patches).

Build: clone at repo.commit, `git apply inject.patch`.
Buggy: cd pygcn && PYTHONPATH=/opt:/mnt python3 train.py --no-cuda  (does not finish in limit)
Fixed: git apply reference/fix.patch; same command with --seed S.
