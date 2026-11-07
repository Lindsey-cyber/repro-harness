Environment: python 3.13, site dir built with `tools/sbx.sh bench/repro/.cache --site bench/repro/.cache/site-torch --net -- python3 -m pip install --target /opt torch --index-url https://download.pytorch.org/whl/cpu` then `... pip install --target /opt numpy scipy scikit-learn psutil pandas texttable tqdm` (see pip_freeze.txt). Runs: `tools/sbx.sh <workdir> --site <site> -- <cmd>` (no network, 1 thread, workdir = clone at repo.commit + patches).

Build: clone at repo.commit, `git apply inject.patch`.
Buggy: python3 citation.py --dataset cora --tuned --seed S
Fixed: `git apply reference/fix.patch` (reverse of inject.patch), same command.
Environment fix as in sgc_citation_pristine (pip install scikit-learn).
