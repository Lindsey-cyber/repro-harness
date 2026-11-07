#!/usr/bin/env python3
"""Generate task.json and reference/ metadata from the reference logs (run from bench/repro)."""
import json, re, math, glob, os, shutil, statistics as st

def vals(pattern, regex, scale=1.0, flags=re.M):
    out = []
    for f in sorted(glob.glob(pattern)):
        t = open(f, errors="replace").read().replace("\r", "\n")
        m = re.findall(regex, t, flags)
        out.append(float(m[-1]) * scale if m else None)
    return out

def tol(value, paper_std, runs):
    n = len(runs); s = st.stdev(runs) if n > 1 else 0.0
    return max(0.01, 2 * (paper_std or 0.0), 2 * s / math.sqrt(n))

def expected(cid, value, paper_std, runs, why, seeds, extra=None):
    mean = sum(runs) / len(runs); s = st.stdev(runs) if len(runs) > 1 else 0.0
    t = tol(value, paper_std, runs)
    status = "reproduced" if abs(mean - value) <= t else "not_reproduced"
    e = {"claim": cid, "status": status,
         "reference": {"mean": round(mean, 4), "std": round(s, 4), "seeds": seeds, "values": [round(r, 4) for r in runs], "tol": round(t, 4)},
         "why": why}
    if extra: e.update(extra)
    return e

def claim(cid, table, ds, method, value, std, metric="accuracy"):
    return {"id": cid, "table": table, "dataset": ds, "method": method, "metric": metric, "value": value, "std": std}

def write(tid, task, freeze=True, patches=None, commands=""):
    d = f"tasks/{tid}"; os.makedirs(f"{d}/reference", exist_ok=True)
    json.dump(task, open(f"{d}/task.json", "w"), indent=2); open(f"{d}/task.json", "a").write("\n")
    if freeze: shutil.copy(".cache/freeze.txt", f"{d}/reference/pip_freeze.txt")
    open(f"{d}/reference/commands.md", "w").write(commands)
    if patches:
        open(f"{d}/reference/patches.tsv", "w").write("".join(f"{p}\t{r}\n" for p, r in patches.items()))

SEEDS = [0, 1, 2]
SITE = ("Environment: python 3.13, site dir built with `tools/sbx.sh bench/repro/.cache --site bench/repro/.cache/site-torch --net -- python3 -m pip install --target /opt "
        "torch --index-url https://download.pytorch.org/whl/cpu` then `... pip install --target /opt numpy scipy scikit-learn psutil pandas texttable tqdm` (see pip_freeze.txt). "
        "Runs: `tools/sbx.sh <workdir> --site <site> -- <cmd>` (no network, 1 thread, workdir = clone at repo.commit + patches).\n\n")
SGC_COMMIT = "2c7a2727e82e462d8ef9d6e57f0b08888e16488f"
SGC_URL = "https://github.com/Tiiiger/SGC"
SGC_TBL = "Table 2 of arXiv 1902.07153 (pdftotext line): 'SGC        81.0 ± 0.0      71.9 ± 0.1      78.9 ± 0.0' (columns Cora, Citeseer, Pubmed; caption: Test accuracy (%) averaged over 10 runs on citation networks)."

# ---- 1 SGC pristine
L = "tasks/sgc_citation_pristine/reference/logs/"
rc = vals(L+"cora_seed*.log", r"Test Accuracy: ([0-9.]+)"); rs = vals(L+"citeseer_seed*.log", r"Test Accuracy: ([0-9.]+)"); rp = vals(L+"pubmed_seed*.log", r"Test Accuracy: ([0-9.]+)")
t = {"id": "sgc_citation_pristine", "variant": "pristine", "paper": {"arxiv": "1902.07153", "title": "Simplifying Graph Convolutional Networks"},
     "repo": {"url": SGC_URL, "commit": SGC_COMMIT},
     "objective": "Reproduce the SGC test accuracy on Cora, Citeseer and Pubmed reported in Table 2 of the paper, using this repository's tuned-hyperparameter commands for the citation networks.",
     "claims": [claim("cora_acc", "Table 2", "Cora", "SGC", 0.810, 0.0), claim("citeseer_acc", "Table 2", "Citeseer", "SGC", 0.719, 0.001), claim("pubmed_acc", "Table 2", "Pubmed", "SGC", 0.789, 0.0)],
     "expected": [
        expected("cora_acc", 0.810, 0.0, rc, "Runs as documented (`citation.py --dataset cora --tuned`) once scikit-learn is installed; the repo's requirements file cannot be installed as written. Accuracy is essentially seed independent.", SEEDS),
        expected("citeseer_acc", 0.719, 0.001, rs, "README command uses `--epochs 150` for Citeseer; with it the number matches.", SEEDS),
        expected("pubmed_acc", 0.789, 0.0, rp, "Matches as documented.", SEEDS)],
     "root_causes": ["env.missing_dependency"], "fix_check": None,
     "notes": SGC_TBL + " README: `pip install -r requirement.txt` (the file is named requirements.txt) which pins networkx==1.11 and hyperopt==0.1.1 (not installable on python 3.13); scikit-learn is imported by metrics.py. Fix = install scikit-learn; no source patch needed (networkx 3.x and torch 2.x work as is, with deprecation warnings). Seeds are passed with `--seed`."}
write("sgc_citation_pristine", t, commands=SITE + "Commands (per seed S in 0 1 2):\n- python3 citation.py --dataset cora --tuned --seed S\n- python3 citation.py --dataset citeseer --tuned --epochs 150 --seed S\n- python3 citation.py --dataset pubmed --tuned --seed S\nEnvironment fix: pip install scikit-learn (no patches).\n")

# ---- SGC injected
def sgc_inj(tid, label, rc_list, buggy_hint, why_bug, extra_notes):
    L = f"tasks/{tid}/reference/logs/"
    fixed = vals(L+"fixed_seed*.log", r"Test Accuracy: ([0-9.]+)"); bug = vals(L+"buggy_seed*.log", r"Test Accuracy: ([0-9.]+)")
    pristine_mean = sum(rc) / len(rc)
    e = expected("cora_acc", 0.810, 0.0, fixed, "After the defect is found and removed the pristine behaviour is restored and the number matches.", SEEDS,
                 {"buggy": {"values": [b for b in bug], "note": buggy_hint}})
    t = {"id": tid, "variant": "injected", "paper": {"arxiv": "1902.07153", "title": "Simplifying Graph Convolutional Networks"},
         "repo": {"url": SGC_URL, "commit": SGC_COMMIT},
         "objective": "Reproduce the SGC test accuracy on Cora reported in Table 2 of the paper, using this repository's tuned-hyperparameter command for Cora.",
         "claims": [claim("cora_acc", "Table 2", "Cora", "SGC", 0.810, 0.0)], "expected": [e], "root_causes": rc_list,
         "fix_check": {"command": ["python3", "citation.py", "--dataset", "cora", "--tuned", "--seed", "{seed}"], "parse": "Test Accuracy: ([0-9.]+)", "claim": "cora_acc",
                       "pristine_reference_mean": round(pristine_mean, 4)},
         "notes": SGC_TBL + " `{seed}` in fix_check.command is the placeholder for each reference seed. Expected status is the status after the injected defect is correctly diagnosed and removed; the buggy numbers are recorded under expected[].buggy. " + extra_notes}
    write(tid, t, commands=SITE + "Build: clone at repo.commit, `git apply inject.patch`.\nBuggy: python3 citation.py --dataset cora --tuned --seed S\nFixed: `git apply reference/fix.patch` (reverse of inject.patch), same command.\nEnvironment fix as in sgc_citation_pristine (pip install scikit-learn).\n",
          patches={"fix.patch": "reverse of inject.patch (the reference fix)"})
sgc_inj("sgc_citation_inj_api_drift", "api", ["env.missing_dependency", "env.version_drift"], "run crashes before printing a metric (all seeds)", "", "Visible failure: crash on current numpy.")
sgc_inj("sgc_citation_inj_overlap", "overlap", ["env.missing_dependency", "data.train_test_overlap"], "inflated: test accuracy ~0.88 on all seeds", "", "Silent defect: number moves up and looks like a 'better than paper' result; validation accuracy is also inflated.")
sgc_inj("sgc_citation_inj_evalpath", "eval", ["env.missing_dependency", "pipeline.eval_path_mismatch"], "deflated: test accuracy ~0.70 on all seeds while validation accuracy still reads 0.794", "", "Silent defect: validation accuracy is unchanged (0.794), only the test number is wrong.")

# ---- 2 pygcn pristine
PY = "1600b5b748b3976413d1e307540ccc62605b4d6d"; PYU = "https://github.com/tkipf/pygcn"
L = "tasks/gcn_pygcn_pristine/reference/logs/"; rg = vals(L+"seed*.log", r"accuracy= ([0-9.]+)")
GCN_TBL = ("Table 2 of arXiv 1609.02907 (pdftotext): 'GCN (this paper)  70.3 (7s)  81.5 (4s)  79.0 (38s)  66.0 (48s)' and 'GCN (rand. splits)  67.9 ± 0.5  80.1 ± 0.5  78.9 ± 0.7  58.4 ± 1.7' "
           "(columns Citeseer, Cora, Pubmed, NELL). The claim here is the Cora entry of the 'GCN (rand. splits)' row.")
t = {"id": "gcn_pygcn_pristine", "variant": "pristine", "paper": {"arxiv": "1609.02907", "title": "Semi-Supervised Classification with Graph Convolutional Networks"},
     "repo": {"url": PYU, "commit": PY},
     "objective": "Reproduce, using this repository, the Cora test accuracy of the GCN in the 'GCN (rand. splits)' row of Table 2.",
     "claims": [claim("cora_rand_acc", "Table 2", "Cora", "GCN (rand. splits)", 0.801, 0.005)],
     "expected": [expected("cora_rand_acc", 0.801, 0.005, rg, "The code runs fine but is a re-implementation with its own fixed split (140 train / 300 val / 1000 test taken from cora.content order) and no random-split evaluation; its test accuracy (about 0.82-0.84) is above the paper's number and outside tolerance. The README itself says it is not intended for reproduction of the paper's results.", [0, 1, 2, 3, 4],
                  {"note": "runs vary by about 0.01-0.02 between repeats even with the same --seed (sparse ops), hence 5 seeds are recorded"})],
     "root_causes": ["protocol.mismatch"], "fix_check": None,
     "notes": GCN_TBL + " Run from the pygcn/ package directory with the repo root on PYTHONPATH (`cd pygcn && PYTHONPATH=/mnt python3 train.py --no-cuda --seed S`): data path is `../data/cora/` and the package is imported as `pygcn.*` (README: `python setup.py install`). No code patch needed. The headline 81.5 entry ('GCN (this paper)') is NOT used as a claim because the repo's mean (~0.825) sits about at the tolerance edge for it and would make the ground truth noisy."}
write("gcn_pygcn_pristine", t, commands=SITE + "Command (per seed S in 0..4): cd pygcn && PYTHONPATH=/opt:/mnt python3 train.py --no-cuda --seed S\nNo patches.\n")

# ---- pygcn resource
L = "tasks/gcn_pygcn_inj_resource/reference/logs/"; fx = vals(L+"fixed_seed*.log", r"accuracy= ([0-9.]+)")
e = expected("cora_rand_acc", 0.801, 0.005, fx, "With the hidden size back at the paper's 16 the repo behaves as in the pristine task: runs, but its protocol differs and the number is outside tolerance.", SEEDS,
             {"buggy": {"note": "documented defaults do not finish: 2 epochs take ~26 s (13 s/epoch, 200 epochs ~ 43 min, limit 15 min); 120 s timeout log in logs/buggy_timeout120s.log"}})
t = {"id": "gcn_pygcn_inj_resource", "variant": "injected", "paper": {"arxiv": "1609.02907", "title": "Semi-Supervised Classification with Graph Convolutional Networks"},
     "repo": {"url": PYU, "commit": PY},
     "objective": "Reproduce, using this repository, the Cora test accuracy of the GCN in the 'GCN (rand. splits)' row of Table 2.",
     "claims": [claim("cora_rand_acc", "Table 2", "Cora", "GCN (rand. splits)", 0.801, 0.005)], "expected": [e],
     "root_causes": ["run.resource_limit", "protocol.mismatch"],
     "fix_check": {"command": ["sh", "-c", "cd pygcn && PYTHONPATH=/opt:/mnt python3 train.py --no-cuda --seed {seed}"], "parse": "accuracy= ([0-9.]+)", "claim": "cora_rand_acc",
                   "pristine_reference_mean": round(sum(rg) / len(rg), 4)},
     "notes": GCN_TBL + " Resource defect: the documented configuration (defaults) exceeds the 15 minute / CPU limit; the paper's setting (16 hidden units) is cheap. Expected status is for the reasonable setting. A system that reports not_runnable without trying the paper's hyperparameter has missed the point. `{seed}` is the per-seed placeholder. Runs are slightly non-deterministic (see pristine task)."}
write("gcn_pygcn_inj_resource", t, commands=SITE + "Build: clone at repo.commit, `git apply inject.patch`.\nBuggy: cd pygcn && PYTHONPATH=/opt:/mnt python3 train.py --no-cuda  (does not finish in limit)\nFixed: git apply reference/fix.patch; same command with --seed S.\n", patches={"fix.patch": "reverse of inject.patch (the reference fix)"})

# ---- 3 fashion
L = "tasks/fashion_mnist_sklearn_pristine/reference/logs/run*.log"
def fm(name): return vals(L, r'done! acc: ([0-9.]+) \(\+/- [0-9.]+\) repeated: \d+ classifier: "%s"' % name)
FT = "Table 3 of arXiv 1708.07747 (pdftotext), columns Fashion / MNIST: 'DecisionTreeClassifier criterion=entropy max_depth=10 splitter=best 0.798 0.873'; 'ExtraTreeClassifier criterion=gini max_depth=10 splitter=best 0.775 0.806'; 'GaussianNB priors=[0.1, ...] 0.511 0.524'. Text: 'All algorithms are repeated 5 times by shuffling the training data and the average accuracy on the test set is reported.'"
t = {"id": "fashion_mnist_sklearn_pristine", "variant": "pristine", "paper": {"arxiv": "1708.07747", "title": "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms"},
     "repo": {"url": "https://github.com/zalandoresearch/fashion-mnist", "commit": "b2617bb6d3ffa2e429640350f613e3291e10b141"},
     "objective": "Using the repository's benchmark runner (benchmark/runner.py and benchmark/baselines.json), reproduce the Fashion-MNIST test accuracy of three entries of Table 3: DecisionTreeClassifier (entropy, max_depth=10, splitter=best), ExtraTreeClassifier (gini, max_depth=10, splitter=best) and GaussianNB (uniform priors).",
     "claims": [claim("dtree_entropy_d10_best", "Table 3", "Fashion-MNIST", "DecisionTreeClassifier criterion=entropy max_depth=10 splitter=best", 0.798, None),
                claim("extratree_gini_d10_best", "Table 3", "Fashion-MNIST", "ExtraTreeClassifier criterion=gini max_depth=10 splitter=best", 0.775, None),
                claim("gnb_uniform_priors", "Table 3", "Fashion-MNIST", "GaussianNB priors=[0.1]*10", 0.511, None)],
     "expected": [expected("dtree_entropy_d10_best", 0.798, None, fm("DecisionTreeClassifier"), "Deterministic 0.811 with current scikit-learn (tree-building changes since 2017); 0.013 above the paper, just outside the 0.01 tolerance.", SEEDS),
                  expected("extratree_gini_d10_best", 0.775, None, fm("ExtraTreeClassifier"), "Within tolerance (mean about 0.78) despite training-set shuffling noise.", SEEDS),
                  expected("gnb_uniform_priors", 0.511, None, fm("GaussianNB"), "Current scikit-learn gives 0.570 (var_smoothing/numerics differ from the 2017 version); the paper's 0.511 is not reached.", SEEDS)],
     "root_causes": ["env.version_drift", "repro.claim_not_met"], "fix_check": None,
     "notes": FT + " Seeds: the runner has no seed option; each of the 3 'seeds' is one invocation of the runner, which itself repeats each configuration twice with shuffled training data (the reported value is that run's mean). The runner never exits by itself (worker loop blocks on the job queue) so reference runs are cut with `timeout 120`; results are read from the 'done!' log lines. Fixes: time.clock was removed in python 3.8 (the runner catches the AttributeError per job and logs 'failed', so every job fails silently) -> 01; the unrestricted baselines.json holds 129 configurations including GaussianProcessClassifier and SVC on 60000 samples which cannot finish in the limits, so the job list is restricted to the three claimed configurations -> 02 (a system may equally write its own driver). Other deprecations (PassiveAggressiveClassifier, loss names) only matter for configs outside the claims."}
write("fashion_mnist_sklearn_pristine", t, patches={"01_time_clock_removed.patch": "time.clock() no longer exists in python>=3.8; replaced by time.perf_counter()", "02_restrict_baselines_to_claimed_configs.patch": "full 129-config baselines cannot run on CPU in limits; keep only the three configs behind the claims"},
      commands=SITE + "Command (per run i in 0 1 2, fresh clone + both patches): timeout 120 python3 benchmark/runner.py   (see tools/fashion_run.sh); parse 'done! acc: X ... classifier: \"Name\"' lines.\n")

# ---- 4 dragen
L = "tasks/gcn_dragen1860_pristine/reference/logs/"; r4 = vals(L+"epochs400_seed*.log", r"^test: ([0-9.]+)"); r2 = vals(L+"epochs200_seed*.log", r"^test: ([0-9.]+)")
t = {"id": "gcn_dragen1860_pristine", "variant": "pristine", "paper": {"arxiv": "1609.02907", "title": "Semi-Supervised Classification with Graph Convolutional Networks"},
     "repo": {"url": "https://github.com/dragen1860/GCN-PyTorch", "commit": "623aa67f0d1c3384fe56f3a20e14e2c13845bcd9"},
     "objective": "Reproduce the Cora test accuracy of the GCN in Table 2 ('GCN (this paper)') using this PyTorch port, following its README.",
     "claims": [claim("cora_acc", "Table 2", "Cora", "GCN (this paper)", 0.815, None)],
     "expected": [expected("cora_acc", 0.815, None, r4, "Reproduced only after (a) three drift/hardware fixes so that it runs on CPU with current scipy/numpy and (b) using 400 epochs as the README hint says ('To achieve 81.7% on Cora dataset, pytorch need 400 epochs'). With the default 200 epochs the five-seed mean is about 0.79 with large variance (0.73-0.82).", [0, 1, 2, 3, 4],
                  {"default_200_epochs": {"values": [round(x, 4) for x in r2], "mean": round(sum(r2)/len(r2), 4)}})],
     "root_causes": ["env.version_drift", "other:hardcoded_cuda", "env.path_or_config"], "fix_check": None,
     "notes": GCN_TBL.replace(" The claim here is the Cora entry of the 'GCN (rand. splits)' row.", " The claim here is the Cora entry of the 'GCN (this paper)' row: 81.5.") + " Fixes: scipy.sparse.linalg.eigen.arpack import removed (01); np.bool/np.int removed from numpy (02); torch.device('cuda') hard-coded (03). env.path_or_config denotes the wrong default epochs (200 instead of the 400 the README says are needed). The seed in train.py is hard-coded (123) and the default run gave 0.572 on this setup; patch 90 is only a measurement aid adding a --seed flag so that >= 3 seeds can be run, it is not a root cause. Reference command: python3 train.py --epochs 400 --seed S."}
write("gcn_dragen1860_pristine", t, patches={"01_scipy_eigsh_import.patch": "scipy.sparse.linalg.eigen.arpack was removed; import eigsh from scipy.sparse.linalg", "02_numpy_removed_aliases.patch": "np.bool / np.int aliases were removed in numpy>=1.24", "03_cpu_device_fallback.patch": "device was hard-coded to cuda; fall back to cpu", "90_measurement_aid_seed_flag.patch": "measurement aid only: expose the hard-coded seed as --seed to run several seeds"},
      commands=SITE + "Command (per seed S in 0..4, patches 01-03,90 applied): python3 train.py --epochs 400 --seed S ; parse '^test: X'. Default-config logs: python3 train.py --seed S (200 epochs).\n")

# ---- 5 gcnii
L = "tasks/gcnii_cora_pristine/reference/logs/"; r5 = vals(L+"seed*.log", r"Test acc\.:([0-9.]+)", 0.01)
t = {"id": "gcnii_cora_pristine", "variant": "pristine", "paper": {"arxiv": "2007.02133", "title": "Simple and Deep Graph Convolutional Networks"},
     "repo": {"url": "https://github.com/chennnM/GCNII", "commit": "ca91f5686c4cd09cc1c6f98431a5d5b7e36acc92"},
     "objective": "Reproduce the semi-supervised Cora test accuracy of GCNII (64 layers) reported in Table 2, using the repository's semi.sh command for Cora.",
     "claims": [claim("cora_acc", "Table 2", "Cora", "GCNII (64 layers)", 0.855, 0.005)],
     "expected": [expected("cora_acc", 0.855, 0.005, r5, "Reproduces after making the device selection CPU-capable; about 8-10 minutes per seed on one CPU thread.", SEEDS)],
     "root_causes": ["other:hardcoded_cuda"], "fix_check": None,
     "notes": "Table 2 of arXiv 2007.02133 (pdftotext): 'GCNII  85.5 ± 0.5 (64)  73.4 ± 0.6 (32)  80.2 ± 0.4 (16)' (Cora, Citeseer, Pubmed; accuracy %). semi.sh first line: `python -u train.py --data cora --layer 64 --test`. Test accuracy is printed as 'Test acc.:85.1' (percent). torch.device('cuda:0') is hard-coded -> patch 01. Reference command: python3 train.py --data cora --layer 64 --test --seed S."}
write("gcnii_cora_pristine", t, patches={"01_cpu_device_fallback.patch": "device hard-coded to cuda:<id>; fall back to cpu when CUDA is unavailable"}, commands=SITE + "Command (per seed S in 0 1 2, patch 01): python3 train.py --data cora --layer 64 --test --seed S ; parse 'Test acc.:X' (percent).\n")

# ---- 6 appnp
L = "tasks/appnp_pyg_port_pristine/reference/logs/"; rx = vals(L+"exact_seed*.log", r"Test accuracy: ([0-9.]+)"); ra = vals(L+"approx_seed*.log", r"Test accuracy: ([0-9.]+)")
t = {"id": "appnp_pyg_port_pristine", "variant": "pristine", "paper": {"arxiv": "1810.05997", "title": "Predict then Propagate: Graph Neural Networks meet Personalized PageRank"},
     "repo": {"url": "https://github.com/benedekrozemberczki/APPNP", "commit": "338d98980ee8748e652630dd422341ef8b2a354a"},
     "objective": "Using this PyTorch implementation on its bundled Cora data, reproduce the Cora-ML test accuracy of PPNP (exact model) and APPNP (approximate model) in Table 2.",
     "claims": [claim("ppnp_cora_ml", "Table 2", "Cora-ML", "PPNP", 0.8529, 0.0025), claim("appnp_cora_ml", "Table 2", "Cora-ML", "APPNP", 0.8509, 0.0025)],
     "expected": [expected("ppnp_cora_ml", 0.8529, 0.0025, rx, "Runs (after replacing torch_sparse) but the repo uses plain Cora, a random 1500/500/rest split with 1500 labelled nodes, not the paper's Cora-ML with 20 labels per class; accuracy is about 0.90.", SEEDS),
                  expected("appnp_cora_ml", 0.8509, 0.0025, ra, "Same protocol difference, with `--model approximate`; about 0.89.", SEEDS)],
     "root_causes": ["env.missing_dependency", "protocol.mismatch"], "fix_check": None,
     "notes": "Table 2 of arXiv 1810.05997 (pdftotext): 'PPNP*  75.83 ± 0.27  85.29 ± 0.25  -' and 'APPNP  75.73 ± 0.30  85.09 ± 0.25  79.73 ± 0.31' (columns Citeseer, Cora-ML, PubMed, MS Academic; average accuracy %, 95% confidence by bootstrapping; the Cora-ML column is the second). The repo's README requires torch-scatter/torch-sparse/torch-geometric 2019 builds; torch_sparse cannot be built here, and only `spmm` is used -> patch 01 reimplements it with torch.sparse.mm. Run from the repo root: python3 src/main.py --model exact|approximate --seed S (default model is exact = PPNP; ~4 min/seed exact, ~1 min approximate)."}
write("appnp_pyg_port_pristine", t, patches={"01_replace_torch_sparse_spmm.patch": "torch_sparse (needed only for spmm) is not installable on current torch; reimplement spmm with torch.sparse.mm"}, commands=SITE + "Commands (seed S in 0 1 2, patch 01): python3 src/main.py --model exact --seed S ; python3 src/main.py --model approximate --seed S ; parse 'Test accuracy: X'.\n")
print("ok")
