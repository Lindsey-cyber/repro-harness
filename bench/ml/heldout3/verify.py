#!/usr/bin/env python3
"""Run each task per its README from a scratch copy; print metrics. usage: verify.py SCRATCH [ids...] [--fixed]"""
import json, os, shutil, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv, args = sys.argv[:1], sys.argv[1:]
import importlib.util
spec = importlib.util.spec_from_file_location("mk", os.path.join(HERE, "make.py")); mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
scratch = args[0]; fixed = "--fixed" in args; ids = [a for a in args[1:] if not a.startswith("--")] or [t for t, _ in mk.TASKS]
env = dict(os.environ)
for t in ids:
    if fixed and t not in mk.DEFECTIVE: continue
    if fixed:
        src = os.path.join(scratch, "_fixed_src", t)
        subprocess.run([sys.executable, os.path.join(HERE, "make.py"), "--fixed", os.path.join(scratch, "_fixed_src")], check=True)
    else:
        src = os.path.join(HERE, "tasks", t, "repo")
    wd = os.path.join(scratch, ("fx_" if fixed else "bg_") + t)
    shutil.rmtree(wd, ignore_errors=True); shutil.copytree(src, wd)
    cmds, files = mk.RUN[t]
    for c in cmds:
        r = subprocess.run(c, shell=True, cwd=wd, capture_output=True, text=True, env=env, timeout=60)
        if r.returncode: print(t, "FAILED", c, r.stderr[-500:]); break
    for f in files:
        p = os.path.join(wd, f)
        print(t, "fixed" if fixed else "buggy", f, json.dumps(json.load(open(p))) if os.path.exists(p) else "MISSING")
