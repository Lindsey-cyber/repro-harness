"""Fake Modal runner for offline end-to-end and failure-injection tests (same JSON contract as
src/mlh/backends/modal_runner.py). Commands run locally on a copy of a trusted synthetic
repository; /work and /cache are mapped to a temporary directory and MLH_FAKE_CACHE.

MLH_FAKE_FAULTS: JSON list of {"when": regex on the command, "nth": [k, ...] (1-based matching
call numbers), "error": text}; matching calls fail with that error instead of running.
MLH_FAKE_ELAPSED: seconds reported per call (default 60), so costs are deterministic.
"""
import json, os, re, shutil, subprocess, sys, tempfile, time

def main():
    req = json.loads(sys.stdin.read())
    cache = os.environ["MLH_FAKE_CACHE"]
    elapsed = float(os.environ.get("MLH_FAKE_ELAPSED", "60"))
    if req["op"] == "check":
        return {"ok": True, "app": "fake"}
    if req["op"] != "run":
        return {"ok": False, "error": f"fake runner: op {req['op']} not supported"}
    cmd = req["cmd"]
    text = " ".join(cmd)
    if "MLH_PROBE" in text:
        real = {"2.5.1-cuda12.4": ("3.11.10", "2.5.1"), "2.1.2-cuda12.1": ("3.10.13", "2.1.2"), "1.13.1-cuda11.6": ("3.10.8", "1.13.1"),
                "1.12.1-cuda11.3": ("3.7.13", "1.12.1"), "1.9.1-cuda11.1": ("3.7.11", "1.10.0"), "1.7.1-cuda11.0": ("3.8.5", "1.7.1")}
        v = next((v for k, v in real.items() if k in (req.get("image") or "")), None)
        return {"ok": True, "exit_code": 0 if v else 1, "stdout": f"MLH_PROBE py={v[0]} torch={v[1]}\n" if v else "", "stderr": "", "elapsed_s": 5, "files": {}}
    os.makedirs(cache, exist_ok=True)
    counts_file = os.path.join(cache, ".fake-counts.json")
    counts = json.load(open(counts_file)) if os.path.exists(counts_file) else {}
    for i, f in enumerate(json.loads(os.environ.get("MLH_FAKE_FAULTS", "[]"))):
        if re.search(f["when"], text):
            k = counts.get(str(i), 0) + 1
            counts[str(i)] = k
            json.dump(counts, open(counts_file, "w"))
            if k in f.get("nth", [1]):
                if "exit" in f:  # the command ran and failed (OOM, timeout): a result, not a platform error
                    return {"ok": True, "exit_code": f["exit"], "stdout": "", "stderr": f.get("stderr", ""), "elapsed_s": elapsed, "files": {}}
                return {"ok": False, "error": f["error"], "elapsed_s": elapsed}
    with open(os.path.join(cache, ".fake-calls.jsonl"), "a") as log:
        log.write(json.dumps({"cmd": cmd, "gpu": req.get("gpu"), "image": req.get("image"), "pip": req.get("pip"), "volume": req.get("cache_volume"), "system": req.get("system"), "memory_mb": req.get("memory_mb"), "links": req.get("links")}) + "\n")
    work = tempfile.mkdtemp(prefix="fake-work-")
    try:
        shutil.copytree(req["repo"], work, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git"))
        m = lambda s: s.replace("/work", work).replace("/cache", cache)
        for r in req.get("restore") or []:
            src, dst = m(r["from"]), os.path.join(work, r["to"])
            if os.path.exists(src):
                shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst) if os.path.isdir(src) else shutil.copy(src, dst)
        for l in req.get("links") or []:
            dst = os.path.join(work, l["path"]); os.makedirs(os.path.dirname(dst) or work, exist_ok=True)
            if os.path.lexists(dst): os.remove(dst) if not os.path.isdir(dst) or os.path.islink(dst) else shutil.rmtree(dst)
            os.symlink(m(l["to"]), dst)
        env = {**os.environ, **{k: m(str(v)) for k, v in (req.get("env") or {}).items()}}
        argv = [m(a) for a in cmd]
        if len(argv) > 1 and argv[1] == "fetch_data.py":
            # The real fetch script, with downloads served from MLH_FAKE_DATA ({url: text}) and markers in the fake cache.
            env["MLH_FETCH_MARKS"] = os.path.join(cache, ".mlh-fetch")
            shim = ("import io, json, os, runpy, sys, urllib.request\n"
                    "data = json.loads(os.environ.get('MLH_FAKE_DATA', '{}'))\n"
                    "class R(io.BytesIO):\n    def __enter__(self): return self\n    def __exit__(self, *a): return False\n"
                    "def urlopen(req, timeout=None):\n    url = req.full_url if hasattr(req, 'full_url') else req\n"
                    "    if url not in data: raise OSError('fake: no such url ' + url)\n    return R(data[url].encode())\n"
                    "urllib.request.urlopen = urlopen\nsys.argv = sys.argv[1:]\nrunpy.run_path('fetch_data.py', run_name='__main__')\n")
            argv = [argv[0], "-c", shim] + argv[1:]
        try:
            p = subprocess.run(argv, cwd=work, env=env, capture_output=True, text=True, timeout=int(req.get("timeout_s", 600)))
            code, out, err = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            code, out, err = 124, "", "timeout"
        for r in req.get("persist") or []:
            src, dst = os.path.join(work, r["from"]), m(r["to"])
            if os.path.exists(src):
                shutil.rmtree(dst, ignore_errors=True); os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copytree(src, dst) if os.path.isdir(src) else shutil.copy(src, dst)
        files = {}
        for rel in req.get("outputs") or []:
            pth = os.path.join(work, rel)
            if os.path.exists(pth):
                files[rel] = open(pth).read()[-200000:]
        return {"ok": True, "exit_code": code, "stdout": out, "stderr": err, "elapsed_s": elapsed, "files": files}
    finally:
        shutil.rmtree(work, ignore_errors=True)

print(json.dumps(main()))
