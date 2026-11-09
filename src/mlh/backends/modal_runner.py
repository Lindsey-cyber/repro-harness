"""Modal execution backend for the reproduction harness: one JSON request on stdin, one JSON result on stdout.

Untrusted repository code runs only inside a Modal Sandbox (an isolated container on Modal's
infrastructure), never on the harness machine. No credential is ever placed inside a sandbox.

Environment: a base image plus pip packages, built by Modal as a cached image layer (the network is
used while building, not while the experiment runs unless the request allows it).

State between stages lives in the persistent Volume "mlh-cache" mounted at /cache: `restore` copies
earlier stages' outputs into the working copy before the command, `persist` copies outputs out after.

ops:
  check  {} -> {app}
  run    {repo, cmd, gpu, timeout_s, network, env, pip, image, restore, persist, outputs}
         -> {exit_code, stdout, stderr, elapsed_s, files: {relpath: text}}
"""
import contextlib
import io
import json
import os
import shlex
import sys
import tarfile
import time

import modal

APP = "mlh-repro"
DEFAULT_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
_images = {}
CREATED_AT = None


def tar_dir(path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".mlh_in")]
            for f in files:
                full = os.path.join(root, f)
                if os.path.islink(full) or os.path.getsize(full) > 100 * 1024 * 1024:
                    continue  # large artifacts belong in the cache volume, not the upload
                t.add(full, arcname=os.path.relpath(full, path))
    return buf.getvalue()


def image_for(base, pip, system=()):
    key = (base, tuple(pip), tuple(system))
    if key not in _images:
        img = modal.Image.from_registry(base)
        # System packages and toolchains the environment declared (e.g. a C++ compiler): image layers,
        # cached by Modal, so every stage and smoke test gets the same environment without re-installing.
        if system:
            img = img.run_commands(*system)
        # torch is already in the default base image; other torch-family wheels come from the CUDA index.
        if pip:
            img = img.pip_install(*pip)
        _images[key] = img
    return _images[key]


def op_check(_req):
    app = modal.App.lookup(APP, create_if_missing=True)
    return {"app": app.app_id}


def op_run(req):
    app = modal.App.lookup(APP, create_if_missing=True)
    image = image_for(req.get("image") or DEFAULT_IMAGE, req.get("pip") or [], req.get("system") or [])
    vol = modal.Volume.from_name(req.get("cache_volume") or "mlh-cache", create_if_missing=True)
    kwargs = dict(app=app, image=image, timeout=int(req["timeout_s"]) + 600, volumes={"/cache": vol},
                  block_network=not req.get("network", False))
    if req.get("gpu"):
        kwargs["gpu"] = req["gpu"]
    if req.get("memory_mb"):
        kwargs["memory"] = int(req["memory_mb"])  # host RAM the plan asked for (GraphSAINT PPI-edge was OOM-killed)
    if req.get("network") and req.get("domains"):
        kwargs["outbound_domain_allowlist"] = req["domains"]  # network restricted to data hosts
    t0 = time.time()
    # Image build output is captured so a failed build reports why (which package, which error).
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf), modal.enable_output():
            sb = modal.Sandbox.create(**kwargs)
        global CREATED_AT
        CREATED_AT = time.time()  # billing starts here: time queued for capacity is not container time
    except Exception as e:
        lines = [l for l in buf.getvalue().splitlines() if l.strip()]
        errs = [l for l in lines if any(k in l for k in ("error", "Error", "ERROR", "failed", "Failed", "No matching distribution"))][-15:]
        raise RuntimeError(f"{type(e).__name__}: {e}\nbuild errors:\n" + "\n".join(errs) + "\nbuild log tail:\n" + "\n".join(lines[-15:])) from e
    try:
        sb.filesystem.write_bytes(tar_dir(req["repo"]), "/tmp/repo.tgz")
        sb.exec("bash", "-c", "mkdir -p /work && tar -xzf /tmp/repo.tgz -C /work", timeout=600).wait()
        # Copy SRC to exactly DST, replacing it (cp -a SRC DST would nest SRC inside an existing DST).
        def put(src, dst):
            q, d = shlex.quote(src), shlex.quote(dst)
            return f"rm -rf {d} && mkdir -p $(dirname {d}) && cp -a {q} {d}"
        pre = " && ".join([put(r["from"], "/work/" + r["to"]) for r in (req.get("restore") or [])]
                          + [f"mkdir -p $(dirname {shlex.quote('/work/' + l['path'])}) && ln -sfn {shlex.quote(l['to'])} {shlex.quote('/work/' + l['path'])}" for l in (req.get("links") or [])]) or "true"
        # A declared output that the command did not produce is reported, never skipped silently.
        def keep(src, dst):
            q = shlex.quote(src)
            return f"if [ -e {q} ]; then {put(src, dst)}; else echo MLH_MISSING_OUTPUT={q} >&2; m=1; fi"
        post = " ; ".join(keep("/work/" + p["from"], p["to"]) for p in (req.get("persist") or [])) or "true"
        env = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in (req.get("env") or {}).items())
        cmd = " ".join(shlex.quote(a) for a in req["cmd"])
        script = f"cd /work && ({pre}) && {('env ' + env + ' ') if env else ''}{cmd}; s=$?; m=0; {post}; sync; [ $s -eq 0 ] && [ $m -ne 0 ] && s=97; exit $s"
        p = sb.exec("bash", "-c", script, timeout=int(req["timeout_s"]))
        out, err = p.stdout.read(), p.stderr.read()
        code = p.wait()
        files = {}
        for rel in req.get("outputs") or []:
            try:
                files[rel] = sb.filesystem.read_text(f"/work/{rel}")[-200000:]
            except Exception:
                pass
        res = {"exit_code": code, "stdout": out[-100000:], "stderr": err[-50000:], "elapsed_s": time.time() - CREATED_AT, "files": files}
        # Long output (per-epoch training logs): the whole of it, compressed, so metrics can be read from
        # every epoch (LoRA r3: only the last 42 of 80 RTE epochs survived the 100 kB tail).
        if len(out) > 100000:
            import base64, gzip
            res["stdout_gz"] = base64.b64encode(gzip.compress(out[-50_000_000:].encode())).decode()
        return res
    finally:
        # Volume changes become visible to other containers only once termination has committed them:
        # wait for it, or the next stage can start on a stale view (GraphSAINT r1, missing build outputs).
        sb.terminate()
        try:
            sb.wait(raise_on_termination=False)
        except Exception:
            pass


if __name__ == "__main__":
    T0 = time.time()
    req = json.loads(sys.stdin.read())
    try:
        res = {"check": op_check, "run": op_run}[req["op"]](req)
        print(json.dumps({"ok": True, **res}))
    except Exception as e:  # reported to the harness, which classifies it (and charges what ran)
        # Only the time a container existed is charged; a capacity error while queued costs nothing.
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {str(e)[:2000]}", "elapsed_s": time.time() - CREATED_AT if CREATED_AT else 0, "queued_s": time.time() - T0}))
