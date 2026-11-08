"""Modal GPU backend for the reproduction harness: one JSON request on stdin, one JSON result on stdout.

Untrusted repository code runs only inside a Modal Sandbox (an isolated container on Modal's
infrastructure), never on the harness machine. The repository is uploaded into the sandbox; a
persistent Modal Volume caches downloaded datasets/checkpoints across runs. Network inside the
sandbox is blocked for experiment runs and allowed only when the request says so (the separately
authorized fetch/install phase).

ops:
  check  {} -> {app}
  run    {repo, cmd, gpu, timeout_s, network, env, setup, outputs, image, cache_volume}
         -> {exit_code, stdout, stderr, elapsed_s, files: {relpath: text}}
"""
import io
import json
import os
import shlex
import sys
import tarfile
import time

import modal

APP = "mlh-repro"
# A CUDA-enabled PyTorch base; repositories install the rest in their setup step.
DEFAULT_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"


def tar_dir(path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".mlh_in")]
            for f in files:
                full = os.path.join(root, f)
                if os.path.getsize(full) > 200 * 1024 * 1024:
                    continue  # large artifacts belong in the cache volume, not the upload
                t.add(full, arcname=os.path.relpath(full, path))
    return buf.getvalue()


def op_check(_req):
    app = modal.App.lookup(APP, create_if_missing=True)
    return {"app": app.app_id}


def op_run(req):
    app = modal.App.lookup(APP, create_if_missing=True)
    image = modal.Image.from_registry(req.get("image") or DEFAULT_IMAGE)
    vol = modal.Volume.from_name(req.get("cache_volume") or "mlh-cache", create_if_missing=True)
    kwargs = dict(app=app, image=image, timeout=int(req["timeout_s"]) + 300, volumes={"/cache": vol},
                  block_network=not req.get("network", False))
    if req.get("gpu"):
        kwargs["gpu"] = req["gpu"]
    t0 = time.time()
    sb = modal.Sandbox.create(**kwargs)
    try:
        sb.filesystem.write_bytes(tar_dir(req["repo"]), "/tmp/repo.tgz")
        sb.exec("bash", "-c", "mkdir -p /work && tar -xzf /tmp/repo.tgz -C /work", timeout=600).wait()
        env = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in (req.get("env") or {}).items())
        cmd = " ".join(shlex.quote(a) for a in req["cmd"])
        script = f"cd /work && ({req.get('setup') or 'true'}) && {('env ' + env + ' ') if env else ''}{cmd}"
        p = sb.exec("bash", "-c", script, timeout=int(req["timeout_s"]))
        out, err = p.stdout.read(), p.stderr.read()
        code = p.wait()
        files = {}
        for rel in req.get("outputs") or []:
            try:
                files[rel] = sb.filesystem.read_text(f"/work/{rel}")[-200000:]
            except Exception:
                pass
        return {"exit_code": code, "stdout": out[-100000:], "stderr": err[-50000:], "elapsed_s": time.time() - t0, "files": files}
    finally:
        sb.terminate()


if __name__ == "__main__":
    req = json.loads(sys.stdin.read())
    try:
        res = {"check": op_check, "run": op_run}[req["op"]](req)
        print(json.dumps({"ok": True, **res}))
    except Exception as e:  # reported to the harness, which classifies it
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {str(e)[:2000]}"}))
