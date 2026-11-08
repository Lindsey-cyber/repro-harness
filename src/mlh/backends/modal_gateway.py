"""mlh execution gateway: a Modal app that runs reproduction jobs in GPU sandboxes.

Why a gateway: the harness's network path only carries HTTP/1.1 HTTPS, while Modal's SDK speaks
gRPC. The gateway keeps Modal's own credentials on Modal; the harness holds only a revocable
gateway token (sent as `Authorization: Bearer ...`), and talks plain HTTPS.

Jobs are asynchronous: POST /start returns a job id at once; GET /result?id=... polls. A job keeps
running on Modal if the harness dies; the harness resumes polling with the id from its journal.

Deploy once, from a machine where `modal setup` has been run:
    modal secret create mlh-gateway MLH_GATEWAY_TOKEN=$(python3 -c "import secrets;print(secrets.token_urlsafe(32))")
    modal deploy src/mlh/backends/modal_gateway.py
The deploy prints the endpoint URL (https://<workspace>--mlh-gateway-api.modal.run).
"""
import base64
import hmac
import io
import json
import os
import shlex
import tarfile
import time

import modal

app = modal.App("mlh-gateway")
web_image = modal.Image.debian_slim().pip_install("fastapi[standard]")
DEFAULT_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"

# Server-side limits, enforced whatever the client asks for.
ALLOWED_GPUS = {"T4", "L4", "A10G", "A100-40GB", "A100-80GB", "H100"}
MAX_TIMEOUT_S = 6 * 3600
MAX_UPLOAD_BYTES = 300 * 1024 * 1024

cache = modal.Volume.from_name("mlh-cache", create_if_missing=True)


@app.function(image=web_image, timeout=MAX_TIMEOUT_S + 900, volumes={"/cache": cache})
def run_job(req: dict) -> dict:
    """Runs one job in a fresh sandbox; untrusted code never runs in this function's container."""
    image = modal.Image.from_registry(req.get("image") or DEFAULT_IMAGE)
    kwargs = dict(app=app, image=image, timeout=int(req["timeout_s"]) + 300, volumes={"/cache": cache},
                  block_network=not req.get("network", False))
    if req.get("gpu"):
        kwargs["gpu"] = req["gpu"]
    t0 = time.time()
    sb = modal.Sandbox.create(**kwargs)
    try:
        with sb.open("/tmp/repo.tgz", "wb") as f:
            f.write(base64.b64decode(req["repo_tgz"]))
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
                with sb.open(f"/work/{rel}", "r") as f:
                    files[rel] = f.read()[-200000:]
            except Exception:
                pass
        return {"exit_code": code, "stdout": out[-100000:], "stderr": err[-50000:], "elapsed_s": time.time() - t0, "files": files}
    finally:
        sb.terminate()


@app.function(image=web_image, secrets=[modal.Secret.from_name("mlh-gateway")])
@modal.asgi_app()
def api():
    from fastapi import FastAPI, HTTPException, Request

    web = FastAPI()
    token = os.environ["MLH_GATEWAY_TOKEN"]

    def auth(request: Request):
        got = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(got, token):
            raise HTTPException(status_code=401, detail="bad gateway token")

    @web.get("/health")
    def health(request: Request):
        auth(request)
        return {"ok": True}

    @web.post("/start")
    async def start(request: Request):
        auth(request)
        req = json.loads(await request.body())
        if req.get("gpu") and req["gpu"] not in ALLOWED_GPUS:
            raise HTTPException(400, f"gpu {req['gpu']} not allowed")
        if int(req.get("timeout_s", 0)) > MAX_TIMEOUT_S or int(req.get("timeout_s", 0)) <= 0:
            raise HTTPException(400, "timeout_s out of range")
        if len(req.get("repo_tgz", "")) * 3 // 4 > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "repository upload too large; put data in the cache volume")
        call = run_job.spawn(req)
        return {"id": call.object_id}

    @web.get("/result")
    def result(request: Request, id: str):
        auth(request)
        call = modal.FunctionCall.from_id(id)
        try:
            return {"done": True, **call.get(timeout=0)}
        except TimeoutError:
            return {"done": False}
        except Exception as e:
            return {"done": True, "error": f"{type(e).__name__}: {str(e)[:2000]}"}

    @web.post("/cancel")
    def cancel(request: Request, id: str):
        auth(request)
        modal.FunctionCall.from_id(id).cancel()
        return {"ok": True}

    return web
