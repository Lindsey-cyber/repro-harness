"""Trusted data fetcher, run by the harness inside a remote sandbox with network (never repository code).

argv[1] is a JSON request: {"items": [{url, to, sha256?, unpack?}], "checks": [{path, minFiles}]}.
Idempotent: an item already fetched (same URL, target present) is skipped. Prints one JSON line:
{"items": [{url, to, sha256, bytes, skipped}], "checks": [{path, files, minFiles, ok}]}.
"""
import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile

MARKS = "/cache/.mlh-fetch"


def count_files(path):
    n = 0
    for _root, _dirs, files in os.walk(path):
        n += len(files)
    return n


def fetch(item):
    url, to = item["url"], item["to"]
    mark = os.path.join(MARKS, hashlib.sha256(url.encode()).hexdigest()[:24] + ".json")
    if os.path.exists(mark) and os.path.exists(to):
        rec = json.load(open(mark))
        return {**rec, "skipped": True}
    os.makedirs(MARKS, exist_ok=True)
    tmp = to.rstrip("/") + ".part"
    os.makedirs(os.path.dirname(tmp) or "/", exist_ok=True)
    h, n = hashlib.sha256(), 0
    req = urllib.request.Request(url, headers={"User-Agent": "mlh-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            h.update(chunk)
            n += len(chunk)
            f.write(chunk)
    digest = h.hexdigest()
    if item.get("sha256") and item["sha256"].lower() != digest:
        os.remove(tmp)
        raise SystemExit(f"sha256 mismatch for {url}: expected {item['sha256']}, got {digest}")
    unpack = item.get("unpack")
    if unpack == "zip":
        os.makedirs(to, exist_ok=True)
        with zipfile.ZipFile(tmp) as z:
            z.extractall(to)
        os.remove(tmp)
    elif unpack in ("tar", "tgz", "tar.gz"):
        os.makedirs(to, exist_ok=True)
        with tarfile.open(tmp) as t:
            t.extractall(to, filter="data")
        os.remove(tmp)
    else:
        shutil.move(tmp, to)
    rec = {"url": url, "to": to, "sha256": digest, "bytes": n}
    json.dump(rec, open(mark, "w"))
    return {**rec, "skipped": False}


if __name__ == "__main__":
    req = json.loads(sys.argv[1])
    items = [fetch(i) for i in req.get("items", [])]
    checks = []
    for c in req.get("checks", []):
        files = count_files(c["path"]) if os.path.exists(c["path"]) else 0
        checks.append({"path": c["path"], "files": files, "minFiles": c["minFiles"], "ok": files >= c["minFiles"]})
    os.sync()
    print(json.dumps({"items": items, "checks": checks}))
