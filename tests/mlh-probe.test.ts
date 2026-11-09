// The download URL probe (planning-time) and the http-with-checksum rule, offline (local server).
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { describe, expect, it } from "vitest";
import { gpuPrice } from "../src/mlh/backends/modal.ts";
import { validateGpuPlan } from "../src/mlh/repro/gpu.ts";
import { probeUrls } from "../src/mlh/repro/skills.ts";

describe("download URL probe", () => {
  it("passes files that answer, rejects 404, falls back to a 1-byte GET where HEAD is refused, names unreachable hosts", async () => {
    const srv = createServer((req, res) => {
      if (req.url === "/ok") { res.statusCode = 200; res.end("x"); return; }
      if (req.url === "/nohead") { res.statusCode = req.method === "HEAD" ? 405 : 206; res.end("x"); return; }
      res.statusCode = 404; res.end();
    }).listen(0);
    const base = `http://127.0.0.1:${(srv.address() as AddressInfo).port}`;
    const saved = { p: process.env.HTTP_PROXY, s: process.env.MLH_SKIP_URL_PROBE };
    delete process.env.MLH_SKIP_URL_PROBE;
    try {
      const bad = await probeUrls([`${base}/ok`, `${base}/nohead`, `${base}/missing`, "https://127.0.0.1:1/x"]);
      expect(bad).toHaveLength(2);
      expect(bad[0]).toMatch(/\/missing answered HTTP 404/);
      expect(bad[1]).toMatch(/not reachable.*http URL with its sha256/);
    } finally { srv.close(); if (saved.s !== undefined) process.env.MLH_SKIP_URL_PROBE = saved.s; }
  });
});

describe("http downloads need a checksum", () => {
  const claims = [{ id: "c", metric: "accuracy", value: 0.9 }] as never;
  const plan = (fetch: unknown[]) => ({ fidelity: "full", stages: [{ id: "s", argv: ["python", "a.py"], timeoutS: 300 }], claims: [{ claim: "c", stage: "s", parse: "acc: ([0-9.]+)" }], fetch });
  const sha = "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05";
  it("accepts http with a sha256 (COCO serves only http), rejects http without one", () => {
    expect(validateGpuPlan(claims, plan([{ url: "http://images.cocodataset.org/zips/val2017.zip", to: "/cache/data/coco", sha256: sha, unpack: "zip" }]), { budgetUsd: 5, price: gpuPrice }).errors).toEqual([]);
    expect(validateGpuPlan(claims, plan([{ url: "http://images.cocodataset.org/zips/val2017.zip", to: "/cache/data/coco" }]), { budgetUsd: 5, price: gpuPrice }).errors.join()).toMatch(/must be https, or http with its sha256/);
  });
});
