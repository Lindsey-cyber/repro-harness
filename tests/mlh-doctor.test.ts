import { describe, expect, it } from "vitest";
import { imageCheck, labelOf } from "../src/mlh/repro/doctor.ts";
import { IMAGES } from "../src/mlh/repro/gpu.ts";

describe("doctor", () => {
  it("every offered image has a label the probe can check", () => {
    for (const k of Object.keys(IMAGES)) expect(labelOf(k), k).toBeDefined();
  });
  it("flags a label that disagrees with the image (the c1 case)", () => {
    expect(imageCheck("py3.8-torch1.12-cu113", "MLH_PROBE py=3.7.13 torch=1.12.1").ok).toBe(false);
    expect(imageCheck("py3.7-torch1.12-cu113", "MLH_PROBE py=3.7.13 torch=1.12.1").ok).toBe(true);
    expect(imageCheck("py3.7-torch1.12-cu113", "boom").ok).toBe(false);
  });
  it("old labels still resolve but are not offered", () => {
    expect(IMAGES["py3.8-torch1.12-cu113"]).toBe(IMAGES["py3.7-torch1.12-cu113"]);
    expect(Object.keys(IMAGES)).not.toContain("py3.8-torch1.12-cu113");
  });
});

describe("image for the repository's torch pin", async () => {
  const { imageForPins } = await import("../src/mlh/repro/gpu.ts");
  it("maps a pin to the image with that torch (SimSiam: torch==1.7.0)", () => {
    expect(imageForPins("torch==1.7.0\ntorchvision==0.8.1")).toEqual({ key: "py3.8-torch1.7-cu110", torch: "1.7" });
    expect(imageForPins("install_requires=['torch>=1.12']")?.key).toBe("py3.7-torch1.12-cu113");
    expect(imageForPins("numpy==1.18")).toBeUndefined();
    expect(imageForPins("torch==0.4.1")).toBeUndefined();
  });
});
