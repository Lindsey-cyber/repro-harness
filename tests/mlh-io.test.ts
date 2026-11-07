import { appendFileSync, existsSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { Journal } from "../src/mlh/io/journal.ts";
import { DEFAULT_LIMITS, Sandbox, selfTest } from "../src/mlh/io/sandbox.ts";

const tmp = () => mkdtempSync(join(tmpdir(), "mlh-"));

describe("journal", () => {
  it("tolerates a torn last line from a crash, but not corruption earlier", () => {
    const p = join(tmp(), "j.jsonl");
    const j = new Journal(p);
    j.append({ t: "done", at: "x" });
    appendFileSync(p, '{"t":"no');
    expect(j.read()).toHaveLength(1);
    writeFileSync(p, '{"t":"no\n{"t":"done","at":"x"}\n');
    expect(() => j.read()).toThrow(/corrupt/);
  });
});

describe("sandbox", async () => {
  const st = await selfTest(tmp());
  it.runIf(st.isolation === "namespace")("blocks network, host writes and home, keeps the copy writable", async () => {
    const src = tmp();
    writeFileSync(join(src, "x.py"), [
      "import os, socket",
      "open('out.txt','w').write('in workspace')",
      "try:\n socket.create_connection(('1.1.1.1', 443), timeout=2); print('NET')\nexcept OSError: print('nonet')",
      "try:\n open(os.path.expanduser('~/.mlh_test_canary'),'w').write('x'); print('home-write-private')\nexcept OSError: print('home-blocked')",
      "try:\n open('/usr/mlh_x','w'); print('ROOTWRITE')\nexcept OSError: print('ro')",
    ].join("\n"));
    const sb = Sandbox.create(tmp(), "t", src, { isolation: "namespace", limits: DEFAULT_LIMITS, stats: { runs: 0, ms: 0 } });
    const r = await sb.run(["python3", "x.py"]);
    expect(r.code).toBe(0);
    expect(r.stdout).toContain("nonet");
    expect(r.stdout).toContain("ro");
    expect(existsSync(join(process.env.HOME ?? "/root", ".mlh_test_canary"))).toBe(false);
    expect(existsSync(sb.path("out.txt"))).toBe(true);
    expect(existsSync(join(src, "out.txt"))).toBe(false); // the source tree is never touched
  });

  it("classifies a runaway experiment as a timeout and kills it", async () => {
    const src = tmp();
    writeFileSync(join(src, "spin.py"), "while True: pass\n");
    const sb = Sandbox.create(tmp(), "t", src, { isolation: st.isolation, limits: { ...DEFAULT_LIMITS, timeoutMs: 1500 }, stats: { runs: 0, ms: 0 } });
    const r = await sb.run(["python3", "spin.py"]);
    expect(r.category).toBe("timeout");
    expect(r.ms).toBeLessThan(5000);
  });

  it("classifies a CPU-limit kill as resource_limit", async () => {
    const src = tmp();
    writeFileSync(join(src, "spin.py"), "while True: pass\n");
    const sb = Sandbox.create(tmp(), "t", src, { isolation: st.isolation, limits: { ...DEFAULT_LIMITS, cpuSec: 1 }, stats: { runs: 0, ms: 0 } });
    expect((await sb.run(["python3", "spin.py"])).category).toBe("resource_limit");
  });
});
