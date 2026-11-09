import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { modalSpent, rebuildTrust } from "../src/mlh/repro/campaign.ts";
import { renderRetro, roundOf, stopReached, type Round } from "../src/mlh/repro/retro.ts";
import { emptyStore } from "../src/mlh/repro/trustscore.ts";

const R = (label: string, success: number, n: number, medianMin: number, blockedBy: string[] = []): Round => ({ label, at: "t", success, n, medianMin, gpuUsd: 0, llmUsd: 0, blockedBy });

describe("round retro", () => {
  it("ranks failures, compares with the previous round and names the next fix", () => {
    const rows = [
      { claims: [{ status: "reproduced" }], wallMin: 20, gpuUsd: 0.4, llmUsd: 0.1, failures: { contract_violation: 2 } },
      { claims: [{ status: "not_runnable" }], wallMin: 80, gpuUsd: 1, llmUsd: 0.2, failures: { contract_violation: 1, timeout: 1 } },
    ];
    const r = roundOf("c2", "t", rows, emptyStore());
    expect(r).toMatchObject({ success: 1, n: 2, topFailure: "contract_violation", gpuUsd: 1.4 });
    const md = renderRetro(r, R("c1", 1, 4, 60), [R("c1", 1, 4, 60), r]);
    expect(md).toMatch(/1\/2 \(previous c1: 1\/4, better\)/);
    expect(md).toMatch(/tests\/mlh-replay/);
    expect(md).toMatch(/Stop predicate: not reached/);
  });

  it("stops only after two consecutive rounds meet every bar", () => {
    expect(stopReached([R("a", 5, 5, 30)]).done).toBe(false);
    expect(stopReached([R("a", 5, 5, 30), R("b", 4, 5, 50)]).done).toBe(false); // too slow
    expect(stopReached([R("a", 5, 5, 30), R("b", 5, 5, 30, ["debugger"])]).done).toBe(false); // a role unproven
    expect(stopReached([R("a", 1, 5, 30), R("b", 4, 5, 40), R("c", 5, 5, 30)]).done).toBe(true);
  });

  it("modal ledger sums every run's spend and survives a torn budget file", () => {
    const root = mkdtempSync(join(tmpdir(), "mlh-ledger-"));
    for (const [l, id, body] of [["c1", "a", '{"spent":1.5}'], ["c1", "b", '{"spent":0.25}'], ["c2", "a", "{torn"]] as const) {
      mkdirSync(join(root, l, id), { recursive: true });
      writeFileSync(join(root, l, id, "budget.json"), body);
    }
    writeFileSync(join(root, "c1", "a.log"), "");
    expect(modalSpent(root)).toBeCloseTo(1.75);
  });

  it("trust is rebuilt from journals in time order; stopped and solo rows are no evidence", () => {
    const root = mkdtempSync(join(tmpdir(), "mlh-rebuild-")), out = join(root, "out"), camp = join(root, "campaign");
    mkdirSync(out);
    const journal = (label: string, paper: string, at: string, envOk: boolean) => {
      mkdirSync(join(camp, label, paper), { recursive: true });
      writeFileSync(join(camp, label, paper, "journal.jsonl"), JSON.stringify({ t: "ok", at, id: "repro.env", output: { smoke: { ok: envOk } } }) + "\n");
    };
    journal("c1", "a", "2026-01-01", true); journal("c2", "a", "2026-01-03", false); journal("c1", "b", "2026-01-02", true); journal("c-solo", "a", "2026-01-04", false); journal("c2", "z", "2026-01-05", false);
    writeFileSync(join(out, "c1.jsonl"), '{"paper":"a"}\n{"paper":"b"}\n');
    writeFileSync(join(out, "c2.jsonl"), '{"paper":"a"}\n{"paper":"z","stopped":"provider_fatal"}\n');
    writeFileSync(join(out, "c-solo.jsonl"), '{"paper":"a"}\n');
    const t = rebuildTrust(out, camp);
    expect(t.env).toMatchObject({ n: 3, ok: 2 });
    expect(t.env.history.map((h) => h.paper)).toEqual(["c1/a", "c1/b", "c2/a"]);
  });
});
