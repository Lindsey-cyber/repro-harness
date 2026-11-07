// Append-only, fsync'd event journal: the run's persistent state and checkpoint.
import { closeSync, existsSync, fsyncSync, mkdirSync, openSync, readFileSync, writeSync } from "node:fs";
import { dirname } from "node:path";
import type { RunEvent } from "../core/dag.ts";

export class Journal {
  constructor(readonly path: string) {}

  read(): RunEvent[] {
    if (!existsSync(this.path)) return [];
    const lines = readFileSync(this.path, "utf8").split("\n").filter(Boolean);
    const out: RunEvent[] = [];
    lines.forEach((l, i) => {
      try {
        out.push(JSON.parse(l) as RunEvent);
      } catch (e) {
        // A crash mid-write can leave a torn last line; anything earlier is corruption.
        if (i !== lines.length - 1) throw new Error(`journal ${this.path} corrupt at line ${i + 1}: ${String(e)}`);
      }
    });
    return out;
  }

  append(e: RunEvent): void {
    mkdirSync(dirname(this.path), { recursive: true });
    const fd = openSync(this.path, "a");
    try {
      writeSync(fd, JSON.stringify(e) + "\n");
      fsyncSync(fd);
    } finally {
      closeSync(fd);
    }
  }
}
