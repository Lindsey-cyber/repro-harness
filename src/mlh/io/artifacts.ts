// Content-addressed artifact store under the run directory.
import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import type { ArtifactRef } from "../core/types.ts";

export class ArtifactStore {
  constructor(readonly dir: string) {}

  put(name: string, file: string): ArtifactRef {
    const buf = readFileSync(file);
    const sha256 = createHash("sha256").update(buf).digest("hex");
    mkdirSync(this.dir, { recursive: true });
    const dest = join(this.dir, sha256);
    if (!existsSync(dest)) copyFileSync(file, dest);
    return { name, sha256, bytes: statSync(dest).size };
  }

  path(ref: ArtifactRef): string {
    return join(this.dir, ref.sha256);
  }
}
