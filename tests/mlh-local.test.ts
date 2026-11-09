import { describe, expect, it } from "vitest";
import { localInspection } from "../src/mlh/agentloop.ts";

describe("read-only inspection runs locally, everything else remote", () => {
  it.each([
    ["sed -n 500,600p examples/NLU/examples/text-classification/run_glue.py", "sed -n 500,600p examples/NLU/examples/text-classification/run_glue.py"],
    ["cd /work && ls output/x/ 2>/dev/null", "cd . && ls output/x/ 2>/dev/null"],
    ['grep -rln "all_results.json\\|mean" examples | head', 'grep -rln "all_results.json\\|mean" examples | head'],
    ["pwd; ls -la; echo ---; ls examples 2>/dev/null | head", "pwd; ls -la; echo ---; ls examples 2>/dev/null | head"],
  ])("local: %s", (cmd, want) => expect(localInspection(cmd)).toBe(want));

  it.each([
    "python -c 'import torch'",
    "ls /cache/hf",
    "cat a > b",
    "echo x >> f",
    "sed -i s/a/b/ f",
    "sed s/a/b/ f",
    "find . -exec rm {} ;",
    "find . -delete",
    "ls $(cat x)",
    "ls `whoami`",
    "cat f | tee g",
    "ls; bash run.sh",
    "awk 'BEGIN{system(\"id\")}'",
    "cat <<EOF",
  ])("remote: %s", (cmd) => expect(localInspection(cmd)).toBeUndefined());
});

