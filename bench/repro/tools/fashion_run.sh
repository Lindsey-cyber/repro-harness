#!/usr/bin/env bash
# fashion_run.sh <workdir> <site> <log> : run the benchmark runner (which never exits by itself) for 120 s
cd "$(dirname "$0")/.."
( time timeout 120 tools/sbx.sh "$1" --site "$2" -- python3 benchmark/runner.py ) > "$3" 2>&1 || true
