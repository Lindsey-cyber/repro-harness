#!/usr/bin/env bash
# mkwork.sh <cache-clone-dir> <commit> <dest> [patch...]  -- build a clean checkout at commit, apply patches in order
set -euo pipefail
src=$(realpath "$1"); commit=$2; dest=$3; shift 3
git clone -q --no-checkout "$src" "$dest"
git -C "$dest" checkout -q "$commit"
for p in "$@"; do git -C "$dest" apply "$(realpath "$p")"; done
