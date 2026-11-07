#!/usr/bin/env bash
# refrun.sh <logfile> <workdir> <site> <timeout-s> -- cmd...   (runs through sbx.sh, no network, records wall time)
set -u
cd "$(dirname "$0")/.."
log=$1; w=$2; site=$3; to=$4; shift 5
mkdir -p "$(dirname "$log")"
s=$(date +%s)
{ echo "\$ $*"; timeout "$to" tools/sbx.sh "$w" --site "$site" -- "$@"; echo "exit=$?"; echo "wall_s=$(( $(date +%s) - s ))"; } > "$log" 2>&1
