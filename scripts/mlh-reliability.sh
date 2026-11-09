#!/usr/bin/env bash
# P5 reliability baseline: the same paper, same configuration, strictly one run at a time.
# 3 clean runs from scratch, then 2 runs killed mid-way (during the env agent; during a GPU stage)
# and resumed. Each run is its own campaign label (doctor gate, watchdog, stop rules, accounting).
#   scripts/mlh-reliability.sh <env-file> <config.json> <prefix>
set -u
envfile=$1; config=$2; prefix=$3
cd "$(dirname "$0")/.."
. "$envfile"
out=bench/complex/results/reliability
mkdir -p "$out"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$out/$prefix.log"; }
campaign() { MLH_ENV_CACHE=".mlh/env-cache-$1" npx tsx src/mlh/cli.ts campaign "$config" --label "$1" --parallel 1 --out "$out" >> ".mlh/campaign-$1.log" 2>&1; }
journal() { cat .mlh/campaign/"$1"/*/journal.jsonl 2>/dev/null; }
stopped() { [ -f ".mlh/campaign/$1/stopped" ] || ls .mlh/campaign/"$1"/*/stopped.json >/dev/null 2>&1; }

for i in 1 2 3; do
  l="$prefix-clean$i"; log "start $l"; campaign "$l"; log "end $l (exit $?)"
  if stopped "$l"; then log "STOP: $l stopped; investigating before more runs"; exit 1; fi
done

# Killed while a node of the given kind is running, then resumed by relaunching the same label.
killed_run() {
  l=$1; pattern=$2
  log "start $l (kill when $pattern starts)"
  campaign "$l" &
  for _ in $(seq 1 720); do
    if journal "$l" | grep -q "\"t\":\"start\",\"at\":\"[^\"]*\",\"id\":\"$pattern"; then break; fi
    sleep 5
  done
  sleep 20
  pkill -9 -f -- "--label $l( |$)"; pkill -9 -f -- "campaign/$l/"; wait
  log "killed $l during $pattern; resuming"
  campaign "$l"; log "end $l (exit $?)"
  if stopped "$l"; then log "STOP: $l stopped after resume"; exit 1; fi
}
killed_run "$prefix-kill-env" "repro.env"
killed_run "$prefix-kill-stage" "repro.stage"
log "all runs done"
