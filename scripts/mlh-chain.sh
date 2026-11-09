#!/usr/bin/env bash
# Runs campaigns one after another, unattended: waits for a campaign label to finish, then starts the next.
#   [MLH_CONFIG=bench/complex/x.json] scripts/mlh-chain.sh <env-file> <wait-label> <next-label> [campaign args...]
# Detached use: setsid nohup scripts/mlh-chain.sh ... &
# Test hooks: MLH_CHAIN_CLI (command prefix), MLH_CHAIN_POLL_S, MLH_CHAIN_MAX_WAIT_S, MLH_CHAIN_ROOT.
set -u
envfile=$1; wait_label=$2; next_label=$3; shift 3
cd "${MLH_CHAIN_ROOT:-$(dirname "$0")/..}"
config=${MLH_CONFIG:-bench/complex/campaign.json}
cli=${MLH_CHAIN_CLI:-npx tsx src/mlh/cli.ts}
poll=${MLH_CHAIN_POLL_S:-20}
max_wait=${MLH_CHAIN_MAX_WAIT_S:-86400}
marker=.mlh/campaign/${wait_label}
mkdir -p .mlh
running() { pgrep -f -- "campaign .*--label ${wait_label}( |$)" >/dev/null; }
log() { echo "$(date -u +%FT%TZ) $*" >> .mlh/chain.log; }
# Finished = a done marker with no process, or seen running and then gone. A round that has not
# started yet has neither. A "stopped" marker (balance, limits, strict stop) ends the chain: the
# next round would hit the same wall. Waiting longer than max_wait ends the chain too (a round that
# crashed between polls without a marker): it is reported, never guessed.
seen=0; waited=0
while :; do
  if running; then seen=1
  elif [ -f "$marker/stopped" ]; then log "${wait_label} stopped; not starting ${next_label}"; exit 1
  elif [ "$seen" = 1 ] || [ -f "$marker/done" ]; then break
  elif [ "$waited" -ge "$max_wait" ]; then log "${wait_label} neither ran nor finished within ${max_wait}s; not starting ${next_label}"; exit 2; fi
  sleep "$poll"; waited=$((waited + poll))
done
# A round that was seen and is gone without a done marker crashed or was killed: do not chain on.
if [ ! -f "$marker/done" ]; then log "${wait_label} ended without a done marker; not starting ${next_label}"; exit 3; fi
. "$envfile"
# Review the finished round (idempotent; covers rounds run by older code), then start the next.
$cli campaign-retro "$wait_label" >> .mlh/chain.log 2>&1
log "${wait_label} finished; starting ${next_label}"
exec $cli campaign "$config" --label "$next_label" "$@" >> ".mlh/campaign-${next_label}.log" 2>&1 < /dev/null
