#!/usr/bin/env bash
# Runs campaigns one after another, unattended: waits for a campaign label to finish, then starts the next.
#   [MLH_CONFIG=bench/complex/x.json] scripts/mlh-chain.sh <env-file> <wait-label> <next-label> [campaign args...]
# Detached use: setsid nohup scripts/mlh-chain.sh ... &
set -u
envfile=$1; wait_label=$2; next_label=$3; shift 3
cd "$(dirname "$0")/.."
config=${MLH_CONFIG:-bench/complex/campaign.json}
marker=.mlh/campaign/${wait_label}
running() { pgrep -f -- "campaign .*--label ${wait_label}( |$)" >/dev/null; }
# Finished = a done marker with no process, or seen running and then gone. A round that has not
# started yet has neither. A "stopped" marker (balance, limit) ends the chain: the next round
# would hit the same wall.
seen=0
while :; do
  if running; then seen=1
  elif [ -f "$marker/stopped" ]; then echo "$(date -u +%FT%TZ) ${wait_label} stopped; not starting ${next_label}" >> .mlh/chain.log; exit 1
  elif [ "$seen" = 1 ] || [ -f "$marker/done" ]; then break; fi
  sleep 20
done
. "$envfile"
# Review the finished round (idempotent; covers rounds run by older code), then start the next.
npx tsx src/mlh/cli.ts campaign-retro "$wait_label" >> .mlh/chain.log 2>&1
echo "$(date -u +%FT%TZ) ${wait_label} finished; starting ${next_label}" >> .mlh/chain.log
exec npx tsx src/mlh/cli.ts campaign "$config" --label "$next_label" "$@" >> ".mlh/campaign-${next_label}.log" 2>&1 < /dev/null
