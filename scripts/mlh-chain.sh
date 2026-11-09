#!/usr/bin/env bash
# Runs campaigns one after another, unattended: waits for a campaign label to finish, then starts the next.
#   [MLH_CONFIG=bench/complex/x.json] scripts/mlh-chain.sh <env-file> <wait-label> <next-label> [campaign args...]
# Detached use: setsid nohup scripts/mlh-chain.sh ... &
set -u
envfile=$1; wait_label=$2; next_label=$3; shift 3
cd "$(dirname "$0")/.."
config=${MLH_CONFIG:-bench/complex/campaign.json}
while pgrep -f "campaign .* --label ${wait_label} " >/dev/null; do sleep 60; done
. "$envfile"
echo "$(date -u +%FT%TZ) ${wait_label} finished; starting ${next_label}" >> .mlh/chain.log
exec npx tsx src/mlh/cli.ts campaign "$config" --label "$next_label" "$@" >> ".mlh/campaign-${next_label}.log" 2>&1 < /dev/null
