#!/usr/bin/env bash
# Two TP4 replicas of one ASR arm on ONE chip pair (cores pair*8+0..3 and +4..7) behind a round-robin front on PORT,
# so the unchanged client and the six-model runner treat "2 x TP4" exactly like "1 x TP8" on the same two chips.
# Usage: serve_replicas.sh <ARM> <PAIR> <PORT>     ARM in R4 (stock config) / RB (fork, 128-token buckets).
# Logs: logs/<ARM>.r0.server.log, logs/<ARM>.r1.server.log; the front's log: logs/<ARM>.proxy.log.
# SUITE_CPUS="a-b" (the arm's CPU range) is split in half between the replicas; the front runs on the first half.
set -uo pipefail
ARM=$1; PAIR=$2; PORT=$3; H=$(cd "$(dirname "$0")" && pwd)
P0=$((PORT + 1000)); P1=$((PORT + 2000)); mkdir -p logs
if [ -n "${SUITE_CPUS:-}" ]; then a=${SUITE_CPUS%-*}; b=${SUITE_CPUS#*-}; m=$(( (a + b + 1) / 2 ))
  C0="$a-$((m - 1))"; C1="$m-$b"; else C0=""; C1=""; fi
pids=()
cleanup() {   # finishes well inside the runner's 5 s stop window
  for p in "${pids[@]}"; do kill -TERM "$p" 2>/dev/null; done
  for _ in 1 2 3 4 5 6; do alive=0; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done; [ $alive = 0 ] && return; sleep 0.25; done
  for p in "${pids[@]}"; do kill -KILL "$p" 2>/dev/null; done
}
trap cleanup EXIT TERM INT
REPLICA=0 REPLICA_CPUS=$C0 LOG_TAG=.r0 bash "$H/serve.sh" "$ARM" "$PAIR" "$P0" & pids+=($!)
REPLICA=1 REPLICA_CPUS=$C1 LOG_TAG=.r1 bash "$H/serve.sh" "$ARM" "$PAIR" "$P1" & pids+=($!)
${SUITE_PYTHON:-python3} "$H/rr_proxy.py" "$PORT" "$P0" "$P1" > "logs/$ARM.proxy.log" 2>&1 & pids+=($!)
# the arm is alive while every child is: a replica that dies takes the arm down (no silent single-replica result)
while :; do for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null || { echo "replica process $p exited" >&2; exit 1; }; done; sleep 0.5; done
