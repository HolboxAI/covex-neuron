#!/usr/bin/env bash
# Separate per-stage pass (not timed): arms S2 and D, concurrency 1 and 8, ~50 requests each.
# (a) request side: /metrics deltas + client E2E -> stage table; py-spy flame graphs (API server + engine core) at c8.
# (b) device side: restart with PROFILE=1, capture Neuron Explorer system+device profiles at c1 and c8, summarize kernels.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); CLIENT_DIR=${CLIENT_DIR:-$HOME/qwen3-asr-inference-cuda-trainium}
declare -A PORT=([S2]=8001 [D]=8002) PAIR=([S2]=1 [D]=2)
wait_up() { for _ in $(seq 1 360); do curl -sf "http://127.0.0.1:$1/health" > /dev/null && return 0; sleep 10; done; return 1; }
load() { # arm conc tag
  python3 "$CLIENT_DIR/benchmark/benchmark.py" --base-url "http://127.0.0.1:${PORT[$1]}" --model qwen3-asr-1.7b \
    --data-dir "$CLIENT_DIR/data" --num-samples 50 --concurrency "$2" --rounds 1 --warmup 3 \
    --phases transcribe --label "stage-$1-c$2" --output "stages/$1-c$2-$3.json" > /dev/null; }
mkdir -p stages profiles
# (a) request side + host flame graphs
for a in S2 D; do LOG_TAG=-stage "$HERE/serve.sh" "$a" "${PAIR[$a]}" "${PORT[$a]}" & done
for a in S2 D; do wait_up "${PORT[$a]}"; done
for a in S2 D; do for c in 1 8; do
  load "$a" "$c" warm
  curl -sf "http://127.0.0.1:${PORT[$a]}/metrics" > "stages/$a-c$c.before.prom"
  if [ "$c" = 8 ] && command -v py-spy > /dev/null; then
    pid=$(pgrep -f "vllm serve.*--port ${PORT[$a]}" | head -1)
    py-spy record --subprocesses --idle -d 25 -o "stages/$a-c8-host.svg" -p "$pid" & SPY=$!
  fi
  load "$a" "$c" run
  curl -sf "http://127.0.0.1:${PORT[$a]}/metrics" > "stages/$a-c$c.after.prom"
  [ -n "${SPY:-}" ] && wait "$SPY" || true; unset SPY
done; done
pkill -f "vllm serve" || true; sleep 15
# (b) device timelines
for a in S2 D; do LOG_TAG=-prof PROFILE=1 "$HERE/serve.sh" "$a" "${PAIR[$a]}" "${PORT[$a]}" & done
for a in S2 D; do wait_up "${PORT[$a]}"; done
for a in S2 D; do for c in 1 8; do
  load "$a" "$c" pwarm
  curl -sf -X POST "http://127.0.0.1:${PORT[$a]}/start_profile" > /dev/null
  load "$a" "$c" prof
  curl -sf -X POST "http://127.0.0.1:${PORT[$a]}/stop_profile" > /dev/null
  mv "profiles/$a" "profiles/$a-c$c"
  neuron-explorer view -d "profiles/$a-c$c" --output-format perfetto
  python3 "$HERE/summarize_profile.py" "profiles/$a-c$c" > "profiles/$a-c$c/STAGES.md" || true
done; done
pkill -f "vllm serve" || true; sleep 15
# tables
for a in S2 D; do { echo "# $a"; for c in 1 8; do
  python3 "$HERE/stage_table.py" "stages/$a-c$c.before.prom" "stages/$a-c$c.after.prom" "stages/$a-c$c-run.json" "$c" "profiles/$a-c$c/STAGES.md"; echo; done; } > "stages/$a.md"; done
echo "stage tables: stages/S2.md stages/D.md; flame graphs: stages/*-host.svg; traces: profiles/*-c*/"
