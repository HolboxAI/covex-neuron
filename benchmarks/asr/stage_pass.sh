#!/usr/bin/env bash
# Separate per-stage pass (not timed): arms S2 and D, concurrency 1 and 8, ~50 requests each.
# (a) request side: /metrics deltas + client E2E -> stage table; py-spy flame graphs (API server + engine core) at c8.
# (b) device side: restart with PROFILE=1, capture Neuron Explorer system+device profiles at c1 and c8, summarize kernels.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); CLIENT_DIR=${CLIENT_DIR:-$HOME/qwen3-asr-inference-cuda-trainium}
declare -A PORT=([S2]=8001 [D]=8002) PAIR=([S2]=1 [D]=2)
source "$HERE/../lib.sh"
declare -A SPID
up_all() {  # tag [env...]: start S2 and D, READY = /health + one real transcription; abort the pass if not ready
  local tag=$1 a why; shift
  for a in S2 D; do env LOG_TAG=-$tag "$@" "$HERE/serve.sh" "$a" "${PAIR[$a]}" "${PORT[$a]}" & SPID[$a]=$!; done
  for a in S2 D; do why=$(wait_ready "${SPID[$a]}" "${PORT[$a]}" 3600 asr qwen3-asr-1.7b "$CLIENT_DIR/data") \
    || { echo "stage pass: $a not ready ($why)" >&2; stop_both; exit 3; }; done; }
stop_both() { local a; for a in S2 D; do [ -n "${SPID[$a]:-}" ] && stop_arm "${SPID[$a]}" "${PORT[$a]}"; done; }   # only our arms
load() { # arm conc tag warmup samples: warm calls (outside any window) use warmup 3; measured calls use warmup 0 --
  # the pinned client sends its warmup waves inside the same process, so a measured call with warmup 3 put 3 (c1) or
  # 24 (c8) extra requests inside the snapshot (53 / 74 server requests for 50 clips).
  python3 "$CLIENT_DIR/benchmark/benchmark.py" --base-url "http://127.0.0.1:${PORT[$1]}" --model qwen3-asr-1.7b \
    --data-dir "$CLIENT_DIR/data" --num-samples "${5:-50}" --concurrency "$2" --rounds 1 --warmup "${4:-0}" \
    --phases transcribe --label "stage-$1-c$2" --output "stages/$1-c$2-$3.json" > /dev/null; }
scrape() { curl -sf --max-time 10 "http://127.0.0.1:${PORT[$1]}/metrics" > "$2" || : > "$2"; }   # empty = failed scrape
mkdir -p stages profiles
# (a) request side + host flame graphs
up_all stage
for a in S2 D; do for c in 1 8; do
  load "$a" "$c" warm 3 $(( c * 2 > 8 ? c * 2 : 8 ))     # warm this concurrency, outside the window
  scrape "$a" "stages/$a-c$c.before.prom"
  if [ "$c" = 8 ] && command -v py-spy > /dev/null; then
    py-spy record --subprocesses --idle -d 25 -o "stages/$a-c8-host.svg" -p "${SPID[$a]}" & SPY=$!   # the PID we started
  fi
  load "$a" "$c" run 0 50 || echo "stage pass: measured load failed for $a c$c" >&2
  sleep 2; scrape "$a" "stages/$a-c$c.after.prom"                 # metrics publication can lag the last response
  [ -n "${SPY:-}" ] && wait "$SPY" || true; unset SPY
done; done
stop_both
# (b) device timelines
up_all prof PROFILE=1
for a in S2 D; do for c in 1 8; do
  load "$a" "$c" pwarm 3 $(( c * 2 > 8 ? c * 2 : 8 ))    # warm outside the capture
  http_post "http://127.0.0.1:${PORT[$a]}/start_profile" 60 || { echo "start_profile failed for $a c$c" >&2; continue; }
  load "$a" "$c" prof 0 50 && lrc=0 || lrc=$?
  http_post "http://127.0.0.1:${PORT[$a]}/stop_profile" 300 || echo "stop_profile failed for $a c$c" >&2   # always stopped
  [ $lrc = 0 ] || echo "stage pass: profiled load failed for $a c$c" >&2
  [ -d "profiles/$a" ] || { echo "no device profile written for $a c$c" >&2; continue; }
  mv "profiles/$a" "profiles/$a-c$c"
  save_evidence "stage-$a-c$c" "profiles/$a-c$c" || echo "evidence upload failed for $a c$c" >&2   # raw traces first
  neuron-explorer view -d "profiles/$a-c$c" --output-format perfetto || echo "trace conversion failed for $a c$c" >&2
  python3 "$HERE/summarize_profile.py" "profiles/$a-c$c" > "profiles/$a-c$c/STAGES.md" || true
done; done
stop_both
# tables
for a in S2 D; do { echo "# $a"; for c in 1 8; do
  python3 "$HERE/stage_table.py" "stages/$a-c$c.before.prom" "stages/$a-c$c.after.prom" "stages/$a-c$c-run.json" "$c" "profiles/$a-c$c/STAGES.md"; echo; done; } > "stages/$a.md"; done
echo "stage tables: stages/S2.md stages/D.md; flame graphs: stages/*-host.svg; traces: profiles/*-c*/"
