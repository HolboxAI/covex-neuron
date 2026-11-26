#!/usr/bin/env bash
# Run the round-1 workload matrix against one running arm. Usage: bench_arm.sh <ARM> <PORT> <OUTDIR>
set -euo pipefail
ARM=$1; PORT=$2; OUT=$3; MODEL=${MODEL:-Qwen/Qwen3-8B}; REPEATS=${REPEATS:-3}
mkdir -p "$OUT"
run() { # wl isl osl conc prompts rep
  vllm bench serve --backend vllm --base-url "http://127.0.0.1:$PORT" --model "$MODEL" \
    --dataset-name random --random-input-len "$2" --random-output-len "$3" --ignore-eos \
    --max-concurrency "$4" --num-prompts "$5" --seed "$6" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90 \
    --save-result --result-dir "$OUT" --result-filename "${ARM}_${1}_c${4}_r${6}.json" > /dev/null
}
max() { [ "$1" -gt "$2" ] && echo "$1" || echo "$2"; }
# c=1 once (same code path in every arm; regression check only), c>=8 REPEATS times.
run W1 1024 190 1 16 1; run W2 1024 1024 1 8 1; run W3 8192 1024 1 8 1
for rep in $(seq 1 "$REPEATS"); do
  for c in 8 16 32 64; do run W1 1024 190  "$c" "$(max 32 $((4*c)))" "$rep"; done
  for c in 16 32 64;   do run W2 1024 1024 "$c" "$(max 16 $((2*c)))" "$rep"; done
  for c in 8 16 32;    do run W3 8192 1024 "$c" "$(max 16 $((2*c)))" "$rep"; done
done
