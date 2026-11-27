#!/usr/bin/env bash
# Capture one steady-state profile window from an arm started with PROFILE=1, then summarize every stage.
# Usage: profile.sh <ARM> <PORT> [CONCURRENCY=8]
set -euo pipefail
ARM=$1; PORT=$2; C=${3:-8}; HERE=$(cd "$(dirname "$0")" && pwd)
CLIENT_DIR=${CLIENT_DIR:-$HOME/qwen3-asr-inference-cuda-trainium}
run() { python3 "$CLIENT_DIR/benchmark/benchmark.py" --base-url "http://127.0.0.1:$PORT" --model qwen3-asr-1.7b \
          --data-dir "$CLIENT_DIR/data" --num-samples 32 --concurrency "$C" --rounds 1 --warmup 3 \
          --phases transcribe --label "profile-$ARM" --output "/tmp/profile-$ARM-$1.json" > /dev/null; }
run warm                                                   # warm every compiled shape first
curl -sf -X POST "http://127.0.0.1:$PORT/start_profile" > /dev/null
run captured
curl -sf -X POST "http://127.0.0.1:$PORT/stop_profile" > /dev/null
neuron-explorer view -d "profiles/$ARM" --output-format perfetto
python3 "$HERE/summarize_profile.py" "profiles/$ARM" | tee "profiles/$ARM/STAGES.md"
