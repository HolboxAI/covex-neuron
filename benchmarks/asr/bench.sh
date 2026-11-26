#!/usr/bin/env bash
# Run Dax's benchmark client (pinned commit) against one running arm.
# Usage: bench.sh <ARM> <PORT> [NUM_SAMPLES=200] [CONCURRENCY=1,4,8,16]
set -euo pipefail
ARM=$1; PORT=$2; N=${3:-200}; CONC=${4:-1,4,8,16}; ROUNDS=${ROUNDS:-3}
# The client runs on the same pinned CPU set as its arm (PAIR = PORT - 8000).
PIN=(taskset -c "$(( (PORT - 8000) * 24 ))-$(( (PORT - 8000) * 24 + 23 ))"); [ "${SOLO:-0}" = 1 ] && PIN=()
CLIENT_DIR=${CLIENT_DIR:-$HOME/qwen3-asr-inference-cuda-trainium}
[ -d "$CLIENT_DIR" ] || git clone -q https://github.com/HolboxAI/qwen3-asr-inference-cuda-trainium "$CLIENT_DIR"
git -C "$CLIENT_DIR" checkout -q 4ce1ddb   # same client and dataset slice as the CUDA (1xL4, 8xL4) baselines
mkdir -p results metrics
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "arm $ARM (port $PORT) is not up; skipping" >&2; exit 0; }
curl -sf "http://127.0.0.1:$PORT/metrics" > "metrics/$ARM.before.prom" || true
"${PIN[@]}" python3 "$CLIENT_DIR/benchmark/benchmark.py" --base-url "http://127.0.0.1:$PORT" --model qwen3-asr-1.7b \
  --data-dir "$CLIENT_DIR/data" --num-samples "$N" --concurrency "$CONC" --rounds "$ROUNDS" --warmup 3 \
  --label "trn2-$ARM" --output "results/$ARM.json"
# vLLM / vllm-neuron Prometheus counters after the run: queue, prefill, decode and TTFT histograms,
# prompt-token histogram, and vllm_neuron:num_batched_tokens_padding / num_seqs_padding (padding waste per bucket).
curl -sf "http://127.0.0.1:$PORT/metrics" > "metrics/$ARM.after.prom" || true
