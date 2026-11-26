#!/usr/bin/env bash
# Start one benchmark arm on one Trainium2 chip. Usage: serve_arm.sh <ARM> <CHIP 0-15> <PORT>
# ARM: S0 S1 S2 C1 C2 C3 (see benchmarks/plans/2x-round1.md). BEST_STOCK_ENV selects S1 or S2 memory for C arms.
set -euo pipefail
ARM=$1; CHIP=$2; PORT=$3
MODEL=${MODEL:-Qwen/Qwen3-8B}
FIRST=$((CHIP * 4)); LAST=$((FIRST + 3))       # TP4 at LNC2 = 4 logical cores = one chip
export NEURON_RT_VISIBLE_CORES="${FIRST}-${LAST}"
export VLLM_NEURON_COMPILATION_TIMEOUT=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
BEST_STOCK_ENV=${BEST_STOCK_ENV:-VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.8}
CFG='{}'
case "$ARM" in
  S0) ;;
  S1) export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.8 ;;
  S2) export VLLM_NEURON_RESERVED_HEADROOM_GIB=4 ;;
  C1) export "$BEST_STOCK_ENV"; CFG='{"kv_cache_overcommit": true}' ;;
  C2) export "$BEST_STOCK_ENV"; CFG='{"num_batched_tokens_bucket_step": 128, "num_seqs_bucket_step": 8}' ;;
  C3) export "$BEST_STOCK_ENV"; CFG='{"kv_cache_overcommit": true, "num_batched_tokens_bucket_step": 128, "num_seqs_bucket_step": 8}' ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
mkdir -p logs
echo "arm=$ARM chip=$CHIP cores=$NEURON_RT_VISIBLE_CORES cfg=$CFG commit=$(git rev-parse --short HEAD)" | tee "logs/${ARM}.meta"
exec vllm serve "$MODEL" \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --max-num-seqs 64 \
  --port "$PORT" \
  --additional-config "{\"neuron_config\": $CFG}" \
  > "logs/${ARM}.server.log" 2>&1
