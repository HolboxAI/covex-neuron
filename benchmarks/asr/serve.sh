#!/usr/bin/env bash
# Serve one arm on two Trainium2 chips (TP8 at LNC2 = 8 logical cores).
# Usage: serve.sh <ARM> <PAIR 0-7> <PORT>       PROFILE=1 adds the built-in Neuron profiler endpoints.
set -euo pipefail
ARM=$1; PAIR=$2; PORT=$3
HERE=$(cd "$(dirname "$0")" && pwd); source "$HERE/arm_config.sh"
MODEL_DIR=${MODEL_DIR:-$HOME/models/Qwen3-ASR-1.7B}
CFG=$(arm_config "$ARM")
export NEURON_RT_VISIBLE_CORES="$((PAIR * 8))-$((PAIR * 8 + 7))"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 NEURON_LIBTORCH_COMPILATION_TIMEOUT=1800 NEURON_SKIP_EFA_AFFINITY=1
PROF_ARGS=()
if [ "${PROFILE:-0}" = 1 ]; then
  # vllm-neuron built-in profiler (docs/design/vllm/neuron-profiling.md): /start_profile and /stop_profile
  # write system + device profiles for every rank, viewable in Neuron Explorer / Perfetto.
  CFG=$(python3 -c 'import json,sys; c=json.loads(sys.argv[1]); c["neuron_profiler"]={"activities":["system_profile","device_profile"],"neuron_cores":list(range(8)),"output_dir":sys.argv[2]}; print(json.dumps(c))' "$CFG" "$PWD/profiles/$ARM")
  PROF_ARGS=(--profiler-config '{"profiler": "cuda"}')
fi
mkdir -p logs
{ echo "arm=$ARM pair=$PAIR cores=$NEURON_RT_VISIBLE_CORES commit=$(git -C "$HERE" rev-parse --short HEAD) profile=${PROFILE:-0}"
  env | grep -E '^(VLLM_NEURON|COVEX|NEURON_RT)' | sort; echo "$CFG"; } > "logs/$ARM.meta"
exec vllm serve "$MODEL_DIR" \
  --served-model-name qwen3-asr-1.7b --dtype bfloat16 \
  --tensor-parallel-size 8 --max-model-len 1024 \
  --max-num-seqs 16 --max-num-batched-tokens 1024 \
  --limit-mm-per-prompt '{"audio": 1}' --no-enable-prefix-caching --mm-processor-cache-gb 0 \
  --port "$PORT" "${PROF_ARGS[@]}" --additional-config "$CFG" > "logs/$ARM.server.log" 2>&1
