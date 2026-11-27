#!/usr/bin/env bash
# Serve one arm on its own two Trainium2 chips (TP8 at LNC2 = 8 logical cores; TP4 arms use the first chip).
# Usage: serve.sh <ARM> <PAIR 0-7> <PORT>       PROFILE=1 adds the built-in Neuron profiler endpoints.
set -euo pipefail
ARM=$1; PAIR=$2; PORT=$3
HERE=$(cd "$(dirname "$0")" && pwd); source "$HERE/arm_config.sh"
MODEL_DIR=${MODEL_DIR:-$HOME/models/Qwen3-ASR-1.7B}
arm_config "$ARM" > /dev/null; CFG=$(arm_config "$ARM")
# vllm-neuron multiproc requires NEURON_VISIBLE_DEVICES (logical-core range "a-b"); NEURON_RT_VISIBLE_CORES is rejected.
export NEURON_VISIBLE_DEVICES="$((PAIR * 8))-$((PAIR * 8 + ARM_TP - 1))"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 NEURON_LIBTORCH_COMPILATION_TIMEOUT=1800 NEURON_SKIP_EFA_AFFINITY=1
PROF_ARGS=()
if [ "${PROFILE:-0}" = 1 ]; then
  # vllm-neuron built-in profiler (docs/design/vllm/neuron-profiling.md): /start_profile and /stop_profile
  # write system + device profiles for every rank, viewable in Neuron Explorer / Perfetto.
  CFG=$(python3 -c 'import json,sys; c=json.loads(sys.argv[1]); c["neuron_profiler"]={"activities":["system_profile","device_profile"],"neuron_cores":list(range(int(sys.argv[3]))),"output_dir":sys.argv[2]}; print(json.dumps(c))' "$CFG" "$PWD/profiles/$ARM" "$ARM_TP")
  PROF_ARGS=(--profiler-config '{"profiler": "cuda"}')
fi
mkdir -p logs
{ echo "arm=$ARM tp=$ARM_TP pair=$PAIR cores=$NEURON_VISIBLE_DEVICES commit=$(git -C "$HERE" rev-parse --short HEAD) profile=${PROFILE:-0}"
  env | grep -E '^(VLLM_NEURON|COVEX|NEURON_RT|NEURON_VISIBLE)' | sort; echo "$CFG"; } > "logs/$ARM${LOG_TAG:-}.meta"
# Equal, disjoint host CPUs per arm (8 arms x 24 vCPUs on the 192-vCPU host) so arms do not steal host time
# from each other. SOLO=1 drops the pinning for the solo S2 re-run.
CPUS="$((PAIR * 24))-$((PAIR * 24 + 23))"; PIN=(taskset -c "$CPUS"); [ "${SOLO:-0}" = 1 ] && PIN=()
echo "cpus=${PIN[*]:-all}" >> "logs/$ARM${LOG_TAG:-}.meta"
exec "${PIN[@]}" vllm serve "$MODEL_DIR" \
  --served-model-name qwen3-asr-1.7b --dtype bfloat16 \
  --tensor-parallel-size "$ARM_TP" --max-model-len 1024 \
  --max-num-seqs 16 --max-num-batched-tokens 1024 \
  --limit-mm-per-prompt '{"audio": 1}' --no-enable-prefix-caching --mm-processor-cache-gb 0 \
  --port "$PORT" "${PROF_ARGS[@]}" --additional-config "$CFG" > "logs/$ARM${LOG_TAG:-}.server.log" 2>&1
