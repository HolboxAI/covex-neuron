#!/usr/bin/env bash
# K9 engine check on an x86 Neuron DLAMI box, CPU mode, small real Qwen3 at ~/models-small/Qwen3-32B (rehearsal layout).
# Starts k9-T (fork, no speculation) and k9-P (fork, predicted outputs), runs cpu_check.py, writes results/k9-cpu.json.
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd); F=$H/../five; source "$H/../lib.sh"
mkdir -p logs results
export VLLM_NEURON_CPU_MODE=1 REHEARSAL_EXTRA="--gpu-memory-utilization 0.2" MODELS_DIR=${MODELS_DIR:-$HOME/models-small}
n=$(nproc); half=$((n / 2))
taskset -c "0-$((half - 1))" "$F/serve.sh" k9-T 8300 & T=$!
taskset -c "$half-$((n - 1))" "$F/serve.sh" k9-P 8301 & P=$!
trap 'stop_arm $T; stop_arm $P' EXIT
rt=$(wait_ready $T 8300 ${K9_READY_SECS:-2400} gen m) || { echo "k9-T not ready: $rt"; exit 2; }
rp=$(wait_ready $P 8301 ${K9_READY_SECS:-2400} gen m) || { echo "k9-P not ready: $rp"; exit 2; }
python3 "$H/cpu_check.py" 8300 8301 results/k9-cpu.json --n 6 --tokens 48
