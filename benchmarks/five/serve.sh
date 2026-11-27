#!/usr/bin/env bash
# Start one arm. Usage: serve.sh <ARM> <PORT>   (ARM = <model>-<T|C>; T = tuned stock, C = COVEX flags)
# Core ranges are fixed per arm so all arms share one trn2.48xlarge (16 chips = 64 logical cores at LNC2).
set -euo pipefail
ARM=$1; PORT=$2; M=$HOME/models
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 NEURON_LIBTORCH_COMPILATION_TIMEOUT=3600 NEURON_SKIP_EFA_AFFINITY=1
export VLLM_NEURON_COMPILATION_TIMEOUT=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
mkdir -p logs
[ "${DRYRUN:-0}" = 1 ] && cd "$(mktemp -d)"
case "$ARM" in
  q32-T|q32-C)  # Qwen3-32B, TP8, 32K context, cap 0.8 (tuned stock); C = dynamic FA + in-kernel KV write
    [ "$ARM" = q32-T ] && CORES=0-7 || { CORES=8-15; export COVEX_DECODE_DYNAMIC_FA=1 COVEX_QWEN3_INKERNEL_KV_WRITE=1; }
    export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.8
    ARGS=("$M/Qwen3-32B" --served-model-name m --dtype bfloat16 --tensor-parallel-size 8 --max-model-len 32768
          --max-num-batched-tokens 8192 --max-num-seqs 8 --no-enable-prefix-caching
          ); CFG='{"neuron_config":{"num_batched_tokens_buckets":[1024,8192],"num_seqs_buckets":[1,2,4,8],"on_device_sampling_config":{"all_greedy":true}}}' ;;
  emb-T|emb-C)  # Qwen3-Embedding-8B, TP8, tutorial config; C = 16 prefills per step
    [ "$ARM" = emb-T ] && { CORES=32-39; EXTRA=''; } || { CORES=40-47; EXTRA='"max_prefills_per_batch":16,'; }
    ARGS=("$M/Qwen3-Embedding-8B" --served-model-name m --runner pooling --dtype bfloat16 --tensor-parallel-size 8
          --max-model-len 4096 --max-num-batched-tokens 4096 --max-num-seqs 32 --no-enable-prefix-caching
          ); CFG="{\"neuron_config\":{${EXTRA}\"num_batched_tokens_buckets\":[128,256,512,1024,2048,4096]}}" ;;
  vl-T|vl-C)  # Qwen3-VL-32B, TP4 (1 chip), tutorial BF16 config; C = vision MLP on NKI kernel + LayerNorm fold
    [ "$ARM" = vl-T ] && CORES=48-51 || { CORES=52-55; export COVEX_VL_VISION_MLP_NKI=1 COVEX_VL_VISION_LN_IN_MLP=1; }
    ARGS=("$M/Qwen3-VL-32B-Instruct" --served-model-name m --max-model-len 8192 --max-num-batched-tokens 8192
          --max-num-seqs 4 --tensor-parallel-size 4
          ); CFG='{"neuron_config":{"quantization":"bf16","num_batched_tokens_buckets":[8192],"num_seqs_buckets":[4],"on_device_sampling_config":{"all_greedy":true}},"encoder_neuron_config":{"num_encoder_input_buckets":[2048],"encoder_attention_block_size":2048}}' ;;
  oss-T)  # GPT-OSS-20B, TP8, tutorial config (stock only; no COVEX change applies to it)
    CORES=56-63
    ARGS=("$M/gpt-oss-20b" --served-model-name m --tensor-parallel-size 8 --max-model-len 16384
          --max-num-batched-tokens 8192 --max-num-seqs 4 --no-enable-prefix-caching --hf-overrides '{"quantization_config": {}}'
          ); CFG='{"neuron_config":{"num_batched_tokens_buckets":[8192],"num_seqs_buckets":[4],"on_device_sampling_config":{"all_greedy":true}}}' ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
export NEURON_VISIBLE_DEVICES=$CORES
PROF=()
if [ "${PROFILE:-0}" = 1 ]; then   # vllm-neuron built-in profiler; records only between /start_profile and /stop_profile
  NCORES=$(( ${CORES#*-} - ${CORES%-*} + 1 ))
  CFG=$(python3 -c 'import json,sys; c=json.loads(sys.argv[1]); c["neuron_profiler"]={"activities":["system_profile","device_profile"],"neuron_cores":list(range(int(sys.argv[3]))),"output_dir":sys.argv[2]}; print(json.dumps(c))' "$CFG" "$PWD/profiles/$ARM" "$NCORES")
  PROF=(--profiler-config '{"profiler": "cuda"}')
fi
python3 -c 'import json,sys; json.loads(sys.argv[1])' "$CFG" || { echo "bad CFG for $ARM" >&2; exit 2; }
if [ "${DRYRUN:-0}" = 1 ]; then echo "$ARM cores=$CORES flags=$(env | grep -oE '^COVEX_[A-Z_]+' | sort | tr '\n' ' ')cfg=$CFG"; exit 0; fi
{ echo "arm=$ARM cores=$CORES profile=${PROFILE:-0}"; env | grep -E '^(COVEX|VLLM_NEURON|NEURON_VISIBLE)' | sort; printf '%q ' "${ARGS[@]}"; echo; echo "$CFG"; } > "logs/$ARM.meta"
exec vllm serve "${ARGS[@]}" --additional-config "$CFG" "${PROF[@]}" --port "$PORT" > "logs/$ARM.server.log" 2>&1
