#!/usr/bin/env bash
# Start one arm. Usage: serve.sh <ARM> <PORT>   (ARM = <model>-<T|C>; T = tuned stock, C = COVEX flags)
# Core ranges are fixed per arm so all arms share one trn2.48xlarge (16 chips = 64 logical cores at LNC2).
set -euo pipefail
ARM=$1; PORT=$2; M=${MODELS_DIR:-$HOME/models}; TAG=${LOG_TAG:-}
# Inherited COVEX flags never leak into an arm: each arm sets exactly its own below.
for v in $(env | grep -oE '^COVEX_[A-Z0-9_]+' | grep -vE '^COVEX_(PROVENANCE_DIR|ARM)$'); do unset "$v"; done
VLLM=(vllm)   # the fork arms; q32-U runs the upstream venv's interpreter (UPSTREAM_PY), see the q32 case
# REHEARSAL_EXTRA: extra vllm args for the CPU-mode rehearsal only (never set on Trainium).
read -r -a REH_ARGS <<< "${REHEARSAL_EXTRA:-}"   # own name: arms use $EXTRA for config fragments
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 NEURON_LIBTORCH_COMPILATION_TIMEOUT=3600 NEURON_SKIP_EFA_AFFINITY=1
export VLLM_NEURON_COMPILATION_TIMEOUT=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
mkdir -p logs
[ "${DRYRUN:-0}" = 1 ] && cd "$(mktemp -d)"
case "$ARM" in
  q32-U|q32-T|q32-K|q32-F|q32-C)  # Qwen3-32B, TP8, 32K context, cap 0.5, identical config in every arm:
    #   U = pinned upstream vllm-neuron in its own venv (UPSTREAM_PY); provenance checked per process by q32/run.sh
    #   T = this fork, flags off   K = in-kernel KV write only   F = dynamic FA only   C = both
    # cap 0.8 failed to load (2026-11-27): 7.64 GiB weights/core + 13.4 GiB KV left < 3 GiB for NEFFs. 0.5 -> 10.8 GiB KV,
    # enough for 8 seqs x 32K (8.6 GiB), ~5.5 GiB headroom. Checked by tests/covex/test_five_arms.py::test_hbm_budget.
    case $ARM in
      # U: its own venv (a copy of the DLAMI venv with only pinned upstream installed); launched by absolute
      # interpreter from an empty working directory, so neither cwd nor the fork's install can supply vllm_neuron.
      q32-U) CORES=${Q32_CORES:-0-7}; VLLM=("${UPSTREAM_PY:?UPSTREAM_PY (upstream venv python) required}" -m vllm.entrypoints.cli.main) ;;
      q32-T) CORES=${Q32_CORES:-0-7} ;;
      q32-K) CORES=${Q32_CORES:-8-15}; export COVEX_QWEN3_INKERNEL_KV_WRITE=1 ;;
      q32-F) CORES=${Q32_CORES:-8-15}; export COVEX_DECODE_DYNAMIC_FA=1 ;;
      q32-C) CORES=${Q32_CORES:-8-15}; export COVEX_DECODE_DYNAMIC_FA=1 COVEX_QWEN3_INKERNEL_KV_WRITE=1 ;;
    esac
    export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.5
    ARGS=("$M/Qwen3-32B" --served-model-name m --dtype bfloat16 --tensor-parallel-size 8 --max-model-len 32768
          --max-num-batched-tokens 8192 --max-num-seqs 8 --no-enable-prefix-caching
          ); CFG='{"neuron_config":{"num_batched_tokens_buckets":[1024,8192],"num_seqs_buckets":[1,2,4,8],"on_device_sampling_config":{"all_greedy":true}}}' ;;
  emb-T|emb-C)  # Qwen3-Embedding-8B, TP8, tutorial config; C = 16 prefills per step
    # KNOWN BROKEN (2026-11-27): packed prefills change max_query_len per step -> torch.compile recompile ->
    # 'fail_on_recompile'. Needs a warmup/padding fix for the pooling path before emb-C can run.
    [ "$ARM" = emb-T ] && { CORES=32-39; EXTRA=''; } || { CORES=40-47; EXTRA='"max_prefills_per_batch":16,'; }
    ARGS=("$M/Qwen3-Embedding-8B" --served-model-name m --runner pooling --dtype bfloat16 --tensor-parallel-size 8
          --max-model-len 4096 --max-num-batched-tokens 4096 --max-num-seqs 32 --no-enable-prefix-caching
          ); CFG="{\"neuron_config\":{${EXTRA}\"num_batched_tokens_buckets\":[128,256,512,1024,2048,4096]}}" ;;
  vl-T|vl-C)  # Qwen3-VL-32B, TP4 (1 chip), tutorial BF16 config; C = vision MLP on NKI kernel + LayerNorm fold
    [ "$ARM" = vl-T ] && CORES=48-51 || { CORES=52-55; export COVEX_VL_VISION_MLP_NKI=1 COVEX_VL_VISION_LN_IN_MLP=1; }
    # 66.7 GB weights over TP4 = 15.5 GiB/core; default cap 0.30 leaves ~2.4 GiB for NEFFs (same margin that failed
    # Qwen3-32B). 0.15 -> 3.2 GiB KV (4 seqs x 8K needs 2 GiB), ~5.3 GiB headroom. Same in T and C.
    export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.15
    # Single-shot prefill (batched tokens == max_model_len) is incompatible with prefix caching (failed 2026-11-27).
    ARGS=("$M/Qwen3-VL-32B-Instruct" --served-model-name m --max-model-len 8192 --max-num-batched-tokens 8192
          --max-num-seqs 4 --tensor-parallel-size 4 --no-enable-prefix-caching
          ); CFG='{"neuron_config":{"quantization":"bf16","num_batched_tokens_buckets":[8192],"num_seqs_buckets":[4],"on_device_sampling_config":{"all_greedy":true}},"encoder_neuron_config":{"num_encoder_input_buckets":[2048],"encoder_attention_block_size":2048}}' ;;
  oss-T)  # GPT-OSS-20B, TP8, tutorial config (stock only; no COVEX change applies to it)
    CORES=56-63
    ARGS=("$M/gpt-oss-20b" --served-model-name m --tensor-parallel-size 8 --max-model-len 16384
          --max-num-batched-tokens 8192 --max-num-seqs 4 --no-enable-prefix-caching --hf-overrides '{"quantization_config": {}}'
          ); CFG='{"neuron_config":{"num_batched_tokens_buckets":[8192],"num_seqs_buckets":[4],"on_device_sampling_config":{"all_greedy":true}}}' ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
# CPU mode runs on host CPUs; core ranges apply on Trainium and in CPU compile (compile-cache prefix).
[ "${VLLM_NEURON_CPU_MODE:-0}" = 1 ] || export NEURON_VISIBLE_DEVICES=$CORES
PROF=()
if [ "${PROFILE:-0}" = 1 ]; then   # vllm-neuron built-in profiler; records only between /start_profile and /stop_profile
  NCORES=$(( ${CORES#*-} - ${CORES%-*} + 1 ))
  CFG=$(python3 -c 'import json,sys; c=json.loads(sys.argv[1]); c["neuron_profiler"]={"activities":["system_profile","device_profile"],"neuron_cores":list(range(int(sys.argv[3]))),"output_dir":sys.argv[2]}; print(json.dumps(c))' "$CFG" "$PWD/profiles/$ARM" "$NCORES")
  PROF=(--profiler-config '{"profiler": "cuda"}')
fi
python3 -c 'import json,sys; json.loads(sys.argv[1])' "$CFG" || { echo "bad CFG for $ARM" >&2; exit 2; }
export COVEX_ARM=$ARM
if [ -n "${COVEX_PROVENANCE_DIR:-}" ]; then   # per-process record of where vllm_neuron is imported from
  export PYTHONPATH="$(cd "$(dirname "$0")/../provenance" && pwd)${PYTHONPATH:+:$PYTHONPATH}"; fi
if [ -n "${ARM_CACHE_BASE:-}" ]; then          # each arm its own compile cache: no arm can run another's NEFFs
  unset NEURON_LIBTORCH_REMOTE_CACHE           # a shared remote cache would bypass the per-arm roots
  export NEURON_LIBTORCH_CACHE_ROOT=$ARM_CACHE_BASE/$ARM VLLM_CACHE_ROOT=$ARM_CACHE_BASE/$ARM/vllm \
         TORCH_NEURONX_NEFF_CACHE_DIR=$ARM_CACHE_BASE/$ARM/neff; fi
if [ "${DRYRUN:-0}" = 1 ]; then
  echo "$ARM cores=$CORES flags=$(env | grep -oE '^COVEX_[A-Z_]+' | grep -vxE 'COVEX_(ARM|PROVENANCE_DIR)' | sort | tr '\n' ' ')cfg=$CFG"
  printf 'argv:'; printf ' %q' "${VLLM[@]}" serve "${ARGS[@]}" ${REH_ARGS[@]+"${REH_ARGS[@]}"} --additional-config "$CFG" ${PROF[@]+"${PROF[@]}"} --port "$PORT"; echo; exit 0; fi
LOGS=$PWD/logs
{ echo "arm=$ARM cores=$CORES profile=${PROFILE:-0} launcher=${VLLM[*]}"; env | grep -E '^(COVEX|VLLM_NEURON|NEURON_VISIBLE|NEURON_LIBTORCH_CACHE_ROOT|VLLM_CACHE_ROOT)' | sort; printf '%q ' "${ARGS[@]}"; echo; echo "$CFG"; } > "$LOGS/$ARM$TAG.meta"
cd "$(mktemp -d)"   # empty working directory: `python -m` would otherwise put a source checkout first on sys.path
exec "${VLLM[@]}" serve "${ARGS[@]}" ${REH_ARGS[@]+"${REH_ARGS[@]}"} --additional-config "$CFG" ${PROF[@]+"${PROF[@]}"} --port "$PORT" > "$LOGS/$ARM$TAG.server.log" 2>&1
