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
  q32-U|q32-Ut|q32-T|q32-K|q32-F|q32-F4|q32-C)  # Qwen3-32B, TP8, 32K context, cap 0.5, identical config in every arm:
    #   U = pinned upstream vllm-neuron in its own venv (UPSTREAM_PY); provenance checked per process by q32/run.sh
    #   Ut = the same upstream + decode_context_length_buckets [2048, 8192] (upstream feature; selects MRv1)
    #   T = this fork, flags off   K = in-kernel KV write only   F = dynamic FA only   C = both
    # cap 0.8 failed to load (2026-11-27): 7.64 GiB weights/core + 13.4 GiB KV left < 3 GiB for NEFFs. 0.5 -> 10.8 GiB KV,
    # enough for 8 seqs x 32K (8.6 GiB), ~5.5 GiB headroom. Checked by tests/covex/test_five_arms.py::test_hbm_budget.
    case $ARM in
      # U: its own venv (a copy of the DLAMI venv with only pinned upstream installed); launched by absolute
      # interpreter from an empty working directory, so neither cwd nor the fork's install can supply vllm_neuron.
      q32-U) CORES=${Q32_CORES:-0-7}; VLLM=("${UPSTREAM_PY:?UPSTREAM_PY (upstream venv python) required}" -m vllm.entrypoints.cli.main) ;;
      q32-Ut) CORES=${Q32_CORES:-16-23}; VLLM=("${UPSTREAM_PY:?UPSTREAM_PY (upstream venv python) required}" -m vllm.entrypoints.cli.main)
              CTX='"decode_context_length_buckets":[2048,8192],' ;;
      q32-T) CORES=${Q32_CORES:-0-7} ;;
      q32-K) CORES=${Q32_CORES:-8-15}; export COVEX_QWEN3_INKERNEL_KV_WRITE=1 ;;
      q32-F4) CORES=${Q32_CORES:-8-15}; export COVEX_DECODE_DYNAMIC_FA=1 COVEX_NKILIB_FA_TILE_SIZE=4096
              export PYTHONPATH="${Q32_NKI_SRC:?set Q32_NKI_SRC to the corrected NKI checkout src directory}${PYTHONPATH:+:$PYTHONPATH}"
              unset NKILIB_FORCE_BUNDLED_LIBRARY ;;
      q32-F) CORES=${Q32_CORES:-8-15}; export COVEX_DECODE_DYNAMIC_FA=1 ;;
      q32-C) CORES=${Q32_CORES:-8-15}; export COVEX_DECODE_DYNAMIC_FA=1 COVEX_QWEN3_INKERNEL_KV_WRITE=1 ;;
    esac
    export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.5
    # Equal explicit headroom in every arm (run 2): KV = 24 GiB - 7.64 GiB weights - 7 GiB = ~9.4 GiB >= 8.6 GiB for
    # 8 x 32K, and 7 GiB for NEFFs + profiler buffers (run 1's cap-0.5 split left ~5.5 GiB and profiling failed).
    # Ut loads extra context-bucket NEFFs, so a fixed KV fraction would leave it less room than the others.
    export VLLM_NEURON_RESERVED_HEADROOM_GIB=${Q32_HEADROOM_GIB:-7}
    ARGS=("$M/Qwen3-32B" --served-model-name m --dtype bfloat16 --tensor-parallel-size 8 --max-model-len 32768
          --max-num-batched-tokens 8192 --max-num-seqs 8 --no-enable-prefix-caching
          ); CFG="{\"neuron_config\":{${CTX:-}\"num_batched_tokens_buckets\":[1024,8192],\"num_seqs_buckets\":[1,2,4,8],\"on_device_sampling_config\":{\"all_greedy\":true}}}" ;;
  k9-U|k9-T|k9-E3|k9-P)  # K9 code-edit benchmark, Qwen3-32B BF16 TP8 at 10,240 context (where upstream validated EAGLE3),
    # 4 sequences, identical config except the speculation:
    #   U  = pinned upstream, no speculation          E3 = pinned upstream + EAGLE3 draft (4 tokens)
    #   P  = fork, predicted outputs / prompt lookup (method ngram, 4 tokens; host-proposed, synchronous scheduling)
    SPEC=()
    case $ARM in
      k9-U)  CORES=${K9_CORES:-0-7};  VLLM=("${UPSTREAM_PY:?UPSTREAM_PY (upstream venv python) required}" -m vllm.entrypoints.cli.main) ;;
      k9-E3) CORES=${K9_CORES:-8-15}; VLLM=("${UPSTREAM_PY:?UPSTREAM_PY (upstream venv python) required}" -m vllm.entrypoints.cli.main)
             SPEC=(--speculative-config "{\"method\":\"eagle3\",\"model\":\"$M/Qwen3-32B-speculator.eagle3\",\"num_speculative_tokens\":4}") ;;
      k9-T)  CORES=${K9_CORES:-24-31} ;;   # fork, no speculation: the exactness reference for k9-P (CPU check)
      k9-P)  CORES=${K9_CORES:-16-23}
             SPEC=(--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":3,"prompt_lookup_max":8}' --no-async-scheduling) ;;
    esac
    # K9_HEADROOM_GIB=none: no explicit headroom (CPU-mode check: the worker then uses fair share x util x cap)
    if [ "${K9_HEADROOM_GIB:-7}" = none ]; then unset VLLM_NEURON_RESERVED_HEADROOM_GIB
    else export VLLM_NEURON_RESERVED_HEADROOM_GIB=${K9_HEADROOM_GIB:-7}; fi
    ARGS=("$M/Qwen3-32B" --served-model-name m --dtype bfloat16 --tensor-parallel-size 8 --max-model-len 10240
          --max-num-batched-tokens 8192 --max-num-seqs 4 --no-enable-prefix-caching ${SPEC[@]+"${SPEC[@]}"}
          ); CFG='{"neuron_config":{"num_batched_tokens_buckets":[1024,8192],"num_seqs_buckets":[1,2,4],"on_device_sampling_config":{"all_greedy":true}}}' ;;
  emb-U|emb-T|emb-C|emb-P1x|emb-P4|emb-P8|emb-P8t|emb-P16)  # Qwen3-Embedding-8B, TP8, tutorial config, identical in every arm except
    # the prefills packed per step: U = pinned upstream (own venv), T = fork 1 per step, P<N> = fork N per step
    # (C = legacy name for P16). Packing was KNOWN BROKEN (fail_on_recompile on max_query_len) until the canonical
    # per-bucket max_query_len fix (bucket_utils.canonical_prefill_max_query_len).
    case $ARM in
      emb-U) CORES=${EMB_CORES:-32-39}; EXTRA=''
             VLLM=("${UPSTREAM_PY:?UPSTREAM_PY (upstream venv python) required}" -m vllm.entrypoints.cli.main) ;;
      emb-T) CORES=${EMB_CORES:-32-39}; EXTRA='' ;;
      # P1x: one prefill per step, but through the packed attention path (diagnostic: kernel numerics vs packing)
      emb-P1x) CORES=${EMB_CORES:-32-39}; EXTRA=''; export COVEX_FORCE_PACKED_ATTENTION=1 ;;
      emb-P4) CORES=${EMB_CORES:-40-47}; EXTRA='"max_prefills_per_batch":4,' ;;
      emb-P8) CORES=${EMB_CORES:-40-47}; EXTRA='"max_prefills_per_batch":8,' ;;
      # P8t: P8 plus one INFO line per prefill step (members, offsets, padding) for the isolation diagnostic
      emb-P8t) CORES=${EMB_CORES:-40-47}; EXTRA='"max_prefills_per_batch":8,'; export COVEX_PACK_TRACE=1 ;;
      emb-C|emb-P16) CORES=${EMB_CORES:-40-47}; EXTRA='"max_prefills_per_batch":16,' ;;
    esac
    ARGS=("$M/Qwen3-Embedding-8B" --served-model-name m --runner pooling --dtype bfloat16 --tensor-parallel-size 8
          --max-model-len 4096 --max-num-batched-tokens 4096 --max-num-seqs 32 --no-enable-prefix-caching
          ); CFG="{\"neuron_config\":{${EXTRA}\"num_batched_tokens_buckets\":[128,256,512,1024,2048,4096]}}" ;;
  vl-U|vl-T|vl-C|vl-Ub|vl-Cb)  # Qwen3-VL-32B, TP4 (1 chip), tutorial BF16 config; C = vision MLP on NKI kernel + LayerNorm fold
    # *b = the same arm with prefill buckets [512..8192] instead of the single 8192 bucket: campaign 3fd284f showed the
    # image is 3% of a 1.58 s request and a text-only 1-token request already takes 1.55 s (every prompt padded to 8192).
    case $ARM in vl-C|vl-Cb) CORES=52-55; export COVEX_VL_VISION_MLP_NKI=1 COVEX_VL_VISION_LN_IN_MLP=1 ;; *) CORES=48-51 ;; esac
    VL_BUCKETS='[8192]'; case $ARM in *b) VL_BUCKETS='[512,1024,2048,4096,8192]' ;; esac
    # 66.7 GB weights over TP4 = 15.5 GiB/core; default cap 0.30 leaves ~2.4 GiB for NEFFs (same margin that failed
    # Qwen3-32B). 0.15 -> 3.2 GiB KV (4 seqs x 8K needs 2 GiB), ~5.3 GiB headroom. Same in T and C.
    export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.15
    # Single-shot prefill (batched tokens == max_model_len) is incompatible with prefix caching (failed 2026-11-27).
    ARGS=("$M/Qwen3-VL-32B-Instruct" --served-model-name m --max-model-len 8192 --max-num-batched-tokens 8192
          --max-num-seqs 4 --tensor-parallel-size 4 --no-enable-prefix-caching
          ); CFG="{\"neuron_config\":{\"quantization\":\"bf16\",\"num_batched_tokens_buckets\":$VL_BUCKETS,\"num_seqs_buckets\":[4],\"on_device_sampling_config\":{\"all_greedy\":true}},\"encoder_neuron_config\":{\"num_encoder_input_buckets\":[2048],\"encoder_attention_block_size\":2048}}" ;;
  oss-U|oss-P|oss-T|oss-U8k|oss-P8k)  # GPT-OSS-20B BF16. P omits padded tokens from prefill expert routing.
    CORES=56-63
    case $ARM in
      oss-U|oss-P) OSS_BUCKETS='[1024,8192]' ;;
      *) OSS_BUCKETS='[8192]' ;;  # Legacy T and the forced-padding diagnostic
    esac
    case $ARM in oss-P|oss-P8k) export COVEX_GPT_OSS_SKIP_PADDED_ROUTES=1 ;; esac
    ARGS=("$M/gpt-oss-20b" --served-model-name m --tensor-parallel-size 8 --max-model-len 16384
          --max-num-batched-tokens 8192 --max-num-seqs 4 --no-enable-prefix-caching --hf-overrides '{"quantization_config": {}}'
          ); CFG="{\"neuron_config\":{\"num_batched_tokens_buckets\":$OSS_BUCKETS,\"num_seqs_buckets\":[4],\"on_device_sampling_config\":{\"all_greedy\":true}}}" ;;
  llama-U|llama-F|llama-K2|llama-E3)
    CORES=0-7
    if [ "$ARM" = llama-F ]; then export COVEX_DECODE_DYNAMIC_FA=1; fi
    ARGS=("$M/Llama-3.1-8B-Instruct" --served-model-name m --dtype bfloat16 --tensor-parallel-size 8
          --max-model-len 16384 --max-num-batched-tokens 8192 --max-num-seqs 8 --no-enable-prefix-caching)
    CFG='{"neuron_config":{"num_batched_tokens_buckets":[1024,8192],"num_seqs_buckets":[1,2,4,8],"on_device_sampling_config":{"all_greedy":true}}}'
    case "$ARM" in
      llama-K2) export COVEX_LLAMA_FUSED_PAIRS=1
                export PYTHONPATH="${LLAMA_NKI_SRC:?set LLAMA_NKI_SRC to the patched transformer checkout src directory}${PYTHONPATH:+:$PYTHONPATH}"
                unset NKILIB_FORCE_BUNDLED_LIBRARY ;;
      llama-E3) draft="${LLAMA_E3_DIR:-$M/Llama-3.1-8B-Instruct-speculator.eagle3}"
                ARGS+=(--speculative-config "$(python3 -c 'import json,sys;print(json.dumps(dict(method="eagle3",model=sys.argv[1],num_speculative_tokens=3)))' "$draft")") ;;
    esac ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
case "$ARM" in vl-U|oss-U|oss-U8k|llama-U|llama-E3) VLLM=("${UPSTREAM_PY:?upstream Python required}" -m vllm.entrypoints.cli.main) ;; esac
if [ -n "${SUITE_PYTHON:-}" ]; then VLLM=("$SUITE_PYTHON" -m vllm.entrypoints.cli.main); fi
CORES=${SUITE_CORES:-$CORES}
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
