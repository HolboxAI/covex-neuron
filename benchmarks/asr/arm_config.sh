#!/usr/bin/env bash
# Print the --additional-config JSON for one arm, and export its env flags when sourced.
# Base = Dax's bs8 serving config (HolboxAI/qwen3-asr-inference-cuda-trainium infra/trainium/neuron_config_bs8.json),
# extended to 16 sequences so every arm can serve concurrency 16. Identical in every arm except the change under test.
#   S    stock vllm-neuron code paths (all COVEX flags off)
#   D    + Dax: audio-encoder MLP on the nkilib NKI kernel           (VLLM_NEURON_ASR_AUDIO_MLP_NKI=1)
#   DS   + Susmit #4 KV overcommit and #5 linear prompt buckets (step 128 instead of [512, 1024])
#   DSM  + Susmit #3 two prefills per step (routes to MRv1 runner; experimental)
arm_config() {
  local arm=$1 tok_buckets='"num_batched_tokens_buckets": [512, 1024],' extra=''
  unset VLLM_NEURON_ASR_AUDIO_MLP_NKI
  case "$arm" in
    S) ;;
    D) export VLLM_NEURON_ASR_AUDIO_MLP_NKI=1 ;;
    DS) export VLLM_NEURON_ASR_AUDIO_MLP_NKI=1
        tok_buckets='"num_batched_tokens_bucket_step": 128,'; extra='"kv_cache_overcommit": true,' ;;
    DSM) export VLLM_NEURON_ASR_AUDIO_MLP_NKI=1
        tok_buckets='"num_batched_tokens_bucket_step": 128,'; extra='"kv_cache_overcommit": true, "max_prefills_per_batch": 2,' ;;
    *) echo "unknown arm $arm" >&2; return 2 ;;
  esac
  cat <<JSON
{
  "neuron_config": {
    "quantization": "bf16",
    $tok_buckets
    "num_seqs_buckets": [1, 2, 4, 8, 16],
    $extra
    "on_device_sampling_config": {"all_greedy": true}
  },
  "encoder_neuron_config": {
    "num_encoder_input_buckets": [4],
    "encoder_attention_block_size": 128,
    "encoder_cache_num_blocks": 64,
    "dp_size": 8
  }
}
JSON
}
