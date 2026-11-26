#!/usr/bin/env bash
# Per-arm settings. Source this, then call: arm_config <ARM>  -> prints --additional-config JSON and sets ARM_TP.
# Base = Dax's bs8 config (qwen3-asr-inference-cuda-trainium infra/trainium/neuron_config_bs8.json), 16 sequences.
#   S    stock code, Dax's buckets [512, 1024], encoder DP8, TP8                       (reference)
#   S2   stock code, prompt buckets [128, 256, 512, 1024]                            (tuned stock; the baseline we claim against)
#   D    S2 + Dax encoder MLP NKI kernel                                             (VLLM_NEURON_ASR_AUDIO_MLP_NKI=1)
#   DB   D  + Susmit #5 linear prompt buckets, step 128 (128..1024)
#   DM   D  + Susmit #3 two prefills per step (MRv1 runner; report ITL)
#   DL   D  + encoder norm2 LayerNorm folded into the audio MLP NKI kernel      (COVEX_ASR_ENCODER_LN_IN_MLP=1)
#   DE1  D  with encoder dp_size 1 (no 8-way block padding + all_gather per clip)    (setting probe)
#   T4   S2 at TP4 on one chip                                                       (setting probe; compare per chip)
arm_config() {
  local arm=$1 buckets='"num_batched_tokens_buckets": [128, 256, 512, 1024],' extra='' enc_dp=8
  ARM_TP=8; unset VLLM_NEURON_ASR_AUDIO_MLP_NKI COVEX_ASR_ENCODER_LN_IN_MLP
  case "$arm" in
    S)   buckets='"num_batched_tokens_buckets": [512, 1024],' ;;
    S2)  ;;
    D)   export VLLM_NEURON_ASR_AUDIO_MLP_NKI=1 ;;
    DB)  export VLLM_NEURON_ASR_AUDIO_MLP_NKI=1; buckets='"num_batched_tokens_bucket_step": 128,' ;;
    DM)  export VLLM_NEURON_ASR_AUDIO_MLP_NKI=1; extra='"max_prefills_per_batch": 2,' ;;
    DL)  export VLLM_NEURON_ASR_AUDIO_MLP_NKI=1 COVEX_ASR_ENCODER_LN_IN_MLP=1 ;;
    DE1) export VLLM_NEURON_ASR_AUDIO_MLP_NKI=1; enc_dp=1 ;;
    T4)  ARM_TP=4; enc_dp=4 ;;
    *) echo "unknown arm $arm" >&2; return 2 ;;
  esac
  cat <<JSON
{
  "neuron_config": {
    "quantization": "bf16",
    $buckets
    "num_seqs_buckets": [1, 2, 4, 8, 16],
    $extra
    "on_device_sampling_config": {"all_greedy": true}
  },
  "encoder_neuron_config": {
    "num_encoder_input_buckets": [4],
    "encoder_attention_block_size": 128,
    "encoder_cache_num_blocks": 64,
    "dp_size": $enc_dp
  }
}
JSON
}
