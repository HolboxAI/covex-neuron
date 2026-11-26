# Qwen3-ASR-1.7B: stock vllm-neuron vs COVEX (Dax + Susmit), stage by stage

Same model, same client and same LibriSpeech slice as Dax's Trainium-vs-CUDA study (`HolboxAI/qwen3-asr-inference-cuda-trainium`, commit `4ce1ddb`). The CUDA numbers below are directly comparable.

## Branch `exp/qwen3-asr`

Upstream `main`, plus these commits. Every change sits behind its own flag and is off by default.

| Commit | Author | Change | Turned on by |
|---|---|---|---|
| `b14137c` | Dax | audio-encoder MLP on the nkilib NKI kernel | `VLLM_NEURON_ASR_AUDIO_MLP_NKI=1` |
| `797bd6e` | Susmit | KV overcommit with recompute preemption | `"kv_cache_overcommit": true` |
| `2061be6` | Susmit | linear bucket steps | `"num_batched_tokens_bucket_step": 128` |
| `650f98f`, `d15a4f4`, `c9c1396`, `b7ed3f5` | Susmit | two prefills per step (MRv1 runner) | `"max_prefills_per_batch": 2` |

## Arms

All arms run on one trn2.48xlarge, each on 2 chips, concurrently. Settings are in `benchmarks/asr/arm_config.sh`.

| Arm | Contents |
|---|---|
| **S** | stock code paths. Dax's bs8 config, extended to 16 sequences. |
| **D** | S + Dax's encoder MLP kernel |
| **DS** | D + Susmit's overcommit and 128-token prompt buckets |
| **DSM** | DS + two prefills per step (experimental, MRv1) |

**Common settings:**
- TP8, `max_model_len` 1024, `max_num_seqs` 16, decode buckets 1/2/4/8/16
- encoder DP8, greedy sampling, no prefix caching

## Measurements

- **Throughput and latency:** Dax's `benchmark.py`, 200 samples, concurrency 1, 4, 8 and 16, 3 rounds. Reports E2E, RTF, TTFT, ITL, audio-s/s and WER.
- **Accuracy:** full LibriSpeech test-clean (2,620 utterances) WER on S and D. This settles the 2.45% vs 2.53% question.
- **Stages:** each arm restarted with `PROFILE=1` (vllm-neuron built-in profiler, system + device profiles on all 8 ranks). One window at concurrency 8.
  - `neuron-explorer view --output-format perfetto` produces the traces.
  - `summarize_profile.py` writes `profiles/<arm>/STAGES.md`: busy time per stage (collective, attention, MLP, matmul, norm, DMA, sampling, host) and the top 25 slices.
  - The raw `.pftrace` files open in https://ui.perfetto.dev.
- **Everything at once:** `benchmarks/asr/run_all.sh` runs the whole round.

## Known numbers (same client and dataset)

| System | Throughput at c8 (audio-s/s) | Source |
|---|---|---|
| Trn2 stock vllm-neuron 0.29, TP8 | 136 | Dax, `results/trainium.json` |
| Trn2 + Dax encoder MLP kernel | 160.7 (+18%) | Dax, susmit-holbox/covex-neuron#2 |
| 1 × NVIDIA L4, vLLM | 107 | Dax, `results/cuda.json` |
| 8 × NVIDIA L4 TP8, vLLM | 202 | Dax, `results/cuda_tp8.json` |

**Expected:**
- **D vs S:** about +18%, if it reproduces on the current base.
- **DS and DSM:** no estimate. Susmit's changes have not run on this model.
- **To beat 8 × L4 (202):** we need about +26% over D.

## Other platforms

- **NVIDIA:** rerun `infra/cuda/serve.sh` from Dax's repo (vLLM on CUDA) with the same client.
  - Already measured on L4 (above).
  - g6e (L40S) is the cheapest next point. A stopped `nki-bench-g6e` already exists in ap-south-2.
- **SGLang:** Qwen3-ASR support in SGLang (CUDA) and in sglang-neuron is not confirmed. sglang-neuron lists no ASR model. Check before planning a run.
- **TPU:** vLLM TPU (tpu-inference) support for Qwen3-ASR is not confirmed. The team has v6e access via Debjyoti's `tpu_rl_benchmark` work. Check model support first.

## Time and cost

- **Time on the instance:** about 2 h (image, model download, compile, the benchmark matrix, the WER run and profiles).
- **Cost:** about $17 at $8.60/h Spot.
- **Hard stop:** 3 h, so at most about $26.

**Launch** with `benchmarks/scripts/launch_round1.sh` from `exp/2x-round1` (`STOP_MIN=180`). Then on the instance:

```bash
git clone -b exp/qwen3-asr https://github.com/HolboxAI/covex-neuron && cd covex-neuron
pip install -e .
benchmarks/asr/run_all.sh
```
