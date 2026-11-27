# Six-model run: stock vs COVEX on every officially supported model

**Branch:** `exp/six-model` = upstream `main@e16d424` plus every COVEX change, each behind its own flag (off = stock code path).
**Kernels:** the HolboxAI/nki-library `covex/2.33` changes are installed only for the arms that use them, each with its own compile cache.

## Baselines and the claim

| Name | What it is |
|---|---|
| **R** (recipe stock) | Upstream code with the upstream tutorial/recipe config |
| **T** (tuned stock) | Upstream code with the same config knobs COVEX uses (cap, buckets, block size, max_model_len), no code changes. **The headline baseline.** |
| **C** (COVEX) | Same config as T, plus COVEX code flags |

Every row reports **C vs T**. C vs R is shown separately and labeled.

## Per-model arms

| Model | Parallelism | Config (T and C) | COVEX flags (C) | Workload | Expected C vs T (estimate) | ≥2x plausible? |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | TP4, 1 chip | max 8192, block 16, cap 0.6 | #77 (in T too; it is a bug fix), `COVEX_DECODE_DYNAMIC_FA`, nki tile 2048 | 1k/256 and 8k/1k, c1/4/16/32 | 1.1–1.4x | No (only vs R, through #77) |
| Qwen3-32B | TP8, 2 chips | max 32768, cap 0.8 | `COVEX_DECODE_DYNAMIC_FA`, `COVEX_QWEN3_INKERNEL_KV_WRITE`, nki tile 2048 | 1k/512, c1/4/8/12 | 1.4–1.7x | Only if the HLO shows the KV write is a full-cache copy |
| Qwen3-VL-32B | TP4, 1 chip | max 8192, block 16 | vision MLP NKI port + LN fold (**to be written**), text dynFA + tile | fixed 200-image set + text, c1/4/8 | TTFT 1.3–2x, throughput 1.2–1.6x | Image TTFT only |
| Qwen3-Embedding-8B | TP8, 2 chips | max 4096, recipe buckets | `max_prefills_per_batch` 4 | 64 / 256 / 1024-token inputs, c32 | 1.7–2.8x at 64 tokens; about 1.0–1.2x at 256 and 1024 | Short inputs only |
| Qwen3-ASR-1.7B | TP8, 2 chips | S2 config (buckets 128–1024, 16 seqs) | Dax MLP + LN fold, two prefills per step, 128-step buckets | LibriSpeech, c1/8/16 | 1.3–1.6x per server | No (about 2x per chip with TP4) |
| GPT-OSS-20B | TP8, 2 chips | max 16384 | dynFA (full-attention layers only) | 1k/256, c1/8/16 | 1.0–1.15x | No. Control row. |

So ≥2x across all six is **not** expected from this code. It is plausible for Embedding at short inputs, for Llama only against recipe stock, and for Qwen3-32B only if the KV-write copy turns out large. The run measures every stage, so the next levers can be chosen from data.

## Gates

A row is reported only if all of these pass:

| Model | Accuracy gate |
|---|---|
| Llama, Qwen3-32B, VL text | GSM8K, 1,319 greedy, within 1 point of T |
| Qwen3-32B in-kernel KV write | Greedy token match vs T on 50 prompts × 256 tokens |
| Embedding | Cosine ≥ 0.9999, batched vs single, same inputs |
| ASR | Full test-clean WER within 0.1 points |
| VL vision | Vision-tower embedding cosine ≥ 0.999 vs T, and caption/VQA exact-match within 1 point |

## Stage and kernel timing

Captured for every arm, at the model's middle concurrency.

- **Request side:** `/metrics` deltas give queue / prefill / decode / e2e per request; client E2E minus server E2E gives host + HTTP.
- **Host side:** py-spy flame graph of the API server and engine core.
- **Device side:** vllm-neuron built-in profiler, system + device, all ranks. `summarize_profile.py` reports time per stage (attention, MLP, collectives, DMA, norm, sampling, host gaps), the top-25 kernels, and each kernel's DMA share and SBUF-spill flags.
- **Graph:** the compile cache (NEFF + input HLO) is synced to S3. Scatter/transpose bytes per model come from the HLO (the #40 check).

## Layout: two instances in parallel

Each instance is a trn2.48xlarge with 16 chips.

| Instance | Arms (T + C per model; R where it differs from T) | Chips |
|---|---|---|
| A | Qwen3-32B T/C (4), ASR R/T/C (6), Llama R/T/C (3) | 13 |
| B | Embedding R/T/C (6), VL T/C (2), GPT-OSS T/C (4) | 12 |

- Separate compile caches for the nki-library arms.
- No pre-seeded caches for flag arms.
- Arms are CPU-pinned (16 vCPUs each); clients run on their arm's CPU set.

## Before launch

All of these must pass, or there is no launch.

1. **Code:** the VL vision MLP port + LN fold is written and unit-checked on CPU against the stock torch path (same outputs within BF16 tolerance).
2. **Local preflight for every model:**
   - weights reachable without a token (Llama 3.1 uses an ungated mirror, or the user provides an HF token)
   - env vars read by `vllm_neuron`
   - no rejected vars
   - branch pushed and clean
   - nki-library sha pinned
3. **Inputs pre-staged in S3:** LibriSpeech (done), GSM8K, a 200-image VL set, embedding text sets. The instances never call `datasets` audio or image decode.
4. **On-instance smoke, first:** one arm per model, 4 real-client requests. Any failure: write FAILED, stop that model's arms. Other models continue.
5. **Hard stop** 3.5 h per instance. Expected ~2.5–3 h each, about **$45–50 total**; cap about $60.

## Output

One table per model: R / T / C, with tok/s (or audio-s/s, seq/s), TTFT / ITL / E2E p50, C vs T, the accuracy gate, and the stage breakdown. Raw JSON, metrics, flame graphs, Perfetto traces and HLO stats go to S3.
