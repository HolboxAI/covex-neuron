# Qwen3-ASR-1.7B, run 4 (2026-11-27): DIRECTIONAL

- **Instance:** trn2.48xlarge Spot, us-east-2, AMI ami-040afe1fbd4796e10 (Deep Learning AMI Neuron Ubuntu 24.04, SDK 2.33.1, 20261112), prebuilt venv vllm 0.29.0.
- **Code:** covex-neuron `28b960d89d` (exp/qwen3-asr = upstream main@e16d424 + flagged changes), installed --no-deps. `pip_freeze.txt` is attached.
- **Version note:** the Qwen3-ASR model code is identical in vllm-neuron release 0.29.0.1.2.0 and main@e16d424.
- **Setup:** 4 arms at once, each on its own 2 chips (TP8, encoder DP8), 16 max seqs, max_model_len 1024, no prefix cache.
- **Client:** Dax's benchmark client `4ce1ddb`. 100 LibriSpeech test-clean clips × 2 rounds = 200 requests per level, c1 / c8 / 16. One repeat, so these results are directional.
- **Prompt tokens per clip:** `{"n": 100, "min": 39, "p50": 82.5, "p90": 183, "max": 318}`. Every prefill fits the 128-token bucket.

## Arms

| Arm | Contents |
|---|---|
| S2 | Tuned stock: prompt buckets [128, 256, 512, 1024]. Baseline. |
| D | S2 + Dax's audio-encoder MLP on the nkilib NKI kernel (`VLLM_NEURON_ASR_AUDIO_MLP_NKI=1`) |
| DM | D + two prefills per step. **Failed:** "_gather_mm_embeddings expects single-request prefill but got 2 requests". Multi-prefill is not supported for multimodal (audio) input. |
| DMBL | DM + 128-step buckets + LN-in-MLP. **Failed at compile:** dynamo "Unresolved GetAttrVariable comparison" from a one-time log guard in the LN-in-MLP code. Fixed in exp/six-model `ddcb44f`. |

## Throughput (client)

| Arm | Conc | audio-s/s (median, min–max, n) | per chip | vs S2 | E2E p50 ms | TTFT p50 ms | ITL p50 ms | WER % |
|---|---|---|---|---|---|---|---|---|
| D | 1 | 58.3 (58.3–58.3, n=1) | 29.2 | -0.3% | 103 | 50 | 2.76 | 2.71 |
| D | 8 | 138.4 (138.4–138.4, n=1) | 69.2 | -0.9% | 362 | 132 | 3.69 | 2.71 |
| D | 16 | 148.4 (148.4–148.4, n=1) | 74.2 | +0.3% | 648 | 261 | 3.85 | 2.71 |
| DM | 1 | 59.4 (59.4–59.4, n=1) | 29.7 | +1.6% | 100 | – | – | 2.71 |
| DM | 8 | 0.0 (0.0–0.0, n=0) | – | – | – | – | – | – |
| DM | 16 | 0.0 (0.0–0.0, n=0) | – | – | – | – | – | – |
| DMBL | 1 | 0.0 (0.0–0.0, n=0) | – | – | – | – | – | – |
| DMBL | 8 | 0.0 (0.0–0.0, n=0) | – | – | – | – | – | – |
| DMBL | 16 | 0.0 (0.0–0.0, n=0) | – | – | – | – | – | – |
| S2 | 1 | 58.5 (58.5–58.5, n=1) | 29.2 | +0.0% | 102 | 52 | 2.76 | 2.70 |
| S2 | 8 | 139.7 (139.7–139.7, n=1) | 69.8 | +0.0% | 355 | 134 | 3.65 | 2.70 |
| S2 | 16 | 147.9 (147.9–147.9, n=1) | 74.0 | +0.0% | 659 | 292 | 2.36 | 2.69 |

## Per-request stages (server /metrics deltas)

Means over the whole benchmark: c1 + c8 + c16, transcribe + streaming.

| Stage | S2 | D | D vs S2 |
|---|---|---|---|
| Queue | 40.7 ms | 40.8 ms | +0.3% |
| Prefill (encoder + prompt) | 34.9 ms | 73.0 ms | +109% |
| Decode | 198.5 ms | 200.6 ms | +1.1% |
| Time to first token (server) | 101.5 ms | 139.1 ms | +37% |
| Inter-token latency | 7.93 ms | 8.02 ms | +1.2% |
| Prompt / generated tokens | 103.8 / 26.0 | 103.8 / 26.0 | |

## Findings

1. **No speedup for D over tuned stock** (−0.3% / −0.9% / +0.3% audio-s/s). Dax's earlier +18% (release 0.29, 32 clips, separate runs) did not reproduce.
2. **Server-side prefill is slower with D** (35 → 73 ms), yet client TTFT barely moves (52 → 50 ms at c1). Needs a device profile. The profiling pass did not finish before the instance was stopped.
3. **Decode dominates per-request server time** (about 200 ms vs about 35 ms prefill), so the next ASR lever is decode (ITL about 8 ms), not the encoder.
