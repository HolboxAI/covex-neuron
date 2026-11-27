| Model | Load | Metric | Stock (T) | COVEX (C) | C / T |
|---|---|---|---|---|---|
| Qwen3-32B | c1 | output tok/s | – | – | – |
| Qwen3-32B | c1 | TPOT p50 ms (lower is better) | – | – | – |
| Qwen3-32B | c8 | output tok/s | – | – | – |
| Qwen3-32B | c8 | TPOT p50 ms (lower is better) | – | – | – |
| GPT-OSS-20B (stock only) | c1 | output tok/s | 82.73 | – | – |
| GPT-OSS-20B (stock only) | c4 | output tok/s | 248.31 | – | – |
| Qwen3-Embedding-8B | t64 c32 | sequences/s | 47.537 | – | – |
| Qwen3-Embedding-8B | t256 c32 | sequences/s | 39.928 | – | – |
| Qwen3-VL-32B | c4 | requests/s | – | – | – |
| Qwen3-VL-32B | c4 | latency p50 ms (lower is better) | – | – | – |
| Qwen3-ASR-1.7B | c1 | audio-s/s | 54.6 | 54.5 | 1.00x |
| Qwen3-ASR-1.7B | c8 | audio-s/s | 137.3 | 123.6 | 0.90x |

**Correctness (COVEX vs stock):**
- q32: not run
- emb: 
- vl: not run
- asr-T WER on 100 clips: 2.7%
- asr-C WER on 100 clips: 2.67%

## Arms that did not produce results

| Arm | Cause |
|---|---|
| q32-T and q32-C (Qwen3-32B) | "Could not load the model status=4 message=Allocation Failure". max_model_len 32768 with cap 0.8 did not fit HBM. |
| vl-T and vl-C (Qwen3-VL-32B) | ValueError: Automatic Prefix Caching requires segmented prefill, but prefill is single-shot (max_num_batched_tokens 8192). Prefix caching should have been disabled. |
| emb-C (Qwen3-Embedding-8B, 16 prefills per step) | Worker exception at 03:28, mid-benchmark |

**Other gaps:**
- No device or kernel profiles: the 105-minute cap ended the run before the profile step.
- ASR C (Dax NKI MLP + LN fold) vs T: 1.00x at c1, 0.90x at c8. No gain.
- Labelled directional.
