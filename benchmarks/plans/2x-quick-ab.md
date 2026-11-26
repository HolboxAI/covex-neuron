# Quick A/B: one chip, one metric, three engines

Run this before round 1. One number per engine, at one setting.

| Item | Value |
|---|---|
| Instance | 1 × trn2.48xlarge Spot. Each engine gets its own chip (TP4, LNC2). The three run at the same time. |
| Model | Qwen/Qwen3-8B, BF16 |
| Workload | random, 1024 in / 256 out, `--ignore-eos`, concurrency 32, 128 prompts. The same setting Pengcheng used for his vllm-neuron vs sglang-neuron baseline. |
| Metric | output tok/s. Median of 3 repeats. |
| Engines | **A** stock vllm-neuron 0.29.0.1.2.0 (AWS DLC sdk2.33.1). **B** covex-neuron `exp/2x-round1` with `kv_cache_overcommit`, same server args as A. **C** sglang-neuron v0.1.0 at its released defaults, TP4. |
| Server limits (A, B) | `--max-model-len 16384 --max-num-seqs 64` |

**Time on the instance:**

| Step | Time |
|---|---|
| Boot, image pull, model download | ~15 min |
| Cold compile, three engines in parallel | ~15–30 min (the unavoidable part on Neuron) |
| Benchmark, 3 × ~1–2 min per engine, run in parallel | ~5 min |
| Total | **~40–60 min ≈ $6–9** at $8.60/h |

**Hard stop:** the instance terminates itself after 2 h, so the most this can cost is about $17.

**Output:** one table with the three output tok/s values and the deltas B vs A and C vs A, plus the exact server commands.
