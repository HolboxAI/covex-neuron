# Quick A/B: one chip, one metric, three engines

Run this before round 1. One number per engine, at one setting.

| Item | Value |
|---|---|
| Instance | 1 × trn2.48xlarge Spot. Each engine gets its own chip (TP4, LNC2). The three run at the same time. |
| Model | Qwen/Qwen3-8B, BF16 |
| Workload | random, 1024 in / 256 out, `--ignore-eos`, at concurrency 1 (16 prompts), 32 (128 prompts) and 64 (256 prompts). c32 matches Pengcheng's vllm-neuron vs sglang-neuron baseline. |
| Metric | output tok/s. Median of 3 repeats. |
| Engines | **A** stock vllm-neuron 0.29.0.1.2.0 (AWS DLC sdk2.33.1), `--max-model-len 16384`. **A2** the same stock engine right-sized, `--max-model-len 2048`. This is the control a fair reviewer would run. **B** covex-neuron `exp/2x-round1` with `kv_cache_overcommit`, same args as A. **C** sglang-neuron v0.1.0 at its released defaults, TP4. |
| Server limits | `--max-num-seqs 64` for A, A2 and B. `--max-model-len` as listed per engine. |

**Time on the instance:**

| Step | Time |
|---|---|
| Boot, image pull, model download | ~15 min |
| Cold compile, three engines in parallel | ~15–30 min (the unavoidable part on Neuron) |
| Benchmark, 3 × ~1–2 min per engine, run in parallel | ~5 min |
| Total | **~40–60 min ≈ $6–9** at $8.60/h |

**Hard stop:** the instance terminates itself after 2 h, so the most this can cost is about $17.

**Output:** one table with the three output tok/s values and the deltas B vs A and C vs A, plus the exact server commands.

## Expected numbers (estimates, written before the run)

**Stock admission math (A):**
- KV per token for Qwen3-8B: 36 layers × 8 KV heads × 128 × 2 × 2 B = 144 KiB, or 36 KiB per TP4 rank.
- Per-rank KV budget: 24 GiB HBM × 0.9 × 0.30 cap ≈ 6.5 GiB, so about 188k tokens.
- 188k ÷ 16,384 means stock admits only about **11 concurrent requests**. The server log's "maximum concurrency" line confirms the real value.

| Engine | Expected output tok/s at c32 | Why |
|---|---|---|
| A stock, 16k | ~500–800 | Queue-bound at about 11 running requests |
| A2 stock, 2k | ~1,300–1,450 | Issue #76: Qwen3-8B TP4 at 6.93 req/s × 190 ≈ 1,317. Pengcheng: sglang 1,421. |
| B overcommit, 16k | ≈ A2, or lower | All 32 requests admitted. Any gap below A2 is the cost of decode attention over 16k padded KV. This is the direct test for context-aware decode attention. |
| C sglang-neuron | ~1,420 | Pengcheng's measurement (Qwen3-8B, 1024/256, c32) |
| all at c1 | about equal | Same decode path; no admission limit at one request |

**Reading the result:**
- B vs A (expected ~2x) is the headline "same settings" number.
- **B vs A2 shows whether we beat stock on engine merit.** We will quote both.
