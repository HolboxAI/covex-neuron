# Qwen3-32B: per-step decode time, stock vs two kernel changes

One model, one instance, one question: **do these changes cut decode step time at identical settings?**

## Changes under test

| Arm | Flag | What it changes | Code |
|---|---|---|---|
| S | none | stock vllm-neuron decode | upstream `e16d424` |
| F | `COVEX_DECODE_DYNAMIC_FA=1` | decode attention stops after the longest real context in the batch (nkilib `max_context_len` early exit) instead of walking all `max_model_len` of KV | `62171a0` (ours) |
| M | `VLLM_NEURON_QWEN3_INTREE_MLP_BF16=1` | Dax's in-tree BF16 decode MLP NKI kernel | `bbadd1f` (Dax) |
| FM | both | | |

## Fixed setup (identical in all arms)

| Item | Value |
|---|---|
| Instance | 1 × trn2.48xlarge Spot. Each arm on its own 2 chips; all four run at once. |
| Image | `pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.1-ubuntu26.04`, with this branch installed |
| Model | `Qwen/Qwen3-32B` BF16, TP8 |
| Server | `--max-model-len 32768` (Qwen3-32B's native context), `--max-num-seqs 12`, `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.8`. Per rank: min(21.6 − 8.2, 0.8 × 21.6) = 13.4 GiB ÷ 32 KiB ÷ 32,768 ≈ 13 sequences, so 12 always fit |
| Workloads | **D1:** 1024 in / 512 out, concurrency 1, 8, 12. **D2:** 16,384 in / 512 out, concurrency 8. `--ignore-eos`, 3 repeats, median. |
| Primary metric | **TPOT p50**, i.e. decode step time |
| Secondary metrics | output tok/s, TTFT p50 |
| Accuracy | GSM8K, 1,319 questions greedy, arms S and FM. Pass: within 1 point. |
| Profile | built-in Neuron profiler (`/start_profile`, `/stop_profile`) at D1 concurrency 12 for S and F, opened in Neuron Explorer: attention vs MLP vs collective time per layer |

## Expected (byte model, written before the run)

**Per rank (TP8):** about 8.2 GB of weights per decode step. KV is 32 KiB per token.

- **Stock (S):** reads `batch × 32,768` tokens of KV, the full span.
- **F at D1:** reads `batch × 8,192`, one 8K tile, because every request stays under 8K.

| Concurrency | S bytes per step | F bytes per step | Expected TPOT gain if memory-bound |
|---|---|---|---|
| 1 | 9.3 GB | 8.5 GB | ~1.1x |
| 8 | 16.8 GB | 10.3 GB | ~1.6x |
| 12 | 21.1 GB | 11.4 GB | ~1.85x by bytes. ~1.45x by time once the fixed per-step overhead is added; see the stage budget. |
| D2, c8 (contexts up to ~17K, 3 of 4 tiles) | 16.8 GB | 14.6 GB | ~1.15x |

**Arm M:** unknown. This is the first hardware run of Dax's kernel.

**Profile check:** attention's share of each step in S at concurrency 12 should be about half or more. If it isn't, F's gain will be smaller than the table.

## Time and cost

| Step | Time |
|---|---|
| Boot and image pull, model download (65 GB) | ~25 min |
| Compile, 4 arms in parallel | ~45–60 min |
| Benchmarks, in parallel | ~30 min |
| GSM8K | ~15 min |
| Profiles | ~10 min |
| Total | ~2–2.5 h, about $20 at $8.60/h |

**Hard stop:** the instance terminates itself after 3 h (`STOP_MIN=180`), so the most this can cost is about $26.

## Decision

An arm ships if it cuts TPOT p50 by at least 5%, or raises output tok/s by at least 3%, at one or more points, with no point more than 2% worse and GSM8K passing.
