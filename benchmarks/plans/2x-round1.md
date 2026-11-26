# 2x program, round 1: measure before building

**Goal:** higher output tok/s than stock vllm-neuron at identical user-visible settings.

**This round answers three questions** on one trn2.48xlarge:

1. How much of Susmit's measured gain (+58–68% at concurrency 16–32) survives against stock with its memory knobs tuned?
2. Which of his two changes (#4 KV overcommit, #5 linear buckets) carries the gain?
3. What share of a decode step is attention over padded KV? This decides whether round 2 builds context-aware decode attention (`max_context_len` / `attention_tkg_seq_packed`).

No new kernel code is built until round 1 is in.

## Fixed setup (identical in every arm)

| Item | Value |
|---|---|
| Instance | trn2.48xlarge Spot, one instance, us-east-2 (fallback us-west-2) |
| Image | `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.1-ubuntu26.04` |
| Code | this branch, `exp/2x-round1` (upstream `e16d424` + #4 + #5, both off by default). Each arm sets only its flags. |
| Model | `Qwen/Qwen3-8B`, BF16 |
| Parallelism | TP4 = one chip, LNC2 |
| Server limits | `--max-model-len 16384 --max-num-seqs 64` |
| Sampling | greedy, `--ignore-eos` on the client so every request generates exactly the output length |

**Why Qwen3-8B TP4 rather than Llama 3.1 8B:**
- Qwen3-8B TP4 is a healthy path in stock vllm-neuron, at 6.93 req/s in vllm-neuron issue #76.
- Llama 3.1 8B TP4 falls back to torch attention in stock (#76/#77), so a Llama win would be a bug fix, not an engine gain. Llama is round 2, after porting #77, with the fix applied to both arms.

**One chip per arm:** each arm runs on its own chip of the same instance (`NEURON_RT_VISIBLE_CORES`).
- All six arms compile **and benchmark at the same time**. Each server and client uses a few of the 192 host vCPUs.
- Arms start in a different order on each repeat. If any arm's c=1 TPOT moves more than 3% between repeats, re-run that point alone.

## Arms

| Arm | Code path | `--additional-config` neuron_config | Env |
|---|---|---|---|
| S0 stock-default | stock | `{}` | none |
| S1 stock-tuned | stock | `{}` | `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.8` |
| S2 stock-headroom | stock | `{}` | `VLLM_NEURON_RESERVED_HEADROOM_GIB=4` |
| C1 overcommit | #4 | `{"kv_cache_overcommit": true}` | same as the best of S1/S2 |
| C2 buckets | #5 | `{"num_batched_tokens_bucket_step": 128, "num_seqs_bucket_step": 8}` | same as the best of S1/S2 |
| C3 both | #4 + #5 | both of the above | same as the best of S1/S2 |

S0 is reported for reference: "vs stock defaults".

**The claim we make is C\* vs the best of S1/S2.** C arms use the same memory setting as the best stock arm, so the only difference is the code.

## Workloads

`vllm bench serve --dataset-name random --ignore-eos`.
- Concurrency ≥ 8: 3 repeats per point, median reported.
- Concurrency 1: run once. It goes through the same code path in every arm, so it's only a regression check.

| Workload | Input / output | Concurrency | Prompts per point |
|---|---|---|---|
| W1 (comparable with upstream #76/#77) | 1024 / 190 | 1, 8, 16, 32, 64 | max(32, 4 × concurrency) |
| W2 chat | 1024 / 1024 | 1, 16, 32, 64 | max(16, 2 × concurrency) |
| W3 long prompt | 8192 / 1024 | 1, 8, 16, 32 | max(16, 2 × concurrency) |

**Metrics:**
- output tok/s and total tok/s
- TTFT p50/p90
- TPOT and ITL p50/p90
- E2E p50
- KV GiB and "maximum concurrency" from the server log
- preemption count (C1/C3)

## Accuracy

GSM8K test (1,319 questions, greedy, chat template) on S1 and on the best C arm.

**Pass:** within 1.0 point of S1. Upstream #77 reports 84.0% for Llama TP4, so expect the low 90s for Qwen3-8B.

## Profile (question 3)

On S1 at W3, concurrency 16:
- Capture one steady-state decode step with `neuron-profile`.
- Report the share of step time and HBM bytes spent in the attention kernel, and the KV bytes read vs the KV bytes holding real tokens.

**Go/no-go:** build context-aware decode attention in round 2 only if attention over padding is at least 15% of the decode step at this setting.

## Decision rules

- **A change ships to `main`** (CONTRIBUTING-COVEX.md section 2) if it is at least 3% faster in output tok/s, or at least 5% better in TTFT/ITL p50, against the best stock arm at one or more points, with no metric more than 2% worse and accuracy passing.
- **The README row** quotes the gain against the best stock arm. "vs stock defaults" may be quoted separately and labeled.

## Budget and safety

**Time on the instance** (all arms in parallel):

| Step | Estimate |
|---|---|
| Boot, image pull (~15 GB), model download (16 GB) | 20 min |
| Compile, 6 arms in parallel | 45–60 min |
| Benchmark matrix, about 25 min per repeat, 3 repeats (c=1 once) | 70–80 min |
| GSM8K, 2 arms × 1,319 questions | 10 min |
| neuron-profile of one decode step | 20 min |
| Slack | 30 min |
| **Total** | **about 3.5–4 h** |

- **Spot price:** $8.60/h (2026-11-26).
- **Expected spend:** about $30–35.
- **Hard stop:** the instance self-terminates after 6 h (`shutdown -h +360`, terminate on shutdown). That caps spend at about $52.
- **Compile cache:** saved to S3 at the end, so round 2 skips most of the compile step.
- **Tags:**
  - `Project=covex-neuron`
  - `Experiment=2x-round1`
  - `Owner=<launcher>`
  - `AutoTerminate=6h`
- **Results** go to `benchmarks/results/2026-11-xx-qwen3-8b-round1/` on this branch: raw JSON, server logs and summary.

## Round 2 (decided by round 1)

1. **Context-aware decode attention:** pass nkilib `max_context_len` into `attention_block_tkg`, behind `COVEX_DECODE_DYNAMIC_FA`. Then try `attention_tkg_seq_packed`. Only if the profile passes.
2. **Overcommit hardening:** a cap on the preemption rate, then host-memory swap instead of recompute.
3. **Llama 3.1 8B** with upstream #77 applied in both arms.
4. **Mixed prefill + decode** (Sarathi-style): design doc first.
