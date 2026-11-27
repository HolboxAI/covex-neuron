# A/B: upstream vllm-neuron #77 (Llama 3 GQA decode at TP < num_kv_heads)

**Change:** commit `aa5cb4e`, cherry-picked from vllm-project/vllm-neuron#77. Author: niklas-palm.
- At TP4, each rank holds 2 KV heads. Stock passes a 2D block table, so `attention_decode` falls back to torch attention (vllm-neuron issue #76).
- The fix builds the per-head 3D table, the same way Qwen3 already does.
- The change is not flag-gated: it only affects the case that falls back today (nkh > 1, attention DP 1, no DCP, KV not packed FP8).

**A/B:** Llama-3.1-8B-Instruct, TP4 (one chip), BF16, trn2.48xlarge, SDK 2.33.1. Each arm runs on its own chip at the same time.
- **A:** `main` (stock)
- **B:** this branch
- **Workload:** 1024 in / 190 out, at 8, 16 and 32 in flight. These are the issue #76 settings.
- **Accuracy:** GSM8K, 1,319 questions greedy, both arms.

**Upstream's numbers to reproduce** (req/s at 8 / 16 / 32 in flight):

| | req/s |
|---|---|
| before | 1.60 / 2.13 / 3.20 |
| after | 3.07 / 5.07 / 6.93 |

GSM8K: 84.0% with the fix vs 83.6% at TP8.

**Pass:** at least +3% req/s at one or more points, no point more than 2% worse, GSM8K within 1 point.
