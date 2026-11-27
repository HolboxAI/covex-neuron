# A/B: COVEX_DECODE_DYNAMIC_FA on Llama 3.1 8B

**Change:**
- Llama `forward_decode` passes nkilib's `max_context_len` plus in-kernel `pos_ids`, so decode attention stops after the batch's longest real context.
- Stock walks `max_blocks_per_seq × block_size` every step. Qwen3 got the same change in `62171a0`.
- **Active only when:**
  - plain BF16 KV
  - no DCP, no attention DP
  - `S_ctx > 8K` and a multiple of 8K. The nkilib 2.33 FA tile is 8K; a smaller tile comes from HolboxAI/nki-library `feat/fa-tile-2k`.
- The GQA per-head path from #77 (TP4) is not covered. Use TP8.
- This branch also contains the #77 port.

**A/B:** Llama-3.1-8B-Instruct, TP8 (2 chips), BF16, trn2.48xlarge, SDK 2.33.1. Both arms run at the same time on separate chips.
- **A:** this branch, flag off
- **B:** this branch, `COVEX_DECODE_DYNAMIC_FA=1`
- **Server:** `--max-model-len 32768`, `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.8`
- **Workloads:** 1024 / 256 at concurrency 1, 8, 16 and 32 (up to the stock admission limit); 8192 / 256 at concurrency 8
- **Metrics:** TPOT p50, output tok/s
- **Accuracy:** GSM8K, both arms

**Expected (estimate):** KV read per step drops from the full 32K span to one 8K tile per request.
- Largest gain at high concurrency with short real contexts
- About 0 at concurrency 1
- 0 if `max_model_len` is 8K or less

**Pass:** the CONTRIBUTING-COVEX.md thresholds.
