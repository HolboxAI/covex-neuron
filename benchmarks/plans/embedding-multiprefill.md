# A/B: two prefills per step on Qwen3-Embedding-8B (Susmit #3)

**Why embeddings:**
- An embedding request is prefill only.
- Stock admits one prefill per scheduler step (`max_prefills_per_batch = 1`, scheduler L282), so the server embeds one sequence per step whatever the load.
- Susmit's change (commits `d9fae73`, `62cdbf6`, `9f4d5a0`, `b652904`) allows N prefills per step. It routes to the MRv1 runner, because MRv2 accepts one prefill row only.

**A/B:** Qwen3-Embedding-8B, TP4 (one chip), BF16, trn2.48xlarge, SDK 2.33.1. Each arm runs on its own chip.
- **A:** this branch, `max_prefills_per_batch` 1 (stock behaviour, MRv2)
- **A1:** this branch, `max_prefills_per_batch` 1, forced MRv1. Separates the runner switch from the gain.
- **B:** this branch, `max_prefills_per_batch` 4 (MRv1)
- **Workload:** `/v1/embeddings` with 64-token and 256-token inputs, at concurrency 1, 8 and 32, with 10 × concurrency requests
- **Metrics:** sequences/s, p50/p90 latency
- **Accuracy:** cosine similarity of B vs A embeddings on 500 inputs. Must be ≥ 0.9999.

**Expected (estimate):** largest gain at 64 tokens, where per-step fixed cost dominates. Smaller at 256.

**Pass:** the CONTRIBUTING-COVEX.md thresholds vs A and vs A1.
