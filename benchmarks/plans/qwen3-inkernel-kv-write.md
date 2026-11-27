# A/B: COVEX_QWEN3_INKERNEL_KV_WRITE (Qwen3 decode KV write inside the attention kernel)

**Stock (code, upstream main):**
- Qwen3 `forward_decode` calls `attention_decode(update_cache=False)`, which returns K/V for the new token.
- It then writes them with `self.k_cache.index_put_` / `self.v_cache.index_put_` on the whole cache tensor, every layer, every step.
- Llama 3 decode instead passes `update_cache=True` plus `kv_cache_update_idx`, and the kernel writes in place.
- vllm-neuron #40 found that this `index_put_` pattern on a cache parameter lowers to a **full-cache scatter + transpose**: 30.94 GB of HLO scatter per step on MiMo. The fix cut that to 12.3 MB and TPOT from 199 to 128.5 ms.

**Change:** when the flag is on, the nkh == 1 path (TP ≥ num_kv_heads, e.g. Qwen3-32B TP8, Qwen3-8B TP8) runs on BF16 KV. It uses Llama's in-kernel write (`kv_cache_update_idx` = slot_mapping) and skips the manual `index_put_`.
- The GQA nkh > 1 path is unchanged.
- FP8 KV (`k_scale` set) is unchanged.

**Risk:** the kernel writes into the squeezed 3D view of the 4D cache. The FX aliasing pass must carry that write back to the cache parameter. A silent no-write would corrupt decode, so correctness is the first gate.

**A/B:** Qwen3-32B, TP8, BF16, trn2.48xlarge, SDK 2.33.1.
- **A:** flag off
- **B:** flag on
- **Gate 1, correctness:** greedy outputs on 50 prompts × 256 tokens must match A token for token. GSM8K must be within 1 point.
- **Gate 2, speed:** TPOT p50 and output tok/s at concurrency 1, 8, 12, with `max_model_len` 8192 and 32768 (cap 0.8 = tuned stock).
- **Offline check:** compare HLO scatter/transpose bytes from both arms' compile caches.

**Expected:** if stock's HLO shows a full-cache scatter, a large TPOT cut, comparable to #40's 1.55x. If XLA already lowers it to a small scatter, about 0. The offline HLO check answers this before any claim.
