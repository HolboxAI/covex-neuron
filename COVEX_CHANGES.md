# COVEX changes vs stock vllm-neuron

Every change COVEX carries on top of upstream, in one place, so customers know what they get and the vllm-neuron community can pick any change. Each row is one squash commit on `main`.

| Change | Flag (set `0` for stock behavior) | Default on for | Result | Upstream status | Commit / PR |
|---|---|---|---|---|---|
| Qwen3-ASR audio-encoder MLP on the nkilib NKI kernel | `VLLM_NEURON_ASR_AUDIO_MLP_NKI` | Qwen3-ASR | +18% audio-s/s at concurrency 8, TTFT −25% to −46% ([results](benchmarks/results/2026-11-24-qwen3-asr-1.7b-audio-mlp-nki.md)) | Candidate: not yet sent | In review |

**Upstream status values:**
- **Candidate:** generally useful, to be sent to vllm-project/vllm-neuron.
- **Sent:** `vllm-project/vllm-neuron#N` is open.
- **Merged upstream:** the change is dropped from COVEX at the next sync that contains it.
- **COVEX-only:** model- or customer-specific, with the reason stated.

**Picking a change into your own vllm-neuron:** each row is one commit. `git cherry-pick <sha>` applies it to stock vllm-neuron. Every commit carries a DCO `Signed-off-by`, like upstream requires.
