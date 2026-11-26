# Weekend run: COVEX vs stock vllm-neuron on officially supported models

One trn2.48xlarge Spot (16 chips). Every model runs stock and COVEX side by side, each on its own chips, all at the same time.

## Matrix

| Model | Parallelism (chips per arm) | Stock arm | COVEX arm |
|---|---|---|---|
| meta-llama/Llama-3.1-8B-Instruct | TP8 (2) | vllm-neuron 0.29.0.1.2.0, AWS DLC sdk2.33.1 | `exp/2x-round1`: `kv_cache_overcommit`, linear buckets (128 / 8) |
| Qwen/Qwen3-32B | TP8 (2) | same | same flags |
| openai/gpt-oss-20b (mxfp4) | TP8 (2) | the same model settings as AWS's `run_tp8.sh`, except the common serving settings below | same flags |
| Qwen/Qwen3-ASR-1.7B | decoder TP8 + encoder DP8 (2) | same | the flags above plus `VLLM_NEURON_ASR_AUDIO_MLP_NKI=1` (Dax) |

- **Llama 3.1 8B runs at TP8, not TP4.** Stock TP4 falls back to torch attention (vllm-neuron issue #76), and we are not posting a bug-fix win.
- **Same serving settings in both arms of each model:** `--max-model-len 8192 --max-num-seqs 64`.
  - AWS's example scripts use `max_num_seqs=4`. Those are smoke tests, not serving configs, so both arms use the common settings.
- **Workload (LLMs):** random 1024 in / 1024 out, `--ignore-eos`, at concurrency 1, 16 and 64. 3 repeats at 16 and 64, median reported. One run at concurrency 1.
- **Workload (ASR):** LibriSpeech test-clean at concurrency 1, 4 and 8, plus the full test-clean WER.
- **Accuracy:** GSM8K, 1,319 questions greedy, on Llama, Qwen3-32B and gpt-oss, both arms.
- **Profile:** vllm-neuron's built-in profiler, `--profiler-config '{"profiler":"cuda"}'` with `neuron_profiler` `{"activities":["system_profile","device_profile"]}`.
  - Capture one window at concurrency 16 per arm with `/start_profile` and `/stop_profile`.
  - View it in Neuron Explorer (`neuron-explorer view --output-format perfetto`).

## Time and cost

| Step | Time |
|---|---|
| Boot, image pull, 4 model downloads | ~25 min |
| Compile, 8 servers in parallel (largest: Qwen3-32B) | ~45–60 min |
| Benchmarks and profiles, in parallel | ~60–75 min |
| GSM8K and WER | ~20 min |
| Total | ~2.5–3 h, about $25 at $8.60/h |

**Hard stop:** the instance terminates itself after 4 h (`STOP_MIN=240`), so the most this can cost is about $35.

## What gets posted

**Only rows where COVEX beats stock** at these settings, by the thresholds in CONTRIBUTING-COVEX.md, with accuracy holding.

**Each posted row includes:**
- the exact commands and image
- both arms' numbers
- a link to the raw JSON

Rows without a gain are reported internally, not posted.
