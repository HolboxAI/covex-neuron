# Benchmark protocol

Use this for every COVEX vs stock vllm-neuron comparison.

0. **The base commit is in the table.** A/B is flag off vs flag on at one commit of `main`, and `git rev-parse HEAD` goes into the results file and the README row. Optionally also compare against the latest vllm-neuron release tag, since that is what users install.
1. **One instance, one SDK.**
   - Run baseline and change back to back on the same instance.
   - Record the instance type, region, Neuron SDK version, `neuronx-cc` version and the vllm-neuron base commit (`git rev-parse upstream-main`).
2. **The baseline is the same code with the flag off.**
   - Same branch, `COVEX_*` / `VLLM_NEURON_*` flag set to `0`.
   - Or the `upstream-main` commit the PR is based on.
3. **Identical server arguments.**
   - Copy the full `vllm serve ...` command into the results file.
   - Use the same compiled-artifact cache policy. Warm up both sides before measuring.
4. **The workload:**
   - **LLMs:** `vllm bench serve` at 1k/1k and 8k/1k, random or ShareGPT, at concurrency 1, 8, 32 and 64 where KV allows. Use at least 10 × concurrency prompts.
   - **Also run 1k in / 190 out at concurrency 8, 16 and 32.** This is the shape used in vllm-neuron issue #76 and PR #77, so results compare directly with upstream numbers.
   - **Accuracy for default-on changes:** use the full standard set, not a sample (e.g. LibriSpeech test-clean, all 2,620 utterances; full GSM8K).
   - **ASR / multimodal:** the model's standard set (e.g. LibriSpeech) at concurrency 1, 4 and 8.
5. **Metrics to report:**
   - output tok/s (or audio-s/s)
   - total tok/s
   - TTFT p50/p90
   - ITL/TPOT p50/p90
   - E2E p50
   - the accuracy check
6. **Repeats:** 3 runs per side. Report the median.

Results files go in `benchmarks/results/`. Each has a header with the settings above, then a table per concurrency with baseline, change and delta.
