## What this changes

<!-- One optimization per PR. Which models does it affect? Which flag controls it? -->

- Flag:
- Default on for models:
- Upstream: <!-- candidate / sent: vllm-project/vllm-neuron#N / COVEX-only (why) -->

## Benchmark

<!-- Required unless the PR has the `no-perf-change` label. Follow benchmarks/README.md. -->

- Instance / SDK / base commit:
- Server command:
- Workload (dataset, ISL/OSL, concurrency, prompts):
- Results file: benchmarks/results/

| Metric | Concurrency | Baseline (stock) | This PR | Change |
|---|---|---|---|---|
| | | | | |

## Accuracy

<!-- WER, GSM8K or logit match: baseline vs this PR -->

## Checklist

- [ ] Same settings for baseline and change; median of 3 runs
- [ ] At least +3% tok/s or −5% TTFT/ITL, with no metric worse by more than 2%
- [ ] Flag off restores stock behavior
- [ ] README results table and COVEX_CHANGES.md row updated
- [ ] Every commit signed off (`git commit -s`, DCO, same as upstream)
- [ ] `upstream-candidate` label added if this should go to vllm-project/vllm-neuron
