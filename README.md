# COVEX Neuron

**COVEX Neuron is a fork of [vllm-project/vllm-neuron](https://github.com/vllm-project/vllm-neuron), maintained by [HolboxAI](https://github.com/HolboxAI). It focuses on higher tokens/s on AWS Trainium.**

- **Same API as vllm-neuron.** Install and serve exactly as described in the upstream README below.
- **Synced with upstream every night.** The `upstream-main` branch mirrors `vllm-project/vllm-neuron` `main`. An automatic PR merges it into `main`.
- **Only measured wins land in `main`.** Every performance change carries a before/after benchmark against stock vllm-neuron at the exact same settings. Optimizations are scoped per model and turn on by default only for the models they were validated on. See [CONTRIBUTING-COVEX.md](CONTRIBUTING-COVEX.md).
- **Every COVEX change is listed in [COVEX_CHANGES.md](COVEX_CHANGES.md)** with its flag, result and upstream status. Each is one signed-off commit that anyone can cherry-pick into stock vllm-neuron.
- **Generally useful changes are sent upstream** to vllm-neuron. COVEX exists so customers get the wins now instead of waiting for the next upstream release.
- **For production, pin a release tag** (`covex-<upstream release>-<n>`), not `main`.

## Install COVEX Neuron

```bash
git clone https://github.com/HolboxAI/covex-neuron.git
cd covex-neuron
pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com -e .
```

This replaces the upstream install step in the README below. Everything else there (instance setup, `vllm serve`, model support) applies unchanged. To get stock vllm-neuron behavior for any COVEX optimization, set its flag to `0` (see the table).

## COVEX vs stock vllm-neuron

Each row compares the same model, instance, Neuron SDK, parallelism, dataset and concurrency. Only the code under test differs. Raw results live in [`benchmarks/results/`](benchmarks/results/). The protocol is in [`benchmarks/README.md`](benchmarks/README.md).

**Baseline:** each row was measured with the optimization off vs on, on the vllm-neuron base named in its results file. Rows measured on an older base are re-run on current `main` (flag off vs on, same commit) before the next release. The Qwen3-ASR row below was measured on vllm-neuron 0.29.0.1.2.0 with 32 LibriSpeech samples. A re-run on current `main` and a full test-clean WER (2,620 utterances) are pending.

| Model | Setup | Metric | vllm-neuron 0.29.0.1.2.0 | COVEX | Change | Status |
|---|---|---|---|---|---|---|
| Qwen3-ASR-1.7B (BF16) | trn2.48xlarge, SDK 2.33.1, decoder TP8 + encoder DP8, LibriSpeech, 32 samples | Throughput, concurrency 8 | 136 audio-s/s | 160.7 audio-s/s | **+18.1%** | In review |
| | | TTFT p50, concurrency 1 / 8 | 0.051 s / – | 0.038 s / – | **−25% / −46%** | |
| | | E2E latency p50, concurrency 1 / 8 | 0.103 s / – | 0.091 s / – | **−11.6% / −26.6%** | |
| | | WER | 2.45% | 2.53% | +0.08 pt | |

**In progress (no numbers yet, so not in `main`):**
- multi-prefill batching ([upstream #124](https://github.com/vllm-project/vllm-neuron/pull/124))
- KV-cache overcommit with recompute preemption
- linear bucket steps to cut padding
- Qwen3-32B BF16 decode MLP NKI kernel
- Qwen3.5 hybrid support plus weight transfer and sleep/wake for RL rollout ([upstream #98](https://github.com/vllm-project/vllm-neuron/pull/98))

## Branches

| Branch | What it is |
|---|---|
| `main` | Upstream plus validated COVEX optimizations. Protected: PR, review and benchmark gate. |
| `upstream-main` | Exact mirror of `vllm-project/vllm-neuron` `main`, updated nightly. Never commit here. |
| `feat/*`, `port/*` | Work in progress, behind env flags. |
| `legacy/runner-sdk` | The earlier COVEX two-line PyTorch/HF to Trainium runner (PyPI `covex-neuron` 0.0.1.dev5). |

---

# Upstream vllm-neuron README

# vLLM Neuron Plugin (Beta)

The vLLM Neuron plugin is the recommended serving solution for large language models on AWS Trainium. It extends [vLLM](https://docs.vllm.ai/) with a Neuron backend, providing the same vLLM APIs and configuration you are already familiar with.

> **If you are using vLLM with NxD Inference (0.5.x plugin version),** see the [release-0.5.3 branch](https://github.com/aws-neuron/private-vllm-neuron/tree/release-0.5.3) and the [NxD Inference + vLLM documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/index.html). To migrate to the plugin based on 0.21.0+, see the [migration guide](docs/getting-started/migration-nxdi-to-vllm-neuron.md).

## Installation

```bash
git clone -b release-0.29.0.1.2.0 https://github.com/vllm-project/vllm-neuron.git
cd vllm-neuron
pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com -e .
```

This installs the vLLM Neuron plugin from source along with vLLM and the required Neuron SDK packages. See [Version](#version) for the full compatibility matrix.

**For detailed environment setup and instance selection**: See the [Setup guide](docs/getting-started/setup-guide.md).

## Quick Start

Run the following command from a shell terminal in the local or instance environment (`vllm` must be in the path) where vLLM is configured:

```bash
vllm serve openai/gpt-oss-20b \
    --tensor-parallel-size 8 \
    --max-model-len 8192 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 4
```

## Supported Models

Models listed below are tested end-to-end on Neuron hardware with correctness validation and performance benchmarking.

| Family | Model | Type | Instance | Correctness | Perf Test | Perf Tuning |
|---|---|---|---|---|---|---|
| Llama 3 | 1B, 8B, 70B | Text | Trn2, Trn3 | ✅ | ✅ | In progress |
| GPT-OSS | 20B, 120B | Text | Trn2, Trn3 | ✅ | ✅ | In progress |
| Qwen3 | 32B (Dense) | Text | Trn2, Trn3 | ✅ | ✅ | In progress |
| Qwen3-VL | 32B (Dense) | Multimodal | Trn2, Trn3 | ✅ | ✅ | In progress |
| Qwen3-Embedding | 8B | Embedding | Trn2, Trn3 | ✅ | ✅ | In progress |
| Qwen3-ASR | 1.7B | Audio | Trn2, Trn3 | ✅ | ✅ | In progress |

- **Correctness** — Accuracy validation passing (logit comparison, dataset accuracy validation)
- **Perf Test** — Performance benchmark tests tracked across releases
- **Perf Tuning** — Active optimization work being done

> **Note:** This list includes only the models currently supported in the latest plugin version and does not include models under development or on roadmap.

For dense Qwen3-32B checkpoints, features, and deployment commands, see the
[model recipe](docs/model-recipes/qwen3-32b.md).

Performance improvements are delivered incrementally with each Neuron release.

### Community Contributions

The [`contrib/`](contrib/) folder hosts community-contributed models. These are maintained by the community and may have limited testing compared to the officially-supported models above. To use one, or to contribute your own model, see [contrib/README.md](contrib/README.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Features

Feature support is at the framework level. See the model cards in [`docs/model-recipes/`](docs/model-recipes/) for per-model feature availability and the [features guide](docs/guides/features-guide.md) for configuration details.

| Category | Feature | Status |
|---|---|---|
| **Serving** | Continuous batching | ✅ |
| | OpenAI-compatible APIs | ✅ |
| | Embedding API (`/v1/embeddings`) | ✅ |
| | Streaming | ✅ |
| | Structured outputs (JSON, regex, grammar) | ✅ |
| | Tool calling | ✅ |
| | LoRA adapters | ❌ |
| | Weight reload API | ❌ |
| | Sleep mode (sleep/wake_up API) | ❌ |
| **Parallelism** | Tensor parallelism (TP) | ✅ |
| | Sequence parallelism (SP) | ✅ |
| | Data parallelism (DP) | ✅ |
| | Expert parallelism (EP) | ✅ |
| | Encoder parallelism | ✅ |
| | Pipeline parallelism (PP) | ❌ |
| | Context parallelism (CP) | ✅ |
| **Performance** | Paged KV cache | ✅ |
| | Prefix caching | ✅ |
| | Segmented prefill | ✅ |
| | Chunked prefill (mixed batching) | ❌ |
| | Disaggregated inference (1P1D, xPyD) | ✅ |
| | Async scheduling | ✅ |
| | On-device sampling (top-k, top-p, temperature) | ✅ |
| | KV cache offloading | ❌ |
| | Disaggregated encoder (EPD) | ✅ |
| **Compilation** | torch.compile (native PyTorch) | ✅ |
| | torch.compile (XLA backend) | ✅ |
| | Compile cache (local/remote) | ✅ |
| | CPU compilation | ✅ |
| **Runtime** | Model Runner V2 (dense models) | ✅ |
| **Speculative Decoding** | EAGLE3 | ✅ |
| | MTP | ❌ |
| **Quantization** | BF16 | ✅ |
| | FP8 | ✅ |
| | Quantized KV cache (FP8) | ✅ |
| | MXFP4 weights (Trn3) | ✅ |
| | MXFP8 weights (Trn3) | ✅ |
| **Multimodal Inputs** | Images + text | ✅ |
| | Videos + text | ✅ |
| | Audio | ✅ |
| **Observability** | Production metrics (`/metrics` endpoint) | ✅ |
| | Profiler API | ✅ |

**Status legend:**

- ✅ Supported — integrated and tested for at least one model
- ❌ Not supported — may be considered for future releases

## Documentation

Full documentation sources are found in [`docs/`](docs/). These docs are published to [awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/index.html).

**Deploying supported models:**

| Folder | What's inside |
|---|---|
| [`docs/getting-started/`](docs/getting-started/) | Setup, quickstarts, migration from NxD Inference |
| [`docs/guides/`](docs/guides/) | Features guide, configuration reference, metrics, profiling |
| [`docs/model-recipes/`](docs/model-recipes/) | Per-model deployment recipes |
| [`docs/tutorials/`](docs/tutorials/) | End-to-end walkthroughs (DI, prefix caching, etc.) |

**Implementing new models & optimizing performance:**

| Folder | What's inside |
|---|---|
| [`docs/model-dev/`](docs/model-dev/) | Model onboarding, CPU development, etc. |
| [`docs/design/`](docs/design/) | Architecture deep dives (parallelism, DI, prefix caching, etc.) |

## Version

Starting with 0.21.0, the vLLM Neuron plugin version follows the format `<vLLM version>.<plugin version>` (e.g., `0.21.0.1.0.0` means vLLM 0.21.0, plugin version 1.0.0). Model implementations live directly in the plugin and NxD Inference is no longer a dependency. Customers on 0.5.x can migrate using the [migration guide](docs/getting-started/migration-nxdi-to-vllm-neuron.md).

| vLLM-Neuron Plugin | vLLM Version | Neuron SDK | NxD Inference | Instance Support | Status | Documentation |
|---|---|---|---|---|---|---|
| 0.29.0.1.2.0 (latest) | 0.29.0 | 2.33 | Not required | Trn2, Trn3 | Beta | [vLLM Neuron docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/index.html) |
| 0.24.0.1.1.0 | 0.24.0 | 2.32 | Not required | Trn2, Trn3 | Beta | [vLLM Neuron docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/index.html) |
| 0.21.0.1.0.0 | 0.21.0 | 2.31 | Not required | Trn2, Trn3 | Beta | [vLLM Neuron docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/index.html) |
| 0.5.3 | 0.16.0 | 2.31 | 0.10.x | Trn1, Trn2, Trn3, Inf2 | Maintenance | [NxDI + vLLM on Neuron docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/index.html) |

## Issues

Report bugs or request features: [GitHub Issues](https://github.com/vllm-project/vllm-neuron/issues)

## Code of Conduct

This project follows the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct). For more information, see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact <opensource-codeofconduct@amazon.com>.

## License

Apache-2.0
