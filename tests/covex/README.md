# COVEX static checks

`pytest tests/covex -q` runs in seconds with plain Python. Each test encodes a failure that cost a Trainium run on 2026-11-27.

| Test | Failure it prevents |
|---|---|
| `test_arm_config_is_valid_json_and_profiler_merges` | broken `--additional-config` JSON, profiler config not merged |
| `test_every_flag_is_read_by_vllm_neuron` | a flag the code never reads, so an arm silently measures stock |
| `test_no_env_var_that_vllm_neuron_rejects` | `NEURON_RT_VISIBLE_CORES` under multiproc (run 2) |
| `test_single_shot_prefill_disables_prefix_caching` | Qwen3-VL: "APC requires segmented prefill" (five-model run) |
| `test_hbm_budget` | Qwen3-32B "Allocation Failure" at cap 0.8 (five-model run): weights + KV must leave at least 5 GiB per core for compiled programs |
| `test_known_broken_arms_are_not_started` | emb-C packed-prefill recompile crash (five-model run) |
| `test_payload_uses_supported_download_cli` | retired `huggingface-cli` (run 1); bad `--exclude` (GPT-OSS weights missing) |
| `test_changed_python_has_no_undefined_names` | `@nki.jit` without `import nki`, which broke every model's import (five-model attempt 1) |
| `test_vllm_neuron_imports_in_cpu_mode` | import-time errors. Needs the Neuron pip packages, so it runs on a CPU box with the Neuron SDK, not in GitHub CI. |

**Still needs the CPU-mode rehearsal** (`VLLM_NEURON_CPU_MODE=1` on an x86 box with the Neuron SDK), before any Trainium launch:
- runtime errors inside the engine, such as the multi-prefill audio-path rejection and the emb-C recompile
- dynamo trace errors

Speed and stage timing need the chip.
