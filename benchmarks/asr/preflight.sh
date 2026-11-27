#!/usr/bin/env bash
# Local preflight: run before launching any instance. Fails on anything that would kill the run on the box.
#  1. model reachable (config.json only)  2. every cloned/fetched input reachable
#  3. every env var the scripts export is read somewhere in vllm_neuron at this commit (catches renamed/rejected vars)
#  4. vllm-neuron's own rejection list (e.g. NEURON_RT_VISIBLE_CORES under multiproc) is not set by us
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"; fail=0
curl -sfI https://huggingface.co/Qwen/Qwen3-ASR-1.7B/resolve/main/config.json > /dev/null && echo "ok   model config reachable" || { echo "FAIL model config"; fail=1; }
git ls-remote https://github.com/HolboxAI/covex-neuron "$(git rev-parse --abbrev-ref HEAD)" > /dev/null && echo "ok   branch on origin" || { echo "FAIL branch not pushed"; fail=1; }
[ -z "$(git status --porcelain benchmarks vllm_neuron)" ] && echo "ok   branch clean" || { echo "FAIL uncommitted changes"; fail=1; }
for v in $(grep -hoE '\bexport [A-Z_]+=|\b(VLLM_NEURON|COVEX|NEURON)_[A-Z_]+=' benchmarks/asr/*.sh | sed -E 's/export //; s/=$//' | sort -u); do
  case "$v" in NEURON_LIBTORCH_COMPILATION_TIMEOUT|NEURON_SKIP_EFA_AFFINITY|TORCH_DEVICE_BACKEND_AUTOLOAD|NEURON_RT_INSPECT_*) continue;; esac
  if git grep -q "\"$v\"\|'$v'\|$v\b" -- vllm_neuron; then echo "ok   env $v is read by vllm_neuron"; else echo "FAIL env $v is not read anywhere in vllm_neuron"; fail=1; fi
done
for v in $(git grep -hoE '[A-Z_]{8,} cannot be used' -- vllm_neuron | awk '{print $1}' | sort -u); do
  if grep -q "export $v=" benchmarks/asr/*.sh; then echo "FAIL scripts export $v, which vllm_neuron rejects"; fail=1; else echo "ok   not setting rejected $v"; fi
done
for s in benchmarks/asr/*.sh; do bash -n "$s" || { echo "FAIL syntax $s"; fail=1; }; done
exit $fail
