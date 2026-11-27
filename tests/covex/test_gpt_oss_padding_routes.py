"""Exercise the production block mapper on CPU. Only accelerator imports are stubbed."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def map_blocks():
    def accelerator_only(*args, **kwargs):
        raise AssertionError("unexpected accelerator dispatch on CPU")

    # Load the leaf module without importing the accelerator-only package initializer.
    stubs = {
        "nki": {"jit": lambda: lambda kernel: kernel},
        "nkilib.core.subkernels.find_nonzero_indices": {"find_nonzero_indices": accelerator_only},
        "nkilib.core.subkernels.indexed_flatten": {"indexed_flatten": accelerator_only},
        "libtorch_neuronx_lite.nki.nki_hop": {"wrap_nki": accelerator_only},
        "vllm_neuron.utils.neuron_utils": {"can_run_kernel": lambda tensor: False},
    }
    modules = {}
    for name, attributes in stubs.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        modules[name] = module
    spec = importlib.util.spec_from_file_location(
        "oss_moe_blockwise", ROOT / "vllm_neuron/functional/moe/moe_blockwise.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module.build_blockwise_mapping


def weighted_routes(mapping, rows, experts, block_size):
    # Reconstruct the effective weight of every token/expert pair from dispatched
    # blocks. Accumulating catches duplicate or dropped assignments.
    weights, token_ids, block_experts, conditions = mapping
    valid = token_ids >= 0
    assert torch.equal(conditions.bool(), valid.view(-1, block_size).any(dim=1))
    slots = token_ids[valid].long() * experts + block_experts.repeat_interleave(block_size)[valid]
    effective = torch.zeros(rows * experts, dtype=weights.dtype)
    effective.index_add_(0, slots, weights.flatten()[slots])
    return effective.view(rows, experts)


@pytest.mark.parametrize(("rows", "length"), [(8192, 1024), (512, 1), (512, 512), (512, 0)])
def test_gpt_oss_prefill_routes(map_blocks, rows, length):
    experts, top_k, block_size = 32, 4, 256
    rng = torch.Generator().manual_seed(83)
    scores = torch.randn(rows, experts, generator=rng)
    values, indices = scores.topk(top_k)
    affinities = torch.zeros_like(scores).scatter_(1, indices, values.softmax(dim=-1))
    mask = torch.arange(rows) < length
    group = SimpleNamespace(rank_in_group_tensor=torch.zeros(1, dtype=torch.int32))
    args = (affinities, experts, top_k, block_size, group)
    stock = map_blocks(*args, padding_mask=mask)
    tuned = map_blocks(*args, padding_mask=mask, skip_padded_routes=True)
    expected = affinities * mask[:, None]
    for mapping in (stock, tuned):
        assert torch.equal(weighted_routes(mapping, rows, experts, block_size), expected)
    assert (stock[1] >= 0).sum().item() == rows * top_k
    assert (tuned[1] >= 0).sum().item() == length * top_k
    assert not (tuned[1] >= length).any()
    if length == rows:
        for before, after in zip(stock, tuned):
            assert torch.equal(before, after)
