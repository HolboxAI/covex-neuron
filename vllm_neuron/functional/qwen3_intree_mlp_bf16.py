# SPDX-License-Identifier: Apache-2.0
"""Torch entry point for the in-tree Qwen3-32B BF16 decode MLP kernel.

Wraps the ``@nki.jit`` kernel in ``vllm_neuron._kernels.mlp`` so the Qwen3-32B
BF16 decode path can call it from the traced model graph, the same way
``qwen3_intree_mlp`` wraps the MXFP8 kernel. The kernel is launched at the
process LNC degree. Each tensor-parallel rank runs it on its own weight shard,
and vLLM does the cross-rank all-reduce.

Disabled by default; enable with ``VLLM_NEURON_QWEN3_INTREE_MLP_BF16=1``.
"""

from __future__ import annotations

from torch import Tensor

from vllm_neuron._kernels.mlp.qwen3_32b_bf16_mlp_tkg import qwen3_32b_bf16_mlp_tkg

__all__ = ["qwen3_intree_mlp_bf16"]


def qwen3_intree_mlp_bf16(
    hidden: Tensor,
    gate_w: Tensor,
    up_w: Tensor,
    down_w: Tensor,
) -> Tensor:
    """Run the in-tree Qwen3-32B BF16 decode MLP for one tensor-parallel rank.

    The model stores gate/up as ``[H, I]`` and down as ``[I, H]``. The kernel's
    stationary operands want the transposes ``[I, H]`` / ``[H, I]``, so this
    wrapper transposes the (constant) weights once per graph capture.

    Args:
        hidden: bf16 ``[T, H]`` activation, already normalized (the caller applies
            the post-attention RMSNorm, matching the BF16 model).
        gate_w: bf16 ``[H, I]`` gate weight.
        up_w: bf16 ``[H, I]`` up weight.
        down_w: bf16 ``[I, H]`` down weight.

    Returns:
        bf16 ``[T, H]`` per-rank MLP output (the caller does the all-reduce).
    """
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from vllm_neuron.utils.hardware_config import get_logical_nc_config

    lnc = get_logical_nc_config()
    result = wrap_nki(qwen3_32b_bf16_mlp_tkg)[lnc](
        hidden=hidden,
        gate_w=gate_w.t().contiguous(),
        up_w=up_w.t().contiguous(),
        down_w=down_w.t().contiguous(),
    )
    return result[0] if isinstance(result, (tuple, list)) else result
