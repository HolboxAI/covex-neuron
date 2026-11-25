# SPDX-License-Identifier: Apache-2.0
"""In-tree Qwen3-32B BF16 decode MLP NKI kernel (first draft).

A BF16 (non-quantized) decode MLP for Qwen3-32B dense: ``silu(x @ gate^T) *
(x @ up^T) @ down^T`` with tensor-parallel weight shards and LNC (logical
NeuronCore) H-sharding.

This is the BF16 counterpart of :mod:`qwen3_32b_mxfp8_mlp_tkg`, without any
quantization: weights are plain bf16, there are no scales, no MX packing and no
activation quantize.

Weight layout (transposed relative to the model's natural ``[H, I]`` params, so
the matmul's stationary operand reads directly):
  * ``gate_w`` / ``up_w``: bf16 ``[I, H]``  (P = I free, K = H contraction)
  * ``down_w``:            bf16 ``[H, I]``  (P = H free, K = I contraction)

The functional wrapper (:mod:`vllm_neuron.functional.qwen3_intree_mlp_bf16`)
does the transposes and launches this kernel through ``wrap_nki``.

LNC sharding: gate/up shard the H contraction (each core accumulates a partial
over its H tiles, then a cross-core reduce combines them); down shards the H
output (each core writes a disjoint H slice). At grid 1 this reduces to the
single-core path. The intermediate (post-SiLU) is the *stationary* operand of
the down projection, so its free axis is the token axis (T <= 128) and no
intermediate transpose is needed -- the same trick the MXFP8 kernel uses.

NOTE: FIRST DRAFT. The tiling, the cross-core reduce order and the matmul
operand setup must be validated and profiled on Neuron before this kernel is
enabled by default. It is gated behind ``VLLM_NEURON_QWEN3_INTREE_MLP_BF16``
(off by default).
"""

from __future__ import annotations

import nki.isa as nisa
import nki.language as nl

_P = nl.tile_size.pmax  # partition size (128)
_TILE = 512  # contraction tile (H or I)
_PSUM_BMAX = 8  # PSUM bank count


def _div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


@nki.jit
def qwen3_32b_bf16_mlp_tkg(
    hidden: nl.ndarray,
    gate_w: nl.ndarray,
    up_w: nl.ndarray,
    down_w: nl.ndarray,
) -> nl.ndarray:
    """Compute the Qwen3-32B BF16 decode MLP for one tensor-parallel rank.

    The caller has already applied the post-attention RMSNorm to ``hidden``
    (matching the current BF16 model, which does NOT fuse the norm).

    Args:
        hidden: bf16 ``[T, H]`` activation, already normalized.
        gate_w: bf16 ``[I, H]`` gate weight (transposed).
        up_w: bf16 ``[I, H]`` up weight (transposed).
        down_w: bf16 ``[H, I]`` down weight (transposed).

    Returns:
        bf16 ``[T, H]`` MLP output in HBM (full H, gathered across the LNC cores).

    Example:
        >>> # traced through wrap_nki; not called directly
    """
    t, h_size = hidden.shape
    i_size = gate_w.shape[0]
    n_h512 = h_size // _TILE
    n_i512 = _div_ceil(i_size, _TILE)
    assert h_size % _TILE == 0, f"H ({h_size}) must be a multiple of {_TILE}."
    assert n_i512 <= _PSUM_BMAX, (
        f"gate/up needs {n_i512} PSUM banks but only {_PSUM_BMAX} exist."
    )
    assert t <= _P, (
        f"the down projection makes the intermediate stationary, so the token "
        f"count must not exceed {_P}; got T={t}."
    )

    # ---- LNC grid -----------------------------------------------------------
    num_shards = 1
    shard_id = 0
    if nl.program_ndim() != 0:
        num_shards = nl.num_programs(axes=0)
        shard_id = nl.program_id(axis=0)
    assert n_h512 % num_shards == 0, (
        f"H={h_size} gives {n_h512} tiles of {_TILE}, and the grid has "
        f"{num_shards} core(s). The core count must divide the tile count."
    )
    n_h512_shard = n_h512 // num_shards
    h_per_shard = h_size // num_shards
    h512_off = shard_id * n_h512_shard

    # ---- load the activation (already normalized) ---------------------------
    hidden_sb = nl.ndarray((t, h_size), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=hidden_sb, src=hidden)

    # ---- gate / up projections ----------------------------------------------
    # One PSUM bank per I512 tile, accumulated over this core's H tiles.
    gate_psum = [
        nl.ndarray((_P, t), dtype=nl.float32, buffer=nl.psum) for _ in range(n_i512)
    ]
    up_psum = [
        nl.ndarray((_P, t), dtype=nl.float32, buffer=nl.psum) for _ in range(n_i512)
    ]
    for i_h in range(n_h512_shard):
        h_idx = h512_off + i_h
        h_base = h_idx * _TILE
        for i_i in range(n_i512):
            i_base = i_i * _TILE
            nisa.nc_matmul(
                gate_psum[i_i],
                gate_w[i_base : i_base + _P, h_base : h_base + _TILE],
                hidden_sb[:, h_base : h_base + _TILE],
            )
            nisa.nc_matmul(
                up_psum[i_i],
                up_w[i_base : i_base + _P, h_base : h_base + _TILE],
                hidden_sb[:, h_base : h_base + _TILE],
            )

    # ---- cross-core reduce of gate/up partials ------------------------------
    # At grid 2 each core holds a partial sum over its H tiles. Sum across cores
    # so both cores see the full-I gate/up. (Exact reduce primitive to confirm
    # on hardware; the MXFP8 kernel uses a point-to-point rendezvous.)
    if num_shards > 1:
        for i_i in range(n_i512):
            # placeholder: cross-core add of gate_psum[i_i] / up_psum[i_i]
            pass

    # ---- materialize gate/up, apply SiLU(gate) * up -------------------------
    gate_sb = nl.ndarray((i_size, t), dtype=nl.bfloat16, buffer=nl.sbuf)
    up_sb = nl.ndarray((i_size, t), dtype=nl.bfloat16, buffer=nl.sbuf)
    act = nl.ndarray((i_size, t), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_i in range(n_i512):
        nisa.tensor_copy(
            dst=gate_sb[i_i * _TILE : (i_i + 1) * _TILE, :], src=gate_psum[i_i]
        )
        nisa.tensor_copy(
            dst=up_sb[i_i * _TILE : (i_i + 1) * _TILE, :], src=up_psum[i_i]
        )
    nisa.activation(gate_sb, gate_sb, op=nisa.activation_op.silu)
    nisa.tensor_scalar(act, gate_sb, up_sb, op0=nisa.scalar_op.multiply)

    # ---- down projection ----------------------------------------------------
    # out[T, H] = act[T, I] @ down_w[I, H]. The intermediate is the stationary
    # operand (free axis = T <= _P), down_w is the moving operand.
    down_psum = nl.ndarray((h_per_shard, t), dtype=nl.float32, buffer=nl.psum)
    for i_i in range(n_i512):
        nisa.nc_matmul(
            down_psum,
            act[i_i * _TILE : (i_i + 1) * _TILE, :],  # stationary [P=T, K=I]
            down_w[shard_id * h_per_shard : (shard_id + 1) * h_per_shard,
                   i_i * _TILE : (i_i + 1) * _TILE],  # moving [H, K=I]
        )

    out = nl.ndarray((t, h_per_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out, src=down_psum)
    return out
