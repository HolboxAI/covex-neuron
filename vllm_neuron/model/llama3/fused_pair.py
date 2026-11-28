"""TP8 BF16 single-token decode for two consecutive Llama 3.1 8B layers."""
import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from .fused_pair_kernel import pair_kernel

from vllm_neuron.functional.attention.attention_decode_mask import (
    gen_attention_decode_mask, maybe_transpose_mask_for_qk_swap,
)


def decode_layers(layers, x, positions, rope, metadata, ranks):
    first = metadata["layers.0.self_attn"]
    table = first["block_table_tensor"]
    batch = table.shape[0]
    if x.shape != (batch, 4096) or x.dtype != torch.bfloat16:
        raise ValueError("Llama fused decode requires one BF16 hidden state per batch slot")
    block = first["block_size"]
    mask = gen_attention_decode_mask(
        positions.reshape(1, batch).float(), bs=batch, q_head=4, s_active=1,
        s_prior=table.shape[1] * block, block_len=block,
    )
    mask = maybe_transpose_mask_for_qk_swap(mask, d_head=128)
    slots = first["slot_mapping"].reshape(batch, 1).to(torch.uint32)
    cos, sin = [t[:, :64].T.contiguous().reshape(64, batch, 1).to(x.dtype) for t in rope]
    blocks = table.to(torch.int32)
    wrapped = wrap_nki(pair_kernel)
    for i in range(0, len(layers), 2):
        weights = []
        caches = []
        for layer in (layers[i], layers[i+1]):
            attn = layer.self_attn
            info = metadata[f"layers.{layer.layer_idx}.self_attn"]
            if (info["block_table_tensor"] is not table or info["slot_mapping"] is not first["slot_mapping"]
                    or info["block_size"] != block):
                raise ValueError("Fused layers require shared page and physical-slot mappings")
            if (attn.k_cache.ndim != 4 or attn.k_cache.shape[1:] != (1, block, 128)
                    or attn.v_cache.shape != attn.k_cache.shape
                    or attn.k_cache.dtype != x.dtype or attn.v_cache.dtype != x.dtype):
                raise ValueError("Expected bound [pages, 1, block, 128] BF16 Llama caches")
            weights.extend((layer.input_layernorm.weight.view(1, -1), attn.qkv_proj_weight, attn.o_proj_weight,
                            layer.post_attention_layernorm.weight.view(1, -1), layer.mlp.gate_proj_weight,
                            layer.mlp.up_proj_weight, layer.mlp.down_proj_weight))
            caches.extend((attn.k_cache, attn.v_cache))
        # Direct bound K/V tensors let the compiler thread the four in-place writes.
        x = wrapped[2](x.reshape(batch, 1, -1), *weights, *caches, cos, sin, mask,
                       blocks, slots, layers[i].rms_norm_eps, ranks)[0]
    return x.reshape(batch, -1)
