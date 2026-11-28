"""Two paged Llama decode layers with TP collectives on Neuron."""
import nki
from nkilib.experimental.transformer.transformer_tkg import transformer_tkg


@nki.jit
def pair_kernel(x, g0, q0, o0, m0, a0, u0, z0, g1, q1, o1, m1, a1, u1, z1,
                k0, v0, k1, v1, cos, sin, mask, table, slots, eps, ranks):
    return transformer_tkg(
        x, [q0, q1], [o0, o1], [a0, a1], [u0, u1], [z0, z1],
        [g0, g1], [m0, m1], [k0, k1], [v0, v1], cos, sin, mask, None,
        num_layers=2, eps=eps, replica_groups=[list(ranks)],
        sbuf_residual_and_cc=True, rmsnorm_X_enabled=False,
        active_blocks_table=table, kv_cache_update_idx=slots, K_cache_transposed=False,
        return_updated_kv=True,
    )
