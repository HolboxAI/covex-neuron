"""Run the serving kernel on eight simulated ranks with a dense PyTorch oracle.

Set LLAMA_NKI_SRC to the patched NKI repository's src directory.
"""
import argparse
import importlib.util
import math
import os
from pathlib import Path
import sys


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-shard", action="store_true")
    parser.add_argument("--device", action="store_true", help="run via torchrun on eight dedicated Neuron cores")
    parser.add_argument("--first-core", type=int, default=0)
    options = parser.parse_args()
    rank = None
    if options.device:
        rank = int(os.environ["LOCAL_RANK"])
        if int(os.environ["WORLD_SIZE"]) != 8 or options.first_core < 0:
            parser.error("device mode requires eight local ranks and a nonnegative first core")
        os.environ.update(NEURON_RT_VISIBLE_CORES=str(options.first_core + rank), NEURON_RT_NUM_CORES="1")

    os.environ.update(NEURON_PLATFORM_TARGET_OVERRIDE="trn2", NEURON_LOGICAL_NC_CONFIG="2", NKI_PRECISE_FP="1")
    src = Path(os.environ["LLAMA_NKI_SRC"]).resolve()
    sys.path[:0] = [str(src), str(src / "nkilib_src")]
    import torch
    if rank is None:
        from nki._simulator import simulate_kernel_multi_rank
        from nki.framework._simulate import SimulatorFrameworkContext
        from nki.compiler.target import resolve_target
    else:
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
        from vllm_neuron.envs import get_compile_backend_name
        from vllm_neuron.parallel.neuron_parallel_state import initialize_neuron_parallel_state

        torch.distributed.init_process_group("gloo")
        initialize_neuron_parallel_state(tp_global_ranks=list(range(8)), local_rank=rank)
    from nkilib.core.attention.gen_mask_tkg_torch import build_full_attention_mask
    from nkilib.core.attention.attention_tkg_utils import is_qk_swapped

    torch.set_num_threads(1)
    ref = load("paged_reference", src.parent / "covex/llama_paged_parity.py")
    kernel = load("serving_kernel", Path(__file__).resolve().parents[2] /
                  "vllm_neuron/model/llama3/fused_pair_kernel.py").pair_kernel
    layers, table, initial, inputs, head = ref.fixture(options.full_shard, simulate=True)
    # Distinct TP shards, shared RMS scales. Scale partial outputs to unit aggregate variance.
    shards = [[tuple(w if j in (0, 3) else torch.roll(w, r * 31, dims=0)
                     * (1 / math.sqrt(8) if j in (2, 6) else 1)
                     for j, w in enumerate(layer)) for layer in layers] for r in range(8)]
    pools = [[p.roll(r, dims=0) for p in initial] for r in range(8)]
    oracle = [[p.clone() for p in rank] for rank in pools]
    batch, d, ctx = ref.BATCH, initial[0].shape[-1], table.shape[1] * ref.BLOCK
    if rank is not None:
        wrapped = wrap_nki(kernel)
        def device_step(*values):
            return wrapped[2](*values)
        compiled = torch.compile(device_step, backend=get_compile_backend_name(), fullgraph=True)
        device_pools = [p.to("neuron:0") for p in pools[rank]]
        addresses = [p.data_ptr() for p in device_pools]

    for step, x in enumerate(inputs):
        pos = torch.tensor([15 + step, 16 + step, 31 + step, 0])
        slots = ref.slots_for(table, pos)
        cos, sin = ref.rope_inputs(pos, d)
        expected = x
        for layer in range(2):
            attn = []
            for r in range(8):
                w = shards[r][layer]
                q, k, v = ref.project(expected, w, d, cos, sin)
                rows = []
                for b, length in enumerate(pos.tolist()):
                    pages = table[b, :(length + ref.BLOCK - 1) // ref.BLOCK]
                    keys, values = [pool[pages, 0].reshape(-1, d)[:length] for pool in oracle[r][2*layer:2*layer+2]]
                    keys = torch.cat((keys, k[b:b+1])).float()
                    values = torch.cat((values, v[b:b+1])).float()
                    rows.append((q[b].float() @ keys.T / math.sqrt(d)).softmax(-1) @ values)
                    if b < ref.LIVE:
                        for pool, update in zip(oracle[r][2*layer:2*layer+2], (k, v)):
                            pool.view(-1, d)[slots[b]] = update[b]
                attn.append(torch.stack(rows).to(x.dtype).view(batch, -1) @ w[2])
            expected = expected + torch.stack(attn).float().sum(0).to(x.dtype)
            mlp = []
            for rank_layers in shards:
                w = rank_layers[layer]
                z = ref.norm(expected, w[3])
                mlp.append((torch.nn.functional.silu(z @ w[4]) * (z @ w[5])) @ w[6])
            expected = expected + torch.stack(mlp).float().sum(0).to(x.dtype)
        swapped = is_qk_swapped(batch, ref.HEADS, d, 1, ctx, 2, 128, True, True, False, False)
        mask = build_full_attention_mask(pos, batch, ref.HEADS, 1, ctx, lnc=2, block_len=ref.BLOCK,
                                         include_active_mask=True, transposed=not swapped).contiguous()
        before = [[p.clone() for p in rank] for rank in pools]
        converters, args = [], []
        for r in (range(8) if rank is None else (rank,)):
            values = (x[:, None], *shards[r][0], *shards[r][1], *pools[r],
                      cos.T.contiguous().unsqueeze(-1), sin.T.contiguous().unsqueeze(-1), mask, table.int(),
                      slots.to(torch.uint32).reshape(batch, 1), ref.EPS, tuple(range(8)))
            if rank is not None:
                # Reuse the bound input pools across steps; a new output allocation is a failure.
                device_values = [v.to("neuron:0") if isinstance(v, torch.Tensor) else v for v in values[:15]]
                device_values.extend(device_pools)
                device_values.extend(v.to("neuron:0") if isinstance(v, torch.Tensor) else v for v in values[19:])
                y, *cache = compiled(*device_values)
                ref.check_storage_aliases(device_pools, cache, addresses)
                ref.compare((y.cpu().reshape(batch, -1), cache), (expected, oracle[r]), before[r], slots, head)
                pools[r] = [p.cpu() for p in device_pools]
                continue
            conv = SimulatorFrameworkContext()
            args.append(conv.convert_args(values, {})[0])
            converters.append(conv)
        if rank is not None:
            torch.distributed.barrier()
            print(f"PASS device rank={rank} step={step+1} logits and four cache aliases", flush=True)
            continue
        result = simulate_kernel_multi_rank(kernel.func, args, [{} for _ in args], 8, lnc=2,
                                            target=resolve_target(kernel.func, None))
        for r, conv in enumerate(converters):
            conv.sync_input_tensors()
            y, *cache = conv.convert_result(result[r])
            ref.compare((y.reshape(batch, -1), cache), (expected, oracle[r]), before[r], slots, head)
            for owner, returned in zip(pools[r], cache, strict=True):
                torch.testing.assert_close(owner, returned, atol=0, rtol=0)
        print(f"PASS step {step+1}, eight TP ranks, two LNC programs per rank, four page caches per rank", flush=True)

    if rank is not None:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
