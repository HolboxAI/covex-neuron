#!/usr/bin/env python3
"""Controlled embedding pair replay (diagnostic, offline engine, one Neuron arm).

The HTTP isolation test could not force a pair into one packed step: the server split 6 of 8 pairs. Here the offline
engine receives every prompt of a case BEFORE its first scheduler step (LLM.embed adds all requests, then steps), so
a pair shares one packed prefill whenever max_prefills_per_batch >= 2. COVEX_PACK_TRACE=1 records each step's members,
offsets and padding; every repeat is checked against the trace and a case only counts when its members really shared
one step.

  python3 replay_pairs.py --model DIR --dataset t64.jsonl --tokenizer DIR --out pairs.json [--packing 8] [--repeat 3]

Cases (A and A' have the same token length, so B's offset is identical in both):
  B                 B alone (packed path, one member)
  A+B, A'+B         B second at offset len(A): peer content changes, offset fixed
  B+A, B+A'         B first at offset 0: peer content changes
  X+B               B second behind a peer of a DIFFERENT length: offset changes, B's content fixed
Verdict (written to the JSON, not decided by the caller):
  follows_peer_content  A+B != A'+B (same offset)     -> cross-request masking / leakage first
  follows_offset        A+B == A'+B but != B alone    -> position ids, bounds, tiling at that offset
  invariant             all B vectors equal B alone   -> packing is exact for B
Exit 0 when every case ran as intended (verified packing); the verdict is reported either way.
"""
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path


def vec_cmp(a, b):
    dot = math.fsum(x * y for x, y in zip(a, b))
    na = math.sqrt(math.fsum(x * x for x in a)); nb = math.sqrt(math.fsum(y * y for y in b))
    return {"cosine": dot / (na * nb) if na and nb else None, "exact": list(a) == list(b),
            "max_abs_delta": max((abs(x - y) for x, y in zip(a, b)), default=0.0)}


TRACE = re.compile(r"COVEX_PACK_TRACE prefill step: members=(\[.*?\]) total_padded=(\d+)")


def trace_steps(text):
    import ast
    out, seen = [], set()
    for m in TRACE.finditer(text):
        members = ast.literal_eval(m.group(1))
        key = tuple(sorted(x["req"] for x in members))
        if key not in seen:          # every TP rank logs the same step
            seen.add(key); out.append(members)
    return out


def pick(texts, tok):
    """A, A' of equal length (B's offset fixed), B, X of a different length than A."""
    lens = {}
    for t in texts:
        lens.setdefault(len(tok.encode(t, add_special_tokens=True)), []).append(t)
    same = sorted((n, v) for n, v in lens.items() if len(v) >= 3)
    if not same:
        raise ValueError("need three prompts of one token length")
    n, v = same[-1]
    other = next((w[0] for m, w in sorted(lens.items(), key=lambda kv: abs(kv[0] - n // 2)) if m != n), None)
    if other is None:
        raise ValueError("need a prompt of a different length for the offset case")
    return {"A": v[0], "A_prime": v[1], "B": v[2], "X": other}, n


CASES = {"B": ["B"], "A+B": ["A", "B"], "A'+B": ["A_prime", "B"], "B+A": ["B", "A"], "B+A'": ["B", "A_prime"], "X+B": ["X", "B"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True); ap.add_argument("--dataset", required=True); ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--packing", type=int, default=8); ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--log", help="file the engine's stderr goes to (for the trace); default: the process's own stderr redirect")
    a = ap.parse_args(argv)
    os.environ.setdefault("COVEX_PACK_TRACE", "1")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    texts = [json.loads(l)["text"] for l in open(a.dataset)]
    prompts, n_tok = pick(texts, tok)
    from vllm import LLM
    cfg = {"neuron_config": {"max_prefills_per_batch": a.packing, "num_batched_tokens_buckets": [128, 256, 512, 1024, 2048, 4096]}}
    llm = LLM(model=a.model, runner="pooling", dtype="bfloat16", tensor_parallel_size=8, max_model_len=4096,
              max_num_batched_tokens=4096, max_num_seqs=32, enable_prefix_caching=False, additional_config=cfg)
    log = Path(a.log) if a.log else None
    vectors, runs = {}, []
    for rep in range(a.repeat):
        for name, order in CASES.items():
            start = log.stat().st_size if log and log.exists() else 0
            outs = llm.embed([prompts[k] for k in order], use_tqdm=False)
            b = list(outs[order.index("B")].outputs.embedding)
            vectors.setdefault(name, []).append(b)
            steps = trace_steps(log.read_text(errors="replace")[start:]) if log else []
            shared = [s for s in steps if len(s) == len(order)]
            runs.append({"case": name, "repeat": rep, "order": order, "steps": steps,
                         "packed_as_intended": len(order) == 1 or (len(shared) == 1 and len(steps) == 1),
                         "b_offset": next((s[order.index("B")]["start"] for s in shared), None)})
    ref = vectors["B"][0]
    cmp = {name: [vec_cmp(ref, v) for v in vs] for name, vs in vectors.items()}
    repeat_stable = all(vec_cmp(vs[0], v)["exact"] for vs in vectors.values() for v in vs[1:])
    same_off = vec_cmp(vectors["A+B"][0], vectors["A'+B"][0])["exact"]
    first_same = vec_cmp(vectors["B+A"][0], vectors["B+A'"][0])["exact"]
    all_eq = all(c["exact"] for cs in cmp.values() for c in cs)
    verdict = ("invariant" if all_eq else "follows_peer_content" if not (same_off and first_same)
               else "follows_offset")
    verified = all(r["packed_as_intended"] for r in runs) if log else False
    res = {"verdict": verdict, "packing_verified": verified, "repeat_stable": repeat_stable, "token_length": n_tok,
           "peer_changed_same_offset_exact": same_off, "peer_changed_B_first_exact": first_same,
           "vs_B_alone": {k: v[0] for k, v in cmp.items()}, "runs": runs}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("verdict", "packing_verified", "repeat_stable", "peer_changed_same_offset_exact",
                                           "peer_changed_B_first_exact")}))
    return 0 if verified else 1


if __name__ == "__main__":
    sys.exit(main())
