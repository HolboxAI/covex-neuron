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
  inconclusive          a case did not run as exactly one step of its own requests, B's offset/padding/bucket was not
                        identical across repeats or between the two cases of a peer comparison, or an identical
                        repeat changed B (exit 1)
B is bound to its own request id (submission order) in the trace, never just by member count.
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
            seen.add(key); out.append({"members": members, "total_padded": int(m.group(2))})
    return out


def _num(req):
    """Offline request ids are consecutive counters (possibly with a suffix); order of submission = numeric order."""
    m = re.search(r"\d+", str(req))
    return int(m.group()) if m else -1


def layout(steps, order):
    """B's physical placement for one case: the single step that holds exactly this case's requests, B bound to its own
    request id (the id submitted at B's position). None when the case did not run as one step."""
    if len(order) == 1:
        if len(steps) != 1 or len(steps[0]["members"]) != 1:
            return None
        m = steps[0]["members"][0]
        return {"b_req": m["req"], "b_offset": m["start"], "b_padded": m["padded"], "bucket": steps[0]["total_padded"]}
    if len(steps) != 1 or len(steps[0]["members"]) != len(order):
        return None
    by_id = sorted(steps[0]["members"], key=lambda m: _num(m["req"]))
    b = by_id[order.index("B")]
    return {"b_req": b["req"], "b_offset": b["start"], "b_padded": b["padded"], "bucket": steps[0]["total_padded"],
            "members": [(m["req"], m["start"], m["tokens"], m["padded"]) for m in steps[0]["members"]]}


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
    if log is None:
        raise SystemExit("--log is required: packing must be verified from the server trace")
    vectors, runs = {}, []
    for rep in range(a.repeat):
        for name, order in CASES.items():
            start = log.stat().st_size if log.exists() else 0
            outs = llm.embed([prompts[k] for k in order], use_tqdm=False)
            vectors.setdefault(name, []).append(list(outs[order.index("B")].outputs.embedding))
            lay = layout(trace_steps(log.read_text(errors="replace")[start:]), order)
            runs.append({"case": name, "repeat": rep, "order": order, "layout": lay})
    why = []
    lay = {}
    for r in runs:
        if r["layout"] is None:
            why.append(f"{r['case']} repeat {r['repeat']}: did not run as one step with exactly its requests")
        else:
            key = (r["layout"]["b_offset"], r["layout"]["b_padded"], r["layout"]["bucket"])
            lay.setdefault(r["case"], set()).add(key)
    for case, keys in lay.items():
        if len(keys) != 1:
            why.append(f"{case}: B's offset/padding/bucket changed across repeats {sorted(keys)}")
    for x, y in (("A+B", "A'+B"), ("B+A", "B+A'")):
        if lay.get(x) and lay.get(y) and lay[x] != lay[y]:
            why.append(f"{x} vs {y}: B's layout differs {sorted(lay[x])} vs {sorted(lay[y])} (peer comparison not controlled)")
    repeat_stable = all(vec_cmp(vs[0], v)["exact"] for vs in vectors.values() for v in vs[1:])
    if not repeat_stable:
        why.append("identical case repeated gave a different B vector")
    ref = vectors["B"][0]
    cmp = {name: [vec_cmp(ref, v) for v in vs] for name, vs in vectors.items()}
    same_off = vec_cmp(vectors["A+B"][0], vectors["A'+B"][0])["exact"]
    first_same = vec_cmp(vectors["B+A"][0], vectors["B+A'"][0])["exact"]
    all_eq = all(c["exact"] for cs in cmp.values() for c in cs)
    if why:
        verdict = "inconclusive"
    else:
        verdict = ("invariant" if all_eq else "follows_peer_content" if not (same_off and first_same) else "follows_offset")
    verified = not why
    res = {"verdict": verdict, "packing_verified": verified, "repeat_stable": repeat_stable, "token_length": n_tok,
           "inconclusive_because": why, "layouts": {k: sorted(v) for k, v in lay.items()},
           "peer_changed_same_offset_exact": same_off, "peer_changed_B_first_exact": first_same,
           "vs_B_alone": {k: v[0] for k, v in cmp.items()}, "runs": runs}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("verdict", "packing_verified", "repeat_stable", "peer_changed_same_offset_exact",
                                           "peer_changed_B_first_exact")}))
    return 0 if verdict != "inconclusive" else 1


if __name__ == "__main__":
    sys.exit(main())
