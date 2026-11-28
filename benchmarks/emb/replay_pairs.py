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
B is bound to the request id returned with its embedding, and the trace must confirm packed=True.
"""
import argparse
import hashlib
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


TRACE = re.compile(r"COVEX_PACK_TRACE prefill step: members=(\[.*?\]) total_padded=(\d+) "
                   r"padded_num_reqs=(\d+) packed=(True|False)")


def trace_steps(text):
    import ast
    out, seen = [], {}
    for m in TRACE.finditer(text):
        members = ast.literal_eval(m.group(1))
        key = tuple(sorted(x["req"] for x in members))
        step = {"members": members, "total_padded": int(m.group(2)), "packed": m.group(4) == "True"}
        if key not in seen:          # every TP rank logs the same step
            seen[key] = step; out.append(step)
        elif seen[key] != step:      # disagreeing rank traces cannot certify a single packed layout
            out.append(step)
    return out


def layout(steps, order, request_ids):
    """B's physical placement for one case: the single step that holds exactly this case's requests, B bound to its own
    request id returned with B's embedding. None when the case did not run as one packed step."""
    if (len(steps) != 1 or not steps[0]["packed"] or len(request_ids) != len(order)
            or any(not isinstance(req, str) or not req for req in request_ids)
            or len(set(request_ids)) != len(order)):
        return None
    members = steps[0]["members"]
    if len(members) != len(order):
        return None
    # vLLM 0.29 returns external IDs from embed(), but its workers normally log
    # external_id + "-" + eight lowercase hex digits. Require a bijection for
    # the whole step in a single mode. Never infer membership from batch order.
    by_output = {m["req"]: m for m in members}
    if set(by_output) != set(request_ids):
        by_output = {}
        for member in members:
            external, _, suffix = member["req"].rpartition("-")
            if not re.fullmatch("[0-9a-f]{8}", suffix) or external in by_output:
                return None
            by_output[external] = member
        if set(by_output) != set(request_ids):
            return None
    b = by_output[request_ids[order.index("B")]]
    return {"b_req": b["req"], "b_offset": b["start"], "b_padded": b["padded"], "bucket": steps[0]["total_padded"],
            "members": [(m["req"], m["start"], m["tokens"], m["padded"]) for m in members]}


def pick(texts, tok, witness=None):
    """A, A' of equal length (B's offset fixed), B, X of a different length than A."""
    lens = {}
    for t in texts:
        lens.setdefault(len(tok.encode(t, add_special_tokens=True)), []).append(t)
    same = sorted((n, v) for n, v in lens.items() if len(v) >= 3)
    if not same:
        raise ValueError("need three prompts of one token length")
    if witness is None:
        n, v = same[-1]
    else:
        v = []
        for name in ("A", "A_prime", "B"):
            record = witness["diagnostic_inputs"][name]
            index = record["input_line_one_based"] - 1
            if index < 0 or index >= len(texts):
                raise ValueError(f"missing original {name} in dataset")
            text = texts[index]
            if hashlib.sha256(text.encode()).hexdigest() != record["text_sha256"]:
                raise ValueError(f"dataset differs from the campaign for {name}")
            v.append(text)
        lengths = {len(tok.encode(text, add_special_tokens=True)) for text in v}
        if len(lengths) != 1 or len(set(v)) != 3:
            raise ValueError("the original peer controls need three distinct equal-length inputs")
        n = lengths.pop()
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
    ap.add_argument("--case-file", help="JSON identities of the original campaign inputs")
    a = ap.parse_args(argv)
    os.environ.setdefault("COVEX_PACK_TRACE", "1")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    texts = [json.loads(l)["text"] for l in open(a.dataset)]
    witness = json.loads(Path(a.case_file).read_text()) if a.case_file else None
    prompts, n_tok = pick(texts, tok, witness)
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
            with log.open("rb") as stream:
                stream.seek(start)
                steps = trace_steps(stream.read().decode("utf-8", errors="replace"))
            lay = layout(steps, order, [getattr(out, "request_id", None) for out in outs])
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
    if witness:
        offset = witness["physical_valid_tokens_in_logged_pair"]
        for name in ("A+B", "A'+B"):
            if any(key[0] != offset for key in lay.get(name, ())):
                why.append(f"{name}: failed to reproduce the original offset {offset}")
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
    res["prompt_sha256"] = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in prompts.items()}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("verdict", "packing_verified", "repeat_stable", "peer_changed_same_offset_exact",
                                           "peer_changed_B_first_exact")}))
    return 0 if verdict != "inconclusive" else 1


if __name__ == "__main__":
    sys.exit(main())
