#!/usr/bin/env python3
"""One steady-state decode step, stock vs a COVEX arm, from Neuron device profiles (all TP ranks).
Usage: decode_step.py --layers 64 ARM=<dir with per-rank *.pftrace> [ARM=<dir> ...]
       (convert first: neuron-explorer view -d <profile_dir> --output-format perfetto; NEFFs must sit in <dir>/neffs)

Per arm it reports, for the median-duration decode execution taken from the middle (steady state) of the capture:
  step ms        elapsed wall time of that step, counted ONCE across ranks: max(end) - min(start) over the 8 ranks
  exposed comm   wall time inside the step where a collective is active on some rank and no compute is active on any
                 rank (the time the step waits on all-reduce), split into "after attention" and "after MLP"
  overlapped     collective time that coincides with compute on some rank
The attention/MLP split assumes Qwen3's two TP all-reduces per layer in order (o_proj, then down_proj) and is reported
only when exactly 2 x layers collectives are found in the step on the reference rank; otherwise it is "unknown". If no
slice name matches a collective, every communication figure is "unknown". Nothing is summed across ranks.
"""
import os, re, sys

COLL = re.compile(r"all.?reduce|all.?gather|reduce.?scatter|collective|\bcc[_ ]|nccom|\bcc\b", re.I)
IDLE = re.compile(r"idle|wait|barrier|semaphore|notify", re.I)


def union(iv):
    """total length of the union of [s, e) intervals"""
    tot, cur_s, cur_e = 0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return tot + ((cur_e - cur_s) if cur_e is not None else 0)


def intersect(a, b):
    """length of union(a) ∩ union(b)"""
    return union(a) + union(b) - union(a + b)


def subtract_len(a, b):
    """length of union(a) minus union(b)"""
    return union(a + b) - union(b)


def pick_steps(ranks):
    """ranks: {rank: [(name, ts, dur, depth)]}. Decode executions = the most frequent depth-0 slice name on rank 0.
    Returns the step index chosen (median duration among the middle half) and per-rank (start, end) of that step."""
    r0 = min(ranks)
    top = [x for x in ranks[r0] if x[3] == 0]
    if not top:
        return None, "no top-level execution slices"
    names = {}
    for n, *_ in top:
        names[n] = names.get(n, 0) + 1
    dec = max(names, key=names.get)
    seqs = {r: sorted((x for x in v if x[3] == 0 and x[0] == dec), key=lambda x: x[1]) for r, v in ranks.items()}
    n = min(len(s) for s in seqs.values())
    if n < 4:
        return None, f"only {n} decode executions on some rank"
    mid = list(range(n // 4, 3 * n // 4))
    durs = [max(seqs[r][i][1] + seqs[r][i][2] for r in seqs) - min(seqs[r][i][1] for r in seqs) for i in mid]
    k = mid[durs.index(sorted(durs)[len(durs) // 2])]
    return k, {r: (seqs[r][k][1], seqs[r][k][1] + seqs[r][k][2]) for r in seqs}


def leaves(v, s, e):
    """slices inside [s, e) that contain no deeper slice: a wrapper spanning the step is not compute"""
    inside = [x for x in v if x[3] > 0 and x[1] < e and x[1] + x[2] > s]
    out = []
    for n, ts, dur, d in inside:
        if not any(d2 > d and ts2 >= ts and ts2 + dur2 <= ts + dur for _, ts2, dur2, d2 in inside):
            out.append((n, ts, dur, d))
    return out


def analyze(ranks, layers, tp=8):
    if len(ranks) != tp:   # every TP rank, or no communication claim at all
        return {"error": f"{len(ranks)} rank traces, expected {tp}; communication split unknown"}
    k, win = pick_steps(ranks)
    if k is None:
        return {"error": win}
    t0, t1 = min(s for s, _ in win.values()), max(e for _, e in win.values())
    coll, comp, per_rank_coll = [], [], {}
    for r, v in ranks.items():
        s, e = win[r]
        for name, ts, dur, depth in leaves(v, s, e):
            iv = (max(ts, s), min(ts + dur, e))
            if COLL.search(name or ""):
                coll.append(iv); per_rank_coll.setdefault(r, []).append(iv)
            elif not IDLE.search(name or ""):
                comp.append(iv)
    out = {"step_index": k, "step_ms": (t1 - t0) / 1e6, "ranks": len(ranks)}
    missing = [r for r in ranks if not per_rank_coll.get(r)]
    if coll and missing:
        out.update(exposed_ms="unknown", attn_ms="unknown", mlp_ms="unknown", overlap_ms="unknown",
                   note=f"ranks {missing} show no collective in the step; split unknown")
        return out
    if not coll:
        out.update(exposed_ms="unknown", attn_ms="unknown", mlp_ms="unknown", overlap_ms="unknown",
                   note="no slice name matches a collective")
        return out
    out["exposed_ms"] = subtract_len(coll, comp) / 1e6
    out["overlap_ms"] = intersect(coll, comp) / 1e6
    out["comm_union_ms"] = union(coll) / 1e6
    ref = per_rank_coll.get(min(ranks), [])
    if len(ref) == 2 * layers:
        order = sorted(ref)
        attn_iv, mlp_iv = order[0::2], order[1::2]
        # attribute exposed time by position: each collective interval keeps its exposure across all ranks' compute
        out["attn_ms"] = sum(subtract_len([iv], comp) for iv in attn_iv) / 1e6
        out["mlp_ms"] = sum(subtract_len([iv], comp) for iv in mlp_iv) / 1e6
    else:
        out["attn_ms"] = out["mlp_ms"] = "unknown"
        out["note"] = f"{len(ref)} collectives on rank {min(ranks)} in the step (expected {2 * layers}); split unknown"
    return out


def load_dir(d):
    """per-rank slices from Perfetto traces (one per NeuronCore); rank = nc index in the file name, else file order"""
    import glob
    from perfetto.trace_processor import TraceProcessor
    ranks = {}
    files = sorted(glob.glob(os.path.join(d, "**", "*.pftrace"), recursive=True))
    for i, f in enumerate(files):
        m = re.search(r"nc_?(\d+)", os.path.basename(f))
        r = int(m.group(1)) if m else i
        tp = TraceProcessor(trace=f)
        ranks[r] = [(x.name, x.ts, x.dur, x.depth) for x in tp.query("select name, ts, dur, depth from slice")]
        tp.close()
    return ranks


def fmt(v):
    return v if isinstance(v, str) else f"{v:.3f}"


if __name__ == "__main__":
    a = sys.argv[1:]
    layers = int(a[a.index("--layers") + 1]) if "--layers" in a else 64
    tp = int(a[a.index("--tp") + 1]) if "--tp" in a else 8
    arms = [x for x in a if "=" in x]
    print("| Arm | Ranks | Step (ms, once across ranks) | Exposed all-reduce wait (ms) | after attention | after MLP | "
          "Comm overlapped with compute (ms) | Note |")
    print("|---|---|---|---|---|---|---|---|")
    for spec in arms:
        arm, d = spec.split("=", 1)
        r = analyze(load_dir(d), layers, tp)
        if "error" in r:
            print(f"| {arm} | – | unknown | unknown | unknown | unknown | unknown | {r['error']} |"); continue
        print(f"| {arm} | {r['ranks']} | {fmt(r['step_ms'])} | {fmt(r['exposed_ms'])} | {fmt(r['attn_ms'])} | "
              f"{fmt(r['mlp_ms'])} | {fmt(r['overlap_ms'])} | {r.get('note', '')} |")
