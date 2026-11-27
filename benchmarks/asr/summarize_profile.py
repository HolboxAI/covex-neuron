#!/usr/bin/env python3
"""Per-stage time table from Neuron Explorer Perfetto traces (*.pftrace).

Usage: summarize_profile.py <profile_dir>      (requires: pip install perfetto)
For every trace it prints (1) busy time per stage, from the slice names, and (2) the top slices by total time,
so each arm's attention / MLP / collectives / DMA / norm / sampling / host time can be compared side by side.
Stage rules are name regexes; anything unmatched is reported as "other" and listed in the top-slices table.
"""
import glob, os, re, sys
from perfetto.trace_processor import TraceProcessor

STAGES = [
    ("collective (all-reduce / all-gather / barrier)", r"all.?reduce|all.?gather|reduce.?scatter|collective|\bcc[_ ]|barrier|nccom|ring"),
    ("attention", r"attn|attention|flash|sdpa|softmax|tkg|cte"),
    ("mlp / ffn", r"\bmlp|ffn|gate|up_proj|down_proj|silu|gelu"),
    ("qkv / o-proj / matmul", r"qkv|o_proj|matmul|gemm|\bdot\b|linear|lm_head"),
    ("norm / elementwise", r"rms|norm|add|mul|residual|rope|rotary"),
    ("dma / copy", r"dma|copy|memcpy|transfer|load|store|h2d|d2h"),
    ("sampling", r"sampl|argmax|topk|top_k|logit"),
    ("host / runtime / launch", r"nrt|exec|submit|launch|queue|host|python|schedule|wait"),
]

def stage_of(name: str) -> str:
    n = (name or "").lower()
    for stage, rx in STAGES:
        if re.search(rx, n):
            return stage
    return "other"

def summarize(path: str) -> None:
    tp = TraceProcessor(trace=path)
    rows = list(tp.query("select name, sum(dur) as total, count(*) as n from slice group by name"))
    span = list(tp.query("select (max(ts + dur) - min(ts)) as span from slice"))[0].span or 1
    tp.close()
    by_stage = {}
    for r in rows:
        s = stage_of(r.name)
        by_stage[s] = by_stage.get(s, 0) + (r.total or 0)
    total = sum(by_stage.values()) or 1
    print(f"\n## {os.path.basename(path)}  (trace span {span/1e6:.2f} ms)\n")
    print("| Stage | Busy time (ms) | Share of slice time |\n|---|---|---|")
    for s, v in sorted(by_stage.items(), key=lambda kv: -kv[1]):
        print(f"| {s} | {v/1e6:.3f} | {100*v/total:.1f}% |")
    print("\n| Top slices | Total (ms) | Count | Stage |\n|---|---|---|---|")
    for r in sorted(rows, key=lambda r: -(r.total or 0))[:25]:
        print(f"| `{(r.name or '')[:70]}` | {(r.total or 0)/1e6:.3f} | {r.n} | {stage_of(r.name)} |")

if __name__ == "__main__":
    d = sys.argv[1]
    traces = sorted(glob.glob(os.path.join(d, "**", "*.pftrace"), recursive=True))
    if not traces:
        sys.exit(f"no .pftrace under {d}; run: neuron-explorer view -d {d} --output-format perfetto")
    print(f"# Stage breakdown: {d}")
    for t in traces:
        summarize(t)
