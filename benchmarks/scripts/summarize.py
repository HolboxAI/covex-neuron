#!/usr/bin/env python3
"""Summarize round-1 results: median of repeats per (arm, workload, concurrency), deltas vs a baseline arm.
Usage: summarize.py <RESULT_DIR> <BASELINE_ARM>  -> prints a Markdown table."""
import glob, json, os, re, statistics, sys
from collections import defaultdict

res_dir, base = sys.argv[1], sys.argv[2]
pts = defaultdict(list)
for f in glob.glob(os.path.join(res_dir, "*.json")):
    m = re.match(r"(\w+?)_(W\d)_c(\d+)_r\d+\.json", os.path.basename(f))
    if m:
        pts[(m[1], m[2], int(m[3]))].append(json.load(open(f)))
keys = ["output_throughput", "median_ttft_ms", "p90_ttft_ms", "median_itl_ms", "median_tpot_ms", "median_e2el_ms"]
med = {k: {x: statistics.median(r.get(x, float("nan")) for r in v) for x in keys} for k, v in pts.items()}
print("| Arm | Workload | Conc | out tok/s | vs %s | TTFT p50 ms | TTFT p90 ms | ITL p50 ms | TPOT p50 ms |" % base)
print("|---|---|---|---|---|---|---|---|---|")
for (arm, wl, c) in sorted(med, key=lambda k: (k[1], k[2], k[0])):
    m, b = med[(arm, wl, c)], med.get((base, wl, c))
    d = f"{(m['output_throughput'] / b['output_throughput'] - 1) * 100:+.1f}%" if b else "n/a"
    print(f"| {arm} | {wl} | {c} | {m['output_throughput']:.1f} | {d} | {m['median_ttft_ms']:.0f} | {m['p90_ttft_ms']:.0f} | {m['median_itl_ms']:.1f} | {m['median_tpot_ms']:.1f} |")
