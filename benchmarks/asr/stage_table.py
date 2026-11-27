#!/usr/bin/env python3
"""Per-request stage times (ms) for one arm at one concurrency, from two /metrics scrapes around the run + the client JSON.
Usage: stage_table.py <before.prom> <after.prom> <client.json> <concurrency> [device_STAGES.md]
Columns: host+HTTP+preprocess (client E2E - server E2E) | queue | prefill (encoder + decoder prefill) | decode | other.
Encoder vs decoder-prefill split and per-kernel times come from the device profile (STAGES.md), appended if given."""
import json, re, sys
from collections import defaultdict

def hist(path):
    s, c = defaultdict(float), defaultdict(float)
    for line in open(path):
        m = re.match(r'^(vllm:[a-z0-9_]+?)_(sum|count)(\{[^}]*\})?\s+([0-9.eE+-]+)$', line.strip())
        if m:
            (s if m[2] == "sum" else c)[m[1]] += float(m[4])
    return s, c

def mean_ms(name, b, a):
    ds, dc = a[0][name] - b[0][name], a[1][name] - b[1][name]
    return 1000 * ds / dc if dc > 0 else None

bef, aft, client, conc = hist(sys.argv[1]), hist(sys.argv[2]), json.load(open(sys.argv[3])), int(sys.argv[4])
tr = next((r for r in client["phases"]["transcribe"] if r["concurrency"] == conc), {})
client_e2e = 1000 * tr.get("e2e_mean_s", float("nan"))
q = mean_ms("vllm:request_queue_time_seconds", bef, aft)
pf = mean_ms("vllm:request_prefill_time_seconds", bef, aft)
dc = mean_ms("vllm:request_decode_time_seconds", bef, aft)
se = mean_ms("vllm:e2e_request_latency_seconds", bef, aft)
ttft = mean_ms("vllm:time_to_first_token_seconds", bef, aft)
f = lambda v: f"{v:.1f}" if isinstance(v, float) and v == v else "–"
host = client_e2e - se if se is not None else None
other = se - sum(x for x in (q, pf, dc) if x is not None) if se is not None else None
print(f"| c | client E2E | host+HTTP+preprocess | queue | prefill (enc+dec) | decode | other (server) | server TTFT |")
print("|---|---|---|---|---|---|---|---|")
print(f"| {conc} | {f(client_e2e)} | {f(host)} | {f(q)} | {f(pf)} | {f(dc)} | {f(other)} | {f(ttft)} |")
if len(sys.argv) > 5:
    print("\nDevice profile (kernel-level split of prefill/decode):\n")
    print(open(sys.argv[5]).read())
