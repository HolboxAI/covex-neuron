#!/usr/bin/env python3
"""Per-request stage times (ms) for one arm at one concurrency, from two /metrics scrapes around the run + the client JSON.
Usage: stage_table.py <before.prom> <after.prom> <client.json> <concurrency> [device_STAGES.md]
Columns: host+HTTP+preprocess (client E2E - server E2E) | queue | prefill (encoder + decoder prefill) | decode | other.
Attribution goes through the same validation as stage_levels.py (stage_levels.level): a missing or failed scrape, a
counter reset, or a server request count that differs from the client's successful requests makes the row
"unavailable" - it never falls back to lifetime counters. Host residuals are derived only from a validated row."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_levels import STAGES, hist, level  # noqa: E402

before, after, client_path, conc = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
try:
    client = json.load(open(client_path))
    tr = [r for r in client.get("phases", {}).get("transcribe", []) if r.get("concurrency") == conc]
except (OSError, ValueError):
    tr = []
n_ok = tr[0].get("n_success") if len(tr) == 1 else None
got, why = level(hist(before), hist(after), n_ok)
print("| c | client E2E | host+HTTP+preprocess | queue | prefill (enc+dec) | decode | other (server) | server TTFT |")
print("|---|---|---|---|---|---|---|---|")
if why:
    print(f"| {conc} | – | – | – | – | – | – | – |\n\nstage attribution unavailable: {why}")
else:
    v = dict(zip([s for s, _ in STAGES], [None if x == "–" else float(x) for x in got[1]]))
    client_e2e = 1000 * tr[0]["e2e_mean_s"] if tr[0].get("e2e_mean_s") is not None else None
    f = lambda x: "–" if x is None else f"{x:.1f}"
    host = client_e2e - v["e2e"] if client_e2e is not None and v["e2e"] is not None else None
    other = v["e2e"] - sum(x for x in (v["queue"], v["prefill"], v["decode"]) if x is not None) if v["e2e"] is not None else None
    print(f"| {conc} | {f(client_e2e)} | {f(host)} | {f(v['queue'])} | {f(v['prefill'])} | {f(v['decode'])} | {f(other)} | {f(v['TTFT'])} |")
if len(sys.argv) > 5 and os.path.exists(sys.argv[5]):
    print("\nDevice profile (kernel-level split of prefill/decode):\n")
    print(open(sys.argv[5]).read())
