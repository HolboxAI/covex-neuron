#!/usr/bin/env python3
"""Qwen3-VL-8B pass-1 summary. Usage: summarize.py <results> <metrics> --arms "vl8-U vl8-Ub vl8-C" --windows 3 --n 40
Per arm: output tok/s per window and median, requests/s, p90 request latency, completed/requested, prompt tokens and
padding (bucket tokens / real tokens, from the exact usage.prompt_tokens and the arm's bucket list), and the server's
mean per-request prefill / decode / queue time from vLLM /metrics deltas. Ratios COVEX vs vanilla (the bar) and vs
tuned stock, printed only when every window of both arms completed all requests and the first 20 answers match
vanilla's. Exit 1 when the COVEX-vs-vanilla comparison is incomplete."""
import json, os, re, statistics, sys

a = sys.argv
R, M = a[1], a[2]
ARMS = a[a.index("--arms") + 1].split(); W = int(a[a.index("--windows") + 1]); N = int(a[a.index("--n") + 1])
BUCKETS = {"vl8-U": [8192], "vl8-Ub": [256, 512, 1024, 2048, 4096, 8192], "vl8-C": [256, 512, 1024, 2048, 4096, 8192]}
failed = {}
if "--status" in a and os.path.exists(a[a.index("--status") + 1]):
    for line in open(a[a.index("--status") + 1]):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and p[2].startswith("FAIL"):
            failed.setdefault(p[0], []).append(f"{p[1]} {p[2]}"[:90])


def load(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def prom(path):
    try:
        t = open(path).read()
    except OSError:
        return None
    if not t:
        return None
    out = {}
    for name in ("request_prefill_time_seconds", "request_decode_time_seconds", "request_queue_time_seconds"):
        for suf in ("sum", "count"):
            k = f"vllm:{name}_{suf}"
            out[k] = sum(float(x) for x in re.findall(rf"^{re.escape(k)}(?:{{[^}}]*}})? ([0-9.eE+-]+)$", t, re.M))
    return out


def stage_means(arm):
    """mean seconds per request over the scored windows, from /metrics deltas; None when any scrape is missing."""
    tot = {"prefill": [0.0, 0.0], "decode": [0.0, 0.0], "queue": [0.0, 0.0]}
    for w in range(1, W + 1):
        b, e = prom(f"{M}/{arm}-w{w}.before.prom"), prom(f"{M}/{arm}-w{w}.after.prom")
        if b is None or e is None:
            return None
        for key, name in (("prefill", "request_prefill_time_seconds"), ("decode", "request_decode_time_seconds"),
                          ("queue", "request_queue_time_seconds")):
            s = e[f"vllm:{name}_sum"] - b[f"vllm:{name}_sum"]; c = e[f"vllm:{name}_count"] - b[f"vllm:{name}_count"]
            if c < 0 or s < 0:
                return None   # counter reset
            tot[key][0] += s; tot[key][1] += c
    return {k: (v[0] / v[1] if v[1] else None) for k, v in tot.items()}


def bucket(t, bs):
    return next((b for b in bs if b >= t), None)


rows, ref_answers = {}, None
for arm in ARMS:
    ds = [load(f"{R}/{arm}-c4-w{w}.json") for w in range(1, W + 1)]
    ok = [d is not None and d.get("n") == N and isinstance(d.get("out_tok_per_s"), (int, float)) and d["out_tok_per_s"] > 0
          and len(d.get("completion_tokens") or []) == N for d in ds]
    tps = [d["out_tok_per_s"] for d, o in zip(ds, ok) if o]
    pt = [t for d in ds if d for t in (d.get("prompt_tokens") or []) if isinstance(t, int)]
    pad = (sum(bucket(t, BUCKETS[arm]) or 0 for t in pt) / sum(pt)) if pt else None
    rows[arm] = {"tps": tps, "median": round(statistics.median(tps), 2) if len(tps) == W else None, "complete": all(ok),
                 "answers": (ds[0] or {}).get("answers"), "req_s": statistics.median([d["req_per_s"] for d in ds if d]) if all(ok) else None,
                 "p90": max((d.get("lat_p90_ms") for d in ds if d), default=None), "prompt_p50": statistics.median(pt) if pt else None,
                 "prompt_max": max(pt) if pt else None, "padding": pad, "stages": stage_means(arm), "stage_probe": load(f"{R}/{arm}-stage.json"),
                 "failed": failed.get(arm, [])}
print("| Arm | out tok/s per window | median | req/s | p90 request ms | prompt tokens p50 / max | padded / real tokens | "
      "mean prefill / decode / queue s per request | serial image cost s | Status |\n|---|---|---|---|---|---|---|---|---|---|")
for arm, r in rows.items():
    st = r["stages"]; sp = r["stage_probe"] or {}
    stg = " / ".join("–" if (st is None or st[k] is None) else f"{st[k]:.3f}" for k in ("prefill", "decode", "queue"))
    pad_s = f"{r['padding']:.1f}x" if r["padding"] else "–"
    print(f"| {arm} | {', '.join(f'{x:.1f}' for x in r['tps']) or '–'} | {r['median'] or '–'} | {r['req_s'] or '–'} | {r['p90'] or '–'} | "
          f"{r['prompt_p50'] or '–'} / {r['prompt_max'] or '–'} | {pad_s} | {stg} | "
          f"{sp.get('image_cost_s', '–') if not isinstance(sp.get('image_cost_s'), float) else round(sp['image_cost_s'], 3)} | "
          f"{'; '.join(r['failed']) or ('complete' if r['complete'] else 'incomplete windows')} |")
ref = rows.get("vl8-U")
incomplete = []
print("\n| Comparison | Ratio | Status |\n|---|---|---|")
for arm, base in (("vl8-C", "vl8-U"), ("vl8-C", "vl8-Ub"), ("vl8-Ub", "vl8-U")):
    x, b = rows.get(arm), rows.get(base)
    why = []
    if not x or not b:
        why.append("arm not run")
    else:
        if not (x["complete"] and b["complete"]) or x["median"] is None or b["median"] is None:
            why.append("incomplete windows")
        if x["failed"] or b["failed"]:
            why.append("failed step")
        if ref and x["answers"] != ref["answers"]:
            why.append("answers differ from vanilla stock")
    ratio = f"{x['median'] / b['median']:.2f}x" if not why else "–"
    print(f"| {arm} vs {base} | {ratio} | {'; '.join(why) or 'complete'} |")
    if why and (arm, base) == ("vl8-C", "vl8-U"):
        incomplete.append("; ".join(why))
for w in incomplete:
    print(f"INCOMPLETE (required): vl8-C vs vl8-U: {w}")
sys.exit(1 if incomplete else 0)
