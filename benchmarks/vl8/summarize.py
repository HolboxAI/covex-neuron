#!/usr/bin/env python3
"""Qwen3-VL-8B pass-1 summary: SHORT-ANSWER VL TPS (each question asks for a single option letter, so output tokens are
few and the number measures prefill-dominated serving, not long-form generation). Usage: summarize.py <results> <metrics> --arms "vl8-U vl8-Ub vl8-C" --windows 3 --n 40
Per arm: output tok/s per window and median, requests/s, p90 request latency, completed/requested, prompt tokens and
padding (bucket tokens / real tokens, from the exact usage.prompt_tokens and the arm's bucket list), and the server's
mean per-request prefill / decode / queue time from vLLM /metrics deltas. Ratios COVEX vs vanilla (the bar) and vs
tuned stock, printed only when every window of both arms completed all requests, ALL answers and per-request output
token counts of EVERY window equal vanilla's (requests are issued in the same order), and the arm's p90 request
latency is within P90_TOL (10%, the campaign's non-Qwen tolerance) of the comparison arm's. Exit 1 when the
COVEX-vs-vanilla comparison is incomplete or disqualified."""
P90_TOL = 1.10
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


def pooled_p90(ds):
    """p90 request latency (ms) over EVERY measured request of every scored window (W x N values), never a max or
    median of per-window p90s (Astra: max-of-window p90 printed 4.00x complete while pooled p90 went 100 -> 250 ms).
    None when any window lacks a full set of finite per-request latencies."""
    import math
    lat = []
    for d in ds:
        v = (d or {}).get("latencies_s")
        if not isinstance(v, list) or len(v) != N or not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in v):
            return None
        lat += v
    lat.sort()
    return 1000 * lat[max(0, math.ceil(0.9 * len(lat)) - 1)]


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
                 "answers_w": [(d or {}).get("answers") for d in ds], "tokens_w": [(d or {}).get("completion_tokens") for d in ds], "req_s": statistics.median([d["req_per_s"] for d in ds if d]) if all(ok) else None,
                 "p90": pooled_p90(ds), "prompt_p50": statistics.median(pt) if pt else None,
                 "prompt_max": max(pt) if pt else None, "padding": pad, "stages": stage_means(arm), "stage_probe": load(f"{R}/{arm}-stage.json"),
                 "failed": failed.get(arm, [])}
print("Arms: vl8-U = stock with an 8,192-token prefill bucket (the bar); vl8-Ub = stock with COVEX's buckets (attribution);")
print("vl8-C = COVEX. p90 = pooled over all measured requests of all scored windows.\n")
print("| Arm | out tok/s per window | median | req/s | pooled p90 request ms | prompt tokens p50 / max | padded / real tokens | "
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
generic_fail = failed.get("bench", [])          # a failed window not attributed to one arm voids every comparison


def answer_problems(side, ref, name):
    for w in range(W):
        xa, ra = side["answers_w"][w], ref["answers_w"][w]
        if not isinstance(xa, list) or len(xa) != N or not isinstance(ra, list) or len(ra) != N:
            return f"{name} w{w + 1}: answers not recorded for all {N} requests"
        blank = sum(1 for t in xa if not isinstance(t, str) or not t.strip())
        if blank:
            return f"{name} w{w + 1}: {blank}/{N} blank answers"
        if xa != ra:
            return f"{name} w{w + 1}: answers differ from vanilla stock on {sum(p != q for p, q in zip(xa, ra))}/{N}"
        if side["tokens_w"][w] != ref["tokens_w"][w]:
            return f"{name} w{w + 1}: per-request output token counts differ from vanilla stock"
    return ""
incomplete = []
print("\n## Short-answer VL-8B (single option letter per question): COVEX vs stock with an 8,192-token prefill bucket\n")
print("| Comparison | Ratio | Status |\n|---|---|---|")
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
        if ref:
            # BOTH arms of every comparison (an attribution ratio vs tuned stock too) must reproduce vanilla's answers
            # and token counts in every window, and every answer must be a real, non-blank reply
            for side, arm_name in ((x, arm), (b, base)):
                bad = answer_problems(side, ref, arm_name)
                if bad:
                    why.append(bad)
        if generic_fail:
            why.append("a window failed: " + "; ".join(generic_fail))
        if x["p90"] is None or b["p90"] is None:
            why.append("pooled p90 unavailable (per-request latencies missing or incomplete)")
        elif x["p90"] > P90_TOL * b["p90"]:
            why.append(f"pooled p90 {x['p90']:.0f} ms > {P90_TOL:.2f} x {b['p90']:.0f} ms of {base}")
    ratio = f"{x['median'] / b['median']:.2f}x" if not why else "–"
    print(f"| {arm} vs {base} | {ratio} | {'; '.join(why) or 'complete'} |")
    if why and (arm, base) == ("vl8-C", "vl8-U"):
        incomplete.append("; ".join(why))
for w in incomplete:
    print(f"INCOMPLETE (required): vl8-C vs vl8-U: {w}")
sys.exit(1 if incomplete else 0)
