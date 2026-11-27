#!/usr/bin/env python3
"""One table: stock (T) vs COVEX (C) per model and load point, from results/*.json. Usage: summarize.py results"""
import glob, json, os, sys
R = sys.argv[1]; J = lambda p: json.load(open(p)) if os.path.exists(p) else None
rows = []
def add(model, point, metric, t, c, unit):
    ratio = f"{c / t:.2f}x" if t and c else "–"
    rows.append(f"| {model} | {point} | {metric} | {t if t is not None else '–'} | {c if c is not None else '–'} | {ratio} |")
for c in ("c1", "c8"):
    T, C = J(f"{R}/q32-T-{c}.json"), J(f"{R}/q32-C-{c}.json")
    g = lambda d, k: round(d[k], 2) if d and k in d else None
    add("Qwen3-32B", c, "output tok/s", g(T, "output_throughput"), g(C, "output_throughput"), "tok/s")
    add("Qwen3-32B", c, "TPOT p50 ms (lower is better)", g(T, "median_tpot_ms"), g(C, "median_tpot_ms"), "ms")
for c in ("c1", "c4"):
    T = J(f"{R}/oss-T-{c}.json"); add("GPT-OSS-20B (stock only)", c, "output tok/s", round(T["output_throughput"], 2) if T else None, None, "tok/s")
for t in ("t64", "t256"):
    T, C = J(f"{R}/emb-T-{t}.json"), J(f"{R}/emb-C-{t}.json")
    add("Qwen3-Embedding-8B", f"{t} c32", "sequences/s", T and T["seq_per_s"], C and C["seq_per_s"], "seq/s")
T, C = J(f"{R}/vl-T.json"), J(f"{R}/vl-C.json")
add("Qwen3-VL-32B", "c4", "requests/s", T and T["req_per_s"], C and C["req_per_s"], "req/s")
add("Qwen3-VL-32B", "c4", "latency p50 ms (lower is better)", T and T["lat_p50_ms"], C and C["lat_p50_ms"], "ms")
for conc in (1, 8):
    def a(arm):
        d = J(f"{R}/{arm}.json")
        if not d: return None
        x = [r for r in d["phases"]["transcribe"] if r["concurrency"] == conc]
        return round(x[0]["audio_seconds_per_s"], 1) if x else None
    add("Qwen3-ASR-1.7B", f"c{conc}", "audio-s/s", a("asr-T"), a("asr-C"), "audio-s/s")
print("| Model | Load | Metric | Stock (T) | COVEX (C) | C / T |\n|---|---|---|---|---|---|")
print("\n".join(rows))
print("\n**Correctness (COVEX vs stock):**")
for m in ("q32", "emb", "vl"):
    p = f"{R}/{m}.match.json"
    print(f"- {m}: {open(p).read().strip() if os.path.exists(p) else 'not run'}")
for arm in ("asr-T", "asr-C"):
    d = J(f"{R}/{arm}.json")
    if d: print(f"- {arm} WER on 100 clips: {round(100 * d['phases']['transcribe'][0]['wer_mean'], 2)}%")
