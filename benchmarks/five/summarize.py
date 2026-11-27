#!/usr/bin/env python3
"""T vs C table from results/*.json. A ratio needs both arms complete, numerically valid, doing equal work, and
correct (gate.py); otherwise the reason. ASR shows WER beside speed; its ratio is quality-qualified only within 0.5 pp.
Usage: summarize.py <results> [--required "q32-T q32-C asr-T ..."] -> exit 1 when a required comparison is incomplete."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate import asr_level_ok, asr_quality_ok, bench_ok, count_ok, load, match_ok, same_work  # noqa: E402

R = sys.argv[1]
REQUIRED = sys.argv[sys.argv.index("--required") + 1].split() if "--required" in sys.argv else []
rows, incomplete = [], []


def add(key, model, point, metric, t, c, checks, lower_better=False, note=""):
    why = [w for ok, w in checks if not ok]
    ratio = f"{(t / c if lower_better else c / t):.2f}x" if not why and t and c else "–"
    if why:
        incomplete.append((key, f"{model} {point} {metric}: " + "; ".join(why)))
    f = lambda v: "–" if v is None else v
    status = "; ".join(why) or ("complete" + (f"; {note}" if note else ""))
    rows.append(f"| {model} | {point} | {metric} | {f(t)} | {f(c)} | {ratio} | {status} |")


tag = lambda pfx, ok: (ok[0], f"{pfx}: {ok[1]}" if not ok[0] else "")
g = lambda d, k: round(d[k], 2) if d and d.get(k) is not None else None
q32_match = match_ok(f"{R}/q32.match.json", 20)
for c, n in (("c1", 8), ("c8", 32)):
    T, C = load(f"{R}/q32-T-{c}.json"), load(f"{R}/q32-C-{c}.json")
    ch = [tag("T", bench_ok(T, n, 256, 1024)), tag("C", bench_ok(C, n, 256, 1024)), same_work(T, C), q32_match]
    add("q32", "Qwen3-32B", c, "output tok/s", g(T, "output_throughput"), g(C, "output_throughput"), ch)
    add("q32", "Qwen3-32B", c, "TPOT p50 ms (lower is better)", g(T, "median_tpot_ms"), g(C, "median_tpot_ms"), ch, lower_better=True)
for c, n in (("c1", 8), ("c4", 16)):
    T = load(f"{R}/oss-T-{c}.json")
    add("oss", "GPT-OSS-20B (stock only)", c, "output tok/s", g(T, "output_throughput"), None,
        [tag("T", bench_ok(T, n, 256)), (False, "no COVEX arm")])
for t in ("t64", "t256"):
    T, C = load(f"{R}/emb-T-{t}.json"), load(f"{R}/emb-C-{t}.json")
    add("emb", "Qwen3-Embedding-8B", f"{t} c32", "sequences/s", g(T, "seq_per_s"), g(C, "seq_per_s"),
        [tag("T", count_ok(T, 256, metric="seq_per_s")), tag("C", count_ok(C, 256, metric="seq_per_s")),
         match_ok(f"{R}/emb.match.json", 20)])
T, C = load(f"{R}/vl-T.json"), load(f"{R}/vl-C.json")
ch = [tag("T", count_ok(T, 40, metric="req_per_s")), tag("C", count_ok(C, 40, metric="req_per_s")), match_ok(f"{R}/vl.match.json", 20)]
add("vl", "Qwen3-VL-32B", "c4", "requests/s", g(T, "req_per_s"), g(C, "req_per_s"), ch)
add("vl", "Qwen3-VL-32B", "c4", "latency p50 ms (lower is better)", g(T, "lat_p50_ms"), g(C, "lat_p50_ms"), ch, lower_better=True)
for conc in (1, 8):
    T, C = load(f"{R}/asr-T.json"), load(f"{R}/asr-C.json")
    a = lambda d: next((round(r["audio_seconds_per_s"], 1) for r in d["phases"]["transcribe"] if r["concurrency"] == conc), None) if d else None
    q = asr_quality_ok(T, C, conc)
    add("asr", "Qwen3-ASR-1.7B", f"c{conc}", "audio-s/s", a(T), a(C),
        [tag("T", asr_level_ok(T, conc, 100)), tag("C", asr_level_ok(C, conc, 100)), q], note=q[1] if q[0] else "")
print("| Model | Load | Metric | Stock (T) | COVEX (C) | C vs T | Status |\n|---|---|---|---|---|---|---|")
print("\n".join(rows))
print("\nRatios only when both arms completed every request with valid numbers, equal work, and agreeing outputs (gate.py).")
need = {a.split("-")[0] for a in REQUIRED}
bad = [msg for key, msg in incomplete if key in need]
for x in bad:
    print(f"INCOMPLETE (required): {x}")
sys.exit(1 if bad else 0)
