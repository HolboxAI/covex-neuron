#!/usr/bin/env python3
"""Qwen3-32B primary experiment summary. Usage: summarize.py <results> --arms "q32-U q32-T q32-K" --windows 3
Per arm and load point: output tok/s of every unprofiled window, the median, completed/requested, TPOT and TTFT p50/p90
(medians over windows). A ratio vs U or T is printed only when every window of both arms completed fully (gate.py) and
the arm's greedy outputs on the exact-length cache prompts equal the reference at c1 and c3. Exit 1 when a required
comparison (K vs U, K vs T) is incomplete."""
import os, statistics, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "five"))
from gate import bench_ok, load, match_ok, same_work  # noqa: E402

a = sys.argv
R = a[1]; ARMS = a[a.index("--arms") + 1].split(); W = int(a[a.index("--windows") + 1]) if "--windows" in a else 3
if W < 1:
    sys.exit("summarize: --windows must be >= 1 (no measurement windows means no comparison)")
STATUS = a[a.index("--status") + 1] if "--status" in a else None
EXPLORATORY = {"q32-F", "q32-C"}   # dynamic FA: throughput shown even when tokens diverge, labelled, never headline
failed_steps = {}   # arms with a failed run step (ready, provenance, profile, evidence, bench) cannot back a ratio
if STATUS and os.path.exists(STATUS):
    for line in open(STATUS):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and p[2].startswith("FAIL"):
            failed_steps.setdefault(p[0], []).append(f"{p[1]} {p[2]}"[:80])
LOADS = (("c1", 8), ("c8", 32)); REQUIRED = [("q32-K", "q32-U"), ("q32-K", "q32-T")]


def windows(arm, c, n):
    ds = [load(f"{R}/{arm}-{c}-w{w}.json") for w in range(1, W + 1)]
    oks = [bench_ok(d, n, 256, 1024) for d in ds]
    bad = [f"w{i + 1}: {why}" for i, (ok, why) in enumerate(oks) if not ok]
    return ds, bad


def med(ds, k):
    v = [d[k] for d in ds if d and d.get(k) is not None]
    return round(statistics.median(v), 2) if v else None


print("## Per arm (unprofiled windows)\n")
print("| Arm | Load | out tok/s per window | median tok/s | completed | TPOT p50 / p90 ms | TTFT p50 / p90 ms | Status |")
print("|---|---|---|---|---|---|---|---|")
summary = {}
for arm in ARMS:
    for c, n in LOADS:
        ds, bad = windows(arm, c, n)
        summary[(arm, c)] = (med(ds, "output_throughput"), bad)
        per = ", ".join(f"{d['output_throughput']:.1f}" if d and isinstance(d.get("output_throughput"), (int, float)) else "invalid"
                        for d in ds)
        done = sum((d or {}).get("completed", 0) for d in ds)
        print(f"| {arm} | {c} | {per} | {summary[(arm, c)][0] or '–'} | {done}/{n * W} | {med(ds, 'median_tpot_ms')} / "
              f"{med(ds, 'p90_tpot_ms')} | {med(ds, 'median_ttft_ms')} / {med(ds, 'p90_ttft_ms')} | {'; '.join(bad) or 'complete'} |")

print("\n## Comparisons (median output tok/s; ratio only when complete and correct)\n")
print("| Arm | vs | Load | Arm tok/s | Ref tok/s | Ratio | Status |\n|---|---|---|---|---|---|---|")
incomplete = []
for arm in ARMS:
    for ref in ("q32-U", "q32-T"):
        if arm == ref or ref not in ARMS:
            continue
        corr = [match_ok(f"{R}/{arm}.vs.{ref}.cache.c{k}.match.json", 18) for k in (1, 3)]
        for c, _ in LOADS:
            (x, bx), (r, br) = summary[(arm, c)], summary[(ref, c)]
            work = [same_work(load(f"{R}/{arm}-{c}-w{w}.json"), load(f"{R}/{ref}-{c}-w{w}.json")) for w in range(1, W + 1)]
            steps = [f"{z} step failed: {f}" for z in (arm, ref) for f in failed_steps.get(z, [])][:2]
            hard = [f"{arm} {b}" for b in bx] + [f"{ref} {b}" for b in br] + [w for ok, w in work if not ok][:1] + steps
            corr_why = [f"cache c{k}: {m[1]}" for k, m in zip((1, 3), corr) if not m[0]]
            if arm in EXPLORATORY and not hard and corr_why and x and r:
                # dynamic FA: keep throughput visible, labelled; tokens differ, so no equivalence or headline claim
                print(f"| {arm} | {ref} | {c} | {x} | {r} | {x / r:.2f}x (exploratory) | tokens differ: {'; '.join(corr_why)} |")
                continue
            why = hard + corr_why
            ratio = f"{x / r:.2f}x" if not why and x and r else "–"
            label = "complete" + (" (exploratory arm)" if arm in EXPLORATORY else "")
            print(f"| {arm} | {ref} | {c} | {x or '–'} | {r or '–'} | {ratio} | {'; '.join(why) or label} |")
            if why and (arm, ref) in REQUIRED:
                incomplete.append(f"{arm} vs {ref} {c}: " + "; ".join(why))
# a requested exploratory arm must have complete, valid windows; only its token agreement is optional
for arm in [x for x in ARMS if x in EXPLORATORY]:
    for c, n in LOADS:
        bad = summary[(arm, c)][1]
        if bad or summary[(arm, c)][0] is None:
            incomplete.append(f"{arm} (requested exploratory arm) {c}: " + ("; ".join(bad) or "no valid windows"))
for x in incomplete:
    print(f"INCOMPLETE (required): {x}")
missing = [p for p in REQUIRED if p[0] not in ARMS or p[1] not in ARMS]
for p in missing:
    print(f"INCOMPLETE (required): {p[0]} vs {p[1]}: arm not run")
sys.exit(1 if incomplete or missing else 0)
