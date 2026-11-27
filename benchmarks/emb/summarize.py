#!/usr/bin/env python3
"""Embedding packing sweep summary. Usage: summarize.py <results> --arms "emb-U emb-T emb-P4 ..." --lens "t64 t256" --windows 3
           [--status STATUS.tsv]
Per arm and length: seq/s of every window and the median. A ratio vs U (headline) or T (attribution) only when every
window of both arms served all 256 requests with a positive rate, the arm had no failed run step, and EVERY vector of
window 1 has cosine >= 0.9999 to the reference. Exit 1 if any requested arm's comparison vs U is incomplete."""
import os, statistics, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "five"))
from gate import count_ok, load, match_ok  # noqa: E402

a = sys.argv; R = a[1]; ARMS = a[a.index("--arms") + 1].split(); LENS = a[a.index("--lens") + 1].split()
W = int(a[a.index("--windows") + 1]) if "--windows" in a else 3
if W < 1:
    sys.exit("--windows must be >= 1")
failed = {}
if "--status" in a and os.path.exists(a[a.index("--status") + 1]):
    for line in open(a[a.index("--status") + 1]):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and p[2].startswith("FAIL"):
            failed.setdefault(p[0], []).append(f"{p[1]} {p[2]}"[:80])
med, bad = {}, {}
print("| Arm | Length | seq/s per window | median seq/s | Status |\n|---|---|---|---|---|")
for arm in ARMS:
    for t in LENS:
        ds = [load(f"{R}/{arm}-{t}-w{w}.json") for w in range(1, W + 1)]
        oks = [count_ok(d, 256, metric="seq_per_s") for d in ds]
        bad[(arm, t)] = [f"w{i + 1}: {w}" for i, (ok, w) in enumerate(oks) if not ok] + failed.get(arm, [])[:1]
        v = [d["seq_per_s"] for d, (ok, _) in zip(ds, oks) if ok]
        med[(arm, t)] = round(statistics.median(v), 2) if len(v) == W else None
        per = ", ".join(f"{d['seq_per_s']:.1f}" if ok else "invalid" for d, (ok, _) in zip(ds, oks))
        print(f"| {arm} | {t} | {per} | {med[(arm, t)] or '–'} | {'; '.join(bad[(arm, t)]) or 'complete'} |")
print("\n| Arm | vs | Length | Arm seq/s | Ref seq/s | Ratio | Status |\n|---|---|---|---|---|---|---|")
incomplete = []
for arm in ARMS:
    for ref in ("emb-U", "emb-T"):
        if arm == ref or ref not in ARMS:
            continue
        for t in LENS:
            # match files carry n = vectors compared; window 1 stores all 256
            m = match_ok(f"{R}/{arm}.vs.{ref}.{t}.match.json", 256)
            why = [f"{arm} {b}" for b in bad[(arm, t)]] + [f"{ref} {b}" for b in bad[(ref, t)]] + ([m[1]] if not m[0] else [])
            x, r = med[(arm, t)], med[(ref, t)]
            ratio = f"{x / r:.2f}x" if not why and x and r else "–"
            print(f"| {arm} | {ref} | {t} | {x or '–'} | {r or '–'} | {ratio} | {'; '.join(why) or 'complete'} |")
            if why and ref == "emb-U":
                incomplete.append(f"{arm} vs emb-U {t}: " + "; ".join(why))
for x in incomplete:
    print(f"INCOMPLETE (required): {x}")
sys.exit(1 if incomplete else 0)
