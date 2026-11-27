#!/usr/bin/env python3
"""Compare arms from Dax-client result JSONs (phases.transcribe + phases.chat_stream).
Files named <ARM>-r<k>.json are repeats: the table shows the median across repeats and the min-max spread.
<ARM>-wer.json (full test-clean) supplies the WER column; otherwise the per-run WER is used.
Usage: compare.py <results_dir> [baseline=S2]     Chips per arm: TP/4 at LNC2 (TP8 = 2 chips, TP4 = 1 chip)."""
import glob, json, os, re, statistics, sys
from collections import defaultdict
d = sys.argv[1]; base = sys.argv[2] if len(sys.argv) > 2 else "S2"
CHIPS = {"T4": 1}
runs = defaultdict(list); wer_full = {}
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    name = os.path.basename(f)[:-5]; j = json.load(open(f)); ph = j.get("phases", {})
    if name.endswith("-wer"):
        tr = ph.get("transcribe", [])
        if tr: wer_full[name[:-4]] = tr[0].get("wer_mean"); continue
    runs[re.sub(r"-r\d+$", "", name)].append(({r["concurrency"]: r for r in ph.get("transcribe", [])},
                                               {r["concurrency"]: r for r in ph.get("chat_stream", [])}))
def med(vals):
    v = [x for x in vals if isinstance(x, (int, float))]
    return statistics.median(v) if v else None
def agg(arm, c):
    reps = runs.get(arm, [])
    t = [r[0].get(c, {}).get("audio_seconds_per_s") for r in reps]
    return {"tput": med(t), "lo": min([x for x in t if x] or [0]), "hi": max([x for x in t if x] or [0]), "n": len([x for x in t if x]),
            "e2e": med([r[0].get(c, {}).get("e2e_p50_s") for r in reps]),
            "ttft": med([r[1].get(c, {}).get("ttft_p50_s") for r in reps]),
            "itl": med([r[1].get(c, {}).get("itl_p50_s") for r in reps]),
            "wer": med([r[0].get(c, {}).get("wer_mean") for r in reps])}
fmt = lambda v, k=1000, p=0: f"{v * k:.{p}f}" if isinstance(v, (int, float)) else "–"
print(f"| Arm | Conc | audio-s/s (median, min–max, n) | per chip | vs {base} | E2E p50 ms | TTFT p50 ms | ITL p50 ms | WER % |")
print("|---|---|---|---|---|---|---|---|---|")
for arm in sorted(runs):
    chips = None if arm.startswith("cuda") else CHIPS.get(arm, 2)
    concs = sorted({c for r in runs[arm] for c in r[0]})
    for c in concs:
        a = agg(arm, c); b = agg(base, c)["tput"] if base in runs else None
        delta = f"{(a['tput'] / b - 1) * 100:+.1f}%" if a["tput"] and b else "–"
        per_chip = f"{a['tput'] / chips:.1f}" if chips and a["tput"] else "–"
        wer = wer_full.get(arm, a["wer"])
        print(f"| {arm} | {c} | {fmt(a['tput'],1,1)} ({a['lo']:.1f}–{a['hi']:.1f}, n={a['n']}) | {per_chip} | {delta} | "
              f"{fmt(a['e2e'])} | {fmt(a['ttft'])} | {fmt(a['itl'],1000,2)} | {fmt(wer,100,2)}{' (full)' if arm in wer_full else ''} |")
