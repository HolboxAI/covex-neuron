#!/usr/bin/env python3
"""Compare arms from Dax-client result JSONs (phases.transcribe + phases.chat_stream).
Prints audio-s/s per server and per chip, delta vs a baseline arm, E2E p50, TTFT/ITL p50 (streaming phase), WER.
Usage: compare.py <results_dir> [baseline=S2]     Chips per arm: TP/4 at LNC2 (TP8 = 2 chips, TP4 = 1 chip)."""
import glob, json, os, sys
d = sys.argv[1]; base = sys.argv[2] if len(sys.argv) > 2 else "S2"
CHIPS = {"T4": 1}  # arms not listed are TP8 = 2 chips; non-Trainium files (cuda*) are reported per server only
def load(path):
    j = json.load(open(path)); ph = j.get("phases", {})
    tr = {r["concurrency"]: r for r in ph.get("transcribe", [])}
    cs = {r["concurrency"]: r for r in ph.get("chat_stream", [])}
    return tr, cs
arms = {os.path.basename(f)[:-5]: load(f) for f in sorted(glob.glob(os.path.join(d, "*.json")))}
ms = lambda v: f"{v * 1000:.0f}" if isinstance(v, (int, float)) else "–"
print(f"| Arm | Conc | audio-s/s | per chip | vs {base} | E2E p50 ms | TTFT p50 ms | ITL p50 ms | WER % |")
print("|---|---|---|---|---|---|---|---|---|")
for arm, (tr, cs) in arms.items():
    chips = None if arm.startswith("cuda") else CHIPS.get(arm.split("-")[0], 2)
    for c in sorted(tr):
        t = tr[c].get("audio_seconds_per_s"); b = arms.get(base, ({}, {}))[0].get(c, {}).get("audio_seconds_per_s")
        delta = f"{(t / b - 1) * 100:+.1f}%" if t and b else "–"
        per_chip = f"{t / chips:.1f}" if chips and t else "–"
        s = cs.get(c, {})
        itl = s.get("itl_p50_s")
        print(f"| {arm} | {c} | {t:.1f} | {per_chip} | {delta} | {ms(tr[c].get('e2e_p50_s'))} | {ms(s.get('ttft_p50_s'))} | "
              f"{(f'{itl*1000:.2f}' if isinstance(itl,(int,float)) else '–')} | {tr[c].get('wer_mean', float('nan')) * 100:.2f} |")
