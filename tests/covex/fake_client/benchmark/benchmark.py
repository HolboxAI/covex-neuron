"""Fake ASR benchmark client (same CLI and JSON schema as the pinned client). Sends warmup*concurrency + samples*rounds
transcription requests, counts failures like the real client, and - like the real client - exits 0 even when every
request failed. Logs each call to $FAKE_EVENTS (if set)."""
import argparse, json, os, urllib.error, urllib.request

p = argparse.ArgumentParser()
for a in ["--base-url", "--model", "--data-dir", "--concurrency", "--phases", "--label", "--output"]:
    p.add_argument(a)
for a in ["--num-samples", "--rounds", "--warmup"]:
    p.add_argument(a, type=int)
x = p.parse_args()
if os.environ.get("FAKE_EVENTS"):
    with open(os.environ["FAKE_EVENTS"], "a") as f:
        f.write(f"CLIENT c={x.concurrency} ph={x.phases} n={x.num_samples} out={os.path.basename(x.output)}\n")


def one():
    try:
        urllib.request.urlopen(urllib.request.Request(x.base_url + "/v1/audio/transcriptions", data=b"x", method="POST"), timeout=30).read()
        return True
    except (urllib.error.URLError, OSError):
        return False


phases, raw = {}, []
for ph in x.phases.split(","):
    phases[ph] = []
    for c in [int(v) for v in x.concurrency.split(",")]:
        for _ in range(x.warmup * c):
            one()
        ok = [one() for _ in range(x.num_samples * x.rounds)]
        n_ok = sum(ok)
        phases[ph].append({"concurrency": c, "n_total": len(ok), "n_success": n_ok, "n_error": len(ok) - n_ok,
                           "audio_seconds_per_s": 1.0 * n_ok, "wer_mean": 0.03})
        raw.append({"phase": ph, "concurrency": c, "ok": ok})
json.dump({"label": x.label, "concurrency_levels": [int(v) for v in x.concurrency.split(",")], "phases": phases,
           "raw_results": raw}, open(x.output, "w"))
