#!/usr/bin/env python3
"""Exact prompt-token counts for the benchmark clips (sets S2's smallest prefill bucket).
Sends each clip once to /v1/chat/completions (non-stream, same audio_url payload as Dax's client) and reads usage.
Usage: prompt_tokens.py <PORT> <CLIENT_DIR> [N=32]   -> prints min / p50 / p90 / max prompt_tokens."""
import base64, json, statistics, sys, urllib.request
from pathlib import Path
port, client_dir = sys.argv[1], Path(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 32
wavs = sorted((client_dir / "data").rglob("*.wav"))[:n]
if not wavs:
    sys.exit(f"no .wav under {client_dir}/data; run bench.sh once first (it downloads the clips)")
counts = []
for w in wavs:
    uri = "data:audio/wav;base64," + base64.b64encode(w.read_bytes()).decode()
    body = {"model": "qwen3-asr-1.7b", "temperature": 0.0, "max_tokens": 1,
            "messages": [{"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": uri}}]}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    counts.append(json.load(urllib.request.urlopen(req, timeout=120))["usage"]["prompt_tokens"])
counts.sort()
q = lambda p: counts[min(len(counts) - 1, int(p * len(counts)))]
print(json.dumps({"n": len(counts), "min": counts[0], "p50": statistics.median(counts), "p90": q(0.9), "max": counts[-1]}))
