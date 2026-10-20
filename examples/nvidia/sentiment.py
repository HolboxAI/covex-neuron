"""Sentiment analysis with a transformers pipeline, as written for an NVIDIA GPU."""

import argparse
import json
import os
import statistics
import time

import torch
from transformers import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
parser.add_argument("--n", type=int, default=1024)
parser.add_argument("--batch-size", type=int, default=32)
args = parser.parse_args()

REVIEWS = [
    "The battery lasts all day and the screen is gorgeous.",
    "Support never answered my emails, worst purchase this year.",
    "It works, I guess, though setup took longer than the manual promised.",
    "Absolutely loved the food, the staff were friendly and fast.",
    "The update broke everything and now the app crashes on launch.",
    "Decent value for the price but the build quality feels cheap.",
    "I would recommend this to anyone who travels a lot for work.",
    "Shipping was late twice and the box arrived damaged.",
]
texts = [f"{REVIEWS[i % len(REVIEWS)]} (review #{i})" for i in range(args.n)]

device = "cuda" if torch.cuda.is_available() else "cpu"

t0 = time.perf_counter()
clf = pipeline("sentiment-analysis", model=args.model, device=device)
load_s = time.perf_counter() - t0

t0 = time.perf_counter()
clf(texts[: args.batch_size], batch_size=args.batch_size)
first_batch_s = time.perf_counter() - t0

clf(texts[0])  # warm-up at batch size 1
latencies = []
for text in texts[:50]:
    t0 = time.perf_counter()
    clf(text)
    latencies.append((time.perf_counter() - t0) * 1000)

t0 = time.perf_counter()
results = clf(texts, batch_size=args.batch_size)
throughput = len(texts) / (time.perf_counter() - t0)

metrics = {
    "model": args.model, "device": device, "items": len(texts),
    "load_s": load_s, "first_batch_s": first_batch_s,
    "warm_latency_ms_p50": statistics.median(latencies),
    "warm_latency_ms_p95": statistics.quantiles(latencies, n=20)[-1],
    "throughput_items_per_s": throughput,
}
out = os.environ.get("SM_MODEL_DIR", "out/sentiment")
os.makedirs(out, exist_ok=True)
json.dump(metrics, open(f"{out}/metrics.json", "w"), indent=2)
json.dump([r["label"] for r in results], open(f"{out}/predictions.json", "w"))
print(json.dumps(metrics, indent=2))
