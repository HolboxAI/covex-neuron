"""Emotion classification with AutoModelForSequenceClassification, as written for an NVIDIA GPU."""

import argparse
import json
import os
import statistics
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="j-hartmann/emotion-english-distilroberta-base")
parser.add_argument("--n", type=int, default=1024)
parser.add_argument("--batch-size", type=int, default=32)
args = parser.parse_args()

MESSAGES = [
    "I can't believe they cancelled the show, I'm furious.",
    "We finally closed on the house today, best day ever!",
    "Walking alone at night down that street really scares me.",
    "My grandmother passed away last week and I miss her.",
    "Wow, I did not expect the package to arrive a week early.",
    "The meeting is at 3pm in the usual room.",
    "That smell from the fridge is absolutely disgusting.",
    "Thank you so much for helping me move, you're amazing.",
]
texts = [f"{MESSAGES[i % len(MESSAGES)]} #{i}" for i in range(args.n)]

device = "cuda" if torch.cuda.is_available() else "cpu"

t0 = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()
load_s = time.perf_counter() - t0


def classify(batch):
    enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    return [model.config.id2label[i] for i in logits.argmax(-1).tolist()]


t0 = time.perf_counter()
classify(texts[: args.batch_size])
first_batch_s = time.perf_counter() - t0

classify(texts[:1])  # warm-up at batch size 1
latencies = []
for text in texts[:50]:
    t0 = time.perf_counter()
    classify([text])
    latencies.append((time.perf_counter() - t0) * 1000)

t0 = time.perf_counter()
labels = []
for i in range(0, len(texts), args.batch_size):
    labels += classify(texts[i : i + args.batch_size])
throughput = len(texts) / (time.perf_counter() - t0)

metrics = {
    "model": args.model, "device": device, "items": len(texts),
    "load_s": load_s, "first_batch_s": first_batch_s,
    "warm_latency_ms_p50": statistics.median(latencies),
    "warm_latency_ms_p95": statistics.quantiles(latencies, n=20)[-1],
    "throughput_items_per_s": throughput,
}
out = os.environ.get("SM_MODEL_DIR", "out/text_classify")
os.makedirs(out, exist_ok=True)
json.dump(metrics, open(f"{out}/metrics.json", "w"), indent=2)
json.dump(labels, open(f"{out}/predictions.json", "w"))
print(json.dumps(metrics, indent=2))
