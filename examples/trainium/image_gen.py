"""Text-to-image with diffusers (SD-Turbo, 1 step), the NVIDIA script plus the covex lines."""

import argparse
import json
import os
import statistics
import time

import torch
from diffusers import AutoPipelineForText2Image

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="stabilityai/sd-turbo")
parser.add_argument("--n", type=int, default=8)
parser.add_argument("--size", type=int, default=512)
args = parser.parse_args()

PROMPTS = [
    "a red fox in fresh snow, morning light, photograph",
    "a lighthouse on a cliff during a storm, oil painting",
    "a bowl of ramen on a wooden table, studio lighting",
    "a futuristic city skyline at dusk, neon reflections",
]

import covex_neuron as cn  # covex: 1/2
device = cn.device()  # covex: "neuron" on Trainium

t0 = time.perf_counter()
pipe = AutoPipelineForText2Image.from_pretrained(args.model, torch_dtype=torch.float16).to(device)
pipe = cn.prepare(pipe)  # covex: 2/2 bf16, torch.compile(backend="neuron")
load_s = time.perf_counter() - t0


def generate(i):
    return pipe(PROMPTS[i % len(PROMPTS)], num_inference_steps=1, guidance_scale=0.0,
                height=args.size, width=args.size, generator=torch.Generator("cpu").manual_seed(i)).images[0]


t0 = time.perf_counter()
generate(0)
first_batch_s = time.perf_counter() - t0

out = os.environ.get("SM_MODEL_DIR", "out/image_gen")
os.makedirs(out, exist_ok=True)
latencies = []
for i in range(args.n):
    t0 = time.perf_counter()
    image = generate(i)
    latencies.append((time.perf_counter() - t0) * 1000)
    image.save(f"{out}/image_{i}.png")

metrics = {
    "model": args.model, "device": device, "items": args.n,
    "load_s": load_s, "first_batch_s": first_batch_s,
    "warm_latency_ms_p50": statistics.median(latencies),
    "warm_latency_ms_p95": max(latencies),
    "throughput_items_per_s": 1000 / statistics.mean(latencies),
}
json.dump(metrics, open(f"{out}/metrics.json", "w"), indent=2)
print(json.dumps(metrics, indent=2))
