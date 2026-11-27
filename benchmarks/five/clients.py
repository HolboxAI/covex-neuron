#!/usr/bin/env python3
"""Small clients for the five-model run. All talk to an OpenAI-compatible server whose model name is "m".
  emb   <port> <texts.jsonl> <n> <conc> <out.json>   embeddings throughput (+ first 20 vectors for the T/C match)
  vl    <port> <vl_dir> <n> <conc> <out.json>        image chat throughput (+ first 20 answers)
  gen   <port> <out.json>                            20 fixed greedy prompts, 64 tokens (T/C output match)
  smoke <gen|emb|vl|asr> <port> [model] [data]   one real request; non-zero exit unless usable (readiness gate)
  match <T.json> <C.json>                            compare saved outputs: exact-text match rate, or embedding cosine
"""
import asyncio, base64, json, os, statistics, sys, time
# PROBE_N / PROBE_TOKENS shrink requests for the CPU-mode rehearsal (CPU decode is ~1 tok/s); unset on Trainium.
PROBE_N = int(os.environ.get("PROBE_N", "0")) or None
PROBE_TOKENS = int(os.environ.get("PROBE_TOKENS", "64"))
from pathlib import Path
import httpx

PROMPTS = [f"Write a short factual paragraph about topic number {i}: " + t for i, t in enumerate(
    ["the water cycle", "binary search", "the French Revolution", "photosynthesis", "TCP handshakes", "black holes",
     "compound interest", "the Pythagorean theorem", "volcanoes", "public-key cryptography", "the immune system",
     "supply and demand", "plate tectonics", "neural networks", "the printing press", "ocean tides", "DNA replication",
     "the speed of light", "hash tables", "the Roman Empire"])]

async def run_pool(n, conc, make):
    sem = asyncio.Semaphore(conc); lat = []; outs = {}
    async with httpx.AsyncClient(timeout=600) as c:
        async def one(i):
            async with sem:
                t = time.perf_counter(); r = await make(c, i); r.raise_for_status(); lat.append(time.perf_counter() - t)
                outs[i] = r.json()
        t0 = time.perf_counter(); await asyncio.gather(*(one(i) for i in range(n))); wall = time.perf_counter() - t0
    return wall, lat, outs

def summary(n, conc, wall, lat, extra):
    return {"n": n, "concurrency": conc, "wall_s": round(wall, 3), "req_per_s": round(n / wall, 3),
            "lat_p50_ms": round(1000 * statistics.median(lat), 1), "lat_p90_ms": round(1000 * sorted(lat)[int(0.9 * len(lat)) - 1], 1), **extra}

def emb(port, path, n, conc, out):
    texts = [json.loads(l)["text"] for l in open(path)]
    make = lambda c, i: c.post(f"http://127.0.0.1:{port}/v1/embeddings", json={"model": "m", "input": texts[i % len(texts)]})
    asyncio.run(run_pool(4, 4, make))  # warm
    wall, lat, outs = asyncio.run(run_pool(n, conc, make))
    vecs = [outs[i]["data"][0]["embedding"] for i in range(min(20, n))]
    json.dump(summary(n, conc, wall, lat, {"seq_per_s": round(n / wall, 3), "vectors": vecs}), open(out, "w"))

def vl(port, d, n, conc, out):
    d = Path(d); man = json.load(open(d / "manifest.json"))
    def make(c, i):
        m = man[i % len(man)]; img = base64.b64encode((d / "images" / m["image"]).read_bytes()).decode()
        content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
                   {"type": "text", "text": m["question"] + "\nAnswer with the option letter only."}]
        return c.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                      json={"model": "m", "messages": [{"role": "user", "content": content}], "max_tokens": PROBE_TOKENS, "temperature": 0})
    asyncio.run(run_pool(2, 2, make))
    wall, lat, outs = asyncio.run(run_pool(n, conc, make))
    toks = sum(o["usage"]["completion_tokens"] for o in outs.values())
    answers = [outs[i]["choices"][0]["message"]["content"] for i in range(min(20, n))]
    json.dump(summary(n, conc, wall, lat, {"out_tok_per_s": round(toks / wall, 2), "answers": answers}), open(out, "w"))

def gen(port, out):
    make = lambda c, i: c.post(f"http://127.0.0.1:{port}/v1/completions",
                               json={"model": "m", "prompt": PROMPTS[i], "max_tokens": PROBE_TOKENS, "temperature": 0})
    n = min(PROBE_N or len(PROMPTS), len(PROMPTS))
    wall, lat, outs = asyncio.run(run_pool(n, 4, make))
    json.dump({"answers": [outs[i]["choices"][0]["text"] for i in range(n)]}, open(out, "w"))

def match(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    if "vectors" in A:
        import math
        cos = [sum(x * y for x, y in zip(u, v)) / (math.sqrt(sum(x * x for x in u)) * math.sqrt(sum(y * y for y in v)))
               for u, v in zip(A["vectors"], B["vectors"])]
        print(json.dumps({"min_cosine": round(min(cos), 6), "pass": min(cos) >= 0.9999}))
    else:
        same = sum(x == y for x, y in zip(A["answers"], B["answers"]))
        print(json.dumps({"exact_match": f"{same}/{len(A['answers'])}", "pass": same == len(A["answers"])}))

def smoke(kind, port, model="m", data=""):
    """One small real request; exits non-zero unless it returns a usable result. The readiness gate after /health."""
    import glob
    base = f"http://127.0.0.1:{port}"
    with httpx.Client(timeout=600) as c:
        if kind == "gen":
            r = c.post(f"{base}/v1/completions", json={"model": model, "prompt": "The capital of France is", "max_tokens": 4, "temperature": 0})
            ok = r.status_code == 200 and isinstance(r.json()["choices"][0]["text"], str)
        elif kind == "emb":
            r = c.post(f"{base}/v1/embeddings", json={"model": model, "input": ["readiness check"]})
            ok = r.status_code == 200 and len(r.json()["data"][0]["embedding"]) > 0
        elif kind == "vl":
            man = json.load(open(os.path.join(data, "manifest.json")))
            img = base64.b64encode(open(os.path.join(data, "images", man[0]["image"]), "rb").read()).decode()
            content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}, {"type": "text", "text": "Describe briefly."}]
            r = c.post(f"{base}/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 4, "temperature": 0})
            ok = r.status_code == 200 and r.json()["choices"][0]["message"] is not None
        elif kind == "asr":
            wav = sorted(glob.glob(os.path.join(data, "**", "*.wav"), recursive=True))[0]
            r = c.post(f"{base}/v1/audio/transcriptions", data={"model": model}, files={"file": (os.path.basename(wav), open(wav, "rb"), "audio/wav")})
            ok = r.status_code == 200 and isinstance(r.json().get("text"), str)
        else:
            raise SystemExit(f"unknown smoke kind {kind}")
    if not ok:
        raise SystemExit(f"smoke {kind} failed: HTTP {r.status_code} {r.text[:200]}")
    print(f"smoke {kind} ok")


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    {"emb": lambda: emb(a[0], a[1], int(a[2]), int(a[3]), a[4]), "vl": lambda: vl(a[0], a[1], int(a[2]), int(a[3]), a[4]),
     "gen": lambda: gen(a[0], a[1]), "match": lambda: match(a[0], a[1]), "smoke": lambda: smoke(*a)}[cmd]()