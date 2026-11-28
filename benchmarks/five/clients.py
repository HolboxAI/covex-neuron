#!/usr/bin/env python3
"""Small clients for the five-model run. All talk to an OpenAI-compatible server whose model name is "m".
  emb   <port> <texts.jsonl> <n> <conc> <out.json>   embeddings throughput (+ first 20 vectors for the T/C match)
  vl    <port> <vl_dir> <n> <conc> <out.json>        image chat throughput (+ first 20 answers)
  gen   <port> <out.json>                            20 fixed greedy prompts, 64 tokens (T/C output match)
  smoke <gen|emb|vl|asr> <port> [model] [data]   one real request; non-zero exit unless usable (readiness gate)
  cachecheck <port> <out.json> [conc] [gen_tokens]  exact-length token-id prompts across cache/bucket/tile boundaries
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
            "latencies_s": lat, "lat_p50_ms": round(1000 * statistics.median(lat), 1), "lat_p90_ms": round(1000 * sorted(lat)[int(0.9 * len(lat)) - 1], 1), **extra}

def emb(port, path, n, conc, out):
    texts = [json.loads(l)["text"] for l in open(path)]
    make = lambda c, i: c.post(f"http://127.0.0.1:{port}/v1/embeddings", json={"model": "m", "input": texts[i % len(texts)]})
    asyncio.run(run_pool(4, 4, make))  # warm
    wall, lat, outs = asyncio.run(run_pool(n, conc, make))
    keep = n if os.environ.get("EMB_ALL_VECTORS") == "1" else min(20, n)   # sweep checks cosine on EVERY vector
    vecs = [outs[i]["data"][0]["embedding"] for i in range(keep)]
    import hashlib   # prompt identity per vector, so a match compares the same prompt's vectors
    ids = [hashlib.sha256(texts[i % len(texts)].encode()).hexdigest()[:16] for i in range(keep)]
    json.dump(summary(n, conc, wall, lat, {"seq_per_s": round(n / wall, 3), "vectors": vecs, "input_ids": ids}),
              open(out, "w"))

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
    token_counts = [outs[i]["usage"]["completion_tokens"] for i in range(n)]
    prompt_tokens = [outs[i]["usage"].get("prompt_tokens") for i in range(n)]   # exact text+image tokens (padding report)
    toks = sum(token_counts)
    # VL_ALL_ANSWERS=1 (vl8 lane): every answer, in request order, so each window is checked in full; default keeps
    # the first 20 that the six-model scorecard expects
    keep = n if os.environ.get("VL_ALL_ANSWERS") == "1" else min(20, n)
    answers = [outs[i]["choices"][0]["message"]["content"] for i in range(keep)]
    json.dump(summary(n, conc, wall, lat, {"out_tok_per_s": round(toks / wall, 2), "answers": answers,
               "completion_tokens": token_counts, "prompt_tokens": prompt_tokens}), open(out, "w"))

def vlstage(port, d, n, out):
    """Serial (c1) stage split for one VL arm: per image, the full request (image + question, max_tokens 1) and the same
    question without the image (max_tokens 1); image cost = full - text-only (vision encoder + image-token prefill),
    and the full answer request gives the decode tail. Medians over n images; diagnostic, not a scored window."""
    d = Path(d); man = json.load(open(d / "manifest.json"))
    def body(m, with_image, max_tokens):
        content = [{"type": "text", "text": m["question"] + "\nAnswer with the option letter only."}]
        if with_image:
            img = base64.b64encode((d / "images" / m["image"]).read_bytes()).decode()
            content.insert(0, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
        return {"model": "m", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0}
    rows = []
    with httpx.Client(timeout=600) as c:
        for i in range(n):
            m = man[i % len(man)]
            t = {}
            for k, (img, mt) in {"text_only_1tok": (False, 1), "image_1tok": (True, 1), "image_full": (True, PROBE_TOKENS)}.items():
                t0 = time.perf_counter(); r = c.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body(m, img, mt))
                r.raise_for_status(); t[k] = time.perf_counter() - t0
            rows.append(t)
    med = {k: statistics.median(r[k] for r in rows) for k in rows[0]}
    full = med["image_full"]
    out_obj = {"n": n, "median_s": med, "image_cost_s": med["image_1tok"] - med["text_only_1tok"],
               "decode_tail_s": full - med["image_1tok"],
               "image_share_of_request": (med["image_1tok"] - med["text_only_1tok"]) / full if full > 0 else None,
               "max_latency_gain_if_image_cost_zero": full / (full - (med["image_1tok"] - med["text_only_1tok"]))
               if full > (med["image_1tok"] - med["text_only_1tok"]) else None}
    json.dump(out_obj, open(out, "w"), indent=1)

def gen(port, out):
    make = lambda c, i: c.post(f"http://127.0.0.1:{port}/v1/completions",
                               json={"model": "m", "prompt": PROMPTS[i], "max_tokens": PROBE_TOKENS, "temperature": 0})
    n = min(PROBE_N or len(PROMPTS), len(PROMPTS))
    wall, lat, outs = asyncio.run(run_pool(n, 4, make))
    json.dump({"answers": [outs[i]["choices"][0]["text"] for i in range(n)]}, open(out, "w"))

# Exact prompt lengths (token ids, no tokenizer) around KV-cache block, prefill bucket and FA-tile boundaries. 8190 + 32
# generated tokens crosses 8192 during decode (the dynamic-FA threshold). Ids 1000..19999: ordinary BPE tokens.
CACHE_LENGTHS = [1, 31, 32, 33, 127, 128, 129, 511, 512, 513, 1023, 1024, 1025, 2049, 4095, 4096, 4097, 8190]

def cache_prompts():
    import random
    rng = random.Random(0)
    return [[rng.randrange(1000, 20000) for _ in range(n)] for n in CACHE_LENGTHS]


def cachecheck(port, out, conc="1", gen_tokens="32"):
    """Greedy output TOKEN IDS for exact-length prompts (return_token_ids -> choices[0].token_ids), so two arms that
    decode to the same text through different ids still differ. conc 3 pads decode batches to the 4-bucket.
    Fails loudly unless every prompt returns exactly gen_tokens ids."""
    prompts = cache_prompts()
    g = int(gen_tokens)
    make = lambda c, i: c.post(f"http://127.0.0.1:{port}/v1/completions",
                               json={"model": "m", "prompt": prompts[i], "max_tokens": g, "temperature": 0,
                                     "ignore_eos": True, "return_token_ids": True})
    n = min(PROBE_N or len(prompts), len(prompts))
    wall, lat, outs = asyncio.run(run_pool(n, int(conc), make))
    ids = []
    for i in range(n):
        ch = outs[i]["choices"][0]
        t, pt = ch.get("token_ids"), ch.get("prompt_token_ids", outs[i].get("prompt_token_ids"))
        if not isinstance(t, list) or len(t) != g or not all(isinstance(x, int) for x in t):
            raise SystemExit(f"cachecheck: prompt {i} (len {CACHE_LENGTHS[i]}) returned {t!r:.80} - expected {g} token ids")
        if pt != prompts[i]:   # the server must have seen exactly the deterministic prompt (length and ids)
            raise SystemExit(f"cachecheck: prompt {i}: server prompt_token_ids differ from the sent {CACHE_LENGTHS[i]} ids "
                             f"(got {len(pt) if isinstance(pt, list) else pt!r})")
        ids.append(t)
    import hashlib
    digest = hashlib.sha256(json.dumps(prompts[:n]).encode()).hexdigest()
    json.dump({"lengths": CACHE_LENGTHS[:n], "concurrency": int(conc), "gen_tokens": g, "prompt_sha256": digest,
               "token_ids": ids}, open(out, "w"))


def vector_match(A, B, cos_min=None, norm_tol=1e-3):
    """Embedding agreement. Cosine alone accepts a truncated or rescaled vector (Astra), so every vector must also have
    the reference's exact dimension, only finite values, a norm within norm_tol (relative), and the same prompt."""
    import math
    thr = float(os.environ.get("EMB_COS_MIN", "0.9999")) if cos_min is None else cos_min
    va, vb = A.get("vectors", []), B.get("vectors", [])
    why = []
    if not va or len(va) != len(vb):
        why.append(f"vector count {len(va)} vs {len(vb)}")
    if A.get("input_ids") is not None or B.get("input_ids") is not None:
        if A.get("input_ids") != B.get("input_ids"):
            why.append("prompt identity differs")
    cos_vals, worst_norm = [], 0.0
    for i, (u, v) in enumerate(zip(va, vb)):
        if len(u) != len(v):
            why.append(f"vector {i}: dimension {len(v)} vs {len(u)}"); break
        if not all(math.isfinite(x) for x in u) or not all(math.isfinite(x) for x in v):
            why.append(f"vector {i}: non-finite values"); break
        nu, nv = math.sqrt(sum(x * x for x in u)), math.sqrt(sum(x * x for x in v))
        if nu == 0 or nv == 0:
            why.append(f"vector {i}: zero norm"); break
        worst_norm = max(worst_norm, abs(nv - nu) / nu)
        cos_vals.append(sum(x * y for x, y in zip(u, v)) / (nu * nv))
    if worst_norm > norm_tol:
        why.append(f"norm differs by {worst_norm:.2e} (> {norm_tol})")
    if cos_vals and min(cos_vals) < thr:
        why.append(f"min cosine {min(cos_vals):.6f} < {thr}")
    return {"min_cosine": round(min(cos_vals), 6) if cos_vals else None, "max_norm_rel_diff": round(worst_norm, 6),
            "n": len(cos_vals), "pass": not why, "why": "; ".join(why)}


def match(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    if "vectors" in A:
        print(json.dumps(vector_match(A, B)))
    elif "token_ids" in A or "token_ids" in B:
        # exact token-id agreement; both sides must hold every prompt with exactly gen_tokens ids
        a_, b_, g = A.get("token_ids", []), B.get("token_ids", []), A.get("gen_tokens")
        full = lambda L: len(L) > 0 and all(isinstance(x, list) and len(x) == g for x in L)
        same = sum(x == y for x, y in zip(a_, b_))
        ok = (len(a_) == len(b_) and full(a_) and full(b_) and g == B.get("gen_tokens") and
              A.get("lengths") == B.get("lengths") and A.get("prompt_sha256") == B.get("prompt_sha256") and same == len(a_))
        first = next(([i, next(j for j, (p, q) in enumerate(zip(x, y)) if p != q)] for i, (x, y) in enumerate(zip(a_, b_))
                      if x != y), None)   # [prompt, first differing generated position]
        print(json.dumps({"exact_match": f"{same}/{len(a_)}", "n": len(a_), "gen_tokens": g, "first_diff": first, "pass": ok}))
    else:
        a_, b_ = A.get("answers", []), B.get("answers", [])
        same = sum(x == y for x, y in zip(a_, b_))
        # Pass only on equal, non-empty answer lists with every answer non-empty (0/0 must not count as a match).
        ok = len(a_) == len(b_) > 0 and all(isinstance(x, str) and x.strip() for x in a_ + b_) and same == len(a_)
        print(json.dumps({"exact_match": f"{same}/{len(a_)}", "n": len(a_), "pass": ok}))

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
     "vlstage": lambda: vlstage(a[0], a[1], int(a[2]), a[3]),
     "gen": lambda: gen(a[0], a[1]), "match": lambda: match(a[0], a[1]), "smoke": lambda: smoke(*a), "cachecheck": lambda: cachecheck(*a)}[cmd]()
