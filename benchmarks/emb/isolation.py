#!/usr/bin/env python3
"""Diagnose packed-embedding cross-request leakage and single-request drift.

This is a diagnostic, not a replacement for the pre-registered cosine gate.
Native embedding batches make A and B part of one HTTP request; confirm from
server logs or a trace that they actually shared a packed Neuron prefill.
"""

import argparse
import hashlib
import json
import math
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else
                          json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def select_texts(dataset, tokenizer):
    groups = defaultdict(list)
    for line in Path(dataset).read_text().splitlines():
        if not line.strip():
            continue
        text = json.loads(line)["text"]
        if not isinstance(text, str) or not text:
            continue
        length = len(tokenizer.encode(text, add_special_tokens=False))
        if text in groups[length]:
            continue
        groups[length].append(text)
        if len(groups[length]) == 3:
            return {"A": groups[length][0], "A_prime": groups[length][1], "B": groups[length][2]}, length
    raise ValueError("dataset has no three distinct prompts with the same token length")


def post_json(base_url, payload):
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/embeddings",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"HTTP {exc.code}: {exc.read(512).decode(errors='replace')}") from exc


def embeddings(base_url, model, prompts):
    response = post_json(base_url, {"model": model, "input": prompts, "encoding_format": "float"})
    data = response.get("data") if isinstance(response, dict) else None
    if not isinstance(data, list) or len(data) != len(prompts):
        raise ValueError(f"expected {len(prompts)} embeddings, got {len(data) if isinstance(data, list) else 'none'}")
    ordered = {}
    for item in data:
        if not isinstance(item, dict) or type(item.get("index")) is not int:
            raise ValueError("embedding index missing")
        index, vector = item["index"], item.get("embedding")
        if index in ordered or index < 0 or index >= len(prompts):
            raise ValueError(f"duplicate or invalid embedding index {index}")
        if (not isinstance(vector, list) or not vector or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in vector)):
            raise ValueError(f"embedding {index} has non-finite or missing values")
        ordered[index] = vector
    if len({len(vector) for vector in ordered.values()}) != 1:
        raise ValueError("embedding dimensions differ within one batch")
    return [ordered[i] for i in range(len(prompts))]


def compare(a, b):
    if len(a) != len(b):
        raise ValueError(f"embedding dimension differs: {len(a)} vs {len(b)}")
    na, nb = math.sqrt(math.fsum(x * x for x in a)), math.sqrt(math.fsum(x * x for x in b))
    if na == 0 or nb == 0:
        raise ValueError("zero-norm embedding")
    return {"exact": a == b,
            "cosine": math.fsum(x * y for x, y in zip(a, b)) / (na * nb),
            "max_abs_delta": max(abs(x - y) for x, y in zip(a, b)),
            "norm_rel_delta": abs(na - nb) / na}


def run(args, tokenizer):
    if args.repeat < 2:
        raise ValueError("--repeat must be at least 2 to establish repeatability")
    texts, token_length = select_texts(args.dataset, tokenizer)
    cases = {
        "stock_B": (args.stock_url, ["B"]),
        "packed_B": (args.packed_url, ["B"]),
        "packed_AB": (args.packed_url, ["A", "B"]),
        "packed_AprimeB": (args.packed_url, ["A_prime", "B"]),
        "packed_BA": (args.packed_url, ["B", "A"]),
        "packed_BAprime": (args.packed_url, ["B", "A_prime"]),
    }
    outputs = {}
    dimension = None
    for name, (url, order) in cases.items():
        vectors = []
        for _ in range(args.repeat):
            batch = embeddings(url, args.model, [texts[key] for key in order])
            vector = batch[order.index("B")]
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ValueError(f"{name}: dimension {len(vector)} differs from {dimension}")
            vectors.append(vector)
        outputs[name] = vectors
    repeat = {name: [compare(vectors[0], vector) for vector in vectors[1:]]
              for name, vectors in outputs.items()}
    checks = {
        "packed_single_vs_stock": compare(outputs["stock_B"][0], outputs["packed_B"][0]),
        "peer_changed_B_second": compare(outputs["packed_AB"][0], outputs["packed_AprimeB"][0]),
        "peer_changed_B_first": compare(outputs["packed_BA"][0], outputs["packed_BAprime"][0]),
        "B_position_swapped": compare(outputs["packed_AB"][0], outputs["packed_BA"][0]),
        "B_alone_vs_pair": compare(outputs["packed_B"][0], outputs["packed_AB"][0]),
    }
    stable = all(item["exact"] for rows in repeat.values() for item in rows)
    invariant = checks["peer_changed_B_second"]["exact"] and checks["peer_changed_B_first"]["exact"]
    result = {"pass": stable and invariant, "repeat_stable": stable, "peer_invariant": invariant,
              "claimable_speedup": False, "packed_step_verified": False,
              "reason": ("client checks passed; verify packed-step membership in the server trace" if stable and invariant
                         else "identical-batch repeat changed" if not stable else "B changed when only its peer changed"),
              "token_length": token_length, "dimension": dimension,
              "prompt_sha256": {key: digest(value) for key, value in texts.items()},
              "cases": {name: {"order": order, "b_sha256": [digest(v) for v in outputs[name]],
                                "repeat": repeat[name]} for name, (_, order) in cases.items()},
              "checks": checks}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-url", required=True)
    parser.add_argument("--packed-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
        result = run(args, tokenizer)
    except (OSError, ValueError, TypeError, KeyError, ImportError, urllib.error.URLError) as exc:
        result = {"pass": False, "claimable_speedup": False, "reason": str(exc) or type(exc).__name__}
    try:
        Path(args.out).write_text(json.dumps(result, sort_keys=True) + "\n")
    except OSError as exc:
        print(json.dumps({"pass": False, "claimable_speedup": False,
                          "reason": f"cannot write {args.out}: {exc}"}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
