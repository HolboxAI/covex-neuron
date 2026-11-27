"""Replay an ordered group of greedy completions and diagnose the first differing token."""
import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

import httpx


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, json.dumps({"pass": False, "reason": message}) + "\n")


def save_json(path, data):
    path = Path(path)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
            tmp = Path(f.name)
            json.dump(data, f, allow_nan=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def finish_report(path, report):
    try:
        save_json(path, report)
    except (OSError, ValueError) as exc:
        print(json.dumps({"pass": False, "reason": f"cannot save {path}: {exc}"}), file=sys.stderr)
        return 1
    if report.get("pass") is False:
        print(json.dumps(report), file=sys.stderr)
        return 1
    return 0


def make_prompts(batch, seed):
    if not 1 <= batch <= 8:
        raise ValueError("--batch must be between 1 and the 8 compiled decode slots")
    # Reconstruct Run 1's two failing inputs byte-for-byte, including earlier RNG draws.
    original = random.Random(0)
    known = {}
    for length in (1, 31, 32, 33, 127, 128, 129, 511, 512, 513, 1023, 1024, 1025, 2049,
                   4095, 4096, 4097, 8190):
        ids = [original.randrange(1000, 20000) for _ in range(length)]
        if length in (33, 127, 128, 8190):
            known[length] = ids
    if batch == 1:
        return [known[8190]]
    # Dynamic attention uses the longest context in the batch. Keep the other
    # slots shorter so the 8190-token prompt can cross 8192 during this replay.
    lengths = (33, 127, 128, 8190, 31, 32, 129, 511)
    rng = random.Random(seed)
    return [known[n] if n in known else [rng.randrange(1000, 20000) for _ in range(n)]
            for n in lengths[:batch]]


def is_id(value):
    return type(value) is int and value >= 0


def valid_pairs(pairs, chosen):
    if not isinstance(pairs, list) or len(pairs) != 2:
        raise ValueError("expected two token ID/logprob pairs")
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2 or not is_id(pair[0]):
            raise ValueError("invalid top-2 token ID")
        if type(pair[1]) not in (int, float) or not math.isfinite(pair[1]):
            raise ValueError("non-finite or invalid top-2 logprob")
    scores = dict(pairs)
    if len(scores) != 2 or chosen not in scores or scores[chosen] != max(scores.values()):
        raise ValueError("top-2 tokens disagree with greedy token ID")
    return scores


def unpack_choice(choice, prompt, tokens):
    if choice.get("prompt_token_ids") != prompt:
        raise ValueError("server echoed different prompt token IDs")
    ids = choice["token_ids"]
    rows = choice["logprobs"]["top_logprobs"]
    if not isinstance(ids, list) or len(ids) != tokens or not all(map(is_id, ids)):
        raise ValueError("missing or incomplete generated token IDs")
    if not isinstance(rows, list) or len(rows) != tokens:
        raise ValueError("missing per-token top-2 logprobs")
    top2 = []
    for row, chosen in zip(rows, ids):
        if not isinstance(row, dict) or len(row) != 2:
            raise ValueError("expected exactly two unambiguous top logprobs")
        pairs = []
        for key, lp in row.items():
            if not re.fullmatch(r"token_id:[0-9]+", key):
                raise ValueError("logprob keys must identify token IDs")
            pairs.append([int(key.split(":")[1]), lp])
        valid_pairs(pairs, chosen)
        top2.append(sorted(pairs, key=lambda p: -p[1]))
    digest = hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()
    return {"prompt_sha256": digest, "prompt_len": len(prompt), "token_ids": ids, "top2": top2}


def run(args):
    prompts = make_prompts(args.batch, args.seed)
    if args.tokens < 1:
        raise ValueError("--tokens must be positive")
    # One ordered, native multi-prompt request. The server controls physical decode scheduling.
    body = {"model": args.model, "prompt": prompts, "n": 1, "temperature": 0, "seed": args.seed,
            "max_tokens": args.tokens, "ignore_eos": True, "logprobs": 2,
            "return_token_ids": True, "return_tokens_as_token_ids": True}
    response = httpx.post(args.base_url.rstrip("/") + "/v1/completions", json=body, timeout=300)
    response.raise_for_status()
    choices = response.json()["choices"]
    if not isinstance(choices, list) or len(choices) != len(prompts):
        raise ValueError("incomplete replay batch")
    by_index = {choice["index"]: choice for choice in choices if is_id(choice["index"])}
    if set(by_index) != set(range(len(prompts))):
        raise ValueError("missing or duplicated completion index")
    rows = [unpack_choice(by_index[i], prompt, args.tokens) for i, prompt in enumerate(prompts)]
    return {"prompts": rows, "completed": len(rows)}


def validated_rows(path):
    with open(path) as f:
        data = json.load(f)
    rows = data["prompts"]
    if not isinstance(rows, list) or not rows or type(data["completed"]) is not int or data["completed"] != len(rows):
        raise ValueError(f"{path}: empty or incomplete replay")
    for row in rows:
        ids, top2 = row["token_ids"], row["top2"]
        if not re.fullmatch("[0-9a-f]{64}", row["prompt_sha256"]) or not is_id(row["prompt_len"]) or not row["prompt_len"]:
            raise ValueError(f"{path}: invalid prompt identity")
        if not isinstance(ids, list) or not ids or not all(map(is_id, ids)) or not isinstance(top2, list) or len(top2) != len(ids):
            raise ValueError(f"{path}: missing token IDs or top-2 data")
        for pairs, chosen in zip(top2, ids):
            valid_pairs(pairs, chosen)
    if len({len(row["token_ids"]) for row in rows}) != 1:
        raise ValueError(f"{path}: incomplete generation in batch")
    return rows


def compare(args):
    left, right = validated_rows(args.a), validated_rows(args.b)
    if len(left) != len(right):
        raise ValueError("different prompt counts")
    mismatches = []
    for i, (a, b) in enumerate(zip(left, right)):
        if a["prompt_sha256"] != b["prompt_sha256"] or a["prompt_len"] != b["prompt_len"]:
            raise ValueError(f"different prompt at index {i}")
        if len(a["token_ids"]) != len(b["token_ids"]):
            raise ValueError(f"different number of generated tokens at index {i}")
        for t, (x, y) in enumerate(zip(a["token_ids"], b["token_ids"])):
            if x == y:
                continue
            # A is the reference. Inspect logits only at the first divergent position.
            scores = dict(a["top2"][t])
            gap = Decimal(str(scores[x])) - Decimal(str(scores[y])) if y in scores else None
            mismatches.append({"i": i, "first_diff": t, "a": x, "b": y,
                               "near_tie": gap is not None and gap < Decimal("0.1"),
                               "gap": float(gap) if gap is not None else None})
            break
    return {"pass": not mismatches, "n": len(left), "mismatches": mismatches}


def main():
    parser = JsonArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    r = subs.add_parser("run")
    r.add_argument("--base-url", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--batch", type=int, default=8)
    r.add_argument("--tokens", type=int, default=32)
    r.add_argument("--seed", type=int, default=1)
    c = subs.add_parser("compare", help="compare reference A with candidate B")
    c.add_argument("a")
    c.add_argument("b")
    for sub in (r, c):
        sub.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        result = run(args) if args.action == "run" else compare(args)
        if result.get("pass") is False:
            result["reason"] = "greedy token mismatch"
    except (OSError, ValueError, TypeError, KeyError, AttributeError, httpx.HTTPError) as exc:
        result = {"pass": False, "reason": str(exc) or type(exc).__name__}
        result.update({"n": 0, "mismatches": []} if args.action == "compare" else {"completed": 0, "prompts": []})
    return finish_report(args.out, result)


if __name__ == "__main__":
    sys.exit(main())
