"""Repeat the campaign's 8,190-token greedy request on two idle Qwen servers."""

import argparse
import hashlib
import json
import sys

import httpx

from replay import is_id, make_prompts, save_json, unpack_choice


def observe(client, url, model, prompt, tokens, scored):
    body = {"model": model, "prompt": prompt, "temperature": 0, "max_tokens": tokens,
            "ignore_eos": True, "return_token_ids": True}
    if scored:
        body.update(logprobs=2, return_tokens_as_token_ids=True)
    response = client.post(url.rstrip("/") + "/v1/completions", json=body)
    response.raise_for_status()
    result = response.json()
    choices = result["choices"]
    if len(choices) != 1 or choices[0]["index"] != 0:
        raise ValueError("expected exactly one completion for the single request")
    choice = choices[0]
    if choice.get("prompt_token_ids") != prompt:
        raise ValueError("server echoed a different input")
    ids = choice["token_ids"]
    if not isinstance(ids, list) or len(ids) != tokens or not all(map(is_id, ids)):
        raise ValueError("server returned invalid or incomplete token IDs")
    observation = {"request_id": result["id"], "ids": ids}
    if scored:
        observation["top2"] = unpack_choice(choice, prompt, tokens)["top2"]
    return observation


def first_difference(left, right):
    if len(left["ids"]) != len(right["ids"]):
        raise ValueError("completion lengths differ")
    for i, (a, b) in enumerate(zip(left["ids"], right["ids"])):
        if a != b:
            return {"token_index": i, "a": a, "b": b,
                    "a_top2": left.get("top2", [None] * len(left["ids"]))[i],
                    "b_top2": right.get("top2", [None] * len(right["ids"]))[i]}
    return None


def run(client, args):
    prompt = make_prompts(1, 0)[0]
    observations = {}
    errors = {}
    urls = {"U": args.stock, "F4": args.fork}
    for rep in range(args.repeat):
        for arm, url in urls.items():
            for mode in ("plain", "scored"):
                key = f"{arm}/{mode}/{rep}"
                try:
                    observations[key] = observe(client, url, args.model, prompt,
                                                args.tokens, mode == "scored")
                except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as exc:
                    errors[key] = str(exc)

    comparisons = []

    def compare(label, a, b):
        if a in observations and b in observations:
            diff = first_difference(observations[a], observations[b])
            if diff is not None:
                diff["predicting_position"] = len(prompt) - 1 + diff["token_index"]
            comparisons.append({"check": label, "left": a, "right": b, "first_difference": diff})

    for rep in range(args.repeat):
        for mode in ("plain", "scored"):
            compare("stock_vs_fork", f"U/{mode}/{rep}", f"F4/{mode}/{rep}")
        for arm in urls:
            compare("logprob_request_effect", f"{arm}/plain/{rep}", f"{arm}/scored/{rep}")
            if rep:
                for mode in ("plain", "scored"):
                    compare("same_arm_repeat", f"{arm}/{mode}/0", f"{arm}/{mode}/{rep}")
    return {"prompt_length": len(prompt),
            "prompt_sha256": hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest(),
            "tokens": args.tokens, "repeats": args.repeat, "complete": not errors, "errors": errors,
            "observations": observations, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", required=True, help="pinned upstream server URL")
    parser.add_argument("--fork", required=True, help="F4 server URL")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="m")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=32)
    args = parser.parse_args()
    if args.stock.rstrip("/") == args.fork.rstrip("/") or args.repeat < 2 or args.tokens < 9:
        parser.error("use distinct servers, at least two repeats, and at least nine output tokens")
    with httpx.Client(timeout=300) as client:
        report = run(client, args)
    save_json(args.out, report)
    if not report["complete"]:
        return 1
    return 0 if all(x["first_difference"] is None for x in report["comparisons"]) else 2


if __name__ == "__main__":
    sys.exit(main())
