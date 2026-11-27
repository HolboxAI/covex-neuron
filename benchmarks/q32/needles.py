"""Check retrieval from precisely sized Qwen prompts using a local model tokenizer."""
import hashlib
import sys

import httpx

from replay import JsonArgumentParser, finish_report


def record(length):
    key = "archive_" + hashlib.sha256(f"key:{length}".encode()).hexdigest()[:8]
    value = str(100000 + int.from_bytes(hashlib.sha256(f"value:{length}".encode()).digest()[:4], "big") % 900000)
    return key, value


def make_prompt(tokenizer, length):
    key, value = record(length)
    intro = "Look up the requested value in the archive. Return only the number.\n"
    fact = f"\n{key} = {value}\n"
    question = f"\nWhat is {key}?\nAnswer:"
    sentence = " The archive contains ordinary field notes."

    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=False))

    # Keep the record deep in the prompt. Size by the complete tokenization, since
    # separately tokenized fragments can merge at a BPE boundary.
    fixed = count(intro + fact + question)
    if length < fixed:
        raise ValueError(f"length {length} cannot contain the {fixed}-token question and record")
    per_sentence = max(1, count(sentence * 10) / 10)
    before = int((length - fixed) * 0.90 / per_sentence)
    prefix = intro + sentence * before + fact
    remaining = length - count(prefix + question)
    if remaining < 0:
        raise ValueError("record exceeds requested prompt length")
    after = max(0, int(remaining / per_sentence) - 1)
    while after and count(prefix + sentence * after + question) > length:
        after -= 1
    body = prefix + sentence * after
    # Find a short suffix with exactly the remaining number of tokens.
    for filler in (" note", " x", " 7", "."):
        repeats = length - count(body + question)
        for _ in range(8):
            if repeats < 0:
                break
            text = body + filler * repeats + question
            actual = count(text)
            if actual == length:
                return text, value
            repeats += length - actual
    raise ValueError(f"could not construct an exact {length}-token prompt with this tokenizer")


def check(client, url, model, tokenizer, length):
    prompt, expected = make_prompt(tokenizer, length)
    response = client.post(url.rstrip("/") + "/v1/completions", json={
        "model": model, "prompt": prompt, "temperature": 0, "max_tokens": 16,
        "add_special_tokens": False, "n": 1,
    })
    response.raise_for_status()
    result = response.json()
    actual = result["usage"]["prompt_tokens"]
    if type(actual) is not int or actual != length:
        raise ValueError(f"server counted {actual!r} prompt tokens, expected {length}")
    choices = result["choices"]
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0].get("text"), str):
        raise ValueError("expected exactly one text completion")
    got = choices[0]["text"]
    passed = got.strip() == expected
    row = {"len": length, "pass": passed, "expected": expected, "got": got}
    if not passed:
        row["reason"] = "answer differs from the stored value"
    return row


def main():
    parser = JsonArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--lengths", default="8204,9644,24044")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    try:
        lengths = [int(n) for n in args.lengths.split(",")]
        if any(n <= 0 for n in lengths) or len(set(lengths)) != len(lengths):
            raise ValueError("lengths must be distinct positive integers")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        with httpx.Client(timeout=300) as client:
            for length in lengths:
                try:
                    row = check(client, args.base_url, args.model, tokenizer, length)
                except (httpx.HTTPError, ValueError, TypeError, KeyError, AttributeError) as exc:
                    row = {"len": length, "pass": False, "expected": record(length)[1], "got": None,
                           "reason": str(exc) or type(exc).__name__}
                rows.append(row)
        report = {"pass": bool(rows) and all(row["pass"] for row in rows), "results": rows}
        if not report["pass"]:
            report["reason"] = "one or more needle checks failed"
    except (OSError, ImportError, RuntimeError, ValueError, TypeError) as exc:
        report = {"pass": False, "results": rows, "reason": str(exc) or type(exc).__name__}
    return finish_report(args.out, report)


if __name__ == "__main__":
    sys.exit(main())
