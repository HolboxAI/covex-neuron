"""Validate campaign replay artifacts and pool warm request latencies."""
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "q32"))
import needles
import replay


def qwen_checks(reference, candidate):
    expected = replay.make_prompts(8, 1)
    paths = (reference / "replay.json", reference / "repeat.json", candidate / "replay.json")
    try:
        ids = []
        for path in paths:
            rows = replay.validated_rows(path)
            if len(rows) != len(expected):
                raise ValueError(f"{path}: incomplete replay")
            for prompt, row in zip(expected, rows):
                digest = hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()
                if (row["prompt_sha256"] != digest or row["prompt_len"] != len(prompt)
                        or len(row["token_ids"]) != 32):
                    raise ValueError(f"{path}: wrong replay workload")
            ids.append([r["token_ids"] for r in rows])
        if ids[0] != ids[1]:
            return ["stock replay differs on repeat"]
        if ids[0] != ids[2]:
            return ["fixed-batch greedy replay differs from stock"]
        for folder in (reference, candidate):
            data = json.loads((folder / "needles.json").read_text())
            lengths = (8204, 9644, 24044)
            if [r["len"] for r in data["results"]] != list(lengths):
                raise ValueError(f"{folder}: wrong needle lengths")
            for length, row in zip(lengths, data["results"]):
                answer = needles.record(length)[1]
                if row["expected"] != answer or row["pass"] is not True or row["got"].strip() != answer:
                    return [f"{folder.name}: needle failed at {length}"]
            if data["pass"] is not True:
                return [f"{folder.name}: missing successful needle verdict"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return [f"Qwen correctness: {exc}"]
    return []


def decoder_p90(windows):
    """Pooled TTFT / TPOT milliseconds. Each completed request contributes once."""
    ttft, tpot = [], []
    for obj in windows:
        n = obj["completed"]
        times, intervals, lengths = obj.get("ttfts"), obj.get("itls"), obj.get("output_lens")
        if not all(isinstance(xs, list) and len(xs) == n for xs in (times, intervals, lengths)):
            raise ValueError("incomplete request latency records")
        for first, rest, length in zip(times, intervals, lengths):
            if (not isinstance(rest, list) or not rest or type(length) is not int or length <= 1 or
                    any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in [first, *rest])):
                raise ValueError("invalid request latency records")
            ttft.append(1000 * first)
            tpot.append(1000 * sum(rest) / (length - 1))
    if not ttft:
        raise ValueError("no measured requests")
    index = math.ceil(0.90 * len(ttft)) - 1
    return {"ttft_ms": sorted(ttft)[index], "tpot_ms": sorted(tpot)[index]}
