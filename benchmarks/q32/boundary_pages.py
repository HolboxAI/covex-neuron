"""Match a single-request boundary replay to host-side page records from all TP8 ranks."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from replay import save_json


def audit(log_path, report, arm):
    marker = "COVEX_QWEN_BOUNDARY "
    events = defaultdict(dict)
    issues = []
    wanted = {row["request_id"]: key for key, row in report["observations"].items()
              if key.startswith(arm + "/")}
    if len(wanted) != 2 * report["repeats"]:
        issues.append("missing observations or repeated HTTP request IDs")
    first = report["prompt_length"]
    # Output token zero is predicted during prefill.
    positions = range(first, min(first + report["tokens"] - 1, 8222))
    with open(log_path, errors="replace") as stream:
        for line in stream:
            if marker not in line:
                continue
            record = json.loads(line.split(marker, 1)[1])
            req = record["request_id"]
            parents = [i for i in wanted if req == i or req.startswith(i + "-")]
            if not parents:
                continue
            if len(parents) != 1:
                raise ValueError(f"ambiguous request id {req}")
            pos = record["position"]
            if pos not in positions or record["group"] != 0:
                continue
            key = wanted[parents[0]]
            rank = record["tp_rank"]
            existing = events[key].get((rank, pos))
            if existing is not None and existing != record:
                issues.append(f"{key} rank {rank} position {pos}: conflicting observations")
            events[key][rank, pos] = record
    coverage = {}
    for key in wanted.values():
        sample = events[key]
        for rank in range(8):
            missing = [p for p in positions if (rank, p) not in sample]
            if missing:
                issues.append(f"{key} rank {rank}: missing positions {missing}")
        for (rank, pos), record in sample.items():
            page, offset = divmod(pos, record["block_size"])
            physical = record["adjacent_pages"].get(str(page))
            expected = physical * record["block_size"] + offset if physical is not None else None
            if (rank not in range(8) or record["active_rows"] != 1 or record["row_bucket"] != 1 or
                    record["logical_page"] != page or page >= record["valid_pages"] or
                    expected is None or record["observed_write_slot"] != expected):
                issues.append(f"{key} rank {rank} position {pos}: invalid page, slot, or graph batch")
        coverage[key] = len(sample)
    return {"arm": arm, "scope": "MRv2 host page tables and write slots",
            "checked_rank_steps": coverage, "issues": issues, "pass": not issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--arm", choices=("U", "F4"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        report = audit(args.log, json.loads(Path(args.replay).read_text()), args.arm)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {"arm": args.arm, "pass": False, "issues": [str(exc)]}
    save_json(args.out, report)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
