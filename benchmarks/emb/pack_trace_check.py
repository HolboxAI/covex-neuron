#!/usr/bin/env python3
"""Verify from the packed arm's server log that the isolation client's pairs really shared one packed Neuron step.
Usage: pack_trace_check.py <server.log> <from_line> <isolation.json>   (updates isolation.json in place)

Needs COVEX_PACK_TRACE=1 on the packed arm. Every two-input request of the client (4 pair cases x repeat) must appear
as a prefill step with exactly two members of equal real token count and packed=True, and every one-input packed
request (packed_B x repeat) as a one-member step. Sets packed_step_verified and, only when the client checks also
passed, keeps pass=True; otherwise records why. Never sets claimable_speedup (that needs the vector gate too).
"""
import ast
import json
import re
import sys

PAT = re.compile(r"COVEX_PACK_TRACE prefill step: members=(\[.*?\]) total_padded=(\d+) padded_num_reqs=(\d+) packed=(\w+)")


def steps(lines):
    out = []
    for l in lines:
        m = PAT.search(l)
        if m:
            out.append({"members": ast.literal_eval(m.group(1)), "total_padded": int(m.group(2)),
                        "padded_num_reqs": int(m.group(3)), "packed": m.group(4) == "True"})
    return out


def check(log_lines, rep):
    st = steps(log_lines)
    # one line per TP rank: keep rank-0 view by de-duplicating identical consecutive records
    uniq = []
    for s in st:
        key = json.dumps([[m["start"], m["tokens"], m["padded"]] for m in s["members"]]) + str(s["total_padded"])
        if not uniq or uniq[-1][0] != key or len({m["req"] for m in s["members"]} ^ uniq[-1][1]):
            uniq.append((key, {m["req"] for m in s["members"]}, s))
    st = [u[2] for u in uniq]
    repeat = len(next(iter(rep["cases"].values()))["b_sha256"])
    pairs = [s for s in st if len(s["members"]) == 2]
    singles = [s for s in st if len(s["members"]) == 1]
    want_pairs, want_singles = 4 * repeat, 2 * repeat     # stock_B is on the stock arm; packed_B singles here
    why = []
    if len(pairs) != want_pairs:
        why.append(f"{len(pairs)} two-member prefill steps, expected {want_pairs}")
    if any(not s["packed"] for s in pairs + singles):
        why.append("a step did not take the packed attention path")
    if any(len({m["tokens"] for m in s["members"]}) != 1 for s in pairs):
        why.append("pair members have different real token counts")
    if any(s["members"][1]["start"] <= s["members"][0]["start"] for s in pairs):
        why.append("pair members do not occupy two slots")
    if len(singles) < repeat:
        why.append(f"{len(singles)} one-member packed steps, expected >= {repeat}")
    return not why, why, {"pair_steps": len(pairs), "single_steps": len(singles),
                          "pair_layouts": sorted({json.dumps([[m["start"], m["padded"]] for m in s["members"]]) for s in pairs})}


def main(a):
    log, start, path = a[1], int(a[2]), a[3]
    lines = open(log, errors="replace").read().splitlines()[start:]
    rep = json.load(open(path))
    if "cases" not in rep:
        rep.update({"packed_step_verified": False, "trace_reason": "isolation client did not complete"})
    else:
        ok, why, detail = check(lines, rep)
        rep.update({"packed_step_verified": ok, "trace": detail, "trace_reason": "; ".join(why) or "pairs shared one packed step"})
        if not ok:
            rep["pass"] = False
    json.dump(rep, open(path, "w"), sort_keys=True)
    print(json.dumps({k: rep.get(k) for k in ("pass", "packed_step_verified", "trace_reason")}))
    return 0 if rep.get("pass") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
