#!/usr/bin/env python3
"""Qwen3-32B primary experiment summary. Usage: summarize.py <results> --arms "q32-U q32-T q32-K" --windows 3
Per arm and load point: output tok/s of every unprofiled window, the median, completed/requested, TPOT and TTFT p50/p90
(medians over windows). A ratio vs U or T is printed only when every window of both arms completed fully (gate.py) and
the arm's greedy outputs on the exact-length cache prompts equal the reference at c1 and c3. Exit 1 when a required
comparison (K vs U, K vs T) is incomplete."""
import os, statistics, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "five"))
from gate import bench_ok, load, match_ok, same_work  # noqa: E402

a = sys.argv
R = a[1]; ARMS = a[a.index("--arms") + 1].split(); W = int(a[a.index("--windows") + 1]) if "--windows" in a else 3
if W < 1:
    sys.exit("summarize: --windows must be >= 1 (no measurement windows means no comparison)")
STATUS = a[a.index("--status") + 1] if "--status" in a else None
EXPLORATORY = {"q32-F", "q32-C"}   # dynamic FA: throughput shown even when tokens diverge, labelled, never headline
failed_steps = {}   # arms with a failed run step (ready, provenance, profile, evidence, bench) cannot back a ratio
if STATUS and os.path.exists(STATUS):
    for line in open(STATUS):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and p[2].startswith("FAIL"):
            failed_steps.setdefault(p[0], []).append(f"{p[1]} {p[2]}"[:80])
# workload tag -> (prompts per window, fixed input length or None for a ranged workload); tags are bench file names
WORKLOADS = {"c1": (8, 1024), "c8": (32, 1024), "m5k-c8": (16, None)}   # m5k = 4K-6K input (5120 +-20%), c8
LOAD_TAGS = a[a.index("--workloads") + 1].split() if "--workloads" in a else ["c1", "c8"]
LOADS = tuple((t, WORKLOADS[t][0]) for t in LOAD_TAGS)
REQUIRED = ([tuple(x.split(":")) for x in a[a.index("--required") + 1].split()] if "--required" in a
            else [("q32-K", "q32-U"), ("q32-K", "q32-T")])
RUN2 = "--run2" in a
REPLAY_ARMS = a[a.index("--replay-arms") + 1].split() if "--replay-arms" in a else []   # correctness only, no timing
NEEDLE_LENS = [8204, 9644, 24044]
HARD = ("ready", "provenance", "bench", "import", "replay", "needles", "cache")   # profile/evidence never disqualify   # run 2 eligibility adds the fixed-batch replay and the needles to the correctness gate


def windows(arm, c, n):
    ds = [load(f"{R}/{arm}-{c}-w{w}.json") for w in range(1, W + 1)]
    oks = [bench_ok(d, n, 256, WORKLOADS[c][1]) for d in ds]
    bad = [f"w{i + 1}: {why}" for i, (ok, why) in enumerate(oks) if not ok]
    return ds, bad


LIMITS = {"ttft_p90_s": 2.0, "tpot_p90_ms": 50.0}   # absolute, 1024/256 workloads only (pre-registered, user + Astra)
RELATIVE = 1.10   # longer workloads: <= 1.1x U's own request p90 TTFT and TPOT on the same workload, same run


def p90(xs):
    xs = sorted(xs)
    return xs[max(0, int(-(-0.9 * len(xs) // 1)) - 1)] if xs else None


def request_p90s(ds):
    """p90 TTFT (s) and TPOT (ms) over INDIVIDUAL requests pooled across windows (vllm bench --save-detailed:
    per-request `ttfts`, `itls`, `output_lens`). None when per-request data is absent or incomplete; never a median
    of p90s."""
    import math
    ok = lambda x: isinstance(x, (int, float)) and math.isfinite(x) and x >= 0
    tt, tp = [], []
    for d in ds:
        if not d or not isinstance(d.get("ttfts"), list) or not isinstance(d.get("itls"), list):
            return None, None
        n = d.get("completed"); ol = d.get("output_lens")
        # exactly one valid, finite record per completed request in EVERY window, or the limit is unknown
        if not isinstance(ol, list) or len(d["ttfts"]) != n or len(d["itls"]) != n or len(ol) != n:
            return None, None
        if not all(ok(x) for x in d["ttfts"]) or not all(type(x) is int and x >= 1 for x in ol):
            return None, None
        if not all(isinstance(x, list) and all(ok(y) for y in x) for x in d["itls"]):
            return None, None
        tt += d["ttfts"]
        # TPOT as pinned vLLM computes it (benchmarks/serve.py): time after the first token / (output tokens - 1).
        # Streaming chunks (itls) can carry several tokens, so dividing by the number of intervals overstates TPOT.
        tp += [1000 * sum(x) / (o - 1) for x, o in zip(d["itls"], ol) if o > 1]
    return p90(tt), p90(tp)


def med(ds, k):
    v = [d[k] for d in ds if d and d.get(k) is not None]
    return round(statistics.median(v), 2) if v else None


def needles_ok(arm):
    d = load(f"{R}/{arm}.needle.json")
    if d is None:
        return False, "needles not run"
    res = d.get("results") or []
    if sorted(r.get("len") for r in res) != NEEDLE_LENS:
        return False, f"needles ran lengths {sorted(r.get('len') for r in res)}, expected {NEEDLE_LENS}"
    if any(not isinstance(r.get("got"), str) for r in res):
        return False, "needle tool error at " + ", ".join(str(r.get("len")) for r in res if not isinstance(r.get("got"), str))
    if not d.get("pass") or not all(r.get("pass") for r in res):
        return False, "needles failed at " + ", ".join(str(r.get("len")) for r in res if not r.get("pass"))
    return True, ""


def replay_ok(arm):
    d = load(f"{R}/{arm}.vs.q32-U.replay.match.json")
    if d is None:
        return False, "replay not run"
    if d.get("n") != 8:
        return False, f"replay covered n={d.get('n')}, expected 8"
    if not d.get("pass"):
        mm = d.get("mismatches") or []
        nt = sum(1 for m in mm if m.get("near_tie"))
        return False, f"replay differs on {len(mm)} prompt(s) ({nt} near-tie)"
    return True, ""


def baseline():
    """Reasons U cannot serve as the reference (empty = valid): it must repeat itself exactly on the replay batch and
    answer all three needles. Without that, no ratio and no equivalence claim is printed against it."""
    if not RUN2:
        return []
    why = [] if replay_ok("q32-U.rerun")[0] else ["U repeat replay: " + replay_ok("q32-U.rerun")[1]]
    ok, w = needles_ok("q32-U")
    return why + ([] if ok else ["U " + w])


BASE = baseline()


def correctness(arm):
    """Reasons an arm's tokens are not established equal to U's (empty = equal on every check run)."""
    why = [f"cache c{k}: {m[1]}" for k in (1, 3) for m in [match_ok(f"{R}/{arm}.vs.q32-U.cache.c{k}.match.json", 18)] if not m[0]]
    if RUN2:
        why += [m[1] for m in (replay_ok(arm), needles_ok(arm)) if not m[0]]
    return why


print("## Per arm (unprofiled windows)\n")
print("| Arm | Load | out tok/s per window | median tok/s | completed | TPOT p50 / p90 ms (window medians) | "
      "TTFT p50 / p90 ms (window medians) | request p90 TTFT / TPOT | limits (1K: 2 s / 50 ms; m5k: <= 1.1x U) | Status |")
print("|---|---|---|---|---|---|---|---|---|---|")
summary, limits = {}, {}
for arm in ARMS:
    for c, n in LOADS:
        ds, bad = windows(arm, c, n)
        summary[(arm, c)] = (med(ds, "output_throughput"), bad)
        per = ", ".join(f"{d['output_throughput']:.1f}" if d and isinstance(d.get("output_throughput"), (int, float)) else "invalid"
                        for d in ds)
        done = sum((d or {}).get("completed", 0) for d in ds)
        rt, rp = request_p90s(ds)
        if WORKLOADS[c][1] is None:   # relative limit against U
            ut, up = request_p90s(windows("q32-U", c, n)[0]) if "q32-U" in ARMS else (None, None)
            lim = ("unknown (no per-request data)" if rt is None else "unknown (no U reference)" if ut is None else
                   "PASS" if rt <= RELATIVE * ut and rp <= RELATIVE * up else "FAIL")
        else:
            lim = ("unknown (no per-request data)" if rt is None else
                   "PASS" if rt <= LIMITS["ttft_p90_s"] and rp <= LIMITS["tpot_p90_ms"] else "FAIL")
        limits[(arm, c)] = lim
        rq = "–" if rt is None else f"{rt:.3f} s / {rp:.2f} ms"
        print(f"| {arm} | {c} | {per} | {summary[(arm, c)][0] or '–'} | {done}/{n * W} | {med(ds, 'median_tpot_ms')} / "
              f"{med(ds, 'p90_tpot_ms')} | {med(ds, 'median_ttft_ms')} / {med(ds, 'p90_ttft_ms')} | {rq} | {lim} | "
              f"{'; '.join(bad) or 'complete'} |")

print("\n## Comparisons (median output tok/s; ratio only when complete and correct)\n")
print("| Arm | vs | Load | Arm tok/s | Ref tok/s | Ratio | Status |\n|---|---|---|---|---|---|---|")
incomplete = []
for arm in ARMS:
    for ref in ("q32-U", "q32-T"):
        if arm == ref or ref not in ARMS:
            continue
        corr = [match_ok(f"{R}/{arm}.vs.{ref}.cache.c{k}.match.json", 18) for k in (1, 3)]
        for c, _ in LOADS:
            (x, bx), (r, br) = summary[(arm, c)], summary[(ref, c)]
            work = [same_work(load(f"{R}/{arm}-{c}-w{w}.json"), load(f"{R}/{ref}-{c}-w{w}.json")) for w in range(1, W + 1)]
            steps = [f"{z} step failed: {f}" for z in (arm, ref) for f in failed_steps.get(z, [])
                     if f.split()[0].startswith(HARD)][:2]
            if ref == "q32-U" and BASE:
                steps.append("U baseline invalid: " + "; ".join(BASE))
            hard = [f"{arm} {b}" for b in bx] + [f"{ref} {b}" for b in br] + [w for ok, w in work if not ok][:1] + steps
            corr_why = [f"cache c{k}: {m[1]}" for k, m in zip((1, 3), corr) if not m[0]]
            if RUN2 and ref == "q32-U":
                corr_why += [m[1] for m in (replay_ok(arm), needles_ok(arm)) if not m[0]]
            if arm in EXPLORATORY and not hard and corr_why and x and r:
                # dynamic FA: keep throughput visible, labelled; tokens differ, so no equivalence or headline claim
                print(f"| {arm} | {ref} | {c} | {x} | {r} | {x / r:.2f}x (exploratory) | tokens differ: {'; '.join(corr_why)} |")
                continue
            why = hard + corr_why
            ratio = f"{x / r:.2f}x" if not why and x and r else "–"
            label = "complete" + (" (exploratory arm)" if arm in EXPLORATORY else "")
            print(f"| {arm} | {ref} | {c} | {x or '–'} | {r or '–'} | {ratio} | {'; '.join(why) or label} |")
            # run 2: a token or needle difference is a finding (ratio suppressed, reported); only missing work, a
            # failed step or an invalid U baseline make the run incomplete
            if (hard if RUN2 else why) and (arm, ref) in REQUIRED:
                incomplete.append(f"{arm} vs {ref} {c}: " + "; ".join(hard if RUN2 else why))
# a requested exploratory arm must have complete, valid windows; only its token agreement is optional
for arm in [x for x in ARMS if x in EXPLORATORY]:
    for c, n in LOADS:
        bad = summary[(arm, c)][1]
        if bad or summary[(arm, c)][0] is None:
            incomplete.append(f"{arm} (requested exploratory arm) {c}: " + ("; ".join(bad) or "no valid windows"))
if RUN2:
    print("\n## Correctness vs U (run 2)\n")
    rr = load(f"{R}/q32-U.rerun.vs.q32-U.replay.match.json")
    print(f"U run twice on the same batch: {'identical' if rr and rr.get('pass') else 'NOT identical: replay diffs are not attributable' if rr else 'not run'}\n")
    print("| Arm | cache c1 | cache c3 | replay | needles 8,204 / 9,644 / 24,044 |\n|---|---|---|---|---|")
    if BASE:
        print("**U baseline invalid: " + "; ".join(BASE) + ".** No ratio or equivalence is reported against U; raw tok/s stay visible.\n")
    for arm in ARMS + REPLAY_ARMS:
        cells = [("ok" if m[0] else m[1]) for m in [match_ok(f"{R}/{arm}.vs.q32-U.cache.c{k}.match.json", 18) for k in (1, 3)]] \
            if arm != "q32-U" else ["reference", "reference"]
        cells += ["reference" if arm == "q32-U" else ("ok" if replay_ok(arm)[0] else replay_ok(arm)[1]),
                  ("not run (replay-only arm)" if arm in REPLAY_ARMS else "ok" if needles_ok(arm)[0] else needles_ok(arm)[1])]
        print(f"| {arm} | " + " | ".join(cells) + " |")
    if os.path.exists(f"{R}/runner.tsv"):
        print("\n## Runner and graphs per arm (from server logs)\n\n| Arm | runner | decode context buckets | selected decode graph per workload | KV budget line |\n|---|---|---|---|---|")
        for line in open(f"{R}/runner.tsv"):
            print("| " + " | ".join(line.rstrip("\n").split("\t")) + " |")
print("\n## Best eligible tested point per arm and workload\n")
print("Eligible = no failed run step (provenance, profile-independent steps included), all windows complete, request-level")
print("p90 within the pre-registered limits, and (for non-U arms) exact cache-check agreement with U at c1 and c3"
      + (", replay identical to U and all needles passing" if RUN2 else "") + ".\n")
print("| Arm | Load | median tok/s | Note |\n|---|---|---|---|")
for arm in ARMS:
    hard_fail = [f for f in failed_steps.get(arm, []) if f.split()[0].startswith(HARD)]
    corr_why = [] if arm == "q32-U" else correctness(arm)
    why = (["failed step: " + hard_fail[0]] if hard_fail else []) + (["U baseline invalid: " + "; ".join(BASE)] if BASE else []) + corr_why
    if why:
        print(f"| {arm} | – | – | not eligible: {'; '.join(why)} |"); continue
    for c, _ in LOADS:   # one row per workload: tok/s across different workloads are not comparable
        v, bad = summary[(arm, c)]
        if v and not bad and limits.get((arm, c)) == "PASS":
            print(f"| {arm} | {c} | {v} | tested points only ({', '.join(LOAD_TAGS)}); not a saturation sweep |")
        else:
            print(f"| {arm} | {c} | – | not eligible here: {'; '.join(bad) or 'limit ' + str(limits.get((arm, c)))} |")
for x in incomplete:
    print(f"INCOMPLETE (required): {x}")
missing = [p for p in REQUIRED if p[0] not in ARMS or p[1] not in ARMS]
for p in missing:
    print(f"INCOMPLETE (required): {p[0]} vs {p[1]}: arm not run")
sys.exit(1 if incomplete or missing else 0)
