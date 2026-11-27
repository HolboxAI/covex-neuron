"""Completion and correctness gates for every reported speed ratio. A ratio is printed only when BOTH arms completed
exactly the requested work without failures and (where defined) their outputs agree; otherwise the reason is printed.
Review finding (2026-11-27): the old summarizer printed 2.00x for an arm that completed 1 of 8 requests and matched
0 of 20 greedy answers."""
import json, math, os


def _pos(v):
    return isinstance(v, (int, float)) and math.isfinite(v) and v > 0


def load(path):
    try:
        return json.load(open(path)) if os.path.exists(path) else None
    except (OSError, ValueError):
        return None


def bench_ok(d, n_expected, out_len, in_len=None):
    """vllm bench serve result: every requested prompt completed, none failed, every output token produced, and (with
    in_len) exactly the declared input workload. Missing totals are a failure, never a pass."""
    if d is None:
        return False, "no result"
    n, done, failed, toks = d.get("num_prompts"), d.get("completed"), d.get("failed", 0), d.get("total_output_tokens")
    if n != n_expected:
        return False, f"requested {n}, expected {n_expected}"
    if done != n or failed:
        return False, f"completed {done}/{n}, failed {failed}"
    if toks != n * out_len:
        return False, f"output tokens {toks} != {n}x{out_len}"
    if in_len is not None and d.get("total_input_tokens") != n * in_len:
        return False, f"input tokens {d.get('total_input_tokens')} != {n}x{in_len}"
    bad = [k for k in ("output_throughput", "median_tpot_ms", "median_ttft_ms") if not _pos(d.get(k))]
    if bad:
        return False, f"missing or non-positive {', '.join(bad)}"
    return True, ""


def same_work(a, b):
    """Both arms measured the same workload (prompt count and, when reported, input tokens)."""
    if not a or not b:
        return True, ""   # the per-arm gates already report the missing side
    for k in ("num_prompts", "total_input_tokens"):
        if k in a and k in b and a[k] != b[k]:
            return False, f"unequal work: {k} {a[k]} vs {b[k]}"
    return True, ""


def count_ok(d, n_expected, key="n", metric=None):
    if d is None:
        return False, "no result"
    if d.get(key) != n_expected:
        return False, f"{key}={d.get(key)}, expected {n_expected}"
    if metric and not _pos(d.get(metric)):
        return False, f"missing or non-positive {metric}"
    return True, ""


def asr_level_ok(d, conc, n_expected, phase="transcribe"):
    if d is None:
        return False, "no result"
    lv = [r for r in d.get("phases", {}).get(phase, []) if r.get("concurrency") == conc]
    if len(lv) != 1:
        return False, f"{len(lv)} results for c{conc}"
    r = lv[0]
    if r.get("n_total") != n_expected or r.get("n_success") != n_expected or r.get("n_error"):
        return False, f"success {r.get('n_success')}/{r.get('n_total')} (expected {n_expected}), errors {r.get('n_error')}"
    if not _pos(r.get("audio_seconds_per_s")):
        return False, "missing or non-positive audio_seconds_per_s"
    if not isinstance(r.get("wer_mean"), (int, float)) or not math.isfinite(r["wer_mean"]):
        return False, "missing WER"
    return True, ""


def asr_quality_ok(t, c, conc, tol=0.005, phase="transcribe"):
    """Quality qualification for an ASR speed ratio: COVEX WER within tol (absolute, 0.5 pp) of stock."""
    w = lambda d: next((r.get("wer_mean") for r in (d or {}).get("phases", {}).get(phase, []) if r.get("concurrency") == conc), None)
    wt, wc = w(t), w(c)
    if wt is None or wc is None:
        return False, "WER missing"
    if wc > wt + tol:
        return False, f"quality: WER {100 * wt:.2f}% (T) vs {100 * wc:.2f}% (C)"
    return True, f"WER {100 * wt:.2f}% vs {100 * wc:.2f}%"


def match_ok(path, n_expected):
    """Agreement on exactly the declared sample count (a truthy n is not enough)."""
    d = load(path)
    if d is None:
        return False, "correctness not run"
    if d.get("n") != n_expected:
        return False, f"correctness covered n={d.get('n')}, expected {n_expected}"
    if not d.get("pass"):
        return False, f"outputs differ ({d.get('exact_match', d.get('min_cosine'))}, n={d.get('n')})"
    return True, ""
