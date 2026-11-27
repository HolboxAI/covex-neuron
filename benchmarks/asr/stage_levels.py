"""Per-level server stage times from bench.sh snapshots. Usage: stage_levels.py <metrics_dir> <results_dir> <ARM>...
Reads metrics/<ARM>.c<C>.<phase>.{before,after}.prom and results/<ARM>.c<C>.<phase>.json (the client's counts).
A level gets stage numbers only when attribution is sound; otherwise "unavailable: <why>":
  - a scrape is missing or empty (a failed before-scrape would otherwise turn lifetime counters into an "average");
  - any counter went down between scrapes (server restart / counter reset);
  - the e2e request-count delta differs from the client's successful measured requests (warmup or foreign traffic
    inside the window, or metrics not yet published).
Means are per request, except ITL, whose histogram counts tokens."""
import glob, json, os, re, sys

STAGES = [("queue", "vllm:request_queue_time_seconds"), ("prefill", "vllm:request_prefill_time_seconds"),
          ("decode", "vllm:request_decode_time_seconds"), ("TTFT", "vllm:time_to_first_token_seconds"),
          ("ITL", "vllm:inter_token_latency_seconds"), ("e2e", "vllm:e2e_request_latency_seconds")]


def hist(path):
    out = {}
    if not os.path.exists(path):
        return None
    for line in open(path):
        m = re.match(r"^(vllm:[a-z0-9_]+?)_(sum|count)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line.strip())
        if m:
            out[(m.group(1), m.group(2))] = out.get((m.group(1), m.group(2)), 0.0) + float(m.group(4))
    return out or None


def level(b, a, n_success):
    """-> (row values, None) or (None, reason)."""
    if b is None or a is None:
        return None, "missing scrape"
    if any(a.get(k, 0.0) < v for k, v in b.items()):
        return None, "counter reset between scrapes"
    n = a.get(("vllm:e2e_request_latency_seconds", "count"), 0) - b.get(("vllm:e2e_request_latency_seconds", "count"), 0)
    if n_success is None:
        return None, "no client result"
    if int(round(n)) != n_success:
        return None, f"server counted {int(round(n))} requests, client {n_success} successful"
    vals = []
    for _, name in STAGES:
        c = a.get((name, "count"), 0) - b.get((name, "count"), 0)
        vals.append("–" if c <= 0 else f"{1000 * (a.get((name, 'sum'), 0) - b.get((name, 'sum'), 0)) / c:.2f}")
    return (int(round(n)), vals), None


def main(mdir, rdir, arms):
    print("| Arm | Conc | Phase | Requests | " + " | ".join(f"{s} ms" for s, _ in STAGES) + " |")
    print("|---|---|---|---|" + "---|" * len(STAGES))
    for arm in arms:
        files = sorted(glob.glob(os.path.join(mdir, f"{arm}.c*.*.before.prom")),
                       key=lambda f: (int(re.search(r"\.c(\d+)\.", f).group(1)), f))
        for bf in files:
            c, ph = re.search(r"\.c(\d+)\.([a-z_]+)\.before\.prom$", bf).groups()
            rj = os.path.join(rdir, f"{arm}.c{c}.{ph}.json")
            n_ok = None
            if os.path.exists(rj):
                lv = [r for r in json.load(open(rj)).get("phases", {}).get(ph, []) if str(r.get("concurrency")) == c]
                n_ok = lv[0].get("n_success") if lv else None
            got, why = level(hist(bf), hist(bf.replace(".before.", ".after.")), n_ok)
            if why:
                print(f"| {arm} | {c} | {ph} | – | " + " | ".join(["–"] * len(STAGES)) + f" |  unavailable: {why}")
            else:
                print(f"| {arm} | {c} | {ph} | {got[0]} | " + " | ".join(got[1]) + " |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
