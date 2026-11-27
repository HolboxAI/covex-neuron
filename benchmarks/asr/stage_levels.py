"""Per-level server stage times from bench.sh snapshots (warmup-free). Usage: stage_levels.py <metrics_dir> <ARM> [<ARM>...]
Reads metrics/<ARM>.c<C>.<phase>.{before,after}.prom; prints one markdown row per (arm, concurrency, phase).
Means are per request (per token for inter-token latency), from histogram _sum/_count deltas summed over label sets."""
import glob, os, re, sys

STAGES = [("queue", "vllm:request_queue_time_seconds"), ("prefill", "vllm:request_prefill_time_seconds"),
          ("decode", "vllm:request_decode_time_seconds"), ("TTFT", "vllm:time_to_first_token_seconds"),
          ("ITL", "vllm:inter_token_latency_seconds"), ("e2e", "vllm:e2e_request_latency_seconds")]


def hist(path):
    out = {}
    for line in open(path):
        m = re.match(r"^(vllm:[a-z_]+?)_(sum|count)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line.strip())
        if m:
            out[(m.group(1), m.group(2))] = out.get((m.group(1), m.group(2)), 0.0) + float(m.group(4))
    return out


def mean_ms(name, b, a):
    n = a.get((name, "count"), 0) - b.get((name, "count"), 0)
    return None if n <= 0 else (1000 * (a.get((name, "sum"), 0) - b.get((name, "sum"), 0)) / n, n)


def main(d, arms):
    print("| Arm | Conc | Phase | Requests | " + " | ".join(f"{s} ms" for s, _ in STAGES) + " |")
    print("|---|---|---|---|" + "---|" * len(STAGES))
    for arm in arms:
        files = sorted(glob.glob(os.path.join(d, f"{arm}.c*.*.before.prom")),
                       key=lambda f: (int(re.search(r"\.c(\d+)\.", f).group(1)), f))
        for bf in files:
            c, ph = re.search(r"\.c(\d+)\.([a-z_]+)\.before\.prom$", bf).groups()
            b, a = hist(bf), hist(bf.replace(".before.", ".after."))
            vals, reqs = [], 0
            for _, name in STAGES:
                r = mean_ms(name, b, a)
                vals.append("–" if r is None else f"{r[0]:.2f}")
                if r and name.endswith("e2e_request_latency_seconds"):
                    reqs = int(r[1])
            print(f"| {arm} | {c} | {ph} | {reqs} | " + " | ".join(vals) + " |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
