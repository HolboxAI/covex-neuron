"""Long-prompt probe for the CPU rehearsal: prompts above max_num_batched_tokens (8192) take the segmented / chunked
prefill path, which run 1 never exercised (its longest prompt was 8,190 tokens).
Usage: long_prompt.py <port> <len,len,...> <max_tokens>
Sends one token-id prompt per length (greedy, ignore_eos), requires HTTP 200 and exactly max_tokens token ids each.
Prints a one-line summary on success; exits nonzero with a JSON reason on any failure (never an empty pass)."""
import json, random, sys, time, urllib.error, urllib.request


def main(port, lengths, k):
    rng = random.Random(1)
    done = []
    for n in lengths:
        prompt = [rng.randrange(1000, 20000) for _ in range(n)]
        body = json.dumps({"model": "m", "prompt": prompt, "max_tokens": k, "temperature": 0, "ignore_eos": True,
                           "return_token_ids": True}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body, {"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=2400) as r:
                code, out = r.status, json.load(r)
        except urllib.error.HTTPError as e:
            sys.exit(json.dumps({"pass": False, "len": n, "reason": f"HTTP {e.code}: {e.read()[:200]!r}"}))
        except Exception as e:  # connection refused, timeout, server died
            sys.exit(json.dumps({"pass": False, "len": n, "reason": f"{type(e).__name__}: {e}"}))
        ids = (out.get("choices") or [{}])[0].get("token_ids")
        if code != 200 or not isinstance(ids, list) or len(ids) != k:
            sys.exit(json.dumps({"pass": False, "len": n, "reason": f"HTTP {code}, token_ids {ids!r:.80} (expected {k})"}))
        done.append(f"{n} tok -> {k} ids in {time.time() - t0:.0f}s")
    print("; ".join(done))


if __name__ == "__main__":
    main(int(sys.argv[1]), [int(x) for x in sys.argv[2].split(",")], int(sys.argv[3]))
