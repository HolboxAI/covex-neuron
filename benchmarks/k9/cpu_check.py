#!/usr/bin/env python3
"""K9 engine check (CPU mode, small real Qwen3): predicted-output decoding must reproduce plain greedy decoding exactly.
Usage: cpu_check.py <ref_port> <k9_port> <out.json> [--n 6] [--tokens 48] [--refs refs.json]
  With --refs: k9_port 0 = only produce and save the reference; ref_port 0 = only check K9 against the saved file
  (the two servers then never run at the same time; two TP8 CPU-mode servers OOM-killed a worker on m7i.16xlarge).

For each prompt, the reference (fork, no speculation) gives greedy token ids. The K9 server then decodes the same prompt
with: the exact reference as the prediction (full acceptance), the reference with one token changed mid-way (partial
rejection, then re-alignment), another prompt's output (wrong), and no prediction (prompt lookup only). Every output
must equal the reference. Also run all of them concurrently (mixed batch: real drafts and pads in one step). /metrics
must show accepted draft tokens for the perfect case (speculation actually ran). Exit 1 with reasons otherwise."""
import concurrent.futures as cf
import json
import re
import sys

import httpx

PROMPTS = [
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, x):\n",
    "import json\n\ndef load_config(path):\n    with open(path) as f:\n",
    "# Merge two sorted lists into one sorted list\ndef merge(a, b):\n",
    "SELECT name, count(*) FROM users\n",
    "The quick brown fox jumps over the lazy dog. The quick brown fox",
]


def gen(c, port, prompt, tokens, pred=None):
    body = {"model": "m", "prompt": prompt, "max_tokens": tokens, "temperature": 0, "ignore_eos": True, "return_token_ids": True}
    if pred is not None:
        body["vllm_xargs"] = {"covex_prediction_ids": pred}
    r = c.post(f"http://127.0.0.1:{port}/v1/completions", json=body)
    r.raise_for_status()
    ids = r.json()["choices"][0].get("token_ids")
    if not isinstance(ids, list):
        raise ValueError("no token_ids in response")
    return ids


def accepted(c, port):
    t = c.get(f"http://127.0.0.1:{port}/metrics").text
    return sum(float(m.group(1)) for m in re.finditer(r"^vllm:spec_decode_num_accepted_tokens(?:_total)?(?:\{[^}]*\})? ([0-9.eE+-]+)$", t, re.M))


def main(a):
    ref_port, k9_port, out = int(a[1]), int(a[2]), a[3]
    n = int(a[a.index("--n") + 1]) if "--n" in a else 6
    tokens = int(a[a.index("--tokens") + 1]) if "--tokens" in a else 48
    problems, rows = [], []
    refs_file = a[a.index("--refs") + 1] if "--refs" in a else None
    try:
        with httpx.Client(timeout=900) as c:
            prompts = PROMPTS[:n]
            if refs_file and k9_port == 0:          # phase 1: reference server only; save its greedy outputs
                ref = [gen(c, ref_port, p, tokens) for p in prompts]
                json.dump({"prompts": prompts, "tokens": tokens, "ref": ref}, open(refs_file, "w"))
                print(json.dumps({"saved_refs": len(ref)})); return 0
            if refs_file and ref_port == 0:         # phase 2: K9 server only, against the saved reference
                saved = json.load(open(refs_file))
                if saved["prompts"] != prompts or saved["tokens"] != tokens:
                    raise ValueError("saved reference was made for different prompts or token count")
                ref = saved["ref"]
            else:
                ref = [gen(c, ref_port, p, tokens) for p in prompts]
            cases = {}
            for i, p in enumerate(prompts):
                perturbed = list(ref[i]); perturbed[len(perturbed) // 3] = (perturbed[len(perturbed) // 3] + 1) % 1000
                cases[f"p{i}-perfect"] = (p, ref[i], i)
                cases[f"p{i}-perturbed"] = (p, perturbed, i)
                cases[f"p{i}-wrong"] = (p, ref[(i + 1) % len(prompts)], i)
                cases[f"p{i}-absent"] = (p, None, i)
            a0 = accepted(c, k9_port)
            per_case = {}
            for name, (p, pred, i) in cases.items():                       # serial: one case per step
                before = accepted(c, k9_port)
                got = gen(c, k9_port, p, tokens, pred)
                per_case[name] = accepted(c, k9_port) - before
                rows.append({"case": name, "exact": got == ref[i], "accepted_draft_tokens": per_case[name]})
                if got != ref[i]:
                    first = next((k for k, (x, y) in enumerate(zip(got, ref[i])) if x != y), min(len(got), len(ref[i])))
                    problems.append(f"{name}: differs from plain greedy at token {first}")
            a1 = accepted(c, k9_port)
            with cf.ThreadPoolExecutor(8) as ex:                             # mixed batch: drafts and pads together
                futs = {name: ex.submit(gen, c, k9_port, p, tokens, pred) for name, (p, pred, i) in cases.items()}
                for name, f in futs.items():
                    ok = f.result() == ref[cases[name][2]]
                    rows.append({"case": name + "-concurrent", "exact": ok})
                    if not ok:
                        problems.append(f"{name}-concurrent: differs from plain greedy")
            if a1 - a0 <= 0:
                problems.append("no accepted draft tokens: speculation did not run")
            # the aggregate counter also rises from prompt lookup, so prove the CLIENT prediction drove acceptance:
            # each perfect-prediction case must accept at least half its tokens as drafts, and more than the same
            # prompt without a prediction
            for i in range(len(prompts)):
                pf, ab = per_case[f"p{i}-perfect"], per_case[f"p{i}-absent"]
                if pf < 0.5 * tokens or pf <= ab:
                    problems.append(f"p{i}: perfect prediction accepted {pf:g} draft tokens (absent {ab:g}); "
                                    f"client-supplied drafts were not used")
            result = {"pass": not problems, "accepted_draft_tokens_serial": a1 - a0, "accepted_by_case": per_case,
                      "cases": rows, "problems": problems,
                      "tokens": tokens, "n_prompts": len(prompts)}
    except Exception as e:  # noqa: BLE001 - any failure is a failed check, reported
        result = {"pass": False, "problems": [f"{type(e).__name__}: {e}"]}
    json.dump(result, open(out, "w"), indent=1)
    print(json.dumps({"pass": result["pass"], "problems": result["problems"][:5]}))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
