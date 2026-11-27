#!/usr/bin/env python3
"""K9 code-edit benchmark: time to an exactly correct edit, stock vs EAGLE3 vs predicted outputs.

  make    --tokenizer DIR --files F1 F2 ... --out edits.jsonl     build edit tasks from real source files
  run     --base-url URL --tasks edits.jsonl --mode M --conc C --out run.json
          M = none (plain greedy) | predict (vllm_xargs covex_prediction_ids = the original file)
            | wrong (another task's file as the prediction) | absent (no prediction; prompt lookup only)
  compare --ref stock.json --runs a.json b.json ... --out summary.json
  report  summary.json                                              markdown table

Each task: a real file plus a deterministic edit instruction (rename one identifier everywhere, or add a one-line
header comment). Greedy (temperature 0); max_tokens = len(file tokens) + 256. Metrics per run:
  - exact: output token ids identical to the reference (stock greedy) run, per task (required for any speed claim);
  - wall time to completion per task, c-level throughput (correct output tokens / s);
  - accepted tokens per verify pass from the server's /metrics deltas (vllm:spec_decode_num_accepted_tokens,
    vllm:spec_decode_num_drafts), when the server is speculating.
Fails loudly: nonzero exit with a JSON reason, never an empty pass.
"""
import argparse
import concurrent.futures as cf
import json
import random
import re
import sys
import time

INSTR = {
    "rename": "Rename the identifier `{a}` to `{b}` everywhere in this file. Output the complete updated file and nothing else.",
    "header": "Add the line `# Reviewed: {b}` as the very first line of this file. Output the complete updated file and nothing else.",
}


def _fail(out, reason):
    if out:
        json.dump({"pass": False, "reason": reason}, open(out, "w"))
    print(json.dumps({"pass": False, "reason": reason}))
    return 1


def make(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rng = random.Random(args.seed)
    tasks = []
    for i, path in enumerate(args.files):
        text = open(path).read()
        ids = [w for w in set(re.findall(r"\b[a-z_][a-z0-9_]{5,}\b", text)) if text.count(w) >= 3]
        kind = "rename" if ids and i % 2 == 0 else "header"
        a = sorted(ids)[rng.randrange(len(ids))] if kind == "rename" else ""
        b = (a + "_v2") if kind == "rename" else f"task-{i}"
        prompt = f"{INSTR[kind].format(a=a, b=b)}\n\n```\n{text}\n```\n"
        file_ids = tok.encode(text, add_special_tokens=False)
        if not file_ids:
            return _fail(args.out, f"{path}: empty file")
        tasks.append({"id": i, "path": path, "kind": kind, "prompt": prompt, "prediction": file_ids,
                      "max_tokens": len(file_ids) + 256})
    with open(args.out, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"{len(tasks)} tasks -> {args.out}")
    return 0


def _metrics(client, base):
    try:
        text = client.get(base.rstrip("/") + "/metrics").text
    except Exception:
        return None
    vals = {}
    for name in ("vllm:spec_decode_num_accepted_tokens", "vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens"):
        vals[name] = sum(float(m.group(1)) for m in re.finditer(rf"^{re.escape(name)}(?:_total)?(?:{{[^}}]*}})? ([0-9.eE+-]+)$", text, re.M))
    return vals


def run(args):
    import httpx
    tasks = [json.loads(l) for l in open(args.tasks)]
    if not tasks:
        return _fail(args.out, "no tasks")
    base = args.base_url.rstrip("/")

    def one(t):
        body = {"model": args.model, "prompt": t["prompt"], "temperature": 0, "max_tokens": t["max_tokens"],
                "return_token_ids": True, "n": 1}
        if args.mode == "predict":
            body["vllm_xargs"] = {"covex_prediction_ids": t["prediction"]}
        elif args.mode == "wrong":
            other = tasks[(tasks.index(t) + 1) % len(tasks)]
            body["vllm_xargs"] = {"covex_prediction_ids": other["prediction"]}
        t0 = time.perf_counter()
        r = client.post(base + "/v1/completions", json=body)
        dt = time.perf_counter() - t0
        r.raise_for_status()
        ch = r.json()["choices"]
        if len(ch) != 1 or not isinstance(ch[0].get("token_ids"), list):
            raise ValueError(f"task {t['id']}: expected one choice with token_ids")
        return {"id": t["id"], "seconds": dt, "token_ids": ch[0]["token_ids"], "finish": ch[0].get("finish_reason")}

    try:
        with httpx.Client(timeout=args.timeout) as client:
            m0 = _metrics(client, base)
            t0 = time.perf_counter()
            with cf.ThreadPoolExecutor(args.conc) as ex:
                rows = list(ex.map(one, tasks))
            wall = time.perf_counter() - t0
            m1 = _metrics(client, base)
    except Exception as e:  # noqa: BLE001 - any failure is a failed run, reported
        return _fail(args.out, f"{type(e).__name__}: {e}")
    spec = None
    if m0 and m1:
        d = {k: m1[k] - m0[k] for k in m0}
        drafts = d["vllm:spec_decode_num_drafts"]
        spec = {"accepted": d["vllm:spec_decode_num_accepted_tokens"], "drafts": drafts,
                "draft_tokens": d["vllm:spec_decode_num_draft_tokens"],
                "tokens_per_pass": (1 + d["vllm:spec_decode_num_accepted_tokens"] / drafts) if drafts else None}
    out = {"pass": True, "mode": args.mode, "conc": args.conc, "wall_s": wall, "tasks": rows, "spec": spec,
           "output_tokens": sum(len(r["token_ids"]) for r in rows)}
    json.dump(out, open(args.out, "w"))
    print(f"{args.mode} c{args.conc}: {len(rows)} tasks in {wall:.1f}s")
    return 0


def compare(args):
    ref = json.load(open(args.ref))
    if not ref.get("pass"):
        return _fail(args.out, "reference run failed")
    rt = {r["id"]: r["token_ids"] for r in ref["tasks"]}
    rows = []
    for p in args.runs:
        d = json.load(open(p))
        if not d.get("pass"):
            rows.append({"run": p, "pass": False, "reason": d.get("reason")}); continue
        got = {r["id"]: r["token_ids"] for r in d["tasks"]}
        exact = [i for i in rt if got.get(i) == rt[i]]
        rows.append({"run": p, "mode": d["mode"], "conc": d["conc"], "exact": len(exact), "n": len(rt),
                     "wall_s": d["wall_s"], "ref_wall_s": ref["wall_s"],
                     "speedup_vs_ref": (ref["wall_s"] / d["wall_s"]) if len(exact) == len(rt) else None,
                     "tokens_per_pass": (d.get("spec") or {}).get("tokens_per_pass"),
                     "correct_tok_per_s": sum(len(rt[i]) for i in exact) / d["wall_s"]})
    out = {"pass": True, "reference": args.ref, "rows": rows}
    json.dump(out, open(args.out, "w"), indent=1)
    return 0


def report(args):
    d = json.load(open(args.summary))
    print("| run | mode | conc | exact | wall s | vs ref | tokens/verify pass | correct tok/s |\n|---|---|---|---|---|---|---|---|")
    for r in d["rows"]:
        if not r.get("pass", True):
            print(f"| {r['run']} | – | – | FAILED: {r.get('reason')} | | | | |"); continue
        sp = f"{r['speedup_vs_ref']:.2f}x" if r["speedup_vs_ref"] else "– (not exact)"
        tp = f"{r['tokens_per_pass']:.2f}" if r["tokens_per_pass"] else "–"
        print(f"| {r['run']} | {r['mode']} | {r['conc']} | {r['exact']}/{r['n']} | {r['wall_s']:.1f} | {sp} | {tp} | {r['correct_tok_per_s']:.1f} |")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("--tokenizer", required=True); m.add_argument("--files", nargs="+", required=True)
    m.add_argument("--out", required=True); m.add_argument("--seed", type=int, default=1)
    r = sub.add_parser("run"); r.add_argument("--base-url", required=True); r.add_argument("--model", default="m")
    r.add_argument("--tasks", required=True); r.add_argument("--mode", choices=["none", "predict", "wrong", "absent"], required=True)
    r.add_argument("--conc", type=int, default=1); r.add_argument("--out", required=True); r.add_argument("--timeout", type=float, default=1800)
    c = sub.add_parser("compare"); c.add_argument("--ref", required=True); c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--out", required=True)
    p = sub.add_parser("report"); p.add_argument("summary")
    a = ap.parse_args()
    return {"make": make, "run": run, "compare": compare, "report": report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
