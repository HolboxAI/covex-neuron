#!/usr/bin/env python3
"""CPU-mode KV budget check for the K9 engine check. Usage: kv_budget_check.py <server.log> <config.json> <max_len> <seqs> <tp>
Every TP rank must log "CPU mode KV cache budget: ... = X GiB" WITHOUT explicit headroom, and X must cover
max_len x seqs tokens of this model's K+V per rank (bf16), with X <= 8 GiB (16 such workers must fit the host).
Prints a JSON verdict; exit 1 on any failure."""
import json, re, sys

log, cfg, max_len, seqs, tp = sys.argv[1], json.load(open(sys.argv[2])), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
text = open(log, errors="replace").read()
rows = re.findall(r"\((?:Worker_)?TP(\d+)[^)]*\).*?CPU mode KV cache budget: (.*?)= ([0-9.]+) GiB", text)
budgets = {int(r): (detail, float(x)) for r, detail, x in rows}
hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
kvh = max(1, cfg["num_key_value_heads"] // tp)
need = max_len * seqs * 2 * cfg["num_hidden_layers"] * kvh * hd * 2 / 2**30
why = []
if sorted(budgets) != list(range(tp)):
    why.append(f"budget lines for ranks {sorted(budgets)}, expected 0..{tp - 1}")
for r, (detail, x) in budgets.items():
    if "headroom (explicit)" in detail:
        why.append(f"TP{r}: explicit headroom still applied")
    if x < need:
        why.append(f"TP{r}: {x:.2f} GiB < {need:.2f} GiB needed for {max_len} x {seqs} tokens")
    if x > 8:
        why.append(f"TP{r}: {x:.2f} GiB per worker would not fit 16 workers on the host")
out = {"pass": not why, "need_gib_per_rank": round(need, 3), "budget_gib": {r: x for r, (_, x) in budgets.items()}, "problems": why}
print(json.dumps(out))
sys.exit(0 if not why else 1)
