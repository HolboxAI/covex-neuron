"""KV-cache aliasing check on a neuronx compile log.
Usage: alias_check.py <server.log> <config.json> <lanes> <tp> [--ref <stock server.log>]

Checks, for every compiled graph (lane) on every rank:
  1. present: all lanes x ranks logged a "Mutated inputs" line (a crashed or partial compile fails);
  2. SPMD-consistent: every rank of a lane mutates the same input list;
  3. count: >= 1 + 2*num_hidden_layers mutated inputs (hidden state + every layer's K and V cache);
  4. structural parity: with --ref (the stock arm's log), the ordinal list must equal stock's for that lane. Ordinals
     are local to each graph signature and the log carries no buffer names, so this is structure, not named K/V
     identity; without a reference the result is UNVERIFIED, never PASS. Conflicting duplicate records FAIL.
Stock Qwen3-32B TP8: 129 = 1 + 2*64 inputs, identical on all 8 ranks of all 6 lanes. The squeeze-view bug left
decode lanes 2-5 with [1]. Prints "<PASS|FAIL|UNVERIFIED>\\t<detail>"; exit 0 PASS, 1 FAIL, 2 UNVERIFIED."""
import json, re, sys


class Conflict(ValueError):
    pass


def parse(text):
    """lane/rank -> mutated-input ordinals. Identical repeats are fine; conflicting repeats raise Conflict."""
    out = {}
    for line in text.splitlines():
        m = re.search(r"\[trace lane=(\d+) rank=(\d+)\] Mutated inputs: \[([^\]]*)\]", line)
        if m:
            k = (int(m.group(1)), int(m.group(2))); v = tuple(int(x) for x in m.group(3).split(",") if x.strip())
            if k in out and out[k] != v:
                raise Conflict(f"conflicting records for lane/rank {k}")
            out[k] = v
    return out


def check(text, num_layers, lanes, tp, ref_text=None):
    """STRUCTURAL check. Ordinals are local to each graph signature, so equality with the stock reference is
    structural parity (same count and positions), not proof that the same named K/V buffers are aliased: the log has
    no buffer names. Named identity is reported separately as UNVERIFIED by the rehearsal."""
    try:
        got = parse(text)
    except Conflict as e:
        return "FAIL", str(e)
    need = 1 + 2 * num_layers
    missing = sorted(k for k in ((l, r) for l in range(lanes) for r in range(tp)) if k not in got)
    if missing:
        return "FAIL", f"missing lane/rank {missing[:8]}{'...' if len(missing) > 8 else ''} ({len(missing)} of {lanes * tp})"
    per_lane = {}
    for l in range(lanes):
        lists = {got[(l, r)] for r in range(tp)}
        if len(lists) != 1:
            return "FAIL", f"lane {l}: ranks disagree on mutated inputs ({len(lists)} distinct lists)"
        per_lane[l] = lists.pop()
    short = {l: len(v) for l, v in per_lane.items() if len(v) < need}
    if short:
        return "FAIL", f"lanes without KV aliasing (need >= {need}): {short}"
    if ref_text is None:
        return "UNVERIFIED", f"{lanes} lanes x {tp} ranks, >= {need} mutated each; buffer identity unverified (no stock reference)"
    try:
        ref = parse(ref_text)
    except Conflict as e:
        return "FAIL", f"reference: {e}"
    diff = [l for l in range(lanes) if any(ref.get((l, r)) != per_lane[l] for r in range(tp))]
    if diff:
        return "FAIL", f"lanes {diff}: mutated-input ordinals differ from the stock reference (structural parity fails)"
    return "PASS", (f"structural parity with stock: {lanes} lanes x {tp} ranks, {need} mutated inputs each "
                    f"(hidden + 2x{num_layers}); named K/V identity not in the log")


if __name__ == "__main__":
    a = sys.argv[1:]
    ref = None
    if "--ref" in a:
        i = a.index("--ref"); ref = open(a[i + 1], errors="replace").read(); a = a[:i] + a[i + 2:]
    log, cfg, lanes, tp = a[0], a[1], int(a[2]), int(a[3])
    c = json.load(open(cfg)); c = c.get("text_config", c)
    st, detail = check(open(log, errors="replace").read(), int(c["num_hidden_layers"]), lanes, tp, ref)
    print(f"{st}\t{detail}")
    sys.exit({"PASS": 0, "FAIL": 1, "UNVERIFIED": 2}[st])
