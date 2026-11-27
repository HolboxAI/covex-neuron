"""KV-cache aliasing check on a compile log. Usage: alias_check.py <server.log> <config.json> <expected_lanes>
Every compiled graph (lane) must list each layer's K and V cache as a mutated (in-place) input: >= 2 * num_layers
mutated inputs besides the hidden state. Stock Qwen3-32B: 129 = 1 + 2*64 in all 6 lanes. The squeeze-view bug
(2026-11-27) left decode lanes with 1, and neuronx-cc then failed with NCC_EOOM002. Prints PASS/FAIL + detail."""
import json, re, sys


def check(log_text, num_layers, expected_lanes):
    lanes = {}
    for line in log_text.splitlines():
        m = re.search(r"\[trace lane=(\d+) rank=0\] Mutated inputs: \[([^\]]*)\]", line)
        if m:
            lanes[int(m.group(1))] = len([x for x in m.group(2).split(",") if x.strip()])
    need = 1 + 2 * num_layers
    missing = sorted(set(range(expected_lanes)) - lanes.keys())
    short = {l: n for l, n in sorted(lanes.items()) if n < need}
    ok = not missing and not short
    detail = f"lanes={sorted(lanes)} need>={need} mutated each; " + (
        "all lanes alias every K/V cache" if ok else f"missing lanes {missing}; lanes without KV aliasing {short}")
    return ok, detail


if __name__ == "__main__":
    log, cfg, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    c = json.load(open(cfg)); c = c.get("text_config", c)
    ok, detail = check(open(log, errors="replace").read(), int(c["num_hidden_layers"]), n)
    print(("PASS" if ok else "FAIL") + "\t" + detail)
    sys.exit(0 if ok else 1)
