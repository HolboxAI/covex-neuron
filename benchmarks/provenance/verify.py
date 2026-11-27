"""Check per-process provenance records for each arm. Usage:
  verify.py <provenance_dir> <min_procs> [--tp N --log ARM=server.log[,replica2.log] ...] ARM=EXPECTED_ROOT[@EXPECTED_PREFIX] [...]
With --tp/--log, TP0..TP(N-1) and EngineCore are taken from vLLM's own log prefixes and each must have a record, so
surplus helper processes cannot stand in for a missing worker rank.
Each arm needs >= min_procs records (API server + engine + TP workers). In every record vllm_neuron must resolve under
the arm's expected source root, no COVEX_* flag may be set unless the arm name says so (K/F/C), and the compile-cache
roots (NEURON_LIBTORCH_CACHE_ROOT, VLLM_CACHE_ROOT, TORCH_NEURONX_NEFF_CACHE_DIR) must be the arm's own and not shared
with another arm (a shared cache fails every arm involved). With @PREFIX, sys.prefix (the venv) must match. Exactly one
row per requested arm; a malformed record fails the arm it names (or every arm, if it names none). Prints "<arm>\\t<PASS|FAIL>\\t<detail>" rows; exit 1 on any FAIL."""
import glob, json, os, re, sys

EXPECTED_FLAGS = {"q32-U": set(), "q32-T": set(), "q32-K": {"COVEX_QWEN3_INKERNEL_KV_WRITE"},
                  "q32-F": {"COVEX_DECODE_DYNAMIC_FA"}, "q32-C": {"COVEX_QWEN3_INKERNEL_KV_WRITE", "COVEX_DECODE_DYNAMIC_FA"},
                  "q32-Ut": set(), "emb-U": set(), "emb-T": set(), "emb-P4": set(), "emb-P8": set(), "emb-P16": set(),
                  "emb-P1x": {"COVEX_FORCE_PACKED_ATTENTION"}, "emb-P8t": {"COVEX_PACK_TRACE"}}


CACHE_KEYS = ("NEURON_LIBTORCH_CACHE_ROOT", "VLLM_CACHE_ROOT", "TORCH_NEURONX_NEFF_CACHE_DIR")


def log_ranks(path):
    """vLLM's own log prefixes: "(Worker_TP3 pid=9478)", "(EngineCore pid=9022)" -> {"TP3": 9478, "EngineCore": ...}."""
    roles = {}
    try:
        for m in re.finditer(r"^\((Worker_TP(\d+)|EngineCore|APIServer)[^)]*?pid=(\d+)\)", open(path, errors="replace").read(), re.M):
            roles.setdefault(f"TP{m.group(2)}" if m.group(2) else m.group(1), int(m.group(3)))
    except OSError:
        pass
    return roles


def verify(pdir, min_procs, expect, tp=0, logs=None):
    recs, broken = {}, {}
    for f in glob.glob(os.path.join(pdir, "*.json")):
        try:
            r = json.load(open(f))
            assert isinstance(r, dict)
        except (ValueError, AssertionError, OSError):
            arm = os.path.basename(f).split(".")[0]
            broken.setdefault(arm, []).append(os.path.basename(f))
            continue
        recs.setdefault(r.get("arm"), []).append(r)
    caches = {}
    for arm in expect:
        caches[arm] = {(k, r.get("cache_roots", {}).get(k)) for r in recs.get(arm, []) for k in CACHE_KEYS
                       if r.get("cache_roots", {}).get(k)}
    rows = []
    for arm, spec in expect.items():
        root, _, prefix = spec.partition("@")
        rs, root = recs.get(arm, []), os.path.realpath(root)
        why = []
        if broken.get(arm) or (broken and None in broken):
            why.append(f"malformed provenance records {broken.get(arm) or broken.get(None)}")
        if len({r.get("pid") for r in rs}) < min_procs:
            why.append(f"{len({r.get('pid') for r in rs})} process records < {min_procs}")
        bad = [r for r in rs if not r.get("vllm_neuron_origin") or
               not os.path.realpath(r["vllm_neuron_origin"]).startswith(root + os.sep)]
        if bad:
            why.append(f"{len(bad)} processes load vllm_neuron from {sorted({str(b.get('vllm_neuron_origin')) for b in bad})[:2]}")
        if prefix and any(os.path.realpath(r.get("prefix", "")) != os.path.realpath(prefix) for r in rs):
            why.append(f"sys.prefix {sorted({r.get('prefix') for r in rs})} != {prefix}")
        want = EXPECTED_FLAGS.get(arm)
        flags = {tuple(r.get("covex_flags", [])) for r in rs}
        if want is not None and any(set(f) != want for f in flags):
            why.append(f"COVEX flags {sorted(flags)} != {sorted(want)}")
        lib = {r.get("cache_roots", {}).get("NEURON_LIBTORCH_CACHE_ROOT") for r in rs}
        if len(lib) != 1 or None in lib:
            why.append(f"NEURON_LIBTORCH_CACHE_ROOT {sorted(map(str, lib))} (need exactly one per arm)")
        if tp and logs and arm in logs:   # every TP rank (and the engine) must be a recorded process, by PID from the log
            pids = {r.get("pid") for r in rs}
            for lp in logs[arm].split(","):   # replicas: one log per replica, each with its own TP ranks + engine
                roles = log_ranks(lp)
                miss = [k for k in [f"TP{i}" for i in range(tp)] + ["EngineCore"] if roles.get(k) not in pids]
                if miss:
                    why.append(f"{os.path.basename(lp)}: no provenance record for {miss} (log pids {[roles.get(k) for k in miss]})")
        shared = sorted(b for b in expect if b != arm and {v for _, v in caches[arm]} & {v for _, v in caches[b]})
        if shared:
            why.append(f"shares a compile-cache path with {shared}")
        rows.append((arm, "FAIL" if why else "PASS",
                     "; ".join(why) or f"{len(rs)} processes, vllm_neuron under {root}, cache {next(iter(lib))}"))
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]
    tp = 0; logs = {}
    if "--tp" in a:
        i = a.index("--tp"); tp = int(a[i + 1]); a = a[:i] + a[i + 2:]
    while "--log" in a:
        i = a.index("--log"); k, v = a[i + 1].split("=", 1); logs[k] = v; a = a[:i] + a[i + 2:]
    pdir, n = a[0], int(a[1])
    expect = dict(x.split("=", 1) for x in a[2:])
    rows = verify(pdir, n, expect, tp, logs)
    for r in rows:
        print("\t".join(r))
    sys.exit(1 if any(r[1] == "FAIL" for r in rows) else 0)
