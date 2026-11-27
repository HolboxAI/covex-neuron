"""Static checks for the five-model benchmark arms. No Neuron hardware or packages needed.
Each test encodes a failure we actually hit on Trainium (2026-11-27), so it cannot recur silently.
Run: pytest tests/covex -q
"""
import json, os, re, shutil, subprocess
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
SERVE = ROOT / "benchmarks/five/serve.sh"
RUN = ROOT / "benchmarks/five/run.sh"
PAYLOAD = ROOT / "benchmarks/five/payload.sh"
ARMS = ["q32-T", "q32-K", "q32-F", "q32-C", "emb-T", "emb-C", "emb-P4", "emb-P8", "emb-P16", "vl-T", "vl-C", "oss-T"]
GIB = 1024 ** 3

# Per-model facts for the HBM check: total BF16 weight bytes, TP degree, KV bytes per token per rank, limits.
# Weights match what Neuron reported at load (Qwen3-32B: 7.64 GiB/core). Trn2 at LNC2: 24 GiB HBM per logical core.
MODELS = {
    "q32": dict(weights=65.5e9, tp=8, kv_tok_rank=64 * 1 * 128 * 2 * 2),   # 64 layers, 8 KV heads / TP8 = 1 per rank
    "emb": dict(weights=15.1e9, tp=8, kv_tok_rank=36 * 1 * 128 * 2 * 2),
    "vl":  dict(weights=66.7e9, tp=4, kv_tok_rank=64 * 2 * 128 * 2 * 2),   # 8 KV heads / TP4 = 2 per rank
    "oss": dict(weights=42.0e9, tp=8, kv_tok_rank=24 * 1 * 64 * 2 * 2),    # dequantized to BF16 by the quantization override
}
HBM_PER_RANK = 24 * GIB
NEFF_RESERVE = 5 * GIB   # Qwen3-32B failed with ~2.4 GiB left after weights + KV; keep >= 5 GiB


def dry(arm, profile="0"):
    env = {"HOME": os.environ.get("HOME", "/tmp"), "PATH": os.environ["PATH"], "DRYRUN": "1", "PROFILE": profile}
    out = subprocess.run(["bash", str(SERVE), arm, "8000"], env=env, capture_output=True, text=True, check=True).stdout
    flags = re.search(r"flags=([A-Z0-9_ ]*)cfg=", out).group(1).split()
    cfg = json.loads(out.split("cfg=", 1)[1].splitlines()[0])
    text = SERVE.read_text()
    # the arm's top-level case block: from its label line to the next top-level label (nested cases have ";;" too)
    block = re.search(rf"\n  [A-Za-z0-9|*-]*{re.escape(arm)}[A-Za-z0-9|-]*\)(.*?)(?=\n  [A-Za-z0-9|*-]+\)|\nesac)", text, re.S).group(1)
    return flags, cfg, block


@pytest.mark.parametrize("arm", ARMS)
def test_arm_config_is_valid_json_and_profiler_merges(arm):
    _, cfg, _ = dry(arm)
    assert "neuron_config" in cfg
    _, cfg_p, _ = dry(arm, profile="1")
    assert cfg_p["neuron_profiler"]["activities"] == ["system_profile", "device_profile"]


def test_every_flag_is_read_by_vllm_neuron():
    names = set(re.findall(r"\b(COVEX_[A-Z0-9_]+|VLLM_NEURON_[A-Z0-9_]+)", SERVE.read_text()))
    names -= {"VLLM_NEURON_COMPILATION_TIMEOUT"}  # documented by AWS quickstart, harmless if unread
    harness = {"COVEX_ARM", "COVEX_PROVENANCE_DIR"}   # read by benchmarks/provenance/sitecustomize.py, not the model
    prov = (ROOT / "benchmarks/provenance/sitecustomize.py").read_text()
    assert all(h in prov for h in harness)
    names -= harness
    src = "\n".join(p.read_text() for p in (ROOT / "vllm_neuron").rglob("*.py"))
    missing = sorted(n for n in names if n not in src)
    assert not missing, f"env vars set by serve.sh but never read by vllm_neuron: {missing}"


def test_no_env_var_that_vllm_neuron_rejects():
    src = "\n".join(p.read_text() for p in (ROOT / "vllm_neuron").rglob("*.py"))
    rejected = set(re.findall(r"([A-Z][A-Z0-9_]{6,}) cannot be used", src)) | {"NEURON_RT_VISIBLE_CORES"}
    scripts = "\n".join(p.read_text() for p in (ROOT / "benchmarks").rglob("*.sh"))
    bad = sorted(v for v in rejected if re.search(rf"export {v}=|\b{v}=\S", scripts))
    assert not bad, f"scripts set env vars vllm-neuron rejects: {bad}"


@pytest.mark.parametrize("arm", ARMS)
def test_single_shot_prefill_disables_prefix_caching(arm):
    _, _, block = dry(arm)
    mml = int(re.search(r"--max-model-len (\d+)", block).group(1))
    mbt = int(re.search(r"--max-num-batched-tokens (\d+)", block).group(1))
    if mbt >= mml:
        assert "--no-enable-prefix-caching" in block, f"{arm}: single-shot prefill needs --no-enable-prefix-caching"


@pytest.mark.parametrize("arm", ARMS)
def test_hbm_budget(arm):
    _, _, block = dry(arm)
    m = MODELS[arm.split("-")[0]]
    cap = float((re.search(r"CAP_FRACTION=([0-9.]+)", block) or [None, "0.30"])[1])
    mml = int(re.search(r"--max-model-len (\d+)", block).group(1))
    seqs = int(re.search(r"--max-num-seqs (\d+)", block).group(1))
    weights = m["weights"] / m["tp"]
    usable = HBM_PER_RANK * 0.9
    kv_budget = min(usable - weights, cap * usable)          # vllm-neuron _compute_kv_budget (fractional mode)
    kv_need = mml * seqs * m["kv_tok_rank"]
    headroom = HBM_PER_RANK - weights - kv_budget
    assert kv_need <= kv_budget, f"{arm}: KV need {kv_need/GIB:.1f} GiB > budget {kv_budget/GIB:.1f} GiB"
    assert headroom >= NEFF_RESERVE, f"{arm}: only {headroom/GIB:.1f} GiB left for compiled programs"


def test_known_broken_arms_are_not_started():
    text = RUN.read_text()
    broken = re.search(r'BROKEN_ARMS="([^"]*)"', text).group(1).split()
    assert "emb-C" in broken  # remove only together with the packed-prefill recompile fix


@pytest.mark.parametrize("payload", sorted((ROOT / "benchmarks").rglob("payload*.sh")), ids=lambda p: p.parent.name)
def test_payload_uses_supported_download_cli(payload):
    text = payload.read_text()
    assert re.search(r"^shutdown -h \+\d+", text, re.M), "payload must hard-stop the instance"
    assert "huggingface-cli" not in text, "huggingface-cli is retired; use `hf download`"
    for line in re.findall(r"hf download[^\n]*", text):
        assert not re.search(r'--exclude "[^"]+" "[^"]+"', line), f"one pattern per --exclude: {line}"
    for model in re.findall(r"hf download \S+ --local-dir ~/models[-a-z]*/(\S+)", text):
        assert "config.json" in text, f"{model}: no post-download check"


def test_changed_python_has_no_undefined_names():
    if not shutil.which("git") or subprocess.run(["python3", "-m", "pyflakes", "--version"], capture_output=True).returncode:
        pytest.skip("pyflakes not installed")
    files = subprocess.run(["git", "diff", "--name-only", "origin/main...HEAD", "--", "*.py"],
                           cwd=ROOT, capture_output=True, text=True).stdout.split()
    files = [f for f in files if (ROOT / f).exists()]
    if not files:
        pytest.skip("no changed python files")
    out = subprocess.run(["python3", "-m", "pyflakes", *files], cwd=ROOT, capture_output=True, text=True).stdout
    bad = [l for l in out.splitlines() if "undefined name" in l or "syntax" in l.lower()]
    assert not bad, "\n".join(bad)


def test_vllm_neuron_imports_in_cpu_mode():
    """Catches import-time errors like the missing `import nki` (needs the Neuron pip packages; skipped otherwise)."""
    pytest.importorskip("nkilib")
    env = dict(os.environ, VLLM_NEURON_CPU_MODE="1")
    r = subprocess.run(["python3", "-c", "import vllm_neuron, vllm_neuron.functional"], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]


def test_shell_functions_defined_before_first_call():
    """run.sh called llm() in the profile block before defining it (caught 2026-11-27, before any launch)."""
    for script in (RUN, ROOT / "benchmarks/asr/run_all.sh"):
        if not script.exists():
            continue
        lines = script.read_text().splitlines()
        defs = {m.group(1): i for i, l in enumerate(lines) if (m := re.match(r"\s*([a-z_][a-z0-9_]*)\(\)\s*\{", l))}
        for name, d in defs.items():
            for i, l in enumerate(lines[:d]):
                code = l.split("#", 1)[0]
                if re.search(rf"(^|[;&|({{]\s*|\s){name}\s", code):
                    pytest.fail(f"{script.name}:{i+1} calls {name}() before its definition at line {d+1}")


def test_qwen3_inkernel_kv_write_passes_bound_cache_buffer():
    """The in-kernel KV write must target self.k_cache/self.v_cache, not a squeeze view (NCC_EOOM002, q32-C 2026-11-27)."""
    src = (ROOT / "vllm_neuron/model/qwen3/model.py").read_text()
    call = src[src.index("update_cache=_INKERNEL_KV") - 3000: src.index("update_cache=_INKERNEL_KV")]
    assert "K_cache=self.k_cache if _INKERNEL_KV" in call and "V_cache=self.v_cache if _INKERNEL_KV" in call


def test_rehearsal_scripts_parse_and_cover_every_arm():
    sh = (ROOT / "benchmarks/rehearsal/rehearse.sh").read_text()
    subprocess.run(["bash", "-n", str(ROOT / "benchmarks/rehearsal/rehearse.sh")], check=True)
    for arm in ["q32-T", "q32-C", "emb-T", "emb-C", "vl-T", "vl-C", "asr-T", "asr-C"]:
        assert arm in sh, f"rehearsal does not cover {arm}"   # vl-* in the compile track only (CPU-mode GQA limit)
    assert "VLLM_NEURON_CPU_MODE=1" in sh and "VLLM_NEURON_CPU_COMPILE=1" in sh


def test_no_bare_wait_outside_subshells():
    """A bare `wait` in the shell that started the servers waits for the servers forever (five-model run 2 hung there)."""
    for script in [RUN, ROOT / "benchmarks/rehearsal/rehearse.sh", ROOT / "benchmarks/asr/run_all.sh"]:
        if not script.exists():
            continue
        depth = 0
        for i, line in enumerate(script.read_text().splitlines(), 1):
            code = line.split("#", 1)[0].strip()
            if code == "(":
                depth += 1
                continue
            if code.startswith(")") and depth:
                depth -= 1
                continue
            if depth == 0 and re.search(r"(^|[;&{]\s*)wait\s*(;|$|\})", code):
                if "start" in script.read_text() or "serve" in script.read_text():
                    pytest.fail(f"{script.name}:{i}: bare `wait` at top level; wait on client PIDs instead")





def test_encoder_mlp_ports_are_kernel_eligible():
    """NF.mlp silently falls back to PyTorch when a shape fails its gate; then a COVEX arm measures stock.
    Re-derive the CTE gate (H % 128, ceil(I/tile) <= PSUM banks, bias disables I-sharding) for the ported encoders."""
    src = (ROOT / "vllm_neuron/functional/mlp.py").read_text()
    banks = int(re.search(r"^_NUM_HW_PSUM_BANKS = (\d+)", src, re.M).group(1))
    tile = int(re.search(r"^_SRC_PROJ_INT_DIM_TILE_SIZE = (\d+)", src, re.M).group(1))
    # (name, hidden H, intermediate I, tensor-parallel degree); both MLPs carry an up bias -> no I-sharding
    for name, H, I, tp in [("Qwen3-VL-32B vision", 1152, 4304, 4), ("Qwen3-ASR-1.7B audio", 1024, 4096, 8)]:
        i_rank = -(-I // tp)
        assert H % 128 == 0, f"{name}: H={H} not 128-aligned -> torch fallback"
        assert -(-i_rank // tile) <= banks, f"{name}: I/rank={i_rank} exceeds {banks}x{tile} -> torch fallback"


def test_covex_flags_default_off():
    src = (ROOT / "vllm_neuron/envs.py").read_text()
    for flag in re.findall(r'"(COVEX_[A-Z0-9_]+)"\s*:', src):
        body = src[src.index(f'"{flag}"'):][:300]
        assert re.search(r'getenv\(\s*"%s"\s*(,\s*"0"\s*)?\)' % flag, body) or '"0"' in body or "False" in body, f"{flag} must default off"


def _rehearsal_env(tmp_path, home, extra):
    """Env for rehearse.sh with fakes: fork arms import a fork stub, U's fake interpreter swaps in an upstream stub."""
    stub = {}
    for k in ("upstream", "fork"):
        stub[k] = tmp_path / f"stub-{k}"; (stub[k] / "vllm_neuron").mkdir(parents=True, exist_ok=True)
        (stub[k] / "vllm_neuron/__init__.py").write_text("")
    upy = tmp_path / "upstream-venv-python"
    upy.write_text(f"""#!/bin/bash
shift 2
export PYTHONPATH="${{PYTHONPATH%%:*}}:{stub['upstream']}"
exec vllm "$@"
""")
    upy.chmod(0o755)
    env = {"HOME": str(home), "PATH": f"{ROOT/'tests/covex/fake_bin'}:{os.environ['PATH']}", "PYTHONPATH": str(stub["fork"]),
           "UPSTREAM_DIR": str(stub["upstream"]), "UPSTREAM_PY": str(upy), "FORK_DIR": str(stub["fork"]),
           "REHEARSAL_MIN_PROCS": "1", "FAKE_LAYERS": "2",
           "REHEARSAL_PAIRS_A": "q32-T q32-K;q32-U q32-C", "REHEARSAL_PAIRS_B": "q32-K q32-T;q32-U q32-C;vl-T vl-C"}
    env.update(extra)
    return env


def test_rehearsal_harness_end_to_end_with_fake_vllm(tmp_path):
    """Runs the real rehearse.sh (q32 arms, both tracks) against a fake vllm that starts 3 s late.
    Catches harness bugs (start-up race, empty-log PASS, wrong waits) before they cost an instance."""
    import sys as _sys
    if _sys.platform != "linux" or not shutil.which("taskset"):
        pytest.skip("needs Linux bash>=4 and taskset (runs in CI)")
    home = tmp_path / "home"
    for d in ["models/Qwen3-32B", "models-small/Qwen3-32B", "models/Qwen3-VL-32B-Instruct"]:
        (home / d).mkdir(parents=True)
    (home / "models/Qwen3-32B/.done").touch()
    (home / "models/Qwen3-VL-32B-Instruct/.done").touch()
    (home / "models/Qwen3-32B/config.json").write_text('{"num_hidden_layers": 2}')
    run = tmp_path / "run"; run.mkdir()
    env = _rehearsal_env(tmp_path, home, {"FAKE_EXIT_AFTER_EXTRACTION_PORTS": "8206", "FAKE_FAILED_LAUNCHER_PORTS": "8207",
                                          "FAKE_ALIAS_BUG_PORTS": "8209"})   # 8209 = q32-K compile: must be caught
    r = subprocess.run(["bash", str(ROOT / "benchmarks/rehearsal/rehearse.sh")], cwd=run, env=env,
                       capture_output=True, text=True, timeout=600)
    tsv = (run / "REHEARSAL.tsv").read_text()
    logs = "\n".join(f"--- {f.name}\n{f.read_text()[-1500:]}" for f in sorted((run / "logs").glob("*.log")))
    rows = {(l.split("\t")[0], l.split("\t")[1]): l.split("\t")[2] for l in tsv.splitlines() if l}
    expect = {(a, t) for a in ("q32-T", "q32-K", "q32-U", "q32-C") for t in ("cpu", "compile", "compile-alias")}
    assert expect <= rows.keys(), f"missing rows: {expect - rows.keys()}\n{tsv}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    ident = {k: v for k, v in rows.items() if k[1] == "kv-identity"}
    assert ident == {(a, "kv-identity"): "UNVERIFIED" for a in ("q32-T", "q32-K", "q32-U", "q32-C")}, ident
    bad = {k: v for k, v in rows.items() if v != "PASS" and k[1] != "kv-identity"}
    assert bad == {("q32-K", "compile-alias"): "FAIL", ("vl-T", "compile"): "FAIL",
                   ("vl-C", "compile"): "FAIL"}, f"unexpected failures: {bad}\n{tsv}\n{logs}"
    assert "vl-T\tcompile\tFAIL\tno post-compilation completion marker" in tsv
    assert "vl-C\tcompile\tFAIL\tlauncher exited 137" in tsv
    md = (run / "REHEARSAL.md").read_text()
    assert r.returncode == 1 and "OVERALL FAIL" in md and "REQUIRED q32-K:compile-alias: FAIL" in md, md
    assert subprocess.run(["pgrep", "-f", "fake_server.py"], capture_output=True).returncode != 0, "servers left running"


# ---------------- behaviour tests (run real code against fakes; Linux CI) ----------------
LINUX = __import__("sys").platform == "linux" and shutil.which("taskset") is not None
FAKE = ROOT / "tests/covex/fake_server.py"


def _bash(script, env=None, timeout=120):
    e = dict(os.environ, **(env or {}))
    return subprocess.run(["bash", "-c", f"source {ROOT/'benchmarks/lib.sh'}\n{script}"], env=e, capture_output=True,
                          text=True, timeout=timeout)


def _alias():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ac", ROOT / "benchmarks/rehearsal/alias_check.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_alias_check_on_real_logs_and_mutations():
    ac = _alias(); fx = ROOT / "tests/covex/fixtures"
    stock = (fx / "alias_stock_ok.log").read_text()          # real q32-T compile log: 6 lanes x 8 ranks, 129 each
    bug = (fx / "alias_inkernel_squeeze_bug.log").read_text()  # real q32-C log: decode lanes [1], 2 ranks never logged
    assert ac.check(stock, 64, 6, 8, ref_text=stock)[0] == "PASS"
    assert ac.check(stock, 64, 6, 8)[0] == "UNVERIFIED", "no reference -> count heuristic only, never PASS"
    assert ac.check(bug, 64, 6, 8, ref_text=stock)[0] == "FAIL"
    lines = stock.splitlines()
    # a nonzero rank missing
    no_rank = "\n".join(l for l in lines if "lane=3 rank=5]" not in l)
    st, d = ac.check(no_rank, 64, 6, 8, ref_text=stock); assert st == "FAIL" and "(3, 5)" in d, d
    # one real cache index swapped for an unrelated one on ALL ranks of lane 2, count unchanged
    swapped = "\n".join(l.replace("Mutated inputs: [1, 13, 14,", "Mutated inputs: [1, 13, 999,") if "lane=2 " in l else l for l in lines)
    assert ac.parse(swapped)[(2, 0)].__len__() == 129
    st, d = ac.check(swapped, 64, 6, 8, ref_text=stock); assert st == "FAIL" and "differ from the stock" in d, d
    # conflicting duplicate record for one lane/rank
    dup = stock + "\n" + lines[0].replace("[1, 13,", "[1, 77,")
    st, d = ac.check(dup, 64, 6, 8, ref_text=stock); assert st == "FAIL" and "conflicting" in d, d
    # a PASS never claims named identity
    st, d = ac.check(stock, 64, 6, 8, ref_text=stock); assert "named K/V identity not in the log" in d, d
    # the same swap on one rank only
    one = "\n".join(l.replace("[1, 13, 14,", "[1, 13, 999,") if "lane=2 rank=5]" in l else l for l in lines)
    st, d = ac.check(one, 64, 6, 8, ref_text=stock); assert st == "FAIL" and "ranks disagree" in d, d
    # decode lanes without KV aliasing, all ranks present
    noalias = "\n".join(re.sub(r"Mutated inputs: \[[^\]]*\]", "Mutated inputs: [1]", l) if re.search(r"lane=[2-5] ", l) else l for l in lines)
    st, d = ac.check(noalias, 64, 6, 8, ref_text=stock); assert st == "FAIL" and "without KV aliasing" in d, d


@pytest.mark.skipif(not LINUX, reason="Linux CI")
def test_lib_readiness_requires_real_inference(tmp_path):
    r = _bash(f"""
      python3 {FAKE} --port 8911 & p1=$!; FAKE_FAIL_INFER=1 python3 {FAKE} --port 8912 & p2=$!
      (sleep 4; exec python3 {FAKE} --port 8913) & p3=$!      # starts late
      why=$(wait_ready $p1 8911 60 gen); echo "healthy=$?"
      why=$(wait_ready $p2 8912 60 gen); echo "health_only=$? $why"
      why=$(wait_ready $p3 8913 60 gen); echo "late=$?"
      (exit 1) & d=$!; why=$(wait_ready $d 8914 30 gen); echo "dead=$?"
      stop_arm $p1 8911; stop_arm $p2 8912; stop_arm $p3 8913""")
    out = r.stdout
    assert "healthy=0" in out and "late=0" in out, out + r.stderr
    assert "health_only=3" in out and "smoke request failed" in out, "health without working inference must not be READY\n" + out
    assert "dead=1" in out, out


@pytest.mark.skipif(not LINUX, reason="Linux CI")
def test_lib_stop_arm_is_targeted_and_wait_all_sees_every_failure():
    r = _bash(f"""
      python3 {FAKE} --port 8921 & a=$!; python3 {FAKE} --port 8922 & b=$!
      wait_health $a 8921 30; wait_health $b 8922 30
      stop_arm $a 8921
      curl -sf localhost:8921/health >/dev/null && echo a_alive || echo a_stopped
      curl -sf localhost:8922/health >/dev/null && echo b_alive || echo b_stopped
      stop_arm $b 8922
      (exit 1) & f=$!; (sleep 1; exit 0) & g=$!; wait_all $f $g; echo "wait_all=$?"
      (exit 0) & f=$!; (exit 0) & g=$!; wait_all $f $g; echo "wait_all_ok=$?" """)
    out = r.stdout
    assert "a_stopped" in out and "b_alive" in out, "stop_arm must stop only its own arm\n" + out + r.stderr
    assert "wait_all=1" in out and "wait_all_ok=0" in out, out


@pytest.mark.skipif(not LINUX, reason="Linux CI")
def test_bench_prewarms_each_level_before_its_snapshot(tmp_path):
    ev = tmp_path / "events"; work = tmp_path / "w"; work.mkdir()
    srv = subprocess.Popen(["python3", str(FAKE), "--port", "8931"], env=dict(os.environ, FAKE_EVENTS=str(ev)))
    try:
        for _ in range(50):
            if subprocess.run(["curl", "-sf", "localhost:8931/health"], capture_output=True).returncode == 0:
                break
            __import__("time").sleep(0.2)
        env = dict(os.environ, FAKE_EVENTS=str(ev), CLIENT_DIR=str(ROOT / "tests/covex/fake_client"), SOLO="1", ROUNDS="1")
        r = subprocess.run(["bash", str(ROOT / "benchmarks/asr/bench.sh"), "X", "8931", "4", "1,8"], cwd=work, env=env,
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stdout + r.stderr
    finally:
        srv.terminate()
    lines = [l for l in ev.read_text().splitlines() if not l.startswith("POST")]
    assert lines[0].startswith("CLIENT") and "prewarm" in lines[0], "global prewarm must precede the first snapshot\n" + "\n".join(lines)
    for c in (1, 8):
        for ph in ("transcribe", "chat_stream"):
            seq = [i for i, l in enumerate(lines) if f"c={c} ph={ph} " in l]
            pre = [i for i in seq if f"X.c{c}.{ph}.prewarm" in lines[i]]
            meas = [i for i in seq if f"out=X.c{c}.{ph}.json" in lines[i]]
            assert pre and meas, f"c{c} {ph}: missing prewarm or measured call\n" + "\n".join(lines)
            p, m = pre[0], meas[0]
            assert p < m and lines[p + 1] == "METRICS" and lines[m + 1] == "METRICS" and m == p + 2, \
                f"c{c} {ph}: order must be prewarm, METRICS(before), measured, METRICS(after)\n" + "\n".join(lines)
    assert (work / "results/X.json").exists()


@pytest.mark.skipif(not LINUX, reason="Linux CI")
def test_bench_fails_loudly_when_server_down(tmp_path):
    r = subprocess.run(["bash", str(ROOT / "benchmarks/asr/bench.sh"), "X", "8939", "4", "1"], cwd=tmp_path,
                       env=dict(os.environ, CLIENT_DIR=str(ROOT / "tests/covex/fake_client"), SOLO="1"),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0, "a down server must not look like a successful benchmark"


def test_no_pkill_by_pattern_in_runners():
    """pkill -f patterns killed the wrong process (the test shell itself); stops must go by PID tree."""
    for f in (ROOT / "benchmarks").rglob("*.sh"):
        code = "\n".join(l.split("#", 1)[0] for l in f.read_text().splitlines())
        assert "pkill" not in code, f"{f.relative_to(ROOT)} uses pkill"


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("extra", ["", "--gpu-memory-utilization 0.2"])
def test_final_vllm_argv_is_clean(arm, extra):
    """The exact argv serve.sh would exec. A variable-name clash ($EXTRA) once injected '' and a JSON fragment as
    positional args (emb-T/emb-C would have died on Trainium too; caught by rehearsal 1)."""
    import shlex
    env = {"HOME": os.environ.get("HOME", "/tmp"), "PATH": os.environ["PATH"], "DRYRUN": "1", "REHEARSAL_EXTRA": extra}
    out = subprocess.run(["bash", str(SERVE), arm, "8000"], env=env, capture_output=True, text=True, check=True).stdout
    argv = shlex.split(out.split("argv:", 1)[1].strip())
    assert argv[:2] == ["vllm", "serve"]
    assert "" not in argv, f"{arm}: empty argument in {argv}"
    value_flags = {"--additional-config", "--profiler-config", "--hf-overrides"}
    for i, tok in enumerate(argv):
        if '":' in tok or tok.startswith("{"):
            assert argv[i - 1] in value_flags, f"{arm}: config fragment {tok!r} passed as a bare argument"
    if extra:
        assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.2"
    else:
        assert "--gpu-memory-utilization" not in argv


def test_mlp_torch_fallback_activations_are_traceable():
    """CPU-mode/torch fallback returned the bare F.gelu object; dynamo saw torch._C._nn.gelu and refused to trace
    (asr-C in rehearsal 1). Every activation must be a Python callable dynamo can inline."""
    src = (ROOT / "vllm_neuron/functional/mlp.py").read_text()
    body = src[src.index("def _get_activation_fn"):]
    body = body[:body.index("\ndef ")]
    assert not re.search(r"return torch\.nn\.functional\.gelu\s*$", body, re.M), "return a lambda wrapper, not F.gelu"



@pytest.mark.skipif(not LINUX, reason="Linux CI")
def test_rehearsal_overall_pass_when_required_rows_pass(tmp_path):
    """Success path: healthy q32 arms -> OVERALL PASS, exit 0, even though non-required VL arms fail."""
    home = tmp_path / "home"
    for d in ["models/Qwen3-32B", "models-small/Qwen3-32B", "models/Qwen3-VL-32B-Instruct"]:
        (home / d).mkdir(parents=True)
    for d in ["models/Qwen3-32B", "models/Qwen3-VL-32B-Instruct"]:
        (home / d / ".done").touch()
    (home / "models/Qwen3-32B/config.json").write_text('{"num_hidden_layers": 2}')
    run = tmp_path / "run"; run.mkdir()
    env = _rehearsal_env(tmp_path, home, {"FAKE_FAILED_LAUNCHER_PORTS": "8207"})
    r = subprocess.run(["bash", str(ROOT / "benchmarks/rehearsal/rehearse.sh")], cwd=run, env=env,
                       capture_output=True, text=True, timeout=600)
    md = (run / "REHEARSAL.md").read_text()
    assert r.returncode == 0 and "OVERALL PASS" in md, md
    tsv = (run / "REHEARSAL.tsv").read_text()
    assert all(f"{a}\tcompile-alias\tPASS" in tsv for a in ("q32-T", "q32-K", "q32-U", "q32-C")), tsv
    assert all(f"{a}\tprovenance\tPASS" in tsv for a in ("q32-T", "q32-K", "q32-U")), tsv
    assert "vl-C\tcompile\tFAIL" in (run / "REHEARSAL.tsv").read_text()


SUMM = ROOT / "tests/covex/fixtures/summ"


@pytest.mark.parametrize("case,ratio,code", [("ok", "2.00x", 0), ("partial", None, 1), ("mismatch", None, 1)])
def test_summary_prints_ratio_only_for_complete_correct_work(case, ratio, code):
    r = subprocess.run(["python3", str(ROOT / "benchmarks/five/summarize.py"), str(SUMM / case), "--required", "q32-T q32-C"],
                       capture_output=True, text=True)
    row = next(l for l in r.stdout.splitlines() if l.startswith("| Qwen3-32B | c1 | output tok/s"))
    assert r.returncode == code, r.stdout
    if ratio:
        assert f"| {ratio} | complete |" in row, row
    else:
        assert "| – |" in row and ("completed 1/8" in row or "outputs differ (0/20" in row), row


def test_match_rejects_empty_and_unequal_outputs(tmp_path):
    def run(a, b):
        (tmp_path / "a.json").write_text(json.dumps({"answers": a})); (tmp_path / "b.json").write_text(json.dumps({"answers": b}))
        out = subprocess.run(["python3", str(ROOT / "benchmarks/five/clients.py"), "match", str(tmp_path / "a.json"),
                              str(tmp_path / "b.json")], capture_output=True, text=True, check=True).stdout
        return json.loads(out)["pass"]
    assert run(["x", "y"], ["x", "y"]) is True
    assert run([], []) is False, "0/0 must not count as a match"
    assert run(["", ""], ["", ""]) is False, "empty answers must not count"
    assert run(["x"], ["x", "y"]) is False


def _five_run(tmp_path, extra_env):
    home = tmp_path / "home"; (home / "models/Qwen3-32B").mkdir(parents=True)
    run = tmp_path / "run"; run.mkdir()
    client = tmp_path / "client"; shutil.copytree(ROOT / "tests/covex/fake_client", client)
    (client / "data").mkdir(); (client / "data/a.wav").write_bytes(b"RIFF fake")
    env = dict(os.environ, HOME=str(home), PATH=f"{ROOT/'tests/covex/fake_bin'}:{os.environ['PATH']}",
               FIVE_ARMS="q32-T q32-C emb-C", CLIENT_DIR=str(client))
    env.update(extra_env)
    r = subprocess.run(["bash", str(ROOT / "benchmarks/five/run.sh")], cwd=run, env=env, capture_output=True, text=True,
                       timeout=600)
    return r, run


@pytest.mark.skipif(not LINUX, reason="Linux CI")
def test_five_run_success_path_with_fakes(tmp_path):
    r, run = _five_run(tmp_path, {})
    res = (run / "RESULTS.md").read_text(); st = (run / "results/STATUS.tsv").read_text()
    assert r.returncode == 0 and "RUN OK" in r.stdout, r.stdout[-3000:] + r.stderr[-2000:]
    assert "| Qwen3-32B | c1 | output tok/s | 100.0 | 200.0 | 2.00x | complete |" in res, res
    assert "emb-C\tstart\tSKIPPED known broken" in st and "FAILED" not in st, st          # skip is not a failure
    ev = (run / "evidence/EVIDENCE.tsv").read_text()
    assert "profile-q32-T\tOK" in ev and (run / "evidence/profile-q32-T/q32-T/rank0.ntff").exists(), ev


@pytest.mark.skipif(not LINUX, reason="Linux CI")
def test_five_run_failed_stock_client_suppresses_ratio(tmp_path):
    r, run = _five_run(tmp_path, {"FAKE_BENCH_FAIL_PORTS": "8000"})          # stock client fails, neighbour succeeds
    res = (run / "RESULTS.md").read_text(); st = (run / "results/STATUS.tsv").read_text()
    assert r.returncode != 0 and "RUN INCOMPLETE" in r.stdout, r.stdout[-3000:]
    assert "q32\tworkload\tFAILED" in st and "q32-C\tbench-c1\tOK" in st, st
    row = next(l for l in res.splitlines() if l.startswith("| Qwen3-32B | c1 | output tok/s"))
    assert "| – |" in row and "T: no result" in row, row


@pytest.mark.skipif(not LINUX, reason="Linux CI")
def test_five_run_partial_completion_suppresses_ratio(tmp_path):
    r, run = _five_run(tmp_path, {"FAKE_BENCH_COMPLETED": "8001:1"})        # valid JSON, 1/8 completed
    res = (run / "RESULTS.md").read_text()
    assert r.returncode != 0
    row = next(l for l in res.splitlines() if l.startswith("| Qwen3-32B | c1 | output tok/s"))
    assert "| – |" in row and "C: completed 1/8" in row, row


def _bench(tmp_path, port, server_env=None, conc="1,8", n="4"):
    """Run the real bench.sh against fake_server.py + the fake client; return (proc, workdir)."""
    import time
    work = tmp_path / "w"; work.mkdir(exist_ok=True)
    srv = subprocess.Popen(["python3", str(FAKE), "--port", str(port)], env=dict(os.environ, **(server_env or {})))
    try:
        for _ in range(50):
            if subprocess.run(["curl", "-sf", f"localhost:{port}/health"], capture_output=True).returncode == 0:
                break
            time.sleep(0.2)
        env = dict(os.environ, CLIENT_DIR=str(ROOT / "tests/covex/fake_client"), SOLO="1", ROUNDS="1")
        r = subprocess.run(["bash", str(ROOT / "benchmarks/asr/bench.sh"), "X", str(port), n, conc], cwd=work, env=env,
                           capture_output=True, text=True, timeout=120)
    finally:
        srv.terminate(); srv.wait()
    return r, work


def _stage_rows(work):
    out = subprocess.run(["python3", str(ROOT / "benchmarks/asr/stage_levels.py"), str(work / "metrics"), str(work / "results"), "X"],
                         capture_output=True, text=True, check=True).stdout
    return {(l.split("|")[2].strip(), l.split("|")[3].strip()): l for l in out.splitlines() if l.startswith("| X |")}


@pytest.mark.skipif(not LINUX, reason="Linux CI")
def test_bench_success_two_phases_two_levels(tmp_path):
    r, work = _bench(tmp_path, 8941)
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads((work / "results/X.json").read_text())
    assert d["concurrency_levels"] == [1, 8] and len(d["raw_results"]) == 4, "merge must keep every part's raw_results/levels"
    assert {ph: [lv["concurrency"] for lv in d["phases"][ph]] for ph in d["phases"]} == {"transcribe": [1, 8], "chat_stream": [1, 8]}
    rows = _stage_rows(work)
    assert len(rows) == 4 and all("unavailable" not in v for v in rows.values()), rows
    assert all(v.split("|")[4].strip() == "4" for v in rows.values()), "server request delta must equal measured requests"


@pytest.mark.skipif(not LINUX, reason="Linux CI")
def test_bench_rejects_healthy_server_that_fails_measurements(tmp_path):
    r, work = _bench(tmp_path, 8942, {"FAKE_FAIL_INFER": "1"})
    assert r.returncode == 4 and "INVALID" in r.stderr, r.stdout + r.stderr   # the client itself exits 0 here
    v = json.loads((work / "results/X.json").read_text())["validation"]
    assert v["problems"] and "success 0/4" in v["problems"][0], v


@pytest.mark.skipif(not LINUX, reason="Linux CI")
def test_stage_levels_unavailable_after_failed_before_scrape(tmp_path):
    # scrape #1 is the c1 transcribe before-scrape (prewarm calls do not scrape)
    r, work = _bench(tmp_path, 8943, {"FAKE_METRICS_FAIL_NTH": "1"})
    assert r.returncode == 0, r.stdout + r.stderr
    rows = _stage_rows(work)
    assert "unavailable: missing scrape" in rows[("1", "transcribe")], rows
    assert all("unavailable" not in v for k, v in rows.items() if k != ("1", "transcribe")), rows


def test_stage_levels_counter_reset_and_count_mismatch(tmp_path):
    m, rd = tmp_path / "m", tmp_path / "r"; m.mkdir(); rd.mkdir()
    prom = lambda n: f"vllm:e2e_request_latency_seconds_sum {0.1 * n}\nvllm:e2e_request_latency_seconds_count {n}\n"
    res = lambda ok, c: json.dumps({"phases": {"transcribe": [{"concurrency": c, "n_success": ok}]}})
    (m / "X.c1.transcribe.before.prom").write_text(prom(100)); (m / "X.c1.transcribe.after.prom").write_text(prom(10))
    (rd / "X.c1.transcribe.json").write_text(res(4, 1))
    (m / "X.c8.transcribe.before.prom").write_text(prom(10)); (m / "X.c8.transcribe.after.prom").write_text(prom(17))
    (rd / "X.c8.transcribe.json").write_text(res(4, 8))          # 7 server requests for 4 measured: warmup leaked in
    out = subprocess.run(["python3", str(ROOT / "benchmarks/asr/stage_levels.py"), str(m), str(rd), "X"],
                         capture_output=True, text=True, check=True).stdout
    assert "unavailable: counter reset" in out and "server counted 7 requests, client 4 successful" in out, out


def _q32_run(tmp_path, extra_env, upstream_venv_has="upstream"):
    """Real q32/run.sh with fakes. U runs a fake interpreter whose import path holds an upstream stub (or, to simulate a
    broken upstream venv, the fork stub); the fork arms import the fork stub. Each fake server imports vllm_neuron."""
    home = tmp_path / "home"; (home / "models/Qwen3-32B").mkdir(parents=True)
    stub = {}
    for k in ("upstream", "fork"):
        stub[k] = tmp_path / f"stub-{k}"; (stub[k] / "vllm_neuron").mkdir(parents=True)
        (stub[k] / "vllm_neuron/__init__.py").write_text("")
    upy = tmp_path / "upstream-venv-python"
    upy.write_text(f"""#!/bin/bash
shift 2   # drop "-m vllm.entrypoints.cli.main"
export PYTHONPATH="${{PYTHONPATH%%:*}}:{stub[upstream_venv_has]}"   # keep the provenance dir, then this venv's package
exec vllm "$@"
""")
    upy.chmod(0o755)
    run = tmp_path / "run"; run.mkdir()
    env = dict(os.environ, HOME=str(home), PATH=f"{ROOT/'tests/covex/fake_bin'}:{os.environ['PATH']}",
               PYTHONPATH=str(stub["fork"]), UPSTREAM_PY=str(upy), UPSTREAM_DIR=str(stub["upstream"]),
               FORK_DIR=str(stub["fork"]), Q32_MIN_PROCS="1", Q32_WINDOWS="2", **extra_env)
    r = subprocess.run(["bash", str(ROOT / "benchmarks/q32/run.sh")], cwd=run, env=env, capture_output=True, text=True, timeout=900)
    return r, run


@pytest.mark.skipif(not LINUX, reason="Linux CI")
def test_q32_run_success_with_fakes(tmp_path):
    r, run = _q32_run(tmp_path, {})
    res = (run / "RESULTS.md").read_text()
    assert r.returncode == 0 and "RUN OK" in r.stdout, r.stdout[-3000:] + r.stderr[-2000:]
    rows = [l for l in res.splitlines() if l.startswith("| q32-K | q32-U |")]
    assert len(rows) == 2 and all("complete" in l and "x |" in l for l in rows), res
    ev = (run / "evidence/EVIDENCE.tsv").read_text()
    assert all(f"profile-{a}\tOK" in ev for a in ("q32-U", "q32-T", "q32-K")), ev
    st = (run / "results/STATUS.tsv").read_text()
    assert all(f"{a}\tprovenance\tPASS" in st for a in ("q32-U", "q32-T", "q32-K")), st


@pytest.mark.skipif(not LINUX, reason="Linux CI")
def test_q32_run_cache_divergence_suppresses_ratio(tmp_path):
    r, run = _q32_run(tmp_path, {"FAKE_DIVERGE_PORTS": "8102"})       # q32-K greedy outputs differ
    res = (run / "RESULTS.md").read_text()
    assert r.returncode == 1, r.stdout[-2000:]
    row = next(l for l in res.splitlines() if l.startswith("| q32-K | q32-U | c1 |"))
    assert "| – |" in row and "cache c1: outputs differ" in row, row


@pytest.mark.skipif(not LINUX, reason="Linux CI")
def test_q32_run_refuses_upstream_arm_that_imports_the_fork(tmp_path):
    r, run = _q32_run(tmp_path, {}, upstream_venv_has="fork")       # the "upstream" venv actually serves the fork
    st = (run / "results/STATUS.tsv").read_text()
    assert r.returncode == 1 and "q32-U\tprovenance\tFAIL" in st and "load vllm_neuron from" in st, st
    assert "INCOMPLETE (required): q32-K vs q32-U" in (run / "RESULTS.md").read_text()


PINS = ["a" * 40, "b" * 40, "c" * 40]


def test_q32_payload_renders_only_with_exact_pins(tmp_path):
    out = tmp_path / "p.sh"
    r = subprocess.run(["bash", str(ROOT / "benchmarks/q32/render_payload.sh"), *PINS, str(out), "q32-F", "3"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert "@@" not in text
    assert f"checkout -q {'a' * 40}" in text and f"checkout -q {'b' * 40}" in text and f"--revision {'c' * 40}" in text
    assert re.search(r"^shutdown -h \+\d+", text, re.M)
    assert 'P="q32/$RUN_ID"' in text and "date -u" in text, "each attempt needs its own S3 prefix"
    assert '--exclude "evidence/*"' in text and "EVIDENCE_UPLOAD=" in text, "evidence goes up immediately, not via the periodic sync"
    assert "ARM_CACHE_BASE=" in text and "UPSTREAM_PY=~/venv-upstream/bin/python" in text and "manifest.json" in text
    assert 'Q32_EXTRA_ARMS="q32-F"' in text
    # U's venv is copied from the pristine DLAMI venv BEFORE the fork is installed into it
    assert text.index('cp -a "$V" ~/venv-upstream') < text.index("pip install -q --no-deps -e covex-neuron")
    for bad in (["exp/six-model", PINS[1], PINS[2]], [PINS[0], "e16d424", PINS[2]], [PINS[0], PINS[1], "main"]):
        r = subprocess.run(["bash", str(ROOT / "benchmarks/q32/render_payload.sh"), *bad, str(tmp_path / "x.sh")],
                           capture_output=True, text=True)
        assert r.returncode == 2 and not (tmp_path / "x.sh").exists(), f"moving/short ref accepted: {bad}"
    r = subprocess.run(["bash", str(ROOT / "benchmarks/q32/render_payload.sh"), *PINS, str(tmp_path / "y.sh"), "q32-Z"],
                       capture_output=True, text=True)
    assert r.returncode == 2


def test_q32_summary_gates_on_windows_and_cache(tmp_path):
    def bench(n, done, tps): return {"num_prompts": n, "completed": done, "failed": n - done, "total_output_tokens": done * 256,
                                     "total_input_tokens": n * 1024, "output_throughput": tps, "median_tpot_ms": 10.0,
                                     "median_ttft_ms": 50.0}
    for arm, tps in (("q32-U", 100.0), ("q32-T", 100.0), ("q32-K", 150.0), ("q32-F", 180.0)):
        for c, n in (("c1", 8), ("c8", 32)):
            for w in (1, 2):
                done = 7 if (arm, c, w) == ("q32-K", "c8", 2) else n          # one partial window on K c8
                (tmp_path / f"{arm}-{c}-w{w}.json").write_text(json.dumps(bench(n, done, tps)))
    for ref in ("q32-U", "q32-T"):
        for k in (1, 3):
            (tmp_path / f"q32-K.vs.{ref}.cache.c{k}.match.json").write_text(json.dumps({"pass": True, "n": 18, "exact_match": "18/18"}))
            (tmp_path / f"q32-F.vs.{ref}.cache.c{k}.match.json").write_text(json.dumps({"pass": False, "n": 18, "exact_match": "16/18"}))
    (tmp_path / "q32-T.vs.q32-U.cache.c1.match.json").write_text(json.dumps({"pass": True, "n": 18}))
    st = tmp_path / "STATUS.tsv"; st.write_text("q32-T\tready\tOK\n")
    r = subprocess.run(["python3", str(ROOT / "benchmarks/q32/summarize.py"), str(tmp_path), "--arms", "q32-U q32-T q32-K q32-F",
                        "--windows", "2", "--status", str(st)], capture_output=True, text=True)
    rows = {tuple(c.strip() for c in l.split("|")[1:4]): l for l in r.stdout.splitlines() if l.startswith("| q32-K | q32-")}
    assert "1.50x" in rows[("q32-K", "q32-U", "c1")], r.stdout
    assert "| – |" in rows[("q32-K", "q32-U", "c8")] and "w2: completed 7/32" in rows[("q32-K", "q32-U", "c8")], r.stdout
    assert r.returncode == 1 and "INCOMPLETE (required): q32-K vs q32-U c8" in r.stdout, r.stdout
    f_row = next(l for l in r.stdout.splitlines() if l.startswith("| q32-F | q32-U | c1 |"))
    assert "1.80x (exploratory)" in f_row and "tokens differ" in f_row, f_row       # visible, labelled, not headline
    # a failed run step on the reference arm removes every ratio that depends on it
    st.write_text("q32-U\tprofile\tFAILED start\n")
    r = subprocess.run(["python3", str(ROOT / "benchmarks/q32/summarize.py"), str(tmp_path), "--arms", "q32-U q32-T q32-K",
                        "--windows", "2", "--status", str(st)], capture_output=True, text=True)
    row = next(l for l in r.stdout.splitlines() if l.startswith("| q32-K | q32-U | c1 |"))
    assert "| – |" in row and "q32-U step failed: profile FAILED start" in row, row
    # zero windows and missing throughput are rejected, never a pass or a crash
    z = subprocess.run(["python3", str(ROOT / "benchmarks/q32/summarize.py"), str(tmp_path), "--arms", "q32-U q32-T q32-K",
                        "--windows", "0"], capture_output=True, text=True)
    assert z.returncode != 0
    j = json.loads((tmp_path / "q32-K-c1-w1.json").read_text()); j.pop("output_throughput")
    (tmp_path / "q32-K-c1-w1.json").write_text(json.dumps(j))
    r = subprocess.run(["python3", str(ROOT / "benchmarks/q32/summarize.py"), str(tmp_path), "--arms", "q32-U q32-T q32-K",
                        "--windows", "2"], capture_output=True, text=True)
    assert "Traceback" not in r.stderr and "invalid" in r.stdout and r.returncode == 1, r.stdout + r.stderr


@pytest.mark.skipif(not LINUX, reason="Linux CI")
def test_q32_run_failed_evidence_upload_makes_run_incomplete(tmp_path):
    r, run = _q32_run(tmp_path, {"EVIDENCE_UPLOAD": "false"})        # every upload fails
    assert r.returncode == 1 and "profile evidence incomplete" in r.stdout, r.stdout[-2000:]
    assert "FAILED rc=2" in (run / "evidence/EVIDENCE.tsv").read_text()



def _prov_case(tmp_path, arm, cwd_is_fork, extra_env=None):
    up, fork, prov = tmp_path / "up", tmp_path / "fork", tmp_path / "prov"
    for r in (up, fork):
        (r / "vllm_neuron").mkdir(parents=True, exist_ok=True); (r / "vllm_neuron/__init__.py").write_text("")
    neutral = tmp_path / "neutral"; neutral.mkdir(exist_ok=True)
    env = dict(os.environ, COVEX_PROVENANCE_DIR=str(prov), COVEX_ARM=arm, NEURON_LIBTORCH_CACHE_ROOT=f"/cache/{arm}",
               PYTHONPATH=f"{ROOT/'benchmarks/provenance'}:{up}")
    env.update(extra_env or {})
    subprocess.run(["python3", "-c", "import vllm_neuron"], cwd=fork if cwd_is_fork else neutral, env=env, check=True)
    return up, fork, prov


def _verify(prov, *pairs, n=1):
    r = subprocess.run(["python3", str(ROOT / "benchmarks/provenance/verify.py"), str(prov), str(n), *pairs],
                       capture_output=True, text=True)
    return r.returncode, r.stdout


def test_provenance_records_the_tree_actually_imported(tmp_path):
    up, fork, prov = _prov_case(tmp_path, "q32-U", cwd_is_fork=False)
    assert _verify(prov, f"q32-U={up}")[0] == 0
    # the cwd trap: launched from the fork checkout, `python -c` imports the fork although upstream is on PYTHONPATH
    up, fork, prov2 = _prov_case(tmp_path / "b", "q32-U", cwd_is_fork=True)
    rc, out = _verify(prov2, f"q32-U={up}")
    assert rc == 1 and "load vllm_neuron from" in out and "/fork/" in out, out


def test_provenance_rejects_flags_shared_cache_and_missing_processes(tmp_path):
    up, fork, prov = _prov_case(tmp_path, "q32-T", cwd_is_fork=False, extra_env={"COVEX_QWEN3_INKERNEL_KV_WRITE": "1"})
    rc, out = _verify(prov, f"q32-T={up}")
    assert rc == 1 and "COVEX flags" in out, out                       # T must run with no COVEX flag
    up, fork, prov = _prov_case(tmp_path / "k", "q32-K", cwd_is_fork=False)
    rc, out = _verify(prov, f"q32-K={up}")
    assert rc == 1 and "COVEX flags" in out, out                       # K must have its flag
    up, fork, prov = _prov_case(tmp_path / "s", "q32-U", cwd_is_fork=False)
    _prov_case(tmp_path / "s", "q32-T", cwd_is_fork=False, extra_env={"NEURON_LIBTORCH_CACHE_ROOT": "/cache/q32-U"})
    rc, out = _verify(prov, f"q32-U={up}", f"q32-T={up}")
    assert rc == 1 and out.count("shares a compile-cache path") == 2, out      # both arms fail
    rc, out = _verify(prov, f"q32-U={up}", n=9)
    assert rc == 1 and "process records < 9" in out, out


@pytest.mark.skipif(not LINUX, reason="Linux CI")
def test_rehearsal_greedy_mismatch_fails_overall_while_servers_are_healthy(tmp_path):
    home = tmp_path / "home"
    for d in ["models/Qwen3-32B", "models-small/Qwen3-32B"]:
        (home / d).mkdir(parents=True)
    (home / "models/Qwen3-32B/.done").touch(); (home / "models/Qwen3-32B/config.json").write_text('{"num_hidden_layers": 2}')
    run = tmp_path / "run"; run.mkdir()
    env = _rehearsal_env(tmp_path, home, {"REHEARSAL_PAIRS_B": "q32-K q32-T;q32-U q32-C",
                                          "FAKE_DIVERGE_PORTS": "8109"})   # 8109 = q32-K in CPU mode: answers differ
    r = subprocess.run(["bash", str(ROOT / "benchmarks/rehearsal/rehearse.sh")], cwd=run, env=env,
                       capture_output=True, text=True, timeout=600)
    tsv = (run / "REHEARSAL.tsv").read_text(); md = (run / "REHEARSAL.md").read_text()
    assert "q32-K\tcpu\tPASS" in tsv, tsv                             # the server itself served fine
    assert "q32-K\tcpu-match\tFAIL" in tsv and "q32-U\tcpu-match\tPASS" in tsv, tsv
    assert r.returncode == 1 and "REQUIRED q32-K:cpu-match: FAIL" in md, md


def _summ(dirpath, required):
    r = subprocess.run(["python3", str(ROOT / "benchmarks/five/summarize.py"), str(dirpath), "--required", required],
                       capture_output=True, text=True)
    return r.returncode, r.stdout


def test_summary_rejects_missing_numbers_wrong_counts_and_unequal_work(tmp_path):
    def case(name, mutate):
        d = tmp_path / name; shutil.copytree(SUMM / "ok", d)
        for f in d.glob("q32-*-c*.json"):
            j = json.loads(f.read_text()); mutate(f.name, j); f.write_text(json.dumps(j))
        return _summ(d, "q32-T q32-C")
    rc, out = case("nothroughput", lambda n, j: j.pop("output_throughput"))
    assert rc == 1 and "missing or non-positive output_throughput" in out, out
    rc, out = case("unequal", lambda n, j: j.update(total_input_tokens=1000 if n.startswith("q32-T") else 999))
    assert rc == 1 and "unequal work" in out, out
    d = tmp_path / "fewmatch"; shutil.copytree(SUMM / "ok", d)
    (d / "q32.match.json").write_text(json.dumps({"exact_match": "1/1", "n": 1, "pass": True}))
    rc, out = _summ(d, "q32-T q32-C")
    assert rc == 1 and "correctness covered n=1, expected 20" in out, out


def test_summary_asr_required_and_quality_qualified(tmp_path):
    rc, out = _summ(tmp_path, "asr-T asr-C")                       # no ASR results at all
    assert rc == 1 and "INCOMPLETE (required): Qwen3-ASR-1.7B c1" in out, out
    def arm(wer, aps):
        return {"phases": {"transcribe": [dict(n_total=100, n_success=100, n_error=0, wer_mean=wer, concurrency=c,
                                               audio_seconds_per_s=aps) for c in (1, 8)]}}
    (tmp_path / "asr-T.json").write_text(json.dumps(arm(0.03, 100.0)))
    (tmp_path / "asr-C.json").write_text(json.dumps(arm(1.00, 200.0)))
    rc, out = _summ(tmp_path, "asr-T asr-C")
    row = next(l for l in out.splitlines() if l.startswith("| Qwen3-ASR-1.7B | c1"))
    assert rc == 1 and "| – |" in row and "quality: WER 3.00% (T) vs 100.00% (C)" in row, row
    (tmp_path / "asr-C.json").write_text(json.dumps(arm(0.031, 200.0)))
    rc, out = _summ(tmp_path, "asr-T asr-C")
    row = next(l for l in out.splitlines() if l.startswith("| Qwen3-ASR-1.7B | c1"))
    assert rc == 0 and "2.00x" in row and "WER 3.00% vs 3.10%" in row, row



@pytest.mark.skipif(not LINUX, reason="Linux CI")
def test_five_run_asr_evidence_from_the_asr_launcher(tmp_path):
    """Real benchmarks/asr/serve.sh (S2 / DL, tags -T / -C) under the combined runner: evidence must hold its files."""
    r, run = _five_run(tmp_path, {"FIVE_ARMS": "asr-T asr-C", "REQUIRED_ARMS": "asr-T asr-C"})
    assert r.returncode == 0 and "RUN OK" in r.stdout, r.stdout[-3000:] + r.stderr[-2000:]
    for arm, base in (("asr-T", "S2-T"), ("asr-C", "DL-C")):
        ev = run / f"evidence/profile-{arm}"
        assert (ev / base.split("-")[0] / "rank0.ntff").exists() and (ev / f"{base}.server.log").exists(), list(ev.rglob("*"))
    assert "Qwen3-ASR-1.7B | c1" in (run / "RESULTS.md").read_text()


@pytest.mark.skipif(not LINUX, reason="Linux CI")
def test_five_run_required_profile_failure_fails_run(tmp_path):
    r, run = _five_run(tmp_path, {"FAKE_FAIL_START_PROFILE_PORTS": "8000"})   # q32-T cannot start profiling
    st = (run / "results/STATUS.tsv").read_text()
    assert "q32-T\tbench-c1\tOK" in st, st                                 # throughput itself succeeded
    assert r.returncode == 1 and "REQUIRED q32-T profile: FAILED start_profile" in r.stdout, r.stdout[-2000:]


def test_stage_table_never_uses_lifetime_counters(tmp_path):
    """Astra's repro: empty before-scrape + 50 measured requests used to print a 500 ms 'server E2E' from 1,100 lifetime
    requests. It must be unavailable instead; a valid pair must still give numbers."""
    prom = lambda n, s: f"vllm:e2e_request_latency_seconds_sum {s}\nvllm:e2e_request_latency_seconds_count {n}\n"
    cl = tmp_path / "c.json"
    cl.write_text(json.dumps({"phases": {"transcribe": [{"concurrency": 8, "n_success": 50, "e2e_mean_s": 0.6}]}}))
    (tmp_path / "empty.prom").write_text(""); (tmp_path / "after.prom").write_text(prom(1100, 550.0))
    run = lambda b, a: subprocess.run(["python3", str(ROOT / "benchmarks/asr/stage_table.py"), str(b), str(a), str(cl), "8"],
                                      capture_output=True, text=True, check=True).stdout
    out = run(tmp_path / "empty.prom", tmp_path / "after.prom")
    assert "stage attribution unavailable: missing scrape" in out and "500.0" not in out, out
    (tmp_path / "before.prom").write_text(prom(1050, 520.0))
    out = run(tmp_path / "before.prom", tmp_path / "after.prom")       # 50 requests, 30 s -> 600 ms server E2E
    assert "| 8 | 600.0 | 0.0 |" in out, out


def test_bench_merge_requires_one_result_per_phase_and_level(tmp_path):
    src = (ROOT / "benchmarks/asr/bench.sh").read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    (tmp_path / "part.json").write_text(json.dumps({"phases": {}, "raw_results": []}))
    r = subprocess.run(["python3", "-", "out.json", "4", "1,8", "part.json"], cwd=tmp_path, input=src, text=True,
                       capture_output=True)
    v = json.loads((tmp_path / "out.json").read_text())["validation"]["problems"]
    assert r.returncode == 4 and any("transcribe c1: 0 results" in p for p in v), (r.stderr, v)



@pytest.mark.skipif(not LINUX, reason="Linux CI")
def test_rehearsal_fails_when_upstream_arm_runs_the_fork(tmp_path):
    home = tmp_path / "home"
    for d in ["models/Qwen3-32B", "models-small/Qwen3-32B"]:
        (home / d).mkdir(parents=True)
    (home / "models/Qwen3-32B/.done").touch(); (home / "models/Qwen3-32B/config.json").write_text('{"num_hidden_layers": 2}')
    run = tmp_path / "run"; run.mkdir()
    env = _rehearsal_env(tmp_path, home, {"REHEARSAL_PAIRS_B": "q32-K q32-T;q32-U q32-C"})
    env["UPSTREAM_PY"] = str(ROOT / "tests/covex/fake_bin/python_as_vllm")   # keeps the fork stub on the path
    r = subprocess.run(["bash", str(ROOT / "benchmarks/rehearsal/rehearse.sh")], cwd=run, env=env, capture_output=True,
                       text=True, timeout=600)
    md = (run / "REHEARSAL.md").read_text()
    assert r.returncode == 1 and "REQUIRED q32-U:provenance: FAIL" in md, md


def test_payload_clone_urls_are_the_real_repositories():
    """aws-neuron/vllm-neuron does not exist; upstream is vllm-project/vllm-neuron (this repo's `upstream` remote)."""
    allowed = {"https://github.com/vllm-project/vllm-neuron", "https://github.com/HolboxAI/covex-neuron",
               "https://github.com/HolboxAI/qwen3-asr-inference-cuda-trainium"}
    for p in (ROOT / "benchmarks").rglob("*.sh"):
        for url in re.findall(r"git clone [^\n]*?(https://github\.com/[\w.-]+/[\w.-]+)", p.read_text()):
            assert url in allowed, f"{p.relative_to(ROOT)} clones {url}"



def test_cachecheck_compares_token_ids_not_text(tmp_path):
    """Equal decoded text through different token ids must not count as agreement; missing ids must fail loudly."""
    import time
    ports = {"same": 8951, "ids": 8952, "noids": 8953}
    envs = {"same": {}, "ids": {"FAKE_DIVERGE_IDS_PORTS": "8952"}, "noids": {"FAKE_NO_TOKEN_IDS_PORTS": "8953"}}
    procs = [subprocess.Popen(["python3", str(FAKE), "--port", str(p)], env=dict(os.environ, **envs[k])) for k, p in ports.items()]
    try:
        for p in ports.values():
            for _ in range(50):
                if subprocess.run(["curl", "-sf", f"localhost:{p}/health"], capture_output=True).returncode == 0:
                    break
                time.sleep(0.2)
        cc = lambda p, out: subprocess.run(["python3", str(ROOT / "benchmarks/five/clients.py"), "cachecheck", str(p), str(out), "3", "32"],
                                           capture_output=True, text=True)
        assert cc(ports["same"], tmp_path / "a.json").returncode == 0
        assert cc(ports["ids"], tmp_path / "b.json").returncode == 0
        r = cc(ports["noids"], tmp_path / "c.json")
        assert r.returncode != 0 and "expected 32 token ids" in r.stderr, r.stderr
        a = json.loads((tmp_path / "a.json").read_text())
        assert len(a["token_ids"]) == 18 and all(len(x) == 32 for x in a["token_ids"])
        m = lambda x, y: json.loads(subprocess.run(["python3", str(ROOT / "benchmarks/five/clients.py"), "match", str(x), str(y)],
                                                   capture_output=True, text=True, check=True).stdout)
        assert m(tmp_path / "a.json", tmp_path / "a.json")["pass"] is True
        assert m(tmp_path / "a.json", tmp_path / "b.json")["pass"] is False      # same text "ok", different ids
        short = json.loads((tmp_path / "a.json").read_text()); short["token_ids"][3] = short["token_ids"][3][:31]
        (tmp_path / "d.json").write_text(json.dumps(short))
        assert m(tmp_path / "d.json", tmp_path / "d.json")["pass"] is False      # one prompt with 31 ids
    finally:
        for p in procs:
            p.terminate()


@pytest.mark.skipif(not LINUX, reason="Linux CI")
def test_rehearsal_missing_tp_rank_fails_alias(tmp_path):
    home = tmp_path / "home"
    for d in ["models/Qwen3-32B", "models-small/Qwen3-32B"]:
        (home / d).mkdir(parents=True)
    (home / "models/Qwen3-32B/.done").touch(); (home / "models/Qwen3-32B/config.json").write_text('{"num_hidden_layers": 2}')
    run = tmp_path / "run"; run.mkdir()
    env = _rehearsal_env(tmp_path, home, {"REHEARSAL_PAIRS_A": " ", "REHEARSAL_PAIRS_B": "q32-K q32-T",
                                          "FAKE_DROP_RANK_PORTS": "8209"})    # q32-K compile: lane 3 rank 5 never logs
    r = subprocess.run(["bash", str(ROOT / "benchmarks/rehearsal/rehearse.sh")], cwd=run, env=env, capture_output=True,
                       text=True, timeout=600)
    tsv = (run / "REHEARSAL.tsv").read_text()
    assert "q32-K\tcompile-alias\tFAIL\tmissing lane/rank [(3, 5)]" in tsv and "q32-T\tcompile-alias\tPASS" in tsv, tsv
    assert r.returncode == 1



def test_provenance_malformed_record_prefix_and_missing_tp_rank(tmp_path):
    up, fork, prov = _prov_case(tmp_path, "q32-U", cwd_is_fork=False)
    (prov / "q32-U.999.json").write_text("{not json")
    rc, out = _verify(prov, f"q32-U={up}")
    assert rc == 1 and "malformed provenance records" in out, out
    up, fork, prov = _prov_case(tmp_path / "p", "q32-U", cwd_is_fork=False)
    rc, out = _verify(prov, f"q32-U={up}@/opt/wrong-venv")
    assert rc == 1 and "sys.prefix" in out, out
    # TP ranks come from vLLM's log: surplus helper records cannot replace a missing worker
    up, fork, prov = _prov_case(tmp_path / "r", "q32-U", cwd_is_fork=False)
    pid = json.loads(next(prov.glob("*.json")).read_text())["pid"]
    for extra in (1, 2, 3):   # surplus helper processes
        (prov / f"q32-U.{pid + extra}.json").write_text(next(prov.glob("*.json")).read_text().replace(str(pid), str(pid + extra), 1))
    log = tmp_path / "u.log"
    log.write_text(f"(EngineCore pid={pid}) x\n" + "".join(f"(Worker_TP{i} pid={pid if i != 5 else 4242}) x\n" for i in range(8)))
    r = subprocess.run(["python3", str(ROOT / "benchmarks/provenance/verify.py"), str(prov), "1", "--tp", "8",
                        "--log", f"q32-U={log}", f"q32-U={up}"], capture_output=True, text=True)
    assert r.returncode == 1 and "no provenance record for ['TP5']" in r.stdout, r.stdout


@pytest.mark.parametrize("fault,expect", [("FAKE_PROFILE_EMPTY_PORTS", "0 non-empty trace files < 8"),
                                          ("FAKE_PROFILE_DROP_RANK_PORTS", "7 non-empty trace files < 8"),
                                          ("FAKE_NO_NEFF_PORTS", "FAILED neff: no non-empty files")])
@pytest.mark.skipif(not LINUX, reason="Linux CI")
def test_q32_run_rejects_missing_profile_artifacts(tmp_path, fault, expect):
    r, run = _q32_run(tmp_path, {fault: "8102"})                       # q32-K: profile endpoints succeed, artifacts missing
    st = (run / "results/STATUS.tsv").read_text()
    assert r.returncode == 1 and expect in st, st



def test_q32_summary_requested_exploratory_arm_must_have_results(tmp_path):
    def bench(n): return {"num_prompts": n, "completed": n, "failed": 0, "total_output_tokens": n * 256,
                          "total_input_tokens": n * 1024, "output_throughput": 100.0, "median_tpot_ms": 10.0, "median_ttft_ms": 50.0}
    for arm in ("q32-U", "q32-T", "q32-K"):                     # q32-F requested but produced nothing
        for c, n in (("c1", 8), ("c8", 32)):
            (tmp_path / f"{arm}-{c}-w1.json").write_text(json.dumps(bench(n)))
    for ref in ("q32-U", "q32-T"):
        for k in (1, 3):
            (tmp_path / f"q32-K.vs.{ref}.cache.c{k}.match.json").write_text(json.dumps({"pass": True, "n": 18}))
    r = subprocess.run(["python3", str(ROOT / "benchmarks/q32/summarize.py"), str(tmp_path), "--arms", "q32-U q32-T q32-K q32-F",
                        "--windows", "1"], capture_output=True, text=True)
    assert r.returncode == 1 and "q32-F (requested exploratory arm) c1" in r.stdout, r.stdout
    r = subprocess.run(["python3", str(ROOT / "benchmarks/q32/summarize.py"), str(tmp_path), "--arms", "q32-U q32-T q32-K",
                        "--windows", "1"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout                          # not requested -> not required


def _ds():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ds", ROOT / "benchmarks/q32/decode_step.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_decode_step_counts_time_once_and_splits_allreduce():
    ds = _ds(); L = 2
    def rank(offset):   # 8 decode executions of 1000 ns; each: attn 0-200, AR 200-300 (100 exposed), mlp 300-700,
        out = []          # AR 650-750 (50 overlapped with mlp, 50 exposed), then the same for layer 2 shifted by 0 (L=2)
        for i in range(8):
            s = i * 2000 + offset
            out.append(("decode_graph", s, 1000, 0))
            for base in (0, 500):
                out += [("attention_tkg", s + base, 200, 1), ("cc_all_reduce", s + base + 200, 50, 1),
                        ("mlp_tkg", s + base + 250, 150, 1), ("cc_all_reduce", s + base + 380, 40, 1)]
        return out
    ranks = {r: rank(offset=r * 10) for r in range(8)}            # ranks skewed by 10 ns each
    out = ds.analyze(ranks, L)
    assert abs(out["step_ms"] - (1000 + 70) / 1e6) < 1e-12, out     # once across ranks: 1000 + max skew 70, not 8x
    assert out["attn_ms"] != "unknown" and out["mlp_ms"] != "unknown", out
    assert out["exposed_ms"] >= 0 and out["overlap_ms"] >= 0


def test_decode_step_marks_unknown_when_trace_cannot_resolve():
    ds = _ds()
    ranks = {r: [("decode_graph", i * 2000, 1000, 0) for i in range(8)] + [("matmul", i * 2000 + 10, 500, 1) for i in range(8)]
             for r in range(8)}
    out = ds.analyze(ranks, 64)
    assert out["exposed_ms"] == "unknown" and "no slice name matches a collective" in out["note"]
    ranks = {r: [("decode_graph", i * 2000, 1000, 0) for i in range(8)] + [("cc_all_reduce", i * 2000 + 10, 5, 1) for i in range(8)]
             for r in range(8)}
    out = ds.analyze(ranks, 64)
    assert out["attn_ms"] == "unknown" and "split unknown" in out["note"], out


def test_packed_prefill_max_query_len_is_one_value_per_bucket():
    """emb-C crashed with fail_on_recompile: the compiled guard was max_query_len == 128 and packed steps passed the
    longest request's padded length instead. Every packed prefill in a bucket must carry the same value; stock
    (single prefill) and decode must be unchanged."""
    import importlib.util, types, sys as _s
    src = (ROOT / "vllm_neuron/utils/bucket_utils.py").read_text()
    fn_src = src[src.index("def canonical_prefill_max_query_len"):]
    ns = {}; exec(fn_src, ns); f = ns["canonical_prefill_max_query_len"]
    bucket = 128
    assert {f(m, bucket, 1, True) for m in (17, 37, 64, 100, 128)} == {bucket}     # any mix -> one value
    assert f(128, 128, 1, False) == 128 and f(37, 128, 1, False) == 37             # single prefill unchanged
    assert f(1, 8, 1, True) == 1 and f(3, 8, 3, True) == 3                         # decode unchanged (incl. spec)
    runner = (ROOT / "vllm_neuron/vllm/worker/neuron_model_runner.py").read_text()
    assert runner.count("canonical_prefill_max_query_len(") == 2, "runtime and warmup must use the same rule"
