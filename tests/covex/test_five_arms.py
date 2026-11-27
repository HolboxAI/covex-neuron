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
ARMS = ["q32-T", "q32-C", "emb-T", "emb-C", "vl-T", "vl-C", "oss-T"]
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
    cfg = json.loads(out.split("cfg=", 1)[1])
    text = SERVE.read_text()
    block = re.search(rf"\n  [^\n]*{re.escape(arm)}[^\n]*\)(.*?);;", text, re.S).group(1)
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
        assert arm in sh, f"rehearsal does not cover {arm}"
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


def test_rehearsal_harness_end_to_end_with_fake_vllm(tmp_path):
    """Runs the real rehearse.sh (q32 arms, both tracks) against a fake vllm that starts 3 s late.
    Catches harness bugs (start-up race, empty-log PASS, wrong waits) before they cost an instance."""
    import sys as _sys
    if _sys.platform != "linux" or not shutil.which("taskset"):
        pytest.skip("needs Linux bash>=4 and taskset (runs in CI)")
    home = tmp_path / "home"
    for d in ["models/Qwen3-32B", "models-small/Qwen3-32B"]:
        (home / d).mkdir(parents=True)
    (home / "models/Qwen3-32B/.done").touch()
    (home / "models/Qwen3-32B/config.json").write_text('{"num_hidden_layers": 2}')
    run = tmp_path / "run"; run.mkdir()
    env = {"HOME": str(home), "PATH": f"{ROOT/'tests/covex/fake_bin'}:{os.environ['PATH']}",
           "REHEARSAL_PAIRS_A": "q32-T q32-C", "REHEARSAL_PAIRS_B": "q32-C q32-T",
           "FAKE_LAYERS": "2", "FAKE_ALIAS_BUG_PORTS": "8201"}   # 8201 = q32-C compile: must be caught
    r = subprocess.run(["bash", str(ROOT / "benchmarks/rehearsal/rehearse.sh")], cwd=run, env=env,
                       capture_output=True, text=True, timeout=600)
    tsv = (run / "REHEARSAL.tsv").read_text()
    logs = "\n".join(f"--- {f.name}\n{f.read_text()[-1500:]}" for f in sorted((run / "logs").glob("*.log")))
    rows = {(l.split("\t")[0], l.split("\t")[1]): l.split("\t")[2] for l in tsv.splitlines() if l}
    expect = {("q32-T", "cpu"), ("q32-C", "cpu"), ("q32-C", "compile"), ("q32-T", "compile"),
              ("q32-C", "compile-alias"), ("q32-T", "compile-alias")}
    assert expect <= rows.keys(), f"missing rows: {expect - rows.keys()}\n{tsv}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    bad = {k: v for k, v in rows.items() if v != "PASS"}
    assert bad == {("q32-C", "compile-alias"): "FAIL"}, f"expected only the injected alias bug to fail, got {bad}\n{tsv}\n{logs}"
    assert (run / "REHEARSAL.md").exists()
    assert subprocess.run(["pgrep", "-f", "fake_server.py"], capture_output=True).returncode != 0, "servers left running"


# ---------------- behaviour tests (run real code against fakes; Linux CI) ----------------
LINUX = __import__("sys").platform == "linux" and shutil.which("taskset") is not None
FAKE = ROOT / "tests/covex/fake_server.py"


def _bash(script, env=None, timeout=120):
    e = dict(os.environ, **(env or {}))
    return subprocess.run(["bash", "-c", f"source {ROOT/'benchmarks/lib.sh'}\n{script}"], env=e, capture_output=True,
                          text=True, timeout=timeout)


def test_alias_check_on_real_logs():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ac", ROOT / "benchmarks/rehearsal/alias_check.py")
    ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)
    fx = ROOT / "tests/covex/fixtures"
    ok, _ = ac.check((fx / "alias_stock_ok.log").read_text(), 64, 6)
    assert ok, "stock Qwen3-32B compile log (129 mutated inputs in all 6 lanes) must pass"
    ok, detail = ac.check((fx / "alias_inkernel_squeeze_bug.log").read_text(), 64, 6)
    assert not ok and "{2: 1, 3: 1, 4: 1, 5: 1}" in detail, detail
    ok, detail = ac.check("\n".join((fx / "alias_stock_ok.log").read_text().splitlines()[:5]), 64, 6)
    assert not ok and "missing lanes" in detail, "a missing graph lane must fail"


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
