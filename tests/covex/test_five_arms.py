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


def test_bench_snapshots_after_prewarm_and_per_level():
    text = (ROOT / "benchmarks/asr/bench.sh").read_text()
    assert text.index("prewarm") < text.index(".before.prom"), "warm the server before the first metrics snapshot"
    assert "metrics/$ARM.c$c.$ph.before.prom" in text, "snapshot each concurrency/phase separately"


def test_liveness_uses_pid_not_pgrep_port():
    """pgrep -f '--port N' misses a server until serve.sh execs vllm; the 2026-11-27 rehearsal killed starting servers."""
    for script in [RUN, ROOT / "benchmarks/rehearsal/rehearse.sh"]:
        code = "\n".join(l.split("#", 1)[0] for l in script.read_text().splitlines())
        assert not re.search(r"pgrep -f -- \"--port", code), f"{script.name}: liveness via pgrep --port"
        assert "kill -0" in code, f"{script.name}: no PID liveness check"


def test_rehearsal_compile_pass_needs_positive_evidence():
    sh = (ROOT / "benchmarks/rehearsal/rehearse.sh").read_text()
    assert sh.index("Extracting graphs for") < sh.index('record $a compile PASS'), "compile PASS must require graph extraction in the log"


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
