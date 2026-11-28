"""Qwen3-VL-8B lane: arm configs, padding / stage accounting and the gated ratio."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_vl8_arms_differ_only_in_buckets_and_engine():
    out = {}
    for a in ("vl8-U", "vl8-Ub", "vl8-C"):
        r = subprocess.run(["bash", str(ROOT / "benchmarks/five/serve.sh"), a, "8400"], capture_output=True, text=True,
                           env=dict(os.environ, DRYRUN="1", UPSTREAM_PY="/up/python"))
        assert r.returncode == 0, r.stderr
        out[a] = r.stdout
    assert '"num_batched_tokens_buckets":[8192]' in out["vl8-U"]
    cfg = lambda o: o.split("cfg=")[1].split("\n")[0]
    assert cfg(out["vl8-Ub"]) == cfg(out["vl8-C"])                                   # tuned stock == COVEX config
    assert "/up/python" in out["vl8-U"] and "/up/python" in out["vl8-Ub"] and "/up/python" not in out["vl8-C"]
    assert len({o.split("cores=")[1].split()[0] for o in out.values()}) == 3          # one chip each


def _fixture(tmp, tps_c=40.0, answers_c=None, prompt=230, p90_c=2, bad_window=None, tokens_c=None):
    r, m = tmp / "r", tmp / "m"
    r.mkdir(); m.mkdir()
    for arm, tps in (("vl8-U", 10.0), ("vl8-Ub", 38.0), ("vl8-C", tps_c)):
        for w in range(1, 4):
            ans = ["A"] * 40
            if arm == "vl8-C" and answers_c and (bad_window is None or w == bad_window):
                ans = answers_c
            (r / f"{arm}-c4-w{w}.json").write_text(json.dumps({
                "n": 40, "concurrency": 4, "wall_s": 1, "req_per_s": tps / 10, "lat_p50_ms": 1,
                "lat_p90_ms": p90_c if arm == "vl8-C" else 2, "latencies_s": [(p90_c if arm == "vl8-C" else 2) / 1000] * 40,
                "out_tok_per_s": tps, "answers": ans,
                "completion_tokens": (tokens_c if (arm == "vl8-C" and tokens_c) else [1] * 40), "prompt_tokens": [prompt] * 40}))
            for k, v in (("before", 0), ("after", 1)):
                (m / f"{arm}-w{w}.{k}.prom").write_text(
                    f"vllm:request_prefill_time_seconds_sum {v * 4.0}\nvllm:request_prefill_time_seconds_count {v * 40}\n"
                    f"vllm:request_decode_time_seconds_sum {v * 0.4}\nvllm:request_decode_time_seconds_count {v * 40}\n"
                    f"vllm:request_queue_time_seconds_sum 0\nvllm:request_queue_time_seconds_count {v * 40}\n")
    return subprocess.run([sys.executable, str(ROOT / "benchmarks/vl8/summarize.py"), str(r), str(m), "--arms",
                           "vl8-U vl8-Ub vl8-C", "--windows", "3", "--n", "40"], capture_output=True, text=True)


def test_vl8_summary_padding_stages_and_ratios(tmp_path):
    p = _fixture(tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr
    u = next(l for l in p.stdout.splitlines() if l.startswith("| vl8-U |"))
    c = next(l for l in p.stdout.splitlines() if l.startswith("| vl8-C |"))
    assert "35.6x" in u and "1.1x" in c, (u, c)                     # 8192/230 vs 256/230
    assert "0.100 / 0.010 / 0.000" in c, c                          # mean prefill / decode / queue per request
    assert "| vl8-C vs vl8-U | 4.00x | complete |" in p.stdout and "| vl8-C vs vl8-Ub | 1.05x |" in p.stdout


def test_vl8_ratio_withheld_when_answers_differ(tmp_path):
    p = _fixture(tmp_path, answers_c=["B"] * 40)
    assert p.returncode == 1 and "answers differ from vanilla stock on 40/40" in p.stdout


def test_vl8_checks_all_40_answers_in_every_window(tmp_path):
    """Astra: only 20 of 40 answers in window 1 were checked. A difference at request 35 of window 3 must disqualify."""
    late = ["A"] * 35 + ["B"] + ["A"] * 4
    p = _fixture(tmp_path, answers_c=late, bad_window=3)
    assert p.returncode == 1 and "w3: answers differ from vanilla stock on 1/40" in p.stdout, p.stdout


def test_vl8_token_counts_and_p90_limit_gate_the_ratio(tmp_path):
    a = tmp_path / "a"; a.mkdir()
    p = _fixture(a, tokens_c=[1] * 39 + [2])
    assert p.returncode == 1 and "output token counts differ" in p.stdout
    b = tmp_path / "b"; b.mkdir()
    p = _fixture(b, p90_c=3)                                           # 3 ms > 1.10 x 2 ms
    assert p.returncode == 1 and "pooled p90 3 ms > 1.10 x 2 ms" in p.stdout, p.stdout
    c = tmp_path / "c"; c.mkdir()
    p = _fixture(c, p90_c=2.2)                                         # within 10%
    assert p.returncode == 0 and "Short-answer VL-8B" in p.stdout and "stock with an 8,192-token prefill bucket" in p.stdout


import shutil
import pytest

LINUX = sys.platform == "linux" and shutil.which("taskset") is not None


@pytest.mark.skipif(not LINUX, reason="Linux CI")
def test_vl8_run_end_to_end_with_fakes(tmp_path):
    home = tmp_path / "home"; (home / "models/Qwen3-VL-8B-Instruct").mkdir(parents=True)
    data = tmp_path / "vl"; (data / "images").mkdir(parents=True)
    (data / "images/a.jpg").write_bytes(b"\xff\xd8fake")
    (data / "manifest.json").write_text(json.dumps([{"id": i, "image": "a.jpg", "question": f"Q{i}?", "answer": "A"} for i in range(5)]))
    stub = {}
    for k in ("upstream", "fork"):
        stub[k] = tmp_path / f"stub-{k}"; (stub[k] / "vllm_neuron").mkdir(parents=True)
        (stub[k] / "vllm_neuron/__init__.py").write_text("")
    upy = tmp_path / "upy"
    upy.write_text(f"#!/bin/bash\nshift 2\nexport PYTHONPATH=\"${{PYTHONPATH%%:*}}:{stub['upstream']}\"\nexec vllm \"$@\"\n"); upy.chmod(0o755)
    run = tmp_path / "run"; run.mkdir()
    env = dict(os.environ, HOME=str(home), PATH=f"{ROOT/'tests/covex/fake_bin'}:{os.environ['PATH']}", PYTHONPATH=str(stub["fork"]),
               UPSTREAM_PY=str(upy), UPSTREAM_DIR=str(stub["upstream"]), FORK_DIR=str(stub["fork"]), VL8_MIN_PROCS="1",
               VL_DATA=str(data), VL8_WINDOWS="1", VL8_N="8", VL8_READY_SECS="120")
    r = subprocess.run(["bash", str(ROOT / "benchmarks/vl8/run.sh")], cwd=run, env=env, capture_output=True, text=True, timeout=600)
    res = (run / "RESULTS.md").read_text()
    assert "| vl8-C vs vl8-U |" in res and "| vl8-U |" in res, res + r.stdout[-2000:] + r.stderr[-1000:]
    st = (run / "results/STATUS.tsv").read_text()
    assert all(f"{a}\tprovenance\tPASS" in st for a in ("vl8-U", "vl8-Ub", "vl8-C")), st
    assert subprocess.run(["pgrep", "-f", "fake_server.py"], capture_output=True).returncode != 0, "servers left running"


def test_vl8_payload_renders_with_pins_and_cap(tmp_path):
    import re
    out = tmp_path / "p.sh"
    pins = ["a" * 40, "b" * 40, "c" * 40]
    r = subprocess.run(["bash", str(ROOT / "benchmarks/q32/render_payload.sh"), *pins, str(out), "", "3"],
                       env=dict(os.environ, TEMPLATE=str(ROOT / "benchmarks/vl8/payload.sh"), READY_MIN="25"), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t = out.read_text()
    assert "@@" not in t and "benchmarks/vl8/run.sh" in t and 'P="vl8/$RUN_ID"' in t
    assert "T hf download Qwen/Qwen3-VL-8B-Instruct --revision " + "c" * 40 in t and "sixdata/vl/manifest.json" in t
    assert "UPSTREAM_PREFIX=$HOME/venv-upstream FORK_PREFIX=$V" in t
    m = re.search(r"^shutdown -h \+(\d+)", t, re.M)
    assert m and int(m.group(1)) * 8.60 / 60 <= 6.0


def _write(tmp, overrides):
    """vl8-U/Ub/C windows with all-'A' answers; overrides[arm] = dict of field -> value applied to every window."""
    r, m = tmp / "r", tmp / "m"
    r.mkdir(); m.mkdir()
    for arm, tps in (("vl8-U", 10.0), ("vl8-Ub", 38.0), ("vl8-C", 40.0)):
        for w in ["warm", 1, 2, 3]:
            d = {"n": 40, "concurrency": 4, "wall_s": 1, "req_per_s": 1, "lat_p50_ms": 1, "lat_p90_ms": 2, "out_tok_per_s": tps,
                 "latencies_s": [0.002] * 40,
                 "answers": ["A"] * 40, "completion_tokens": [1] * 40, "prompt_tokens": [230] * 40}
            d.update(overrides.get(arm, {}))
            (r / f"{arm}-c4-w{w}.json").write_text(json.dumps(d))
    return r, m


def _summ(r, m, status=None):
    cmd = [sys.executable, str(ROOT / "benchmarks/vl8/summarize.py"), str(r), str(m), "--arms", "vl8-U vl8-Ub vl8-C", "--windows", "3", "--n", "40"]
    if status:
        cmd += ["--status", str(status)]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_vl8_tuned_attribution_requires_tuned_stock_to_be_correct(tmp_path):
    """Astra: tuned stock wrong on every request still gave a 'complete' COVEX-vs-tuned ratio."""
    r, m = _write(tmp_path, {"vl8-Ub": {"answers": ["B"] * 40}})
    p = _summ(r, m)
    row = next(l for l in p.stdout.splitlines() if l.startswith("| vl8-C vs vl8-Ub |"))
    assert "| – |" in row and "vl8-Ub w1: answers differ from vanilla stock on 40/40" in row, row
    assert next(l for l in p.stdout.splitlines() if l.startswith("| vl8-C vs vl8-U |")).split("|")[2].strip().endswith("x")


def test_vl8_blank_answers_are_not_complete(tmp_path):
    """Astra: all three arms returning blank answers with positive token counts were accepted."""
    blank = {"answers": [""] * 40}
    r, m = _write(tmp_path, {"vl8-U": blank, "vl8-Ub": blank, "vl8-C": blank})
    p = _summ(r, m)
    assert p.returncode == 1 and "40/40 blank answers" in p.stdout, p.stdout


def test_vl8_failed_warmup_voids_the_arm(tmp_path):
    """Astra: a failed warm-up was recorded under a generic status the per-arm gate did not read."""
    r, m = _write(tmp_path, {})
    st = tmp_path / "STATUS.tsv"
    st.write_text("vl8-C\tbench-c4-wwarm\tFAILED rc=1\n")
    p = _summ(r, m, st)
    assert p.returncode == 1 and "failed step" in next(l for l in p.stdout.splitlines() if l.startswith("| vl8-C vs vl8-U |"))
    st.write_text("bench\tc4-w2\tFAILED a client failed\n")       # unattributed window failure voids every comparison
    p = _summ(r, m, st)
    assert p.returncode == 1 and "a window failed" in p.stdout


def test_vl8_run_records_per_arm_window_status():
    s = (ROOT / "benchmarks/vl8/run.sh").read_text()
    assert '[ $rc = 0 ] || status $a "bench-c4-w$w"' in s and "stage-probe" in s



def test_vl8_pooled_p90_catches_a_regression_hidden_by_per_window_p90s(tmp_path):
    """Astra: every window's own p90 looked fine, but pooled over all 120 requests COVEX's p90 went 100 -> 250 ms."""
    import json as _j
    base = [0.100] * 40
    slow = [0.050] * 30 + [0.250] * 10          # each window's own p90 sits at the 36th value (0.25 s)...
    r, m = _write(tmp_path, {"vl8-U": {"latencies_s": base, "lat_p90_ms": 100}, "vl8-Ub": {"latencies_s": base, "lat_p90_ms": 100},
                             "vl8-C": {"latencies_s": slow, "lat_p90_ms": 50}})     # ...but the recorded lat_p90_ms lies
    p = _summ(r, m)
    row = next(l for l in p.stdout.splitlines() if l.startswith("| vl8-C vs vl8-U |"))
    assert "| – |" in row and "pooled p90 250 ms > 1.10 x 100 ms" in row, row
    d = _j.loads((r / "vl8-C-c4-w1.json").read_text()); d["latencies_s"] = d["latencies_s"][:39]
    (r / "vl8-C-c4-w1.json").write_text(_j.dumps(d))
    assert "pooled p90 unavailable" in _summ(r, m).stdout
