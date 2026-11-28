import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "benchmarks/six/run.py"


def test_six_plan_keeps_gpt_diagnostic_separate():
    plan = subprocess.check_output([sys.executable, str(SUITE), "--plan"], text=True)
    assert len(plan.splitlines()) == 7   # + ASR 2 x TP4 replica row (Claude, equal-layout comparator)
    assert "oss-U vs oss-P," in plan and "oss-P8k" not in plan
    diag = subprocess.check_output([sys.executable, str(SUITE), "--plan", "--gpt-padded-diagnostic-only"], text=True)
    assert "diagnostic only" in diag and "oss-U8k vs oss-P8k," in diag
    assert diag.count("wave ") == 1


@pytest.mark.parametrize(("diagnostic", "menu"), [(False, [1024, 8192]), (True, [8192])])
def test_gpt_launch_and_greedy_gate(tmp_path, diagnostic, menu):
    suite = runpy.run_path(str(SUITE))
    model = suite["GPT_PADDED"] if diagnostic else next(m for wave in suite["WAVES"] for m in wave if m.stock == "oss-U")
    from clients import CACHE_LENGTHS, cache_prompts

    places, configs = {}, []
    for arm in (model.stock, model.fork):
        cwd = tmp_path / arm
        cwd.mkdir()
        places[arm] = cwd
        python = "/up/python" if arm == model.stock else "/fork/python"
        env = dict(os.environ, DRYRUN="1", UPSTREAM_PY="/up/python", SUITE_PYTHON=python,
                   COVEX_GPT_OSS_SKIP_PADDED_ROUTES="1", PROFILE="0")
        output = subprocess.check_output(["bash", str(ROOT / "benchmarks/five/serve.sh"), arm, "8610"],
                                         env=env, cwd=cwd, text=True)
        header, line = output.splitlines()
        assert ("COVEX_GPT_OSS_SKIP_PADDED_ROUTES" in header) == (arm == model.fork)
        argv = shlex.split(line.removeprefix("argv:"))
        assert argv[0] == python
        config = json.loads(argv[argv.index("--additional-config") + 1])
        assert config["neuron_config"]["num_batched_tokens_buckets"] == menu
        configs.append(argv[1:])
        for c in model.loads:
            n = 8 if c == 1 else 4*c
            result = dict(num_prompts=n, completed=n, failed=0, max_concurrency=c, total_input_tokens=n*1024,
                          total_output_tokens=n*256, output_throughput=10, median_tpot_ms=5, median_ttft_ms=8,
                          input_lens=[1024]*n, output_lens=[256]*n, generated_texts=["example"]*n)
            result.update(ttfts=[0.010]*n, itls=[[0.250] for _ in range(n)])
            for length in model.input_lengths:
                result.update(input_lens=[length]*n, total_input_tokens=n*length)
                (cwd / suite["window_file"](length, c, 1)).write_text(json.dumps(result))
        for conc in (1, 3):
            probe = dict(lengths=CACHE_LENGTHS, concurrency=conc, gen_tokens=32,
                         prompt_sha256=hashlib.sha256(json.dumps(cache_prompts()).encode()).hexdigest(),
                         token_ids=[[1]*32 for _ in CACHE_LENGTHS], diagnostic_port=arm)
            (cwd / f"cache-c{conc}.json").write_text(json.dumps(probe))
    assert configs[0] == configs[1]
    rows = list(suite["record_rows"](model, places, {}, 1))
    assert [r["ratio"] for r in rows] == [1.0] * len(rows)
    assert all(("diagnostic" in r["status"]) == diagnostic for r in rows)
    window = places[model.fork] / "c1-w1.json"
    original = window.read_text()
    changed = json.loads(original)
    changed["generated_texts"][0] += " mismatch"
    window.write_text(json.dumps(changed))
    assert [r["ratio"] for r in suite["record_rows"](model, places, {}, 1)] == [None] + [1.0] * (len(rows)-1)
    window.write_text(original)
    cache = places[model.fork] / "cache-c3.json"
    different = json.loads(cache.read_text())
    different["token_ids"][-1][-1] = 2
    cache.write_text(json.dumps(different))
    assert all(r["ratio"] is None for r in suite["record_rows"](model, places, {}, 1))
    for arm in (model.stock, model.fork):
        (places[arm] / "cache-c3.json").write_text((places[arm] / "cache-c1.json").read_text())
    assert all("wrong or missing" in r["status"] and r["ratio"] is None
               for r in suite["record_rows"](model, places, {}, 1))
    cache.unlink()
    assert all(r["stock"] == r["covex"] == 10 and r["ratio"] is None and "cache-c3.json" in r["status"]
               for r in suite["record_rows"](model, places, {}, 1))
