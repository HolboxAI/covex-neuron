import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/q32"))
import boundary_pages
import boundary_replay


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_replay_follows_same_prefix_and_retains_http_failures():
    count = 0
    expected = boundary_replay.make_prompts(1, 0)[0]

    def handle(request):
        nonlocal count
        count += 1
        body = json.loads(request.content)
        assert body["prompt"] == expected and len(body["prompt"]) == 8190
        assert body["temperature"] == 0 and body["ignore_eos"]
        scored = "logprobs" in body
        if count == 8:
            return httpx.Response(500)
        last = 319 if request.url.host == "fork" else 871
        ids = [77] * 8 + [last]
        choice = {"index": 0, "prompt_token_ids": expected, "token_ids": ids}
        if scored:
            choice["logprobs"] = {"top_logprobs": [{f"token_id:{tok}": -1.0, "token_id:9": -1.1}
                                                   for tok in ids]}
        return httpx.Response(200, json={"choices": [choice], "id": f"cmpl-{count}"})

    args = SimpleNamespace(stock="http://stock", fork="http://fork", model="m", tokens=9, repeat=2)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        report = boundary_replay.run(client, args)
    assert not report["complete"] and set(report["errors"]) == {"F4/scored/1"}
    comparison = next(x for x in report["comparisons"] if x["left"] == "U/scored/0"
                      and x["right"] == "F4/scored/0")
    diff = comparison["first_difference"]
    assert (diff["token_index"], diff["predicting_position"], diff["a"], diff["b"]) == (8, 8197, 871, 319)
    assert diff["a_top2"][0][0] == 871 and diff["b_top2"][0][0] == 319


def test_host_pages_require_every_rank_and_step_and_check_real_write_slots(tmp_path):
    trace = load_module("boundary_trace", "vllm_neuron/vllm/worker/qwen_boundary_trace.py")
    observations = {"U/plain/0": {"request_id": "cmpl-a"}, "U/scored/0": {"request_id": "cmpl-b"}}
    report = {"repeats": 1, "prompt_length": 8190, "tokens": 9, "observations": observations}
    table = np.random.default_rng(123).permutation(300).astype(np.int32)
    lines = []
    for request in observations.values():
        for rank in range(8):
            for pos in range(8190, 8198):
                page, offset = divmod(pos, 32)
                r = trace.page_record(table, 258, 32, pos, int(table[page]) * 32 + offset)
                r.update(request_id=request["request_id"] + "-0-worker", tp_rank=rank, group=0,
                         active_rows=1, row_bucket=1)
                lines.append("worker COVEX_QWEN_BOUNDARY " + json.dumps(r))
    log = tmp_path / "server.log"
    log.write_text("\n".join(lines + lines))
    assert boundary_pages.audit(log, report, "U")["pass"]
    log.write_text("\n".join(lines[1:]))
    assert "missing positions [8190]" in str(boundary_pages.audit(log, report, "U")["issues"])
    prefix, event = lines[-1].split("COVEX_QWEN_BOUNDARY ")
    wrong = json.loads(event)
    wrong["observed_write_slot"] += 1
    log.write_text("\n".join(lines[:-1] + [prefix + "COVEX_QWEN_BOUNDARY " + json.dumps(wrong)]))
    assert "invalid page, slot" in str(boundary_pages.audit(log, report, "U")["issues"])


def test_campaign_retains_timed_windows_when_tuned_logprobs_fail(monkeypatch, tmp_path):
    campaign = load_module("campaign_boundary_test", "benchmarks/six/run.py")
    calls = []

    def command(argv, *unused):
        calls.append(argv)
        if any(str(a).endswith("/replay.py") for a in argv):
            raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(campaign, "command", command)
    model = campaign.WAVES[0][0]
    with pytest.raises(subprocess.CalledProcessError):
        campaign.measure(model, "q32-Ut", 8601, tmp_path, {"FORK_PY": "python"}, tmp_path, tmp_path, 3, "0-7", 60)
    assert any(any(a.endswith("/needles.py") for a in cmd) for cmd in calls)
    dest = [cmd[cmd.index("--result-filename") + 1] for cmd in calls if "--result-filename" in cmd]
    assert dest == [f"c{c}-w{w}.json" for c in [1, 8] for w in ["warm", 1, 2, 3]]
