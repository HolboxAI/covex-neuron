"""Exercise the replay CLI using indexed OpenAI completions and saved reports."""
import copy
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/q32"))
import replay


def run_client(monkeypatch, tmp_path, change=None):
    requests = []

    def post(url, *, json, timeout):
        requests.append(json)
        choices = []
        for i, prompt in enumerate(json["prompt"]):
            choices.append({"index": i, "prompt_token_ids": prompt, "token_ids": [4, 4],
                            "logprobs": {"top_logprobs": [{"token_id:4": -0.1, "token_id:9": -0.15}
                                                         for _ in range(2)]}})
        if change:
            change(choices)
        assert url == "http://local/v1/completions"
        return httpx.Response(200, json={"choices": choices[::-1]}, request=httpx.Request("POST", url))

    monkeypatch.setattr(replay.httpx, "post", post)
    path = tmp_path / "batch.json"
    monkeypatch.setattr(sys, "argv", ["replay.py", "run", "--base-url", "http://local", "--model", "m",
                                       "--tokens", "2", "--out", str(path)])
    return replay.main(), json.loads(path.read_text()), requests


def test_native_batch_reorders_by_index_and_replays_original_prompts(monkeypatch, tmp_path):
    rc, result, requests = run_client(monkeypatch, tmp_path)
    assert rc == 0 and result["completed"] == 8
    body, = requests
    assert [len(ids) for ids in body["prompt"]] == [33, 127, 128, 8190, 31, 32, 129, 511]
    assert body["temperature"] == 0 and body["logprobs"] == 2
    assert body["return_token_ids"] and body["return_tokens_as_token_ids"]
    rows = result["prompts"]
    assert rows[0]["prompt_sha256"] == "fcebafd8debac2d173fc22bedab236a27e94e225f56eea20fc7f0084cbd9b198"
    assert rows[3]["prompt_sha256"] == "5bea8f6236c42a808f058df64b73a7e6454fbd4c198a23c2d300ec8cb613d22a"
    assert replay.make_prompts(1, 1) == [body["prompt"][3]]
    assert replay.make_prompts(3, 1) == body["prompt"][:3]
    assert rows[0]["top2"][0] == [[4, -0.1], [9, -0.15]]


@pytest.mark.parametrize("change", [
    lambda choices: choices.pop(),
    lambda choices: choices[0].update(index=1),
    lambda choices: choices[0].update(prompt_token_ids=[123]),
    lambda choices: choices[0].pop("token_ids"),
    lambda choices: choices[0].update(token_ids=[4]),
    lambda choices: choices[0]["logprobs"].update(top_logprobs=[]),
    lambda choices: choices[0]["logprobs"]["top_logprobs"][0].update({"text": -0.2}),
])
def test_failed_run_replaces_stale_success(monkeypatch, tmp_path, change, capsys):
    (tmp_path / "batch.json").write_text('{"completed":8}')
    rc, result, _ = run_client(monkeypatch, tmp_path, change)
    assert rc != 0 and result["completed"] == 0 and result["reason"]
    assert json.loads(capsys.readouterr().err)["pass"] is False


def compare(monkeypatch, tmp_path, left, right):
    a, b, out = (tmp_path / name for name in ["U.json", "F.json", "F.vs.U.replay.match.json"])
    for path, data in [(a, left), (b, right)]:
        path.write_text(json.dumps(data))
    monkeypatch.setattr(sys, "argv", ["replay.py", "compare", str(a), str(b), "--out", str(out)])
    return replay.main(), json.loads(out.read_text())


def report():
    return {"completed": 1, "prompts": [{"prompt_sha256": "a" * 64, "prompt_len": 10,
            "token_ids": [4, 4, 4], "top2": [[[4, -0.1], [9, -0.15]] for _ in range(3)]}]}


@pytest.mark.parametrize(("other", "second_lp", "near", "gap"), [
    (9, -0.15, True, 0.05), (9, -0.2, False, 0.1), (14, -0.15, False, None)])
def test_first_difference_never_passes(monkeypatch, tmp_path, other, second_lp, near, gap):
    left = report()
    right = copy.deepcopy(left)
    left["prompts"][0]["top2"][1][1][1] = second_lp
    for pos in [1, 2]:
        right["prompts"][0]["token_ids"][pos] = other
        right["prompts"][0]["top2"][pos] = [[other, -0.1], [4, -0.15]]
    rc, result = compare(monkeypatch, tmp_path, left, right)
    assert rc != 0 and result["pass"] is False and result["n"] == 1
    assert result["mismatches"] == [{"i": 0, "first_diff": 1, "a": 4, "b": other, "near_tie": near, "gap": gap}]


def test_self_comparison(monkeypatch, tmp_path):
    rc, result = compare(monkeypatch, tmp_path, report(), report())
    assert rc == 0 and result == {"pass": True, "n": 1, "mismatches": []}


@pytest.mark.parametrize("change", [
    lambda data: data.update(completed=0, prompts=[]),
    lambda data: data.update(completed=8),
    lambda data: data["prompts"][0].update(prompt_sha256="b" * 64),
    lambda data: data["prompts"][0]["top2"][0][0].__setitem__(1, float("nan")),
    lambda data: data["prompts"][0].update(top2=[]),
])
def test_invalid_input_cannot_pass(monkeypatch, tmp_path, change):
    right = report()
    change(right)
    rc, result = compare(monkeypatch, tmp_path, report(), right)
    assert rc != 0 and not result["pass"] and result["reason"]
