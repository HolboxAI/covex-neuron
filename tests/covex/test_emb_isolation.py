"""The isolation probe must catch peer-dependent vectors and invalid batches."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


SOURCE = Path(__file__).resolve().parents[2] / "benchmarks/emb/isolation.py"
spec = importlib.util.spec_from_file_location("emb_isolation", SOURCE)
isolation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(isolation)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def setup(tmp_path):
    data = tmp_path / "t64.jsonl"
    data.write_text("".join(json.dumps({"text": text}) + "\n" for text in (
        "alpha forest", "beta river", "gamma stone", "extra longer prompt")))
    return SimpleNamespace(dataset=str(data), stock_url="http://stock", packed_url="http://packed",
                           model="m", repeat=2)


def fake_response(url, payload, leak=False):
    prompts = payload["input"]
    assert isinstance(prompts, list) and payload["encoding_format"] == "float"
    data = []
    for index, prompt in enumerate(prompts):
        if prompt == "gamma stone":
            vector = [1.0, 0.0] if url == "http://stock" else [1.0, 0.01]
            if leak and "beta river" in prompts:
                vector = [1.0, 0.1]
        else:
            vector = [0.0, 1.0]
        data.append({"index": index, "embedding": vector})
    return {"data": data}


def test_stable_peer_invariance_reports_packed_path_shift(monkeypatch, tmp_path):
    args = setup(tmp_path)
    calls = []

    def post(url, payload):
        calls.append((url, tuple(payload["input"])))
        return fake_response(url, payload)

    monkeypatch.setattr(isolation, "post_json", post)
    result = isolation.run(args, FakeTokenizer())
    assert result["pass"] and result["repeat_stable"] and result["peer_invariant"]
    assert result["claimable_speedup"] is False and result["packed_step_verified"] is False
    assert result["token_length"] == 2 and result["dimension"] == 2
    assert result["checks"]["packed_single_vs_stock"]["exact"] is False
    assert result["checks"]["peer_changed_B_second"]["exact"] is True
    assert len(calls) == 12


def test_peer_dependent_B_fails_even_when_each_batch_repeats(monkeypatch, tmp_path):
    args = setup(tmp_path)
    monkeypatch.setattr(isolation, "post_json", lambda url, payload: fake_response(url, payload, leak=True))
    result = isolation.run(args, FakeTokenizer())
    assert result["pass"] is False and result["repeat_stable"] is True
    assert result["peer_invariant"] is False
    assert result["checks"]["peer_changed_B_second"]["max_abs_delta"] > 0


def test_missing_or_duplicate_embedding_index_is_rejected(monkeypatch):
    monkeypatch.setattr(isolation, "post_json", lambda url, payload: {
        "data": [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}]})
    try:
        isolation.embeddings("http://packed", "m", ["A", "B"])
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate index must fail")


def test_no_equal_length_triple_fails(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text('{"text":"one"}\n{"text":"two words"}\n')
    try:
        isolation.select_texts(data, FakeTokenizer())
    except ValueError as exc:
        assert "three distinct" in str(exc)
    else:
        raise AssertionError("insufficient prompt matches must fail")


def test_cli_writes_json_failure_for_incomplete_server_batch(monkeypatch, tmp_path):
    args = setup(tmp_path)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: FakeTokenizer())))
    monkeypatch.setattr(isolation, "post_json", lambda url, payload: {"data": []})
    out = tmp_path / "result.json"
    rc = isolation.main(["--stock-url", args.stock_url, "--packed-url", args.packed_url,
                         "--model", args.model, "--dataset", args.dataset,
                         "--tokenizer", str(tmp_path), "--out", str(out)])
    result = json.loads(out.read_text())
    assert rc == 1 and result["pass"] is False and result["claimable_speedup"] is False
    assert "expected 1 embeddings" in result["reason"]
