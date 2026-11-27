"""Check end-to-end prompt sizing, needle scoring and CLI error reporting."""
import json
import re
import sys
import types
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/q32"))
import needles


class WhitespaceTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return text.split()


def exercise(monkeypatch, tmp_path, lengths, failure=None):
    def from_pretrained(path, *, local_files_only):
        assert local_files_only and path == "local-tok"
        return WhitespaceTokenizer()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=from_pretrained)))
    seen = []

    def completion(request):
        data = json.loads(request.content)
        assert request.url.path == "/v1/completions"
        assert data["max_tokens"] == 16 and data["temperature"] == 0 and data["add_special_tokens"] is False
        prompt = data["prompt"]
        n = len(prompt.split())
        key, value = re.findall(r"(archive_\w+) = (\d+)", prompt)[0]
        assert prompt.count(value) == 1 and prompt.count(key) == 2
        assert prompt.endswith(f"What is {key}?\nAnswer:")
        assert 0.85 < len(prompt.split(key)[0].split()) / n < 0.95
        seen.append((n, value))
        response = {"usage": {"prompt_tokens": n}, "choices": [{"text": f" {value}\n"}]}
        if failure and n == 9644:
            if failure == "wrong":
                response["choices"][0]["text"] = "wrong-" + value
            elif failure == "count":
                response["usage"]["prompt_tokens"] -= 1
            elif failure == "missing":
                response.pop("usage")
            elif failure == "http":
                return httpx.Response(500)
        return httpx.Response(200, json=response)

    client = httpx.Client(transport=httpx.MockTransport(completion))
    monkeypatch.setattr(needles.httpx, "Client", lambda **kwargs: client)
    out = tmp_path / "needle.json"
    monkeypatch.setattr(sys, "argv", ["needles.py", "--base-url", "http://local", "--model", "m",
                                      "--tokenizer", "local-tok", "--lengths", lengths, "--out", str(out)])
    return needles.main(), json.loads(out.read_text()), seen


def test_exact_lengths_and_distinct_answer_per_length(monkeypatch, tmp_path):
    rc, result, seen = exercise(monkeypatch, tmp_path, "8204,9644,24044")
    assert rc == 0 and result["pass"]
    assert [n for n, _ in seen] == [8204, 9644, 24044]
    assert len({value for _, value in seen}) == 3
    assert all(row["got"].strip() == row["expected"] and row["pass"] for row in result["results"])


@pytest.mark.parametrize("failure", ["wrong", "count", "missing", "http"])
def test_reports_every_length_after_failure(monkeypatch, tmp_path, failure, capsys):
    rc, result, seen = exercise(monkeypatch, tmp_path, "8204,9644,24044", failure)
    assert rc != 0 and not result["pass"] and len(seen) == 3
    assert [row["pass"] for row in result["results"]] == [True, False, True]
    assert result["results"][1]["reason"] and result["results"][1]["expected"] == seen[1][1]
    assert json.loads(capsys.readouterr().err)["reason"]


@pytest.mark.parametrize("lengths", ["", "0", "8204,8204", "oops", "1"])
def test_bad_lengths_never_pass(monkeypatch, tmp_path, lengths):
    rc, result, seen = exercise(monkeypatch, tmp_path, lengths)
    assert rc != 0 and not result["pass"] and result["reason"] and not seen


def test_sizes_by_encoded_tokens():
    class ByteTokenizer:
        def encode(self, text, *, add_special_tokens):
            return text.encode()

    tokenizer = ByteTokenizer()
    text, _ = needles.make_prompt(tokenizer, 8204)
    assert len(text.encode()) == 8204
