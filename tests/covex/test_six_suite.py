"""Use the embedding-only attempt without rerunning or conflating its packing arms."""
import json
from pathlib import Path
import runpy

SUITE = Path(__file__).resolve().parents[2] / "benchmarks/six/run.py"


def test_reuse_embedding_results_independently(tmp_path):
    suite = runpy.run_path(str(SUITE))
    results = tmp_path / "results"
    results.mkdir()
    manifest = {
        "run_prefix": "emb/fixture", "upstream_vllm_neuron": suite["UPSTREAM_COMMIT"],
        "covex_neuron": "032aadebc0d109c2e4d54d2e8b11b4249e910c39",
        "model": "Qwen/Qwen3-Embedding-8B", "model_revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
        "windows": 3, "lens": ["t64"], "concurrency": 32, "requests_per_window": 256, "chips_per_arm": 2,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    statuses = []
    for arm, rate in (("emb-U", 30), ("emb-T", 30), ("emb-P4", 60), ("emb-P8", 90)):
        statuses.extend((f"{arm}\tready\tOK", f"{arm}\tprovenance\tPASS source and interpreter"))
        for w in (1, 2, 3):
            result = {"n": 256, "concurrency": 32, "seq_per_s": rate, "vectors": [[1, 0]]*256,
                      "input_ids": list(map(str, range(256)))}
            (results / f"{arm}-t64-w{w}.json").write_text(json.dumps(result))
    (results / "STATUS.tsv").write_text("\n".join(statuses))
    read = suite["import_embedding"]
    assert read(tmp_path, 3, "emb-P4")["covex"] == 60
    for revision, ratio in (("unknown", None), ("63b0d89fdecb9a8df314c927a3ac9d99a319b85e", None)):
        manifest["covex_neuron"] = revision
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        assert read(tmp_path, 3, "emb-P4")["ratio"] == ratio
    assert "isolation and pooled p90 unavailable" in read(tmp_path, 3, "emb-P8")["status"]
    path = results / "emb-P8-t64-w1.json"
    obj = json.loads(path.read_text())
    obj["vectors"][17] = [0, 1]
    path.write_text(json.dumps(obj))
    rows = [read(tmp_path, 3, arm) for arm in ("emb-P4", "emb-P8")]
    assert [r["ratio"] for r in rows] == [None, None]
    suite["write_summary"](tmp_path, rows)
    assert "packed vectors differ" in (tmp_path / "RESULTS.md").read_text()


def test_live_modalities_require_matched_work(tmp_path):
    suite = runpy.run_path(str(SUITE))
    waves = suite["WAVES"]
    for model in (m for wave in waves for m in wave if m.kind in ("emb", "asr", "vl")):
        source = {}
        dirs = {}
        for arm in (model.stock, model.fork):
            dirs[arm] = tmp_path / arm
            dirs[arm].mkdir()
            for c in model.loads:
                if model.kind == "emb":
                    (dirs[arm] / "isolation.json").write_text(json.dumps(dict(
                        peer_invariant=True, repeat_stable=True, packed_step_verified=True,
                        **{"pass": True})))
                    result = dict(n=256, concurrency=c, seq_per_s=2, latencies_s=[0.2]*256,
                                  input_ids=list(map(str, range(256))), vectors=[[1.0, 2.0]]*256)
                elif model.kind == "asr":
                    result = dict(phases={"transcribe": [dict(concurrency=c, n_total=100,
                                  n_success=100, n_error=0, wer_mean=.06, audio_seconds_per_s=2)]},
                                  raw_results=[dict(phase="transcribe", concurrency=c, is_warmup=False, success=True,
                                               round_idx=0, sample_id=str(i), audio_duration_s=1.5, e2e_latency_s=.25) for i in range(100)])
                else:
                    result = dict(n=20, concurrency=c, out_tok_per_s=2, latencies_s=[.2]*20, answers=["A"]*20, completion_tokens=[1]*20)
                (dirs[arm] / f"c{c}-w1.json").write_text(json.dumps(result))
                source[c] = result
        report = suite["record_rows"]
        assert all(r["ratio"] == 1 for r in report(model, dirs, {}, 1))
        for c, result in source.items():
            if model.kind == "emb":
                del result["input_ids"]
                (dirs[model.stock] / f"c{c}-w1.json").write_text(json.dumps(result))
            elif model.kind == "asr":
                result["raw_results"][0]["sample_id"] = "different clip"
            else:
                result["completion_tokens"][3] += 1
            (dirs[model.fork] / f"c{c}-w1.json").write_text(json.dumps(result))
        assert all(r["ratio"] is None for r in report(model, dirs, {}, 1))
        (dirs[model.fork] / f"c{model.loads[0]}-w1.json").write_text("[]")
        assert "must contain a JSON object" in next(report(model, dirs, {}, 1))["status"]



def test_asr_replica_row_layout_and_labels():
    import importlib.util
    spec = importlib.util.spec_from_file_location("six", Path(__file__).resolve().parents[2] / "benchmarks/six/run.py")
    six = importlib.util.module_from_spec(spec); spec.loader.exec_module(six)
    asr = [m for w in six.WAVES for m in w if m.kind == "asr"]
    assert {(m.stock, m.fork, m.tp, m.replicas) for m in asr} == {("asr-U", "asr-DMB", 8, 1), ("asr-U2", "asr-R", 4, 2)}
    assert all(six.ASR_LABELS[a] for m in asr for a in (m.stock, m.fork))
    for w in six.WAVES:
        six.check_wave_fits(w)
    import pytest
    with pytest.raises(ValueError):
        six.check_wave_fits(tuple(w for wave in six.WAVES for w in wave))   # all models in one wave cannot fit


def test_vlstage_splits_image_cost_from_decode(tmp_path, monkeypatch):
    import httpx
    import pytest
    from types import SimpleNamespace
    runpy.run_path(str(SUITE))
    import clients

    clock = 0.0
    def transport(request):
        nonlocal clock
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        image = any(part["type"] == "image_url" for part in body["messages"][0]["content"])
        clock += .05 + (.2 if image else 0) + .01 * body["max_tokens"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "A"}}]})

    client = httpx.Client(transport=httpx.MockTransport(transport))
    monkeypatch.setattr(clients, "httpx", SimpleNamespace(Client=lambda **_: client))
    monkeypatch.setattr(clients, "time", SimpleNamespace(perf_counter=lambda: clock))
    monkeypatch.setattr(clients, "PROBE_TOKENS", 8)
    (tmp_path / "images").mkdir()
    (tmp_path / "images/a.jpg").write_bytes(b"sample")
    (tmp_path / "manifest.json").write_text(json.dumps([{"image": "a.jpg", "question": "Q?"}]))
    out = tmp_path / "timings.json"
    clients.vlstage(8600, tmp_path, 3, out)
    result = json.loads(out.read_text())
    assert result["image_cost_s"] == pytest.approx(.2)
    assert result["decode_tail_s"] == pytest.approx(.07)
    assert result["max_latency_gain_if_image_cost_zero"] == pytest.approx(.33 / .13)
