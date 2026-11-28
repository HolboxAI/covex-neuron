import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

SIX = Path(__file__).resolve().parents[2] / "benchmarks/six"
sys.path.insert(0, str(SIX))
import run as suite


def test_followup_launches_matched_arms_in_disjoint_regions(tmp_path, monkeypatch):
    wave = suite.TUNED_WAVES[1]
    assert [m.arms() for m in wave] == [
        ("asr-U", "asr-Ut", "asr-B8"), ("asr-U2", "asr-U2t", "asr-R"), ("vl-U", "vl-Ub", "vl-Cb")]
    ports, cores, cpus, launched = set(), set(), set(), set()
    def stop_after_launch(argv, cwd, env, stream):
        port = int(argv[-1])
        assert port not in ports
        ports.add(port)
        arm = env["COVEX_ARM"]
        model = next(m for m in wave if arm in m.arms())
        assert env["SUITE_PYTHON"] == ("fork" if arm == model.fork else "upstream")
        first, last = map(int, env["SUITE_CORES"].split("-"))
        cpu_first, cpu_last = map(int, env["SUITE_CPUS"].split("-"))
        core_range = set(range(first, last + 1))
        cpu_range = set(range(cpu_first, cpu_last + 1))
        assert len(core_range) == model.tp * model.replicas and first % len(core_range) == 0
        assert len(cpu_range) == 3 * len(core_range)
        assert cores.isdisjoint(core_range) and cpus.isdisjoint(cpu_range)
        if model.kind == "asr":
            assert 8 * int(argv[-2]) == first
            assert argv[-3] == suite.ASR_LABELS[arm]
        cores.update(core_range)
        cpus.update(cpu_range)
        launched.add(arm)
        raise OSError("recorded launch")

    for folder in {m.folder for m in wave}:
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "config.json").write_text("{}")
    for f in ("benchmark/benchmark.py", "data/manifest.json"):
        path = tmp_path / f
        path.parent.mkdir()
        path.touch()
    monkeypatch.setattr(suite, "service", stop_after_launch)
    monkeypatch.setattr(suite, "verify_checkpoint", lambda *_: None)
    out = tmp_path / "out"
    out.mkdir()
    suite.run_wave(wave, out, SimpleNamespace(windows=1),
                   dict(MODELS_DIR=str(tmp_path), CLIENT_DIR=str(tmp_path), FORK_PY="fork", UPSTREAM_PY="upstream"),
                   (tmp_path, tmp_path), ("upstream", "fork"))
    assert len(launched) == 9 and cores == set(range(60)) and cpus == set(range(180))

    pins = json.loads(subprocess.check_output(
        [sys.executable, SIX / "run.py", "--tuned-followup", "--print-pins"]))
    assert set(pins) == set(suite.MODEL_PINS) - {"openai/gpt-oss-20b"}


def test_asr_tuned_ratio_requires_direct_wer_match(tmp_path):
    model = suite.TUNED_WAVES[1][0]
    places = {}
    for arm, wer in zip(model.arms(), (.10, .096, .104)):
        path = tmp_path / arm
        path.mkdir()
        places[arm] = path
        for c in model.loads:
            obs = dict(phases={"transcribe": [dict(concurrency=c, n_total=100, n_success=100, n_error=0,
                                                    wer_mean=wer, audio_seconds_per_s=40)]},
                       raw_results=[dict(phase="transcribe", concurrency=c, is_warmup=False, success=True,
                                         round_idx=0, sample_id=str(i), audio_duration_s=2, e2e_latency_s=.1)
                                    for i in range(100)])
            (path / f"c{c}-w1.json").write_text(json.dumps(obs))
    for row in suite.record_rows(model, places, {}, 1):
        assert row["ratio"] == 1 and row["ratio_vs_tuned"] is None
        assert "WER" in row["tuned_status"]
