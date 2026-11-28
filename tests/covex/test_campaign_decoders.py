import hashlib
import json
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[2]
SIX = ROOT / "benchmarks/six"


def test_best_correct_upstream_and_latency(tmp_path):
    suite = runpy.run_path(str(SIX / "run.py"))
    from clients import CACHE_LENGTHS, cache_prompts
    model = next(m for wave in suite["WAVES"] for m in wave if m.stock == "llama-U")
    dirs = {}
    for arm, rate in ((model.stock, 10), (model.tuned, 14), (model.fork, 21)):
        folder = dirs[arm] = tmp_path / arm
        folder.mkdir()
        for c in (1, 3):
            probe = dict(lengths=CACHE_LENGTHS, gen_tokens=32, concurrency=c,
                         prompt_sha256=hashlib.sha256(json.dumps(cache_prompts()).encode()).hexdigest(),
                         token_ids=[[1]*32 for _ in CACHE_LENGTHS])
            (folder / f"cache-c{c}.json").write_text(json.dumps(probe))
        for c in model.loads:
            n = 8 if c == 1 else 4*c
            obs = dict(completed=n, failed=0, max_concurrency=c, num_prompts=n, output_throughput=rate,
                       input_lens=[1024]*n, output_lens=[256]*n, total_input_tokens=1024*n, total_output_tokens=256*n,
                       ttfts=[.10]*n, itls=[[1.0] for _ in range(n)], median_ttft_ms=100, median_tpot_ms=4,
                       generated_texts=["text"]*n)
            (folder / f"c{c}-w1.json").write_text(json.dumps(obs))
    def read():
        return list(suite["record_rows"](model, dirs, {}, 1))
    assert [(r["comparator"], r["ratio"]) for r in read()] == [(model.tuned, 1.5)]*2
    f = dirs[model.fork] / "c1-w1.json"
    changed = json.loads(f.read_text())
    changed["ttfts"][1] = .30
    f.write_text(json.dumps(changed))
    assert read()[0]["ratio"] is None
    assert "p90" in read()[0]["status"]
    (dirs[model.tuned] / "cache-c3.json").unlink()
    assert all(r["ratio"] is None and r["covex"] == 21 for r in read())


def test_fixed_replay_and_answers(tmp_path):
    runpy.run_path(str(SIX / "run.py"))
    from decoder_checks import qwen_checks, replay, needles
    u, f = (tmp_path / name for name in ("u", "f"))
    u.mkdir()
    f.mkdir()
    rows = [dict(prompt_len=len(p), prompt_sha256=hashlib.sha256(json.dumps(p, separators=(",", ":")).encode()).hexdigest(),
                 token_ids=[1]*32, top2=[[[1, -.1], [2, -1.]]]*32) for p in replay.make_prompts(8, 1)]
    for path in (u / "replay.json", u / "repeat.json", f / "replay.json"):
        path.write_text(json.dumps(dict(prompts=rows, completed=8)))
    answers = [{"len": n, "expected": needles.record(n)[1], "got": needles.record(n)[1], "pass": True}
               for n in (8204, 9644, 24044)]
    for path in (u, f):
        (path / "needles.json").write_text(json.dumps({"pass": True, "results": answers}))
    assert qwen_checks(u, f) == []
    answers[-1]["got"] = "incorrect"
    (f / "needles.json").write_text(json.dumps({"pass": True, "results": answers}))
    assert "24044" in qwen_checks(u, f)[0]
    (u / "repeat.json").unlink()
    assert "repeat.json" in qwen_checks(u, f)[0]


def test_each_rank_requires_expected_source_and_tile(tmp_path):
    mod = runpy.run_path(str(ROOT / "benchmarks/provenance/kernel_files.py"))
    roles = {f"TP{r}": 100+r for r in range(8)}
    for pid in roles.values():
        folder = tmp_path / "kernels/q32-F4" / str(pid)
        folder.mkdir(parents=True)
        for name, sha in mod["QWEN_4K"].items():
            (folder / f"{name}.json").write_text(json.dumps({"sha256": sha, "tile": "4096"}))
    verify = mod["verify_sources"]
    verify(tmp_path, "q32-F4", roles, 8)
    (folder / f"{name}.json").write_text(json.dumps({"sha256": sha, "tile": "8192"}))
    with pytest.raises(ValueError, match="TP7"):
        verify(tmp_path, "q32-F4", roles, 8)


def test_chip_pair_limit_counts_tuned_arms():
    suite = runpy.run_path(str(SIX / "run.py"))
    first, second = suite["WAVES"]
    assert [sum(len(m.arms()) for m in wave) for wave in (first, second)] == [8, 8]
    with pytest.raises(ValueError, match="10 chip pairs"):
        suite["check_wave_fits"]((*first, second[0]))
