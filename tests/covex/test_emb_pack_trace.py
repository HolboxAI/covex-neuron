"""pack_trace_check: proves the isolation pairs shared one packed step, from the server's COVEX_PACK_TRACE lines."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _line(members, packed=True, rank=0):
    m = [{"req": r, "start": s, "tokens": t, "padded": p} for r, s, t, p in members]
    return (f"(Worker_TP{rank} pid=1) INFO neuron_model_runner.py:4240 COVEX_PACK_TRACE prefill step: members={m!r} "
            f"total_padded={sum(x[3] for x in members)} padded_num_reqs=8 packed={packed}")


def _run(tmp, lines, repeat=2):
    log = tmp / "s.log"; log.write_text("noise\n" + "\n".join(lines) + "\n")
    rep = tmp / "iso.json"
    rep.write_text(json.dumps({"pass": True, "cases": {"packed_B": {"b_sha256": ["x"] * repeat}}, "checks": {}}))
    r = subprocess.run([sys.executable, str(ROOT / "benchmarks/emb/pack_trace_check.py"), str(log), "1", str(rep)],
                       capture_output=True, text=True)
    return r.returncode, json.load(open(rep))


def _good(rank=0):
    out = []
    for i in range(2):                                  # packed_B x2 (one member)
        out.append(_line([(f"b{i}", 0, 64, 128)], rank=rank))
    for i in range(8):                                  # 4 pair cases x repeat 2
        out.append(_line([(f"p{i}a", 0, 64, 128), (f"p{i}b", 128, 64, 128)], rank=rank))
    return out


def test_verified_when_every_pair_is_one_packed_step(tmp_path):
    rc, d = _run(tmp_path, _good())
    assert rc == 0 and d["packed_step_verified"] is True and d["trace"]["pair_steps"] == 8, d


def test_tp_rank_duplicates_are_counted_once(tmp_path):
    lines = [l for pair in zip(_good(0), _good(1)) for l in pair]   # every step logged by two ranks
    rc, d = _run(tmp_path, lines)
    assert rc == 0 and d["trace"]["pair_steps"] == 8, d


def test_pairs_split_into_separate_steps_fail(tmp_path):
    lines = _good()[:2] + [_line([(f"s{i}", 0, 64, 128)]) for i in range(16)]   # scheduler ran each input alone
    rc, d = _run(tmp_path, lines)
    assert rc == 1 and d["packed_step_verified"] is False and d["pass"] is False and "two-member" in d["trace_reason"]


def test_unpacked_path_fails(tmp_path):
    lines = [l.replace("packed=True", "packed=False") for l in _good()]
    rc, d = _run(tmp_path, lines)
    assert rc == 1 and "packed attention path" in d["trace_reason"]


def test_incomplete_client_report_is_not_verified(tmp_path):
    log = tmp_path / "s.log"; log.write_text("\n".join(_good()))
    rep = tmp_path / "iso.json"; rep.write_text(json.dumps({"pass": False, "reason": "HTTP 500"}))
    r = subprocess.run([sys.executable, str(ROOT / "benchmarks/emb/pack_trace_check.py"), str(log), "0", str(rep)], capture_output=True, text=True)
    assert r.returncode == 1 and json.load(open(rep))["packed_step_verified"] is False
