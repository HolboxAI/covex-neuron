"""replay_pairs.py with stub vllm/transformers: verdict logic and packing verification from the trace."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

STUB_VLLM = r'''
import hashlib, os, sys, types
MODE = os.environ["STUB_MODE"]; LOG = os.environ["STUB_LOG"]; N = [0]
class _O:
    def __init__(s, e, request_id): s.outputs = types.SimpleNamespace(embedding=e); s.request_id = request_id
class LLM:
    def __init__(s, **kw): pass
    def embed(s, prompts, use_tqdm=False):
        ids = []
        for _ in prompts:
            N[0] += 1; ids.append(f"opaque-{1000-N[0]}" if MODE == "opaque" else str(N[0]))
        start = 0; mem = []
        for i, p in zip(ids, prompts):
            n = len(p.split()); mem.append({"req": i, "start": start, "tokens": n, "padded": n}); start += n
        if MODE == "reorder":
            mem = mem[::-1]                                       # trace lists members in another order than submitted
        split = MODE == "split" and len(prompts) == 2
        with open(LOG, "a") as f:
            groups = [[m] for m in mem] if split else [mem]
            for g in groups:
                tp = sum(m["padded"] for m in g) + (8 if MODE == "bucket_drift" and N[0] % 3 == 0 else 0)
                for rank in (0, 1):
                    f.write(f"(Worker_TP{rank}) INFO COVEX_PACK_TRACE prefill step: members={g!r} total_padded={tp} padded_num_reqs=8 packed={MODE != 'not_packed'}\n")
        out = []
        for k, p in enumerate(prompts):
            v = [float(b) for b in hashlib.sha256(p.encode()).digest()[:4]]
            if MODE == "unstable" and N[0] % 2: v[1] += 1e-6
            if len(prompts) > 1 and not split:
                if MODE == "offset" and k > 0: v[0] += 1e-3                       # depends on position only
                if MODE == "peer" and k > 0: v[0] += hashlib.sha256(prompts[0].encode()).digest()[0] * 1e-4 + 1e-4
            out.append(_O(v, "missing-from-trace" if MODE == "wrong_id" and len(ids) > 1 and k == 1 else ids[k]))
        return out
'''
STUB_TOK = '''
class _T:
    def encode(self, t, add_special_tokens=True): return t.split()
class AutoTokenizer:
    @staticmethod
    def from_pretrained(p, **k): return _T()
'''


def _run(tmp, mode):
    st = tmp / "stub"; (st / "vllm").mkdir(parents=True, exist_ok=True); (st / "transformers").mkdir(exist_ok=True)
    (st / "vllm/__init__.py").write_text(STUB_VLLM); (st / "transformers/__init__.py").write_text(STUB_TOK)
    data = tmp / "d.jsonl"
    data.write_text("".join(json.dumps({"text": t}) + "\n" for t in
                            ["a b c d e", "f g h i j", "k l m n o", "p q", "r s t u v"]))
    log = tmp / f"{mode}.log"; log.write_text("")
    import os
    env = dict(os.environ, PYTHONPATH=str(st), STUB_MODE=mode, STUB_LOG=str(log))
    out = tmp / f"{mode}.json"
    r = subprocess.run([sys.executable, str(ROOT / "benchmarks/emb/replay_pairs.py"), "--model", "m", "--dataset", str(data),
                        "--tokenizer", "t", "--out", str(out), "--log", str(log), "--repeat", "2"], env=env,
                       capture_output=True, text=True, timeout=60)
    return r.returncode, json.load(open(out))


def test_invariant_when_packing_is_exact(tmp_path):
    rc, d = _run(tmp_path, "exact")
    assert rc == 0 and d["packing_verified"] and d["verdict"] == "invariant", d


def test_offset_dependent_shift_is_classified(tmp_path):
    rc, d = _run(tmp_path, "offset")
    assert rc == 0 and d["verdict"] == "follows_offset" and d["peer_changed_same_offset_exact"], d


def test_peer_dependent_shift_is_classified(tmp_path):
    rc, d = _run(tmp_path, "peer")
    assert rc == 0 and d["verdict"] == "follows_peer_content", d


def test_split_pairs_fail_verification(tmp_path):
    rc, d = _run(tmp_path, "split")
    assert rc == 1 and d["packing_verified"] is False and d["verdict"] == "inconclusive"


def test_unstable_repeats_are_inconclusive(tmp_path):
    rc, d = _run(tmp_path, "unstable")
    assert rc == 1 and d["verdict"] == "inconclusive" and not d["repeat_stable"], d


def test_bucket_drift_between_repeats_is_inconclusive(tmp_path):
    rc, d = _run(tmp_path, "bucket_drift")
    assert rc == 1 and d["verdict"] == "inconclusive" and any("bucket" in w for w in d["inconclusive_because"]), d


def test_b_is_bound_by_request_id_not_trace_order(tmp_path):
    rc, d = _run(tmp_path, "reorder")
    lay = d["layouts"]
    assert rc == 0 and lay["A+B"][0][0] == 5 and lay["B+A"][0][0] == 0, lay        # B second at offset len(A)=5


def test_opaque_request_ids_bind_b_from_output_not_numeric_sort(tmp_path):
    rc, d = _run(tmp_path, "opaque")
    assert rc == 0 and d["packing_verified"], d
    assert d["layouts"]["A+B"][0][0] == 5 and d["layouts"]["B+A"][0][0] == 0, d


def test_unpacked_step_is_inconclusive_even_with_both_members(tmp_path):
    rc, d = _run(tmp_path, "not_packed")
    assert rc == 1 and d["verdict"] == "inconclusive" and not d["packing_verified"], d


def test_output_id_must_match_the_packed_trace(tmp_path):
    rc, d = _run(tmp_path, "wrong_id")
    assert rc == 1 and d["verdict"] == "inconclusive", d


def test_replays_the_pinned_inputs_even_when_a_longer_group_exists():
    import hashlib
    import runpy
    import pytest
    from types import SimpleNamespace

    pick = runpy.run_path(str(ROOT / "benchmarks/emb/replay_pairs.py"))["pick"]
    texts = ["short", "red sun", "blue sky", "gold moon", "a long phrase", "a high hill", "a deep cave"]
    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens: text.split())
    witness = {"diagnostic_inputs": {
        name: {"input_line_one_based": i + 1, "text_sha256": hashlib.sha256(texts[i].encode()).hexdigest()}
        for name, i in (("A", 1), ("A_prime", 2), ("B", 3))
    }}
    prompts, length = pick(texts, tokenizer, witness)
    assert (prompts["A"], prompts["A_prime"], prompts["B"], length) == ("red sun", "blue sky", "gold moon", 2)
    with pytest.raises(ValueError, match="differs from the campaign"):
        pick(list(reversed(texts)), tokenizer, witness)
