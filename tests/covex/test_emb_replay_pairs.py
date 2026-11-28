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
    def __init__(s, e): s.outputs = types.SimpleNamespace(embedding=e)
class LLM:
    def __init__(s, **kw): pass
    def embed(s, prompts, use_tqdm=False):
        N[0] += 1; ids = [f"r{N[0]}-{i}" for i in range(len(prompts))]; start = 0; mem = []
        for i, p in zip(ids, prompts):
            n = len(p.split()); mem.append({"req": i, "start": start, "tokens": n, "padded": n}); start += n
        split = MODE == "split" and len(prompts) == 2
        with open(LOG, "a") as f:
            groups = [[m] for m in mem] if split else [mem]
            for g in groups:
                for rank in (0, 1):
                    f.write(f"(Worker_TP{rank}) INFO COVEX_PACK_TRACE prefill step: members={g!r} total_padded=1 padded_num_reqs=8 packed=True\n")
        out = []
        for k, p in enumerate(prompts):
            v = [float(b) for b in hashlib.sha256(p.encode()).digest()[:4]]
            if len(prompts) > 1 and not split:
                if MODE == "offset" and k > 0: v[0] += 1e-3                       # depends on position only
                if MODE == "peer" and k > 0: v[0] += hashlib.sha256(prompts[0].encode()).digest()[0] * 1e-4 + 1e-4
            out.append(_O(v))
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
    assert rc == 1 and d["packing_verified"] is False
