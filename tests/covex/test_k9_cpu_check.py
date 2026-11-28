"""K9 CPU-mode exactness checker against two stub servers (reference greedy and a K9 server)."""
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _server(k9=False, diverge=False, accept=True):
    st = {"acc": 0}
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, b, ctype="application/json"):
            b = b.encode() if isinstance(b, str) else json.dumps(b).encode()
            self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            self._send(f"vllm:spec_decode_num_accepted_tokens_total {st['acc']}\n", "text/plain")
        def do_POST(self):
            b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            ids = [(len(b["prompt"]) * 7 + i) % 500 for i in range(b["max_tokens"])]
            pred = (b.get("vllm_xargs") or {}).get("covex_prediction_ids")
            if k9 and pred and accept:
                st["acc"] += 4
            if k9 and diverge and pred and pred != ids:
                ids[5] += 1                                   # a wrong prediction corrupts the output: the bug to catch
            self._send({"choices": [{"token_ids": ids}]})
    s = ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


def _check(tmp, **kw):
    ref, k9 = _server(), _server(k9=True, **kw)
    try:
        out = tmp / "k.json"
        r = subprocess.run([sys.executable, str(ROOT / "benchmarks/k9/cpu_check.py"), str(ref.server_port), str(k9.server_port),
                            str(out), "--n", "3", "--tokens", "12"], capture_output=True, text=True, timeout=120)
        return r.returncode, json.load(open(out))
    finally:
        ref.shutdown(); k9.shutdown()


def test_exact_k9_passes(tmp_path):
    rc, d = _check(tmp_path)
    assert rc == 0 and d["pass"] and len(d["cases"]) == 24, d


def test_prediction_dependent_output_fails(tmp_path):
    rc, d = _check(tmp_path, diverge=True)
    assert rc == 1 and any("perturbed" in p or "wrong" in p for p in d["problems"]), d


def test_no_speculation_fails(tmp_path):
    rc, d = _check(tmp_path, accept=False)
    assert rc == 1 and "speculation did not run" in " ".join(d["problems"])
