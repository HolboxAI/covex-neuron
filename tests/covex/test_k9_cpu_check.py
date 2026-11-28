"""K9 CPU-mode exactness checker against two stub servers (reference greedy and a K9 server)."""
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


LOOKUP = True


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
                st["acc"] += (b["max_tokens"] - 1) if pred == ids else 1      # a perfect prediction drafts almost all
            if k9 and not pred and LOOKUP:
                st["acc"] += 2                                                  # prompt lookup alone also counts
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
    msg = " ".join(d["problems"])
    assert rc == 1 and ("speculation did not run" in msg or "client-supplied drafts were not used" in msg), d



def test_two_phase_refs_file_matches_single_phase(tmp_path):
    ref, k9 = _server(), _server(k9=True)
    try:
        refs = tmp_path / "refs.json"
        r1 = subprocess.run([sys.executable, str(ROOT / "benchmarks/k9/cpu_check.py"), str(ref.server_port), "0", str(tmp_path / "a.json"),
                             "--n", "3", "--tokens", "12", "--refs", str(refs)], capture_output=True, text=True, timeout=60)
        assert r1.returncode == 0 and json.load(open(refs))["ref"], r1.stdout + r1.stderr
        r2 = subprocess.run([sys.executable, str(ROOT / "benchmarks/k9/cpu_check.py"), "0", str(k9.server_port), str(tmp_path / "k.json"),
                             "--n", "3", "--tokens", "12", "--refs", str(refs)], capture_output=True, text=True, timeout=60)
        d = json.load(open(tmp_path / "k.json"))
        assert r2.returncode == 0 and d["pass"] and len(d["cases"]) == 24, d
        r3 = subprocess.run([sys.executable, str(ROOT / "benchmarks/k9/cpu_check.py"), "0", str(k9.server_port), str(tmp_path / "x.json"),
                             "--n", "2", "--tokens", "12", "--refs", str(refs)], capture_output=True, text=True, timeout=60)
        assert r3.returncode == 1 and "different prompts" in json.load(open(tmp_path / "x.json"))["problems"][0]
    finally:
        ref.shutdown(); k9.shutdown()



def test_acceptance_must_come_from_the_client_prediction(tmp_path):
    """Astra: the aggregate counter also rises from prompt lookup; a server that ignores the prediction but accepts
    lookup drafts must fail."""
    ref = _server()
    class Ignore(BaseHTTPRequestHandler):
        st = {"acc": 0}
        def log_message(self, *a): pass
        def _send(self, b, ctype="application/json"):
            b = b.encode() if isinstance(b, str) else json.dumps(b).encode()
            self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            self._send(f"vllm:spec_decode_num_accepted_tokens_total {Ignore.st['acc']}\n", "text/plain")
        def do_POST(self):
            b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            Ignore.st["acc"] += 2                                                # lookup-only acceptance, prediction ignored
            self._send({"choices": [{"token_ids": [(len(b["prompt"]) * 7 + i) % 500 for i in range(b["max_tokens"])]}]})
    k9 = ThreadingHTTPServer(("127.0.0.1", 0), Ignore); threading.Thread(target=k9.serve_forever, daemon=True).start()
    try:
        out = tmp_path / "k.json"
        r = subprocess.run([sys.executable, str(ROOT / "benchmarks/k9/cpu_check.py"), str(ref.server_port), str(k9.server_port),
                            str(out), "--n", "3", "--tokens", "12"], capture_output=True, text=True, timeout=120)
        d = json.load(open(out))
        assert r.returncode == 1 and any("client-supplied drafts were not used" in p for p in d["problems"]), d
    finally:
        ref.shutdown(); k9.shutdown()
