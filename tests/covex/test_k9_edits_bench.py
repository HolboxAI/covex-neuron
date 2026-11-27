"""K9 edit benchmark client against an in-test HTTP stub (no vLLM)."""
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "benchmarks/k9/edits.py"


def _server(diverge=False, accepted=0.0):
    state = {"acc": 0.0, "drafts": 0.0}
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, code, body, ctype="application/json"):
            b = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            self._send(200, f"vllm:spec_decode_num_accepted_tokens_total {state['acc']}\n"
                            f"vllm:spec_decode_num_drafts_total {state['drafts']}\nvllm:spec_decode_num_draft_tokens_total 0\n", "text/plain")
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            ids = [len(body["prompt"]) % 97, 1, 2, 3]
            if diverge and "vllm_xargs" in body:
                ids[2] = 99
            if "vllm_xargs" in body:
                state["acc"] += accepted; state["drafts"] += 1
            self._send(200, {"choices": [{"token_ids": ids, "finish_reason": "stop"}]})
    srv = HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _tasks(tmp):
    p = tmp / "t.jsonl"
    p.write_text("".join(json.dumps({"id": i, "prompt": "x" * (10 + i), "prediction": [5, 6, 7], "max_tokens": 8}) + "\n"
                         for i in range(3)))
    return p


def _run(port, tasks, mode, out, conc=1):
    return subprocess.run([sys.executable, str(E), "run", "--base-url", f"http://127.0.0.1:{port}", "--tasks", str(tasks),
                           "--mode", mode, "--conc", str(conc), "--out", str(out)], capture_output=True, text=True, timeout=60)


def test_exact_runs_get_a_speedup_and_tokens_per_pass(tmp_path):
    srv = _server(accepted=3.0)
    try:
        t = _tasks(tmp_path)
        assert _run(srv.server_port, t, "none", tmp_path / "ref.json").returncode == 0
        assert _run(srv.server_port, t, "predict", tmp_path / "k9.json", conc=4).returncode == 0
    finally:
        srv.shutdown()
    r = subprocess.run([sys.executable, str(E), "compare", "--ref", str(tmp_path / "ref.json"), "--runs", str(tmp_path / "k9.json"),
                        "--out", str(tmp_path / "s.json")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    row = json.load(open(tmp_path / "s.json"))["rows"][0]
    assert row["exact"] == 3 and row["speedup_vs_ref"] is not None and abs(row["tokens_per_pass"] - 4.0) < 1e-9, row
    rep = subprocess.run([sys.executable, str(E), "report", str(tmp_path / "s.json")], capture_output=True, text=True).stdout
    assert "| predict | 4 | 3/3 |" in rep, rep


def test_inexact_output_gets_no_speedup(tmp_path):
    srv = _server(diverge=True)
    try:
        t = _tasks(tmp_path)
        _run(srv.server_port, t, "none", tmp_path / "ref.json")
        _run(srv.server_port, t, "predict", tmp_path / "k9.json")
    finally:
        srv.shutdown()
    subprocess.run([sys.executable, str(E), "compare", "--ref", str(tmp_path / "ref.json"), "--runs", str(tmp_path / "k9.json"),
                    "--out", str(tmp_path / "s.json")], check=True)
    row = json.load(open(tmp_path / "s.json"))["rows"][0]
    assert row["exact"] == 0 and row["speedup_vs_ref"] is None


def test_unreachable_server_fails_loudly(tmp_path):
    r = _run(1, _tasks(tmp_path), "predict", tmp_path / "x.json")
    assert r.returncode == 1 and json.load(open(tmp_path / "x.json"))["pass"] is False
