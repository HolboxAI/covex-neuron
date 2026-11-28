"""Round-robin front for the ASR replica layout, against two in-test backends."""
import json
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]


def _backend(name, healthy=True):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass
        def _send(self, code, body, ctype="application/json"):
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            if self.path == "/health":
                return self._send(200 if healthy else 503, b"")
            if self.path == "/metrics":
                return self._send(200, b"vllm:e2e_request_latency_seconds_count 5\n", "text/plain")
        def do_POST(self):
            n = int(self.headers["Content-Length"]); body = json.loads(self.rfile.read(n))
            if body.get("stream"):
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Connection", "close")
                self.end_headers()
                for i in range(3):
                    self.wfile.write(f"data: {name}-{i}\n\n".encode()); self.wfile.flush(); time.sleep(0.05)
                self.close_connection = True
                return
            self._send(200, json.dumps({"backend": name, "echo": body}).encode())
    class Srv(ThreadingHTTPServer):
        request_queue_size = 2048   # like vLLM's uvicorn backlog; the default 5 resets bursts on Linux (502 at the proxy)
        daemon_threads = True
    s = Srv(("127.0.0.1", 0), H); threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


def _free():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _proxy(ports):
    lp = _free()
    p = subprocess.Popen([sys.executable, str(ROOT / "benchmarks/asr/rr_proxy.py"), str(lp), *map(str, ports)])
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", lp), timeout=0.2).close(); break
        except OSError:
            time.sleep(0.05)
    return p, lp


def test_round_robin_streaming_metrics_and_health():
    a, b = _backend("a"), _backend("b")
    p, lp = _proxy([a.server_port, b.server_port])
    try:
        with httpx.Client() as c:                          # one kept-alive client: balancing must be per request
            got = [c.post(f"http://127.0.0.1:{lp}/v1/audio/transcriptions", json={"i": i}).json()["backend"] for i in range(6)]
            assert got.count("a") == 3 and got.count("b") == 3, got
            with c.stream("POST", f"http://127.0.0.1:{lp}/v1/chat/completions", json={"stream": True}) as r:
                text = "".join(r.iter_text())
            assert text.count("data: ") == 3
            m = c.get(f"http://127.0.0.1:{lp}/metrics").text
            assert m.count("vllm:e2e_request_latency_seconds_count 5") == 2            # both replicas, summable
            assert c.get(f"http://127.0.0.1:{lp}/health").status_code == 200
            st = c.get(f"http://127.0.0.1:{lp}/proxy_stats").json()
            assert sorted(v["requests"] for v in st.values()) == [3, 4], st        # 6 transcriptions + 1 stream
            assert all(v["inflight"] == 0 and v["busy_s"] > 0 and v["errors"] == 0 for v in st.values()), st
    finally:
        p.kill(); p.wait(); a.shutdown(); b.shutdown()


def test_health_fails_when_any_replica_is_down():
    a, b = _backend("a"), _backend("b", healthy=False)
    p, lp = _proxy([a.server_port, b.server_port])
    try:
        assert httpx.get(f"http://127.0.0.1:{lp}/health").status_code == 503
    finally:
        p.kill(); p.wait(); a.shutdown(); b.shutdown()


def test_concurrent_burst_is_queued_not_reset():
    import concurrent.futures as cf
    a, b = _backend("a"), _backend("b")
    p, lp = _proxy([a.server_port, b.server_port])
    try:
        def one(i):
            return httpx.post(f"http://127.0.0.1:{lp}/v1/audio/transcriptions", json={"i": i}, timeout=30).status_code
        with cf.ThreadPoolExecutor(64) as ex:
            assert set(ex.map(one, range(200))) == {200}
    finally:
        p.kill(); p.wait(); a.shutdown(); b.shutdown()


import os
import shutil
import pytest

LINUX = sys.platform == "linux" and shutil.which("taskset") is not None


@pytest.mark.skipif(not LINUX, reason="Linux CI (taskset)")
def test_serve_replicas_two_tp4_behind_one_port_and_clean_stop(tmp_path):
    env = dict(os.environ, PATH=f"{ROOT/'tests/covex/fake_bin'}:{os.environ['PATH']}", MODEL_DIR=str(tmp_path / "m"),
               SOLO="1", FAKE_VLLM_DELAY="1")
    port = _free()
    p = subprocess.Popen(["bash", str(ROOT / "benchmarks/asr/serve_replicas.sh"), "R4", "0", str(port)], cwd=tmp_path, env=env,
                         start_new_session=True)
    try:
        for _ in range(120):
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        assert httpx.get(f"http://127.0.0.1:{port}/health").status_code == 200
        r = httpx.post(f"http://127.0.0.1:{port}/v1/audio/transcriptions", json={}, timeout=10)
        assert r.status_code == 200
        for r_ in (0, 1):
            meta = (tmp_path / f"logs/R4.r{r_}.meta").read_text()
            assert f"cores={4 * r_}-{4 * r_ + 3}" in meta, meta          # replica r on cores 4r..4r+3 of pair 0
            assert (tmp_path / f"logs/R4.r{r_}.server.log").exists()
    finally:
        os.killpg(p.pid, 15); p.wait(timeout=30)
    time.sleep(4)
    assert subprocess.run(["pgrep", "-f", f"fake_server.py --port {port + 1000}"], capture_output=True).returncode != 0
