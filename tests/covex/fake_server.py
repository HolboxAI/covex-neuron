"""Minimal OpenAI-style fake for harness tests: /health, /metrics, completions, embeddings, chat, transcriptions.
FAKE_FAIL_INFER=1 -> /health OK but every inference request returns 500 (the 'healthy but cannot serve' case).
FAKE_EVENTS=<file> -> append one line per request ("METRICS" / "POST <path>") to observe ordering."""
import http.server, json, os, sys

PORT = int(sys.argv[sys.argv.index("--port") + 1])
EVENTS = os.environ.get("FAKE_EVENTS")


def event(s):
    if EVENTS:
        with open(EVENTS, "a") as f:
            f.write(s + "\n")


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {})
        if self.path == "/metrics":
            event("METRICS")
            return self._send(200, "vllm:e2e_request_latency_seconds_sum 1.0\nvllm:e2e_request_latency_seconds_count 1.0\n", "text/plain")
        self._send(404, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); self.rfile.read(n)
        event(f"POST {self.path}")
        if os.environ.get("FAKE_FAIL_INFER") == "1":
            return self._send(500, {"error": "fake inference failure"})
        if self.path == "/v1/embeddings":
            return self._send(200, {"data": [{"embedding": [0.1, 0.2]}]})
        if self.path == "/v1/chat/completions":
            return self._send(200, {"choices": [{"message": {"content": "ok"}}]})
        if self.path == "/v1/audio/transcriptions":
            return self._send(200, {"text": "ok"})
        self._send(200, {"choices": [{"text": "ok"}], "usage": {"completion_tokens": 2}})


http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
