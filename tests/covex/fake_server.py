"""Minimal OpenAI-style server for the rehearsal harness test: /health and /v1/completions."""
import http.server, json, sys


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._send({}) if self.path == "/health" else self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self._send({"choices": [{"text": "ok " + str(body.get("prompt", ""))[:8]}], "usage": {"completion_tokens": 2}})


http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[sys.argv.index("--port") + 1])), H).serve_forever()
