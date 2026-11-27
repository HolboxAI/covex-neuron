"""Minimal OpenAI-style fake for harness tests: /health, /metrics, completions, embeddings, chat, transcriptions.
FAKE_FAIL_INFER=1 -> /health OK but every inference request returns 500 (the 'healthy but cannot serve' case).
FAKE_EVENTS=<file> -> append one line per request ("METRICS" / "POST <path>") to observe ordering."""
import http.server, json, os, sys

PORT = int(sys.argv[sys.argv.index("--port") + 1])
if os.environ.get("COVEX_PROVENANCE_DIR"):   # like a real arm process: import the plugin package (recorded by provenance)
    import importlib
    try:
        importlib.import_module("vllm_neuron")
    except ImportError:
        pass
EVENTS = os.environ.get("FAKE_EVENTS")
STATE = {"served": 0, "scrapes": 0}   # served requests -> vllm:e2e_request_latency_seconds_count
FAIL_SCRAPES = {int(v) for v in os.environ.get("FAKE_METRICS_FAIL_NTH", "").split()}   # 1-based scrape numbers to fail


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
            event("METRICS"); STATE["scrapes"] += 1
            if STATE["scrapes"] in FAIL_SCRAPES:
                return self._send(500, {"error": "fake scrape failure"})
            n = STATE["served"]
            body = "".join(f"vllm:{m}_sum {0.05 * n}\nvllm:{m}_count {n}\n" for m in (
                "e2e_request_latency_seconds", "request_queue_time_seconds", "request_prefill_time_seconds",
                "request_decode_time_seconds", "time_to_first_token_seconds"))
            body += f"vllm:inter_token_latency_seconds_sum {0.01 * 4 * n}\nvllm:inter_token_latency_seconds_count {4 * n}\n"
            return self._send(200, body, "text/plain")
        self._send(404, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); raw = self.rfile.read(n)
        try:
            req = json.loads(raw or b"{}")
        except ValueError:
            req = {}
        event(f"POST {self.path}")
        if self.path == "/start_profile" and str(PORT) in os.environ.get("FAKE_FAIL_START_PROFILE_PORTS", "").split():
            return self._send(500, {"error": "fake start_profile failure"})
        if self.path in ("/start_profile", "/stop_profile"):
            d = os.environ.get("FAKE_PROFILE_DIR")
            if self.path == "/stop_profile" and d:   # what the real profiler leaves: one device trace per TP rank
                os.makedirs(d, exist_ok=True)
                ports = lambda k: str(PORT) in os.environ.get(k, "").split()
                if not ports("FAKE_PROFILE_EMPTY_PORTS"):
                    for r in range(8):
                        if r == 5 and ports("FAKE_PROFILE_DROP_RANK_PORTS"):
                            continue
                        open(os.path.join(d, f"rank{r}.ntff"), "w").write("fake")
            return self._send(200, {})
        if os.environ.get("FAKE_FAIL_INFER") == "1":
            return self._send(500, {"error": "fake inference failure"})
        STATE["served"] += 1
        if self.path == "/v1/embeddings":
            return self._send(200, {"data": [{"embedding": [0.1, 0.2]}]})
        if self.path == "/v1/chat/completions":
            return self._send(200, {"choices": [{"message": {"content": "ok"}}]})
        if self.path == "/v1/audio/transcriptions":
            return self._send(200, {"text": "ok"})
        text = "ok-diverged" if str(PORT) in os.environ.get("FAKE_DIVERGE_PORTS", "").split() else "ok"
        choice = {"text": text}
        if req.get("return_token_ids") and str(PORT) not in os.environ.get("FAKE_NO_TOKEN_IDS_PORTS", "").split():
            k = int(req.get("max_tokens", 1))
            base = 7 if str(PORT) in os.environ.get("FAKE_DIVERGE_IDS_PORTS", "").split() else 5   # same text, other ids
            choice["token_ids"] = [base] * k if "ok-diverged" != text else [9] * k
            choice["prompt_token_ids"] = req.get("prompt") if isinstance(req.get("prompt"), list) else None
        self._send(200, {"choices": [choice], "usage": {"completion_tokens": 2}})


http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
