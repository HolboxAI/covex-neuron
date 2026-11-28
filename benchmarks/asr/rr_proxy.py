#!/usr/bin/env python3
"""Round-robin HTTP front for N replicas of one model, so the unchanged ASR client can drive "2 x TP4 on two chips"
exactly like "1 x TP8". Usage: rr_proxy.py <listen_port> <backend_port> [<backend_port> ...]

- Each request goes to the next replica (per request, not per connection: the client keeps connections alive).
- Responses are streamed through as they arrive (chat_stream phase), close-delimited (HTTP/1.0 style).
- GET /health is 200 only when every replica is healthy.
- GET /metrics concatenates every replica's /metrics, so counter sums (e.g. e2e_request_latency count) cover both.
- Backlog 2048: concurrent bursts queue instead of being reset.
The proxy adds one local hop; its overhead is measured by running the 1-replica layout through it as well.
"""
import http.server
import itertools
import json
import sys
import threading
import time

import httpx

HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding",
       "upgrade", "content-length", "host"}


def make_handler(backends):
    rr = itertools.cycle(backends)
    lock = threading.Lock()
    # per-replica accounting (Kiln found one engine serving 63 of 64 requests): requests, request bytes (audio size),
    # busy seconds (wall time with >= 1 request in flight) and in-flight peak; GET /proxy_stats returns it as JSON
    stats = {b: {"requests": 0, "errors": 0, "request_bytes": 0, "busy_s": 0.0, "inflight": 0, "inflight_peak": 0,
                 "_busy_since": None} for b in backends}

    def begin(port, nbytes):
        with lock:
            st = stats[port]; st["requests"] += 1; st["request_bytes"] += nbytes; st["inflight"] += 1
            st["inflight_peak"] = max(st["inflight_peak"], st["inflight"])
            if st["inflight"] == 1:
                st["_busy_since"] = time.monotonic()

    def end(port, ok):
        with lock:
            st = stats[port]; st["inflight"] -= 1; st["errors"] += 0 if ok else 1
            if st["inflight"] == 0 and st["_busy_since"] is not None:
                st["busy_s"] += time.monotonic() - st["_busy_since"]; st["_busy_since"] = None

    def snapshot():
        with lock:
            now = time.monotonic()
            return {str(b): {k: (round(v + (now - st["_busy_since"] if k == "busy_s" and st["_busy_since"] else 0), 3)
                                 if k == "busy_s" else v) for k, v in st.items() if not k.startswith("_")}
                    for b, st in stats.items()}
    client = httpx.Client(timeout=httpx.Timeout(1800, connect=10), limits=httpx.Limits(max_connections=512))

    def nxt():
        with lock:
            return next(rr)

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):
            pass

        def _all(self, path):
            outs = []
            for b in backends:
                try:
                    r = client.get(f"http://127.0.0.1:{b}{path}", timeout=10)
                    outs.append((r.status_code, r.content))
                except httpx.HTTPError:
                    outs.append((502, b""))
            return outs

        def _reply(self, code, body, ctype="text/plain"):
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)

        def _forward(self, method):
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else None
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            port = nxt()
            begin(port, n); ok = False
            try:
                with client.stream(method, f"http://127.0.0.1:{port}{self.path}", content=body, headers=headers) as r:
                    self.send_response(r.status_code)
                    for k, v in r.headers.items():
                        if k.lower() not in HOP:
                            self.send_header(k, v)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    for chunk in r.iter_raw():
                        self.wfile.write(chunk); self.wfile.flush()
                    ok = r.status_code < 400
            except httpx.HTTPError as e:
                self._reply(502, f"replica {port}: {type(e).__name__}".encode())
            finally:
                end(port, ok)

        def do_GET(self):
            if self.path == "/health":
                outs = self._all("/health")
                return self._reply(200 if all(c == 200 for c, _ in outs) else 503, b"")
            if self.path == "/proxy_stats":
                return self._reply(200, json.dumps(snapshot()).encode(), "application/json")
            if self.path == "/metrics":
                outs = self._all("/metrics")
                if not all(c == 200 for c, _ in outs):
                    return self._reply(502, b"")
                return self._reply(200, b"\n".join(b for _, b in outs))
            self._forward("GET")

        def do_POST(self):
            self._forward("POST")

    return H


class Srv(http.server.ThreadingHTTPServer):
    request_queue_size = 2048
    daemon_threads = True


def main(argv):
    if len(argv) < 3:
        sys.exit("usage: rr_proxy.py <listen_port> <backend_port> [<backend_port> ...]")
    listen, backends = int(argv[1]), [int(x) for x in argv[2:]]
    Srv(("127.0.0.1", listen), make_handler(backends)).serve_forever()


if __name__ == "__main__":
    main(sys.argv)
