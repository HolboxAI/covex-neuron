"""Minimal OpenAI-style fake for harness tests: /health, /metrics, completions, embeddings, chat, transcriptions.
FAKE_FAIL_INFER=1 -> /health OK but every inference request returns 500 (the 'healthy but cannot serve' case).
FAKE_EVENTS=<file> -> append one line per request ("METRICS" / "POST <path>") to observe ordering.

Run 2 additions (all per port, space-separated lists, so one env drives several arms):
  Token-id completions (replay.py): prompt = list of ids (or a list of such lists). Greedy ids are a deterministic
    function of the prompt, so every arm agrees unless told otherwise. logprobs=N -> vLLM completions logprobs
    (tokens / token_logprobs / top_logprobs; keys "token_id:<id>" when return_tokens_as_token_ids, else "tok<id>").
  FAKE_REPLAY_DIVERGE="<port>:<prompt_len>:<pos>[:near|far]" -> on that arm, prompts of that length change their
    token at <pos> (and everything after). near = the new token is the reference's 2nd choice with a 0.05 gap;
    far = a token outside the reference top-2.
  Needles (needles.py): a text prompt containing "The pass key is <digits>" (any case; "passkey: N" too) gets the
    digits back; otherwise "<key> = <value>" plus a question naming <key> gets <value>.
    FAKE_NEEDLE_WRONG="<port>:<prompt_tokens>" -> that arm answers wrong when usage.prompt_tokens equals it.
    Text prompts report prompt_tokens = whitespace-split word count, unless FAKE_PROMPT_TOKENS=<n> overrides.
  FAKE_FAIL_INFER_PORTS -> inference 500s on those arms only.
  FAKE_DIE_AFTER="<port>:<n>" -> the arm's process exits after serving n inference requests (server dies mid-run).
  FAKE_PROFILE_ALLOC_FAIL_PORTS -> /start_profile returns 200 but logs run 1's failure mode (no device buffer) to
    stderr, and /stop_profile writes no traces.
"""
import hashlib, http.server, json, os, re, sys

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
VOCAB = 151936


def on_port(var):
    return str(PORT) in os.environ.get(var, "").split()


def port_specs(var):
    """'<port>:a:b ...' entries for this port -> list of the remaining fields."""
    out = []
    for spec in os.environ.get(var, "").split():
        f = spec.split(":")
        if f[0] == str(PORT):
            out.append(f[1:])
    return out


def event(s):
    if EVENTS:
        with open(EVENTS, "a") as f:
            f.write(s + "\n")


def ref_choices(prompt_ids, pos):
    """Reference greedy token and runner-up at position pos, deterministic in (prompt, pos)."""
    h = hashlib.sha256((",".join(map(str, prompt_ids)) + f"|{pos}").encode()).digest()
    a = 1000 + int.from_bytes(h[:4], "big") % (VOCAB - 2000)
    b = a + 1 + h[4] % 7
    return a, b


def greedy(prompt_ids, k):
    """-> list of (token, logprob, [(id, lp), (id, lp)]) for k positions, with any configured divergence applied."""
    div = None
    for f in port_specs("FAKE_REPLAY_DIVERGE"):
        if int(f[0]) == len(prompt_ids):
            div = (int(f[1]), (f[2] if len(f) > 2 else "far"))
    out = []
    for pos in range(k):
        a, b = ref_choices(prompt_ids, pos)
        top = [(a, -0.10), (b, -0.15)]          # reference: a wins by 0.05 (a near tie)
        if div and pos >= div[0]:
            if div[1] == "near":
                top = [(b, -0.10), (a, -0.15)]  # same pair, order flipped by a 0.05 gap
            else:
                c = b + 11                      # outside the reference top-2
                top = [(c, -0.10), (a, -2.50)]
        out.append((top[0][0], top[0][1], top))
    return out


def completion_choice(req, prompt, index):
    k = int(req.get("max_tokens", 1))
    nlp = req.get("logprobs")
    as_ids = bool(req.get("return_tokens_as_token_ids"))
    key = (lambda t: f"token_id:{t}") if as_ids else (lambda t: f"tok{t}")
    if isinstance(prompt, list):                           # token-id prompt (replay / cache check)
        g = greedy(prompt, k)
        ids = [t for t, _, _ in g]
        if on_port("FAKE_DIVERGE_IDS_PORTS"):   # legacy knobs (cachecheck tests): other ids, same / different text
            ids = [7] * k
        elif on_port("FAKE_DIVERGE_PORTS"):
            ids = [9] * k
        choice = {"index": index, "text": " ".join(key(t) for t in ids), "finish_reason": "length"}
        if req.get("return_token_ids") and not on_port("FAKE_NO_TOKEN_IDS_PORTS"):
            choice["token_ids"] = ids
            choice["prompt_token_ids"] = prompt
        if nlp:
            choice["logprobs"] = {"tokens": [key(t) for t in ids], "token_logprobs": [lp for _, lp, _ in g],
                                  "top_logprobs": [{key(i): lp for i, lp in top[:int(nlp)]} for _, _, top in g],
                                  "text_offset": list(range(k))}
        return choice, len(prompt), k
    text = str(prompt)                                     # text prompt (needles, legacy clients)
    n_tok = int(os.environ.get("FAKE_PROMPT_TOKENS", 0)) or len(text.split())
    pairs = dict(re.findall(r"(\b[A-Za-z][\w-]*)\s*=\s*([\w-]+)", text))
    asked = [kk for kk in pairs if re.search(rf"\b{re.escape(kk)}\b\s*\?", text)]
    passkeys = re.findall(r"pass ?key\s*(?:is|:|=)\s*(\d+)", text, re.I)   # passkey-retrieval style needle
    if passkeys or asked:
        ans = passkeys[-1] if passkeys else pairs[asked[-1]]
        if any(int(f[0]) == n_tok for f in port_specs("FAKE_NEEDLE_WRONG")):
            ans = "wrong-" + ans
        out = " " + ans
    else:
        out = "ok-diverged" if on_port("FAKE_DIVERGE_PORTS") else "ok"
    choice = {"index": index, "text": out, "finish_reason": "stop"}
    if req.get("return_token_ids") and not on_port("FAKE_NO_TOKEN_IDS_PORTS"):   # legacy cachecheck path
        base = 7 if on_port("FAKE_DIVERGE_IDS_PORTS") else 5
        choice["token_ids"] = [base] * k if out != "ok-diverged" else [9] * k
        choice["prompt_token_ids"] = None
    return choice, n_tok, max(1, len(out.split()))


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
        if self.path == "/start_profile" and on_port("FAKE_FAIL_START_PROFILE_PORTS"):
            return self._send(500, {"error": "fake start_profile failure"})
        if self.path in ("/start_profile", "/stop_profile"):
            if on_port("FAKE_PROFILE_ALLOC_FAIL_PORTS"):   # run 1: the profiler cannot get its device buffer
                if self.path == "/start_profile":
                    print("(Worker_TP0 pid=%d) ERROR neuron_profiler: profile capture failed: RuntimeError: "
                          "Failed to allocate 128 MB device buffer for the profiler (out of device memory)" % os.getpid(),
                          file=sys.stderr, flush=True)
                return self._send(200, {})
            d = os.environ.get("FAKE_PROFILE_DIR")
            if self.path == "/stop_profile" and d:   # what the real profiler leaves: one device trace per TP rank
                os.makedirs(d, exist_ok=True)
                if not on_port("FAKE_PROFILE_EMPTY_PORTS"):
                    for r in range(8):
                        if r == 5 and on_port("FAKE_PROFILE_DROP_RANK_PORTS"):
                            continue
                        open(os.path.join(d, f"rank{r}.ntff"), "w").write("fake")
            return self._send(200, {})
        if os.environ.get("FAKE_FAIL_INFER") == "1" or on_port("FAKE_FAIL_INFER_PORTS"):
            return self._send(500, {"error": "fake inference failure"})
        STATE["served"] += 1
        for f in port_specs("FAKE_DIE_AFTER"):
            if STATE["served"] > int(f[0]):
                print(f"(EngineCore pid={os.getpid()}) ERROR fake engine died after {f[0]} requests", file=sys.stderr, flush=True)
                os._exit(1)
        if self.path == "/v1/embeddings":
            return self._send(200, {"data": [{"embedding": [0.1, 0.2]}]})
        if self.path == "/v1/chat/completions":
            return self._send(200, {"choices": [{"message": {"content": "ok"}}],
                                    "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}})
        if self.path == "/v1/audio/transcriptions":
            return self._send(200, {"text": "ok"})
        prompt = req.get("prompt", "")
        batch = prompt if isinstance(prompt, list) and prompt and isinstance(prompt[0], (list, str)) else [prompt]
        choices, pt, ct = [], 0, 0
        for i, p in enumerate(batch):
            c, a, b = completion_choice(req, p, i)
            choices.append(c); pt += a; ct += b
        self._send(200, {"choices": choices, "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                                                       "total_tokens": pt + ct}})


http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
