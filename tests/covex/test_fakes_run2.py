"""Run-2 fakes (Claude-B): the fake server / fake vllm behave like the real ones for every run-2 client and failure.
Runs on macOS and Linux with plain Python (no Neuron)."""
import json, os, socket, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FAKE = ROOT / "tests/covex/fake_server.py"
VLLM = ROOT / "tests/covex/fake_bin/vllm"
LONG = ROOT / "benchmarks/rehearsal/long_prompt.py"


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class Server:
    def __init__(self, env=None, port=None):
        self.port = port or free_port()
        self.p = subprocess.Popen([sys.executable, str(FAKE), "--port", str(self.port)], env=dict(os.environ, **(env or {})),
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1); return
            except Exception:
                time.sleep(0.05)
        raise RuntimeError("fake server did not start")

    def post(self, path, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.load(r)

    def stop(self):
        self.p.kill(); out, err = self.p.communicate(); return err

    def __enter__(self):
        return self

    def __exit__(self, *a):
        if self.p.poll() is None:
            self.stop()


def completion(s, prompt, k=8, **kw):
    return s.post("/v1/completions", dict(model="m", prompt=prompt, max_tokens=k, temperature=0, logprobs=2,
                                          return_token_ids=True, return_tokens_as_token_ids=True, **kw))[1]


P33 = list(range(1000, 1033)); P40 = list(range(2000, 2040))


def test_token_id_prompts_are_deterministic_across_arms_with_top2_logprobs():
    with Server() as a, Server() as b:
        ca, cb = completion(a, P33), completion(b, P33)
        ch = ca["choices"][0]
        assert ch["token_ids"] == cb["choices"][0]["token_ids"] and len(ch["token_ids"]) == 8
        assert ch["prompt_token_ids"] == P33 and ca["usage"]["prompt_tokens"] == 33
        lp = ch["logprobs"]
        assert len(lp["top_logprobs"]) == 8 and all(len(t) == 2 for t in lp["top_logprobs"])
        assert all(k.startswith("token_id:") for t in lp["top_logprobs"] for k in t)
        assert lp["tokens"][0] == f"token_id:{ch['token_ids'][0]}"
        # the greedy token is the top-1 entry
        assert all(max(t, key=t.get) == f"token_id:{i}" for t, i in zip(lp["top_logprobs"], ch["token_ids"]))


def test_batched_prompt_list_returns_one_choice_per_prompt_in_order():
    with Server() as s:
        out = completion(s, [P33, P40])
        assert [c["index"] for c in out["choices"]] == [0, 1]
        assert out["choices"][0]["token_ids"] == completion(s, P33)["choices"][0]["token_ids"]
        assert out["usage"]["prompt_tokens"] == 73


def test_replay_divergence_near_and_far():
    with Server() as ref:
        r = completion(ref, P33, k=32)["choices"][0]
        port = free_port()
        with Server({"FAKE_REPLAY_DIVERGE": f"{port}:33:15:near"}, port=port) as near:
            n = completion(near, P33, k=32)["choices"][0]
            assert n["token_ids"][:15] == r["token_ids"][:15] and n["token_ids"][15] != r["token_ids"][15]
            # near tie: F's token is the reference's 2nd choice, gap < 0.1
            top = r["logprobs"]["top_logprobs"][15]
            alt = f"token_id:{n['token_ids'][15]}"
            assert alt in top and abs(top[max(top, key=top.get)] - top[alt]) < 0.1
            assert completion(near, P40, k=32)["choices"][0]["token_ids"] == completion(ref, P40, k=32)["choices"][0]["token_ids"]
        port = free_port()
        with Server({"FAKE_REPLAY_DIVERGE": f"{port}:33:8:far"}, port=port) as far:
            f = completion(far, P33, k=32)["choices"][0]
            assert f["token_ids"][:8] == r["token_ids"][:8]
            assert f"token_id:{f['token_ids'][8]}" not in r["logprobs"]["top_logprobs"][8]


def test_needle_answer_and_injected_wrong_answer():
    text = "filler " * 50 + "zebra-key = 7f3a9c " + "filler " * 50 + "What is zebra-key ?"
    with Server() as s:
        out = s.post("/v1/completions", {"model": "m", "prompt": text, "max_tokens": 16, "temperature": 0})[1]
        assert out["choices"][0]["text"].strip() == "7f3a9c" and out["usage"]["prompt_tokens"] == len(text.split())
    port = free_port()
    with Server({"FAKE_NEEDLE_WRONG": f"{port}:8204", "FAKE_PROMPT_TOKENS": "8204"}, port=port) as w:
        out = w.post("/v1/completions", {"model": "m", "prompt": text, "max_tokens": 16})[1]
        assert out["choices"][0]["text"].strip() != "7f3a9c" and out["usage"]["prompt_tokens"] == 8204


def test_numeric_passkey_needle():
    text = "The grass is green. " * 40 + "The pass key is 68120. Remember it. " + "The sky is blue. " * 40 + \
           "What is the pass key? The pass key is"
    with Server() as s:
        out = s.post("/v1/completions", {"model": "m", "prompt": text, "max_tokens": 16, "temperature": 0})[1]
        assert out["choices"][0]["text"].strip() == "68120"
    port = free_port()
    with Server({"FAKE_PROMPT_TOKENS": "9644", "FAKE_NEEDLE_WRONG": f"{port}:9644"}, port=port) as w:
        out = w.post("/v1/completions", {"model": "m", "prompt": text, "max_tokens": 16})[1]
        assert out["usage"]["prompt_tokens"] == 9644 and out["choices"][0]["text"].strip() != "68120"


def test_legacy_cachecheck_knobs_still_diverge():
    port = free_port()
    with Server({"FAKE_DIVERGE_PORTS": str(port)}, port=port) as s, Server() as ref:
        assert completion(s, P33)["choices"][0]["token_ids"] != completion(ref, P33)["choices"][0]["token_ids"]


def test_injected_failures_server_dies_client_500_profile_alloc():
    port = free_port()
    with Server({"FAKE_DIE_AFTER": f"{port}:2"}, port=port) as s:
        completion(s, P33); completion(s, P33)
        with pytest.raises(Exception):
            completion(s, P33)
        s.p.wait(timeout=5); assert s.p.returncode == 1
    port = free_port()
    with Server({"FAKE_FAIL_INFER_PORTS": str(port)}, port=port) as s:
        with pytest.raises(urllib.error.HTTPError) as e:
            completion(s, P33)
        assert e.value.code == 500
    port = free_port()
    with Server({"FAKE_PROFILE_ALLOC_FAIL_PORTS": str(port), "FAKE_PROFILE_DIR": "/nonexistent/should-not-write"}, port=port) as s:
        assert s.post("/start_profile", {})[0] == 200 and s.post("/stop_profile", {})[0] == 200
        err = s.stop()
        assert "profil" in err.lower() and "error" in err.lower() and "128 MB" in err


def test_profile_success_writes_eight_distinct_rank_traces(tmp_path):
    d = tmp_path / "profiles/q32-U"
    with Server({"FAKE_PROFILE_DIR": str(d)}) as s:
        s.post("/start_profile", {}); s.post("/stop_profile", {})
    assert sorted(p.name for p in d.glob("*.ntff")) == [f"rank{r}.ntff" for r in range(8)]
    assert (d / "neffs/graph.neff").exists()


def _serve(cfg, env):
    """Start the fake vllm serve and return (process, port, log lines up to the ready point)."""
    port = free_port()
    p = subprocess.Popen(["bash", str(VLLM), "serve", "m", "--port", str(port), "--additional-config", cfg],
                         env=dict(os.environ, FAKE_VLLM_DELAY="0", **env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p, port


def _serve_log(cfg, env):
    p, port = _serve(cfg, env)
    for _ in range(200):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1); break
        except Exception:
            time.sleep(0.05)
    p.kill(); return p.communicate()[0]


def test_fake_serve_logs_runner_choice_and_headroom_verbatim():
    ut = _serve_log('{"neuron_config":{"decode_context_length_buckets":[2048,8192]}}', {"VLLM_NEURON_RESERVED_HEADROOM_GIB": "7"})
    assert "Neuron Model Runner V2 does not yet support decode context-length bucketing; using V1." in ut
    assert "Using decode_context_length_buckets from config: [2048, 8192]" in ut
    assert "standalone V2" not in ut
    assert ut.count("GiB headroom (explicit)") == 8 and "- 7.00 GiB headroom (explicit) = 9.36 GiB" in ut
    u = _serve_log('{"neuron_config":{"num_seqs_buckets":[1,2,4,8]}}', {})
    assert "Using the standalone V2 Neuron model runner." in u and "using V1" not in u
    assert "GiB headroom (default)" in u


def test_fake_serve_slow_ready_is_per_port_and_bounded():
    p, port = _serve("{}", {})
    p.kill(); p.communicate()
    t0 = time.time()
    p = subprocess.Popen(["bash", str(VLLM), "serve", "m", "--port", str(port), "--additional-config", "{}"],
                         env=dict(os.environ, FAKE_VLLM_DELAY="0", FAKE_SLOW_READY=f"{port}:2"),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    up = None
    for _ in range(200):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1); up = time.time() - t0; break
        except Exception:
            time.sleep(0.05)
    p.kill(); p.wait()
    assert up is not None and up >= 2.0


def _bench(tmp_path, port, extra_args=(), env=None):
    rd = tmp_path / f"res{len(list(tmp_path.glob('res*')))}"   # a fresh result dir per call: no stale result
    cmd = ["bash", str(VLLM), "bench", "serve", "--base-url", f"http://127.0.0.1:{port}", "--num-prompts", "8",
           "--random-input-len", "5120", "--random-output-len", "256", "--result-dir", str(rd), "--result-filename", "x.json",
           *extra_args]
    r = subprocess.run(cmd, env=dict(os.environ, **(env or {})), capture_output=True, text=True)
    return r, (json.loads((rd / "x.json").read_text()) if (rd / "x.json").exists() else None)


def test_fake_bench_detailed_fields_and_range_ratio(tmp_path):
    r, d = _bench(tmp_path, 8100, ["--save-detailed", "--random-range-ratio", '{"input":0.2,"output":0}'])
    assert r.returncode == 0, r.stderr
    for k in ("ttfts", "itls", "output_lens", "input_lens", "e2els"):
        assert isinstance(d[k], list) and len(d[k]) == d["completed"] == 8, k
    assert all(4096 <= n <= 6144 for n in d["input_lens"]) and len(set(d["input_lens"])) > 1
    assert d["total_input_tokens"] == sum(d["input_lens"]) and d["output_lens"] == [256] * 8
    assert all(len(x) == 255 for x in d["itls"])
    # vLLM's TPOT formula: (e2e - ttft) / (output_len - 1)
    tpot = [(e - t) / (o - 1) for e, t, o in zip(d["e2els"], d["ttfts"], d["output_lens"])]
    assert all(abs(x * 1000 - d["median_tpot_ms"]) < 1e-6 for x in tpot)


def test_fake_bench_chunked_streaming_and_latency_knobs(tmp_path):
    env = {"FAKE_BENCH_ITL_CHUNKS": "8100:100", "FAKE_BENCH_TPOT_MS": "8100:40", "FAKE_BENCH_TTFT": "8100:1.5"}
    r, d = _bench(tmp_path, 8100, ["--save-detailed"], env)
    assert r.returncode == 0 and all(len(x) == 100 for x in d["itls"]), r.stderr
    # Astra's repro: 100 intervals over 255 tokens of decode -> naive per-interval mean is 102 ms, true TPOT 40 ms
    assert abs(sum(d["itls"][0]) / 100 * 1000 - 102.0) < 0.01 and abs(sum(d["itls"][0]) / 255 * 1000 - 40.0) < 1e-6
    assert d["ttfts"] == [1.5] * 8
    r, d = _bench(tmp_path, 8100, ["--save-detailed"], {"FAKE_BENCH_COMPLETED": "8100:3"})
    assert d["completed"] == 3 and d["failed"] == 5 and len(d["ttfts"]) == 3
    r, d = _bench(tmp_path, 8100, [], {"FAKE_BENCH_FAIL_PORTS": "8100"})
    assert r.returncode != 0 and d is None


def test_long_prompt_probe_passes_and_fails_loudly():
    with Server() as s:
        r = subprocess.run([sys.executable, str(LONG), str(s.port), "9644,24044", "8"], capture_output=True, text=True)
        assert r.returncode == 0 and "9644 tok -> 8 ids" in r.stdout and "24044 tok -> 8 ids" in r.stdout, r.stderr
    port = free_port()
    with Server({"FAKE_FAIL_INFER_PORTS": str(port)}, port=port) as s:
        r = subprocess.run([sys.executable, str(LONG), str(port), "9644", "8"], capture_output=True, text=True)
        assert r.returncode != 0 and json.loads(r.stderr.strip().splitlines()[-1])["pass"] is False
    r = subprocess.run([sys.executable, str(LONG), str(free_port()), "9644", "8"], capture_output=True, text=True)
    assert r.returncode != 0 and "pass" in r.stderr


def test_rehearsal_covers_run2_arms_and_requires_long_prompts():
    sh = (ROOT / "benchmarks/rehearsal/rehearse.sh").read_text()
    subprocess.run(["bash", "-n", str(ROOT / "benchmarks/rehearsal/rehearse.sh")], check=True)
    assert "q32-Ut) echo 8011" in sh and "long_prompt.py" in sh and "q32-Ut:compile-alias" in sh
    assert 'LONG_ARMS=${REHEARSAL_LONG_ARMS:-"q32-U q32-Ut q32-F"}' in sh
    assert "RECOMPILE_RE=" in sh
    # exact arm matching: no substring test that would let q32-U match q32-Ut
    assert '== *"$a"* ]]' not in sh


def test_fake_bench_range_ratio_matches_vllm_029(tmp_path):
    """A bare ratio varies output lengths too (vLLM 0.29); the JSON form keeps output exactly 256."""
    import json, os, subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    def bench(rr):
        r = subprocess.run([str(root / "tests/covex/fake_bin/vllm"), "bench", "serve", "--base-url", "http://127.0.0.1:8100",
                            "--num-prompts", "16", "--random-input-len", "5120", "--random-output-len", "256",
                            "--random-range-ratio", rr, "--save-detailed", "--result-dir", str(tmp_path), "--result-filename", "x.json"],
                           capture_output=True, text=True, env=dict(os.environ))
        assert r.returncode == 0, r.stderr
        return json.load(open(tmp_path / "x.json"))
    d = bench("0.2")
    assert len(set(d["output_lens"])) > 1 and d["total_output_tokens"] != 16 * 256
    d = bench('{"input":0.2,"output":0}')
    assert d["output_lens"] == [256] * 16 and d["total_output_tokens"] == 4096
    assert min(d["input_lens"]) >= 4096 and max(d["input_lens"]) <= 6144 and len(set(d["input_lens"])) > 1
    assert all(len(x) == 255 for x in d["itls"])


def test_rehearsal_recompile_regex_passes_real_stock_log_and_catches_recompiles(tmp_path):
    """Astra: `recompil` as a substring matched "Precompile: OFF" in every successful stock log (a false FAIL)."""
    import re, subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    rx = re.search(r"^RECOMPILE_RE='([^']+)'", (root / "benchmarks/rehearsal/rehearse.sh").read_text(), re.M).group(1)
    grep = lambda f: subprocess.run(["grep", "-qiE", rx, str(f)]).returncode == 0
    real = root / "tests/covex/fixtures/real_stock_precompile_line.log"
    assert "Precompile: OFF" in real.read_text() and not grep(real)
    for bad in ("torch._dynamo hit config.recompile_limit (8)", "RuntimeError: fail_on_recompile is set",
                "Recompiling function forward in model.py"):
        f = tmp_path / "x.log"; f.write_text(real.read_text() + bad + "\n")
        assert grep(f), bad


def test_long_prompt_probe_rejects_truncated_prompt_and_null_ids(tmp_path):
    """Astra: one input token and eight null ids passed as '24044 tok -> 8 ids'."""
    import json, subprocess, sys, threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    mode = {}
    class Hd(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            p = body["prompt"]
            c = {"null": {"token_ids": [None] * 8, "prompt_token_ids": p},
                 "trunc": {"token_ids": [1] * 8, "prompt_token_ids": p[:1]},
                 "ok": {"token_ids": [1] * 8, "prompt_token_ids": p}}[mode["m"]]
            out = json.dumps({"choices": [c], "usage": {"prompt_tokens": len(c["prompt_token_ids"])}}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
    srv = HTTPServer(("127.0.0.1", 0), Hd); threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        for m, want in (("null", 1), ("trunc", 1), ("ok", 0)):
            mode["m"] = m
            r = subprocess.run([sys.executable, str(root / "benchmarks/rehearsal/long_prompt.py"), str(srv.server_port), "9644", "8"],
                               capture_output=True, text=True, timeout=60)
            assert (r.returncode != 0) == bool(want), (m, r.stdout, r.stderr)
    finally:
        srv.shutdown()
