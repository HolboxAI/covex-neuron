"""K6/K9 host drafter: runner-facing behaviour, without torch/vllm (the runner module needs a Neuron venv)."""
import importlib.util
import textwrap
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _pp():
    spec = importlib.util.spec_from_file_location("pp", ROOT / "vllm_neuron/vllm/spec_decode/prediction_proposer.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _runner_fn(pp):
    """_propose_host_drafts as written in the runner, bound to a fake runner."""
    src = (ROOT / "vllm_neuron/vllm/worker/neuron_model_runner.py").read_text()
    i = src.index("    def _propose_host_drafts(self)"); j = src.index("    def _propose_draft_token_ids(", i)
    ns = {"HostDraftProposer": pp.HostDraftProposer}
    exec(textwrap.dedent(src[i:j]), ns)
    return ns["_propose_host_drafts"]


def _req(prompt, out, extra=None):
    return types.SimpleNamespace(prompt_token_ids=prompt, output_token_ids=out,
                                 sampling_params=types.SimpleNamespace(extra_args=extra))


def test_runner_drafts_prediction_and_prompt_lookup_per_request():
    pp = _pp(); fn = _runner_fn(pp)
    d = pp.HostDraftProposer(4, vocab_size=1000, min_ngram=3, max_ngram=5)
    file = [10, 11, 12, 13, 14, 15, 16, 17]
    prompt_b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 50, 51, 52, 53]
    reqs = {"a": _req([900, 901], [10, 11], {"covex_prediction_ids": file}),     # K9: client prediction
            "b": _req(prompt_b, [2, 3, 4]),                                       # K6: prompt lookup
            "c": _req([7, 7, 7], [9]),                                            # nothing to match
            "bad": _req([1], [], {"covex_prediction_ids": [5, 99999]})}           # out-of-vocab: ignored
    runner = types.SimpleNamespace(drafter=d, requests=reqs, input_batch=types.SimpleNamespace(req_ids=["a", "b", "c", "bad"]))
    drafts = fn(runner)
    assert drafts[0] == [12, 13, 14, 15]
    assert drafts[1] == [5, 6, 7, 8]                     # output 2,3,4 found in the prompt -> continue from it
    assert drafts[2] == [0, 0, 0, 0] and len(drafts) == 4 and all(len(x) == 4 for x in drafts)
    assert drafts[3] == [0, 0, 0, 0]
    del reqs["a"]; runner.input_batch.req_ids = ["b"]   # finished request: its state is dropped
    fn(runner)
    assert "a" not in d.known and "a" not in d.proposer.states


def test_prompt_lookup_does_not_assume_output_starts_at_prompt():
    pp = _pp(); d = pp.HostDraftProposer(3, vocab_size=1000, min_ngram=3, max_ngram=5)
    # the first decode step must not draft prompt[0:3] (the output does not begin with the prompt)
    assert d.propose_host([("r", None, [5, 6, 7, 8, 9], [1])]) == [[0, 0, 0]]


def test_host_drafter_has_no_model_for_the_runner_guards():
    pp = _pp(); d = pp.HostDraftProposer(4, vocab_size=10)
    assert d.model is None and d.num_speculative_tokens == 4


def test_runner_skips_every_draft_model_step_for_host_drafts():
    src = (ROOT / "vllm_neuron/vllm/worker/neuron_model_runner.py").read_text()
    for marker in ('logger.info("Spec decode enabled. Loading draft model ...")', "# Draft model capture",
                   "# === Draft model graph capture ===", "# === Draft model warmup ==="):
        i = src.index(marker)
        window = src[max(0, i - 120):i + 120]
        assert "not self.is_host_draft_spec" in window, marker
    assert 'self.speculative_config.method == "ngram"' in src and "needs synchronous scheduling" in src
    i = src.index("    def _propose_draft_token_ids(")
    assert "if self.is_host_draft_spec:\n            return self._propose_host_drafts()" in src[i:i + 900]



def test_decode_warmup_draft_calls_are_noops_for_host_drafts():
    """Astra: the two decode-warmup calls (graph_extract, warmup) sit inside the block that also compiles the
    target's verify graphs, so the host drafter must accept them as no-ops (not AttributeError)."""
    pp = _pp(); d = pp.HostDraftProposer(4, vocab_size=10)
    assert d.graph_extract(num_tokens=40, num_reqs=8, attn_metadata={}) is None
    assert d.warmup(num_tokens=40, num_reqs=8, attn_metadata={}) is None
    src = (ROOT / "vllm_neuron/vllm/worker/neuron_model_runner.py").read_text()
    for call in ("self.drafter.graph_extract(", "self.drafter.warmup("):
        assert src.count(call) == 2, call                                    # prefill (guarded) + decode (no-op)


def test_k9_serve_arms_differ_only_in_speculation():
    import os, subprocess
    out = {}
    for a in ("k9-U", "k9-E3", "k9-P"):
        r = subprocess.run(["bash", str(ROOT / "benchmarks/five/serve.sh"), a, "8100"], capture_output=True, text=True,
                           env=dict(os.environ, DRYRUN="1", UPSTREAM_PY="/up/python"))
        assert r.returncode == 0, r.stderr
        out[a] = r.stdout.splitlines()[-1]
    assert "--speculative-config" not in out["k9-U"] and "eagle3" in out["k9-E3"] and "/up/python" in out["k9-E3"]
    assert "ngram" in out["k9-P"] and "--no-async-scheduling" in out["k9-P"] and "/up/python" not in out["k9-P"]
    import re
    strip = lambda s: re.sub(r" --speculative-config \S+| --no-async-scheduling", "", s.split(" serve ", 1)[1])
    assert strip(out["k9-U"]) == strip(out["k9-E3"]) == strip(out["k9-P"])        # same model/context/batch config
