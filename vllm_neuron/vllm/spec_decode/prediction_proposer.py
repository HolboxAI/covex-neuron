# SPDX-License-Identifier: Apache-2.0
"""K9: predicted-output proposer (host-only, no draft model).

Coding/edit clients usually know most of the output in advance: the original file. The client sends it as the
*prediction*; each decode step drafts the next ``k`` tokens of that prediction, and the target model verifies them
in one multi-token step (vllm-neuron's existing EAGLE3/DFlash verify path, S_decode = 1 + k).

Correctness:
- **Greedy:** verification accepts a drafted token only when it equals the target's own argmax, so the output is
  identical to non-speculative greedy decoding whatever the prediction. Padding with any id is safe, because an
  accepted pad equals the target's own token.
- **Sampling (temperature > 0):** vLLM's rejection sampler treats host drafts as deterministic proposals
  (draft_probs=None). That preserves the output *distribution*, not identical tokens. K9 benchmarks are greedy only.

Transport: vLLM 0.29's OpenAI server has no ``prediction`` field. Clients pass ``vllm_xargs``
``{"covex_prediction_ids": [token ids]}`` (a flat list of ints), copied into ``SamplingParams.extra_args`` for both
completions and chat. Ids are validated against the vocabulary: an out-of-range id would reach the embedding lookup
of the verify step.

Cost: per-request state is built once at admission (an n-gram index plus a cursor). A step is O(k) on the aligned
path; re-alignment examines a bounded number of candidate positions near the cursor.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence

PREDICTION_KEY = "covex_prediction_ids"


def prediction_from_extra_args(extra_args: dict | None, vocab_size: int) -> list[int] | None:
    """The request's predicted token ids, or None when absent or invalid (any bool, non-int or out-of-vocab id)."""
    if not extra_args or PREDICTION_KEY not in extra_args:
        return None
    ids = extra_args[PREDICTION_KEY]
    if not isinstance(ids, (list, tuple)):
        return None
    if not all(type(x) is int and 0 <= x < vocab_size for x in ids):   # type() check rejects bools
        return None
    return list(ids)


class PredictionState:
    """Per-request alignment state between the generated output and the prediction."""

    def __init__(self, prediction: Sequence[int], min_ngram: int = 3, max_ngram: int = 8, max_candidates: int = 16,
                 aligned_start: bool = True):
        self.pred = list(prediction)
        self.min_ngram, self.max_ngram, self.max_candidates = min_ngram, max_ngram, max_candidates
        self.cursor = 0          # index in pred of the next expected token
        self.seen = 0            # output tokens already consumed
        # True for a client prediction (the output starts where the file starts); False for prompt lookup (the prompt
        # is a source of n-grams, not the beginning of the output)
        self.aligned = aligned_start
        # index of every min_ngram window: tuple -> sorted END positions (index just after the window)
        self.index: dict[tuple, list[int]] = {}
        for end in range(min_ngram, len(self.pred) + 1):
            self.index.setdefault(tuple(self.pred[end - min_ngram:end]), []).append(end)

    def _realign(self, output: Sequence[int]) -> None:
        n = self.min_ngram
        if len(output) < n:
            self.aligned = False
            return
        ends = self.index.get(tuple(output[-n:]))
        if not ends:
            self.aligned = False
            return
        i = bisect.bisect_left(ends, self.cursor)
        after = ends[i:i + self.max_candidates]                              # prefer moving forward through the file
        before = ends[max(0, i - self.max_candidates):i][::-1]               # then the nearest before the cursor
        best, best_key = None, None
        for rank, end in enumerate(after + before):
            match = n                                                        # extend the match backwards
            while (match < self.max_ngram and match < len(output) and end - match - 1 >= 0
                   and self.pred[end - match - 1] == output[-match - 1]):
                match += 1
            key = (match, end >= self.cursor, -abs(end - self.cursor), -rank)
            if best_key is None or key > best_key:
                best, best_key = end, key
        self.cursor, self.aligned = best, True

    def update(self, output: Sequence[int]) -> None:
        """Consume output tokens produced since the last call (accepted drafts + bonus tokens)."""
        new = output[self.seen:]
        diverged = False
        for tok in new:
            if self.aligned and self.cursor < len(self.pred) and self.pred[self.cursor] == tok:
                self.cursor += 1
            else:
                diverged = True
        self.seen = len(output)
        if diverged or not self.aligned:
            self._realign(output)

    def draft(self, k: int) -> list[int]:
        return self.pred[self.cursor:self.cursor + k] if self.aligned else []


class PredictionProposer:
    """Drafts exactly ``k`` ids per request (the fixed compiled verify width), padding short or absent drafts."""

    def __init__(self, num_speculative_tokens: int, pad_token_id: int = 0, min_ngram: int = 3, max_ngram: int = 8):
        if num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be >= 1")
        if not 1 <= min_ngram <= max_ngram:
            raise ValueError("need 1 <= min_ngram <= max_ngram")
        self.k, self.pad, self.min_ngram, self.max_ngram = num_speculative_tokens, pad_token_id, min_ngram, max_ngram
        self.states: dict[str, PredictionState] = {}

    def add_request(self, req_id: str, prediction: Sequence[int] | None, aligned_start: bool = True) -> None:
        if prediction:
            self.states[req_id] = PredictionState(prediction, self.min_ngram, self.max_ngram, aligned_start=aligned_start)

    def remove_request(self, req_id: str) -> None:
        self.states.pop(req_id, None)

    def propose_one(self, req_id: str, output: Sequence[int]) -> tuple[list[int], int]:
        """-> (exactly k draft ids, number of real predicted ids among them)."""
        st = self.states.get(req_id)
        if st is None:
            return [self.pad] * self.k, 0
        st.update(output)
        real = st.draft(self.k)
        return real + [self.pad] * (self.k - len(real)), len(real)

    def propose(self, req_ids: Sequence[str], outputs: Sequence[Sequence[int]]) -> list[list[int]]:
        if len(req_ids) != len(outputs):
            raise ValueError("one output per request")
        return [self.propose_one(r, o)[0] for r, o in zip(req_ids, outputs)]


class HostDraftProposer:
    """Runner-facing drafter with no draft model (speculative method "ngram").

    Plugs into vllm-neuron's existing multi-token verify path (S_decode = 1 + k, rejection sampler, KV rollback via
    num_computed_tokens) the way EagleProposer does, but proposes on the host:
      - the client's prediction (``vllm_xargs`` ``covex_prediction_ids``) when the request has one (K9);
      - otherwise prompt lookup: n-grams of the request's own prompt (K6).
    ``model`` is None: the runner skips draft-model loading, graph capture and warmup for it.
    """

    model = None

    def __init__(self, num_speculative_tokens: int, vocab_size: int, pad_token_id: int = 0, prompt_lookup: bool = True,
                 min_ngram: int = 3, max_ngram: int = 8):
        self.num_speculative_tokens = num_speculative_tokens
        self.vocab_size = vocab_size
        self.prompt_lookup = prompt_lookup
        self.proposer = PredictionProposer(num_speculative_tokens, pad_token_id, min_ngram, max_ngram)
        self.known: set[str] = set()

    def _admit(self, req_id: str, extra_args: dict | None, prompt_ids: Sequence[int] | None) -> None:
        pred = prediction_from_extra_args(extra_args, self.vocab_size)
        if pred:
            self.proposer.add_request(req_id, pred, aligned_start=True)
        elif self.prompt_lookup and prompt_ids:
            self.proposer.add_request(req_id, prompt_ids, aligned_start=False)
        self.known.add(req_id)

    def propose_host(self, requests: Sequence[tuple[str, dict | None, Sequence[int] | None, Sequence[int]]],
                     live: set[str] | None = None) -> list[list[int]]:
        """requests: (req_id, sampling extra_args, prompt token ids, output token ids so far), in batch order.
        live: ids of every request the runner still holds; state for any other id is dropped. -> k ids per request."""
        if live is not None:
            for r in list(self.known - live):
                self.known.discard(r); self.proposer.remove_request(r)
        out = []
        for req_id, extra, prompt_ids, output in requests:
            if req_id not in self.known:
                self._admit(req_id, extra, prompt_ids)
            out.append(self.proposer.propose_one(req_id, output)[0])
        return out
