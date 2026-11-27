# SPDX-License-Identifier: Apache-2.0
"""K9: predicted-output proposer (host-only, no draft model).

Coding/edit clients usually know most of the output in advance: the original file. The client sends it as the
*prediction*; each decode step drafts the next ``k`` tokens of that prediction, and the target model verifies them
in one multi-token step (vllm-neuron's existing EAGLE3/DFlash verify path, S_decode = 1 + k).

Correctness never depends on the prediction. Greedy verification accepts a drafted token only when it equals the
target's own argmax, so the output is identical to non-speculative greedy decoding. A wrong, short or absent
prediction only costs speed. Padding with any id is therefore safe: an accepted pad equals the target's own token.

Transport: vLLM 0.29's OpenAI server has no ``prediction`` field, so clients pass ``vllm_xargs``
``{"covex_prediction_ids": [...]}``, which the server copies into ``SamplingParams.extra_args``.
"""

from __future__ import annotations

from collections.abc import Sequence

PREDICTION_KEY = "covex_prediction_ids"


def prediction_from_extra_args(extra_args: dict | None) -> list[int] | None:
    """The request's predicted token ids, or None. Non-int entries make the prediction invalid (ignored)."""
    if not extra_args or PREDICTION_KEY not in extra_args:
        return None
    ids = extra_args[PREDICTION_KEY]
    if not isinstance(ids, (list, tuple)) or not all(isinstance(x, int) and x >= 0 for x in ids):
        return None
    return list(ids)


class PredictionProposer:
    """Drafts the next ``k`` predicted tokens for each request.

    Alignment:
      1. Fast path: the output so far equals the prediction's prefix, so draft the tokens that follow it.
      2. Re-align after an edit: find the latest occurrence in the prediction of the output's last ``n`` tokens
         (n from ``max_ngram`` down to ``min_ngram``) and continue from just after it. This is how an edit that
         inserts or deletes text resynchronises with the unchanged rest of the file.
      3. No match, or no prediction: no draft (all pad), i.e. normal decoding.
    Always returns exactly ``k`` ids per request, the fixed compiled verify width.
    """

    def __init__(self, num_speculative_tokens: int, pad_token_id: int = 0, min_ngram: int = 3, max_ngram: int = 8):
        if num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be >= 1")
        if not 1 <= min_ngram <= max_ngram:
            raise ValueError("need 1 <= min_ngram <= max_ngram")
        self.k = num_speculative_tokens
        self.pad = pad_token_id
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram

    def _continuation_start(self, prediction: Sequence[int], output: Sequence[int]) -> int | None:
        n_out = len(output)
        if n_out <= len(prediction) and list(prediction[:n_out]) == list(output):
            return n_out
        for n in range(min(self.max_ngram, n_out), self.min_ngram - 1, -1):
            tail = list(output[-n:])
            # latest match wins: edits move forward through the file
            for start in range(len(prediction) - n, -1, -1):
                if list(prediction[start:start + n]) == tail:
                    return start + n
        return None

    def propose_one(self, prediction: Sequence[int] | None, output: Sequence[int]) -> tuple[list[int], int]:
        """-> (exactly k draft ids, number of real predicted ids among them)."""
        if not prediction:
            return [self.pad] * self.k, 0
        start = self._continuation_start(prediction, output)
        if start is None:
            return [self.pad] * self.k, 0
        real = list(prediction[start:start + self.k])
        return real + [self.pad] * (self.k - len(real)), len(real)

    def propose(self, predictions: Sequence[Sequence[int] | None], outputs: Sequence[Sequence[int]]) -> list[list[int]]:
        if len(predictions) != len(outputs):
            raise ValueError("one prediction slot per request")
        return [self.propose_one(p, o)[0] for p, o in zip(predictions, outputs)]
