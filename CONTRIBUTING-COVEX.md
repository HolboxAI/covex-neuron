# COVEX Neuron contribution rules

The goal of this repo is simple: **`main` must always be at least as fast as stock vllm-neuron, and faster wherever we have measured it.**

## 1. Where changes go

- **No direct pushes to `main`.** Everything goes in through a pull request.
- **Branch from `main`.** Use one of these names:
  - `feat/<topic>` for new optimizations
  - `port/<topic>` for bringing in work from a personal fork
  - `fix/<topic>` for correctness fixes
- **Never commit to `upstream-main`.** It is a mirror that the nightly sync overwrites.
- **One optimization per PR.** Keep PRs small enough to benchmark in isolation.

## 2. What it takes to merge a performance change

A PR that changes runtime behavior merges only when all of these hold:

1. **There is a benchmark against stock vllm-neuron at the exact same settings.**
   - Settings that must match: model, instance type, Neuron SDK, parallelism, max-model-len, buckets, dataset, input/output lengths, concurrency and number of prompts.
   - The baseline is the `upstream-main` commit this PR is based on, run with the change turned off.
   - Follow [`benchmarks/README.md`](benchmarks/README.md).
2. **The win is real.**
   - At least **+3% throughput (tok/s)**, or at least **−5% on TTFT or ITL p50**, at one or more concurrency levels.
   - Use the median of 3 runs for each side.
3. **Nothing regresses.**
   - No other reported metric, at any tested concurrency, gets worse by more than **2%**.
4. **Accuracy holds.**
   - Report the model's accuracy check: WER for ASR, GSM8K or a logit match for LLMs.
   - Any change must be within the noise of the baseline, and you must say so in the PR.
5. **The raw results are committed** under `benchmarks/results/<date>-<model>-<topic>.md` (or `.json`), from both runs.
6. **It sits behind a flag.**
   - Add a `VLLM_NEURON_*` / `COVEX_*` env var in `vllm_neuron/envs.py`.
   - The default is **on only for the models listed in the results file**, and off everywhere else.
   - Setting the flag to `0` must restore stock behavior.
7. **The README table is updated** with the new row.

PRs that don't change runtime behavior (docs, CI, tests, the upstream sync) take the `no-perf-change` label instead. The benchmark gate then skips them.

## 3. Sign-off and the change registry

- **DCO sign-off on every commit** (`git commit -s`). vllm-project/vllm-neuron requires it, and the `dco` check enforces it here too, so any change can go upstream without rewriting history.
- **Squash-merge feature PRs.** One optimization becomes one commit on `main` that customers and the community can cherry-pick.
- **Add or update the row in [COVEX_CHANGES.md](COVEX_CHANGES.md)** in the same PR: flag, default-on models, result, upstream status.

## 4. Releases for customers

- **Customers pin a release tag, not `main`.**
- Tag format: `covex-<upstream release>-<n>`, e.g. `covex-0.29.0.1.2.0-1`.
- **Each GitHub release states:**
  - the upstream commit it is based on and the validated Neuron SDK
  - the COVEX_CHANGES rows it contains
  - benchmark links
- **Cut a release when one of these happens:**
  - a new validated win merges
  - upstream ships a release (rebase the COVEX commits onto it in the release branch `covex/<upstream release>`)
- **Never re-tag.** A fix gets a new `-<n+1>`.

## 5. Review

- Each PR needs one approving review from a code owner (see `.github/CODEOWNERS`).
- The **benchmark-gate** check must pass. It verifies that the PR template's benchmark section is filled in and that a results file is in the diff.

## 6. Upstream sync

- **Nightly:** `upstream-main` fast-forwards to `vllm-project/vllm-neuron` `main`, and a PR `upstream-main → main` is opened or updated.
- **Merging the sync PR:**
  - A clean merge goes in after a smoke test.
  - On a conflict, the author of the conflicting COVEX change resolves it.
- **Keep our diff small and contained:**
  - prefer new files and flags over edits to upstream files
  - put model-specific code in its model folder

## 7. Upstreaming

- **The goal is to send every generally useful win upstream.** Examples: scheduler, bucketing, attention and kernel fixes that help any model.
- Tag those PRs `upstream-candidate`. After merge, open the same commit against `vllm-project/vllm-neuron` from a branch cut from `upstream-main`.
- Set the COVEX_CHANGES row to **Sent**, and to **Merged upstream** when it lands. The next sync then drops our copy.
- COVEX exists so customers get the wins before the next upstream release, not to diverge.
