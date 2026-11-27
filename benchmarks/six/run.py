#!/usr/bin/env python3
"""Benchmark six pinned-upstream/fork pairs on a provisioned trn2.48xlarge.

python benchmarks/six/run.py --plan
python benchmarks/six/run.py --out /path/to/new/attempt

Set UPSTREAM_DIR, UPSTREAM_PY and FORK_PY (absolute paths to separate installed
environments). MODELS_DIR defaults to ~/models, SIX_DATA to ~/data/sixdata,
CLIENT_DIR to ~/qwen3-asr-inference-cuda-trainium (prepared ASR client and data).
The runner starts only local services. It does not provision instances.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH / "five"))
from gate import asr_level_ok, asr_quality_ok, bench_ok, count_ok, same_work

UPSTREAM_COMMIT = "e16d424681c7ac6afa2d834b2255cbb12309ecaa"
CLIENT = BENCH / "five/clients.py"


@dataclass(frozen=True)
class Model:
    folder: str
    stock: str
    fork: str
    kind: str
    tp: int
    loads: tuple
    unit: str
    replicas: int = 1     # >1: that many TP-sized replicas share the arm's chip pair behind asr/rr_proxy.py


# ASR arm -> asr/arm_config.sh label. DL (encoder LN-in-MLP kernel) lost 10% at c8, so the fork arms use the
# scheduling changes instead: DMB = 2 prefills/step + linear 128-token buckets; R* = one TP4 replica each.
ASR_LABELS = {"asr-U": "S2", "asr-DMB": "DMB", "asr-U2": "R4", "asr-R": "RDMB"}


WAVES = (
    (Model("Qwen3-32B", "q32-U", "q32-F", "gen", 8, (1, 8), "output tok/s"),
     Model("Qwen3-Embedding-8B", "emb-U", "emb-P8", "emb", 8, (32,), "sequences/s"),
     Model("Qwen3-ASR-1.7B", "asr-U", "asr-DMB", "asr", 8, (1, 8), "audio seconds/s"),
     # same two chips: upstream 2 x TP4 vs fork 2 x TP4 + scheduling (equal layout); asr-U vs asr-U2 is the
     # layout-only gain upstream can also take, reported in the same table
     Model("Qwen3-ASR-1.7B", "asr-U2", "asr-R", "asr", 4, (1, 8), "audio seconds/s", replicas=2)),
    (Model("Llama-3.1-8B-Instruct", "llama-U", "llama-F", "gen", 8, (1, 8), "output tok/s"),
     Model("Qwen3-VL-32B-Instruct", "vl-U", "vl-C", "vl", 4, (4,), "output tok/s"),
     Model("gpt-oss-20b", "oss-U", "oss-P", "gen", 8, (1, 4), "output tok/s")),
)

GPT_PADDED = Model("gpt-oss-20b", "oss-U8k", "oss-P8k", "gen", 8, (1, 4), "output tok/s")


def write_json(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def command(args, cwd, env, log, timeout):
    with log.open("a") as stream:
        subprocess.run(args, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                       check=True, timeout=timeout)


def git_head(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


@contextmanager
def service(argv, cwd, env, stream):
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        yield process
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        except ProcessLookupError:
            pass
        finally:
            process.wait()


def wait_ready(process, port, kind, extra, cwd, env, timeout):
    until = time.monotonic() + timeout
    while process.poll() is None and time.monotonic() < until:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    break
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    if process.poll() is not None or time.monotonic() >= until:
        raise RuntimeError("server exited or readiness timed out")
    name = "qwen3-asr-1.7b" if kind == "asr" else "m"
    command([sys.executable, str(CLIENT), "smoke", kind, str(port), name, str(extra)],
            cwd, env, cwd / "smoke.log", max(1, until - time.monotonic()))


def measure(model, arm, port, cwd, env, model_path, data, windows, cpus, limit):
    def call(args):
        command(["taskset", "-c", cpus, *map(str, args)], cwd, env, cwd / "measure.log", limit)

    if model.kind == "gen":
        for c in (1, 3):
            call([sys.executable, CLIENT, "cachecheck", port, f"cache-c{c}.json", c, 32])
    if model.kind == "vl":   # serial stage split (diagnostic, before the scored windows): how much is the image
        call([sys.executable, CLIENT, "vlstage", port, data / "vl", 10, "vl-stage.json"])
    for c in model.loads:
        for w in ("warm", *range(1, windows + 1)):
            dest = f"c{c}-w{w}.json"
            if model.kind == "gen":
                call([env["FORK_PY"], "-m", "vllm.entrypoints.cli.main", "bench", "serve",
                      "--backend", "vllm", "--base-url", f"http://127.0.0.1:{port}", "--model", "m",
                      "--tokenizer", model_path, "--dataset-name", "random", "--random-input-len", 1024,
                      "--random-output-len", 256, "--ignore-eos", "--max-concurrency", c, "--num-prompts",
                      8 if c == 1 else 4*c, "--seed", 1, "--temperature", 0, "--save-result", "--save-detailed", "--result-dir", cwd,
                      "--result-filename", dest])
            elif model.kind == "emb":
                call([sys.executable, CLIENT, "emb", port, data / "embed/t64.jsonl", 256, c, dest])
            elif model.kind == "vl":
                # The image client preserves answers for its first 20 images.
                call([sys.executable, CLIENT, "vl", port, data / "vl", 20, c, dest])
            else:
                label = f"{arm}-c{c}-w{w}"
                call(["bash", BENCH / "asr/bench.sh", label, port, 100, c])
                (cwd / "results" / f"{label}.json").rename(cwd / dest)


def read_object(path):
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return obj


def embedding_match(u, f, count):
    from clients import vector_match
    for obs in (u, f):
        ids, vectors = obs.get("input_ids"), obs.get("vectors")
        if (not isinstance(ids, list) or len(ids) != count or not all(isinstance(i, str) and i for i in ids)
                or not isinstance(vectors, list) or len(vectors) != count):
            return False
    try:
        return vector_match(u, f, cos_min=0.9999)["pass"] is True
    except (TypeError, ValueError, OverflowError):
        return False


def greedy_probe_problems(model, places):
    from clients import CACHE_LENGTHS, cache_prompts
    digest = hashlib.sha256(json.dumps(cache_prompts()).encode()).hexdigest()
    problems = []
    for conc in (1, 3):
        outputs = []
        for arm in (model.stock, model.fork):
            try:
                probe = read_object(places[arm] / f"cache-c{conc}.json")
                if (probe.get("lengths") != CACHE_LENGTHS or probe.get("gen_tokens") != 32 or
                        probe.get("concurrency") != conc or probe.get("prompt_sha256") != digest):
                    raise ValueError("wrong or missing prompt identity, length, or concurrency")
                ids = probe.get("token_ids")
                if (not isinstance(ids, list) or len(ids) != len(CACHE_LENGTHS) or
                        any(not isinstance(row, list) or len(row) != 32 or
                            any(type(i) is not int or i < 0 for i in row) for row in ids)):
                    raise ValueError("incomplete or invalid generated token IDs")
                outputs.append(ids)
            except (OSError, ValueError) as exc:
                problems.append(f"{arm} greedy probe c{conc}: {exc}")
        if len(outputs) == 2 and outputs[0] != outputs[1]:
            i, j = next((i, j) for i, (u, f) in enumerate(zip(*outputs))
                        for j, (a, b) in enumerate(zip(u, f)) if a != b)
            problems.append(f"greedy token check c{conc} differs at prompt {i} (length {CACHE_LENGTHS[i]}), token {j}")
    return problems


def audio_inputs(obj, concurrency):
    clips = {}
    for r in obj.get("raw_results", []):
        if r["phase"] != "transcribe" or r["concurrency"] != concurrency or r["is_warmup"]:
            continue
        clip, duration = r["sample_id"], r["audio_duration_s"]
        if (not isinstance(clip, str) or not clip or clip in clips or r["success"] is not True
                or r["round_idx"] != 0 or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration <= 0):
            raise ValueError("invalid or duplicate ASR clip record")
        clips[clip] = duration
    if len(clips) != 100:
        raise ValueError(f"expected 100 measured ASR clips, found {len(clips)}")
    return clips


def record_rows(model, places, errors, windows):
    common = list(errors.values())
    if model.kind == "gen" and not common:
        common.extend(greedy_probe_problems(model, places))
    for c in model.loads:
        rates = {}
        problems = list(common)
        observations = {}
        for arm in (model.stock, model.fork):
            sample = []
            observations[arm] = []
            for w in range(1, windows + 1):
                try:
                    obj = read_object(places[arm] / f"c{c}-w{w}.json")
                    if model.kind == "gen":
                        ok, why = bench_ok(obj, 8 if c == 1 else 4*c, 256, 1024)
                        if obj.get("max_concurrency") != c or obj.get("failed") != 0:
                            raise ValueError("missing or incorrect benchmark concurrency/failure count")
                        metric = obj.get("output_throughput")
                    elif model.kind == "asr":
                        ok, why = asr_level_ok(obj, c, 100)
                        metric = next((r["audio_seconds_per_s"] for r in obj["phases"]["transcribe"]
                                       if r["concurrency"] == c), None)
                    else:
                        key, n = ("seq_per_s", 256) if model.kind == "emb" else ("out_tok_per_s", 20)
                        ok, why = count_ok(obj, n, metric=key)
                        if obj.get("concurrency") != c:
                            raise ValueError("missing or incorrect benchmark concurrency")
                        metric = obj.get(key)
                    if not ok:
                        raise ValueError(why)
                    observations[arm].append(obj)
                    sample.append(metric)
                except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
                    problems.append(f"{arm} window {w}: {exc}")
                    break
            rates[arm] = statistics.median(sample) if len(sample) == windows else None
        if not problems:
            for u, f in zip(observations[model.stock], observations[model.fork]):
                if model.kind == "emb":
                    if not embedding_match(u, f, 256):
                        problems.append("embedding vectors or prompt identities differ or are incomplete")
                elif model.kind == "asr":
                    try:
                        if audio_inputs(u, c) != audio_inputs(f, c):
                            raise ValueError("ASR clip IDs or durations differ")
                        ok, why = asr_quality_ok(u, f, c)
                        if not ok:
                            problems.append(why)
                    except (KeyError, TypeError, ValueError) as exc:
                        problems.append(f"ASR correctness: {exc}")
                elif model.kind == "vl":
                    answers = u.get("answers")
                    tokens = u.get("completion_tokens")
                    if (not isinstance(answers, list) or len(answers) != 20 or
                            not all(isinstance(a, str) and a.strip() for a in answers) or answers != f.get("answers") or
                            not isinstance(tokens, list) or len(tokens) != 20 or any(type(t) is not int or t <= 0 for t in tokens)
                            or tokens != f.get("completion_tokens")):
                        problems.append("image answers or per-image output token counts differ or are incomplete")
                else:
                    ok, why = same_work(u, f)
                    if not ok:
                        problems.append(why)
                    n = 8 if c == 1 else 4*c
                    outputs = u.get("generated_texts")
                    if (any(o.get("input_lens") != [1024]*n or o.get("output_lens") != [256]*n for o in (u, f))
                            or not isinstance(outputs, list) or len(outputs) != n
                            or not all(isinstance(text, str) and text for text in outputs)
                            or outputs != f.get("generated_texts")):
                        problems.append("per-request lengths or measured greedy outputs differ or are missing")
        yield dict(model=model.folder, arm=model.fork, concurrency=c, unit=model.unit, stock=rates[model.stock],
                   covex=rates[model.fork], ratio=None if problems else rates[model.fork] / rates[model.stock],
                   status="; ".join(dict.fromkeys(problems)) or (
                       "OK (diagnostic: forced 8192-token prefill)" if model == GPT_PADDED else "OK"))


def write_summary(out, rows):
    write_json(out / "results.json", rows)
    lines = ["| Model | COVEX arm | Concurrency | Unit | Upstream | COVEX | Ratio | Status |",
             "|---|---|---:|---|---:|---:|---:|---|"]
    for row in rows:
        nums = ["—" if row[k] is None else f"{row[k]:.2f}" for k in ("stock", "covex", "ratio")]
        status = row["status"].replace("|", "/").replace("\n", " ")
        lines.append(f"| {row['model']} | {row['arm']} | {row['concurrency']} | {row['unit']} | " + " | ".join(nums) + f" | {status} |")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")


def check_wave_fits(wave):
    """2 arms per model, one chip pair (8 logical cores) and 24 vCPUs per arm: at most 8 arms per trn2.48xlarge."""
    if 2 * len(wave) > 8:
        raise ValueError(f"wave of {len(wave)} models needs {2 * len(wave)} chip pairs; the host has 8")
    for m in wave:
        if m.tp * m.replicas > 8:
            raise ValueError(f"{m.folder}: {m.replicas} x TP{m.tp} does not fit one chip pair")


for _w in WAVES:
    check_wave_fits(_w)


def run_wave(wave, out, args, base, roots, prefixes):
    models_dir = Path(base.get("MODELS_DIR", Path.home() / "models")).resolve()
    data = Path(base.get("SIX_DATA", Path.home() / "data/sixdata")).resolve()
    asr = Path(base["CLIENT_DIR"]) / "data"
    jobs, errors, places = {}, {}, {}
    with ThreadPoolExecutor(max_workers=6) as pool, ExitStack() as stack:
        ready = {}
        for index, model in enumerate(wave):
            for side, arm in enumerate((model.stock, model.fork)):
                slot = 2 * index + side
                port = 8600 + slot
                # 8 arm slots per wave (16 chips): 24 host vCPUs each fill the 192-vCPU host exactly
                cpus = f"{24*slot}-{24*slot+23}"
                cwd = out / arm
                cwd.mkdir()
                places[arm] = cwd
                prov = cwd / "provenance"
                prov.mkdir()
                cache = cwd / "cache"
                interpreter = base["UPSTREAM_PY" if side == 0 else "FORK_PY"]
                env = dict(base, SUITE_PYTHON=interpreter, SUITE_CPUS=cpus,
                           SUITE_CORES=f"{8*slot}-{8*slot+model.tp*model.replicas-1}",
                           MODELS_DIR=str(models_dir), MODEL_DIR=str(models_dir / model.folder),
                           COVEX_ARM=arm, COVEX_PROVENANCE_DIR=str(prov), ARM_CACHE_BASE=str(cache),
                           PYTHONPATH=str(BENCH / "provenance"), SOLO="1", ROUNDS="1", EMB_ALL_VECTORS="1",
                           NEURON_LIBTORCH_CACHE_ROOT=str(cache / arm), VLLM_CACHE_ROOT=str(cache / arm / "vllm"),
                           TORCH_NEURONX_NEFF_CACHE_DIR=str(cache / arm / "neff"))
                model_path = models_dir / model.folder
                try:
                    cfg = model_path / "config.json"
                    if not cfg.is_file():
                        raise ValueError(f"checkpoint missing: {cfg}")
                    meta = model_path / ".cache/huggingface/download/config.json.metadata"
                    rev = meta.read_text().splitlines()[0] if meta.is_file() else None
                    write_json(cwd / "settings.json", dict(model=str(model_path), revision=rev,
                               config_sha256=hashlib.sha256(cfg.read_bytes()).hexdigest(), python=interpreter,
                               cores=env["SUITE_CORES"], cpus=cpus, upstream=side == 0))
                    if model.kind == "asr":
                        if not (Path(base["CLIENT_DIR"]) / "benchmark/benchmark.py").is_file() or not (asr / "manifest.json").is_file():
                            raise ValueError("ASR client and audio dataset must be staged on the host")
                        client_dir = Path(base["CLIENT_DIR"])
                        if (client_dir / ".git").exists() and not git_head(client_dir).startswith("4ce1ddb"):
                            raise ValueError("prepare ASR benchmark client at 4ce1ddb before starting the suite")
                        label = ASR_LABELS[arm]
                        script = "asr/serve_replicas.sh" if model.replicas > 1 else "asr/serve.sh"
                        argv = [BENCH / script, label, str(slot), str(port)]
                    else:
                        label = arm
                        argv = [BENCH / "five/serve.sh", arm, str(port)]
                    stream = stack.enter_context((cwd / "launcher.log").open("w"))
                    proc = stack.enter_context(service(["taskset", "-c", cpus, "bash", *argv], cwd, env, stream))
                    extra = asr if model.kind == "asr" else data / "vl"
                    ready[arm] = pool.submit(wait_ready, proc, port, model.kind, extra, cwd, env, args.ready_seconds)
                    jobs[arm] = (model, port, cwd, env, model_path, cpus, label, side)
                except (OSError, ValueError) as exc:
                    errors[arm] = str(exc)
        for arm, future in ready.items():
            try:
                future.result()
                model, _, cwd, _, _, _, label, side = jobs[arm]
                logs = ",".join(f"{cwd}/logs/{label}.r{r}.server.log" for r in range(model.replicas)) \
                    if model.replicas > 1 else f"{cwd}/logs/{label}.server.log"
                command([sys.executable, str(BENCH / "provenance/verify.py"), str(cwd / "provenance"),
                         str((model.tp + 2) * model.replicas), "--tp", str(model.tp), "--log", f"{arm}={logs}",
                         f"{arm}={roots[side]}@{prefixes[side]}"], cwd, base, cwd / "provenance.log", 30)
            except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
                errors[arm] = f"{arm}: {exc}"
        measured = {}
        for arm, (model, port, cwd, env, path, cpus, _, _) in jobs.items():
            if arm not in errors:
                measured[arm] = pool.submit(measure, model, arm, port, cwd, env, path, data,
                                            args.windows, cpus, args.client_seconds)
        for arm, future in measured.items():
            try:
                future.result()
            except (OSError, subprocess.SubprocessError) as exc:
                errors[arm] = f"{arm}: {exc}"
    rows = []
    for model in wave:
        arm_errors = {a: errors[a] for a in (model.stock, model.fork) if a in errors}
        try:
            rows.extend(list(record_rows(model, places, arm_errors, args.windows)))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            for c in model.loads:
                rows.append(dict(model=model.folder, arm=model.fork, concurrency=c, unit=model.unit, stock=None,
                                 covex=None, ratio=None, status=f"invalid artifacts: {exc}"))
    return rows


def import_embedding(path, windows, packed):
    """Reuse the completed, independently timed embedding experiment as its own row."""
    m = WAVES[0][1]
    row = dict(model=m.folder, arm=packed, concurrency=32, unit=m.unit, stock=None, covex=None, ratio=None)
    try:
        manifest = read_object(path / "manifest.json")
        pins = {"upstream_vllm_neuron": UPSTREAM_COMMIT, "model": "Qwen/Qwen3-Embedding-8B",
                "model_revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
                "windows": windows,
                "concurrency": 32, "requests_per_window": 256, "chips_per_arm": 2, "lens": ["t64"]}
        # Both revisions have identical production code, model configuration and workloads.
        reviewed = ("032aadebc0d109c2e4d54d2e8b11b4249e910c39", "63b0d89fdecb9a8df314c927a3ac9d99a319b85e")
        if manifest.get("covex_neuron") not in reviewed or any(manifest.get(k) != v for k, v in pins.items()):
            raise ValueError("saved workload or commit differs")
        folder = path / "results"
        status = [s.split("\t") for s in (folder / "STATUS.tsv").read_text().splitlines()]
        saved, medians = {}, {}
        for arm in (m.stock, "emb-T", packed):
            steps = [r[1:] for r in status if r and r[0] == arm]
            if (["ready", "OK"] not in steps or not any(k == "provenance" and v.split()[0] == "PASS" for k, v in steps)
                    or any(v.startswith("FAIL") for _, v in steps)):
                raise ValueError(f"unverified saved arm {arm}")
            rates = []
            saved[arm] = []
            for w in range(1, windows + 1):
                obs = read_object(folder / f"{arm}-t64-w{w}.json")
                ok, why = count_ok(obs, 256, metric="seq_per_s")
                if not ok or obs.get("concurrency") != 32:
                    raise ValueError(f"{arm} window {w}: {why or 'wrong concurrency'}")
                rates.append(obs["seq_per_s"])
                saved[arm].append(obs)
            medians[arm] = statistics.median(rates)
        row.update(stock=medians[m.stock], covex=medians[packed])
        for arm in (m.stock, "emb-T"):
            for w, (ref, candidate) in enumerate(zip(saved[arm], saved[packed]), 1):
                # This archived client saved all 256 in w1 and the first 20 thereafter.
                count = len(ref.get("vectors", []))
                if ((count != 256 and (w == 1 or count != 20)) or not embedding_match(ref, candidate, count)):
                    raise ValueError(f"packed vectors differ from {arm} or comparison incomplete in window {w}")
        row.update(ratio=row["covex"] / row["stock"], status=f"OK (saved attempt {manifest['run_prefix']})")
    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        row["status"] = f"saved embedding attempt: {exc}"
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--out", type=Path)
    parser.add_argument("--embedding-run", type=Path, help="local copy of a reviewed embedding attempt; reuse P4 and P8 vs upstream")
    parser.add_argument("--gpt-padded-diagnostic-only", action="store_true",
                        help="run only the GPT-OSS pair with forced 8192-token prefills; excluded from the six-model result")
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--ready-seconds", type=int, default=1800)
    parser.add_argument("--client-seconds", type=int, default=900)
    args = parser.parse_args()
    if min(args.windows, args.ready_seconds, args.client_seconds) < 1:
        parser.error("windows and timeouts must be positive")
    if args.gpt_padded_diagnostic_only and args.embedding_run:
        parser.error("the isolated GPT diagnostic cannot import an embedding run")
    waves = ((GPT_PADDED,),) if args.gpt_padded_diagnostic_only else WAVES
    if args.plan:
        if args.gpt_padded_diagnostic_only:
            print("GPT-OSS diagnostic only (forced 8192-token prefill)")
        for i, wave in enumerate(waves, 1):
            for m in wave:
                if m.kind == "emb" and args.embedding_run:
                    print(f"reuse {args.embedding_run}: {m.folder} stock vs pack-4 and pack-8")
                    continue
                print(f"wave {i}  {m.folder}: {m.stock} vs {m.fork}, TP{m.tp}, c{m.loads}, {m.unit}")
        return 0
    needed = ("UPSTREAM_DIR", "UPSTREAM_PY", "FORK_PY")
    if any(not os.path.isabs(os.environ.get(k, "")) for k in needed):
        parser.error("set absolute UPSTREAM_DIR, UPSTREAM_PY and FORK_PY")
    roots = (Path(os.environ["UPSTREAM_DIR"]).resolve(), BENCH.parent)
    if git_head(roots[0]) != UPSTREAM_COMMIT:
        parser.error(f"UPSTREAM_DIR must be at {UPSTREAM_COMMIT}")
    if subprocess.check_output(["git", "-C", str(roots[0]), "status", "--porcelain", "--", "vllm_neuron"]).strip():
        parser.error("upstream engine source has local changes")
    prefixes = [subprocess.check_output([os.environ[k], "-c", "import sys;print(sys.prefix)"], text=True).strip()
                for k in ("UPSTREAM_PY", "FORK_PY")]
    if prefixes[0] == prefixes[1]:
        parser.error("use distinct upstream and fork environments")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    base = {k: v for k, v in os.environ.items() if not k.startswith("COVEX_") and k not in
            ("NEURON_LIBTORCH_REMOTE_CACHE", "VLLM_NEURON_CPU_MODE", "VLLM_NEURON_CPU_COMPILE", "PROFILE",
             "LOG_TAG", "REHEARSAL_EXTRA", "DRYRUN", "PROBE_N", "PROBE_TOKENS", "EMB_COS_MIN",
             "PYTHONPATH", "VLLM_NEURON_ASR_AUDIO_MLP_NKI")}
    base["CLIENT_DIR"] = str(Path(base.get("CLIENT_DIR", Path.home() / "qwen3-asr-inference-cuda-trainium")).resolve())
    base["PATH"] = str(Path(base["FORK_PY"]).parent) + os.pathsep + base["PATH"]
    write_json(out / "manifest.json", dict(upstream=UPSTREAM_COMMIT, fork=git_head(roots[1]),
               windows=args.windows, python_prefixes=prefixes, embedding_run=str(args.embedding_run) if args.embedding_run else None,
               gpt_padded_diagnostic_only=args.gpt_padded_diagnostic_only,
               waves=[[vars(m) for m in wave] for wave in waves]))
    rows = []
    for wave in waves:
        if args.embedding_run and WAVES[0][1] in wave:
            rows.extend(import_embedding(args.embedding_run, args.windows, arm) for arm in ("emb-P4", "emb-P8"))
            wave = tuple(m for m in wave if m.kind != "emb")
        rows.extend(run_wave(wave, out, args, base, roots, prefixes))
        write_summary(out, rows)
    print((out / "RESULTS.md").read_text())
    return int(any(row["ratio"] is None for row in rows))


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    sys.exit(main())
