"""Stage pinned campaign checkpoints and persist outcomes before any service starts."""
import argparse
import errno
import json
import multiprocessing
from pathlib import Path
import socket
import sys

DRAFTER = "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3"


def verify_checkpoint(folder, revision):
    meta = folder / ".cache/huggingface/download/config.json.metadata"
    if not (folder / "config.json").is_file() or meta.read_text().splitlines()[:1] != [revision]:
        raise ValueError(f"missing config or wrong revision in {folder}")
    for filename in ("model.safetensors", "pytorch_model.bin"):
        index = folder / (filename + ".index.json")
        if index.is_file():
            contents = json.loads(index.read_text())
            weights = contents.get("weight_map") if isinstance(contents, dict) else None
            if not isinstance(weights, dict) or not weights:
                raise ValueError(f"empty checkpoint index in {folder}")
            if any(not isinstance(name, str) or not name or Path(name).name != name for name in weights.values()):
                raise ValueError(f"invalid checkpoint shard name in {index}")
            shards = set(weights.values())
        elif (folder / filename).is_file():
            shards = {filename}
        else:
            continue
        for name in shards:
            if not (folder / name).is_file() or (folder / name).stat().st_size == 0:
                raise ValueError(f"missing or empty weight shard {name} in {folder}")
        return
    raise ValueError(f"no model weights in {folder}")


def describe_failure(exc):
    import httpx
    from urllib.error import URLError

    seen = set()
    network = False
    # Hub/Xet can wrap transport and OS errors. Keep their scope and most
    # specific cause without saving a signed download URL in the report.
    detail = type(exc).__name__
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        message = str(exc).splitlines()[0] if str(exc) else ""
        if "://" not in message:
            detail = f"{type(exc).__name__}: {message[:250]}"
        if (isinstance(exc, ImportError)
                or isinstance(exc, OSError) and exc.errno in {
                    errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.EROFS, errno.ENOMEM,
                    errno.ENETDOWN, errno.ENETUNREACH, errno.EMFILE, errno.ENFILE,
                } or "no space left on device" in message.lower()):
            return "shared", detail
        status = getattr(getattr(exc, "response", None), "status_code", None)
        network |= (isinstance(exc, (httpx.TransportError, ConnectionError, TimeoutError, socket.gaierror, URLError))
                    or status in (429, 500, 502, 503, 504)
                    or any(issubclass(t, OSError) and t.__module__.startswith("requests")
                           and t.__name__ in ("ConnectionError", "Timeout") for t in type(exc).__mro__))
        exc = exc.__cause__ or exc.__context__
    return ("network" if network else "model"), detail


def stage_one(job):
    repo, revision, folder = job
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, revision=revision, local_dir=folder, ignore_patterns=["original/*", "*.gguf"])
        verify_checkpoint(folder, revision)
    except Exception as exc:
        scope, detail = describe_failure(exc)
        return repo, scope, "FAILED " + detail
    return repo, "ok", "ok"


def stage(pins, model_dir, report):
    results = {repo: "PENDING" for repo in pins}
    def save():
        temp = report.with_suffix(".tmp")
        temp.write_text(json.dumps(results, indent=2) + "\n")
        temp.replace(report)

    save()
    jobs = [(repo, rev, model_dir / repo.split("/")[1]) for repo, rev in pins.items()]
    network_failures = []
    # A fatal result terminates the remaining downloads, including blocked HTTP workers.
    with multiprocessing.get_context("spawn").Pool(min(len(jobs), 3)) as pool:
        for repo, scope, outcome in pool.imap_unordered(stage_one, jobs):
            results[repo] = outcome
            if scope == "network":
                network_failures.append(repo)
            if scope == "shared" or len(network_failures) >= 2:
                detail = f"{repo}: {outcome}" if scope == "shared" else "network failures for " + ", ".join(network_failures)
                results["_shared_failure"] = detail
                save()
                print(detail, file=sys.stderr, flush=True)
                return 1
            save()
    if not any(results[repo] == "ok" for repo in pins if repo != DRAFTER):
        results["_shared_failure"] = "no target checkpoint downloaded successfully"
        save()
        return 1
    return 0


def load_outcomes(path, pins):
    report = json.loads(path.read_text())
    if not isinstance(report, dict):
        raise ValueError("invalid checkpoint download report")
    if "_shared_failure" in report:
        raise ValueError(f"shared checkpoint setup failure: {report['_shared_failure']}")
    if not any(report.get(repo) == "ok" for repo in pins if repo != DRAFTER):
        raise ValueError("download report has no successful target checkpoint")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pins", "model-dir", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.exit(stage(json.loads(args.pins.read_text()), args.model_dir, args.report))
