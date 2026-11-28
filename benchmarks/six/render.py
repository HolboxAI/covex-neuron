"""Package committed engines and the tested NKI trees locally. Prints the S3 key for later staging."""
import argparse
import gzip
import hashlib
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "provenance"))
from kernel_files import LLAMA_PAIR, QWEN_4K
from run import UPSTREAM_COMMIT


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *map(str, args)], text=True).strip()


def stable_header(info):
    info.uid = info.gid = info.mtime = 0
    info.uname = info.gname = ""
    info.pax_headers = {}
    return info


def add_sources(archive, prefix, src, checks):
    package = src / "nkilib_src/nkilib"
    for name, digest in checks.items():
        path = package / (name.replace(".", "/") + ".py")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"wrong kernel source: {path}")
    for path in sorted(src.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"unexpected symlink in kernel tree: {path}")
        if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts:
            archive.add(path, arcname=f"{prefix}/{path.relative_to(src)}", recursive=False, filter=stable_header)


def render(args):
    if not re.fullmatch(r"s3://[a-z0-9.-]+(?:/[A-Za-z0-9./_-]+)?", args.s3_root):
        raise ValueError("invalid S3 staging prefix")
    if not re.fullmatch(r"[a-z]{2}(-[a-z0-9]+)+-[0-9]+", args.region) or args.minutes < 1:
        raise ValueError("invalid region or time budget")
    repo = HERE.parents[1]
    if git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("commit the engine and benchmark changes before rendering")
    fork = git(repo, "rev-parse", "HEAD")
    args.out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as tmp:
        bare = Path(tmp) / "repo"
        subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True)
        git(bare, "fetch", "-q", "--no-tags", repo, fork + ":refs/heads/campaign", UPSTREAM_COMMIT + ":refs/heads/upstream")
        bundle = Path(tmp) / "engines.bundle"
        git(bare, "bundle", "create", bundle, "--all")
        archive_path = args.out / "sources.tgz"
        with archive_path.open("wb") as output, gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as tar:
                tar.add(bundle, arcname="engines.bundle", filter=stable_header)
                add_sources(tar, "qwen", args.qwen_src, QWEN_4K)
                add_sources(tar, "llama", args.llama_src, LLAMA_PAIR)
                tar.add(args.llama_src.parent / "covex/llama_paged_parity.py", arcname="covex/llama_paged_parity.py", filter=stable_header)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    blob = archive_path.rename(args.out / f"sources-{digest}.tgz")
    root = args.s3_root.rstrip("/")
    values = dict(ROOT=root, REGION=args.region, MINUTES=str(args.minutes), BLOB=blob.name,
                  SHA=digest, FORK=fork, UPSTREAM=UPSTREAM_COMMIT)
    script = (HERE / "payload.sh.in").read_text()
    for key, value in values.items():
        script = script.replace("@@" + key + "@@", shlex.quote(value))
    if "@@" in script:
        raise ValueError("unfilled payload template")
    target = args.out / "payload.sh"
    target.write_text(script)
    subprocess.run(["bash", "-n", str(target)], check=True)
    print(f"{root}/inputs/{blob.name}\npayload SHA256 {hashlib.sha256(target.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("qwen-src", "llama-src", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--s3-root", required=True)
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--minutes", type=int, required=True, help="total minutes from boot")
    render(parser.parse_args())
