import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SIX = ROOT / "benchmarks/six"


def module():
    spec = importlib.util.spec_from_file_location("campaign", SIX / "run.py")
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    return suite


def checkpoint(folder, pin):
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    metadata = folder / ".cache/huggingface/download/config.json.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(pin + "\n")


@pytest.mark.parametrize("gpt_status", ["ok", "FAILED network interrupted"])
def test_partial_and_failed_downloads_never_spawn_an_arm(tmp_path, monkeypatch, gpt_status):
    suite = module()
    llama = next(m for wave in suite.WAVES for m in wave if m.fork == "llama-K2")
    gpt = next(m for wave in suite.WAVES for m in wave if m.fork == "oss-P")
    folders = tmp_path / "models"
    pins = suite.MODEL_PINS
    for repo in (suite.REPOS[llama.folder], suite.DRAFTER, suite.REPOS[gpt.folder]):
        folder = folders / repo.split("/")[1]
        checkpoint(folder, pins[repo])
        (folder / "model.safetensors").write_bytes(b"existing weights")
    # Even a locally complete-looking snapshot cannot override this attempt's FAILED verdict.
    failed = {**dict.fromkeys(pins, "ok"), suite.DRAFTER: "FAILED interrupted transfer",
              suite.REPOS[gpt.folder]: gpt_status}
    # The GPT snapshot has a pinned config, one old weight file, and a missing indexed shard.
    folder = folders / gpt.folder
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "shard2.safetensors"}}))
    path = tmp_path / "downloads.json"
    path.write_text(json.dumps(failed))
    attempted = []
    def service(argv, cwd, env, stream):
        attempted.append(env["COVEX_ARM"])
        raise OSError("test recorded launch")
    monkeypatch.setattr(suite, "service", service)
    out = tmp_path / "out"
    out.mkdir()
    rows = suite.run_wave((llama, gpt), out, SimpleNamespace(windows=1),
                          dict(MODELS_DIR=str(folders), CLIENT_DIR="asr", FORK_PY=sys.executable,
                               UPSTREAM_PY=sys.executable), (tmp_path, tmp_path), ("up", "fork"),
                          suite.load_outcomes(path, pins))
    assert attempted == ["llama-U", "llama-K2"]
    assert all(row["ratio"] is None for row in rows)
    assert all("interrupted transfer" in row["status"] for row in rows if row["model"] == llama.folder)
    reason = "shard2" if gpt_status == "ok" else gpt_status
    assert all(reason in row["status"] for row in rows if row["model"] == gpt.folder)


FAKE_HUB = '''
import errno, json, os, socket, time
from pathlib import Path

def snapshot_download(repo, revision, local_dir, ignore_patterns):
    Path(local_dir).mkdir(parents=True)
    m = local_dir / ".cache/huggingface/download/config.json.metadata"
    m.parent.mkdir(parents=True)
    m.write_text(revision + "\\n")
    (local_dir / "config.json").write_text("{}")
    if repo.endswith("/disk"):
        raise RuntimeError("transfer aborted") from OSError(errno.ENOSPC, "disk full")
    if "/dns" in repo:
        raise RuntimeError("hub unavailable") from socket.gaierror(-2, "name resolution failed")
    if repo.endswith("/slow"):
        time.sleep(60)
    if repo.endswith("/partial"):
        return local_dir
    (local_dir / "model.safetensors").write_bytes(b"weights")
    return local_dir
'''


@pytest.mark.parametrize(("names", "exit_code"), [
    (["good", "partial"], 0), (["disk", "slow", "good"], 1), (["dns1", "slow", "dns2"], 1),
])
def test_real_staging_cli_saves_verdicts_and_aborts_shared_failures(tmp_path, names, exit_code):
    (tmp_path / "huggingface_hub.py").write_text(FAKE_HUB)
    pins = {"org/" + n: "1"*40 for n in names}
    path = tmp_path / "pins.json"
    path.write_text(json.dumps(pins))
    report = tmp_path / "downloads.json"
    env = dict(os.environ, PYTHONPATH=str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run([sys.executable, str(SIX / "downloads.py"), "--pins", str(path), "--model-dir",
                           str(tmp_path / "models"), "--report", str(report)], env=env,
                          capture_output=True, text=True, timeout=15)
    assert proc.returncode == exit_code, proc.stderr
    data = json.loads(report.read_text())
    suite = module()
    if exit_code:
        assert "disk full" in data["org/disk"] if "disk" in names else "network failures" in data["_shared_failure"]
        assert data["org/slow"] == "PENDING"
        with pytest.raises(ValueError, match="shared checkpoint setup failure"):
            suite.load_outcomes(report, pins)
        # The normal runner entry must stop before any environment checks or server startup.
        run = subprocess.run([sys.executable, str(SIX / "run.py"), "--out", str(tmp_path / "attempt"),
                              "--downloads", str(report)], capture_output=True, text=True)
        assert run.returncode != 0 and "shared checkpoint setup failure" in run.stderr
        assert not (tmp_path / "attempt").exists()
    else:
        assert data["org/good"] == "ok" and "no model weights" in data["org/partial"]
        assert suite.load_outcomes(report, pins) == data
