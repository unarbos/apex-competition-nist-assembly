"""Unit coverage for the built-image wall harness using a local Docker stub."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _stub(tmp_path, body):
    directory = tmp_path / "bin"
    directory.mkdir()
    docker = directory / "docker"
    docker.write_text("#!/usr/bin/env python3\n" + body)
    docker.chmod(docker.stat().st_mode | stat.S_IXUSR)
    return directory


def _run(tmp_path, stub_dir):
    out = tmp_path / "out"
    out.mkdir()
    out.chmod(0o777)
    env = dict(os.environ)
    env["PATH"] = f"{stub_dir}:/usr/bin:/bin"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "docker_wall_v08.py"),
         "--image", "unit-stub", "--policy", str(ROOT / "baselines" / "baseline.onnx"),
         "--out", str(out), "--timeout", "15"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    return proc, json.loads((out / "wall.json").read_text())


def test_wall_harness_accepts_complete_64_episode_result(tmp_path):
    stub = _stub(tmp_path, """import json, pathlib, sys
out = pathlib.Path(sys.argv[sys.argv.index('-v') + 1].split(':')[0])
(out / 'result.json').write_text(json.dumps({
    'raw_scores': [0.0], 'terminal_reason': 'scored',
    'metadata': {'summary': [{'key': 'episodes', 'value': 64}], 'details': {}},
}))
""")
    proc, result = _run(tmp_path, stub)
    assert proc.returncode == 0, proc.stderr
    assert result["episodes"] == 64
    assert result["under_600s"] is True
    assert result["built_image"] == "unit-stub"
    assert (tmp_path / "out" / "result.json").is_file()


def test_wall_harness_rejects_timeout_and_cleans_player(tmp_path):
    stub = _stub(tmp_path, "import time\ntime.sleep(60)\n")
    proc, result = _run(tmp_path, stub)
    assert proc.returncode != 0
    assert result["under_600s"] is False
    assert "error" in result
    players = subprocess.run(
        ["ps", "-eo", "args="], capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    # No server started by this test may remain after subprocess.TimeoutExpired.
    assert not any(str(ROOT / "player" / "launch.py") in line for line in players)
