"""Slow: a policy memorising one round must not carry over to new rounds.

v0.2.0 note: the overfit builder still targets the multi-peg board; set RUN_SLOW=1.

    RUN_SLOW=1 pytest -q tests/test_rotation.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1")


def score(policy: Path, seed: int, out: Path) -> float:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/local_run.py"),
            "--policy",
            str(policy),
            "--seed",
            str(seed),
            "--out",
            str(out),
        ],
        check=True,
    )
    return json.loads((out / "result.json").read_text())["raw_scores"][0]


def test_memorised_round_does_not_generalise(tmp_path):
    pol = tmp_path / "overfit.onnx"
    subprocess.run(
        [sys.executable, str(ROOT / "tools/make_overfit.py"), "--seed", "11", "--out", str(pol)],
        check=True,
    )
    base = ROOT / "baselines/baseline.onnx"
    own = score(pol, 11, tmp_path / "own")
    # On its own round the table may or may not beat the weak baseline; off-round it must not.
    for s in (12, 13):
        assert score(pol, s, tmp_path / f"o{s}") <= score(base, s, tmp_path / f"b{s}") + 0.05
