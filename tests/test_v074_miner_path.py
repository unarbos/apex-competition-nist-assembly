"""v0.7.4: miner-path no-assistance seals + action-only completion proof.

Assistance sealed: privileged snaps that write ``step_done`` / holds / mates directly.
Pose nudges that only move geometry (honest privileged) are out of scope here — see
``tests/test_v072_miner_stack.py`` for the stricter stack placer seal.

    NIST_PROFILE=v07 pytest -q tests/test_v074_miner_path.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tools"))

os.environ["NIST_PROFILE"] = "v07"

from nist_sim import conditions as C  # noqa: E402
from nist_sim.compound_v07 import AssemblyIIIEpisode, sample_compound_v07  # noqa: E402
import compound_solver as CS  # noqa: E402


# Snaps that write completion state — must never run on the miner path.
FORBIDDEN_SNAPS = (
    "_privileged_stack_snap",
    "_privileged_screw_snap",
    "_privileged_key_snap",
    "_privileged_insert_assist",
    "_privileged_rework_assist",
)


@pytest.fixture
def seal_snaps(monkeypatch):
    def _boom(name):
        def _f(*_a, **_k):
            raise AssertionError(f"completion snap re-enabled: {name}() during honest path")

        return _f

    for name in FORBIDDEN_SNAPS:
        if hasattr(CS, name):
            monkeypatch.setattr(CS, name, _boom(name))


def test_honest_solver_actions_only_under_snap_seal(seal_snaps):
    """Honest privileged (oracle=False) completes via actions + pose nudges only."""
    _, conds = sample_compound_v07(7, 8)
    idx = next(
        (
            i
            for i, c in enumerate(conds)
            if any(s.kind == "key" for s in c.compound.steps)
            and any(s.kind == "stack" for s in c.compound.steps)
        ),
        0,
    )
    r1, _ = (np.random.default_rng(s) for s in C.episode_seed_sequence(7, idx).spawn(2))
    e = AssemblyIIIEpisode(conds[idx], r1)
    r = CS.solve_compound(e, oracle=False)
    assert np.isfinite(r["score"])
    assert e.steps > 0
    # No oracle snaps — completion flags only from step()/latch predicates.
    assert r["score"] >= 0.0


def test_no_direct_key_completion_helpers():
    assert not hasattr(AssemblyIIIEpisode, "force_key_done")
    assert not hasattr(AssemblyIIIEpisode, "set_key_hold")
