"""The success predicate: a seated part must be upright, not merely low and centred.

    pytest -q tests/test_success.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
import mujoco  # noqa: E402
from nist_sim.conditions import sample_episodes  # noqa: E402
from nist_sim.env import BLOCK_H, PEG_LEN, SEAT_TOL_Z, Episode  # noqa: E402

OPEN_AND_HOLD = np.array([0, 0, 0, 0, 1.0])


def _one_per_part():
    _, conds = sample_episodes(7, 1)
    seen = {}
    for c in conds:
        seen.setdefault((c.shape, c.diameter), c)
    return list(seen.values())


def _place(e: Episode, xyz, quat):
    q = e.part_qadr
    e.data.qpos[q : q + 3] = xyz
    e.data.qpos[q + 3 : q + 7] = quat
    e.data.qvel[e.part_vadr : e.part_vadr + 6] = 0
    hx, hy = e.c.hole_xy
    e.data.qpos[0:3] = [hx, hy, 0.12]  # hand well above the part
    e.target[0:3] = [hx, hy, 0.12]
    mujoco.mj_forward(e.model, e.data)


def _hold(e: Episode, steps: int = 15) -> None:
    for _ in range(steps):
        e.step(OPEN_AND_HOLD)
        if e.done:
            break


def _lying_across(c) -> Episode:
    e = Episode(c, np.random.default_rng(0))
    hx, hy = c.hole_xy
    # On its side, centred over the hole, resting on the block's top face.
    _place(e, [hx, hy, BLOCK_H + c.diameter / 2 + 2e-4], [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0])
    return e


# Lying across the mouth puts the part's centre at BLOCK_H + r, i.e. its "bottom" (centre - PEG_LEN/2)
# at r - 5 mm: within the 1 mm seat tolerance for every part up to 12 mm. Those are the cases the
# position-only predicate accepted.
LYING_ACCEPTED_BY_POSITION = [c for c in _one_per_part() if c.diameter / 2 - (PEG_LEN / 2 - BLOCK_H) <= SEAT_TOL_Z]


@pytest.mark.parametrize("c", LYING_ACCEPTED_BY_POSITION, ids=lambda c: f"{c.shape}_{round(c.diameter * 1000)}mm")
def test_part_lying_across_hole_is_not_assembled(c):
    e = _lying_across(c)
    _hold(e)
    assert not e.success, e.terminal_reason


# Within those, the pose physically settles and scores for every part except the 8 mm round peg,
# which keeps moving on the mouth of its hole. The control below runs on the cases where the
# position-only predicate really did pay out.
LYING_EXPLOITABLE = [c for c in LYING_ACCEPTED_BY_POSITION if not (c.shape == "round" and round(c.diameter * 1000) == 8)]


@pytest.mark.parametrize("c", LYING_EXPLOITABLE, ids=lambda c: f"{c.shape}_{round(c.diameter * 1000)}mm")
def test_only_the_upright_check_rejects_lying_across(c, monkeypatch):
    # With the upright check disabled the same pose must count as assembled; otherwise the test
    # above would pass for an unrelated reason (the part rolling off, an unstable step, ...).
    monkeypatch.setattr(Episode, "_upright", lambda self: True)
    e = _lying_across(c)
    _hold(e)
    assert e.success and e.terminal_reason == "assembled", e.terminal_reason


def test_lying_cases_cover_three_sizes():
    assert {round(c.diameter * 1000) for c in LYING_ACCEPTED_BY_POSITION} == {4, 8, 12}


@pytest.mark.parametrize("c", _one_per_part(), ids=lambda c: f"{c.shape}_{round(c.diameter * 1000)}mm")
@pytest.mark.parametrize("flipped", [False, True])
def test_upright_seated_part_is_assembled(c, flipped):
    e = Episode(c, np.random.default_rng(0))
    hx, hy = c.hole_xy
    quat = [0, 1, 0, 0] if flipped else [1, 0, 0, 0]  # the peg is end-symmetric
    _place(e, [hx, hy, PEG_LEN / 2 + 2e-4], quat)
    _hold(e)
    assert e.success and e.terminal_reason == "assembled", e.terminal_reason
