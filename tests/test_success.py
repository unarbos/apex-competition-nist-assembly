"""Success / partial-credit predicates for the multi-peg board.

    pytest -q tests/test_success.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
from nist_sim.conditions import EpisodeConditions, PegStart  # noqa: E402
from nist_sim.env import BLOCK_H, PEG_LEN, SEAT_TOL_Z, Episode  # noqa: E402
from nist_sim.scoring import episode_score  # noqa: E402

OPEN_AND_HOLD = np.array([0, 0, 0, 0, 1.0])


def _ep(slots=(0,), clearance=0.0003) -> Episode:
    pegs = [
        PegStart(slot=s, start_type="loose", offset_xy=(0.06, 0.0), height=0.02, yaw_err=0.0)
        for s in slots
    ]
    c = EpisodeConditions(
        n_pegs=len(pegs),
        clutter_bucket="low",
        clearance=clearance,
        friction=1.0,
        peg_mass=0.02,
        gain_scale=1.0,
        obs_noise=0.0,
        obs_noise_rad=0.0,
        dropout_prob=0.0,
        act_delay=0,
        disturbance=0,
        board_xy=(0.0, 0.0),
        board_yaw=0.0,
        pegs=pegs,
        clutter=[],
        max_steps=200 * len(pegs),
    )
    return Episode(c, np.random.default_rng(0))


def _place_peg(e: Episode, i: int, xyz, quat):
    q = e.peg_qadr[i]
    e.data.qpos[q : q + 3] = xyz
    e.data.qpos[q + 3 : q + 7] = quat
    e.data.qvel[e.peg_vadr[i] : e.peg_vadr[i] + 6] = 0
    e.data.qpos[0:3] = [xyz[0], xyz[1], 0.12]
    e.target[0:3] = [xyz[0], xyz[1], 0.12]
    mujoco.mj_forward(e.model, e.data)


def test_lying_across_hole_is_not_seated():
    e = _ep(slots=(0,))
    hx, hy = e.slot_xy[0]
    # On its side over the hole.
    _place_peg(
        e,
        0,
        [hx, hy, BLOCK_H + 0.002 + 0.002],
        [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0],
    )
    for _ in range(20):
        e.step(OPEN_AND_HOLD)
    assert not e.seated_now[0]


def test_wrong_slot_does_not_count():
    """Peg for slot 0 placed into slot 1's hole → not seated for peg 0."""
    e = _ep(slots=(0, 1))
    # Seat peg 0 into peg 1's hole.
    hx1, hy1 = e.slot_xy[1]
    _place_peg(e, 0, [hx1, hy1, PEG_LEN / 2], [1, 0, 0, 0])
    for _ in range(20):
        e.step(OPEN_AND_HOLD)
    assert not e.seated_now[0]


def test_knocked_out_loses_credit():
    # Two pegs so seating peg 0 does not end the episode (all-peg success).
    e = _ep(slots=(0, 1))
    hx, hy = e.slot_xy[0]
    _place_peg(e, 0, [hx, hy, PEG_LEN / 2], [1, 0, 0, 0])
    for _ in range(10):
        e.step(OPEN_AND_HOLD)
    assert e.seated_now[0] and not e.done
    first = e.first_seat_step[0]
    # Yank peg 0 out of its hole.
    _place_peg(e, 0, [hx + 0.08, hy, PEG_LEN / 2], [1, 0, 0, 0])
    for _ in range(10):
        e.step(OPEN_AND_HOLD)
    assert not e.seated_now[0]
    seated = e.peg_seated_at_end()
    firsts = e.peg_first_seat_steps()
    assert seated[0] is False
    assert firsts[0] is None
    assert episode_score(seated, firsts, e.c.max_steps) < 0.5
    assert first is not None  # had been seated earlier


def test_upright_seat_in_own_slot_counts():
    e = _ep(slots=(0,), clearance=0.0004)
    hx, hy = e.slot_xy[0]
    _place_peg(e, 0, [hx, hy, PEG_LEN / 2], [1, 0, 0, 0])
    for _ in range(15):
        e.step(OPEN_AND_HOLD)
        if e.done:
            break
    assert e.seated_now[0]
    assert e.peg_seated_at_end() == [True]
