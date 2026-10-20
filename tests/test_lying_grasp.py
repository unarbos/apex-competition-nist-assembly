"""Lying 4 mm pegs must be reachable on the multi-peg board.

    pytest -q tests/test_lying_grasp.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
from nist_sim.conditions import EpisodeConditions, PegStart  # noqa: E402
from nist_sim.env import HAND_Z_MIN, Episode  # noqa: E402
from nist_sim.policies import StrongScripted  # noqa: E402

LIFT_Z = 0.012


def _lying_4mm(slot: int, seed: int, yaw: float) -> Episode:
    # Place in the open south apron of the board, clear of every hole opening.
    pegs = [
        PegStart(slot=slot, start_type="lying", offset_xy=(0.0, -0.12), height=0.02, yaw_err=yaw)
    ]
    c = EpisodeConditions(
        n_pegs=1,
        clutter_bucket="low",
        clearance=0.0003,
        friction=1.2,
        peg_mass=0.008,
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
        max_steps=400,
    )
    return Episode(c, np.random.default_rng(seed))


@pytest.mark.parametrize("slot", [0, 4])  # round 4mm, square 4mm
@pytest.mark.parametrize("yaw", [0.0, math.pi / 2])
@pytest.mark.parametrize("seed", [0, 1])
def test_strong_scripted_lifts_lying_4mm(slot, yaw, seed):
    e = _lying_4mm(slot, seed, yaw)
    pol = StrongScripted()
    noise = np.random.default_rng(seed)
    lifted = False
    for _ in range(160):
        e.step(np.clip(pol.act(e, e.observe(noise)), -1, 1))
        if float(e.peg_pose(0)[0][2]) > LIFT_Z:
            lifted = True
            break
        if e.done:
            break
    assert lifted, (
        f"failed to lift lying 4 mm slot={slot} (yaw={yaw}, seed={seed}): "
        f"part_z={e.peg_pose(0)[0][2]:.4f} phase={pol.phase}/{pol.sub} reason={e.terminal_reason}"
    )


def test_hand_z_floor_allows_pad_overlap_on_lying_4mm():
    peg_top = 0.004
    pad_bottom_at_floor = HAND_Z_MIN + 0.004
    assert pad_bottom_at_floor < peg_top
    assert pad_bottom_at_floor > 0.0
