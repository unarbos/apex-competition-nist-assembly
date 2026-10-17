"""Lying 4 mm pegs must be reachable: a scripted policy can pick them up and lift them.

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
from nist_sim.env import (  # noqa: E402
    DXYZ,
    DYAW,
    HAND_Z_MIN,
    Episode,
    EpisodeConditions,
)

# Part is "off the board" once its centre is well above its lying rest height (r + epsilon).
LIFT_Z = 0.012


def _long_xy(pquat: np.ndarray) -> np.ndarray:
    """World XY of the peg's long axis from a (w, x, y, z) quaternion."""
    w, x, y, z = pquat
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x)])


def _scripted_pick_lying(e: Episode) -> bool:
    """Open, align yaw with the short axis, lower to HAND_Z_MIN, close, lift.

    Pad normals are along hand +/-Y. For a side grasp they must squeeze the peg's
    diameter, so pad normals stay perpendicular to the peg's long axis.
    """
    for _ in range(50):
        ppos, pq = e.part_pose()
        hand = e.data.qpos[0:4]
        long_xy = _long_xy(pq)
        if float(np.linalg.norm(long_xy)) > 1e-6:
            want_yaw = math.atan2(long_xy[1], long_xy[0])
        else:
            want_yaw = float(hand[3])
        yaw_err = (want_yaw - hand[3] + math.pi) % (2 * math.pi) - math.pi
        ax = float(np.clip((ppos[0] - hand[0]) / DXYZ, -1, 1))
        ay = float(np.clip((ppos[1] - hand[1]) / DXYZ, -1, 1))
        az = float(np.clip((0.03 - hand[2]) / DXYZ, -1, 1))
        ayaw = float(np.clip(yaw_err / DYAW, -1, 1))
        e.step(np.array([ax, ay, az, ayaw, 1.0]))
        if e.done:
            return False
    for _ in range(40):
        ppos, _ = e.part_pose()
        hand = e.data.qpos[0:3]
        ax = float(np.clip((ppos[0] - hand[0]) / DXYZ, -1, 1))
        ay = float(np.clip((ppos[1] - hand[1]) / DXYZ, -1, 1))
        az = float(np.clip((HAND_Z_MIN - hand[2]) / DXYZ, -1, 1))
        e.step(np.array([ax, ay, az, 0.0, 1.0]))
        if e.done:
            return False
    for _ in range(25):
        e.step(np.array([0.0, 0.0, 0.0, 0.0, -1.0]))
        if e.done:
            return False
    for _ in range(45):
        hand = e.data.qpos[0:3]
        az = float(np.clip((0.06 - hand[2]) / DXYZ, -1, 1))
        e.step(np.array([0.0, 0.0, az, 0.0, -1.0]))
        if e.done:
            return False
    return float(e.part_pose()[0][2]) > LIFT_Z


def _lying_4mm(shape: str, seed: int, yaw: float) -> Episode:
    c = EpisodeConditions(
        shape=shape,
        diameter=0.004,
        clearance=0.0003,
        friction=1.2,
        peg_mass=0.008,
        gain_scale=1.0,
        obs_noise=0.0,
        reset_type="lying",
        hole_xy=(0.0, 0.0),
        offset_xy=(0.05, 0.0),
        height=0.02,
        yaw_err=yaw,
    )
    return Episode(c, np.random.default_rng(seed))


@pytest.mark.parametrize("shape", ["round", "square"])
@pytest.mark.parametrize("yaw", [0.0, 0.7, 1.5, -1.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scripted_policy_lifts_lying_4mm(shape, yaw, seed):
    e = _lying_4mm(shape, seed, yaw)
    assert _scripted_pick_lying(e), (
        f"failed to lift lying 4 mm {shape} (yaw={yaw}, seed={seed}): "
        f"part_z={e.part_pose()[0][2]:.4f} reason={e.terminal_reason}"
    )


def test_hand_z_floor_allows_pad_overlap_on_lying_4mm():
    """At HAND_Z_MIN the pad bottom sits below the peg top and above the board."""
    peg_top = 0.004
    pad_bottom_at_floor = HAND_Z_MIN + 0.004  # PAD_BOTTOM_OFFSET
    assert pad_bottom_at_floor < peg_top
    assert pad_bottom_at_floor > 0.0  # pads do not go through the board
