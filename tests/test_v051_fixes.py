"""v0.5.1: fixes from the Macrocosmos v0.5.0 validation notes (issue #13) and the
observation hardening.

    pytest -q tests/test_v051_fixes.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
from nist_sim import conditions  # noqa: E402
from nist_sim.conditions import (  # noqa: E402
    LAYOUT_WALL,
    _start_clear,
    draw_layout,
    episode_seed_sequence,
    sample_episodes,
)
from nist_sim.env import Episode, grasp_hand_yaw  # noqa: E402


def _overlap(a, b) -> bool:
    need = a.outer_radius() + b.outer_radius() + 2 * (LAYOUT_WALL + 0.001)
    return math.hypot(a.x - b.x, a.y - b.y) < need


def test_layout_never_overlaps():
    rng = np.random.default_rng(5)
    for _ in range(3000):
        lay = draw_layout(rng)
        for i in range(len(lay)):
            for j in range(i + 1, len(lay)):
                assert not _overlap(lay[i], lay[j])


def test_layout_redraws_then_raises(monkeypatch):
    # An impossible board region: every pass fails, so draw_layout must raise
    # instead of accepting an overlap.
    monkeypatch.setattr(conditions, "LAYOUT_X", (-0.001, 0.001))
    monkeypatch.setattr(conditions, "LAYOUT_Y", (-0.001, 0.001))
    monkeypatch.setattr(conditions, "LAYOUT_REDRAWS", 3)
    with pytest.raises(RuntimeError):
        draw_layout(np.random.default_rng(0))


def test_free_starts_clear_other_slots():
    bad = total = 0
    for seed in range(12):
        _, eps = sample_episodes(seed, 4)
        for c in eps:
            for p in c.pegs:
                if p.start_type in ("grasped", "near_hole"):
                    continue
                total += 1
                lx, ly = p.spec.x + p.offset_xy[0], p.spec.y + p.offset_xy[1]
                if not _start_clear(lx, ly, p.spec.outer_radius(), c.slots, 0.0):
                    bad += 1
    assert total > 300
    assert bad == 0, f"{bad}/{total} free starts overlap a slot"


@pytest.mark.parametrize("yaw", [0.0, 1.0, -1.0, 3.05, -3.05, 3.4, -3.4, 2 * math.pi + 0.3, 4.0])
def test_grasp_hand_yaw_wraps_into_joint_range(yaw):
    h = grasp_hand_yaw(yaw)
    assert -math.pi / 2 - 1e-9 <= h <= math.pi / 2 + 1e-9
    # Same grasp: differs by a whole number of half turns.
    k = (yaw - h) / math.pi
    assert abs(k - round(k)) < 1e-9


def test_grasped_start_near_pi_holds_peg():
    """A pre-grasped peg whose slot yaw is near ±pi is still held after settle."""
    for seed in range(40):
        _, eps = sample_episodes(seed, 2)
        for i, c in enumerate(eps):
            g = [k for k, p in enumerate(c.pegs) if p.start_type == "grasped"]
            if not g:
                continue
            p = c.pegs[g[0]]
            yaw = p.yaw_err + c.board_yaw + p.spec.yaw
            if abs(abs(math.remainder(yaw, 2 * math.pi)) - math.pi) > 0.25:
                continue
            r1, _ = (np.random.default_rng(s) for s in episode_seed_sequence(seed, i).spawn(2))
            e = Episode(c, r1)
            hyaw = float(e.data.qpos[3])
            assert abs(hyaw) <= math.pi / 2 + 0.05
            ppos, _ = e.peg_pose(g[0])
            assert ppos[2] > 0.03  # still lifted in the fingers
            return
    pytest.skip("no grasped start near ±pi in the sampled seeds")


def test_drawn_start_type_recorded():
    _, eps = sample_episodes(3, 2)
    c = eps[0]
    for p in c.pegs:
        assert p.drawn_start_type == p.start_type
    r1, _ = (np.random.default_rng(s) for s in episode_seed_sequence(3, 0).spawn(2))
    e = Episode(c, r1)
    e._force_safe_start()  # the spawn fallback may change the effective start
    pub = c.public()["pegs"]
    for p, d in zip(c.pegs, pub):
        assert d["start_type"] == p.start_type
        assert d["drawn_start_type"] == p.drawn_start_type
        assert d["drawn_start_type"] in conditions.PEG_START_TYPES


def test_glitch_bursts_happen_and_end():
    _, eps = sample_episodes(2, 2)
    c = next(c for c in eps if c.difficulty_tier == "hard")
    r1, r2 = (np.random.default_rng(s) for s in episode_seed_sequence(2, 0).spawn(2))
    e = Episode(c, r1)
    gains = [e._glitch_step(r2) for _ in range(3000)]
    on = np.mean([g > 1.0 for g in gains])
    assert 0.1 < on < 0.6
    assert gains.count(1.0) > 0


def test_more_lying_starts():
    n = lying = 0
    for seed in range(6):
        _, eps = sample_episodes(seed, 8)
        for c in eps:
            if c.difficulty_tier == "hard":
                n += len(c.pegs)
                lying += sum(p.drawn_start_type == "lying" for p in c.pegs)
    assert 0.12 < lying / n < 0.32
