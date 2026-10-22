"""v0.5.0 generalization hardening: per-episode board layout and peg shapes.

There is no fixed slot table: slot positions, slot yaw, cross-section family and size
are drawn per episode, and the observation carries no shape label. These tests pin
that, plus the geometric validity of the drawn layouts.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
from nist_sim import conditions  # noqa: E402
from nist_sim.conditions import LAYOUT_WALL, N_SLOTS, sample_episodes  # noqa: E402
from nist_sim.env import ACT_DIM, PEG_FEAT, PROPRIO_DIM, Episode  # noqa: E402

PEG0 = PROPRIO_DIM + ACT_DIM


def _layouts(seed=0, per=4):
    _, eps = sample_episodes(seed, per)
    return eps


def test_layout_varies_per_episode():
    """No shared template: slot centres of two episodes do not line up."""
    eps = _layouts()
    assert all(len(c.slots) == N_SLOTS for c in eps)
    gaps = []
    for a, b in zip(eps, eps[1:]):
        pa = np.array([[s.x, s.y] for s in a.slots])
        pb = np.array([[s.x, s.y] for s in b.slots])
        d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1).min(axis=1)
        gaps.append(float(np.median(d)))
    # A fixed table would give ~0 mm; random layouts leave a cm-scale median gap.
    assert np.median(gaps) > 0.008, np.median(gaps)


def test_slot_index_carries_no_geometry():
    """Mean position of slot k over many episodes is near the board centre for every k."""
    eps = []
    for seed in range(4):
        eps += _layouts(seed, 8)
    pos = np.array([[[s.x, s.y] for s in c.slots] for c in eps])  # (E, 8, 2)
    mean_per_slot = pos.mean(axis=0)
    assert float(np.abs(mean_per_slot).max()) < 0.02, mean_per_slot


def test_layout_slots_do_not_overlap():
    for c in _layouts(1, 8):
        for i, a in enumerate(c.slots):
            for b in c.slots[i + 1 :]:
                d = math.hypot(a.x - b.x, a.y - b.y)
                assert d >= a.outer_radius() + b.outer_radius() + 2 * LAYOUT_WALL, (a, b)


def test_live_shape_variety_continuous():
    """Several families, continuous sizes, no single dominant family."""
    eps = []
    for seed in range(3):
        eps += _layouts(seed, 8)
    fam = [p.shape() for c in eps for p in c.pegs]
    sizes = {round(p.diameter(), 8) for c in eps for p in c.pegs}
    assert set(fam) == set(conditions.LIVE_SHAPES)
    assert max(fam.count(f) for f in set(fam)) / len(fam) < 0.45
    assert len(sizes) > 0.9 * len(fam)


def test_obs_has_no_shape_label():
    """Peg features 8:12 are slot-yaw estimate + outer size, identical layout for every family."""
    _, eps = sample_episodes(2, 2)
    for i, c in enumerate(eps[:6]):
        ss = np.random.SeedSequence([2, 0x5245, i])
        r1, r2 = (np.random.default_rng(x) for x in ss.spawn(2))
        e = Episode(c, r1)
        o = e.observe(r2)
        for k, peg in enumerate(c.pegs):
            f = o[PEG0 + k * PEG_FEAT : PEG0 + (k + 1) * PEG_FEAT]
            assert abs(f[8] ** 2 + f[9] ** 2 - 1.0) < 1e-3  # sin/cos of an angle, not one-hot
            assert f[11] == 0.0
            assert abs(f[12] / 50 - peg.diameter()) < 1e-5


def test_slot_yaw_estimate_is_biased_vs_peg_yaw():
    """The slot-yaw channel and peg-yaw channel do not share one offset (per-peg residual)."""
    _, eps = sample_episodes(3, 4)
    hard = [c for c in eps if c.difficulty_tier == "hard"]
    mags = [abs(d) for c in hard for d in c.peg_yaw_delta_list]
    assert mags and min(mags) >= conditions.PEG_YAW_DELTA[0] - 1e-9


def test_aligned_pegs_seat_in_every_family():
    """Geometry sanity: a peg dropped aligned onto its hole seats, for every family."""
    import mujoco

    from nist_sim.env import PEG_LEN, _yaw_quat

    seen = set()
    _, eps = sample_episodes(4, 2)
    for c in eps:
        e = Episode(c, np.random.default_rng(0))
        for i in range(e.n_pegs):
            hx, hy = e.slot_xy[i]
            e._set_free(e.peg_qadr[i], e.peg_vadr[i], [hx, hy, PEG_LEN / 2 + 0.0015], _yaw_quat(e.slot_yaw[i]))
        mujoco.mj_forward(e.model, e.data)
        e.target[:] = [c.board_xy[0], c.board_xy[1], 0.15, e.target[3]]
        for _ in range(25):
            e.step(np.array([0, 0, 1, 0, 1.0]))
        for i, p in enumerate(c.pegs):
            assert e.seated_now[i], (p.shape(), p.spec.public())
            seen.add(p.shape())
    assert seen == set(conditions.LIVE_SHAPES)
