"""Regression guards for the observation hardening (v0.3.0, kept in v0.5.0).

Slot, clutter and peg-vs-slot estimate offsets are drawn independently per slot /
per block / per peg. These tests pin that structure and check that the per-peg
relative residual (what insertion has to resolve) keeps its designed magnitude.
Since v0.5.0 the slot layout is drawn per episode; the global-fit checks below use
the episode's own true layout, i.e. even a policy that knew the layout exactly could
not recover the hole poses from one shared offset.
"""

from __future__ import annotations

import math

import numpy as np

from nist_sim.conditions import sample_episodes, slot_world_xy
from nist_sim.env import ACT_DIM, PEG_FEAT, PROPRIO_DIM, Episode

PEG0 = PROPRIO_DIM + ACT_DIM


def _peg_feat(obs: np.ndarray, i: int) -> np.ndarray:
    return obs[PEG0 + i * PEG_FEAT : PEG0 + (i + 1) * PEG_FEAT]


def _global_fit_slot_error(obs: np.ndarray, cond) -> float:
    """Best single (yaw, offset) fit of the slot channels; returns the max per-slot
    error against the true hole position (board frame)."""
    est, lay = [], []
    for i, peg in enumerate(cond.pegs):
        f = _peg_feat(obs, i)
        if f[0] < 0.5:
            continue
        est.append([f[13] / 10.0, f[14] / 10.0])
        lay.append([peg.spec.x, peg.spec.y])
    est = np.array(est)
    lay = np.array(lay)
    em, lm = est.mean(0), lay.mean(0)
    de, dl = est - em, lay - lm
    yaw = math.atan2(
        np.sum(dl[:, 0] * de[:, 1] - dl[:, 1] * de[:, 0]),
        np.sum(dl[:, 0] * de[:, 0] + dl[:, 1] * de[:, 1]),
    )
    c, s = math.cos(yaw), math.sin(yaw)
    global_bias = em - np.array([c * lm[0] - s * lm[1], s * lm[0] + c * lm[1]])
    worst = 0.0
    k = 0
    for i, peg in enumerate(cond.pegs):
        f = _peg_feat(obs, i)
        if f[0] < 0.5:
            continue
        rec = est[k] - global_bias
        k += 1
        tx, ty = slot_world_xy(cond.board_xy, cond.board_yaw, peg.spec)
        true_rel = np.array([tx - cond.board_xy[0], ty - cond.board_xy[1]])
        worst = max(worst, float(np.hypot(*(rec - true_rel))))
    return worst


def _hard_episodes(seed: int, per: int = 4):
    _, conds = sample_episodes(seed, per, "live")
    return [(i, c) for i, c in enumerate(conds) if c.difficulty_tier == "hard" and len(c.pegs) >= 3]


def test_slot_biases_are_independent_per_slot():
    """A hard episode must not share a single global slot bias."""
    _, conds = sample_episodes(3, 4, "live")
    hard = [c for c in conds if c.difficulty_tier == "hard" and len(c.pegs) >= 4]
    assert hard, "expected at least one multi-peg hard episode"
    c = hard[0]
    biases = np.array(c.slot_bias_list)
    assert biases.shape[0] == len(c.pegs)
    # Not all equal: max pairwise distance is several mm, not ~0.
    maxpair = 0.0
    for a in range(len(biases)):
        for b in range(a + 1, len(biases)):
            maxpair = max(maxpair, float(np.hypot(*(biases[a] - biases[b]))))
    assert maxpair > 0.002, f"slot biases look shared (max pairwise {maxpair*1e3:.2f} mm)"


def test_true_slot_poses_not_readable_from_obs():
    """A single global fit of the slot channels does not give exact hole poses."""
    worst_over_eps = []
    for seed in range(4):
        for i, c in _hard_episodes(seed):
            ss = np.random.SeedSequence([seed, 0x5245, i])
            r1, r2 = (np.random.default_rng(x) for x in ss.spawn(2))
            e = Episode(c, r1)
            errs = [_global_fit_slot_error(e.observe(r2), c) for _ in range(8)]
            worst_over_eps.append(min(errs))
    worst_over_eps = np.array(worst_over_eps)
    # On hard tier the irreducible per-slot bias keeps the exact hole pose several
    # mm away from any global fit — above the clearance floor (0.25 mm) by a wide
    # margin, so contact search is still required.
    assert worst_over_eps.mean() > 0.0025, (
        f"global fit too close (mean worst {worst_over_eps.mean()*1e3:.2f} mm)"
    )
    assert np.median(worst_over_eps) > 0.0020


def test_relative_residual_preserved():
    """Per-peg peg-vs-own-slot residual (what insertion must resolve) is intact."""
    _, conds = sample_episodes(5, 4, "live")
    hard = [c for c in conds if c.difficulty_tier == "hard"]
    mags = [math.hypot(*d) for c in hard for d in c.peg_bias_delta_list]
    assert mags
    # Design residual band ~1.5-5 mm (scaled by tier); hard tier keeps full scale.
    assert 0.0012 < float(np.mean(mags)) < 0.0060


def test_shared_offset_premise_regression():
    """Guard the premise: with one shared slot offset the global fit is exact, so the
    per-slot offsets must stay independent."""
    seed = 2
    hard = _hard_episodes(seed)
    assert hard
    i, c = hard[0]
    ss = np.random.SeedSequence([seed, 0x5245, i])
    r1, r2 = (np.random.default_rng(x) for x in ss.spawn(2))
    e = Episode(c, r1)
    # Force the pre-fix behaviour: one global slot bias shared by every slot.
    shared = e._slot_bias_xy[0].copy()
    e._slot_bias_xy = np.tile(shared, (e.n_pegs, 1))
    e._peg_bias_xy = e._slot_bias_xy + np.array(c.peg_bias_delta_list)
    errs = [_global_fit_slot_error(e.observe(r2), c) for _ in range(8)]
    assert min(errs) < 0.0010, f"shared-offset premise not reproduced (min {min(errs)*1e3:.2f} mm)"


def test_peg_residuals_are_independent_per_peg():
    """The peg-vs-slot residual is drawn independently per peg."""
    _, conds = sample_episodes(3, 4, "live")
    hard = [c for c in conds if c.difficulty_tier == "hard" and len(c.pegs) >= 4]
    assert hard
    gaps = []
    for c in hard:
        d = np.array(c.peg_bias_delta_list)
        assert d.shape[0] == len(c.pegs)
        gaps.append(max(float(np.hypot(*(d[a] - d[b]))) for a in range(len(d)) for b in range(a + 1, len(d))))
    assert float(np.mean(gaps)) > 0.002


def test_seated_residual_does_not_predict_other_pegs():
    """Residuals of different pegs in one episode are not interchangeable."""
    errs = []
    for seed in range(3):
        for i, c in _hard_episodes(seed):
            ss = np.random.SeedSequence([seed, 0x5245, i])
            r1, r2 = (np.random.default_rng(x) for x in ss.spawn(2))
            e = Episode(c, r1)
            d = e._peg_bias_xy - e._slot_bias_xy  # per-peg residual actually applied
            for j in range(1, len(c.pegs)):
                errs.append(float(np.hypot(*(d[j] - d[0]))))
    assert np.median(errs) > 0.0015, f"per-peg residuals too similar ({np.median(errs)*1e3:.2f} mm)"
