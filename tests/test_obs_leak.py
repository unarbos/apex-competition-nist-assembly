"""Regression guard for the observation slot-estimate anti-scripting.

The slot layout is a fixed, public geometry. If every slot shared ONE global
pose-estimate bias, that bias (and the board yaw) could be solved from two or
more slot channels and the exact hole poses read straight out of the
observation, collapsing the intended contact-search difficulty.

These tests assert that each active slot now carries its OWN independent
estimate bias, so no affine (single-yaw + single-bias) inversion of the slot
channels recovers the true hole poses to within the fit, while the per-peg
relative residual that insertion must resolve is preserved.
"""

from __future__ import annotations

import math

import numpy as np

from nist_sim.conditions import SLOT_SPECS, sample_episodes, slot_world_xy
from nist_sim.env import ACT_DIM, PEG_FEAT, PROPRIO_DIM, Episode

PEG0 = PROPRIO_DIM + ACT_DIM


def _peg_feat(obs: np.ndarray, i: int) -> np.ndarray:
    return obs[PEG0 + i * PEG_FEAT : PEG0 + (i + 1) * PEG_FEAT]


def _best_affine_slot_recovery_error(obs: np.ndarray, cond) -> float:
    """Attacker view: best single (yaw, global-bias) fit of the slot channels,
    returning the max per-slot error between the recovered and the true hole
    position (board frame). Large => the exact poses are NOT readable."""
    est, lay = [], []
    for i, peg in enumerate(cond.pegs):
        f = _peg_feat(obs, i)
        if f[0] < 0.5:
            continue
        est.append([f[13] / 10.0, f[14] / 10.0])
        _, _, lx, ly = SLOT_SPECS[peg.slot]
        lay.append([lx, ly])
    est = np.array(est)
    lay = np.array(lay)
    em, lm = est.mean(0), lay.mean(0)
    de, dl = est - em, lay - lm
    yaw = math.atan2(
        np.sum(dl[:, 0] * de[:, 1] - dl[:, 1] * de[:, 0]),
        np.sum(dl[:, 0] * de[:, 0] + dl[:, 1] * de[:, 1]),
    )
    c, s = math.cos(yaw), math.sin(yaw)
    # Best single global bias the attacker can subtract from every slot estimate.
    global_bias = em - np.array([c * lm[0] - s * lm[1], s * lm[0] + c * lm[1]])
    worst = 0.0
    k = 0
    for i, peg in enumerate(cond.pegs):
        f = _peg_feat(obs, i)
        if f[0] < 0.5:
            continue
        # Attacker's best estimate of the true hole: slot estimate minus the one
        # global bias they could solve for (the per-slot bias is unknown to them).
        rec = est[k] - global_bias
        k += 1
        tx, ty = slot_world_xy(cond.board_xy, cond.board_yaw, peg.slot)
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
    """No affine inversion of the slot channels recovers exact hole poses."""
    worst_over_eps = []
    for seed in range(4):
        for i, c in _hard_episodes(seed):
            ss = np.random.SeedSequence([seed, 0x5245, i])
            r1, r2 = (np.random.default_rng(x) for x in ss.spawn(2))
            e = Episode(c, r1)
            # Average several observations to beat down white noise, as an attacker would.
            errs = [_best_affine_slot_recovery_error(e.observe(r2), c) for _ in range(8)]
            worst_over_eps.append(min(errs))  # attacker keeps their best fit
    worst_over_eps = np.array(worst_over_eps)
    # On hard tier the irreducible per-slot bias keeps the exact hole pose several
    # mm away from any global fit — above the clearance floor (0.25 mm) by a wide
    # margin, so contact search is still required.
    assert worst_over_eps.mean() > 0.0025, (
        f"slot poses too recoverable (mean worst {worst_over_eps.mean()*1e3:.2f} mm)"
    )
    assert np.median(worst_over_eps) > 0.0020


def test_relative_residual_preserved():
    """Per-peg peg-vs-own-slot residual (what insertion must resolve) is intact."""
    _, conds = sample_episodes(5, 4, "live")
    hard = [c for c in conds if c.difficulty_tier == "hard"]
    mags = [math.hypot(*c.peg_bias_delta_xy) for c in hard]
    assert mags
    # Design residual band ~1.5-5 mm (scaled by tier); hard tier keeps full scale.
    assert 0.0012 < float(np.mean(mags)) < 0.0060


def test_shared_global_bias_would_leak_regression():
    """Guard the premise: if a single shared slot bias is reintroduced, the exact
    hole poses become readable again (recovery error collapses to the noise floor).
    This is what the per-slot bias prevents; keep them independent."""
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
    e._peg_bias_xy = e._slot_bias_xy + np.array(c.peg_bias_delta_xy)
    errs = [_best_affine_slot_recovery_error(e.observe(r2), c) for _ in range(8)]
    # With a shared bias the global fit removes it entirely: exact poses recovered
    # to the obs-noise floor (sub-mm). This is the leak the fix closes.
    assert min(errs) < 0.0010, f"shared-bias leak not reproduced (min {min(errs)*1e3:.2f} mm)"
