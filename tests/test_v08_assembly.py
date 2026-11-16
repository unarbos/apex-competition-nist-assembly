"""v0.8 Assembly IV: interface, anti-aligned keys, no answer cues, scoring seals."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))

os.environ["NIST_PROFILE"] = "v08"

from nist_sim.compound_v08 import (  # noqa: E402
    KEY_ANTI_ALIGN,
    OBS_DIM_V08,
    AssemblyIVEpisode,
    compound_score_v08,
    sample_compound_v08,
    sample_key_yaw_v08,
    wrap,
)
from nist_sim.conditions import episode_seed_sequence  # noqa: E402


@pytest.fixture(scope="module")
def conds():
    _, c = sample_compound_v08(7, 8, "live")
    return c


def test_obs_dim_is_320(conds):
    ss = episode_seed_sequence(7, 0)
    r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
    ep = AssemblyIVEpisode(conds[0], r1)
    o = ep.observe(r2)
    assert o.shape == (OBS_DIM_V08,)
    assert OBS_DIM_V08 == 320


def test_blueprint_longer_than_v07(conds):
    mean = sum(len(c.compound.steps) for c in conds) / len(conds)
    assert mean >= 8.0
    assert max(len(c.compound.steps) for c in conds) >= 12


def test_key_yaw_anti_aligned(conds):
    for c in conds:
        for s in c.compound.steps:
            if s.kind != "key":
                continue
            peg = c.pegs[s.peg]
            slot_yaw = float(peg.spec.yaw + c.board_yaw)
            sym = int(peg.spec.symmetry())
            n = max(1, sym) if sym > 0 else 1
            forbidden = [slot_yaw + (2.0 * np.pi * k) / n for k in range(n)] if sym > 0 else [slot_yaw]
            forbidden = list(forbidden) + [0.0]
            dmin = min(abs(wrap(s.key_yaw - f)) for f in forbidden)
            assert dmin >= KEY_ANTI_ALIGN - 1e-6, (s.key_yaw, dmin, forbidden)


def test_no_closeness_or_detent_in_obs(conds):
    """Key task feats must not publish closeness / retarget pulse / key_tol."""
    ss = episode_seed_sequence(7, 1)
    r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
    ep = AssemblyIVEpisode(conds[1], r1)
    # Force probe+touch on a key peg if present so v0.7 would have published cues.
    key_pegs = [s.peg for s in ep.cs.steps if s.kind == "key"]
    if not key_pegs:
        pytest.skip("no key in episode")
    peg = key_pegs[0]
    ep._peg_touched[peg] = True
    ep._peg_probe_frames[peg] = 99
    ep._key_index_armed[peg] = True
    ep._key_retarget_pulse[peg] = 20
    o = ep.observe(r2)
    task0 = 184
    b = task0 + peg * 6
    # feat2 may be present-mark 0.5; feat3 must stay 0 (no closeness); feat5 not key_tol.
    assert float(o[b + 3]) == 0.0
    assert float(o[b + 5]) in (0.0, 1.0, 2.0)  # collar flag or empty — not tol*10 (~0.4)


def test_equal_weight_scoring_caps_insert_only(conds):
    """Finishing only inserts cannot dominate when the blueprint has many hard steps."""
    ss = episode_seed_sequence(7, 0)
    r1, _ = (np.random.default_rng(s) for s in ss.spawn(2))
    ep = AssemblyIVEpisode(conds[0], r1)
    n = len(ep.cs.steps)
    # Pretend all inserts done in order; nothing else.
    for k, s in enumerate(ep.cs.steps):
        if s.kind == "insert":
            ep.step_done[k] = True
            ep.step_in_order[k] = True
            ep.step_done_at[k] = 10
            ep._stability_ok[k] = True
            ep._precision[k] = 1.0
        else:
            ep.step_done[k] = False
            ep.step_progress[k] = 0.0
    ep._finalized = True
    ep.done = True
    sc = compound_score_v08(ep)["score"]
    n_ins = sum(1 for s in ep.cs.steps if s.kind == "insert")
    # Upper bound: inserts get ≤1.0 each, mean ≤ n_ins/n
    assert sc <= (n_ins / n) + 1e-6
    assert sc < 0.55  # blueprints are dense enough that inserts alone stay mid/low


def test_zero_policy_near_zero(conds):
    ss = episode_seed_sequence(7, 2)
    r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
    ep = AssemblyIVEpisode(conds[2], r1)
    z = np.zeros(5)
    while not ep.done:
        ep.observe(r2)
        ep.step(z)
    sc = compound_score_v08(ep)["score"]
    assert sc < 0.15


def test_sample_key_yaw_helper_reachable():
    rng = np.random.default_rng(0)
    _, conds = sample_compound_v08(3, 2, "live")
    for c in conds[:4]:
        for i in range(c.n_pegs):
            yaw = sample_key_yaw_v08(rng, c, i)
            assert abs(yaw) <= 2.95 + 1e-6
