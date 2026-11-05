"""v0.7 Assembly III: generator, obs layout, new step kinds, v06 unchanged under profile v06.

    NIST_PROFILE=v07 pytest -q tests/test_v07_assembly.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))

# This module exercises v07; set before importing profile-sensitive modules in workers.
os.environ["NIST_PROFILE"] = "v07"

from nist_sim import profile  # noqa: E402
from nist_sim.compound import OBS_DIM_V06, sample_compound  # noqa: E402
from nist_sim.compound_v07 import (  # noqa: E402
    OBS_DIM_V07,
    STEP_KINDS_V07,
    AssemblyIIIEpisode,
    compound_score_v07,
    sample_compound_v07,
)
from nist_sim import conditions as C  # noqa: E402


def _ep(seed: int, i: int, per: int = 4) -> AssemblyIIIEpisode:
    _, conds = sample_compound_v07(seed, per)
    r1, _ = (np.random.default_rng(s) for s in C.episode_seed_sequence(seed, i).spawn(2))
    return AssemblyIIIEpisode(conds[i], r1)


def test_profile_v07_registered():
    p = profile.get("v07")
    assert p.name == "v07" and p.physics_dt == 0.002
    assert OBS_DIM_V07 == OBS_DIM_V06 == 288


def test_v07_deterministic_and_all_tiers():
    _, a = sample_compound_v07(3, 4)
    _, b = sample_compound_v07(3, 4)
    assert [c.compound.public() for c in a] == [c.compound.public() for c in b]
    assert {c.compound.tier for c in a} == {"c1", "c2", "c3", "c4"}


def test_v07_has_new_step_kinds_on_hard_tiers():
    _, conds = sample_compound_v07(11, 8)
    kinds = {s.kind for c in conds for s in c.compound.steps}
    # Across a full draw we should see the new vocabulary (not every episode).
    assert kinds & {"rework", "screw", "stack", "key"}
    assert kinds <= set(STEP_KINDS_V07)


def test_v07_dag_and_wave_consistent_edges():
    _, conds = sample_compound_v07(5, 4)
    for c in conds:
        steps = c.compound.steps
        indeg = [len(s.prereq) for s in steps]
        ready = [k for k, d in enumerate(indeg) if d == 0]
        seen = 0
        while ready:
            u = ready.pop()
            seen += 1
            for v, s in enumerate(steps):
                if u in s.prereq:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        ready.append(v)
        assert seen == len(steps)
        # Wave-consistent: prerequisite wave <= dependent wave.
        for k, s in enumerate(steps):
            sw = c.compound.collars[s.part].wave if s.kind in ("collar", "stack") else c.compound.peg_wave[s.peg]
            for p in s.prereq:
                pre = steps[p]
                pw = c.compound.collars[pre.part].wave if pre.kind in ("collar", "stack") else c.compound.peg_wave[pre.peg]
                assert pw <= sw


def test_v07_obs_layout_and_idle_score():
    e = _ep(1, 0)
    r = np.random.default_rng(0)
    o = e.observe(r)
    assert o.shape == (288,)
    # Idle a few steps; score stays in partial band.
    for _ in range(8):
        e.step(np.array([0.0, 0.0, 0.3, 0.0, 1.0]))
    sc = compound_score_v07(e)
    assert 0.0 <= sc["score"] <= 0.30 + 1e-6


def test_v07_rework_starts_in_wrong_slot():
    _, conds = sample_compound_v07(9, 8)
    idx = next(
        (i for i, c in enumerate(conds) if any(s.kind == "rework" for s in c.compound.steps)),
        None,
    )
    if idx is None:
        pytest.skip("no rework in this draw")
    e = _ep(9, idx, per=8)
    s = next(s for s in e.cs.steps if s.kind == "rework")
    p, _ = e.peg_pose(s.peg)
    spec = e.c.slots[s.wrong_slot]
    from nist_sim.conditions import slot_world_xy

    wx, wy = slot_world_xy(e.c.board_xy, e.c.board_yaw, spec)
    assert float(np.hypot(p[0] - wx, p[1] - wy)) < 0.008
    # Not seated in the true hole yet.
    assert not e.seated_now[s.peg]


def test_v06_sample_unchanged_under_v06_env(monkeypatch):
    """v06 generator path must still be reachable and produce only insert/collar/twist."""
    monkeypatch.setenv("NIST_PROFILE", "v06")
    # sample_compound does not read the profile for graph shape; it is the v06 generator.
    _, conds = sample_compound(3, 4)
    kinds = {s.kind for c in conds for s in c.compound.steps}
    assert kinds <= {"insert", "collar", "twist"}
