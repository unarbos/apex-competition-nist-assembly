"""v0.6 compound tasks: generator determinism, task-graph invariants, waves, obs layout, rubric.

    pytest -q tests/test_v06_compound.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
import nist_sim.conditions as C  # noqa: E402
import mujoco  # noqa: E402
from nist_sim.compound import (  # noqa: E402
    BLOCK_H,
    COLLAR_H,
    DONE_BASE,
    OBS_DIM_V06,
    OUT_OF_ORDER,
    PARTIAL_CAP,
    CompoundEpisode,
    compound_score,
    sample_compound,
    step_credit,
)


def _ep(seed: int, i: int, per: int = 4) -> CompoundEpisode:
    _, conds = sample_compound(seed, per)
    r1, _ = (np.random.default_rng(s) for s in C.episode_seed_sequence(seed, i).spawn(2))
    return CompoundEpisode(conds[i], r1)


def test_deterministic_and_all_tiers():
    _, a = sample_compound(3, 4)
    _, b = sample_compound(3, 4)
    assert [c.compound.public() for c in a] == [c.compound.public() for c in b]
    assert {c.compound.tier for c in a} == {"c1", "c2", "c3", "c4"}


def test_task_graph_is_a_dag_with_mandatory_edges():
    _, conds = sample_compound(5, 4)
    for c in conds:
        steps = c.compound.steps
        for k, s in enumerate(steps):
            if s.kind in ("collar", "twist"):
                assert s.peg in s.prereq  # insert(peg) has index == peg
        # Kahn's algorithm consumes every node iff acyclic.
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
        assert c.max_steps > 0


def test_obs_layout_and_parked_parts_hidden():
    _, conds = sample_compound(1, 4)
    idx = next(i for i, c in enumerate(conds) if c.compound.n_waves > 1)
    e = _ep(1, idx)
    r = np.random.default_rng(0)
    o = e.observe(r)
    assert o.shape == (OBS_DIM_V06,)
    for i, parked in enumerate(e.peg_parked):
        if parked:
            assert not np.any(o[21 + i * 17 : 21 + (i + 1) * 17])


def test_wave_arrives_by_its_deadline():
    _, conds = sample_compound(1, 4)
    idx = next(i for i, c in enumerate(conds) if c.compound.n_waves > 1)
    e = _ep(1, idx)
    t = e.cs.wave_time[1]
    while not e.done and e.steps <= t:
        e.step(np.array([0.0, 0.0, 0.3, 0.0, 1.0]))
    assert e.wave_arrived[1] or e.done


def test_rubric():
    full = DONE_BASE + 0.30
    assert step_credit(True, True, 0, 100, 0.0) == full
    # Out-of-order is max(PARTIAL_CAP, OUT_OF_ORDER × in-order); never below partial.
    ooo = step_credit(True, False, 0, 100, 0.0)
    assert ooo == max(PARTIAL_CAP, OUT_OF_ORDER * full) >= PARTIAL_CAP
    assert step_credit(False, False, None, 100, 0.9) == PARTIAL_CAP
    e = _ep(1, 0)
    assert 0.0 <= compound_score(e)["score"] <= PARTIAL_CAP


def test_collar_partial_requires_peg_in_hole():
    """0.20 / 0.30 collar tiers need the peg seated; a ring on an empty slot stays ≤ 0.05."""
    _, conds = sample_compound(4, 8)
    idx = next(i for i, c in enumerate(conds) if c.compound.collars)
    e = _ep(4, idx, per=8)
    # Pick a collar whose peg starts unseated (not grasped into the hole).
    k = next(k for k, s in enumerate(e.cs.steps) if s.kind == "collar")
    j = e.cs.steps[k].part
    peg_i = e.cs.collars[j].peg
    if e.collar_parked[j]:
        e._arrive(e.cs.collars[j].wave)
    # Place the collar low over the empty slot without seating the peg.
    hx, hy = e.slot_xy[peg_i]
    e._set_free(e.collar_qadr[j], e.collar_vadr[j], [hx, hy, BLOCK_H + COLLAR_H / 2], [1.0, 0, 0, 0])
    # Keep peg away from the hole if it is not already out.
    if e._peg_in_hole(peg_i):
        p, q = e.peg_pose(peg_i)
        e._set_free(e.peg_qadr[peg_i], e.peg_vadr[peg_i], [hx + 0.05, hy + 0.05, p[2]], q)
    mujoco.mj_forward(e.model, e.data)
    done, prog = e._step_state(k)
    assert not done
    assert prog <= 0.05 + 1e-9
