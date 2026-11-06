"""v0.7.2: stack action-mate latch + twist/screw Bernoulli draw.

    NIST_PROFILE=v07 pytest -q tests/test_v072_stack_mate.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tools"))

os.environ["NIST_PROFILE"] = "v07"

from nist_sim import conditions as C  # noqa: E402
from nist_sim.compound_v07 import (  # noqa: E402
    COLLAR_H,
    STACK_INTERFERENCE,
    STACK_MATE_HOLD,
    AssemblyIIIEpisode,
    sample_compound_v07,
)
from nist_sim.env import PEG_LEN, SEATED_HOLD, _yaw_quat  # noqa: E402
from compound_solver import collar_routine, _nudge_seat_peg, solve_compound  # noqa: E402
import mujoco  # noqa: E402


def _ep(seed: int, i: int, per: int = 8) -> AssemblyIIIEpisode:
    _, conds = sample_compound_v07(seed, per)
    r1, _ = (np.random.default_rng(s) for s in C.episode_seed_sequence(seed, i).spawn(2))
    return AssemblyIIIEpisode(conds[i], r1)


def _first_stack_ep(seed: int = 7, per: int = 8) -> tuple[AssemblyIIIEpisode, int, int]:
    """Prefer a stack whose peg does not start lying (easier to force-seat for unit tests)."""
    _, conds = sample_compound_v07(seed, per)
    idx = None
    for i, c in enumerate(conds):
        for s in c.compound.steps:
            if s.kind != "stack":
                continue
            if c.pegs[s.peg].start_type == "lying":
                continue
            idx = i
            break
        if idx is not None:
            break
    if idx is None:
        idx = next(
            (i for i, c in enumerate(conds) if any(s.kind == "stack" for s in c.compound.steps)),
            None,
        )
    if idx is None:
        pytest.skip("no stack in draw")
    ep = _ep(seed, idx, per=per)
    sk = next(k for k, s in enumerate(ep.cs.steps) if s.kind == "stack")
    return ep, sk, idx


def _force_seat(ep: AssemblyIIIEpisode, peg: int) -> None:
    for k, s in enumerate(ep.cs.steps):
        if s.kind in ("insert", "rework") and s.peg == peg:
            ep.step_progress[k] = max(ep.step_progress[k], 0.20)
            break
    hx, hy = ep.slot_xy[peg]
    # Deep upright seat (bottom near 0) then idle so seated_now latches.
    z = PEG_LEN / 2 - 0.0005
    ep._set_free(ep.peg_qadr[peg], ep.peg_vadr[peg], [hx, hy, z], _yaw_quat(0.0))
    ep.data.qvel[ep.peg_vadr[peg] : ep.peg_vadr[peg] + 6] = 0.0
    ep.data.qpos[0:3] = [hx + 0.08, hy + 0.08, 0.14]
    ep.data.qpos[4] = 0.012
    mujoco.mj_forward(ep.model, ep.data)
    _nudge_seat_peg(ep, peg, min_prog=0.0)
    for _ in range(SEATED_HOLD + 25):
        if not (ep.seated_now[peg] or ep._peg_in_hole(peg)):
            ep._set_free(ep.peg_qadr[peg], ep.peg_vadr[peg], [hx, hy, z], _yaw_quat(0.0))
            ep.data.qvel[ep.peg_vadr[peg] : ep.peg_vadr[peg] + 6] = 0.0
            mujoco.mj_forward(ep.model, ep.data)
        ep.step(np.array([0.0, 0.0, 0.3, 0.0, 1.0]))
    assert ep.seated_now[peg] or ep._peg_in_hole(peg)


def test_stack_rings_are_interference_fit():
    _, conds = sample_compound_v07(7, 8)
    seen = 0
    for c in conds:
        for col in c.compound.collars:
            if not col.is_stack:
                continue
            outer = c.pegs[col.peg].spec.outer_radius()
            assert col.r_in < outer
            assert STACK_INTERFERENCE[0] - 1e-9 <= (outer - col.r_in) <= STACK_INTERFERENCE[1] + 1e-6
            seen += 1
    assert seen > 0


def test_twist_screw_bernoulli_appear_on_live_draw():
    """``round(frac * n)`` was always 0; per-peg Bernoulli must produce some twists/screws."""
    _, conds = sample_compound_v07(7, 8)
    kinds = {s.kind for c in conds for s in c.compound.steps}
    assert "twist" in kinds or "screw" in kinds
    n_ts = sum(1 for c in conds for s in c.compound.steps if s.kind in ("twist", "screw"))
    assert n_ts >= 1


def test_scripted_ungripped_hold_mates_stack():
    """Armed ring + ≥STACK_MATE_HOLD released frames in zone pins via `_maybe_mate_stacks`."""
    ep, sk, _ = _first_stack_ep()
    s = ep.cs.steps[sk]
    j, i = s.part, s.peg
    _force_seat(ep, i)
    hx, hy = ep.slot_xy[i]
    peg_p, _ = ep.peg_pose(i)
    peg_top = peg_p[2] + PEG_LEN / 2
    z_ring = peg_top + COLLAR_H / 2
    ep.collar_parked[j] = False
    ep._set_body_active(ep.collar_body[j], True)
    ep._set_free(ep.collar_qadr[j], ep.collar_vadr[j], [hx, hy, z_ring], _yaw_quat(0.0))
    ep.data.qvel[ep.collar_vadr[j] : ep.collar_vadr[j] + 6] = 0.0
    ep.data.qpos[0:3] = [hx + 0.06, hy + 0.06, 0.14]
    ep.data.qpos[4] = 0.012
    mujoco.mj_forward(ep.model, ep.data)
    # Seed arm as a prior deliberate closed-grasp place would (contact-free teleport
    # does not create pad touch, so we cannot arm through physics here).
    ep._stack_mate_arm[j] = 2
    assert ep._stack_in_mate_zone(j)
    assert ep._stack_release_clear(j)
    for _ in range(STACK_MATE_HOLD + SEATED_HOLD + 6):
        ep.step(np.array([0.0, 0.0, 0.3, 0.0, 1.0]))
        if ep.step_done[sk]:
            break
    assert ep._stack_mated[j]
    assert ep.step_done[sk]


def test_single_frame_sweep_does_not_free_pin():
    ep, sk, _ = _first_stack_ep()
    s = ep.cs.steps[sk]
    j, i = s.part, s.peg
    _force_seat(ep, i)
    hx, hy = ep.slot_xy[i]
    peg_p, _ = ep.peg_pose(i)
    peg_top = peg_p[2] + PEG_LEN / 2
    ep.collar_parked[j] = False
    ep._set_body_active(ep.collar_body[j], True)
    # One frame in zone.
    ep._set_free(ep.collar_qadr[j], ep.collar_vadr[j], [hx, hy, peg_top + COLLAR_H / 2], _yaw_quat(0.0))
    ep.data.qvel[ep.collar_vadr[j] : ep.collar_vadr[j] + 6] = 0.0
    ep.data.qpos[0:3] = [hx + 0.08, hy + 0.08, 0.14]
    ep.data.qpos[4] = 0.012
    mujoco.mj_forward(ep.model, ep.data)
    ep.step(np.array([0.0, 0.0, 0.3, 0.0, 1.0]))
    assert ep._stack_mate_run[j] <= 1
    # Yank away before hold completes.
    ep._set_free(ep.collar_qadr[j], ep.collar_vadr[j], [hx + 0.07, hy, peg_top + 0.04], _yaw_quat(0.0))
    for _ in range(6):
        ep.step(np.array([0.0, 0.0, 0.3, 0.0, 1.0]))
    assert not ep._stack_mated[j]
    assert not ep.step_done[sk]


def test_collar_routine_mates_without_nudge():
    """Privileged place-and-release completes via the latch only (no `_nudge_stack_mate`)."""
    ep, sk, _ = _first_stack_ep()
    s = ep.cs.steps[sk]
    j, i = s.part, s.peg
    _force_seat(ep, i)
    for a in collar_routine(ep, j, stack=True):
        if ep.done or ep.steps >= ep.c.max_steps - 2:
            break
        ep.step(a)
        if ep.step_done[sk]:
            break
    for _ in range(30):
        if ep.step_done[sk] or ep.done:
            break
        ep.step(np.array([0.0, 0.0, 0.3, 0.0, 1.0]))
    assert ep._stack_mated[j]
    assert ep.step_done[sk]


def test_honest_solver_no_stack_pin_teleport():
    """Honest path may pose-nudge; must not pin/snap-complete stacks."""
    src = (ROOT / "tools" / "compound_solver.py").read_text()
    assert "_nudge_stack_mate" not in src
    assert "def _nudge_stack_pose" in src
    assert "def _privileged_stack_snap" in src
    pose_fn = src.split("def _nudge_stack_pose", 1)[1].split("\ndef ", 1)[0]
    # May read completion flags; must not write them.
    assert "step_done[k] = " not in pose_fn
    assert "collar_run[" not in pose_fn
    assert "_stack_mated[j] = " not in pose_fn
    assert "_pin_stack_on_peg" not in pose_fn


@pytest.mark.slow
def test_honest_privileged_stack_completion_rate_seed7():
    """Live-sized seed-7 / 64-ep honest run: complete ≥50% of drawn stacks (no stack teleport)."""
    seed, per = 7, 8
    _, conds = sample_compound_v07(seed, per)
    drawn = completed = 0
    for i, c in enumerate(conds):
        stack_steps = [k for k, s in enumerate(c.compound.steps) if s.kind == "stack"]
        if not stack_steps:
            continue
        drawn += len(stack_steps)
        r1, _ = (np.random.default_rng(s) for s in C.episode_seed_sequence(seed, i).spawn(2))
        ep = AssemblyIIIEpisode(c, r1)
        solve_compound(ep, oracle=False)
        completed += sum(1 for k in stack_steps if ep.step_done[k])
    assert drawn > 0
    rate = completed / drawn
    assert rate >= 0.50, f"stack completed/drawn={completed}/{drawn} ({rate:.2%}) < 50%"
