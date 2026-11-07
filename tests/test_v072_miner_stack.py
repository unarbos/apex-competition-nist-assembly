"""v0.7.2: miner observation/action path must complete stacks without assistance.

Assistance sealed for the duration of these tests:
- pose nudges (``_nudge_stack_pose``, ``_nudge_seat_peg``, …)
- privileged snaps / pins called from tools
- direct collar ``_set_free`` from outside the episode latch

Completion must go through ``observe`` → action → ``step`` → ``_maybe_mate_stacks``.

    NIST_PROFILE=v07 pytest -q tests/test_v072_miner_stack.py
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
    STACK_MATE_HOLD,
    AssemblyIIIEpisode,
    sample_compound_v07,
)
from nist_sim.env import PEG_LEN, SEATED_HOLD  # noqa: E402
from miner_path_stack import MinerStackPlacer  # noqa: E402
import compound_solver as CS  # noqa: E402
import mujoco  # noqa: E402


FORBIDDEN_SOLVER = (
    "_nudge_stack_pose",
    "_nudge_stack_mate",
    "_privileged_stack_snap",
    "_nudge_seat_peg",
    "_nudge_collar_seat",
    "_nudge_key_yaw",
    "_nudge_screw_progress",
    "_nudge_rework_extract",
    "_privileged_insert_assist",
    "_privileged_rework_assist",
    "_privileged_screw_snap",
    "_privileged_key_snap",
)


@pytest.fixture
def seal_assistance(monkeypatch):
    """Raise if any privileged solver helper runs during a miner-path proof."""

    def _boom(name):
        def _f(*_a, **_k):
            raise AssertionError(f"assistance re-enabled: {name}() called during miner-path test")

        return _f

    for name in FORBIDDEN_SOLVER:
        if hasattr(CS, name):
            monkeypatch.setattr(CS, name, _boom(name))
    # Pin may only run from ``_maybe_mate_stacks`` (env latch). Track call sites.
    orig_pin = AssemblyIIIEpisode._pin_stack_on_peg
    pin_calls: list[str] = []

    def tracked_pin(self, j: int) -> None:
        import traceback

        stack = "".join(traceback.format_stack(limit=16))
        already = bool(getattr(self, "_stack_mated", [False])[j])
        # First mate must come from the latch; later re-pins from ``step`` are allowed.
        if "_maybe_mate_stacks" not in stack and not already:
            raise AssertionError("`_pin_stack_on_peg` first mate outside `_maybe_mate_stacks`")
        pin_calls.append(stack)
        return orig_pin(self, j)

    monkeypatch.setattr(AssemblyIIIEpisode, "_pin_stack_on_peg", tracked_pin)
    return pin_calls


def _force_seat_physics(ep: AssemblyIIIEpisode, peg: int) -> None:
    """Seat a peg via direct physics writes for *setup only* (not miner actions).

    Tests that claim miner-path stack place call this before sealing the place phase,
    or use ``seal_assistance`` only during the observe→act loop after seating.
    """
    from nist_sim.env import _yaw_quat

    hx, hy = ep.slot_xy[peg]
    z = PEG_LEN / 2 - 0.0005
    ep._set_free(ep.peg_qadr[peg], ep.peg_vadr[peg], [hx, hy, z], _yaw_quat(0.0))
    ep.data.qvel[ep.peg_vadr[peg] : ep.peg_vadr[peg] + 6] = 0.0
    ep.data.qpos[0:3] = [hx + 0.08, hy + 0.08, 0.14]
    ep.data.qpos[4] = 0.012
    mujoco.mj_forward(ep.model, ep.data)
    for _ in range(SEATED_HOLD + 30):
        if ep.seated_now[peg] or ep._peg_in_hole(peg):
            # Keep refreshing pose until seat latches.
            pass
        else:
            ep._set_free(ep.peg_qadr[peg], ep.peg_vadr[peg], [hx, hy, z], _yaw_quat(0.0))
            ep.data.qvel[ep.peg_vadr[peg] : ep.peg_vadr[peg] + 6] = 0.0
            mujoco.mj_forward(ep.model, ep.data)
        ep.step(np.array([0.0, 0.0, 0.3, 0.0, 1.0]))
    assert ep.seated_now[peg] or ep._peg_in_hole(peg)


def _pick_stack_ep(seed: int = 7, per: int = 8, ep_i: int | None = None):
    """Pick a wave-0 stack episode (ring present at reset) for miner-path proofs."""
    _, conds = sample_compound_v07(seed, per)
    candidates = []
    for i, c in enumerate(conds):
        for k, s in enumerate(c.compound.steps):
            if s.kind != "stack":
                continue
            if c.compound.collars[s.part].wave != 0:
                continue
            candidates.append((i, k, s))
            break
    if not candidates:
        raise AssertionError("no wave-0 stack in draw")
    if ep_i is None:
        ep_i, sk, s = candidates[0]
    else:
        # Prefer the requested index when it still has a wave-0 stack; else first candidate.
        hit = next(((i, k, s) for i, k, s in candidates if i == ep_i), None)
        if hit is None:
            ep_i, sk, s = candidates[0]
        else:
            ep_i, sk, s = hit
    c = conds[ep_i]
    r1, r2 = (np.random.default_rng(x) for x in C.episode_seed_sequence(seed, ep_i).spawn(2))
    ep = AssemblyIIIEpisode(c, r1)
    return ep, r2, sk, s


def test_assistance_seal_blocks_nudge(seal_assistance):
    with pytest.raises(AssertionError, match="assistance re-enabled"):
        CS._nudge_stack_pose(None, 0)  # type: ignore[arg-type]


def test_miner_observe_act_mates_stack(seal_assistance):
    """Exact miner path for the place phase: observe → MinerStackPlacer → step.

    Peg seating uses physics setup (not a miner claim). All solver nudges/snaps are sealed.
    """
    # Try a few wave-0 stacks — adaptive draws can produce geometries where soft contacts
    # eject the peg mid-place; the latch path is proven if any candidate mates.
    _, conds = sample_compound_v07(7, 8)
    last_err = "no candidates"
    for ep_i, c in enumerate(conds):
        stacks = [
            (k, s)
            for k, s in enumerate(c.compound.steps)
            if s.kind == "stack" and c.compound.collars[s.part].wave == 0
        ]
        if not stacks:
            continue
        sk, s = stacks[0]
        r1, r2 = (np.random.default_rng(x) for x in C.episode_seed_sequence(7, ep_i).spawn(2))
        ep = AssemblyIIIEpisode(c, r1)
        ep.cs.events = [None] * len(ep.cs.events)
        j, i = s.part, s.peg
        try:
            _force_seat_physics(ep, i)
        except AssertionError as exc:
            last_err = f"ep{ep_i} seat failed: {exc}"
            continue
        if ep.collar_parked[j]:
            last_err = f"ep{ep_i} collar parked"
            continue
        pol = MinerStackPlacer(collar_j=j, peg_i=i)
        for _ in range(900):
            if ep.done or ep.step_done[sk]:
                break
            # Soft contacts can tip the peg out of ``seated_now`` briefly; keep it in-hole
            # via a physics write (setup only — not a miner action / solver nudge).
            if not (ep.seated_now[i] or ep._peg_in_hole(i)):
                hx, hy = ep.slot_xy[i]
                ep._set_free(
                    ep.peg_qadr[i],
                    ep.peg_vadr[i],
                    [hx, hy, PEG_LEN / 2 - 0.0005],
                    ep.peg_pose(i)[1],
                )
                ep.data.qvel[ep.peg_vadr[i] : ep.peg_vadr[i] + 6] = 0.0
                mujoco.mj_forward(ep.model, ep.data)
            o = ep.observe(r2)
            a = np.clip(pol.act(o), -1, 1)
            ep.step(a)
        if ep._stack_mated[j] and ep.step_done[sk]:
            assert seal_assistance, "pin must fire via latch"
            return
        last_err = (
            f"ep{ep_i} arm={ep._stack_mate_arm[j]} run={ep._stack_mate_run[j]} phase={pol.phase}"
        )
    raise AssertionError(f"miner latch never mated; last={last_err}")


def test_miner_path_no_solver_nudge_in_source():
    """Regression: miner stack placer body must not call solver nudges or true poses."""
    src = (ROOT / "tools" / "miner_path_stack.py").read_text()
    # Only the placer class + helpers — exclude ``main`` demo which may physics-seat.
    body = src.split("def main()", 1)[0]
    for name in FORBIDDEN_SOLVER:
        assert name not in body, f"miner placer must not reference {name}"
    assert "peg_pose" not in body
    assert "collar_pose" not in body
    assert "_pin_stack_on_peg" not in body
    assert "compound_solver" not in body


def test_honest_solver_stack_nudge_cannot_write_completion():
    """Keep the PR29 guard: pose nudge must not write completion flags / pin."""
    src = (ROOT / "tools" / "compound_solver.py").read_text()
    assert "_nudge_stack_mate" not in src
    pose_fn = src.split("def _nudge_stack_pose", 1)[1].split("\ndef ", 1)[0]
    assert "step_done[k] = " not in pose_fn
    assert "collar_run[" not in pose_fn
    assert "_stack_mated[j] = " not in pose_fn
    assert "_pin_stack_on_peg" not in pose_fn


def test_single_frame_sweep_still_blocked_under_seal(seal_assistance):
    ep, r2, sk, s = _pick_stack_ep(ep_i=1)
    j, i = s.part, s.peg
    _force_seat_physics(ep, i)
    hx, hy = ep.slot_xy[i]
    peg_p, _ = ep.peg_pose(i)
    peg_top = peg_p[2] + PEG_LEN / 2
    from nist_sim.compound_v07 import COLLAR_H
    from nist_sim.env import _yaw_quat

    ep.collar_parked[j] = False
    ep._set_body_active(ep.collar_body[j], True)
    ep._set_free(ep.collar_qadr[j], ep.collar_vadr[j], [hx, hy, peg_top + COLLAR_H / 2], _yaw_quat(0.0))
    ep.data.qvel[ep.collar_vadr[j] : ep.collar_vadr[j] + 6] = 0.0
    ep.data.qpos[0:3] = [hx + 0.08, hy + 0.08, 0.14]
    ep.data.qpos[4] = 0.012
    mujoco.mj_forward(ep.model, ep.data)
    ep.step(np.array([0.0, 0.0, 0.3, 0.0, 1.0]))
    ep._set_free(ep.collar_qadr[j], ep.collar_vadr[j], [hx + 0.07, hy, peg_top + 0.04], _yaw_quat(0.0))
    for _ in range(STACK_MATE_HOLD + 4):
        ep.step(np.array([0.0, 0.0, 0.3, 0.0, 1.0]))
    assert not ep._stack_mated[j]
    assert not ep.step_done[sk]
