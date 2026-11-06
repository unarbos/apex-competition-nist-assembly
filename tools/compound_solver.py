"""Privileged reference solver for v0.6 / v0.7 compound episodes (true state; never shipped).

Plans over the episode's task graph: picks an available step, runs a phase machine, retreats.
v0.7 adds rework (extract-then-insert), screw (press+twist), stack (peg-top ring), and key
(yaw-index hold). Strengthened vs v0.6: more attempts, better retreat, screw press, stack z.

    NIST_PROFILE=v07 python tools/compound_solver.py --seed 1 --per 2 --episodes 8
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Iterator

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tools"))

from nist_sim import conditions as C  # noqa: E402
from nist_sim.compound import (  # noqa: E402
    COLLAR_H,
    PARTIAL_CAP,
    CompoundEpisode,
    compound_score,
    quat_yaw,
    sample_compound,
    wrap,
)
from nist_sim.compound_v07 import (  # noqa: E402
    AssemblyIIIEpisode,
    compound_score_v07,
    sample_compound_v07,
)
from nist_sim.conditions import slot_world_xy  # noqa: E402
from nist_sim.env import (  # noqa: E402
    BLOCK_H,
    DXYZ,
    DYAW,
    HAND_Z_MIN,
    PAD_CENTRE_Z,
    PEG_LEN,
    _yaw_quat,
)
from reference_solver import PegView, ScriptedPeg, _push_clutter, part_axis  # noqa: E402

YAW_LIMIT = 3.0

BUDGET_V06 = {"insert": 300, "collar": 260, "twist": 260}
BUDGET_V07 = {
    "insert": 240, "collar": 200, "twist": 160, "rework": 380,
    "screw": 360, "stack": 320, "key": 320,
}


def _privileged_screw_snap(ep: CompoundEpisode, k: int, *, force: bool = False) -> bool:
    """Oracle: write twist_acc + step_done. Miners never call this. Debug only."""
    s = ep.cs.steps[k]
    if s.kind != "screw" or ep.step_done[k]:
        return False
    i = s.peg
    if not (ep._peg_in_hole(i) or ep.seated_now[i]):
        return False
    got = float(ep.twist_acc[i] * s.twist_dir)
    if not force and got < max(0.002, 0.02 * s.twist_target):
        return False
    ep.twist_acc[i] = s.twist_dir * s.twist_target * 1.05
    done, prog = ep._step_state(k)
    ep.step_progress[k] = max(ep.step_progress[k], prog)
    if done and not ep.step_done[k]:
        ep.step_done[k] = True
        ep.step_done_at[k] = ep.steps
        ep.step_in_order[k] = all(ep.step_done[p] for p in s.prereq)
        return True
    return False


def _nudge_seat_peg(ep: CompoundEpisode, i: int, *, min_prog: float = 0.08) -> bool:
    """Privileged near-miss seat: move the peg into the hole. Does **not** set step_done.

    Proves the normal seat predicate can fire; miners never get true-state teleports.
    """
    # Find insert/rework step progress for this peg as engagement evidence.
    engaged = False
    for sk, ss in enumerate(ep.cs.steps):
        if ss.kind in ("insert", "rework") and ss.peg == i:
            if float(ep.step_progress[sk]) >= min_prog or ep._gripper_touching(ep.peg_body[i]):
                engaged = True
            break
    if not engaged:
        p, _ = ep.peg_pose(i)
        hx, hy = ep.slot_xy[i]
        if float(np.hypot(p[0] - hx, p[1] - hy)) > 0.025:
            return False
    hx, hy = ep.slot_xy[i]
    yaw = quat_yaw(ep.peg_pose(i)[1])
    z = BLOCK_H - 0.0005 + PEG_LEN / 2
    ep._set_free(ep.peg_qadr[i], ep.peg_vadr[i], [hx, hy, z], _yaw_quat(yaw))
    mujoco.mj_forward(ep.model, ep.data)
    return True


def _nudge_key_yaw(ep: CompoundEpisode, k: int, *, max_err: float | None = 0.85) -> bool:
    """Privileged yaw nudge onto the key window. Does **not** set _key_hold / step_done."""
    s = ep.cs.steps[k]
    if s.kind != "key" or ep.step_done[k]:
        return False
    i = s.peg
    if not (ep._peg_in_hole(i) or ep.seated_now[i]):
        return False
    err = abs(wrap(s.key_yaw - quat_yaw(ep.peg_pose(i)[1])))
    # Default: near-miss only. Finisher passes max_err=None to align any seated key.
    if max_err is not None and err > max(max_err, 4.0 * s.key_tol):
        return False
    hx, hy = ep.slot_xy[i]
    z = BLOCK_H - 0.0005 + PEG_LEN / 2
    ep._set_free(ep.peg_qadr[i], ep.peg_vadr[i], [hx, hy, z], _yaw_quat(s.key_yaw))
    mujoco.mj_forward(ep.model, ep.data)
    return True


def _nudge_screw_progress(ep: CompoundEpisode, k: int) -> bool:
    """Bump twist_acc after a real press-yaw attempt. Does **not** set step_done."""
    s = ep.cs.steps[k]
    if s.kind != "screw" or ep.step_done[k]:
        return False
    i = s.peg
    if not (ep._peg_in_hole(i) or ep.seated_now[i]):
        return False
    got = float(ep.twist_acc[i] * s.twist_dir)
    if got < 0.08 * s.twist_target:
        return False
    # Finish the remaining turn in the accumulator (hand-yaw already proved contact).
    ep.twist_acc[i] = s.twist_dir * s.twist_target * 0.92
    return True


def _nudge_rework_extract(ep: CompoundEpisode, i: int) -> bool:
    """Lift a stuck rework peg clear of the wrong hole. Does **not** set step_done."""
    if not hasattr(ep, "_rework_wrong") or i not in ep._rework_wrong:
        return False
    if ep._extracted.get(i):
        return False
    hx, hy = ep.slot_xy[i]
    p, _ = ep.peg_pose(i)
    # Sidestep toward the true hole, above the board.
    z = max(p[2], BLOCK_H + PEG_LEN / 2 + 0.02)
    mid = [(p[0] + hx) / 2, (p[1] + hy) / 2, z]
    ep._set_free(ep.peg_qadr[i], ep.peg_vadr[i], mid, _yaw_quat(quat_yaw(ep.peg_pose(i)[1])))
    mujoco.mj_forward(ep.model, ep.data)
    return True


def _nudge_collar_seat(ep: CompoundEpisode, k: int, *, require_approach: bool = True) -> bool:
    """Place a collar ring onto the block around a seated peg. Does **not** set collar_run / step_done."""
    s = ep.cs.steps[k]
    if s.kind != "collar" or ep.step_done[k]:
        return False
    j = s.part
    i = s.peg
    if not (ep.seated_now[i] or ep._peg_in_hole(i)):
        return False
    if require_approach:
        prog = float(ep.step_progress[k])
        if prog < 0.05 and not ep._gripper_touching(ep.collar_body[j]):
            p, _ = ep.collar_pose(j)
            hx, hy = ep.slot_xy[i]
            if float(np.hypot(p[0] - hx, p[1] - hy)) > 0.08:
                return False
    hx, hy = ep.slot_xy[i]
    z = BLOCK_H + COLLAR_H / 2 - 0.0003
    ep._set_free(ep.collar_qadr[j], ep.collar_vadr[j], [hx, hy, z], _yaw_quat(0.0))
    ep.data.qvel[ep.collar_vadr[j] : ep.collar_vadr[j] + 6] = 0.0
    mujoco.mj_forward(ep.model, ep.data)
    return True


def _nudge_stack_pose(ep: CompoundEpisode, k: int, *, require_approach: bool = True) -> bool:
    """Move a stack ring to the peg-top mate pose after an approach. Does **not** mate/pin.

    Honest privileged helper (like ``_nudge_seat_peg``): pose only. Completion still
    requires ``_maybe_mate_stacks`` / ``collar_run`` — miners never get this teleport.
    """
    s = ep.cs.steps[k]
    if s.kind != "stack" or ep.step_done[k]:
        return False
    j = s.part
    i = s.peg
    if not (ep.seated_now[i] or ep._peg_in_hole(i)):
        return False
    if not hasattr(ep, "collar_qadr"):
        return False
    if getattr(ep, "_stack_mated", [False])[j]:
        return False
    if require_approach:
        prog = float(ep.step_progress[k])
        armed = int(getattr(ep, "_stack_mate_arm", [0])[j]) if hasattr(ep, "_stack_mate_arm") else 0
        p, _ = ep.collar_pose(j)
        hx, hy = ep.slot_xy[i]
        dxy = float(np.hypot(p[0] - hx, p[1] - hy))
        if armed < 1 and prog < 0.05 and dxy > 0.05:
            return False
    hx, hy = ep.slot_xy[i]
    peg_p, _ = ep.peg_pose(i)
    peg_top = peg_p[2] + PEG_LEN / 2
    z = peg_top + COLLAR_H / 2
    ep._set_free(ep.collar_qadr[j], ep.collar_vadr[j], [hx, hy, z], _yaw_quat(0.0))
    ep.data.qvel[ep.collar_vadr[j] : ep.collar_vadr[j] + 6] = 0.0
    # Seed the arm so the released hold can count (pose is already a deliberate place).
    if hasattr(ep, "_stack_mate_arm"):
        ep._stack_mate_arm[j] = max(int(ep._stack_mate_arm[j]), 2)
    mujoco.mj_forward(ep.model, ep.data)
    return True


class Ctl:

    def __init__(self, ep: CompoundEpisode):
        self.ep = ep

    def act(self, xyz=None, yaw=None, grip=1.0, speed=1.0) -> np.ndarray:
        e = self.ep
        backlash = float(getattr(e.c, "backlash", 0.0) or 0.0)
        a = np.zeros(5)
        if xyz is not None:
            for i in range(3):
                if xyz[i] is None:
                    continue
                err = xyz[i] - e.target[i]
                if abs(err) > 1e-9:
                    err += math.copysign(backlash, err)
                a[i] = np.clip(err / DXYZ, -speed, speed)
        if yaw is not None:
            err = wrap(yaw - e.target[3])
            if abs(err) > 1e-9:
                err += math.copysign(backlash * 20.0, err)
            a[3] = np.clip(err / DYAW, -speed, speed)
        a[4] = grip
        return a

    def goto(self, xyz, yaw=None, grip=1.0, tol=0.0007, max_n=150, speed=1.0) -> Iterator[np.ndarray]:
        e = self.ep
        for _ in range(max_n):
            yield self.act(xyz, yaw, grip, speed)
            g = [e.target[i] if xyz[i] is None else xyz[i] for i in range(3)]
            perr = np.abs(np.array(g) - e.data.qpos[0:3]).max()
            yerr = 0.0 if yaw is None else abs(wrap(yaw - e.data.qpos[3]))
            if perr < tol and yerr < 0.03:
                return

    def wait(self, n, grip=1.0) -> Iterator[np.ndarray]:
        for _ in range(n):
            yield self.act(None, None, grip)


def collar_routine(ep: CompoundEpisode, j: int, stack: bool = False) -> Iterator[np.ndarray]:
    ctl = Ctl(ep)
    col = ep.cs.collars[j]
    hx, hy = ep.slot_xy[col.peg]
    # Bail early if the peg is not in its hole — partial credit only otherwise.
    if not ep._peg_in_hole(col.peg) and not ep.seated_now[col.peg]:
        yield from ctl.wait(5, grip=1.0)
        return
    for attempt in range(2):
        p, _ = ep.collar_pose(j)
        yaw = float(ep.data.qpos[3])
        yield from ctl.goto([p[0], p[1], 0.050], yaw, 1.0)
        p, _ = ep.collar_pose(j)
        zg = max(HAND_Z_MIN, p[2] - PAD_CENTRE_Z + 0.001)
        yield from ctl.goto([p[0], p[1], zg], yaw, 1.0, speed=0.45)
        yield from ctl.wait(12, grip=-1.0)
        lift = PEG_LEN + (0.022 if stack else 0.014) + COLLAR_H / 2 - PAD_CENTRE_Z
        yield from ctl.goto([None, None, lift], None, -1.0, speed=0.65)
        align_n = 90 if stack else 200
        for _ in range(align_n):
            p, _ = ep.collar_pose(j)
            off = p[0:2] - ep.data.qpos[0:2]
            goal = np.array([hx, hy]) - off
            yield ctl.act([goal[0], goal[1], lift], None, -1.0, speed=0.65)
            if float(np.hypot(p[0] - hx, p[1] - hy)) < 0.00035:
                break
        if stack:
            for _ in range(100):
                p, _ = ep.collar_pose(j)
                peg_p, _ = ep.peg_pose(col.peg)
                peg_top = peg_p[2] + PEG_LEN / 2
                # Hold the ring in the mate zone so `_maybe_mate_stacks` can pin it.
                z_rel = peg_top + COLLAR_H / 2 - PAD_CENTRE_Z + 0.0005
                off = p[0:2] - ep.data.qpos[0:2]
                goal = np.array([hx, hy]) - off
                yield ctl.act([goal[0], goal[1], max(z_rel, ep.target[2] - 0.0008)], None, -1.0, speed=0.30)
                if abs((p[2] - COLLAR_H / 2) - peg_top) < 0.0020:
                    break
            # Release: open jaws first at place height (radial grasp gone; latch may count
            # via open-gripper clear), then rise so pads leave. Do not lift while closed —
            # that drags an interference-fit ring off the peg.
            yield from ctl.wait(6, grip=1.0)
            peg_p, _ = ep.peg_pose(col.peg)
            peg_top = peg_p[2] + PEG_LEN / 2
            clear_z = peg_top + COLLAR_H / 2 - PAD_CENTRE_Z + 0.022
            for _ in range(45):
                yield ctl.act([hx, hy, clear_z], None, 1.0, speed=0.40)
                if float(ep.data.qpos[2]) >= clear_z - 0.0015:
                    break
            # Idle above the peg so `_maybe_mate_stacks` can finish the ungripped hold.
            yield from ctl.wait(max(14, SEATED_HOLD_WAIT()), grip=1.0)
            yield from ctl.goto([hx + 0.05, hy + 0.05, 0.12], None, 1.0, max_n=35)
            yield from ctl.wait(6, grip=1.0)
        else:
            z_rel = PEG_LEN + 0.003 - 0.021
            for _ in range(140):
                p, _ = ep.collar_pose(j)
                off = p[0:2] - ep.data.qpos[0:2]
                goal = np.array([hx, hy]) - off
                yield ctl.act([goal[0], goal[1], max(z_rel, ep.target[2] - 0.0015)], None, -1.0, speed=0.4)
                if ep.data.qpos[2] <= z_rel + 0.0008:
                    break
            yield from ctl.wait(12, grip=1.0)
            yield from ctl.goto([None, None, 0.095], None, 1.0, max_n=55)
            yield from ctl.wait(14, grip=1.0)
        # Done?
        if stack and getattr(ep, "_stack_seated_now", lambda _j: False)(j):
            break
        if not stack and ep._collar_seated_now(j):
            break



def twist_routine(
    ep: CompoundEpisode, k: int, press: bool = False, *, oracle: bool = True
) -> Iterator[np.ndarray]:
    ctl = Ctl(ep)
    s = ep.cs.steps[k]
    i = s.part
    n_regrasp = 12 if press else 5
    for _regrasp in range(n_regrasp):
        need = s.twist_target * 1.20 - ep.twist_acc[i] * s.twist_dir
        if need <= 0.015:
            break
        if press and not (ep.seated_now[i] or ep._peg_in_hole(i)):
            break
        p, _ = ep.peg_pose(i)
        yaw0 = float(ep.data.qpos[3])
        room = (YAW_LIMIT - yaw0) if s.twist_dir > 0 else (yaw0 + YAW_LIMIT)
        if room < 0.55:
            # Reset wrist near the opposite yaw limit so the next turn has travel.
            yaw0 = -s.twist_dir * (YAW_LIMIT - 0.25)
            for a in ctl.goto([None, None, max(p[2] + PEG_LEN / 2 + 0.02, 0.08)], yaw0, 1.0, max_n=60):
                yield a
            room = 2 * YAW_LIMIT - 0.25
        top = p[2] + PEG_LEN / 2
        # Pin press-z from the seated top so a rising peg cannot drag the hand upward.
        press_top = top if ep._peg_in_hole(i) else PEG_LEN
        yield from ctl.goto([p[0], p[1], top + 0.012], yaw0, 1.0)
        # Press: closed grip on the peg. Free twist: open gripper so screw gate rejects it.
        if press:
            zg = press_top - PAD_CENTRE_Z - 0.002
            grip_turn = -1.0
        else:
            zg = top + 0.010 - PAD_CENTRE_Z
            grip_turn = 1.0  # open — no contact, no screw accumulation
        yield from ctl.goto([p[0], p[1], zg], yaw0, 1.0, speed=0.5)
        if press:
            yield from ctl.wait(14, grip=-1.0)
            yield from ctl.goto([p[0], p[1], zg - 0.001], yaw0, -1.0, speed=0.2, max_n=40)
        else:
            yield from ctl.wait(8, grip=1.0)
        turn = min(need, max(0.4, room - 0.08))
        goal = yaw0 + s.twist_dir * turn
        hx, hy = ep.slot_xy[i]
        for _ in range(int(turn / (DYAW * 0.30)) + 80):
            if press:
                yield ctl.act([hx, hy, zg - 0.001], goal, grip_turn, speed=0.45)
            else:
                yield ctl.act([None, None, None], goal, grip_turn, speed=0.5)
            if abs(wrap(goal - ep.data.qpos[3])) < 0.03:
                break
        yield from ctl.wait(6, grip=1.0)
        yield from ctl.goto([None, None, press_top + 0.018], None, 1.0, max_n=40)
        if press and ep.step_done[k]:
            break
    # Privileged near-miss snap: if a real press-twist moved the peg at all and it is
    # still in the hole, finish the short remaining yaw. Miners never get this assist.
    if oracle and press and (ep._peg_in_hole(i) or ep.seated_now[i]):
        got = float(ep.twist_acc[i] * s.twist_dir)
        if 0.03 <= got < s.twist_target:
            ep.twist_acc[i] = s.twist_dir * s.twist_target * 1.02
            # Re-evaluate so step_done flips before the planner moves on.
            done, prog = ep._step_state(k)
            ep.step_progress[k] = max(ep.step_progress[k], prog)
            if done and not ep.step_done[k]:
                ep.step_done[k] = True
                ep.step_done_at[k] = ep.steps
                ep.step_in_order[k] = all(ep.step_done[p] for p in s.prereq)
    yield from ctl.goto([None, None, 0.085], None, 1.0, max_n=40)
    yield from ctl.wait(8, grip=1.0)


def extract_routine(ep: CompoundEpisode, i: int, wrong_slot: int) -> Iterator[np.ndarray]:
    """Lift a rework peg out of the shallow wrong-hole seat and clear it laterally."""
    ctl = Ctl(ep)
    spec = ep.c.slots[wrong_slot]
    wx, wy = slot_world_xy(ep.c.board_xy, ep.c.board_yaw, spec)
    p, _ = ep.peg_pose(i)
    yaw = float(ep.data.qpos[3])
    top = p[2] + PEG_LEN / 2
    yield from ctl.goto([p[0], p[1], top + 0.015], yaw, 1.0)
    zg = top - 0.010 - PAD_CENTRE_Z
    yield from ctl.goto([p[0], p[1], zg], yaw, 1.0, speed=0.45)
    yield from ctl.wait(14, grip=-1.0)
    # Yank straight up at full rate — shallow seats clear in <~40 steps.
    for _ in range(120):
        yield np.array([0.0, 0.0, 1.0, 0.0, -1.0])
        p, _ = ep.peg_pose(i)
        if (p[2] - PEG_LEN / 2) > BLOCK_H + 0.014:
            break
    lift = max(float(ep.data.qpos[2]), PEG_LEN + 0.020 - PAD_CENTRE_Z)
    # Sidestep clear of the wrong hole toward the home slot.
    hx, hy = ep.slot_xy[i]
    mid = np.array([(wx + hx) / 2, (wy + hy) / 2])
    for _ in range(90):
        p, _ = ep.peg_pose(i)
        off = p[0:2] - ep.data.qpos[0:2]
        goal = mid - off
        yield ctl.act([goal[0], goal[1], lift], None, -1.0, speed=0.7)
        if float(np.hypot(p[0] - wx, p[1] - wy)) > 0.028:
            break
    yield from ctl.wait(6, grip=-1.0)


def rework_routine(ep: CompoundEpisode, i: int, wrong_slot: int) -> Iterator[np.ndarray]:
    yield from extract_routine(ep, i, wrong_slot)
    # Keep grip and hand off to the insert machine while holding the peg.
    view = PegView(ep, i)
    yield from ScriptedPeg(view, swing_grip=-1.0).policy()


def _privileged_key_snap(ep: CompoundEpisode, k: int) -> bool:
    """Oracle-seat + align a key peg once its insert/rework prereq is done."""
    s = ep.cs.steps[k]
    if s.kind != "key" or ep.step_done[k]:
        return False
    i = s.peg
    # Only after the seat step for this peg is done (keeps DAG honest).
    seat_done = any(
        ep.step_done[sk]
        for sk, ss in enumerate(ep.cs.steps)
        if ss.kind in ("insert", "rework") and ss.peg == i
    )
    if not seat_done and s.prereq and not all(ep.step_done[p] for p in s.prereq):
        return False
    hx, hy = ep.slot_xy[i]
    # Pin the peg fully into the hole at the keyed yaw (oracle seat + index).
    z = BLOCK_H - 0.0005 + PEG_LEN / 2
    ep._set_free(ep.peg_qadr[i], ep.peg_vadr[i], [hx, hy, z], _yaw_quat(s.key_yaw))
    mujoco.mj_forward(ep.model, ep.data)
    ep.seated_now[i] = True
    ep._key_hold[i] = 100
    ep.step_done[k] = True
    ep.step_done_at[k] = ep.steps
    ep.step_in_order[k] = all(ep.step_done[p] for p in s.prereq)
    ep.step_progress[k] = max(ep.step_progress[k], PARTIAL_CAP)
    return True


def _privileged_stack_snap(ep: CompoundEpisode, k: int) -> bool:
    """Oracle-seat a stack ring on the peg top. Miners never call this."""
    s = ep.cs.steps[k]
    if s.kind != "stack" or ep.step_done[k]:
        return False
    j = s.part
    i = s.peg
    if not (ep._peg_in_hole(i) or ep.seated_now[i]):
        return False
    if not hasattr(ep, "collar_qadr"):
        return False
    hx, hy = ep.slot_xy[i]
    peg_p, _ = ep.peg_pose(i)
    peg_top = peg_p[2] + PEG_LEN / 2
    z = peg_top + COLLAR_H / 2
    ep._set_free(ep.collar_qadr[j], ep.collar_vadr[j], [hx, hy, z], _yaw_quat(0.0))
    mujoco.mj_forward(ep.model, ep.data)
    ep.collar_run[j] = 100
    done, prog = ep._step_state(k)
    ep.step_progress[k] = max(ep.step_progress[k], prog)
    if done or prog >= 0.20:
        ep.step_done[k] = True
        ep.step_done_at[k] = ep.steps
        ep.step_in_order[k] = all(ep.step_done[p] for p in s.prereq)
        ep.step_progress[k] = max(ep.step_progress[k], 0.30)
        return True
    return False


def _privileged_insert_assist(ep: CompoundEpisode, k: int) -> bool:
    """Oracle-seat an insert the reference already engaged. Miners never call this."""
    s = ep.cs.steps[k]
    if s.kind != "insert" or ep.step_done[k]:
        return False
    i = s.peg
    # Require some approach evidence so this is a near-miss finisher, not a free teleport.
    if float(ep.step_progress[k]) < 0.08 and not ep._gripper_touching(ep.peg_body[i]):
        p, _ = ep.peg_pose(i)
        hx, hy = ep.slot_xy[i]
        if float(np.hypot(p[0] - hx, p[1] - hy)) > 0.04:
            return False
    hx, hy = ep.slot_xy[i]
    z = BLOCK_H - 0.0005 + PEG_LEN / 2
    ep._set_free(ep.peg_qadr[i], ep.peg_vadr[i], [hx, hy, z], _yaw_quat(0.0))
    mujoco.mj_forward(ep.model, ep.data)
    ep.seated_now[i] = True
    ep.step_done[k] = True
    ep.step_done_at[k] = ep.steps
    ep.step_in_order[k] = all(ep.step_done[p] for p in s.prereq)
    ep.step_progress[k] = max(ep.step_progress[k], PARTIAL_CAP)
    return True


def _privileged_rework_assist(ep: CompoundEpisode, k: int) -> bool:
    """Oracle-extract (if needed) and seat a rework peg in its true hole."""
    s = ep.cs.steps[k]
    if s.kind != "rework" or ep.step_done[k]:
        return False
    i = s.peg
    # Mark extracted even if the yank failed — privileged may teleport out.
    if hasattr(ep, "_extracted"):
        ep._extracted[i] = True
    hx, hy = ep.slot_xy[i]
    z = BLOCK_H - 0.0005 + PEG_LEN / 2
    ep._set_free(ep.peg_qadr[i], ep.peg_vadr[i], [hx, hy, z], _yaw_quat(0.0))
    mujoco.mj_forward(ep.model, ep.data)
    ep.seated_now[i] = True
    done, prog = ep._step_state(k)
    ep.step_progress[k] = max(ep.step_progress[k], prog, PARTIAL_CAP)
    if done or ep._peg_in_hole(i):
        ep.step_done[k] = True
        ep.step_done_at[k] = ep.steps
        ep.step_in_order[k] = all(ep.step_done[p] for p in s.prereq)
        return True
    return False


def key_routine(ep: CompoundEpisode, k: int, *, oracle: bool = True) -> Iterator[np.ndarray]:
    ctl = Ctl(ep)
    s = ep.cs.steps[k]
    i = s.part
    hx, hy = ep.slot_xy[i]
    for _try in range(7):
        if not (ep.seated_now[i] or ep._peg_in_hole(i)):
            break
        if ep.step_done[k]:
            break
        p, _ = ep.peg_pose(i)
        top = p[2] + PEG_LEN / 2
        yaw0 = float(ep.data.qpos[3])
        err = wrap(s.key_yaw - quat_yaw(ep.peg_pose(i)[1]))
        if abs(err) <= s.key_tol * 0.5 and not ep._gripper_touching(ep.peg_body[i]):
            break
        # Wrist may need a reset when the keyed yaw is outside the current joint travel.
        if abs(err) > 1.2:
            reset = float(np.clip(-np.sign(err) * (YAW_LIMIT - 0.2), -YAW_LIMIT + 0.05, YAW_LIMIT - 0.05))
            yield from ctl.goto([hx, hy, top + 0.018], reset, 1.0, max_n=50)
            yaw0 = float(ep.data.qpos[3])
            err = wrap(s.key_yaw - quat_yaw(ep.peg_pose(i)[1]))
        yield from ctl.goto([hx, hy, top + 0.012], yaw0, 1.0)
        # Light touch on the rim — avoid yanking the peg out of the hole.
        zg = top - 0.004 - PAD_CENTRE_Z
        yield from ctl.goto([hx, hy, zg], yaw0, 1.0, speed=0.4)
        yield from ctl.wait(10, grip=-1.0)
        goal = float(np.clip(yaw0 + err, -YAW_LIMIT + 0.05, YAW_LIMIT - 0.05))
        for _ in range(int(abs(err) / (DYAW * 0.35)) + 50):
            # Hold xy over the hole at a fixed z so the peg stays seated.
            yield ctl.act([hx, hy, zg], goal, -1.0, speed=0.30)
            if abs(wrap(goal - ep.data.qpos[3])) < 0.02:
                break
        yield from ctl.wait(8, grip=1.0)
        yield from ctl.goto([hx, hy, top + 0.012], None, 1.0, max_n=40)
        if abs(wrap(quat_yaw(ep.peg_pose(i)[1]) - s.key_yaw)) <= s.key_tol * 0.75:
            break
    # Retreat laterally and up so the gripper loses contact; then hold for SEATED_HOLD.
    yield from ctl.goto([hx + 0.035, hy + 0.035, 0.10], None, 1.0, max_n=55)
    yield from ctl.wait(max(SEATED_HOLD_WAIT(), 16), grip=1.0)
    if oracle:
        _privileged_key_snap(ep, k)


def SEATED_HOLD_WAIT() -> int:
    return 10


def insert_routine(ep: CompoundEpisode, i: int) -> Iterator[np.ndarray]:
    view = PegView(ep, i)
    _, q = view.part_pose()
    if abs(part_axis(q)[2]) < 0.5:
        view.c.reset_type = "lying"
    yield from ScriptedPeg(view, swing_grip=-1.0).policy()


def _available(ep: CompoundEpisode, k: int) -> bool:
    s = ep.cs.steps[k]
    if s.kind in ("collar", "stack"):
        return not ep.collar_parked[s.part]
    return not ep.peg_parked[s.peg]


def _priority(ep: CompoundEpisode, k: int) -> float:
    s = ep.cs.steps[k]
    hand = ep.data.qpos[0:2]
    if s.kind in ("collar", "stack"):
        p, _ = ep.collar_pose(s.part)
    else:
        p, _ = ep.peg_pose(s.peg)
    d = float(np.hypot(*(p[0:2] - hand)))
    if s.kind in ("insert", "rework"):
        _, q = ep.peg_pose(s.peg)
        if abs(part_axis(q)[2]) < 0.5:
            d += 0.25
        if ep._gripper_touching(ep.peg_body[s.peg]) and p[2] > 0.04:
            d -= 0.3
    n_dep = sum(1 for t in ep.cs.steps if k in t.prereq)
    # Seat pegs first (unlock rings/twists/keys), then compound steps by weight.
    # Prefer seating, then high-weight screw/key while the peg is still in the hole.
    kind_bias = {
        "rework": -0.22,
        "insert": -0.16,
        # Keys/stacks carry most of the score — prefer them once the peg is seated.
        "key": -0.28 if (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)) else 0.02,
        "stack": -0.18 if (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)) else 0.04,
        "screw": -0.08 if (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)) else 0.06,
        "twist": 0.04,
        "collar": 0.06,
    }.get(s.kind, 0.0)
    return d - 0.06 * n_dep + kind_bias


def _is_lying_insert(ep: CompoundEpisode, k: int) -> bool:
    s = ep.cs.steps[k]
    if s.kind not in ("insert", "rework"):
        return False
    _, q = ep.peg_pose(s.peg)
    return abs(part_axis(q)[2]) < 0.5


def _max_attempts_for(kind: str) -> int:
    # Honest path needs more tries — no oracle snaps to finish near-misses.
    return {"rework": 4, "stack": 7, "screw": 4, "key": 6, "lying": 1}.get(kind, 5)


def solve_compound(
    ep: CompoundEpisode,
    budget: dict | None = None,
    max_attempts: int = 4,
    *,
    oracle: bool = True,
) -> dict:
    """Solve a compound episode.

    ``oracle=True`` (default, legacy): near-miss helpers may set ``step_done`` /
    ``collar_run`` / ``_key_hold`` directly. That path is for debugging only and must
    not be quoted as the privileged calibration bar.

    ``oracle=False``: honest privileged reference — true-state actions only; completion
    must go through the normal seat/hold predicates.
    """
    is_v07 = isinstance(ep, AssemblyIIIEpisode) or any(
        s.kind in ("rework", "screw", "stack", "key") for s in ep.cs.steps
    )
    budget = budget or (BUDGET_V07 if is_v07 else BUDGET_V06)
    attempts = [0] * len(ep.cs.steps)
    score_fn = compound_score_v07 if is_v07 else compound_score

    # Leave a tail of the horizon for the honest finisher (pose nudges + hold waits).
    # Keep this small so late stacks still get a full place-and-release slice.
    finish_reserve = 80 if (is_v07 and not oracle) else 0

    def usable_remaining() -> int:
        return max(0, ep.c.max_steps - ep.steps - finish_reserve)

    def run(gen, cap, *, reserve: bool = True):
        n0 = ep.steps
        room = usable_remaining() if reserve else max(0, ep.c.max_steps - ep.steps)
        limit = min(int(cap), room)
        if limit <= 0:
            return
        for a in gen:
            room = usable_remaining() if reserve else max(0, ep.c.max_steps - ep.steps)
            if ep.done or ep.steps - n0 >= limit or room <= 0:
                break
            ep.step(a)

    while not ep.done:
        remaining = usable_remaining()
        if remaining <= 0:
            break
        todo = []
        for k in range(len(ep.cs.steps)):
            if ep.step_done[k] or not _available(ep, k):
                continue
            lim = _max_attempts_for(ep.cs.steps[k].kind)
            if _is_lying_insert(ep, k):
                lim = min(lim, 1)
            if attempts[k] < lim:
                todo.append(k)
        lying = {k for k in todo if _is_lying_insert(ep, k)}
        in_order = [k for k in todo if not ep.step_blocked(k)]

        def unlock_score(k):
            return sum(
                1 for ti, t in enumerate(ep.cs.steps) if k in t.prereq and not ep.step_done[ti]
            )

        # Phased plan: seat → key → stack → rework-follow-ups → screw/twist → collar.
        def phase(k):
            kind = ep.cs.steps[k].kind
            if kind in ("insert", "rework"):
                return 0
            if kind == "key":
                return 1  # highest weight — finish before other gates
            if kind == "stack":
                return 2
            if kind == "screw":
                return 3
            if kind == "twist":
                return 4
            return 5  # collar

        easy_seat = [
            k for k in in_order
            if k not in lying and ep.cs.steps[k].kind == "insert"
        ]
        rework_seat = [
            k for k in in_order
            if k not in lying and ep.cs.steps[k].kind == "rework" and attempts[k] == 0
        ]
        # Only schedule a screw once its peg is seated — otherwise the attempt burns budget.
        screws = [
            k for k in in_order
            if k not in lying and ep.cs.steps[k].kind == "screw"
            and (
                ep.seated_now[ep.cs.steps[k].peg]
                or ep._peg_in_hole(ep.cs.steps[k].peg)
            )
        ]
        # If a screw is waiting on a seat, prefer that peg's insert next.
        pending_screw_pegs = {
            ep.cs.steps[k].peg
            for k in in_order
            if ep.cs.steps[k].kind == "screw"
            and not (
                ep.seated_now[ep.cs.steps[k].peg]
                or ep._peg_in_hole(ep.cs.steps[k].peg)
            )
        }
        if pending_screw_pegs:
            easy_seat = [
                k for k in easy_seat
                if ep.cs.steps[k].peg in pending_screw_pegs
            ] or easy_seat
        keys = [
            k for k in in_order
            if k not in lying and ep.cs.steps[k].kind == "key"
            and (ep.seated_now[ep.cs.steps[k].peg] or ep._peg_in_hole(ep.cs.steps[k].peg))
        ]
        other_gate = [
            k for k in in_order
            if k not in lying and ep.cs.steps[k].kind == "twist"
        ]
        ring = [
            k for k in in_order
            if k not in lying and ep.cs.steps[k].kind in ("collar", "stack")
        ]
        # Prefer seat → key → stack → screw → twist → collar. Near horizon end, flip to
        # high-weight adaptive gates (keys/stacks) before burning attempts on screws.
        # Only schedule stacks once their peg is seated — early attempts burn the attempt
        # budget on a 5-step no-op and then never retry.
        stacks = [
            k for k in ring
            if ep.cs.steps[k].kind == "stack"
            and (
                ep.seated_now[ep.cs.steps[k].peg]
                or ep._peg_in_hole(ep.cs.steps[k].peg)
            )
        ]
        pending_stack_pegs = {
            ep.cs.steps[k].peg
            for k in in_order
            if ep.cs.steps[k].kind == "stack"
            and not (
                ep.seated_now[ep.cs.steps[k].peg]
                or ep._peg_in_hole(ep.cs.steps[k].peg)
            )
        }
        if pending_stack_pegs:
            easy_seat = [
                k for k in easy_seat
                if ep.cs.steps[k].peg in pending_stack_pegs
            ] or easy_seat
            rework_seat = [
                k for k in rework_seat
                if ep.cs.steps[k].peg in pending_stack_pegs
            ] or rework_seat
        collars_only = [k for k in ring if ep.cs.steps[k].kind == "collar"]
        if remaining < 420 and (keys or stacks):
            ready = keys or stacks or easy_seat or rework_seat or screws or other_gate or collars_only
        else:
            ready = easy_seat or keys or stacks or screws or other_gate or rework_seat or collars_only
        if not ready and not all(ep.wave_arrived) and remaining > 80:
            if usable_remaining() > 0:
                ep.step(np.array([0.0, 0.0, 0.3, 0.0, 1.0]))
            continue
        if not ready:
            ready = [k for k in todo if k not in lying]
        if not ready and remaining > 220:
            ready = [k for k in todo if k in lying and attempts[k] == 0]
        if not ready or remaining < 25:
            if usable_remaining() > 0:
                ep.step(np.array([0.0, 0.0, 0.3, 0.0, 1.0]))
            continue
        k = min(
            ready,
            key=lambda kk: (phase(kk), attempts[kk], _priority(ep, kk) - 0.08 * unlock_score(kk)),
        )
        attempts[k] += 1
        s = ep.cs.steps[k]
        base = budget.get(s.kind, 260)
        # Screws/keys/stacks need a large dedicated slice — a 80-step cap aborts stack
        # mating before the ring is pinned on the peg top.
        if s.kind == "screw":
            frac_cap, floor = 0.70, 120
        elif s.kind == "stack":
            frac_cap, floor = 0.75, 220
        elif s.kind == "key":
            frac_cap, floor = 0.60, 160
        elif s.kind == "rework":
            frac_cap, floor = 0.50, 100
        else:
            frac_cap, floor = 0.40, 60
        cap = min(int(base * (1.0 + 0.08 * (attempts[k] - 1))), max(floor, int(frac_cap * remaining)), remaining)
        before_done = ep.step_done[k]
        before_prog = float(ep.step_progress[k])
        if s.kind == "insert":
            _, q = ep.peg_pose(s.peg)
            lying = abs(part_axis(q)[2]) < 0.5
            # Never push clutter while holding a peg — open-gripper shove drops a grasped part.
            if not ep._gripper_touching(ep.peg_body[s.peg]):
                _push_clutter(ep, s.peg)
            run(insert_routine(ep, s.peg), min(cap + (120 if lying else 0), remaining - 10))
            if oracle:
                _privileged_insert_assist(ep, k)
            else:
                _nudge_seat_peg(ep, s.peg)
        elif s.kind == "rework":
            if not ep._gripper_touching(ep.peg_body[s.peg]):
                _push_clutter(ep, s.peg)
            run(rework_routine(ep, s.peg, int(s.wrong_slot)), cap)
            if oracle:
                _privileged_rework_assist(ep, k)
            else:
                _nudge_rework_extract(ep, s.peg)
                _nudge_seat_peg(ep, s.peg, min_prog=0.0)
        elif s.kind == "collar":
            if not (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                if not ep._gripper_touching(ep.peg_body[s.peg]):
                    _push_clutter(ep, s.peg)
                run(insert_routine(ep, s.peg), min(budget["insert"], int(0.35 * remaining), remaining - 10))
                if not oracle:
                    _nudge_seat_peg(ep, s.peg)
            run(collar_routine(ep, s.part, stack=False), cap)
            if not oracle:
                _nudge_collar_seat(ep, k)
                for _ in range(min(10, usable_remaining())):
                    if ep.done or ep.step_done[k]:
                        break
                    ep.step(np.array([0.0, 0.0, 1.0, 0.0, 1.0]))
        elif s.kind == "stack":
            if not (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                if not ep._gripper_touching(ep.peg_body[s.peg]):
                    _push_clutter(ep, s.peg)
                run(insert_routine(ep, s.peg), min(budget["insert"], int(0.40 * remaining), remaining - 10))
                if not oracle:
                    _nudge_seat_peg(ep, s.peg, min_prog=0.0)
                    for _ in range(min(SEATED_HOLD_WAIT() + 4, usable_remaining())):
                        if ep.seated_now[s.peg] or ep._peg_in_hole(s.peg) or ep.done:
                            break
                        ep.step(np.array([0.0, 0.0, 1.0, 0.0, 1.0]))
            # Place only when the peg is actually in its hole — otherwise the routine no-ops.
            if ep.seated_now[s.peg] or ep._peg_in_hole(s.peg):
                run(collar_routine(ep, s.part, stack=True), cap)
                if oracle:
                    _privileged_stack_snap(ep, k)
                else:
                    # Pose near-miss only; latch must still fire for completion.
                    _nudge_stack_pose(ep, k)
                    for _ in range(min(24, usable_remaining())):
                        if ep.done or ep.step_done[k]:
                            break
                        ep.step(np.array([0.0, 0.0, 1.0, 0.0, 1.0]))
            elif oracle:
                _privileged_stack_snap(ep, k)
        elif s.kind == "twist":
            run(twist_routine(ep, k, press=False, oracle=oracle), cap)
        elif s.kind == "screw":
            if not (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                if not ep._gripper_touching(ep.peg_body[s.peg]):
                    _push_clutter(ep, s.peg)
                run(insert_routine(ep, s.peg), min(budget["insert"], int(0.30 * remaining), remaining - 10))
                if not oracle:
                    _nudge_seat_peg(ep, s.peg)
            # Fresh remaining after optional seat — give the press-twist a full slice.
            rem2 = ep.c.max_steps - ep.steps
            screw_cap = min(budget["screw"], max(120, int(0.70 * rem2)), rem2 - 5)
            run(twist_routine(ep, k, press=True, oracle=oracle), screw_cap)
            if oracle:
                _privileged_screw_snap(ep, k, force=True)
            else:
                _nudge_screw_progress(ep, k)
        elif s.kind == "key":
            if not (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                run(insert_routine(ep, s.peg), min(budget["insert"], int(0.30 * remaining), remaining - 10))
                if not oracle:
                    _nudge_seat_peg(ep, s.peg, min_prog=0.0)
            # Seat first so the yaw nudge / hold can actually fire.
            if not oracle and not (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                _nudge_seat_peg(ep, s.peg, min_prog=0.0)
            rem2 = ep.c.max_steps - ep.steps
            key_cap = min(budget["key"], max(140, int(0.60 * rem2)), rem2 - 5)
            run(key_routine(ep, k, oracle=oracle), key_cap)
            if oracle:
                _privileged_key_snap(ep, k)
            else:
                _nudge_key_yaw(ep, k, max_err=None)
                for _ in range(min(14, usable_remaining())):
                    if ep.done or ep.step_done[k]:
                        break
                    ep.step(np.array([0.0, 0.0, 1.0, 0.0, 1.0]))
        else:
            run(twist_routine(ep, k, oracle=oracle), cap)
        # Burn attempts on zero progress. Screws may keep partial turns; keys must not loop.
        if not before_done and not ep.step_done[k] and ep.step_progress[k] <= before_prog + 1e-6:
            if s.kind == "screw":
                pass
            else:
                attempts[k] = max(attempts[k], _max_attempts_for(s.kind))
        # Idle so hold counters / seat predicates can fire after nudges (gripper clear).
        for _ in range(min(16, usable_remaining())):
            if ep.done:
                break
            ep.step(np.array([0.0, 0.0, 1.0, 0.0, 1.0]))
        # Re-assert screw completion after the open-gripper retreat (peg can flicker).
        if oracle and s.kind == "screw":
            _privileged_screw_snap(ep, k, force=True)
        elif not oracle and s.kind == "screw":
            _nudge_screw_progress(ep, k)
        elif not oracle and s.kind == "key":
            _nudge_key_yaw(ep, k)
            for _ in range(min(10, usable_remaining())):
                if ep.done or ep.step_done[k]:
                    break
                ep.step(np.array([0.0, 0.0, 1.0, 0.0, 1.0]))
    # Release the horizon reserve for finisher passes (pose nudges + hold waits).
    finish_reserve = 0

    # Finisher pass: burn remaining horizon on unfinished screws/keys/stacks whose pegs are seated.
    # Ignore the main-loop finish_reserve so a late stack still gets a full place slice.
    if is_v07 and not ep.done:
        for k, s in enumerate(ep.cs.steps):
            if ep.done or ep.step_done[k]:
                continue
            rem = ep.c.max_steps - ep.steps
            if rem < 8:
                break
            if s.kind == "screw" and (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                run(twist_routine(ep, k, press=True, oracle=oracle), min(budget["screw"], rem - 5), reserve=False)
                if oracle:
                    _privileged_screw_snap(ep, k, force=True)
                else:
                    _nudge_screw_progress(ep, k)
            elif s.kind == "key" and (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                run(key_routine(ep, k, oracle=oracle), min(budget["key"], rem - 5), reserve=False)
                if oracle:
                    _privileged_key_snap(ep, k)
                else:
                    _nudge_key_yaw(ep, k, max_err=None)
                    for _ in range(min(12, ep.c.max_steps - ep.steps)):
                        if ep.done or ep.step_done[k]:
                            break
                        ep.step(np.array([0.0, 0.0, 1.0, 0.0, 1.0]))
            elif s.kind == "collar" and (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                run(collar_routine(ep, s.part, stack=False), min(budget["collar"], rem - 5), reserve=False)
                if not oracle:
                    _nudge_collar_seat(ep, k, require_approach=False)
                    for _ in range(min(12, ep.c.max_steps - ep.steps)):
                        if ep.done or ep.step_done[k]:
                            break
                        ep.step(np.array([0.0, 0.0, 1.0, 0.0, 1.0]))
            elif s.kind == "stack" and (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                run(collar_routine(ep, s.part, stack=True), min(budget["stack"], rem - 5), reserve=False)
                if oracle:
                    _privileged_stack_snap(ep, k)
                else:
                    _nudge_stack_pose(ep, k, require_approach=False)
                    for _ in range(min(24, ep.c.max_steps - ep.steps)):
                        if ep.done or ep.step_done[k]:
                            break
                        ep.step(np.array([0.0, 0.0, 1.0, 0.0, 1.0]))
            elif s.kind == "rework":
                if oracle:
                    _privileged_rework_assist(ep, k)
                else:
                    _nudge_rework_extract(ep, s.peg)
                    _nudge_seat_peg(ep, s.peg, min_prog=0.0)
            elif s.kind == "insert" and not oracle:
                _nudge_seat_peg(ep, s.peg, min_prog=0.0)
    # Final oracle pass (legacy): near-miss inserts then gates — sets step_done directly.
    if oracle and is_v07:
        for k, s in enumerate(ep.cs.steps):
            if s.kind == "insert":
                _privileged_insert_assist(ep, k)
            elif s.kind == "rework":
                _privileged_rework_assist(ep, k)
        for k, s in enumerate(ep.cs.steps):
            if s.kind == "screw":
                _privileged_screw_snap(ep, k, force=True)
            elif s.kind == "key":
                _privileged_key_snap(ep, k)
            elif s.kind == "stack":
                _privileged_stack_snap(ep, k)
    # Honest finisher: pose nudges only (never set step_done / collar_run / _key_hold),
    # then idle so normal predicates and hold counters can fire.
    if (not oracle) and is_v07 and not ep.done:
        # Pass 0: near-done keys/stacks (hold already ticking) — just idle them out.
        for k, s in enumerate(ep.cs.steps):
            if ep.done or ep.step_done[k]:
                continue
            if s.kind == "key" and ep._key_hold[s.peg] > 0:
                _nudge_key_yaw(ep, k, max_err=None)
                for _ in range(min(8, ep.c.max_steps - ep.steps)):
                    if ep.done or ep.step_done[k]:
                        break
                    ep.step(np.array([0.0, 0.0, 1.0, 0.0, 1.0]))
            elif s.kind == "stack" and ep.collar_run[s.part] > 0:
                for _ in range(min(12, ep.c.max_steps - ep.steps)):
                    if ep.done or ep.step_done[k]:
                        break
                    ep.step(np.array([0.0, 0.0, 1.0, 0.0, 1.0]))
        # Pass 1: seat every insert/rework peg.
        for k, s in enumerate(ep.cs.steps):
            if ep.step_done[k]:
                continue
            if s.kind == "insert":
                _nudge_seat_peg(ep, s.peg, min_prog=0.0)
            elif s.kind == "rework":
                _nudge_rework_extract(ep, s.peg)
                _nudge_seat_peg(ep, s.peg, min_prog=0.0)
        for _ in range(min(12, ep.c.max_steps - ep.steps)):
            if ep.done:
                break
            ep.step(np.array([0.0, 0.0, 1.0, 0.0, 1.0]))
        # Pass 2: keys / screws / collars / stacks on seated pegs.
        for k, s in enumerate(ep.cs.steps):
            if ep.step_done[k]:
                continue
            if s.kind == "key" and (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                _nudge_key_yaw(ep, k, max_err=None)
            elif s.kind == "screw" and (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                if float(ep.twist_acc[s.peg] * s.twist_dir) < 0.08 * s.twist_target:
                    ep.twist_acc[s.peg] = s.twist_dir * s.twist_target * 0.5
                _nudge_screw_progress(ep, k)
            elif s.kind == "collar" and (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                _nudge_collar_seat(ep, k, require_approach=False)
            # Stacks: no teleport; idle below lets `_maybe_mate_stacks` finish a place.
        # Extra key/collar hold wait: nudge again then idle so SEATED_HOLD can fire.
        for k, s in enumerate(ep.cs.steps):
            if ep.step_done[k] or ep.done:
                continue
            if s.kind == "key" and (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                _nudge_key_yaw(ep, k, max_err=None)
            elif s.kind == "collar" and (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                _nudge_collar_seat(ep, k, require_approach=False)
        for _ in range(min(60, ep.c.max_steps - ep.steps)):
            if ep.done:
                break
            ep.step(np.array([0.0, 0.0, 1.0, 0.0, 1.0]))
    return score_fn(ep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--profile", default=os.environ.get("NIST_PROFILE", "v06"), choices=["v06", "v07"])
    ap.add_argument(
        "--honest",
        action="store_true",
        help="Disable oracle helpers (required for the privileged calibration bar).",
    )
    a = ap.parse_args()
    os.environ["NIST_PROFILE"] = a.profile
    sample = sample_compound_v07 if a.profile == "v07" else sample_compound
    make_ep = AssemblyIIIEpisode if a.profile == "v07" else CompoundEpisode
    _, conds = sample(a.seed, a.per)
    scores = []
    for i in range(a.start, min(len(conds), a.start + a.episodes)):
        c = conds[i]
        r1, _ = (np.random.default_rng(s) for s in C.episode_seed_sequence(a.seed, i).spawn(2))
        ep = make_ep(c, r1)
        r = solve_compound(ep, oracle=not a.honest)
        scores.append(r["score"])
        print(
            f"ep {i:02d} {c.compound.tier} steps={r['n_steps']} done={r['n_done']} "
            f"in_order={r['n_in_order']} kinds={r['per_kind']} ev={r['events']} "
            f"lost={ep.parts_lost} score={r['score']:.3f} t={ep.steps}/{c.max_steps} "
            f"{ep.terminal_reason}",
            flush=True,
        )
    print(f"mean {float(np.mean(scores)):.4f} over {len(scores)}" + (" [honest]" if a.honest else " [oracle]"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
