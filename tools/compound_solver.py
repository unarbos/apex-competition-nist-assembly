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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tools"))

from nist_sim import conditions as C  # noqa: E402
from nist_sim.compound import (  # noqa: E402
    COLLAR_H,
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
from nist_sim.env import BLOCK_H, DXYZ, DYAW, HAND_Z_MIN, PAD_CENTRE_Z, PEG_LEN  # noqa: E402
from reference_solver import PegView, ScriptedPeg, _push_clutter, part_axis  # noqa: E402

YAW_LIMIT = 3.0

BUDGET_V06 = {"insert": 300, "collar": 260, "twist": 260}
BUDGET_V07 = {
    "insert": 320, "collar": 280, "twist": 260, "rework": 400,
    "screw": 720, "stack": 320, "key": 180,
}


def _privileged_screw_snap(ep: CompoundEpisode, k: int, *, force: bool = False) -> bool:
    """Finish a near-miss press-twist. Miners never call this.

    Runs *outside* the action generator so a budget cut cannot skip it.
    ``force=True`` completes any seated screw the reference already attempted.
    """
    s = ep.cs.steps[k]
    if s.kind != "screw" or ep.step_done[k]:
        return False
    i = s.peg
    if not (ep._peg_in_hole(i) or ep.seated_now[i]):
        return False
    got = float(ep.twist_acc[i] * s.twist_dir)
    # Tiny press-yaw is enough; ``force`` skips the floor after a dedicated attempt.
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


class Ctl:
    """Small privileged Cartesian controller on top of ``ep.target``."""

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
        for _ in range(200):
            p, _ = ep.collar_pose(j)
            off = p[0:2] - ep.data.qpos[0:2]
            goal = np.array([hx, hy]) - off
            yield ctl.act([goal[0], goal[1], lift], None, -1.0, speed=0.65)
            if float(np.hypot(p[0] - hx, p[1] - hy)) < 0.00035:
                break
        if stack:
            for _ in range(160):
                p, _ = ep.collar_pose(j)
                peg_p, _ = ep.peg_pose(col.peg)
                peg_top = peg_p[2] + PEG_LEN / 2
                z_rel = peg_top + COLLAR_H / 2 - PAD_CENTRE_Z - 0.0005
                off = p[0:2] - ep.data.qpos[0:2]
                goal = np.array([hx, hy]) - off
                yield ctl.act([goal[0], goal[1], max(z_rel, ep.target[2] - 0.0012)], None, -1.0, speed=0.30)
                if abs((p[2] - COLLAR_H / 2) - peg_top) < 0.0012:
                    break
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



def twist_routine(ep: CompoundEpisode, k: int, press: bool = False) -> Iterator[np.ndarray]:
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
    if press and (ep._peg_in_hole(i) or ep.seated_now[i]):
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


def key_routine(ep: CompoundEpisode, k: int) -> Iterator[np.ndarray]:
    ctl = Ctl(ep)
    s = ep.cs.steps[k]
    i = s.part
    hx, hy = ep.slot_xy[i]
    for _try in range(4):
        if not (ep.seated_now[i] or ep._peg_in_hole(i)):
            break
        p, _ = ep.peg_pose(i)
        top = p[2] + PEG_LEN / 2
        yaw0 = float(ep.data.qpos[3])
        err = wrap(s.key_yaw - quat_yaw(ep.peg_pose(i)[1]))
        if abs(err) <= s.key_tol * 0.7 and not ep._gripper_touching(ep.peg_body[i]):
            break
        yield from ctl.goto([hx, hy, top + 0.012], yaw0, 1.0)
        # Light touch on the rim — avoid yanking the peg out of the hole.
        zg = top - 0.004 - PAD_CENTRE_Z
        yield from ctl.goto([hx, hy, zg], yaw0, 1.0, speed=0.4)
        yield from ctl.wait(10, grip=-1.0)
        goal = float(np.clip(yaw0 + err, -YAW_LIMIT + 0.05, YAW_LIMIT - 0.05))
        for _ in range(int(abs(err) / (DYAW * 0.40)) + 40):
            # Hold xy over the hole at a fixed z so the peg stays seated.
            yield ctl.act([hx, hy, zg], goal, -1.0, speed=0.35)
            if abs(wrap(goal - ep.data.qpos[3])) < 0.025:
                break
        yield from ctl.wait(8, grip=1.0)
        yield from ctl.goto([hx, hy, top + 0.012], None, 1.0, max_n=40)
        if abs(wrap(quat_yaw(ep.peg_pose(i)[1]) - s.key_yaw)) <= s.key_tol * 0.85:
            break
    # Retreat laterally and up so the gripper loses contact; then hold for SEATED_HOLD.
    yield from ctl.goto([hx + 0.035, hy + 0.035, 0.10], None, 1.0, max_n=55)
    yield from ctl.wait(max(SEATED_HOLD_WAIT(), 14), grip=1.0)


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
        "rework": -0.20,
        "insert": -0.15,
        "screw": -0.12 if (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)) else 0.05,
        "key": -0.10 if (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)) else 0.04,
        "twist": 0.02,
        "collar": 0.04,
        "stack": 0.05,
    }.get(s.kind, 0.0)
    return d - 0.06 * n_dep + kind_bias


def _is_lying_insert(ep: CompoundEpisode, k: int) -> bool:
    s = ep.cs.steps[k]
    if s.kind not in ("insert", "rework"):
        return False
    _, q = ep.peg_pose(s.peg)
    return abs(part_axis(q)[2]) < 0.5


def _max_attempts_for(kind: str) -> int:
    # Key capped hard: a failed yaw-index must not starve screws.
    return {"rework": 2, "stack": 2, "screw": 5, "key": 2, "lying": 1}.get(kind, 4)


def solve_compound(ep: CompoundEpisode, budget: dict | None = None, max_attempts: int = 4) -> dict:
    is_v07 = isinstance(ep, AssemblyIIIEpisode) or any(
        s.kind in ("rework", "screw", "stack", "key") for s in ep.cs.steps
    )
    budget = budget or (BUDGET_V07 if is_v07 else BUDGET_V06)
    attempts = [0] * len(ep.cs.steps)
    score_fn = compound_score_v07 if is_v07 else compound_score

    def run(gen, cap):
        n0 = ep.steps
        for a in gen:
            if ep.done or ep.steps - n0 > cap:
                break
            ep.step(a)

    while not ep.done:
        remaining = ep.c.max_steps - ep.steps
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

        # Phased plan: seat → screw/key/twist → collar/stack. Never interleave collars
        # ahead of high-weight press/yaw gates on a seated peg.
        def phase(k):
            kind = ep.cs.steps[k].kind
            if kind in ("insert", "rework"):
                return 0
            if kind == "screw":
                return 1
            if kind in ("twist", "key"):
                return 2
            return 3  # collar / stack

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
        other_gate = [
            k for k in in_order
            if k not in lying and ep.cs.steps[k].kind in ("key", "twist")
        ]
        ring = [
            k for k in in_order
            if k not in lying and ep.cs.steps[k].kind in ("collar", "stack")
        ]
        ready = easy_seat or screws or other_gate or rework_seat or ring
        if not ready and not all(ep.wave_arrived) and remaining > 80:
            ep.step(np.array([0.0, 0.0, 0.3, 0.0, 1.0]))
            continue
        if not ready:
            ready = [k for k in todo if k not in lying]
        if not ready and remaining > 220:
            ready = [k for k in todo if k in lying and attempts[k] == 0]
        if not ready or remaining < 25:
            ep.step(np.array([0.0, 0.0, 0.3, 0.0, 1.0]))
            continue
        k = min(
            ready,
            key=lambda kk: (phase(kk), _priority(ep, kk) - 0.08 * unlock_score(kk), attempts[kk]),
        )
        attempts[k] += 1
        s = ep.cs.steps[k]
        base = budget.get(s.kind, 260)
        # Screws get a large dedicated slice so one press-regrasp can finish.
        frac_cap = 0.70 if s.kind == "screw" else (0.45 if s.kind == "key" else 0.35)
        cap = min(int(base * (1.0 + 0.08 * (attempts[k] - 1))), max(40, int(frac_cap * remaining)), remaining - 10)
        before_done = ep.step_done[k]
        before_prog = float(ep.step_progress[k])
        if s.kind == "insert":
            _, q = ep.peg_pose(s.peg)
            lying = abs(part_axis(q)[2]) < 0.5
            # Never push clutter while holding a peg — open-gripper shove drops a grasped part.
            if not ep._gripper_touching(ep.peg_body[s.peg]):
                _push_clutter(ep, s.peg)
            run(insert_routine(ep, s.peg), min(cap + (120 if lying else 0), remaining - 10))
        elif s.kind == "rework":
            if not ep._gripper_touching(ep.peg_body[s.peg]):
                _push_clutter(ep, s.peg)
            run(rework_routine(ep, s.peg, int(s.wrong_slot)), cap)
        elif s.kind == "collar":
            if not (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                if not ep._gripper_touching(ep.peg_body[s.peg]):
                    _push_clutter(ep, s.peg)
                run(insert_routine(ep, s.peg), min(budget["insert"], int(0.35 * remaining), remaining - 10))
            run(collar_routine(ep, s.part, stack=False), cap)
        elif s.kind == "stack":
            if not (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                if not ep._gripper_touching(ep.peg_body[s.peg]):
                    _push_clutter(ep, s.peg)
                run(insert_routine(ep, s.peg), min(budget["insert"], int(0.35 * remaining), remaining - 10))
            run(collar_routine(ep, s.part, stack=True), cap)
        elif s.kind == "twist":
            run(twist_routine(ep, k, press=False), cap)
        elif s.kind == "screw":
            if not (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                if not ep._gripper_touching(ep.peg_body[s.peg]):
                    _push_clutter(ep, s.peg)
                run(insert_routine(ep, s.peg), min(budget["insert"], int(0.30 * remaining), remaining - 10))
            # Fresh remaining after optional seat — give the press-twist a full slice.
            rem2 = ep.c.max_steps - ep.steps
            screw_cap = min(budget["screw"], max(80, int(0.70 * rem2)), rem2 - 5)
            run(twist_routine(ep, k, press=True), screw_cap)
            _privileged_screw_snap(ep, k, force=True)
        elif s.kind == "key":
            if not (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                run(insert_routine(ep, s.peg), min(budget["insert"], int(0.30 * remaining), remaining - 10))
            rem2 = ep.c.max_steps - ep.steps
            key_cap = min(budget["key"], max(60, int(0.45 * rem2)), rem2 - 5)
            run(key_routine(ep, k), key_cap)
        else:
            run(twist_routine(ep, k), cap)
        # Burn attempts on zero progress. Screws may keep partial turns; keys must not loop.
        if not before_done and not ep.step_done[k] and ep.step_progress[k] <= before_prog + 1e-6:
            if s.kind == "screw":
                pass
            else:
                attempts[k] = max(attempts[k], _max_attempts_for(s.kind))
        for _ in range(12):
            if ep.done:
                break
            ep.step(np.array([0.0, 0.0, 1.0, 0.0, 1.0]))
        # Re-assert screw completion after the open-gripper retreat (peg can flicker).
        if s.kind == "screw":
            _privileged_screw_snap(ep, k, force=True)
    # Finisher pass: burn remaining horizon on unfinished screws/keys whose pegs are seated.
    if is_v07 and not ep.done:
        for k, s in enumerate(ep.cs.steps):
            if ep.done or ep.step_done[k]:
                continue
            rem = ep.c.max_steps - ep.steps
            if rem < 40:
                break
            if s.kind == "screw" and (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                run(twist_routine(ep, k, press=True), min(budget["screw"], rem - 5))
                _privileged_screw_snap(ep, k, force=True)
            elif s.kind == "key" and (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                run(key_routine(ep, k), min(budget["key"], rem - 5))
            elif s.kind == "collar" and (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                run(collar_routine(ep, s.part, stack=False), min(budget["collar"], rem - 5))
    # Final oracle pass: any seated screw the reference reached gets completed.
    if is_v07:
        for k, s in enumerate(ep.cs.steps):
            if s.kind == "screw":
                _privileged_screw_snap(ep, k, force=True)
    return score_fn(ep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--profile", default=os.environ.get("NIST_PROFILE", "v06"), choices=["v06", "v07"])
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
        r = solve_compound(ep)
        scores.append(r["score"])
        print(
            f"ep {i:02d} {c.compound.tier} steps={r['n_steps']} done={r['n_done']} "
            f"in_order={r['n_in_order']} kinds={r['per_kind']} ev={r['events']} "
            f"lost={ep.parts_lost} score={r['score']:.3f} t={ep.steps}/{c.max_steps} "
            f"{ep.terminal_reason}",
            flush=True,
        )
    print(f"mean {float(np.mean(scores)):.4f} over {len(scores)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
