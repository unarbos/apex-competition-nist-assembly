"""Privileged reference solver for v0.6 compound episodes (reads true state; never shipped).

Plans over the episode's task graph: repeatedly picks an available step (its part has arrived and
its prerequisites are done), runs a phase machine for it, and retreats. Inserts reuse the v0.5
``ScriptedPeg`` machine; collars and twists have their own routines. If nothing in order is
available it waits for the next wave, and near the end it tries out-of-order steps for partial
credit.

    python tools/compound_solver.py --seed 1 --per 2 --episodes 8
"""

from __future__ import annotations

import argparse
import math
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
from nist_sim.env import BLOCK_H, DXYZ, DYAW, HAND_Z_MIN, PAD_CENTRE_Z, PEG_LEN  # noqa: E402
from reference_solver import PegView, ScriptedPeg, _push_clutter, part_axis  # noqa: E402

YAW_LIMIT = 3.0


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


def collar_routine(ep: CompoundEpisode, j: int) -> Iterator[np.ndarray]:
    ctl = Ctl(ep)
    col = ep.cs.collars[j]
    hx, hy = ep.slot_xy[col.peg]
    p, _ = ep.collar_pose(j)
    yaw = float(ep.data.qpos[3])
    yield from ctl.goto([p[0], p[1], 0.045], yaw, 1.0)
    p, _ = ep.collar_pose(j)
    zg = max(HAND_Z_MIN, p[2] - PAD_CENTRE_Z + 0.001)
    yield from ctl.goto([p[0], p[1], zg], yaw, 1.0, speed=0.5)
    yield from ctl.wait(10, grip=-1.0)
    # Lift clear of every peg top, then carry over the target slot.
    lift = PEG_LEN + 0.012 + COLLAR_H / 2 - PAD_CENTRE_Z
    yield from ctl.goto([None, None, lift], None, -1.0, speed=0.7)
    for _ in range(160):
        p, _ = ep.collar_pose(j)
        off = p[0:2] - ep.data.qpos[0:2]
        goal = np.array([hx, hy]) - off
        yield ctl.act([goal[0], goal[1], lift], None, -1.0, speed=0.7)
        if float(np.hypot(p[0] - hx, p[1] - hy)) < 0.0004:
            break
    # Descend around the peg until the palm nearly reaches the peg top.
    z_rel = PEG_LEN + 0.003 - 0.021  # palm bottom (hand z + 0.021) stays above the peg top
    for _ in range(120):
        p, _ = ep.collar_pose(j)
        off = p[0:2] - ep.data.qpos[0:2]
        goal = np.array([hx, hy]) - off
        yield ctl.act([goal[0], goal[1], max(z_rel, ep.target[2] - 0.002)], None, -1.0, speed=0.5)
        if ep.data.qpos[2] <= z_rel + 0.0008:
            break
    yield from ctl.wait(8, grip=1.0)
    yield from ctl.goto([None, None, 0.085], None, 1.0, max_n=40)
    yield from ctl.wait(12, grip=1.0)


def twist_routine(ep: CompoundEpisode, k: int) -> Iterator[np.ndarray]:
    ctl = Ctl(ep)
    s = ep.cs.steps[k]
    i = s.part
    for _regrasp in range(4):
        need = s.twist_target * 1.08 - ep.twist_acc[i] * s.twist_dir
        if need <= 0:
            break
        p, _ = ep.peg_pose(i)
        yaw0 = float(ep.data.qpos[3])
        room = (YAW_LIMIT - yaw0) if s.twist_dir > 0 else (yaw0 + YAW_LIMIT)
        if room < 0.5:  # unwind the open hand first
            yaw0 = -s.twist_dir * (YAW_LIMIT - 0.3)
            room = 2 * YAW_LIMIT - 0.3
        top = p[2] + PEG_LEN / 2
        yield from ctl.goto([p[0], p[1], top + 0.012], yaw0, 1.0)
        zg = top - 0.008 - PAD_CENTRE_Z
        yield from ctl.goto([p[0], p[1], zg], yaw0, 1.0, speed=0.5)
        yield from ctl.wait(10, grip=-1.0)
        turn = min(need, room - 0.05)
        goal = yaw0 + s.twist_dir * turn
        for _ in range(int(turn / (DYAW * 0.5)) + 40):
            yield ctl.act([None, None, None], goal, -1.0, speed=0.5)
            if abs(wrap(goal - ep.data.qpos[3])) < 0.03:
                break
        yield from ctl.wait(6, grip=1.0)
        yield from ctl.goto([None, None, top + 0.015], None, 1.0, max_n=40)
    yield from ctl.goto([None, None, 0.085], None, 1.0, max_n=40)
    yield from ctl.wait(8, grip=1.0)


def insert_routine(ep: CompoundEpisode, i: int) -> Iterator[np.ndarray]:
    view = PegView(ep, i)
    _, q = view.part_pose()
    if abs(part_axis(q)[2]) < 0.5:
        view.c.reset_type = "lying"
    yield from ScriptedPeg(view, swing_grip=-1.0).policy()


def _available(ep: CompoundEpisode, k: int) -> bool:
    s = ep.cs.steps[k]
    if s.kind == "collar":
        return not ep.collar_parked[s.part]
    return not ep.peg_parked[s.peg]


def _priority(ep: CompoundEpisode, k: int) -> float:
    s = ep.cs.steps[k]
    hand = ep.data.qpos[0:2]
    if s.kind == "collar":
        p, _ = ep.collar_pose(s.part)
    else:
        p, _ = ep.peg_pose(s.peg)
    d = float(np.hypot(*(p[0:2] - hand)))
    if s.kind == "insert":
        _, q = ep.peg_pose(s.peg)
        if abs(part_axis(q)[2]) < 0.5:
            d += 0.25
        if ep._gripper_touching(ep.peg_body[s.peg]) and p[2] > 0.04:
            d -= 0.3
    # Unblock the most dependants first.
    n_dep = sum(1 for t in ep.cs.steps if k in t.prereq)
    return d - 0.03 * n_dep


def _is_lying_insert(ep: CompoundEpisode, k: int) -> bool:
    s = ep.cs.steps[k]
    if s.kind != "insert":
        return False
    _, q = ep.peg_pose(s.peg)
    return abs(part_axis(q)[2]) < 0.5


def solve_compound(ep: CompoundEpisode, budget: dict | None = None, max_attempts: int = 2) -> dict:
    budget = budget or {"insert": 300, "collar": 260, "twist": 260}
    attempts = [0] * len(ep.cs.steps)

    def run(gen, cap):
        n0 = ep.steps
        for a in gen:
            if ep.done or ep.steps - n0 > cap:
                break
            ep.step(a)

    while not ep.done:
        remaining = ep.c.max_steps - ep.steps
        todo = [k for k in range(len(ep.cs.steps)) if not ep.step_done[k] and _available(ep, k)
                and attempts[k] < max_attempts]
        lying = {k for k in todo if _is_lying_insert(ep, k)}
        in_order = [k for k in todo if not ep.step_blocked(k)]
        # In-order steps first, then wait for the next wave, then out-of-order steps (partial
        # credit, and they unlock dependants), and a lying-peg regrasp only as a last resort.
        ready = [k for k in in_order if k not in lying]
        if not ready and not all(ep.wave_arrived) and remaining > 150:
            ep.step(np.array([0.0, 0.0, 0.3, 0.0, 1.0]))
            continue
        if not ready:
            ready = [k for k in todo if k not in lying]
        if not ready and remaining > 350:
            ready = [k for k in todo if k in lying and attempts[k] == 0]
        if not ready or remaining < 40:
            ep.step(np.array([0.0, 0.0, 0.3, 0.0, 1.0]))
            continue
        k = min(ready, key=lambda k: _priority(ep, k))
        attempts[k] += 1
        s = ep.cs.steps[k]
        if s.kind == "insert":
            _, q = ep.peg_pose(s.peg)
            lying = abs(part_axis(q)[2]) < 0.5
            _push_clutter(ep, s.peg)
            run(insert_routine(ep, s.peg), min(budget["insert"] + (150 if lying else 0), remaining - 20))
        elif s.kind == "collar":
            run(collar_routine(ep, s.part), min(budget["collar"], remaining - 20))
        else:
            run(twist_routine(ep, k), min(budget["twist"], remaining - 20))
        for _ in range(10):
            if ep.done:
                break
            ep.step(np.array([0.0, 0.0, 1.0, 0.0, 1.0]))
    return compound_score(ep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    _, conds = sample_compound(a.seed, a.per)
    scores = []
    for i in range(a.start, min(len(conds), a.start + a.episodes)):
        c = conds[i]
        r1, _ = (np.random.default_rng(s) for s in C.episode_seed_sequence(a.seed, i).spawn(2))
        ep = CompoundEpisode(c, r1)
        r = solve_compound(ep)
        scores.append(r["score"])
        print(f"ep {i:02d} {c.compound.tier} steps={r['n_steps']} done={r['n_done']} in_order={r['n_in_order']} "
              f"kinds={r['per_kind']} ev={r['events']} lost={ep.parts_lost} score={r['score']:.3f} "
              f"t={ep.steps}/{c.max_steps} {ep.terminal_reason}", flush=True)
    print(f"mean {float(np.mean(scores)):.4f} over {len(scores)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
