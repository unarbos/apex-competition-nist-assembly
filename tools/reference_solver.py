"""Privileged reference solver for the v0.2 multi-peg board.

Hand-written phase machine that reads true MuJoCo state (not observations).
Adapted from the v0.1.5 scripted controller (approach / grasp / yaw align /
compliant descend / release / retreat) plus multi-peg sequencing and clutter
push. Lives under tools/ — never shipped in the player image.

    PYTHONPATH=referee python tools/reference_solver.py --seed 0
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))

from nist_sim.conditions import sample_episodes  # noqa: E402
from nist_sim.env import (  # noqa: E402
    BLOCK_H,
    DXYZ,
    DYAW,
    HAND_Z_MIN,
    MAX_OPEN,
    MAX_SEAT_TILT,
    PAD_BOTTOM_OFFSET,
    PAD_CENTRE_Z,
    PEG_LEN,
    SEAT_TOL_Z,
    SEATED_HOLD,
    WALL_T,
    Episode,
)
from nist_sim.scoring import episode_score  # noqa: E402


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def part_axis(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def part_yaw(q) -> float:
    w, x, y, z = q
    return 2 * math.atan2(z, w)


@dataclass
class _PegCond:
    """Minimal condition view for one peg (scripted controller API)."""

    shape: str
    diameter: float
    hole_xy: tuple[float, float]
    reset_type: str


class PegView:
    """Duck-types the single-peg Episode surface used by the scripted phase machine."""

    def __init__(self, ep: Episode, peg_i: int):
        self.ep = ep
        self.i = peg_i
        peg = ep.c.pegs[peg_i]
        hx, hy = ep.slot_xy[peg_i]
        self.c = _PegCond(
            shape=peg.shape(),
            diameter=peg.diameter(),
            hole_xy=(hx, hy),
            reset_type=peg.start_type,
        )
        self.r_hole = ep.r_hole[peg_i]
        self.hole = np.array([hx, hy, BLOCK_H])
        self.part_body = ep.peg_body[peg_i]
        self.grip_bodies = ep.grip_bodies
        self.model = ep.model
        self.data = ep.data
        self.target = ep.target
        self._seated_run = 0
        self.done = False
        self.success = False

    @property
    def steps(self) -> int:
        return self.ep.steps

    @property
    def grip(self) -> float:
        return self.ep.grip

    @grip.setter
    def grip(self, v: float) -> None:
        self.ep.grip = v

    def part_pose(self):
        return self.ep.peg_pose(self.i)

    def _touching(self) -> bool:
        return self.ep._gripper_touching(self.part_body)

    def step(self, a: np.ndarray) -> None:
        self.ep.step(a)
        # Local assembled flag for the active peg (episode may continue).
        p, q = self.part_pose()
        d = self.data
        clear = (
            d.qpos[2] + PAD_BOTTOM_OFFSET >= p[2] + PEG_LEN / 2 + 0.002
            and not self._touching()
        )
        up = abs(part_axis(q)[2]) >= math.cos(MAX_SEAT_TILT)
        seated = (
            p[2] - PEG_LEN / 2 <= SEAT_TOL_Z
            and float(np.hypot(*(p[0:2] - self.hole[0:2]))) <= self.r_hole
            and clear
            and up
        )
        self._seated_run = self._seated_run + 1 if seated else 0
        if self._seated_run >= SEATED_HOLD:
            self.done = True
            self.success = True
        if self.ep.done and not self.success:
            self.done = True


class ScriptedPeg:
    """Phase machine for one peg. Uses true sim state (privileged)."""

    def __init__(
        self,
        e: PegView,
        speed: float = 1.0,
        lying_offset: float = 0.0175,
        swing_grip: float = -0.8,
        press: float = 0.002,
        pivot_steps: int = 22,
        wall_z: float = 0.010,
        straighten_steps: int = 15,
        th_end_deg: float = 90.0,
    ):
        self.th_end = math.radians(th_end_deg)
        self.straighten_steps = straighten_steps
        self.wall_z = wall_z
        self.press = press
        self.pivot_steps = pivot_steps
        self.e = e
        self.speed = speed
        self.lo = lying_offset
        self.swing_grip = swing_grip
        self._sgn: float | None = None
        self.log: list = []

    def _act(self, goal_xyz=None, goal_yaw=None, grip=1.0, speed=None):
        e = self.e
        s = self.speed if speed is None else speed
        # Compensate Cartesian backlash: deadzone eats small commands, so pad
        # the requested delta when the parent episode exposes backlash.
        parent = getattr(e, "ep", e)
        backlash = float(getattr(getattr(parent, "c", None), "backlash", 0.0) or 0.0)
        a = np.zeros(5)
        if goal_xyz is not None:
            g = np.array(goal_xyz, float)
            for i in range(3):
                if not np.isnan(g[i]):
                    err = g[i] - e.target[i]
                    if abs(err) > 1e-9:
                        err = err + math.copysign(backlash, err)
                    a[i] = np.clip(err / DXYZ, -s, s)
        if goal_yaw is not None:
            err = wrap(goal_yaw - e.target[3])
            if abs(err) > 1e-9:
                err = err + math.copysign(backlash * 20.0, err)
            a[3] = np.clip(err / DYAW, -s, s)
        a[4] = grip
        return a

    def wait(self, n, **kw):
        for _ in range(n):
            yield self._act(**kw)

    def goto(self, xyz, yaw=None, grip=1.0, tol=0.0006, max_n=200, speed=None):
        e = self.e
        delay = int(getattr(getattr(e, "ep", e).c, "act_delay", 0) or 0)
        for _ in range(max_n):
            xyz_ = [e.target[i] if (xyz[i] is None) else xyz[i] for i in range(3)]
            yield self._act(xyz_, yaw, grip, speed)
            terr = np.abs(np.array(xyz_) - e.target[0:3]).max()
            qerr = np.abs(np.array(xyz_) - e.data.qpos[0:3]).max()
            yerr = 0 if yaw is None else abs(wrap(yaw - e.data.qpos[3]))
            if terr < tol and yerr < 0.02:
                # Drain any delayed actions, then confirm the plant settled.
                for _ in range(max(1, delay + 1)):
                    yield self._act(xyz_, yaw, grip, speed=0.0)
                qerr = np.abs(np.array(xyz_) - e.data.qpos[0:3]).max()
                if qerr < tol * 3:
                    return

    def hole_yaw_target(self) -> float:
        e = self.e
        shape = e.c.shape
        if shape == "round":
            return float(e.target[3])
        _, q = e.part_pose()
        py = part_yaw(q)
        if shape == "hex":
            # Align to nearest 60°.
            err = wrap(-py)
            err = (err + math.pi / 6) % (math.pi / 3) - math.pi / 6
            return float(e.data.qpos[3] + err)
        if shape in ("square", "dflat"):
            err = wrap(-py)
            err = (err + math.pi / 4) % (math.pi / 2) - math.pi / 4
            return float(e.data.qpos[3] + err)
        return float(e.target[3])

    def insert(self) -> Iterator[np.ndarray]:
        e = self.e
        hx, hy = e.c.hole_xy

        def part_off():
            p, q = e.part_pose()
            ax = part_axis(q)
            if ax[2] < 0:
                ax = -ax
            tip = p - ax * PEG_LEN / 2
            return tip[0:2] - e.data.qpos[0:2]

        def tip_xy():
            return e.data.qpos[0:2] + part_off()

        p, _ = e.part_pose()
        dz_bottom = (p[2] - PEG_LEN / 2) - (BLOCK_H + 0.004)
        if dz_bottom < 0:
            yield from self.goto([None, None, e.target[2] - dz_bottom], grip=-1.0)
        # Refresh yaw once above the hole, then again just before the lip.
        yaw = self.hole_yaw_target()
        for _ in range(140):
            o = part_off()
            yaw = self.hole_yaw_target()
            yield self._act([hx - o[0], hy - o[1], e.target[2]], yaw, -1.0)
            if (
                np.hypot(*(tip_xy() - np.array([hx, hy]))) < 0.0006
                and abs(wrap(yaw - e.data.qpos[3])) < 0.03
            ):
                break
        stuck = 0
        for k in range(200):
            p, _ = e.part_pose()
            o = part_off()
            bottom = p[2] - PEG_LEN / 2
            yaw = self.hole_yaw_target()
            # Spiral / rock search when the tip sits on the hole lip (common for
            # sharp square corners and sub-mm clearance in MuJoCo).
            wx = wy = wyaw = 0.0
            if bottom <= BLOCK_H + 0.0015 and bottom > 0.001:
                stuck += 1
                amp = min(0.0025, 0.0004 + 0.00008 * stuck)
                wx = amp * math.sin(stuck * 0.9)
                wy = amp * math.cos(stuck * 0.7)
                wyaw = 0.04 * math.sin(stuck * 0.5)
            else:
                stuck = 0
            sp = 0.7 if bottom > BLOCK_H + 0.003 else 0.35
            yield self._act(
                [hx - o[0] + wx, hy - o[1] + wy, e.target[2] - 0.004],
                yaw + wyaw,
                -1.0,
                speed=sp,
            )
            if bottom < 0.0006:
                break
            parent = getattr(e, "ep", None)
            idx = getattr(e, "i", None)
            if parent is not None and idx is not None and parent.seated_now[idx]:
                break
        yield from self.wait(8, grip=1.0)
        yield from self.goto([None, None, PEG_LEN + 0.03], grip=1.0, max_n=60)
        yield from self.wait(SEATED_HOLD + 8, grip=1.0)

    def pick_upright(self) -> Iterator[np.ndarray]:
        e = self.e
        p, q = e.part_pose()
        yaw = float(e.data.qpos[3])
        if e.c.shape in ("square", "hex", "dflat"):
            yaw = self.hole_yaw_target()
        top = p[2] + PEG_LEN / 2
        yield from self.goto([p[0], p[1], top + 0.01], yaw, 1.0)
        p, _ = e.part_pose()
        z_grasp = p[2] + (PEG_LEN / 2 - 0.008) - PAD_CENTRE_Z
        yield from self.goto([p[0], p[1], z_grasp], yaw, 1.0, speed=0.4)
        yield from self.wait(12, grip=-1.0)
        yield from self.goto([None, None, 0.075], grip=-1.0, speed=0.5)

    def _lying_grasp_goal(self):
        e = self.e
        p, q = e.part_pose()
        ax = part_axis(q)
        lxy = ax[0:2] / (np.linalg.norm(ax[0:2]) + 1e-9)
        if self._sgn is None:
            hole = np.array(e.c.hole_xy)
            self._sgn = -1.0 if np.dot(hole - p[0:2], lxy) > 0 else 1.0
        gxy = p[0:2] + self._sgn * self.lo * lxy
        yaw = math.atan2(lxy[1], lxy[0])
        cur = float(e.target[3])
        if abs(wrap(yaw - cur)) > math.pi / 2:
            yaw = wrap(yaw + math.pi)
        return gxy, cur + wrap(yaw - cur)

    def pick_lying(self) -> Iterator[np.ndarray]:
        """Side-grasp, lay on block, pivot upright against the wall (v0.1.5 script)."""
        e = self.e
        H = np.array(e.c.hole_xy)
        self._sgn = 1.0
        lo_save, self.lo = self.lo, 0.0
        for z, sp, n in [(0.03, None, 80), (HAND_Z_MIN, 0.5, 40)]:
            for _ in range(n):
                gxy, yaw = self._lying_grasp_goal()
                yield self._act([gxy[0], gxy[1], z], yaw, 1.0, sp)
                if (
                    abs(e.data.qpos[2] - z) < 0.0008
                    and np.hypot(*(gxy - e.data.qpos[0:2])) < 0.0008
                    and abs(wrap(yaw - e.data.qpos[3])) < 0.03
                ):
                    break
        self.lo = lo_save
        yield from self.wait(6, grip=-1.0)
        p, _ = e.part_pose()
        dz = float(e.data.qpos[2] - p[2])
        r = e.c.diameter / 2
        w = np.array([1.0 if p[0] >= H[0] else -1.0, 0.0])
        top = BLOCK_H + r
        yield from self.goto([None, None, top + 0.008 + dz], grip=-1.0, speed=1.0, tol=0.002)
        cur = float(e.target[3])
        cands = [cur + wrap(a - cur) for a in (0.0, math.pi)]
        yaw = min(cands, key=lambda a: abs(a - cur))
        yield from self.goto([H[0], H[1], top + 0.008 + dz], yaw, -1.0, speed=1.0, tol=0.001, max_n=60)
        yield from self.goto([H[0], H[1], top + 0.0005 + dz], yaw, -1.0, speed=0.5, tol=0.0008, max_n=15)
        yield from self.wait(4, grip=1.0)
        p, _ = e.part_pose()
        g = p[0:2] + w * self.lo
        zg = p[2] - PAD_CENTRE_Z
        yield from self.goto([None, None, p[2] + 0.012], None, 1.0, speed=1.0, tol=0.002, max_n=20)
        yield from self.goto([g[0], g[1], p[2] + 0.012], None, 1.0, speed=1.0, tol=0.001, max_n=30)
        yield from self.goto([g[0], g[1], zg], None, 1.0, speed=1.0, tol=0.0008, max_n=30)
        yield from self.wait(6, grip=-1.0)
        zc = self.wall_z
        dz = -PAD_CENTRE_Z
        Rw = e.r_hole + WALL_T
        d_tip = self.lo + PEG_LEN / 2
        hx = H + w * (Rw + 0.004 + d_tip)
        yield from self.goto([g[0], g[1], p[2] + 0.006 + dz], None, -1.0, speed=1.0, tol=0.002, max_n=10)
        yield from self.goto([hx[0], hx[1], p[2] + 0.006 + dz], None, -1.0, speed=1.0, tol=0.001, max_n=30)
        yield from self.goto([hx[0], hx[1], zc + dz], None, -1.0, speed=1.0, tol=0.001, max_n=20)
        v = -w
        P = H - v * (Rw - self.press)
        L = d_tip
        hp = P - v * L
        yield from self.goto([hp[0], hp[1], zc + dz], None, -1.0, speed=0.5, tol=0.0008, max_n=10)
        sg = self.swing_grip
        yield from self.wait(2, grip=sg)
        n = self.pivot_steps
        th = 0.0
        for k in range(1, n + 1):
            th = self.th_end * (0.5 - 0.5 * math.cos(math.pi * k / n))
            hxy = P - v * L * math.cos(th)
            hz = zc + dz + L * math.sin(th)
            yield self._act([hxy[0], hxy[1], hz], None, sg, speed=1.0)
        for _ in range(self.straighten_steps):
            _, q = e.part_pose()
            ax = part_axis(q)
            down = -ax if ax[2] > 0 else ax
            lean = float(np.dot(down[0:2], v))
            if abs(lean) < math.sin(math.radians(0.8)):
                break
            th += math.copysign(math.radians(1.0), lean)
            hxy = P - v * L * math.cos(th)
            hz = zc + dz + L * math.sin(th)
            yield self._act([hxy[0], hxy[1], hz], None, sg, speed=1.0)
        yield from self.wait(2, grip=sg)
        yield from self.wait(3, grip=-1.0)
        self.log.append(("upright", e.steps))
        yield from self.goto([None, None, e.target[2] + BLOCK_H + 0.006], grip=-1.0, speed=1.0, tol=0.002, max_n=30)

    def policy(self) -> Iterator[np.ndarray]:
        e = self.e
        rt = e.c.reset_type
        p, q = e.part_pose()
        lying = abs(part_axis(q)[2]) < 0.5
        # Already grasped above the hole: skip pick.
        holding = e._touching() and (e.data.qpos[4] + e.data.qpos[5]) < 0.014
        if holding and p[2] > 0.04:
            yield from self.insert()
            return
        if lying or rt == "lying":
            yield from self.pick_lying()
        elif rt in ("loose", "tray", "near_hole") or p[2] < 0.04:
            yield from self.pick_upright()
        yield from self.insert()


def _push_clutter(ep: Episode, peg_i: int, max_steps: int = 28) -> None:
    """Gently push the nearest clutter block away from the target hole approach."""
    hx, hy = ep.slot_xy[peg_i]
    best, best_d = -1, 1e9
    for i in range(ep.n_clutter):
        p, _ = ep.clutter_pose(i)
        d = float(np.hypot(p[0] - hx, p[1] - hy))
        if d < best_d and p[2] < 0.045:
            best_d, best = d, i
    if best < 0 or best_d > 0.028:
        return
    for t in range(max_steps):
        if ep.done:
            return
        p, _ = ep.clutter_pose(best)
        if float(np.hypot(p[0] - hx, p[1] - hy)) > 0.04 or abs(p[0]) > 0.20 or abs(p[1]) > 0.20:
            return
        # Abort if any peg is near the board edge (avoid part_lost cascades).
        for j in range(ep.n_pegs):
            pj, _ = ep.peg_pose(j)
            if abs(pj[0]) > 0.22 or abs(pj[1]) > 0.22:
                return
        hand = ep.data.qpos[0:3]
        away = np.array([p[0] - hx, p[1] - hy])
        n = float(np.linalg.norm(away)) + 1e-6
        away /= n
        if t < 10:
            goal = p[0:2]
            z = 0.032
        else:
            goal = p[0:2] + away * 0.035
            z = 0.026
        a = np.array([
            np.clip((goal[0] - hand[0]) / DXYZ, -0.55, 0.55),
            np.clip((goal[1] - hand[1]) / DXYZ, -0.55, 0.55),
            np.clip((z - hand[2]) / DXYZ, -0.55, 0.55),
            0.0,
            1.0,
        ])
        ep.step(a)


def _peg_order(ep: Episode) -> list[int]:
    """Prefer already-held, then upright near their hole; defer lying/tray tips."""
    hand = ep.data.qpos[0:2]
    scored = []
    for i in range(ep.n_pegs):
        if ep.seated_now[i]:
            continue
        p, q = ep.peg_pose(i)
        d = float(np.hypot(*(p[0:2] - hand)))
        lying = abs(part_axis(q)[2]) < 0.5
        holding = ep._gripper_touching(ep.peg_body[i]) and (ep.data.qpos[4] + ep.data.qpos[5]) < 0.014
        if holding and p[2] > 0.04:
            d -= 0.20  # finish the held peg first
        if lying:
            d += 0.25  # expensive; do upright pegs first for partial credit
        d += 0.35 * float(np.hypot(p[0] - ep.slot_xy[i][0], p[1] - ep.slot_xy[i][1]))
        scored.append((d, i))
    scored.sort()
    return [i for _, i in scored]


def solve_episode(ep: Episode, budget_per_peg: int | None = None) -> dict:
    """Run the privileged solver on one Episode. Mutates ep in place."""
    if budget_per_peg is None:
        # Leave room for K pegs; lying attempts get a shorter cap so upright
        # pegs still get a turn within the episode horizon.
        budget_per_peg = max(280, ep.c.max_steps // max(1, ep.n_pegs))
    for peg_i in _peg_order(ep):
        if ep.done:
            break
        if ep.seated_now[peg_i]:
            continue
        remaining = ep.c.max_steps - ep.steps
        if remaining < 40:
            break
        _push_clutter(ep, peg_i)
        if ep.done:
            break
        view = PegView(ep, peg_i)
        _, q = view.part_pose()
        lying = abs(part_axis(q)[2]) < 0.5
        if lying:
            # Lying regrasp is brittle on the multi-peg board (pivots fling
            # neighbours). Skip unless we have ample budget and few pegs left.
            n_left = sum(1 for s in ep.seated_now if not s)
            if remaining < 350 or n_left > 3:
                continue
            view.c.reset_type = "lying"
            peg_budget = min(budget_per_peg, max(150, remaining // 2))
        else:
            peg_budget = min(budget_per_peg, remaining - 20)
        script = ScriptedPeg(view, swing_grip=-1.0)
        n0 = ep.steps
        for a in script.policy():
            if ep.done or ep.seated_now[peg_i]:
                break
            if ep.steps - n0 > peg_budget:
                break
            ep.step(a)
        # Retreat between pegs (open gripper, lift) so seated pegs stay clear.
        for _ in range(12):
            if ep.done:
                break
            ep.step(np.array([0.0, 0.0, 1.0, 0.0, 1.0]))
    while not ep.done:
        ep.step(np.array([0.0, 0.0, 0.2, 0.0, 1.0]))
    seated = ep.peg_seated_at_end()
    first = ep.peg_first_seat_steps()
    msa = ep.action_l2 / max(1, ep.steps)
    return {
        "score": episode_score(seated, first, ep.c.max_steps, msa),
        "seated": seated,
        "n_seated": int(sum(seated)),
        "n_pegs": ep.n_pegs,
        "steps": ep.steps,
        "success": bool(ep.success),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=8, help="first N episodes of the round")
    a = ap.parse_args()
    _, conds = sample_episodes(a.seed, a.per)
    scores = []
    for i, c in enumerate(conds[: a.episodes]):
        ss = np.random.SeedSequence([a.seed, 0x5245, i])
        e = Episode(c, np.random.default_rng(ss.spawn(2)[0]))
        r = solve_episode(e)
        scores.append(r["score"])
        print(
            f"ep {i:02d} K={c.n_pegs} clutter={len(c.clutter)} clear={c.clearance*1e3:.2f}mm "
            f"seated={r['n_seated']}/{r['n_pegs']} score={r['score']:.3f} steps={r['steps']}"
        )
    print(f"mean score {float(np.mean(scores)):.4f} over {len(scores)} episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
