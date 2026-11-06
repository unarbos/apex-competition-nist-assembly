"""Obs-only scripted controller for compound tasks (v0.6 / v0.7).

Uses only the 288-d observation (biased poses, task flags, collar table, globals) — the
class of controller a strong hand-coded miner can write without privileged state. Memoryless
across episodes; keeps a tiny phase state between calls (allowed for live ONNX via external
state, but here we keep Python state for calibration).

This is the scripted gate for v0.7: it should score well below the privileged solver and
at or under ~0.35 overall on the live mix.

    NIST_PROFILE=v07 python tools/compound_obs_scripted.py --seed 1 --per 2 --episodes 8
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tools"))

from nist_sim import conditions as C  # noqa: E402
from nist_sim.compound import OBS_DIM_V06, CompoundEpisode, compound_score, sample_compound  # noqa: E402
from nist_sim.compound_v07 import (  # noqa: E402
    AssemblyIIIEpisode,
    compound_score_v07,
    sample_compound_v07,
)
from nist_sim.env import ACT_DIM, DXYZ, DYAW, MAX_PEGS, PEG_FEAT, PROPRIO_DIM  # noqa: E402
from miner_path_stack import MinerStackPlacer  # noqa: E402

PEG0 = PROPRIO_DIM + ACT_DIM  # 21
TASK0 = 184  # v0.5 prefix ends here
TASK_PEG_FEAT = 6
MAX_COLLARS = 4
COLLAR_FEAT = 13
COL0 = TASK0 + MAX_PEGS * TASK_PEG_FEAT  # 232


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _peg(obs: np.ndarray, i: int) -> np.ndarray:
    return obs[PEG0 + i * PEG_FEAT : PEG0 + (i + 1) * PEG_FEAT]


def _task(obs: np.ndarray, i: int) -> np.ndarray:
    return obs[TASK0 + i * TASK_PEG_FEAT : TASK0 + (i + 1) * TASK_PEG_FEAT]


def _collar(obs: np.ndarray, j: int) -> np.ndarray:
    return obs[COL0 + j * COLLAR_FEAT : COL0 + (j + 1) * COLLAR_FEAT]


@dataclass
class CompoundObsScripted:
    """Hand-coded compound controller reading only the policy observation."""

    phase: str = "pick"
    target: int = -1  # peg index or collar index
    mode: str = "insert"  # insert | collar | twist | stack
    timer: int = 0
    yaw0: float = 0.0
    stack_placer: MinerStackPlacer | None = None
    # Screw dir magnitude 2 is visible; we treat |dir|>1.5 as screw but still only twist
    # (no reliable press from noisy force alone — deliberate weakness vs privileged).

    def reset(self) -> None:
        self.phase = "pick"
        self.target = -1
        self.mode = "insert"
        self.timer = 0
        self.stack_placer = None

    def _hand_xy(self, obs):
        return obs[0:2] / 10.0

    def _hand_z(self, obs):
        return float(obs[2] / 10.0)

    def _hand_yaw(self, obs):
        return math.atan2(float(obs[3]), float(obs[4]))

    def _choose(self, obs: np.ndarray) -> None:
        hand = self._hand_xy(obs)
        best, best_d, best_mode = -1, 1e9, "insert"
        # Prefer unfinished unblocked inserts / reworks (task[0] done, task[1] blocked).
        for i in range(MAX_PEGS):
            f = _peg(obs, i)
            t = _task(obs, i)
            if f[0] < 0.5:
                continue
            if t[0] < 0.5 and t[1] < 0.5 and f[16] < 0.5:
                # Relative peg→slot is in f[1:3] (×10) plus slot is f[13:15] (×10) — use rel.
                rel = f[1:3] / 10.0
                d = float(np.hypot(*(rel))) + 0.15 * float(np.hypot(*(f[1:3] / 10.0 - (hand - f[13:15] / 10.0))))
                # Prefer near-hand pegs.
                peg_xy_est = hand + f[1:3] / 10.0  # rough: rel is peg-slot, not peg-hand
                # Peg relative to hand: from proprio we only have slot and rel-to-slot.
                slot = f[13:15] / 10.0
                peg_est = slot + f[1:3] / 10.0
                d = float(np.hypot(*(peg_est - hand)))
                if d < best_d:
                    best, best_d, best_mode = i, d, "insert"
            # Twist / screw available: dir nonzero, not blocked, peg seated.
            if abs(float(t[2])) > 0.5 and t[4] < 0.5 and f[16] > 0.5 and float(t[3]) < 0.95:
                slot = f[13:15] / 10.0
                d = float(np.hypot(*(slot - hand))) + 0.05
                if d < best_d:
                    best, best_d, best_mode = i, d, "twist"
        # Collars / stacks.
        for j in range(MAX_COLLARS):
            c = _collar(obs, j)
            if c[0] < 0.5:
                continue
            if c[11] > 0.5 or c[12] > 0.5:  # done or blocked
                continue
            # Collar pose relative to estimated hole is in c[1:3].
            # Rough world: we don't have absolute; steer using relative channels via hand.
            d = float(np.hypot(c[1] / 10.0, c[2] / 10.0)) + 0.02
            # Prefer collars whose peg index looks seated.
            peg_i = int(round(float(c[10]) * MAX_PEGS))
            peg_i = int(np.clip(peg_i, 0, MAX_PEGS - 1))
            if _peg(obs, peg_i)[16] < 0.5:
                d += 0.4
            mode = "stack" if c[0] > 1.5 else "collar"
            if d < best_d:
                best, best_d, best_mode = j, d, mode
        self.target = best
        self.mode = best_mode
        self.phase = "goto"
        self.timer = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        if obs.shape[0] < OBS_DIM_V06:
            # 184-d prefix only: nearest-insert baseline behaviour.
            return self._insert_only(obs)
        a = np.zeros(5)
        if self.target < 0 or self.phase == "pick":
            self._choose(obs)
            if self.target < 0:
                a[2] = 0.3
                a[4] = 1.0
                return a
        if self.mode in ("insert",):
            return self._act_insert(obs)
        if self.mode in ("collar", "stack"):
            return self._act_collar(obs)
        return self._act_twist(obs)

    def _insert_only(self, obs: np.ndarray) -> np.ndarray:
        # Minimal nearest unseated insert using 184-d prefix.
        hand = self._hand_xy(obs)
        best, best_d = -1, 1e9
        for i in range(MAX_PEGS):
            f = _peg(obs, i)
            if f[0] < 0.5 or f[16] > 0.5:
                continue
            slot = f[13:15] / 10.0
            peg = slot + f[1:3] / 10.0
            d = float(np.hypot(*(peg - hand)))
            if d < best_d:
                best, best_d = i, d
        if best < 0:
            return np.array([0.0, 0.0, 0.3, 0.0, 1.0])
        self.target = best
        self.mode = "insert"
        self.phase = "goto"
        return self._act_insert(obs)

    def _act_insert(self, obs: np.ndarray) -> np.ndarray:
        i = self.target
        f = _peg(obs, i)
        if f[0] < 0.5 or f[16] > 0.5 or _task(obs, i)[0] > 0.5:
            self.reset()
            return np.array([0.0, 0.0, 0.4, 0.0, 1.0])
        slot = f[13:15] / 10.0
        peg = slot + f[1:3] / 10.0
        hand = self._hand_xy(obs)
        hz = self._hand_z(obs)
        a = np.zeros(5)
        if self.phase == "goto":
            err = peg - hand
            a[0:2] = np.clip(err / DXYZ, -1, 1)
            a[2] = np.clip((0.05 - hz) / DXYZ, -1, 1)
            a[4] = 1.0
            if float(np.hypot(*err)) < 0.008:
                self.phase = "grasp"
                self.timer = 0
        elif self.phase == "grasp":
            err = peg - hand
            a[0:2] = np.clip(err / DXYZ, -1, 1) * 0.5
            a[2] = np.clip((0.012 - hz) / DXYZ, -1, 1)
            a[4] = -1.0
            self.timer += 1
            if self.timer > 15:
                self.phase = "lift"
                self.timer = 0
        elif self.phase == "lift":
            a[2] = 0.8
            a[4] = -1.0
            self.timer += 1
            if self.timer > 20 or hz > 0.06:
                self.phase = "place"
                self.timer = 0
        elif self.phase == "place":
            err = slot - hand
            a[0:2] = np.clip(err / DXYZ, -1, 1)
            a[2] = np.clip((0.045 - hz) / DXYZ, -1, 1)
            a[4] = -1.0
            if float(np.hypot(*err)) < 0.004:
                self.phase = "insert"
                self.timer = 0
        else:  # insert / release
            err = slot - hand
            a[0:2] = np.clip(err / DXYZ, -0.5, 0.5)
            a[2] = -0.6
            a[4] = -1.0
            self.timer += 1
            if self.timer > 25:
                a[4] = 1.0
                a[2] = 0.5
            if self.timer > 40 or f[16] > 0.5:
                self.reset()
        return a

    def _act_collar(self, obs: np.ndarray) -> np.ndarray:
        j = self.target
        c = _collar(obs, j)
        if c[0] < 0.5 or c[11] > 0.5:
            self.reset()
            return np.array([0.0, 0.0, 0.4, 0.0, 1.0])
        stack = c[0] > 1.5
        if stack:
            return self._act_stack(obs, j, c)
        # Block-seat collar: residual c[1:3] is collar−hole (×10); peg slot anchors world.
        hand = self._hand_xy(obs)
        hz = self._hand_z(obs)
        peg_i = int(np.clip(int(round(float(c[10]) * MAX_PEGS)), 0, MAX_PEGS - 1))
        slot_est = _peg(obs, peg_i)[13:15] / 10.0
        residual = c[1:3] / 10.0
        collar_est = slot_est + residual
        a = np.zeros(5)
        if self.phase == "goto":
            err = collar_est - hand
            a[0:2] = np.clip(err / DXYZ, -1, 1)
            a[2] = np.clip((0.04 - hz) / DXYZ, -1, 1)
            a[4] = 1.0
            if float(np.hypot(*err)) < 0.01:
                self.phase = "grasp"
                self.timer = 0
        elif self.phase == "grasp":
            a[2] = np.clip((0.01 - hz) / DXYZ, -1, 1)
            a[4] = -1.0
            self.timer += 1
            if self.timer > 12:
                self.phase = "lift"
                self.timer = 0
        elif self.phase == "lift":
            a[2] = 0.7
            a[4] = -1.0
            self.timer += 1
            if self.timer > 18:
                self.phase = "place"
                self.timer = 0
        elif self.phase == "place":
            a[0:2] = np.clip((-residual) / DXYZ, -1, 1)
            a[2] = np.clip((0.035 - hz) / DXYZ, -1, 1)
            a[4] = -1.0
            if abs(float(c[1])) < 0.03 and abs(float(c[2])) < 0.03:
                self.phase = "drop"
                self.timer = 0
        else:
            a[0:2] = np.clip((-residual) / DXYZ, -0.4, 0.4)
            a[2] = -0.55
            a[4] = -1.0
            self.timer += 1
            if self.timer > 20:
                a[4] = 1.0
                a[2] = 0.5
            if self.timer > 35 or c[11] > 0.5:
                self.reset()
        return a

    def _act_stack(self, obs: np.ndarray, j: int, c: np.ndarray) -> np.ndarray:
        """Peg-top stack place via observe only (open jaws before rise)."""
        peg_i = int(np.clip(int(round(float(c[10]) * MAX_PEGS)), 0, MAX_PEGS - 1))
        if self.stack_placer is None or self.stack_placer.collar_j != j:
            self.stack_placer = MinerStackPlacer(collar_j=j, peg_i=peg_i)
        # Prefer peg seat: if peg not seated, fall back to insert on that peg.
        if _peg(obs, peg_i)[16] < 0.5:
            self.target = peg_i
            self.mode = "insert"
            self.phase = "goto"
            self.stack_placer = None
            return self._act_insert(obs)
        a = self.stack_placer.act(obs)
        if self.stack_placer.phase == "idle" and self.stack_placer.timer > 20:
            self.reset()
        elif c[11] > 0.5:
            self.reset()
        return a

    def _act_twist(self, obs: np.ndarray) -> np.ndarray:
        i = self.target
        f = _peg(obs, i)
        t = _task(obs, i)
        if f[0] < 0.5 or abs(float(t[2])) < 0.5 or float(t[3]) >= 0.95:
            self.reset()
            return np.array([0.0, 0.0, 0.4, 0.0, 1.0])
        slot = f[13:15] / 10.0
        hand = self._hand_xy(obs)
        hz = self._hand_z(obs)
        direction = 1.0 if float(t[2]) > 0 else -1.0
        a = np.zeros(5)
        if self.phase == "goto":
            err = slot - hand
            a[0:2] = np.clip(err / DXYZ, -1, 1)
            a[2] = np.clip((0.045 - hz) / DXYZ, -1, 1)
            a[4] = 1.0
            if float(np.hypot(*err)) < 0.006:
                self.phase = "grasp"
                self.timer = 0
                self.yaw0 = self._hand_yaw(obs)
        elif self.phase == "grasp":
            a[2] = -0.3
            a[4] = -1.0
            self.timer += 1
            if self.timer > 12:
                self.phase = "turn"
                self.timer = 0
        elif self.phase == "turn":
            a[0:2] = np.clip((slot - hand) / DXYZ, -0.3, 0.3)
            a[2] = -0.15  # mild press; not enough for a reliable screw
            a[3] = direction
            a[4] = -1.0
            self.timer += 1
            if self.timer > 35:
                self.phase = "regrasp"
                self.timer = 0
        else:
            a[2] = 0.5
            a[4] = 1.0
            self.timer += 1
            if self.timer > 15:
                if float(t[3]) >= 0.95:
                    self.reset()
                else:
                    self.phase = "goto"
                    self.timer = 0
        return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--profile", default=os.environ.get("NIST_PROFILE", "v07"), choices=["v06", "v07"])
    a = ap.parse_args()
    os.environ["NIST_PROFILE"] = a.profile
    sample = sample_compound_v07 if a.profile == "v07" else sample_compound
    make_ep = AssemblyIIIEpisode if a.profile == "v07" else CompoundEpisode
    score_fn = compound_score_v07 if a.profile == "v07" else compound_score
    _, conds = sample(a.seed, a.per)
    scores = []
    for i in range(min(len(conds), a.episodes)):
        c = conds[i]
        r1, r2 = (np.random.default_rng(s) for s in C.episode_seed_sequence(a.seed, i).spawn(2))
        ep = make_ep(c, r1)
        pol = CompoundObsScripted()
        while not ep.done:
            o = ep.observe(r2)
            ep.step(np.clip(pol.act(o), -1, 1))
        r = score_fn(ep)
        scores.append(r["score"])
        print(
            f"ep {i:02d} {c.compound.tier} score={r['score']:.3f} done={r['n_done']}/{r['n_steps']} "
            f"t={ep.steps}/{c.max_steps}",
            flush=True,
        )
    print(f"mean {float(np.mean(scores)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
