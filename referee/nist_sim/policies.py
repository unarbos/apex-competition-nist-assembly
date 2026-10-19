"""Reference policies for the multi-peg board.

NearestFirstBaseline: memoryless nearest-unseated insert/release over the obs
table (ONNX twin in tools/make_policies.py).

ObsOnlyScripted: strongest hand-coded controller that uses **only** the policy
observation (biased/noisy estimates + wrist wrench) — the anti-scripting
calibration target. Must score well below a learned policy's ceiling.

StrongScripted: privileged state-based planner for diagnostics only (not shipped).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .conditions import SLOT_SPECS, SHAPE_NAMES
from .env import (
    ACT_DIM,
    BLOCK_H,
    CLUTTER_FEAT,
    DXYZ,
    DYAW,
    HAND_Z_MIN,
    MAX_CLUTTER,
    MAX_PEGS,
    OBS_DIM,
    PEG_FEAT,
    PEG_LEN,
    PROPRIO_DIM,
    Episode,
)

PEG_TABLE0 = PROPRIO_DIM + ACT_DIM  # 21
CLUTTER_TABLE0 = PEG_TABLE0 + MAX_PEGS * PEG_FEAT
# Peg feature layout (PEG_FEAT=17):
# 0 present | 1:4 rel*10 | 4:8 quat | 8:12 shape_oh | 12 size | 13:16 slot*10 | 16 seated
FEAT_PRESENT = 0
FEAT_REL = slice(1, 4)
FEAT_QUAT = slice(4, 8)
FEAT_SHAPE = slice(8, 12)
FEAT_SIZE = 12
FEAT_SLOT = slice(13, 16)
FEAT_SEATED = 16
# Proprio: 0:3 hand*10, 3:5 sin/cos yaw, 5:8 vel, 8 gap, 9:12 force, 12:15 torque, 15 pad_f
PROP_FORCE = slice(9, 12)
PROP_PAD = 15


def _peg_slice(obs: np.ndarray, i: int) -> np.ndarray:
    return obs[PEG_TABLE0 + i * PEG_FEAT : PEG_TABLE0 + (i + 1) * PEG_FEAT]


def _hand_xy(obs: np.ndarray) -> np.ndarray:
    return obs[0:2] / 10.0


def _hand_z(obs: np.ndarray) -> float:
    return float(obs[2] / 10.0)


def _hand_yaw(obs: np.ndarray) -> float:
    return math.atan2(float(obs[3]), float(obs[4]))


@dataclass
class NearestFirstBaseline:
    """Nearest unseated peg → insert/release. No clutter, no lying regrasp."""

    target: int = -1
    phase: str = "goto"
    timer: int = 0

    def reset(self) -> None:
        self.target = -1
        self.phase = "goto"
        self.timer = 0

    def _choose(self, obs: np.ndarray) -> int:
        hand = _hand_xy(obs)
        best, best_d = -1, 1e9
        for i in range(MAX_PEGS):
            f = _peg_slice(obs, i)
            if f[FEAT_PRESENT] < 0.5 or f[FEAT_SEATED] > 0.5:
                continue
            peg_xy = f[FEAT_SLOT].copy()
            peg_xy = peg_xy[0:2] / 10.0 + f[FEAT_REL].copy()[0:2] / 10.0
            d = float(np.hypot(*(peg_xy - hand)))
            if d < best_d:
                best_d, best = d, i
        return best

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        a = np.zeros(ACT_DIM)
        if (
            self.target < 0
            or _peg_slice(obs, self.target)[FEAT_SEATED] > 0.5
            or _peg_slice(obs, self.target)[FEAT_PRESENT] < 0.5
        ):
            self.target = self._choose(obs)
            self.phase = "goto"
            self.timer = 0
        if self.target < 0:
            a[4] = 1.0
            return a
        f = _peg_slice(obs, self.target)
        rel = f[FEAT_REL] / 10.0
        slot_xy = f[FEAT_SLOT][0:2] / 10.0
        hand = np.array([*_hand_xy(obs), _hand_z(obs)])
        qz = float(f[7])  # quat z
        if self.phase == "goto":
            a[0] = float(np.clip((slot_xy[0] - hand[0]) / DXYZ, -1, 1))
            a[1] = float(np.clip((slot_xy[1] - hand[1]) / DXYZ, -1, 1))
            a[2] = float(np.clip((0.07 - hand[2]) / DXYZ, -1, 1))
            a[3] = float(np.clip(-12.5 * qz, -1, 1))
            a[4] = -1.0
            if abs(slot_xy[0] - hand[0]) < 0.006 and abs(slot_xy[1] - hand[1]) < 0.006:
                self.phase = "down"
                self.timer = 0
            if abs(rel[0]) < 0.004 and abs(rel[1]) < 0.004:
                self.phase = "down"
                self.timer = 0
            return a
        if self.phase == "down":
            a[0] = float(np.clip(-rel[0] / DXYZ, -1, 1))
            a[1] = float(np.clip(-rel[1] / DXYZ, -1, 1))
            a[2] = -0.85
            a[3] = float(np.clip(-12.5 * qz, -1, 1))
            a[4] = -1.0
            self.timer += 1
            if rel[2] < PEG_LEN / 2 - BLOCK_H + 0.005 or self.timer > 60:
                self.phase = "release"
                self.timer = 0
            return a
        a[2] = 0.9
        a[4] = 1.0
        self.timer += 1
        if self.timer > 15:
            self.target = -1
            self.phase = "goto"
        return a


@dataclass
class ObsOnlyScripted:
    """Strongest reasonable hand-coded controller using **only** observations.

    Uses the biased peg/slot estimates, wrist force for contact detection, and a
    spiral search when force rises at the lip. Cannot read true MuJoCo state.
    This is the anti-scripting calibration target (must stay ≤ ~0.25).
    """

    target: int = -1
    phase: str = "pick"  # pick | align | insert | spiral | release | retreat
    timer: int = 0
    spiral_t: int = 0
    force_ref: float = 0.0

    def reset(self) -> None:
        self.target = -1
        self.phase = "pick"
        self.timer = 0
        self.spiral_t = 0
        self.force_ref = 0.0

    def _choose(self, obs: np.ndarray) -> int:
        hand = _hand_xy(obs)
        best, best_d = -1, 1e9
        for i in range(MAX_PEGS):
            f = _peg_slice(obs, i)
            if f[FEAT_PRESENT] < 0.5 or f[FEAT_SEATED] > 0.5:
                continue
            # Prefer upright-looking pegs (quat w large) and near estimated slot.
            quat = f[FEAT_QUAT]
            upright = abs(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2))
            peg_xy = f[FEAT_SLOT][0:2] / 10.0 + f[FEAT_REL][0:2] / 10.0
            d = float(np.hypot(*(peg_xy - hand)))
            d += 0.05 * (1.0 - upright)  # defer lying
            d += 0.3 * float(np.linalg.norm(f[FEAT_REL][0:2] / 10.0))
            if d < best_d:
                best_d, best = d, i
        return best

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        a = np.zeros(ACT_DIM)
        force = float(np.linalg.norm(obs[PROP_FORCE]))
        pad = float(obs[PROP_PAD])

        if (
            self.target < 0
            or _peg_slice(obs, self.target)[FEAT_SEATED] > 0.5
            or _peg_slice(obs, self.target)[FEAT_PRESENT] < 0.5
        ):
            self.target = self._choose(obs)
            self.phase = "pick"
            self.timer = 0
            self.spiral_t = 0
        if self.target < 0:
            a[4] = 1.0
            a[2] = 0.2
            return a

        f = _peg_slice(obs, self.target)
        rel = f[FEAT_REL] / 10.0
        slot = f[FEAT_SLOT][0:2] / 10.0
        hand = np.array([*_hand_xy(obs), _hand_z(obs)])
        quat = f[FEAT_QUAT]
        qz = float(quat[3])
        upright = abs(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2))

        if self.phase == "pick":
            # Approach estimated peg (rel + slot ≈ peg in board frame).
            peg_xy = slot + rel[0:2]
            z_goal = max(0.03, float(rel[2]) + BLOCK_H + 0.02)
            a[0] = float(np.clip((peg_xy[0] - hand[0]) / DXYZ, -1, 1))
            a[1] = float(np.clip((peg_xy[1] - hand[1]) / DXYZ, -1, 1))
            a[2] = float(np.clip((z_goal - hand[2]) / DXYZ, -1, 1))
            a[3] = float(np.clip(-10.0 * qz, -1, 1))
            a[4] = 1.0 if pad < 0.05 else -1.0
            self.timer += 1
            if pad > 0.08 or (abs(peg_xy[0] - hand[0]) < 0.008 and self.timer > 25):
                a[4] = -1.0
                if self.timer > 40 or pad > 0.15:
                    self.phase = "align"
                    self.timer = 0
            if upright < 0.5 and self.timer > 80:
                # Give up on this lying peg estimate; try another.
                self.target = -1
                self.timer = 0
            return a

        if self.phase == "align":
            a[0] = float(np.clip((slot[0] - hand[0]) / DXYZ, -1, 1))
            a[1] = float(np.clip((slot[1] - hand[1]) / DXYZ, -1, 1))
            a[2] = float(np.clip((0.065 - hand[2]) / DXYZ, -1, 1))
            a[3] = float(np.clip(-12.0 * qz, -1, 1))
            a[4] = -1.0
            self.timer += 1
            if abs(slot[0] - hand[0]) < 0.005 and abs(slot[1] - hand[1]) < 0.005:
                self.phase = "insert"
                self.timer = 0
                self.force_ref = force
            if self.timer > 100:
                self.phase = "insert"
                self.timer = 0
            return a

        if self.phase == "insert":
            a[0] = float(np.clip(-rel[0] / DXYZ, -1, 1))
            a[1] = float(np.clip(-rel[1] / DXYZ, -1, 1))
            a[2] = -0.7
            a[3] = float(np.clip(-12.0 * qz, -1, 1))
            a[4] = -1.0
            self.timer += 1
            # Contact at the lip → spiral search (bias makes pure servo miss).
            if force > self.force_ref + 0.05 and abs(rel[2]) < 0.03:
                self.phase = "spiral"
                self.spiral_t = 0
            if rel[2] < -0.015 or self.timer > 80:
                self.phase = "release"
                self.timer = 0
            return a

        if self.phase == "spiral":
            rad = min(0.004, 0.0003 * (1 + self.spiral_t // 3))
            ang = 0.55 * self.spiral_t
            goal = slot + rad * np.array([math.cos(ang), math.sin(ang)])
            a[0] = float(np.clip((goal[0] - hand[0]) / DXYZ, -1, 1))
            a[1] = float(np.clip((goal[1] - hand[1]) / DXYZ, -1, 1))
            a[2] = -0.55 if force > self.force_ref + 0.08 else -0.85
            a[3] = float(np.clip(-8.0 * qz + 0.15 * math.sin(ang), -1, 1))
            a[4] = -1.0
            self.spiral_t += 1
            if rel[2] < -0.015 or self.spiral_t > 120 or f[FEAT_SEATED] > 0.5:
                self.phase = "release"
                self.timer = 0
            return a

        if self.phase == "release":
            a[2] = 0.3
            a[4] = 1.0
            self.timer += 1
            if self.timer > 10:
                self.phase = "retreat"
                self.timer = 0
            return a

        # retreat
        a[2] = 1.0
        a[4] = 1.0
        self.timer += 1
        if self.timer > 12 or f[FEAT_SEATED] > 0.5:
            self.target = -1
            self.phase = "pick"
            self.timer = 0
        return a


@dataclass
class StrongScripted:
    """Privileged heuristic (true state). Calibration diagnostic only."""

    mode: str = "select"
    target: int = -1
    clutter_i: int = -1
    timer: int = 0
    spiral_t: int = 0
    servo: NearestFirstBaseline = field(default_factory=NearestFirstBaseline)
    sub: str = ""

    def reset(self) -> None:
        self.mode = "select"
        self.target = -1
        self.clutter_i = -1
        self.timer = 0
        self.spiral_t = 0
        self.servo.reset()
        self.sub = ""

    def act(self, e: Episode, obs: np.ndarray) -> np.ndarray:
        if self.mode == "select":
            self.target = self._next_peg(e)
            if self.target < 0:
                return np.array([0.0, 0.0, 0.2, 0.0, 1.0])
            self.clutter_i = self._blocking_clutter(e, self.target)
            if self.clutter_i >= 0:
                self.mode = "push"
                self.timer = 0
            else:
                p, q = e.peg_pose(self.target)
                if self._is_lying(q):
                    self.mode = "upright"
                    self.sub = "approach"
                    self.timer = 0
                else:
                    self.mode = "servo"
                    self.servo.reset()
                    self.spiral_t = 0

        if self.mode == "push":
            return self._push(e)
        if self.mode == "upright":
            return self._upright_regrasp(e)
        return self._servo(e)

    def _next_peg(self, e: Episode) -> int:
        hand_xy = e.data.qpos[0:2]
        best, best_d = -1, 1e9
        for i in range(e.n_pegs):
            if e.seated_now[i]:
                continue
            p, q = e.peg_pose(i)
            d = float(np.hypot(*(p[0:2] - hand_xy)))
            if self._is_lying(q):
                d += 0.04
            d += 0.5 * float(np.hypot(p[0] - e.slot_xy[i][0], p[1] - e.slot_xy[i][1]))
            if d < best_d:
                best_d, best = d, i
        return best

    def _blocking_clutter(self, e: Episode, peg_i: int) -> int:
        hx, hy = e.slot_xy[peg_i]
        for i in range(e.n_clutter):
            p, _ = e.clutter_pose(i)
            if float(np.hypot(p[0] - hx, p[1] - hy)) < 0.03 and p[2] < 0.04:
                return i
        return -1

    def _is_lying(self, quat: np.ndarray) -> bool:
        return abs(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)) < 0.5

    def _long_xy(self, quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat
        return np.array([2 * (x * z + w * y), 2 * (y * z - w * x)])

    def _push(self, e: Episode) -> np.ndarray:
        a = np.zeros(ACT_DIM)
        hand = e.data.qpos[0:4]
        p, _ = e.clutter_pose(self.clutter_i)
        hx, hy = e.slot_xy[self.target]
        away = np.array([p[0] - hx, p[1] - hy])
        n = float(np.linalg.norm(away)) + 1e-6
        away /= n
        goal = p[0:2] if self.timer < 18 else p[0:2] + away * 0.06
        z = 0.025 if self.timer < 18 else 0.02
        a[0] = float(np.clip((goal[0] - hand[0]) / DXYZ, -1, 1))
        a[1] = float(np.clip((goal[1] - hand[1]) / DXYZ, -1, 1))
        a[2] = float(np.clip((z - hand[2]) / DXYZ, -1, 1))
        a[4] = 1.0
        self.timer += 1
        if self.timer > 50:
            p2, q2 = e.peg_pose(self.target)
            self.mode = "upright" if self._is_lying(q2) else "servo"
            self.sub = "approach"
            self.timer = 0
            self.servo.reset()
            self.clutter_i = -1
        return a

    def _upright_regrasp(self, e: Episode) -> np.ndarray:
        a = np.zeros(ACT_DIM)
        i = self.target
        p, q = e.peg_pose(i)
        hand = e.data.qpos[0:4]
        if not self._is_lying(q) and p[2] > 0.02:
            self.mode = "servo"
            self.servo.reset()
            self.timer = 0
            return a
        long_xy = self._long_xy(q)
        want_yaw = (
            math.atan2(long_xy[1], long_xy[0])
            if float(np.linalg.norm(long_xy)) > 1e-6
            else float(hand[3])
        )
        if self.sub == "approach":
            a[0] = float(np.clip((p[0] - hand[0]) / DXYZ, -1, 1))
            a[1] = float(np.clip((p[1] - hand[1]) / DXYZ, -1, 1))
            a[2] = float(np.clip((0.04 - hand[2]) / DXYZ, -1, 1))
            yaw_err = (want_yaw - hand[3] + math.pi) % (2 * math.pi) - math.pi
            a[3] = float(np.clip(yaw_err / DYAW, -1, 1))
            a[4] = 1.0
            if abs(p[0] - hand[0]) < 0.008 and abs(p[1] - hand[1]) < 0.008:
                self.sub = "descend"
                self.timer = 0
            return a
        if self.sub == "descend":
            a[0] = float(np.clip((p[0] - hand[0]) / DXYZ, -1, 1))
            a[1] = float(np.clip((p[1] - hand[1]) / DXYZ, -1, 1))
            a[2] = float(np.clip((HAND_Z_MIN - hand[2]) / DXYZ, -1, 1))
            a[4] = 1.0
            self.timer += 1
            if self.timer > 25:
                self.sub = "close"
                self.timer = 0
            return a
        if self.sub == "close":
            a[4] = -1.0
            self.timer += 1
            if self.timer > 20:
                self.sub = "lift"
                self.timer = 0
            return a
        if self.sub == "lift":
            a[4] = -1.0
            a[2] = float(np.clip((0.07 - hand[2]) / DXYZ, -1, 1))
            self.timer += 1
            if hand[2] > 0.055 or self.timer > 35:
                self.sub = "drop"
                self.timer = 0
            return a
        a[2] = -0.5
        a[4] = 1.0
        self.timer += 1
        p2, q2 = e.peg_pose(i)
        if (not self._is_lying(q2)) or self.timer > 40:
            self.mode = "servo"
            self.servo.reset()
            self.timer = 0
        return a

    def _clean_obs(self, e: Episode) -> np.ndarray:
        """Noise-free, unbiased observation from true state (privileged)."""
        from .conditions import shape_one_hot

        d = e.data
        bx, by = e.c.board_xy
        hand = d.qpos[0:3].copy()
        hand[0] -= bx
        hand[1] -= by
        yaw = d.qpos[3]
        proprio = np.concatenate([
            hand * 10,
            [math.sin(yaw), math.cos(yaw)],
            d.qvel[0:3],
            [(d.qpos[4] + d.qpos[5]) * 50],
            np.zeros(3),
            np.zeros(3),
            [0.0],
        ])
        peg_feats = np.zeros(MAX_PEGS * PEG_FEAT)
        for i, peg in enumerate(e.c.pegs):
            ppos, pquat = e.peg_pose(i)
            hx, hy = e.slot_xy[i]
            prel = np.array([ppos[0] - hx, ppos[1] - hy, ppos[2] - BLOCK_H])
            peg_feats[i * PEG_FEAT : (i + 1) * PEG_FEAT] = np.concatenate([
                [1.0],
                prel * 10,
                pquat,
                shape_one_hot(peg.shape()),
                [peg.diameter() * 50],
                [(hx - bx) * 10, (hy - by) * 10, BLOCK_H * 10],
                [1.0 if e.seated_now[i] else 0.0],
            ])
        clutter_feats = np.zeros(MAX_CLUTTER * CLUTTER_FEAT)
        for i, cl in enumerate(e.c.clutter):
            ppos, pquat = e.clutter_pose(i)
            prel = np.array([ppos[0] - bx, ppos[1] - by, ppos[2]])
            clutter_feats[i * CLUTTER_FEAT : (i + 1) * CLUTTER_FEAT] = np.concatenate([
                [1.0],
                prel * 10,
                pquat,
                [max(cl.size_xy) * 50],
            ])
        return np.concatenate([proprio, e.last_action, peg_feats, clutter_feats]).astype(np.float64)

    def _servo(self, e: Episode) -> np.ndarray:
        i = self.target
        if i < 0 or e.seated_now[i]:
            self.mode = "select"
            self.target = -1
            return np.zeros(ACT_DIM)
        clean = self._clean_obs(e)
        for j in range(MAX_PEGS):
            if j != i:
                clean[PEG_TABLE0 + j * PEG_FEAT] = 0.0
        a = self.servo.act(clean)
        p, q = e.peg_pose(i)
        hx, hy = e.slot_xy[i]
        xy_err = float(np.hypot(p[0] - hx, p[1] - hy))
        if xy_err < 0.006 and self.servo.phase == "down":
            rad = min(0.0025, 0.0002 * (1 + self.spiral_t // 4))
            ang = 0.4 * self.spiral_t
            a[0] = float(np.clip((hx + rad * math.cos(ang) - e.data.qpos[0]) / DXYZ, -1, 1))
            a[1] = float(np.clip((hy + rad * math.sin(ang) - e.data.qpos[1]) / DXYZ, -1, 1))
            a[2] = -0.95
            a[4] = -1.0
            self.spiral_t += 1
        bottom = p[2] - PEG_LEN / 2
        if bottom <= 0.002 and xy_err <= e.r_hole[i] + 0.0015:
            a[:] = [0.0, 0.0, 1.0, 0.0, 1.0]
            self.timer += 1
            if self.timer > 12 or e.seated_now[i]:
                self.mode = "select"
                self.target = -1
                self.timer = 0
        if self.spiral_t > 300:
            self.mode = "select"
            self.target = -1
            self.spiral_t = 0
        return a
