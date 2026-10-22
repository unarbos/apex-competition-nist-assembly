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
# 0 present | 1:4 rel*10 | 4:8 quat | 8:10 est slot yaw sin/cos | 10 outer size*50 | 11 zero
# | 12 size | 13:16 slot*10 | 16 seated
FEAT_PRESENT = 0
FEAT_REL = slice(1, 4)
FEAT_QUAT = slice(4, 8)
FEAT_SLOT_YAW = slice(8, 10)
FEAT_OUTER = 10
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


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _yaw_cmd(f: np.ndarray, gain: float = 0.6) -> float:
    """Yaw command that turns the (estimated) peg yaw onto the (estimated) slot yaw."""
    q = f[FEAT_QUAT]
    peg_yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
    slot_yaw = math.atan2(float(f[8]), float(f[9]))
    return float(np.clip(gain * _wrap(slot_yaw - peg_yaw) / DYAW, -1, 1))


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
            a[3] = _yaw_cmd(f)
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
            a[3] = _yaw_cmd(f)
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
    """Miner-grade hand-coded controller using **only** observations.

    Strong-miner recipe:
    - EMA on wrist wrench (Fz/Fx/Fy, pad force)
    - Prefer near-hole / already-held pegs (partial-credit greedy)
    - Skip pick when pad force shows a grasped-start hold
    - Coarse approach to the (biased) slot estimate, then servo on relative
      peg−slot while descending (residual bias remains when peg/slot biases differ)
    - On sustained lip jam: expanding spiral + lateral compliance (−Fx/−Fy),
      yaw wiggle, jam count, retry with phase shift
    - Drop / release and retarget

    Anti-scripting gate: must stay ≤ ~0.25 after difficulty tuning.
    """

    target: int = -1
    phase: str = "goto"
    timer: int = 0
    spiral_t: int = 0
    pad_ema: float = 0.0
    fz_ema: float = 0.0
    fx_ema: float = 0.0
    fy_ema: float = 0.0
    force_ema: float = 0.0
    free_fz: float = 0.0
    jam_count: int = 0
    holding: bool = False
    searching: bool = False
    retry: int = 0

    def reset(self) -> None:
        self.target = -1
        self.phase = "goto"
        self.timer = 0
        self.spiral_t = 0
        self.pad_ema = 0.0
        self.fz_ema = 0.0
        self.fx_ema = 0.0
        self.fy_ema = 0.0
        self.force_ema = 0.0
        self.free_fz = 0.0
        self.jam_count = 0
        self.holding = False
        self.searching = False
        self.retry = 0

    def _choose(self, obs: np.ndarray, raw_pad: float) -> int:
        hand = _hand_xy(obs)
        best, best_d = -1, 1e9
        for i in range(MAX_PEGS):
            f = _peg_slice(obs, i)
            if f[FEAT_PRESENT] < 0.5 or f[FEAT_SEATED] > 0.5:
                continue
            peg_xy = f[FEAT_SLOT][0:2] / 10.0 + f[FEAT_REL][0:2] / 10.0
            d = float(np.hypot(*(peg_xy - hand)))
            # Prefer near-hole pegs (partial credit).
            d += 0.25 * float(np.linalg.norm(f[FEAT_REL][0:2] / 10.0))
            if raw_pad > 0.08:
                d -= 0.25 * max(0.0, 0.03 - d)
            if d < best_d:
                best_d, best = d, i
        return best

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        a = np.zeros(ACT_DIM)
        raw_pad = float(obs[PROP_PAD])
        self.pad_ema = 0.75 * self.pad_ema + 0.25 * raw_pad
        fv = obs[PROP_FORCE]
        self.force_ema = 0.75 * self.force_ema + 0.25 * float(np.linalg.norm(fv))
        self.fz_ema = 0.75 * self.fz_ema + 0.25 * float(abs(fv[2]))
        self.fx_ema = 0.7 * self.fx_ema + 0.3 * float(fv[0])
        self.fy_ema = 0.7 * self.fy_ema + 0.3 * float(fv[1])
        hand = np.array([*_hand_xy(obs), _hand_z(obs)])

        if (
            self.target < 0
            or _peg_slice(obs, self.target)[FEAT_SEATED] > 0.5
            or _peg_slice(obs, self.target)[FEAT_PRESENT] < 0.5
        ):
            self.target = self._choose(obs, raw_pad)
            self.phase = "goto"
            self.timer = 0
            self.spiral_t = 0
            self.jam_count = 0
            self.searching = False
            # Grasped start: already holding — go straight to insert.
            if self.target >= 0 and raw_pad > 0.08 and hand[2] > 0.018:
                self.holding = True
                self.phase = "down"
                self.free_fz = self.fz_ema

        if self.target < 0:
            a[4] = 1.0
            return a

        f = _peg_slice(obs, self.target)
        rel = f[FEAT_REL] / 10.0
        slot_xy = f[FEAT_SLOT][0:2] / 10.0
        qz = float(f[7])

        if self.phase == "goto":
            # Coarse approach to biased slot estimate (same prior as baseline).
            a[0] = float(np.clip((slot_xy[0] - hand[0]) / DXYZ, -1, 1))
            a[1] = float(np.clip((slot_xy[1] - hand[1]) / DXYZ, -1, 1))
            a[2] = float(np.clip((0.07 - hand[2]) / DXYZ, -1, 1))
            a[3] = _yaw_cmd(f)
            a[4] = -1.0
            if abs(slot_xy[0] - hand[0]) < 0.006 and abs(slot_xy[1] - hand[1]) < 0.006:
                self.phase = "down"
                self.timer = 0
                self.free_fz = self.fz_ema
            if abs(float(rel[0])) < 0.004 and abs(float(rel[1])) < 0.004:
                self.phase = "down"
                self.timer = 0
                self.free_fz = self.fz_ema
            return a

        if self.phase == "down":
            # Primary: servo on relative peg−slot (residual bias remains).
            a[0] = float(np.clip(-rel[0] / DXYZ, -1, 1))
            a[1] = float(np.clip(-rel[1] / DXYZ, -1, 1))
            # Sustained lip jam → expanding spiral + compliance.
            if self.timer > 18 and self.fz_ema > self.free_fz + 0.12:
                self.searching = True
            if self.searching:
                rad = min(0.009, 0.0004 + 0.00016 * self.spiral_t)
                ang = 0.5 * self.spiral_t + 0.3 * self.retry
                a[0] = float(np.clip(a[0] + rad * math.cos(ang) / DXYZ - 0.8 * self.fx_ema, -1, 1))
                a[1] = float(np.clip(a[1] + rad * math.sin(ang) / DXYZ - 0.8 * self.fy_ema, -1, 1))
                self.spiral_t += 1
                if self.fz_ema > self.free_fz + 0.12:
                    a[2] = 0.25
                    self.jam_count += 1
                else:
                    a[2] = -0.8
            else:
                a[2] = -0.85
            a[3] = float(np.clip(_yaw_cmd(f) + 0.1 * math.sin(0.8 * self.timer), -1, 1))
            a[4] = -1.0
            self.timer += 1
            if (
                f[FEAT_SEATED] > 0.5
                or float(rel[2]) < PEG_LEN / 2 - BLOCK_H + 0.005
                or self.timer > 70
            ):
                self.phase = "release"
                self.timer = 0
            elif self.jam_count > 24:
                self.retry += 1
                self.jam_count = 0
                self.spiral_t = 0
                if self.retry > 2:
                    self.phase = "release"
                    self.timer = 0
                else:
                    self.phase = "goto"
                    self.timer = 0
                    self.searching = True
            return a

        # release / retreat
        a[2] = 0.9
        a[4] = 1.0
        self.holding = False
        self.timer += 1
        if self.timer > 15:
            self.target = -1
            self.phase = "goto"
            self.retry = 0
            self.searching = False
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
                [math.sin(e.slot_yaw[i]), math.cos(e.slot_yaw[i]), 2 * peg.spec.outer_radius() * 50, 0.0],
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
