"""Observation-only key searcher for v0.7.6.

Two eval paths:

1. **Actions-only (default, miner-path):** ``observe → act → step`` only. The policy
   seats pegs, press-probes, and yaw-searches with its own actions. No true-state
   reads and no private env writes (no ``_set_free``, qpos teleports, ``step_done``
   edits, or ``_peg_probe_frames`` fills).

2. **Fixture (``--fixture``):** privileged test setup seats the peg, clears prereqs,
   and finishes press-probe via env internals, then the yaw policy runs from obs.
   Use only to isolate observability of the closeness channel — **not** a miner-path
   claim. Macrocosmos rejected this for the hold review.

    NIST_PROFILE=v07 python tools/obs_key_searcher.py --seed 7 --per 8 --episodes 64
    NIST_PROFILE=v07 python tools/obs_key_searcher.py --fixture --seed 7 --per 8 --episodes 64
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tools"))

from nist_sim import conditions as C  # noqa: E402
from nist_sim.compound_v07 import (  # noqa: E402
    KEY_HOLD,
    KEY_PRESENT_MARK,
    KEY_PROBE_FRAMES,
    AssemblyIIIEpisode,
    compound_score_v07,
    sample_compound_v07,
)
from nist_sim.env import (  # noqa: E402
    ACT_DIM,
    BLOCK_H,
    DXYZ,
    DYAW,
    MAX_PEGS,
    PEG_FEAT,
    PEG_LEN,
    PROPRIO_DIM,
    _yaw_quat,
)

PEG0 = PROPRIO_DIM + ACT_DIM
TASK0 = 184
TASK_PEG_FEAT = 6


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _peg(obs: np.ndarray, i: int) -> np.ndarray:
    return obs[PEG0 + i * PEG_FEAT : PEG0 + (i + 1) * PEG_FEAT]


def _task(obs: np.ndarray, i: int) -> np.ndarray:
    return obs[TASK0 + i * TASK_PEG_FEAT : TASK0 + (i + 1) * TASK_PEG_FEAT]


def fixture_seat_key_peg(ep: AssemblyIIIEpisode, key_k: int) -> bool:
    """PRIVILEGED test-setup: seat peg, clear prereqs, finish press-probe.

    Writes ``_set_free``, hand ``qpos``, ``step_done``, and ``_peg_probe_frames``.
    Not miner-path. Kept for ``--fixture`` observability isolation only.
    """
    from nist_sim.env import PAD_CENTRE_Z

    s = ep.cs.steps[key_k]
    if s.kind != "key":
        return False
    i = s.peg
    hx, hy = ep.slot_xy[i]
    z = BLOCK_H - 0.0005 + PEG_LEN / 2
    # Peg must seat at a physically reachable yaw (non-round = slot symmetry ≈ key_yaw).
    # Only the **hand** starts offset so the policy must search the window.
    peg_yaw = float(s.key_yaw)
    hand_yaw0 = _wrap(s.key_yaw + math.copysign(max(0.55, 6.0 * s.key_tol), 1.0))
    ep._set_free(ep.peg_qadr[i], ep.peg_vadr[i], [hx, hy, z], _yaw_quat(peg_yaw))
    mujoco.mj_forward(ep.model, ep.data)
    ep.data.qpos[0] = hx + 0.04
    ep.data.qpos[1] = hy + 0.04
    ep.data.qpos[2] = 0.10
    ep.data.qpos[3] = float(np.clip(hand_yaw0, -2.8, 2.8))
    if hasattr(ep, "target") and len(ep.target) > 3:
        ep.target[0] = float(ep.data.qpos[0])
        ep.target[1] = float(ep.data.qpos[1])
        ep.target[2] = float(ep.data.qpos[2])
        ep.target[3] = float(ep.data.qpos[3])
    ep.data.qpos[4] = 0.012
    ep.data.qpos[5] = 0.012
    mujoco.mj_forward(ep.model, ep.data)
    for _ in range(14):
        if ep.done:
            break
        ep.step(np.array([0.0, 0.0, 0.4, 0.0, 1.0]))
    if not (ep.seated_now[i] or ep._peg_in_hole(i)):
        ep._set_free(ep.peg_qadr[i], ep.peg_vadr[i], [hx, hy, z], _yaw_quat(peg_yaw))
        mujoco.mj_forward(ep.model, ep.data)
        for _ in range(12):
            ep.step(np.array([0.0, 0.0, 0.4, 0.0, 1.0]))
    if not (ep.seated_now[i] or ep._peg_in_hole(i)):
        return False
    for p in s.prereq:
        ep.step_done[p] = True
        ep.step_done_at[p] = max(0, ep.steps)
        ep.step_in_order[p] = True
    if s.reveal_after is not None:
        ep.step_done[s.reveal_after] = True
        ep.step_done_at[s.reveal_after] = max(0, ep.steps)
    ep._pick_active_key(prefer=key_k)
    p, _ = ep.peg_pose(i)
    top = float(p[2] + PEG_LEN / 2)
    zg = top - 0.004 - PAD_CENTRE_Z
    for _ in range(100):
        if ep._peg_probe_frames[i] >= KEY_PROBE_FRAMES:
            break
        hand = ep.data.qpos[0:3]
        a = np.zeros(5)
        a[0] = float(np.clip((hx - hand[0]) / DXYZ, -1, 1))
        a[1] = float(np.clip((hy - hand[1]) / DXYZ, -1, 1))
        a[2] = float(np.clip((zg - hand[2]) / DXYZ, -1, 1))
        a[4] = -1.0
        ep.step(a)
    if ep._peg_probe_frames[i] < KEY_PROBE_FRAMES:
        ep._peg_probe_frames[i] = KEY_PROBE_FRAMES
        ep._peg_touched[i] = True
    for _ in range(10):
        ep.step(np.array([0.2, 0.2, 0.7, 0.0, 1.0]))
    # Start search from a hand yaw far from the window.
    ep.data.qpos[3] = float(np.clip(hand_yaw0, -2.8, 2.8))
    if hasattr(ep, "target") and len(ep.target) > 3:
        ep.target[3] = float(ep.data.qpos[3])
    mujoco.mj_forward(ep.model, ep.data)
    return True


@dataclass
class ObsKeySearcher:
    """Coarse yaw grid search + fine hill-climb on closeness, then open-hold.

    Obs + actions only. Assumes the peg is already seated and press-probed
    (closeness channel live). Used after fixture setup or after actions-only
    seat/probe phases.
    """

    phase: str = "seek"
    target: int = -1
    timer: int = 0
    close_ema: float = 0.0
    best_close: float = 0.0
    best_yaw: float = 0.0
    cand_i: int = 0
    candidates: list = field(default_factory=list)
    armed_jog: bool = False
    sweep_dir: float = 1.0

    def reset(self) -> None:
        self.phase = "seek"
        self.target = -1
        self.timer = 0
        self.close_ema = 0.0
        self.best_close = 0.0
        self.best_yaw = 0.0
        self.cand_i = 0
        self.candidates = []
        self.armed_jog = False

    def bind(self, peg_i: int) -> None:
        self.reset()
        self.target = int(peg_i)
        # Continuous sweep first (fast full-circle), then short dwell grid if needed.
        self.phase = "seek"
        self.sweep_dir = 1.0

    def _hand_xy(self, obs):
        return obs[0:2] / 10.0

    def _hand_z(self, obs):
        return float(obs[2] / 10.0)

    def _hand_yaw(self, obs):
        return math.atan2(float(obs[3]), float(obs[4]))

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        if self.target < 0:
            for i in range(MAX_PEGS):
                f = _peg(obs, i)
                t = _task(obs, i)
                if f[0] < 0.5 or t[4] > 0.5:
                    continue
                if (
                    (f[16] > 0.5 or float(t[0]) > 0.5)
                    and (
                        float(t[3]) > 0.12
                        or float(t[2]) >= KEY_PRESENT_MARK - 1e-3
                        or 0.70 < float(t[5]) < 1.60
                    )
                ):
                    self.bind(i)
                    break
            if self.target < 0:
                return np.array([0.0, 0.0, 0.3, 0.0, 1.0])
        return self._act_key(obs)

    def _act_key(self, obs: np.ndarray) -> np.ndarray:
        i = self.target
        f = _peg(obs, i)
        t = _task(obs, i)
        a = np.zeros(5)
        if f[0] < 0.5 or t[4] > 0.5:
            self.reset()
            return np.array([0.0, 0.0, 0.4, 0.0, 1.0])

        close = float(t[3])
        if float(t[2]) >= 1.0 - 1e-3:
            # Retarget pulse (1.0) — restart sweep. Key-present marker is 0.5.
            self.bind(i)
            close = float(t[3])
        self.close_ema = 0.55 * self.close_ema + 0.45 * close
        hz = self._hand_z(obs)
        hyaw = self._hand_yaw(obs)

        # Hold XY still while yaw-searching. Biased slot/peg estimates will yank a
        # seated peg out of the hole if we servo them (cm-scale residuals).
        if self.phase == "seek":
            # Continuous yaw sweep at full DYAW until closeness rises above noise.
            # Full circle ≈ 2π/0.08 ≈ 79 frames — much faster than a dwell grid.
            a[0:2] = 0.0
            a[2] = np.clip((0.052 - hz) / DXYZ, -0.5, 0.5)
            a[3] = float(self.sweep_dir)
            a[4] = 0.6
            self.timer += 1
            if self.close_ema > self.best_close:
                self.best_close = self.close_ema
                self.best_yaw = hyaw
            # Soft cue rises inside KEY_CLOSE_SOFT*tol; 0.22 clears the noise floor.
            if self.close_ema >= 0.22:
                self.phase = "fine"
                self.timer = 0
            elif self.timer > 95:
                # One full circle with no hit — reverse and try again from best.
                self.sweep_dir *= -1.0
                self.timer = 0
                if self.best_close >= 0.12:
                    self.phase = "fine"
            return a

        if self.phase == "fine":
            # Hill-climb on closeness around the rising edge.
            a[0:2] = 0.0
            a[2] = np.clip((0.050 - hz) / DXYZ, -0.4, 0.4)
            if self.close_ema < self.best_close - 0.04:
                self.sweep_dir *= -1.0
            step = self.sweep_dir * (0.030 if self.close_ema < 0.70 else 0.010)
            goal = _wrap(hyaw + step)
            a[3] = float(np.clip(_wrap(goal - hyaw) / DYAW, -1, 1))
            a[4] = 0.4
            if self.close_ema > self.best_close:
                self.best_close = self.close_ema
                self.best_yaw = hyaw
            self.timer += 1
            if self.close_ema >= 0.78 and self.timer > 2:
                self.phase = "hold"
                self.timer = 0
            elif self.timer > 80:
                self.phase = "seek"
                self.timer = 0
            return a

        if self.phase == "jog":
            # Leave window once for index arming (yaw only).
            a[0:2] = 0.0
            a[2] = np.clip((0.050 - hz) / DXYZ, -0.3, 0.3)
            a[3] = float(np.clip(self.sweep_dir * 0.8, -1, 1))
            a[4] = 0.5
            self.timer += 1
            if self.close_ema < 0.25 or self.timer > 14:
                self.armed_jog = True
                self.phase = "reenter"
                self.timer = 0
            return a

        if self.phase == "reenter":
            a[0:2] = 0.0
            a[2] = np.clip((0.050 - hz) / DXYZ, -0.3, 0.3)
            a[3] = float(np.clip(_wrap(self.best_yaw - hyaw) / DYAW, -1, 1))
            a[4] = 0.3
            self.timer += 1
            if self.close_ema >= 0.80 or self.timer > 40:
                self.phase = "hold"
                self.timer = 0
            return a

        # hold: open, lift slightly, keep yaw — no XY chase.
        a[0:2] = 0.0
        a[2] = np.clip((0.065 - hz) / DXYZ, -0.2, 0.6)
        a[3] = float(np.clip(_wrap(self.best_yaw - hyaw) / DYAW, -0.5, 0.5))
        a[4] = 1.0
        self.timer += 1
        if self.close_ema < 0.35 and self.timer > 4:
            self.phase = "fine"
            self.timer = 0
        elif self.timer > KEY_HOLD + 12 and not self.armed_jog:
            # Hold didn't latch — need leave/reenter arming.
            self.phase = "jog"
            self.timer = 0
        elif self.timer > KEY_HOLD + 24:
            self.phase = "seek"
            self.timer = 0
        return a


@dataclass
class ActionsOnlyKeyPolicy:
    """Miner-path policy: seat + press-probe + key search from observations only.

    Never reads true state. Never writes env internals — only returns actions.

    Important: task feat 0 is the **insert** done flag for that peg, not key done.
    After a successful seat feat 0 becomes 1.0 — that is when key work should start.
    Key completion is observed via blocked/inactive (feat 4) once the active key rotates.
    """

    mode: str = "pick"  # pick | insert | probe | key
    target: int = -1
    phase: str = "goto"
    timer: int = 0
    yaw_dir: float = 1.0
    spiral_i: int = 0
    freeze_xy: bool = False
    # Grasp-align centre (hand XY at first holding press). Archimedean spiral around
    # this — never chase the biased slot estimate (cm-scale obs bias > tight r_hole).
    align_xy: np.ndarray | None = None
    spiral_theta: float = 0.0
    skip: set = field(default_factory=set)
    key_pol: ObsKeySearcher = field(default_factory=ObsKeySearcher)

    def reset(self) -> None:
        # Soft reset keeps skip set so we rotate through pegs within an episode.
        self.mode = "pick"
        self.target = -1
        self.phase = "goto"
        self.timer = 0
        self.yaw_dir = 1.0
        self.spiral_i = 0
        self.freeze_xy = False
        self.align_xy = None
        self.spiral_theta = 0.0
        self.key_pol.reset()

    def hard_reset(self) -> None:
        self.skip = set()
        self.reset()

    def _hand_xy(self, obs):
        return obs[0:2] / 10.0

    def _hand_z(self, obs):
        return float(obs[2] / 10.0)

    def _finger_gap(self, obs):
        # Proprio finger gap ×50 in obs[5] (see README); small ⇒ likely grasping.
        return float(obs[5]) / 50.0 if obs.shape[0] > 5 else 0.02

    def _looks_like_key(self, t: np.ndarray, f: np.ndarray) -> bool:
        s2, s3, s5 = float(t[2]), float(t[3]), float(t[5])
        if float(t[4]) > 0.5:
            return False
        # key_tol*10 when collar/stack flag slot is free (~0.9–1.5 under v0.7.6 KEY_TOL).
        if 0.70 < s5 < 1.60:
            return True
        # v0.7.6 key-present marker (0.5) or retarget pulse (1.0).
        if f[16] > 0.5 and s2 >= KEY_PRESENT_MARK - 1e-3:
            return True
        # Post-probe closeness (threshold above far-yaw noise floor).
        if f[16] > 0.5 and s3 > 0.12:
            return True
        return False

    def _choose(self, obs: np.ndarray) -> None:
        hand = self._hand_xy(obs)
        gap = self._finger_gap(obs)
        best, best_d, best_mode = -1, 1e9, "insert"
        for i in range(MAX_PEGS):
            if i in self.skip:
                continue
            f = _peg(obs, i)
            t = _task(obs, i)
            if f[0] < 0.5:
                continue
            # Seated peg with key cues — highest priority (insert feat0 may be 1.0).
            # Present-mark alone is enough to enter key: closeness is ~0 at far yaw, so
            # gating on t[3] would never start the sweep.
            if (f[16] > 0.5 or float(t[0]) > 0.5) and self._looks_like_key(t, f):
                slot = f[13:15] / 10.0
                d = float(np.hypot(*(slot - hand))) - 0.80
                mode = "key"
                if d < best_d:
                    best, best_d, best_mode = i, d, mode
                continue
            # Seated / insert latched, no key cue yet — brief probe to discover key-present.
            if f[16] > 0.5 and float(t[0]) > 0.5 and float(t[4]) < 0.5:
                slot = f[13:15] / 10.0
                d = float(np.hypot(*(slot - hand))) - 0.25
                if d < best_d:
                    best, best_d, best_mode = i, d, "probe"
                continue
            # Unseated unfinished insert.
            if float(t[0]) < 0.5 and float(t[1]) < 0.5 and f[16] < 0.5:
                slot = f[13:15] / 10.0
                peg_est = slot + f[1:3] / 10.0
                d = float(np.hypot(*(peg_est - hand)))
                # Prefer already-in-hand / near-hand pegs (grasped starts).
                if float(np.hypot(*(peg_est - hand))) < 0.025 and gap < 0.018:
                    d -= 0.50
                # Prefer near-hole (small peg-vs-slot residual in obs).
                if float(np.hypot(f[1], f[2])) < 0.35:
                    d -= 0.20
                if d < best_d:
                    best, best_d, best_mode = i, d, "insert"
        if best < 0 and self.skip:
            # All skips exhausted — clear and retry.
            self.skip = set()
            return self._choose(obs)
        self.target = best
        self.mode = best_mode if best >= 0 else "pick"
        self.phase = "goto"
        self.timer = 0
        self.spiral_i = 0
        if self.mode == "key" and best >= 0:
            self.key_pol.bind(best)

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        if self.target < 0 or self.mode == "pick":
            self._choose(obs)
            if self.target < 0:
                return np.array([0.0, 0.0, 0.3, 0.0, 1.0])
        if self.mode == "insert":
            return self._act_insert(obs)
        if self.mode == "probe":
            return self._act_probe(obs)
        return self._act_key(obs)

    def _give_up_target(self) -> np.ndarray:
        if self.target >= 0:
            self.skip.add(int(self.target))
        self.reset()
        return np.array([0.0, 0.0, 0.4, 0.0, 1.0])

    def _act_insert(self, obs: np.ndarray) -> np.ndarray:
        i = self.target
        f = _peg(obs, i)
        t = _task(obs, i)
        if f[0] < 0.5:
            return self._give_up_target()
        # Seated / insert latched → key if present-mark, else probe for identity.
        # Seated obs can flicker for a frame after release — do NOT skip the peg.
        # Do NOT gate key on closeness: far-yaw closeness is ~0 by design.
        if f[16] > 0.5 or float(t[0]) > 0.5:
            if self._looks_like_key(t, f):
                self.mode = "key"
                self.key_pol.bind(i)
                self.timer = 0
                return self.act(obs)
            if float(t[0]) > 0.5:
                self.mode = "probe"
                self.phase = "goto"
                self.timer = 0
                return self.act(obs)
            # Seated flicker without insert latch: try a short re-seat press.
            if self.phase != "release" and f[16] < 0.5:
                self.phase = "place"
                self.timer = 0
        slot = f[13:15] / 10.0
        peg = slot + f[1:3] / 10.0
        hand = self._hand_xy(obs)
        hz = self._hand_z(obs)
        gap = self._finger_gap(obs)
        a = np.zeros(5)
        # Already holding (grasped start): jaws nearly closed. Gap threshold is loose —
        # some grasped starts sit at ~0.016. Do NOT chase biased peg_est (cm-scale noise).
        holding = gap < 0.020
        if self.phase == "goto" and holding:
            self.phase = "insert"
            self.timer = 0
            self.freeze_xy = True
            if self.align_xy is None:
                self.align_xy = np.array(hand, dtype=np.float64)
        if self.phase == "goto":
            err = peg - hand
            a[0:2] = np.clip(err / DXYZ, -1, 1)
            a[2] = np.clip((0.05 - hz) / DXYZ, -1, 1)
            a[4] = 1.0
            self.timer += 1
            if float(np.hypot(*err)) < 0.012:
                self.phase = "grasp"
                self.timer = 0
            elif self.timer > 80:
                return self._give_up_target()
        elif self.phase == "grasp":
            err = peg - hand
            a[0:2] = np.clip(err / DXYZ, -1, 1) * 0.5
            a[2] = np.clip((0.010 - hz) / DXYZ, -1, 1)
            a[4] = -1.0
            self.timer += 1
            if self.timer > 18:
                self.phase = "lift"
                self.timer = 0
        elif self.phase == "lift":
            a[2] = 0.85
            a[4] = -1.0
            self.timer += 1
            if self.timer > 22 or hz > 0.065:
                self.phase = "place"
                self.timer = 0
        elif self.phase == "place":
            # Far picks only (grasped uses continuous insert spiral).
            ang = 1.25 * self.spiral_i
            rad = 0.0025 * (1 + self.spiral_i // 2)
            goal = slot + np.array([rad * math.cos(ang), rad * math.sin(ang)])
            err = goal - hand
            a[0:2] = np.clip(err / DXYZ, -1, 1)
            a[2] = np.clip((0.034 - hz) / DXYZ, -1, 1)
            a[3] = float(self.yaw_dir) * 0.35
            a[4] = -1.0
            self.timer += 1
            if float(np.hypot(*err)) < 0.0020:
                self.phase = "insert"
                self.timer = 0
            elif self.timer > 18:
                self.spiral_i += 1
                self.timer = 0
                if self.spiral_i > 48:
                    return self._give_up_target()
        elif self.phase == "insert":
            if holding and self.align_xy is None:
                self.align_xy = np.array(hand, dtype=np.float64)
                self.freeze_xy = True
                self.spiral_theta = 0.0
            if self.freeze_xy and self.align_xy is not None:
                # Grasped starts are spawned inside r_hole (v0.7.6). Freeze XY, press,
                # then open-lift so the peg drops in. Tiny spiral only on retry.
                ang = 1.2 * self.spiral_i
                rad = 0.0006 * self.spiral_i
                goal = self.align_xy + np.array([rad * math.cos(ang), rad * math.sin(ang)])
                err = goal - hand
                a[0:2] = np.clip(err / DXYZ, -0.35, 0.35)
                a[2] = -1.0
                a[3] = float(self.yaw_dir) * (0.10 if self.spiral_i > 0 else 0.0)
                a[4] = -1.0
                self.timer += 1
                if self.timer >= 22:
                    self.phase = "release"
                    self.timer = 0
            else:
                ang = 0.9 * self.spiral_i + 0.35 * self.timer
                rad = 0.0018 * (1 + self.spiral_i // 3)
                goal = slot + np.array([rad * math.cos(ang), rad * math.sin(ang)])
                err = goal - hand
                a[0:2] = np.clip(err / DXYZ, -0.55, 0.55)
                a[2] = -1.0
                a[3] = float(self.yaw_dir) * 0.40
                a[4] = -1.0
                self.timer += 1
                if self.timer > 20:
                    self.phase = "release"
                    self.timer = 0
                elif self.timer % 15 == 0:
                    self.spiral_i += 1
                    self.yaw_dir *= -1.0
                    if self.spiral_i > 48:
                        return self._give_up_target()
        else:
            # release: open first while staying low so the peg can drop fully into the
            # hole; lifting while still pinched wedges many pegs at bot≈12 mm.
            a[0:2] = 0.0
            a[3] = 0.0
            a[4] = 1.0
            if self.timer < 14:
                a[2] = np.clip((0.028 - hz) / DXYZ, -0.4, 0.35)
            else:
                a[2] = 0.75
            self.timer += 1
            if f[16] > 0.5 or float(t[0]) > 0.5:
                if self._looks_like_key(t, f):
                    self.mode = "key"
                    self.key_pol.bind(i)
                else:
                    self.mode = "probe"
                self.phase = "goto"
                self.timer = 0
            elif self.timer > 48:
                if self.freeze_xy:
                    # Retry once with a tiny XY nudge; then abort.
                    if self.spiral_i < 6:
                        self.phase = "insert"
                        self.timer = 0
                        self.spiral_i += 1
                        self.yaw_dir *= -1.0
                    else:
                        return self._give_up_target()
                else:
                    self.phase = "place"
                    self.timer = 0
                    self.spiral_i += 1
                    self.yaw_dir *= -1.0
                    if self.spiral_i > 48:
                        return self._give_up_target()
        return a

    def _act_probe(self, obs: np.ndarray) -> np.ndarray:
        """Press on a seated peg until key identity / closeness appears."""
        i = self.target
        f = _peg(obs, i)
        t = _task(obs, i)
        if f[0] < 0.5:
            return self._give_up_target()
        if float(t[4]) > 0.5:
            # Inactive / blocked — key done or not the active key.
            return self._give_up_target()
        # Seated flicker: if insert is done / key-present, keep pressing near the slot.
        if f[16] < 0.5 and float(t[0]) < 0.5 and not self._looks_like_key(t, f):
            self.mode = "insert"
            self.phase = "goto"
            self.timer = 0
            return self._act_insert(obs)
        # Present-mark (post-touch + reveal) ⇒ start yaw sweep even if closeness is 0.
        # Closeness is a *local* cue around the window; far yaw reads ~0 by design.
        if self._looks_like_key(t, f):
            self.mode = "key"
            self.key_pol.bind(i)
            self.timer = 0
            return self.key_pol.act(obs)
        slot = f[13:15] / 10.0
        hand = self._hand_xy(obs)
        hz = self._hand_z(obs)
        a = np.zeros(5)
        # Move over the slot estimate first. Freezing XY when the hand is elsewhere
        # (e.g. peg seated as a side-effect while working another peg) never touches,
        # so key-present never appears. Once near the slot, freeze and press.
        e = slot - hand
        dist = float(np.hypot(*e))
        if dist > 0.012:
            a[0:2] = np.clip(e / DXYZ, -1, 1)
            a[2] = np.clip((0.040 - hz) / DXYZ, -0.7, 0.7)
            a[4] = 0.3
        else:
            a[0:2] = np.clip(e / DXYZ, -0.25, 0.25)
            a[2] = np.clip((0.026 - hz) / DXYZ, -0.85, 0.5)
            a[4] = -1.0
        self.timer += 1
        if self.timer > 140:
            # Still no present-mark → not an active key (or press failed).
            return self._give_up_target()
        elif self.timer > 55 and dist <= 0.012 and self.timer % 25 < 5:
            # Brief open/re-press to re-trigger touch contact.
            a[2] = 0.35
            a[4] = 1.0
        return a

    def _act_key(self, obs: np.ndarray) -> np.ndarray:
        i = self.target
        f = _peg(obs, i)
        t = _task(obs, i)
        if f[0] < 0.5:
            return self._give_up_target()
        # feat 0 is insert-done, NOT key-done. Key done / inactive → blocked feat 4.
        if float(t[4]) > 0.5:
            return self._give_up_target()
        # Lost seat and insert not latched → re-seat. Do NOT fall back to probe just
        # because closeness is low: far-yaw closeness is supposed to be ~0.
        if f[16] < 0.5 and float(t[0]) < 0.5:
            self.mode = "insert"
            self.phase = "goto"
            self.timer = 0
            return self._act_insert(obs)
        # Present-mark gone while insert still latched → need another press for touch.
        if not self._looks_like_key(t, f) and float(t[3]) <= 0.08:
            self.mode = "probe"
            self.phase = "goto"
            self.timer = 0
            return self._act_probe(obs)
        if self.key_pol.target != i:
            self.key_pol.bind(i)
        a = self.key_pol.act(obs)
        if self.key_pol.target < 0:
            return self._give_up_target()
        return a


def run_eval(
    seed: int,
    per: int,
    episodes: int,
    start: int = 0,
    *,
    budget: int = 500,
) -> dict:
    """Fixture-assisted yaw search (privileged seat+probe). Not miner-path."""
    _, conds = sample_compound_v07(seed, per)
    n = min(len(conds), start + episodes)
    keys_total = 0
    keys_done = 0
    per_key_frames: list[int] = []
    for i in range(start, n):
        c = conds[i]
        key_idx = [k for k, s in enumerate(c.compound.steps) if s.kind == "key"]
        for key_k in key_idx:
            keys_total += 1
            r1, r2 = (np.random.default_rng(s) for s in C.episode_seed_sequence(seed, i).spawn(2))
            ep = AssemblyIIIEpisode(c, r1)
            if not fixture_seat_key_peg(ep, key_k):
                per_key_frames.append(0)
                continue
            pol = ObsKeySearcher()
            pol.bind(ep.cs.steps[key_k].peg)
            used = 0
            while not ep.done and used < budget and ep.steps < ep.c.max_steps:
                o = ep.observe(r2)
                a = pol.act(o)
                ep.step(a)
                used += 1
                if ep.step_done[key_k]:
                    break
            per_key_frames.append(used)
            if ep.step_done[key_k]:
                keys_done += 1
    rate = keys_done / max(1, keys_total)
    return dict(
        seed=seed,
        episodes=n - start,
        keys_total=keys_total,
        keys_done=keys_done,
        key_rate=rate,
        mean_frames=float(np.mean(per_key_frames)) if per_key_frames else 0.0,
        median_frames=float(np.median(per_key_frames)) if per_key_frames else 0.0,
        path="fixture",
    )


def run_eval_actions_only(
    seed: int,
    per: int,
    episodes: int,
    start: int = 0,
) -> dict:
    """Full-episode miner-path eval: observe → act → step only.

    Reports key completions as keys_done/keys_total (key *steps*, not episodes)
    and mean episode score.
    """
    _, conds = sample_compound_v07(seed, per)
    n = min(len(conds), start + episodes)
    keys_total = 0
    keys_done = 0
    scores: list[float] = []
    for i in range(start, n):
        c = conds[i]
        key_idx = [k for k, s in enumerate(c.compound.steps) if s.kind == "key"]
        keys_total += len(key_idx)
        r1, r2 = (np.random.default_rng(s) for s in C.episode_seed_sequence(seed, i).spawn(2))
        ep = AssemblyIIIEpisode(c, r1)
        pol = ActionsOnlyKeyPolicy()
        while not ep.done:
            o = ep.observe(r2)
            ep.step(np.clip(pol.act(o), -1.0, 1.0))
        for k in key_idx:
            if ep.step_done[k]:
                keys_done += 1
        scores.append(float(compound_score_v07(ep)["score"]))
    rate = keys_done / max(1, keys_total)
    return dict(
        seed=seed,
        episodes=n - start,
        keys_total=keys_total,
        keys_done=keys_done,
        key_rate=rate,
        score_mean=float(np.mean(scores)) if scores else 0.0,
        score_std=float(np.std(scores)) if scores else 0.0,
        path="actions_only",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument(
        "--fixture",
        action="store_true",
        help="Privileged seat+probe fixture then yaw search (not miner-path).",
    )
    a = ap.parse_args()
    os.environ["NIST_PROFILE"] = "v07"
    if a.fixture:
        r = run_eval(a.seed, a.per, a.episodes, a.start, budget=a.budget)
        print(
            f"obs_key_searcher[fixture] seed={r['seed']} eps={r['episodes']} "
            f"keys={r['keys_done']}/{r['keys_total']} rate={r['key_rate']:.3f} "
            f"mean_frames={r['mean_frames']:.1f} median_frames={r['median_frames']:.1f}"
        )
    else:
        r = run_eval_actions_only(a.seed, a.per, a.episodes, a.start)
        print(
            f"obs_key_searcher[actions_only] seed={r['seed']} eps={r['episodes']} "
            f"keys={r['keys_done']}/{r['keys_total']} rate={r['key_rate']:.3f} "
            f"score_mean={r['score_mean']:.3f} score_std={r['score_std']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
