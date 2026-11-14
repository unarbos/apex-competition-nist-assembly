"""Observation-only key searcher for v0.7.5 (no true-state reads / writes / teleports).

v0.7.5 keys index **hand yaw** against a hidden window. After fixture seat+probe, the
policy sweeps wrist yaw and hill-climbs on the noisy closeness channel (and wrist detent),
then leave/reenter-arms and open-holds. Absolute key yaw is never read.

    NIST_PROFILE=v07 python tools/obs_key_searcher.py --seed 7 --per 8 --episodes 64
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
    KEY_PROBE_FRAMES,
    AssemblyIIIEpisode,
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
    """Test-setup: seat peg, clear prereqs, finish press-probe. Policy finds the yaw."""
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
    """Coarse yaw grid search + fine hill-climb on closeness, then open-hold."""

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
        # Coarse grid over ±π at ~2.5× typical tol spacing.
        step = 0.12
        self.candidates = [ _wrap(-math.pi + k * step) for k in range(int(2 * math.pi / step)) ]
        self.phase = "seek"

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
                if f[16] > 0.5 or float(t[3]) > 0.0 or float(t[5]) > 0.2:
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
        if float(t[2]) > 0.5:
            # Retarget — restart grid search.
            self.bind(i)
            close = float(t[3])
        self.close_ema = 0.6 * self.close_ema + 0.4 * close
        slot = f[13:15] / 10.0
        peg_xy = slot + f[1:3] / 10.0
        hand = self._hand_xy(obs)
        hz = self._hand_z(obs)
        hyaw = self._hand_yaw(obs)

        if self.phase == "seek":
            if self.cand_i >= len(self.candidates):
                # Restart from best, finer.
                self.candidates = [
                    _wrap(self.best_yaw + d) for d in (-0.20, -0.12, -0.06, 0.0, 0.06, 0.12, 0.20)
                ]
                self.cand_i = 0
            goal = self.candidates[self.cand_i]
            e = peg_xy - hand
            a[0:2] = np.clip(e / DXYZ, -1, 1)
            a[2] = np.clip((0.050 - hz) / DXYZ, -1, 1)
            a[3] = float(np.clip(_wrap(goal - hyaw) / DYAW, -1, 1))
            a[4] = 0.5
            self.timer += 1
            if abs(_wrap(goal - hyaw)) < 0.05 or self.timer > 25:
                # Dwell to read closeness.
                a[3] = float(np.clip(_wrap(goal - hyaw) / DYAW, -0.4, 0.4))
                if self.timer > 8:
                    if self.close_ema > self.best_close:
                        self.best_close = self.close_ema
                        self.best_yaw = hyaw
                    if self.close_ema >= 0.50:
                        self.phase = "fine"
                        self.timer = 0
                    else:
                        self.cand_i += 1
                        self.timer = 0
            return a

        if self.phase == "fine":
            # Hill-climb around best_yaw.
            e = peg_xy - hand
            a[0:2] = np.clip(e / DXYZ, -0.5, 0.5)
            a[2] = np.clip((0.048 - hz) / DXYZ, -0.5, 0.5)
            if self.close_ema < self.best_close - 0.05:
                self.sweep_dir *= -1.0
            step = self.sweep_dir * (0.035 if self.close_ema < 0.75 else 0.012)
            goal = _wrap(hyaw + step)
            a[3] = float(np.clip(_wrap(goal - hyaw) / DYAW, -1, 1))
            a[4] = 0.3
            if self.close_ema > self.best_close:
                self.best_close = self.close_ema
                self.best_yaw = hyaw
            self.timer += 1
            if self.close_ema >= 0.82 and self.timer > 3:
                # Try hold first (D-flat auto-arms). If hold fails, jog to arm leave/reenter.
                self.phase = "hold"
                self.timer = 0
            elif self.timer > 100:
                self.phase = "seek"
                self.timer = 0
            return a

        if self.phase == "jog":
            # Leave window once for index arming.
            e = peg_xy - hand
            a[0:2] = np.clip(e / DXYZ, -0.4, 0.4)
            a[2] = np.clip((0.050 - hz) / DXYZ, -0.4, 0.4)
            a[3] = float(np.clip(self.sweep_dir * 0.8, -1, 1))
            a[4] = 0.5
            self.timer += 1
            if self.close_ema < 0.25 or self.timer > 14:
                self.armed_jog = True
                self.phase = "reenter"
                self.timer = 0
            return a

        if self.phase == "reenter":
            e = peg_xy - hand
            a[0:2] = np.clip(e / DXYZ, -0.4, 0.4)
            a[2] = np.clip((0.050 - hz) / DXYZ, -0.4, 0.4)
            a[3] = float(np.clip(_wrap(self.best_yaw - hyaw) / DYAW, -1, 1))
            a[4] = 0.3
            self.timer += 1
            if self.close_ema >= 0.85 or self.timer > 50:
                self.phase = "hold"
                self.timer = 0
            return a

        # hold
        e = slot - hand
        a[0:2] = np.clip(e / DXYZ, -0.25, 0.25)
        a[2] = np.clip((0.065 - hz) / DXYZ, -0.3, 0.6)
        a[3] = float(np.clip(_wrap(self.best_yaw - hyaw) / DYAW, -0.5, 0.5))
        a[4] = 1.0
        self.timer += 1
        if self.close_ema < 0.40 and self.timer > 4:
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


def run_eval(seed: int, per: int, episodes: int, start: int = 0, *, budget: int = 500) -> dict:
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
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--budget", type=int, default=500)
    a = ap.parse_args()
    os.environ["NIST_PROFILE"] = "v07"
    r = run_eval(a.seed, a.per, a.episodes, a.start, budget=a.budget)
    print(
        f"obs_key_searcher seed={r['seed']} eps={r['episodes']} "
        f"keys={r['keys_done']}/{r['keys_total']} rate={r['key_rate']:.3f} "
        f"mean_frames={r['mean_frames']:.1f} median_frames={r['median_frames']:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
