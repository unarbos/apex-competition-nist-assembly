"""Observation-only pick/place baseline for v0.8 calibration.

Uses only the 320-d public observation. It does not read the simulator, true
part poses, hidden key yaw, or task object state. Each peg is approached above
its estimated position, grasped, lifted, and placed using the independently
biased slot estimate. This is a baseline, not a reference solution.

    NIST_PROFILE=v08 python tools/obs_scripted_v08.py --seed 7 --episodes 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))

from nist_sim.compound import COLLAR_FEAT, GLOBAL_FEAT, MAX_COLLARS, TASK_PEG_FEAT  # noqa: E402
from nist_sim.compound_v08 import OBS_DIM_V08, AssemblyIVEpisode, compound_score_v08, sample_compound_v08  # noqa: E402
from nist_sim.conditions import episode_seed_sequence  # noqa: E402
from nist_sim.env import ACT_DIM, MAX_PEGS, OBS_DIM, PAD_CENTRE_Z, PEG_FEAT, PEG_LEN  # noqa: E402

PROPRIO = 16
TASK0 = OBS_DIM
COL0 = TASK0 + MAX_PEGS * TASK_PEG_FEAT
GLOB0 = COL0 + MAX_COLLARS * COLLAR_FEAT
BP0 = GLOB0 + GLOBAL_FEAT
PEG0 = PROPRIO + ACT_DIM
KIND_BY_CODE = {1: "insert", 2: "collar", 3: "twist", 4: "rework", 5: "screw", 6: "stack", 7: "key"}

# Public baseline uses a fixed high grasp bias. CEM/residual search cancels it
# toward the mid-shaft grasp; see tools/train_rl_v08.py.
GRASP_Z_BIAS_M = 0.008
PLACE_Z_M = 0.028
LIFT_Z_M = 0.105


def blueprint_step(o: np.ndarray, k: int):
    """Decode only public blueprint channels; no simulator task access."""
    if k >= 12:
        return None
    kind = KIND_BY_CODE.get(int(round(float(o[BP0 + 4 + 2 * k] * 10.0))))
    packed = float(o[BP0 + 5 + 2 * k])
    peg = min(MAX_PEGS - 1, int(math.floor(packed + 1e-6)))
    flags = packed - peg
    blocked = flags >= 0.49
    done = (flags - 0.5 * blocked) >= 0.20
    return kind, peg, blocked, done


class ObsScriptedPolicy:
    """Stateful only through public observations and its own action history."""

    def __init__(
        self,
        seed: int = 0,
        grasp_bias_m: float | None = None,
        place_z_m: float | None = None,
        lift_z_m: float | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        # None → public strongest defaults; residual learners pass overrides.
        self.grasp_bias_m = GRASP_Z_BIAS_M if grasp_bias_m is None else float(grasp_bias_m)
        self.place_z_m = PLACE_Z_M if place_z_m is None else float(place_z_m)
        self.lift_z_m = LIFT_Z_M if lift_z_m is None else float(lift_z_m)
        self.reset()

    def reset(self):
        self.t = 0
        self.phase = "select"
        self.phase_t = 0
        self.target_i = -1
        self.target = np.zeros(3)
        self.command = None
        self.estimate = np.zeros(3)
        self.grasp_z = 0.0
        self.grasp_target = np.zeros(3)
        self.last_action = np.zeros(ACT_DIM)
        self.lost = 0
        self.slot = np.zeros(2)

    def _set_phase(self, phase: str):
        self.phase = phase
        self.phase_t = 0

    def _peg(self, o: np.ndarray, i: int):
        b = PEG0 + i * PEG_FEAT
        hand = o[:3] / 10.0
        slot = o[b + 13:b + 15] / 10.0
        rel = o[b + 1:b + 4] / 10.0
        pos = np.r_[slot + rel[:2], 0.020 + rel[2]]
        return pos, slot, rel

    def _go(self, o: np.ndarray, goal, grip: float, speed: float = 1.0):
        hand = o[:3] / 10.0
        if self.command is None:
            self.command = hand.copy()
        # Actions increment the controller target; they are not absolute poses.
        delta = np.clip((np.asarray(goal) - self.command) / 0.004, -speed, speed)
        self.command = np.clip(self.command + 0.004 * delta, [-0.25, -0.25, -0.003], [0.25, 0.25, 0.25])
        self.target = self.command.copy()
        a = np.zeros(ACT_DIM)
        a[:3] = delta
        a[4] = grip
        self.last_action = a
        return a

    def act(self, o: np.ndarray):
        self.t += 1
        self.phase_t += 1
        hand = o[:3] / 10.0
        gap = float(o[8]) / 50.0
        if self.phase == "select":
            best = None
            candidates = [blueprint_step(o, k) for k in range(12)]
            # Prefer insert over rework so the baseline spends time on graspable upright pegs.
            for kinds in (("insert",), ("insert", "rework")):
                allowed = {
                    row[1] for row in candidates
                    if row is not None and row[0] in kinds and not row[2] and not row[3]
                }
                for i in range(MAX_PEGS):
                    b = PEG0 + i * PEG_FEAT
                    if o[b] < 0.5 or o[b + 16] > 0.5 or i not in allowed:
                        continue
                    pos, slot, _ = self._peg(o, i)
                    diam = max(0.003, float(o[b + 12]) / 50.0)
                    if diam <= 0.0025 or not (0.022 <= float(pos[2]) <= 0.080):
                        continue
                    # Upright parts (local z close to world z) are graspable by
                    # this deliberately simple baseline; skip lying-down parts.
                    upright = abs(1.0 - 2.0 * (o[b + 5] ** 2 + o[b + 6] ** 2))
                    if upright < 0.9:
                        continue
                    dist = float(np.linalg.norm(pos[:2] - hand[:2]))
                    score = dist - 0.35 * diam
                    if best is None or score < best[0]:
                        best = (score, i, pos, slot)
                if best is not None:
                    break
            if best is None:
                return self._go(o, [hand[0], hand[1], 0.12], 1.0)
            _, self.target_i, self.estimate, self.slot = best
            self._set_phase("above")

        if self.target_i < 0:
            return self._go(o, [hand[0], hand[1], 0.12], 1.0)
        pos, slot, _ = self._peg(o, self.target_i)
        self.estimate = pos
        self.slot = slot
        if self.phase == "above":
            goal = np.r_[pos[:2], 0.085]
            a = self._go(o, goal, 1.0)
            if self.phase_t >= 12 and np.max(np.abs(goal - hand)) < 0.008:
                self._set_phase("descend")
            elif self.phase_t > 70:
                self._set_phase("select")
        elif self.phase == "descend":
            # Mid-shaft aim plus a configurable high bias (see grasp_bias_m).
            self.grasp_z = float(
                np.clip(
                    pos[2] + (PEG_LEN / 2 - 0.008) - PAD_CENTRE_Z + self.grasp_bias_m,
                    0.024,
                    0.070,
                )
            )
            goal = np.r_[pos[:2], self.grasp_z]
            a = self._go(o, goal, 1.0, 0.40)
            xy_ok = float(np.linalg.norm(goal[:2] - hand[:2])) < 0.006
            z_ok = abs(goal[2] - hand[2]) < 0.007
            if (self.phase_t >= 20 and xy_ok and z_ok) or self.phase_t >= 55:
                self._set_phase("close")
        elif self.phase == "close":
            a = self._go(o, np.r_[pos[:2], self.grasp_z + 0.003], -1.0, 0.25)
            if self.phase_t >= 18:
                self._set_phase("lift")
        elif self.phase == "lift":
            a = self._go(o, np.r_[pos[:2], self.lift_z_m], -1.0, 0.28)
            if self.phase_t >= 50:
                self._set_phase("carry")
        elif self.phase == "carry":
            goal = np.r_[self.slot[:2], max(0.070, self.lift_z_m - 0.015)]
            a = self._go(o, goal, -1.0, 0.50)
            if self.phase_t >= 45:
                self._set_phase("place")
        elif self.phase == "place":
            a = self._go(o, np.r_[self.slot[:2], self.place_z_m], -1.0, 0.32)
            if self.phase_t >= 28:
                self._set_phase("release")
        elif self.phase == "release":
            a = self._go(o, self.target, 1.0)
            if self.phase_t >= 12:
                self._set_phase("retreat")
        elif self.phase == "retreat":
            a = self._go(o, self.target + np.array([0.0, 0.0, 0.08]), 1.0, 0.55)
            if self.phase_t >= 22:
                self._set_phase("select")
        else:
            self._set_phase("select")
            a = self._go(o, [hand[0], hand[1], 0.10], 1.0)
        return np.clip(a, -1.0, 1.0)


def act_from_obs(o: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Compatibility wrapper; use ``ObsScriptedPolicy`` for stateful evaluation."""
    if not hasattr(act_from_obs, "_policy"):
        act_from_obs._policy = ObsScriptedPolicy(int(rng.integers(2**31)))
    return act_from_obs._policy.act(o)


def eval_seed(seed: int, per: int, episodes: int, start: int = 0) -> dict:
    _, conds = sample_compound_v08(seed, per, "live")
    scores = []
    t0 = time.monotonic()
    n = min(len(conds), start + episodes)
    for i in range(start, n):
        ss = episode_seed_sequence(seed, i)
        r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
        ep = AssemblyIVEpisode(conds[i], r1)
        policy = ObsScriptedPolicy(int(ss.generate_state(1)[0]))
        while not ep.done:
            o = ep.observe(r2)
            assert o.shape[0] == OBS_DIM_V08
            ep.step(policy.act(o))
        sc = compound_score_v08(ep)["score"]
        scores.append(sc)
        print(f"episode={i} score={sc:.6f} terminal={ep.terminal_reason}", flush=True)
    return {"mean": float(np.mean(scores)) if scores else 0.0, "scores": scores, "wall_s": time.monotonic() - t0, "n": len(scores)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    os.environ.setdefault("NIST_PROFILE", "v08")
    r = eval_seed(a.seed, a.per, a.episodes, a.start)
    print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
