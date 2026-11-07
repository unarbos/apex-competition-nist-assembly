"""Miner-path stack place-and-release (observe → action → step only).

Uses the 288-d observation channels a live miner sees. Never reads true MuJoCo poses,
never calls pose nudges / pins / snaps. Completion goes through ``_maybe_mate_stacks``.

Intended for:
- unit proof that the stack latch is reachable without privileged assistance
- a strong scripted baseline for RL residual training

    NIST_PROFILE=v07 python tools/miner_path_stack.py --seed 7 --ep 1
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tools"))

from nist_sim.compound import BLOCK_H, COLLAR_H, OBS_DIM_V06  # noqa: E402
from nist_sim.env import (  # noqa: E402
    ACT_DIM,
    DXYZ,
    HAND_Z_MIN,
    MAX_PEGS,
    PAD_CENTRE_Z,
    PEG_FEAT,
    PEG_LEN,
    PROPRIO_DIM,
)

PEG0 = PROPRIO_DIM + ACT_DIM  # 21
TASK0 = 184
TASK_PEG_FEAT = 6
MAX_COLLARS = 4
COLLAR_FEAT = 13
COL0 = TASK0 + MAX_PEGS * TASK_PEG_FEAT  # 232


def _peg(obs: np.ndarray, i: int) -> np.ndarray:
    return obs[PEG0 + i * PEG_FEAT : PEG0 + (i + 1) * PEG_FEAT]


def _collar(obs: np.ndarray, j: int) -> np.ndarray:
    return obs[COL0 + j * COLLAR_FEAT : COL0 + (j + 1) * COLLAR_FEAT]


@dataclass
class MinerStackPlacer:
    """Obs-only controller that place-and-releases one stack ring onto its peg."""

    collar_j: int
    peg_i: int
    phase: str = "seek"
    timer: int = 0

    def reset(self) -> None:
        self.phase = "seek"
        self.timer = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        cf = _collar(obs, self.collar_j)
        pf = _peg(obs, self.peg_i)
        a = np.zeros(5)
        if cf[0] < 1.5:
            # Ring not visible / not a stack — idle up.
            a[2] = 0.3
            a[4] = 1.0
            return a
        hand = obs[0:2] / 10.0
        hz = float(obs[2] / 10.0)
        slot_est = pf[13:15] / 10.0
        residual = cf[1:3] / 10.0
        collar_est = slot_est + residual
        collar_z = BLOCK_H + float(cf[3]) / 10.0
        grasp_z = max(HAND_Z_MIN, collar_z - PAD_CENTRE_Z + 0.001)
        # Seated peg top ≈ PEG_LEN above z=0; place ring bottom on peg top.
        place_z = PEG_LEN + COLLAR_H / 2 - PAD_CENTRE_Z + 0.0005
        lift_z = PEG_LEN + 0.022 + COLLAR_H / 2 - PAD_CENTRE_Z

        if self.phase == "seek":
            err = collar_est - hand
            a[0:2] = np.clip(err / DXYZ, -1, 1)
            a[2] = np.clip((0.05 - hz) / DXYZ, -1, 1)
            a[4] = 1.0
            if float(np.hypot(*err)) < 0.010:
                self.phase = "descend"
                self.timer = 0
        elif self.phase == "descend":
            err = collar_est - hand
            a[0:2] = np.clip(err / DXYZ, -0.6, 0.6)
            a[2] = np.clip((grasp_z - hz) / DXYZ, -1, 1)
            a[4] = 1.0
            self.timer += 1
            if abs(hz - grasp_z) < 0.003 or self.timer > 50:
                self.phase = "grasp"
                self.timer = 0
        elif self.phase == "grasp":
            err = collar_est - hand
            a[0:2] = np.clip(err / DXYZ, -0.3, 0.3)
            a[2] = np.clip((grasp_z - hz) / DXYZ, -0.5, 0.5)
            a[4] = -1.0
            self.timer += 1
            if self.timer > 22:
                self.phase = "lift"
                self.timer = 0
        elif self.phase == "lift":
            a[2] = 0.9
            a[4] = -1.0
            self.timer += 1
            if self.timer > 35 or hz > 0.08:
                self.phase = "align"
                self.timer = 0
        elif self.phase == "align":
            a[0:2] = np.clip((-residual) / DXYZ, -1, 1)
            a[2] = np.clip((lift_z - hz) / DXYZ, -1, 1)
            a[4] = -1.0
            self.timer += 1
            if float(np.hypot(*residual)) < 0.0012 and self.timer > 10:
                self.phase = "place"
                self.timer = 0
        elif self.phase == "place":
            # Drive into the peg-top mate zone under a closed grasp.
            a[0:2] = np.clip((-residual) / DXYZ, -0.45, 0.45)
            a[2] = np.clip((place_z - hz) / DXYZ, -0.25, 0.25)
            a[4] = -1.0
            self.timer += 1
            # Tighter mate zone (STACK_MATE_XY≈2.8 mm): demand smaller residual before hold.
            aligned = float(np.hypot(*residual)) < 0.0010 and abs(hz - place_z) < 0.004
            # Do not open on the first aligned frame — enter ``hold`` so the latch can arm.
            if aligned and self.timer > 25:
                self.phase = "hold"
                self.timer = 0
            elif self.timer > 140:
                self.phase = "hold"
                self.timer = 0
        elif self.phase == "hold":
            # Closed grasp, stay put — `_maybe_mate_stacks` needs ≥2 closed-in-zone frames.
            a[0:2] = np.clip((-residual) / DXYZ, -0.2, 0.2)
            a[2] = np.clip((place_z - hz) / DXYZ, -0.15, 0.15)
            a[4] = -1.0
            self.timer += 1
            if self.timer > 14:
                self.phase = "open"
                self.timer = 0
        elif self.phase == "open":
            # Open jaws in place first — do not lift while closed (drags interference ring).
            a[0:2] = np.clip((-residual) / DXYZ, -0.10, 0.10)
            a[2] = 0.0
            a[4] = 1.0
            self.timer += 1
            if self.timer > 16:
                self.phase = "rise"
                self.timer = 0
        elif self.phase == "rise":
            # Slow rise while holding XY; latch counts released frames (STACK_MATE_HOLD=6).
            a[0:2] = np.clip((-residual) / DXYZ, -0.06, 0.06)
            a[2] = 0.15
            a[4] = 1.0
            self.timer += 1
            if self.timer > 70:
                self.phase = "idle"
                self.timer = 0
        else:
            a[2] = 0.15
            a[4] = 1.0
            self.timer += 1
        return a


def first_visible_stack(obs: np.ndarray) -> tuple[int, int] | None:
    """Return (collar_j, peg_i) for the first unfinished stack ring in the obs table."""
    for j in range(MAX_COLLARS):
        c = _collar(obs, j)
        if c[0] < 1.5 or c[11] > 0.5 or c[12] > 0.5:
            continue
        peg_i = int(round(float(c[10]) * MAX_PEGS))
        peg_i = int(np.clip(peg_i, 0, MAX_PEGS - 1))
        return j, peg_i
    return None


def main() -> int:
    """Demo: physics-seat a peg, then miner-path place (no solver nudges)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--per", type=int, default=8)
    a = ap.parse_args()
    os.environ["NIST_PROFILE"] = "v07"
    from nist_sim import conditions as C
    from nist_sim.compound_v07 import AssemblyIIIEpisode, sample_compound_v07
    from nist_sim.env import SEATED_HOLD, _yaw_quat
    import mujoco

    _, conds = sample_compound_v07(a.seed, a.per)
    c = conds[a.ep]
    sk = next(k for k, s in enumerate(c.compound.steps) if s.kind == "stack")
    s = c.compound.steps[sk]
    r1, r2 = (np.random.default_rng(x) for x in C.episode_seed_sequence(a.seed, a.ep).spawn(2))
    ep = AssemblyIIIEpisode(c, r1)
    # Physics seat only (tests seal solver nudges separately).
    hx, hy = ep.slot_xy[s.peg]
    z = PEG_LEN / 2 - 0.0005
    ep._set_free(ep.peg_qadr[s.peg], ep.peg_vadr[s.peg], [hx, hy, z], _yaw_quat(0.0))
    ep.data.qvel[ep.peg_vadr[s.peg] : ep.peg_vadr[s.peg] + 6] = 0.0
    ep.data.qpos[0:3] = [hx + 0.08, hy + 0.08, 0.14]
    ep.data.qpos[4] = 0.012
    mujoco.mj_forward(ep.model, ep.data)
    for _ in range(SEATED_HOLD + 30):
        if not (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
            ep._set_free(ep.peg_qadr[s.peg], ep.peg_vadr[s.peg], [hx, hy, z], _yaw_quat(0.0))
            ep.data.qvel[ep.peg_vadr[s.peg] : ep.peg_vadr[s.peg] + 6] = 0.0
            mujoco.mj_forward(ep.model, ep.data)
        ep.step(np.array([0.0, 0.0, 0.3, 0.0, 1.0]))
    pol = MinerStackPlacer(collar_j=s.part, peg_i=s.peg)
    for _ in range(900):
        if ep.done or ep.step_done[sk]:
            break
        o = ep.observe(r2)
        ep.step(np.clip(pol.act(o), -1, 1))
    print(
        f"ep={a.ep} done={ep.step_done[sk]} mated={ep._stack_mated[s.part]} "
        f"arm={ep._stack_mate_arm[s.part]} phase={pol.phase}"
    )
    return 0 if ep.step_done[sk] else 1


if __name__ == "__main__":
    raise SystemExit(main())
