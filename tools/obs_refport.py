"""Obs-only port of the reference solver (strong scripted calibration point).

Runs the same phase machine as ``tools/reference_solver.py`` but every privileged
read is replaced by what the policy observation provides: smoothed peg / slot /
clutter pose estimates, the slot-yaw estimate, the nominal size, pad force instead
of contact queries, and no knowledge of hidden dynamics (backlash, delay) or of the
cross-section family. This is the "port the reference into a script" strategy a
strong hand-coded entry would use, so it is the scripted gate for calibration.

    PYTHONPATH=referee python tools/obs_refport.py --seed 0 --episodes 8
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tools"))

from nist_sim.conditions import sample_episodes  # noqa: E402
from nist_sim.env import ACT_DIM, BLOCK_H, CLUTTER_FEAT, MAX_CLUTTER, PEG_FEAT, PROPRIO_DIM, Episode  # noqa: E402
import reference_solver as rs  # noqa: E402

PEG0 = PROPRIO_DIM + ACT_DIM
CL0 = PEG0 + 8 * PEG_FEAT
NOMINAL_CLEARANCE = 0.0005


@dataclass
class _BlindSpec:
    """What the obs reveals about a slot: sizes only (no family, no symmetry)."""

    size: float
    outer: float

    def symmetry(self) -> int:
        # Round-looking (outer ~ inscribed) -> any yaw; otherwise align to the estimate.
        return 0 if self.outer <= self.size * 1.02 else 1

    def outer_radius(self) -> float:
        return self.outer / 2

    def half_width_y(self) -> float:
        return self.size / 2


class _PegC:
    def __init__(self, start_type: str, spec: _BlindSpec):
        self.start_type = start_type
        self.spec = spec

    def shape(self) -> str:
        return "unknown"

    def diameter(self) -> float:
        return self.spec.size


class _CondView:
    def __init__(self, c, pegs):
        self.max_steps = c.max_steps
        self.n_pegs = c.n_pegs
        self.backlash = 0.0  # hidden dynamics are not observable
        self.act_delay = 0
        self.pegs = pegs


class ObsEpisode:
    """Episode proxy: physics and scoring stay real, every read is an estimate."""

    def __init__(self, ep: Episode, noise_rng: np.random.Generator, alpha: float = 0.25):
        self._ep = ep
        self._rng = noise_rng
        self._alpha = alpha
        self._slot = None
        self._syaw = None
        # Optional hooks (residual RL on top of this controller, tools/train_ppo.py):
        # act_hook(obs, base_action, view) -> action; post_hook(next_obs, done).
        self.act_hook = None
        self.post_hook = None
        self.phase_flag = None  # "insert" while the phase machine aligns / descends on a hole
        self._update(first=True)

    def __getattr__(self, name):
        return getattr(self._ep, name)

    # ---- estimates ---------------------------------------------------------
    def _update(self, first: bool = False) -> None:
        ep = self._ep
        o = ep.observe(self._rng).astype(np.float64)
        self.last_obs = o
        bx, by = ep.c.board_xy  # translation only: obs is board-relative
        n = ep.n_pegs
        slot = np.zeros((n, 2))
        syaw = np.zeros((n, 2))
        self._peg_pos = np.zeros((n, 3))
        self._peg_quat = np.zeros((n, 4))
        sizes = []
        for i in range(n):
            f = o[PEG0 + i * PEG_FEAT : PEG0 + (i + 1) * PEG_FEAT]
            slot[i] = [f[13] / 10 + bx, f[14] / 10 + by]
            syaw[i] = [f[8], f[9]]
            rel = f[1:4] / 10
            self._peg_pos[i] = [slot[i][0] + rel[0], slot[i][1] + rel[1], rel[2] + BLOCK_H]
            self._peg_quat[i] = f[4:8]
            sizes.append((f[12] / 50, f[10] / 50))
        if first:
            self._slot, self._syaw = slot, syaw
            self._pegs = [
                _PegC("lying" if abs(rs.part_axis(self._peg_quat[i])[2]) < 0.5 else "loose", _BlindSpec(*sizes[i]))
                for i in range(n)
            ]
            self.c = _CondView(ep.c, self._pegs)
        else:
            a = self._alpha
            self._slot = (1 - a) * self._slot + a * slot
            self._syaw = (1 - a) * self._syaw + a * syaw
        self._cl = []
        for j in range(ep.n_clutter):
            g = o[CL0 + j * CLUTTER_FEAT : CL0 + (j + 1) * CLUTTER_FEAT]
            self._cl.append((np.array([g[1] / 10 + bx, g[2] / 10 + by, g[3] / 10]), g[4:8]))
        self._pad = float(o[15])
        self.slot_xy = [tuple(x) for x in self._slot]
        self.slot_yaw = [math.atan2(s, c) for s, c in self._syaw]
        self.r_hole = [self._pegs[i].spec.size / 2 + NOMINAL_CLEARANCE for i in range(n)]

    def step(self, a) -> None:
        if self.act_hook is not None:
            a = self.act_hook(self.last_obs, np.clip(np.asarray(a, dtype=float), -1, 1), self)
        self._ep.step(a)
        if not self._ep.done:
            self._update()
        if self.post_hook is not None:
            self.post_hook(self.last_obs, self._ep.done)

    def peg_pose(self, i: int):
        return self._peg_pos[i].copy(), self._peg_quat[i].copy()

    def clutter_pose(self, i: int):
        p, q = self._cl[i]
        return p.copy(), np.asarray(q).copy()

    def _gripper_touching(self, body_id: int) -> bool:
        return self._pad > 0.05


# Wider, longer lip search than the privileged solver: the estimate residual is mm-scale.
SEARCH_KW = {"search_amp": 0.006, "search_steps": 320}


def solve_obs_episode(
    ep: Episode, noise_rng: np.random.Generator, script_kw: dict | None = None, act_hook=None, post_hook=None
) -> dict:
    kw = SEARCH_KW if script_kw is None else script_kw
    view = ObsEpisode(ep, noise_rng)
    view.act_hook, view.post_hook = act_hook, post_hook
    return rs.solve_episode(view, script_kw=kw)  # type: ignore[arg-type]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=8)
    a = ap.parse_args()
    _, conds = sample_episodes(a.seed, a.per)
    scores = []
    for i, c in enumerate(conds[: a.episodes]):
        ss = np.random.SeedSequence([a.seed, 0x5245, i])
        r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
        r = solve_obs_episode(Episode(c, r1), r2)
        scores.append(r["score"])
        print(f"ep {i:02d} K={c.n_pegs} seated={r['n_seated']}/{r['n_pegs']} score={r['score']:.3f}")
    print(f"mean {np.mean(scores):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
