"""Strong v0.8 script restricted to the public 320-D observation.

The controller reuses the reference motion generators through a narrow adapter.
All poses are board-relative and observation-derived. Unknown attributes fail
closed instead of forwarding to the simulator. Diagnostics are returned only by
the caller and never enter control decisions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "referee"), str(ROOT / "tools")]

from nist_sim.compound_v08 import MAX_HORIZON_V08, AssemblyIVEpisode, compound_score_v08, sample_compound_v08
from nist_sim.conditions import episode_seed_sequence
from nist_sim.env import ACT_DIM, BLOCK_H, MAX_PEGS, PEG_FEAT
from obs_refport import _BlindSpec, _PegC, rs
from obs_scripted_v08 import blueprint_step

PEG0 = ACT_DIM + 16


class _ObservedHand:
    """Proprioceptive hand state, not a MuJoCo data object."""

    def __init__(self, o):
        self.qpos = np.r_[
            np.asarray(o[:3], dtype=np.float64) / 10.0,
            math.atan2(float(o[3]), float(o[4])),
            float(o[8]) / 100.0,
            0.0,
        ]


class V08ObsEpisode:
    """Fail-closed observation adapter for the upright motion primitives."""

    def __init__(self, ep, noise_rng, initial_obs=None):
        self._ep = ep
        self._rng = noise_rng
        self._initial_obs = initial_obs
        self._alpha = 0.25
        self.phase_flag = None
        self.done = False
        self.attempted = set()
        self._update(first=True)

    def __getattr__(self, name):
        raise AttributeError(f"privileged or unknown episode attribute: {name}")

    @property
    def data(self):
        return _ObservedHand(self.last_obs)

    @property
    def target(self):
        # Public hand feedback substitutes for the hidden actuator target.
        return self.data.qpos[:4].copy()

    @property
    def seated_now(self):
        return [self.last_obs[PEG0 + i * PEG_FEAT + 16] > 0.5 for i in range(MAX_PEGS)]

    def _update(self, first=False, obs=None):
        if obs is not None:
            o = np.asarray(obs, dtype=np.float64)
        elif first and self._initial_obs is not None:
            o = np.asarray(self._initial_obs, dtype=np.float64)
        else:
            o = self._ep.observe(self._rng).astype(np.float64)
        if o.shape != (320,) or not np.all(np.isfinite(o)):
            raise ValueError("invalid public observation")
        self.last_obs = o
        slot = np.zeros((MAX_PEGS, 2))
        syaw = np.zeros((MAX_PEGS, 2))
        self._peg_pos = np.zeros((MAX_PEGS, 3))
        self._peg_quat = np.zeros((MAX_PEGS, 4))
        sizes = []
        for i in range(MAX_PEGS):
            f = o[PEG0 + i * PEG_FEAT : PEG0 + (i + 1) * PEG_FEAT]
            slot[i] = f[13:15] / 10.0
            syaw[i] = [f[8], f[9]]
            rel = f[1:4] / 10.0
            self._peg_pos[i] = [slot[i, 0] + rel[0], slot[i, 1] + rel[1], rel[2] + BLOCK_H]
            self._peg_quat[i] = f[4:8] if f[0] > 0.5 else np.array([1.0, 0.0, 0.0, 0.0])
            sizes.append((max(f[12] / 50.0, 0.001), max(f[10] / 50.0, 0.001)))
        if first:
            self._slot, self._syaw = slot, syaw
            self._pegs = [
                _PegC(
                    "lying" if abs(rs.part_axis(self._peg_quat[i])[2]) < 0.5 else "loose",
                    _BlindSpec(*sizes[i]),
                )
                for i in range(MAX_PEGS)
            ]
            # Purely observational metadata used by the generic motion code.
            self.c = SimpleNamespace(
                max_steps=MAX_HORIZON_V08,
                n_pegs=MAX_PEGS,
                backlash=0.0,
                act_delay=0,
                pegs=self._pegs,
            )
        else:
            a = self._alpha
            self._slot = (1 - a) * self._slot + a * slot
            self._syaw = (1 - a) * self._syaw + a * syaw
        self._pad = float(o[15])
        self.steps = int(round(float(o[286]) * MAX_HORIZON_V08))
        self.slot_xy = [tuple(x) for x in self._slot]
        self.slot_yaw = [math.atan2(s, c) for s, c in self._syaw]
        self.r_hole = [self._pegs[i].spec.size / 2 + 0.0005 for i in range(MAX_PEGS)]

    def peg_pose(self, i):
        return self._peg_pos[i].copy(), self._peg_quat[i].copy()

    def _gripper_touching(self, body_id):
        return self._pad > 0.05

    def step(self, action):
        self._ep.step(np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0))
        self.done = self._ep.done
        if not self.done:
            self._update()


class ObservedPegView:
    """Only the public fields consumed by ``ScriptedPeg.pick_upright/insert``."""

    def __init__(self, ep, peg_i):
        self.ep = ep
        self.i = peg_i
        self.c = SimpleNamespace(
            hole_xy=tuple(ep.slot_xy[peg_i]),
            hole_yaw=float(ep.slot_yaw[peg_i]),
            symmetry=int(ep.c.pegs[peg_i].spec.symmetry()),
        )
        self.r_hole = ep.r_hole[peg_i]
        self.part_body = peg_i  # opaque index; contact uses observed pad force

    @property
    def data(self):
        return self.ep.data

    @property
    def target(self):
        return self.ep.target

    @property
    def steps(self):
        return 0  # only diagnostic for the two upright generators used here

    def part_pose(self):
        return self.ep.peg_pose(self.i)

    def _touching(self):
        return self.ep._gripper_touching(self.part_body)


def _choices(view, attempted):
    o = view.last_obs
    choices = []
    for k in range(12):
        row = blueprint_step(o, k)
        if row is None or row[0] not in ("insert", "rework") or row[2] or row[3]:
            continue
        kind, i, _, _ = row
        if i in attempted or o[PEG0 + i * PEG_FEAT] < 0.5:
            continue
        p, q = view.peg_pose(i)
        if abs(rs.part_axis(q)[2]) < 0.90:
            continue
        dist = float(np.linalg.norm(p[:2] - view.data.qpos[:2]))
        choices.append((dist + (-0.08 if kind == "rework" else 0.0), i))
    return sorted(choices)


def solve_obs_episode(ep, noise_rng, *, trace=False):
    """Run the exact controller used beneath residual policies.

    ``trace`` records only public observations and controller state; diagnostics
    never enter the action path. The observation boundary owns terminal state.
    """
    from strong_residual_v08 import StrongResidualController

    obs = ep.observe(noise_rng)
    controller = StrongResidualController(ep, noise_rng, initial_obs=obs)
    rows = []
    steps = 0
    while not ep.done:
        action = controller.act()
        if trace:
            rows.append({"step": steps, "stage": controller.stage, "peg": controller.i,
                         "observation": obs.astype(float).tolist()})
        ep.step(action)
        steps += 1
        if not ep.done:
            obs = ep.observe(noise_rng)
            controller.post_step(False, obs, action=action)
    return {"steps": steps, "attempts": rows}


def eval_seed(seed, per, episodes):
    _, conds = sample_compound_v08(seed, per, "live")
    scores, rows = [], []
    t0 = time.monotonic()
    for i in range(min(len(conds), episodes)):
        ss = episode_seed_sequence(seed, i)
        r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
        ep = AssemblyIVEpisode(conds[i], r1)
        from obs_boundary_v08 import ObservationBoundary
        solve_obs_episode(ObservationBoundary(ep), r2)
        score = float(compound_score_v08(ep)["score"])
        scores.append(score)
        rows.append({"episode": i, "score": score, "terminal_reason": ep.terminal_reason})
        print(f"episode={i} score={score:.9f} terminal={ep.terminal_reason}", flush=True)
    arr = np.asarray(scores)
    rng = np.random.default_rng(20261117)
    means = np.mean(rng.choice(arr, (10000, len(arr)), replace=True), axis=1)
    return {
        "seed": seed,
        "n": len(scores),
        "mean": float(np.mean(arr)),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "scores": scores,
        "episodes": rows,
        "wall_s": time.monotonic() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    os.environ.setdefault("NIST_PROFILE", "v08")
    result = eval_seed(args.seed, args.per, args.episodes)
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
