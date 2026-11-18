"""Collect privileged-teacher residual demonstrations on training seeds only.

Only public observations and action residuals are saved. The teacher is honest
(``oracle=False``) and is used exclusively during training. Evaluation uses the
fail-closed observation boundary and never loads the teacher.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INSERT_SEARCH_STEPS_V08 = 160


def matched_insert_teacher(ep, peg_i):
    """Construct the honest training-only teacher with script-matched search."""
    from compound_solver import PegView, ScriptedPeg

    return ScriptedPeg(
        PegView(ep, peg_i), search_amp=0.006,
        search_steps=INSERT_SEARCH_STEPS_V08, swing_grip=-1.0,
    )


def collect(job):
    seed, per, i, scale, teacher_mode = job
    os.environ["NIST_PROFILE"] = "v08"
    sys.path[:0] = [str(ROOT / "referee"), str(ROOT / "tools")]
    from nist_sim.compound_v08 import AssemblyIVEpisode, compound_score_v08, sample_compound_v08
    from nist_sim.conditions import episode_seed_sequence
    from compound_solver import solve_compound
    from obs_boundary_v08 import ObservationBoundary
    from strong_residual_v08 import StrongResidualController

    _, conds = sample_compound_v08(seed, per, "live")
    ss = episode_seed_sequence(seed, i)
    r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
    ep = AssemblyIVEpisode(conds[i], r1)
    boundary = ObservationBoundary(ep)
    obs = boundary.observe(r2)
    controller = StrongResidualController(boundary, r2, initial_obs=obs)
    X, Y = [], []
    physical_step = ep.step
    teacher = None

    def record_step(action):
        nonlocal obs, teacher
        if ep.done:
            return
        if teacher_mode == "insert":
            # Stay on the observation-only script until its observation-derived
            # lift check succeeds. Only then start an honest true-state teacher
            # insertion in the same physical state. No teacher state or target
            # is saved, and no teleport or latch helper is invoked.
            if teacher is None and controller.phase != "insert":
                action = controller.act()
            else:
                if teacher is None:
                    teacher = iter(matched_insert_teacher(ep, controller.i).insert())
                try:
                    action = next(teacher)
                except StopIteration:
                    teacher = None
                    action = controller.act()
        teacher_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        base_action = controller.act()
        X.append(obs.astype(np.float32))
        Y.append(np.clip((teacher_action - base_action) / scale, -1.0, 1.0).astype(np.float32))
        physical_step(teacher_action)
        boundary.done = ep.done
        if not ep.done:
            obs = boundary.observe(r2)
            controller.post_step(False, obs, action=teacher_action)

    ep.step = record_step
    solve_compound(ep, oracle=False)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), float(compound_score_v08(ep)["score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seeds", default="101,102,103,104")
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=8, help="episodes per training seed")
    ap.add_argument("--scale", type=float, default=0.35)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--teacher-mode", choices=("trajectory", "insert"), default="trajectory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seeds = [int(s) for s in args.train_seeds.split(",")]
    jobs = [(seed, args.per, i, args.scale, args.teacher_mode) for seed in seeds for i in range(args.episodes)]
    t0 = time.monotonic()
    with mp.get_context("fork").Pool(args.workers) as pool:
        rows = pool.map(collect, jobs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=np.concatenate([r[0] for r in rows]), Y=np.concatenate([r[1] for r in rows]))
    print({
        "episodes": len(rows),
        "samples": sum(len(r[0]) for r in rows),
        "teacher_mean": float(np.mean([r[2] for r in rows])),
        "teacher_scores": [r[2] for r in rows],
        "train_seeds": seeds,
        "teacher_mode": args.teacher_mode,
        "wall_s": time.monotonic() - t0,
    })


if __name__ == "__main__":
    main()
