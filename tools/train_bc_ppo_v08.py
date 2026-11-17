"""Collect observation-only scripted residual demonstrations for PPO warm start.

The target is ``(scripted_action + small deterministic perturbation) / scale``.
This teaches a residual policy to reproduce and locally explore around the
public scripted baseline. It is not privileged teacher data and is not itself
an RL result. ``tools/train_ppo_v08.py --residual`` performs PPO fine-tuning.
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


def collect(job):
    seed, per, i, scale = job
    os.environ["NIST_PROFILE"] = "v08"
    sys.path[:0] = [str(ROOT / "referee"), str(ROOT / "tools")]
    from nist_sim.compound_v08 import AssemblyIVEpisode, compound_score_v08, sample_compound_v08
    from nist_sim.conditions import episode_seed_sequence
    from obs_scripted_v08 import ObsScriptedPolicy

    _, conds = sample_compound_v08(seed, per)
    ss = episode_seed_sequence(seed, i)
    r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
    ep = AssemblyIVEpisode(conds[i], r1)
    base = ObsScriptedPolicy(int(ss.generate_state(1)[0]))
    rng = np.random.default_rng(seed * 1000 + i)
    X, Y = [], []
    while not ep.done:
        obs = ep.observe(r2)
        action = base.act(obs)
        residual = np.clip(rng.normal(0, 0.025, 5), -0.06, 0.06)
        X.append(obs)
        Y.append((action + residual) / scale)
        ep.step(np.clip(action + residual, -1, 1))
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), float(compound_score_v08(ep)["score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--scale", type=float, default=0.35)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.monotonic()
    with mp.get_context("fork").Pool(args.workers) as pool:
        rows = pool.map(collect, [(args.seed, args.per, i, args.scale) for i in range(args.episodes)])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=np.concatenate([r[0] for r in rows]), Y=np.concatenate([r[1] for r in rows]))
    print({"episodes": len(rows), "samples": sum(len(r[0]) for r in rows), "scores": [r[2] for r in rows], "wall_s": time.monotonic() - t0})


if __name__ == "__main__":
    main()
