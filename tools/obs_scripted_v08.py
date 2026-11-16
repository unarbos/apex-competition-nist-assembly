"""Strong observation-only scripted controller for v0.8 calibration (≤~0.03 gate).

Uses only the published observation (and the public blueprint encoding). No env
internals, no true poses, no key yaw. Tries insert → collar/stack heuristics from
biased peg/slot estimates.

    NIST_PROFILE=v08 python tools/obs_scripted_v08.py --seed 7 --per 8 --episodes 64
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))

from nist_sim.compound import COLLAR_FEAT, GLOBAL_FEAT, MAX_COLLARS, TASK_PEG_FEAT  # noqa: E402
from nist_sim.compound_v08 import (  # noqa: E402
    BLUEPRINT_FEAT,
    OBS_DIM_V08,
    AssemblyIVEpisode,
    compound_score_v08,
    sample_compound_v08,
)
from nist_sim.conditions import episode_seed_sequence  # noqa: E402
from nist_sim.env import ACT_DIM, MAX_PEGS, OBS_DIM, PEG_FEAT  # noqa: E402

PROPRIO = 16
TASK0 = OBS_DIM  # 184
COL0 = TASK0 + MAX_PEGS * TASK_PEG_FEAT
GLOB0 = COL0 + MAX_COLLARS * COLLAR_FEAT
BP0 = GLOB0 + GLOBAL_FEAT


def act_from_obs(o: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Heuristic: chase the first unfinished insert-like peg using biased relative pose."""
    a = np.zeros(ACT_DIM, dtype=np.float64)
    hand = o[0:3] / 10.0
    # Find a peg row that looks present and not seated.
    best_i, best_score = -1, -1e9
    for i in range(MAX_PEGS):
        b = PROPRIO + ACT_DIM + i * PEG_FEAT
        present = float(o[b + 0])
        if present < 0.5:
            continue
        seated = float(o[b + 16])
        if seated > 0.5:
            continue
        rel = o[b + 1 : b + 4] / 10.0  # est peg - something; use as chase vector
        # Prefer smaller XY error (closer).
        score = -float(np.hypot(rel[0], rel[1])) + 0.1 * present
        if score > best_score:
            best_score, best_i = score, i
    if best_i < 0:
        # Nothing to do: random small jitter (does not farm score).
        a[:3] = rng.normal(0, 0.05, 3)
        a[4] = 1.0
        return np.clip(a, -1, 1)

    b = PROPRIO + ACT_DIM + best_i * PEG_FEAT
    rel = o[b + 1 : b + 4] / 10.0
    # Servo toward peg estimate, then press and open.
    a[0] = float(np.clip(rel[0] / 0.004, -1, 1))
    a[1] = float(np.clip(rel[1] / 0.004, -1, 1))
    z_err = float(rel[2])
    if abs(rel[0]) < 0.008 and abs(rel[1]) < 0.008:
        a[2] = -0.8 if z_err > -0.01 else 0.4
        a[4] = -1.0 if z_err > -0.005 else 1.0
    else:
        a[2] = 0.3
        a[4] = -1.0
    # Mild yaw search — no closeness cue to guide it (blind).
    a[3] = float(np.clip(0.35 * np.sin(0.15 * (o[GLOB0] * 100.0 + best_i)), -1, 1))
    return np.clip(a, -1, 1)


def eval_seed(seed: int, per: int, episodes: int, start: int = 0) -> dict:
    _, conds = sample_compound_v08(seed, per, "live")
    scores = []
    t0 = time.time()
    n = min(len(conds), start + episodes)
    for i in range(start, n):
        ss = episode_seed_sequence(seed, i)
        r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
        ep = AssemblyIVEpisode(conds[i], r1)
        noise = r2
        act_rng = np.random.default_rng(ss.spawn(1)[0])
        while not ep.done:
            o = ep.observe(noise)
            assert o.shape[0] == OBS_DIM_V08
            a = act_from_obs(o, act_rng)
            ep.step(a)
        sc = compound_score_v08(ep)["score"]
        scores.append(sc)
    return {
        "mean": float(np.mean(scores)) if scores else 0.0,
        "scores": scores,
        "wall_s": time.time() - t0,
        "n": len(scores),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    os.environ.setdefault("NIST_PROFILE", "v08")
    r = eval_seed(a.seed, a.per, a.episodes, a.start)
    print(f"obs_scripted_v08 seed={a.seed} mean={r['mean']:.4f} n={r['n']} wall={r['wall_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
