"""Per-control-step cost of physics vs the v0.6 wrist depth patch on live-mix episodes.

    python tools/probe_v06_cost.py --seed 1 --episodes 8 --steps 150 --n 16

Prints mean ms/step for physics (40 substeps), observe(), and the depth patch (+ corrupt), and
the projected share the depth patch adds to a round. Random-walk actions keep the hand moving
over the board.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
from nist_sim.conditions import episode_seed_sequence, sample_episodes  # noqa: E402
from nist_sim.env import Episode  # noqa: E402
from nist_sim.perception import WristDepth, corrupt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    _, conds = sample_episodes(a.seed, 1, "live")
    t_phys = t_obs = t_depth = 0.0
    nsteps = 0
    for i, c in enumerate(conds[: a.episodes]):
        r1, r2 = (np.random.default_rng(s) for s in episode_seed_sequence(a.seed, i).spawn(2))
        e = Episode(c, r1)
        cam = WristDepth(e.model, n=a.n)
        rng = np.random.default_rng(i)
        act = np.zeros(5)
        for _ in range(a.steps):
            if e.done:
                break
            t0 = time.perf_counter(); e.observe(r2); t1 = time.perf_counter()
            corrupt(cam(e.data, e.data.qpos[0:3], float(e.data.qpos[3])), r2); t2 = time.perf_counter()
            act = np.clip(0.9 * act + rng.normal(0, 0.3, 5), -1, 1)
            e.step(act); t3 = time.perf_counter()
            t_obs += t1 - t0; t_depth += t2 - t1; t_phys += t3 - t2; nsteps += 1
    out = dict(steps=nsteps, n=a.n, ms_phys=1e3 * t_phys / nsteps, ms_obs=1e3 * t_obs / nsteps,
               ms_depth=1e3 * t_depth / nsteps)
    out["depth_overhead_frac"] = out["ms_depth"] / (out["ms_phys"] + out["ms_obs"])
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}))


if __name__ == "__main__":
    main()
