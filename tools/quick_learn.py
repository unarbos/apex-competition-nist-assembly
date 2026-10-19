"""Short CPU hill-climb on a linear policy — evidence that learning helps.

No torch required. Evolves a clipped linear map obs→action for a few hundred
episode evaluations and compares against ObsOnlyScripted.

    PYTHONPATH=referee python tools/quick_learn.py --iters 40 --episodes 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))

from nist_sim.conditions import sample_episodes  # noqa: E402
from nist_sim.env import ACT_DIM, OBS_DIM, Episode  # noqa: E402
from nist_sim.policies import ObsOnlyScripted  # noqa: E402
from nist_sim.scoring import episode_score  # noqa: E402


def _eval_linear(W: np.ndarray, b: np.ndarray, seed: int, n_eps: int) -> float:
    _, conds = sample_episodes(seed, 1)
    scores = []
    for i, c in enumerate(conds[:n_eps]):
        ss = np.random.SeedSequence([seed, 0x11EA5, i])
        r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
        e = Episode(c, r1)
        while not e.done:
            o = e.observe(r2)
            a = np.clip(o @ W + b, -1, 1)
            e.step(a)
        seated = e.peg_seated_at_end()
        first = e.peg_first_seat_steps()
        msa = e.action_l2 / max(1, e.steps)
        scores.append(episode_score(seated, first, c.max_steps, msa))
    return float(np.mean(scores))


def _eval_obs_scripted(seed: int, n_eps: int) -> float:
    _, conds = sample_episodes(seed, 1)
    scores = []
    for i, c in enumerate(conds[:n_eps]):
        ss = np.random.SeedSequence([seed, 0xLEAR, i])
        r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
        e = Episode(c, r1)
        pol = ObsOnlyScripted()
        while not e.done:
            e.step(np.clip(pol.act(e.observe(r2)), -1, 1))
        seated = e.peg_seated_at_end()
        first = e.peg_first_seat_steps()
        msa = e.action_l2 / max(1, e.steps)
        scores.append(episode_score(seated, first, c.max_steps, msa))
    return float(np.mean(scores))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed + 7)
    # Warm-start near the distilled baseline if present, else zeros.
    W = np.zeros((OBS_DIM, ACT_DIM), np.float64)
    b = np.zeros(ACT_DIM, np.float64)
    best = _eval_linear(W, b, a.seed, a.episodes)
    print(f"init linear={best:.4f}", flush=True)
    obs_score = _eval_obs_scripted(a.seed, a.episodes)
    print(f"obs_scripted={obs_score:.4f}", flush=True)

    history = [best]
    for t in range(a.iters):
        scale = 0.08 * (0.92 ** t)
        W2 = W + rng.normal(0, scale, W.shape)
        b2 = b + rng.normal(0, scale, b.shape)
        sc = _eval_linear(W2, b2, a.seed + (t % 3), a.episodes)
        if sc >= best:
            best, W, b = sc, W2, b2
            print(f"iter {t:02d} improved → {best:.4f}", flush=True)
        history.append(best)

    print(
        f"RESULT: linear_hillclimb={best:.4f} obs_scripted={obs_score:.4f} "
        f"delta={best - obs_score:+.4f}",
        flush=True,
    )
    out = ROOT / "out" / "quick_learn.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, W=W, b=b, best=best, obs_scripted=obs_score, history=np.array(history))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
