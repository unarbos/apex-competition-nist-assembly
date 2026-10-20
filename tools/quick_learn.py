"""Short CPU hill-climb on a linear policy — evidence that learning helps.

Warm-starts from a ridge fit of NearestFirstBaseline rollouts (same recipe as
``make_policies.py``), then mutates weights for a few dozen iterations.
Compares against ObsOnlyScripted on the same episode batch.

    PYTHONPATH=referee python tools/quick_learn.py --iters 25 --episodes 16
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
from nist_sim.policies import NearestFirstBaseline, ObsOnlyScripted  # noqa: E402
from nist_sim.scoring import episode_score  # noqa: E402


def _rollout_score(act_fn, seed: int, n_eps: int, tag: int = 0) -> float:
    # per=2 → 16 episodes across all strata.
    _, conds = sample_episodes(seed, 2)
    scores = []
    for i, c in enumerate(conds[:n_eps]):
        ss = np.random.SeedSequence([seed, 0x11EA5 + tag, i])
        r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
        e = Episode(c, r1)
        while not e.done:
            o = e.observe(r2)
            e.step(np.clip(act_fn(o), -1, 1))
        seated = e.peg_seated_at_end()
        first = e.peg_first_seat_steps()
        msa = e.action_l2 / max(1, e.steps)
        scores.append(episode_score(seated, first, c.max_steps, msa))
    return float(np.mean(scores))


def _distill_baseline(seeds=(0, 1), max_eps=12, max_steps=200) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for seed in seeds:
        _, conds = sample_episodes(seed, 2)
        for i, c in enumerate(conds[:max_eps]):
            ss = np.random.SeedSequence([seed, 0x5245, i])
            r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
            e = Episode(c, r1)
            pol = NearestFirstBaseline()
            steps = 0
            while not e.done and steps < max_steps:
                o = e.observe(r2)
                a = np.clip(pol.act(o), -1, 1)
                xs.append(o)
                ys.append(a)
                e.step(a)
                steps += 1
    X = np.asarray(xs, np.float64)
    Y = np.asarray(ys, np.float64)
    lam = 1e-2
    W = np.linalg.solve(X.T @ X + lam * np.eye(OBS_DIM), X.T @ Y)
    b = Y.mean(axis=0) - X.mean(axis=0) @ W
    return W.astype(np.float64), b.astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print("distilling baseline warm-start…", flush=True)
    W, b = _distill_baseline()
    rng = np.random.default_rng(a.seed + 7)

    def linear_act(o, W=W, b=b):
        return o @ W + b

    warm = _rollout_score(linear_act, a.seed, a.episodes, tag=1)
    print(f"warm linear={warm:.4f}", flush=True)
    obs_score = _rollout_score(lambda o: ObsOnlyScripted().act(o), a.seed, a.episodes, tag=1)
    # Fresh policy instance each step would reset — fix: one policy per episode inside helper.
    # Re-eval obs scripted properly:
    def obs_fn_factory():
        pol = {"p": ObsOnlyScripted()}

        def fn(o):
            return pol["p"].act(o)

        return fn

    # Proper obs eval with persistent policy state per episode:
    _, conds = sample_episodes(a.seed, 2)
    obs_scores = []
    for i, c in enumerate(conds[: a.episodes]):
        ss = np.random.SeedSequence([a.seed, 0x11EA5 + 1, i])
        r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
        e = Episode(c, r1)
        pol = ObsOnlyScripted()
        while not e.done:
            e.step(np.clip(pol.act(e.observe(r2)), -1, 1))
        seated = e.peg_seated_at_end()
        first = e.peg_first_seat_steps()
        msa = e.action_l2 / max(1, e.steps)
        obs_scores.append(episode_score(seated, first, c.max_steps, msa))
    obs_score = float(np.mean(obs_scores))
    print(f"obs_scripted={obs_score:.4f}", flush=True)

    best = warm
    history = [best]
    for t in range(a.iters):
        scale = 0.05 * (0.94 ** t)
        W2 = W + rng.normal(0, scale, W.shape)
        b2 = b + rng.normal(0, scale * 0.5, b.shape)

        def act2(o, W2=W2, b2=b2):
            return o @ W2 + b2

        sc = _rollout_score(act2, a.seed + (t % 5), a.episodes, tag=2 + t)
        if sc > best + 1e-6:
            best, W, b = sc, W2, b2
            print(f"iter {t:02d} improved → {best:.4f}", flush=True)
        history.append(best)

    # Final compare on a held seed
    final = _rollout_score(lambda o: o @ W + b, a.seed + 11, a.episodes, tag=99)
    obs_hold = []
    _, conds = sample_episodes(a.seed + 11, 2)
    for i, c in enumerate(conds[: a.episodes]):
        ss = np.random.SeedSequence([a.seed + 11, 0x11EA5 + 99, i])
        r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
        e = Episode(c, r1)
        pol = ObsOnlyScripted()
        while not e.done:
            e.step(np.clip(pol.act(e.observe(r2)), -1, 1))
        seated = e.peg_seated_at_end()
        first = e.peg_first_seat_steps()
        msa = e.action_l2 / max(1, e.steps)
        obs_hold.append(episode_score(seated, first, c.max_steps, msa))
    obs_hold_m = float(np.mean(obs_hold))
    print(
        f"RESULT: warm={warm:.4f} hillclimb_best={best:.4f} "
        f"held_linear={final:.4f} held_obs_scripted={obs_hold_m:.4f} "
        f"delta={final - obs_hold_m:+.4f}",
        flush=True,
    )
    out = ROOT / "out" / "quick_learn.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        W=W,
        b=b,
        warm=warm,
        best=best,
        held_linear=final,
        held_obs=obs_hold_m,
        history=np.array(history),
    )
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
