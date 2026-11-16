"""Short CPU PPO-style baseline for v0.8 (session-budget learnability check).

Trains a tiny linear+tanh policy with a REINFORCE/PPO-lite update on sparse
episode returns. Not competitive — only to show a learnable gradient exists.

    NIST_PROFILE=v08 python tools/train_rl_v08.py --steps 200 --eval-episodes 16
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

os.environ.setdefault("NIST_PROFILE", "v08")

from nist_sim.compound_v08 import (  # noqa: E402
    OBS_DIM_V08,
    AssemblyIVEpisode,
    compound_score_v08,
    sample_compound_v08,
)
from nist_sim.conditions import episode_seed_sequence  # noqa: E402
from nist_sim.env import ACT_DIM  # noqa: E402


class LinearPolicy:
    def __init__(self, obs_dim: int, act_dim: int, rng: np.random.Generator):
        self.W = rng.normal(0, 0.02, (obs_dim, act_dim)).astype(np.float64)
        self.b = np.zeros(act_dim, dtype=np.float64)
        self.log_std = np.full(act_dim, -0.7, dtype=np.float64)

    def act(self, o: np.ndarray, rng: np.random.Generator, *, deterministic: bool = False):
        mu = np.tanh(o @ self.W + self.b)
        if deterministic:
            return np.clip(mu, -1, 1), mu, 0.0
        std = np.exp(self.log_std)
        a = mu + std * rng.normal(0, 1, mu.shape)
        a = np.clip(a, -1, 1)
        # Gaussian logprob under pre-clip (approx).
        logp = -0.5 * np.sum(((a - mu) / std) ** 2 + 2 * self.log_std + np.log(2 * np.pi))
        return a, mu, float(logp)

    def update(self, batch_o, batch_a, batch_adv, lr: float = 3e-3):
        # Simple policy gradient on tanh-linear mean.
        for o, a, adv in zip(batch_o, batch_a, batch_adv):
            if abs(adv) < 1e-8:
                continue
            mu = np.tanh(o @ self.W + self.b)
            # d(tanh)/dz = 1 - tanh^2
            g_mu = adv * (a - mu)  # push mean toward actions that got positive advantage
            g_pre = g_mu * (1.0 - mu ** 2)
            self.W += lr * np.outer(o, g_pre)
            self.b += lr * g_pre


def rollout(policy: LinearPolicy, seed: int, ep_i: int, cond, *, train: bool):
    ss = episode_seed_sequence(seed, ep_i)
    r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
    act_rng = np.random.default_rng(ss.spawn(1)[0])
    ep = AssemblyIVEpisode(cond, r1)
    obs, acts, logps = [], [], []
    while not ep.done:
        o = ep.observe(r2).astype(np.float64)
        a, _, logp = policy.act(o, act_rng, deterministic=not train)
        ep.step(a)
        if train:
            obs.append(o)
            acts.append(a)
            logps.append(logp)
    score = float(compound_score_v08(ep)["score"])
    return score, obs, acts, logps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seed", type=int, default=1)
    ap.add_argument("--eval-seed", type=int, default=7)
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--steps", type=int, default=120)  # episode updates
    ap.add_argument("--eval-episodes", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    policy = LinearPolicy(OBS_DIM_V08, ACT_DIM, rng)
    _, train_conds = sample_compound_v08(a.train_seed, a.per, "live")
    _, eval_conds = sample_compound_v08(a.eval_seed, a.per, "live")

    t0 = time.time()
    returns = []
    for t in range(a.steps):
        i = int(rng.integers(0, len(train_conds)))
        sc, obs, acts, _ = rollout(policy, a.train_seed, i, train_conds[i], train=True)
        returns.append(sc)
        # Advantage = return - baseline (running mean).
        baseline = float(np.mean(returns[-32:]))
        adv = sc - baseline
        if obs:
            policy.update(obs, acts, [adv] * len(obs), lr=a.lr)
        if (t + 1) % 20 == 0:
            print(f"train step {t+1}/{a.steps} recent_mean={float(np.mean(returns[-20:])):.4f}", flush=True)

    # Eval
    eval_scores = []
    for i in range(min(a.eval_episodes, len(eval_conds))):
        sc, _, _, _ = rollout(policy, a.eval_seed, i, eval_conds[i], train=False)
        eval_scores.append(sc)
    print(
        f"rl_v08 eval_mean={float(np.mean(eval_scores)):.4f} "
        f"train_last20={float(np.mean(returns[-20:])):.4f} "
        f"wall={time.time()-t0:.1f}s n_eval={len(eval_scores)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
