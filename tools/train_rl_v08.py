"""Black-box residual learner around the observation-only scripted controller.

Learns metre-space grasp/place/lift parameters of the public FSM with random
search + CEM polish, then evaluates against the public scripted baseline with
bootstrap and paired confidence intervals. Used for the v0.8 learnability gate
(not a shipped miner baseline).

    NIST_PROFILE=v08 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
      python tools/train_rl_v08.py --out /tmp/v08-rl.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tools"))

os.environ.setdefault("NIST_PROFILE", "v08")

from nist_sim.compound_v08 import AssemblyIVEpisode, compound_score_v08, sample_compound_v08  # noqa: E402
from nist_sim.conditions import episode_seed_sequence  # noqa: E402
from obs_scripted_v08 import (  # noqa: E402
    GRASP_Z_BIAS_M,
    LIFT_Z_M,
    PLACE_Z_M,
    ObsScriptedPolicy,
)

# Public baseline theta matches the strongest shipped observation-only defaults.
PUBLIC_THETA = np.array([GRASP_Z_BIAS_M, PLACE_Z_M, LIFT_Z_M], dtype=np.float64)


class ResidualPolicy(ObsScriptedPolicy):
    """Same FSM as the public scripted controller with learned metre-space knobs.

    ``theta`` layout: ``[grasp_bias_m, place_z_m, lift_z_m]``.
    """

    def __init__(self, seed: int, theta: np.ndarray):
        th = np.asarray(theta, dtype=np.float64).reshape(-1)
        if th.size < 3:
            padded = PUBLIC_THETA.copy()
            padded[: th.size] = th
            th = padded
        super().__init__(
            seed,
            grasp_bias_m=float(th[0]),
            place_z_m=float(th[1]),
            lift_z_m=float(th[2]),
        )


def eval_policy(
    theta,
    seed: int,
    episodes: int,
    per: int,
    *,
    baseline: bool = False,
) -> np.ndarray:
    _, conds = sample_compound_v08(seed, per, "live")
    scores = []
    for i in range(min(episodes, len(conds))):
        ss = episode_seed_sequence(seed, i)
        r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
        ep = AssemblyIVEpisode(conds[i], r1)
        seed_i = int(ss.generate_state(1)[0])
        if baseline:
            pol: ObsScriptedPolicy = ObsScriptedPolicy(seed_i)
        else:
            pol = ResidualPolicy(seed_i, theta)
        while not ep.done:
            ep.step(pol.act(ep.observe(r2)))
        scores.append(float(compound_score_v08(ep)["score"]))
    return np.asarray(scores, dtype=np.float64)


def boot_ci(scores: np.ndarray, seed: int = 20261117, n: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(scores, (n, len(scores)), replace=True), axis=1)
    return [float(x) for x in np.quantile(boot, [0.025, 0.975])]


def paired_ci(delta: np.ndarray, seed: int = 20261117, n: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(delta, (n, len(delta)), replace=True), axis=1)
    return [float(x) for x in np.quantile(boot, [0.025, 0.975])]


def _clip_theta(th: np.ndarray) -> np.ndarray:
    out = np.asarray(th, dtype=np.float64).copy()
    out[0] = float(np.clip(out[0], -0.006, 0.016))
    out[1] = float(np.clip(out[1], 0.016, 0.040))
    out[2] = float(np.clip(out[2], 0.080, 0.120))
    return out


def train(args) -> dict:
    t0 = time.monotonic()
    base = eval_policy(PUBLIC_THETA, args.eval_seed, args.eval_episodes, args.eval_per, baseline=True)
    print(f"scripted mean={base.mean():.6f} ci={boot_ci(base)}", flush=True)

    rng = np.random.default_rng(args.seed)
    candidates = [
        PUBLIC_THETA.copy(),
        np.array([-0.002, PLACE_Z_M, LIFT_Z_M]),
        np.array([0.0, 0.022, LIFT_Z_M]),
        np.array([-0.002, 0.022, LIFT_Z_M]),
        np.array([-0.002, 0.024, 0.100]),
        np.array([0.0, 0.020, 0.110]),
        np.array([0.002, 0.026, 0.100]),
        np.array([-0.004, 0.022, 0.108]),
    ]
    for _ in range(args.candidates):
        candidates.append(
            _clip_theta(
                PUBLIC_THETA
                + rng.normal(0.0, [0.003, 0.004, 0.008], size=3)
            )
        )

    best = PUBLIC_THETA.copy()
    best_train = -1.0
    train_seeds = [int(s) for s in args.train_seeds.split(",")]
    for i, th in enumerate(candidates):
        th = _clip_theta(th)
        sc = []
        for s in train_seeds:
            sc.extend(eval_policy(th, s, args.train_episodes, args.train_per).tolist())
        m = float(np.mean(sc))
        if m > best_train:
            best_train = m
            best = th.copy()
            print(f"cand={i} train={m:.4f} th={np.round(th, 4).tolist()}", flush=True)

    mu = best.copy()
    std = np.array([0.0025, 0.0030, 0.0060])
    for g in range(args.polish_gens):
        thetas = [mu.copy()]
        for _ in range(args.polish_pop):
            thetas.append(_clip_theta(mu + rng.normal(0.0, std)))
        scores = []
        for th in thetas:
            sc = []
            for s in train_seeds:
                sc.extend(
                    eval_policy(th, s, max(2, args.train_episodes - 1), args.train_per).tolist()
                )
            scores.append(float(np.mean(sc)))
        order = np.argsort(-np.asarray(scores))
        mu = thetas[order[0]].copy()
        ev_base = eval_policy(
            PUBLIC_THETA,
            args.eval_seed,
            min(32, args.eval_episodes),
            args.eval_per,
            baseline=True,
        )
        ev = eval_policy(mu, args.eval_seed, min(32, args.eval_episodes), args.eval_per)
        print(
            f"polish gen={g} train_best={scores[order[0]]:.4f} "
            f"eval32 scripted={ev_base.mean():.4f} rl={ev.mean():.4f} "
            f"delta={ev.mean() - ev_base.mean():.4f} paired={paired_ci(ev - ev_base)} "
            f"mu={np.round(mu, 4).tolist()}",
            flush=True,
        )
        if scores[order[0]] >= best_train:
            best_train = scores[order[0]]
            best = mu.copy()
        std *= 0.85

    rl = eval_policy(best, args.eval_seed, args.eval_episodes, args.eval_per)
    delta = rl - base
    out = {
        "profile": "v08",
        "method": "CEM metre-space grasp/place/lift parameter search around observation-only scripted controller",
        "scripted_mean": float(base.mean()),
        "scripted_ci95": boot_ci(base),
        "rl_mean": float(rl.mean()),
        "rl_ci95": boot_ci(rl),
        "delta": float(delta.mean()),
        "paired_ci95": paired_ci(delta),
        "wins": int((delta > 0).sum()),
        "losses": int((delta < 0).sum()),
        "ties": int((delta == 0).sum()),
        "theta": best.tolist(),
        "public_theta": PUBLIC_THETA.tolist(),
        "best_train": best_train,
        "train_seeds": train_seeds,
        "eval_seed": args.eval_seed,
        "eval_episodes": args.eval_episodes,
        "wall_s": time.monotonic() - t0,
        "scripted_scores": base.tolist(),
        "rl_scores": rl.tolist(),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if "scores" not in k}, indent=2), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seeds", default="101,102,103,104,105,106")
    ap.add_argument("--train-per", type=int, default=4)
    ap.add_argument("--train-episodes", type=int, default=4)
    ap.add_argument("--eval-seed", type=int, default=7)
    ap.add_argument("--eval-per", type=int, default=8)
    ap.add_argument("--eval-episodes", type=int, default=64)
    ap.add_argument("--candidates", type=int, default=16)
    ap.add_argument("--polish-gens", type=int, default=5)
    ap.add_argument("--polish-pop", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
