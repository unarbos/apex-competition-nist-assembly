"""Accept-only evolutionary residual tuning around the public scripted controller.

Selection uses training seeds plus disjoint validation seeds, never the reported
test seed. The initial zero-residual incumbent is retained unless a candidate
improves both validation mean and validation minus baseline mean. Evaluation
uses only observations and a small observation-history linear residual model.

    NIST_PROFILE=v08 python tools/evolve_residual_v08.py \
      --generations 12 --population 4 --train-seeds 101,102 --validation-seeds 201,202 \
      --train-per 4 --validation-per 4 --eval-seed 7 --eval-episodes 8 --out /tmp/v08-es
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "referee"), str(ROOT / "tools")]
os.environ.setdefault("NIST_PROFILE", "v08")

from nist_sim.compound_v08 import (  # noqa: E402
    OBS_DIM_V08,
    AssemblyIVEpisode,
    compound_score_v08,
    sample_compound_v08,
)
from nist_sim.conditions import episode_seed_sequence  # noqa: E402
from nist_sim.env import ACT_DIM  # noqa: E402
from obs_scripted_v08 import ObsScriptedPolicy  # noqa: E402


def _episode(job):
    seed, per, i, weights, scale, feature_dim, cond = job
    if cond is None:
        _, conds = sample_compound_v08(seed, per, "live")
        cond = conds[i]
    ss = episode_seed_sequence(seed, i)
    r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
    ep = AssemblyIVEpisode(cond, r1)
    base = ObsScriptedPolicy(int(ss.generate_state(1)[0]))
    theta = np.asarray(weights, dtype=np.float64).reshape(feature_dim, ACT_DIM)
    zero_residual = not np.any(theta)
    history = np.zeros((3, OBS_DIM_V08), dtype=np.float64)
    while not ep.done:
        obs = ep.observe(r2)
        if zero_residual:
            # The incumbent must be exactly the public controller. Do not even
            # round-trip its observation through a residual feature dtype.
            ep.step(base.act(obs))
            continue
        history[1:] = history[:-1]
        history[0] = obs
        features = np.concatenate([obs.astype(np.float64), np.mean(history, axis=0)])
        residual = np.clip(np.tanh(features @ theta), -1.0, 1.0)
        action = np.clip(base.act(obs) + scale * residual, -1.0, 1.0)
        ep.step(action)
    score = float(compound_score_v08(ep)["score"])
    return {"episode": i, "score": score, "terminal_reason": ep.terminal_reason}


def _summary(scores, seed=20261117):
    arr = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(arr, (10000, len(arr)), replace=True), axis=1)
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
        "ci_method": "episode bootstrap, 10000 resamples, seed 20261117",
    }


class Evaluator:
    def __init__(self, seed, per, episodes, scale, workers):
        self.seed, self.per, self.episodes = seed, per, episodes
        self.scale, self.workers = scale, workers
        self.feature_dim = 2 * OBS_DIM_V08
        self.zero = np.zeros((self.feature_dim, ACT_DIM), dtype=np.float64).ravel()
        self.cache = {}
        self.rollouts = 0
        self.eval_wall_s = 0.0
        self.conditions = {}
        for s in self.seed:
            _, conds = sample_compound_v08(s, self.per, "live")
            for i, cond in enumerate(conds[:self.per]):
                self.conditions[(s, i)] = cond

    def evaluate(self, weights):
        key = np.asarray(weights, dtype=np.float64).tobytes()
        if key in self.cache:
            return self.cache[key]
        t0 = time.monotonic()
        jobs = [
            (s, self.per, i, weights, self.scale, self.feature_dim, self.conditions[(s, i)])
            for s in self.seed
            for i in range(min(self.episodes, self.per))
        ]
        if self.workers > 1:
            ctx = __import__("multiprocessing").get_context("fork")
            with concurrent.futures.ProcessPoolExecutor(self.workers, mp_context=ctx) as pool:
                rows = list(pool.map(_episode, jobs))
        else:
            rows = [_episode(job) for job in jobs]
        self.rollouts += len(rows)
        self.eval_wall_s += time.monotonic() - t0
        scores = [float(r["score"]) for r in rows]
        out = {**_summary(scores), "scores": scores, "episodes": rows}
        self.cache[key] = out
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--population", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--decay", type=float, default=0.92)
    ap.add_argument("--scale", type=float, default=0.35)
    ap.add_argument("--train-seeds", default="101,102")
    ap.add_argument("--validation-seeds", default="201,202")
    ap.add_argument("--train-per", type=int, default=4)
    ap.add_argument("--validation-per", type=int, default=4)
    ap.add_argument("--train-episodes", type=int, default=8)
    ap.add_argument("--validation-episodes", type=int, default=8)
    ap.add_argument("--eval-seed", type=int, default=7)
    ap.add_argument("--eval-per", type=int, default=8)
    ap.add_argument("--eval-episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    rng = np.random.default_rng(args.seed)
    train = Evaluator([int(s) for s in args.train_seeds.split(",")], args.train_per, args.train_episodes, args.scale, args.workers)
    validation = Evaluator([int(s) for s in args.validation_seeds.split(",")], args.validation_per, args.validation_episodes, args.scale, args.workers)
    test = Evaluator([args.eval_seed], args.eval_per, args.eval_episodes, args.scale, args.workers)
    incumbent = train.zero.copy()
    train_base = train.evaluate(incumbent)
    val_base = validation.evaluate(incumbent)
    test_base = test.evaluate(incumbent)
    best = incumbent.copy()
    best_train = train_base["mean"]
    best_val = val_base["mean"]
    history = []
    t0 = time.monotonic()
    print("baseline", json.dumps({"train": train_base["mean"], "validation": best_val, "test": test_base["mean"]}), flush=True)
    for generation in range(args.generations):
        sigma = args.sigma * args.decay ** generation
        accepted = 0
        for member in range(args.population):
            delta = rng.normal(0.0, sigma, incumbent.shape)
            candidate = np.clip(incumbent + delta, -0.5, 0.5)
            train_result = train.evaluate(candidate)
            # Use the cheaper training screen only to avoid spending validation
            # rollouts on clear regressions. It cannot by itself accept a model.
            if train_result["mean"] < best_train - 1e-12:
                history.append({"generation": generation, "member": member, "sigma": sigma, "accepted": False, "train": train_result["mean"], "screen": "reject"})
                continue
            val_result = validation.evaluate(candidate)
            # Both disjoint sets must improve: no acceptance from a single
            # training sample, and no degradation of the incumbent on validation.
            accept = val_result["mean"] > best_val + 1e-12 and train_result["mean"] > best_train + 1e-12
            history.append({
                "generation": generation, "member": member, "sigma": sigma,
                "accepted": accept, "train": train_result["mean"], "validation": val_result["mean"],
            })
            if accept:
                incumbent = candidate
                best = candidate.copy()
                best_train = train_result["mean"]
                best_val = val_result["mean"]
                accepted += 1
        print(json.dumps({"generation": generation, "sigma": sigma, "accepted": accepted, "train": best_train, "validation": best_val, "wall_s": time.monotonic() - t0}), flush=True)
        np.save(out / "best.npy", best)
        (out / "history.json").write_text(json.dumps(history, indent=2))
    test_result = test.evaluate(best)
    result = {
        "algorithm": "accept-only evolutionary linear residual around observation-only scripted controller",
        "feature_contract": "current 320-d observation plus running three-frame mean; no true state",
        "parameters": int(best.size),
        "accepted": sum(bool(row["accepted"]) for row in history),
        "train": train.evaluate(best),
        "validation": validation.evaluate(best),
        "test": test_result,
        "test_baseline": test_base,
        "train_baseline": train_base,
        "validation_baseline": val_base,
        "wall_s": time.monotonic() - t0,
        "rollouts": train.rollouts + validation.rollouts + test.rollouts,
        "args": vars(args),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("accepted", "wall_s", "rollouts")}), flush=True)
    print("test", json.dumps(test_result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
