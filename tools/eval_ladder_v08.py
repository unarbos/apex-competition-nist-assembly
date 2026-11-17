"""Matched held-out policy ladder and referee-seed separation report.

All policies see identical conditions for a given referee seed. Selection and
training are performed elsewhere; this tool only evaluates supplied artifacts.
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
sys.path[:0] = [str(ROOT / "referee"), str(ROOT / "tools")]
os.environ.setdefault("NIST_PROFILE", "v08")

from nist_sim.compound_v08 import AssemblyIVEpisode, compound_score_v08, sample_compound_v08
from nist_sim.conditions import episode_seed_sequence
from obs_scripted_v08 import ObsScriptedPolicy
from obs_boundary_v08 import ObservationBoundary
from obs_strong_v08 import solve_obs_episode


def summary(scores, seed=20261117):
    arr = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(arr, (10000, len(arr)), replace=True), axis=1)
    return {"n": len(arr), "mean": float(np.mean(arr)), "ci95": np.quantile(means, [0.025, 0.975]).tolist(), "scores": arr.tolist()}


def run(seed, per, episodes, kind, model_path="", residual_scale=0.35, residual_policy="script"):
    _, conds = sample_compound_v08(seed, per, "live")
    model = None
    if kind == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device="cpu")
    scores, rows = [], []
    t0 = time.monotonic()
    for i in range(min(episodes, len(conds))):
        ss = episode_seed_sequence(seed, i)
        r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
        ep = AssemblyIVEpisode(conds[i], r1)
        if kind == "privileged":
            from compound_solver import solve_compound
            solve_compound(ep, oracle=False)
        else:
            boundary = ObservationBoundary(ep)
            if kind == "strong_script":
                solve_obs_episode(boundary, r2)
            elif kind == "ppo" and residual_policy == "strong":
                from strong_residual_v08 import StrongResidualController
                obs = boundary.observe(r2)
                strong = StrongResidualController(boundary, r2, initial_obs=obs)
                while not boundary.done:
                    residual, _ = model.predict(obs, deterministic=True)
                    action = np.clip(strong.act() + residual_scale * residual, -1.0, 1.0)
                    boundary.step(action)
                    if not boundary.done:
                        obs = boundary.observe(r2)
                        strong.post_step(False, obs, action=action)
            else:
                base = ObsScriptedPolicy(int(ss.generate_state(1)[0])) if kind in ("script", "ppo") else None
                while not boundary.done:
                    obs = boundary.observe(r2)
                    if kind == "zero":
                        action = np.zeros(5)
                    elif kind == "script":
                        action = base.act(obs)
                    else:
                        residual, _ = model.predict(obs, deterministic=True)
                        action = np.clip(base.act(obs) + residual_scale * residual, -1.0, 1.0)
                    boundary.step(action)
        score = float(compound_score_v08(ep)["score"])
        scores.append(score)
        rows.append({"episode": i, "score": score, "terminal_reason": ep.terminal_reason})
        print(f"{kind} seed={seed} episode={i} score={score:.9f}", flush=True)
    return {**summary(scores), "seed": seed, "episodes": rows, "wall_s": time.monotonic() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7,19,31")
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--kind", choices=("zero", "script", "strong_script", "ppo", "privileged"), required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--residual-scale", type=float, default=0.35)
    ap.add_argument("--residual-policy", choices=("script", "strong"), default="script")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default="", help="matched strongest-script report for paired separation")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    rounds = [run(seed, args.per, args.episodes, args.kind, args.model, args.residual_scale, args.residual_policy) for seed in seeds]
    pooled = [s for r in rounds for s in r["scores"]]
    report = {
        "kind": args.kind,
        "seeds": seeds,
        "episodes_per_round": args.episodes,
        "round_means": [r["mean"] for r in rounds],
        "round_std_population": float(np.std([r["mean"] for r in rounds])),
        "round_std_sample": float(np.std([r["mean"] for r in rounds], ddof=1)) if len(rounds) > 1 else 0.0,
        "pooled": summary(pooled),
        "rounds": rounds,
        "model": args.model,
        "residual_scale": args.residual_scale,
        "residual_policy": args.residual_policy,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    if args.compare:
        other = json.loads(Path(args.compare).read_text())
        assert other["seeds"] == seeds and other["episodes_per_round"] == args.episodes
        baseline = [s for r in other["rounds"] for s in r["scores"]]
        assert len(baseline) == len(pooled)
        delta = np.asarray(pooled) - np.asarray(baseline)
        rng = np.random.default_rng(20261117)
        boot = np.mean(rng.choice(delta, (10000, len(delta)), replace=True), axis=1)
        report["matched_improvement"] = {
            "baseline_kind": other["kind"],
            "mean": float(np.mean(delta)),
            "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "method": "paired episode bootstrap, 10000 resamples, seed 20261117",
            "pooled_cis_nonoverlap": bool(report["pooled"]["ci95"][0] > other["pooled"]["ci95"][1]),
            "paired_ci_excludes_zero": bool(np.quantile(boot, 0.025) > 0.0),
        }
        Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("kind", "round_means", "round_std_sample", "pooled")}))


if __name__ == "__main__":
    main()
