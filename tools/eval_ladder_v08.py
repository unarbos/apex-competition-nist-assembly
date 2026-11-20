"""Matched held-out policy ladder and referee-seed separation report.

All policies see identical conditions for a given referee seed. Selection and
training are performed elsewhere; this tool only evaluates supplied artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "referee"), str(ROOT / "tools")]
os.environ.setdefault("NIST_PROFILE", "v08")

from nist_sim.compound_v08 import (
    AssemblyIVEpisode,
    compound_score_v08,
    sample_compound_v08,
    step_results_v08,
)
from nist_sim.conditions import episode_seed_sequence
from obs_scripted_v08 import ObsScriptedPolicy
from obs_boundary_v08 import ObservationBoundary
from obs_strong_v08 import solve_obs_episode

SIMULATOR_FILES = (
    "referee/nist_sim/compound_v08.py",
    "referee/nist_sim/compound_v07.py",
    "referee/nist_sim/compound.py",
    "referee/nist_sim/env.py",
    "referee/nist_sim/conditions.py",
    "referee/nist_sim/profile.py",
)


def simulator_sha256() -> str:
    """Hash every inherited implementation used by Assembly IV."""
    digest = hashlib.sha256()
    for name in SIMULATOR_FILES:
        digest.update(name.encode("utf-8") + b"\0")
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    """Compatibility guards must remain active under ``python -O``."""
    if not condition:
        raise ValueError(message)


def summary(scores, seed=20261117):
    arr = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(arr, (10000, len(arr)), replace=True), axis=1) if len(arr) else np.zeros(1)
    return {"n": len(arr), "mean": float(np.mean(arr)) if len(arr) else 0.0, "ci95": np.quantile(means, [0.025, 0.975]).tolist(), "scores": arr.tolist()}


def matched_improvement(scores, baseline):
    """Paired episode bootstrap on identically ordered matched scores."""
    delta = np.asarray(scores, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    if not len(delta):
        return {"mean": 0.0, "ci95": [0.0, 0.0], "n": 0}
    rng = np.random.default_rng(20261117)
    boot = np.mean(rng.choice(delta, (10000, len(delta)), replace=True), axis=1)
    return {
        "mean": float(np.mean(delta)),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "n": len(delta),
        "method": "paired episode bootstrap, 10000 resamples, seed 20261117",
    }


def make_report(kind, seeds, episodes, start_episode, rounds, model_path="", residual_scale=0.35, residual_policy="script", privileged_mode="assisted", per=8):
    pooled = [s for r in rounds for s in r["scores"]]
    means = [r["mean"] for r in rounds]
    return {
        "kind": kind,
        "seeds": seeds,
        "episodes_per_round": episodes,
        "per": per,
        "start_episode": start_episode,
        "round_means": means,
        "round_std_population": float(np.std(means)),
        "round_std_sample": float(np.std(means, ddof=1)) if len(rounds) > 1 else 0.0,
        "pooled": summary(pooled),
        "rounds": rounds,
        "model": model_path,
        "residual_scale": residual_scale,
        "residual_policy": residual_policy,
        "privileged_mode": privileged_mode,
        "complete_rounds": len(rounds),
        "simulator_sha256": simulator_sha256(),
        "controller_sha256": hashlib.sha256((ROOT / "tools" / "compound_solver.py").read_bytes()).hexdigest() if kind == "privileged" else None,
        "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest() if model_path else None,
    }


def run(seed, per, episodes, kind, model_path="", residual_scale=0.35, residual_policy="script", privileged_mode="assisted", start_episode=0):
    if episodes <= 0 or start_episode < 0:
        raise ValueError("matched evaluation requires a positive episode count and non-negative start")
    _, conds = sample_compound_v08(seed, per, "live")
    model = None
    if kind == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device="cpu")
    elif kind == "cem":
        if not model_path:
            raise ValueError("CEM evaluation requires --model with selected theta; use --kind script for the public baseline")
        from train_rl_v08 import ResidualPolicy
        artifact = json.loads(Path(model_path).read_text())
        theta = np.asarray(artifact["theta"], dtype=np.float64)
    scores, rows = [], []
    t0 = time.monotonic()
    for i in range(start_episode, min(start_episode + episodes, len(conds))):
        ss = episode_seed_sequence(seed, i)
        r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
        ep = AssemblyIVEpisode(conds[i], r1)
        if kind == "privileged":
            if privileged_mode != "assisted":
                raise ValueError("this scoring-fix PR measures the legacy assisted privileged reference")
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
                if kind == "cem":
                    base = ResidualPolicy(int(ss.generate_state(1)[0]), theta)
                while not boundary.done:
                    obs = boundary.observe(r2)
                    if kind == "zero":
                        action = np.zeros(5)
                    elif kind == "random":
                        rng = np.random.default_rng(
                            int(ss.generate_state(1)[0]) ^ (i * 0x9E3779B9) ^ ep.steps
                        )
                        action = rng.uniform(-1.0, 1.0, 5)
                    elif kind in ("script", "cem"):
                        action = base.act(obs)
                    else:
                        residual, _ = model.predict(obs, deterministic=True)
                        action = np.clip(base.act(obs) + residual_scale * residual, -1.0, 1.0)
                    boundary.step(action)
        info = compound_score_v08(ep)
        score = float(info["score"])
        scores.append(score)
        milestones = {
            name: sum(bool(row["partial"][name]) for row in step_results_v08(ep))
            for name in ("lift", "transport", "insert", "align", "stable")
        }
        rows.append({
            "episode": i,
            "score": score,
            "terminal_reason": ep.terminal_reason,
            "n_steps": info["n_steps"],
            "n_done": info["n_done"],
            "completed_kinds": [ep.cs.steps[k].kind for k, done in enumerate(ep.step_done) if done],
            "physical_milestones": milestones,
        })
        print(f"{kind} seed={seed} episode={i} score={score:.9f}", flush=True)
    if len(scores) != episodes:
        raise ValueError(f"requested {episodes} episodes but only {len(scores)} conditions are available")
    return {**summary(scores), "seed": seed, "episodes": rows,
            "episode_score_std_sample": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "wall_s": time.monotonic() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7,19,31")
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--start-episode", type=int, default=0, help="first matched episode index for a resumable run")
    ap.add_argument("--kind", choices=("zero", "random", "script", "strong_script", "ppo", "cem", "privileged"), required=True)
    ap.add_argument("--model", default="", help="PPO model or CEM JSON with theta")
    ap.add_argument("--residual-scale", type=float, default=0.35)
    ap.add_argument("--residual-policy", choices=("script", "strong"), default="script")
    ap.add_argument("--privileged-mode", choices=("assisted",), default="assisted", help="legacy privileged reference; includes simulator-state assists")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="", help="optional round-level report updated after each completed seed")
    ap.add_argument("--compare", default="", help="matched strongest-script report for paired separation")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out)
    checkpoint = Path(args.checkpoint) if args.checkpoint else out
    rounds = []
    if checkpoint.exists():
        previous = json.loads(checkpoint.read_text())
        require(previous["kind"] == args.kind, "checkpoint kind differs")
        require(previous["seeds"] == seeds, "checkpoint seeds differ")
        require(previous["episodes_per_round"] == args.episodes, "checkpoint episode count differs")
        require(previous.get("per") == args.per, "checkpoint conditions-per-seed differs")
        require(previous.get("start_episode", 0) == args.start_episode, "checkpoint start differs")
        require(previous.get("model", "") == args.model, "checkpoint model differs")
        require(previous.get("residual_scale", 0.35) == args.residual_scale, "checkpoint residual scale differs")
        require(previous.get("residual_policy", "script") == args.residual_policy, "checkpoint residual policy differs")
        require(previous.get("privileged_mode", "assisted") == args.privileged_mode, "checkpoint privileged mode differs")
        require(previous.get("simulator_sha256") == simulator_sha256(), "refuse to resume a different scorer")
        if args.kind == "privileged":
            require(previous["controller_sha256"] == hashlib.sha256((ROOT / "tools" / "compound_solver.py").read_bytes()).hexdigest(), "checkpoint controller differs")
        if args.model:
            require(previous["model_sha256"] == hashlib.sha256(Path(args.model).read_bytes()).hexdigest(), "checkpoint model hash differs")
        rounds = previous["rounds"]
        require([r["seed"] for r in rounds] == seeds[:len(rounds)], "checkpoint round order differs")
        for rr in rounds:
            indices = [row["episode"] for row in rr["episodes"]]
            require(indices == list(range(args.start_episode, args.start_episode + args.episodes)), "checkpoint episode indices are not contiguous")
    for seed in seeds[len(rounds):]:
        rounds.append(run(seed, args.per, args.episodes, args.kind, args.model, args.residual_scale, args.residual_policy, args.privileged_mode, args.start_episode))
        report = make_report(args.kind, seeds, args.episodes, args.start_episode, rounds, args.model, args.residual_scale, args.residual_policy, args.privileged_mode, args.per)
        checkpoint.write_text(json.dumps(report, indent=2))
        print(f"checkpoint completed_rounds={len(rounds)} path={checkpoint}", flush=True)
    report = make_report(args.kind, seeds, args.episodes, args.start_episode, rounds, args.model, args.residual_scale, args.residual_policy, args.privileged_mode, args.per)
    pooled = [s for r in rounds for s in r["scores"]]
    if args.compare:
        other = json.loads(Path(args.compare).read_text())
        require(other["seeds"] == seeds and other["episodes_per_round"] == args.episodes, "matched episode layout differs")
        require(other.get("per") == args.per, "matched reports must use the same conditions per seed")
        require(other.get("start_episode", 0) == args.start_episode, "matched start differs")
        require(other.get("complete_rounds", len(other["rounds"])) == len(other["rounds"]), "matched report is incomplete")
        require([r["seed"] for r in other["rounds"]] == seeds, "matched seed order differs")
        require(other.get("simulator_sha256") == report["simulator_sha256"], "matched reports must use the same scorer")
        for rr in other["rounds"]:
            require([row["episode"] for row in rr["episodes"]] == list(range(args.start_episode, args.start_episode + args.episodes)), "matched episode indices differ")
        baseline = [s for r in other["rounds"] for s in r["scores"]]
        require(len(baseline) == len(pooled), "matched score counts differ")
        report["matched_improvement"] = {
            "baseline_kind": other["kind"],
            **matched_improvement(pooled, baseline),
            "pooled_cis_nonoverlap": bool(report["pooled"]["ci95"][0] > other["pooled"]["ci95"][1]),
        }
        report["matched_improvement"]["paired_ci_excludes_zero"] = bool(report["matched_improvement"]["ci95"][0] > 0.0)
        out.write_text(json.dumps(report, indent=2))
    if checkpoint != out:
        out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("kind", "round_means", "round_std_sample", "pooled")}))


if __name__ == "__main__":
    main()
