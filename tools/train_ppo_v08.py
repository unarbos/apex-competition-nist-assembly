"""Measured CPU PPO baseline for the v0.8 observation/action contract.

This is a calibration tool, not a shipped baseline. Requires the optional
training dependencies pinned below; the referee image needs only NumPy/MuJoCo.

    pip install torch==2.15.0+cpu stable-baselines3==2.9.0 gymnasium==1.3.0
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NIST_PROFILE=v08 \
      python tools/train_ppo_v08.py --timesteps 250000 --train-seeds 1,2,3 \
      --eval-seed 7 --eval-episodes 32 --out /tmp/v08-ppo
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

from nist_sim.compound_v08 import (  # noqa: E402
    OBS_DIM_V08,
    AssemblyIVEpisode,
    compound_score_v08,
    sample_compound_v08,
    step_results_v08,
)
from nist_sim.conditions import episode_seed_sequence  # noqa: E402
from nist_sim.env import ACT_DIM  # noqa: E402


def train_and_eval(args):
    import gymnasium as gym
    import torch
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv

    class V08Env(gym.Env):
        """Terminal task score, with optional small per-control-step progress deltas."""

        metadata = {"render_modes": []}

        def __init__(self, seed, per, dense, residual=False):
            self.seed, self.per, self.dense, self.residual = seed, per, dense, residual
            _, self.conds = sample_compound_v08(seed, per, "live")
            self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM_V08,), np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
            self.i = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.i = int(self.np_random.integers(0, len(self.conds)))
            self.ss = episode_seed_sequence(self.seed, self.i)
            r1, r2 = (np.random.default_rng(s) for s in self.ss.spawn(2))
            self.ep = AssemblyIVEpisode(self.conds[self.i], r1)
            if self.residual:
                from obs_scripted_v08 import ObsScriptedPolicy
                self.base = ObsScriptedPolicy(int(self.ss.generate_state(1)[0]))
            self.noise = r2
            self.prev = np.zeros(len(self.ep.cs.steps))
            return self.ep.observe(self.noise), {"episode": self.i}

        def step(self, action):
            raw_action = np.asarray(action, dtype=np.float64)
            o = self.ep.observe(self.noise)
            if self.residual:
                raw_action = np.clip(self.base.act(o) + args.residual_scale * raw_action, -1, 1)
            self.ep.step(raw_action)
            if not self.ep.done:
                reward = 0.0
                if self.dense:
                    current = np.array([sum(r["partial"].values()) * 0.06 for r in step_results_v08(self.ep)])
                    reward = float(np.sum(current - self.prev))
                    self.prev = current
                return self.ep.observe(self.noise), reward, False, False, {}
            if self.residual:
                self.base.act(o)  # advance baseline phase even on terminal actions
            info = compound_score_v08(self.ep)
            score = float(info["score"])
            reward = score
            if self.dense:
                current = np.array([r["credit"] for r in step_results_v08(self.ep)])
                reward += float(np.sum(current - self.prev))
            return np.zeros(OBS_DIM_V08, np.float32), reward, True, False, {
                "score": score,
                "episode": self.i,
                "terminal_reason": self.ep.terminal_reason,
                "n_steps": info["n_steps"],
            }

    class Recorder(BaseCallback):
        def __init__(self):
            super().__init__()
            self.scores = []

        def _on_step(self):
            for info in self.locals.get("infos", []):
                if "score" in info:
                    self.scores.append(float(info["score"]))
            return True

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.train_seeds.split(",")]
    recorder = Recorder()
    t0 = time.monotonic()
    env = SubprocVecEnv([
        (lambda seed=seed: Monitor(V08Env(seed, args.per, args.dense, args.residual)))
        for seed in seeds
    ], start_method="fork")
    model = PPO(
        "MlpPolicy", env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[128, 128], activation_fn=torch.nn.Tanh),
        seed=args.ppo_seed,
        verbose=1,
        device="cpu",
    )
    bc_samples = 0
    if args.residual and args.zero_residual:
        # Start exactly at the functioning observation-only controller. Small
        # exploration then learns corrections without destroying its phase machine.
        torch.nn.init.zeros_(model.policy.action_net.weight)
        torch.nn.init.zeros_(model.policy.action_net.bias)
        with torch.no_grad():
            model.policy.log_std.fill_(args.residual_log_std)
    if args.bc_demos:
        data = np.load(args.bc_demos)
        X = torch.as_tensor(data["X"], dtype=torch.float32)
        Y = torch.as_tensor(data["Y"], dtype=torch.float32)
        if args.residual:
            # Demonstration actions are residual corrections to the public
            # scripted baseline, not privileged actions at evaluation.
            Y = Y / args.residual_scale
        opt = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
        for epoch in range(args.bc_epochs):
            for start in range(0, len(X), args.batch_size):
                values, log_probs, entropy = model.policy.evaluate_actions(
                    X[start:start + args.batch_size], Y[start:start + args.batch_size]
                )
                loss = -log_probs.mean() + 0.001 * (values ** 2).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
                opt.step()
            if epoch % 5 == 0 or epoch == args.bc_epochs - 1:
                print(f"bc epoch={epoch} loss={float(loss):.6f}", flush=True)
        bc_samples = len(X)
    model.learn(total_timesteps=args.timesteps, callback=recorder, progress_bar=False)
    train_wall_s = time.monotonic() - t0
    model.save(str(out / "model.zip"))
    env.close()

    _, eval_conds = sample_compound_v08(args.eval_seed, args.eval_per, "live")
    scores, rows = [], []
    eval_t0 = time.monotonic()
    for i in range(min(args.eval_episodes, len(eval_conds))):
        ss = episode_seed_sequence(args.eval_seed, i)
        r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
        ep = AssemblyIVEpisode(eval_conds[i], r1)
        base = None
        if args.residual:
            from obs_scripted_v08 import ObsScriptedPolicy
            base = ObsScriptedPolicy(int(ss.generate_state(1)[0]))
        while not ep.done:
            obs = ep.observe(r2)
            action, _ = model.predict(obs, deterministic=True)
            if base is not None:
                action = np.clip(base.act(obs) + args.residual_scale * action, -1, 1)
            ep.step(action)
        info = compound_score_v08(ep)
        scores.append(float(info["score"]))
        rows.append({
            "episode": i,
            "score": float(info["score"]),
            "n_steps": info["n_steps"],
            "n_done": info["n_done"],
            "terminal_reason": ep.terminal_reason,
            "partials": {
                name: sum(bool(r["partial"][name]) for r in step_results_v08(ep))
                for name in ("lift", "align", "stable")
            },
        })
        print(f"eval episode={i} score={scores[-1]:.6f}", flush=True)
    arr = np.asarray(scores)
    rng = np.random.default_rng(args.ci_seed)
    boot = np.mean(rng.choice(arr, (args.bootstrap, len(arr)), replace=True), axis=1) if len(arr) else np.zeros(1)
    result = {
        "profile": "v08",
        "algorithm": "residual PPO around observation-only scripted controller" if args.residual else "stable-baselines3 PPO",
        "bc_demos": args.bc_demos,
        "bc_samples": bc_samples,
        "bc_epochs": args.bc_epochs,
        "residual_scale": args.residual_scale,
        "train_seeds": seeds,
        "train_conditions_per_seed": len(seeds) * 8 * args.per,
        "eval_seed": args.eval_seed,
        "eval_episodes": len(scores),
        "timesteps": args.timesteps,
        "dense": args.dense,
        "mean": float(np.mean(arr)) if len(arr) else 0.0,
        "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "ci_method": f"episode bootstrap, {args.bootstrap} resamples, seed {args.ci_seed}",
        "train_wall_s": train_wall_s,
        "eval_wall_s": time.monotonic() - eval_t0,
        "train_last20_mean": float(np.mean(recorder.scores[-20:])) if recorder.scores else None,
        "training_episodes": len(recorder.scores),
        "scores": scores,
        "episodes": rows,
        "versions": {"torch": torch.__version__},
        "args": vars(args),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("scores", "episodes")}, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=250000)
    ap.add_argument("--train-seeds", default="1,2,3")
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--eval-seed", type=int, default=7)
    ap.add_argument("--eval-per", type=int, default=8)
    ap.add_argument("--eval-episodes", type=int, default=32)
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--residual", action="store_true")
    ap.add_argument("--residual-scale", type=float, default=0.35)
    ap.add_argument("--zero-residual", action="store_true")
    ap.add_argument("--residual-log-std", type=float, default=-2.0)
    ap.add_argument("--bc-demos", default="")
    ap.add_argument("--bc-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--ppo-seed", type=int, default=0)
    ap.add_argument("--ci-seed", type=int, default=20261117)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    train_and_eval(args)


if __name__ == "__main__":
    main()
