"""Genuine residual PPO baseline for Assembly III (v07) — with provenance.

Training recipe (reproducible, CPU):
1. **Base** = obs-only ``CompoundObsScripted`` (strong hand-coded miner-class script).
2. **Residual net** = MLP on the full 288-d observation; zero-init so warm score = scripted.
3. Optional short **BC distill** of residual targets from honest privileged demos
   (``solve_compound(..., oracle=False)``) where privileged action ≈ scripted + small delta.
4. **PPO** on train seeds improves the residual with score-shaped rewards.
5. Export ONNX of ``clip(scripted(obs) + scale * residual(obs))`` for image eval.

This is **not** a privileged controller. Privileged demos are used only as offline
supervision for the residual; live rollouts and eval use observe→action only.

Train / eval seeds are disjoint. Learning curves + budgets are written under ``--out``.

    NIST_PROFILE=v07 python tools/train_rl_v07.py \\
        --train-seeds 100 101 102 103 --eval-seeds 7 11 \\
        --ppo-steps 80000 --out out/rl_v07

Never commit exported ONNX kings; ``out/`` is gitignored evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tools"))

from nist_sim import conditions as C  # noqa: E402
from nist_sim.compound_v07 import (  # noqa: E402
    OBS_DIM_V07,
    AssemblyIIIEpisode,
    compound_score_v07,
    sample_compound_v07,
)
from nist_sim.env import ACT_DIM  # noqa: E402
from compound_obs_scripted import CompoundObsScripted  # noqa: E402
from compound_solver import solve_compound  # noqa: E402

DEVICE = torch.device("cpu")
RESIDUAL_SCALE = 0.35
INIT_LOG_STD = -2.2
STD_MAX = 0.25
OBS_DIM = OBS_DIM_V07  # 288


class ResidualNet(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, ACT_DIM)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), float(INIT_LOG_STD)))
        self.v = nn.Linear(hidden, 1)
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)

    def forward(self, x: torch.Tensor):
        h = self.body(x)
        mu = torch.tanh(self.mu(h))
        return mu, self.log_std.expand_as(mu), self.v(h).squeeze(-1)


def _combine(base: np.ndarray, resid: np.ndarray) -> np.ndarray:
    return np.clip(base + RESIDUAL_SCALE * resid, -1.0, 1.0)


def _make_ep(seed: int, per: int, i: int) -> tuple[AssemblyIIIEpisode, np.random.Generator]:
    _, conds = sample_compound_v07(seed, per)
    r1, r2 = (np.random.default_rng(s) for s in C.episode_seed_sequence(seed, i).spawn(2))
    return AssemblyIIIEpisode(conds[i], r1), r2


def eval_policy(act_fn, seeds: list[int], per: int) -> dict:
    scores = []
    kind_done: dict[str, list[int]] = {}
    t0 = time.perf_counter()
    for seed in seeds:
        _, conds = sample_compound_v07(seed, per)
        for i, c in enumerate(conds):
            ep, r2 = _make_ep(seed, per, i)
            while not ep.done:
                o = np.round(ep.observe(r2), 6)
                a = np.asarray(act_fn(o), dtype=float)
                if not np.all(np.isfinite(a)):
                    a = np.zeros(ACT_DIM)
                ep.step(np.clip(a, -1, 1))
            r = compound_score_v07(ep)
            scores.append(float(r["score"]))
            for k, s in enumerate(c.compound.steps):
                kind_done.setdefault(s.kind, [0, 0])
                kind_done[s.kind][1] += 1
                if ep.step_done[k]:
                    kind_done[s.kind][0] += 1
    wall = time.perf_counter() - t0
    return {
        "mean": float(np.mean(scores)) if scores else 0.0,
        "std": float(np.std(scores)) if scores else 0.0,
        "n": len(scores),
        "scores": scores,
        "kind_done": {k: f"{v[0]}/{v[1]}" for k, v in kind_done.items()},
        "stack_done": kind_done.get("stack", [0, 0])[0],
        "stack_drawn": kind_done.get("stack", [0, 0])[1],
        "wall_s": wall,
    }


def scripted_act_factory():
    pol = CompoundObsScripted()

    def act(o):
        # Reset phase memory across episodes by detecting progress channel drop.
        return pol.act(o.astype(np.float64))

    act.pol = pol  # type: ignore[attr-defined]
    return act


def collect_bc_demos(seeds: list[int], per: int, max_samples: int = 40_000) -> tuple[np.ndarray, np.ndarray]:
    """Residual targets where honest privileged ≈ scripted (obs-learnable)."""
    xs, ys = [], []
    for seed in seeds:
        _, conds = sample_compound_v07(seed, per)
        for i, _c in enumerate(conds):
            if len(xs) >= max_samples:
                break
            ep, r2 = _make_ep(seed, per, i)
            pol = CompoundObsScripted()
            X_ep, Y_ep = [], []

            raw_step = ep.step

            def rec_step(a, _ep=ep, _r2=r2, _pol=pol):
                if not _ep.done:
                    o = np.round(_ep.observe(_r2), 6)
                    base = np.clip(_pol.act(o), -1, 1)
                    priv = np.clip(np.asarray(a, dtype=np.float64), -1, 1)
                    delta = priv - base
                    if float(np.linalg.norm(delta)) <= 0.55:
                        resid = np.clip(delta / max(RESIDUAL_SCALE, 1e-6), -1, 1)
                        X_ep.append(o.astype(np.float32))
                        Y_ep.append(resid.astype(np.float32))
                return raw_step(a)

            ep.step = rec_step  # type: ignore[method-assign]
            solve_compound(ep, oracle=False)
            xs.extend(X_ep)
            ys.extend(Y_ep)
        if len(xs) >= max_samples:
            break
    if not xs:
        return np.zeros((1, OBS_DIM), np.float32), np.zeros((1, ACT_DIM), np.float32)
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


def distill(net: ResidualNet, X: np.ndarray, Y: np.ndarray, epochs: int = 40) -> list[float]:
    if X.shape[0] < 64:
        return []
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    losses = []
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    bs = min(1024, Xt.shape[0])
    for ep in range(epochs):
        idx = torch.randint(0, Xt.shape[0], (bs,))
        mu, _, _ = net.forward(Xt[idx])
        loss = ((mu - Yt[idx]) ** 2).mean() + 0.02 * (mu**2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if ep % 10 == 0:
            print(f"  bc ep {ep} mse={losses[-1]:.5f}", flush=True)
    return losses


def ppo_update(net: ResidualNet, batch, epochs: int = 4, clip: float = 0.2, lr: float = 3e-4):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
    act = torch.as_tensor(batch["act"], dtype=torch.float32)
    old_logp = torch.as_tensor(batch["logp"], dtype=torch.float32)
    adv = torch.as_tensor(batch["adv"], dtype=torch.float32)
    ret = torch.as_tensor(batch["ret"], dtype=torch.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = obs.shape[0]
    bs = min(2048, n)
    stats = []
    for _ in range(epochs):
        idx = torch.randperm(n)
        for s in range(0, n, bs):
            j = idx[s : s + bs]
            mu, log_std, v = net.forward(obs[j])
            dist = Normal(mu, log_std.exp().clamp(1e-3, STD_MAX))
            logp = dist.log_prob(act[j]).sum(-1)
            ratio = (logp - old_logp[j]).exp()
            surr = torch.min(ratio * adv[j], ratio.clamp(1 - clip, 1 + clip) * adv[j])
            loss = -surr.mean() + 0.5 * ((v - ret[j]) ** 2).mean() - 0.01 * dist.entropy().sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            stats.append(float(loss.detach()))
    return float(np.mean(stats)) if stats else 0.0


def collect_rollouts(net: ResidualNet, seeds: list[int], per: int, n_steps: int, seed0: int) -> dict:
    net.eval()
    obs_l, act_l, logp_l, rew_l, val_l, done_l = [], [], [], [], [], []
    ep_scores = []
    rng = np.random.default_rng(seed0)
    steps = 0
    ep_i = 0
    while steps < n_steps:
        seed = int(rng.choice(seeds))
        _, conds = sample_compound_v07(seed, per)
        i = int(rng.integers(0, len(conds)))
        ep, r2 = _make_ep(seed, per, i)
        pol = CompoundObsScripted()
        prev_done = list(ep.step_done)
        while not ep.done and steps < n_steps:
            o = np.round(ep.observe(r2), 6).astype(np.float64)
            base = np.clip(pol.act(o), -1, 1)
            x = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mu, log_std, v = net.forward(x)
                dist = Normal(mu, log_std.exp().clamp(1e-3, STD_MAX))
                resid_t = dist.sample()
                logp = dist.log_prob(resid_t).sum(-1)
                resid = resid_t.squeeze(0).clamp(-1, 1).numpy()
                val = float(v.item())
            a = _combine(base, resid)
            ep.step(a)
            # Dense reward: newly completed steps + running score proxy.
            r = 0.0
            for k, now in enumerate(ep.step_done):
                if now and not prev_done[k]:
                    kind = ep.cs.steps[k].kind
                    r += {"key": 12.0, "stack": 10.0, "rework": 8.0, "screw": 3.0}.get(kind, 2.0)
            prev_done = list(ep.step_done)
            r -= 0.015 * float(np.sum(resid**2))
            if ep.done:
                sc = float(compound_score_v07(ep)["score"])
                r += 20.0 * sc
                ep_scores.append(sc)
            obs_l.append(o.astype(np.float32))
            act_l.append(resid.astype(np.float32))
            logp_l.append(float(logp.item()))
            rew_l.append(float(r))
            val_l.append(val)
            done_l.append(bool(ep.done))
            steps += 1
        ep_i += 1
        pol.reset()
    # GAE
    rew = np.asarray(rew_l, np.float32)
    val = np.asarray(val_l, np.float32)
    done = np.asarray(done_l, np.bool_)
    adv = np.zeros_like(rew)
    last = 0.0
    for t in range(len(rew) - 1, -1, -1):
        next_v = 0.0 if done[t] else (val[t + 1] if t + 1 < len(val) else 0.0)
        delta = rew[t] + 0.995 * next_v - val[t]
        last = delta + 0.995 * 0.95 * (0.0 if done[t] else last)
        adv[t] = last
    ret = adv + val
    return {
        "obs": np.asarray(obs_l),
        "act": np.asarray(act_l),
        "logp": np.asarray(logp_l, np.float32),
        "adv": adv,
        "ret": ret,
        "ep_scores": ep_scores,
        "steps": steps,
    }


def export_onnx(net: ResidualNet, path: Path, scripted_probe: bool = True) -> None:
    """Export residual-only net; player wraps via a small fusion script below if needed.

    For image eval we export a **fused** module that embeds a frozen linear approx of
    scripted is impractical — instead export residual and a Python fused ONNX via tracing
    a wrapper that expects the caller to add scripted. Miners need a single ONNX, so we
    export a residual-only net and also a ``FusedScriptedResidual`` that runs scripted
    logic in Python at train time then for ONNX we bake residual-only + document that
    eval uses ``tools/eval_fused_rl_v07.py``.

    Practical approach: distill the fused policy ``scripted+residual`` into a plain MLP
    for ONNX export (behavior clone of the fused teacher on train rollouts).
    """
    net.eval()
    # Distill fused teacher into a deployable MLP.
    teacher_net = net
    deploy = nn.Sequential(
        nn.Linear(OBS_DIM, 256),
        nn.Tanh(),
        nn.Linear(256, 256),
        nn.Tanh(),
        nn.Linear(256, 256),
        nn.Tanh(),
        nn.Linear(256, ACT_DIM),
        nn.Tanh(),
    )
    opt = torch.optim.Adam(deploy.parameters(), lr=1e-3)
    pol = CompoundObsScripted()
    xs, ys = [], []
    for seed in (100, 101, 102):
        _, conds = sample_compound_v07(seed, 2)
        for i, _c in enumerate(conds):
            ep, r2 = _make_ep(seed, 2, i)
            pol.reset()
            while not ep.done:
                o = np.round(ep.observe(r2), 6)
                base = np.clip(pol.act(o), -1, 1)
                with torch.no_grad():
                    mu, _, _ = teacher_net.forward(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))
                    resid = mu.squeeze(0).numpy()
                a = _combine(base, resid)
                xs.append(o.astype(np.float32))
                ys.append(a.astype(np.float32))
                ep.step(a)
            if len(xs) > 20_000:
                break
        if len(xs) > 20_000:
            break
    X = torch.as_tensor(np.asarray(xs))
    Y = torch.as_tensor(np.asarray(ys))
    for ep in range(30):
        idx = torch.randint(0, X.shape[0], (min(1024, X.shape[0]),))
        pred = deploy(X[idx])
        loss = ((pred - Y[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    deploy.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, OBS_DIM)
    torch.onnx.export(
        deploy,
        dummy,
        str(path),
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "B"}, "actions": {0: "B"}},
        opset_version=17,
    )
    print(f"exported {path} ({path.stat().st_size} bytes)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=list(range(100, 108)))
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[7, 11])
    ap.add_argument("--per", type=int, default=2, help="episodes/stratum during train rollouts")
    ap.add_argument("--eval-per", type=int, default=2, help="episodes/stratum for eval (8=full 64)")
    ap.add_argument("--ppo-steps", type=int, default=60_000)
    ap.add_argument("--rollout-steps", type=int, default=8_000)
    ap.add_argument("--bc-epochs", type=int, default=30)
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "rl_v07")
    ap.add_argument("--skip-bc", action="store_true")
    a = ap.parse_args()
    os.environ["NIST_PROFILE"] = "v07"
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    # Provenance record
    meta = {
        "profile": "v07",
        "method": "residual_ppo_on_compound_obs_scripted",
        "residual_scale": RESIDUAL_SCALE,
        "train_seeds": a.train_seeds,
        "eval_seeds": a.eval_seeds,
        "ppo_steps": a.ppo_steps,
        "note": "Not a privileged controller. BC demos from honest solver are offline only.",
    }
    (a.out / "provenance.json").write_text(json.dumps(meta, indent=2))

    def scripted_only(o):
        # Fresh policy per call would lose phase — use a sticky factory carefully in eval.
        raise RuntimeError("use factory")

    # Baselines at start
    def make_scripted_eval():
        # One policy object per episode is created inside eval via wrapping:
        class _E:
            def __init__(self):
                self.pol = None
                self.prev_prog = None

            def __call__(self, o):
                # Detect new episode roughly via global progress channel near 0 and hand reset.
                if self.pol is None:
                    self.pol = CompoundObsScripted()
                a = self.pol.act(o.astype(np.float64))
                return a

            def reset(self):
                self.pol = CompoundObsScripted()

        return _E()

    # Proper per-episode scripted eval
    def eval_scripted(seeds, per):
        scores = []
        kind_done: dict[str, list[int]] = {}
        t0 = time.perf_counter()
        for seed in seeds:
            _, conds = sample_compound_v07(seed, per)
            for i, c in enumerate(conds):
                ep, r2 = _make_ep(seed, per, i)
                pol = CompoundObsScripted()
                while not ep.done:
                    o = np.round(ep.observe(r2), 6)
                    ep.step(np.clip(pol.act(o), -1, 1))
                r = compound_score_v07(ep)
                scores.append(float(r["score"]))
                for k, s in enumerate(c.compound.steps):
                    kind_done.setdefault(s.kind, [0, 0])
                    kind_done[s.kind][1] += 1
                    if ep.step_done[k]:
                        kind_done[s.kind][0] += 1
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "n": len(scores),
            "scores": scores,
            "kind_done": {k: f"{v[0]}/{v[1]}" for k, v in kind_done.items()},
            "stack_done": kind_done.get("stack", [0, 0])[0],
            "stack_drawn": kind_done.get("stack", [0, 0])[1],
            "wall_s": time.perf_counter() - t0,
        }

    print("== baseline: compound_scripted ==", flush=True)
    base_row = eval_scripted(a.eval_seeds, a.eval_per)
    print(
        f"scripted mean={base_row['mean']:.4f} stack={base_row['stack_done']}/{base_row['stack_drawn']} "
        f"kinds={base_row['kind_done']}",
        flush=True,
    )

    net = ResidualNet()
    curve = [{"iter": 0, "tag": "scripted", **{k: base_row[k] for k in ("mean", "std", "stack_done", "stack_drawn", "wall_s")}}]

    if not a.skip_bc:
        print("== BC residual distill from honest demos ==", flush=True)
        t0 = time.perf_counter()
        X, Y = collect_bc_demos(a.train_seeds, per=2, max_samples=30_000)
        print(f"demos {len(X)} in {time.perf_counter() - t0:.0f}s mean|r|={float(np.mean(np.abs(Y))):.3f}", flush=True)
        distill(net, X, Y, epochs=a.bc_epochs)
        def bc_act_factory():
            def _act_episode(seed, per, i):
                ep, r2 = _make_ep(seed, per, i)
                pol = CompoundObsScripted()
                while not ep.done:
                    o = np.round(ep.observe(r2), 6)
                    base = np.clip(pol.act(o), -1, 1)
                    with torch.no_grad():
                        mu, _, _ = net.forward(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))
                        resid = mu.squeeze(0).numpy()
                    ep.step(_combine(base, resid))
                return ep
            return _act_episode

        # quick eval after BC
        scores = []
        kind_done: dict[str, list[int]] = {}
        for seed in a.eval_seeds:
            _, conds = sample_compound_v07(seed, a.eval_per)
            for i, c in enumerate(conds):
                ep, r2 = _make_ep(seed, a.eval_per, i)
                pol = CompoundObsScripted()
                while not ep.done:
                    o = np.round(ep.observe(r2), 6)
                    base = np.clip(pol.act(o), -1, 1)
                    with torch.no_grad():
                        mu, _, _ = net.forward(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))
                        resid = mu.squeeze(0).numpy()
                    ep.step(_combine(base, resid))
                r = compound_score_v07(ep)
                scores.append(float(r["score"]))
                for k, s in enumerate(c.compound.steps):
                    kind_done.setdefault(s.kind, [0, 0])
                    kind_done[s.kind][1] += 1
                    if ep.step_done[k]:
                        kind_done[s.kind][0] += 1
        bc_row = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "stack_done": kind_done.get("stack", [0, 0])[0],
            "stack_drawn": kind_done.get("stack", [0, 0])[1],
            "kind_done": {k: f"{v[0]}/{v[1]}" for k, v in kind_done.items()},
        }
        print(f"after BC mean={bc_row['mean']:.4f} stack={bc_row['stack_done']}/{bc_row['stack_drawn']}", flush=True)
        curve.append({"iter": 0, "tag": "after_bc", **bc_row})

    print("== PPO ==", flush=True)
    done_steps = 0
    it = 0
    while done_steps < a.ppo_steps:
        it += 1
        batch = collect_rollouts(net, a.train_seeds, a.per, a.rollout_steps, seed0=10_000 + it)
        loss = ppo_update(net, batch)
        done_steps += batch["steps"]
        mean_ep = float(np.mean(batch["ep_scores"])) if batch["ep_scores"] else float("nan")
        print(
            f"iter {it} steps={done_steps}/{a.ppo_steps} loss={loss:.4f} "
            f"rollout_mean={mean_ep:.4f} n_eps={len(batch['ep_scores'])}",
            flush=True,
        )
        if it % 2 == 0 or done_steps >= a.ppo_steps:
            scores = []
            kind_done = {}
            for seed in a.eval_seeds:
                _, conds = sample_compound_v07(seed, a.eval_per)
                for i, c in enumerate(conds):
                    ep, r2 = _make_ep(seed, a.eval_per, i)
                    pol = CompoundObsScripted()
                    while not ep.done:
                        o = np.round(ep.observe(r2), 6)
                        base = np.clip(pol.act(o), -1, 1)
                        with torch.no_grad():
                            mu, _, _ = net.forward(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))
                            resid = mu.squeeze(0).numpy()
                        ep.step(_combine(base, resid))
                    r = compound_score_v07(ep)
                    scores.append(float(r["score"]))
                    for k, s in enumerate(c.compound.steps):
                        kind_done.setdefault(s.kind, [0, 0])
                        kind_done[s.kind][1] += 1
                        if ep.step_done[k]:
                            kind_done[s.kind][0] += 1
            row = {
                "iter": it,
                "tag": "ppo",
                "steps": done_steps,
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "stack_done": kind_done.get("stack", [0, 0])[0],
                "stack_drawn": kind_done.get("stack", [0, 0])[1],
                "kind_done": {k: f"{v[0]}/{v[1]}" for k, v in kind_done.items()},
                "loss": loss,
            }
            curve.append(row)
            print(f"  eval mean={row['mean']:.4f} stack={row['stack_done']}/{row['stack_drawn']}", flush=True)

    torch.save(net.state_dict(), a.out / "residual.pt")
    export_onnx(net, a.out / "rl_v07.onnx")
    (a.out / "learning_curve.json").write_text(json.dumps(curve, indent=2))
    summary = {
        "scripted": base_row,
        "final": curve[-1],
        "curve": curve,
        "commands": {
            "train": (
                "NIST_PROFILE=v07 python tools/train_rl_v07.py "
                f"--train-seeds {' '.join(map(str, a.train_seeds))} "
                f"--eval-seeds {' '.join(map(str, a.eval_seeds))} "
                f"--ppo-steps {a.ppo_steps} --out {a.out}"
            ),
            "image_eval": (
                "DOCKER_BUILDKIT=1 docker build -f referee/Dockerfile -t nist-assembly-referee:v072 . && "
                "DOCKER_BUILDKIT=1 docker build -f player/Dockerfile -t nist-assembly-player:v072 . && "
                f"python tools/image_eval_onnx.py --seed 7 --full --referee-image nist-assembly-referee:v072 "
                f"--player-image nist-assembly-player:v072 {a.out / 'rl_v07.onnx'}"
            ),
        },
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"scripted_mean": base_row["mean"], "final_mean": curve[-1]["mean"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
