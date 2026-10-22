"""Residual PPO learnability harness for v0.2 tiered live mix.

Policy is a **zero-init residual** on top of ``ObsOnlyScripted``:

    a = clip(scripted(obs) + scale * residual_net(obs), -1, 1)

So warm-start live score equals the obs-only scripted gate. Privileged-solver
demos supervise the residual only when they nearly agree with scripted
(obs-learnable corrections miners can collect offline). PPO then improves the
residual on the tiered live mix with easy/medium oversampling.

    PYTHONPATH=referee:tools python tools/train_ppo.py --steps 100000 --workers 2
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tools"))

from nist_sim.conditions import sample_episodes  # noqa: E402
from nist_sim.env import ACT_DIM, OBS_DIM, Episode  # noqa: E402
from nist_sim.policies import (  # noqa: E402
    FEAT_PRESENT,
    FEAT_REL,
    FEAT_SEATED,
    MAX_PEGS,
    NearestFirstBaseline,
    ObsOnlyScripted,
    PEG_FEAT,
    PEG_TABLE0,
    PROP_PAD,
)
from nist_sim.scoring import episode_score  # noqa: E402
from reference_solver import solve_episode  # noqa: E402

DEVICE = torch.device("cpu")
RESIDUAL_SCALE = 0.30
INIT_LOG_STD = -2.5
PRIV_SHAPING = False
STD_MAX = 0.2


def _insert_potential(e) -> float:
    """Privileged training-time shaping (simulator state; the policy never sees it):
    seated pegs count 1, unseated pegs count by how far their tip is down inside their own
    hole, weighted by how well centred they are."""
    from nist_sim.env import BLOCK_H, PEG_LEN  # noqa: PLC0415

    phi = 0.0
    for i in range(e.n_pegs):
        if e.seated_now[i]:
            phi += 1.0
            continue
        p, _ = e.peg_pose(i)
        hx, hy = e.slot_xy[i]
        xy = float(np.hypot(p[0] - hx, p[1] - hy))
        bottom = float(p[2] - PEG_LEN / 2)
        depth = float(np.clip((BLOCK_H + 0.004 - bottom) / (BLOCK_H + 0.004), 0.0, 1.0))
        phi += 0.5 * math.exp(-xy / 0.003) * (0.3 + 0.7 * depth) if bottom < BLOCK_H + 0.02 else 0.0
    return phi


def _clearance_bin(c: float) -> str:
    mm = c * 1e3
    if mm < 0.40:
        return "tight"
    if mm < 0.60:
        return "mid"
    return "loose"


def _start_mix(c) -> str:
    n_lying = sum(1 for p in c.pegs if p.start_type == "lying")
    n_up = sum(1 for p in c.pegs if p.start_type in ("loose", "near_hole", "grasped"))
    if n_lying == len(c.pegs):
        return "all_lying"
    if n_lying > 0:
        return "mixed_lying"
    if n_up == len(c.pegs):
        return "all_upright"
    return "has_tray"


def _shaped_reward(obs, prev_obs, seated_now, prev_seated) -> float:
    r = 0.0
    if prev_seated is not None:
        for a, b in zip(prev_seated, seated_now):
            if b and not a:
                r += 10.0
    r += 0.12 * float(sum(seated_now))
    best = 1e9
    for i in range(MAX_PEGS):
        f = obs[PEG_TABLE0 + i * PEG_FEAT : PEG_TABLE0 + (i + 1) * PEG_FEAT]
        if f[FEAT_PRESENT] < 0.5 or f[FEAT_SEATED] > 0.5:
            continue
        best = min(best, float(np.linalg.norm(f[FEAT_REL][0:2] / 10.0)))
    if best < 1e8:
        r += 1.0 * math.exp(-best / 0.012)
        r += 0.1 * float(obs[PROP_PAD])
        if prev_obs is not None:
            prev_best = 1e9
            for i in range(MAX_PEGS):
                f = prev_obs[PEG_TABLE0 + i * PEG_FEAT : PEG_TABLE0 + (i + 1) * PEG_FEAT]
                if f[FEAT_PRESENT] < 0.5 or f[FEAT_SEATED] > 0.5:
                    continue
                prev_best = min(prev_best, float(np.linalg.norm(f[FEAT_REL][0:2] / 10.0)))
            if prev_best < 1e8:
                r += 6.0 * (prev_best - best)
    return float(r)


def _combine(base: np.ndarray, resid: np.ndarray) -> np.ndarray:
    return np.clip(base + RESIDUAL_SCALE * resid, -1.0, 1.0)


class ResidualNet(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(INIT_LOG_STD)))
        self.v = nn.Linear(hidden, 1)
        # Near-zero residual at init → warm score ≈ scripted.
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)

    def forward(self, x: torch.Tensor):
        h = self.body(x)
        mu = torch.tanh(self.mu(h))
        return mu, self.log_std.expand_as(mu), self.v(h).squeeze(-1)

    def residual(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            mu, log_std, _ = self.forward(x)
            if deterministic:
                a = mu
            else:
                a = Normal(mu, log_std.exp().clamp(1e-3, STD_MAX)).sample()
            a = a.squeeze(0).clamp(-1, 1)
        return a.cpu().numpy().astype(np.float64)

    def state_dict_cpu(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}


def _collect_residual_demos(n_max: int = 12_000) -> tuple[np.ndarray, np.ndarray]:
    """Small privileged residual demos where privileged ≈ scripted (learnable)."""
    xs, ys = [], []
    for seed in range(10):
        _, conds = sample_episodes(seed, 2)
        for i, c in enumerate(conds):
            if c.difficulty_tier == "hard":
                continue
            if len(xs) >= n_max:
                break
            ss = np.random.SeedSequence([seed, 0xBC40, i])
            r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
            e = Episode(c, r1)
            rec: list = []
            solve_episode(e, record=rec, noise_rng=r2)
            pol = ObsOnlyScripted()
            for o, priv_a in rec:
                base = np.clip(pol.act(o), -1, 1)
                delta = priv_a - base
                if float(np.linalg.norm(delta)) > 0.35:
                    continue  # not obs-learnable
                resid = np.clip(delta / max(RESIDUAL_SCALE, 1e-6), -1, 1)
                xs.append(o)
                ys.append(resid)
        if len(xs) >= n_max:
            break
    if not xs:
        return np.zeros((1, OBS_DIM), np.float32), np.zeros((1, ACT_DIM), np.float32)
    X = np.asarray(xs, np.float32)
    Y = np.asarray(ys, np.float32)
    print(f"residual demos: {X.shape[0]} (mean|r|={float(np.mean(np.abs(Y))):.3f})", flush=True)
    return X, Y


def _light_distill(net: ResidualNet, X: np.ndarray, Y: np.ndarray, epochs: int = 80) -> None:
    if X.shape[0] < 32:
        print("skip distill (too few demos)", flush=True)
        return
    opt = torch.optim.Adam(net.parameters(), lr=8e-4)
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    bs = min(512, Xt.shape[0])
    for ep in range(epochs):
        idx = torch.randint(0, Xt.shape[0], (bs,))
        mu, _, _ = net.forward(Xt[idx])
        # Stay close to zero: only learn small corrections.
        loss = ((mu - Yt[idx]) ** 2).mean() + 0.05 * (mu**2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 20 == 0:
            print(f"  distill ep {ep} mse={float(loss.detach()):.5f}", flush=True)


def _rollout_worker(args):
    """Spawn worker — late imports required for ProcessPoolExecutor safety."""
    weights, n_steps, seed, stage_bias, root, base_kind = args
    sys.path.insert(0, root)
    sys.path.insert(0, str(Path(root) / "referee"))
    sys.path.insert(0, str(Path(root) / "tools"))
    import torch as _torch  # noqa: PLC0415
    from nist_sim.conditions import sample_episodes as _sample  # noqa: PLC0415
    from nist_sim.env import Episode as _Episode  # noqa: PLC0415
    from nist_sim.policies import ObsOnlyScripted as _Scripted  # noqa: PLC0415
    from nist_sim.scoring import episode_score as _score  # noqa: PLC0415
    from train_ppo import (  # noqa: PLC0415
        ResidualNet as _Net,
        _combine as _comb,
        _shaped_reward as _rew,
    )

    _torch.set_num_threads(1)
    import train_ppo as _tp  # noqa: PLC0415

    _tp.PRIV_SHAPING = os.environ.get("NIST_PPO_PRIV_SHAPING") == "1"
    _tp.STD_MAX = float(os.environ.get("NIST_PPO_STD_MAX", _tp.STD_MAX))
    net = _Net()
    net.load_state_dict(weights)
    net.eval()
    rng = np.random.default_rng(seed)
    obs_l, act_l, logp_l, rew_l, val_l, done_l = [], [], [], [], [], []
    ep_scores = []
    steps = 0
    ep_i = 0
    while steps < n_steps:
        _, conds = _sample(int(rng.integers(0, 100_000)), 2)
        roll = rng.random()
        if roll < stage_bias:
            pool = [c for c in conds if c.difficulty_tier == "easy"] or [
                c for c in conds if c.difficulty_tier == "medium"
            ] or conds
        elif roll < stage_bias + 0.25:
            pool = [c for c in conds if c.difficulty_tier == "medium"] or conds
        else:
            pool = conds
        c = pool[ep_i % len(pool)]
        ss = np.random.SeedSequence([int(seed), 0x5050, int(ep_i)])
        r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
        e = _Episode(c, r1)
        if base_kind == "refport":
            steps = _refport_rollout(net, e, c, r2, n_steps, steps, obs_l, act_l, logp_l, rew_l, val_l, done_l, ep_scores)
            ep_i += 1
            continue
        scripted = _Scripted()
        prev = None
        prev_seated = [False] * e.n_pegs
        while not e.done and steps < n_steps:
            o = e.observe(r2).astype(np.float64)
            base = np.clip(scripted.act(o), -1, 1)
            x = _torch.as_tensor(o, dtype=_torch.float32).unsqueeze(0)
            with _torch.no_grad():
                mu, log_std, v = net.forward(x)
                dist = _torch.distributions.Normal(mu, log_std.exp().clamp(1e-3, STD_MAX))
                resid_t = dist.sample()
                logp = dist.log_prob(resid_t).sum(-1)
                resid = resid_t.squeeze(0).clamp(-1, 1).numpy()
                val = float(v.item())
            a = _comb(base, resid)
            e.step(a)
            nxt = e.observe(r2) if not e.done else o
            r = _rew(nxt, prev, e.seated_now, prev_seated)
            # Soft penalty for large residuals (keep scripted intact).
            r -= 0.02 * float(np.sum(resid**2))
            if e.done:
                sc = _score(
                    e.peg_seated_at_end(),
                    e.peg_first_seat_steps(),
                    c.max_steps,
                    e.action_l2 / max(1, e.steps),
                )
                r += 18.0 * sc
                ep_scores.append(sc)
            obs_l.append(o)
            act_l.append(resid.astype(np.float32))
            logp_l.append(float(logp.item()))
            rew_l.append(r)
            val_l.append(val)
            done_l.append(float(e.done))
            prev = o
            prev_seated = list(e.seated_now)
            steps += 1
        ep_i += 1
    return {
        "obs": np.asarray(obs_l, np.float32),
        "acts": np.asarray(act_l, np.float32),
        "logp": np.asarray(logp_l, np.float32),
        "rews": np.asarray(rew_l, np.float32),
        "vals": np.asarray(val_l, np.float32),
        "dones": np.asarray(done_l, np.float32),
        "ep_mean": float(np.mean(ep_scores)) if ep_scores else 0.0,
        "n_eps": len(ep_scores),
    }


class _StopRollout(Exception):
    pass


def _refport_rollout(net, e, c, r2, n_steps, steps, obs_l, act_l, logp_l, rew_l, val_l, done_l, ep_scores):
    """One episode of residual PPO on top of the obs-only refport controller."""
    from obs_refport import solve_obs_episode  # noqa: PLC0415
    from nist_sim.scoring import episode_score as _score  # noqa: PLC0415

    st = {"steps": steps, "prev": None, "prev_seated": [False] * e.n_pegs, "o": None, "phi": _insert_potential(e)}

    def act_hook(o, base):
        if st["steps"] >= n_steps:
            raise _StopRollout
        x = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mu, log_std, v = net.forward(x)
            dist = Normal(mu, log_std.exp().clamp(1e-3, STD_MAX))
            resid_t = dist.sample()
            logp = dist.log_prob(resid_t).sum(-1)
        resid = resid_t.squeeze(0).clamp(-1, 1).numpy()
        obs_l.append(np.asarray(o, np.float64))
        act_l.append(resid.astype(np.float32))
        logp_l.append(float(logp.item()))
        val_l.append(float(v.item()))
        st["o"] = o
        st["resid"] = resid
        return _combine(base, resid)

    def post_hook(nxt, done):
        if PRIV_SHAPING:
            phi = _insert_potential(e)
            r = 6.0 * (phi - st["phi"]) + 8.0 * sum(1 for a, b in zip(st["prev_seated"], e.seated_now) if b and not a)
            st["phi"] = phi
        else:
            r = _shaped_reward(nxt, st["prev"], e.seated_now, st["prev_seated"])
        r -= 0.02 * float(np.sum(st["resid"] ** 2))
        if done:
            sc = _score(e.peg_seated_at_end(), e.peg_first_seat_steps(), c.max_steps, e.action_l2 / max(1, e.steps))
            r += 18.0 * sc
            ep_scores.append(sc)
        rew_l.append(r)
        done_l.append(float(done))
        st["prev"] = st["o"]
        st["prev_seated"] = list(e.seated_now)
        st["steps"] += 1

    try:
        solve_obs_episode(e, r2, act_hook=act_hook, post_hook=post_hook)
    except _StopRollout:
        pass
    # Drop a dangling transition without a reward (stop inside a step).
    n = min(len(obs_l), len(rew_l))
    del obs_l[n:], act_l[n:], logp_l[n:], val_l[n:]
    if done_l and len(done_l) == n and st["steps"] >= n_steps:
        done_l[-1] = 1.0
    return st["steps"]


def _parallel_collect(net: ResidualNet, n_steps: int, workers: int, seed: int, stage_bias: float, base_kind: str = "obs"):
    weights = net.state_dict_cpu()
    per = max(256, n_steps // max(1, workers))
    root = str(ROOT)
    args = [(weights, per, seed + 17 * w, stage_bias, root, base_kind) for w in range(workers)]
    if workers <= 1:
        parts = [_rollout_worker(args[0])]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_rollout_worker, args))
    return {
        "obs": np.concatenate([p["obs"] for p in parts]),
        "acts": np.concatenate([p["acts"] for p in parts]),
        "logp": np.concatenate([p["logp"] for p in parts]),
        "rews": np.concatenate([p["rews"] for p in parts]),
        "vals": np.concatenate([p["vals"] for p in parts]),
        "dones": np.concatenate([p["dones"] for p in parts]),
        "ep_mean": float(np.mean([p["ep_mean"] for p in parts if p["n_eps"] > 0]) or 0.0),
    }


def _gae(rews, vals, dones, gamma=0.995, lam=0.95):
    adv = np.zeros_like(rews)
    last = 0.0
    for t in reversed(range(len(rews))):
        next_v = 0.0 if t == len(rews) - 1 or dones[t] else vals[t + 1]
        delta = rews[t] + gamma * next_v * (1.0 - dones[t]) - vals[t]
        last = delta + gamma * lam * (1.0 - dones[t]) * last
        adv[t] = last
    ret = adv + vals
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv.astype(np.float32), ret.astype(np.float32)


def _ppo_update(net, opt, batch, epochs: int = 4, l2_coef: float = 0.02):
    adv, ret = _gae(batch["rews"], batch["vals"], batch["dones"])
    obs = torch.as_tensor(batch["obs"])
    acts = torch.as_tensor(batch["acts"])
    old_logp = torch.as_tensor(batch["logp"])
    adv_t = torch.as_tensor(adv)
    ret_t = torch.as_tensor(ret)
    n = obs.shape[0]
    for _ in range(epochs):
        idx = torch.randperm(n)
        for s in range(0, n, 512):
            b = idx[s : s + 512]
            mu, log_std, v = net.forward(obs[b])
            dist = Normal(mu, log_std.exp().clamp(1e-3, STD_MAX))
            logp = dist.log_prob(acts[b]).sum(-1)
            ratio = (logp - old_logp[b]).exp()
            surr1 = ratio * adv_t[b]
            surr2 = ratio.clamp(0.88, 1.12) * adv_t[b]
            pg = -torch.min(surr1, surr2).mean()
            vf = ((v - ret_t[b]) ** 2).mean()
            ent = dist.entropy().sum(-1).mean()
            l2 = (mu**2).mean()
            loss = pg + 0.5 * vf - 0.008 * ent + l2_coef * l2
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()


def _eval_policy(act_fn, seeds, per=2) -> dict:
    scores = []
    by_k, by_clear, by_clutter, by_start, by_tier = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    for seed in seeds:
        _, conds = sample_episodes(seed, per)
        for i, c in enumerate(conds):
            ss = np.random.SeedSequence([seed, 0x5245, i])
            r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
            e = Episode(c, r1)
            state = {"pol": None}
            while not e.done:
                o = e.observe(r2)
                a = np.clip(act_fn(o, state), -1, 1)
                e.step(a)
            sc = episode_score(
                e.peg_seated_at_end(),
                e.peg_first_seat_steps(),
                c.max_steps,
                e.action_l2 / max(1, e.steps),
            )
            scores.append(sc)
            by_k[str(c.n_pegs)].append(sc)
            by_clear[_clearance_bin(c.clearance)].append(sc)
            by_clutter[c.clutter_bucket].append(sc)
            by_start[_start_mix(c)].append(sc)
            by_tier[c.difficulty_tier].append(sc)
    return {
        "mean": float(np.mean(scores)),
        "sd": float(np.std(scores)),
        "n": len(scores),
        "by_k": {k: float(np.mean(v)) for k, v in sorted(by_k.items())},
        "by_clear": {k: float(np.mean(v)) for k, v in sorted(by_clear.items())},
        "by_clutter": {k: float(np.mean(v)) for k, v in sorted(by_clutter.items())},
        "by_start": {k: float(np.mean(v)) for k, v in sorted(by_start.items())},
        "by_tier": {k: float(np.mean(v)) for k, v in sorted(by_tier.items())},
    }


def _eval_worker(args):
    weights, base_kind, use_resid, seed, per, i, root = args
    sys.path[:0] = [str(Path(root) / "referee"), str(Path(root) / "tools")]
    from nist_sim.conditions import sample_episodes as _sample  # noqa: PLC0415
    from nist_sim.env import Episode as _Episode  # noqa: PLC0415
    from nist_sim.policies import ObsOnlyScripted as _Scripted  # noqa: PLC0415
    from nist_sim.scoring import episode_score as _score  # noqa: PLC0415

    torch.set_num_threads(1)
    net = ResidualNet()
    net.load_state_dict(weights)
    net.eval()
    _, conds = _sample(seed, per)
    c = conds[i]
    ss = np.random.SeedSequence([seed, 0x5245, i])
    r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
    e = _Episode(c, r1)
    if base_kind == "refport":
        from obs_refport import solve_obs_episode  # noqa: PLC0415

        hook = (lambda o, b: _combine(b, net.residual(o, deterministic=True))) if use_resid else None
        return solve_obs_episode(e, r2, act_hook=hook)["score"], c.difficulty_tier
    pol = _Scripted()
    while not e.done:
        o = e.observe(r2).astype(np.float64)
        b = np.clip(pol.act(o), -1, 1)
        e.step(_combine(b, net.residual(o, deterministic=True)) if use_resid else b)
    return _score(e.peg_seated_at_end(), e.peg_first_seat_steps(), c.max_steps, e.action_l2 / max(1, e.steps)), c.difficulty_tier


def eval_parallel(net, base_kind: str, use_resid: bool, seeds, per: int, workers: int) -> dict:
    w = net.state_dict_cpu()
    jobs = [(w, base_kind, use_resid, s, per, i, str(ROOT)) for s in seeds for i in range(8 * per)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(_eval_worker, jobs))
    sc = [r[0] for r in res]
    tiers = defaultdict(list)
    for v, t in res:
        tiers[t].append(v)
    return {"mean": float(np.mean(sc)), "scores": sc, "n": len(sc), "by_tier": {k: round(float(np.mean(v)), 4) for k, v in sorted(tiers.items())}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--rollout", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=8_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", choices=("obs", "refport"), default="obs",
                    help="scripted base controller under the residual")
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--eval-per", type=int, default=2)
    ap.add_argument("--final-seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    ap.add_argument("--final-per", type=int, default=4)
    ap.add_argument("--demos", type=int, default=12_000, help="0 = no privileged residual demos")
    ap.add_argument("--out", default=str(ROOT / "out"))
    ap.add_argument("--parallel-eval", action="store_true", help="paired parallel eval (default for refport)")
    ap.add_argument("--priv-shaping", action="store_true", help="privileged insertion-potential reward (training only)")
    ap.add_argument("--init-log-std", type=float, default=-2.5)
    ap.add_argument("--std-max", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    global INIT_LOG_STD, STD_MAX, PRIV_SHAPING
    INIT_LOG_STD, STD_MAX, PRIV_SHAPING = a.init_log_std, a.std_max, a.priv_shaping
    os.environ["NIST_PPO_PRIV_SHAPING"] = "1" if a.priv_shaping else "0"
    os.environ["NIST_PPO_STD_MAX"] = str(a.std_max)
    if a.base == "refport" or a.parallel_eval:
        return _main_parallel(a)

    net = ResidualNet().to(DEVICE)
    print("collecting small privileged residual demos…", flush=True)
    X, Y = _collect_residual_demos(a.demos)
    print("light residual distill…", flush=True)
    _light_distill(net, X, Y)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)

    def ppo_act(o, state):
        if state["pol"] is None:
            state["pol"] = ObsOnlyScripted()
        base = np.clip(state["pol"].act(o), -1, 1)
        return _combine(base, net.residual(o, deterministic=True))

    def obs_act(o, state):
        if state["pol"] is None:
            state["pol"] = ObsOnlyScripted()
        return state["pol"].act(o)

    def base_act(o, state):
        if state["pol"] is None:
            state["pol"] = NearestFirstBaseline()
        return state["pol"].act(o)

    eval_seeds = [0, 1, 2]
    print("live eval (warm / scripted)…", flush=True)
    warm = _eval_policy(ppo_act, eval_seeds, per=2)
    obs_ev = _eval_policy(obs_act, eval_seeds, per=2)
    base_ev = _eval_policy(base_act, eval_seeds, per=2)
    print(
        f"START live ppo={warm['mean']:.4f} obs={obs_ev['mean']:.4f} "
        f"base={base_ev['mean']:.4f} ppo_tiers={warm['by_tier']}",
        flush=True,
    )

    curve = [{"step": 0, "live": warm["mean"], "train_ep": warm["mean"]}]
    t0 = time.time()
    total = 0
    best_live = warm["mean"]
    best_state = net.state_dict_cpu()

    while total < a.steps:
        frac = total / max(1, a.steps)
        stage_bias = max(0.25, 0.85 - 0.5 * frac)
        l2_coef = 0.03 * (1.0 - 0.5 * frac)
        batch = _parallel_collect(net, a.rollout, a.workers, a.seed + total, stage_bias)
        _ppo_update(net, opt, batch, l2_coef=l2_coef)
        total += batch["obs"].shape[0]
        if total // a.eval_every > (total - batch["obs"].shape[0]) // a.eval_every:
            live = _eval_policy(ppo_act, eval_seeds, per=2)
            curve.append({"step": total, "live": live["mean"], "train_ep": batch["ep_mean"]})
            if live["mean"] > best_live:
                best_live = live["mean"]
                best_state = net.state_dict_cpu()
            print(
                f"step={total} live={live['mean']:.4f} tiers={live['by_tier']} "
                f"train_ep={batch['ep_mean']:.4f} best={best_live:.4f} "
                f"wall={time.time()-t0:.0f}s",
                flush=True,
            )

    net.load_state_dict(best_state)
    print("\n=== FINAL live (5 seeds × 16 eps) ===", flush=True)
    final = _eval_policy(ppo_act, list(range(5)), per=2)
    obs_f = _eval_policy(obs_act, list(range(5)), per=2)
    base_f = _eval_policy(base_act, list(range(5)), per=2)
    for name, ev in (("ppo", final), ("obs_scripted", obs_f), ("baseline", base_f)):
        print(
            f"{name}: mean={ev['mean']:.4f} sd={ev['sd']:.4f} "
            f"by_tier={ev['by_tier']} by_k={ev['by_k']} by_start={ev['by_start']}",
            flush=True,
        )
    print(
        f"delta ppo-obs={final['mean']-obs_f['mean']:+.4f} "
        f"ppo-base={final['mean']-base_f['mean']:+.4f}",
        flush=True,
    )
    print("curve_live:", [(c["step"], round(c["live"], 4)) for c in curve], flush=True)

    out = ROOT / "out" / "ppo_curve.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        steps=np.array([c["step"] for c in curve]),
        live=np.array([c["live"] for c in curve]),
        final_ppo=final["mean"],
        final_obs=obs_f["mean"],
        final_base=base_f["mean"],
    )
    torch.save(best_state, ROOT / "out" / "ppo_policy.pt")
    print("wrote", out, flush=True)
    print(f"total wall={time.time()-t0:.0f}s", flush=True)
    return 0


def _main_parallel(a) -> int:
    """Residual PPO on the refport base with parallel paired evaluation."""
    torch.manual_seed(a.seed)
    net = ResidualNet().to(DEVICE)
    if a.base == "obs" and a.demos > 0:
        X, Y = _collect_residual_demos(a.demos)
        _light_distill(net, X, Y)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    t0 = time.time()
    warm = eval_parallel(net, a.base, True, a.eval_seeds, a.eval_per, a.workers)
    print(f"START live={warm['mean']:.4f} tiers={warm['by_tier']}", flush=True)
    curve = [{"step": 0, "live": warm["mean"]}]
    best_live, best_state = warm["mean"], net.state_dict_cpu()
    total = 0
    while total < a.steps:
        frac = total / max(1, a.steps)
        stage_bias = max(0.25, 0.85 - 0.5 * frac)
        batch = _parallel_collect(net, a.rollout, a.workers, a.seed + total, stage_bias, a.base)
        _ppo_update(net, opt, batch, l2_coef=0.03 * (1.0 - 0.5 * frac))
        total += batch["obs"].shape[0]
        if total // a.eval_every > (total - batch["obs"].shape[0]) // a.eval_every:
            live = eval_parallel(net, a.base, True, a.eval_seeds, a.eval_per, a.workers)
            curve.append({"step": total, "live": live["mean"]})
            if live["mean"] > best_live:
                best_live, best_state = live["mean"], net.state_dict_cpu()
            print(f"step={total} live={live['mean']:.4f} tiers={live['by_tier']} train_ep={batch['ep_mean']:.4f} "
                  f"best={best_live:.4f} wall={time.time()-t0:.0f}s", flush=True)
    net.load_state_dict(best_state)
    fin = eval_parallel(net, a.base, True, a.final_seeds, a.final_per, a.workers)
    ref = eval_parallel(net, a.base, False, a.final_seeds, a.final_per, a.workers)
    d = np.array(fin["scores"]) - np.array(ref["scores"])
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    print(f"FINAL ppo={fin['mean']:.4f} {a.base}={ref['mean']:.4f} delta={d.mean():+.4f} (paired se {se:.4f}, n={len(d)}) "
          f"ppo_tiers={fin['by_tier']} base_tiers={ref['by_tier']}", flush=True)
    print("curve_live:", [(c["step"], round(c["live"], 4)) for c in curve], flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"ppo_{a.base}_curve.npz", steps=np.array([c["step"] for c in curve]),
             live=np.array([c["live"] for c in curve]), final_ppo=fin["mean"], final_base=ref["mean"])
    torch.save(best_state, out / f"ppo_{a.base}_policy.pt")
    print(f"total wall={time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    raise SystemExit(main())
