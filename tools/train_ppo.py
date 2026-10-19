"""Small CPU PPO (BC-then-RL) — learnability evidence for v0.2.

1. Distill NearestFirstBaseline into a small MLP (BC warm-start).
2. Curriculum: easy grasped/near-hole + lower residual bias → full live mix.
3. PPO with KL pull toward the BC policy so early updates do not wipe the clone.
4. Report learning curve on an easy held eval and a live eval, vs ObsOnlyScripted
   and NearestFirstBaseline (with per-stratum breakdown).

    PYTHONPATH=referee python tools/train_ppo.py --steps 30000
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))

from nist_sim.conditions import EpisodeConditions, PegStart, sample_episodes  # noqa: E402
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

DEVICE = torch.device("cpu")


def _clearance_bin(c: float) -> str:
    mm = c * 1e3
    if mm < 0.40:
        return "tight"
    if mm < 0.60:
        return "mid"
    return "loose"


def _start_mix(c: EpisodeConditions) -> str:
    n_lying = sum(1 for p in c.pegs if p.start_type == "lying")
    n_up = sum(1 for p in c.pegs if p.start_type in ("loose", "near_hole", "grasped"))
    if n_lying == len(c.pegs):
        return "all_lying"
    if n_lying > 0:
        return "mixed_lying"
    if n_up == len(c.pegs):
        return "all_upright"
    return "has_tray"


def _ease_condition(c: EpisodeConditions, rng: np.random.Generator, stage: int) -> EpisodeConditions:
    """stage 0 = grasped/near-hole + low residual; 1 = milder; 2 = live."""
    if stage >= 2:
        return c
    pegs = list(c.pegs)
    if stage == 0:
        for i, p in enumerate(pegs):
            if i == 0:
                pegs[i] = PegStart(p.slot, "grasped", (0.0, 0.0), 0.02, 0.0, p.shape_override)
            else:
                ang = 2 * math.pi * i / max(1, len(pegs))
                pegs[i] = PegStart(
                    p.slot,
                    "near_hole",
                    (0.012 * math.cos(ang), 0.012 * math.sin(ang)),
                    0.0003,
                    0.0,
                    p.shape_override,
                )
        delta_scale, noise_scale, drift = 0.30, 0.35, 0.0
        clutter: list = []
        clearance = max(c.clearance, 0.0005)
        chamfer = max(c.chamfer, 0.7)
        act_delay, disturbance = 0, 0
    else:
        for i, p in enumerate(pegs):
            if p.start_type == "lying" and rng.random() < 0.65:
                pegs[i] = PegStart(p.slot, "near_hole", (0.012, 0.0), 0.0003, 0.0, p.shape_override)
        delta_scale, noise_scale, drift = 0.55, 0.65, 0.4 * c.bias_drift
        clutter = c.clutter
        clearance, chamfer = c.clearance, c.chamfer
        act_delay, disturbance = c.act_delay, 0
    dx, dy = c.peg_bias_delta_xy
    return EpisodeConditions(
        n_pegs=c.n_pegs,
        clutter_bucket=c.clutter_bucket if clutter else "low",
        clearance=clearance,
        friction=c.friction,
        peg_mass=c.peg_mass,
        gain_scale=c.gain_scale,
        obs_noise=c.obs_noise * noise_scale,
        obs_noise_rad=c.obs_noise_rad * noise_scale,
        dropout_prob=c.dropout_prob * (0.25 if stage == 0 else 0.6),
        act_delay=act_delay,
        disturbance=disturbance,
        board_xy=c.board_xy,
        board_yaw=c.board_yaw,
        pose_bias_xy=c.pose_bias_xy,
        pose_bias_yaw=c.pose_bias_yaw,
        peg_bias_delta_xy=(dx * delta_scale, dy * delta_scale),
        bias_drift=drift,
        backlash=c.backlash * (0.25 if stage == 0 else 0.7),
        chamfer=chamfer,
        search_pitch=c.search_pitch,
        pegs=pegs,
        clutter=clutter,
        disturbance_step=0,
        disturbance_target="none",
        max_steps=c.max_steps,
    )


def _shaped_reward(
    obs: np.ndarray,
    prev_obs: np.ndarray | None,
    seated_now: list[bool],
    prev_seated: list[bool] | None,
) -> float:
    r = 0.0
    # Big bonus when a peg newly seats (partial-credit gradient).
    if prev_seated is not None:
        for a, b in zip(prev_seated, seated_now):
            if b and not a:
                r += 5.0
    r += 0.05 * float(sum(seated_now))
    best = 1e9
    for i in range(MAX_PEGS):
        f = obs[PEG_TABLE0 + i * PEG_FEAT : PEG_TABLE0 + (i + 1) * PEG_FEAT]
        if f[FEAT_PRESENT] < 0.5 or f[FEAT_SEATED] > 0.5:
            continue
        d = float(np.linalg.norm(f[FEAT_REL][0:2] / 10.0))
        best = min(best, d)
    if best < 1e8:
        r += 0.5 * math.exp(-best / 0.012)
        r += 0.08 * float(obs[PROP_PAD])
        if prev_obs is not None:
            prev_best = 1e9
            for i in range(MAX_PEGS):
                f = prev_obs[PEG_TABLE0 + i * PEG_FEAT : PEG_TABLE0 + (i + 1) * PEG_FEAT]
                if f[FEAT_PRESENT] < 0.5 or f[FEAT_SEATED] > 0.5:
                    continue
                prev_best = min(prev_best, float(np.linalg.norm(f[FEAT_REL][0:2] / 10.0)))
            if prev_best < 1e8:
                r += 3.0 * (prev_best - best)
    return float(r)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        # Low initial noise so BC warm-start is not immediately destroyed.
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.8))
        self.v = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.body(x)
        mu = self.mu(h)
        return mu, self.log_std.expand_as(mu), self.v(h).squeeze(-1)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            mu, log_std, _ = self.forward(x)
            if deterministic:
                a = mu
            else:
                a = Normal(mu, log_std.exp().clamp(min=1e-3)).sample()
            a = a.squeeze(0).clamp(-1, 1)
        return a.cpu().numpy().astype(np.float64)


def _distill_baseline() -> ActorCritic:
    xs, ys = [], []
    rng = np.random.default_rng(0)
    for seed in range(4):
        _, conds = sample_episodes(seed, 2)
        for i, c0 in enumerate(conds):
            # Mix live + easy demos.
            c = c0 if i % 2 else _ease_condition(c0, rng, 0)
            ss = np.random.SeedSequence([seed, 0xBC01, i])
            r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
            e = Episode(c, r1)
            pol = NearestFirstBaseline()
            steps = 0
            while not e.done and steps < 180:
                o = e.observe(r2)
                a = np.clip(pol.act(o), -1, 1)
                xs.append(o)
                ys.append(a)
                e.step(a)
                steps += 1
    X = torch.as_tensor(np.asarray(xs), dtype=torch.float32)
    Y = torch.as_tensor(np.asarray(ys), dtype=torch.float32)
    net = ActorCritic().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(120):
        idx = torch.randint(0, X.shape[0], (min(512, X.shape[0]),))
        mu, _, _ = net.forward(X[idx])
        loss = ((mu - Y[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


@dataclass
class Rollout:
    obs: list
    acts: list
    logp: list
    rews: list
    vals: list
    dones: list
    bc_mu: list


def _collect(net: ActorCritic, bc: ActorCritic, n_steps: int, stage: int, seed: int):
    roll = Rollout([], [], [], [], [], [], [])
    rng = np.random.default_rng(seed)
    ep_scores: list[float] = []
    steps = 0
    ep_i = 0
    while steps < n_steps:
        _, conds = sample_episodes(int(rng.integers(0, 50_000)), 1)
        c = _ease_condition(conds[ep_i % len(conds)], rng, stage)
        ss = np.random.SeedSequence([int(seed), 0x50504F, int(ep_i), int(stage)])
        r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
        e = Episode(c, r1)
        prev = None
        prev_seated = [False] * e.n_pegs
        while not e.done and steps < n_steps:
            o = e.observe(r2).astype(np.float64)
            x = torch.as_tensor(o, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                mu, log_std, v = net.forward(x)
                bc_mu, _, _ = bc.forward(x)
                dist = Normal(mu, log_std.exp().clamp(min=1e-3))
                a_t = dist.sample()
                logp = dist.log_prob(a_t).sum(-1)
                a = a_t.squeeze(0).clamp(-1, 1).cpu().numpy()
                val = float(v.item())
            e.step(a)
            next_obs = e.observe(r2) if not e.done else o
            r = _shaped_reward(next_obs, prev, e.seated_now, prev_seated)
            if e.done:
                seated = e.peg_seated_at_end()
                first = e.peg_first_seat_steps()
                msa = e.action_l2 / max(1, e.steps)
                sc = episode_score(seated, first, c.max_steps, msa)
                r += 10.0 * sc
                ep_scores.append(sc)
            roll.obs.append(o)
            roll.acts.append(a)
            roll.logp.append(float(logp.item()))
            roll.rews.append(r)
            roll.vals.append(val)
            roll.dones.append(float(e.done))
            roll.bc_mu.append(bc_mu.squeeze(0).cpu().numpy())
            prev = o
            prev_seated = list(e.seated_now)
            steps += 1
        ep_i += 1
    return roll, float(np.mean(ep_scores)) if ep_scores else 0.0


def _gae(roll: Rollout, gamma: float = 0.995, lam: float = 0.95):
    rews = np.asarray(roll.rews, np.float64)
    vals = np.asarray(roll.vals, np.float64)
    dones = np.asarray(roll.dones, np.float64)
    adv = np.zeros_like(rews)
    last = 0.0
    for t in reversed(range(len(rews))):
        next_v = 0.0 if t == len(rews) - 1 or dones[t] else vals[t + 1]
        delta = rews[t] + gamma * next_v * (1.0 - dones[t]) - vals[t]
        last = delta + gamma * lam * (1.0 - dones[t]) * last
        adv[t] = last
    ret = adv + vals
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret


def _ppo_update(
    net: ActorCritic,
    opt: torch.optim.Optimizer,
    roll: Rollout,
    bc_coef: float,
    epochs: int = 3,
):
    adv, ret = _gae(roll)
    obs = torch.as_tensor(np.asarray(roll.obs), dtype=torch.float32)
    acts = torch.as_tensor(np.asarray(roll.acts), dtype=torch.float32)
    old_logp = torch.as_tensor(np.asarray(roll.logp), dtype=torch.float32)
    bc_mu = torch.as_tensor(np.asarray(roll.bc_mu), dtype=torch.float32)
    adv_t = torch.as_tensor(adv, dtype=torch.float32)
    ret_t = torch.as_tensor(ret, dtype=torch.float32)
    n = obs.shape[0]
    for _ in range(epochs):
        idx = torch.randperm(n)
        for start in range(0, n, 256):
            b = idx[start : start + 256]
            mu, log_std, v = net.forward(obs[b])
            dist = Normal(mu, log_std.exp().clamp(min=1e-3))
            logp = dist.log_prob(acts[b]).sum(-1)
            ratio = (logp - old_logp[b]).exp()
            surr1 = ratio * adv_t[b]
            surr2 = ratio.clamp(0.85, 1.15) * adv_t[b]
            pg = -torch.min(surr1, surr2).mean()
            vf = ((v - ret_t[b]) ** 2).mean()
            ent = dist.entropy().sum(-1).mean()
            bc_loss = ((mu - bc_mu[b]) ** 2).mean()
            loss = pg + 0.5 * vf - 0.005 * ent + bc_coef * bc_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()


def _eval_policy(act_fn, seeds: list[int], per: int = 1, easy_stage: int | None = None) -> dict:
    scores = []
    by_k: dict[str, list] = defaultdict(list)
    by_clear: dict[str, list] = defaultdict(list)
    by_clutter: dict[str, list] = defaultdict(list)
    by_start: dict[str, list] = defaultdict(list)
    rng = np.random.default_rng(0)
    for seed in seeds:
        _, conds = sample_episodes(seed, per)
        for i, c0 in enumerate(conds):
            c = _ease_condition(c0, rng, easy_stage) if easy_stage is not None else c0
            ss = np.random.SeedSequence([seed, 0x5245, i, 0 if easy_stage is None else easy_stage + 1])
            r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
            e = Episode(c, r1)
            state = {"pol": None}
            while not e.done:
                o = e.observe(r2)
                a = np.clip(act_fn(o, state), -1, 1)
                e.step(a)
            seated = e.peg_seated_at_end()
            first = e.peg_first_seat_steps()
            msa = e.action_l2 / max(1, e.steps)
            sc = episode_score(seated, first, c.max_steps, msa)
            scores.append(sc)
            by_k[str(c.n_pegs)].append(sc)
            by_clear[_clearance_bin(c.clearance)].append(sc)
            by_clutter[c.clutter_bucket].append(sc)
            by_start[_start_mix(c)].append(sc)
    return {
        "mean": float(np.mean(scores)),
        "sd": float(np.std(scores)),
        "n": len(scores),
        "by_k": {k: float(np.mean(v)) for k, v in sorted(by_k.items())},
        "by_clear": {k: float(np.mean(v)) for k, v in sorted(by_clear.items())},
        "by_clutter": {k: float(np.mean(v)) for k, v in sorted(by_clutter.items())},
        "by_start": {k: float(np.mean(v)) for k, v in sorted(by_start.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--rollout", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print("distilling baseline BC warm-start…", flush=True)
    net = _distill_baseline()
    bc = ActorCritic().to(DEVICE)
    bc.load_state_dict({k: v.clone() for k, v in net.state_dict().items()})
    for p in bc.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)

    def ppo_act(o, state):
        return net.act(o, deterministic=True)

    def obs_act(o, state):
        if state["pol"] is None:
            state["pol"] = ObsOnlyScripted()
        return state["pol"].act(o)

    def base_act(o, state):
        if state["pol"] is None:
            state["pol"] = NearestFirstBaseline()
        return state["pol"].act(o)

    eval_seeds = [0, 1]
    print("eval warm / scripted on easy + live…", flush=True)
    easy_warm = _eval_policy(ppo_act, eval_seeds, per=1, easy_stage=0)
    live_warm = _eval_policy(ppo_act, eval_seeds, per=1, easy_stage=None)
    easy_obs = _eval_policy(obs_act, eval_seeds, per=1, easy_stage=0)
    live_obs = _eval_policy(obs_act, eval_seeds, per=1, easy_stage=None)
    easy_base = _eval_policy(base_act, eval_seeds, per=1, easy_stage=0)
    live_base = _eval_policy(base_act, eval_seeds, per=1, easy_stage=None)
    print(
        f"START easy: ppo={easy_warm['mean']:.4f} obs={easy_obs['mean']:.4f} base={easy_base['mean']:.4f}",
        flush=True,
    )
    print(
        f"START live: ppo={live_warm['mean']:.4f} obs={live_obs['mean']:.4f} base={live_base['mean']:.4f}",
        flush=True,
    )

    curve = [
        {
            "step": 0,
            "easy": easy_warm["mean"],
            "live": live_warm["mean"],
            "train_ep": easy_warm["mean"],
            "stage": 0,
        }
    ]
    t0 = time.time()
    total = 0
    while total < a.steps:
        stage = 0 if total < a.steps // 2 else (1 if total < 3 * a.steps // 4 else 2)
        bc_coef = 0.5 if stage == 0 else (0.2 if stage == 1 else 0.05)
        roll, train_ep = _collect(net, bc, a.rollout, stage, a.seed + total)
        _ppo_update(net, opt, roll, bc_coef=bc_coef)
        total += a.rollout
        if total % a.eval_every < a.rollout:
            easy_ev = _eval_policy(ppo_act, eval_seeds, per=1, easy_stage=0)
            live_ev = _eval_policy(ppo_act, eval_seeds, per=1, easy_stage=None)
            curve.append(
                {
                    "step": total,
                    "easy": easy_ev["mean"],
                    "live": live_ev["mean"],
                    "train_ep": train_ep,
                    "stage": stage,
                }
            )
            print(
                f"step={total} stage={stage} easy={easy_ev['mean']:.4f} "
                f"live={live_ev['mean']:.4f} train_ep={train_ep:.4f} "
                f"wall={time.time()-t0:.0f}s",
                flush=True,
            )

    print("\n=== FINAL live (4 seeds × 16 eps) ===", flush=True)
    final = _eval_policy(ppo_act, list(range(4)), per=2)
    obs_final = _eval_policy(obs_act, list(range(4)), per=2)
    base_final = _eval_policy(base_act, list(range(4)), per=2)
    easy_final = _eval_policy(ppo_act, list(range(4)), per=1, easy_stage=0)
    for name, ev in (("ppo_live", final), ("obs_scripted", obs_final), ("baseline", base_final)):
        print(
            f"{name}: mean={ev['mean']:.4f} sd={ev['sd']:.4f} "
            f"by_k={ev['by_k']} by_clear={ev['by_clear']} "
            f"by_clutter={ev['by_clutter']} by_start={ev['by_start']}",
            flush=True,
        )
    print(f"ppo_easy_final={easy_final['mean']:.4f}", flush=True)
    print(
        f"delta ppo-obs={final['mean']-obs_final['mean']:+.4f} "
        f"ppo-base={final['mean']-base_final['mean']:+.4f}",
        flush=True,
    )
    print(
        "curve_easy:",
        [(c["step"], round(c["easy"], 4)) for c in curve],
        flush=True,
    )
    print(
        "curve_live:",
        [(c["step"], round(c["live"], 4)) for c in curve],
        flush=True,
    )

    out = ROOT / "out" / "ppo_curve.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        steps=np.array([c["step"] for c in curve]),
        easy=np.array([c["easy"] for c in curve]),
        live=np.array([c["live"] for c in curve]),
        final_ppo=final["mean"],
        final_obs=obs_final["mean"],
        final_base=base_final["mean"],
        final_ppo_easy=easy_final["mean"],
    )
    torch.save(net.state_dict(), ROOT / "out" / "ppo_policy.pt")
    print("wrote", out, flush=True)
    print(f"total wall={time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
