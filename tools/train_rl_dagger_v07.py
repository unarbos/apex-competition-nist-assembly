"""Recurrent BC → DAgger → PPO for Assembly III (v07) — miner-path sealed.

Goal: a learned policy that beats strong scripted baselines (~0.20 kings / v06-style
proxy) on held-out unassisted miner-path evals (observe → action only).

Recipe
------
1. **Teacher demos**: honest privileged ``solve_compound(..., oracle=False)`` trajectories
   recorded as (obs, action) sequences — offline only; never shipped as a controller.
2. **GRU policy**: recurrent actor-critic on the 288-d observation (phase/belief memory).
3. **BC pretrain** on teacher sequences (truncated BPTT).
4. **DAgger**: beta-mixture rollouts; every visited state is labelled by a **reactive
   privileged teacher** (true-state one-step actions). Labels never enter the observation.
5. **PPO fine-tune** with dense shaping from **observable** progress deltas (seated /
   task-done / collar-done / twist-progress channels in the obs) plus terminal score.
   No oracle completion writes.

Eval is always miner-path: ``observe → policy → step``. Train/eval seeds are disjoint.

    NIST_PROFILE=v07 python tools/train_rl_dagger_v07.py \\
        --train-seeds 100 101 102 103 104 105 106 107 \\
        --eval-seeds 7 11 --ppo-steps 120000 --out out/rl_dagger_v07

Do not claim RL advantage without provenance in ``--out/summary.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tools"))

from nist_sim import conditions as C  # noqa: E402
from nist_sim.compound import quat_yaw, wrap  # noqa: E402
from nist_sim.compound_v07 import (  # noqa: E402
    OBS_DIM_V07,
    STEP_WEIGHT_V07,
    AssemblyIIIEpisode,
    compound_score_v07,
    sample_compound_v07,
)
from nist_sim.env import ACT_DIM, DXYZ, DYAW, PEG_LEN  # noqa: E402
from compound_obs_scripted import CompoundObsScripted  # noqa: E402
from compound_solver import (  # noqa: E402
    BUDGET_V07,
    collar_routine,
    insert_routine,
    key_routine,
    rework_routine,
    solve_compound,
    twist_routine,
    _available,
    _is_lying_insert,
    _max_attempts_for,
    _priority,
)
from reference_solver import _push_clutter, part_axis  # noqa: E402

DEVICE = torch.device("cpu")
OBS_DIM = OBS_DIM_V07
HID = 256
INIT_LOG_STD = -1.8
STD_MAX = 0.35
GAMMA = 0.995
LAM = 0.95

# Obs layout helpers (must match compound observe).
PEG0 = 21
PEG_FEAT = 17
TASK0 = 184
TASK_FEAT = 6
COL0 = 232
COL_FEAT = 13
MAX_PEGS = 8
MAX_COLLARS = 4


def _init_worker() -> None:
    os.environ["NIST_PROFILE"] = "v07"
    sys.path.insert(0, str(ROOT / "referee"))
    sys.path.insert(0, str(ROOT / "tools"))


def _make_ep(seed: int, per: int, i: int) -> tuple[AssemblyIIIEpisode, np.random.Generator]:
    _, conds = sample_compound_v07(seed, per)
    r1, r2 = (np.random.default_rng(s) for s in C.episode_seed_sequence(seed, i).spawn(2))
    return AssemblyIIIEpisode(conds[i], r1), r2


# ---- stateful privileged teacher (labels only; full routine generators) ---------------------
class StatefulPrivilegedTeacher:
    """Step-wise honest privileged controller for DAgger labels.

    Reuses the same routine generators as ``solve_compound(..., oracle=False)``.
    Internal phase state lives only in this teacher — never in the observation.
    """

    def __init__(self) -> None:
        self.gen = None
        self.attempts: list[int] | None = None
        self.idle = np.array([0.0, 0.0, 0.35, 0.0, 1.0])

    def reset(self) -> None:
        self.gen = None
        self.attempts = None

    def _start_routine(self, ep: AssemblyIIIEpisode):
        if self.attempts is None:
            self.attempts = [0] * len(ep.cs.steps)
        todo = [
            k
            for k in range(len(ep.cs.steps))
            if not ep.step_done[k]
            and _available(ep, k)
            and self.attempts[k] < _max_attempts_for(ep.cs.steps[k].kind)
        ]
        lying = {k for k in todo if _is_lying_insert(ep, k)}
        in_order = [k for k in todo if not ep.step_blocked(k)]

        def phase(k):
            kind = ep.cs.steps[k].kind
            return {"insert": 0, "rework": 0, "key": 1, "stack": 2, "screw": 3, "twist": 4}.get(kind, 5)

        ready = [k for k in in_order if k not in lying]
        if not ready:
            ready = [k for k in todo if k not in lying] or list(todo)
        if not ready:
            self.gen = iter([self.idle.copy() for _ in range(8)])
            return
        k = min(ready, key=lambda kk: (phase(kk), self.attempts[kk], _priority(ep, kk)))
        self.attempts[k] += 1
        s = ep.cs.steps[k]
        if s.kind == "insert":
            if not ep._gripper_touching(ep.peg_body[s.peg]):
                _push_clutter(ep, s.peg)
            self.gen = insert_routine(ep, s.peg)
        elif s.kind == "rework":
            if not ep._gripper_touching(ep.peg_body[s.peg]):
                _push_clutter(ep, s.peg)
            self.gen = rework_routine(ep, s.peg, int(s.wrong_slot))
        elif s.kind == "stack":
            if not (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                self.gen = insert_routine(ep, s.peg)
            else:
                self.gen = collar_routine(ep, s.part, stack=True)
        elif s.kind == "collar":
            if not (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                self.gen = insert_routine(ep, s.peg)
            else:
                self.gen = collar_routine(ep, s.part, stack=False)
        elif s.kind == "key":
            if not (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                self.gen = insert_routine(ep, s.peg)
            else:
                self.gen = key_routine(ep, k, oracle=False)
        elif s.kind == "screw":
            if not (ep.seated_now[s.peg] or ep._peg_in_hole(s.peg)):
                self.gen = insert_routine(ep, s.peg)
            else:
                self.gen = twist_routine(ep, k, press=True, oracle=False)
        elif s.kind == "twist":
            self.gen = twist_routine(ep, k, press=False, oracle=False)
        else:
            self.gen = iter([self.idle.copy() for _ in range(4)])

    def act(self, ep: AssemblyIIIEpisode) -> np.ndarray:
        for _ in range(3):
            if self.gen is None:
                self._start_routine(ep)
            try:
                a = next(self.gen)
                return np.clip(np.asarray(a, dtype=np.float64), -1, 1)
            except StopIteration:
                self.gen = None
        return self.idle.copy()


# ---- observable progress (no privileged completion leak) ------------------------------------
def obs_progress_features(o: np.ndarray) -> np.ndarray:
    """Compact progress vector derived only from the miner observation."""
    o = np.asarray(o, dtype=np.float64).reshape(-1)
    seated = np.array([o[PEG0 + i * PEG_FEAT + 16] for i in range(MAX_PEGS)])
    task_done = np.array([o[TASK0 + i * TASK_FEAT + 0] for i in range(MAX_PEGS)])
    twist_prog = np.array([o[TASK0 + i * TASK_FEAT + 3] for i in range(MAX_PEGS)])
    col_done = np.array([o[COL0 + j * COL_FEAT + 11] for j in range(MAX_COLLARS)])
    glob = o[-4:]
    return np.concatenate([seated, task_done, twist_prog, col_done, glob])


def dense_reward(prev_feat: np.ndarray, feat: np.ndarray, resid_pen: float, terminal_score: float | None) -> float:
    d = feat - prev_feat
    # Observable deltas: seating, task/collar done flags, twist/key progress, global done frac.
    r = (
        2.0 * float(np.sum(np.clip(d[0:8], 0, 1)))  # seated
        + 4.0 * float(np.sum(np.clip(d[8:16], 0, 1)))  # task done channel
        + 1.5 * float(np.sum(np.clip(d[16:24], 0, 1)))  # twist/key progress
        + 6.0 * float(np.sum(np.clip(d[24:28], 0, 1)))  # collar/stack done
        + 3.0 * float(np.clip(d[29], 0, 1))  # global done frac
    )
    r -= 0.01 * resid_pen
    if terminal_score is not None:
        r += 25.0 * float(terminal_score)
    return r


# ---- network --------------------------------------------------------------------------------
class GRUActorCritic(nn.Module):
    def __init__(self, hidden: int = HID):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.Tanh())
        self.gru = nn.GRUCell(hidden, hidden)
        self.mu = nn.Linear(hidden, ACT_DIM)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), float(INIT_LOG_STD)))
        self.v = nn.Linear(hidden, 1)
        self.register_buffer("obs_mu", torch.zeros(OBS_DIM))
        self.register_buffer("obs_sd", torch.ones(OBS_DIM))
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)

    def set_obs_norm(self, mu: np.ndarray, sd: np.ndarray) -> None:
        self.obs_mu.copy_(torch.as_tensor(mu, dtype=torch.float32))
        self.obs_sd.copy_(torch.as_tensor(sd, dtype=torch.float32))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.obs_mu) / self.obs_sd

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None, normalized: bool = False):
        """x: [B, obs] or [T, B, obs]; h: [B, H]. Returns mu, log_std, v, h_n."""
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        if not normalized:
            x = self.normalize(x)
        T, B, _ = x.shape
        if h is None:
            h = torch.zeros(B, self.gru.hidden_size, device=x.device)
        mus, vs = [], []
        for t in range(T):
            e = self.enc(x[t])
            h = self.gru(e, h)
            mus.append(torch.tanh(self.mu(h)))
            vs.append(self.v(h).squeeze(-1))
        mu = torch.stack(mus, 0)
        v = torch.stack(vs, 0)
        log_std = self.log_std.expand_as(mu)
        if squeeze:
            return mu.squeeze(0), log_std.squeeze(0), v.squeeze(0), h
        return mu, log_std, v, h

    def act_np(self, o: np.ndarray, h: torch.Tensor | None, deterministic: bool = True):
        x = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mu, log_std, v, h_n = self.forward(x, h)
            if deterministic:
                a = mu.squeeze(0)
                logp = torch.zeros(())
            else:
                dist = Normal(mu, log_std.exp().clamp(1e-3, STD_MAX))
                a = dist.sample().squeeze(0)
                logp = dist.log_prob(a).sum()
            return (
                a.clamp(-1, 1).numpy().astype(np.float64),
                float(logp),
                float(v.item()),
                h_n,
            )


# ---- data collection ------------------------------------------------------------------------
def _collect_teacher_job(job: tuple) -> tuple:
    seed, per, i = job
    import numpy as np

    from compound_solver import solve_compound
    from nist_sim import conditions as C
    from nist_sim.compound_v07 import AssemblyIIIEpisode, sample_compound_v07

    _, conds = sample_compound_v07(seed, per)
    r1, r2 = (np.random.default_rng(s) for s in C.episode_seed_sequence(seed, i).spawn(2))
    ep = AssemblyIIIEpisode(conds[i], r1)
    xs, ys = [], []
    raw = ep.step

    def rec(a):
        if not ep.done:
            xs.append(np.round(ep.observe(r2), 6).astype(np.float32))
            ys.append(np.clip(np.asarray(a, np.float32), -1, 1))
        return raw(a)

    ep.step = rec  # type: ignore[method-assign]
    r = solve_compound(ep, oracle=False)
    return (
        np.asarray(xs, np.float32),
        np.asarray(ys, np.float32),
        float(r["score"]),
        int(sum(ep.step_done)),
        int(len(ep.cs.steps)),
    )


def collect_teacher_demos(seeds: list[int], per: int, workers: int) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    jobs = [(s, per, i) for s in seeds for i in range(8 * per)]
    seq_x, seq_y = [], []
    scores = []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init_worker) as ex:
        for X, Y, sc, _dn, _nt in ex.map(_collect_teacher_job, jobs, chunksize=1):
            if len(X) > 8:
                seq_x.append(X)
                seq_y.append(Y)
                scores.append(sc)
    return seq_x, seq_y, float(np.mean(scores)) if scores else 0.0


def _obs_stats(seq_x: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    flat = np.concatenate(seq_x, axis=0)
    mu = flat.mean(0).astype(np.float32)
    sd = (flat.std(0) + 1e-3).astype(np.float32)
    return mu, sd


def bc_train(
    net: GRUActorCritic,
    seq_x: list[np.ndarray],
    seq_y: list[np.ndarray],
    epochs: int,
    trunc: int = 64,
    obs_mu: np.ndarray | None = None,
    obs_sd: np.ndarray | None = None,
) -> list[float]:
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    if obs_mu is None or obs_sd is None:
        obs_mu, obs_sd = _obs_stats(seq_x)
    net.set_obs_norm(obs_mu, obs_sd)
    losses = []
    for ep in range(epochs):
        order = np.random.permutation(len(seq_x))
        ep_loss = []
        for si in order:
            X = torch.as_tensor(seq_x[si])
            Y = torch.as_tensor(seq_y[si])
            h = None
            for t0 in range(0, X.shape[0], trunc):
                xb = X[t0 : t0 + trunc].unsqueeze(1)  # [T,1,obs]
                yb = Y[t0 : t0 + trunc]
                mu, _, _, h = net.forward(xb, h)
                h = h.detach()
                loss = ((mu.squeeze(1) - yb) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                ep_loss.append(float(loss.detach()))
        losses.append(float(np.mean(ep_loss)) if ep_loss else 0.0)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  bc epoch {ep} mse={losses[-1]:.5f}", flush=True)
    return losses


def dagger_round(
    net: GRUActorCritic,
    seeds: list[int],
    per: int,
    beta: float,
    n_episodes: int,
    seed0: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Noise-augmented full privileged demos (DAgger-style state coverage).

    Runs honest ``solve_compound`` while injecting action noise with probability
    ``1-beta``. Labels are the clean teacher actions before noise. This keeps the
    strong planner in the loop (unlike a weak reactive teacher) while visiting
    slightly off-demo states the learner must recover from.
    """
    rng = np.random.default_rng(seed0)
    seq_x, seq_y, scores = [], [], []
    for _ in range(n_episodes):
        seed = int(rng.choice(seeds))
        _, conds = sample_compound_v07(seed, per)
        i = int(rng.integers(0, len(conds)))
        ep, r2 = _make_ep(seed, per, i)
        xs, ys = [], []
        raw = ep.step

        def rec(a, _ep=ep, _r2=r2, _rng=rng, _beta=beta):
            if not _ep.done:
                o = np.round(_ep.observe(_r2), 6).astype(np.float32)
                clean = np.clip(np.asarray(a, dtype=np.float32), -1, 1)
                xs.append(o)
                ys.append(clean)
                if _rng.random() > _beta:
                    a = np.clip(clean + _rng.normal(0, 0.20, size=ACT_DIM).astype(np.float32), -1, 1)
            return raw(a)

        ep.step = rec  # type: ignore[method-assign]
        r = solve_compound(ep, oracle=False)
        scores.append(float(r["score"]))
        if len(xs) > 8:
            seq_x.append(np.asarray(xs, np.float32))
            seq_y.append(np.asarray(ys, np.float32))
        # Keep net warm (unused here but signature stable).
        _ = net
    return seq_x, seq_y, scores


def ppo_collect(net: GRUActorCritic, seeds: list[int], per: int, n_steps: int, seed0: int) -> dict:
    net.eval()
    rng = np.random.default_rng(seed0)
    obs_l, act_l, logp_l, rew_l, val_l, done_l, h_reset = [], [], [], [], [], [], []
    ep_scores = []
    steps = 0
    while steps < n_steps:
        seed = int(rng.choice(seeds))
        _, conds = sample_compound_v07(seed, per)
        i = int(rng.integers(0, len(conds)))
        ep, r2 = _make_ep(seed, per, i)
        h = None
        o = np.round(ep.observe(r2), 6).astype(np.float32)
        prev = obs_progress_features(o)
        while not ep.done and steps < n_steps:
            a, logp, val, h = net.act_np(o, h, deterministic=False)
            ep.step(np.clip(a, -1, 1))
            o2 = np.round(ep.observe(r2), 6).astype(np.float32) if not ep.done else o
            feat = obs_progress_features(o2)
            term = float(compound_score_v07(ep)["score"]) if ep.done else None
            if ep.done:
                ep_scores.append(term)
            r = dense_reward(prev, feat, float(np.sum(a**2)), term)
            obs_l.append(o)
            act_l.append(a.astype(np.float32))
            logp_l.append(logp)
            rew_l.append(r)
            val_l.append(val)
            done_l.append(bool(ep.done))
            h_reset.append(False)
            o = o2
            prev = feat
            steps += 1
        if obs_l:
            h_reset[-1] = True  # mark episode boundary for GAE bootstrap
    rew = np.asarray(rew_l, np.float32)
    val = np.asarray(val_l, np.float32)
    done = np.asarray(done_l, np.bool_)
    adv = np.zeros_like(rew)
    last = 0.0
    for t in range(len(rew) - 1, -1, -1):
        next_v = 0.0 if done[t] else (val[t + 1] if t + 1 < len(val) else 0.0)
        delta = rew[t] + GAMMA * next_v - val[t]
        last = delta + GAMMA * LAM * (0.0 if done[t] else last)
        adv[t] = last
    return {
        "obs": np.asarray(obs_l, np.float32),
        "act": np.asarray(act_l, np.float32),
        "logp": np.asarray(logp_l, np.float32),
        "adv": adv,
        "ret": adv + val,
        "ep_scores": ep_scores,
        "steps": steps,
    }


def ppo_update(net: GRUActorCritic, batch: dict, epochs: int = 4, clip: float = 0.2, lr: float = 2.5e-4) -> float:
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    obs = torch.as_tensor(batch["obs"])
    act = torch.as_tensor(batch["act"])
    old_logp = torch.as_tensor(batch["logp"])
    adv = torch.as_tensor(batch["adv"])
    ret = torch.as_tensor(batch["ret"])
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = obs.shape[0]
    # Treat as independent transitions with h=0 (memoryless PPO update on GRU encoder).
    # Full RTRL is expensive; BC/DAgger already put temporal structure in the weights.
    bs = min(2048, n)
    stats = []
    for _ in range(epochs):
        idx = torch.randperm(n)
        for s in range(0, n, bs):
            j = idx[s : s + bs]
            mu, log_std, v, _ = net.forward(obs[j], None)
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


# ---- eval -----------------------------------------------------------------------------------
def eval_policy(net: GRUActorCritic | None, mode: str, seeds: list[int], per: int) -> dict:
    scores = []
    kind_done: dict[str, list[int]] = {}
    t0 = time.perf_counter()
    for seed in seeds:
        _, conds = sample_compound_v07(seed, per)
        for i, c in enumerate(conds):
            ep, r2 = _make_ep(seed, per, i)
            if mode == "scripted":
                pol = CompoundObsScripted()
                while not ep.done:
                    o = np.round(ep.observe(r2), 6)
                    ep.step(np.clip(pol.act(o), -1, 1))
            elif mode == "priv":
                from compound_solver import solve_compound as _sc

                _sc(ep, oracle=False)
            elif mode == "v06_style":
                from v06_style_scripted import solve_v06_style

                solve_v06_style(ep)
            elif mode == "learned":
                assert net is not None
                h = None
                while not ep.done:
                    o = np.round(ep.observe(r2), 6).astype(np.float32)
                    a, _, _, h = net.act_np(o, h, deterministic=True)
                    ep.step(np.clip(a, -1, 1))
            else:
                raise ValueError(mode)
            r = compound_score_v07(ep)
            scores.append(float(r["score"]))
            for k, s in enumerate(c.compound.steps):
                kind_done.setdefault(s.kind, [0, 0])
                kind_done[s.kind][1] += 1
                if ep.step_done[k]:
                    kind_done[s.kind][0] += 1
    return {
        "mean": float(np.mean(scores)) if scores else 0.0,
        "std": float(np.std(scores)) if scores else 0.0,
        "n": len(scores),
        "scores": scores,
        "kind_done": {k: f"{v[0]}/{v[1]}" for k, v in kind_done.items()},
        "stack_done": kind_done.get("stack", [0, 0])[0],
        "stack_drawn": kind_done.get("stack", [0, 0])[1],
        "key_done": kind_done.get("key", [0, 0])[0],
        "key_drawn": kind_done.get("key", [0, 0])[1],
        "rework_done": kind_done.get("rework", [0, 0])[0],
        "rework_drawn": kind_done.get("rework", [0, 0])[1],
        "wall_s": time.perf_counter() - t0,
    }


class DeployGRU(nn.Module):
    """Memoryless distill target OR stateful wrapper for in-process eval export."""

    def __init__(self, net: GRUActorCritic):
        super().__init__()
        self.net = net

    def forward(self, x):
        mu, _, _, _ = self.net.forward(x, None)
        return mu


def export_onnx_memoryless(net: GRUActorCritic, path: Path, seq_x: list[np.ndarray], epochs: int = 40) -> None:
    """Distill recurrent teacher into a memoryless MLP for the live ONNX player."""
    deploy = nn.Sequential(
        nn.Linear(OBS_DIM, 512),
        nn.Tanh(),
        nn.Linear(512, 512),
        nn.Tanh(),
        nn.Linear(512, 256),
        nn.Tanh(),
        nn.Linear(256, ACT_DIM),
        nn.Tanh(),
    )
    opt = torch.optim.Adam(deploy.parameters(), lr=1e-3)
    xs, ys = [], []
    net.eval()
    for X in seq_x[: min(64, len(seq_x))]:
        h = None
        for o in X:
            a, _, _, h = net.act_np(o, h, deterministic=True)
            xs.append(o)
            ys.append(a.astype(np.float32))
            if len(xs) > 40_000:
                break
        if len(xs) > 40_000:
            break
    if len(xs) < 64:
        return
    Xt = torch.as_tensor(np.asarray(xs))
    Yt = torch.as_tensor(np.asarray(ys))
    for ep in range(epochs):
        idx = torch.randint(0, Xt.shape[0], (min(2048, Xt.shape[0]),))
        loss = ((deploy(Xt[idx]) - Yt[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    deploy.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        deploy,
        torch.zeros(1, OBS_DIM),
        str(path),
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "B"}, "actions": {0: "B"}},
        opset_version=17,
    )
    print(f"exported memoryless distill {path} ({path.stat().st_size} B)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=list(range(100, 116)))
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[7, 11])
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--eval-per", type=int, default=2)
    ap.add_argument("--bc-epochs", type=int, default=25)
    ap.add_argument("--dagger-rounds", type=int, default=4)
    ap.add_argument("--dagger-eps", type=int, default=24)
    ap.add_argument("--ppo-steps", type=int, default=80_000)
    ap.add_argument("--rollout-steps", type=int, default=6_000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "rl_dagger_v07")
    ap.add_argument("--skip-ppo", action="store_true")
    ap.add_argument("--final-eval-per", type=int, default=8, help="8 → 64 eps/seed for final compare")
    a = ap.parse_args()
    os.environ["NIST_PROFILE"] = "v07"
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    meta = {
        "profile": "v07",
        "method": "gru_bc_dagger_ppo",
        "train_seeds": a.train_seeds,
        "eval_seeds": a.eval_seeds,
        "note": (
            "Learned GRU policy; teacher demos from honest privileged offline only. "
            "Eval is observe→action (miner-path). Dense rewards use obs progress channels only."
        ),
        "weights": STEP_WEIGHT_V07,
    }
    (a.out / "provenance.json").write_text(json.dumps(meta, indent=2))

    curve = []
    print("== baselines (miner-path / refs) ==", flush=True)
    for mode in ("scripted", "v06_style"):
        row = eval_policy(None, mode, a.eval_seeds, a.eval_per)
        print(
            f"{mode}: mean={row['mean']:.4f} key={row['key_done']}/{row['key_drawn']} "
            f"stack={row['stack_done']}/{row['stack_drawn']} rework={row['rework_done']}/{row['rework_drawn']} "
            f"wall={row['wall_s']:.0f}s",
            flush=True,
        )
        curve.append({"tag": mode, **{k: row[k] for k in row if k != "scores"}})

    print("== teacher demos (honest privileged) ==", flush=True)
    t0 = time.perf_counter()
    seq_x, seq_y, priv_mean = collect_teacher_demos(a.train_seeds, a.per, a.workers)
    print(
        f"demos seqs={len(seq_x)} steps={sum(len(x) for x in seq_x)} priv_mean={priv_mean:.3f} "
        f"t={time.perf_counter()-t0:.0f}s",
        flush=True,
    )
    (a.out / "demo_stats.json").write_text(
        json.dumps({"n_seq": len(seq_x), "n_steps": sum(len(x) for x in seq_x), "priv_mean": priv_mean}, indent=2)
    )

    net = GRUActorCritic()
    print("== BC pretrain ==", flush=True)
    bc_losses = bc_train(net, seq_x, seq_y, epochs=a.bc_epochs)
    row = eval_policy(net, "learned", a.eval_seeds, a.eval_per)
    print(
        f"after BC mean={row['mean']:.4f} key={row['key_done']}/{row['key_drawn']} "
        f"stack={row['stack_done']}/{row['stack_drawn']}",
        flush=True,
    )
    curve.append({"tag": "after_bc", "bc_mse": bc_losses[-1] if bc_losses else None, **{k: row[k] for k in row if k != "scores"}})
    torch.save(net.state_dict(), a.out / "gru_after_bc.pt")

    print("== DAgger ==", flush=True)
    beta = 0.85
    for rnd in range(a.dagger_rounds):
        dx, dy, dscores = dagger_round(net, a.train_seeds, a.per, beta=beta, n_episodes=a.dagger_eps, seed0=20_000 + rnd)
        print(
            f"  dagger {rnd} beta={beta:.2f} new_seq={len(dx)} mix_mean={float(np.mean(dscores)):.4f}",
            flush=True,
        )
        seq_x.extend(dx)
        seq_y.extend(dy)
        # Keep a cap on dataset size (prefer recent).
        if len(seq_x) > 400:
            seq_x = seq_x[-400:]
            seq_y = seq_y[-400:]
        bc_train(net, seq_x[-120:], seq_y[-120:], epochs=8)
        row = eval_policy(net, "learned", a.eval_seeds, a.eval_per)
        print(
            f"  after dagger {rnd} mean={row['mean']:.4f} key={row['key_done']}/{row['key_drawn']} "
            f"stack={row['stack_done']}/{row['stack_drawn']}",
            flush=True,
        )
        curve.append({"tag": f"dagger_{rnd}", "beta": beta, **{k: row[k] for k in row if k != "scores"}})
        torch.save(net.state_dict(), a.out / f"gru_dagger_{rnd}.pt")
        beta = max(0.15, beta * 0.65)

    if not a.skip_ppo:
        print("== PPO fine-tune ==", flush=True)
        done_steps = 0
        it = 0
        while done_steps < a.ppo_steps:
            it += 1
            batch = ppo_collect(net, a.train_seeds, a.per, a.rollout_steps, seed0=30_000 + it)
            loss = ppo_update(net, batch)
            done_steps += batch["steps"]
            mean_ep = float(np.mean(batch["ep_scores"])) if batch["ep_scores"] else float("nan")
            print(
                f"iter {it} steps={done_steps}/{a.ppo_steps} loss={loss:.4f} "
                f"rollout_mean={mean_ep:.4f} n_eps={len(batch['ep_scores'])}",
                flush=True,
            )
            if it % 3 == 0 or done_steps >= a.ppo_steps:
                row = eval_policy(net, "learned", a.eval_seeds, a.eval_per)
                print(
                    f"  eval mean={row['mean']:.4f} key={row['key_done']}/{row['key_drawn']} "
                    f"stack={row['stack_done']}/{row['stack_drawn']}",
                    flush=True,
                )
                curve.append(
                    {
                        "tag": "ppo",
                        "iter": it,
                        "steps": done_steps,
                        **{k: row[k] for k in row if k != "scores"},
                        "loss": loss,
                    }
                )

    torch.save(net.state_dict(), a.out / "gru_final.pt")
    export_onnx_memoryless(net, a.out / "rl_dagger_v07.onnx", seq_x)

    print("== final held-out compare ==", flush=True)
    final = {}
    for mode in ("scripted", "v06_style", "learned"):
        row = eval_policy(net if mode == "learned" else None, mode, a.eval_seeds, a.final_eval_per)
        final[mode] = {k: row[k] for k in row if k != "scores"}
        print(
            f"FINAL {mode}: mean={row['mean']:.4f} ±{row['std']:.4f} n={row['n']} "
            f"key={row['key_done']}/{row['key_drawn']} stack={row['stack_done']}/{row['stack_drawn']} "
            f"rework={row['rework_done']}/{row['rework_drawn']} wall={row['wall_s']:.0f}s",
            flush=True,
        )
        (a.out / f"final_{mode}_scores.json").write_text(json.dumps(row["scores"]))

    # Ship-readiness for RL claim: learned > strong script proxy on held-out miner-path.
    strong = final["v06_style"]["mean"]
    learned = final["learned"]["mean"]
    scripted = final["scripted"]["mean"]
    # Published image-through kings on this harden ≈ 0.189–0.213; require clear margin over max(proxy, 0.20).
    king_bar = 0.20
    beats = learned > max(strong, king_bar) + 0.01
    readiness = {
        "rl_claim_ship_ready": bool(beats),
        "learned_mean": learned,
        "v06_style_mean": strong,
        "compound_scripted_mean": scripted,
        "king_bar": king_bar,
        "margin_over_strong": learned - max(strong, king_bar),
        "eval_seeds": a.eval_seeds,
        "eval_per": a.final_eval_per,
        "path": "miner-path in-process (observe→GRU→step); ONNX distill is memoryless approx",
    }
    summary = {
        "provenance": meta,
        "curve": curve,
        "final": final,
        "readiness": readiness,
        "commands": {
            "train": (
                "NIST_PROFILE=v07 python tools/train_rl_dagger_v07.py "
                f"--train-seeds {' '.join(map(str, a.train_seeds))} "
                f"--eval-seeds {' '.join(map(str, a.eval_seeds))} "
                f"--ppo-steps {a.ppo_steps} --out {a.out}"
            ),
        },
    }
    (a.out / "learning_curve.json").write_text(json.dumps(curve, indent=2, default=str))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(readiness, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
