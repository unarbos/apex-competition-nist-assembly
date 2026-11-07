"""Residual GRU PPO on CompoundObsScripted — miner-path sealed.

Warm-start equals the obs-only scripted controller. A zero-init GRU residual is
trained with PPO using dense rewards from observable progress channels only.
Optional short BC of residual targets from honest privileged deltas (when small).

    NIST_PROFILE=v07 python tools/train_rl_residual_gru_v07.py \\
        --train-seeds 100 101 102 103 104 105 106 107 \\
        --eval-seeds 7 11 --ppo-steps 120000 --out out/rl_resid_gru_v07
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
from train_rl_dagger_v07 import (  # noqa: E402
    COL0,
    COL_FEAT,
    MAX_COLLARS,
    MAX_PEGS,
    PEG0,
    PEG_FEAT,
    TASK0,
    TASK_FEAT,
    dense_reward,
    eval_policy,
    obs_progress_features,
)
import math  # noqa: E402


def key_alignment_bonus(o: np.ndarray) -> float:
    """Observable: how well peg yaw matches key sin/cos (0=bad, 1=aligned)."""
    best = 0.0
    for i in range(MAX_PEGS):
        f = o[PEG0 + i * PEG_FEAT : PEG0 + (i + 1) * PEG_FEAT]
        t = o[TASK0 + i * TASK_FEAT : TASK0 + (i + 1) * TASK_FEAT]
        if f[0] < 0.5 or f[16] < 0.5:
            continue
        s2, s3, s5 = float(t[2]), float(t[3]), float(t[5])
        # key_tol*10 packs into t[5] ∈ ~(0.5, 0.9); collar flags are 1.0 / 2.0.
        if not (0.25 < s5 < 0.95):
            continue
        key_yaw = math.atan2(s2, s3)
        qw, qx, qy, qz = (float(f[4 + k]) for k in range(4))
        peg_yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        err = abs((key_yaw - peg_yaw + math.pi) % (2 * math.pi) - math.pi)
        tol = max(0.05, s5 / 10.0)
        best = max(best, float(np.clip(1.0 - err / max(tol, 1e-3), 0.0, 1.0)))
    return best

OBS_DIM = OBS_DIM_V07
RESIDUAL_SCALE = 0.55
INIT_LOG_STD = -1.6
STD_MAX = 0.40
HID = 256
GAMMA = 0.995
LAM = 0.95


class ResidGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(OBS_DIM, HID), nn.Tanh())
        self.gru = nn.GRUCell(HID, HID)
        self.mu = nn.Linear(HID, ACT_DIM)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), float(INIT_LOG_STD)))
        self.v = nn.Linear(HID, 1)
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if h is None:
            h = torch.zeros(x.shape[0], self.gru.hidden_size, device=x.device)
        h = self.gru(self.enc(x), h)
        mu = torch.tanh(self.mu(h))
        return mu, self.log_std.expand_as(mu), self.v(h).squeeze(-1), h

    def act_np(self, o, h, deterministic=True):
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
            return a.clamp(-1, 1).numpy(), float(logp), float(v.item()), h_n


def _combine(base: np.ndarray, resid: np.ndarray) -> np.ndarray:
    return np.clip(base + RESIDUAL_SCALE * resid, -1.0, 1.0)


def _make_ep(seed: int, per: int, i: int):
    _, conds = sample_compound_v07(seed, per)
    r1, r2 = (np.random.default_rng(s) for s in C.episode_seed_sequence(seed, i).spawn(2))
    return AssemblyIIIEpisode(conds[i], r1), r2


def eval_fused(net: ResidGRU, seeds, per) -> dict:
    scores = []
    kind_done: dict[str, list[int]] = {}
    t0 = time.perf_counter()
    for seed in seeds:
        _, conds = sample_compound_v07(seed, per)
        for i, c in enumerate(conds):
            ep, r2 = _make_ep(seed, per, i)
            pol = CompoundObsScripted()
            h = None
            while not ep.done:
                o = np.round(ep.observe(r2), 6).astype(np.float32)
                base = np.clip(pol.act(o.astype(np.float64)), -1, 1)
                resid, _, _, h = net.act_np(o, h, deterministic=True)
                ep.step(_combine(base, resid))
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
        "kind_done": {k: f"{v[0]}/{v[1]}" for k, v in kind_done.items()},
        "stack_done": kind_done.get("stack", [0, 0])[0],
        "stack_drawn": kind_done.get("stack", [0, 0])[1],
        "key_done": kind_done.get("key", [0, 0])[0],
        "key_drawn": kind_done.get("key", [0, 0])[1],
        "rework_done": kind_done.get("rework", [0, 0])[0],
        "rework_drawn": kind_done.get("rework", [0, 0])[1],
        "wall_s": time.perf_counter() - t0,
        "scores": scores,
    }


def collect(net, seeds, per, n_steps, seed0):
    net.eval()
    rng = np.random.default_rng(seed0)
    obs_l, act_l, logp_l, rew_l, val_l, done_l = [], [], [], [], [], []
    ep_scores = []
    steps = 0
    while steps < n_steps:
        seed = int(rng.choice(seeds))
        _, conds = sample_compound_v07(seed, per)
        i = int(rng.integers(0, len(conds)))
        ep, r2 = _make_ep(seed, per, i)
        pol = CompoundObsScripted()
        h = None
        o = np.round(ep.observe(r2), 6).astype(np.float32)
        prev = obs_progress_features(o)
        while not ep.done and steps < n_steps:
            base = np.clip(pol.act(o.astype(np.float64)), -1, 1)
            resid, logp, val, h = net.act_np(o, h, deterministic=False)
            a = _combine(base, resid)
            ep.step(a)
            o2 = np.round(ep.observe(r2), 6).astype(np.float32) if not ep.done else o
            feat = obs_progress_features(o2)
            term = float(compound_score_v07(ep)["score"]) if ep.done else None
            if ep.done:
                ep_scores.append(term)
            # Extra bonus for observable key/stack/rework-ish progress (collar done / seated).
            r = dense_reward(prev, feat, float(np.sum(resid**2)), term)
            # Curriculum bonus: seating unlocks hard steps.
            r += 1.0 * float(np.sum(np.clip(feat[0:8] - prev[0:8], 0, 1)))
            # Observable key yaw alignment potential (no privileged writes).
            ka = key_alignment_bonus(o2)
            r += 4.0 * max(0.0, ka - key_alignment_bonus(o))
            obs_l.append(o)
            act_l.append(resid.astype(np.float32))
            logp_l.append(logp)
            rew_l.append(r)
            val_l.append(val)
            done_l.append(bool(ep.done))
            o, prev = o2, feat
            steps += 1
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


def ppo_update(net, batch, epochs=4, clip=0.2, lr=3e-4):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    obs = torch.as_tensor(batch["obs"])
    act = torch.as_tensor(batch["act"])
    old_logp = torch.as_tensor(batch["logp"])
    adv = torch.as_tensor(batch["adv"])
    ret = torch.as_tensor(batch["ret"])
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = obs.shape[0]
    bs = min(2048, n)
    stats = []
    for _ in range(epochs):
        idx = torch.randperm(n)
        for s in range(0, n, bs):
            j = idx[s : s + bs]
            # Memoryless PPO update on encoder+head (h=0); temporal credit via GAE on rollouts.
            mu, log_std, v, _ = net.forward(obs[j], None)
            dist = Normal(mu, log_std.exp().clamp(1e-3, STD_MAX))
            logp = dist.log_prob(act[j]).sum(-1)
            ratio = (logp - old_logp[j]).exp()
            surr = torch.min(ratio * adv[j], ratio.clamp(1 - clip, 1 + clip) * adv[j])
            loss = -surr.mean() + 0.5 * ((v - ret[j]) ** 2).mean() - 0.015 * dist.entropy().sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            stats.append(float(loss.detach()))
    return float(np.mean(stats)) if stats else 0.0


def export_onnx(net: ResidGRU, path: Path, seeds=(100, 101, 102)):
    """Distill fused scripted+residual into memoryless MLP for the ONNX player."""
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
    for seed in seeds:
        _, conds = sample_compound_v07(seed, 2)
        for i, _ in enumerate(conds):
            ep, r2 = _make_ep(seed, 2, i)
            pol = CompoundObsScripted()
            h = None
            while not ep.done:
                o = np.round(ep.observe(r2), 6).astype(np.float32)
                base = np.clip(pol.act(o.astype(np.float64)), -1, 1)
                resid, _, _, h = net.act_np(o, h, True)
                a = _combine(base, resid)
                xs.append(o)
                ys.append(a.astype(np.float32))
                ep.step(a)
            if len(xs) > 30_000:
                break
        if len(xs) > 30_000:
            break
    Xt, Yt = torch.as_tensor(np.asarray(xs)), torch.as_tensor(np.asarray(ys))
    for _ in range(40):
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
    print(f"exported {path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=list(range(100, 116)))
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[7, 11])
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--eval-per", type=int, default=2)
    ap.add_argument("--ppo-steps", type=int, default=120_000)
    ap.add_argument("--rollout-steps", type=int, default=8_000)
    ap.add_argument("--final-eval-per", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "rl_resid_gru_v07")
    a = ap.parse_args()
    os.environ["NIST_PROFILE"] = "v07"
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    meta = {
        "method": "residual_gru_ppo_on_compound_obs_scripted",
        "residual_scale": RESIDUAL_SCALE,
        "train_seeds": a.train_seeds,
        "eval_seeds": a.eval_seeds,
        "note": "Fused policy = scripted + scale*GRU residual. Eval miner-path only.",
    }
    (a.out / "provenance.json").write_text(json.dumps(meta, indent=2))

    curve = []
    print("== baselines ==", flush=True)
    for mode in ("scripted", "v06_style"):
        row = eval_policy(None, mode, a.eval_seeds, a.eval_per)
        print(f"{mode}: mean={row['mean']:.4f} key={row['key_done']}/{row['key_drawn']} stack={row['stack_done']}/{row['stack_drawn']}", flush=True)
        curve.append({"tag": mode, **{k: row[k] for k in row if k != "scores"}})

    net = ResidGRU()
    row = eval_fused(net, a.eval_seeds, a.eval_per)
    print(f"warm residual(=scripted): mean={row['mean']:.4f}", flush=True)
    curve.append({"tag": "warm", **{k: row[k] for k in row if k != "scores"}})

    print("== PPO ==", flush=True)
    done_steps = 0
    it = 0
    best = row["mean"]
    while done_steps < a.ppo_steps:
        it += 1
        batch = collect(net, a.train_seeds, a.per, a.rollout_steps, seed0=40_000 + it)
        loss = ppo_update(net, batch)
        done_steps += batch["steps"]
        mean_ep = float(np.mean(batch["ep_scores"])) if batch["ep_scores"] else float("nan")
        print(
            f"iter {it} steps={done_steps}/{a.ppo_steps} loss={loss:.4f} "
            f"rollout_mean={mean_ep:.4f} n_eps={len(batch['ep_scores'])}",
            flush=True,
        )
        if it % 2 == 0 or done_steps >= a.ppo_steps:
            row = eval_fused(net, a.eval_seeds, a.eval_per)
            print(
                f"  eval mean={row['mean']:.4f} key={row['key_done']}/{row['key_drawn']} "
                f"stack={row['stack_done']}/{row['stack_drawn']}",
                flush=True,
            )
            curve.append({"tag": "ppo", "iter": it, "steps": done_steps, **{k: row[k] for k in row if k != "scores"}})
            if row["mean"] >= best:
                best = row["mean"]
                torch.save(net.state_dict(), a.out / "resid_gru_best.pt")

    torch.save(net.state_dict(), a.out / "resid_gru_final.pt")
    export_onnx(net, a.out / "rl_resid_gru_v07.onnx")

    print("== final ==", flush=True)
    final = {}
    for mode in ("scripted", "v06_style"):
        row = eval_policy(None, mode, a.eval_seeds, a.final_eval_per)
        final[mode] = {k: row[k] for k in row if k != "scores"}
        print(f"FINAL {mode}: mean={row['mean']:.4f} n={row['n']} kinds={row['kind_done']}", flush=True)
    row = eval_fused(net, a.eval_seeds, a.final_eval_per)
    final["learned_fused"] = {k: row[k] for k in row if k != "scores"}
    print(
        f"FINAL fused: mean={row['mean']:.4f} key={row['key_done']}/{row['key_drawn']} "
        f"stack={row['stack_done']}/{row['stack_drawn']} rework={row['rework_done']}/{row['rework_drawn']}",
        flush=True,
    )
    king_bar = 0.20
    strong = max(final["v06_style"]["mean"], king_bar)
    learned = final["learned_fused"]["mean"]
    readiness = {
        "rl_claim_ship_ready": bool(learned > strong + 0.01),
        "learned_mean": learned,
        "v06_style_mean": final["v06_style"]["mean"],
        "scripted_mean": final["scripted"]["mean"],
        "king_bar": king_bar,
        "margin": learned - strong,
        "eval_seeds": a.eval_seeds,
        "eval_per": a.final_eval_per,
        "path": "miner-path fused scripted+GRU residual",
    }
    summary = {"provenance": meta, "curve": curve, "final": final, "readiness": readiness}
    (a.out / "learning_curve.json").write_text(json.dumps(curve, indent=2, default=str))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(readiness, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
