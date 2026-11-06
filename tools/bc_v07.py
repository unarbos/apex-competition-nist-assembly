"""Behavior cloning from honest privileged demos on Assembly III (v07).

Collects (noisy observation, privileged action) under ``solve_compound(..., oracle=False)``
on train seeds, fits a memoryless MLP ``[B,288] → [B,5]``, exports ONNX (opset 13).

This is a **learned** policy with clear provenance (imitation of the honest solver).
It is **not** RL by itself — pair with ``tools/train_rl_v07.py`` for residual PPO.
It is **not** a privileged controller at eval (observe → ONNX only).

    NIST_PROFILE=v07 python tools/bc_v07.py \\
        --train-seeds 100 101 102 103 104 105 106 107 \\
        --eval-seeds 7 11 --per 2 --epochs 60 --out out/bc_v07
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _init() -> None:
    os.environ["NIST_PROFILE"] = "v07"
    sys.path.insert(0, str(ROOT / "referee"))
    sys.path.insert(0, str(ROOT / "tools"))


def collect(job: tuple) -> tuple:
    seed, per, i = job
    import numpy as np

    from compound_solver import solve_compound
    from nist_sim import conditions as C
    from nist_sim.compound_v07 import AssemblyIIIEpisode, sample_compound_v07

    _, conds = sample_compound_v07(seed, per)
    r1, r2 = (np.random.default_rng(s) for s in C.episode_seed_sequence(seed, i).spawn(2))
    ep = AssemblyIIIEpisode(conds[i], r1)
    X, Y = [], []
    raw = ep.step

    def rec(a):
        if not ep.done:
            X.append(np.round(ep.observe(r2), 6).astype(np.float32))
            Y.append(np.clip(np.asarray(a, np.float32), -1, 1))
        return raw(a)

    ep.step = rec  # type: ignore[method-assign]
    r = solve_compound(ep, oracle=False)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), float(r["score"])


def eval_policy(act_fn, seeds: list[int], per: int) -> dict:
    from nist_sim import conditions as C
    from nist_sim.compound_v07 import AssemblyIIIEpisode, compound_score_v07, sample_compound_v07

    scores = []
    kd: dict[str, list[int]] = {}
    for seed in seeds:
        _, conds = sample_compound_v07(seed, per)
        for i, c in enumerate(conds):
            r1, r2 = (np.random.default_rng(s) for s in C.episode_seed_sequence(seed, i).spawn(2))
            ep = AssemblyIIIEpisode(c, r1)
            while not ep.done:
                o = np.round(ep.observe(r2), 6).astype(np.float32)
                a = np.asarray(act_fn(o), dtype=float)
                if not np.all(np.isfinite(a)):
                    a = np.zeros(5)
                ep.step(np.clip(a, -1, 1))
            r = compound_score_v07(ep)
            scores.append(float(r["score"]))
            for k, s in enumerate(c.compound.steps):
                kd.setdefault(s.kind, [0, 0])
                kd[s.kind][1] += 1
                if ep.step_done[k]:
                    kd[s.kind][0] += 1
    return {
        "mean": float(np.mean(scores)) if scores else 0.0,
        "std": float(np.std(scores)) if scores else 0.0,
        "n": len(scores),
        "scores": scores,
        "kind_done": {k: f"{v[0]}/{v[1]}" for k, v in kd.items()},
        "stack_done": kd.get("stack", [0, 0])[0],
        "stack_drawn": kd.get("stack", [0, 0])[1],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=list(range(100, 116)))
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[7, 11])
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--eval-per", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "bc_v07")
    a = ap.parse_args()
    _init()
    a.out.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn as nn
    from nist_sim.compound_v07 import OBS_DIM_V07

    jobs = [(s, a.per, i) for s in a.train_seeds for i in range(8 * a.per)]
    t0 = time.time()
    with ProcessPoolExecutor(
        a.workers, mp_context=mp.get_context("spawn"), initializer=_init
    ) as ex:
        res = list(ex.map(collect, jobs, chunksize=1))
    X = np.concatenate([r[0] for r in res if len(r[0])])
    Y = np.concatenate([r[1] for r in res if len(r[1])])
    priv_mean = float(np.mean([r[2] for r in res]))
    print(
        f"demos eps={len(res)} samples={len(X)} priv_mean={priv_mean:.3f} t={time.time() - t0:.0f}s",
        flush=True,
    )
    mu, sd = X.mean(0), X.std(0) + 1e-3
    Xn = (X - mu) / sd
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Linear(OBS_DIM_V07, a.hidden),
        nn.ReLU(),
        nn.Linear(a.hidden, a.hidden),
        nn.ReLU(),
        nn.Linear(a.hidden, a.hidden // 2),
        nn.ReLU(),
        nn.Linear(a.hidden // 2, 5),
        nn.Tanh(),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    bs = min(2048, len(Xn))
    curve = []
    for ep in range(a.epochs):
        idx = np.random.randint(0, len(Xn), bs)
        pred = net(torch.as_tensor(Xn[idx]))
        loss = ((pred - torch.as_tensor(Y[idx])) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 5 == 0 or ep == a.epochs - 1:
            row = {"ep": ep, "mse": float(loss.detach())}
            curve.append(row)
            print(f"bc {ep} mse={row['mse']:.5f}", flush=True)

    class Deploy(nn.Module):
        def __init__(self, body, mu_a, sd_a):
            super().__init__()
            self.body = body
            self.register_buffer("mu", torch.as_tensor(mu_a))
            self.register_buffer("sd", torch.as_tensor(sd_a))

        def forward(self, x):
            return self.body((x - self.mu) / self.sd)

    deploy = Deploy(net, mu.astype(np.float32), sd.astype(np.float32)).eval()

    def act_fn(o):
        with torch.no_grad():
            return deploy(torch.as_tensor(o).unsqueeze(0)).squeeze(0).numpy()

    print("evaluating held-out…", flush=True)
    ev = eval_policy(act_fn, a.eval_seeds, a.eval_per)
    print(
        f"eval mean={ev['mean']:.4f} stack={ev['stack_done']}/{ev['stack_drawn']} kinds={ev['kind_done']}",
        flush=True,
    )

    onnx_path = a.out / "bc_honest_v07.onnx"
    dummy = torch.zeros(1, OBS_DIM_V07)
    torch.onnx.export(
        deploy,
        dummy,
        str(onnx_path),
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "B"}, "actions": {0: "B"}},
        opset_version=13,
    )
    summary = {
        "method": "behavior_cloning_honest_privileged_actions",
        "provenance": (
            "Demos from solve_compound(oracle=False). Eval is observe→MLP only. "
            "Not a privileged controller; not RL (see train_rl_v07.py for PPO)."
        ),
        "train_seeds": a.train_seeds,
        "eval_seeds": a.eval_seeds,
        "per": a.per,
        "samples": int(len(X)),
        "priv_demo_mean": priv_mean,
        "epochs": a.epochs,
        "eval": {k: ev[k] for k in ("mean", "std", "n", "kind_done", "stack_done", "stack_drawn")},
        "curve": curve,
        "onnx": str(onnx_path),
        "commands": {
            "train": (
                "NIST_PROFILE=v07 python tools/bc_v07.py "
                f"--train-seeds {' '.join(map(str, a.train_seeds))} "
                f"--eval-seeds {' '.join(map(str, a.eval_seeds))} --out {a.out}"
            ),
            "image_eval": (
                f"python tools/image_eval_onnx.py --seed 7 --full "
                f"--referee-image nist-assembly-referee:v072 "
                f"--player-image nist-assembly-player:v072 {onnx_path}"
            ),
        },
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (a.out / "learning_curve.json").write_text(json.dumps(curve, indent=2))
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
