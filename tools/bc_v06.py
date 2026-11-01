"""Learnability sanity check for v0.6 compound tasks: behaviour cloning from the privileged solver.

Collects (observation, privileged action) pairs on training seeds (disjoint from calibration seeds),
fits a memoryless MLP ``[B,288] -> [B,5]`` and exports ONNX for ``tools/calib_v06.py``.

    python tools/bc_v06.py --seeds 100 101 102 103 104 105 106 107 --per 2 --out /tmp/bc_v06.onnx

This is a floor for "a learned policy can climb", not a tuned recipe (no DAgger, no RL fine-tune).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _init() -> None:
    os.environ["NIST_PROFILE"] = "v06"
    sys.path.insert(0, str(ROOT / "referee"))
    sys.path.insert(0, str(ROOT / "tools"))


def collect(job):
    seed, per, i = job
    import nist_sim.conditions as C
    from compound_solver import solve_compound
    from nist_sim.compound import CompoundEpisode, sample_compound

    _, conds = sample_compound(seed, per)
    r1, r2 = (np.random.default_rng(s) for s in C.episode_seed_sequence(seed, i).spawn(2))
    ep = CompoundEpisode(conds[i], r1)
    X, Y = [], []
    raw_step = ep.step

    def rec_step(a):
        if not ep.done:
            X.append(np.round(ep.observe(r2), 6))
            Y.append(np.clip(np.asarray(a, dtype=np.float32), -1, 1))
        raw_step(a)

    ep.step = rec_step
    r = solve_compound(ep)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), float(r["score"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 108)))
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--out", default="bc_v06.onnx")
    a = ap.parse_args()
    _init()
    import torch
    import torch.nn as nn

    t = time.time()
    jobs = [(s, a.per, i) for s in a.seeds for i in range(8 * a.per)]
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn"), initializer=_init) as ex:
        res = list(ex.map(collect, jobs, chunksize=1))
    X = np.concatenate([r[0] for r in res if len(r[0])])
    Y = np.concatenate([r[1] for r in res if len(r[1])])
    print(f"demos: {len(res)} episodes, {len(X)} samples, privileged mean {np.mean([r[2] for r in res]):.3f}, "
          f"{time.time() - t:.0f} s", flush=True)
    mu, sd = X.mean(0), X.std(0) + 1e-3
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(X.shape[1], a.hidden), nn.ReLU(), nn.Linear(a.hidden, a.hidden), nn.ReLU(),
                        nn.Linear(a.hidden, a.hidden), nn.ReLU(), nn.Linear(a.hidden, 5), nn.Tanh())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.tensor((X - mu) / sd)
    Yt = torch.tensor(Y)
    n = len(Xt)
    for ep in range(a.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for b in range(0, n, 1024):
            idx = perm[b : b + 1024]
            loss = ((net(Xt[idx]) - Yt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"epoch {ep} mse {tot / n:.4f}", flush=True)

    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.mu = nn.Parameter(torch.tensor(mu), requires_grad=False)
            self.sd = nn.Parameter(torch.tensor(sd), requires_grad=False)
            self.net = net

        def forward(self, x):
            return self.net((x - self.mu) / self.sd)

    m = Wrapped().eval()
    torch.save(m.state_dict(), a.out + ".pt")  # keep the weights if the ONNX export fails
    torch.onnx.export(m, torch.zeros(1, X.shape[1]), a.out, input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "B"}, "action": {0: "B"}}, opset_version=17, dynamo=False)
    print("wrote", a.out, f"{time.time() - t:.0f} s total")


if __name__ == "__main__":
    main()
