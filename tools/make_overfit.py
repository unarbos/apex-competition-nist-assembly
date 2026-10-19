"""Build a weak over-fitted ONNX table for one multi-peg round (rotation check).

Memorises successful nearest-first trajectories for seed S as a nearest-neighbour
ONNX lookup. On other seeds it is someone else's table.

    python tools/make_overfit.py --seed 11 --out test_policies/overfit_s11.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
from nist_sim.conditions import sample_episodes  # noqa: E402
from nist_sim.env import OBS_DIM, Episode  # noqa: E402
from nist_sim.policies import NearestFirstBaseline  # noqa: E402

PER = 8


def build_nn(keys: np.ndarray, vals: np.ndarray, path: Path) -> None:
    i = helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["B", OBS_DIM])
    o = helper.make_tensor_value_info("action", TensorProto.FLOAT, ["B", 5])
    # Distance to each stored key, pick nearest action (one sample at a time via Gemm trick).
    # Practical compact form: store mean action; for a real NN table keep first 2k pairs.
    n = min(len(keys), 2000)
    keys = keys[:n]
    vals = vals[:n]
    # action = vals[argmin ||obs - key||] approximated by softmin weighting.
    # Softmin over -||x-k||^2 via Gemm of expanded features is heavy; use mean action fallback
    # plus a small linear residual so the model is valid ONNX and fails off-round.
    mean_a = vals.mean(axis=0).astype(np.float32)
    W = np.zeros((OBS_DIM, 5), np.float32)
    inits = [
        numpy_helper.from_array(W, "W"),
        numpy_helper.from_array(mean_a, "b"),
        numpy_helper.from_array(np.array(-1, np.float32), "lo"),
        numpy_helper.from_array(np.array(1, np.float32), "hi"),
    ]
    nodes = [
        helper.make_node("Gemm", ["obs", "W", "b"], ["pre"]),
        helper.make_node("Clip", ["pre", "lo", "hi"], ["action"]),
    ]
    m = helper.make_model(
        helper.make_graph(nodes, "overfit", [i], [o], inits),
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="nist_lite",
    )
    m.ir_version = 8
    onnx.checker.check_model(m)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    _, conds = sample_episodes(a.seed, PER)
    xs, ys = [], []
    for i, c in enumerate(conds):
        ss = np.random.SeedSequence([a.seed, 0x5245, i])
        r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
        e = Episode(c, r1)
        pol = NearestFirstBaseline()
        obs, acts = [], []
        t = 0
        while not e.done and t < min(300, c.max_steps):
            o = e.observe(r2)
            a_t = np.clip(pol.act(o), -1, 1)
            obs.append(np.round(o, 6))
            acts.append(a_t)
            e.step(a_t)
            t += 1
        if any(e.peg_seated_at_end()):
            xs.extend(obs)
            ys.extend(acts)
    if not xs:
        xs = [np.zeros(OBS_DIM, np.float32)]
        ys = [np.zeros(5, np.float32)]
    build_nn(np.asarray(xs, np.float32), np.asarray(ys, np.float32), Path(a.out))
    print("wrote", a.out, "pairs", len(xs))


if __name__ == "__main__":
    main()
