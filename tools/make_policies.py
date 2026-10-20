"""Build the reference baseline and the test-set ONNX policies (no torch needed).

    python tools/make_policies.py   # writes baselines/*.onnx and test_policies/*.onnx

The baseline ONNX is a ridge-linear distillation of NearestFirstBaseline rollouts
(soft-nearest insert/release, no clutter handling, no lying regrasp).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OBS, ACT = 184, 5
ROOT = Path(__file__).resolve().parents[1]
OPSET = [helper.make_opsetid("", 17)]


def _io():
    return (
        helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["B", OBS]),
        helper.make_tensor_value_info("action", TensorProto.FLOAT, ["B", ACT]),
    )


def _save(graph, path: Path):
    m = helper.make_model(graph, opset_imports=OPSET, producer_name="nist_lite")
    m.ir_version = 8
    onnx.checker.check_model(m)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(path))
    print("wrote", path.relative_to(ROOT))


def linear(name: str, W: np.ndarray, b: np.ndarray, path: Path, clip=True):
    i, o = _io()
    inits = [
        numpy_helper.from_array(W.astype(np.float32), "W"),
        numpy_helper.from_array(b.astype(np.float32), "b"),
    ]
    nodes = [helper.make_node("Gemm", ["obs", "W", "b"], ["pre" if clip else "action"])]
    if clip:
        inits += [
            numpy_helper.from_array(np.array(-1, np.float32), "lo"),
            numpy_helper.from_array(np.array(1, np.float32), "hi"),
        ]
        nodes.append(helper.make_node("Clip", ["pre", "lo", "hi"], ["action"]))
    _save(helper.make_graph(nodes, name, [i], [o], inits), path)


def baseline(path: Path):
    """Distill NearestFirstBaseline into a clipped linear ONNX policy."""
    sys.path.insert(0, str(ROOT / "referee"))
    from nist_sim.conditions import sample_episodes
    from nist_sim.env import Episode
    from nist_sim.policies import NearestFirstBaseline

    xs, ys = [], []
    for seed in (0, 1, 2, 3):
        _, conds = sample_episodes(seed, 8)
        for i, c in enumerate(conds[:24]):
            ss = np.random.SeedSequence([seed, 0x5245, i])
            r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
            e = Episode(c, r1)
            pol = NearestFirstBaseline()
            steps = 0
            while not e.done and steps < min(350, c.max_steps):
                o = e.observe(r2)
                a = np.clip(pol.act(o), -1, 1)
                xs.append(o)
                ys.append(a)
                e.step(a)
                steps += 1
    X = np.asarray(xs, np.float64)
    Y = np.asarray(ys, np.float64)
    lam = 1e-2
    W = np.linalg.solve(X.T @ X + lam * np.eye(OBS), X.T @ Y).astype(np.float32)
    b = (Y.mean(axis=0) - X.mean(axis=0) @ W).astype(np.float32)
    linear("baseline", W, b, path)


def wrong_shape(path: Path):
    i = helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["B", OBS])
    o = helper.make_tensor_value_info("action", TensorProto.FLOAT, ["B", 3])
    W = numpy_helper.from_array(np.zeros((OBS, 3), np.float32), "W")
    b = numpy_helper.from_array(np.zeros(3, np.float32), "b")
    _save(
        helper.make_graph(
            [helper.make_node("Gemm", ["obs", "W", "b"], ["action"])], "wrong", [i], [o], [W, b]
        ),
        path,
    )


def slow(path: Path, reps: int = 400):
    i, o = _io()
    big = numpy_helper.from_array(np.full((OBS, 2048), 1e-4, np.float32), "U")
    sq = numpy_helper.from_array(np.eye(2048, dtype=np.float32), "S")
    down = numpy_helper.from_array(np.zeros((2048, ACT), np.float32), "D")
    nodes = [helper.make_node("MatMul", ["obs", "U"], ["h0"])]
    for k in range(reps):
        nodes.append(helper.make_node("MatMul", [f"h{k}", "S"], [f"h{k + 1}"]))
    nodes.append(helper.make_node("MatMul", [f"h{reps}", "D"], ["action"]))
    _save(helper.make_graph(nodes, "slow", [i], [o], [big, sq, down]), path)


def const_batch2(path: Path):
    i, o = _io()
    c = numpy_helper.from_array(np.zeros((2, ACT), np.float32), "c")
    nodes = [
        helper.make_node("Shape", ["obs"], ["s"]),
        helper.make_node("Constant", [], ["action"], value=c),
    ]
    _save(helper.make_graph(nodes, "const_batch2", [i], [o]), path)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    baseline(ROOT / "baselines" / "baseline.onnx")
    adv = ROOT / "test_policies"
    linear("zero", np.zeros((OBS, ACT)), np.zeros(ACT), adv / "zero.onnx")
    linear(
        "constant_push",
        np.zeros((OBS, ACT)),
        np.array([0, 0, -1, 0, -1]),
        adv / "constant_push.onnx",
    )
    linear("random", rng.normal(0, 3, (OBS, ACT)), rng.normal(0, 1, ACT), adv / "random.onnx")
    linear(
        "huge",
        np.zeros((OBS, ACT)),
        np.array([1e30, -1e30, -1e30, 1e30, 1e30]),
        adv / "huge.onnx",
        clip=False,
    )
    linear("nan", np.zeros((OBS, ACT)), np.array([np.nan] * ACT), adv / "nan.onnx", clip=False)
    wrong_shape(adv / "wrong_shape.onnx")
    slow(adv / "slow.onnx")
    const_batch2(adv / "const_batch2.onnx")
