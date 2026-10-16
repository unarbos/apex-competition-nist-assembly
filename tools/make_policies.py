"""Build the reference baseline and the test-set ONNX policies (no torch needed).

    python tools/make_policies.py   # writes baselines/*.onnx and test_policies/*.onnx
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OBS, ACT = 26, 5
ROOT = Path(__file__).resolve().parents[1]
OPSET = [helper.make_opsetid("", 17)]


def _io():
    return (helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["B", OBS]),
            helper.make_tensor_value_info("action", TensorProto.FLOAT, ["B", ACT]))


def _save(graph, path: Path):
    m = helper.make_model(graph, opset_imports=OPSET, producer_name="nist_lite")
    m.ir_version = 8
    onnx.checker.check_model(m)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(path))
    print("wrote", path.relative_to(ROOT))


def linear(name: str, W: np.ndarray, b: np.ndarray, path: Path, clip=True):
    i, o = _io()
    inits = [numpy_helper.from_array(W.astype(np.float32), "W"), numpy_helper.from_array(b.astype(np.float32), "b")]
    nodes = [helper.make_node("Gemm", ["obs", "W", "b"], ["pre" if clip else "action"])]
    if clip:
        inits += [numpy_helper.from_array(np.array(-1, np.float32), "lo"), numpy_helper.from_array(np.array(1, np.float32), "hi")]
        nodes.append(helper.make_node("Clip", ["pre", "lo", "hi"], ["action"]))
    _save(helper.make_graph(nodes, name, [i], [o], inits), path)


def baseline(path: Path):
    """Insert-then-release: servo the part over the hole and push down; once the part is seated,
    open the gripper and lift. It never grasps loose parts, so those episodes fail."""
    i, o = _io()
    W = np.zeros((OBS, ACT), np.float32)
    W[9, 0] = W[10, 1] = -20.0  # obs[9:12] = (part - hole) * 10 -> a_xy = -0.8 * err / 4mm
    W[15, 3] = -12.5            # part quat z -> yaw correction
    b_ins = np.array([0, 0, -0.5, 0, -1], np.float32)
    act_seated = np.array([[0, 0, 1, 0, 1]], np.float32)
    inits = [
        numpy_helper.from_array(W, "W"), numpy_helper.from_array(b_ins, "b"),
        numpy_helper.from_array(np.array(-1, np.float32), "lo"), numpy_helper.from_array(np.array(1, np.float32), "hi"),
        numpy_helper.from_array(np.array([11], np.int64), "zidx"),
        numpy_helper.from_array(np.array(0.265, np.float32), "zseat"),  # (25 mm + 1.5 mm) * 10
        numpy_helper.from_array(act_seated, "aseat"),
    ]
    nodes = [
        helper.make_node("Gemm", ["obs", "W", "b"], ["pre"]),
        helper.make_node("Clip", ["pre", "lo", "hi"], ["ains"]),
        helper.make_node("Gather", ["obs", "zidx"], ["z"], axis=1),
        helper.make_node("Less", ["z", "zseat"], ["seated"]),
        helper.make_node("Where", ["seated", "aseat", "ains"], ["action"]),
    ]
    _save(helper.make_graph(nodes, "baseline", [i], [o], inits), path)


def wrong_shape(path: Path):
    i = helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["B", OBS])
    o = helper.make_tensor_value_info("action", TensorProto.FLOAT, ["B", 3])
    W = numpy_helper.from_array(np.zeros((OBS, 3), np.float32), "W")
    b = numpy_helper.from_array(np.zeros(3, np.float32), "b")
    _save(helper.make_graph([helper.make_node("Gemm", ["obs", "W", "b"], ["action"])], "wrong", [i], [o], [W, b]), path)


def slow(path: Path, reps: int = 400):
    """A large chain of matmuls so each call misses the per-call deadline."""
    i, o = _io()
    big = numpy_helper.from_array(np.full((OBS, 2048), 1e-4, np.float32), "U")
    sq = numpy_helper.from_array(np.eye(2048, dtype=np.float32), "S")
    down = numpy_helper.from_array(np.zeros((2048, ACT), np.float32), "D")
    nodes = [helper.make_node("MatMul", ["obs", "U"], ["h0"])]
    for k in range(reps):
        nodes.append(helper.make_node("MatMul", [f"h{k}", "S"], [f"h{k+1}"]))
    nodes.append(helper.make_node("MatMul", [f"h{reps}", "D"], ["action"]))
    _save(helper.make_graph(nodes, "slow", [i], [o], [big, sq, down]), path)


def const_batch2(path: Path):
    """Returns a constant [2, 5] tensor: passes the loader's batch-2 probe, then breaks the
    player on any other batch size (the connection drops mid-response)."""
    i, o = _io()
    c = numpy_helper.from_array(np.zeros((2, ACT), np.float32), "c")
    nodes = [helper.make_node("Shape", ["obs"], ["s"]), helper.make_node("Constant", [], ["action"], value=c)]
    _save(helper.make_graph(nodes, "const_batch2", [i], [o]), path)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    baseline(ROOT / "baselines" / "baseline.onnx")
    adv = ROOT / "test_policies"
    linear("zero", np.zeros((OBS, ACT)), np.zeros(ACT), adv / "zero.onnx")
    linear("constant_push", np.zeros((OBS, ACT)), np.array([0, 0, -1, 0, -1]), adv / "constant_push.onnx")
    linear("random", rng.normal(0, 3, (OBS, ACT)), rng.normal(0, 1, ACT), adv / "random.onnx")
    linear("huge", np.zeros((OBS, ACT)), np.array([1e30, -1e30, -1e30, 1e30, 1e30]), adv / "huge.onnx", clip=False)
    linear("nan", np.zeros((OBS, ACT)), np.array([np.nan] * ACT), adv / "nan.onnx", clip=False)
    wrong_shape(adv / "wrong_shape.onnx")
    slow(adv / "slow.onnx")
    const_batch2(adv / "const_batch2.onnx")
