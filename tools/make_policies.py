"""Build the reference baseline and the test-set ONNX policies (no torch needed).

    python tools/make_policies.py   # writes baselines/*.onnx and test_policies/*.onnx

The baseline ONNX is a memoryless rule-based twin of NearestFirstBaseline
(nearest insert/release, no clutter handling, no lying regrasp).
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
    """Memoryless ONNX twin of NearestFirstBaseline (batched, rule-based graph).

    Target = nearest present, unseated peg (by estimated peg xy). Phases are read from
    the observation instead of an internal state: goto (above the slot estimate,
    gripper closed) -> down (servo on -rel while descending) -> release (open, lift)
    once the peg is low. The last action in the obs gives the down phase hysteresis.
    """
    sys.path.insert(0, str(ROOT / "referee"))
    from nist_sim.env import BLOCK_H, DXYZ, PEG_FEAT, PEG_LEN, PROPRIO_DIM, MAX_PEGS

    P0 = PROPRIO_DIM + ACT
    nodes, inits = [], []

    def c(name, val, dtype=np.float32):
        inits.append(numpy_helper.from_array(np.asarray(val, dtype), name))
        return name

    def n(op, ins, out, **kw):
        nodes.append(helper.make_node(op, ins, [out], **kw))
        return out

    c("ax1", [1], np.int64)
    # peg table [B, 8, 17]
    n("Slice", ["obs", c("pt_s", [P0], np.int64), c("pt_e", [P0 + MAX_PEGS * PEG_FEAT], np.int64), "ax1"], "pt_flat")
    n("Reshape", ["pt_flat", c("pt_shape", [-1, MAX_PEGS, PEG_FEAT], np.int64)], "pt")
    c("ax2", [2], np.int64)

    def feat(name, a, b):
        return n("Slice", ["pt", c(name + "_s", [a], np.int64), c(name + "_e", [b], np.int64), "ax2"], name)

    def col(name, a):  # obs column -> [B, 1]
        return n("Slice", ["obs", c(name + "_s", [a], np.int64), c(name + "_e", [a + 1], np.int64), "ax1"], name)

    tenth = c("tenth", 0.1)
    feat("pres", 0, 1); feat("seat", 16, 17)
    feat("rel10", 1, 4); n("Mul", ["rel10", tenth], "rel")           # [B,8,3]
    feat("slot10", 13, 15); n("Mul", ["slot10", tenth], "slot")     # [B,8,2]
    feat("qz", 7, 8)                                                # [B,8,1]
    n("Slice", ["obs", c("h_s", [0], np.int64), c("h_e", [3], np.int64), "ax1"], "hand10")
    n("Mul", ["hand10", tenth], "hand")                             # [B,3]
    n("Slice", ["hand", c("hxy_s", [0], np.int64), c("hxy_e", [2], np.int64), "ax1"], "hxy")
    n("Unsqueeze", ["hxy", "ax1"], "hxy3")                          # [B,1,2]
    n("Slice", ["rel", c("rxy_s", [0], np.int64), c("rxy_e", [2], np.int64), "ax2"], "relxy")
    n("Add", ["slot", "relxy"], "pegxy")
    n("Sub", ["pegxy", "hxy3"], "dp"); n("Mul", ["dp", "dp"], "dp2")
    n("ReduceSum", ["dp2", c("ax2b", [2], np.int64)], "dist", keepdims=1)   # [B,8,1]
    half = c("half", 0.5)
    n("Greater", ["pres", half], "is_p"); n("Less", ["seat", half], "not_s")
    n("And", ["is_p", "not_s"], "valid")
    n("Where", ["valid", "dist", c("big", 1e6)], "dmask")
    n("ArgMin", ["dmask"], "idx", axis=1, keepdims=1)               # [B,1,1]
    n("Equal", [c("ar", np.arange(MAX_PEGS).reshape(1, MAX_PEGS, 1), np.int64), "idx"], "eq")
    n("And", ["eq", "valid"], "oh_b"); n("Cast", ["oh_b"], "oh", to=TensorProto.FLOAT)  # [B,8,1]
    n("ReduceSum", ["oh", "ax1"], "anyt3", keepdims=0)              # [B,1]
    for name, src in (("r", "rel"), ("s", "slot"), ("q", "qz")):
        n("Mul", [src, "oh"], name + "_m")
        n("ReduceSum", [name + "_m", "ax1"], name, keepdims=0)      # r [B,3], s [B,2], q [B,1]

    def pick(name, src, a):
        return n("Slice", [src, c(name + "_s", [a], np.int64), c(name + "_e", [a + 1], np.int64), "ax1"], name)

    pick("rx", "r", 0); pick("ry", "r", 1); pick("rz", "r", 2)
    pick("hx", "hand", 0); pick("hy", "hand", 1); pick("hz", "hand", 2)
    n("Sub", ["s", "hxy"], "ds"); pick("dsx", "ds", 0); pick("dsy", "ds", 1)
    n("Abs", ["ds"], "ads"); n("ReduceMax", ["ads"], "ads_m", axes=[1], keepdims=1)
    n("Slice", ["r", c("rr_s", [0], np.int64), c("rr_e", [2], np.int64), "ax1"], "rxy")
    n("Abs", ["rxy"], "arxy"); n("ReduceMax", ["arxy"], "arxy_m", axes=[1], keepdims=1)
    n("GreaterOrEqual", ["ads_m", c("t6", 0.006)], "far_s")
    n("GreaterOrEqual", ["arxy_m", c("t4", 0.004)], "far_r")
    n("And", ["far_s", "far_r"], "far1")
    n("GreaterOrEqual", ["ads_m", c("t15", 0.015)], "far2")
    n("And", ["far1", "far2"], "far_h")
    col("last_z", PROPRIO_DIM + 2)
    n("Less", ["last_z", c("mhalf", -0.5)], "was_down")
    n("Not", ["was_down"], "not_down")
    n("And", ["was_down", "far_h"], "far_a"); n("And", ["not_down", "far1"], "far_b")
    n("Or", ["far_a", "far_b"], "far")
    n("Less", ["rz", c("zrel", PEG_LEN / 2 - BLOCK_H + 0.005)], "low")
    inv = c("inv", 1.0 / DXYZ)
    one = c("one", 1.0); mone = c("mone", -1.0)

    def clip(name, x):
        return n("Clip", [x, mone, one], name)

    n("Mul", ["q", c("k_yaw", -12.5)], "yaw_p"); clip("yaw", "yaw_p")
    n("Mul", ["dsx", inv], "gx_p"); clip("gx", "gx_p")
    n("Mul", ["dsy", inv], "gy_p"); clip("gy", "gy_p")
    n("Sub", [c("z07", 0.07), "hz"], "gz_d"); n("Mul", ["gz_d", inv], "gz_p"); clip("gz", "gz_p")
    n("Mul", ["hx", c("zero_f", 0.0)], "zcol")                      # [B,1] zeros
    n("Add", ["zcol", mone], "mcol"); n("Add", ["zcol", one], "pcol")
    n("Concat", ["gx", "gy", "gz", "yaw", "mcol"], "a_goto", axis=1)
    n("Mul", ["rx", c("minv", -1.0 / DXYZ)], "dx_p"); clip("dx", "dx_p")
    n("Mul", ["ry", "minv"], "dy_p"); clip("dy", "dy_p")
    n("Add", ["zcol", c("m085", -0.85)], "dz")
    n("Concat", ["dx", "dy", "dz", "yaw", "mcol"], "a_down", axis=1)
    n("Add", ["zcol", c("p09", 0.9)], "rz_up")
    n("Concat", ["zcol", "zcol", "rz_up", "zcol", "pcol"], "a_rel", axis=1)
    n("Concat", ["zcol", "zcol", "zcol", "zcol", "pcol"], "a_idle", axis=1)
    n("Where", ["low", "a_rel", "a_down"], "a_dr")
    n("Where", ["far", "a_goto", "a_dr"], "a_t")
    n("Greater", ["anyt3", half], "has_t")
    n("Where", ["has_t", "a_t", "a_idle"], "a_raw")
    clip("action", "a_raw")
    i, o = _io()
    _save(helper.make_graph(nodes, "baseline", [i], [o], inits), path)


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
