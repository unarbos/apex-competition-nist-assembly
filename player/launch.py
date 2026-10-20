"""nist_lite PLAYER server: serves the miner's ONNX policy over gym_v1.

The platform writes the miner's policy to /app/policy.onnx. The referee calls /act once per
control step with a batch of observations (shape [B, OBS_DIM]); the policy returns [B, 5] actions.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

from gym_v1.player import Player, serve

POLICY_PATH = os.environ.get("NIST_POLICY_PATH", "/app/policy.onnx")
# Must match referee/nist_sim/env.py OBS_DIM / ACT_DIM (v0.2.0 multi-peg board).
OBS_DIM, ACT_DIM = 184, 5
ALLOWED_DOMAINS = {"", "ai.onnx"}
MIN_OPSET, MAX_OPSET = 13, 21


class PolicyRejected(RuntimeError):
    pass


def load_policy(path: str) -> ort.InferenceSession:
    model = onnx.load(path, load_external_data=False)
    for init in model.graph.initializer:
        if init.data_location == onnx.TensorProto.EXTERNAL:
            raise PolicyRejected("policy must be a single self-contained .onnx file")
    for imp in model.opset_import:
        if imp.domain not in ALLOWED_DOMAINS:
            raise PolicyRejected(f"operator domain not allowed: {imp.domain}")
        if imp.domain in ("", "ai.onnx") and not (MIN_OPSET <= imp.version <= MAX_OPSET):
            raise PolicyRejected(f"opset {imp.version} outside [{MIN_OPSET}, {MAX_OPSET}]")
    for node in model.graph.node:
        if node.domain not in ALLOWED_DOMAINS:
            raise PolicyRejected(f"operator domain not allowed: {node.domain}")
    onnx.checker.check_model(model)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    ins, outs = sess.get_inputs(), sess.get_outputs()
    if len(ins) != 1 or len(outs) < 1:
        raise PolicyRejected("policy must have exactly one input and at least one output")
    probe = sess.run(None, {ins[0].name: np.zeros((2, OBS_DIM), np.float32)})[0]
    if probe.shape != (2, ACT_DIM):
        raise PolicyRejected(f"policy output shape {probe.shape} != (batch, {ACT_DIM})")
    return sess


class OnnxPolicyPlayer(Player):
    def __init__(self) -> None:
        self.sess = load_policy(POLICY_PATH)  # failure here = not ready = submission fault
        self.inp = self.sess.get_inputs()[0].name

    def reset(self, match_id: str, player_index: int, seed: int, config: dict[str, Any]) -> None:
        pass

    def act(self, observation: Any, deadline_ms: int) -> Any:  # noqa: ARG002
        obs = np.asarray(observation, dtype=np.float32).reshape(-1, OBS_DIM)
        out = self.sess.run(None, {self.inp: obs})[0]
        return np.asarray(out, dtype=np.float64).reshape(obs.shape[0], ACT_DIM).tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        player = OnnxPolicyPlayer()
    except Exception as e:
        print(f"policy rejected: {e}", file=sys.stderr)
        raise SystemExit(1)
    serve(player, port=args.port, readiness_path="/health")
