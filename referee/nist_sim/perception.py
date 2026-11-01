"""v0.6-dev wrist depth patch (profile ``v06``; not used by the live v05 task).

A downward-looking pinhole "depth camera" mounted on the hand: ``n x n`` rays from one point,
cast with a single ``mujoco.mj_multiRay`` call, rotated with the hand yaw. Returns ray distances
(m) clipped to ``cutoff``; misses read as ``cutoff``. Gripper fingers are visible to the camera
(they occlude part of the view, as on a real wrist camera); the hand body itself is excluded.

Noise / dropout / quantization are applied by :func:`corrupt`, driven by the episode noise RNG.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

DEPTH_N = 16
DEPTH_FOV = math.radians(70.0)
DEPTH_CUTOFF = 0.30
CAM_OFFSET = np.array([0.0, 0.0, 0.035])  # above the hand origin, looking down


def ray_dirs(n: int = DEPTH_N, fov: float = DEPTH_FOV) -> np.ndarray:
    """Unit ray directions in the hand frame, row-major ``n*n x 3`` (camera looks down -z)."""
    half = math.tan(fov / 2)
    u = np.linspace(-half, half, n)
    xx, yy = np.meshgrid(u, u, indexing="ij")
    v = np.stack([xx, yy, -np.ones_like(xx)], axis=-1).reshape(-1, 3)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class WristDepth:
    def __init__(self, model: mujoco.MjModel, n: int = DEPTH_N, fov: float = DEPTH_FOV,
                 cutoff: float = DEPTH_CUTOFF):
        self.m = model
        self.n = n
        self.cutoff = cutoff
        self.local = ray_dirs(n, fov)
        self.hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.geomid = np.zeros(n * n, dtype=np.int32)
        self.dist = np.zeros(n * n, dtype=np.float64)

    def __call__(self, data: mujoco.MjData, hand_xyz: np.ndarray, hand_yaw: float) -> np.ndarray:
        c, s = math.cos(hand_yaw), math.sin(hand_yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        vec = (self.local @ R.T).reshape(-1)
        pnt = np.asarray(hand_xyz, dtype=np.float64) + CAM_OFFSET
        mujoco.mj_multiRay(self.m, data, pnt, vec, None, 1, self.hand, self.geomid, self.dist, None,
                           self.n * self.n, self.cutoff)
        d = np.where(self.dist < 0, self.cutoff, self.dist)
        return np.minimum(d, self.cutoff).reshape(self.n, self.n)


def corrupt(depth: np.ndarray, rng: np.random.Generator, sigma: float = 0.0015,
            dropout: float = 0.05, quant: float = 0.0005, cutoff: float = DEPTH_CUTOFF) -> np.ndarray:
    """Gaussian range noise, per-pixel dropout (reads as ``cutoff``) and range quantization."""
    d = depth + rng.normal(0.0, sigma, depth.shape)
    if quant > 0:
        d = np.round(d / quant) * quant
    d = np.where(rng.random(depth.shape) < dropout, cutoff, d)
    return np.clip(d, 0.0, cutoff)
