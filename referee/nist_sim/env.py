"""NIST-lite multi-peg board in MuJoCo (CPU).

One episode is a peg board with 8 slots whose positions, yaws, cross-section families
(round / rect / n-gon / D-flat live; held-out families on the reserved split only) and
sizes are drawn per episode, with K in {3..6}
active pegs, 0–3 clutter blocks, and a 2-finger parallel gripper on a 4-DoF
(x, y, z, yaw) Cartesian stage.

Observations are intentionally partial: peg and slot pose channels carry
*different* per-episode biases (plus optional mid-episode drift), noise, and
dropout — relative peg−slot does not cancel. Wrist force/torque supports
contact-driven search. The privileged reference solver in tools/ reads true state.

Control at 25 Hz; physics at 1 kHz. Partial credit is per peg.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque

import mujoco
import numpy as np

from .conditions import (
    GLITCH_LEN,
    GLITCH_NOISE_GAIN,
    GLITCH_OFFSET,
    LAYOUT_WALL,
    MAX_CLUTTER,
    N_SLOTS,
    EpisodeConditions,
    PegStart,
    SlotSpec,
    slot_world_xy,
)

# ---- observation layout -------------------------------------------------------
# proprio (16) + last_action (5) + 8 pegs × 17 + 3 clutter × 9 = 184
PROPRIO_DIM = 16  # hand xyz*10 (3), sin/cos yaw (2), vel (3), gap (1), force (3), torque (3), pad_f (1)
ACT_DIM = 5
# present, est_rel_pos*10 (3), quat (4), est_slot_yaw sin/cos (2), outer_size*50, 0,
# size*50, est_slot*10 (3), seated. v0.5: no shape label; slot yaw is an estimate.
PEG_FEAT = 17
CLUTTER_FEAT = 9  # present, pos*10 (3), quat (4), size*50
MAX_PEGS = N_SLOTS
OBS_DIM = PROPRIO_DIM + ACT_DIM + MAX_PEGS * PEG_FEAT + MAX_CLUTTER * CLUTTER_FEAT  # 184

CONTROL_HZ = 25
PHYSICS_DT = 0.001
SUBSTEPS = int(round(1.0 / (CONTROL_HZ * PHYSICS_DT)))  # 40
# Contact width must stay below the minimum radial clearance so soft contacts
# cannot cheat the fit. 0.02 mm << 0.2 mm live lower bound.
SOLIMP = "0.95 0.99 0.00002"

BLOCK_H = 0.020
PEG_LEN = 0.050
WALL_T = 0.006
MAX_OPEN = 0.012
DXYZ = 0.004
DYAW = 0.08
SEAT_TOL_Z = 0.001
SEATED_HOLD = 5
MAX_SEAT_TILT = math.radians(10.0)
# Catches numerical blow-ups only. Soft-contact pops (a dropped 4 mm peg landing across a
# hole lip) can briefly reach ~2.1 m/s in a recoverable way; 2.0 ended those episodes as
# 'unstable' (seed 7 / episode 8 under a no-op). Real blow-ups go far higher or non-finite.
MAX_PART_SPEED = 4.0
HAND_Z_MIN = -0.003
PAD_BOTTOM_OFFSET = 0.004
PAD_CENTRE_Z = 0.012
MAX_SPAWN_ATTEMPTS = 48
SPAWN_MAX_SPEED = 0.25
SPAWN_BOARD_MARGIN = 0.18
SPAWN_MAX_DRIFT = 0.04  # m from intended XY after settle
SPAWN_MAX_PENETRATION = 0.0015

# Round holes use 12 walls (was 16) to cut contact pairs on the full board.
ROUND_HOLE_WALLS = 12
HEX_SIDES = 6
FORCE_SCALE = 0.05  # map Newtons → obs units
TORQUE_SCALE = 0.5


def _peg_shape(peg: PegStart) -> str:
    return peg.shape()


def _poly_walls(poly: list[tuple[float, float]], clearance: float, z0: float, hz: float, prefix: str) -> list[str]:
    """One box wall per edge of a convex polygon, inner face offset outward by ``clearance``."""
    out = []
    n = len(poly)
    for k in range(n):
        x0, y0 = poly[k]
        x1, y1 = poly[(k + 1) % n]
        ex, ey = x1 - x0, y1 - y0
        L = math.hypot(ex, ey)
        if L < 1e-6:
            continue
        nx, ny = ey / L, -ex / L  # outward normal for counter-clockwise order
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if nx * mx + ny * my < 0:
            nx, ny = -nx, -ny
        off = clearance + WALL_T / 2
        cx, cy = mx + nx * off, my + ny * off
        th = math.atan2(ny, nx)
        out.append(
            f'<geom name="{prefix}{k}" type="box" pos="{cx:.6f} {cy:.6f} {z0}" '
            f'euler="0 0 {th:.6f}" size="{WALL_T / 2} {L / 2 + WALL_T * 0.5:.6f} {hz}" class="fixed"/>'
        )
    return out


def _slot_walls(spec: SlotSpec, clearance: float, prefix: str, chamfer: float = 0.0) -> str:
    """Hole walls for one slot (any family); upper lip slightly wider (entry funnel)."""
    poly = spec.polygon()
    if poly is None:
        return _hole_walls("round", spec.r + clearance, prefix, chamfer=chamfer)
    h2 = BLOCK_H / 2
    funnel = 0.00015 + 0.0004 * float(np.clip(chamfer, 0.0, 1.0))
    geoms = _poly_walls(poly, clearance, h2 * 0.7, h2 * 0.7, f"{prefix}_w")
    geoms += _poly_walls(poly, clearance + funnel, BLOCK_H - h2 * 0.35, h2 * 0.35, f"{prefix}_lip")
    return "\n".join(geoms)


def _hole_walls(shape: str, r_hole: float, prefix: str, chamfer: float = 0.0) -> str:
    """Hole wall geoms. ``chamfer`` in [0,1] adds a slightly wider upper lip."""
    geoms = []
    h2 = BLOCK_H / 2
    # Entry funnel: upper third of the hole is wider by up to ~0.4 mm.
    funnel = 0.00015 + 0.0004 * float(np.clip(chamfer, 0.0, 1.0))

    def _square(r: float, z0: float, hz: float, tag: str) -> None:
        outer = r + WALL_T
        for i, (sx, sy, hx, hy) in enumerate([
            (0, r + WALL_T / 2, outer, WALL_T / 2),
            (0, -(r + WALL_T / 2), outer, WALL_T / 2),
            (r + WALL_T / 2, 0, WALL_T / 2, r),
            (-(r + WALL_T / 2), 0, WALL_T / 2, r),
        ]):
            geoms.append(
                f'<geom name="{prefix}_{tag}{i}" type="box" pos="{sx} {sy} {z0}" '
                f'size="{hx} {hy} {hz}" class="fixed"/>'
            )

    def _round(r: float, z0: float, hz: float, tag: str, n: int = ROUND_HOLE_WALLS) -> None:
        half_w = (r + WALL_T) * math.tan(math.pi / n) + 1e-4
        for i in range(n):
            th = 2 * math.pi * i / n
            c = r + WALL_T / 2
            geoms.append(
                f'<geom name="{prefix}_{tag}{i}" type="box" '
                f'pos="{c * math.cos(th):.6f} {c * math.sin(th):.6f} {z0}" '
                f'euler="0 0 {th:.6f}" size="{WALL_T / 2} {half_w:.6f} {hz}" class="fixed"/>'
            )

    def _hex(r: float, z0: float, hz: float, tag: str) -> None:
        # r = apothem (centre to flat).
        half_w = r * math.tan(math.pi / HEX_SIDES) + 1e-4
        for i in range(HEX_SIDES):
            th = 2 * math.pi * i / HEX_SIDES
            c = r + WALL_T / 2
            geoms.append(
                f'<geom name="{prefix}_{tag}{i}" type="box" '
                f'pos="{c * math.cos(th):.6f} {c * math.sin(th):.6f} {z0}" '
                f'euler="0 0 {th:.6f}" size="{WALL_T / 2} {half_w:.6f} {hz}" class="fixed"/>'
            )

    def _dflat(r: float, z0: float, hz: float, tag: str) -> None:
        # Round hole with a flat keyway along +x (D-profile).
        _round(r, z0, hz, tag, n=10)
        # Flat face inset from the +x side.
        flat_x = r * 0.55
        geoms.append(
            f'<geom name="{prefix}_{tag}_flat" type="box" '
            f'pos="{flat_x + WALL_T / 2:.6f} 0 {z0}" '
            f'size="{WALL_T / 2} {r * 0.9:.6f} {hz}" class="fixed"/>'
        )

    z_main = h2 * 0.7
    hz_main = h2 * 0.7
    z_lip = BLOCK_H - h2 * 0.35
    hz_lip = h2 * 0.35
    builders = {
        "square": _square,
        "round": _round,
        "hex": _hex,
        "dflat": _dflat,
    }
    build = builders.get(shape, _round)
    build(r_hole, z_main, hz_main, "w")
    build(r_hole + funnel, z_lip, hz_lip, "lip")
    return "\n".join(geoms)


def _peg_mesh_asset(spec: SlotSpec, name: str) -> str:
    poly = spec.polygon()
    if poly is None or spec.shape == "rect":
        return ""
    half = PEG_LEN / 2
    verts = " ".join(f"{x:.6f} {y:.6f} {z}" for z in (-half, half) for x, y in poly)
    return f'<mesh name="{name}_mesh" vertex="{verts}"/>'


def _spec_peg_geoms(spec: SlotSpec, name: str) -> str:
    half = PEG_LEN / 2
    r = spec.r
    if spec.shape == "round":
        return f'<geom name="{name}" type="cylinder" size="{r} {half}" class="part"/>'
    if spec.shape == "rect":
        ax = r * float(spec.params.get("aspect", 1.0))
        return f'<geom name="{name}" type="box" size="{ax:.6f} {r:.6f} {half}" class="part"/>'
    return f'<geom name="{name}" type="mesh" mesh="{name}_mesh" class="part"/>'


def _peg_geoms(shape: str, r: float, name: str) -> str:
    half = PEG_LEN / 2
    if shape == "square":
        return f'<geom name="{name}" type="box" size="{r} {r} {half}" class="part"/>'
    if shape == "hex":
        # Six thin boxes forming a hexagonal prism (apothem = r).
        parts = []
        half_w = r * math.tan(math.pi / HEX_SIDES)
        for i in range(HEX_SIDES):
            th = 2 * math.pi * i / HEX_SIDES + math.pi / HEX_SIDES
            cx = (r * 0.55) * math.cos(th)
            cy = (r * 0.55) * math.sin(th)
            parts.append(
                f'<geom name="{name}_{i}" type="box" pos="{cx:.6f} {cy:.6f} 0" '
                f'euler="0 0 {th:.6f}" size="{r * 0.55:.6f} {half_w:.6f} {half}" class="part"/>'
            )
        return "\n".join(parts)
    if shape == "dflat":
        # Cylinder plus a flat chord (key) along +x.
        return (
            f'<geom name="{name}" type="cylinder" size="{r} {half}" class="part"/>'
            f'<geom name="{name}_key" type="box" pos="{r * 0.35:.6f} 0 0" '
            f'size="{r * 0.35:.6f} {r * 0.75:.6f} {half}" class="part"/>'
        )
    return f'<geom name="{name}" type="cylinder" size="{r} {half}" class="part"/>'


def _board_holes_xml(slots: list[SlotSpec], clearance: float, chamfer: float) -> str:
    parts = []
    for i, sp in enumerate(slots):
        parts.append(
            f'<body name="hole_{i}" pos="{sp.x:.6f} {sp.y:.6f} 0" euler="0 0 {sp.yaw:.6f}">\n'
            f"{_slot_walls(sp, clearance, f'h{i}', chamfer=chamfer)}\n</body>"
        )
    return "\n".join(parts)


def build_xml(c: EpisodeConditions) -> str:
    """Build one MuJoCo model: board + 8 holes + K pegs + clutter + gripper."""
    kp = 2000.0 * c.gain_scale
    kv = 2 * math.sqrt(kp * 0.6)
    bx, by = c.board_xy
    byaw = c.board_yaw
    meshes = []
    peg_bodies = []
    for i, peg in enumerate(c.pegs):
        r = peg.spec.outer_radius()
        geom = _spec_peg_geoms(peg.spec, f"peg{i}")
        meshes.append(_peg_mesh_asset(peg.spec, f"peg{i}"))
        peg_bodies.append(
            f"""
    <body name="peg{i}" pos="0 0 0.15">
      <freejoint name="peg{i}"/>
      <inertial pos="0 0 0" mass="{c.peg_mass}"
        diaginertia="{c.peg_mass * r * r / 2 + 1e-8} {c.peg_mass * r * r / 2 + 1e-8} {c.peg_mass * r * r / 2 + 1e-8}"/>
      {geom}
    </body>"""
        )

    clutter_bodies = []
    for i, cl in enumerate(c.clutter):
        sx, sy = cl.size_xy
        hz = cl.height / 2
        clutter_bodies.append(
            f"""
    <body name="clutter{i}" pos="0 0 0.15">
      <freejoint name="clutter{i}"/>
      <inertial pos="0 0 0" mass="{max(0.01, c.peg_mass)}"
        diaginertia="1e-6 1e-6 1e-6"/>
      <geom name="clutter{i}" type="box" size="{sx} {sy} {hz}" class="part"
        rgba="0.4 0.55 0.35 1"/>
    </body>"""
        )

    # Contype/conaffinity: board+holes = bit 1, parts = bit 2, gripper = bit 4.
    # Far-apart hole walls still collide with parts (needed for insertion); parts
    # collide with each other and the gripper. Gripper does not collide with itself.
    return f"""
<mujoco model="nist_lite_board">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{PHYSICS_DT}" cone="elliptic" impratio="10" integrator="implicitfast"/>
  <asset>{"".join(meshes)}</asset>
  <default>
    <geom friction="{c.friction} 0.01 0.001" solref="0.002 1" solimp="{SOLIMP}"/>
    <default class="fixed"><geom contype="1" conaffinity="2" rgba="0.5 0.5 0.55 1"/></default>
    <default class="part"><geom condim="4" contype="2" conaffinity="7" rgba="0.8 0.6 0.2 1"/></default>
    <default class="pad"><geom condim="4" contype="4" conaffinity="3"
      rgba="0.2 0.2 0.2 1" friction="{c.friction * 1.5} 0.02 0.001"/></default>
  </default>
  <worldbody>
    <body name="board" pos="{bx} {by} 0" euler="0 0 {byaw}">
      <geom name="board_top" type="box" pos="0 0 -0.005" size="0.22 0.18 0.005" class="fixed"/>
      {_board_holes_xml(c.slots, c.clearance, c.chamfer)}
    </body>
    <body name="hand" pos="0 0 0" gravcomp="1">
      <joint name="jx" type="slide" axis="1 0 0" damping="{kv}" range="-0.28 0.28"/>
      <joint name="jy" type="slide" axis="0 1 0" damping="{kv}" range="-0.28 0.28"/>
      <joint name="jz" type="slide" axis="0 0 1" damping="{kv}" range="{HAND_Z_MIN} 0.3"/>
      <joint name="jyaw" type="hinge" axis="0 0 1" damping="0.5" range="-3.2 3.2"/>
      <inertial pos="0 0 0.03" mass="0.6" diaginertia="0.001 0.001 0.001"/>
      <geom name="palm" type="box" pos="0 0 0.027" size="0.02 0.03 0.006"
        contype="4" conaffinity="3" rgba="0.3 0.3 0.8 1"/>
      <body name="fl" pos="0 0 0" gravcomp="1">
        <joint name="jfl" type="slide" axis="0 1 0" limited="true" range="0 {MAX_OPEN}"
          damping="2" armature="0.001"/>
        <inertial pos="0 0.003 0.012" mass="0.02" diaginertia="1e-6 1e-6 1e-6"/>
        <geom name="padl" type="box" pos="0 0.003 0.012" size="0.008 0.003 0.008" class="pad"/>
      </body>
      <body name="fr" pos="0 0 0" gravcomp="1">
        <joint name="jfr" type="slide" axis="0 -1 0" limited="true" range="0 {MAX_OPEN}"
          damping="2" armature="0.001"/>
        <inertial pos="0 -0.003 0.012" mass="0.02" diaginertia="1e-6 1e-6 1e-6"/>
        <geom name="padr" type="box" pos="0 -0.003 0.012" size="0.008 0.003 0.008" class="pad"/>
      </body>
    </body>
    {"".join(peg_bodies)}
    {"".join(clutter_bodies)}
  </worldbody>
  <contact>
    <exclude body1="hand" body2="fl"/><exclude body1="hand" body2="fr"/><exclude body1="fl" body2="fr"/>
  </contact>
  <actuator>
    <position name="ax" joint="jx" kp="{kp}"/>
    <position name="ay" joint="jy" kp="{kp}"/>
    <position name="az" joint="jz" kp="{kp}"/>
    <position name="ayaw" joint="jyaw" kp="{2.0 * c.gain_scale}"/>
    <position name="afl" joint="jfl" kp="400" forcerange="-15 15" ctrlrange="0 {MAX_OPEN}"/>
    <position name="afr" joint="jfr" kp="400" forcerange="-15 15" ctrlrange="0 {MAX_OPEN}"/>
  </actuator>
</mujoco>
"""


def grasp_hand_yaw(yaw: float) -> float:
    """Hand yaw for a pre-grasped peg, wrapped into [-pi/2, pi/2].

    The 2-finger gripper is symmetric under a half turn, so the same grasp is
    reachable inside the yaw joint range (±3.2 rad) and the ±3.1 command clip
    even when the slot yaw is near ±pi (v0.5.1).
    """
    return float((yaw + math.pi / 2) % math.pi - math.pi / 2)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q1, q2)
    return out


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _lying_quat(yaw: float) -> np.ndarray:
    qx = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])
    return _quat_mul(_yaw_quat(yaw), qx)


class Episode:
    """One multi-peg episode. Steps only through ``step(action)`` at 25 Hz."""

    def __init__(self, cond: EpisodeConditions, rng: np.random.Generator):
        self.c = cond
        self.rng = rng
        self.model = mujoco.MjModel.from_xml_string(build_xml(cond))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.n_pegs = len(cond.pegs)
        self.n_clutter = len(cond.clutter)
        self.peg_qadr = []
        self.peg_vadr = []
        self.peg_body = []
        for i in range(self.n_pegs):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"peg{i}")
            self.peg_qadr.append(m.jnt_qposadr[jid])
            self.peg_vadr.append(m.jnt_dofadr[jid])
            self.peg_body.append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"peg{i}"))
        self.clutter_qadr = []
        self.clutter_vadr = []
        self.clutter_body = []
        for i in range(self.n_clutter):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"clutter{i}")
            self.clutter_qadr.append(m.jnt_qposadr[jid])
            self.clutter_vadr.append(m.jnt_dofadr[jid])
            self.clutter_body.append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"clutter{i}"))
        self.grip_bodies = {
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("hand", "fl", "fr")
        }
        self.slot_xy = [
            slot_world_xy(cond.board_xy, cond.board_yaw, peg.spec) for peg in cond.pegs
        ]
        self.slot_yaw = [float(cond.board_yaw + peg.spec.yaw) for peg in cond.pegs]
        self.r_hole = [
            peg.diameter() / 2 + cond.clearance for peg in cond.pegs
        ]
        self.last_action = np.zeros(ACT_DIM)
        self.steps = 0
        self.seated_run = [0] * self.n_pegs
        self.first_seat_step: list[int | None] = [None] * self.n_pegs
        self.seated_now = [False] * self.n_pegs
        self.done = False
        self.success = False  # True iff all pegs seated at end (diagnostic)
        self.success_step: int | None = None
        self.terminal_reason = "running"
        self.action_l2 = 0.0
        self._act_delay = int(cond.act_delay)
        # FIFO of length act_delay: the command applied this step was issued
        # act_delay control steps ago (zeros until the buffer fills).
        self._act_buf: Deque[np.ndarray] = deque(
            [np.zeros(ACT_DIM) for _ in range(self._act_delay)]
        )
        # Pose dropout: last emitted pose features per peg / clutter.
        self._last_peg_obs = [None] * self.n_pegs
        self._last_clutter_obs = [None] * self.n_clutter
        self._disturbance_applied = False
        # Pose-estimate biases (never revealed; applied in observe). Slot and peg
        # biases differ so relative features retain a residual of several mm.
        # Per-slot / per-clutter / per-peg independent estimate biases (observation hardening).
        if getattr(cond, 'peg_bias_delta_list', None):
            _delta = np.array(cond.peg_bias_delta_list, dtype=float)
        else:
            _delta = np.array(cond.peg_bias_delta_xy, dtype=float)
        if getattr(cond, 'slot_bias_list', None):
            self._slot_bias_xy = np.array(cond.slot_bias_list, dtype=float)
        else:
            self._slot_bias_xy = np.tile(np.array(cond.pose_bias_xy, dtype=float), (max(1, self.n_pegs), 1))
        self._peg_bias_xy = self._slot_bias_xy + _delta
        if getattr(cond, 'slot_bias_yaw_list', None):
            self._slot_bias_yaw = np.array(cond.slot_bias_yaw_list, dtype=float)
        else:
            self._slot_bias_yaw = np.full(max(1, self.n_pegs), float(cond.pose_bias_yaw))
        if getattr(cond, 'clutter_bias_list', None):
            self._clutter_bias_xy = np.array(cond.clutter_bias_list, dtype=float)
        else:
            self._clutter_bias_xy = np.tile(np.array(cond.pose_bias_xy, dtype=float), (max(1, self.n_clutter), 1))
        if getattr(cond, 'peg_yaw_delta_list', None):
            self._slot_yaw_delta = np.array(cond.peg_yaw_delta_list, dtype=float)
        else:
            self._slot_yaw_delta = np.zeros(max(1, self.n_pegs))
        self._pose_bias_yaw = float(cond.pose_bias_yaw)  # retained for compatibility
        self._bias_drift = float(cond.bias_drift)
        self._backlash = float(cond.backlash)
        self._cmd_applied = np.zeros(4)  # last Cartesian cmd that passed backlash
        self._wrist_force = np.zeros(3)
        self._wrist_torque = np.zeros(3)
        self._pad_force = 0.0
        # Estimate glitch bursts (v0.5.1): remaining steps and per-object offsets.
        self._glitch_rate = float(getattr(cond, "glitch_rate", 0.0))
        self._glitch_left = 0
        self._glitch_peg = np.zeros((max(1, self.n_pegs), 2))
        self._glitch_slot = np.zeros((max(1, self.n_pegs), 2))
        self._glitch_clutter = np.zeros((max(1, self.n_clutter), 2))
        self._reset()

    # ---- hand / part helpers -------------------------------------------------
    def _set_hand(self, xyz, yaw, open_frac):
        d = self.data
        d.qpos[0:3] = xyz
        d.qpos[3] = yaw
        d.qpos[4] = d.qpos[5] = open_frac * MAX_OPEN
        self.target = np.array([*xyz, yaw], dtype=float)
        self.grip = open_frac * MAX_OPEN

    def _set_free(self, qadr: int, vadr: int, xyz, quat):
        self.data.qpos[qadr : qadr + 3] = xyz
        self.data.qpos[qadr + 3 : qadr + 7] = quat
        self.data.qvel[vadr : vadr + 6] = 0

    def _apply_ctrl(self):
        d = self.data
        d.ctrl[0:4] = self.target
        d.ctrl[4] = d.ctrl[5] = self.grip

    def peg_pose(self, i: int):
        q = self.peg_qadr[i]
        return self.data.qpos[q : q + 3].copy(), self.data.qpos[q + 3 : q + 7].copy()

    def clutter_pose(self, i: int):
        q = self.clutter_qadr[i]
        return self.data.qpos[q : q + 3].copy(), self.data.qpos[q + 3 : q + 7].copy()

    # ---- reset ---------------------------------------------------------------
    def _slot_origin(self, peg: PegStart) -> tuple[float, float]:
        return slot_world_xy(self.c.board_xy, self.c.board_yaw, peg.spec)

    def _place_peg(self, i: int, peg: PegStart) -> None:
        hx, hy = self._slot_origin(peg)
        ox, oy = peg.offset_xy
        # Rotate offset into world by board yaw.
        c, s = math.cos(self.c.board_yaw), math.sin(self.c.board_yaw)
        wx = hx + c * ox - s * oy
        wy = hy + s * ox + c * oy
        r = peg.spec.outer_radius()
        yaw = peg.yaw_err + self.c.board_yaw + peg.spec.yaw
        if peg.start_type == "lying":
            self._set_free(self.peg_qadr[i], self.peg_vadr[i], [wx, wy, r + 1e-4], _lying_quat(yaw))
        elif peg.start_type == "grasped":
            pz = BLOCK_H + peg.height + PEG_LEN / 2
            self._set_free(self.peg_qadr[i], self.peg_vadr[i], [wx, wy, pz], _yaw_quat(yaw))
        else:
            # loose / tray / near_hole: upright on the board plane, clear of the hole void.
            self._set_free(
                self.peg_qadr[i], self.peg_vadr[i], [wx, wy, PEG_LEN / 2 + 1e-4], _yaw_quat(yaw)
            )

    def _place_clutter(self, i: int) -> None:
        cl = self.c.clutter[i]
        bx, by = self.c.board_xy
        c, s = math.cos(self.c.board_yaw), math.sin(self.c.board_yaw)
        ox, oy = cl.offset_xy
        wx = bx + c * ox - s * oy
        wy = by + s * ox + c * oy
        hz = cl.height / 2 + 1e-4
        self._set_free(
            self.clutter_qadr[i],
            self.clutter_vadr[i],
            [wx, wy, hz],
            _yaw_quat(cl.yaw + self.c.board_yaw),
        )

    def _place_and_settle(self) -> None:
        c = self.c
        grasped = [i for i, p in enumerate(c.pegs) if p.start_type == "grasped"]
        for i, peg in enumerate(c.pegs):
            self._place_peg(i, peg)
        for i in range(self.n_clutter):
            self._place_clutter(i)

        if grasped:
            i = grasped[0]
            peg = c.pegs[i]
            ppos, _ = self.peg_pose(i)
            r = peg.spec.half_width_y()
            open_frac = min(1.0, (r + 0.0005) / MAX_OPEN)
            finger_z = PEG_LEN / 2 - 0.008
            self._set_hand(
                [ppos[0], ppos[1], ppos[2] + finger_z - PAD_CENTRE_Z],
                grasp_hand_yaw(peg.yaw_err + c.board_yaw + peg.spec.yaw),
                open_frac,
            )
            settle_grip = 0.0
        else:
            # Park the hand above board centre.
            bx, by = c.board_xy
            self._set_hand([bx, by, 0.12], c.board_yaw, 1.0)
            settle_grip = MAX_OPEN

        mujoco.mj_forward(self.model, self.data)
        self.grip = settle_grip
        hold = set(grasped)
        for _ in range(150):
            self._apply_ctrl()
            for i in hold:
                q = self.peg_qadr[i]
                ppos, _ = self.peg_pose(i)
                self.data.qvel[self.peg_vadr[i] : self.peg_vadr[i] + 6] = 0
                # Keep XY/Z of the grasped peg fixed while fingers close.
                self.data.qpos[q : q + 3] = ppos
            mujoco.mj_step(self.model, self.data)
        self.grip = settle_grip if grasped else MAX_OPEN
        for _ in range(80):
            self._apply_ctrl()
            mujoco.mj_step(self.model, self.data)
        # Kill residual part velocities so a settled start cannot walk off the board.
        for i in range(self.n_pegs):
            self.data.qvel[self.peg_vadr[i] : self.peg_vadr[i] + 6] = 0
        for i in range(self.n_clutter):
            self.data.qvel[self.clutter_vadr[i] : self.clutter_vadr[i] + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def _intended_peg_xy(self, peg: PegStart) -> tuple[float, float]:
        hx, hy = self._slot_origin(peg)
        ox, oy = peg.offset_xy
        c, s = math.cos(self.c.board_yaw), math.sin(self.c.board_yaw)
        return hx + c * ox - s * oy, hy + s * ox + c * oy

    def _spawn_ok(self) -> bool:
        d = self.data
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            return False
        for i, peg in enumerate(self.c.pegs):
            ppos, pquat = self.peg_pose(i)
            pv = d.qvel[self.peg_vadr[i] : self.peg_vadr[i] + 3]
            if float(np.linalg.norm(pv)) > SPAWN_MAX_SPEED:
                return False
            if abs(ppos[0]) > SPAWN_BOARD_MARGIN or abs(ppos[1]) > SPAWN_BOARD_MARGIN or ppos[2] < -0.005:
                return False
            ix, iy = self._intended_peg_xy(peg)
            if float(np.hypot(ppos[0] - ix, ppos[1] - iy)) > SPAWN_MAX_DRIFT:
                return False
            # Non-lying starts must stay roughly upright after settle.
            if peg.start_type != "lying":
                w, x, y, z = pquat
                if abs(1.0 - 2.0 * (x * x + y * y)) < 0.85:
                    return False
        for i, cl in enumerate(self.c.clutter):
            ppos, _ = self.clutter_pose(i)
            if abs(ppos[0]) > SPAWN_BOARD_MARGIN or abs(ppos[1]) > SPAWN_BOARD_MARGIN or ppos[2] < -0.005:
                return False
            bx, by = self.c.board_xy
            c, s = math.cos(self.c.board_yaw), math.sin(self.c.board_yaw)
            ox, oy = cl.offset_xy
            ix, iy = bx + c * ox - s * oy, by + s * ox + c * oy
            if float(np.hypot(ppos[0] - ix, ppos[1] - iy)) > SPAWN_MAX_DRIFT:
                return False
        part_bodies = set(self.peg_body) | set(self.clutter_body)
        for i in range(d.ncon):
            con = d.contact[i]
            b1 = self.model.geom_bodyid[con.geom1]
            b2 = self.model.geom_bodyid[con.geom2]
            if (b1 in part_bodies or b2 in part_bodies) and con.dist < -SPAWN_MAX_PENETRATION:
                # Allow light peg-peg / clutter contacts in tray clusters.
                if b1 in part_bodies and b2 in part_bodies and con.dist > -0.003:
                    continue
                return False
        return True

    def _nudge_offsets(self) -> None:
        """Re-jitter free peg / clutter offsets within the same stratum identity."""
        rng = self.rng
        for peg in self.c.pegs:
            if peg.start_type == "grasped":
                continue
            peg.offset_xy = (
                peg.offset_xy[0] + float(rng.uniform(-0.01, 0.01)),
                peg.offset_xy[1] + float(rng.uniform(-0.01, 0.01)),
            )
            peg.yaw_err += float(rng.uniform(-0.2, 0.2))
        for cl in self.c.clutter:
            cl.offset_xy = (
                cl.offset_xy[0] + float(rng.uniform(-0.015, 0.015)),
                cl.offset_xy[1] + float(rng.uniform(-0.015, 0.015)),
            )

    def _force_safe_start(self) -> None:
        """Spread parts away from holes so settle cannot interlock."""
        for i, peg in enumerate(self.c.pegs):
            if peg.start_type == "grasped":
                continue
            ang = 2 * math.pi * i / max(1, self.n_pegs)
            # Short radial offset from the peg's own hole; v0.5.1 rotates the spot
            # until it clears every slot (a free peg must not start in another hole).
            for k in range(24):
                a = ang + k * (2 * math.pi / 24)
                off = (0.055 * math.cos(a), 0.055 * math.sin(a))
                if self._board_spot_clear(peg.spec.x + off[0], peg.spec.y + off[1], peg.spec.outer_radius()):
                    ang = a
                    break
            peg.offset_xy = (0.055 * math.cos(ang), 0.055 * math.sin(ang))
            if peg.start_type == "lying":
                peg.yaw_err = ang + math.pi / 2  # long axis tangential to radial
            else:
                peg.start_type = "loose"
                peg.yaw_err = 0.0
        for i, cl in enumerate(self.c.clutter):
            cl.offset_xy = (0.10, -0.10 + 0.035 * i)

    def _teleport_safe_grid(self) -> None:
        """Last-resort: snap free parts onto a clear grid on the board and freeze."""
        bx, by = self.c.board_xy
        cells = self._clear_grid_cells()
        for i, peg in enumerate(self.c.pegs):
            if peg.start_type == "grasped":
                continue
            r = peg.spec.outer_radius()
            wx, wy = cells[i % len(cells)]
            if peg.start_type == "lying":
                self._set_free(
                    self.peg_qadr[i],
                    self.peg_vadr[i],
                    [wx, wy, r + 1e-4],
                    _lying_quat(math.pi / 2),
                )
            else:
                peg.start_type = "loose"
                self._set_free(
                    self.peg_qadr[i],
                    self.peg_vadr[i],
                    [wx, wy, PEG_LEN / 2 + 1e-4],
                    _yaw_quat(0.0),
                )
        for i, cl in enumerate(self.c.clutter):
            wx = bx + 0.10
            wy = by - 0.10 + 0.04 * i
            self._set_free(
                self.clutter_qadr[i],
                self.clutter_vadr[i],
                [wx, wy, cl.height / 2 + 1e-4],
                _yaw_quat(0.0),
            )
        if not any(p.start_type == "grasped" for p in self.c.pegs):
            self._set_hand([bx, by, 0.12], self.c.board_yaw, 1.0)
            self.grip = MAX_OPEN
        mujoco.mj_forward(self.model, self.data)
        self._apply_ctrl()
        for _ in range(30):
            self._apply_ctrl()
            mujoco.mj_step(self.model, self.data)
        for i in range(self.n_pegs):
            self.data.qvel[self.peg_vadr[i] : self.peg_vadr[i] + 6] = 0
        for i in range(self.n_clutter):
            self.data.qvel[self.clutter_vadr[i] : self.clutter_vadr[i] + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def _board_spot_clear(self, lx: float, ly: float, peg_outer: float, margin: float = 0.002) -> bool:
        """Board-local spot clears every slot's outer wall (v0.5.1 spawn check)."""
        for sp in self.c.slots:
            need = sp.outer_radius() + 0.0012 + LAYOUT_WALL + peg_outer + margin
            if math.hypot(lx - sp.x, ly - sp.y) < need:
                return False
        return True

    def _clear_grid_cells(self) -> list[tuple[float, float]]:
        """World-frame teleport cells that clear every slot (fallback: the legacy grid)."""
        bx, by = self.c.board_xy
        c, s = math.cos(self.c.board_yaw), math.sin(self.c.board_yaw)
        legacy = [(bx - 0.06 + 0.06 * col, by - 0.12 + 0.04 * row) for row in range(2) for col in range(3)]
        out = []
        po = max((p.spec.outer_radius() for p in self.c.pegs), default=0.008)
        for gy in np.arange(-0.13, 0.131, 0.026):
            for gx in np.arange(-0.15, 0.151, 0.03):
                wx, wy = bx + gx, by + gy
                # world -> board-local
                lx = c * (wx - bx) + s * (wy - by)
                ly = -s * (wx - bx) + c * (wy - by)
                if self._board_spot_clear(lx, ly, po, 0.003) and all(
                    math.hypot(wx - ox, wy - oy) >= 0.026 for ox, oy in out
                ):
                    out.append((float(wx), float(wy)))
        return out if len(out) >= self.n_pegs else legacy

    def _reset(self) -> None:
        for attempt in range(MAX_SPAWN_ATTEMPTS):
            if attempt > 0:
                self._nudge_offsets()
            self._place_and_settle()
            if self._spawn_ok():
                return
        self._force_safe_start()
        self._place_and_settle()
        if not self._spawn_ok():
            self._teleport_safe_grid()

    # ---- observation ---------------------------------------------------------
    def _upright(self, quat) -> bool:
        w, x, y, z = quat
        return abs(1.0 - 2.0 * (x * x + y * y)) >= math.cos(MAX_SEAT_TILT)

    def _gripper_touching(self, body_id: int) -> bool:
        m, d = self.model, self.data
        for i in range(d.ncon):
            con = d.contact[i]
            b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
            if body_id in (b1, b2) and (b1 in self.grip_bodies or b2 in self.grip_bodies):
                return True
        return False

    def _peg_seated(self, i: int) -> bool:
        ppos, pquat = self.peg_pose(i)
        hx, hy = self.slot_xy[i]
        bottom = ppos[2] - PEG_LEN / 2
        xy_err = float(np.hypot(ppos[0] - hx, ppos[1] - hy))
        pad_bottom = self.data.qpos[2] + PAD_BOTTOM_OFFSET
        clear = pad_bottom >= ppos[2] + PEG_LEN / 2 + 0.002 and not self._gripper_touching(
            self.peg_body[i]
        )
        return (
            bottom <= SEAT_TOL_Z
            and xy_err <= self.r_hole[i]
            and clear
            and self._upright(pquat)
        )

    def _glitch_step(self, noise_rng: np.random.Generator) -> float:
        """Advance the estimate-glitch burst state; returns the noise gain for this step.

        A burst adds extra pose noise and an independent transient offset to every
        peg / slot / clutter estimate for a few control steps (v0.5.1).
        """
        if self._glitch_rate <= 0.0:
            return 1.0
        if self._glitch_left > 0:
            self._glitch_left -= 1
            if self._glitch_left == 0:
                self._glitch_peg[:] = 0.0
                self._glitch_slot[:] = 0.0
                self._glitch_clutter[:] = 0.0
                return 1.0
            return GLITCH_NOISE_GAIN
        if noise_rng.random() >= self._glitch_rate:
            return 1.0
        self._glitch_left = int(noise_rng.integers(GLITCH_LEN[0], GLITCH_LEN[1] + 1))

        def _offs(n: int) -> np.ndarray:
            mag = noise_rng.uniform(GLITCH_OFFSET[0], GLITCH_OFFSET[1], n)
            ang = noise_rng.uniform(0.0, 2 * math.pi, n)
            return np.stack([mag * np.cos(ang), mag * np.sin(ang)], axis=1)

        self._glitch_peg = _offs(self._glitch_peg.shape[0])
        self._glitch_slot = _offs(self._glitch_slot.shape[0])
        self._glitch_clutter = _offs(self._glitch_clutter.shape[0])
        return GLITCH_NOISE_GAIN

    def _noisy_pose(
        self, pos, quat, noise_rng, bias_xy: np.ndarray | None = None, yaw_bias: float = 0.0, gain: float = 1.0
    ):
        p = np.asarray(pos, dtype=float).copy()
        if bias_xy is not None:
            p[0] += float(bias_xy[0])
            p[1] += float(bias_xy[1])
        p = p + noise_rng.normal(0, self.c.obs_noise * gain, 3)
        dth = noise_rng.normal(0, self.c.obs_noise_rad * gain, 3)
        yaw_extra = float(dth[2]) + (float(yaw_bias) if bias_xy is not None else 0.0)
        dq = _yaw_quat(yaw_extra)
        q = _quat_mul(dq, quat)
        n = float(np.linalg.norm(q))
        if n > 0:
            q = q / n
        return p, q

    def _drift_biases(self) -> None:
        """Non-stationary estimate bias: random-walk so EMA−rel scripts go stale."""
        if self._bias_drift <= 0.0:
            return
        self._slot_bias_xy = self._slot_bias_xy + self.rng.normal(0.0, self._bias_drift, self._slot_bias_xy.shape)
        # Peg bias walks independently (distractor: residual keeps moving).
        self._peg_bias_xy = self._peg_bias_xy + self.rng.normal(0.0, self._bias_drift, self._peg_bias_xy.shape)

    def _update_wrist_wrench(self) -> None:
        """Estimate wrist force/torque from contacts on hand / pad geoms."""
        m, d = self.model, self.data
        force = np.zeros(3)
        torque = np.zeros(3)
        pad_f = 0.0
        hand_pos = d.qpos[0:3]
        for i in range(d.ncon):
            con = d.contact[i]
            b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
            if not (b1 in self.grip_bodies or b2 in self.grip_bodies):
                continue
            cforce = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, cforce)
            # Contact frame force (normal along x of contact frame).
            fn = float(cforce[0])
            # Approximate world-frame push along contact normal.
            frame = con.frame.reshape(3, 3)
            f_world = frame[:, 0] * fn
            force += f_world
            r = con.pos - hand_pos
            torque += np.cross(r, f_world)
            if b1 in self.grip_bodies and b2 in self.grip_bodies:
                continue
            pad_f += abs(fn)
        # Light EMA so the signal is usable at 25 Hz.
        self._wrist_force = 0.7 * self._wrist_force + 0.3 * force
        self._wrist_torque = 0.7 * self._wrist_torque + 0.3 * torque
        self._pad_force = 0.7 * self._pad_force + 0.3 * pad_f

    def observe(self, noise_rng: np.random.Generator) -> np.ndarray:
        """Partial observation: differently-biased peg/slot estimates + wrench.

        True hole/peg poses are never revealed. Slot and peg estimates use
        independent biases (plus optional drift), so relative peg−slot retains
        a several-mm residual that contact search must resolve.
        """
        d = self.data
        bx, by = self.c.board_xy
        glitch_gain = self._glitch_step(noise_rng)
        hand = d.qpos[0:3].copy()
        hand[0] -= bx
        hand[1] -= by
        yaw = d.qpos[3]
        hand_v = d.qvel[0:3]
        gap = d.qpos[4] + d.qpos[5]
        proprio = np.concatenate([
            hand * 10,
            [math.sin(yaw), math.cos(yaw)],
            hand_v,
            [gap * 50],
            self._wrist_force * FORCE_SCALE,
            self._wrist_torque * TORQUE_SCALE,
            [self._pad_force * FORCE_SCALE],
        ])

        peg_feats = np.zeros(MAX_PEGS * PEG_FEAT, dtype=np.float64)
        for i, peg in enumerate(self.c.pegs):
            base = i * PEG_FEAT
            ppos, pquat = self.peg_pose(i)
            ppos_n, pquat_n = self._noisy_pose(
                ppos, pquat, noise_rng, bias_xy=self._peg_bias_xy[i] + self._glitch_peg[i],
                yaw_bias=float(self._slot_bias_yaw[i]), gain=glitch_gain,
            )
            hx, hy = self.slot_xy[i]
            # Slot estimate uses slot bias (≠ peg bias) — no exact hole coordinates.
            sn = self.c.obs_noise * glitch_gain
            hx_e = hx + self._slot_bias_xy[i][0] + self._glitch_slot[i][0] + float(noise_rng.normal(0, sn))
            hy_e = hy + self._slot_bias_xy[i][1] + self._glitch_slot[i][1] + float(noise_rng.normal(0, sn))
            prel = np.array([ppos_n[0] - hx_e, ppos_n[1] - hy_e, ppos_n[2] - BLOCK_H])
            # Slot yaw estimate: slot bias plus a per-peg residual vs the peg yaw channel.
            syaw = (
                self.slot_yaw[i]
                + float(self._slot_bias_yaw[i])
                + float(self._slot_yaw_delta[i])
                + float(noise_rng.normal(0, self.c.obs_noise_rad))
            )
            feat = np.concatenate([
                [1.0],
                prel * 10,
                pquat_n,
                [math.sin(syaw), math.cos(syaw), 2 * peg.spec.outer_radius() * 50, 0.0],
                [peg.diameter() * 50],
                [(hx_e - bx) * 10, (hy_e - by) * 10, BLOCK_H * 10],
                [1.0 if self.seated_now[i] else 0.0],
            ])
            if noise_rng.random() < self.c.dropout_prob and self._last_peg_obs[i] is not None:
                feat = self._last_peg_obs[i]
            else:
                self._last_peg_obs[i] = feat.copy()
            peg_feats[base : base + PEG_FEAT] = feat

        clutter_feats = np.zeros(MAX_CLUTTER * CLUTTER_FEAT, dtype=np.float64)
        for i, cl in enumerate(self.c.clutter):
            base = i * CLUTTER_FEAT
            ppos, pquat = self.clutter_pose(i)
            ppos_n, pquat_n = self._noisy_pose(
                ppos, pquat, noise_rng, bias_xy=self._clutter_bias_xy[i] + self._glitch_clutter[i], yaw_bias=0.0,
                gain=glitch_gain,
            )
            prel = np.array([ppos_n[0] - bx, ppos_n[1] - by, ppos_n[2]])
            size = max(cl.size_xy) * 50
            feat = np.concatenate([[1.0], prel * 10, pquat_n, [size]])
            if noise_rng.random() < self.c.dropout_prob and self._last_clutter_obs[i] is not None:
                feat = self._last_clutter_obs[i]
            else:
                self._last_clutter_obs[i] = feat.copy()
            clutter_feats[base : base + CLUTTER_FEAT] = feat

        o = np.concatenate([proprio, self.last_action, peg_feats, clutter_feats])
        assert o.shape[0] == OBS_DIM, (o.shape[0], OBS_DIM)
        return o.astype(np.float32)

    # ---- step ----------------------------------------------------------------
    def fail(self, reason: str) -> None:
        self.done = True
        self.success = False
        self.terminal_reason = reason

    def _apply_disturbance(self) -> None:
        if self._disturbance_applied or self.c.disturbance <= 0:
            return
        if self.steps != self.c.disturbance_step:
            return
        self._disturbance_applied = True
        tgt = self.c.disturbance_target
        if tgt.startswith("peg:"):
            i = int(tgt.split(":")[1])
            if 0 <= i < self.n_pegs and not self.seated_now[i]:
                self.data.qvel[self.peg_vadr[i] : self.peg_vadr[i] + 3] += self.rng.normal(0, 0.18, 3)
        elif tgt == "drop":
            # Slip: open gripper and nudge any held peg downward.
            self.grip = MAX_OPEN
            self.target[2] = min(0.25, self.target[2] + 0.01)
            for i in range(self.n_pegs):
                if self.seated_now[i]:
                    continue
                if self._gripper_touching(self.peg_body[i]):
                    self.data.qvel[self.peg_vadr[i] + 2] -= 0.25
                    self.data.qvel[self.peg_vadr[i] : self.peg_vadr[i] + 2] += self.rng.normal(0, 0.05, 2)
        elif tgt == "board":
            self.target[0] += float(self.rng.uniform(-0.005, 0.005))
            self.target[1] += float(self.rng.uniform(-0.005, 0.005))

    def step(self, action: np.ndarray) -> None:
        if self.done:
            return
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.action_l2 += float(np.sum(a * a))
        self.last_action = a.copy()

        if self._act_delay > 0:
            self._act_buf.append(a.copy())
            applied = self._act_buf.popleft()  # issued act_delay steps ago
        else:
            applied = a

        # Backlash / deadzone on Cartesian commands: small desired moves are eaten.
        delta = applied[0:3] * DXYZ
        for i in range(3):
            if abs(delta[i]) < self._backlash:
                delta[i] = 0.0
            else:
                delta[i] -= math.copysign(self._backlash, delta[i])
        self.target[0:3] += delta
        self.target[0] = np.clip(self.target[0], -0.25, 0.25)
        self.target[1] = np.clip(self.target[1], -0.25, 0.25)
        self.target[2] = np.clip(self.target[2], HAND_Z_MIN, 0.25)
        dyaw = applied[3] * DYAW
        if abs(dyaw) < self._backlash * 20:  # ~yaw deadzone scaled
            dyaw = 0.0
        self.target[3] = np.clip(self.target[3] + dyaw, -3.1, 3.1)
        self.grip = (applied[4] + 1.0) / 2.0 * MAX_OPEN

        for _ in range(SUBSTEPS):
            self._apply_ctrl()
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        self._drift_biases()
        self._update_wrist_wrench()
        self._apply_disturbance()

        d = self.data
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            return self.fail("unstable")

        for i in range(self.n_pegs):
            ppos, _ = self.peg_pose(i)
            pv = d.qvel[self.peg_vadr[i] : self.peg_vadr[i] + 3]
            if float(np.linalg.norm(pv)) > MAX_PART_SPEED:
                return self.fail("unstable")
            if abs(ppos[0]) > 0.30 or abs(ppos[1]) > 0.30 or ppos[2] < -0.01:
                return self.fail("part_lost")

        for i in range(self.n_pegs):
            seated = self._peg_seated(i)
            self.seated_run[i] = self.seated_run[i] + 1 if seated else 0
            if self.seated_run[i] >= SEATED_HOLD:
                if self.first_seat_step[i] is None:
                    self.first_seat_step[i] = self.steps
                self.seated_now[i] = True
            else:
                # Knocked out after a prior seat → no longer seated at end.
                self.seated_now[i] = False

        if all(self.seated_now) and self.n_pegs > 0:
            # Gripper already clear by seated predicate; end early.
            self.done = True
            self.success = True
            self.success_step = self.steps
            self.terminal_reason = "assembled"
            return

        if self.steps >= self.c.max_steps:
            self.done = True
            self.success = all(self.seated_now) and self.n_pegs > 0
            self.terminal_reason = "assembled" if self.success else "time_limit"
            if self.success:
                self.success_step = self.steps

    def peg_seated_at_end(self) -> list[bool]:
        return [bool(x) for x in self.seated_now]

    def peg_first_seat_steps(self) -> list[int | None]:
        # Only credit first-seat time if still seated at end.
        out: list[int | None] = []
        for i in range(self.n_pegs):
            if self.seated_now[i]:
                out.append(self.first_seat_step[i])
            else:
                out.append(None)
        return out
