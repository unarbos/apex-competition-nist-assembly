"""NIST-lite multi-peg board in MuJoCo (CPU).

One episode is a full ATB-1-style peg board (8 slots: round+square × 4/8/12/16 mm)
with K in {3..6} active pegs, 0–3 clutter blocks, and a 2-finger parallel gripper
on a 4-DoF (x, y, z, yaw) Cartesian stage.

Control at 25 Hz; physics at 1 kHz (or 500 Hz if tight fits require it — see
PHYSICS_DT). Partial credit is per peg; the episode ends early when every active
peg is seated and the gripper is clear.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque

import mujoco
import numpy as np

from .conditions import (
    MAX_CLUTTER,
    N_SLOTS,
    SLOT_SPECS,
    EpisodeConditions,
    PegStart,
    slot_world_xy,
)

# ---- observation layout -------------------------------------------------------
# proprio (9) + last_action (5) + 8 pegs × 15 + 3 clutter × 9 = 161
PROPRIO_DIM = 9
ACT_DIM = 5
PEG_FEAT = 15  # present, pos*10 (3), quat (4), shape_oh (2), size*50, slot*10 (3), seated
CLUTTER_FEAT = 9  # present, pos*10 (3), quat (4), size*50
MAX_PEGS = N_SLOTS
OBS_DIM = PROPRIO_DIM + ACT_DIM + MAX_PEGS * PEG_FEAT + MAX_CLUTTER * CLUTTER_FEAT  # 161

CONTROL_HZ = 25
# 1 ms is stable down to ~0.05 mm clearance with the solimp width below.
PHYSICS_DT = 0.001
SUBSTEPS = int(round(1.0 / (CONTROL_HZ * PHYSICS_DT)))  # 40
# Contact width must stay below the minimum radial clearance so a 0.05 mm fit
# cannot be cheated by the soft constraint. 0.02 mm < 0.05 mm.
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
MAX_PART_SPEED = 2.0
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


def _hole_walls(shape: str, r_hole: float, prefix: str) -> str:
    geoms = []
    h2 = BLOCK_H / 2
    if shape == "square":
        outer = r_hole + WALL_T
        for i, (sx, sy, hx, hy) in enumerate([
            (0, r_hole + WALL_T / 2, outer, WALL_T / 2),
            (0, -(r_hole + WALL_T / 2), outer, WALL_T / 2),
            (r_hole + WALL_T / 2, 0, WALL_T / 2, r_hole),
            (-(r_hole + WALL_T / 2), 0, WALL_T / 2, r_hole),
        ]):
            geoms.append(
                f'<geom name="{prefix}_w{i}" type="box" pos="{sx} {sy} {h2}" '
                f'size="{hx} {hy} {h2}" class="fixed"/>'
            )
    else:
        n = ROUND_HOLE_WALLS
        half_w = (r_hole + WALL_T) * math.tan(math.pi / n) + 1e-4
        for i in range(n):
            th = 2 * math.pi * i / n
            c = r_hole + WALL_T / 2
            geoms.append(
                f'<geom name="{prefix}_w{i}" type="box" '
                f'pos="{c * math.cos(th):.6f} {c * math.sin(th):.6f} {h2}" '
                f'euler="0 0 {th:.6f}" size="{WALL_T / 2} {half_w:.6f} {h2}" class="fixed"/>'
            )
    return "\n".join(geoms)


def _board_holes_xml(clearance: float) -> str:
    parts = []
    for i, (shape, dia, lx, ly) in enumerate(SLOT_SPECS):
        r_hole = dia / 2 + clearance
        parts.append(
            f'<body name="hole_{i}" pos="{lx} {ly} 0">\n'
            f"{_hole_walls(shape, r_hole, f'h{i}')}\n</body>"
        )
    return "\n".join(parts)


def build_xml(c: EpisodeConditions) -> str:
    """Build one MuJoCo model: board + 8 holes + K pegs + clutter + gripper."""
    kp = 2000.0 * c.gain_scale
    kv = 2 * math.sqrt(kp * 0.6)
    bx, by = c.board_xy
    byaw = c.board_yaw

    peg_bodies = []
    for i, peg in enumerate(c.pegs):
        shape, dia, _, _ = SLOT_SPECS[peg.slot]
        r = dia / 2
        if shape == "square":
            geom = f'<geom name="peg{i}" type="box" size="{r} {r} {PEG_LEN / 2}" class="part"/>'
        else:
            geom = f'<geom name="peg{i}" type="cylinder" size="{r} {PEG_LEN / 2}" class="part"/>'
        # Per-part friction via geom; mass via inertial.
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
      {_board_holes_xml(c.clearance)}
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
            slot_world_xy(cond.board_xy, cond.board_yaw, peg.slot) for peg in cond.pegs
        ]
        self.r_hole = [
            SLOT_SPECS[peg.slot][1] / 2 + cond.clearance for peg in cond.pegs
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
        return slot_world_xy(self.c.board_xy, self.c.board_yaw, peg.slot)

    def _place_peg(self, i: int, peg: PegStart) -> None:
        hx, hy = self._slot_origin(peg)
        ox, oy = peg.offset_xy
        # Rotate offset into world by board yaw.
        c, s = math.cos(self.c.board_yaw), math.sin(self.c.board_yaw)
        wx = hx + c * ox - s * oy
        wy = hy + s * ox + c * oy
        shape, dia, _, _ = SLOT_SPECS[peg.slot]
        r = dia / 2
        yaw = peg.yaw_err + self.c.board_yaw
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
            r = SLOT_SPECS[peg.slot][1] / 2
            open_frac = min(1.0, (r + 0.0005) / MAX_OPEN)
            finger_z = PEG_LEN / 2 - 0.008
            self._set_hand(
                [ppos[0], ppos[1], ppos[2] + finger_z - PAD_CENTRE_Z],
                peg.yaw_err + c.board_yaw,
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
            # Place in board corners relative to each hole, short radial offset.
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
        for i, peg in enumerate(self.c.pegs):
            if peg.start_type == "grasped":
                continue
            shape, dia, _, _ = SLOT_SPECS[peg.slot]
            r = dia / 2
            row, col = divmod(i, 3)
            wx = bx - 0.06 + 0.06 * col
            wy = by - 0.12 + 0.04 * row
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

    def _noisy_pose(self, pos, quat, noise_rng):
        p = pos + noise_rng.normal(0, self.c.obs_noise, 3)
        # Small angular noise via incremental yaw/pitch/roll.
        dth = noise_rng.normal(0, self.c.obs_noise_rad, 3)
        dq = _yaw_quat(float(dth[2]))
        q = _quat_mul(dq, quat)
        n = float(np.linalg.norm(q))
        if n > 0:
            q = q / n
        return p, q

    def observe(self, noise_rng: np.random.Generator) -> np.ndarray:
        d = self.data
        bx, by = self.c.board_xy
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
        ])

        peg_feats = np.zeros(MAX_PEGS * PEG_FEAT, dtype=np.float64)
        for i, peg in enumerate(self.c.pegs):
            base = i * PEG_FEAT
            ppos, pquat = self.peg_pose(i)
            ppos_n, pquat_n = self._noisy_pose(ppos, pquat, noise_rng)
            hx, hy = self.slot_xy[i]
            # Pose relative to matching slot top centre.
            prel = np.array([ppos_n[0] - hx, ppos_n[1] - hy, ppos_n[2] - BLOCK_H])
            shape, dia, _, _ = SLOT_SPECS[peg.slot]
            feat = np.concatenate([
                [1.0],
                prel * 10,
                pquat_n,
                [1.0 if shape == "round" else 0.0, 1.0 if shape == "square" else 0.0],
                [dia * 50],
                [(hx - bx) * 10, (hy - by) * 10, BLOCK_H * 10],
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
            ppos_n, pquat_n = self._noisy_pose(ppos, pquat, noise_rng)
            prel = np.array([ppos_n[0] - bx, ppos_n[1] - by, ppos_n[2]])
            size = max(cl.size_xy) * 50
            feat = np.concatenate([[1.0], prel * 10, pquat_n, [size]])
            if noise_rng.random() < self.c.dropout_prob and self._last_clutter_obs[i] is not None:
                feat = self._last_clutter_obs[i]
            else:
                self._last_clutter_obs[i] = feat.copy()
            clutter_feats[base : base + CLUTTER_FEAT] = feat

        o = np.concatenate([proprio, self.last_action, peg_feats, clutter_feats])
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
                # Small impulse on a not-yet-seated peg.
                self.data.qvel[self.peg_vadr[i] : self.peg_vadr[i] + 3] += self.rng.normal(0, 0.15, 3)
        elif tgt == "board":
            # Nudge the hand target slightly (board is welded; nudge via part of grasp chain).
            self.target[0] += float(self.rng.uniform(-0.004, 0.004))
            self.target[1] += float(self.rng.uniform(-0.004, 0.004))

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

        self.target[0:3] += applied[0:3] * DXYZ
        self.target[0] = np.clip(self.target[0], -0.25, 0.25)
        self.target[1] = np.clip(self.target[1], -0.25, 0.25)
        self.target[2] = np.clip(self.target[2], HAND_Z_MIN, 0.25)
        self.target[3] = np.clip(self.target[3] + applied[3] * DYAW, -3.1, 3.1)
        self.grip = (applied[4] + 1.0) / 2.0 * MAX_OPEN

        for _ in range(SUBSTEPS):
            self._apply_ctrl()
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
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
