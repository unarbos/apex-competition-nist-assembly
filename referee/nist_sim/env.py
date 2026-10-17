"""NIST-lite peg insertion in MuJoCo (CPU).

Each episode is one assembly pair from the NIST Assembly Task Board #1 peg family
(round / square pegs, 4-16 mm), built procedurally from primitives, plus a
2-finger parallel gripper on a 4-DoF (x, y, z, yaw) Cartesian stage.

Control at 25 Hz; physics at 1 kHz. One success flag per episode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import mujoco
import numpy as np

OBS_DIM = 26
ACT_DIM = 5
CONTROL_HZ = 25
PHYSICS_DT = 0.001
SUBSTEPS = int(round(1.0 / (CONTROL_HZ * PHYSICS_DT)))  # 40

BLOCK_H = 0.020  # hole depth / block height (m)
PEG_LEN = 0.050
WALL_T = 0.006
MAX_OPEN = 0.012  # per-finger opening (m)
DXYZ = 0.004  # m per control step at |a|=1
DYAW = 0.08  # rad per control step at |a|=1
SEAT_TOL_Z = 0.001  # within 1 mm of fully seated
SEATED_HOLD = 5  # consecutive control steps
# Upright tolerance for "seated". A peg fully inside the 20 mm hole cannot tilt more than
# ~3.4 deg at the widest clearance (atan(2 * 0.6 / 20)); a peg lying across the mouth is at 90.
MAX_SEAT_TILT = math.radians(10.0)
MAX_PART_SPEED = 2.0  # m/s
# Hand z floor. Pads are centred 12 mm above the hand with an 8 mm half-height, so their
# bottom is at hand_z + PAD_BOTTOM_OFFSET. A lying 4 mm peg tops out at ~4 mm; allowing the
# hand slightly below the board plane gives the pads ~3 mm of vertical overlap without the
# pads (bottom at +1 mm) or palm (bottom at +21 mm) penetrating the board.
HAND_Z_MIN = -0.003
PAD_BOTTOM_OFFSET = 0.004  # pad centre z (0.012) minus pad half-height (0.008)
PAD_CENTRE_Z = 0.012
MAX_SPAWN_ATTEMPTS = 48
# After the settle phase a valid spawn is nearly still. Well below MAX_PART_SPEED so
# blow-ups are caught before the first control step.
SPAWN_MAX_SPEED = 0.2  # m/s
SPAWN_MAX_DRIFT = 0.05  # m from the intended hole+offset XY after settle
SPAWN_BOARD_MARGIN = 0.22  # m; leave headroom vs the 0.28 part_lost limit
SPAWN_MAX_PENETRATION = 0.001  # m; deeper part contacts mean the start is interlocking


@dataclass
class EpisodeConditions:
    shape: str  # "round" | "square"
    diameter: float  # m (peg nominal width)
    clearance: float  # m (radial)
    friction: float
    peg_mass: float
    gain_scale: float
    obs_noise: float  # m, std on part position observations
    reset_type: str  # grasped_above | grasped_far | near_hole | loose | lying
    hole_xy: tuple[float, float]
    offset_xy: tuple[float, float]
    height: float
    yaw_err: float
    max_steps: int = 250

    def public(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}


def _hole_walls(shape: str, r_hole: float) -> str:
    """Return MJCF geoms forming the hole block walls around (0,0)."""
    geoms = []
    h2 = BLOCK_H / 2
    if shape == "square":
        outer = r_hole + WALL_T
        # four walls
        for sx, sy, hx, hy in [
            (0, r_hole + WALL_T / 2, outer, WALL_T / 2),
            (0, -(r_hole + WALL_T / 2), outer, WALL_T / 2),
            (r_hole + WALL_T / 2, 0, WALL_T / 2, r_hole),
            (-(r_hole + WALL_T / 2), 0, WALL_T / 2, r_hole),
        ]:
            geoms.append(
                f'<geom type="box" pos="{sx} {sy} {h2}" size="{hx} {hy} {h2}" class="fixed"/>'
            )
    else:
        n = 16
        half_w = (r_hole + WALL_T) * math.tan(math.pi / n) + 1e-4
        for i in range(n):
            th = 2 * math.pi * i / n
            c = r_hole + WALL_T / 2
            geoms.append(
                f'<geom type="box" pos="{c*math.cos(th):.6f} {c*math.sin(th):.6f} {h2}" '
                f'euler="0 0 {th:.6f}" size="{WALL_T/2} {half_w:.6f} {h2}" class="fixed"/>'
            )
    return "\n".join(geoms)


def build_xml(c: EpisodeConditions) -> str:
    r = c.diameter / 2
    r_hole = r + c.clearance
    if c.shape == "square":
        peg_geom = f'<geom name="peg" type="box" size="{r} {r} {PEG_LEN/2}" class="part"/>'
    else:
        peg_geom = f'<geom name="peg" type="cylinder" size="{r} {PEG_LEN/2}" class="part"/>'
    kp = 2000.0 * c.gain_scale
    kv = 2 * math.sqrt(kp * 0.6)
    hx, hy = c.hole_xy
    return f"""
<mujoco model="nist_lite">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{PHYSICS_DT}" cone="elliptic" impratio="10" integrator="implicitfast"/>
  <default>
    <geom friction="{c.friction} 0.01 0.001" solref="0.002 1" solimp="0.95 0.99 0.0005"/>
    <default class="fixed"><geom rgba="0.5 0.5 0.55 1"/></default>
    <default class="part"><geom condim="4" rgba="0.8 0.6 0.2 1"/></default>
    <default class="pad"><geom condim="4" rgba="0.2 0.2 0.2 1" friction="{c.friction*1.5} 0.02 0.001"/></default>
  </default>
  <worldbody>
    <geom name="board" type="box" pos="0 0 -0.005" size="0.3 0.3 0.005" class="fixed"/>
    <body name="hole" pos="{hx} {hy} 0">
      {_hole_walls(c.shape, r_hole)}
    </body>
    <body name="hand" pos="0 0 0" gravcomp="1">
      <joint name="jx" type="slide" axis="1 0 0" damping="{kv}" range="-0.25 0.25"/>
      <joint name="jy" type="slide" axis="0 1 0" damping="{kv}" range="-0.25 0.25"/>
      <joint name="jz" type="slide" axis="0 0 1" damping="{kv}" range="{HAND_Z_MIN} 0.3"/>
      <joint name="jyaw" type="hinge" axis="0 0 1" damping="0.5" range="-3.2 3.2"/>
      <inertial pos="0 0 0.03" mass="0.6" diaginertia="0.001 0.001 0.001"/>
      <geom name="palm" type="box" pos="0 0 0.027" size="0.02 0.03 0.006" rgba="0.3 0.3 0.8 1"/>
      <body name="fl" pos="0 0 0" gravcomp="1">
        <joint name="jfl" type="slide" axis="0 1 0" limited="true" range="0 {MAX_OPEN}" damping="2" armature="0.001"/>
        <inertial pos="0 0.003 0.012" mass="0.02" diaginertia="1e-6 1e-6 1e-6"/>
        <geom name="padl" type="box" pos="0 0.003 0.012" size="0.008 0.003 0.008" class="pad"/>
      </body>
      <body name="fr" pos="0 0 0" gravcomp="1">
        <joint name="jfr" type="slide" axis="0 -1 0" limited="true" range="0 {MAX_OPEN}" damping="2" armature="0.001"/>
        <inertial pos="0 -0.003 0.012" mass="0.02" diaginertia="1e-6 1e-6 1e-6"/>
        <geom name="padr" type="box" pos="0 -0.003 0.012" size="0.008 0.003 0.008" class="pad"/>
      </body>
    </body>
    <body name="part" pos="0 0 0.1">
      <freejoint name="part"/>
      <inertial pos="0 0 0" mass="{c.peg_mass}" diaginertia="{c.peg_mass*r*r/2+1e-8} {c.peg_mass*r*r/2+1e-8} {c.peg_mass*r*r/2+1e-8}"/>
      {peg_geom}
    </body>
  </worldbody>
  <contact>
    <exclude body1="hand" body2="fl"/><exclude body1="hand" body2="fr"/><exclude body1="fl" body2="fr"/>
  </contact>
  <actuator>
    <position name="ax" joint="jx" kp="{kp}"/>
    <position name="ay" joint="jy" kp="{kp}"/>
    <position name="az" joint="jz" kp="{kp}"/>
    <position name="ayaw" joint="jyaw" kp="{2.0*c.gain_scale}"/>
    <position name="afl" joint="jfl" kp="400" forcerange="-15 15" ctrlrange="0 {MAX_OPEN}"/>
    <position name="afr" joint="jfr" kp="400" forcerange="-15 15" ctrlrange="0 {MAX_OPEN}"/>
  </actuator>
</mujoco>
"""


class Episode:
    """One simulated episode. Steps only through `step(action)` at 25 Hz."""

    def __init__(self, cond: EpisodeConditions, rng: np.random.Generator):
        self.c = cond
        self.rng = rng
        self.model = mujoco.MjModel.from_xml_string(build_xml(cond))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.part_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "part")]
        self.part_vadr = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "part")]
        self.part_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "part")
        self.grip_bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("hand", "fl", "fr")}
        self.hole = np.array([cond.hole_xy[0], cond.hole_xy[1], BLOCK_H])
        self.r_hole = cond.diameter / 2 + cond.clearance
        self.last_action = np.zeros(ACT_DIM)
        self.steps = 0
        self.seated_run = 0
        self.done = False
        self.success = False
        self.success_step: int | None = None
        self.terminal_reason = "running"
        self.action_l2 = 0.0
        self._reset()

    # ---- reset -----------------------------------------------------------------
    def _set_hand(self, xyz, yaw, open_frac):
        d = self.data
        d.qpos[0:3] = xyz
        d.qpos[3] = yaw
        d.qpos[4] = d.qpos[5] = open_frac * MAX_OPEN
        self.target = np.array([*xyz, yaw], dtype=float)
        self.grip = open_frac * MAX_OPEN

    def _set_part(self, xyz, yaw):
        q = self.part_qadr
        self.data.qpos[q : q + 3] = xyz
        self.data.qpos[q + 3 : q + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.data.qvel[self.part_vadr : self.part_vadr + 6] = 0

    def _apply_ctrl(self):
        d = self.data
        d.ctrl[0:4] = self.target
        d.ctrl[4] = d.ctrl[5] = self.grip

    def _resample_start(self) -> None:
        """Re-draw start offsets from the same ranges as conditions.sample_episodes.

        Consumes self.rng so the whole reset stays deterministic from the platform seed.
        Stratum identity (shape, diameter, reset_type) is unchanged.
        """
        c = self.c
        rng = self.rng
        rt = c.reset_type
        if rt == "grasped_above":
            r = float(rng.uniform(0.0, 0.005))
            th = float(rng.uniform(0, 2 * math.pi))
            c.offset_xy = (r * math.cos(th), r * math.sin(th))
            c.height = float(rng.uniform(0.005, 0.025))
            c.yaw_err = float(rng.uniform(-0.15, 0.15)) if c.shape == "square" else 0.0
        elif rt == "grasped_far":
            r = float(rng.uniform(0.01, 0.03))
            th = float(rng.uniform(0, 2 * math.pi))
            c.offset_xy = (r * math.cos(th), r * math.sin(th))
            c.height = float(rng.uniform(0.02, 0.05))
            c.yaw_err = float(rng.uniform(-0.5, 0.5)) if c.shape == "square" else 0.0
        elif rt == "near_hole":
            r = float(rng.uniform(0.0008, 0.0025))
            th = float(rng.uniform(0, 2 * math.pi))
            c.offset_xy = (r * math.cos(th), r * math.sin(th))
            c.height = 0.0003
            c.yaw_err = float(rng.uniform(-0.1, 0.1)) if c.shape == "square" else 0.0
        elif rt == "lying":
            r = float(rng.uniform(0.03, 0.06))
            th = float(rng.uniform(0, 2 * math.pi))
            c.offset_xy = (r * math.cos(th), r * math.sin(th))
            c.height = float(rng.uniform(0.01, 0.03))
            c.yaw_err = float(rng.uniform(-math.pi, math.pi))
        elif rt == "loose":
            r = float(rng.uniform(0.03, 0.06))
            th = float(rng.uniform(0, 2 * math.pi))
            c.offset_xy = (r * math.cos(th), r * math.sin(th))
            c.height = float(rng.uniform(0.01, 0.03))
            c.yaw_err = float(rng.uniform(-math.pi / 4, math.pi / 4))
        else:
            raise ValueError(f"unknown reset_type: {rt!r}")

    def _spawn_ok(self) -> bool:
        """True if the settled start is finite, on-board, slow, and not interlocking."""
        d = self.data
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            return False
        ppos, _ = self.part_pose()
        pv = d.qvel[self.part_vadr : self.part_vadr + 3]
        if float(np.linalg.norm(pv)) > SPAWN_MAX_SPEED:
            return False
        if abs(ppos[0]) > SPAWN_BOARD_MARGIN or abs(ppos[1]) > SPAWN_BOARD_MARGIN or ppos[2] < -0.01:
            return False
        hx, hy = self.c.hole_xy
        ox, oy = self.c.offset_xy
        if float(np.hypot(ppos[0] - (hx + ox), ppos[1] - (hy + oy))) > SPAWN_MAX_DRIFT:
            return False
        for i in range(d.ncon):
            con = d.contact[i]
            b1 = self.model.geom_bodyid[con.geom1]
            b2 = self.model.geom_bodyid[con.geom2]
            if self.part_body in (b1, b2) and con.dist < -SPAWN_MAX_PENETRATION:
                return False
        return True

    def _place_and_settle(self) -> None:
        c = self.c
        hx, hy = c.hole_xy
        ox, oy = c.offset_xy
        r = c.diameter / 2
        finger_z_on_part = PEG_LEN / 2 - 0.008  # grasp point (part frame)
        if c.reset_type in ("loose", "lying"):
            px, py = hx + ox, hy + oy
            if c.reset_type == "loose":
                self._set_part([px, py, PEG_LEN / 2 + 1e-4], c.yaw_err)
            else:  # on its side: rotate 90 deg about x, then yaw
                q = self.part_qadr
                self.data.qpos[q : q + 3] = [px, py, r + 1e-4]
                qx = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0])
                qz = np.array([math.cos(c.yaw_err / 2), 0, 0, math.sin(c.yaw_err / 2)])
                out = np.zeros(4)
                mujoco.mju_mulQuat(out, qz, qx)
                self.data.qpos[q + 3 : q + 7] = out
                self.data.qvel[self.part_vadr : self.part_vadr + 6] = 0
            self._set_hand([px + self.rng.uniform(-0.004, 0.004), py + self.rng.uniform(-0.004, 0.004),
                            PEG_LEN + 0.01 + c.height], 0.0, 1.0)
            settle_grip = MAX_OPEN
            pz = None
        else:
            bottom = BLOCK_H + c.height
            pz = bottom + PEG_LEN / 2
            px, py = hx + ox, hy + oy
            self._set_part([px, py, pz], c.yaw_err)
            open_frac = min(1.0, (r + 0.0005) / MAX_OPEN)
            self._set_hand([px, py, pz + finger_z_on_part - PAD_CENTRE_Z], c.yaw_err, open_frac)
            settle_grip = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.grip = settle_grip
        for _ in range(150):
            self._apply_ctrl()
            if c.reset_type not in ("loose", "lying"):
                # hold the part in place while the fingers close
                q = self.part_qadr
                self.data.qvel[self.part_vadr : self.part_vadr + 6] = 0
                self.data.qpos[q : q + 3] = [px, py, pz]
            mujoco.mj_step(self.model, self.data)
        for _ in range(50):
            self._apply_ctrl()
            mujoco.mj_step(self.model, self.data)

    def _force_safe_start(self) -> None:
        """Last-resort placement that clears the hole block for loose/lying parts."""
        c = self.c
        if c.reset_type in ("loose", "lying"):
            # Far from the hole, long axis tangential so a 50 mm peg cannot hit the walls.
            c.offset_xy = (0.06, 0.0)
            c.height = 0.02
            c.yaw_err = math.pi / 2 if c.reset_type == "lying" else 0.0
        else:
            self._resample_start()

    def _reset(self):
        # First attempt keeps the Latin-hypercube draw. Failures (blow-ups, wall
        # interlocking) re-sample offsets from the episode rng within the same stratum.
        for attempt in range(MAX_SPAWN_ATTEMPTS):
            if attempt > 0:
                self._resample_start()
            self._place_and_settle()
            if self._spawn_ok():
                return
        self._force_safe_start()
        self._place_and_settle()

    # ---- observation -----------------------------------------------------------
    def part_pose(self):
        q = self.part_qadr
        return self.data.qpos[q : q + 3].copy(), self.data.qpos[q + 3 : q + 7].copy()

    def observe(self, noise_rng: np.random.Generator) -> np.ndarray:
        d = self.data
        hand = d.qpos[0:3] - self.hole
        yaw = d.qpos[3]
        hand_v = d.qvel[0:3]
        gap = d.qpos[4] + d.qpos[5]
        ppos, pquat = self.part_pose()
        prel = ppos - self.hole + noise_rng.normal(0, self.c.obs_noise, 3)
        pvel = d.qvel[self.part_vadr : self.part_vadr + 3]
        o = np.concatenate([
            hand * 10, [math.sin(yaw), math.cos(yaw)], hand_v, [gap * 50],
            prel * 10, pquat, pvel,
            [self.c.diameter * 50, 1.0 if self.c.shape == "square" else 0.0],
            self.last_action,
        ])
        return o.astype(np.float32)

    def _gripper_touching_part(self) -> bool:
        m, d = self.model, self.data
        for i in range(d.ncon):
            con = d.contact[i]
            b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
            if self.part_body in (b1, b2) and (b1 in self.grip_bodies or b2 in self.grip_bodies):
                return True
        return False

    def _upright(self) -> bool:
        # World-z component of the part's long axis is 1 - 2(x^2 + y^2) for quat (w, x, y, z).
        # The peg is end-symmetric, so either end down counts.
        _, (w, x, y, z) = self.part_pose()
        return abs(1.0 - 2.0 * (x * x + y * y)) >= math.cos(MAX_SEAT_TILT)

    # ---- step ------------------------------------------------------------------
    def fail(self, reason: str):
        self.done = True
        self.success = False
        self.terminal_reason = reason

    def step(self, action: np.ndarray) -> None:
        if self.done:
            return
        a = np.clip(action, -1.0, 1.0)
        self.action_l2 += float(np.sum(a * a))
        self.last_action = a
        self.target[0:3] += a[0:3] * DXYZ
        self.target[0] = np.clip(self.target[0], -0.2, 0.2)
        self.target[1] = np.clip(self.target[1], -0.2, 0.2)
        self.target[2] = np.clip(self.target[2], HAND_Z_MIN, 0.25)
        self.target[3] = np.clip(self.target[3] + a[3] * DYAW, -3.1, 3.1)
        self.grip = (a[4] + 1.0) / 2.0 * MAX_OPEN  # +1 = open, -1 = closed
        for _ in range(SUBSTEPS):
            self._apply_ctrl()
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        d = self.data
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            return self.fail("unstable")
        ppos, _ = self.part_pose()
        pv = d.qvel[self.part_vadr : self.part_vadr + 3]
        if float(np.linalg.norm(pv)) > MAX_PART_SPEED:
            return self.fail("unstable")
        if abs(ppos[0]) > 0.28 or abs(ppos[1]) > 0.28 or ppos[2] < -0.01:
            return self.fail("part_lost")
        bottom = ppos[2] - PEG_LEN / 2
        xy_err = float(np.hypot(ppos[0] - self.hole[0], ppos[1] - self.hole[1]))
        # Assembled = seated within 1 mm, inside the hole, and the gripper released and clear
        # of the part (fingertips above the part's top face, no gripper contact).
        pad_bottom = d.qpos[2] + PAD_BOTTOM_OFFSET
        clear = pad_bottom >= ppos[2] + PEG_LEN / 2 + 0.002 and not self._gripper_touching_part()
        seated = bottom <= SEAT_TOL_Z and xy_err <= self.r_hole and clear and self._upright()
        self.seated_run = self.seated_run + 1 if seated else 0
        if self.seated_run >= SEATED_HOLD:
            self.done = True
            self.success = True
            self.success_step = self.steps
            self.terminal_reason = "assembled"
            return
        if self.steps >= self.c.max_steps:
            self.fail("time_limit")
