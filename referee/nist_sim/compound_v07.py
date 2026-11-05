"""v0.7 "Assembly III": harder compound assembly (profile ``v07``).

Builds on v0.6 (``compound.py``) with new step kinds, denser graphs, late reveals, and
stronger between-step dynamics. Observation stays 288-d with the v0.6 layout; new kinds
reuse twist/collar/task channels (documented encodings below).

Encodings in the 288-d observation (v0.6 prefix unchanged):
- Twist channels (per-peg task feats 2..4): used for ``twist`` and ``screw``. ``screw`` sets
  ``twist_dir`` to ±2 so a script can tell them apart; progress is still in [0, 1].
- Collar table: ``stack`` rings use present=2.0 (collar uses 1.0). Seat target for stacks is
  the peg top, not the slot block.
- Hidden params: when ``reveal_after`` is set and that step is not done, twist_dir / progress /
  key fields are zeroed in the observation.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from . import conditions as C
from . import profile as _profile
from .compound import (
    COLLAR_H,
    COLLAR_MAX_PEG_OUTER,
    COLLAR_T,
    EVENT_KINDS,
    GLOBAL_FEAT,
    LYING_EXTRA,
    MAX_COLLARS,
    OBS_DIM_V06,
    PARTIAL_CAP,
    TASK_PEG_FEAT,
    TWIST_TARGET,
    WAVE_EXTRA,
    CollarSpec,
    CompoundEpisode,
    CompoundSpec,
    TaskStep,
    _assign,
    _collar_spot,
    _extend_pegs,
    _step_wave_of,
    _topo_random,
    compound_score as _compound_score_v06,
    quat_yaw,
    step_credit,
    wrap,
)
from .conditions import EpisodeConditions, sample_episodes, slot_world_xy
from .env import (
    BLOCK_H,
    MAX_PEGS,
    PEG_LEN,
    SEATED_HOLD,
    SPAWN_BOARD_MARGIN,
    SPAWN_MAX_DRIFT,
    SPAWN_MAX_SPEED,
    _yaw_quat,
)

# ---- v0.7 vocabulary ----------------------------------------------------------------------------
STEP_KINDS_V07 = ("insert", "collar", "twist", "rework", "screw", "stack", "key")
STEP_WEIGHT_V07 = {
    "insert": 0.25,
    "collar": 1.2,
    "twist": 0.9,
    "rework": 1.4,
    "screw": 1.6,
    "stack": 1.5,
    "key": 1.3,
}
# Shorter budgets than v0.6: hardness from density + new kinds, not long horizons (eval cost).
STEP_BUDGET_V07 = {
    "insert": 200,
    "collar": 160,
    "twist": 110,
    "rework": 280,
    "screw": 180,
    "stack": 170,
    "key": 140,
}
REWORK_EXTRA = 80
TIER_CFG_V07 = {
    # Hardness from new kinds + events + tight slack, not from unbounded step counts (eval cost).
    # key_frac is intentionally modest: each key adds a full step after insert.
    "c1": dict(
        extra_pegs=0, collar_frac=0.30, twist_frac=0.25, screw_frac=0.10, rework_frac=0.10,
        stack_frac=0.10, key_frac=0.10, waves=(1, 1), edge_p=0.05, event_p=0.10, slack=1.05,
        reveal_p=0.05,
    ),
    "c2": dict(
        extra_pegs=0, collar_frac=0.50, twist_frac=0.45, screw_frac=0.25, rework_frac=0.25,
        stack_frac=0.25, key_frac=0.18, waves=(1, 2), edge_p=0.18, event_p=0.35, slack=0.88,
        reveal_p=0.20,
    ),
    "c3": dict(
        extra_pegs=1, collar_frac=0.70, twist_frac=0.55, screw_frac=0.40, rework_frac=0.40,
        stack_frac=0.40, key_frac=0.25, waves=(2, 3), edge_p=0.30, event_p=0.60, slack=0.65,
        reveal_p=0.40,
    ),
    "c4": dict(
        extra_pegs=2, collar_frac=0.90, twist_frac=0.70, screw_frac=0.55, rework_frac=0.55,
        stack_frac=0.55, key_frac=0.35, waves=(3, 4), edge_p=0.42, event_p=0.85, slack=0.48,
        reveal_p=0.55,
    ),
}
NOISE_SCALE_V07 = dict(slot_bias=0.55, peg_delta=0.45, glitch=0.55, drift=0.55)
EVENT_KINDS_V07 = EVENT_KINDS + ("swap_bias",)
EVENT_WEIGHTS_V07 = np.array([0.14, 0.14, 0.12, 0.10, 0.18, 0.16, 0.16])
SCREW_TARGET = (1.4, 2.4)  # rad; a bit longer than twist
KEY_TOL = (0.06, 0.14)  # rad half-width
STACK_SEAT_XY = 0.0035
# Stack ring rests near the peg top (peg centre at ~PEG_LEN/2 when seated → top ≈ PEG_LEN).
STACK_SEAT_Z_CLEAR = 0.002  # |bottom - peg_top| tolerance
PRESS_FORCE = 0.8  # |wrist fz| threshold (obs units are Newtons-ish from env)
PRESS_Z_MARGIN = 0.006  # hand must be this close to peg top while gripping

OBS_DIM_V07 = OBS_DIM_V06  # 288; encodings reuse v0.6 channels


def _is_v07() -> bool:
    return _profile.get().name == "v07"


def _soften_noise_v07(cond: EpisodeConditions) -> None:
    s = NOISE_SCALE_V07
    cond.slot_bias_list = [(x * s["slot_bias"], y * s["slot_bias"]) for x, y in cond.slot_bias_list]
    cond.clutter_bias_list = [(x * s["slot_bias"], y * s["slot_bias"]) for x, y in cond.clutter_bias_list]
    cond.peg_bias_delta_list = [(x * s["peg_delta"], y * s["peg_delta"]) for x, y in cond.peg_bias_delta_list]
    cond.glitch_rate *= s["glitch"]
    cond.bias_drift *= s["drift"]


def _draw_event_v07(rng: np.random.Generator) -> dict:
    kind = str(rng.choice(EVENT_KINDS_V07, p=EVENT_WEIGHTS_V07 / EVENT_WEIGHTS_V07.sum()))
    if kind == "friction":
        return dict(kind=kind, scale=float(rng.uniform(0.40, 1.9)))
    if kind == "gain":
        return dict(kind=kind, scale=float(rng.uniform(0.45, 1.55)))
    if kind == "mass":
        return dict(kind=kind, scale=float(rng.uniform(0.55, 2.4)))
    if kind == "delay":
        return dict(kind=kind, steps=int(rng.integers(0, 4)))
    if kind == "rebias":
        return dict(kind=kind, mag=float(rng.uniform(0.0015, 0.006)), ang=float(rng.uniform(0, 2 * math.pi)))
    if kind == "swap_bias":
        return dict(kind=kind)  # reshuffle peg-vs-slot residuals (handled in episode)
    return dict(kind=kind, sigma=float(rng.uniform(0.12, 0.28)))


def make_compound_v07(rng: np.random.Generator, cond: EpisodeConditions, tier: str) -> CompoundSpec:
    cfg = TIER_CFG_V07[tier]
    _soften_noise_v07(cond)
    _extend_pegs(rng, cond, cfg["extra_pegs"])
    K = cond.n_pegs
    W = int(rng.integers(cfg["waves"][0], cfg["waves"][1] + 1))
    W = max(1, min(W, K))
    order = list(rng.permutation(K))
    grasped = [i for i, p in enumerate(cond.pegs) if p.start_type == "grasped"]
    if grasped:
        order.remove(grasped[0])
        order.insert(0, grasped[0])
    peg_wave = [0] * K
    for rank, i in enumerate(order):
        peg_wave[i] = 0 if rank == 0 else (int(rng.integers(0, W)) if rank >= W else rank)
    for i, p in enumerate(cond.pegs):
        if peg_wave[i] > 0 and p.start_type in ("grasped", "tray"):
            p.start_type = "loose"

    # --- ring parts: collars and stacks on narrow pegs -----------------------------------------
    cands = [i for i, p in enumerate(cond.pegs) if p.spec.outer_radius() <= COLLAR_MAX_PEG_OUTER]
    n_col = min(MAX_COLLARS, len(cands), int(round(cfg["collar_frac"] * max(1, len(cands)))))
    n_stack = min(
        MAX_COLLARS - n_col,
        max(0, len(cands) - n_col),
        int(round(cfg["stack_frac"] * max(1, len(cands)))),
    )
    ring_pegs: list[tuple[int, str]] = []
    if n_col:
        for i in rng.choice(cands, size=n_col, replace=False).tolist():
            ring_pegs.append((int(i), "collar"))
    remain = [i for i in cands if i not in {p for p, _ in ring_pegs}]
    if n_stack and remain:
        for i in rng.choice(remain, size=min(n_stack, len(remain)), replace=False).tolist():
            ring_pegs.append((int(i), "stack"))

    collars: list[CollarSpec] = []
    taken: list[tuple[float, float]] = []
    for peg_i, kind in ring_pegs:
        r_in = cond.pegs[peg_i].spec.outer_radius() + float(rng.uniform(0.0006, 0.0012))
        spot = _collar_spot(rng, cond, r_in + COLLAR_T, taken)
        taken.append(spot)
        w = int(rng.integers(peg_wave[peg_i], W))
        collars.append(CollarSpec(peg=peg_i, r_in=r_in, offset_xy=spot, wave=w, kind=kind))

    # --- rework: peg starts in a wrong empty slot ----------------------------------------------
    used_slots = {p.slot for p in cond.pegs}
    free_slots = [s for s in range(C.N_SLOTS) if s not in used_slots]
    rework_n = min(len(free_slots), int(round(cfg["rework_frac"] * K)))
    # Never rework the pre-grasped peg (hand already holds it).
    rework_cands = [i for i in range(K) if i not in grasped and peg_wave[i] == 0]
    rework_pegs: dict[int, int] = {}
    if rework_n and rework_cands and free_slots:
        pick = rng.choice(rework_cands, size=min(rework_n, len(rework_cands)), replace=False).tolist()
        wrongs = rng.choice(free_slots, size=len(pick), replace=False).tolist()
        for i, ws in zip(pick, wrongs):
            rework_pegs[int(i)] = int(ws)

    # --- twists / screws on round pegs ---------------------------------------------------------
    rounds = [i for i, p in enumerate(cond.pegs) if p.spec.shape == "round"]
    n_tw = int(round(cfg["twist_frac"] * max(1, len(rounds))))
    n_sc = int(round(cfg["screw_frac"] * max(1, len(rounds))))
    twist_pegs: list[int] = []
    screw_pegs: list[int] = []
    if rounds:
        # Prefer disjoint sets; screws take priority on hard tiers for ceiling.
        n_sc = min(n_sc, len(rounds))
        screw_pegs = rng.choice(rounds, size=n_sc, replace=False).tolist() if n_sc else []
        left = [i for i in rounds if i not in screw_pegs]
        n_tw = min(n_tw, len(left))
        twist_pegs = rng.choice(left, size=n_tw, replace=False).tolist() if n_tw else []

    # --- key gates on a subset of non-round inserts (and some round) ---------------------------
    key_cands = [i for i in range(K) if i not in rework_pegs]
    n_key = min(len(key_cands), int(round(cfg["key_frac"] * K)))
    key_pegs = set(rng.choice(key_cands, size=n_key, replace=False).tolist() if n_key else [])

    # Guarantee every episode is compound.
    if not collars and not twist_pegs and not screw_pegs and not rework_pegs and not key_pegs:
        if cands:
            peg_i = int(cands[0])
            r_in = cond.pegs[peg_i].spec.outer_radius() + 0.0008
            spot = _collar_spot(rng, cond, r_in + COLLAR_T, taken)
            collars.append(CollarSpec(peg=peg_i, r_in=r_in, offset_xy=spot, wave=peg_wave[peg_i], kind="collar"))
        elif rounds:
            twist_pegs = [int(rounds[0])]
        elif free_slots and rework_cands:
            rework_pegs[int(rework_cands[0])] = int(free_slots[0])

    # --- build steps --------------------------------------------------------------------------
    steps: list[TaskStep] = []
    edges: set[tuple[int, int]] = set()
    insert_idx = {}
    for i in range(K):
        if i in rework_pegs:
            steps.append(TaskStep("rework", i, i, wrong_slot=rework_pegs[i]))
        else:
            steps.append(TaskStep("insert", i, i))
        insert_idx[i] = len(steps) - 1
        if i in key_pegs:
            # Key is a follow-on yaw-index gate after the peg is in its true hole.
            ky = float(cond.slots[cond.pegs[i].slot].yaw + cond.board_yaw + rng.uniform(-0.4, 0.4))
            steps.append(
                TaskStep(
                    "key",
                    i,
                    i,
                    key_yaw=ky,
                    key_tol=float(rng.uniform(*KEY_TOL)),
                )
            )
            edges.add((insert_idx[i], len(steps) - 1))

    for i in twist_pegs:
        steps.append(
            TaskStep(
                "twist",
                i,
                i,
                twist_target=float(rng.uniform(*TWIST_TARGET)),
                twist_dir=1.0 if rng.random() < 0.5 else -1.0,
            )
        )
        edges.add((insert_idx[i], len(steps) - 1))
    for i in screw_pegs:
        steps.append(
            TaskStep(
                "screw",
                i,
                i,
                twist_target=float(rng.uniform(*SCREW_TARGET)),
                twist_dir=1.0 if rng.random() < 0.5 else -1.0,
                press=True,
            )
        )
        edges.add((insert_idx[i], len(steps) - 1))

    for j, col in enumerate(collars):
        kind = "stack" if col.is_stack else "collar"
        steps.append(TaskStep(kind, j, col.peg))
        sc = len(steps) - 1
        edges.add((insert_idx[col.peg], sc))
        # Optional: stack/collar after twist/screw on same peg.
        tw = [k for k, s in enumerate(steps) if s.kind in ("twist", "screw") and s.peg == col.peg]
        if tw and rng.random() < 0.55:
            tw_i = tw[0]
            if col.wave > peg_wave[col.peg]:
                edges.add((tw_i, sc))
            else:
                edges.add((tw_i, sc) if rng.random() < 0.5 else (sc, tw_i))

    # Random extra precedence (wave-consistent).
    topo = _topo_random(rng, len(steps), edges)
    pos = {s: k for k, s in enumerate(topo)}
    for a in range(len(steps)):
        for b in range(len(steps)):
            if (
                pos[a] < pos[b]
                and (a, b) not in edges
                and rng.random() < cfg["edge_p"]
                and _step_wave_of(steps[a], peg_wave, collars) <= _step_wave_of(steps[b], peg_wave, collars)
            ):
                edges.add((a, b))
    if grasped:
        edges = {(a, b) for a, b in edges if b != insert_idx[grasped[0]]}
    edges = {
        (a, b)
        for a, b in edges
        if _step_wave_of(steps[a], peg_wave, collars) <= _step_wave_of(steps[b], peg_wave, collars)
    }
    for a, b in edges:
        steps[b].prereq.append(a)
    for s in steps:
        s.prereq.sort()

    # Late reveals: hide twist/screw/key params until a random earlier prerequisite (or insert) is done.
    for k, s in enumerate(steps):
        if s.kind not in ("twist", "screw", "key"):
            continue
        if rng.random() >= cfg["reveal_p"]:
            continue
        gates = list(s.prereq) if s.prereq else ([insert_idx[s.peg]] if s.peg in insert_idx else [])
        if gates:
            s.reveal_after = int(gates[int(rng.integers(0, len(gates)))])

    # Horizon.
    budget = 0
    for s in steps:
        budget += STEP_BUDGET_V07[s.kind]
        if s.kind in ("insert", "rework") and cond.pegs[s.peg].start_type == "lying":
            budget += LYING_EXTRA
        if s.kind == "rework":
            budget += REWORK_EXTRA
    max_steps = int(cfg["slack"] * budget) + WAVE_EXTRA * max(0, W - 1)
    cond.max_steps = max_steps
    wave_time = [0] + [int(max_steps * min(0.80, w * float(rng.uniform(0.14, 0.28)))) for w in range(1, W)]
    for w in range(2, W):
        wave_time[w] = max(wave_time[w], wave_time[w - 1] + 20)
    events = [(_draw_event_v07(rng) if rng.random() < cfg["event_p"] else None) for _ in steps]
    return CompoundSpec(
        tier=tier, steps=steps, collars=collars, peg_wave=peg_wave, n_waves=W,
        wave_time=wave_time, events=events,
    )


def sample_compound_v07(seed: int, episodes_per_stratum: int, split: str = "live"):
    """v0.5 board draw + v0.7 Assembly III task graph per episode."""
    ranges, conds = sample_episodes(seed, episodes_per_stratum, split)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x41333307, *C.build_salt()]))
    per = episodes_per_stratum
    for s in range(0, len(conds), per):
        tiers = _assign(rng, len(conds[s : s + per]))
        for c, t in zip(conds[s : s + per], tiers):
            c.compound = make_compound_v07(rng, c, t)
    return ranges, conds


# ---- episode ------------------------------------------------------------------------------------
class AssemblyIIIEpisode(CompoundEpisode):
    """v0.7 episode: rework spawns, screw press gate, stack seat, key yaw, late reveals."""

    def __init__(self, cond: EpisodeConditions, rng: np.random.Generator):
        self._rework_wrong = {
            s.peg: s.wrong_slot for s in cond.compound.steps if s.kind == "rework" and s.wrong_slot is not None
        }
        self._extracted = {i: False for i in self._rework_wrong}
        self._key_hold = [0] * cond.n_pegs
        super().__init__(cond, rng)

    def _seat_in_wrong_slot(self, i: int) -> None:
        ws = self._rework_wrong[i]
        spec = self.c.slots[ws]
        hx, hy = slot_world_xy(self.c.board_xy, self.c.board_yaw, spec)
        yaw = spec.yaw + self.c.board_yaw
        self._set_free(
            self.peg_qadr[i], self.peg_vadr[i],
            [hx, hy, PEG_LEN / 2 + 1e-4],
            _yaw_quat(yaw),
        )

    def _place_peg(self, i: int, peg) -> None:
        if self.peg_parked[i]:
            self._set_free(self.peg_qadr[i], self.peg_vadr[i], self._park_pose(), [1.0, 0, 0, 0])
            return
        if i in self._rework_wrong:
            self._seat_in_wrong_slot(i)
            return
        super()._place_peg(i, peg)

    def _pin_rework_pegs(self) -> None:
        """Seat rework pegs in their wrong holes (spawn checks otherwise teleport them out)."""
        if not self._rework_wrong:
            return
        for i in self._rework_wrong:
            if self.peg_parked[i]:
                continue
            self._seat_in_wrong_slot(i)
            self.data.qvel[self.peg_vadr[i] : self.peg_vadr[i] + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def _place_and_settle(self) -> None:
        super()._place_and_settle()
        self._pin_rework_pegs()

    def _spawn_ok(self) -> bool:
        # Rework pegs intentionally start inside a foreign hole, which trips the parent's
        # contact-penetration gate. Validate drift against the wrong-slot target instead.
        if not self._rework_wrong:
            return super()._spawn_ok()
        d = self.data
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            return False
        for i, peg in enumerate(self.c.pegs):
            if self.peg_parked[i]:
                continue
            ppos, pquat = self.peg_pose(i)
            pv = d.qvel[self.peg_vadr[i] : self.peg_vadr[i] + 3]
            if float(np.linalg.norm(pv)) > SPAWN_MAX_SPEED:
                return False
            if abs(ppos[0]) > SPAWN_BOARD_MARGIN or abs(ppos[1]) > SPAWN_BOARD_MARGIN or ppos[2] < -0.005:
                return False
            ix, iy = self._intended_peg_xy(peg)
            if float(np.hypot(ppos[0] - ix, ppos[1] - iy)) > SPAWN_MAX_DRIFT:
                return False
            if peg.start_type != "lying" and i not in self._rework_wrong:
                w, x, y, z = pquat
                if abs(1.0 - 2.0 * (x * x + y * y)) < 0.85:
                    return False
        return True

    def _reset(self) -> None:
        super()._reset()
        # Teleport fallback can still move rework pegs; pin them after the spawn loop.
        self._pin_rework_pegs()
        for i in self._rework_wrong:
            self._extracted[i] = False

    def _intended_peg_xy(self, peg) -> tuple[float, float]:
        i = self.c.pegs.index(peg)
        if self.peg_parked[i]:
            p = self._park_pose()
            return p[0], p[1]
        if i in self._rework_wrong:
            spec = self.c.slots[self._rework_wrong[i]]
            return slot_world_xy(self.c.board_xy, self.c.board_yaw, spec)
        return super()._intended_peg_xy(peg)

    def _pressing_on_peg(self, i: int) -> bool:
        """True when the gripper holds the peg and presses it downward into the hole."""
        if not self._gripper_touching(self.peg_body[i]):
            return False
        p, _ = self.peg_pose(i)
        top = p[2] + PEG_LEN / 2
        hand_z = float(self.data.qpos[2])
        # Palm near the peg top and wrist reporting downward force.
        if hand_z > top - PRESS_Z_MARGIN + 0.01:
            return False
        fz = float(self._wrist_force[2]) if hasattr(self, "_wrist_force") else 0.0
        # Env wrench: positive fz is up on the wrist; pressing down → negative or large |f|.
        return fz < -PRESS_FORCE or abs(fz) > PRESS_FORCE and hand_z < top - 0.002

    def _stack_seated_now(self, j: int) -> bool:
        if self.collar_parked[j]:
            return False
        col = self.cs.collars[j]
        if not col.is_stack:
            return False
        p, q = self.collar_pose(j)
        hx, hy = self.slot_xy[col.peg]
        peg_p, _ = self.peg_pose(col.peg)
        peg_top = peg_p[2] + PEG_LEN / 2
        bottom = p[2] - COLLAR_H / 2
        return (
            self.seated_now[col.peg]
            and float(np.hypot(p[0] - hx, p[1] - hy)) <= STACK_SEAT_XY
            and abs(bottom - peg_top) <= STACK_SEAT_Z_CLEAR + 0.001
            and bottom > BLOCK_H + 0.015  # must not be resting on the block (that is a collar)
            and self._upright(q)
            and not self._gripper_touching(self.collar_body[j])
        )

    def _key_aligned(self, s: TaskStep) -> bool:
        if s.key_tol <= 0:
            return True
        yaw = quat_yaw(self.peg_pose(s.peg)[1])
        return abs(wrap(yaw - s.key_yaw)) <= s.key_tol

    def _rework_extracted(self, i: int) -> bool:
        if self._extracted.get(i):
            return True
        if i not in self._rework_wrong:
            return True
        p, _ = self.peg_pose(i)
        spec = self.c.slots[self._rework_wrong[i]]
        wx, wy = slot_world_xy(self.c.board_xy, self.c.board_yaw, spec)
        # Cleared the wrong hole: laterally away or lifted well above the block.
        clear = float(np.hypot(p[0] - wx, p[1] - wy)) > max(0.012, 2.5 * spec.outer_radius())
        lifted = (p[2] - PEG_LEN / 2) > BLOCK_H + 0.012
        if clear or lifted:
            self._extracted[i] = True
            return True
        return False

    def _fire_event(self, ev: dict) -> None:
        if ev["kind"] == "swap_bias":
            # Reshuffle peg-vs-slot residuals (same magnitudes, new directions).
            for i in range(self.n_pegs):
                mag = float(np.linalg.norm(self._peg_bias_xy[i]))
                ang = float(self.rng.uniform(0, 2 * math.pi))
                self._peg_bias_xy[i] = np.array([mag * math.cos(ang), mag * math.sin(ang)])
                smag = float(np.linalg.norm(self._slot_bias_xy[i]))
                sang = float(self.rng.uniform(0, 2 * math.pi))
                self._slot_bias_xy[i] = np.array([smag * math.cos(sang), smag * math.sin(sang)])
            self.events_fired.append(dict(ev, step=self.steps))
            return
        super()._fire_event(ev)

    def _step_state(self, k: int) -> tuple[bool, float]:
        s = self.cs.steps[k]
        if s.kind == "insert":
            return super()._step_state(k)
        if s.kind == "rework":
            i = s.part
            if self.peg_parked[i]:
                return False, 0.0
            extracted = self._rework_extracted(i)
            if self.seated_now[i] or (self.step_done[k] and self._peg_in_hole(i)):
                return True, PARTIAL_CAP
            p, _ = self.peg_pose(i)
            hx, hy = self.slot_xy[i]
            bottom = p[2] - PEG_LEN / 2
            dxy = float(np.hypot(p[0] - hx, p[1] - hy))
            prog = 0.0
            if extracted:
                prog = 0.10
            if self._gripper_touching(self.peg_body[i]) and bottom > 0.003:
                prog = max(prog, 0.12)
            if extracted and dxy < 0.003 and bottom < BLOCK_H + 0.010:
                prog = 0.20
            if extracted and dxy <= self.r_hole[i] and bottom < BLOCK_H:
                prog = 0.20 + 0.10 * (1.0 - max(0.0, bottom) / BLOCK_H)
            return False, prog
        if s.kind == "twist":
            return super()._step_state(k)
        if s.kind == "screw":
            i = s.part
            frac = float(np.clip(self.twist_acc[i] * s.twist_dir / max(1e-6, s.twist_target), 0.0, 1.0))
            done = frac >= 1.0 and (self.seated_now[i] or (self.step_done[k] and self._peg_in_hole(i)))
            return done, PARTIAL_CAP * frac
        if s.kind == "key":
            i = s.part
            if not (self.seated_now[i] or self._peg_in_hole(i)):
                return False, 0.0
            aligned = self._key_aligned(s)
            self._key_hold[i] = self._key_hold[i] + 1 if aligned and not self._gripper_touching(self.peg_body[i]) else 0
            done = self._key_hold[i] >= SEATED_HOLD or (self.step_done[k] and aligned and self._peg_in_hole(i))
            yaw = quat_yaw(self.peg_pose(i)[1])
            err = abs(wrap(yaw - s.key_yaw))
            frac = float(np.clip(1.0 - err / max(s.key_tol, 1e-3), 0.0, 1.0))
            return done, PARTIAL_CAP * frac
        if s.kind == "stack":
            j = s.part
            if self.collar_parked[j]:
                return False, 0.0
            done = self.collar_run[j] >= SEATED_HOLD or (
                self.step_done[k] and self._stack_seated_now(j)
            )
            p, _ = self.collar_pose(j)
            hx, hy = self.slot_xy[self.cs.collars[j].peg]
            peg_p, _ = self.peg_pose(self.cs.collars[j].peg)
            peg_top = peg_p[2] + PEG_LEN / 2
            bottom = p[2] - COLLAR_H / 2
            dxy = float(np.hypot(p[0] - hx, p[1] - hy))
            peg_ok = self._peg_in_hole(self.cs.collars[j].peg)
            prog = 0.0
            if self._gripper_touching(self.collar_body[j]) and bottom > BLOCK_H + 0.01:
                prog = 0.05
            if peg_ok and dxy < STACK_SEAT_XY and abs(bottom - peg_top) < 0.012:
                prog = 0.20
            if peg_ok and dxy < STACK_SEAT_XY and abs(bottom - peg_top) <= STACK_SEAT_Z_CLEAR + 0.002:
                prog = PARTIAL_CAP
            return done, prog
        # collar (and anything else): parent
        return super()._step_state(k)

    def step(self, action: np.ndarray) -> None:
        if self.done:
            return
        # Screw yaw must not free-spin: clear prev so the parent skip-accumulates, then apply
        # a press-gated delta *before* we re-evaluate screw step completion.
        twist_before = list(self.twist_acc)
        screw_pegs = {s.peg for s in self.cs.steps if s.kind == "screw"}
        saved_prev = {i: self._twist_prev[i] for i in screw_pegs}
        for i in screw_pegs:
            self._twist_prev[i] = None
        super().step(action)
        if self.terminal_reason in ("unstable",) or (
            self.done and self.terminal_reason not in ("assembled", "time_limit")
        ):
            return
        for i in screw_pegs:
            y = quat_yaw(self.peg_pose(i)[1])
            prev = saved_prev[i]
            if prev is not None and self._peg_in_hole(i) and self._pressing_on_peg(i):
                self.twist_acc[i] = twist_before[i] + wrap(y - prev)
            else:
                self.twist_acc[i] = twist_before[i]
            self._twist_prev[i] = y
        for j, col in enumerate(self.cs.collars):
            if col.is_stack:
                self.collar_run[j] = self.collar_run[j] + 1 if self._stack_seated_now(j) else 0
        # Re-evaluate screw / stack / key completion with the corrected accumulators.
        for k, s in enumerate(self.cs.steps):
            if s.kind not in ("screw", "stack", "key", "rework"):
                continue
            done, prog = self._step_state(k)
            self.step_progress[k] = max(self.step_progress[k], prog)
            if done and not self.step_done[k]:
                self.step_done[k] = True
                self.step_done_at[k] = self.steps
                self.step_in_order[k] = all(self.step_done[p] for p in s.prereq)
                ev = self.cs.events[self.n_completions] if self.n_completions < len(self.cs.events) else None
                self.n_completions += 1
                if ev is not None:
                    self._fire_event(ev)
            elif not done and self.step_done[k]:
                self.step_done[k] = False
        if all(self.step_done) and all(self.wave_arrived) and not self.done:
            self.done = True
            self.success = True
            self.success_step = self.steps
            self.terminal_reason = "assembled"

    def _params_revealed(self, s: TaskStep) -> bool:
        if s.reveal_after is None:
            return True
        return bool(self.step_done[s.reveal_after])

    def observe(self, noise_rng: np.random.Generator) -> np.ndarray:
        o = super().observe(noise_rng).astype(np.float64)
        # Rewrite task / collar channels for v0.7 encodings and late reveals.
        base_len = o.shape[0] - (MAX_PEGS * TASK_PEG_FEAT + MAX_COLLARS * 13 + GLOBAL_FEAT)
        # Parent already wrote task/collar/global; patch in place.
        task0 = base_len
        for k, s in enumerate(self.cs.steps):
            b = task0 + s.peg * TASK_PEG_FEAT
            revealed = self._params_revealed(s)
            if s.kind == "rework":
                o[b + 0] = float(self.step_done[k])
                o[b + 1] = float(self.step_blocked(k))
                # Signal rework: put extracted flag in twist_progress slot when idle of twist.
                if not any(t.kind in ("twist", "screw") and t.peg == s.peg for t in self.cs.steps):
                    o[b + 3] = float(self._extracted.get(s.peg, False))
            elif s.kind in ("twist", "screw"):
                if revealed:
                    # screw dir encoded as ±2; twist as ±1
                    o[b + 2] = s.twist_dir * (2.0 if s.kind == "screw" else 1.0)
                    o[b + 3] = float(np.clip(self.twist_acc[s.peg] * s.twist_dir / max(1e-6, s.twist_target), 0.0, 1.0))
                else:
                    o[b + 2] = 0.0
                    o[b + 3] = 0.0
                o[b + 4] = float(self.step_blocked(k))
            elif s.kind == "key":
                if revealed:
                    o[b + 2] = math.sin(s.key_yaw)
                    o[b + 3] = math.cos(s.key_yaw)
                    o[b + 4] = float(self.step_blocked(k))
                    # Pack tol into the collar-flag slot when unused.
                    if o[b + 5] == 0.0:
                        o[b + 5] = s.key_tol * 10.0
                else:
                    o[b + 2] = 0.0
                    o[b + 3] = 0.0
                    o[b + 4] = float(self.step_blocked(k))
            elif s.kind in ("collar", "stack"):
                o[b + 5] = 1.0 if s.kind == "collar" else 2.0
        # Collar present channel: 2.0 for stacks.
        col0 = task0 + MAX_PEGS * TASK_PEG_FEAT
        for k, s in enumerate(self.cs.steps):
            if s.kind not in ("collar", "stack"):
                continue
            j = s.part
            if self.collar_parked[j]:
                continue
            b = col0 + j * 13
            if s.kind == "stack":
                o[b + 0] = 2.0
        assert o.shape[0] == OBS_DIM_V07
        return o.astype(np.float32)


def compound_score_v07(ep: AssemblyIIIEpisode | CompoundEpisode) -> dict:
    """Weighted per-step score using v0.7 weights when new kinds are present."""
    kinds = {s.kind for s in ep.cs.steps}
    if kinds <= {"insert", "collar", "twist"}:
        return _compound_score_v06(ep)
    total = wsum = 0.0
    per_kind: dict[str, list[float]] = {k: [] for k in STEP_KINDS_V07}
    for k, s in enumerate(ep.cs.steps):
        cr = step_credit(ep.step_done[k], ep.step_in_order[k], ep.step_done_at[k], ep.c.max_steps, ep.step_progress[k])
        w = STEP_WEIGHT_V07.get(s.kind, 1.0)
        total += w * cr
        wsum += w
        per_kind.setdefault(s.kind, []).append(cr)
    return dict(
        score=total / wsum if wsum else 0.0,
        n_steps=len(ep.cs.steps),
        n_done=int(sum(ep.step_done)),
        n_in_order=int(sum(1 for d, o in zip(ep.step_done, ep.step_in_order) if d and o)),
        per_kind={k: (round(float(np.mean(v)), 3) if v else None) for k, v in per_kind.items()},
        events=len(ep.events_fired),
    )
