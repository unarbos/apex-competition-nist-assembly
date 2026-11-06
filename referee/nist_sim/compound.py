"""v0.6 "Assembly II": compound, randomized multi-step assembly (profile ``v06``; dev only).

One episode is a randomized task graph on the v0.5 board:

- **Step kinds.** ``insert`` (peg into its slot, as in v0.5), ``collar`` (drop a ring over a seated
  peg so it rests on the slot block) and ``twist`` (turn a seated round peg by a target angle in a
  given direction, a bayonet / fastener stand-in).
- **Precedence.** Mandatory edges (a peg before its collar / twist) plus random extra edges drawn
  per episode (only when the prerequisite's wave is at most the dependent's). A step completed while
  one of its prerequisites is not done earns ``max(PARTIAL_CAP, OUT_OF_ORDER × in-order credit)``.
- **Waves.** Parts arrive in 1-4 waves. Later-wave parts are parked off the scene (no contacts, not
  observed) and arrive when the earlier waves are finished or at a random time, at a random free spot.
- **Randomness between subtasks.** After each completed step a pre-drawn event may fire: friction,
  controller gain, part mass, actuation delay, estimate re-bias, or a shove to a loose part.
- **Complexity tiers** ``c1``..``c4`` (every episode has at least one collar or twist) set part counts, collar / twist share, waves, edge density,
  event rate and time slack. ``c4`` is meant to sit well below 1.0 even for the privileged solver.

Scoring (``compound_score``) is per step with partial-progress milestones, so the gradient survives
on episodes nobody finishes. Everything derives from the round SEED, like ``sample_episodes``.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import conditions as C
from .conditions import (
    GLITCH_NOISE_GAIN,
    GLITCH_OFFSET,
    EpisodeConditions,
    PegStart,
    sample_episodes,
)
from .env import (
    ACT_DIM,
    BLOCK_H,
    MAX_PEGS,
    OBS_DIM,
    PEG_FEAT,
    PEG_LEN,
    SEAT_TOL_Z,
    SEATED_HOLD,
    Episode,
    _yaw_quat,
    build_xml,
)

# ---- task vocabulary ----------------------------------------------------------------------------
STEP_KINDS = ("insert", "collar", "twist")
# Compound steps carry most of the credit: a controller that only inserts pegs is capped well below 1.
STEP_WEIGHT = {"insert": 0.3, "collar": 1.5, "twist": 1.0}
STEP_BUDGET = {"insert": 260, "collar": 200, "twist": 130}  # control steps used to size the horizon
LYING_EXTRA = 140
WAVE_EXTRA = 40
MAX_COLLARS = 4
COLLAR_T = 0.0025  # ring wall thickness
COLLAR_H = 0.008  # ring height
COLLAR_NSEG = 8  # 12 -> 8 segments: ~20% fewer contacts, same fit (inner faces at r_in)
COLLAR_MAX_PEG_OUTER = 0.0072  # ring outer radius <= ~10.9 mm: the 24 mm gripper must straddle it
COLLAR_SEAT_XY = 0.003
COLLAR_SEAT_Z = 0.0015
TWIST_TARGET = (1.2, 2.0)  # rad
PARK_Z = 0.6
MASS_MIN, MASS_MAX = 0.008, 0.08  # kg, after a mass event
EVENT_KINDS = ("friction", "gain", "mass", "delay", "rebias", "shove")
EVENT_WEIGHTS = np.array([0.18, 0.18, 0.14, 0.12, 0.22, 0.16])

TIERS = ("c1", "c2", "c3", "c4")
TIER_FRACTIONS = (0.15, 0.30, 0.30, 0.25)
TIER_CFG = {
    "c1": dict(extra_pegs=0, collar_frac=0.25, twist_frac=0.30, waves=(1, 1), edge_p=0.00, event_p=0.00, slack=1.20),
    "c2": dict(extra_pegs=0, collar_frac=0.50, twist_frac=0.60, waves=(1, 2), edge_p=0.10, event_p=0.25, slack=1.00),
    "c3": dict(extra_pegs=1, collar_frac=0.70, twist_frac=0.80, waves=(2, 3), edge_p=0.20, event_p=0.45, slack=0.85),
    "c4": dict(extra_pegs=2, collar_frac=0.90, twist_frac=1.00, waves=(3, 4), edge_p=0.30, event_p=0.65, slack=0.70),
}
# Perception is minor noise in v0.6: estimate biases / glitches scaled down from v0.5.1.
NOISE_SCALE = dict(slot_bias=0.5, peg_delta=0.4, glitch=0.5, drift=0.5)

# Credit rubric (per step, then weighted mean over steps).
DONE_BASE, DONE_TIME = 0.70, 0.30
# Multiplier on in-order credit when a step finishes with open prerequisites. Floored at
# PARTIAL_CAP so finishing out of order never pays less than stopping just short of done.
OUT_OF_ORDER = 0.15
PARTIAL_CAP = 0.30

# Observation: v0.5 layout (184) + per-peg task (8 x 6) + collars (4 x 13) + globals (4).
TASK_PEG_FEAT = 6
COLLAR_FEAT = 13
GLOBAL_FEAT = 4
OBS_DIM_V06 = OBS_DIM + MAX_PEGS * TASK_PEG_FEAT + MAX_COLLARS * COLLAR_FEAT + GLOBAL_FEAT  # 288


@dataclass
class CollarSpec:
    peg: int  # index into cond.pegs
    r_in: float
    offset_xy: tuple[float, float]  # board frame, absolute (not relative to a slot)
    wave: int = 0
    # v0.7: "collar" seats on the slot block; "stack" seats on the peg top.
    kind: str = "collar"

    @property
    def r_out(self) -> float:
        return self.r_in + COLLAR_T

    @property
    def is_stack(self) -> bool:
        return self.kind == "stack"


@dataclass
class TaskStep:
    kind: str
    part: int  # peg index (insert / twist / screw / rework / key) or collar index (collar / stack)
    peg: int  # peg the step belongs to
    prereq: list[int] = field(default_factory=list)
    twist_target: float = 0.0
    twist_dir: float = 1.0
    # v0.7 extensions (defaults keep v0.6 graphs unchanged).
    key_yaw: float = 0.0  # target peg yaw (world) for key-gated done checks; 0 = unused
    key_tol: float = 0.0  # half-width (rad); 0 = no key gate
    reveal_after: int | None = None  # hide twist/key params in obs until this step is done
    wrong_slot: int | None = None  # rework: layout slot index where the peg starts seated
    press: bool = False  # screw: yaw only accumulates while pressing down on the peg


@dataclass
class CompoundSpec:
    tier: str
    steps: list[TaskStep]
    collars: list[CollarSpec]
    peg_wave: list[int]
    n_waves: int
    wave_time: list[int]  # control step at which wave w arrives at the latest (w >= 1)
    events: list[dict | None]  # indexed by completion count (0-based)

    def public(self) -> dict:
        """Compact task summary (used in logs / older tooling)."""
        return {
            "tier": self.tier,
            "n_steps": len(self.steps),
            "kinds": [s.kind for s in self.steps],
            "n_edges": sum(len(s.prereq) for s in self.steps),
            "n_waves": self.n_waves,
            "n_collars": len(self.collars),
        }

    def record_task(self) -> dict:
        """Full task graph for ``nist_lite_record/4`` episode rows (replay viewer)."""
        steps = []
        for s in self.steps:
            row = {"kind": s.kind, "part": int(s.part), "peg": int(s.peg), "prereq": list(s.prereq)}
            if s.kind in ("twist", "screw"):
                row["twist_target"] = round(float(s.twist_target), 6)
                row["twist_dir"] = int(s.twist_dir)
            if s.key_tol > 0:
                row["key_yaw"] = round(float(s.key_yaw), 6)
                row["key_tol"] = round(float(s.key_tol), 6)
            if s.reveal_after is not None:
                row["reveal_after"] = int(s.reveal_after)
            if s.wrong_slot is not None:
                row["wrong_slot"] = int(s.wrong_slot)
            if s.press:
                row["press"] = True
            steps.append(row)
        return {
            "tier": self.tier,
            "n_waves": int(self.n_waves),
            "wave_time": [int(t) for t in self.wave_time],
            "steps": steps,
        }

    def record_collars(self) -> list[dict]:
        """Public collar specs for ``conditions.collars`` in ``nist_lite_record/4``."""
        out = []
        for c in self.collars:
            row = {
                "peg": int(c.peg),
                "r_in": round(float(c.r_in), 6),
                "r_out": round(float(c.r_out), 6),
                "height": COLLAR_H,
                "wave": int(c.wave),
            }
            # v0.6 collars omit kind (record schema lock). v0.7 stacks tag themselves.
            if c.kind != "collar":
                row["kind"] = c.kind
            out.append(row)
        return out


# ---- generator ----------------------------------------------------------------------------------
def _assign(rng: np.random.Generator, n: int) -> list[str]:
    raw = [f * n for f in TIER_FRACTIONS]
    counts = [int(x) for x in raw]
    order = sorted(range(len(TIERS)), key=lambda i: raw[i] - counts[i], reverse=True)
    for k in range(n - sum(counts)):
        counts[order[k % len(TIERS)]] += 1
    tiers = [t for t, c in zip(TIERS, counts) for _ in range(c)]
    rng.shuffle(tiers)
    return tiers


def _topo_random(rng: np.random.Generator, n: int, edges: set[tuple[int, int]]) -> list[int]:
    indeg = [0] * n
    for _, v in edges:
        indeg[v] += 1
    ready = [i for i in range(n) if indeg[i] == 0]
    out = []
    while ready:
        k = int(rng.integers(0, len(ready)))
        u = ready.pop(k)
        out.append(u)
        for a, b in edges:
            if a == u:
                indeg[b] -= 1
                if indeg[b] == 0:
                    ready.append(b)
    return out


def _collar_spot(rng, cond: EpisodeConditions, r_out: float, taken: list[tuple[float, float]]):
    for _ in range(200):
        lx, ly = float(rng.uniform(-0.15, 0.15)), float(rng.uniform(-0.12, 0.12))
        if not C._start_clear(lx, ly, r_out, cond.slots, 0.004):
            continue
        if any(math.hypot(lx - x, ly - y) < 0.03 for x, y in taken):
            continue
        return lx, ly
    return float(rng.uniform(-0.15, 0.15)), float(rng.uniform(-0.12, 0.12))


def _extend_pegs(rng, cond: EpisodeConditions, n_extra: int) -> None:
    used = {p.slot for p in cond.pegs}
    free = [s for s in range(C.N_SLOTS) if s not in used]
    n_extra = min(n_extra, len(free))
    if n_extra <= 0:
        return
    extra = sorted(rng.choice(free, size=n_extra, replace=False).tolist())
    w = np.array([0.32, 0.12, 0.06, 0.50, 0.0])  # never a second grasped start
    new = C._draw_peg_starts_with_weights(rng, extra, cond.slots, w)
    for p in new:
        # Independent directions at the same magnitudes as an existing peg (do not copy biases).
        j = int(rng.integers(0, len(cond.pegs)))
        mag = math.hypot(*cond.slot_bias_list[j])
        ang = float(rng.uniform(0, 2 * math.pi))
        cond.slot_bias_list.append((mag * math.cos(ang), mag * math.sin(ang)))
        yaw_mag = abs(cond.slot_bias_yaw_list[j])
        cond.slot_bias_yaw_list.append(yaw_mag * (1.0 if rng.random() < 0.5 else -1.0))
        dmag = math.hypot(*cond.peg_bias_delta_list[j])
        dang = float(rng.uniform(0, 2 * math.pi))
        cond.peg_bias_delta_list.append((dmag * math.cos(dang), dmag * math.sin(dang)))
        ymag = abs(cond.peg_yaw_delta_list[j])
        cond.peg_yaw_delta_list.append(ymag * (1.0 if rng.random() < 0.5 else -1.0))
        cond.pegs.append(p)
    cond.n_pegs = len(cond.pegs)


def _step_wave_of(s: TaskStep, peg_wave: list[int], collars: list[CollarSpec]) -> int:
    """Wave index of the part a step acts on (collar/stack wave, else peg wave)."""
    if s.kind in ("collar", "stack"):
        return collars[s.part].wave
    return peg_wave[s.peg]


def _soften_noise(cond: EpisodeConditions) -> None:
    s = NOISE_SCALE
    cond.slot_bias_list = [(x * s["slot_bias"], y * s["slot_bias"]) for x, y in cond.slot_bias_list]
    cond.clutter_bias_list = [(x * s["slot_bias"], y * s["slot_bias"]) for x, y in cond.clutter_bias_list]
    cond.peg_bias_delta_list = [(x * s["peg_delta"], y * s["peg_delta"]) for x, y in cond.peg_bias_delta_list]
    cond.glitch_rate *= s["glitch"]
    cond.bias_drift *= s["drift"]


def _draw_event(rng: np.random.Generator) -> dict:
    kind = str(rng.choice(EVENT_KINDS, p=EVENT_WEIGHTS / EVENT_WEIGHTS.sum()))
    if kind == "friction":
        return dict(kind=kind, scale=float(rng.uniform(0.55, 1.6)))
    if kind == "gain":
        return dict(kind=kind, scale=float(rng.uniform(0.6, 1.4)))
    if kind == "mass":
        return dict(kind=kind, scale=float(rng.uniform(0.7, 2.0)))
    if kind == "delay":
        return dict(kind=kind, steps=int(rng.integers(0, 3)))
    if kind == "rebias":
        return dict(kind=kind, mag=float(rng.uniform(0.001, 0.004)), ang=float(rng.uniform(0, 2 * math.pi)))
    return dict(kind=kind, sigma=float(rng.uniform(0.08, 0.2)))


def make_compound(rng: np.random.Generator, cond: EpisodeConditions, tier: str) -> CompoundSpec:
    cfg = TIER_CFG[tier]
    _soften_noise(cond)
    _extend_pegs(rng, cond, cfg["extra_pegs"])
    K = cond.n_pegs
    # Waves: wave 0 always holds the pre-grasped peg (if any) and at least one peg.
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
    for i, p in enumerate(cond.pegs):  # a later-wave part arrives as a loose / lying part
        if peg_wave[i] > 0 and p.start_type in ("grasped", "tray"):
            p.start_type = "loose"
    # Collars on a random subset of narrow pegs.
    cands = [i for i, p in enumerate(cond.pegs) if p.spec.outer_radius() <= COLLAR_MAX_PEG_OUTER]
    n_col = min(MAX_COLLARS, len(cands), int(round(cfg["collar_frac"] * K)))
    rounds = [i for i, p in enumerate(cond.pegs) if p.spec.shape == "round"]
    n_tw = int(round(cfg["twist_frac"] * len(rounds)))
    if n_col == 0 and n_tw == 0:  # every episode is compound: at least one collar or twist
        if cands:
            n_col = 1
        elif rounds:
            n_tw = 1
    collars: list[CollarSpec] = []
    taken: list[tuple[float, float]] = []
    for i in (rng.choice(cands, size=n_col, replace=False).tolist() if n_col else []):
        r_in = cond.pegs[i].spec.outer_radius() + float(rng.uniform(0.0006, 0.0012))
        spot = _collar_spot(rng, cond, r_in + COLLAR_T, taken)
        taken.append(spot)
        w = int(rng.integers(peg_wave[i], W))
        collars.append(CollarSpec(peg=int(i), r_in=r_in, offset_xy=spot, wave=w))
    # Twists on round pegs.
    twist_pegs = rng.choice(rounds, size=n_tw, replace=False).tolist() if n_tw else []
    # Steps + mandatory edges.
    steps: list[TaskStep] = [TaskStep("insert", i, i) for i in range(K)]
    edges: set[tuple[int, int]] = set()
    for i in twist_pegs:
        steps.append(TaskStep("twist", i, i, twist_target=float(rng.uniform(*TWIST_TARGET)),
                              twist_dir=1.0 if rng.random() < 0.5 else -1.0))
        edges.add((i, len(steps) - 1))
    for j, col in enumerate(collars):
        steps.append(TaskStep("collar", j, col.peg))
        sc = len(steps) - 1
        edges.add((col.peg, sc))
        tw = [k for k, s in enumerate(steps) if s.kind == "twist" and s.peg == col.peg]
        if tw and rng.random() < 0.5:
            # Collar wave is ≥ peg wave; only allow collar→twist when waves are equal.
            tw_i = tw[0]
            if collars[j].wave > peg_wave[col.peg]:
                edges.add((tw_i, sc))
            else:
                edges.add((tw_i, sc) if rng.random() < 0.5 else (sc, tw_i))
    # Random extra precedence consistent with a random topological order.
    # Only a→b when wave(a) ≤ wave(b), so an honest policy is never forced to wait on a later wave.
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
    if grasped:  # the pre-grasped peg can always be inserted first
        edges = {(a, b) for a, b in edges if b != grasped[0]}
    # Hard invariant: no edge from a later-wave step to an earlier-wave step.
    edges = {
        (a, b)
        for a, b in edges
        if _step_wave_of(steps[a], peg_wave, collars) <= _step_wave_of(steps[b], peg_wave, collars)
    }
    for a, b in edges:
        steps[b].prereq.append(a)
    for s in steps:
        s.prereq.sort()
    # Horizon.
    budget = 0
    for s in steps:
        budget += STEP_BUDGET[s.kind]
        if s.kind == "insert" and cond.pegs[s.peg].start_type == "lying":
            budget += LYING_EXTRA
    max_steps = int(cfg["slack"] * budget) + WAVE_EXTRA * (W - 1)
    cond.max_steps = max_steps
    wave_time = [0] + [int(max_steps * min(0.85, w * float(rng.uniform(0.18, 0.32)))) for w in range(1, W)]
    for w in range(2, W):
        wave_time[w] = max(wave_time[w], wave_time[w - 1] + 20)
    events = [(_draw_event(rng) if rng.random() < cfg["event_p"] else None) for _ in steps]
    return CompoundSpec(tier=tier, steps=steps, collars=collars, peg_wave=peg_wave, n_waves=W,
                        wave_time=wave_time, events=events)


def sample_compound(seed: int, episodes_per_stratum: int, split: str = "live"):
    """v0.5 board draw (layouts, starts, dynamics) + a compound task graph per episode."""
    ranges, conds = sample_episodes(seed, episodes_per_stratum, split)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x43504E44, *C.build_salt()]))
    per = episodes_per_stratum
    for s in range(0, len(conds), per):
        tiers = _assign(rng, len(conds[s : s + per]))
        for c, t in zip(conds[s : s + per], tiers):
            c.compound = make_compound(rng, c, t)
    return ranges, conds


# ---- episode ------------------------------------------------------------------------------------
def _collar_xml(j: int, col: CollarSpec) -> str:
    rc = col.r_in + COLLAR_T / 2
    half_tan = (col.r_in + COLLAR_T) * math.tan(math.pi / COLLAR_NSEG) + 1e-5
    segs = []
    for k in range(COLLAR_NSEG):
        th = 2 * math.pi * k / COLLAR_NSEG
        segs.append(
            f'<geom name="collar{j}_{k}" type="box" pos="{rc * math.cos(th):.6f} {rc * math.sin(th):.6f} 0" '
            f'euler="0 0 {th:.6f}" size="{COLLAR_T / 2:.6f} {half_tan:.6f} {COLLAR_H / 2}" class="part" '
            f'rgba="0.25 0.45 0.85 1"/>'
        )
    return f"""
    <body name="collar{j}" pos="0 0 0.3">
      <freejoint name="collar{j}"/>
      <inertial pos="0 0 0" mass="0.008" diaginertia="4e-7 4e-7 6e-7"/>
      {"".join(segs)}
    </body>"""


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def quat_yaw(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class CompoundEpisode(Episode):
    """A v0.5 Episode plus collars, task graph, waves and inter-step events."""

    def __init__(self, cond: EpisodeConditions, rng: np.random.Generator):
        self.cs: CompoundSpec = cond.compound
        self.n_collars = len(self.cs.collars)
        self.peg_parked = [w > 0 for w in self.cs.peg_wave]
        self.collar_parked = [c.wave > 0 for c in self.cs.collars]
        self._ids_ready = False
        super().__init__(cond, rng)
        n = len(self.cs.steps)
        self.step_done = [False] * n
        self.step_done_at: list[int | None] = [None] * n
        self.step_in_order = [False] * n
        self.step_progress = [0.0] * n
        self.n_completions = 0
        self.events_fired: list[dict] = []
        self.wave_arrived = [True] + [False] * (self.cs.n_waves - 1)
        self.collar_run = [0] * self.n_collars
        self.parts_lost = 0
        self.twist_acc = [0.0] * self.n_pegs
        self._twist_prev = [None] * self.n_pegs
        # Collar estimate glitch / dropout (same burst process as peg rows).
        self._glitch_collar = np.zeros((max(1, self.n_collars), 2))
        self._last_collar_obs = [None] * self.n_collars
        self._release_lists()

    # -- model ---------------------------------------------------------------------------------
    def _model_xml(self, cond: EpisodeConditions) -> str:
        xml = build_xml(cond)
        collars = "".join(_collar_xml(j, c) for j, c in enumerate(cond.compound.collars))
        return xml.replace("  </worldbody>", collars + "\n  </worldbody>", 1)

    def _all_seated_ends_episode(self) -> bool:
        return False  # compound episodes end when every step is done (see step)

    def _setup_ids(self) -> None:
        if self._ids_ready:
            return
        m = self.model
        self.collar_qadr, self.collar_vadr, self.collar_body = [], [], []
        for j in range(self.n_collars):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"collar{j}")
            self.collar_qadr.append(m.jnt_qposadr[jid])
            self.collar_vadr.append(m.jnt_dofadr[jid])
            self.collar_body.append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"collar{j}"))
        self._geom_contype = m.geom_contype.copy()
        self._geom_conaff = m.geom_conaffinity.copy()
        # Compile-time body collision masks (OR of the body's geom masks, all parts active).
        self._body_contype = m.body_contype.copy()
        self._body_conaff = m.body_conaffinity.copy()
        self._base_friction = m.geom_friction[:, 0].copy()
        self._base_mass = m.body_mass.copy()
        self._base_inertia = m.body_inertia.copy()
        self._base_gain = m.actuator_gainprm[:, 0].copy()
        self._base_bias = m.actuator_biasprm[:, 1].copy()
        self._ids_ready = True

    def _release_lists(self) -> None:
        pass

    def _set_body_active(self, body: int, active: bool) -> None:
        m = self.model
        g0, ng = m.body_geomadr[body], m.body_geomnum[body]
        for g in range(g0, g0 + ng):
            m.geom_contype[g] = self._geom_contype[g] if active else 0
            m.geom_conaffinity[g] = self._geom_conaff[g] if active else 0
        m.body_gravcomp[body] = 0.0 if active else 1.0

    def _park_pose(self) -> list[float]:
        bx, by = self.c.board_xy
        return [bx, by, PARK_Z]

    def collar_pose(self, j: int):
        q = self.collar_qadr[j]
        return self.data.qpos[q : q + 3].copy(), self.data.qpos[q + 3 : q + 7].copy()

    def _place_collar(self, j: int) -> None:
        col = self.cs.collars[j]
        bx, by = self.c.board_xy
        c, s = math.cos(self.c.board_yaw), math.sin(self.c.board_yaw)
        ox, oy = col.offset_xy
        self._set_free(self.collar_qadr[j], self.collar_vadr[j],
                       [bx + c * ox - s * oy, by + s * ox + c * oy, COLLAR_H / 2 + 1e-4], [1.0, 0, 0, 0])

    def _pin_parked(self) -> None:
        for i, parked in enumerate(self.peg_parked):
            if parked:
                self._set_free(self.peg_qadr[i], self.peg_vadr[i], self._park_pose(), [1.0, 0, 0, 0])
        for j, parked in enumerate(self.collar_parked):
            if parked:
                self._set_free(self.collar_qadr[j], self.collar_vadr[j], self._park_pose(), [1.0, 0, 0, 0])

    # -- spawn (parent hooks) ------------------------------------------------------------------------
    def _place_peg(self, i: int, peg: PegStart) -> None:
        if self.peg_parked[i]:
            self._set_free(self.peg_qadr[i], self.peg_vadr[i], self._park_pose(), [1.0, 0, 0, 0])
            return
        super()._place_peg(i, peg)

    def _intended_peg_xy(self, peg: PegStart) -> tuple[float, float]:
        i = self.c.pegs.index(peg)
        if self.peg_parked[i]:
            p = self._park_pose()
            return p[0], p[1]
        return super()._intended_peg_xy(peg)

    def _place_and_settle(self) -> None:
        self._setup_ids()
        for i in range(self.n_pegs):
            self._set_body_active(self.peg_body[i], not self.peg_parked[i])
        for j in range(self.n_collars):
            self._set_body_active(self.collar_body[j], not self.collar_parked[j])
            if self.collar_parked[j]:
                self._set_free(self.collar_qadr[j], self.collar_vadr[j], self._park_pose(), [1.0, 0, 0, 0])
            else:
                self._place_collar(j)
        super()._place_and_settle()
        for j in range(self.n_collars):
            self.data.qvel[self.collar_vadr[j] : self.collar_vadr[j] + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    # -- arrivals and events -----------------------------------------------------------------------
    def _active_xy(self) -> list[np.ndarray]:
        pts = [self.data.qpos[0:2].copy()]
        for i in range(self.n_pegs):
            if not self.peg_parked[i]:
                pts.append(self.peg_pose(i)[0][0:2])
        for j in range(self.n_collars):
            if not self.collar_parked[j]:
                pts.append(self.collar_pose(j)[0][0:2])
        for i in range(self.n_clutter):
            pts.append(self.clutter_pose(i)[0][0:2])
        return pts

    def _free_world_spot(self, lx: float, ly: float, r: float) -> tuple[float, float]:
        bx, by = self.c.board_xy
        c, s = math.cos(self.c.board_yaw), math.sin(self.c.board_yaw)
        pts = self._active_xy()
        for k in range(120):
            if k > 0:
                lx, ly = float(self.rng.uniform(-0.15, 0.15)), float(self.rng.uniform(-0.12, 0.12))
            wx, wy = bx + c * lx - s * ly, by + s * lx + c * ly
            if not C._start_clear(lx, ly, r, self.c.slots, 0.003):
                continue
            if all(float(np.hypot(wx - p[0], wy - p[1])) >= 0.028 for p in pts):
                return lx, ly
        return lx, ly

    def _arrive(self, w: int) -> None:
        self.wave_arrived[w] = True
        for i in range(self.n_pegs):
            if self.peg_parked[i] and self.cs.peg_wave[i] == w:
                peg = self.c.pegs[i]
                hx, hy = peg.spec.x + peg.offset_xy[0], peg.spec.y + peg.offset_xy[1]
                lx, ly = self._free_world_spot(hx, hy, peg.spec.outer_radius() + 0.004)
                peg.offset_xy = (lx - peg.spec.x, ly - peg.spec.y)
                self.peg_parked[i] = False
                self._set_body_active(self.peg_body[i], True)
                super()._place_peg(i, peg)
        for j in range(self.n_collars):
            col = self.cs.collars[j]
            if self.collar_parked[j] and col.wave == w:
                col.offset_xy = self._free_world_spot(col.offset_xy[0], col.offset_xy[1], col.r_out + 0.004)
                self.collar_parked[j] = False
                self._set_body_active(self.collar_body[j], True)
                self._place_collar(j)
        mujoco.mj_forward(self.model, self.data)

    def _fire_event(self, ev: dict) -> None:
        m, kind = self.model, ev["kind"]
        part_bodies = set(self.peg_body) | set(self.collar_body) | set(self.clutter_body)
        if kind == "friction":
            for g in range(m.ngeom):
                if m.geom_bodyid[g] in part_bodies:
                    m.geom_friction[g, 0] = self._base_friction[g] * ev["scale"]
        elif kind == "gain":
            for a in range(3):
                m.actuator_gainprm[a, 0] = self._base_gain[a] * ev["scale"]
                m.actuator_biasprm[a, 1] = self._base_bias[a] * ev["scale"]
        elif kind == "mass":
            for b in self.peg_body:
                # Very light pegs (< 8 g) with tiny inertia blow up in contact; clamp the result.
                new = float(np.clip(self._base_mass[b] * ev["scale"], MASS_MIN, MASS_MAX))
                ratio = new / max(1e-9, float(self._base_mass[b]))
                m.body_mass[b] = new
                m.body_inertia[b] = self._base_inertia[b] * ratio
            # v0.6.2: refresh the derived constants MuJoCo computes from mass/inertia at compile
            # time (body_subtreemass, dof_M0, body/dof_invweight0). Stale values made the next grasp
            # oscillate under implicitfast and could end the episode "unstable" (reported by
            # msulab). A scratch MjData is used because mj_setConst runs kinematics at qpos0 on the
            # data it is given; the live episode state is untouched. Nothing else in the model is
            # changed (parking, friction and gain edits are not derived quantities).
            mujoco.mj_setConst(m, mujoco.MjData(m))
            # v0.6.3: mj_setConst also re-derives body_contype/body_conaffinity from the *current*
            # geom masks. Parked (later-wave) parts have their geoms switched off, so their body
            # masks became 0 and stayed 0 after the wave arrived: the broad phase skipped them and
            # they fell through the board (reported by OddK). Restore the compile-time body masks;
            # parked geoms are still filtered at the geom level.
            m.body_contype[:] = self._body_contype
            m.body_conaffinity[:] = self._body_conaff
        elif kind == "delay":
            self._act_delay = int(ev["steps"])
            self.c.act_delay = self._act_delay  # privileged tooling reads the cond
            self._act_buf = deque([np.zeros(ACT_DIM) for _ in range(self._act_delay)])
        elif kind == "rebias":
            d = np.array([math.cos(ev["ang"]), math.sin(ev["ang"])]) * ev["mag"]
            self._slot_bias_xy = self._slot_bias_xy + d
            self._peg_bias_xy = self._peg_bias_xy + d
        elif kind == "shove":
            loose = [self.peg_vadr[i] for i in range(self.n_pegs) if not self.peg_parked[i] and not self.seated_now[i]]
            loose += [self.clutter_vadr[i] for i in range(self.n_clutter)]
            if loose:
                v = loose[int(self.rng.integers(0, len(loose)))]
                self.data.qvel[v : v + 3] += self.rng.normal(0, ev["sigma"], 3)
        self.events_fired.append(dict(ev, step=self.steps))

    # -- predicates ----------------------------------------------------------------------------------
    def _collar_seated_now(self, j: int) -> bool:
        if self.collar_parked[j]:
            return False
        col = self.cs.collars[j]
        p, q = self.collar_pose(j)
        hx, hy = self.slot_xy[col.peg]
        return (
            self.seated_now[col.peg]
            and float(np.hypot(p[0] - hx, p[1] - hy)) <= COLLAR_SEAT_XY
            and p[2] - COLLAR_H / 2 <= BLOCK_H + COLLAR_SEAT_Z
            and self._upright(q)
            and not self._gripper_touching(self.collar_body[j])
        )

    def _peg_in_hole(self, i: int) -> bool:
        p, _ = self.peg_pose(i)
        hx, hy = self.slot_xy[i]
        return p[2] - PEG_LEN / 2 <= SEAT_TOL_Z + 0.002 and float(np.hypot(p[0] - hx, p[1] - hy)) <= self.r_hole[i]

    def _step_state(self, k: int) -> tuple[bool, float]:
        """(done now, partial-progress milestone in [0, PARTIAL_CAP])."""
        s = self.cs.steps[k]
        if s.kind == "insert":
            i = s.part
            if self.peg_parked[i]:
                return False, 0.0
            # Done once seated (gripper clear); stays done while the peg remains in its hole.
            if self.seated_now[i] or (self.step_done[k] and self._peg_in_hole(i)):
                return True, PARTIAL_CAP
            p, _ = self.peg_pose(i)
            hx, hy = self.slot_xy[i]
            bottom = p[2] - PEG_LEN / 2
            dxy = float(np.hypot(p[0] - hx, p[1] - hy))
            prog = 0.0
            if self._gripper_touching(self.peg_body[i]) and bottom > 0.003:
                prog = 0.05
            if dxy < 0.003 and bottom < BLOCK_H + 0.010:
                prog = 0.15
            if dxy <= self.r_hole[i] and bottom < BLOCK_H:
                prog = 0.15 + 0.15 * (1.0 - max(0.0, bottom) / BLOCK_H)
            return False, prog
        if s.kind == "twist":
            i = s.part
            frac = float(np.clip(self.twist_acc[i] * s.twist_dir / s.twist_target, 0.0, 1.0))
            done = frac >= 1.0 and (self.seated_now[i] or (self.step_done[k] and self._peg_in_hole(i)))
            return done, PARTIAL_CAP * frac
        j = s.part
        if self.collar_parked[j]:
            return False, 0.0
        p, _ = self.collar_pose(j)
        hx, hy = self.slot_xy[self.cs.collars[j].peg]
        bottom = p[2] - COLLAR_H / 2
        dxy = float(np.hypot(p[0] - hx, p[1] - hy))
        peg_i = self.cs.collars[j].peg
        peg_ok = self._peg_in_hole(peg_i)
        still = dxy <= COLLAR_SEAT_XY and bottom <= BLOCK_H + COLLAR_SEAT_Z and peg_ok
        done = self.collar_run[j] >= SEATED_HOLD or (self.step_done[k] and still)
        prog = 0.0
        if self._gripper_touching(self.collar_body[j]) and bottom > 0.003:
            prog = 0.05
        # 0.20 / 0.30 tiers require the collar's peg to be in its hole (empty-slot ring ≠ near seat).
        if peg_ok and dxy < COLLAR_SEAT_XY and bottom < BLOCK_H + PEG_LEN - BLOCK_H:
            prog = 0.20
        if peg_ok and dxy < COLLAR_SEAT_XY and bottom < BLOCK_H + 0.010:
            prog = PARTIAL_CAP
        return done, prog

    # -- step ----------------------------------------------------------------------------------------
    def fail(self, reason: str) -> None:
        """A part that leaves the work area is lost (its steps can no longer finish); the episode goes on.

        Numerical blow-ups (``unstable``) still end the episode.
        """
        if reason != "part_lost":
            return super().fail(reason)
        lost = False
        for i in range(self.n_pegs):
            p, _ = self.peg_pose(i)
            if not self.peg_parked[i] and (abs(p[0]) > 0.30 or abs(p[1]) > 0.30 or p[2] < -0.01):
                self.peg_parked[i] = True
                self.parts_lost += 1
                self._set_body_active(self.peg_body[i], False)
                lost = True
        for j in range(self.n_collars):
            p, _ = self.collar_pose(j)
            if not self.collar_parked[j] and (abs(p[0]) > 0.30 or abs(p[1]) > 0.30 or p[2] < -0.01):
                self.collar_parked[j] = True
                self.parts_lost += 1
                self._set_body_active(self.collar_body[j], False)
                lost = True
        if not lost:  # clutter or something else left the area: keep v0.5 semantics
            return super().fail(reason)
        self._pin_parked()
        mujoco.mj_forward(self.model, self.data)

    def step(self, action: np.ndarray) -> None:
        if self.done:
            return
        super().step(action)
        self._pin_parked()
        if self.terminal_reason in ("unstable", "part_lost"):
            return
        if self.done and self.terminal_reason == "running":
            self.terminal_reason = "time_limit"
        d = self.data
        for j in range(self.n_collars):
            if self.collar_parked[j]:
                continue
            p, _ = self.collar_pose(j)
            v = d.qvel[self.collar_vadr[j] : self.collar_vadr[j] + 3]
            if float(np.linalg.norm(v)) > 4.0:
                return self.fail("unstable")
            if abs(p[0]) > 0.30 or abs(p[1]) > 0.30 or p[2] < -0.01:
                return self.fail("part_lost")
        # Twist accumulators (signed yaw while the peg sits in its hole).
        for i in range(self.n_pegs):
            if self.peg_parked[i]:
                continue
            y = quat_yaw(self.peg_pose(i)[1])
            if self._peg_in_hole(i) and self._twist_prev[i] is not None:
                self.twist_acc[i] += wrap(y - self._twist_prev[i])
            self._twist_prev[i] = y
        for j in range(self.n_collars):
            # Stack rings seat on the peg top; AssemblyIIIEpisode updates their hold
            # counters. Resetting them from the block-seat test would pin the run at 0/1.
            if self.cs.collars[j].is_stack:
                continue
            self.collar_run[j] = self.collar_run[j] + 1 if self._collar_seated_now(j) else 0
        # Step bookkeeping.
        for k, s in enumerate(self.cs.steps):
            # v0.7 hard steps: AssemblyIIIEpisode re-evaluates after correcting screw /
            # stack / key state. Do not un-mark or fire their completion here — that
            # would flicker done_at (and double-count key hold via _step_state).
            if s.kind in ("screw", "key", "stack", "rework"):
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
                self.step_done[k] = False  # knocked out / undone
        # Waves.
        for w in range(1, self.cs.n_waves):
            if self.wave_arrived[w]:
                continue
            earlier = [k for k, s in enumerate(self.cs.steps) if self._step_wave(s) < w]
            if self.steps >= self.cs.wave_time[w] or all(self.step_done[k] for k in earlier):
                self._arrive(w)
            break  # waves arrive in order
        if all(self.step_done) and all(self.wave_arrived) and not self.done:
            self.done = True
            self.success = True
            self.success_step = self.steps
            self.terminal_reason = "assembled"
        elif self.done and self.terminal_reason in ("time_limit", "assembled"):
            # The base class labels "all pegs seated" as assembled; a compound episode is assembled
            # only when every step is done.
            self.success = all(self.step_done) and all(self.wave_arrived)
            self.terminal_reason = "assembled" if self.success else "time_limit"
            self.success_step = self.steps if self.success else None

    def _step_wave(self, s: TaskStep) -> int:
        if s.kind in ("collar", "stack"):
            return self.cs.collars[s.part].wave
        return self.cs.peg_wave[s.peg]

    def step_blocked(self, k: int) -> bool:
        return not all(self.step_done[p] for p in self.cs.steps[k].prereq)

    def _glitch_step(self, noise_rng: np.random.Generator) -> float:
        """Extend peg/slot/clutter glitch bursts with independent collar offsets."""
        was_active = self._glitch_left > 0
        gain = super()._glitch_step(noise_rng)
        if self._glitch_left == 0:
            self._glitch_collar[:] = 0.0
        elif not was_active:
            n = self._glitch_collar.shape[0]
            mag = noise_rng.uniform(GLITCH_OFFSET[0], GLITCH_OFFSET[1], n)
            ang = noise_rng.uniform(0.0, 2 * math.pi, n)
            self._glitch_collar = np.stack([mag * np.cos(ang), mag * np.sin(ang)], axis=1)
        return gain

    # -- observation -----------------------------------------------------------------------------------
    def observe(self, noise_rng: np.random.Generator) -> np.ndarray:
        base = super().observe(noise_rng).astype(np.float64)
        # Parent already advanced the glitch burst; gain matches peg-row treatment this step.
        glitch_gain = GLITCH_NOISE_GAIN if self._glitch_left > 0 else 1.0
        p0 = 16 + ACT_DIM
        for i in range(self.n_pegs):
            if self.peg_parked[i]:
                base[p0 + i * PEG_FEAT : p0 + (i + 1) * PEG_FEAT] = 0.0
        task = np.zeros(MAX_PEGS * TASK_PEG_FEAT)
        for k, s in enumerate(self.cs.steps):
            b = s.peg * TASK_PEG_FEAT
            if s.kind == "insert":
                task[b + 0] = float(self.step_done[k])
                task[b + 1] = float(self.step_blocked(k))
            elif s.kind == "twist":
                task[b + 2] = s.twist_dir
                task[b + 3] = float(np.clip(self.twist_acc[s.peg] * s.twist_dir / s.twist_target, 0.0, 1.0))
                task[b + 4] = float(self.step_blocked(k))
            elif s.kind in ("collar", "stack"):
                # Collar flag only — do not write 1.0 for key/screw/rework (those reuse
                # twist channels; key_tol packs into b+5 when this flag is unused).
                task[b + 5] = 1.0 if s.kind == "collar" else 2.0
        col = np.zeros(MAX_COLLARS * COLLAR_FEAT)
        for k, s in enumerate(self.cs.steps):
            if s.kind not in ("collar", "stack"):
                continue
            j, i = s.part, s.peg
            b = j * COLLAR_FEAT
            if self.collar_parked[j]:
                continue
            p, q = self.collar_pose(j)
            pn, qn = self._noisy_pose(
                p, q, noise_rng,
                bias_xy=self._peg_bias_xy[i] + self._glitch_collar[j],
                yaw_bias=0.0,
                gain=glitch_gain,
            )
            hx, hy = self.slot_xy[i]
            sn = self.c.obs_noise * glitch_gain
            hx_e = hx + self._slot_bias_xy[i][0] + self._glitch_slot[i][0] + float(noise_rng.normal(0, sn))
            hy_e = hy + self._slot_bias_xy[i][1] + self._glitch_slot[i][1] + float(noise_rng.normal(0, sn))
            c = self.cs.collars[j]
            # present: 1 = collar (block seat), 2 = stack (peg-top seat). v0.6 only ever writes 1.
            present = 2.0 if s.kind == "stack" else 1.0
            feat = np.array(
                [present, (pn[0] - hx_e) * 10, (pn[1] - hy_e) * 10, (pn[2] - BLOCK_H) * 10,
                 *qn, c.r_in * 100, c.r_out * 100, i / MAX_PEGS,
                 float(self.step_done[k]), float(self.step_blocked(k))],
                dtype=np.float64,
            )
            if noise_rng.random() < self.c.dropout_prob and self._last_collar_obs[j] is not None:
                feat = self._last_collar_obs[j]
            else:
                self._last_collar_obs[j] = feat.copy()
            col[b : b + COLLAR_FEAT] = feat
        glob = np.array([
            self.steps / max(1, self.c.max_steps),
            sum(self.step_done) / max(1, len(self.step_done)),
            sum(self.wave_arrived) / self.cs.n_waves,
            len(self.cs.steps) / 16.0,
        ])
        o = np.concatenate([base, task, col, glob])
        assert o.shape[0] == OBS_DIM_V06, o.shape
        return o.astype(np.float32)


# ---- scoring ----------------------------------------------------------------------------------------
def in_order_credit(done_at: int | None, horizon: int) -> float:
    """Full in-order done credit: ``DONE_BASE + DONE_TIME * (1 - done_at/horizon)``."""
    t = 0.0 if done_at is None else max(0.0, 1.0 - done_at / max(1, horizon))
    return DONE_BASE + DONE_TIME * t


def step_credit(done: bool, in_order: bool, done_at: int | None, horizon: int, progress: float) -> float:
    if done:
        full = in_order_credit(done_at, horizon)
        if not in_order:
            # Never pay less for finishing out of order than for stopping just short of done.
            return max(PARTIAL_CAP, OUT_OF_ORDER * full)
        return full
    return min(PARTIAL_CAP, progress)


def step_results(ep: CompoundEpisode) -> list[dict]:
    """Per-step result rows for ``nist_lite_record/4`` (aligned with ``task.steps``)."""
    rows = []
    for k, s in enumerate(ep.cs.steps):
        cr = step_credit(
            ep.step_done[k], ep.step_in_order[k], ep.step_done_at[k], ep.c.max_steps, ep.step_progress[k]
        )
        rows.append(
            {
                "done": bool(ep.step_done[k]),
                "done_at": ep.step_done_at[k],
                "in_order": bool(ep.step_in_order[k]),
                "credit": round(float(cr), 5),
            }
        )
    return rows


def record_events(ep: CompoundEpisode) -> list[dict]:
    """Events that fired, for ``nist_lite_record/4`` (``t`` = control step, world-frame sim)."""
    out = []
    for ev in ep.events_fired:
        row = {"t": int(ev.get("step", 0)), "kind": ev["kind"]}
        for key, val in ev.items():
            if key in ("step", "kind"):
                continue
            row[key] = val
        out.append(row)
    return out


def compound_score(ep: CompoundEpisode) -> dict:
    total = wsum = 0.0
    per_kind: dict[str, list[float]] = {k: [] for k in STEP_KINDS}
    for k, s in enumerate(ep.cs.steps):
        cr = step_credit(ep.step_done[k], ep.step_in_order[k], ep.step_done_at[k], ep.c.max_steps,
                         ep.step_progress[k])
        w = STEP_WEIGHT[s.kind]
        total += w * cr
        wsum += w
        per_kind[s.kind].append(cr)
    return dict(
        score=total / wsum if wsum else 0.0,
        n_steps=len(ep.cs.steps),
        n_done=int(sum(ep.step_done)),
        n_in_order=int(sum(1 for d, o in zip(ep.step_done, ep.step_in_order) if d and o)),
        per_kind={k: (round(float(np.mean(v)), 3) if v else None) for k, v in per_kind.items()},
        events=len(ep.events_fired),
    )
