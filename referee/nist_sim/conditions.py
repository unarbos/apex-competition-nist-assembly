"""Per-round condition sampling for the multi-peg board. Everything derives from SEED.

Strata are (K, clutter_bucket) with K in {3,4,5,6} and clutter_bucket in {low, high}.
Eight strata × episodes_per_stratum (default 8) = 64 episodes; live rounds use 16 per
stratum (128 episodes). Mean K = 4.5.

Within each stratum a Latin hypercube covers the live ranges, then each episode is
assigned a **difficulty tier** so every round has a learnable gradient:

- ~25% **easy** — grasped / near-hole starts, low noise & residual bias, little clutter
- ~35% **medium** — mixed upright starts, moderate anti-scripting
- ~40% **hard** — full residual bias / drift / disturbances; reserved split puts
  held-out shapes here

Observation hardening (v0.5.1): larger per-peg estimate residuals, faster estimate
drift, short estimate "glitch" bursts (extra noise plus a transient per-object offset)
and more lying-down starts on the medium / hard tiers.

Board layout (v0.5): every episode draws its own 8-slot layout -- slot positions, slot
yaw, peg cross-section family and size -- from continuous distributions. There is no
fixed public slot table and no shape label in the observation.

Held-out: reserved LHS sub-ranges plus the stadium / trapezoid cross-section families
(reserved split only).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

RESERVED_PATH = Path(os.environ.get("NIST_RESERVED_PATH", Path(__file__).with_name("reserved.json")))

# Board slots (v0.5): drawn per episode, see ``draw_layout``. Live cross-section
# families; "rect" covers squares (aspect 1) and rectangles, "ngon" regular polygons,
# "dflat" a cylinder with a flat chord. Held-out families appear only on the reserved split.
N_SLOTS = 8
MAX_CLUTTER = 3
K_VALUES = (3, 4, 5, 6)
CLUTTER_BUCKETS = ("low", "high")  # low -> 0-1 blocks; high -> 2-3
PEG_START_TYPES = ("loose", "lying", "tray", "near_hole", "grasped")
LIVE_SHAPES = ("round", "rect", "ngon", "dflat")
LIVE_SHAPE_WEIGHTS = (0.28, 0.27, 0.25, 0.20)
NGON_SIDES = (5, 6, 8)
HELD_OUT_SHAPES = ("stadium", "trapezoid")  # reserved split only
SHAPE_NAMES = LIVE_SHAPES + HELD_OUT_SHAPES
# Inscribed radius (m) of the peg cross-section: 2-8 mm (4-16 mm across flats).
SIZE_RADIUS = (0.002, 0.008)
# Board-local region for slot centres and the minimum wall-to-wall gap between slots.
LAYOUT_X = (-0.105, 0.105)
LAYOUT_Y = (-0.072, 0.072)
LAYOUT_GAP = 0.010
LAYOUT_WALL = 0.006  # hole wall thickness (mirrors env.WALL_T)


@dataclass
class SlotSpec:
    """One board slot: cross-section family, size and board-local pose.

    ``size`` is the inscribed diameter (across flats). ``params`` holds the family
    parameters (``aspect`` for rect/stadium, ``n`` for ngon, ``flat`` for dflat,
    ``taper`` for trapezoid).
    """

    shape: str
    size: float
    x: float
    y: float
    yaw: float
    params: dict = field(default_factory=dict)

    @property
    def r(self) -> float:
        return self.size / 2

    def polygon(self) -> list[tuple[float, float]] | None:
        """Convex cross-section (counter-clockwise, slot frame). None for round."""
        return shape_polygon(self.shape, self.r, self.params)

    def outer_radius(self) -> float:
        poly = self.polygon()
        if poly is None:
            return self.r
        return max(math.hypot(x, y) for x, y in poly)

    def half_width_y(self) -> float:
        """Half-extent along the slot-frame y axis (the gripper closing axis)."""
        poly = self.polygon()
        if poly is None:
            return self.r
        return max(abs(y) for _, y in poly)

    def symmetry(self) -> int:
        """Rotational symmetry order (0 = any yaw). Privileged tooling only."""
        if self.shape == "round":
            return 0
        if self.shape == "rect":
            return 4 if abs(self.params.get("aspect", 1.0) - 1.0) < 1e-6 else 2
        if self.shape == "ngon":
            return int(self.params["n"])
        if self.shape == "stadium":
            return 2
        return 1

    def public(self) -> dict:
        return {
            "shape": self.shape,
            "size": round(float(self.size), 6),
            "xy": [round(float(self.x), 6), round(float(self.y), 6)],
            "yaw": round(float(self.yaw), 6),
            "params": {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in self.params.items()},
        }


def shape_polygon(shape: str, r: float, params: dict) -> list[tuple[float, float]] | None:
    if shape == "round":
        return None
    if shape == "rect":
        ax = r * float(params.get("aspect", 1.0))
        return [(ax, -r), (ax, r), (-ax, r), (-ax, -r)]
    if shape == "ngon":
        n = int(params["n"])
        rc = r / math.cos(math.pi / n)
        return [(rc * math.cos(2 * math.pi * k / n + math.pi / n), rc * math.sin(2 * math.pi * k / n + math.pi / n)) for k in range(n)]
    if shape == "dflat":
        # Circle of radius r with a flat chord at x = flat * r (size = 2 r).
        f = float(params["flat"])
        th0 = math.acos(f)
        m = 14
        pts = [(r * math.cos(th0 + (2 * math.pi - 2 * th0) * k / m), r * math.sin(th0 + (2 * math.pi - 2 * th0) * k / m)) for k in range(m + 1)]
        return pts
    if shape == "stadium":
        half = r * float(params.get("aspect", 1.5))
        L = half - r
        pts = []
        m = 6
        for k in range(m + 1):
            a = -math.pi / 2 + math.pi * k / m
            pts.append((L + r * math.cos(a), r * math.sin(a)))
        for k in range(m + 1):
            a = math.pi / 2 + math.pi * k / m
            pts.append((-L + r * math.cos(a), r * math.sin(a)))
        return pts
    if shape == "trapezoid":
        t = float(params.get("taper", 0.6))
        w = r * 1.25
        return [(r, -w * t), (r, w * t), (-r, w), (-r, -w)]
    raise ValueError(shape)


def _draw_slot_shape(rng: np.random.Generator, family: str) -> tuple[float, dict]:
    lo, hi = SIZE_RADIUS
    if family == "round":
        return 2 * _u(rng, lo, hi), {}
    if family == "rect":
        aspect = 1.0 if rng.random() < 0.35 else _u(rng, 1.0, 1.7)
        r = _u(rng, lo, hi / aspect if aspect > 1.25 else hi)
        return 2 * r, {"aspect": aspect}
    if family == "ngon":
        n = int(rng.choice(NGON_SIDES))
        return 2 * _u(rng, lo + 0.0005, hi - 0.0005), {"n": n}
    if family == "dflat":
        return 2 * _u(rng, lo + 0.0005, hi), {"flat": _u(rng, 0.45, 0.8)}
    if family == "stadium":
        aspect = _u(rng, 1.3, 1.8)
        return 2 * _u(rng, lo, hi / aspect + 0.001), {"aspect": aspect}
    if family == "trapezoid":
        return 2 * _u(rng, lo + 0.0005, hi - 0.001), {"taper": _u(rng, 0.45, 0.75)}
    raise ValueError(family)


def fixed_test_layout() -> list[SlotSpec]:
    """Deterministic two-row board (round row, square row; 4/8/12/16 mm) for hand-built
    unit-test conditions only. Never used by ``sample_episodes``."""
    out = []
    for shape, y in (("round", 0.045), ("rect", -0.045)):
        for size, x in zip((0.004, 0.008, 0.012, 0.016), (-0.075, -0.025, 0.025, 0.075)):
            out.append(SlotSpec(shape=shape, size=size, x=x, y=y, yaw=0.0, params={"aspect": 1.0} if shape == "rect" else {}))
    return out


def draw_layout(rng: np.random.Generator, held_out_slot: int | None = None) -> list[SlotSpec]:
    """Per-episode 8-slot layout: families, sizes, positions and yaws.

    Positions are rejection-sampled inside the board region with a minimum
    wall-to-wall gap; slot order is random, so the slot index carries no geometry.
    """
    fams = list(rng.choice(LIVE_SHAPES, size=N_SLOTS, p=np.array(LIVE_SHAPE_WEIGHTS) / sum(LIVE_SHAPE_WEIGHTS)))
    if held_out_slot is not None:
        fams[held_out_slot] = str(rng.choice(HELD_OUT_SHAPES))
    specs: list[SlotSpec] = []
    for fam in fams:
        size, params = _draw_slot_shape(rng, str(fam))
        specs.append(SlotSpec(shape=str(fam), size=float(size), x=0.0, y=0.0, yaw=_u(rng, -math.pi, math.pi), params=params))
    # Place larger slots first. If any slot can't be placed without overlap, the
    # whole placement is redrawn (v0.5.1: the old fallback accepted the overlap).
    order = sorted(range(N_SLOTS), key=lambda i: -specs[i].outer_radius())
    for _layout_try in range(LAYOUT_REDRAWS):
        if _place_slots(rng, specs, order):
            return specs
    raise RuntimeError("draw_layout: could not place 8 non-overlapping slots")


LAYOUT_REDRAWS = 50


def _slots_clear(specs: list[SlotSpec], i: int, x: float, y: float, placed: list[int], gap: float) -> bool:
    ro = specs[i].outer_radius() + LAYOUT_WALL + 0.001
    return all(
        math.hypot(x - specs[j].x, y - specs[j].y) >= ro + specs[j].outer_radius() + LAYOUT_WALL + 0.001 + gap
        for j in placed
    )


def _place_slots(rng: np.random.Generator, specs: list[SlotSpec], order: list[int]) -> bool:
    """One placement pass; True iff every slot found a non-overlapping position."""
    placed: list[int] = []
    for i in order:
        ok = False
        for attempt in range(400):
            x = _u(rng, LAYOUT_X[0], LAYOUT_X[1])
            y = _u(rng, LAYOUT_Y[0], LAYOUT_Y[1])
            gap = LAYOUT_GAP if attempt < 300 else LAYOUT_GAP * 0.5
            if _slots_clear(specs, i, x, y, placed, gap):
                ok = True
                break
        if not ok:
            return False
        specs[i].x, specs[i].y = x, y
        placed.append(i)
    return True


# Live ranges. Continuous axes are LHS-drawn; discrete axes are mapped from LHS then rounded.
# Clearance floor is 0.25 mm: sub-0.1 mm square fits were unstable under MuJoCo soft contact.
RANGES = {
    "clearance": (0.00025, 0.0008),  # 0.25–0.8 mm; needs wiggle on the tight end
    "friction": (0.3, 1.3),
    "peg_mass": (0.005, 0.05),
    "gain_scale": (0.6, 1.4),
    "obs_noise": (0.0008, 0.0035),  # 0.8–3.5 mm white noise on pose estimates (v0.5.1)
    "obs_noise_rad": (0.015, 0.06),
    "dropout_prob": (0.03, 0.18),
    # Delay/backlash stay mild so the privileged solver can prove solvability;
    # anti-scripting comes primarily from pose-estimate bias (below).
    "act_delay": (0.0, 0.45),  # rounds to 0 almost always
    "disturbance": (0.0, 1.0),
    # Per-episode bias on slot pose *estimates* (metres / rad). Peg estimates get
    # an *additional* independent delta so relative peg−slot does NOT cancel —
    # a script that servos on −rel still misses by several mm and must search.
    "pose_bias_xy": (0.004, 0.010),  # 4–10 mm slot-bias magnitude; direction drawn separately
    "pose_bias_yaw": (0.12, 0.28),  # ~7–16 deg
    "peg_bias_delta": (0.0050, 0.0115),  # 5–11.5 mm peg-vs-slot residual (v0.5.1; was 1.5–5)
    "bias_drift": (0.000010, 0.000060),  # m/step random-walk on estimate bias (v0.5.1; was 0–0.025 mm)
    "backlash": (0.00005, 0.0004),  # Cartesian command deadzone (m)
    "chamfer": (0.35, 1.0),  # entry funnel; still contact-rich at the low end
}

# Slot-yaw vs peg-yaw estimate residual (rad), scaled by tier. Not an LHS axis.
PEG_YAW_DELTA = (0.05, 0.14)

# Estimate "glitch" bursts (v0.5.1): per control step, chance to start a burst of
# extra pose noise plus a transient per-object offset, scaled by tier.
GLITCH_RATE = 0.04  # hard tier; easy / medium scale it down
GLITCH_LEN = (8, 30)  # control steps
GLITCH_NOISE_GAIN = 4.0
GLITCH_OFFSET = (0.002, 0.006)  # m, independent direction per object

STEPS_PER_PEG = 280  # 11.2 s @ 25 Hz — room for contact search + sequencing
DEFAULT_EPISODES_PER_STRATUM = 8
WALL_CLEAR = 0.028
# Per-stratum tier mix ≈ 25% / 35% / 40% (exact for episodes_per_stratum=8 → 2/3/3).
DIFFICULTY_TIERS = ("easy", "medium", "hard")
TIER_FRACTIONS = (0.25, 0.35, 0.40)


@dataclass
class PegStart:
    """One active peg: which board slot it belongs to, and how it starts."""

    slot: int  # index into the episode's slot layout (EpisodeConditions.slots)
    start_type: str  # PEG_START_TYPES
    offset_xy: tuple[float, float]
    height: float
    yaw_err: float  # start yaw relative to the slot yaw
    spec: SlotSpec | None = None  # this peg's slot (shape / size / pose)
    # Held-out family on the reserved split (also set on ``spec``); None on live.
    shape_override: str | None = None
    # Start type as drawn. ``start_type`` is the effective one: a spawn fallback
    # (``Episode._force_safe_start`` / teleport grid) may change it.
    drawn_start_type: str | None = None

    def __post_init__(self) -> None:
        if self.drawn_start_type is None:
            self.drawn_start_type = self.start_type

    def shape(self) -> str:
        return self.spec.shape

    def diameter(self) -> float:
        return self.spec.size

    def public(self) -> dict:
        # Miner-visible record: no true start pose (offset / height / yaw error).
        return {
            "slot": int(self.slot),
            "start_type": self.start_type,
            "drawn_start_type": self.drawn_start_type,
            "shape": self.shape(),
            "size": round(float(self.diameter()), 6),
        }


@dataclass
class ClutterStart:
    offset_xy: tuple[float, float]
    yaw: float
    size_xy: tuple[float, float]  # half-extents (m)
    height: float  # full height (m)

    def public(self) -> dict:
        # Miner-visible record (nist_lite_record/2 shape for the replay viewer): exact
        # start pose and size. Clutter placement is not a held-out axis, is re-drawn
        # every episode, and records only publish after the round.
        return {
            "offset_xy": [round(float(x), 6) for x in self.offset_xy],
            "yaw": round(float(self.yaw), 6),
            "size_xy": [round(float(x), 6) for x in self.size_xy],
            "height": round(float(self.height), 6),
        }


@dataclass
class EpisodeConditions:
    """Public per-episode draw. The platform seed is never stored here."""

    n_pegs: int
    clutter_bucket: str
    clearance: float
    friction: float
    peg_mass: float
    gain_scale: float
    obs_noise: float
    obs_noise_rad: float
    dropout_prob: float
    act_delay: int
    disturbance: int  # 0 or 1
    board_xy: tuple[float, float]
    board_yaw: float
    pose_bias_xy: tuple[float, float] = (0.0, 0.0)  # slot estimate bias (m)
    pose_bias_yaw: float = 0.0
    peg_bias_delta_xy: tuple[float, float] = (0.0, 0.0)  # peg-only extra bias (m)
    # Per-slot / per-clutter independent estimate biases (observation hardening).
    # Empty => fall back to the global pose_bias_xy.
    slot_bias_list: list[tuple[float, float]] = field(default_factory=list)
    slot_bias_yaw_list: list[float] = field(default_factory=list)
    clutter_bias_list: list[tuple[float, float]] = field(default_factory=list)
    # Per-peg independent peg-vs-slot residual (same magnitude, independent direction),
    # so the residual learned on one seated peg says nothing about the next one.
    peg_bias_delta_list: list[tuple[float, float]] = field(default_factory=list)
    # Per-peg slot-yaw estimate residual (rad): the slot-yaw channel and the peg yaw
    # channel do not share the same offset.
    peg_yaw_delta_list: list[float] = field(default_factory=list)
    slots: list[SlotSpec] = field(default_factory=list)  # per-episode board layout
    bias_drift: float = 0.0  # random-walk step (m) on estimate biases
    glitch_rate: float = 0.0  # per-step chance of an estimate glitch burst (v0.5.1)
    backlash: float = 0.0
    chamfer: float = 0.5
    # Hidden search-scale cue (not in obs): spiral pitch that fits this episode's
    # clearance/chamfer. Scripts cannot read it; learners can infer from contact.
    search_pitch: float = 0.0002
    difficulty_tier: str = "hard"  # easy | medium | hard — baked into every live round
    pegs: list[PegStart] = field(default_factory=list)
    clutter: list[ClutterStart] = field(default_factory=list)
    disturbance_step: int = 0
    disturbance_target: str = "none"  # "peg:{i}" | "board" | "drop" | "none"
    max_steps: int = 0

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            self.max_steps = STEPS_PER_PEG * int(self.n_pegs)
        # Hand-built conditions (unit tests / tools) without a layout get the fixed
        # two-row test board; sampled rounds always carry their own drawn layout.
        if not self.slots:
            self.slots = fixed_test_layout()
        for p in self.pegs:
            if p.spec is None:
                p.spec = self.slots[p.slot]

    def public(self) -> dict:
        """Miner-visible per-episode record (``nist_lite_record/2`` shape).

        Exact: board pose and clutter start poses (not held-out axes, re-drawn every
        round). Coarse: ``clearance`` is the midpoint of the episode's clearance class
        and ``chamfer`` is a fixed nominal value, so the continuous held-out axes and
        their reserved sub-ranges stay hidden. No estimate-bias values, no exact values
        of any continuous LHS axis, no true peg start poses.
        """
        return {
            "n_pegs": self.n_pegs,
            "k": int(self.n_pegs),
            "clutter_bucket": self.clutter_bucket,
            "n_clutter": len(self.clutter),
            "difficulty_tier": self.difficulty_tier,
            "clearance_class": clearance_class(self.clearance),
            "clearance": coarse_clearance(self.clearance),
            "chamfer": NOMINAL_CHAMFER,
            "board_xy": [round(float(self.board_xy[0]), 6), round(float(self.board_xy[1]), 6)],
            "board_yaw": round(float(self.board_yaw), 6),
            "disturbance": int(self.disturbance),
            "max_steps": int(self.max_steps),
            "pegs": [p.public() for p in self.pegs],
            "clutter": [c.public() for c in self.clutter],
            # Post-round only; the layout is re-drawn every episode.
            "slots": [s.public() for s in self.slots],
        }


# Coarse public stand-ins for the continuous clearance / chamfer axes (records only;
# the simulator always uses the exact per-episode values).
CLEARANCE_CLASS_MIDPOINT = {"tight": 0.000325, "mid": 0.0005, "loose": 0.0007}
NOMINAL_CHAMFER = 0.5


def coarse_clearance(c: float) -> float:
    return CLEARANCE_CLASS_MIDPOINT[clearance_class(c)]


def clearance_class(c: float) -> str:
    if c >= 0.00060:
        return "loose"
    if c >= 0.00040:
        return "mid"
    return "tight"


def build_salt() -> list[int]:
    """Private per-build entropy (from the referee build secret; absent in dev builds).

    Mixed into every round/episode RNG so the episodes of a round cannot be reproduced
    offline from the platform SEED and the public simulator alone.
    """
    try:
        v = json.loads(RESERVED_PATH.read_text()).get("salt")
    except FileNotFoundError:
        return []
    return [int(v)] if v is not None else []


def episode_seed_sequence(seed: int, episode: int) -> np.random.SeedSequence:
    """Per-episode RNG root (spawn / noise streams) used by the referee."""
    return np.random.SeedSequence([seed, 0x5245, episode, *build_salt()])


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def load_reserved() -> dict[str, list[float]]:
    """Sub-ranges excluded from live rounds (absent in development builds)."""
    try:
        return {k: [float(a), float(b)] for k, (a, b) in json.loads(RESERVED_PATH.read_text())["ranges"].items()}
    except FileNotFoundError:
        return {}


def _map(u: np.ndarray, lo: float, hi: float, gap: list[float] | None, inside: bool) -> np.ndarray:
    if gap is None:
        return lo + u * (hi - lo)
    a, b = gap
    if inside:
        return a + u * (b - a)
    x = lo + u * ((hi - lo) - (b - a))
    return np.where(x >= a, x + (b - a), x)


def _lhs(rng: np.random.Generator, k: int, reserved: dict, inside: bool) -> dict[str, np.ndarray]:
    out = {}
    for name, (lo, hi) in RANGES.items():
        u = (rng.permutation(k) + rng.uniform(0, 1, k)) / k
        out[name] = _map(u, lo, hi, reserved.get(name), inside)
    return out


def strata() -> list[tuple[int, str]]:
    return [(k, bucket) for k in K_VALUES for bucket in CLUTTER_BUCKETS]


def _draw_clutter_count(rng: np.random.Generator, bucket: str) -> int:
    if bucket == "low":
        return int(rng.integers(0, 2))  # 0 or 1
    return int(rng.integers(2, 4))  # 2 or 3


def _start_clear(lx: float, ly: float, peg_outer: float, layout: list[SlotSpec], margin: float) -> bool:
    """True if a free peg centred at board-local (lx, ly) clears every slot's walls."""
    if abs(lx) > 0.17 or abs(ly) > 0.14:
        return False
    for sp in layout:
        need = sp.outer_radius() + 0.0012 + LAYOUT_WALL + peg_outer + margin
        if math.hypot(lx - sp.x, ly - sp.y) < need:
            return False
    return True


def _free_offset(rng, spec: SlotSpec, layout: list[SlotSpec], r_lo: float, r_hi: float, margin: float):
    """Random offset (board frame) at r in [r_lo, r_hi] from the slot, clear of all walls."""
    po = spec.outer_radius()
    for _ in range(30):
        r = _u(rng, r_lo, r_hi)
        th = _u(rng, 0, 2 * math.pi)
        off = (r * math.cos(th), r * math.sin(th))
        if _start_clear(spec.x + off[0], spec.y + off[1], po, layout, margin):
            return off
    # v0.5.1: never fall back to a spot inside another slot. Widen the ring, then
    # relax the margin, then take the candidate farthest from every slot wall.
    best, best_c = (r_hi, 0.0), -1e9
    for r_hi2, m2 in ((r_hi + 0.03, margin), (r_hi + 0.06, 0.0)):
        for _ in range(60):
            r = _u(rng, r_lo, r_hi2)
            th = _u(rng, 0, 2 * math.pi)
            off = (r * math.cos(th), r * math.sin(th))
            if _start_clear(spec.x + off[0], spec.y + off[1], po, layout, m2):
                return off
            c = _slot_clearance(spec.x + off[0], spec.y + off[1], po, layout)
            if c > best_c and abs(spec.x + off[0]) <= 0.17 and abs(spec.y + off[1]) <= 0.14:
                best, best_c = off, c
    return best


def _slot_clearance(lx: float, ly: float, peg_outer: float, layout: list[SlotSpec]) -> float:
    """Min distance (m) between a free peg at board-local (lx, ly) and any slot's outer wall."""
    return min(
        math.hypot(lx - sp.x, ly - sp.y) - (sp.outer_radius() + 0.0012 + LAYOUT_WALL + peg_outer)
        for sp in layout
    )


def _draw_peg_starts_with_weights(
    rng: np.random.Generator,
    slot_ids: list[int],
    layout: list[SlotSpec],
    weights: np.ndarray,
) -> list[PegStart]:
    """Assign start types and start poses. At most one peg may be pre-grasped."""
    n = len(slot_ids)
    types = list(rng.choice(PEG_START_TYPES, size=n, replace=True, p=weights / weights.sum()))
    grasped_idx = [i for i, t in enumerate(types) if t == "grasped"]
    if len(grasped_idx) > 1:
        keep = int(rng.choice(grasped_idx))
        for i in grasped_idx:
            if i != keep:
                types[i] = str(rng.choice(["loose", "near_hole", "tray"]))
    tray_centre = (_u(rng, -0.03, 0.03), _u(rng, -0.095, -0.070))
    tray_i = 0
    pegs: list[PegStart] = []
    for local_i, slot in enumerate(slot_ids):
        st = types[local_i]
        spec = layout[slot]
        keyed = spec.shape != "round"
        if st == "grasped":
            r = _u(rng, 0.0, 0.01)
            th = _u(rng, 0, 2 * math.pi)
            off = (r * math.cos(th), r * math.sin(th))
            h = _u(rng, 0.01, 0.04)
            yaw = _u(rng, -0.3, 0.3) if keyed else 0.0
        elif st == "near_hole":
            base = 2 * spec.outer_radius() + 0.0012 + LAYOUT_WALL
            off = _free_offset(rng, spec, layout, base + 0.012, min(0.058, base + 0.018), 0.004)
            h = 0.0003
            yaw = _u(rng, -0.15, 0.15) if keyed else 0.0
        elif st == "lying":
            off = _free_offset(rng, spec, layout, 0.05, 0.07, 0.012)
            h = _u(rng, 0.01, 0.03)
            yaw = _u(rng, -math.pi, math.pi)
        elif st == "tray":
            n_tray = max(1, sum(1 for t in types if t == "tray"))
            ang = tray_i * (2 * math.pi / n_tray)
            rad = 0.016 + 1.2 * spec.outer_radius()
            off = (tray_centre[0] + rad * math.cos(ang), tray_centre[1] + rad * math.sin(ang))
            if not _start_clear(spec.x + off[0], spec.y + off[1], spec.outer_radius(), layout, 0.002):
                # v0.5.1: a tray spot over another slot's hole is moved clear.
                off = _free_offset(rng, spec, layout, 0.04, 0.07, 0.004)
            h = _u(rng, 0.005, 0.02)
            yaw = _u(rng, -math.pi / 2, math.pi / 2)
            tray_i += 1
        else:
            off = _free_offset(rng, spec, layout, 0.04, 0.07, 0.004)
            h = _u(rng, 0.01, 0.03)
            yaw = _u(rng, -math.pi / 2, math.pi / 2)
        pegs.append(
            PegStart(
                slot=int(slot),
                start_type=st,
                offset_xy=(float(off[0]), float(off[1])),
                height=float(h),
                yaw_err=float(yaw),
                spec=spec,
            )
        )
    return pegs


def _draw_clutter(rng: np.random.Generator, n: int, pegs: list[PegStart]) -> list[ClutterStart]:
    """Place 0–3 free blocks, sometimes near or over a slot approach."""
    out: list[ClutterStart] = []
    for j in range(n):
        if pegs and rng.random() < 0.55:
            sp = pegs[int(rng.integers(0, len(pegs)))].spec
            hx, hy = sp.x, sp.y
            off = (hx + _u(rng, -0.02, 0.02), hy + _u(rng, -0.02, 0.02))
        else:
            off = (_u(rng, -0.12, 0.12), _u(rng, -0.12, 0.12))
        sx = _u(rng, 0.008, 0.018)
        sy = _u(rng, 0.008, 0.018)
        hz = _u(rng, 0.012, 0.025)
        out.append(
            ClutterStart(
                offset_xy=(float(off[0]), float(off[1])),
                yaw=_u(rng, -math.pi, math.pi),
                size_xy=(float(sx), float(sy)),
                height=float(hz),
            )
        )
    return out


def _tier_counts(n: int) -> tuple[int, int, int]:
    """Integer (easy, medium, hard) counts summing to n ≈ 25/35/40.

    Small ``n`` is special-cased so every live round keeps an easy↔hard
    gradient (largest-remainder alone can drop easy or hard at n≤3).
    """
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 0, 0, 1
    if n == 2:
        return 1, 0, 1  # easy + hard
    if n == 3:
        return 1, 1, 1
    raw = [f * n for f in TIER_FRACTIONS]
    counts = [int(x) for x in raw]  # floor
    rem = n - sum(counts)
    order = sorted(range(3), key=lambda i: (raw[i] - counts[i], -i), reverse=True)
    for k in range(rem):
        counts[order[k % 3]] += 1
    return counts[0], counts[1], counts[2]


def _assign_tiers(rng: np.random.Generator, n: int) -> list[str]:
    n_easy, n_med, n_hard = _tier_counts(n)
    tiers = (["easy"] * n_easy) + (["medium"] * n_med) + (["hard"] * n_hard)
    rng.shuffle(tiers)
    return tiers


def _easy_peg_starts(rng: np.random.Generator, slot_ids: list[int], layout: list[SlotSpec]) -> list[PegStart]:
    """Grasped / near_hole only — partial-credit friendly, low lateral offset."""
    n = len(slot_ids)
    # One grasped + near-hole neighbours so a single successful insert scores.
    types = ["grasped"] + ["near_hole"] * (n - 1)
    rng.shuffle(types)
    pegs: list[PegStart] = []
    for local_i, slot in enumerate(slot_ids):
        st = types[local_i]
        spec = layout[slot]
        if st == "grasped":
            # Hover near the hole — privileged succeeds; residual bias still bites −rel.
            r = _u(rng, 0.0, 0.006)
            th = _u(rng, 0, 2 * math.pi)
            off = (r * math.cos(th), r * math.sin(th))
            h = _u(rng, 0.012, 0.028)
            yaw = _u(rng, -0.05, 0.05) if spec.shape != "round" else 0.0
        else:
            # Near-hole but outside the chamfer funnel so policies must approach + search.
            base = 2 * spec.outer_radius() + 0.0012 + LAYOUT_WALL
            off = _free_offset(rng, spec, layout, base + 0.012, min(0.05, base + 0.017), 0.004)
            h = 0.0003
            yaw = _u(rng, -0.06, 0.06) if spec.shape != "round" else 0.0
        pegs.append(
            PegStart(
                slot=int(slot),
                start_type=st,
                offset_xy=(float(off[0]), float(off[1])),
                height=float(h),
                yaw_err=float(yaw),
                spec=spec,
            )
        )
    return pegs


# v0.5.1: more lying-down starts (hard 8% -> 22%, medium 2% -> 12%).
HARD_START_WEIGHTS = np.array([0.24, 0.22, 0.10, 0.26, 0.18])  # loose/lying/tray/near/grasped
MEDIUM_START_WEIGHTS = np.array([0.26, 0.12, 0.06, 0.36, 0.20])


def sample_episodes(
    seed: int, episodes_per_stratum: int, split: str = "live"
) -> tuple[dict[str, list[float]], list[EpisodeConditions]]:
    reserved = load_reserved()
    inside = split == "reserved"
    if inside and not reserved:
        raise RuntimeError("no reserved ranges in this build")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4E495354, *build_salt()]))
    rc = {k: list(v) for k, v in RANGES.items()}
    eps: list[EpisodeConditions] = []
    for k_pegs, bucket in strata():
        lhs = _lhs(rng, episodes_per_stratum, reserved, inside)
        tiers = _assign_tiers(rng, episodes_per_stratum)
        for j in range(episodes_per_stratum):
            tier = tiers[j]
            slot_ids = sorted(rng.choice(N_SLOTS, size=k_pegs, replace=False).tolist())
            # Held-out families: one active slot on the reserved hard tier.
            held_slot = int(rng.choice(slot_ids)) if (inside and tier == "hard") else None
            layout = draw_layout(rng, held_slot)
            # ---- tier-dependent starts / clutter / anti-scripting scales ----
            if tier == "easy":
                pegs = _easy_peg_starts(rng, slot_ids, layout)
                n_clutter = 0 if bucket == "low" else int(rng.integers(0, 2))
                # Residual peg−slot bias stays material so −rel scripts miss; roomy
                # chamfer + one grasped start let the privileged solver score.
                noise_scale, delta_scale, drift_scale = 0.50, 0.90, 0.50
                glitch_scale = 0.50
                drop_scale, disturb_gate = 0.22, 1.01  # never disturbance
                clear_boost, chamfer_floor = 0.00018, 0.75
                act_delay_force0 = True
            elif tier == "medium":
                pegs = _draw_peg_starts_with_weights(rng, slot_ids, layout, MEDIUM_START_WEIGHTS)
                n_clutter = _draw_clutter_count(rng, bucket)
                if n_clutter > 2:
                    n_clutter = 2
                noise_scale, delta_scale, drift_scale = 0.75, 1.00, 0.80
                glitch_scale = 0.80
                drop_scale, disturb_gate = 0.55, 0.85
                clear_boost, chamfer_floor = 0.00008, 0.55
                act_delay_force0 = False
            else:  # hard — full difficulty; held-out shapes on reserved split
                pegs = _draw_peg_starts_with_weights(rng, slot_ids, layout, HARD_START_WEIGHTS)
                if held_slot is not None:
                    for p_ in pegs:
                        if p_.slot == held_slot:
                            p_.shape_override = layout[held_slot].shape
                n_clutter = _draw_clutter_count(rng, bucket)
                noise_scale, delta_scale, drift_scale = 1.0, 1.0, 1.0
                glitch_scale = 1.0
                drop_scale, disturb_gate = 1.0, 0.65
                clear_boost, chamfer_floor = 0.0, 0.35
                act_delay_force0 = False
            clutter = _draw_clutter(rng, n_clutter, pegs)
            act_delay = 0 if act_delay_force0 else int(np.clip(round(float(lhs["act_delay"][j])), 0, 1))
            disturbance = 1 if float(lhs["disturbance"][j]) >= disturb_gate else 0
            max_steps = STEPS_PER_PEG * k_pegs
            if disturbance:
                d_step = int(rng.integers(max(1, max_steps // 5), max(2, (4 * max_steps) // 5)))
                roll = rng.random()
                if roll < 0.50:
                    d_target = f"peg:{int(rng.integers(0, k_pegs))}"
                elif roll < 0.70:
                    d_target = "drop"
                else:
                    d_target = "board"
            else:
                d_step = 0
                d_target = "none"
            bias_mag = float(lhs["pose_bias_xy"][j])
            # Easy/medium shrink slot bias; residual peg delta (above) still bites scripts.
            if tier == "easy":
                bias_mag = RANGES["pose_bias_xy"][0] * 0.40 + bias_mag * 0.18
            elif tier == "medium":
                bias_mag = RANGES["pose_bias_xy"][0] * 0.40 + bias_mag * 0.50
            bias_ang = _u(rng, 0, 2 * math.pi)
            bias_xy = (bias_mag * math.cos(bias_ang), bias_mag * math.sin(bias_ang))
            bias_yaw = float(lhs["pose_bias_yaw"][j]) * (1.0 if rng.random() < 0.5 else -1.0)
            if tier == "easy":
                bias_yaw *= 0.30
            elif tier == "medium":
                bias_yaw *= 0.60
            dmag = float(lhs["peg_bias_delta"][j]) * delta_scale
            dang = _u(rng, 0, 2 * math.pi)
            peg_delta = (dmag * math.cos(dang), dmag * math.sin(dang))
            clr = min(RANGES["clearance"][1], float(lhs["clearance"][j]) + clear_boost)
            ch = max(chamfer_floor, float(lhs["chamfer"][j]))
            search_pitch = 0.00008 + 0.55 * clr + 0.00012 * ch
            lo_n, hi_n = RANGES["obs_noise"]
            obs_noise = float(np.clip(float(lhs["obs_noise"][j]) * noise_scale, lo_n * 0.25, hi_n))
            lo_r, hi_r = RANGES["obs_noise_rad"]
            obs_noise_rad = float(
                np.clip(float(lhs["obs_noise_rad"][j]) * noise_scale, lo_r * 0.25, hi_r)
            )
            # Independent per-slot / per-clutter / per-peg estimate offsets: same
            # magnitudes as the episode draw, independent directions.
            slot_bias_list = []
            slot_bias_yaw_list = []
            for _si in range(len(pegs)):
                _a = _u(rng, 0, 2 * math.pi)
                slot_bias_list.append((bias_mag * math.cos(_a), bias_mag * math.sin(_a)))
                slot_bias_yaw_list.append(abs(bias_yaw) * (1.0 if rng.random() < 0.5 else -1.0))
            clutter_bias_list = []
            for _ci in range(len(clutter)):
                _a = _u(rng, 0, 2 * math.pi)
                clutter_bias_list.append((bias_mag * math.cos(_a), bias_mag * math.sin(_a)))
            peg_bias_delta_list = []
            for _pi in range(len(pegs)):
                _a = _u(rng, 0, 2 * math.pi)
                peg_bias_delta_list.append((dmag * math.cos(_a), dmag * math.sin(_a)))
            ymag = _u(rng, *PEG_YAW_DELTA) * delta_scale
            peg_yaw_delta_list = [ymag * (1.0 if rng.random() < 0.5 else -1.0) for _ in pegs]
            eps.append(
                EpisodeConditions(
                    n_pegs=k_pegs,
                    clutter_bucket=bucket,
                    clearance=clr,
                    friction=float(lhs["friction"][j]),
                    peg_mass=float(lhs["peg_mass"][j]),
                    gain_scale=float(lhs["gain_scale"][j]),
                    obs_noise=obs_noise,
                    obs_noise_rad=obs_noise_rad,
                    dropout_prob=float(lhs["dropout_prob"][j]) * drop_scale,
                    act_delay=act_delay,
                    disturbance=disturbance,
                    board_xy=(_u(rng, -0.03, 0.03), _u(rng, -0.03, 0.03)),
                    board_yaw=_u(rng, -0.15, 0.15),
                    pose_bias_xy=(float(bias_xy[0]), float(bias_xy[1])),
                    pose_bias_yaw=float(bias_yaw),
                    peg_bias_delta_xy=(float(peg_delta[0]), float(peg_delta[1])),
                    slot_bias_list=slot_bias_list,
                    slot_bias_yaw_list=slot_bias_yaw_list,
                    clutter_bias_list=clutter_bias_list,
                    peg_bias_delta_list=peg_bias_delta_list,
                    peg_yaw_delta_list=peg_yaw_delta_list,
                    slots=layout,
                    bias_drift=float(lhs["bias_drift"][j]) * drift_scale,
                    glitch_rate=GLITCH_RATE * glitch_scale,
                    backlash=float(lhs["backlash"][j])
                    * (0.15 if tier == "easy" else 0.55 if tier == "medium" else 1.0),
                    chamfer=ch,
                    search_pitch=float(search_pitch),
                    difficulty_tier=tier,
                    pegs=pegs,
                    clutter=clutter,
                    disturbance_step=d_step,
                    disturbance_target=d_target,
                    max_steps=max_steps,
                )
            )
    return rc, eps


def slot_world_xy(board_xy: tuple[float, float], board_yaw: float, spec: SlotSpec) -> tuple[float, float]:
    """World-frame hole centre for a slot given board pose."""
    lx, ly = spec.x, spec.y
    c, s = math.cos(board_yaw), math.sin(board_yaw)
    return board_xy[0] + c * lx - s * ly, board_xy[1] + s * lx + c * ly
