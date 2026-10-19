"""Per-round condition sampling for the multi-peg board. Everything derives from SEED.

Strata are (K, clutter_bucket) with K in {3,4,5,6} and clutter_bucket in {low, high}.
Eight strata × episodes_per_stratum (default 8) = 64 episodes; mean K = 4.5.

Within each stratum a Latin hypercube covers the live ranges of clearance, friction,
mass, gain, observation noise, angular noise, dropout, actuation delay, and whether
a mid-episode disturbance fires. Peg identities, start poses, and board pose rotate
with the seed. Records reveal conditions; they never reveal the seed.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

RESERVED_PATH = Path(os.environ.get("NIST_RESERVED_PATH", Path(__file__).with_name("reserved.json")))

# NIST ATB-1 peg family: round + square × 4/8/12/16 mm. Fixed board-frame centres.
# Layout is two rows (round / square), four sizes left-to-right — ATB-1-inspired spacing.
SLOT_SPECS: tuple[tuple[str, float, float, float], ...] = (
    ("round", 0.004, -0.075, 0.045),
    ("round", 0.008, -0.025, 0.045),
    ("round", 0.012, 0.025, 0.045),
    ("round", 0.016, 0.075, 0.045),
    ("square", 0.004, -0.075, -0.045),
    ("square", 0.008, -0.025, -0.045),
    ("square", 0.012, 0.025, -0.045),
    ("square", 0.016, 0.075, -0.045),
)
N_SLOTS = len(SLOT_SPECS)
MAX_CLUTTER = 3
K_VALUES = (3, 4, 5, 6)
CLUTTER_BUCKETS = ("low", "high")  # low → 0–1 blocks; high → 2–3
PEG_START_TYPES = ("loose", "lying", "tray", "near_hole", "grasped")

# Live ranges. Continuous axes are LHS-drawn; discrete axes are mapped from LHS then rounded.
RANGES = {
    "clearance": (0.00005, 0.0006),  # 0.05–0.6 mm; tightened vs v0.1.x after solimp check
    "friction": (0.3, 1.2),
    "peg_mass": (0.005, 0.05),
    "gain_scale": (0.6, 1.4),
    "obs_noise": (0.0, 0.001),  # 0–1 mm on part poses
    "obs_noise_rad": (0.0, 0.03),
    "dropout_prob": (0.0, 0.12),
    "act_delay": (0.0, 3.0),  # rounded to {0,1,2,3} control steps
    "disturbance": (0.0, 1.0),  # rounded: 0 or 1 mid-episode impulse
}

STEPS_PER_PEG = 200  # 8 s @ 25 Hz
DEFAULT_EPISODES_PER_STRATUM = 8
# Extra radial gap beyond the hole opening so a "near_hole" upright peg does not
# tip into the void during settle (hole wall thickness + margin).
WALL_CLEAR = 0.028  # keep near_hole pegs from tipping into the opening under noop


@dataclass
class PegStart:
    """One active peg: which board slot it belongs to, and how it starts."""

    slot: int  # index into SLOT_SPECS / board holes
    start_type: str  # PEG_START_TYPES
    offset_xy: tuple[float, float]
    height: float
    yaw_err: float

    def public(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}


@dataclass
class ClutterStart:
    offset_xy: tuple[float, float]
    yaw: float
    size_xy: tuple[float, float]  # half-extents (m)
    height: float  # full height (m)

    def public(self) -> dict:
        d = asdict(self)
        out = {}
        for k, v in d.items():
            if isinstance(v, float):
                out[k] = round(v, 6)
            elif isinstance(v, tuple):
                out[k] = [round(x, 6) for x in v]
            else:
                out[k] = v
        return out


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
    pegs: list[PegStart] = field(default_factory=list)
    clutter: list[ClutterStart] = field(default_factory=list)
    disturbance_step: int = 0
    disturbance_target: str = "none"  # "peg:{i}" | "board" | "none"
    max_steps: int = 0

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            self.max_steps = STEPS_PER_PEG * int(self.n_pegs)

    def public(self) -> dict:
        return {
            "n_pegs": self.n_pegs,
            "clutter_bucket": self.clutter_bucket,
            "clearance": round(self.clearance, 7),
            "friction": round(self.friction, 6),
            "peg_mass": round(self.peg_mass, 6),
            "gain_scale": round(self.gain_scale, 6),
            "obs_noise": round(self.obs_noise, 7),
            "obs_noise_rad": round(self.obs_noise_rad, 6),
            "dropout_prob": round(self.dropout_prob, 6),
            "act_delay": int(self.act_delay),
            "disturbance": int(self.disturbance),
            "board_xy": [round(self.board_xy[0], 6), round(self.board_xy[1], 6)],
            "board_yaw": round(self.board_yaw, 6),
            "pegs": [p.public() for p in self.pegs],
            "clutter": [c.public() for c in self.clutter],
            "disturbance_step": int(self.disturbance_step),
            "disturbance_target": self.disturbance_target,
            "max_steps": int(self.max_steps),
            # Convenience mirrors for aggregators / reserve tests.
            "k": int(self.n_pegs),
        }


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


def _draw_peg_starts(
    rng: np.random.Generator, slot_ids: list[int], n_clutter: int
) -> list[PegStart]:
    """Assign start types. At most one peg may be pre-grasped."""
    n = len(slot_ids)
    # Mix of starts; grasped is rare (≤1). Slightly more near_hole/grasped so a
    # competent scripted policy has partial-credit headroom without saturating.
    weights = np.array([0.24, 0.18, 0.18, 0.24, 0.16])  # loose lying tray near grasped
    types = list(rng.choice(PEG_START_TYPES, size=n, replace=True, p=weights / weights.sum()))
    grasped_idx = [i for i, t in enumerate(types) if t == "grasped"]
    if len(grasped_idx) > 1:
        keep = int(rng.choice(grasped_idx))
        for i in grasped_idx:
            if i != keep:
                types[i] = str(rng.choice(["loose", "lying", "tray", "near_hole"]))
    # Tray cluster: pegs share a common tray centre so they touch.
    tray_centre = (_u(rng, -0.03, 0.03), _u(rng, -0.095, -0.070))
    tray_i = 0
    pegs: list[PegStart] = []
    for local_i, slot in enumerate(slot_ids):
        st = types[local_i]
        shape, dia, _, _ = SLOT_SPECS[slot]
        if st == "grasped":
            r = _u(rng, 0.0, 0.01)
            th = _u(rng, 0, 2 * math.pi)
            off = (r * math.cos(th), r * math.sin(th))
            h = _u(rng, 0.01, 0.04)
            yaw = _u(rng, -0.3, 0.3) if shape == "square" else 0.0
        elif st == "near_hole":
            # Rest beside the hole on the board, clear of the wall lip so settle cannot tip it in.
            # Cap radius so large pegs stay on-board relative to their slot.
            r = min(0.055, (dia / 2) + WALL_CLEAR + _u(rng, 0.0, 0.004))
            th = _u(rng, 0, 2 * math.pi)
            off = (r * math.cos(th), r * math.sin(th))
            h = 0.0003
            yaw = _u(rng, -0.15, 0.15) if shape == "square" else 0.0
        elif st == "lying":
            # Keep lying pegs on-board and clear of hole walls (long axis ~50 mm).
            r = _u(rng, 0.05, 0.07)
            th = _u(rng, 0, 2 * math.pi)
            off = (r * math.cos(th), r * math.sin(th))
            h = _u(rng, 0.01, 0.03)
            yaw = _u(rng, -math.pi, math.pi)
        elif st == "tray":
            # Fan pegs around the tray centre so neighbours touch but do not interlock.
            n_tray = max(1, sum(1 for t in types if t == "tray"))
            ang = tray_i * (2 * math.pi / n_tray)
            rad = 0.016 + 0.6 * dia
            off = (tray_centre[0] + rad * math.cos(ang), tray_centre[1] + rad * math.sin(ang))
            h = _u(rng, 0.005, 0.02)
            yaw = _u(rng, -math.pi / 4, math.pi / 4)
            tray_i += 1
        else:  # loose upright
            r = _u(rng, 0.04, 0.07)
            th = _u(rng, 0, 2 * math.pi)
            off = (r * math.cos(th), r * math.sin(th))
            h = _u(rng, 0.01, 0.03)
            yaw = _u(rng, -math.pi / 4, math.pi / 4)
        pegs.append(
            PegStart(
                slot=int(slot),
                start_type=st,
                offset_xy=(float(off[0]), float(off[1])),
                height=float(h),
                yaw_err=float(yaw),
            )
        )
    # n_clutter is unused for peg starts but kept so call sites stay explicit.
    _ = n_clutter
    return pegs


def _draw_clutter(rng: np.random.Generator, n: int, pegs: list[PegStart]) -> list[ClutterStart]:
    """Place 0–3 free blocks, sometimes near or over a slot approach."""
    out: list[ClutterStart] = []
    for j in range(n):
        if pegs and rng.random() < 0.55:
            # Near a peg's matching slot approach (board frame, relative to that hole).
            slot = pegs[int(rng.integers(0, len(pegs)))].slot
            _, _, hx, hy = SLOT_SPECS[slot]
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


def sample_episodes(
    seed: int, episodes_per_stratum: int, split: str = "live"
) -> tuple[dict[str, list[float]], list[EpisodeConditions]]:
    reserved = load_reserved()
    inside = split == "reserved"
    if inside and not reserved:
        raise RuntimeError("no reserved ranges in this build")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4E495354]))
    rc = {k: list(v) for k, v in RANGES.items()}
    eps: list[EpisodeConditions] = []
    for k_pegs, bucket in strata():
        lhs = _lhs(rng, episodes_per_stratum, reserved, inside)
        for j in range(episodes_per_stratum):
            slot_ids = sorted(rng.choice(N_SLOTS, size=k_pegs, replace=False).tolist())
            n_clutter = _draw_clutter_count(rng, bucket)
            pegs = _draw_peg_starts(rng, slot_ids, n_clutter)
            clutter = _draw_clutter(rng, n_clutter, pegs)
            act_delay = int(np.clip(round(float(lhs["act_delay"][j])), 0, 3))
            disturbance = int(round(float(lhs["disturbance"][j])))
            disturbance = 1 if disturbance >= 1 else 0
            max_steps = STEPS_PER_PEG * k_pegs
            if disturbance:
                d_step = int(rng.integers(max(1, max_steps // 5), max(2, (4 * max_steps) // 5)))
                if rng.random() < 0.7:
                    d_target = f"peg:{int(rng.integers(0, k_pegs))}"
                else:
                    d_target = "board"
            else:
                d_step = 0
                d_target = "none"
            eps.append(
                EpisodeConditions(
                    n_pegs=k_pegs,
                    clutter_bucket=bucket,
                    clearance=float(lhs["clearance"][j]),
                    friction=float(lhs["friction"][j]),
                    peg_mass=float(lhs["peg_mass"][j]),
                    gain_scale=float(lhs["gain_scale"][j]),
                    obs_noise=float(lhs["obs_noise"][j]),
                    obs_noise_rad=float(lhs["obs_noise_rad"][j]),
                    dropout_prob=float(lhs["dropout_prob"][j]),
                    act_delay=act_delay,
                    disturbance=disturbance,
                    board_xy=(_u(rng, -0.03, 0.03), _u(rng, -0.03, 0.03)),
                    board_yaw=_u(rng, -0.15, 0.15),
                    pegs=pegs,
                    clutter=clutter,
                    disturbance_step=d_step,
                    disturbance_target=d_target,
                    max_steps=max_steps,
                )
            )
    return rc, eps


def slot_world_xy(board_xy: tuple[float, float], board_yaw: float, slot: int) -> tuple[float, float]:
    """World-frame hole centre for a slot given board pose."""
    _, _, lx, ly = SLOT_SPECS[slot]
    c, s = math.cos(board_yaw), math.sin(board_yaw)
    return board_xy[0] + c * lx - s * ly, board_xy[1] + s * lx + c * ly
