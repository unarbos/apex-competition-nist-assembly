"""Per-round condition sampling for the multi-peg board. Everything derives from SEED.

Strata are (K, clutter_bucket) with K in {3,4,5,6} and clutter_bucket in {low, high}.
Eight strata × episodes_per_stratum (default 8) = 64 episodes; mean K = 4.5.

Within each stratum a Latin hypercube covers the live ranges of clearance, friction,
mass, gain, observation noise, angular noise, dropout, actuation delay, backlash,
pose-estimate bias, chamfer, and whether a mid-episode disturbance fires. Peg
identities, start poses, and board pose rotate with the seed. Records reveal
conditions; they never reveal the seed.

Held-out: reserved sub-ranges plus a D-flat keyed shape family that never appears
in live rounds (baked into the private referee via reserved.json).
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

# Board slots: round + square (live). Hex and D-flat keyed pegs are held-out only.
# Layout is two rows, four sizes left-to-right — ATB-1-inspired spacing.
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
LIVE_SHAPES = ("round", "square")
HELD_OUT_SHAPES = ("hex", "dflat")  # reserved split only
SHAPE_NAMES = ("round", "square", "hex", "dflat")

# Live ranges. Continuous axes are LHS-drawn; discrete axes are mapped from LHS then rounded.
# Clearance floor is 0.25 mm: sub-0.1 mm square fits were unstable under MuJoCo soft contact.
RANGES = {
    "clearance": (0.00025, 0.0008),  # 0.25–0.8 mm; needs wiggle on the tight end
    "friction": (0.3, 1.3),
    "peg_mass": (0.005, 0.05),
    "gain_scale": (0.6, 1.4),
    "obs_noise": (0.0004, 0.0020),  # 0.4–2.0 mm white noise on pose estimates
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
    "peg_bias_delta": (0.0015, 0.0050),  # 1.5–5 mm peg-vs-slot residual (defeats pure −rel)
    "bias_drift": (0.0, 0.000025),  # m/step random-walk on estimate bias (non-stationary)
    "backlash": (0.00005, 0.0004),  # Cartesian command deadzone (m)
    "chamfer": (0.35, 1.0),  # entry funnel; still contact-rich at the low end
}

STEPS_PER_PEG = 280  # 11.2 s @ 25 Hz — room for contact search + sequencing
DEFAULT_EPISODES_PER_STRATUM = 8
WALL_CLEAR = 0.028


@dataclass
class PegStart:
    """One active peg: which board slot it belongs to, and how it starts."""

    slot: int  # index into SLOT_SPECS / board holes
    start_type: str  # PEG_START_TYPES
    offset_xy: tuple[float, float]
    height: float
    yaw_err: float
    # Held-out shape override (e.g. "dflat"); None → use SLOT_SPECS[slot][0].
    shape_override: str | None = None

    def shape(self) -> str:
        if self.shape_override:
            return self.shape_override
        return SLOT_SPECS[self.slot][0]

    def diameter(self) -> float:
        return SLOT_SPECS[self.slot][1]

    def public(self) -> dict:
        d = asdict(self)
        d["shape"] = self.shape()
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
    pose_bias_xy: tuple[float, float] = (0.0, 0.0)  # slot estimate bias (m)
    pose_bias_yaw: float = 0.0
    peg_bias_delta_xy: tuple[float, float] = (0.0, 0.0)  # peg-only extra bias (m)
    bias_drift: float = 0.0  # random-walk step (m) on estimate biases
    backlash: float = 0.0
    chamfer: float = 0.5
    # Hidden search-scale cue (not in obs): spiral pitch that fits this episode's
    # clearance/chamfer. Scripts cannot read it; learners can infer from contact.
    search_pitch: float = 0.0002
    pegs: list[PegStart] = field(default_factory=list)
    clutter: list[ClutterStart] = field(default_factory=list)
    disturbance_step: int = 0
    disturbance_target: str = "none"  # "peg:{i}" | "board" | "drop" | "none"
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
            "pose_bias_xy": [round(self.pose_bias_xy[0], 6), round(self.pose_bias_xy[1], 6)],
            "pose_bias_yaw": round(self.pose_bias_yaw, 6),
            "peg_bias_delta_xy": [
                round(self.peg_bias_delta_xy[0], 6),
                round(self.peg_bias_delta_xy[1], 6),
            ],
            "bias_drift": round(self.bias_drift, 8),
            "backlash": round(self.backlash, 7),
            "chamfer": round(self.chamfer, 4),
            "search_pitch": round(self.search_pitch, 7),
            "pegs": [p.public() for p in self.pegs],
            "clutter": [c.public() for c in self.clutter],
            "disturbance_step": int(self.disturbance_step),
            "disturbance_target": self.disturbance_target,
            "max_steps": int(self.max_steps),
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
    rng: np.random.Generator, slot_ids: list[int], n_clutter: int, held_out: bool
) -> list[PegStart]:
    """Assign start types. At most one peg may be pre-grasped."""
    n = len(slot_ids)
    # Mix keeps upright/near-hole common so the privileged solver can show solvability,
    # while lying/tray remain present as hard cases for obs-only scripts.
    weights = np.array([0.28, 0.08, 0.10, 0.34, 0.20])  # favor loose/near_hole/grasped
    types = list(rng.choice(PEG_START_TYPES, size=n, replace=True, p=weights / weights.sum()))
    grasped_idx = [i for i, t in enumerate(types) if t == "grasped"]
    if len(grasped_idx) > 1:
        keep = int(rng.choice(grasped_idx))
        for i in grasped_idx:
            if i != keep:
                types[i] = str(rng.choice(["loose", "lying", "tray", "near_hole"]))
    tray_centre = (_u(rng, -0.03, 0.03), _u(rng, -0.095, -0.070))
    tray_i = 0
    pegs: list[PegStart] = []
    for local_i, slot in enumerate(slot_ids):
        st = types[local_i]
        shape, dia, _, _ = SLOT_SPECS[slot]
        keyed = shape in ("square", "hex", "dflat")
        if st == "grasped":
            r = _u(rng, 0.0, 0.01)
            th = _u(rng, 0, 2 * math.pi)
            off = (r * math.cos(th), r * math.sin(th))
            h = _u(rng, 0.01, 0.04)
            yaw = _u(rng, -0.3, 0.3) if keyed else 0.0
        elif st == "near_hole":
            r = min(0.055, (dia / 2) + WALL_CLEAR + _u(rng, 0.0, 0.004))
            th = _u(rng, 0, 2 * math.pi)
            off = (r * math.cos(th), r * math.sin(th))
            h = 0.0003
            yaw = _u(rng, -0.15, 0.15) if keyed else 0.0
        elif st == "lying":
            r = _u(rng, 0.05, 0.07)
            th = _u(rng, 0, 2 * math.pi)
            off = (r * math.cos(th), r * math.sin(th))
            h = _u(rng, 0.01, 0.03)
            yaw = _u(rng, -math.pi, math.pi)
        elif st == "tray":
            n_tray = max(1, sum(1 for t in types if t == "tray"))
            ang = tray_i * (2 * math.pi / n_tray)
            rad = 0.016 + 0.6 * dia
            off = (tray_centre[0] + rad * math.cos(ang), tray_centre[1] + rad * math.sin(ang))
            h = _u(rng, 0.005, 0.02)
            yaw = _u(rng, -math.pi / 4, math.pi / 4)
            tray_i += 1
        else:
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
    # Held-out split: force one peg into a held-out family (hex or D-flat).
    if held_out and pegs:
        i = int(rng.integers(0, len(pegs)))
        pegs[i].shape_override = str(rng.choice(HELD_OUT_SHAPES))
    _ = n_clutter
    return pegs


def _draw_clutter(rng: np.random.Generator, n: int, pegs: list[PegStart]) -> list[ClutterStart]:
    """Place 0–3 free blocks, sometimes near or over a slot approach."""
    out: list[ClutterStart] = []
    for j in range(n):
        if pegs and rng.random() < 0.55:
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
            pegs = _draw_peg_starts(rng, slot_ids, n_clutter, held_out=inside)
            clutter = _draw_clutter(rng, n_clutter, pegs)
            act_delay = int(np.clip(round(float(lhs["act_delay"][j])), 0, 1))
            disturbance = 1 if float(lhs["disturbance"][j]) >= 0.65 else 0
            max_steps = STEPS_PER_PEG * k_pegs
            if disturbance:
                d_step = int(rng.integers(max(1, max_steps // 5), max(2, (4 * max_steps) // 5)))
                roll = rng.random()
                if roll < 0.50:
                    d_target = f"peg:{int(rng.integers(0, k_pegs))}"
                elif roll < 0.70:
                    d_target = "drop"  # open gripper / slip mid-episode
                else:
                    d_target = "board"
            else:
                d_step = 0
                d_target = "none"
            # Slot bias + independent peg delta so relative features retain residual.
            bias_mag = float(lhs["pose_bias_xy"][j])
            bias_ang = _u(rng, 0, 2 * math.pi)
            bias_xy = (bias_mag * math.cos(bias_ang), bias_mag * math.sin(bias_ang))
            bias_yaw = float(lhs["pose_bias_yaw"][j]) * (1.0 if rng.random() < 0.5 else -1.0)
            dmag = float(lhs["peg_bias_delta"][j])
            dang = _u(rng, 0, 2 * math.pi)
            peg_delta = (dmag * math.cos(dang), dmag * math.sin(dang))
            # Hidden search pitch scales with clearance / chamfer (not observable).
            clr = float(lhs["clearance"][j])
            ch = float(lhs["chamfer"][j])
            search_pitch = 0.00008 + 0.55 * clr + 0.00012 * ch
            eps.append(
                EpisodeConditions(
                    n_pegs=k_pegs,
                    clutter_bucket=bucket,
                    clearance=clr,
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
                    pose_bias_xy=(float(bias_xy[0]), float(bias_xy[1])),
                    pose_bias_yaw=float(bias_yaw),
                    peg_bias_delta_xy=(float(peg_delta[0]), float(peg_delta[1])),
                    bias_drift=float(lhs["bias_drift"][j]),
                    backlash=float(lhs["backlash"][j]),
                    chamfer=ch,
                    search_pitch=float(search_pitch),
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


def shape_one_hot(shape: str) -> list[float]:
    return [1.0 if shape == name else 0.0 for name in SHAPE_NAMES]
