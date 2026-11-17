"""v0.8 "Assembly IV": frontier-lab hard compound assembly (profile ``v08``).

Hard reset vs v0.7.x. Design goals (Macrocosmos / crux + owner brief):

- Procedural blueprints of many steps (target ~8–28), dense precedence, waves.
- Randomized geometry, tolerances, and dynamics (mass / friction / gain / latency).
- Partial observability: blueprint structure is published; answer-leaking cues are not.
  No key closeness, wrist detent, or retarget pulse. Absolute key yaw stays hidden.
- Exploit-resistant scoring: equal per-step weight; completed credit requires a
  physically verified end state that survives an end-of-episode perturbation.
  Unfinished steps have three small, action-earned physical milestones (lift,
  target alignment, stable release), with no credit for reset-only states.
  Ceiling of 1.0 is essentially unreachable.
- Closes the v0.7.5 hole class: slot-aligned key freebies, D-flat auto-arm, and cue
  hill-climbs that completed keys without a real index search.

Observation is **320-d** (288-d Assembly III layout + 32-d blueprint encoding). Action
stays 5-d. Old 288-input ONNX policies are rejected by the player image.

Eval cost stays sim-horizon bounded (``MAX_HORIZON_V08``) for the ~600 s / 64-ep / 2-CPU bar.
"""

from __future__ import annotations

import copy
import math

import mujoco
import numpy as np

from . import conditions as C
from . import profile as _profile
from .compound import (
    COLLAR_FEAT,
    COLLAR_H,
    COLLAR_MAX_PEG_OUTER,
    COLLAR_SEAT_XY,
    COLLAR_T,
    GLOBAL_FEAT,
    LYING_EXTRA,
    MAX_COLLARS,
    OBS_DIM_V06,
    TASK_PEG_FEAT,
    TIERS,
    TWIST_TARGET,
    WAVE_EXTRA,
    CollarSpec,
    CompoundEpisode,
    CompoundSpec,
    TaskStep,
    _collar_spot,
    _extend_pegs,
    _step_wave_of,
    _topo_random,
    wrap,
)
from .compound_v07 import (
    EVENT_KINDS_V07,
    KEY_HOLD,
    KEY_PROBE_FRAMES,
    REWORK_EXTRA,
    SCREW_TARGET,
    STACK_INTERFERENCE,
    STACK_SEAT_XY,
    AssemblyIIIEpisode,
    _KEY_YAW_REACH,
)
from .conditions import EpisodeConditions, sample_episodes
from .env import (
    BLOCK_H,
    MAX_PEGS,
    MAX_OPEN,
    PEG_LEN,
    SUBSTEPS,
)

# ---- vocabulary / scoring -----------------------------------------------------------------------
STEP_KINDS_V08 = ("insert", "collar", "twist", "rework", "screw", "stack", "key")
# Equal weights: farming easy inserts cannot dominate. Score is a mean over blueprint steps.
STEP_WEIGHT_V08 = {k: 1.0 for k in STEP_KINDS_V08}
STEP_BUDGET_V08 = {
    "insert": 90,
    "collar": 70,
    "twist": 55,
    "rework": 130,
    "screw": 110,
    "stack": 110,
    "key": 120,
}
MAX_HORIZON_V08 = 920
# Bounded, action-earned milestones on an unfinished step. Full completion still
# requires the end-state and perturbation checks below. These are fractions of a step,
# not additional reward on top of a completed step.
LIFT_CREDIT_V08 = 0.06
ALIGN_CREDIT_V08 = 0.06
STABLE_CREDIT_V08 = 0.06
PARTIAL_CAP_V08 = LIFT_CREDIT_V08 + ALIGN_CREDIT_V08 + STABLE_CREDIT_V08
PARTIAL_WEIGHTS_V08 = {"lift": LIFT_CREDIT_V08, "align": ALIGN_CREDIT_V08, "stable": STABLE_CREDIT_V08}
LIFT_CLEARANCE_V08 = 0.008  # part bottom above its support plane, in metres
GRASP_MIN_FRAMES_V08 = 2  # reject a single-frame contact transient
POSE_TOL_SCALE_V08 = 2.0  # near-target alignment is deliberately not a seat
STABLE_MIN_FRAMES_V08 = 8  # final 8 of 12 perturbation-settle frames
# Out-of-order completed steps earn almost nothing (dependency order matters).
OOO_CREDIT_V08 = 0.03
# End-state done credit floor/ceiling before precision × stability.
DONE_BASE_V08 = 0.80
DONE_TIME_V08 = 0.20

# Blueprint encoding appended after the 288-d v0.7 layout.
BLUEPRINT_FEAT = 32
OBS_DIM_V08 = OBS_DIM_V06 + BLUEPRINT_FEAT  # 320
KIND_CODE = {
    "insert": 1.0,
    "collar": 2.0,
    "twist": 3.0,
    "rework": 4.0,
    "screw": 5.0,
    "stack": 6.0,
    "key": 7.0,
}

# Key index: tighter windows; sampled *away* from slot symmetries (anti freebie).
KEY_TOL_V08 = (0.035, 0.055)
# Minimum angular distance from any slot symmetry / natural seat yaw.
KEY_ANTI_ALIGN = 0.22  # rad (~12.6°)

# Perturbation test at episode end.
PERTURB_FRAMES = 12
PERTURB_XY = 0.0018  # m impulse on free bodies
PERTURB_SETTLE_ZERO = True

# Tier mix: more hard episodes than v0.7 so the ceiling stays far away.
TIER_FRACTIONS_V08 = (0.20, 0.30, 0.30, 0.20)
TIER_CFG_V08 = {
    # Target step counts: c1 ~8–12, c2 ~10–16, c3 ~14–20, c4 ~18–28.
    "c1": dict(
        extra_pegs=1, collar_frac=0.20, twist_frac=0.15, screw_frac=0.25, rework_frac=0.22,
        stack_frac=0.45, key_frac=0.55, waves=(1, 2), edge_p=0.22, event_p=0.28, slack=1.55,
        reveal_p=1.00, touch_reveal_p=1.00, max_keys=2,
        post_insert_retarget_p=0.45, post_stack_retarget_p=0.40,
    ),
    "c2": dict(
        extra_pegs=1, collar_frac=0.22, twist_frac=0.18, screw_frac=0.30, rework_frac=0.26,
        stack_frac=0.50, key_frac=0.60, waves=(2, 2), edge_p=0.26, event_p=0.34, slack=1.40,
        reveal_p=1.00, touch_reveal_p=1.00, max_keys=2,
        post_insert_retarget_p=0.55, post_stack_retarget_p=0.48,
    ),
    "c3": dict(
        extra_pegs=2, collar_frac=0.25, twist_frac=0.20, screw_frac=0.35, rework_frac=0.30,
        stack_frac=0.55, key_frac=0.70, waves=(2, 3), edge_p=0.30, event_p=0.40, slack=1.25,
        reveal_p=1.00, touch_reveal_p=1.00, max_keys=3,
        post_insert_retarget_p=0.65, post_stack_retarget_p=0.55,
    ),
    "c4": dict(
        extra_pegs=2, collar_frac=0.28, twist_frac=0.22, screw_frac=0.40, rework_frac=0.34,
        stack_frac=0.60, key_frac=0.80, waves=(3, 4), edge_p=0.34, event_p=0.48, slack=1.10,
        reveal_p=1.00, touch_reveal_p=1.00, max_keys=3,
        post_insert_retarget_p=0.75, post_stack_retarget_p=0.65,
    ),
}
# Stronger perception noise than v0.7 (still secondary to graph/dynamics hardness).
# Drift retains the inherited easy-tier cap: the v0.8 public range itself is wider.
NOISE_SCALE_V08 = dict(slot_bias=0.85, peg_delta=0.80, glitch=0.75, drift=0.625)
# Dynamics event mix: emphasize physics identity changes over cue-facing retargets.
EVENT_WEIGHTS_V08 = np.array(
    # friction, gain, mass, delay, rebias, shove, swap_bias, misseat,
    # retarget, reorder, flip_twist, regraph, activate
    [0.12, 0.12, 0.12, 0.08, 0.08, 0.06, 0.10, 0.06, 0.08, 0.06, 0.04, 0.05, 0.03]
)
# Per-episode dynamics stretch applied on top of LHS draws (system ID required).
DYN_FRICTION = (0.55, 1.85)
DYN_GAIN = (0.45, 1.65)
DYN_MASS = (0.55, 2.20)
DYN_DELAY_EXTRA = (0, 3)  # extra control-step latency drawn once per episode


def _is_v08() -> bool:
    return _profile.get().name == "v08"


def _assign_v08(rng: np.random.Generator, n: int) -> list[str]:
    raw = [f * n for f in TIER_FRACTIONS_V08]
    counts = [int(x) for x in raw]
    order = sorted(range(len(TIERS)), key=lambda i: raw[i] - counts[i], reverse=True)
    for k in range(n - sum(counts)):
        counts[order[k % len(TIERS)]] += 1
    if n >= 4:
        for hard_i in (2, 3):
            if counts[hard_i] == 0:
                donor = max(range(hard_i), key=lambda i: counts[i])
                if counts[donor] >= 1:
                    counts[donor] -= 1
                    counts[hard_i] = 1
    tiers = [t for t, c in zip(TIERS, counts) for _ in range(c)]
    rng.shuffle(tiers)
    return tiers


def _harden_noise_v08(cond: EpisodeConditions) -> None:
    s = NOISE_SCALE_V08
    cond.slot_bias_list = [(x * s["slot_bias"], y * s["slot_bias"]) for x, y in cond.slot_bias_list]
    cond.clutter_bias_list = [(x * s["slot_bias"], y * s["slot_bias"]) for x, y in cond.clutter_bias_list]
    cond.peg_bias_delta_list = [(x * s["peg_delta"], y * s["peg_delta"]) for x, y in cond.peg_bias_delta_list]
    cond.glitch_rate *= s["glitch"]
    # Keep the v0.5 tier invariant: easy drift <= half the global range. The
    # global CI suite intentionally samples raw conditions as well as v08 graphs.
    # Preserve the easy-tier invariant even when LHS draw × tier scale is 0.8.
    cond.bias_drift *= s["drift"]


def _harden_starts_v08(cond: EpisodeConditions) -> None:
    """No held, near-hole, or correct-hole part at reset.

    Regenerate those starts through the normal loose/lying/tray sampler and then
    revalidate every peg against the actual slot geometry. This does not weaken
    v0.5's easy-tier distribution: that historical distribution is covered by
    ``tests/test_conditions.py``; v0.8 deliberately has no free-start stratum.
    """
    loose_w = np.array([0.68, 0.24, 0.08, 0.0, 0.0])
    for i, p in enumerate(cond.pegs):
        if p.start_type not in ("grasped", "near_hole"):
            continue
        fresh = C._draw_peg_starts_with_weights(
            np.random.default_rng(np.random.SeedSequence([
                0x53544152, i, p.slot, len(cond.pegs),
            ])),
            [p.slot], cond.slots, loose_w,
        )[0]
        p.start_type = fresh.start_type
        p.offset_xy = fresh.offset_xy
        p.height = fresh.height
        p.yaw_err = fresh.yaw_err
    # Independent geometry check (the sampler can make clearance tradeoffs in
    # dense boards). Move any too-close start onto a known clear grid cell.
    rng = np.random.default_rng(np.random.SeedSequence([
        0x464c4f4f, len(cond.pegs), len(cond.slots),
    ]))
    for p in cond.pegs:
        if p.start_type not in ("loose", "tray", "lying"):
            raise RuntimeError("v08 free-start sampler returned a privileged start")
        if str(p.drawn_start_type) in ("grasped", "near_hole"):
            p.drawn_start_type = p.start_type  # reflect the effective reset layout
        need = 2.0 * p.spec.outer_radius() + C.LAYOUT_WALL + 0.006
        if math.hypot(*p.offset_xy) >= need:
            continue
        for _try in range(256):
            off = (float(rng.uniform(-0.15, 0.15)), float(rng.uniform(-0.12, 0.12)))
            if C._start_clear(p.spec.x + off[0], p.spec.y + off[1], p.spec.outer_radius(), cond.slots, 0.006):
                p.offset_xy = off
                p.height = 0.0003 if p.start_type != "tray" else p.height
                break
        else:
            raise RuntimeError("v08 could not find an interaction-required peg spawn")


def _stretch_dynamics_v08(rng: np.random.Generator, cond: EpisodeConditions) -> None:
    """Per-episode system-ID stretch: mass / friction / gain / latency."""
    cond.friction = float(np.clip(cond.friction * rng.uniform(*DYN_FRICTION), 0.15, 2.5))
    cond.gain_scale = float(np.clip(cond.gain_scale * rng.uniform(*DYN_GAIN), 0.25, 2.2))
    cond.peg_mass = float(np.clip(cond.peg_mass * rng.uniform(*DYN_MASS), 0.003, 0.12))
    extra = int(rng.integers(DYN_DELAY_EXTRA[0], DYN_DELAY_EXTRA[1] + 1))
    cond.act_delay = int(min(4, int(cond.act_delay) + extra))


def _draw_event_v08(rng: np.random.Generator) -> dict:
    kind = str(rng.choice(EVENT_KINDS_V07, p=EVENT_WEIGHTS_V08 / EVENT_WEIGHTS_V08.sum()))
    if kind == "friction":
        return dict(kind=kind, scale=float(rng.uniform(0.25, 2.6)))
    if kind == "gain":
        return dict(kind=kind, scale=float(rng.uniform(0.30, 2.0)))
    if kind == "mass":
        return dict(kind=kind, scale=float(rng.uniform(0.40, 3.0)))
    if kind == "delay":
        return dict(kind=kind, steps=int(rng.integers(0, 6)))
    if kind == "rebias":
        return dict(kind=kind, mag=float(rng.uniform(0.0025, 0.010)), ang=float(rng.uniform(0, 2 * math.pi)))
    if kind == "swap_bias":
        return dict(kind=kind)
    if kind == "misseat":
        return dict(kind=kind, lift=float(rng.uniform(0.007, 0.016)))
    if kind in ("retarget", "reorder", "flip_twist", "regraph", "activate"):
        return dict(kind=kind)
    return dict(kind=kind, sigma=float(rng.uniform(0.16, 0.36)))


def _symmetry_angles(slot_yaw: float, n: int) -> list[float]:
    if n <= 0:
        return [float(slot_yaw)]
    return [float(wrap(slot_yaw + (2.0 * math.pi * k) / n)) for k in range(n)]


def sample_key_yaw_v08(rng: np.random.Generator, cond: EpisodeConditions, peg_i: int) -> float:
    """Hand-yaw window centre that is reachable but *not* near a natural seat yaw.

    Closes the v0.7.5 freebie class where non-round ``key_yaw`` sat on a slot symmetry
    (±~0.03 jitter) inside ``KEY_TOL``, so insert-at-slot-yaw + probe completed the key.
    """
    peg = cond.pegs[peg_i]
    spec = peg.spec
    slot_yaw = float(spec.yaw + cond.board_yaw)
    sym = int(spec.symmetry())
    forbidden = _symmetry_angles(slot_yaw, max(1, sym)) if sym > 0 else [slot_yaw, 0.0]
    # Also keep clear of wrist-zero (common open-loop rest).
    forbidden = list(forbidden) + [0.0]

    def _ok(yaw: float) -> bool:
        if abs(yaw) > _KEY_YAW_REACH:
            return False
        return all(abs(wrap(yaw - f)) >= KEY_ANTI_ALIGN for f in forbidden)

    for _ in range(48):
        if spec.shape == "round" or sym == 0:
            yaw = float(rng.uniform(-_KEY_YAW_REACH, _KEY_YAW_REACH))
        else:
            # Midway between neighbouring symmetries, plus jitter that stays off-axis.
            k_sym = int(rng.integers(0, max(1, sym)))
            mid = slot_yaw + (2.0 * math.pi * (k_sym + 0.5)) / max(1, sym)
            yaw = float(wrap(mid + float(rng.uniform(-0.04, 0.04))))
            yaw = float(np.clip(yaw, -_KEY_YAW_REACH, _KEY_YAW_REACH))
        if _ok(yaw):
            return yaw
    # Fallback: farthest reachable point from the nearest forbidden angle.
    best, best_d = 0.8, -1.0
    for cand in np.linspace(-_KEY_YAW_REACH, _KEY_YAW_REACH, 61):
        d = min(abs(wrap(float(cand) - f)) for f in forbidden)
        if d > best_d:
            best_d, best = d, float(cand)
    return best


def make_compound_v08(rng: np.random.Generator, cond: EpisodeConditions, tier: str) -> CompoundSpec:
    cfg = TIER_CFG_V08[tier]
    _harden_noise_v08(cond)
    _harden_starts_v08(cond)
    _stretch_dynamics_v08(rng, cond)
    _extend_pegs(rng, cond, cfg["extra_pegs"])
    _harden_starts_v08(cond)  # extra pegs must obey the same no-free-start rule
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

    cands = [i for i, p in enumerate(cond.pegs) if p.spec.outer_radius() <= COLLAR_MAX_PEG_OUTER]
    n_col = min(MAX_COLLARS, len(cands), int(round(cfg["collar_frac"] * max(1, len(cands)))))
    n_stack = min(
        MAX_COLLARS - n_col,
        max(0, len(cands) - n_col),
        max(1, int(round(cfg["stack_frac"] * max(1, len(cands))))),
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
        outer = cond.pegs[peg_i].spec.outer_radius()
        if kind == "stack":
            r_in = max(0.0010, outer - float(rng.uniform(*STACK_INTERFERENCE)))
        else:
            r_in = outer + float(rng.uniform(0.0005, 0.0010))
        spot = _collar_spot(rng, cond, r_in + COLLAR_T, taken)
        taken.append(spot)
        w = int(rng.integers(peg_wave[peg_i], W))
        collars.append(CollarSpec(peg=peg_i, r_in=r_in, offset_xy=spot, wave=w, kind=kind))

    used_slots = {p.slot for p in cond.pegs}
    free_slots = [s for s in range(C.N_SLOTS) if s not in used_slots]
    rework_n = min(len(free_slots), max(1, int(round(cfg["rework_frac"] * K))) if K >= 3 else 0)
    rework_cands = [i for i in range(K) if i not in grasped and peg_wave[i] == 0]
    rework_pegs: dict[int, int] = {}
    if rework_n and rework_cands and free_slots:
        pick = rng.choice(rework_cands, size=min(rework_n, len(rework_cands)), replace=False).tolist()
        wrongs = rng.choice(free_slots, size=len(pick), replace=False).tolist()
        for i, ws in zip(pick, wrongs):
            rework_pegs[int(i)] = int(ws)

    rounds = [i for i, p in enumerate(cond.pegs) if p.spec.shape == "round"]
    rounds_easy = [
        i for i in rounds
        if peg_wave[i] == 0 and cond.pegs[i].start_type not in ("lying",)
    ] or rounds
    screw_pegs = [int(i) for i in rounds_easy if rng.random() < cfg["screw_frac"]]
    twist_pegs = [
        int(i) for i in rounds
        if i not in screw_pegs and rng.random() < cfg["twist_frac"]
    ]

    twist_or_screw = set(twist_pegs) | set(screw_pegs)
    key_cands = [i for i in range(K) if i not in rework_pegs and i not in twist_or_screw]
    max_keys = int(cfg.get("max_keys", 2))
    n_key = min(len(key_cands), max_keys, max(1, int(round(cfg["key_frac"] * K))))
    key_pegs = set(rng.choice(key_cands, size=n_key, replace=False).tolist() if n_key and key_cands else [])
    if not key_pegs and key_cands:
        key_pegs.add(int(rng.choice(key_cands)))
    if len(key_pegs) > max_keys:
        key_pegs = set(list(key_pegs)[:max_keys])

    def _add_stack_on(peg_i: int) -> bool:
        if len(collars) >= MAX_COLLARS:
            return False
        if any(p == peg_i for p, _ in ring_pegs):
            return False
        if peg_i not in cands and cond.pegs[peg_i].spec.outer_radius() > COLLAR_MAX_PEG_OUTER:
            return False
        outer = cond.pegs[peg_i].spec.outer_radius()
        r_in = max(0.0010, outer - float(rng.uniform(*STACK_INTERFERENCE)))
        spot = _collar_spot(rng, cond, r_in + COLLAR_T, taken)
        taken.append(spot)
        w = int(rng.integers(peg_wave[peg_i], W))
        collars.append(CollarSpec(peg=peg_i, r_in=r_in, offset_xy=spot, wave=w, kind="stack"))
        ring_pegs.append((peg_i, "stack"))
        return True

    # Pair stacks onto ~40% of key pegs (harder than v0.7.5's 25%).
    for peg_i in list(key_pegs):
        if any(p == peg_i and k == "stack" for p, k in ring_pegs):
            continue
        if rng.random() < 0.40:
            _add_stack_on(int(peg_i))
    if not any(k == "stack" for _, k in ring_pegs):
        non_key = [i for i in cands if i not in key_pegs and i not in {p for p, _ in ring_pegs}]
        prefer = non_key or [i for i in cands if i not in {p for p, _ in ring_pegs}] or list(cands)
        for peg_i in prefer:
            if _add_stack_on(int(peg_i)):
                break

    steps: list[TaskStep] = []
    edges: set[tuple[int, int]] = set()
    insert_idx: dict[int, int] = {}
    touch_p = float(cfg.get("touch_reveal_p", 1.0))
    for i in range(K):
        if i in rework_pegs:
            steps.append(TaskStep("rework", i, i, wrong_slot=rework_pegs[i]))
        else:
            steps.append(TaskStep("insert", i, i))
        insert_idx[i] = len(steps) - 1
        if i in key_pegs:
            steps.append(
                TaskStep(
                    "key",
                    i,
                    i,
                    key_yaw=sample_key_yaw_v08(rng, cond, i),
                    key_tol=float(rng.uniform(*KEY_TOL_V08)),
                    touch_reveal=bool(rng.random() < touch_p),
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
                touch_reveal=bool(rng.random() < touch_p * 0.5),
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
                touch_reveal=bool(rng.random() < touch_p * 0.5),
            )
        )
        edges.add((insert_idx[i], len(steps) - 1))

    for j, col in enumerate(collars):
        kind = "stack" if col.is_stack else "collar"
        steps.append(TaskStep(kind, j, col.peg))
        sc = len(steps) - 1
        edges.add((insert_idx[col.peg], sc))
        # Stack/collar after twist/screw often; keys after stacks when paired.
        tw = [k for k, s in enumerate(steps) if s.kind in ("twist", "screw") and s.peg == col.peg]
        if tw and rng.random() < 0.60:
            edges.add((tw[0], sc) if rng.random() < 0.55 else (sc, tw[0]))
        if kind == "stack":
            keys = [k for k, s in enumerate(steps) if s.kind == "key" and s.peg == col.peg]
            if keys and rng.random() < 0.55:
                edges.add((sc, keys[0]))

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
        s.prereq = sorted(set(s.prereq))

    # Keys keep insert/rework + optional stack prereqs (harder than v0.7.4 strip-to-insert).
    for s in steps:
        if s.kind != "key":
            continue
        keep = [
            p for p in s.prereq
            if steps[p].kind in ("insert", "rework", "stack") and steps[p].peg == s.peg
        ]
        if not keep and s.peg in insert_idx:
            keep = [int(insert_idx[s.peg])]
        s.prereq = sorted(set(keep))

    for k, s in enumerate(steps):
        if s.kind not in ("twist", "screw", "key"):
            continue
        if rng.random() >= cfg["reveal_p"]:
            continue
        gates = list(s.prereq) if s.prereq else ([insert_idx[s.peg]] if s.peg in insert_idx else [])
        if gates:
            s.reveal_after = int(gates[int(rng.integers(0, len(gates)))])
        elif s.kind == "key" and s.peg in insert_idx:
            s.reveal_after = int(insert_idx[s.peg])

    budget = 0
    for s in steps:
        budget += STEP_BUDGET_V08[s.kind]
        if s.kind in ("insert", "rework") and cond.pegs[s.peg].start_type == "lying":
            budget += LYING_EXTRA
        if s.kind == "rework":
            budget += REWORK_EXTRA
    max_steps = int(cfg["slack"] * budget) + WAVE_EXTRA * max(0, W - 1)
    cond.max_steps = min(MAX_HORIZON_V08, max(400, max_steps))
    wave_time = [0] + [int(cond.max_steps * min(0.80, w * float(rng.uniform(0.12, 0.24)))) for w in range(1, W)]
    for w in range(2, W):
        wave_time[w] = max(wave_time[w], wave_time[w - 1] + 18)
    events = [(_draw_event_v08(rng) if rng.random() < cfg["event_p"] else None) for _ in steps]
    return CompoundSpec(
        tier=tier, steps=steps, collars=collars, peg_wave=peg_wave, n_waves=W,
        wave_time=wave_time, events=events,
    )


def sample_compound_v08(seed: int, episodes_per_stratum: int, split: str = "live"):
    """v0.5 board draw + v0.8 Assembly IV task graph per episode."""
    ranges, conds = sample_episodes(seed, episodes_per_stratum, split)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x41333408, *C.build_salt()]))
    per = episodes_per_stratum
    for s in range(0, len(conds), per):
        tiers = _assign_v08(rng, len(conds[s : s + per]))
        for c, t in zip(conds[s : s + per], tiers):
            c.compound = make_compound_v08(rng, c, t)
    return ranges, conds


# ---- episode ------------------------------------------------------------------------------------
class AssemblyIVEpisode(AssemblyIIIEpisode):
    """v0.8 episode: anti-aligned keys, no answer cues, end-state + perturbation scoring hooks."""

    def __init__(self, cond: EpisodeConditions, rng: np.random.Generator):
        # Parent spawn fallbacks and dynamics events mutate conditions. Keep
        # sampled round definitions immutable across workers and repeat evals.
        cond = copy.deepcopy(cond)
        super().__init__(cond, rng)
        self._stability_ok = [True] * len(cond.compound.steps)
        self._precision = [1.0] * len(cond.compound.steps)
        self._finalized = False
        self._milestones_v08 = [dict(lift=False, align=False, stable=False) for _ in cond.compound.steps]
        self._grasp_run_v08 = {}
        # Only post-action states may earn progress; the reset state is never a baseline.
        self._action_earned_v08 = False
        # Patch tier cfg lookups used by parent retarget helpers.
        self._tier_cfg_override = TIER_CFG_V08

    def _step_state(self, k: int) -> tuple[bool, float]:
        """v0.8 physical latch; scoring uses explicit, bounded evidence separately."""
        done, _ = super()._step_state(k)
        return bool(done), 0.0

    def _snapshot_parts_v08(self) -> None:
        self._part_prev_z_v08 = {}
        for k, s in enumerate(self.cs.steps):
            ev = self._part_evidence_v08(s)
            if ev is not None:
                self._part_prev_z_v08[k] = float(ev[1][2])

    def _part_evidence_v08(self, s: TaskStep):
        if s.kind in ("collar", "stack"):
            j = s.part
            if self.collar_parked[j]:
                return None
            p, q = self.collar_pose(j)
            return self.collar_body[j], p, q, COLLAR_H, self.cs.collars[j].peg
        i = s.peg
        if self.peg_parked[i]:
            return None
        p, q = self.peg_pose(i)
        return self.peg_body[i], p, q, PEG_LEN, i

    def _update_milestones_v08(self, *, settling: bool = False) -> None:
        """Record action-earned grasp/lift and near-target evidence.

        A lift requires a closed jaw, a clear part bottom, and upward part motion
        relative to its pre-action state. Contact alone, wrist motion alone, and
        spawn height do not qualify. Near-target credit requires a prior earned
        lift for that part and a held upright part, never a passive near-hole part.
        """
        if not self._action_earned_v08:
            return
        for k, s in enumerate(self.cs.steps):
            ms = self._milestones_v08[k]
            ev = self._part_evidence_v08(s)
            if ev is None or self.step_done[k]:
                self._grasp_run_v08[k] = 0
                continue
            body, p, q, length, peg_i = ev
            touching = self._gripper_touching(body)
            closed = float(self.data.qpos[4] + self.data.qpos[5]) < 0.014
            bottom = float(p[2]) - length / 2.0
            support = BLOCK_H if s.kind == "stack" else 0.0
            lifted = bottom >= support + LIFT_CLEARANCE_V08
            prev_z = getattr(self, "_part_prev_z_v08", {}).get(k)
            rising = prev_z is not None and float(p[2]) - prev_z > 0.00015
            if touching and closed and lifted and rising and not settling:
                self._grasp_run_v08[k] = self._grasp_run_v08.get(k, 0) + 1
            else:
                self._grasp_run_v08[k] = 0
            if self._grasp_run_v08[k] >= GRASP_MIN_FRAMES_V08:
                ms["lift"] = True
            hx, hy = self.slot_xy[peg_i]
            dxy = float(np.hypot(p[0] - hx, p[1] - hy))
            if s.kind == "collar":
                tol = POSE_TOL_SCALE_V08 * COLLAR_SEAT_XY
                target_z = BLOCK_H + length / 2.0
                ztol = 0.008
            elif s.kind == "stack":
                tol = POSE_TOL_SCALE_V08 * STACK_SEAT_XY
                pp, _ = self.peg_pose(peg_i)
                target_z = float(pp[2]) + PEG_LEN / 2.0 + length / 2.0
                ztol = 0.008
            else:
                tol = max(0.003, POSE_TOL_SCALE_V08 * float(self.r_hole[peg_i]))
                target_z = BLOCK_H + PEG_LEN / 2.0
                ztol = 0.010
            near_pose = dxy <= tol and abs(float(p[2]) - target_z) <= ztol and self._upright(q)
            # Target alignment while held follows a real lift; no reset-time or
            # vibration-only path can earn it.
            if ms["lift"] and touching and closed and near_pose and not settling:
                ms["align"] = True
            # Final stability is evaluated after release, not inferred from a latch.
            # ``_finalize_stability`` calls this only for the last settle frames.
            if settling and near_pose and not touching and ms["align"]:
                ms["stable_frames"] = ms.get("stable_frames", 0) + 1
                if ms["stable_frames"] >= STABLE_MIN_FRAMES_V08:
                    ms["stable"] = True

    def _key_auto_arm(self, peg_i: int) -> bool:
        """v0.8: never auto-arm. Leave/reenter is mandatory for every key family."""
        return False

    def _retarget_key(self, k: int, *, reason: str = "retarget") -> dict | None:
        s = self.cs.steps[k]
        if s.kind != "key" or self.step_done[k]:
            return None
        old = float(s.key_yaw)
        new = old
        for _ in range(12):
            cand = sample_key_yaw_v08(self.rng, self.c, s.peg)
            if abs(wrap(cand - old)) > max(0.15, 2.5 * s.key_tol):
                new = cand
                break
            new = cand
        s.key_yaw = new
        if self.rng.random() < 0.75:
            s.key_tol = float(self.rng.uniform(*KEY_TOL_V08))
        self._key_hold[s.peg] = 0
        self._key_index_armed[s.peg] = False
        # Pulse kept internally for bookkeeping but **not** published in observe().
        self._key_retarget_pulse[s.peg] = 0
        return dict(kind=reason, step=self.steps, peg=int(s.peg), key_step=int(k), old_yaw=old, new_yaw=float(s.key_yaw))

    def _maybe_post_insert_retarget(self, seat_k: int) -> None:
        s = self.cs.steps[seat_k]
        if s.kind not in ("insert", "rework"):
            return
        cfg = TIER_CFG_V08.get(self.cs.tier, {})
        p = float(cfg.get("post_insert_retarget_p", 0.0))
        if p <= 0.0 or self.rng.random() >= p:
            return
        for k, ks in enumerate(self.cs.steps):
            if ks.kind == "key" and ks.peg == s.peg and not self.step_done[k]:
                payload = self._retarget_key(k, reason="retarget")
                if payload is not None:
                    payload["auto"] = "post_insert"
                    self.events_fired.append(payload)
                    self._pick_active_key(prefer=k)
                break

    def _maybe_post_stack_retarget(self, stack_k: int) -> None:
        s = self.cs.steps[stack_k]
        if s.kind != "stack":
            return
        cfg = TIER_CFG_V08.get(self.cs.tier, {})
        p = float(cfg.get("post_stack_retarget_p", 0.0))
        if p <= 0.0 or self.rng.random() >= p:
            return
        for k, ks in enumerate(self.cs.steps):
            if ks.kind == "key" and ks.peg == s.peg and not self.step_done[k]:
                payload = self._retarget_key(k, reason="retarget")
                if payload is not None:
                    payload["auto"] = "post_stack"
                    self.events_fired.append(payload)
                    self._pick_active_key(prefer=k)
                break

    def step(self, action: np.ndarray) -> None:
        if self.done:
            return
        self._snapshot_parts_v08()
        self._action_earned_v08 = True
        # Run parent Assembly III step, then strip answer-leaking detent torque.
        wrist_before = float(self._wrist_torque[2]) if hasattr(self, "_wrist_torque") else 0.0
        super().step(action)
        # Remove detent torque the parent may have injected (v0.7.5 cue).
        if hasattr(self, "_wrist_torque"):
            # Recompute: keep physical torque only — drop synthetic detent by restoring
            # any jump larger than a soft-contact scale while pressing a keyed peg.
            # Safer: zero the detent by subtracting KEY_DETENT if we detect key press path.
            # Parent adds ~0.22 N·m; clamp wrist torque z to the pre-step value when the
            # only change was the synthetic detent (no real contact impulse that large).
            # Simplest seal: never publish detent — observe uses forces after this, so
            # overwrite torque z with a contact-only estimate from mj_data if available.
            try:
                # Prefer raw sensor-less contact torque proxy already in parent; strip bump.
                if abs(float(self._wrist_torque[2]) - wrist_before) > 0.10:
                    # Likely detent; revert toward pre-step.
                    self._wrist_torque[2] = wrist_before
            except Exception:
                pass
        self._update_milestones_v08()
        # No insert latch that survives pull-out for *scoring* — physical verify at finalize.
        # (Parent may still latch step_done; finalize / score re-check end state.)
        if self.done and not self._finalized:
            self._finalize_stability()

    def _physical_step_ok(self, k: int) -> tuple[bool, float]:
        """End-state physical check for step k. Returns (ok, precision in [0,1])."""
        if not self._action_earned_v08:
            return False, 0.0
        s = self.cs.steps[k]
        if s.kind == "insert":
            i = s.part
            if self.peg_parked[i]:
                return False, 0.0
            ok = bool(self.seated_now[i] or self._peg_in_hole(i)) and not self._gripper_touching(self.peg_body[i])
            if not ok:
                return False, 0.0
            p, _ = self.peg_pose(i)
            hx, hy = self.slot_xy[i]
            dxy = float(np.hypot(p[0] - hx, p[1] - hy))
            prec = float(np.clip(1.0 - dxy / max(self.r_hole[i], 1e-4), 0.0, 1.0))
            return True, prec
        if s.kind == "rework":
            i = s.part
            if self.peg_parked[i]:
                return False, 0.0
            if not self._rework_extracted(i):
                return False, 0.0
            ok = bool(self.seated_now[i] or self._peg_in_hole(i)) and not self._gripper_touching(self.peg_body[i])
            if not ok:
                return False, 0.0
            p, _ = self.peg_pose(i)
            hx, hy = self.slot_xy[i]
            dxy = float(np.hypot(p[0] - hx, p[1] - hy))
            return True, float(np.clip(1.0 - dxy / max(self.r_hole[i], 1e-4), 0.0, 1.0))
        if s.kind == "collar":
            j = s.part
            if self.collar_parked[j]:
                return False, 0.0
            ok, _ = CompoundEpisode._step_state(self, k)
            if not ok:
                return False, 0.0
            p, _ = self.collar_pose(j)
            hx, hy = self.slot_xy[self.cs.collars[j].peg]
            dxy = float(np.hypot(p[0] - hx, p[1] - hy))
            return True, float(np.clip(1.0 - dxy / 0.003, 0.0, 1.0))
        if s.kind == "stack":
            j = s.part
            if self.collar_parked[j] or not self._stack_mated[j]:
                return False, 0.0
            if not self._stack_seated_now(j):
                return False, 0.0
            p, _ = self.collar_pose(j)
            hx, hy = self.slot_xy[self.cs.collars[j].peg]
            dxy = float(np.hypot(p[0] - hx, p[1] - hy))
            return True, float(np.clip(1.0 - dxy / 0.003, 0.0, 1.0))
        if s.kind in ("twist", "screw"):
            i = s.part
            frac = float(np.clip(self.twist_acc[i] * s.twist_dir / max(1e-6, s.twist_target), 0.0, 1.0))
            ok = frac >= 0.80 and (self._peg_in_hole(i) or self.seated_now[i])
            return ok, float(np.clip(frac, 0.0, 1.0))
        if s.kind == "key":
            i = s.part
            if not (self.seated_now[i] or self._peg_in_hole(i)):
                return False, 0.0
            if self._peg_probe_frames[i] < KEY_PROBE_FRAMES:
                return False, 0.0
            if not self._key_index_armed[i]:
                return False, 0.0
            if not self._key_aligned(s):
                return False, 0.0
            if self._gripper_touching(self.peg_body[i]):
                return False, 0.0
            # Require hold latch or prior done with continued alignment.
            if self._key_hold[i] < KEY_HOLD and not self.step_done[k]:
                return False, 0.0
            err = abs(wrap(self._key_hand_yaw() - s.key_yaw))
            prec = float(np.clip(1.0 - err / max(s.key_tol, 1e-3), 0.0, 1.0))
            return True, prec
        return bool(self.step_done[k]), 1.0 if self.step_done[k] else 0.0

    def _finalize_stability(self) -> None:
        """Perturb seated parts and re-check end states. Call once when the episode ends."""
        if self._finalized:
            return
        self._finalized = True
        if not self._action_earned_v08:
            # Reset-only probes can inspect state, but cannot earn score.
            self.step_done = [False] * len(self.cs.steps)
            self._stability_ok = [False] * len(self.cs.steps)
            return
        n = len(self.cs.steps)
        self._stability_ok = [True] * n
        self._precision = [0.0] * n
        # Snapshot claimed completions.
        claimed = []
        for k in range(n):
            ok, prec = self._physical_step_ok(k)
            self._precision[k] = prec
            if ok and self.step_done[k]:
                claimed.append(k)
            elif ok and not self.step_done[k]:
                # Do not synthesize completion from a passive end pose. The
                # physical latch must have fired during an action transition.
                self._stability_ok[k] = False
            elif self.step_done[k] and not ok:
                # Latch lied (e.g. insert pull-out) — clear for scoring.
                self.step_done[k] = False
                self._stability_ok[k] = False

        # Apply a small XY impulse to free pegs/collars that look seated, then settle.
        for i in range(self.n_pegs):
            if self.peg_parked[i]:
                continue
            if not (self.seated_now[i] or self._peg_in_hole(i)):
                continue
            ang = float(self.rng.uniform(0, 2 * math.pi))
            vx = PERTURB_XY * math.cos(ang) / max(PHYSICS_PROXY_DT(), 1e-3)
            vy = PERTURB_XY * math.sin(ang) / max(PHYSICS_PROXY_DT(), 1e-3)
            adr = self.peg_vadr[i]
            self.data.qvel[adr] += vx
            self.data.qvel[adr + 1] += vy
        for j in range(self.n_collars):
            if self.collar_parked[j]:
                continue
            ang = float(self.rng.uniform(0, 2 * math.pi))
            vx = 0.7 * PERTURB_XY * math.cos(ang) / max(PHYSICS_PROXY_DT(), 1e-3)
            vy = 0.7 * PERTURB_XY * math.sin(ang) / max(PHYSICS_PROXY_DT(), 1e-3)
            adr = self.collar_vadr[j]
            self.data.qvel[adr] += vx
            self.data.qvel[adr + 1] += vy
        mujoco.mj_forward(self.model, self.data)
        # Settle with open gripper, no motion (does not advance episode score clock meaningfully —
        # episode already done; we only mutate poses for the stability check).
        saved_steps = self.steps
        saved_done = self.done
        saved_ctrl = self.data.ctrl.copy()
        saved_grip = float(self.grip)
        self.grip = MAX_OPEN
        self._apply_ctrl()  # open the jaws; ctrl=0 would command a closed grasp
        self.done = False  # allow physics substeps via a private settle
        for settle_i in range(PERTURB_FRAMES):
            self._touching_bodies_cache = None
            # Direct physics settle without full control bookkeeping.
            for _s in range(SUBSTEPS):
                mujoco.mj_step(self.model, self.data)
            # Refresh seated flags lightly.
            for i in range(self.n_pegs):
                if self.peg_parked[i]:
                    continue
                self.seated_now[i] = bool(self._peg_in_hole(i) and self._upright(self.peg_pose(i)[1]))
            if settle_i >= PERTURB_FRAMES - STABLE_MIN_FRAMES_V08:
                self._update_milestones_v08(settling=True)
        self.done = saved_done
        self.steps = saved_steps
        self.grip = saved_grip
        self.data.ctrl[:] = saved_ctrl
        self._touching_bodies_cache = None
        mujoco.mj_forward(self.model, self.data)

        for k in claimed:
            ok, prec = self._physical_step_ok(k)
            self._precision[k] = min(self._precision[k], prec) if ok else 0.0
            if not ok:
                self._stability_ok[k] = False
                self.step_done[k] = False

    def observe(self, noise_rng: np.random.Generator) -> np.ndarray:
        # Start from v0.7 observe, then strip answer cues and append blueprint features.
        o7 = super().observe(noise_rng).astype(np.float64)
        assert o7.shape[0] == OBS_DIM_V06
        task0 = OBS_DIM_V06 - (MAX_PEGS * TASK_PEG_FEAT + MAX_COLLARS * COLLAR_FEAT + GLOBAL_FEAT)
        for k, s in enumerate(self.cs.steps):
            if s.kind != "key":
                continue
            b = task0 + s.peg * TASK_PEG_FEAT
            # Strip closeness / retarget pulse / key_tol leak. Keep blocked flag only.
            o7[b + 2] = 0.0
            o7[b + 3] = 0.0
            # feat 4 already blocked; clear tol packing in feat 5 if it held key_tol.
            if o7[b + 5] != 0.0 and o7[b + 5] < 2.5:
                # key_tol*10 is typically 0.35–0.55; collar flags are 1 or 2.
                o7[b + 5] = 0.0
            # Publish a coarse "key present / needs index" mark without answer geometry.
            revealed = self._params_revealed(s)
            if revealed and self._peg_touched[s.peg] and self._key_is_active(k):
                o7[b + 2] = 0.5  # present-mark only (not closeness)
        bp = self._blueprint_encoding()
        o = np.concatenate([o7, bp])
        assert o.shape[0] == OBS_DIM_V08
        return o.astype(np.float32)

    def _blueprint_encoding(self) -> np.ndarray:
        """Compact published blueprint: kinds, peg ids, blocked/done, graph size."""
        out = np.zeros(BLUEPRINT_FEAT, dtype=np.float64)
        steps = self.cs.steps
        n = len(steps)
        out[0] = n / 40.0
        out[1] = {"c1": 0.25, "c2": 0.50, "c3": 0.75, "c4": 1.0}.get(self.cs.tier, 0.0)
        out[2] = self.cs.n_waves / 4.0
        out[3] = sum(self.step_done) / max(1, n)
        # Up to 12 steps × 2 floats: kind_code/10, integer peg id + 0.5*blocked + 0.25*done
        for i, s in enumerate(steps[:12]):
            out[4 + 2 * i] = KIND_CODE.get(s.kind, 0.0) / 10.0
            # Integer peg id plus disjoint flag bits. The old peg/8 packing
            # aliased (peg 1 + done + blocked) with (peg 3 + blocked).
            flags = float(s.peg) + 0.5 * float(self.step_blocked(i)) + 0.25 * float(self.step_done[i])
            out[4 + 2 * i + 1] = flags
        # Remaining: mean prereq count, active-key index (normalized), horizon fraction.
        out[28] = float(np.mean([len(s.prereq) for s in steps])) / 4.0 if steps else 0.0
        ak = -1 if self._active_key is None else int(self._active_key)
        out[29] = (ak + 1) / max(1, n)
        out[30] = self.steps / max(1, self.c.max_steps)
        out[31] = float(self.parts_lost) / max(1, self.n_pegs)
        return out


def PHYSICS_PROXY_DT() -> float:
    return float(_profile.get().physics_dt)


def step_credit_v08(
    done: bool,
    in_order: bool,
    done_at: int | None,
    horizon: int,
    progress: float | dict | None = None,
    *,
    precision: float = 1.0,
    stable: bool = True,
) -> float:
    if done and stable:
        time_term = 0.0
        if done_at is not None and int(done_at) >= 0:
            time_term = max(0.0, 1.0 - float(done_at) / max(1, horizon))
        base = DONE_BASE_V08 + DONE_TIME_V08 * time_term
        if in_order:
            return float(base * (0.70 + 0.30 * float(np.clip(precision, 0.0, 1.0))))
        return float(OOO_CREDIT_V08)
    if isinstance(progress, dict):
        return float(sum(PARTIAL_WEIGHTS_V08[name] for name in PARTIAL_WEIGHTS_V08 if progress.get(name, False)))
    # Legacy numeric progress is not a valid v0.8 evidence source.
    return 0.0


def _partial_credit_v08(ep: AssemblyIVEpisode | CompoundEpisode, k: int) -> float:
    if not isinstance(ep, AssemblyIVEpisode) or not ep._action_earned_v08:
        return 0.0
    if not all(ep.step_done[p] for p in ep.cs.steps[k].prereq):
        return 0.0
    evidence = ep._milestones_v08[k]
    return float(sum(w for name, w in PARTIAL_WEIGHTS_V08.items() if evidence.get(name, False)))


def step_results_v08(ep: AssemblyIVEpisode | CompoundEpisode) -> list[dict]:
    if isinstance(ep, AssemblyIVEpisode) and ep.done and not ep._finalized:
        ep._finalize_stability()
    stability = getattr(ep, "_stability_ok", [True] * len(ep.cs.steps))
    precision = getattr(ep, "_precision", [1.0] * len(ep.cs.steps))
    rows = []
    for k, s in enumerate(ep.cs.steps):
        phys_done = bool(ep.step_done[k] and stability[k])
        cr = (
            step_credit_v08(
                True, bool(ep.step_in_order[k]), ep.step_done_at[k], ep.c.max_steps,
                precision=float(precision[k]), stable=True,
            )
            if phys_done else _partial_credit_v08(ep, k)
        )
        evidence = getattr(ep, "_milestones_v08", [{} for _ in ep.cs.steps])[k]
        rows.append(
            {
                "done": phys_done,
                "done_at": ep.step_done_at[k],
                "in_order": bool(ep.step_in_order[k]),
                "credit": round(float(cr), 5),
                "precision": round(float(precision[k]), 5),
                "stable": bool(stability[k]),
                "partial": {
                    name: bool(evidence.get(name, False))
                    for name in PARTIAL_WEIGHTS_V08
                },
            }
        )
    return rows


def compound_score_v08(ep: AssemblyIVEpisode | AssemblyIIIEpisode | CompoundEpisode) -> dict:
    """Equal-weight end-state score with precision × stability.

    For each blueprint step:
    - Must be physically verified at episode end (and survive perturbation when finalized).
    - In-order completions earn ``DONE_BASE + DONE_TIME * time_term``, scaled by precision.
    - Out-of-order completions earn ``OOO_CREDIT_V08`` only.
    - Unfinished frontier steps may earn up to ``PARTIAL_CAP_V08`` from progress.
    - Episode score = mean over steps (equal weights). Ceiling 1.0 requires every step
      done in order at t≈0 with perfect precision and stability — effectively unreachable.
    """
    if isinstance(ep, AssemblyIVEpisode) and ep.done and not ep._finalized:
        ep._finalize_stability()

    n = len(ep.cs.steps)
    if n == 0:
        return dict(score=0.0, n_steps=0, n_done=0, n_in_order=0, per_kind={}, events=0)

    total = 0.0
    per_kind: dict[str, list[float]] = {k: [] for k in STEP_KINDS_V08}
    stability = getattr(ep, "_stability_ok", [True] * n)
    precision = getattr(ep, "_precision", [1.0] * n)

    for k, s in enumerate(ep.cs.steps):
        # Re-verify physical state for non-IV episodes (tools may pass III).
        if isinstance(ep, AssemblyIVEpisode):
            phys_ok = bool(ep._action_earned_v08 and ep.step_done[k] and stability[k])
            prec = float(precision[k]) if phys_ok else 0.0
        else:
            phys_ok = bool(ep.step_done[k])
            prec = 1.0 if phys_ok else 0.0

        if phys_ok:
            time_term = 0.0
            at = ep.step_done_at[k]
            if at is not None and int(at) >= 0:
                time_term = max(0.0, 1.0 - float(at) / max(1, ep.c.max_steps))
            base = DONE_BASE_V08 + DONE_TIME_V08 * time_term
            if ep.step_in_order[k]:
                cr = base * (0.70 + 0.30 * prec)
            else:
                cr = OOO_CREDIT_V08
        else:
            # Only explicit post-action physical evidence earns partial credit.
            cr = _partial_credit_v08(ep, k)

        total += cr
        per_kind.setdefault(s.kind, []).append(cr)

    score = total / n
    # Mild stability factor: fraction of claimed completions that survived (already in flags).
    return dict(
        score=float(score),
        n_steps=n,
        n_done=int(sum(1 for k in range(n) if ep.step_done[k] and stability[k])),
        n_in_order=int(sum(1 for k in range(n) if ep.step_done[k] and ep.step_in_order[k] and stability[k])),
        per_kind={k: (round(float(np.mean(v)), 3) if v else None) for k, v in per_kind.items()},
        events=len(ep.events_fired),
    )
