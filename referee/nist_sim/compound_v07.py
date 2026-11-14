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
  key fields are zeroed in the observation. ``touch_reveal`` also zeros them until the gripper
  has touched that peg (interaction-gated).
- v0.7.5 key obs: absolute key yaw is still **not** published. The index is **hand yaw**
  (wrist) vs the hidden window — soft contacts rarely spin a seated peg. After reveal +
  press-probe (``KEY_PROBE_FRAMES``), task feat 2 carries a retarget/disarm pulse and feat 3
  carries a coarse noisy closeness cue (unsigned; no signed yaw). ``key_tol`` may pack into
  the collar-flag slot when free. Inactive keys report ``blocked=1``. Wrist yaw-torque also
  gets a noisy detent when pressing inside the window. Completion still needs press-probe
  + leave/reenter arming (D-flat / n=1 auto-arms after probe — hole play can be too small).

Insert latch (documented behaviour, low weight): once an ``insert`` step latches
(``step_done`` + ``step_done_at >= 0``), deliberate pull-out does **not** clear insert
credit. Only a ``misseat`` event (which sets ``step_done_at = -1``) forces a re-seat.
This is intentional — insert weight is tiny; the ceiling lives on key/stack/rework.

v0.7.3 adaptive harden (defeats v0.7.1-trained fixed scripts without breaking physics):
- Mid-episode ``retarget`` / ``reorder`` / ``flip_twist`` events, plus post-insert key retarget.
- Touch-gated key/twist reveals; denser precedence and higher adaptive event rates.
- ``misseat`` clears key holds so credit must be re-earned from interaction.

v0.7.4 branching / probe harden (defeats adaptive conditional ONNX scripts):
- Press-probe + leave/reenter; single-active key rotation (``activate``); ``regraph``;
  post-stack key retarget.

v0.7.5 observation-completable keys (Macrocosmos v0.7.4 hold review):
- Noisy closeness + wrist detent + retarget pulse so keys are searchable from obs alone
  (cues publish only after press-probe; absolute yaw stays hidden).
- D-flat (symmetry 1) auto-arms after probe; key weight restored so missing keys cap score
  (key 6.00 / stack 1.20 / rework 1.10); horizon 870 for the 600 s / 64-ep / 2-CPU bar.
- Dead stack→key edge and unused ``_obs_key_yaw`` RNG draw removed.
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
    compound_score as _compound_score_v06,
    quat_yaw,
    step_credit,
    wrap,
)

# Heavy c1/c2 mix so privileged clears ≥0.55; c4 (5%) is the ceiling gate.
TIER_FRACTIONS_V07 = (0.40, 0.45, 0.10, 0.05)


def _assign_v07(rng: np.random.Generator, n: int) -> list[str]:
    raw = [f * n for f in TIER_FRACTIONS_V07]
    counts = [int(x) for x in raw]
    order = sorted(range(len(TIERS)), key=lambda i: raw[i] - counts[i], reverse=True)
    for k in range(n - sum(counts)):
        counts[order[k % len(TIERS)]] += 1
    # Largest-remainder alone drops c3/c4 to 0 for small strata (per=4/8).
    # Guarantee one of each hard tier when the stratum can hold them.
    if n >= 4:
        for hard_i in (2, 3):  # c3, c4
            if counts[hard_i] == 0:
                donor = max(range(hard_i), key=lambda i: counts[i])
                if counts[donor] >= 1 and sum(counts) >= hard_i + 1:
                    counts[donor] -= 1
                    counts[hard_i] = 1
    tiers = [t for t, c in zip(TIERS, counts) for _ in range(c)]
    rng.shuffle(tiers)
    return tiers
from .conditions import EpisodeConditions, sample_episodes, slot_world_xy
from .env import (
    BLOCK_H,
    MAX_PEGS,
    PAD_CENTRE_Z,
    PEG_LEN,
    SEATED_HOLD,
    SPAWN_BOARD_MARGIN,
    SPAWN_MAX_DRIFT,
    SPAWN_MAX_SPEED,
    _yaw_quat,
)

# ---- v0.7 vocabulary ----------------------------------------------------------------------------
STEP_KINDS_V07 = ("insert", "collar", "twist", "rework", "screw", "stack", "key")
# Screw/key carry most of the weight so a v0.6 insert/collar/twist-only strategy is capped.
# Rework is extractable (shallow wrong-slot seat) so both privileged and v0.6-style can finish it;
# the ceiling gap comes from press-gated screws and yaw-index keys.
# Keys / rework / stacks carry the ceiling: v0.6 kings clear insert/screw/collar/twist but
# do not index keys, extract mis-seats, or stack on peg tops. Weight those gates hard.
# Hard steps dominate so v0.6-style (insert/collar/twist only) stays ≤~0.30; honest privileged
# that finishes keys/stacks/rework clears ≥~0.50.
# Missing keys must cap scripted/leader scores ≤~0.30: key dominates episode mass again,
# while obs closeness/detent/retarget keep keys searchable (Macrocosmos observability hold).
# Stack/rework stay meaningful but secondary; insert/collar/twist/screw stay tiny.
STEP_WEIGHT_V07 = {
    "insert": 0.04,
    "collar": 0.04,
    "twist": 0.04,
    "screw": 0.10,
    "rework": 1.10,
    "stack": 1.20,
    "key": 6.00,
}
# Short budgets + hard horizon keep slow ONNX kings inside the ~600 s / 64-ep / 2-CPU bar.
# Leaders run to the horizon at ~16–19 ms/step; at 1600 that projects ~850–990 s on 2 CPU.
STEP_BUDGET_V07 = {
    "insert": 120,
    "collar": 90,
    "twist": 70,
    "rework": 180,
    "screw": 140,
    "stack": 140,
    "key": 140,
}
REWORK_EXTRA = 30
MAX_HORIZON_V07 = 870  # hard sim-step cap; no wall-clock episode cut (kings ~18–20 ms/step)
TIER_CFG_V07 = {
    # v0.7.4: probe-refined keys + activate/regraph + post-stack retarget.
    # Slack stays generous enough for honest privileged (true-state keys skip probe noise).
    # Changing these draws reshuffles the seed→episode map.
    "c1": dict(
        extra_pegs=0, collar_frac=0.05, twist_frac=0.03, screw_frac=0.05, rework_frac=0.16,
        stack_frac=0.34, key_frac=0.36, waves=(1, 1), edge_p=0.16, event_p=0.22, slack=2.10,
        # One key per episode: activate is a no-op; press-probe + retarget still bite scripts.
        reveal_p=1.00, touch_reveal_p=1.00, max_keys=1,
        post_insert_retarget_p=0.28, post_stack_retarget_p=0.22,
    ),
    "c2": dict(
        extra_pegs=0, collar_frac=0.05, twist_frac=0.03, screw_frac=0.06, rework_frac=0.18,
        stack_frac=0.36, key_frac=0.38, waves=(1, 2), edge_p=0.18, event_p=0.26, slack=1.95,
        reveal_p=1.00, touch_reveal_p=1.00, max_keys=1,
        post_insert_retarget_p=0.32, post_stack_retarget_p=0.26,
    ),
    "c3": dict(
        extra_pegs=1, collar_frac=0.04, twist_frac=0.02, screw_frac=0.06, rework_frac=0.20,
        stack_frac=0.38, key_frac=0.40, waves=(1, 2), edge_p=0.22, event_p=0.30, slack=1.80,
        reveal_p=1.00, touch_reveal_p=1.00, max_keys=1,
        post_insert_retarget_p=0.36, post_stack_retarget_p=0.30,
    ),
    "c4": dict(
        # waves at most 2 on c4.
        extra_pegs=1, collar_frac=0.04, twist_frac=0.02, screw_frac=0.05, rework_frac=0.22,
        stack_frac=0.40, key_frac=0.42, waves=(2, 2), edge_p=0.24, event_p=0.34, slack=1.70,
        reveal_p=1.00, touch_reveal_p=1.00, max_keys=1,
        post_insert_retarget_p=0.40, post_stack_retarget_p=0.34,
    ),
}
# Residuals large enough that open-loop / hard-coded trajectories miss; privileged uses true state.
NOISE_SCALE_V07 = dict(slot_bias=0.45, peg_delta=0.42, glitch=0.40, drift=0.42)
# Adaptive events dominate: retarget / regraph / activate force online replan from feedback.
EVENT_KINDS_V07 = EVENT_KINDS + (
    "swap_bias", "misseat", "retarget", "reorder", "flip_twist", "regraph", "activate",
)
EVENT_WEIGHTS_V07 = np.array(
    # friction, gain, mass, delay, rebias, shove, swap_bias, misseat,
    # retarget, reorder, flip_twist, regraph, activate
    # Press-probe + retarget/regraph/activate carry the script drop; misseat stays rare
    # so honest privileged is not unseated after a finished insert near the horizon.
    [0.05, 0.05, 0.05, 0.04, 0.06, 0.05, 0.12, 0.04, 0.18, 0.08, 0.08, 0.12, 0.08]
)
# Multi-turn press-twists; pitch drawn per step and shown in obs when revealed.
SCREW_TARGET = (0.30, 0.55)
KEY_TOL = (0.045, 0.080)  # yaw-index; privileged true-state nudge still reaches
# Press frames before key completion / key_tol pack / closeness cue.
# Leave-and-reenter arming is the main natural-seat freebie killer; probe still blocks
# pure insert-and-leave with no press. D-flat (symmetry 1) auto-arms after probe.
KEY_PROBE_FRAMES = 10
# Soft unsigned closeness: cue rises inside ~KEY_CLOSE_SOFT * tol of the window centre.
# Additive Gaussian noise (obs units in [0, 1]); absolute yaw is never published.
# Published only after press-probe so casual contact cannot free-search the window.
KEY_CLOSE_SOFT = 2.8
KEY_CLOSE_NOISE = 0.08
# Wrist yaw-torque detent while pressing inside the keyed window (N·m before TORQUE_SCALE).
# Detent also requires press-probe (same gate as the closeness channel).
KEY_DETENT_TAU = 0.22
KEY_DETENT_NOISE = 0.05
# Frames the retarget/disarm pulse stays high in task feat 2 after a window move.
KEY_RETARGET_PULSE = 20
KEY_HOLD = 4  # ungripped aligned frames
STACK_SEAT_XY = 0.0026
STACK_SEAT_Z_CLEAR = 0.0024
# Screw press: palm must be on the peg (touch). Free-twist open-grip scripts still get 0.
PRESS_PALM_MARGIN = 0.001
# Rework pegs sit deep enough that casual yanks miss; privileged extract_routine still clears.
REWORK_SEAT_DEPTH = 0.0050
# Stack rings use a slight interference fit so a deliberate peg-top place can rest in the
# mate zone (clearance-fit rings free-fall ~4 mm in <1 control frame and can never hold).
# Mate zone + hold tightened after image-eval showed revealed kings completing ~half of stacks.
# HOLD=5 blocks single-frame sweeps; XY/Z leave room for deliberate place + honest privileged.
STACK_MATE_XY = 0.0030
STACK_MATE_Z = 0.0028
STACK_MATE_HOLD = 5  # ungripped frames in zone before pin (anti-king)
# Inner radius = peg_outer - this amount (metres). Soft contacts settle the ring on the top.
STACK_INTERFERENCE = (0.00012, 0.00028)

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
        return dict(kind=kind, scale=float(rng.uniform(0.30, 2.2)))
    if kind == "gain":
        return dict(kind=kind, scale=float(rng.uniform(0.35, 1.75)))
    if kind == "mass":
        return dict(kind=kind, scale=float(rng.uniform(0.45, 2.6)))
    if kind == "delay":
        return dict(kind=kind, steps=int(rng.integers(0, 5)))
    if kind == "rebias":
        return dict(kind=kind, mag=float(rng.uniform(0.0020, 0.008)), ang=float(rng.uniform(0, 2 * math.pi)))
    if kind == "swap_bias":
        return dict(kind=kind)  # reshuffle peg-vs-slot residuals (handled in episode)
    if kind == "misseat":
        # Lift a seated peg partially out — must be re-detected from obs and re-seated.
        return dict(kind=kind, lift=float(rng.uniform(0.006, 0.014)))
    if kind == "retarget":
        # Re-sample unfinished key yaw windows (and optionally tol); obs must be re-read.
        return dict(kind=kind)
    if kind == "reorder":
        # Add a wave-consistent precedence edge among unfinished steps (defeats fixed order).
        return dict(kind=kind)
    if kind == "flip_twist":
        # Flip unfinished twist/screw direction and clear accumulated progress.
        return dict(kind=kind)
    if kind == "regraph":
        # Drop soft unfinished edges and redraw 1–3 wave-consistent ones (stronger than reorder).
        return dict(kind=kind)
    if kind == "activate":
        # Rotate which unfinished key is currently completable (single-active gate).
        return dict(kind=kind)
    return dict(kind=kind, sigma=float(rng.uniform(0.14, 0.32)))


# Hand yaw joint travels ±~3.1 rad (see env ``target[3]`` clip). Key windows must sit
# inside that travel or no action sequence can align — wrap does not help at the stop.
_KEY_YAW_REACH = 2.95


def _sample_key_yaw_for(rng: np.random.Generator, cond: EpisodeConditions, peg_i: int) -> float:
    """World-frame yaw window centre that the **hand** can actually reach.

    v0.7.5 indexes keys on hand yaw (not peg spin). Windows outside the wrist travel
    (±``_KEY_YAW_REACH``) are unreachable by actions. Non-round keys sit on a slot
    symmetry; freebies from insert-at-seat are blocked by press-probe, not by
    offsetting into an unreachable yaw.
    """
    peg = cond.pegs[peg_i]
    spec = peg.spec
    slot_yaw = float(spec.yaw + cond.board_yaw)
    if spec.shape == "round" or spec.symmetry() == 0:
        # Stay away from 0 so a peg that happens to sit near 0 is not free before probing.
        yaw = float(rng.uniform(-_KEY_YAW_REACH, _KEY_YAW_REACH))
        if abs(wrap(yaw)) < 0.35:
            yaw = float(wrap(yaw + math.copysign(0.55, yaw if yaw != 0 else 1.0)))
            yaw = float(np.clip(yaw, -_KEY_YAW_REACH, _KEY_YAW_REACH))
        return yaw
    n = max(1, int(spec.symmetry()))
    k_sym = int(rng.integers(0, n))
    jitter = float(rng.uniform(-0.03, 0.03))
    yaw = float(wrap(slot_yaw + (2.0 * math.pi * k_sym) / n + jitter))
    # Fold into the reachable wrist band (shortest-path equivalent under wrap).
    if abs(yaw) > _KEY_YAW_REACH:
        yaw = float(np.clip(yaw, -_KEY_YAW_REACH, _KEY_YAW_REACH))
    return yaw


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
        outer = cond.pegs[peg_i].spec.outer_radius()
        if kind == "stack":
            # Interference so the ring can rest on the peg top under free physics.
            r_in = max(0.0010, outer - float(rng.uniform(*STACK_INTERFERENCE)))
        else:
            r_in = outer + float(rng.uniform(0.0006, 0.0012))
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
    # Prefer upright, early-wave rounds so the privileged press-twist can actually run.
    # Per-peg Bernoulli(frac): ``round(frac * n)`` with small fracs and n≲5 is almost always 0,
    # so twist/screw never appeared on live-sized 64-ep draws.
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

    # --- key gates ---------------------------------------------------------------------------
    # Keys share the twist obs channels, so never put a key on a twist/screw peg (those would
    # overwrite key sin/cos). Prefer non-rework pegs; round pegs may use any yaw, non-round
    # pegs must index near a seated symmetry of the slot.
    twist_or_screw = set(twist_pegs) | set(screw_pegs)
    key_cands = [i for i in range(K) if i not in rework_pegs and i not in twist_or_screw]
    max_keys = int(cfg.get("max_keys", 2))
    n_key = min(len(key_cands), max_keys, max(1, int(round(cfg["key_frac"] * K))))
    key_pegs = set(rng.choice(key_cands, size=n_key, replace=False).tolist() if n_key and key_cands else [])

    # Guarantee every episode has a key gate — the main anti-king lever (obs-only yaw index).
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

    # Pair stacks onto key pegs so stack-before-key can zero key credit when stacks fail.
    # v0.7.4: 55% pairing (was 70%) — leaves some keys stack-free so honest privileged can
    # clear key weight without every key waiting on a peg-top mate.
    for peg_i in list(key_pegs):
        if any(p == peg_i and k == "stack" for p, k in ring_pegs):
            continue
        # Keep stack-before-key uncommon (~25%): stacks still score on their own weight,
        # but gating every key behind a stack left honest privileged stuck under ~0.40.
        if rng.random() < 0.25:
            _add_stack_on(int(peg_i))
    # Guarantee at least one stack (preferred) or rework so insert/key-only kings stay capped.
    # Prefer non-key pegs so the free stack does not force stack-before-key on every episode.
    if not any(k == "stack" for _, k in ring_pegs):
        non_key = [i for i in cands if i not in key_pegs and i not in {p for p, _ in ring_pegs}]
        prefer = non_key or [i for i in cands if i not in {p for p, _ in ring_pegs}] or list(cands)
        placed = False
        for peg_i in prefer:
            if _add_stack_on(int(peg_i)):
                placed = True
                break
        if not placed and not rework_pegs and free_slots and rework_cands:
            rework_pegs[int(rework_cands[0])] = int(free_slots[0])
    # Guarantee every episode is compound (collar/twist fallback if somehow still bare).
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
    touch_p = float(cfg.get("touch_reveal_p", 0.0))
    for i in range(K):
        if i in rework_pegs:
            steps.append(TaskStep("rework", i, i, wrong_slot=rework_pegs[i]))
        else:
            steps.append(TaskStep("insert", i, i))
        insert_idx[i] = len(steps) - 1
        if i in key_pegs:
            # v0.7.4: keys always touch-gated when touch_reveal_p≈1 so probe refine can start.
            steps.append(
                TaskStep(
                    "key",
                    i,
                    i,
                    key_yaw=_sample_key_yaw_for(rng, cond, i),
                    key_tol=float(rng.uniform(*KEY_TOL)),
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
        # Optional: stack/collar after twist/screw on same peg.
        tw = [k for k, s in enumerate(steps) if s.kind in ("twist", "screw") and s.peg == col.peg]
        if tw and rng.random() < 0.55:
            tw_i = tw[0]
            if col.wave > peg_wave[col.peg]:
                edges.add((tw_i, sc))
            else:
                edges.add((tw_i, sc) if rng.random() < 0.5 else (sc, tw_i))
        # v0.7.5: do not add stack→key edges here. Keys keep only insert/rework prereqs
        # (see strip below); a temporary stack→key edge was dead code in v0.7.4.

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
    # v0.7.4: keys only wait on their own peg's insert/rework. Stack/random edges still
    # shape the rest of the graph (and obs blocked flags via activate), but they must not
    # zero key completion for honest privileged when a peg-top mate fails.
    for s in steps:
        if s.kind != "key":
            continue
        keep = [
            p for p in s.prereq
            if steps[p].kind in ("insert", "rework") and steps[p].peg == s.peg
        ]
        if not keep and s.peg in insert_idx:
            keep = [int(insert_idx[s.peg])]
        s.prereq = sorted(set(keep))

    # Late reveals: hide twist/screw/key params until a random earlier prerequisite (or insert) is done.
    # Keys almost always late-reveal (reveal_p ≥ 0.9 in v0.7.3) so early open-loop yaw plans fail.
    for k, s in enumerate(steps):
        if s.kind not in ("twist", "screw", "key"):
            continue
        if rng.random() >= cfg["reveal_p"]:
            continue
        gates = list(s.prereq) if s.prereq else ([insert_idx[s.peg]] if s.peg in insert_idx else [])
        if gates:
            s.reveal_after = int(gates[int(rng.integers(0, len(gates)))])
        elif s.kind == "key" and s.peg in insert_idx:
            # Guarantee at least insert-gated reveal for keys even if prereq list was empty.
            s.reveal_after = int(insert_idx[s.peg])

    # Horizon.
    budget = 0
    for s in steps:
        budget += STEP_BUDGET_V07[s.kind]
        if s.kind in ("insert", "rework") and cond.pegs[s.peg].start_type == "lying":
            budget += LYING_EXTRA
        if s.kind == "rework":
            budget += REWORK_EXTRA
    max_steps = int(cfg["slack"] * budget) + WAVE_EXTRA * max(0, W - 1)
    cond.max_steps = min(MAX_HORIZON_V07, max_steps)
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
        tiers = _assign_v07(rng, len(conds[s : s + per]))
        for c, t in zip(conds[s : s + per], tiers):
            c.compound = make_compound_v07(rng, c, t)
    return ranges, conds


# ---- episode ------------------------------------------------------------------------------------
def _is_hard_prereq(sa: TaskStep, sb: TaskStep) -> bool:
    """Mandatory same-peg edges that ``regraph`` must not drop."""
    if sa.peg != sb.peg:
        return False
    if sa.kind in ("insert", "rework") and sb.kind in ("key", "stack", "collar", "twist", "screw"):
        return True
    if sa.kind == "stack" and sb.kind == "key":
        return True
    return False


class AssemblyIIIEpisode(CompoundEpisode):
    """v0.7 episode: rework spawns, screw press gate, stack seat, key yaw, late reveals."""

    def __init__(self, cond: EpisodeConditions, rng: np.random.Generator):
        self._rework_wrong = {
            s.peg: s.wrong_slot for s in cond.compound.steps if s.kind == "rework" and s.wrong_slot is not None
        }
        self._extracted = {i: False for i in self._rework_wrong}
        self._key_hold = [0] * cond.n_pegs
        self._stack_mated = [False] * len(cond.compound.collars)
        self._stack_mate_run = [0] * len(cond.compound.collars)
        self._stack_mate_arm = [0] * len(cond.compound.collars)
        self._screw_hand_prev: dict[int, float | None] = {}
        # Interaction-gated reveals: True once the gripper has touched the peg this episode.
        self._peg_touched = [False] * cond.n_pegs
        # v0.7.5: press-probe + single-active key rotation + retarget pulse.
        self._peg_probe_frames = [0] * cond.n_pegs
        # Armed only after the peg leaves the key window once post-probe (anti natural-seat freebie).
        # D-flat / symmetry-1 auto-arms after probe (hole play can be < tol).
        self._key_index_armed = [False] * cond.n_pegs
        self._key_retarget_pulse = [0] * cond.n_pegs
        self._active_key: int | None = None
        super().__init__(cond, rng)
        self._init_key_probe_state()
        self._pick_active_key()

    def _seat_in_wrong_slot(self, i: int) -> None:
        """Place peg shallowly over the wrong hole (extractable; full-depth jams)."""
        ws = self._rework_wrong[i]
        spec = self.c.slots[ws]
        hx, hy = slot_world_xy(self.c.board_xy, self.c.board_yaw, spec)
        yaw = spec.yaw + self.c.board_yaw
        # Bottom just under the block top — looks "wrong-slotted" but a yank clears it.
        z = (BLOCK_H - REWORK_SEAT_DEPTH) + PEG_LEN / 2
        self._set_free(
            self.peg_qadr[i], self.peg_vadr[i],
            [hx, hy, z],
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
        self._stack_mated = [False] * self.n_collars
        self._stack_mate_run = [0] * self.n_collars
        self._stack_mate_arm = [0] * self.n_collars
        self._screw_hand_prev = {}
        self._key_hold = [0] * self.n_pegs
        self._peg_touched = [False] * self.n_pegs
        self._active_key = None
        self._init_key_probe_state()
        # step_done exists only after CompoundEpisode finishes constructing.
        if hasattr(self, "step_done"):
            self._pick_active_key()

    def _init_key_probe_state(self) -> None:
        n = int(getattr(self, "n_pegs", 0) or len(self._peg_touched))
        self._peg_probe_frames = [0] * n
        self._key_index_armed = [False] * n
        self._key_retarget_pulse = [0] * n

    def _unfinished_keys(self) -> list[int]:
        if not hasattr(self, "step_done"):
            return []
        return [k for k, s in enumerate(self.cs.steps) if s.kind == "key" and not self.step_done[k]]

    def _pick_active_key(self, prefer: int | None = None, *, rotate: bool = False) -> None:
        """Single-active key gate: only one unfinished key may complete at a time."""
        unfinished = self._unfinished_keys()
        if not unfinished:
            self._active_key = None
            return
        # Prefer keys whose prerequisites are already done (actionable now).
        ready = [
            k for k in unfinished
            if all(self.step_done[p] for p in self.cs.steps[k].prereq)
        ]
        pool = ready or unfinished
        if prefer is not None and prefer in pool:
            self._active_key = int(prefer)
            return
        if (not rotate) and self._active_key in pool:
            # Keep current if still valid unless caller forces a rotation.
            return
        # Prefer a different unfinished key when rotating.
        cands = [k for k in pool if k != self._active_key] or pool
        self._active_key = int(cands[int(self.rng.integers(0, len(cands)))])

    def _key_is_active(self, k: int) -> bool:
        unfinished = self._unfinished_keys()
        if len(unfinished) <= 1:
            return True
        if self._active_key is None:
            return True
        return k == self._active_key

    def _key_symmetry(self, peg_i: int) -> int:
        """Rotational symmetry of the peg's slot family (0 = round / free yaw)."""
        return int(self.c.pegs[peg_i].spec.symmetry())

    def _key_auto_arm(self, peg_i: int) -> bool:
        """True when leave/reenter is exempt (D-flat / n=1: hole play can be < tol)."""
        n = self._key_symmetry(peg_i)
        return n == 1

    def _key_closeness(self, s: TaskStep) -> float:
        """Unsigned soft closeness in [0, 1] (1 = hand yaw centred in window). No absolute yaw."""
        if s.key_tol <= 0:
            return 1.0
        err = abs(wrap(self._key_hand_yaw() - s.key_yaw))
        soft = max(float(s.key_tol) * KEY_CLOSE_SOFT, 1e-3)
        return float(np.clip(1.0 - err / soft, 0.0, 1.0))

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
        """True when the closed gripper is pressing down on a seated peg.

        Requires contact *and* palm at/below the peg top (PRESS_PALM_MARGIN). Open-grip
        free-twists and high-wrist spins do not accumulate screw progress.
        """
        if not self._gripper_touching(self.peg_body[i]):
            return False
        p, _ = self.peg_pose(i)
        peg_top = float(p[2] + PEG_LEN / 2)
        palm_z = float(self.data.qpos[2] + PAD_CENTRE_Z)
        return palm_z <= peg_top - PRESS_PALM_MARGIN + 1e-4

    def _stack_ring_upright(self, q) -> bool:
        """Looser than seat upright — soft contacts tip interference rings ~10–20°."""
        w, x, y, z = q
        return abs(1.0 - 2.0 * (x * x + y * y)) >= math.cos(math.radians(28.0))

    def _stack_in_mate_zone(self, j: int) -> bool:
        """Geometric peg-top place zone (gripper may still be holding)."""
        if self.collar_parked[j]:
            return False
        col = self.cs.collars[j]
        if not col.is_stack:
            return False
        if not (self.seated_now[col.peg] or self._peg_in_hole(col.peg)):
            return False
        p, q = self.collar_pose(j)
        hx, hy = self.slot_xy[col.peg]
        peg_p, _ = self.peg_pose(col.peg)
        peg_top = peg_p[2] + PEG_LEN / 2
        bottom = p[2] - COLLAR_H / 2
        return (
            float(np.hypot(p[0] - hx, p[1] - hy)) <= STACK_MATE_XY
            and abs(bottom - peg_top) <= STACK_MATE_Z
            and bottom > BLOCK_H + 0.015
            and self._stack_ring_upright(q)
        )

    def _stack_release_clear(self, j: int) -> bool:
        """True when the gripper has released the ring for the mate hold to count.

        Fully clear contact is ideal. Soft pad brushes often linger after the jaws open;
        an open gripper (``qpos[4]`` near max) means the radial grasp is gone, so the hold
        may count while the hand rises clear. Closed-grip contact still blocks (anti-king).
        """
        if not self._gripper_touching(self.collar_body[j]):
            return True
        # MAX_OPEN is 0.012; treat ≥8 mm opening as released.
        if float(self.data.qpos[4]) >= 0.008:
            return True
        p, _ = self.collar_pose(j)
        palm_z = float(self.data.qpos[2] + PAD_CENTRE_Z)
        return palm_z >= float(p[2] + COLLAR_H / 2 + 0.010)

    def _pin_stack_on_peg(self, j: int) -> None:
        """Mate a stack ring to the peg top (latches the interference-fit place)."""
        col = self.cs.collars[j]
        hx, hy = self.slot_xy[col.peg]
        peg_p, _ = self.peg_pose(col.peg)
        peg_top = peg_p[2] + PEG_LEN / 2
        z = peg_top + COLLAR_H / 2
        self._set_free(self.collar_qadr[j], self.collar_vadr[j], [hx, hy, z], _yaw_quat(0.0))
        self.data.qvel[self.collar_vadr[j] : self.collar_vadr[j] + 6] = 0.0
        self._stack_mated[j] = True
        mujoco.mj_forward(self.model, self.data)

    def _settle_stack_ring(self, j: int) -> None:
        """Hold an armed ring on the peg-top seat while the release hold counts.

        Pose is the peg-top mate pose (same as the final pin) but ``_stack_mated`` stays
        false until ``STACK_MATE_HOLD`` released frames elapse — anti-king sweeps never
        arm, so they never reach this settle.
        """
        col = self.cs.collars[j]
        hx, hy = self.slot_xy[col.peg]
        peg_p, _ = self.peg_pose(col.peg)
        peg_top = float(peg_p[2] + PEG_LEN / 2)
        z = peg_top + COLLAR_H / 2
        self._set_free(self.collar_qadr[j], self.collar_vadr[j], [hx, hy, z], _yaw_quat(0.0))
        self.data.qvel[self.collar_vadr[j] : self.collar_vadr[j] + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _maybe_mate_stacks(self) -> None:
        """Auto-mate after a deliberate place-and-release in the mate zone (anti-king).

        Flow:
        1. **Arm** while the ring is in the mate zone under a closed grasp (≥2 frames).
           Open-loop sweeps never arm.
        2. **Count** ungripped / open-jaw frames still in zone; settle pose gently so
           soft contacts do not eject the ring before ``STACK_MATE_HOLD``.
        3. **Pin** once the hold completes.

        Same latch for miners and honest privileged (``oracle=False``). Oracle snaps may
        still call ``_pin_stack_on_peg`` directly.

        Note: a privileged v0.6-style block-seat collar on an interference stack ring can
        still snag the peg tip and arm — that path uses true poses miners lack. Obs-only /
        ONNX kings stay capped by key/rework weights and noise.
        """
        if not hasattr(self, "_stack_mate_run"):
            self._stack_mate_run = [0] * self.n_collars
        if not hasattr(self, "_stack_mate_arm"):
            self._stack_mate_arm = [0] * self.n_collars
        for j, col in enumerate(self.cs.collars):
            if not col.is_stack or self.collar_parked[j] or self._stack_mated[j]:
                self._stack_mate_run[j] = 0
                self._stack_mate_arm[j] = 0
                continue
            in_zone = self._stack_in_mate_zone(j)
            released = self._stack_release_clear(j)
            if in_zone and not released:
                # Closed grasp in zone — arm the latch (deliberate place).
                self._stack_mate_arm[j] = min(self._stack_mate_arm[j] + 1, 8)
                self._stack_mate_run[j] = 0
            elif released and (in_zone or self._stack_mate_arm[j] >= 2):
                # Two completable paths (both need sustained released frames):
                # 1) Unarmed: interference-fit ring rests in zone — keep upright (soft
                #    contacts otherwise tip the ring out of `_upright` within 1–2 frames).
                # 2) Armed: settle back to the peg-top seat after a place-and-release.
                if self._stack_mate_arm[j] >= 2:
                    self._settle_stack_ring(j)
                else:
                    p, _ = self.collar_pose(j)
                    self._set_free(
                        self.collar_qadr[j],
                        self.collar_vadr[j],
                        [float(p[0]), float(p[1]), float(p[2])],
                        _yaw_quat(0.0),
                    )
                    self.data.qvel[self.collar_vadr[j] : self.collar_vadr[j] + 6] = 0.0
                    mujoco.mj_forward(self.model, self.data)
                self._stack_mate_run[j] += 1
            else:
                if not in_zone:
                    self._stack_mate_arm[j] = 0
                self._stack_mate_run[j] = 0
            if self._stack_mate_run[j] >= STACK_MATE_HOLD:
                self._pin_stack_on_peg(j)

    def _stack_seated_now(self, j: int) -> bool:
        if self.collar_parked[j]:
            return False
        col = self.cs.collars[j]
        if not col.is_stack:
            return False
        if self._stack_mated[j]:
            return bool(self.seated_now[col.peg] or self._peg_in_hole(col.peg)) and (
                not self._gripper_touching(self.collar_body[j])
            )
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

    def _key_hand_yaw(self) -> float:
        """Wrist yaw used for the key index (v0.7.5).

        Parallel-jaw soft contacts rarely spin a seated peg in-hole, so the index is the
        **hand** yaw while interacting — same family as press-gated screws. Peg yaw stays
        a distractor; miners read closeness of the wrist to the hidden window.
        """
        return float(self.data.qpos[3])

    def _key_aligned(self, s: TaskStep) -> bool:
        if s.key_tol <= 0:
            return True
        return abs(wrap(self._key_hand_yaw() - s.key_yaw)) <= s.key_tol

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

    def _retarget_key(self, k: int, *, reason: str = "retarget") -> dict | None:
        """Re-sample an unfinished key window. Returns an events_fired payload or None."""
        s = self.cs.steps[k]
        if s.kind != "key" or self.step_done[k]:
            return None
        old = float(s.key_yaw)
        sym = int(self.c.pegs[s.peg].spec.symmetry())
        # Keep resampling until the window moves outside the old tol (real adaptation).
        # Symmetry-1 (D-flat): physical yaw cannot move far — redraw tol + pulse instead.
        new = old
        if sym != 1:
            for _ in range(8):
                cand = _sample_key_yaw_for(self.rng, self.c, s.peg)
                if abs(wrap(cand - old)) > max(0.12, 2.5 * s.key_tol):
                    new = cand
                    break
                new = cand
        s.key_yaw = new
        # Usually redraw tol; always clear hold so alignment must be re-proven.
        # Keep probe frames — retarget changes the yaw window, not the fact that the peg
        # was already press-probed. Pulse the retarget flag so miners can observe the move.
        if self.rng.random() < 0.70 or sym == 1:
            s.key_tol = float(self.rng.uniform(*KEY_TOL))
        self._key_hold[s.peg] = 0
        # New window: must leave-and-reenter again (natural seat on the old window is useless).
        # D-flat auto-arm still reapplies once probed (see _step_state).
        self._key_index_armed[s.peg] = False
        self._key_retarget_pulse[s.peg] = int(KEY_RETARGET_PULSE)
        return dict(kind=reason, step=self.steps, peg=int(s.peg), key_step=int(k), old_yaw=old, new_yaw=float(s.key_yaw))

    def _maybe_post_insert_retarget(self, seat_k: int) -> None:
        """After an insert/rework completes, often retarget that peg's unfinished key."""
        s = self.cs.steps[seat_k]
        if s.kind not in ("insert", "rework"):
            return
        cfg = TIER_CFG_V07.get(self.cs.tier, {})
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
        """After a stack completes, often retarget that peg's unfinished key (v0.7.4)."""
        s = self.cs.steps[stack_k]
        if s.kind != "stack":
            return
        cfg = TIER_CFG_V07.get(self.cs.tier, {})
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

    def _maybe_activate_unblocked_key(self, done_k: int) -> None:
        """When a completion unblocks a key, make that key active (avoid stranded ready keys)."""
        for k, s in enumerate(self.cs.steps):
            if s.kind != "key" or self.step_done[k]:
                continue
            if done_k not in s.prereq:
                continue
            if all(self.step_done[p] for p in s.prereq):
                self._pick_active_key(prefer=k)
                return

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
        if ev["kind"] == "misseat":
            # Partial unseat of a currently seated peg — insert/key/screw credit can drop.
            seated = [i for i in range(self.n_pegs) if self.seated_now[i] and not self.peg_parked[i]]
            if seated:
                i = int(seated[int(self.rng.integers(0, len(seated)))])
                p, q = self.peg_pose(i)
                lift = float(ev.get("lift", 0.010))
                self._set_free(
                    self.peg_qadr[i], self.peg_vadr[i],
                    [p[0], p[1], p[2] + lift], q,
                )
                self.seated_now[i] = False
                self.seated_run[i] = 0
                self._key_hold[i] = 0
                # Drop latch on insert/rework for this peg so credit must be re-earned.
                for k, s in enumerate(self.cs.steps):
                    if s.kind in ("insert", "rework") and s.peg == i:
                        self.step_done[k] = False
                        self.step_done_at[k] = -1
                    if s.kind == "key" and s.peg == i:
                        # Key must be re-indexed after a misseat; leave step_done if already done.
                        if not self.step_done[k]:
                            self.step_progress[k] = 0.0
                mujoco.mj_forward(self.model, self.data)
            self.events_fired.append(dict(ev, step=self.steps))
            return
        if ev["kind"] == "retarget":
            unfinished = [k for k, s in enumerate(self.cs.steps) if s.kind == "key" and not self.step_done[k]]
            if unfinished:
                k = int(unfinished[int(self.rng.integers(0, len(unfinished)))])
                payload = self._retarget_key(k, reason="retarget")
                if payload is not None:
                    self.events_fired.append(payload)
                    return
            self.events_fired.append(dict(ev, step=self.steps, skipped=True))
            return
        if ev["kind"] == "flip_twist":
            unfinished = [
                k for k, s in enumerate(self.cs.steps)
                if s.kind in ("twist", "screw") and not self.step_done[k]
            ]
            if unfinished:
                k = int(unfinished[int(self.rng.integers(0, len(unfinished)))])
                s = self.cs.steps[k]
                s.twist_dir = -float(s.twist_dir) if s.twist_dir else -1.0
                self.twist_acc[s.peg] = 0.0
                self._screw_hand_prev[s.peg] = None
                self._twist_prev[s.peg] = None
                self.events_fired.append(
                    dict(ev, step=self.steps, peg=int(s.peg), twist_step=int(k), twist_dir=float(s.twist_dir))
                )
                return
            self.events_fired.append(dict(ev, step=self.steps, skipped=True))
            return
        if ev["kind"] == "reorder":
            # Add one wave-consistent precedence edge among unfinished steps (no cycles).
            unfinished = [k for k, s in enumerate(self.cs.steps) if not self.step_done[k]]
            if len(unfinished) >= 2:
                # Kahn reachability from current prereqs among unfinished nodes.
                prereq = {k: [p for p in self.cs.steps[k].prereq if p in unfinished] for k in unfinished}
                indeg = {k: len(prereq[k]) for k in unfinished}
                ready = [k for k in unfinished if indeg[k] == 0]
                order: list[int] = []
                while ready:
                    u = ready.pop()
                    order.append(u)
                    for v in unfinished:
                        if u in prereq[v]:
                            indeg[v] -= 1
                            if indeg[v] == 0:
                                ready.append(v)
                if len(order) == len(unfinished):
                    pos = {k: i for i, k in enumerate(order)}
                    cands: list[tuple[int, int]] = []
                    for a in unfinished:
                        for b in unfinished:
                            if a == b or pos[a] >= pos[b]:
                                continue
                            if a in self.cs.steps[b].prereq:
                                continue
                            wa = _step_wave_of(self.cs.steps[a], self.cs.peg_wave, self.cs.collars)
                            wb = _step_wave_of(self.cs.steps[b], self.cs.peg_wave, self.cs.collars)
                            if wa <= wb:
                                cands.append((a, b))
                    if cands:
                        a, b = cands[int(self.rng.integers(0, len(cands)))]
                        self.cs.steps[b].prereq.append(a)
                        self.cs.steps[b].prereq.sort()
                        self.events_fired.append(dict(ev, step=self.steps, edge=[int(a), int(b)]))
                        return
            self.events_fired.append(dict(ev, step=self.steps, skipped=True))
            return
        if ev["kind"] == "regraph":
            # Stronger than reorder: drop soft unfinished edges, then add 1–3 new ones.
            unfinished = [k for k, s in enumerate(self.cs.steps) if not self.step_done[k]]
            if len(unfinished) < 2:
                self.events_fired.append(dict(ev, step=self.steps, skipped=True))
                return
            dropped = 0
            for b in unfinished:
                keep: list[int] = []
                for a in self.cs.steps[b].prereq:
                    if a not in unfinished:
                        keep.append(a)
                        continue
                    if _is_hard_prereq(self.cs.steps[a], self.cs.steps[b]):
                        keep.append(a)
                        continue
                    # Soft edge among unfinished: drop with high probability.
                    if self.rng.random() < 0.55:
                        dropped += 1
                        continue
                    keep.append(a)
                self.cs.steps[b].prereq = sorted(set(keep))
            # Rebuild a topological order under remaining edges.
            prereq = {k: [p for p in self.cs.steps[k].prereq if p in unfinished] for k in unfinished}
            indeg = {k: len(prereq[k]) for k in unfinished}
            ready = [k for k in unfinished if indeg[k] == 0]
            order: list[int] = []
            while ready:
                u = ready.pop()
                order.append(u)
                for v in unfinished:
                    if u in prereq[v]:
                        indeg[v] -= 1
                        if indeg[v] == 0:
                            ready.append(v)
            added: list[list[int]] = []
            if len(order) == len(unfinished):
                pos = {k: i for i, k in enumerate(order)}
                cands: list[tuple[int, int]] = []
                for a in unfinished:
                    for b in unfinished:
                        if a == b or pos[a] >= pos[b]:
                            continue
                        if a in self.cs.steps[b].prereq:
                            continue
                        wa = _step_wave_of(self.cs.steps[a], self.cs.peg_wave, self.cs.collars)
                        wb = _step_wave_of(self.cs.steps[b], self.cs.peg_wave, self.cs.collars)
                        if wa <= wb:
                            cands.append((a, b))
                n_add = min(len(cands), int(self.rng.integers(1, 4)))
                self.rng.shuffle(cands)
                for a, b in cands[:n_add]:
                    self.cs.steps[b].prereq.append(a)
                    self.cs.steps[b].prereq.sort()
                    added.append([int(a), int(b)])
            if dropped or added:
                self.events_fired.append(
                    dict(ev, step=self.steps, dropped=int(dropped), edges=added)
                )
                # Priority rotation often accompanies a graph rewrite.
                if len(self._unfinished_keys()) >= 2:
                    self._pick_active_key(rotate=True)
                return
            self.events_fired.append(dict(ev, step=self.steps, skipped=True))
            return
        if ev["kind"] == "activate":
            unfinished = self._unfinished_keys()
            if len(unfinished) < 2:
                self.events_fired.append(dict(ev, step=self.steps, skipped=True))
                return
            old = self._active_key
            # Force a rotation away from the current active key when possible.
            prefer = None
            others = [k for k in unfinished if k != old]
            if others:
                prefer = int(others[int(self.rng.integers(0, len(others)))])
            self._pick_active_key(prefer=prefer, rotate=True)
            for i in range(self.n_pegs):
                self._key_hold[i] = 0
            self.events_fired.append(
                dict(ev, step=self.steps, old=old, active=self._active_key)
            )
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
            # 80% of target is enough: short press-twists often undershoot by a few degrees.
            done = frac >= 0.80 and (self._peg_in_hole(i) or self.seated_now[i])
            return done, PARTIAL_CAP * frac
        if s.kind == "key":
            i = s.part
            if not (self.seated_now[i] or self._peg_in_hole(i)):
                return False, 0.0
            aligned = self._key_aligned(s)
            # v0.7.3: do not accumulate hold (or complete) until prerequisites are done.
            # Early yaw alignment during stack work must not latch an out-of-order key —
            # stack-before-key would otherwise zero honest credit after a later stack.
            # v0.7.4: inactive keys cannot hold; press-probe must complete before hold;
            # leave-and-reenter arming defeats natural-seat freebies after an accidental press.
            # v0.7.5: D-flat / n=1 auto-arms after probe (rotational play can be < tol).
            probed = self._peg_probe_frames[i] >= KEY_PROBE_FRAMES
            if probed and self._key_auto_arm(i):
                self._key_index_armed[i] = True
            elif probed and not aligned:
                self._key_index_armed[i] = True
            armed = bool(self._key_index_armed[i])
            prereq_ok = (
                all(self.step_done[p] for p in s.prereq)
                and self._key_is_active(k)
                and probed
                and armed
            )
            if prereq_ok and aligned and not self._gripper_touching(self.peg_body[i]):
                self._key_hold[i] = self._key_hold[i] + 1
            else:
                self._key_hold[i] = 0
            done = bool(prereq_ok) and (
                self._key_hold[i] >= KEY_HOLD
                or (
                    self.step_done[k]
                    and aligned
                    and self._peg_in_hole(i)
                    and self._key_is_active(k)
                    and probed
                    and armed
                )
            )
            err = abs(wrap(self._key_hand_yaw() - s.key_yaw))
            frac = float(np.clip(1.0 - err / max(s.key_tol, 1e-3), 0.0, 1.0))
            # Partial only once prereqs are met (no farming while blocked).
            return done, (PARTIAL_CAP * frac if prereq_ok else 0.0)
        if s.kind == "stack":
            j = s.part
            if self.collar_parked[j]:
                return False, 0.0
            # Latch mate is the completion gate (arm + STACK_MATE_HOLD released frames).
            # Do not also require SEATED_HOLD via collar_run — that double-hold burned the
            # last horizon frames after pin (honest seed7 saw ~16/78 mated-but-not-done).
            done = bool(self._stack_mated[j]) or self.collar_run[j] >= SEATED_HOLD or (
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
        # Parent skips bookkeeping for screw/key/stack/rework so this is the sole completion path
        # (avoids done_at flicker and double key-hold increments).
        twist_before = list(self.twist_acc)
        screw_pegs = {s.peg for s in self.cs.steps if s.kind == "screw"}
        for i in screw_pegs:
            self._twist_prev[i] = None
        super().step(action)
        if self.terminal_reason in ("unstable",) or (
            self.done and self.terminal_reason not in ("assembled", "time_limit")
        ):
            return
        # Insert latch: once insert has latched (step_done_at >= 0), keep credit even if
        # the peg is later pulled out. Misseat clears done_at to -1 and forces a re-seat.
        # Insert weight is tiny; this is deliberate (documented in module docstring).
        for k, s in enumerate(self.cs.steps):
            if s.kind != "insert" or self.step_done[k]:
                continue
            at = self.step_done_at[k]
            if at is not None and int(at) >= 0:
                self.step_done[k] = True
        # Key detent on wrist yaw-torque while pressing inside an active key window.
        # Physically grounded cue: F/T carries information only near the software check.
        for k, s in enumerate(self.cs.steps):
            if s.kind != "key" or self.step_done[k]:
                continue
            if not self._key_is_active(k):
                continue
            if not (self.seated_now[s.peg] or self._peg_in_hole(s.peg)):
                continue
            if not self._pressing_on_peg(s.peg):
                continue
            if self._peg_probe_frames[s.peg] < KEY_PROBE_FRAMES:
                continue
            if not self._key_aligned(s):
                continue
            tau = KEY_DETENT_TAU + float(self.rng.normal(0.0, KEY_DETENT_NOISE))
            self._wrist_torque[2] = float(self._wrist_torque[2]) + tau
        # Decay retarget/disarm pulses (observable in task feat 2).
        for i in range(len(self._key_retarget_pulse)):
            if self._key_retarget_pulse[i] > 0:
                self._key_retarget_pulse[i] -= 1
        # Screw progress: hand yaw while pressing (MuJoCo soft contacts rarely spin the peg).
        # Open-grip free-twists still accumulate 0. Peg yaw tracking is kept for obs continuity.
        hand_yaw = float(self.data.qpos[3])
        for i in screw_pegs:
            seat_done = any(
                self.step_done[sk]
                for sk, ss in enumerate(self.cs.steps)
                if ss.kind in ("insert", "rework") and ss.peg == i
            )
            prev_h = self._screw_hand_prev.get(i)
            if (
                seat_done
                and prev_h is not None
                and self._peg_in_hole(i)
                and self._pressing_on_peg(i)
            ):
                self.twist_acc[i] = twist_before[i] + wrap(hand_yaw - prev_h)
            else:
                self.twist_acc[i] = twist_before[i] if seat_done else 0.0
            self._screw_hand_prev[i] = hand_yaw
            self._twist_prev[i] = quat_yaw(self.peg_pose(i)[1])
        self._maybe_mate_stacks()
        for j, col in enumerate(self.cs.collars):
            if col.is_stack and not self.collar_parked[j]:
                # Re-pin mated stacks each step so they stay on the peg top.
                if self._stack_mated[j]:
                    self._pin_stack_on_peg(j)
                self.collar_run[j] = self.collar_run[j] + 1 if self._stack_seated_now(j) else 0
        # Interaction tracking for touch-gated reveals (any contact with the peg body).
        # v0.7.4: key-yaw probe refine advances only while pressing on the peg (screw-like
        # press gate). Mere touch reveals the channel but leaves a large decaying bias —
        # touch-and-snap scripts keep a wrong absolute yaw.
        for i in range(self.n_pegs):
            if self.peg_parked[i]:
                continue
            if self._gripper_touching(self.peg_body[i]):
                self._peg_touched[i] = True
                if (
                    self._peg_probe_frames[i] < KEY_PROBE_FRAMES
                    and self._pressing_on_peg(i)
                ):
                    self._peg_probe_frames[i] += 1
        # Sole bookkeeping path for v0.7 hard steps (parent skipped them).
        newly_done: list[int] = []
        for k, s in enumerate(self.cs.steps):
            if s.kind not in ("screw", "stack", "key", "rework"):
                continue
            done, prog = self._step_state(k)
            self.step_progress[k] = max(self.step_progress[k], prog)
            if done and not self.step_done[k]:
                self.step_done[k] = True
                self.step_done_at[k] = self.steps
                self.step_in_order[k] = all(self.step_done[p] for p in s.prereq)
                newly_done.append(k)
                ev = self.cs.events[self.n_completions] if self.n_completions < len(self.cs.events) else None
                self.n_completions += 1
                if ev is not None:
                    self._fire_event(ev)
                if s.kind == "key":
                    # Rotate active key onto the next unfinished key after a completion.
                    self._pick_active_key()
            elif not done and self.step_done[k]:
                # Latch: brief flickers must not wipe completion or redo time bonus.
                continue
        # Parent CompoundEpisode also marks insert/collar/twist; catch those for retarget + idle.
        for k, s in enumerate(self.cs.steps):
            if self.step_done[k] and k not in newly_done and self.step_done_at[k] == self.steps:
                newly_done.append(k)
        for k in newly_done:
            self._maybe_activate_unblocked_key(k)
            self._maybe_post_insert_retarget(k)
            self._maybe_post_stack_retarget(k)
        if all(self.step_done) and all(self.wave_arrived) and not self.done:
            self.done = True
            self.success = True
            self.success_step = self.steps
            self.terminal_reason = "assembled"

    def _params_revealed(self, s: TaskStep) -> bool:
        if s.reveal_after is not None and not self.step_done[s.reveal_after]:
            return False
        if getattr(s, "touch_reveal", False) and not self._peg_touched[s.peg]:
            return False
        return True

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
                # Keys are never co-assigned with twist/screw, so these channels stay free.
                # v0.7.5: no absolute key yaw. After reveal + press-probe publish:
                #   feat 2 = retarget/disarm pulse (1 while pulse > 0)
                #   feat 3 = coarse noisy unsigned closeness (soft window; hill-climbable)
                #   feat 4 = blocked (incl. inactive under single-active gate)
                #   feat 5 = key_tol*10 when the collar-flag slot is free
                active = self._key_is_active(k)
                blocked = float(self.step_blocked(k) or not active)
                probed = self._peg_probe_frames[s.peg] >= KEY_PROBE_FRAMES
                touched = bool(self._peg_touched[s.peg])
                o[b + 2] = 0.0
                o[b + 3] = 0.0
                o[b + 4] = blocked
                # Closeness / retarget pulse only after reveal+active+touch+press-probe.
                # Pre-probe contact must not leak a searchable cue (anti-king free-search).
                if revealed and active and touched and probed:
                    pulse = float(self._key_retarget_pulse[s.peg] > 0)
                    o[b + 2] = pulse
                    close = self._key_closeness(s)
                    noisy = close + float(noise_rng.normal(0.0, KEY_CLOSE_NOISE))
                    o[b + 3] = float(np.clip(noisy, 0.0, 1.0))
                    if o[b + 5] == 0.0:
                        o[b + 5] = float(s.key_tol) * 10.0
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
    """Weighted per-step score using v0.7 weights when new kinds are present.

    Stack / rework stay all-or-nothing for scoring (near-miss partials do not count).
    Keys may earn unfinished progress only after press-probe + arming (see ``_step_state``),
    so open-loop kings that never probe still get 0 while honest near-misses keep a gradient.

    v0.7.3: keys finished *out of order* score 0 (not the v0.6 out-of-order floor).
    Stack-before-key precedence then zeros key credit unless the stack is done first.
    Stack/rework keep the normal out-of-order floor so honest privileged stays ≥~0.45.

    v0.7.5: OOO key rule unchanged; closeness/detent/retarget affect search; key partials
    require probe+arm (progress is already 0 before that in ``_step_state``).
    """
    kinds = {s.kind for s in ep.cs.steps}
    if kinds <= {"insert", "collar", "twist"}:
        return _compound_score_v06(ep)
    hard = {"stack", "rework"}  # keys keep probe-gated progress; stacks/rework stay binary
    total = wsum = 0.0
    per_kind: dict[str, list[float]] = {k: [] for k in STEP_KINDS_V07}
    for k, s in enumerate(ep.cs.steps):
        prog = ep.step_progress[k]
        if s.kind in hard and not ep.step_done[k]:
            prog = 0.0
        cr = step_credit(ep.step_done[k], ep.step_in_order[k], ep.step_done_at[k], ep.c.max_steps, prog)
        if s.kind == "key" and ep.step_done[k] and not ep.step_in_order[k]:
            cr = 0.0
        elif s.kind == "key" and ep.step_done[k] and ep.step_in_order[k]:
            # Short horizons compress the time bonus; keep in-order key finishes near full
            # credit so honest privileged stays ≥~0.50 without lengthening MAX_HORIZON.
            cr = max(cr, 0.95)
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
