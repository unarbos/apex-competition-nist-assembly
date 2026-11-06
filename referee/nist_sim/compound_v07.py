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
# Hard adaptive steps dominate total weight. Image-eval of revealed v0.6 kings on the
# prior weights still scored ~0.37 via inserts + stacks + some reworks — so insert/screw
# share is cut again and key/rework/stack carry more of the ceiling.
STEP_WEIGHT_V07 = {
    "insert": 0.07,
    "collar": 0.07,
    "twist": 0.07,
    "screw": 0.10,
    "rework": 3.60,
    "stack": 1.70,
    "key": 5.20,
}
# Short budgets + hard horizon keep slow ONNX kings inside the ~600 s / 64-ep / 2-CPU bar.
STEP_BUDGET_V07 = {
    "insert": 140,
    "collar": 100,
    "twist": 80,
    "rework": 220,
    "screw": 160,
    "stack": 160,
    "key": 130,
}
REWORK_EXTRA = 40
MAX_HORIZON_V07 = 1400  # hard sim-step cap; no wall-clock episode cut
TIER_CFG_V07 = {
    # Density: keys dominate; stacks require precise mate; reworks are secondary.
    # edge_p / event_p / reveal_p force obs-conditioned adaptation; slack keeps privileged room.
    "c1": dict(
        extra_pegs=0, collar_frac=0.10, twist_frac=0.03, screw_frac=0.06, rework_frac=0.28,
        stack_frac=0.32, key_frac=0.34, waves=(1, 1), edge_p=0.12, event_p=0.18, slack=1.55,
        reveal_p=0.40, max_keys=1,
    ),
    "c2": dict(
        extra_pegs=0, collar_frac=0.10, twist_frac=0.03, screw_frac=0.08, rework_frac=0.30,
        stack_frac=0.34, key_frac=0.36, waves=(1, 2), edge_p=0.16, event_p=0.22, slack=1.40,
        reveal_p=0.50, max_keys=2,
    ),
    "c3": dict(
        extra_pegs=1, collar_frac=0.08, twist_frac=0.02, screw_frac=0.08, rework_frac=0.32,
        stack_frac=0.34, key_frac=0.40, waves=(1, 2), edge_p=0.16, event_p=0.24, slack=1.35,
        reveal_p=0.55, max_keys=2,
    ),
    "c4": dict(
        # waves=(2, 2): at most 2 waves on c4 (docs previously claimed "up to 5"; code wins).
        extra_pegs=1, collar_frac=0.08, twist_frac=0.02, screw_frac=0.06, rework_frac=0.34,
        stack_frac=0.36, key_frac=0.44, waves=(2, 2), edge_p=0.18, event_p=0.28, slack=1.25,
        reveal_p=0.60, max_keys=2,
    ),
}
# Residuals large enough that open-loop / hard-coded trajectories miss; privileged uses true state.
NOISE_SCALE_V07 = dict(slot_bias=0.45, peg_delta=0.42, glitch=0.40, drift=0.42)
EVENT_KINDS_V07 = EVENT_KINDS + ("swap_bias", "misseat")
EVENT_WEIGHTS_V07 = np.array([0.13, 0.13, 0.11, 0.09, 0.15, 0.13, 0.14, 0.12])
# Multi-turn press-twists; pitch drawn per step and shown in obs when revealed.
SCREW_TARGET = (0.30, 0.55)
KEY_TOL = (0.05, 0.09)  # tighter yaw-index; privileged true-state nudge still reaches
STACK_SEAT_XY = 0.0030
STACK_SEAT_Z_CLEAR = 0.0028
# Screw press: palm must be on the peg (touch). Free-twist open-grip scripts still get 0.
PRESS_PALM_MARGIN = 0.001
# Rework pegs sit deep enough that casual yanks miss; privileged extract_routine still clears.
REWORK_SEAT_DEPTH = 0.0045
# Stack rings use a slight interference fit so a deliberate peg-top place can rest in the
# mate zone (clearance-fit rings free-fall ~4 mm in <1 control frame and can never hold).
# Mate zone + hold tightened after image-eval showed revealed kings completing ~half of stacks.
STACK_MATE_XY = 0.0030
STACK_MATE_Z = 0.0030
STACK_MATE_HOLD = 5  # ungripped frames in zone before pin (anti-king single-frame sweep)
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
    return dict(kind=kind, sigma=float(rng.uniform(0.14, 0.32)))


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
    # Guarantee at least one stack or rework so insert/collar/screw-only kings stay capped.
    if not rework_pegs and not any(k == "stack" for _, k in ring_pegs):
        if remain or [i for i in cands if i not in {p for p, _ in ring_pegs}]:
            pool = [i for i in cands if i not in {p for p, _ in ring_pegs}] or list(cands)
            peg_i = int(rng.choice(pool))
            if len(collars) < MAX_COLLARS:
                outer = cond.pegs[peg_i].spec.outer_radius()
                r_in = max(0.0010, outer - float(rng.uniform(*STACK_INTERFERENCE)))
                spot = _collar_spot(rng, cond, r_in + COLLAR_T, taken)
                taken.append(spot)
                w = int(rng.integers(peg_wave[peg_i], W))
                collars.append(CollarSpec(peg=peg_i, r_in=r_in, offset_xy=spot, wave=w, kind="stack"))
                ring_pegs.append((peg_i, "stack"))
        elif free_slots and rework_cands:
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

    def _sample_key_yaw(peg_i: int) -> float:
        """World-frame yaw window centre that a seated peg can actually reach."""
        peg = cond.pegs[peg_i]
        spec = peg.spec
        slot_yaw = float(spec.yaw + cond.board_yaw)
        if spec.shape == "round" or spec.symmetry() == 0:
            return float(rng.uniform(-math.pi, math.pi))
        # Discrete seated yaws: slot_yaw + 2π k / n for rotational symmetry n.
        n = max(1, int(spec.symmetry()))
        k_sym = int(rng.integers(0, n))
        # Small jitter inside the key window so the target is not exactly the slot yaw
        # (obs must still be read), but stays reachable while seated.
        jitter = float(rng.uniform(-0.04, 0.04))
        return float(wrap(slot_yaw + (2.0 * math.pi * k_sym) / n + jitter))

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
            steps.append(
                TaskStep(
                    "key",
                    i,
                    i,
                    key_yaw=_sample_key_yaw(i),
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
        super().__init__(cond, rng)

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
                # Drop latch on insert/rework for this peg so credit must be re-earned.
                for k, s in enumerate(self.cs.steps):
                    if s.kind in ("insert", "rework") and s.peg == i:
                        self.step_done[k] = False
                        self.step_done_at[k] = -1
                mujoco.mj_forward(self.model, self.data)
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
            # 80% of target is enough: short press-twists often undershoot by a few degrees.
            done = frac >= 0.80 and (self._peg_in_hole(i) or self.seated_now[i])
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
        # Sole bookkeeping path for v0.7 hard steps (parent skipped them).
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
                # Latch: brief flickers must not wipe completion or redo time bonus.
                continue
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
                # Keys are never co-assigned with twist/screw, so these channels stay free.
                if revealed:
                    o[b + 2] = math.sin(s.key_yaw)
                    o[b + 3] = math.cos(s.key_yaw)
                    o[b + 4] = float(self.step_blocked(k))
                    # Pack tol into the collar-flag slot when that peg has no collar/stack.
                    if o[b + 5] == 0.0:
                        o[b + 5] = float(s.key_tol) * 10.0
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
    """Weighted per-step score using v0.7 weights when new kinds are present.

    Key / stack / rework are all-or-nothing for scoring: near-miss partials do not
    count. That keeps open-loop v0.6 kings from farming partial credit on adaptive
    gates they never finish, while honest privileged still earns full weight on done.
    """
    kinds = {s.kind for s in ep.cs.steps}
    if kinds <= {"insert", "collar", "twist"}:
        return _compound_score_v06(ep)
    hard = {"key", "stack", "rework"}
    total = wsum = 0.0
    per_kind: dict[str, list[float]] = {k: [] for k in STEP_KINDS_V07}
    for k, s in enumerate(ep.cs.steps):
        prog = ep.step_progress[k]
        if s.kind in hard and not ep.step_done[k]:
            prog = 0.0
        cr = step_credit(ep.step_done[k], ep.step_in_order[k], ep.step_done_at[k], ep.c.max_steps, prog)
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
