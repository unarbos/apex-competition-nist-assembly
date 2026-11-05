"""Port of a strong v0.6 insert→collar→twist scripted strategy.

Models the class of controller that saturates live v0.6 (insert every peg, then collar /
twist on that peg; ~85% step completion on v0.6). Uses **privileged state** for those three
kinds only — the same information a well-tuned compiled controller extracts from the 288-d
observation on v0.6. On v0.7 it does **not** implement rework / screw / stack / key: screws are
attempted as plain twists (no press), stacks as block-seat collars, reworks as normal inserts
(no extract), and keys are ignored.

Calibration gate: this should score high on ``v06`` and ≤ ~0.35 on ``v07``.

    NIST_PROFILE=v07 python tools/v06_style_scripted.py --seed 20 --per 2 --episodes 8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tools"))

from nist_sim import conditions as C  # noqa: E402
from nist_sim.compound import CompoundEpisode, compound_score, sample_compound  # noqa: E402
from nist_sim.compound_v07 import (  # noqa: E402
    AssemblyIIIEpisode,
    compound_score_v07,
    sample_compound_v07,
)
from compound_solver import (  # noqa: E402
    _available,
    _is_lying_insert,
    _priority,
    collar_routine,
    insert_routine,
    twist_routine,
)
from reference_solver import _push_clutter, part_axis  # noqa: E402


def solve_v06_style(ep: CompoundEpisode, max_attempts: int = 2) -> dict:
    """Run only insert / collar / twist routines; map screw→twist, stack→collar, skip key/rework extract."""
    is_v07 = isinstance(ep, AssemblyIIIEpisode) or any(
        s.kind in ("rework", "screw", "stack", "key") for s in ep.cs.steps
    )
    score_fn = compound_score_v07 if is_v07 else compound_score
    budget = {"insert": 300, "collar": 260, "twist": 260, "rework": 300, "screw": 260, "stack": 260, "key": 40}
    attempts = [0] * len(ep.cs.steps)

    def run(gen, cap):
        n0 = ep.steps
        for a in gen:
            if ep.done or ep.steps - n0 > cap:
                break
            ep.step(a)

    while not ep.done:
        remaining = ep.c.max_steps - ep.steps
        todo = [
            k for k in range(len(ep.cs.steps))
            if not ep.step_done[k] and _available(ep, k) and attempts[k] < max_attempts
            and ep.cs.steps[k].kind != "key"  # v0.6 strategy has no key concept
        ]
        lying = {k for k in todo if _is_lying_insert(ep, k)}
        in_order = [k for k in todo if not ep.step_blocked(k)]
        ready = [k for k in in_order if k not in lying]
        if not ready and not all(ep.wave_arrived) and remaining > 120:
            ep.step(np.array([0.0, 0.0, 0.3, 0.0, 1.0]))
            continue
        if not ready:
            ready = [k for k in todo if k not in lying]
        if not ready and remaining > 280:
            ready = [k for k in todo if k in lying and attempts[k] == 0]
        if not ready or remaining < 35:
            ep.step(np.array([0.0, 0.0, 0.3, 0.0, 1.0]))
            continue
        k = min(ready, key=lambda kk: _priority(ep, kk))
        attempts[k] += 1
        s = ep.cs.steps[k]
        cap = min(budget.get(s.kind, 260), remaining - 15)
        if s.kind in ("insert", "rework"):
            # Rework treated as plain insert — no extract. That is the v0.6 blind spot.
            _, q = ep.peg_pose(s.peg)
            lying = abs(part_axis(q)[2]) < 0.5
            _push_clutter(ep, s.peg)
            run(insert_routine(ep, s.peg), min(cap + (150 if lying else 0), remaining - 15))
        elif s.kind in ("collar", "stack"):
            # Stack attempted as a block-seat collar — wrong z target on v0.7.
            run(collar_routine(ep, s.part, stack=False), cap)
        elif s.kind in ("twist", "screw"):
            # Screw attempted as a free twist — no press gate.
            run(twist_routine(ep, k, press=False), cap)
        else:
            ep.step(np.array([0.0, 0.0, 0.3, 0.0, 1.0]))
        for _ in range(10):
            if ep.done:
                break
            ep.step(np.array([0.0, 0.0, 1.0, 0.0, 1.0]))
    return score_fn(ep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--profile", default=os.environ.get("NIST_PROFILE", "v07"), choices=["v06", "v07"])
    a = ap.parse_args()
    os.environ["NIST_PROFILE"] = a.profile
    sample = sample_compound_v07 if a.profile == "v07" else sample_compound
    make_ep = AssemblyIIIEpisode if a.profile == "v07" else CompoundEpisode
    _, conds = sample(a.seed, a.per)
    scores = []
    for i in range(min(len(conds), a.episodes)):
        c = conds[i]
        r1, _ = (np.random.default_rng(s) for s in C.episode_seed_sequence(a.seed, i).spawn(2))
        ep = make_ep(c, r1)
        r = solve_v06_style(ep)
        scores.append(r["score"])
        print(
            f"ep {i:02d} {c.compound.tier} score={r['score']:.3f} done={r['n_done']}/{r['n_steps']} "
            f"kinds={r['per_kind']} t={ep.steps}/{c.max_steps}",
            flush=True,
        )
    print(f"mean {float(np.mean(scores)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
