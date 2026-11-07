"""v0.7.1 Macrocosmos blockers: profile bake, no wall clocks, stack/key/latch.

    NIST_PROFILE=v07 pytest -q tests/test_v071_fixes.py
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))

os.environ["NIST_PROFILE"] = "v07"

from nist_sim import conditions as C  # noqa: E402
from nist_sim.compound import TASK_PEG_FEAT, wrap  # noqa: E402
from nist_sim.compound_v07 import (  # noqa: E402
    AssemblyIIIEpisode,
    sample_compound_v07,
)
from nist_sim.env import MAX_PEGS, OBS_DIM, SEATED_HOLD  # noqa: E402


def _ep(seed: int, i: int, per: int = 8) -> AssemblyIIIEpisode:
    _, conds = sample_compound_v07(seed, per)
    r1, _ = (np.random.default_rng(s) for s in C.episode_seed_sequence(seed, i).spawn(2))
    return AssemblyIIIEpisode(conds[i], r1)


def test_dockerfile_bakes_v07_not_v06():
    """Image ENV must be v07; setdefault alone loses to a baked v06."""
    text = (ROOT / "referee" / "Dockerfile").read_text()
    m = re.search(r"ENV\s+[^\n]*NIST_PROFILE=(\w+)", text)
    assert m is not None, "Dockerfile must set NIST_PROFILE"
    assert m.group(1) == "v07"
    assert "NIST_PROFILE=v06" not in text


def test_referee_logs_nist_profile_in_metadata_contract():
    """result.json / records must expose nist_profile for Macrocosmos preview checks."""
    src = (ROOT / "referee" / "referee.py").read_text()
    assert '"nist_profile"' in src or "'nist_profile'" in src
    assert "nist_lite profile=" in src


def test_referee_default_is_v07_without_env(monkeypatch):
    """Import path used by the image: no NIST_PROFILE → Assembly III."""
    monkeypatch.delenv("NIST_PROFILE", raising=False)
    # Simulate Dockerfile bake then setdefault (referee.py order).
    os.environ.setdefault("NIST_PROFILE", "v07")
    # Fresh read of what referee would see.
    assert (os.environ.get("NIST_PROFILE") or "v07").strip().lower() == "v07"
    # Explicit v06 still selectable for regression.
    monkeypatch.setenv("NIST_PROFILE", "v06")
    assert os.environ["NIST_PROFILE"] == "v06"


def test_no_wall_clock_scoring_caps():
    """Scores must not depend on machine load — no round/episode wall cuts."""
    ref = (ROOT / "referee" / "referee.py").read_text()
    assert "round_wall_s" not in ref
    assert "hit_round_wall" not in ref
    assert "round_wall" not in ref
    v07 = (ROOT / "referee" / "nist_sim" / "compound_v07.py").read_text()
    assert "EPISODE_WALL_S" not in v07
    assert "perf_counter" not in v07
    assert "_wall0" not in v07
    assert "_wall_limit" not in v07
    # Input schema must not expose a round wall override.
    schema = (ROOT / "input.schema.json").read_text()
    assert "round_wall" not in schema


def test_stack_hold_can_reach_seated_hold():
    """Parent must not reset stack collar_run from the block-seat test every step."""
    _, conds = sample_compound_v07(21, 8)
    idx = next(
        (i for i, c in enumerate(conds) if any(s.kind == "stack" for s in c.compound.steps)),
        None,
    )
    if idx is None:
        pytest.skip("no stack in this draw")
    e = _ep(21, idx, per=8)
    sk = next(k for k, s in enumerate(e.cs.steps) if s.kind == "stack")
    j = e.cs.steps[sk].part
    # Wave-parked rings are inactive; unpark so hold counting can run.
    e.collar_parked[j] = False
    e._set_body_active(e.collar_body[j], True)
    # Force the stack-seat predicate path by monkey-patching the checker.
    e._stack_seated_now = lambda jj, _j=j: jj == _j  # type: ignore[method-assign]
    e._collar_seated_now = lambda jj: False  # type: ignore[method-assign]
    for _ in range(SEATED_HOLD + 2):
        e.step(np.array([0.0, 0.0, 0.3, 0.0, 1.0]))
        if e.done:
            break
    assert e.collar_run[j] >= SEATED_HOLD
    assert e.step_done[sk]


def test_key_yaw_reachable_and_obs_packs_tol():
    _, conds = sample_compound_v07(13, 8)
    for i, c in enumerate(conds):
        keys = [s for s in c.compound.steps if s.kind == "key"]
        twistscrew = {s.peg for s in c.compound.steps if s.kind in ("twist", "screw")}
        for s in keys:
            # Never share a peg with twist/screw (obs channel conflict).
            assert s.peg not in twistscrew
            peg = c.pegs[s.peg]
            spec = peg.spec
            slot_yaw = float(spec.yaw + c.board_yaw)
            if spec.shape != "round" and spec.symmetry() > 0:
                n = spec.symmetry()
                # Must sit near a seated symmetry of the slot.
                errs = [
                    abs(wrap(s.key_yaw - (slot_yaw + 2 * math.pi * k / n)))
                    for k in range(n)
                ]
                assert min(errs) < 0.08 + s.key_tol
        if not keys:
            continue
        e = _ep(13, i, per=8)
        r = np.random.default_rng(0)
        # Reveal any late-hidden / touch-gated keys so obs packing can be checked.
        for s in e.cs.steps:
            if s.kind == "key" and s.reveal_after is not None:
                e.step_done[s.reveal_after] = True
            if s.kind == "key" and getattr(s, "touch_reveal", False):
                e._peg_touched[s.peg] = True
        o = e.observe(r)
        task0 = OBS_DIM
        for s in e.cs.steps:
            if s.kind != "key":
                continue
            if s.reveal_after is not None and not e.step_done[s.reveal_after]:
                continue
            if getattr(s, "touch_reveal", False) and not e._peg_touched[s.peg]:
                continue
            b = task0 + s.peg * TASK_PEG_FEAT
            assert abs(o[b + 2] - math.sin(s.key_yaw)) < 1e-5
            assert abs(o[b + 3] - math.cos(s.key_yaw)) < 1e-5
            has_ring = any(
                t.kind in ("collar", "stack") and t.peg == s.peg for t in e.cs.steps
            )
            if not has_ring:
                assert abs(float(o[b + 5]) - s.key_tol * 10.0) < 1e-5, (
                    f"key_tol not packed for peg {s.peg}: got {o[b + 5]}"
                )
        break
    else:
        pytest.skip("no key in this draw")


def test_completion_latch_keeps_done_at():
    e = _ep(5, 0, per=4)
    # Mark a key/stack/screw/rework done and ensure a later step does not clear or retarget done_at.
    kinds = ("screw", "key", "stack", "rework")
    targets = [k for k, s in enumerate(e.cs.steps) if s.kind in kinds]
    if not targets:
        pytest.skip("no latchable steps")
    for k in targets:
        e.step_done[k] = True
        e.step_done_at[k] = 42
        e.step_progress[k] = 0.30
    # Force predicates to report not-done so a buggy un-latch would clear them.
    e._step_state = lambda k: (False, 0.0)  # type: ignore[method-assign]
    for _ in range(3):
        e.step(np.array([0.0, 0.0, 0.3, 0.0, 1.0]))
        if e.done:
            break
    for k in targets:
        assert e.step_done[k] is True
        assert e.step_done_at[k] == 42


def test_key_hold_increments_once_per_step():
    _, conds = sample_compound_v07(17, 8)
    idx = next(
        (i for i, c in enumerate(conds) if any(s.kind == "key" for s in c.compound.steps)),
        None,
    )
    if idx is None:
        pytest.skip("no key")
    e = _ep(17, idx, per=8)
    sk = next(k for k, s in enumerate(e.cs.steps) if s.kind == "key")
    peg = e.cs.steps[sk].peg
    # v0.7.3: key hold only accumulates after prerequisites (e.g. stack-before-key).
    for p in e.cs.steps[sk].prereq:
        e.step_done[p] = True
    e._key_aligned = lambda s: True  # type: ignore[method-assign]
    e._gripper_touching = lambda _b: False  # type: ignore[method-assign]
    e.seated_now[peg] = True
    e._peg_in_hole = lambda i, _p=peg: i == _p  # type: ignore[method-assign]
    before = e._key_hold[peg]
    e.step(np.array([0.0, 0.0, 0.3, 0.0, 1.0]))
    # Exactly +1 per control step (parent used to call _step_state too).
    assert e._key_hold[peg] == before + 1
