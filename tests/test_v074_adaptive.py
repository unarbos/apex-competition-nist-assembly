"""v0.7.4 branching / probe harden: activate, regraph, probe refine, post-stack retarget.

    NIST_PROFILE=v07 pytest -q tests/test_v074_adaptive.py
"""

from __future__ import annotations

import math
import os
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
    EVENT_KINDS_V07,
    KEY_CLOSE_NOISE,
    KEY_HOLD,
    KEY_PROBE_FRAMES,
    TIER_CFG_V07,
    AssemblyIIIEpisode,
    sample_compound_v07,
)
from nist_sim.env import OBS_DIM  # noqa: E402


def _ep(seed: int, i: int, per: int = 8) -> AssemblyIIIEpisode:
    _, conds = sample_compound_v07(seed, per)
    r1, _ = (np.random.default_rng(s) for s in C.episode_seed_sequence(seed, i).spawn(2))
    return AssemblyIIIEpisode(conds[i], r1)


def test_v074_events_registered():
    assert "regraph" in EVENT_KINDS_V07
    assert "activate" in EVENT_KINDS_V07
    for tier, cfg in TIER_CFG_V07.items():
        assert cfg["event_p"] >= 0.32, tier
        assert cfg["touch_reveal_p"] >= 1.0 - 1e-9, tier
        assert cfg["post_insert_retarget_p"] >= 0.55, tier
        assert cfg["post_stack_retarget_p"] >= 0.40, tier
        assert cfg["reveal_p"] >= 1.0 - 1e-9, tier
        assert int(cfg["max_keys"]) >= 1, tier
        assert int(cfg["max_keys"]) <= 2, tier


def test_key_obs_hides_absolute_yaw_publishes_closeness():
    """v0.7.5: absolute yaw stays hidden; coarse noisy closeness appears after probe."""
    _, conds = sample_compound_v07(53, 8)
    idx = next(
        (i for i, c in enumerate(conds) if any(s.kind == "key" for s in c.compound.steps)),
        None,
    )
    if idx is None:
        pytest.skip("no key")
    e = _ep(53, idx, per=8)
    k = next(k for k, s in enumerate(e.cs.steps) if s.kind == "key")
    s = e.cs.steps[k]
    if s.reveal_after is not None:
        e.step_done[s.reveal_after] = True
    for p in s.prereq:
        e.step_done[p] = True
    e._active_key = k
    e._peg_touched[s.peg] = True
    e._peg_probe_frames[s.peg] = KEY_PROBE_FRAMES
    e.seated_now[s.peg] = True
    r = np.random.default_rng(0)
    o1 = e.observe(r)
    b = OBS_DIM + s.peg * TASK_PEG_FEAT
    # feat 2 is retarget pulse (0 unless recently retargeted), not sin(key_yaw).
    assert float(o1[b + 2]) in (0.0, 1.0)
    assert 0.0 <= float(o1[b + 3]) <= 1.0  # noisy closeness
    has_ring = any(t.kind in ("collar", "stack") and t.peg == s.peg for t in e.cs.steps)
    if not has_ring:
        assert abs(float(o1[b + 5]) - s.key_tol * 10.0) < 1e-5
    assert KEY_CLOSE_NOISE > 0.0


def _dual_key_episode() -> AssemblyIIIEpisode:
    for seed in range(59, 160):
        _, conds = sample_compound_v07(seed, 8)
        for i, c in enumerate(conds):
            if sum(1 for s in c.compound.steps if s.kind == "key") >= 2:
                return _ep(seed, i, per=8)
    pytest.skip("no dual-key episode")


def test_inactive_key_blocked_in_obs_and_cannot_hold():
    e = _dual_key_episode()
    keys = [k for k, s in enumerate(e.cs.steps) if s.kind == "key"]
    assert len(keys) >= 2
    # Ensure both keys remain unfinished so single-active gating applies.
    for k in keys:
        e.step_done[k] = False
    e._active_key = keys[0]
    inactive = keys[1]
    # Mark prereqs done so only activate gate remains.
    for p in e.cs.steps[inactive].prereq:
        e.step_done[p] = True
    e.seated_now[e.cs.steps[inactive].peg] = True
    assert len(e._unfinished_keys()) >= 2
    assert e._key_is_active(inactive) is False
    r = np.random.default_rng(1)
    if e.cs.steps[inactive].reveal_after is not None:
        e.step_done[e.cs.steps[inactive].reveal_after] = True
    e._peg_touched[e.cs.steps[inactive].peg] = True
    e._peg_probe_frames[e.cs.steps[inactive].peg] = KEY_PROBE_FRAMES
    o = e.observe(r)
    b = OBS_DIM + e.cs.steps[inactive].peg * TASK_PEG_FEAT
    assert float(o[b + 4]) == 1.0  # blocked
    # Hold must not accumulate while inactive (progress forced to 0).
    e._key_hold[e.cs.steps[inactive].peg] = 0
    done, prog = e._step_state(inactive)
    assert done is False
    assert prog == 0.0


def test_activate_event_rotates_active_key():
    e = None
    for seed in range(70, 140):
        _, conds = sample_compound_v07(seed, 8)
        idx = next(
            (
                i
                for i, c in enumerate(conds)
                if sum(1 for s in c.compound.steps if s.kind == "key") >= 2
            ),
            None,
        )
        if idx is None:
            continue
        e = _ep(seed, idx, per=8)
        break
    if e is None:
        pytest.skip("no dual-key episode")
    keys = [k for k, s in enumerate(e.cs.steps) if s.kind == "key"]
    e._active_key = keys[0]
    e._fire_event(dict(kind="activate"))
    fired = [ev for ev in e.events_fired if ev.get("kind") == "activate"]
    assert fired
    if fired[-1].get("skipped"):
        pytest.skip("activate skipped")
    assert e._active_key in keys
    assert e._active_key != keys[0] or len(keys) == 1


def test_regraph_event_rewrites_soft_edges():
    e = _ep(77, 0, per=8)
    before = [list(s.prereq) for s in e.cs.steps]
    e._fire_event(dict(kind="regraph"))
    fired = [ev for ev in e.events_fired if ev.get("kind") == "regraph"]
    assert fired
    if fired[-1].get("skipped"):
        pytest.skip("regraph skipped")
    after = [list(s.prereq) for s in e.cs.steps]
    assert before != after or fired[-1].get("dropped", 0) > 0 or fired[-1].get("edges")


def test_post_stack_retarget_fires():
    e = _ep(83, 0, per=8)
    stack = next((k for k, s in enumerate(e.cs.steps) if s.kind == "stack"), None)
    key = next((k for k, s in enumerate(e.cs.steps) if s.kind == "key"), None)
    if stack is None or key is None:
        pytest.skip("need stack+key")
    e.cs.steps[key].peg = e.cs.steps[stack].peg
    old = float(e.cs.steps[key].key_yaw)
    hit = False
    for _ in range(40):
        e.cs.steps[key].key_yaw = old
        e.events_fired.clear()
        e._maybe_post_stack_retarget(stack)
        if any(ev.get("kind") == "retarget" and ev.get("auto") == "post_stack" for ev in e.events_fired):
            hit = True
            assert abs(wrap(e.cs.steps[key].key_yaw - old)) > 0.05
            break
    assert hit, "post_stack retarget never fired in 40 draws"


def test_retarget_clears_hold_and_index_arm_keeps_probe():
    e = _ep(89, 0, per=8)
    key = next((k for k, s in enumerate(e.cs.steps) if s.kind == "key"), None)
    if key is None:
        pytest.skip("no key")
    s = e.cs.steps[key]
    e._peg_probe_frames[s.peg] = KEY_PROBE_FRAMES
    e._key_hold[s.peg] = 4
    e._key_index_armed[s.peg] = True
    old = float(s.key_yaw)
    e._retarget_key(key)
    assert e._key_hold[s.peg] == 0
    assert e._key_index_armed[s.peg] is False
    assert e._peg_probe_frames[s.peg] == KEY_PROBE_FRAMES
    assert abs(wrap(s.key_yaw - old)) > 0.05


def test_key_requires_leave_and_reenter_after_probe():
    """Natural-seat freebie: probed+aligned without leaving the window must not complete."""
    e = _dual_key_episode()
    k = next(k for k, s in enumerate(e.cs.steps) if s.kind == "key")
    s = e.cs.steps[k]
    for p in s.prereq:
        e.step_done[p] = True
    e._active_key = k
    e._peg_probe_frames[s.peg] = KEY_PROBE_FRAMES
    e._key_index_armed[s.peg] = False
    e.seated_now[s.peg] = True
    e._key_aligned = lambda _s: True  # type: ignore[method-assign]
    e._gripper_touching = lambda _b: False  # type: ignore[method-assign]
    done, prog = e._step_state(k)
    assert done is False
    assert e._key_hold[s.peg] == 0
    # Leaving the window arms the index gate.
    e._key_aligned = lambda _s: False  # type: ignore[method-assign]
    e._step_state(k)
    assert e._key_index_armed[s.peg] is True
    e._key_aligned = lambda _s: True  # type: ignore[method-assign]
    e._key_hold[s.peg] = 0
    for _ in range(KEY_PROBE_FRAMES):  # enough idle frames; hold needs KEY_HOLD
        done, _ = e._step_state(k)
        if done:
            break
    # Hold accumulates only when armed+aligned; simulate KEY_HOLD ticks.
    e._key_hold[s.peg] = 0
    for _ in range(KEY_HOLD):
        done, _ = e._step_state(k)
    assert done is True
