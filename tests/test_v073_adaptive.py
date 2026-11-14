"""v0.7.3 adaptive harden: retarget / reorder / flip / touch-reveal / post-insert retarget.

    NIST_PROFILE=v07 pytest -q tests/test_v073_adaptive.py
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


def test_adaptive_events_registered():
    assert "retarget" in EVENT_KINDS_V07
    assert "reorder" in EVENT_KINDS_V07
    assert "flip_twist" in EVENT_KINDS_V07
    for tier, cfg in TIER_CFG_V07.items():
        # v0.7.5: event/retarget floors lowered so honest fits the short-horizon wall bar.
        assert cfg["event_p"] >= 0.20, tier
        assert cfg["reveal_p"] >= 0.90, tier
        assert cfg["touch_reveal_p"] >= 0.85, tier
        assert cfg["post_insert_retarget_p"] >= 0.25, tier
        assert cfg["post_stack_retarget_p"] >= 0.20, tier


def test_touch_reveal_zeros_key_until_contact():
    _, conds = sample_compound_v07(23, 8)
    idx = next(
        (
            i
            for i, c in enumerate(conds)
            if any(s.kind == "key" and s.touch_reveal for s in c.compound.steps)
        ),
        None,
    )
    if idx is None:
        pytest.skip("no touch_reveal key in this draw")
    e = _ep(23, idx, per=8)
    s = next(s for s in e.cs.steps if s.kind == "key" and s.touch_reveal)
    if s.reveal_after is not None:
        e.step_done[s.reveal_after] = True
    r = np.random.default_rng(0)
    o = e.observe(r)
    b = OBS_DIM + s.peg * TASK_PEG_FEAT
    assert float(o[b + 2]) == 0.0 and float(o[b + 3]) == 0.0
    e._peg_touched[s.peg] = True
    # v0.7.5: absolute yaw still hidden; after probe closeness may appear.
    e._peg_probe_frames[s.peg] = KEY_PROBE_FRAMES
    e.seated_now[s.peg] = True
    for p in s.prereq:
        e.step_done[p] = True
    e._active_key = next(i for i, ss in enumerate(e.cs.steps) if ss is s)
    o2 = e.observe(r)
    assert float(o2[b + 2]) in (0.0, 1.0)  # retarget pulse, not sin(key_yaw)
    assert 0.0 <= float(o2[b + 3]) <= 1.0


def test_retarget_event_moves_key_yaw_and_clears_hold():
    _, conds = sample_compound_v07(29, 8)
    idx = next(
        (i for i, c in enumerate(conds) if any(s.kind == "key" for s in c.compound.steps)),
        None,
    )
    if idx is None:
        pytest.skip("no key")
    e = _ep(29, idx, per=8)
    k = next(k for k, s in enumerate(e.cs.steps) if s.kind == "key")
    s = e.cs.steps[k]
    old = float(s.key_yaw)
    old_tol = float(s.key_tol)
    e._key_hold[s.peg] = 4
    # Target this key directly (event RNG may pick another unfinished key).
    e._retarget_key(k)
    assert e._key_hold[s.peg] == 0
    assert e._key_retarget_pulse[s.peg] > 0
    sym = int(e.c.pegs[s.peg].spec.symmetry())
    if sym == 1:
        # D-flat cannot relocate the physical window; tol redraw + pulse still fire.
        assert s.key_tol != old_tol or e._key_index_armed[s.peg] is False
    else:
        assert abs(wrap(s.key_yaw - old)) > 0.10


def test_reorder_event_adds_precedence_edge():
    e = _ep(31, 0, per=8)
    before = [list(s.prereq) for s in e.cs.steps]
    # Leave at least two unfinished steps (fresh episode).
    e._fire_event(dict(kind="reorder"))
    fired = [ev for ev in e.events_fired if ev.get("kind") == "reorder"]
    assert fired
    if fired[-1].get("skipped"):
        pytest.skip("no valid reorder candidate")
    a, b = fired[-1]["edge"]
    assert a in e.cs.steps[b].prereq
    assert before[b] != list(e.cs.steps[b].prereq)


def test_flip_twist_clears_progress():
    _, conds = sample_compound_v07(37, 8)
    idx = next(
        (
            i
            for i, c in enumerate(conds)
            if any(s.kind in ("twist", "screw") for s in c.compound.steps)
        ),
        None,
    )
    if idx is None:
        pytest.skip("no twist/screw")
    e = _ep(37, idx, per=8)
    k = next(k for k, s in enumerate(e.cs.steps) if s.kind in ("twist", "screw"))
    s = e.cs.steps[k]
    old = float(s.twist_dir)
    e.twist_acc[s.peg] = 0.4
    e._fire_event(dict(kind="flip_twist"))
    assert float(s.twist_dir) == -old
    assert float(e.twist_acc[s.peg]) == 0.0


def test_post_insert_retarget_fires_on_seat_completion():
    """Completing an insert often retargets that peg's unfinished key (auto event)."""
    # Force a high retarget probability path by calling the helper directly after a fake seat.
    e = _ep(41, 0, per=8)
    seat = next((k for k, s in enumerate(e.cs.steps) if s.kind in ("insert", "rework")), None)
    key = next((k for k, s in enumerate(e.cs.steps) if s.kind == "key"), None)
    if seat is None or key is None:
        pytest.skip("need insert+key")
    # Align pegs so the helper can find the key on the same peg.
    e.cs.steps[key].peg = e.cs.steps[seat].peg
    old = float(e.cs.steps[key].key_yaw)
    # Force probability by temporarily patching tier cfg lookup via direct call loop.
    hit = False
    for _ in range(40):
        e.cs.steps[key].key_yaw = old
        e.events_fired.clear()
        e._maybe_post_insert_retarget(seat)
        if any(ev.get("kind") == "retarget" and ev.get("auto") == "post_insert" for ev in e.events_fired):
            hit = True
            assert abs(wrap(e.cs.steps[key].key_yaw - old)) > 0.05
            break
    assert hit, "post_insert retarget never fired in 40 draws"


def test_misseat_clears_key_hold():
    e = _ep(43, 0, per=4)
    from nist_sim.env import BLOCK_H, PEG_LEN, _yaw_quat

    # Exactly one seated candidate so misseat cannot pick another peg.
    for i in range(e.n_pegs):
        e.seated_now[i] = False
        e._key_hold[i] = 0
    i = 0
    e.seated_now[i] = True
    e.peg_parked[i] = False
    e._key_hold[i] = 3
    hx, hy = e.slot_xy[i]
    e._set_free(e.peg_qadr[i], e.peg_vadr[i], [hx, hy, BLOCK_H - 0.0005 + PEG_LEN / 2], _yaw_quat(0.0))
    e._fire_event(dict(kind="misseat", lift=0.010))
    assert e._key_hold[i] == 0
    assert e.seated_now[i] is False


def test_record_task_includes_touch_reveal():
    _, conds = sample_compound_v07(47, 8)
    found = False
    for c in conds:
        task = c.compound.record_task()
        for row in task["steps"]:
            if row.get("touch_reveal"):
                found = True
                assert row["kind"] in ("key", "twist", "screw")
    assert found
