"""v0.8 seals for the v0.7.5 R32 audit hole classes (PR #40 findings).

Hole classes:
1. Per-step cues (closeness / detent) that a yaw sweep can hill-climb.
2. Key windows on slot symmetries → natural seat yaw freebie (esp. D-flat).
3. One heavily weighted subtask dominating the episode score.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))

os.environ["NIST_PROFILE"] = "v08"

from nist_sim.compound import wrap  # noqa: E402
from nist_sim.compound_v07 import KEY_HOLD, KEY_PROBE_FRAMES  # noqa: E402
from nist_sim.compound_v08 import (  # noqa: E402
    KEY_ANTI_ALIGN,
    STEP_WEIGHT_V08,
    AssemblyIVEpisode,
    compound_score_v08,
    sample_compound_v08,
)
from nist_sim.conditions import episode_seed_sequence  # noqa: E402


@pytest.fixture(scope="module")
def conds():
    _, c = sample_compound_v08(7, 8, "live")
    return c


def test_equal_weights_no_dominant_kind():
    assert set(STEP_WEIGHT_V08.values()) == {1.0}
    assert len(STEP_WEIGHT_V08) >= 7


def test_single_kind_cannot_dominate_score(conds):
    """Even finishing every key (or every insert) in order stays well below 1.0."""
    for ep_i in (0, 1, 7, 31):
        ss = episode_seed_sequence(7, ep_i)
        r1, _ = (np.random.default_rng(s) for s in ss.spawn(2))
        ep = AssemblyIVEpisode(conds[ep_i], r1)
        n = len(ep.cs.steps)
        for kind in ("key", "insert", "stack", "rework"):
            for k, s in enumerate(ep.cs.steps):
                ep.step_done[k] = False
                ep.step_progress[k] = 0.0
                ep._stability_ok[k] = True
                ep._precision[k] = 1.0
            for k, s in enumerate(ep.cs.steps):
                if s.kind == kind:
                    ep.step_done[k] = True
                    ep.step_in_order[k] = True
                    ep.step_done_at[k] = 10
            ep._finalized = True
            ep.done = True
            sc = compound_score_v08(ep)["score"]
            n_kind = sum(1 for s in ep.cs.steps if s.kind == kind)
            assert sc <= (n_kind / max(1, n)) + 1e-6
            # Blueprint length keeps any one kind from carrying most of the score.
            assert sc < 0.55, (ep_i, kind, sc, n_kind, n)


def test_no_hillclimbable_key_cue_in_obs(conds):
    """After probe+touch, feat3 (ex-closeness) stays flat; no detent torque bump."""
    found = False
    for ep_i in range(min(16, len(conds))):
        ss = episode_seed_sequence(7, ep_i)
        r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
        ep = AssemblyIVEpisode(conds[ep_i], r1)
        keys = [(k, s) for k, s in enumerate(ep.cs.steps) if s.kind == "key"]
        if not keys:
            continue
        found = True
        k, s = keys[0]
        peg = s.peg
        # Seat-ish + full probe/arm so v0.7.5 would have published closeness.
        ep.seated_now[peg] = True
        ep._peg_touched[peg] = True
        ep._peg_probe_frames[peg] = KEY_PROBE_FRAMES + 5
        ep._key_index_armed[peg] = True
        ep._active_key = k
        # Sweep hand yaw across the circle; closeness channel must not rise near the window.
        task0 = 184
        b = task0 + peg * 6
        samples = []
        for yaw in np.linspace(-2.9, 2.9, 61):
            ep.data.qpos[3] = float(yaw)
            # Strip any parent detent by observing after a zero step bookkeeping path.
            o = ep.observe(r2)
            samples.append(float(o[b + 3]))
        assert max(samples) <= 1e-6, max(samples)
        # Present-mark may be 0.5; must not vary with yaw (not a search gradient).
        marks = []
        for yaw in (-2.0, float(s.key_yaw), 2.0):
            ep.data.qpos[3] = float(yaw)
            o = ep.observe(r2)
            marks.append(float(o[b + 2]))
        assert len(set(round(m, 3) for m in marks)) == 1
        break
    assert found


def test_natural_seat_yaw_does_not_complete_key(conds):
    """Seat at slot symmetry + probe + hold must not complete an anti-aligned key."""
    from nist_sim.compound_v07 import KEY_HOLD

    checked = 0
    for ep_i in range(len(conds)):
        ss = episode_seed_sequence(7, ep_i)
        r1, _ = (np.random.default_rng(s) for s in ss.spawn(2))
        ep = AssemblyIVEpisode(conds[ep_i], r1)
        for k, s in enumerate(ep.cs.steps):
            if s.kind != "key":
                continue
            peg = s.peg
            slot_yaw = float(ep.c.pegs[peg].spec.yaw + ep.c.board_yaw)
            # Confirm window is off the natural seat.
            assert abs(wrap(s.key_yaw - slot_yaw)) >= KEY_ANTI_ALIGN - 1e-6
            # Give insert prereq credit so key is unblocked.
            for p in s.prereq:
                ep.step_done[p] = True
                ep.step_done_at[p] = 1
            ep.seated_now[peg] = True
            ep._peg_touched[peg] = True
            ep._peg_probe_frames[peg] = KEY_PROBE_FRAMES + 5
            # Leave/reenter arm: misalign then align to natural seat (not key window).
            ep.data.qpos[3] = float(wrap(slot_yaw + 0.5))
            ep._key_index_armed[peg] = False
            done, _ = ep._step_state(k)
            assert not done
            # Arm via leave
            ep._key_index_armed[peg] = True
            ep.data.qpos[3] = float(slot_yaw)  # natural seat yaw
            ep._key_hold[peg] = 0
            for _ in range(KEY_HOLD + 2):
                done, _ = ep._step_state(k)
            assert not done, (ep_i, k, s.key_yaw, slot_yaw, ep._key_hand_yaw())
            assert not ep.step_done[k]
            checked += 1
            if checked >= 8:
                break
        if checked >= 8:
            break
    assert checked >= 8


def test_no_dflat_auto_arm(conds):
    ep_i = 0
    ss = episode_seed_sequence(7, ep_i)
    r1, _ = (np.random.default_rng(s) for s in ss.spawn(2))
    ep = AssemblyIVEpisode(conds[ep_i], r1)
    for i in range(ep.n_pegs):
        assert ep._key_auto_arm(i) is False
