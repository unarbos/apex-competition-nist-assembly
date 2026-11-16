"""Regression: key windows must not complete from natural seat yaw alone.

    NIST_PROFILE=v07 pytest -q tests/test_key_window_accounting.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tools"))

os.environ["NIST_PROFILE"] = "v07"

from nist_sim import conditions as C  # noqa: E402
from nist_sim.compound import wrap  # noqa: E402
from nist_sim.compound_v07 import (  # noqa: E402
    KEY_HOLD,
    KEY_PROBE_FRAMES,
    KEY_TOL,
    _KEY_YAW_SEAT_CLEAR,
    _sample_key_yaw_for,
    AssemblyIIIEpisode,
    sample_compound_v07,
)
from obs_key_searcher import fixture_seat_key_peg  # noqa: E402


def test_key_windows_clear_of_seat_yaw():
    """Sampled hand-yaw windows stay clear of slot / symmetry seat poses."""
    rng = np.random.default_rng(0)
    _, conds = sample_compound_v07(7, 8)
    checked = 0
    for c in conds:
        for s in c.compound.steps:
            if s.kind != "key":
                continue
            slot_yaw = float(c.pegs[s.peg].spec.yaw + c.board_yaw)
            clear = max(float(_KEY_YAW_SEAT_CLEAR), 2.5 * float(KEY_TOL[1]))
            assert abs(wrap(s.key_yaw - slot_yaw)) >= clear - 1e-6
            # Fresh draws from the sampler also respect the clearance.
            for _ in range(5):
                y = _sample_key_yaw_for(rng, c, s.peg)
                assert abs(wrap(y - slot_yaw)) >= clear - 1e-6
            checked += 1
    assert checked > 0


def test_seat_yaw_hold_does_not_complete_key():
    """After seat+probe, holding at slot yaw must not complete the key step.

    Completion requires wrist search to the hidden hand-yaw window; seating
    orientation alone must not satisfy key accounting.
    """
    _, conds = sample_compound_v07(7, 8)
    tried = 0
    completed = 0
    for i, c in enumerate(conds[:32]):
        for k, s in enumerate(c.compound.steps):
            if s.kind != "key":
                continue
            r1, _ = (np.random.default_rng(ss) for ss in C.episode_seed_sequence(7, i).spawn(2))
            ep = AssemblyIIIEpisode(c, r1)
            if not fixture_seat_key_peg(ep, k):
                continue
            tried += 1
            slot_yaw = float(c.pegs[s.peg].spec.yaw + c.board_yaw)
            ep._peg_probe_frames[s.peg] = KEY_PROBE_FRAMES
            for p in s.prereq:
                ep.step_done[p] = True
            ep._pick_active_key(prefer=k)
            # Natural seat / insert hand yaw.
            ep.data.qpos[3] = float(np.clip(slot_yaw, -2.95, 2.95))
            if hasattr(ep, "target") and len(ep.target) > 3:
                ep.target[3] = float(ep.data.qpos[3])
            mujoco.mj_forward(ep.model, ep.data)
            for _ in range(KEY_HOLD + 24):
                if ep.step_done[k]:
                    completed += 1
                    break
                ep.step(np.array([0.0, 0.0, 0.55, 0.0, 1.0]))
            if tried >= 12:
                break
        if tried >= 12:
            break
    assert tried >= 8, tried
    # Clearance makes accidental seat-yaw hits rare; zero is the bar.
    assert completed == 0, (completed, tried)
