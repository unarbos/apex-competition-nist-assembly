"""v0.7.6: actions-only key completability + assistance seals.

    NIST_PROFILE=v07 pytest -q tests/test_v076_keys.py
"""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tools"))

os.environ["NIST_PROFILE"] = "v07"

from nist_sim import conditions as C  # noqa: E402
from nist_sim.compound import TASK_PEG_FEAT  # noqa: E402
from nist_sim.compound_v07 import (  # noqa: E402
    KEY_CLOSE_NOISE,
    KEY_CLOSE_SOFT,
    KEY_PRESENT_MARK,
    KEY_PROBE_FRAMES,
    KEY_TOL,
    AssemblyIIIEpisode,
    sample_compound_v07,
)
from nist_sim.env import OBS_DIM  # noqa: E402
import obs_key_searcher as OKS  # noqa: E402


def test_key_tol_and_closeness_usable_but_nontrivial():
    # Window ~1.1–1.9× DYAW (0.08): reachable, not a free snap.
    assert KEY_TOL[0] >= 0.080
    assert KEY_TOL[1] <= 0.160
    assert KEY_TOL[0] < KEY_TOL[1]
    # Soft cue wider than the hard window so continuous sweep can lock on.
    assert KEY_CLOSE_SOFT >= 4.0
    assert KEY_CLOSE_NOISE <= 0.05
    assert 0.4 <= KEY_PRESENT_MARK <= 0.6


def test_key_present_mark_before_probe():
    """Identity cue publishes after reveal+touch+active, before press-probe."""
    _, conds = sample_compound_v07(7, 8)
    idx = next(
        (i for i, c in enumerate(conds) if any(s.kind == "key" for s in c.compound.steps)),
        None,
    )
    if idx is None:
        pytest.skip("no key")
    r1, _ = (np.random.default_rng(s) for s in C.episode_seed_sequence(7, idx).spawn(2))
    e = AssemblyIIIEpisode(conds[idx], r1)
    k = next(k for k, s in enumerate(e.cs.steps) if s.kind == "key")
    s = e.cs.steps[k]
    if s.reveal_after is not None:
        e.step_done[s.reveal_after] = True
    for p in s.prereq:
        e.step_done[p] = True
    e._active_key = k
    e._peg_touched[s.peg] = True
    e._peg_probe_frames[s.peg] = 0  # not yet probed
    e.seated_now[s.peg] = True
    o = e.observe(np.random.default_rng(0))
    b = OBS_DIM + s.peg * TASK_PEG_FEAT
    assert abs(float(o[b + 2]) - KEY_PRESENT_MARK) < 1e-5
    # Closeness stays off until press-probe finishes.
    assert float(o[b + 3]) == 0.0


def test_grasped_key_starts_inside_hole():
    """v0.7.6: grasped key pegs spawn with centre inside r_hole."""
    _, conds = sample_compound_v07(7, 8)
    checked = 0
    for i, c in enumerate(conds):
        for s in c.compound.steps:
            if s.kind != "key" or c.pegs[s.peg].start_type != "grasped":
                continue
            r1, _ = (np.random.default_rng(ss) for ss in C.episode_seed_sequence(7, i).spawn(2))
            e = AssemblyIIIEpisode(c, r1)
            p, _ = e.peg_pose(s.peg)
            hx, hy = e.slot_xy[s.peg]
            dxy = float(np.hypot(p[0] - hx, p[1] - hy))
            assert dxy <= e.r_hole[s.peg] + 1e-4, (i, dxy, e.r_hole[s.peg])
            checked += 1
            if checked >= 5:
                return
    if checked == 0:
        pytest.skip("no grasped key")


def test_actions_only_searcher_completes_some_keys():
    """Miner-path observe→act→step completes keys on seed 7 (short CI slice)."""
    # 24 eps keeps CI light but avoids a flaky 12-ep edge under seat-clear windows;
    # full 64-ep bar is still ~15+ (see CALIBRATION.md).
    r = OKS.run_eval_actions_only(seed=7, per=8, episodes=24, start=0)
    assert r["path"] == "actions_only"
    assert r["keys_total"] > 0
    assert r["keys_done"] >= 1, r


def test_actions_only_cannot_enable_fixture_or_nudges():
    """Static seal: actions-only path never references fixture seat or pose nudges."""
    src = inspect.getsource(OKS.ActionsOnlyKeyPolicy) + inspect.getsource(OKS.run_eval_actions_only)
    for tok in (
        "fixture_seat_key_peg",
        "_nudge_",
        "_set_free",
        "_peg_probe_frames",
        "peg_qadr",
        "oracle",
    ):
        assert tok not in src, tok


def test_fixture_path_is_explicitly_privileged():
    src = inspect.getsource(OKS.fixture_seat_key_peg)
    assert "_set_free" in src and "step_done" in src and "_peg_probe_frames" in src
