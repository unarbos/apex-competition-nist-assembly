"""v0.7.5: observation-completable keys — closeness, detent, retarget pulse, D-flat arm.

    NIST_PROFILE=v07 pytest -q tests/test_v075_keys.py
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
sys.path.insert(0, str(ROOT / "tools"))

os.environ["NIST_PROFILE"] = "v07"

from nist_sim import conditions as C  # noqa: E402
from nist_sim.compound import TASK_PEG_FEAT, wrap  # noqa: E402
from nist_sim.compound_v07 import (  # noqa: E402
    KEY_CLOSE_NOISE,
    KEY_CLOSE_SOFT,
    KEY_DETENT_TAU,
    KEY_HOLD,
    KEY_PROBE_FRAMES,
    KEY_RETARGET_PULSE,
    STEP_WEIGHT_V07,
    AssemblyIIIEpisode,
    sample_compound_v07,
)
from nist_sim.env import OBS_DIM  # noqa: E402
import obs_key_searcher as OKS  # noqa: E402


def _ep(seed: int, i: int, per: int = 8) -> AssemblyIIIEpisode:
    _, conds = sample_compound_v07(seed, per)
    r1, _ = (np.random.default_rng(s) for s in C.episode_seed_sequence(seed, i).spawn(2))
    return AssemblyIIIEpisode(conds[i], r1)


def test_weights_missing_key_caps_score():
    # Missing keys must dominate: key leads, stack/rework secondary, inserts tiny.
    w = STEP_WEIGHT_V07
    typical = 3 * w["insert"] + w["stack"] + w["key"]
    assert w["key"] / typical >= 0.70
    assert w["key"] > w["stack"] >= w["rework"]
    assert w["screw"] > w["insert"]
    assert abs(w["key"] - 6.00) < 1e-9
    assert abs(w["stack"] - 1.20) < 1e-9
    assert abs(w["rework"] - 1.10) < 1e-9
    assert KEY_HOLD == 4
    # Model: all non-key done @0.85, keys unfinished → ≤~0.30.
    nonkey = 3 * w["insert"] + w["stack"] + w["rework"] + w["screw"]
    cap = (nonkey * 0.85) / (nonkey + w["key"])
    assert cap <= 0.30, cap


def test_key_obs_publishes_noisy_closeness_not_absolute_yaw():
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
    # v0.7.5: closeness is hand-yaw vs key window — point the wrist at key_yaw.
    import mujoco

    e.data.qpos[3] = float(s.key_yaw)
    mujoco.mj_forward(e.model, e.data)
    r = np.random.default_rng(0)
    o = e.observe(r)
    b = OBS_DIM + s.peg * TASK_PEG_FEAT
    # Absolute yaw still hidden: feat 2 is retarget pulse (0 here), not sin(key_yaw).
    assert float(o[b + 2]) == 0.0
    close = float(o[b + 3])
    assert close > 0.5, close
    assert KEY_CLOSE_SOFT >= 2.0
    assert KEY_CLOSE_NOISE > 0.0
    assert KEY_DETENT_TAU > 0.0
    assert abs(e._key_hand_yaw() - s.key_yaw) < 1e-5


def test_retarget_sets_observable_pulse():
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
    assert e._key_retarget_pulse[s.peg] == KEY_RETARGET_PULSE
    # Non-round D-flat may keep yaw and only redraw tol; otherwise window must move.
    if e._key_symmetry(s.peg) != 1:
        assert abs(wrap(s.key_yaw - old)) > 0.05
    # Pulse visible in obs after probe+reveal.
    if s.reveal_after is not None:
        e.step_done[s.reveal_after] = True
    for p in s.prereq:
        e.step_done[p] = True
    e._active_key = key
    e._peg_touched[s.peg] = True
    e.seated_now[s.peg] = True
    o = e.observe(np.random.default_rng(1))
    b = OBS_DIM + s.peg * TASK_PEG_FEAT
    assert float(o[b + 2]) == 1.0


def test_dflat_keys_auto_arm_across_tol_range():
    """D-flat (symmetry 1) must arm after probe without leaving the window."""
    found = 0
    for seed in range(11, 200):
        _, conds = sample_compound_v07(seed, 8)
        for i, c in enumerate(conds):
            for k, s in enumerate(c.compound.steps):
                if s.kind != "key":
                    continue
                if c.pegs[s.peg].spec.symmetry() != 1:
                    continue
                e = _ep(seed, i, per=8)
                # Use the live step's tol; also stress low/high ends of KEY_TOL.
                for tol in (0.045, float(s.key_tol), 0.080):
                    e.cs.steps[k].key_tol = tol
                    for p in e.cs.steps[k].prereq:
                        e.step_done[p] = True
                    e._active_key = k
                    e._peg_probe_frames[s.peg] = KEY_PROBE_FRAMES
                    e._key_index_armed[s.peg] = False
                    e.seated_now[s.peg] = True
                    # Stay aligned (cannot leave window) — auto-arm must still fire.
                    e._key_aligned = lambda _s: True  # type: ignore[method-assign]
                    e._gripper_touching = lambda _b: False  # type: ignore[method-assign]
                    done, _ = e._step_state(k)
                    assert e._key_index_armed[s.peg] is True
                    assert e._key_auto_arm(s.peg) is True
                    # Hold can now accumulate.
                    e._key_hold[s.peg] = 0
                    for _ in range(KEY_HOLD):
                        done, _ = e._step_state(k)
                    assert done is True
                found += 1
                if found >= 3:
                    return
    if found == 0:
        pytest.skip("no dflat key in draw")


def test_obs_key_searcher_completes_keys_seed7():
    """Fixture+obs yaw search: closeness channel is searchable after privileged seat."""
    # Privileged fixture seats/probes; policy is obs+actions only for the yaw search.
    # 8 eps keeps CI light. Cue is noisy — rate can be modest; >0 is the observability bar.
    r = OKS.run_eval(seed=7, per=8, episodes=8, start=0, budget=500)
    assert r["keys_total"] > 0
    assert r["keys_done"] > 0, r
    assert r["key_rate"] >= 0.10, r
    # Still costs real frames (not an instant snap).
    assert r["mean_frames"] > 20.0


def test_no_obs_key_yaw_helper():
    assert not hasattr(AssemblyIIIEpisode, "_obs_key_yaw")


def test_dead_stack_key_edge_not_added():
    # Keys only keep insert/rework prereqs — stack must never remain as a key prereq.
    _, conds = sample_compound_v07(41, 8)
    for c in conds:
        for s in c.compound.steps:
            if s.kind != "key":
                continue
            for p in s.prereq:
                assert c.compound.steps[p].kind in ("insert", "rework")
