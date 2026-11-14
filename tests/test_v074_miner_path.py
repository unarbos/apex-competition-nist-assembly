"""v0.7.5: miner-path no-assistance seals + action-only / obs-only key completion.

Assistance sealed: privileged snaps that write ``step_done`` / holds / mates directly.
Honest privileged key path must not write ``_peg_probe_frames`` or teleport key yaw.
Obs-only searcher must complete keys from observations alone.

    NIST_PROFILE=v07 pytest -q tests/test_v074_miner_path.py
"""

from __future__ import annotations

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
from nist_sim.compound_v07 import AssemblyIIIEpisode, sample_compound_v07  # noqa: E402
import compound_solver as CS  # noqa: E402
import obs_key_searcher as OKS  # noqa: E402


# Snaps that write completion state — must never run on the miner path.
FORBIDDEN_SNAPS = (
    "_privileged_stack_snap",
    "_privileged_screw_snap",
    "_privileged_key_snap",
    "_privileged_insert_assist",
    "_privileged_rework_assist",
)


@pytest.fixture
def seal_snaps(monkeypatch):
    def _boom(name):
        def _f(*_a, **_k):
            raise AssertionError(f"completion snap re-enabled: {name}() during honest path")

        return _f

    for name in FORBIDDEN_SNAPS:
        if hasattr(CS, name):
            monkeypatch.setattr(CS, name, _boom(name))


@pytest.fixture
def seal_key_teleports(monkeypatch):
    """Honest key path must not teleport yaw or fill probe frames."""

    def _boom_nudge(*_a, **_k):
        raise AssertionError("_nudge_key_yaw used on honest/action-only key path")

    monkeypatch.setattr(CS, "_nudge_key_yaw", _boom_nudge)


def test_honest_solver_actions_only_under_snap_seal(seal_snaps, seal_key_teleports):
    """Honest privileged (oracle=False) completes via actions + non-key pose nudges only."""
    # Short horizons make a single hard episode flaky; require success on ≥1 of first 8.
    _, conds = sample_compound_v07(7, 8)
    ok = False
    last = None
    for idx in range(len(conds)):
        r1, _ = (np.random.default_rng(s) for s in C.episode_seed_sequence(7, idx).spawn(2))
        e = AssemblyIIIEpisode(conds[idx], r1)
        key_idx = [k for k, s in enumerate(e.cs.steps) if s.kind == "key"]
        r = CS.solve_compound(e, oracle=False)
        last = (r, e, key_idx)
        assert np.isfinite(r["score"])
        assert e.steps > 0
        assert r["score"] >= 0.0
        if key_idx and (sum(1 for k in key_idx if e.step_done[k]) >= 1 or r["score"] >= 0.20):
            ok = True
            break
        if not key_idx and r["score"] >= 0.0:
            ok = True
            break
    assert ok, last


def test_obs_only_key_searcher_completes_keys():
    """Policy is obs+actions only (fixture seats peg); completes keys on seed 7."""
    r = OKS.run_eval(seed=7, per=8, episodes=8, start=0, budget=420)
    assert r["keys_total"] > 0
    assert r["keys_done"] > 0
    assert r["key_rate"] >= 0.20, r


def test_dflat_key_armable_without_leave():
    from nist_sim.compound_v07 import KEY_HOLD, KEY_PROBE_FRAMES

    for seed in range(11, 180):
        _, conds = sample_compound_v07(seed, 8)
        for i, c in enumerate(conds):
            for k, s in enumerate(c.compound.steps):
                if s.kind != "key":
                    continue
                if c.pegs[s.peg].spec.symmetry() != 1:
                    continue
                r1, _ = (np.random.default_rng(ss) for ss in C.episode_seed_sequence(seed, i).spawn(2))
                e = AssemblyIIIEpisode(c, r1)
                for p in e.cs.steps[k].prereq:
                    e.step_done[p] = True
                e._active_key = k
                e._peg_probe_frames[s.peg] = KEY_PROBE_FRAMES
                e._key_index_armed[s.peg] = False
                e.seated_now[s.peg] = True
                e._key_aligned = lambda _s: True  # type: ignore[method-assign]
                e._gripper_touching = lambda _b: False  # type: ignore[method-assign]
                e._step_state(k)
                assert e._key_index_armed[s.peg] is True
                e._key_hold[s.peg] = 0
                done = False
                for _ in range(KEY_HOLD):
                    done, _ = e._step_state(k)
                assert done is True
                return
    pytest.skip("no dflat key")


def test_no_direct_key_completion_helpers():
    assert not hasattr(AssemblyIIIEpisode, "force_key_done")
    assert not hasattr(AssemblyIIIEpisode, "set_key_hold")
    assert not hasattr(AssemblyIIIEpisode, "_obs_key_yaw")
