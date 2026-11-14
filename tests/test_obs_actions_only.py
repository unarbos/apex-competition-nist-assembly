"""Actions-only obs key searcher must not touch private env state.

Miner-path contract: observe → action → step only. No ``_set_free`` / fixture seat /
probe-frame fills / ``step_done`` teleports from the searcher path.

    NIST_PROFILE=v07 pytest -q tests/test_obs_actions_only.py
"""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tools"))

os.environ["NIST_PROFILE"] = "v07"

from nist_sim.compound_v07 import AssemblyIIIEpisode  # noqa: E402
import obs_key_searcher as OKS  # noqa: E402


# Policy must never touch these. Eval harness may read step_done after the episode
# for calibration counts only (not during the observe→act→step loop).
POLICY_FORBIDDEN = (
    "_set_free",
    "_peg_probe_frames",
    "step_done",
    "fixture_seat",
    "qpos[",
    "peg_qadr",
    "key_yaw",  # true-state yaw — actions-only must hill-climb from obs
)
EVAL_FORBIDDEN = (
    "_set_free",
    "_peg_probe_frames",
    "fixture_seat",
    "qpos[",
    "peg_qadr",
    "_nudge_",
)


def test_actions_only_policy_source_has_no_private_writes():
    """Static seal: actions-only policy must not reference private env APIs."""
    src = inspect.getsource(OKS.ActionsOnlyKeyPolicy)
    for tok in POLICY_FORBIDDEN:
        assert tok not in src, f"actions-only policy references private API {tok!r}"
    eval_src = inspect.getsource(OKS.run_eval_actions_only)
    for tok in EVAL_FORBIDDEN:
        assert tok not in eval_src, f"actions-only eval references private API {tok!r}"
    # Loop body is observe → act → step only (no fixture).
    assert "ep.observe" in eval_src and "ep.step" in eval_src
    assert "fixture_seat_key_peg" not in eval_src


def test_actions_only_eval_sealed_set_free_and_fixture(monkeypatch):
    """Runtime seal: non-env callers of ``_set_free`` raise; fixture must not run."""

    real_set_free = AssemblyIIIEpisode._set_free

    def sealed_set_free(self, *a, **k):
        # Env-internal (nist_sim.*) may teleport for waves/parking; searcher/tools must not.
        frame = sys._getframe(1)
        mod = frame.f_globals.get("__name__", "")
        if not str(mod).startswith("nist_sim"):
            raise AssertionError(f"_set_free called from non-env module: {mod}")
        return real_set_free(self, *a, **k)

    monkeypatch.setattr(AssemblyIIIEpisode, "_set_free", sealed_set_free)

    def boom_fixture(*_a, **_k):
        raise AssertionError("fixture_seat_key_peg used on actions-only path")

    monkeypatch.setattr(OKS, "fixture_seat_key_peg", boom_fixture)

    # Short run keeps CI light; seal must hold for the whole loop.
    r = OKS.run_eval_actions_only(seed=7, per=8, episodes=2, start=0)
    assert r["path"] == "actions_only"
    assert r["episodes"] == 2
    assert r["keys_total"] >= 0
    assert "score_mean" in r


def test_fixture_path_still_uses_privileged_helpers():
    """Documented contrast: fixture path *does* call private seat/probe setup."""
    src = inspect.getsource(OKS.fixture_seat_key_peg)
    assert "_set_free" in src
    assert "_peg_probe_frames" in src
    assert "step_done" in src
