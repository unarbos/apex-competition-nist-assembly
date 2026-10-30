"""Miner-visible records must not carry hidden episode conditions.

Per-episode records (history/episodes.jsonl, traj_*.json and result metadata) are
published after a round in the ``nist_lite_record/2`` shape the replay viewer renders.
Board pose and clutter start poses are exact (not held-out axes). ``clearance`` and
``chamfer`` are coarse stand-ins (clearance-class midpoint, fixed nominal chamfer).
Records must not contain estimate-bias values, true peg start poses, or exact values
of the continuous condition axes (which over many rounds would expose the held-out
sub-ranges as gaps). The round RNG also mixes in a private
per-build salt so a round cannot be reproduced from the platform SEED alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
from nist_sim import conditions  # noqa: E402
from nist_sim.reserve import reserved_from_secret  # noqa: E402

ALLOWED_TOP = {
    "n_pegs", "k", "clutter_bucket", "n_clutter", "difficulty_tier", "clearance_class",
    "clearance", "chamfer", "board_xy", "board_yaw",
    "disturbance", "max_steps", "pegs", "clutter", "slots",
}
ALLOWED_PEG = {"slot", "start_type", "drawn_start_type", "shape", "size"}
ALLOWED_SLOT = {"shape", "size", "xy", "yaw", "params"}
ALLOWED_CLUTTER = {"offset_xy", "yaw", "size_xy", "height"}
COARSE_CLEARANCE = {"tight": 0.000325, "mid": 0.0005, "loose": 0.0007}


def _all_floats(obj):
    if isinstance(obj, float):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _all_floats(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _all_floats(v)


def test_public_conditions_whitelist():
    _, eps = conditions.sample_episodes(0, 4)
    for c in eps:
        d = c.public()
        json.dumps(d)
        assert set(d) == ALLOWED_TOP
        for p in d["pegs"]:
            assert set(p) == ALLOWED_PEG
        for cl in d["clutter"]:
            assert set(cl) == ALLOWED_CLUTTER
        assert len(d["slots"]) == conditions.N_SLOTS
        for sl in d["slots"]:
            assert set(sl) == ALLOWED_SLOT


def test_public_conditions_carry_no_hidden_values():
    """No published number equals any hidden per-episode value."""
    _, eps = conditions.sample_episodes(1, 4)
    for c in eps:
        # Board pose and clutter start poses are intentionally exact (not held-out).
        hidden = [
            *c.pose_bias_xy, c.pose_bias_yaw, *c.peg_bias_delta_xy,
            c.clearance, c.friction, c.peg_mass, c.gain_scale, c.obs_noise, c.obs_noise_rad,
            c.dropout_prob, c.bias_drift, c.backlash, c.chamfer, c.search_pitch,
        ]
        hidden += [x for b in c.slot_bias_list for x in b]
        hidden += [x for b in c.peg_bias_delta_list for x in b]
        hidden += [x for p in c.pegs for x in (*p.offset_xy, p.height, p.yaw_err)]
        hidden += [x for b in c.clutter_bias_list for x in b]
        hidden += list(c.slot_bias_yaw_list)
        hidden += list(c.peg_yaw_delta_list)
        hidden = [h for h in hidden if abs(h) > 1e-9]
        for v in _all_floats(c.public()):
            assert not any(abs(v - h) < 1e-7 for h in hidden), v


def test_records_written_by_referee_are_whitelisted(tmp_path, monkeypatch):
    """End-to-end: run the referee worker summary path and check rows."""
    import referee as ref  # referee/referee.py

    _, conds = conditions.sample_episodes(2, 1)
    for c in conds:
        assert set(c.public()) == ALLOWED_TOP
    assert ref.RECORD_FORMAT == "nist_lite_record/2"


def test_coarse_clearance_and_chamfer():
    """clearance = class midpoint, chamfer = fixed nominal; only 3+1 distinct values."""
    seen_clr, seen_ch = set(), set()
    for seed in range(3):
        _, eps = conditions.sample_episodes(seed, 8)
        for c in eps:
            d = c.public()
            assert d["clearance"] == COARSE_CLEARANCE[d["clearance_class"]]
            assert d["chamfer"] == 0.5
            seen_clr.add(d["clearance"])
            seen_ch.add(d["chamfer"])
    assert seen_clr <= set(COARSE_CLEARANCE.values())
    assert seen_ch == {0.5}


def test_coarse_values_inside_class_and_range():
    lo, hi = conditions.RANGES["clearance"]
    for v in COARSE_CLEARANCE.values():
        assert lo < v < hi
    for name, v in COARSE_CLEARANCE.items():
        assert conditions.clearance_class(v) == name
    clo, chi = conditions.RANGES["chamfer"]
    assert clo < conditions.NOMINAL_CHAMFER < chi


def test_coarse_values_independent_of_reserved_ranges(tmp_path, monkeypatch):
    """Published clearance/chamfer depend only on the class, never on the build secret."""
    outs = []
    for secret in ("secret-a", "secret-b", None):
        p = tmp_path / f"r-{secret}.json"
        if secret:
            p.write_text(json.dumps(reserved_from_secret(secret)))
        monkeypatch.setattr(conditions, "RESERVED_PATH", p)
        _, eps = conditions.sample_episodes(3, 4)
        outs.append({(c.public()["clearance_class"], c.public()["clearance"], c.public()["chamfer"]) for c in eps})
    allowed = {(k, v, 0.5) for k, v in COARSE_CLEARANCE.items()}
    for o in outs:
        assert o <= allowed


def test_exact_board_and_clutter_pose_for_viewer():
    _, eps = conditions.sample_episodes(4, 2)
    for c in eps:
        d = c.public()
        assert d["board_xy"] == [round(c.board_xy[0], 6), round(c.board_xy[1], 6)]
        assert d["board_yaw"] == round(c.board_yaw, 6)
        assert len(d["clutter"]) == len(c.clutter) == d["n_clutter"]
        for pub, cl in zip(d["clutter"], c.clutter):
            assert pub["offset_xy"] == [round(x, 6) for x in cl.offset_xy]
            assert pub["yaw"] == round(cl.yaw, 6)
            assert pub["size_xy"] == [round(x, 6) for x in cl.size_xy]
            assert pub["height"] == round(cl.height, 6)
            assert all(np.isfinite(x) for x in (*pub["offset_xy"], pub["yaw"], *pub["size_xy"], pub["height"]))


def test_no_salt_or_reserved_in_public(salted):
    reserved = json.loads(salted.read_text())
    _, eps = conditions.sample_episodes(6, 2)
    blob = json.dumps([c.public() for c in eps])
    assert str(reserved["salt"]) not in blob
    pub_floats = [v for c in eps for v in _all_floats(c.public())]
    for a, b in reserved["ranges"].values():
        assert not any(abs(v - a) < 1e-9 or abs(v - b) < 1e-9 for v in pub_floats)


@pytest.fixture
def salted(tmp_path, monkeypatch):
    p = tmp_path / "reserved.json"
    p.write_text(json.dumps(reserved_from_secret("test-only-secret")))
    monkeypatch.setattr(conditions, "RESERVED_PATH", p)
    return p


def _fingerprint(eps):
    return [(c.board_xy, c.friction, tuple(p.offset_xy for p in c.pegs)) for c in eps]


def test_dev_build_has_no_salt(tmp_path, monkeypatch):
    monkeypatch.setattr(conditions, "RESERVED_PATH", tmp_path / "missing.json")
    assert conditions.build_salt() == []


def test_build_salt_changes_rounds(salted, monkeypatch, tmp_path):
    _, a = conditions.sample_episodes(5, 1)
    ss_a = conditions.episode_seed_sequence(5, 0).generate_state(4)
    monkeypatch.setattr(conditions, "RESERVED_PATH", tmp_path / "missing.json")
    _, b = conditions.sample_episodes(5, 1)
    ss_b = conditions.episode_seed_sequence(5, 0).generate_state(4)
    assert _fingerprint(a) != _fingerprint(b)
    assert not np.array_equal(ss_a, ss_b)


def test_salted_rounds_still_deterministic(salted):
    _, a = conditions.sample_episodes(5, 1)
    _, b = conditions.sample_episodes(5, 1)
    assert _fingerprint(a) == _fingerprint(b)


def test_salt_independent_of_ranges():
    r1 = reserved_from_secret("secret-one")
    r2 = reserved_from_secret("secret-two")
    assert r1["salt"] != r2["salt"]
    assert isinstance(r1["salt"], int) and r1["salt"] > 0
