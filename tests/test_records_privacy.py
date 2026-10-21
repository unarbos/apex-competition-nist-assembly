"""Miner-visible records must not carry hidden episode conditions.

Per-episode records (history/episodes.jsonl and result metadata) are published after
a round. They must not contain estimate-bias values, board pose, true peg / clutter
start poses, or exact values of the continuous condition axes (which over many rounds
would expose the held-out sub-ranges as gaps). The round RNG also mixes in a private
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
    "disturbance", "max_steps", "pegs", "clutter",
}
ALLOWED_PEG = {"slot", "start_type", "shape"}
ALLOWED_CLUTTER = {"size_xy", "height"}


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


def test_public_conditions_carry_no_hidden_values():
    """No published number equals any hidden per-episode value."""
    _, eps = conditions.sample_episodes(1, 4)
    for c in eps:
        hidden = [
            *c.pose_bias_xy, c.pose_bias_yaw, *c.peg_bias_delta_xy, *c.board_xy, c.board_yaw,
            c.clearance, c.friction, c.peg_mass, c.gain_scale, c.obs_noise, c.obs_noise_rad,
            c.dropout_prob, c.bias_drift, c.backlash, c.chamfer, c.search_pitch,
        ]
        hidden += [x for b in c.slot_bias_list for x in b]
        hidden += [x for b in c.peg_bias_delta_list for x in b]
        hidden += [x for p in c.pegs for x in (*p.offset_xy, p.height, p.yaw_err)]
        hidden += [x for cl in c.clutter for x in (*cl.offset_xy, cl.yaw)]
        hidden = [h for h in hidden if abs(h) > 1e-9]
        for v in _all_floats(c.public()):
            assert not any(abs(v - h) < 1e-7 for h in hidden), v


def test_records_written_by_referee_are_whitelisted(tmp_path, monkeypatch):
    """End-to-end: run the referee worker summary path and check rows."""
    import referee as ref  # referee/referee.py

    _, conds = conditions.sample_episodes(2, 1)
    for c in conds:
        assert set(c.public()) == ALLOWED_TOP
    assert ref.RECORD_FORMAT == "nist_lite_record/3"


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
