"""Fast checks on round condition sampling (no player, no HTTP)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
from nist_sim import conditions  # noqa: E402
from nist_sim.reserve import reserved_from_secret  # noqa: E402


@pytest.fixture
def reserved(tmp_path, monkeypatch):
    spec = reserved_from_secret("test-only-secret")
    p = tmp_path / "reserved.json"
    p.write_text(json.dumps(spec))
    monkeypatch.setattr(conditions, "RESERVED_PATH", p)
    return spec["ranges"]


def test_same_seed_same_round():
    _, a = conditions.sample_episodes(5, 2)
    _, b = conditions.sample_episodes(5, 2)
    assert [c.public() for c in a] == [c.public() for c in b]


def test_rounds_differ():
    _, a = conditions.sample_episodes(5, 2)
    _, b = conditions.sample_episodes(6, 2)
    assert [c.public() for c in a] != [c.public() for c in b]


def test_strata_fixed():
    for seed in (1, 2, 3):
        _, eps = conditions.sample_episodes(seed, 4)
        # 4 K values × 2 clutter buckets × 4 = 32
        assert len(eps) == 8 * 4
        ks = [e.n_pegs for e in eps]
        assert set(ks) == {3, 4, 5, 6}
        assert abs(sum(ks) / len(ks) - 4.5) < 1e-9


def test_at_most_one_grasped():
    _, eps = conditions.sample_episodes(7, 8)
    for e in eps:
        grasped = sum(1 for p in e.pegs if p.start_type == "grasped")
        assert grasped <= 1


def test_live_rounds_avoid_reserved(reserved):
    for seed in range(5):
        _, eps = conditions.sample_episodes(seed, 8)
        for name, (a, b) in reserved.items():
            assert not any(a <= getattr(e, name) <= b for e in eps)


def test_reserved_split_inside(reserved):
    _, eps = conditions.sample_episodes(0, 8, "reserved")
    for name, (a, b) in reserved.items():
        assert all(a <= getattr(e, name) <= b for e in eps)


def test_reserved_split_needs_build_input(tmp_path, monkeypatch):
    monkeypatch.setattr(conditions, "RESERVED_PATH", tmp_path / "missing.json")
    with pytest.raises(RuntimeError):
        conditions.sample_episodes(0, 1, "reserved")


def test_ranges_extended_for_v020():
    # 0.05 mm square fits were unstable under MuJoCo soft contact; live lower
    # bound is 0.2 mm. Upper bound stays sub-millimetre for a usable gradient.
    lo, hi = conditions.RANGES["clearance"]
    assert lo >= 0.00015 - 1e-12
    assert hi <= 0.001 + 1e-12
    assert "obs_noise_rad" in conditions.RANGES
    assert "act_delay" in conditions.RANGES
    assert "disturbance" in conditions.RANGES
    assert "dropout_prob" in conditions.RANGES
