"""v0.6-dev profile scaffolding: live profile unchanged, action quantization deterministic.

    pytest -q tests/test_v06_profile.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tools"))
from nist_sim import profile  # noqa: E402
from nist_sim.conditions import episode_seed_sequence, sample_episodes  # noqa: E402
from nist_sim.env import Episode  # noqa: E402


def test_named_profiles_and_quantize_identity():
    p = profile.get("v05")
    assert p.action_quant == 0.0 and profile.get("v06").action_quant == 0.0
    assert profile.get("v07").name == "v07" and profile.get("v07").physics_dt == 0.002
    assert profile.ACTIVE.name in profile.PROFILES
    a = np.array([0.1234567, -0.5, 1.0, -1.0, 0.0])
    assert profile.quantize_action(a, 0.0) is a


def test_release_build_ignores_nist_dev_overrides(tmp_path, monkeypatch):
    """When reserved.json is present (release image), NIST_PHYSICS_DT / NIST_ACTION_QUANT are ignored."""
    reserved = tmp_path / "reserved.json"
    reserved.write_text('{"salt": 1, "ranges": {}}')
    monkeypatch.setattr(profile, "_RESERVED_PATH", reserved)
    monkeypatch.setenv("NIST_PHYSICS_DT", "0.001")
    monkeypatch.setenv("NIST_ACTION_QUANT", "0.05")
    p = profile.get()  # ACTIVE name from env; overrides must not apply
    assert p.physics_dt == profile.PROFILES[p.name].physics_dt
    assert p.action_quant == profile.PROFILES[p.name].action_quant
    # Dev path (no reserved file) still honours overrides.
    monkeypatch.setattr(profile, "_RESERVED_PATH", tmp_path / "missing.json")
    p_dev = profile.get()
    assert p_dev.physics_dt == 0.001
    assert p_dev.action_quant == 0.05


def test_quantize_snaps_and_clips():
    q = profile.quantize_action(np.array([0.01, 0.03, -0.99, 0.999, -0.5]), 1 / 32)
    assert np.allclose(q * 32, np.round(q * 32))
    assert np.all(np.abs(q) <= 1.0)


def _rollout(quant: float, eps: float, steps: int = 60):
    _, conds = sample_episodes(3, 1, "live")
    r1, r2 = (np.random.default_rng(s) for s in episode_seed_sequence(3, 0).spawn(2))
    e = Episode(conds[0], r1)
    e.action_quant = quant
    rng = np.random.default_rng(0)
    for _ in range(steps):
        if e.done:
            break
        e.observe(r2)
        a = np.clip(rng.uniform(-0.8, 0.8, 5) + eps, -1, 1)
        e.step(a)
    return e.data.qpos.copy()


def test_quantization_makes_epsilon_copies_identical():
    # Tiny action perturbation: different trajectory without quantization, identical with it.
    base_q = _rollout(1 / 32, 0.0)
    copy_q = _rollout(1 / 32, 1e-6)
    assert np.array_equal(base_q, copy_q)
    assert not np.array_equal(_rollout(0.0, 0.0), _rollout(0.0, 1e-6))
