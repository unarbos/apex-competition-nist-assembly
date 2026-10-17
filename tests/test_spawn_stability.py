"""Every settled episode start must be stable under a no-op policy for the first steps.

    pytest -q tests/test_spawn_stability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
from nist_sim.conditions import RESET_TYPES, sample_episodes  # noqa: E402
from nist_sim.env import Episode  # noqa: E402

NOOP = np.zeros(5)
# Seeds and first-step horizon that previously showed ~3-4 blow-ups per round.
SCAN_SEEDS = range(20)
EARLY_STEPS = 5


def test_no_early_unstable_or_part_lost_under_noop():
    failures: list[tuple] = []
    for seed in SCAN_SEEDS:
        _, conds = sample_episodes(seed, 16)
        assert len(conds) == 640
        for i, c in enumerate(conds):
            ss = np.random.SeedSequence([seed, 0x5245, i])
            r1 = np.random.default_rng(ss.spawn(2)[0])
            e = Episode(c, r1)
            for _ in range(EARLY_STEPS):
                e.step(NOOP)
                if e.done:
                    if e.terminal_reason in ("unstable", "part_lost"):
                        failures.append(
                            (seed, i, e.terminal_reason, c.reset_type, c.shape, c.diameter)
                        )
                    break
    assert not failures, f"{len(failures)} early spawn failures, e.g. {failures[:5]}"


def test_spawn_validation_keeps_stratum_counts():
    """Re-sampling offsets must not change shape x diameter x reset_type counts."""
    for seed in (0, 1, 7, 11):
        _, conds = sample_episodes(seed, 16)
        # Build episodes (may re-sample offsets) then check strata on the condition objects.
        for i, c in enumerate(conds):
            ss = np.random.SeedSequence([seed, 0x5245, i])
            Episode(c, np.random.default_rng(ss.spawn(2)[0]))
        counts: dict[tuple, int] = {}
        for c in conds:
            key = (c.shape, round(c.diameter * 1000), c.reset_type)
            counts[key] = counts.get(key, 0) + 1
        assert len(counts) == 2 * 4 * len(RESET_TYPES)
        assert all(v == 16 for v in counts.values())


def test_spawn_reset_is_deterministic():
    """Same platform seed and episode index → identical start state after validation."""
    seed, idx = 11, 310  # previously an unstable lying 16 mm spawn on seed 0-like draws
    _, conds_a = sample_episodes(seed, 16)
    _, conds_b = sample_episodes(seed, 16)

    def build(conds):
        ss = np.random.SeedSequence([seed, 0x5245, idx])
        e = Episode(conds[idx], np.random.default_rng(ss.spawn(2)[0]))
        return e.data.qpos.copy(), e.c.offset_xy, e.c.yaw_err

    qa, off_a, yaw_a = build(conds_a)
    qb, off_b, yaw_b = build(conds_b)
    assert np.allclose(qa, qb)
    assert off_a == off_b and yaw_a == yaw_b
