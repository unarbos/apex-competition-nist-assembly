"""Every settled multi-peg start must be stable under a no-op for the first steps.

    pytest -q tests/test_spawn_stability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
from nist_sim.conditions import sample_episodes, strata  # noqa: E402
from nist_sim.env import Episode  # noqa: E402

NOOP = np.zeros(5)
SCAN_SEEDS = range(10)
EARLY_STEPS = 5


def test_no_early_unstable_or_part_lost_under_noop():
    failures: list[tuple] = []
    for seed in SCAN_SEEDS:
        _, conds = sample_episodes(seed, 8)
        assert len(conds) == 64
        for i, c in enumerate(conds):
            ss = np.random.SeedSequence([seed, 0x5245, i])
            r1 = np.random.default_rng(ss.spawn(2)[0])
            e = Episode(c, r1)
            for _ in range(EARLY_STEPS):
                e.step(NOOP)
                if e.done:
                    if e.terminal_reason in ("unstable", "part_lost"):
                        failures.append(
                            (seed, i, e.terminal_reason, c.n_pegs, [p.start_type for p in c.pegs])
                        )
                    break
    assert not failures, f"{len(failures)} early spawn failures, e.g. {failures[:5]}"


def test_spawn_validation_keeps_stratum_counts():
    for seed in (0, 1, 7, 11):
        _, conds = sample_episodes(seed, 8)
        for i, c in enumerate(conds):
            ss = np.random.SeedSequence([seed, 0x5245, i])
            Episode(c, np.random.default_rng(ss.spawn(2)[0]))
        counts: dict[tuple, int] = {}
        for c in conds:
            key = (c.n_pegs, c.clutter_bucket)
            counts[key] = counts.get(key, 0) + 1
        assert len(counts) == len(strata())
        assert all(v == 8 for v in counts.values())


def test_spawn_reset_is_deterministic():
    seed, idx = 11, 10
    _, conds_a = sample_episodes(seed, 8)
    _, conds_b = sample_episodes(seed, 8)

    def build(conds):
        ss = np.random.SeedSequence([seed, 0x5245, idx])
        e = Episode(conds[idx], np.random.default_rng(ss.spawn(2)[0]))
        return e.data.qpos.copy(), [p.offset_xy for p in e.c.pegs]

    qa, off_a = build(conds_a)
    qb, off_b = build(conds_b)
    assert np.allclose(qa, qb)
    assert off_a == off_b
