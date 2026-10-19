"""Unit tests for the v0.1.6 efficiency score (no MuJoCo / player needed).

    pytest -q tests/test_efficiency_scoring.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))

from nist_sim.scoring import SUCCESS_WEIGHT, episode_score, raw_score_from_episodes  # noqa: E402

MAX_STEPS = 250
N = 640
# Platform takeover rule: challenger must beat the incumbent by this factor.
TAKEOVER = 1.01


def test_failure_is_zero():
    assert episode_score(False, None, MAX_STEPS) == 0.0
    assert episode_score(False, 10, MAX_STEPS) == 0.0


def test_success_bounds_and_monotonicity():
    at_limit = episode_score(True, MAX_STEPS, MAX_STEPS)
    early = episode_score(True, 5, MAX_STEPS)
    mid = episode_score(True, 100, MAX_STEPS)
    assert at_limit == pytest.approx(SUCCESS_WEIGHT)
    assert early == pytest.approx(SUCCESS_WEIGHT + (1 - SUCCESS_WEIGHT) * (MAX_STEPS - 5) / MAX_STEPS)
    assert SUCCESS_WEIGHT <= at_limit <= mid <= early <= 1.0
    assert early > mid > at_limit


def test_raw_score_mean_and_deterministic():
    rows = [
        {"success": True, "success_step": 50, "conditions": {"max_steps": MAX_STEPS}},
        {"success": False, "success_step": None, "conditions": {"max_steps": MAX_STEPS}},
        {"success": True, "success_step": 150, "conditions": {"max_steps": MAX_STEPS}},
    ]
    expected = (
        episode_score(True, 50, MAX_STEPS)
        + 0.0
        + episode_score(True, 150, MAX_STEPS)
    ) / 3
    assert raw_score_from_episodes(rows) == pytest.approx(expected)
    assert raw_score_from_episodes(rows) == raw_score_from_episodes(list(rows))


def test_more_successes_outrank_fewer_at_matched_speed():
    """Success stays dominant when speed is held fixed: +1 assembly beats any speed tie."""
    slow = [{"success": True, "success_step": 200, "conditions": {"max_steps": MAX_STEPS}}] * 100
    fewer = list(slow[:-1]) + [{"success": False, "success_step": None, "conditions": {"max_steps": MAX_STEPS}}]
    assert raw_score_from_episodes(slow) > raw_score_from_episodes(fewer)


def test_headroom_faster_perfect_beats_slower_perfect_by_more_than_1pct():
    """Two always-succeeding policies: finishing in 50 steps vs 150 steps.

    50 vs 150 is a meaningful speed gap inside the 10 s limit (2 s vs 6 s) and
    is slower than the reference baseline's typical ~20-step grasped successes,
    so it is a fair stand-in for hard starts (loose / lying) where policies still
    differ.
    """
    fast = [{"success": True, "success_step": 50, "conditions": {"max_steps": MAX_STEPS}}] * N
    slow = [{"success": True, "success_step": 150, "conditions": {"max_steps": MAX_STEPS}}] * N
    fast_s = raw_score_from_episodes(fast)
    slow_s = raw_score_from_episodes(slow)
    assert fast_s == pytest.approx(SUCCESS_WEIGHT + (1 - SUCCESS_WEIGHT) * (MAX_STEPS - 50) / MAX_STEPS)
    assert slow_s == pytest.approx(SUCCESS_WEIGHT + (1 - SUCCESS_WEIGHT) * (MAX_STEPS - 150) / MAX_STEPS)
    assert fast_s > slow_s * TAKEOVER


def test_headroom_above_v015_style_639_of_640():
    """A 639/640 king that only barely makes the time limit must not freeze the board.

    Under v0.1.5 that king scored 0.9984375 and no score could clear +1%.
    Under this formula its re-scored value sits well below the theoretical max,
    so a perfect, faster policy can take over.
    """
    king = (
        [{"success": True, "success_step": MAX_STEPS, "conditions": {"max_steps": MAX_STEPS}}] * 639
        + [{"success": False, "success_step": None, "conditions": {"max_steps": MAX_STEPS}}]
    )
    king_s = raw_score_from_episodes(king)
    theoretical_max = 1.0  # every episode succeeds on the earliest possible step budget
    # Even an always-succeeding mid-speed policy clears the 1% margin over that king.
    challenger = [{"success": True, "success_step": 100, "conditions": {"max_steps": MAX_STEPS}}] * N
    chall_s = raw_score_from_episodes(challenger)
    assert king_s == pytest.approx(639 * SUCCESS_WEIGHT / N)
    assert theoretical_max > king_s * TAKEOVER
    assert chall_s > king_s * TAKEOVER
    # And the absolute ceiling is far above what that re-scored king holds.
    assert theoretical_max - king_s > 0.15
