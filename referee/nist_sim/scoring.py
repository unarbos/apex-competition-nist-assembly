"""Per-peg and per-episode scores for NIST-lite multi-peg board (v0.2.0).

For an episode with K active pegs, each peg k scores independently:

- 0 if peg k is not seated in its own matching slot at episode end
  (wrong slot, knocked out, never seated, or episode faulted before seat).
- Otherwise ``0.75 + 0.25 * time_term``, where
  ``time_term = (1 - t_k / T_e)`` and ``t_k`` is the control step when the
  peg was first stably seated, ``T_e`` the episode horizon.

An optional small action-effort factor lives inside the 0.25 budget so a
violent policy cannot match a calm one at the same seat times, but success
always dominates: a seated peg is worth at least 0.75.

Episode score ``S_e = (1/K) * sum_k s_k``. Round ``raw_score`` is the mean
of ``S_e`` over the fixed episode count (failed / empty rows contribute 0).
"""

from __future__ import annotations

from typing import Any, Sequence

SUCCESS_WEIGHT = 0.75
TIME_WEIGHT = 0.25
# Share of the time/effort budget used by mean squared action (rest is time).
EFFORT_SHARE = 0.15


def peg_score(
    seated_at_end: bool,
    first_seat_step: int | None,
    horizon: int,
    mean_sq_action: float = 0.0,
) -> float:
    """Score one peg in [0, 1].

    Parameters
    ----------
    seated_at_end:
        True if the peg is in its own slot, upright, released, gripper clear
        at episode end.
    first_seat_step:
        Control step when the peg first became stably seated (1 .. horizon).
        Ignored when seated_at_end is False.
    horizon:
        Episode time limit in control steps (``8 s × K × 25 Hz``).
    mean_sq_action:
        Mean of ``sum(a_i^2)`` over control steps for this episode. Used only
        for a small effort discount inside the 0.25 budget.
    """
    if not seated_at_end:
        return 0.0
    if horizon <= 0:
        return float(SUCCESS_WEIGHT)
    if first_seat_step is None:
        return float(SUCCESS_WEIGHT)
    step = int(first_seat_step)
    if step < 0:
        step = 0
    if step > horizon:
        step = horizon
    time_term = (horizon - step) / horizon
    # mean_sq_action is in [0, ACT_DIM] for |a_i|<=1; normalise softly.
    effort_term = max(0.0, 1.0 - float(mean_sq_action) / 5.0)
    blend = (1.0 - EFFORT_SHARE) * time_term + EFFORT_SHARE * effort_term
    return float(SUCCESS_WEIGHT + TIME_WEIGHT * blend)


def episode_score(
    peg_seated: Sequence[bool],
    peg_first_seat_step: Sequence[int | None],
    horizon: int,
    mean_sq_action: float = 0.0,
) -> float:
    """Mean peg score for one episode (fixed K = len(peg_seated))."""
    k = len(peg_seated)
    if k == 0:
        return 0.0
    if len(peg_first_seat_step) != k:
        raise ValueError("peg_seated and peg_first_seat_step length mismatch")
    total = 0.0
    for seated, t in zip(peg_seated, peg_first_seat_step):
        total += peg_score(bool(seated), t, horizon, mean_sq_action)
    return total / k


def _row_horizon(row: dict[str, Any]) -> int:
    return int(row.get("conditions", {}).get("max_steps", row.get("horizon", 1)))


def raw_score_from_episodes(rows: list[dict[str, Any]]) -> float:
    """Mean episode score over a fixed-length row list."""
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        total += float(row.get("score", 0.0))
    return total / len(rows)


def score_row_from_episode_state(row_partial: dict[str, Any]) -> float:
    """Compute episode score from a summary row that already carries peg fields."""
    seated = row_partial.get("peg_seated_at_end") or []
    first = row_partial.get("peg_first_seat_step") or []
    return episode_score(
        seated,
        first,
        _row_horizon(row_partial),
        float(row_partial.get("mean_sq_action", 0.0)),
    )
