"""Per-episode and round scores for NIST-lite.

A failed episode is always 0. A successful episode scores in
[SUCCESS_WEIGHT, 1]: the floor is a flat reward for assembling, and the
remainder grows as the part is seated in fewer control steps.

raw_score is the mean episode score over the fixed episode count (higher
is better, always in [0, 1]). Success stays the dominant term: each
assembly is worth at least SUCCESS_WEIGHT, so policies that assemble
substantially more parts still outscore policies that assemble fewer
unless the gap is tiny and the fewer-success policy is much faster.
"""

from __future__ import annotations

from typing import Any

# Floor credit for a successful assembly. The leftover (1 - SUCCESS_WEIGHT) is
# the efficiency budget. 0.8 leaves enough budget that a meaningfully faster
# always-succeeding policy clears the platform's 1% takeover margin over a
# slower one, while keeping success ~4× larger than the max speed bonus.
SUCCESS_WEIGHT = 0.8

DEFAULT_MAX_STEPS = 250


def episode_score(success: bool, success_step: int | None, max_steps: int) -> float:
    """Score one episode in [0, 1].

    Parameters
    ----------
    success:
        True if the part was assembled (seated, upright, gripper clear).
    success_step:
        Control step index when success was declared (1 .. max_steps).
        Ignored when success is False.
    max_steps:
        Episode time limit in control steps (10 s × 25 Hz = 250 today).
    """
    if not success:
        return 0.0
    if max_steps <= 0:
        return float(SUCCESS_WEIGHT)
    # Missing step on a success should not happen; give the floor only.
    if success_step is None:
        return float(SUCCESS_WEIGHT)
    step = int(success_step)
    if step < 0:
        step = 0
    if step > max_steps:
        step = max_steps
    speed = (max_steps - step) / max_steps
    return float(SUCCESS_WEIGHT + (1.0 - SUCCESS_WEIGHT) * speed)


def _row_max_steps(row: dict[str, Any]) -> int:
    return int(row.get("conditions", {}).get("max_steps", DEFAULT_MAX_STEPS))


def raw_score_from_episodes(rows: list[dict[str, Any]]) -> float:
    """Mean episode score over a fixed-length row list (failed → 0)."""
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        total += episode_score(
            bool(row["success"]),
            row.get("success_step"),
            _row_max_steps(row),
        )
    return total / len(rows)
