"""Fail-closed observation boundary for observation-only policy evaluation.

The wrapped simulator is deliberately inaccessible to controller code. Only the
initial/step observations, the action method, and terminal notification cross
this boundary. Scoring remains outside in the trusted evaluator.
"""

from __future__ import annotations

import numpy as np


class ObservationBoundary:
    def __init__(self, episode):
        self._episode = episode
        self.done = False

    def observe(self, rng):
        # One conversion rule for standalone scripts and residual evaluation.
        return self._episode.observe(rng).astype(np.float64)

    def step(self, action):
        self._episode.step(action)
        self.done = self._episode.done

    def __getattr__(self, name):
        raise AssertionError(f"privileged episode read: {name}")
