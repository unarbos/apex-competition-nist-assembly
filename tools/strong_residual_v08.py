"""Residual wrapper around the strong observation-only upright controller.

The caller owns physics stepping and supplies exactly one observation after each
step. The controller sees no simulator state; a zero residual reproduces the
standalone strong script action for action.
"""

from __future__ import annotations

import numpy as np

from obs_strong_v08 import ObservedPegView, V08ObsEpisode, _choices, rs


class StrongResidualController:
    def __init__(self, ep, noise_rng, initial_obs=None):
        self.view = V08ObsEpisode(ep, noise_rng, initial_obs=initial_obs)
        self.script = None
        self.i = None
        self.phase = "select"
        self.retreat = 0

    def _motion(self, method):
        search_steps = 140 if method == "pick_upright" else 160
        script = rs.ScriptedPeg(
            ObservedPegView(self.view, self.i),
            search_amp=0.006,
            search_steps=search_steps,
            swing_grip=-1.0,
        )
        return getattr(script, method)()

    def _goto_retreat(self, grip=1.0, max_n=25):
        script = rs.ScriptedPeg(ObservedPegView(self.view, self.i), swing_grip=-1.0)
        return script.goto([None, None, 0.095], grip=grip, max_n=max_n)

    def act(self):
        if self.script is not None:
            try:
                return next(self.script)
            except StopIteration:
                self.script = None
                if self.phase == "pick":
                    self.phase = "check"
                elif self.phase == "insert":
                    self.phase = "retreat"
                    self.script = iter(self._goto_retreat(max_n=25))
                    return next(self.script)
                elif self.phase == "lift_fail":
                    self.phase = "select"
                    self.i = None
                elif self.phase == "retreat":
                    self.phase = "select"
                    self.i = None

        if self.phase == "check":
            p, _ = self.view.peg_pose(self.i)
            if p[2] < 0.043:
                self.phase = "lift_fail"
                self.script = iter(self._goto_retreat(max_n=24))
                return next(self.script)
            self.phase = "insert"
            self.script = iter(self._motion("insert"))
            return next(self.script)

        if self.phase == "retreat":
            self.phase = "select"
            self.i = None

        choices = _choices(self.view, self.view.attempted)
        if not choices:
            return np.array([0.0, 0.0, 0.15, 0.0, 1.0])
        self.i = choices[0][1]
        self.view.attempted.add(self.i)
        self.phase = "pick"
        self.script = iter(self._motion("pick_upright"))
        return next(self.script)

    def post_step(self, done, obs=None, action=None):
        self.last_action = None if action is None else np.asarray(action, dtype=np.float64).copy()
        self.view.done = bool(done)
        if not done:
            self.view._update(obs=obs)
