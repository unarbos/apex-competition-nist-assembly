"""Task profile switch for v0.6 development (``NIST_PROFILE``).

``v05`` (default) is the live v0.5.x task, bit-for-bit. ``v06`` is the v0.6 "Assembly II"
development profile; it is built up step by step on the ``v0.6-dev`` branch and is not used
by any released image. Code reads the active profile through :data:`ACTIVE`.

Only fields that are already wired into the simulator have an effect; the rest document the
v0.6 targets so calibration tooling can report them (see docs/v0.6/DESIGN.md).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class Profile:
    name: str
    # --- wired ---
    # Referee-side action quantization step (0 = off). Each action component is snapped to a
    # multiple of ``action_quant`` after clipping, so policies that behave the same up to tiny
    # numeric differences produce identical commands, trajectories and scores.
    action_quant: float = 0.0
    # --- v0.6 targets (not wired yet) ---
    n_slots: int = 8
    k_values: tuple[int, ...] = (3, 4, 5, 6)
    clearance: tuple[float, float] = (0.00025, 0.0008)
    steps_per_part: int = 280
    episodes_per_round: int = 128
    depth_patch: int = 0  # wrist height-map side length (0 = no depth obs)
    policy_state_dim: int = 0  # recurrent state carried by the referee (0 = memoryless)
    collars: bool = False
    held_out_families_per_round: int = 0

    def public(self) -> dict:
        return asdict(self)


PROFILES: dict[str, Profile] = {
    "v05": Profile(name="v05"),
    "v06": Profile(
        name="v06",
        action_quant=0.0,  # off until the stateful-policy protocol lands (rounding alone breaks memoryless policies)
        n_slots=12,
        k_values=(4, 6, 8),
        clearance=(0.00010, 0.00050),
        steps_per_part=240,
        episodes_per_round=96,
        depth_patch=16,
        policy_state_dim=256,
        collars=True,
        held_out_families_per_round=2,
    ),
}


def get(name: str | None = None) -> Profile:
    key = (name or os.environ.get("NIST_PROFILE") or "v05").strip().lower()
    if key not in PROFILES:
        raise ValueError(f"unknown NIST_PROFILE {key!r}; expected one of {sorted(PROFILES)}")
    p = PROFILES[key]
    # Dev-only override for calibration sweeps (never set in released images).
    q = os.environ.get("NIST_ACTION_QUANT")
    if name is None and q:
        p = replace(p, action_quant=float(q))
    return p


ACTIVE: Profile = get()


def quantize_action(a, step: float):
    """Snap each component of ``a`` (already clipped to [-1, 1]) to a multiple of ``step``."""
    import numpy as np

    if step <= 0.0:
        return a
    q = np.round(np.asarray(a, dtype=float) / step) * step
    return np.clip(q, -1.0, 1.0)
