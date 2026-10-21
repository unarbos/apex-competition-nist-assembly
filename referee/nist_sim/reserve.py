"""Build-time helper: derive reserved sub-ranges from a build secret.

    python -m nist_sim.reserve <secret> > nist_sim/reserved.json
"""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

from .conditions import RANGES


def reserved_from_secret(secret: str, n_axes: int = 2, width: float = 0.15) -> dict:
    seed = int.from_bytes(hashlib.sha256(secret.strip().encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    axes = sorted(rng.choice(sorted(RANGES), size=n_axes, replace=False).tolist())
    out = {}
    for name in axes:
        lo, hi = RANGES[name]
        span = hi - lo
        a = lo + float(rng.uniform(0.05, 0.95 - width)) * span
        out[name] = [round(a, 7), round(a + width * span, 7)]
    # Independent per-build salt for the round RNG (different hash domain from the ranges).
    salt = int.from_bytes(hashlib.sha256(b"nist-round-salt:" + secret.strip().encode()).digest()[:8], "little")
    return {"version": 2, "ranges": out, "salt": salt}


if __name__ == "__main__":
    json.dump(reserved_from_secret(sys.argv[1]), sys.stdout)
