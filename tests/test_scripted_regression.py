"""v0.5.0 generalization hardening: hand-coded controllers must stay well below the
learned / privileged range.

- ``ObsOnlyScripted`` (referee/nist_sim/policies.py) and the obs-only reference port
  (tools/obs_refport.py) are always checked.
- Additional scripted ONNX policies can be checked from a local directory given by
  ``NIST_SCRIPTED_FIXTURES`` (every ``*.onnx`` in it). These fixtures are kept
  outside the repository; the test is skipped when the variable is unset.

Each must score <= SCRIPTED_CEILING on a fixed 16-episode slice of the live mix.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tools"))

SCRIPTED_CEILING = 0.30
SEED, PER = 11, 2  # 16 episodes, all tiers


def _episode_score(kind: str, i: int) -> float:
    sys.path[:0] = [str(ROOT / "referee"), str(ROOT / "tools")]
    from nist_sim.conditions import episode_seed_sequence, sample_episodes
    from nist_sim.env import Episode
    from nist_sim.policies import ObsOnlyScripted
    from nist_sim.scoring import episode_score

    _, conds = sample_episodes(SEED, PER)
    c = conds[i]
    r1, r2 = (np.random.default_rng(s) for s in episode_seed_sequence(SEED, i).spawn(2))
    e = Episode(c, r1)
    if kind == "refport":
        from obs_refport import solve_obs_episode

        return float(solve_obs_episode(e, r2)["score"])
    if kind == "obs_scripted":
        pol = ObsOnlyScripted()
        act = pol.act
    else:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        sess = ort.InferenceSession(kind, so, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name

        def act(o):
            x = np.round(np.asarray(o, np.float64), 6).astype(np.float32)[None]
            a = np.asarray(sess.run(None, {name: x})[0], dtype=np.float64).reshape(-1)[:5]
            return a if a.shape == (5,) and np.all(np.isfinite(a)) else np.zeros(5)

    while not e.done:
        e.step(np.clip(act(e.observe(r2)), -1, 1))
    return float(
        episode_score(e.peg_seated_at_end(), e.peg_first_seat_steps(), c.max_steps, e.action_l2 / max(1, e.steps))
    )


def _mean_score(kind: str) -> float:
    n = 8 * PER
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as ex:
        return float(np.mean(list(ex.map(_episode_score, [kind] * n, range(n)))))


@pytest.mark.parametrize("kind", ["obs_scripted", "refport"])
def test_builtin_scripted_controllers_stay_low(kind):
    s = _mean_score(kind)
    assert s <= SCRIPTED_CEILING, f"{kind} scored {s:.3f} > {SCRIPTED_CEILING}"


def _fixture_policies() -> list[str]:
    d = os.environ.get("NIST_SCRIPTED_FIXTURES")
    if not d or not Path(d).is_dir():
        return []
    return sorted(str(p) for p in Path(d).glob("*.onnx"))


@pytest.mark.skipif(not _fixture_policies(), reason="NIST_SCRIPTED_FIXTURES not set (private fixtures)")
@pytest.mark.parametrize(
    "path", _fixture_policies() or ["none"], ids=[f"fixture{i}" for i in range(max(1, len(_fixture_policies())))]
)
def test_fixture_scripted_policies_stay_low(path):
    s = _mean_score(path)
    assert s <= SCRIPTED_CEILING, f"fixture scored {s:.3f} > {SCRIPTED_CEILING}"
