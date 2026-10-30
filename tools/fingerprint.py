"""Action fingerprints for near-duplicate ONNX policies.

Two submissions that differ only by a negligible change (an added constant, a
re-export, graph wrapping) give nearly the same action on almost every
observation, even though their full episodes diverge in a contact-rich sim.
This tool compares policies on a fixed probe set of observations:

    python tools/fingerprint.py --probe-seed 12345 a.onnx b.onnx [c.onnx ...]

The probe set is built from simulator rollouts of the built-in scripted
controller on episodes drawn from ``--probe-seed``. A platform should keep its
probe seed private so the probe set can't be targeted. Output: one JSON line per
pair with ``match`` = share of probe observations whose actions agree within
``--atol`` (max abs difference over the 5 action dims), plus mean abs difference.
A match of about 0.95 or more means "same policy for crown purposes".
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))


def probe_observations(seed: int, n_episodes: int = 16, every: int = 7) -> np.ndarray:
    """Observations visited by the built-in scripted controller (plus action noise)."""
    from nist_sim.conditions import episode_seed_sequence, sample_episodes
    from nist_sim.env import Episode
    from nist_sim.policies import ObsOnlyScripted

    _, conds = sample_episodes(seed, max(1, n_episodes // 8))
    out = []
    for i, c in enumerate(conds[:n_episodes]):
        r1, r2 = (np.random.default_rng(s) for s in episode_seed_sequence(seed, i).spawn(2))
        e, pol = Episode(c, r1), ObsOnlyScripted()
        jitter = np.random.default_rng([seed, i, 7])
        while not e.done:
            o = e.observe(r2)
            if e.steps % every == 0:
                out.append(np.round(o, 6))
            a = np.asarray(pol.act(np.round(o, 6).astype(np.float32)), dtype=float)
            e.step(np.clip(a + jitter.normal(0, 0.3, 5), -1, 1))
    return np.asarray(out, dtype=np.float32)


def policy_actions(path: str, obs: np.ndarray, batch: int = 256) -> np.ndarray:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    acts = []
    for k in range(0, len(obs), batch):
        x = obs[k : k + batch]
        try:
            a = np.asarray(sess.run(None, {name: x})[0], dtype=np.float64).reshape(len(x), -1)[:, :5]
        except Exception:  # some exports only accept batch 1
            a = np.stack(
                [np.asarray(sess.run(None, {name: r[None]})[0], dtype=np.float64).reshape(-1)[:5] for r in x]
            )
        acts.append(np.clip(np.nan_to_num(a), -1, 1))
    return np.concatenate(acts)


def compare(a: np.ndarray, b: np.ndarray, atol: float) -> dict:
    d = np.abs(a - b).max(axis=1)
    return {"match": round(float(np.mean(d <= atol)), 4), "mean_abs_diff": round(float(np.abs(a - b).mean()), 5)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("policies", nargs="+")
    ap.add_argument("--probe-seed", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--atol", type=float, default=0.02)
    a = ap.parse_args()
    obs = probe_observations(a.probe_seed, a.episodes)
    acts = {p: policy_actions(p, obs) for p in a.policies}
    print(json.dumps({"probe_observations": int(len(obs))}))
    for p, q in itertools.combinations(a.policies, 2):
        print(json.dumps({"a": Path(p).name, "b": Path(q).name, **compare(acts[p], acts[q], a.atol)}))


if __name__ == "__main__":
    main()
