"""v0.8 calibration harness: privileged / scripted / zero / red-team / determinism / wall.

Runs in-process against the v08 code path (same as the image bake). Does **not**
load revealed leader ONNX into the repo; pass ``--leaders-dir`` pointing at a private
path outside the tree.

    NIST_PROFILE=v08 python tools/calib_v08.py --seeds 7,19 --per 8 --episodes 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tools"))

os.environ.setdefault("NIST_PROFILE", "v08")

from nist_sim.compound_v08 import (  # noqa: E402
    OBS_DIM_V08,
    AssemblyIVEpisode,
    compound_score_v08,
    sample_compound_v08,
)
from nist_sim.conditions import episode_seed_sequence  # noqa: E402
from compound_solver import solve_compound  # noqa: E402
from obs_scripted_v08 import act_from_obs  # noqa: E402


def _run_policy(seed: int, per: int, episodes: int, policy_fn) -> dict:
    _, conds = sample_compound_v08(seed, per, "live")
    scores = []
    t0 = time.time()
    for i in range(min(len(conds), episodes)):
        ss = episode_seed_sequence(seed, i)
        r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
        ep = AssemblyIVEpisode(conds[i], r1)
        noise = r2
        while not ep.done:
            o = ep.observe(noise)
            a = policy_fn(ep, o, noise, ss, i)
            ep.step(np.asarray(a, dtype=np.float64))
        scores.append(float(compound_score_v08(ep)["score"]))
    return {"mean": float(np.mean(scores)), "scores": scores, "wall_s": time.time() - t0}


def zero_policy(ep, o, noise, ss, i):
    return np.zeros(5)


def random_policy(ep, o, noise, ss, i):
    rng = np.random.default_rng(int(ss.generate_state(1)[0]) ^ (i * 0x9E3779B9) ^ ep.steps)
    return rng.uniform(-1, 1, 5)


def scripted_policy(ep, o, noise, ss, i):
    rng = np.random.default_rng(ss.spawn(1)[0])
    return act_from_obs(o, rng)


def privileged_run(seed: int, per: int, episodes: int, *, honest: bool) -> dict:
    _, conds = sample_compound_v08(seed, per, "live")
    scores = []
    t0 = time.time()
    for i in range(min(len(conds), episodes)):
        ss = episode_seed_sequence(seed, i)
        r1, _ = (np.random.default_rng(s) for s in ss.spawn(2))
        ep = AssemblyIVEpisode(conds[i], r1)
        r = solve_compound(ep, oracle=not honest)
        scores.append(float(r["score"]))
    return {"mean": float(np.mean(scores)), "scores": scores, "wall_s": time.time() - t0}


def red_team(seed: int, per: int, episodes: int = 16) -> list[dict]:
    """Try common reward-hacks; report scores (should stay near 0)."""
    strategies = []

    def jitter(ep, o, noise, ss, i):
        rng = np.random.default_rng(ep.steps + i)
        return rng.normal(0, 0.9, 5)

    def vibrate(ep, o, noise, ss, i):
        a = np.zeros(5)
        a[0] = 0.8 if (ep.steps % 2) == 0 else -0.8
        a[1] = -a[0]
        a[4] = -1.0
        return a

    def slam(ep, o, noise, ss, i):
        return np.array([0.0, 0.0, -1.0, 0.0, -1.0])

    def spin(ep, o, noise, ss, i):
        return np.array([0.0, 0.0, -0.3, 1.0 if (ep.steps // 3) % 2 == 0 else -1.0, -1.0])

    def extreme(ep, o, noise, ss, i):
        return np.array([1e6, -1e6, 1e6, -1e6, 1e6])

    for name, fn in [
        ("jitter", jitter),
        ("vibrate_parts", vibrate),
        ("slam_down", slam),
        ("yaw_spin", spin),
        ("extreme_actions", extreme),
    ]:
        r = _run_policy(seed, per, episodes, fn)
        strategies.append({"name": name, "mean": r["mean"], "wall_s": r["wall_s"], "n": len(r["scores"])})
    return strategies


def leaders_eval(leaders_dir: Path, seed: int, per: int, episodes: int) -> list[dict]:
    """Attempt to load private leader ONNX; expect rejection on width≠320."""
    import onnxruntime as ort

    out = []
    for path in sorted(leaders_dir.glob("*.onnx")):
        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
            shp = sess.get_inputs()[0].shape
            width = shp[1] if len(shp) == 2 else None
            if width != OBS_DIM_V08:
                out.append({"file": path.name, "status": "rejected", "input_width": width, "mean": None})
                continue
            # Compatible width: run (should still score near 0 on v08 distribution).
            inp = sess.get_inputs()[0].name

            def policy(ep, o, noise, ss, i, _sess=sess, _inp=inp):
                a = _sess.run(None, {_inp: o.reshape(1, -1).astype(np.float32)})[0]
                return np.asarray(a, dtype=np.float64).reshape(-1)[:5]

            r = _run_policy(seed, per, min(episodes, 16), policy)
            out.append({"file": path.name, "status": "ran", "input_width": width, "mean": r["mean"]})
        except Exception as e:
            out.append({"file": path.name, "status": f"error:{type(e).__name__}", "mean": None})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7,19")
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--leaders-dir", default="")
    ap.add_argument("--skip-privileged", action="store_true")
    ap.add_argument("--skip-redteam", action="store_true")
    ap.add_argument("--quick", action="store_true", help="8 episodes / seed only")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    eps = 8 if a.quick else a.episodes
    report: dict = {"profile": "v08", "obs_dim": OBS_DIM_V08, "seeds": seeds, "episodes": eps, "gates": {}}

    # Zero
    zero = {}
    for s in seeds:
        zero[s] = _run_policy(s, a.per, eps, zero_policy)
        print(f"zero seed={s} mean={zero[s]['mean']:.4f} wall={zero[s]['wall_s']:.1f}s", flush=True)
    report["gates"]["zero"] = {str(k): {"mean": v["mean"], "wall_s": v["wall_s"]} for k, v in zero.items()}

    # Scripted
    scripted = {}
    for s in seeds:
        scripted[s] = _run_policy(s, a.per, eps, scripted_policy)
        print(f"scripted seed={s} mean={scripted[s]['mean']:.4f} wall={scripted[s]['wall_s']:.1f}s", flush=True)
    report["gates"]["obs_scripted"] = {
        str(k): {"mean": v["mean"], "wall_s": v["wall_s"]} for k, v in scripted.items()
    }

    # Privileged honest
    if not a.skip_privileged:
        priv = {}
        for s in seeds:
            priv[s] = privileged_run(s, a.per, eps, honest=True)
            print(f"privileged_honest seed={s} mean={priv[s]['mean']:.4f} wall={priv[s]['wall_s']:.1f}s", flush=True)
        report["gates"]["privileged_honest"] = {
            str(k): {"mean": v["mean"], "wall_s": v["wall_s"]} for k, v in priv.items()
        }

    # Determinism
    d1 = _run_policy(seeds[0], a.per, min(8, eps), zero_policy)
    d2 = _run_policy(seeds[0], a.per, min(8, eps), zero_policy)
    report["gates"]["determinism_zero"] = {
        "match": d1["scores"] == d2["scores"],
        "scores": d1["scores"],
    }
    print(f"determinism match={report['gates']['determinism_zero']['match']}", flush=True)

    if not a.skip_redteam:
        rt = red_team(seeds[0], a.per, min(16, eps))
        report["gates"]["red_team"] = rt
        for row in rt:
            print(f"red_team {row['name']} mean={row['mean']:.4f}", flush=True)

    if a.leaders_dir:
        ld = Path(a.leaders_dir)
        report["gates"]["leaders"] = leaders_eval(ld, seeds[0], a.per, eps)
        for row in report["gates"]["leaders"]:
            print(f"leader {row}", flush=True)

    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
        print("wrote", a.out)
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
