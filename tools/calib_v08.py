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
from obs_scripted_v08 import ObsScriptedPolicy  # noqa: E402


def _summary(scores, *, bootstrap: int = 10000) -> dict:
    arr = np.asarray(scores, dtype=np.float64)
    if len(arr) == 0:
        return {"n": 0, "mean": 0.0, "ci95": [0.0, 0.0]}
    rng = np.random.default_rng(20261117)
    means = np.mean(rng.choice(arr, (bootstrap, len(arr)), replace=True), axis=1)
    return {
        "n": len(arr), "mean": float(np.mean(arr)),
        "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
        "ci_method": f"episode bootstrap, {bootstrap} resamples, seed 20261117",
    }


def _run_policy(seed: int, per: int, episodes: int, policy_fn) -> dict:
    _, conds = sample_compound_v08(seed, per, "live")
    scores = []
    rows = []
    t0 = time.monotonic()
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
        rows.append({"episode": i, "score": scores[-1], "terminal_reason": ep.terminal_reason, "steps": ep.steps})
    return {**_summary(scores), "scores": scores, "episodes": rows, "wall_s": time.monotonic() - t0}


def zero_policy(ep, o, noise, ss, i):
    return np.zeros(5)


def random_policy(ep, o, noise, ss, i):
    rng = np.random.default_rng(int(ss.generate_state(1)[0]) ^ (i * 0x9E3779B9) ^ ep.steps)
    return rng.uniform(-1, 1, 5)


def scripted_policy(ep, o, noise, ss, i):
    if ep.steps == 0:
        scripted_policy._controllers[i] = ObsScriptedPolicy(int(ss.generate_state(1)[0]))
    return scripted_policy._controllers[i].act(o)


scripted_policy._controllers = {}


def privileged_run(seed: int, per: int, episodes: int, *, honest: bool) -> dict:
    _, conds = sample_compound_v08(seed, per, "live")
    scores = []
    t0 = time.monotonic()
    for i in range(min(len(conds), episodes)):
        ss = episode_seed_sequence(seed, i)
        r1, _ = (np.random.default_rng(s) for s in ss.spawn(2))
        ep = AssemblyIVEpisode(conds[i], r1)
        r = solve_compound(ep, oracle=not honest)
        scores.append(float(r["score"]))
    return {**_summary(scores), "scores": scores, "wall_s": time.monotonic() - t0}


def red_team(seed: int, per: int, episodes: int = 16) -> list[dict]:
    """Try common reward-hacks + audit hole-class attacks; report scores (near 0)."""
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

    def cue_hillclimb_sweep(ep, o, noise, ss, i):
        """v0.7.5-style closeness/detent sweep: continuous yaw search while pressing.

        On v0.8 there is no closeness/detent gradient, so this should not farm keys.
        """
        a = np.zeros(5)
        a[2] = -0.7
        a[3] = 0.85 if (ep.steps % 40) < 20 else -0.85
        a[4] = -1.0
        return a

    def nan_actions(ep, o, noise, ss, i):
        return np.array([np.nan, 0.0, 0.0, 0.0, 0.0])

    def sensor_spoofing(ep, o, noise, ss, i):
        # Adversary can falsify its own sensor estimate, but receives no hidden
        # state or reward. A blind target-directed spoof should not earn credit.
        return np.array([0.65 * np.sin(0.09 * ep.steps), 0.65 * np.cos(0.11 * ep.steps), -0.4, 0.5, -1.0])

    def early_termination(ep, o, noise, ss, i):
        if ep.steps == 20:
            ep.fail("player_fault")
        return np.zeros(5)

    def solver_blowup(ep, o, noise, ss, i):
        if ep.steps == 5:
            ep.data.qvel[ep.peg_vadr[0]] = 1e6
        return np.array([1e100, -1e100, 1e100, -1e100, 1e100])

    for name, fn in [
        ("jitter", jitter),
        ("vibrate_parts", vibrate),
        ("slam_down", slam),
        ("yaw_spin", spin),
        ("extreme_actions", extreme),
        ("nan_actions", nan_actions),
        ("sensor_spoofing", sensor_spoofing),
        ("early_termination", early_termination),
        ("solver_blowup", solver_blowup),
        ("cue_hillclimb_sweep", cue_hillclimb_sweep),
    ]:
        r = _run_policy(seed, per, episodes, fn)
        strategies.append({
            "name": name, "mean": r["mean"], "ci95": r["ci95"], "wall_s": r["wall_s"], "n": len(r["scores"]),
            "terminal_reasons": {reason: sum(row["terminal_reason"] == reason for row in r["episodes"]) for reason in sorted({row["terminal_reason"] for row in r["episodes"]})},
        })

    # Privileged seat-at-symmetry freebie attack (audit hole class 2).
    strategies.append(red_team_natural_seat_freebie(seed, per, episodes))
    # Single-kind cap is an analytic construction, not a physical rollout. It
    # deliberately isolates the equal-weight formula and is labelled as such.
    strategies.append(red_team_single_kind_cap(seed, per, episodes))
    return strategies


def red_team_natural_seat_freebie(seed: int, per: int, episodes: int) -> dict:
    """After seating, hold hand at slot symmetry yaw and press-probe; keys must not complete."""
    from nist_sim.compound import wrap
    from nist_sim.compound_v07 import KEY_HOLD, KEY_PROBE_FRAMES
    from nist_sim.compound_v08 import KEY_ANTI_ALIGN

    _, conds = sample_compound_v08(seed, per, "live")
    key_attempts = key_done = 0
    t0 = time.time()
    for i in range(min(len(conds), episodes)):
        ss = episode_seed_sequence(seed, i)
        r1, _ = (np.random.default_rng(s) for s in ss.spawn(2))
        ep = AssemblyIVEpisode(conds[i], r1)
        for k, s in enumerate(ep.cs.steps):
            if s.kind != "key":
                continue
            peg = s.peg
            slot_yaw = float(ep.c.pegs[peg].spec.yaw + ep.c.board_yaw)
            if abs(wrap(s.key_yaw - slot_yaw)) < KEY_ANTI_ALIGN - 1e-6:
                continue  # would be a generator bug
            for p in s.prereq:
                ep.step_done[p] = True
                ep.step_done_at[p] = 1
            ep.seated_now[peg] = True
            ep._peg_touched[peg] = True
            ep._peg_probe_frames[peg] = KEY_PROBE_FRAMES + 5
            ep._key_index_armed[peg] = True
            ep._active_key = k
            ep.data.qpos[3] = float(slot_yaw)
            ep._key_hold[peg] = 0
            key_attempts += 1
            for _ in range(KEY_HOLD + 3):
                done, _ = ep._step_state(k)
                if done:
                    key_done += 1
                    break
    rate = (key_done / key_attempts) if key_attempts else 0.0
    return {
        "name": "natural_seat_key_freebie",
        "mean": float(rate),  # fraction of keys completed by freebie (target 0)
        "wall_s": time.time() - t0,
        "n": key_attempts,
        "note": "fraction of keys completed at slot-symmetry yaw after probe/hold",
    }


def red_team_single_kind_cap(seed: int, per: int, episodes: int) -> dict:
    """Max score obtainable by finishing only one step kind (equal-weight seal)."""
    _, conds = sample_compound_v08(seed, per, "live")
    worst = 0.0
    t0 = time.time()
    for i in range(min(len(conds), episodes)):
        ss = episode_seed_sequence(seed, i)
        r1, _ = (np.random.default_rng(s) for s in ss.spawn(2))
        ep = AssemblyIVEpisode(conds[i], r1)
        ep._action_earned_v08 = True  # analytic completed-step bound, not reset credit
        n = len(ep.cs.steps)
        for kind in ("key", "insert", "stack", "rework", "screw", "collar", "twist"):
            for k, s in enumerate(ep.cs.steps):
                ep.step_done[k] = s.kind == kind
                ep.step_in_order[k] = s.kind == kind
                ep.step_done_at[k] = 10 if s.kind == kind else None
                ep.step_progress[k] = 0.0
                ep._stability_ok[k] = True
                ep._precision[k] = 1.0
            ep._finalized = True
            ep.done = True
            sc = float(compound_score_v08(ep)["score"])
            worst = max(worst, sc)
    return {
        "name": "single_kind_dominance_cap",
        "mean": worst,
        "wall_s": time.time() - t0,
        "n": min(len(conds), episodes),
        "note": "max score if only one step kind is finished (target << 1, ideally <0.55)",
    }


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
    report["gates"]["zero"] = {str(k): {key: v[key] for key in ("mean", "ci95", "wall_s", "scores")} for k, v in zero.items()}

    # Uniform random actions on exactly the same sampled episodes.
    random = {}
    for s in seeds:
        random[s] = _run_policy(s, a.per, eps, random_policy)
        print(f"random seed={s} mean={random[s]['mean']:.4f} wall={random[s]['wall_s']:.1f}s", flush=True)
    report["gates"]["random"] = {str(k): {key: v[key] for key in ("mean", "ci95", "wall_s", "scores")} for k, v in random.items()}

    # Scripted
    scripted = {}
    for s in seeds:
        scripted[s] = _run_policy(s, a.per, eps, scripted_policy)
        print(f"scripted seed={s} mean={scripted[s]['mean']:.4f} wall={scripted[s]['wall_s']:.1f}s", flush=True)
    report["gates"]["obs_scripted"] = {
        str(k): {key: v[key] for key in ("mean", "ci95", "wall_s", "scores")} for k, v in scripted.items()
    }

    # Privileged honest
    if not a.skip_privileged:
        priv = {}
        for s in seeds:
            priv[s] = privileged_run(s, a.per, eps, honest=True)
            print(f"privileged_honest seed={s} mean={priv[s]['mean']:.4f} wall={priv[s]['wall_s']:.1f}s", flush=True)
        report["gates"]["privileged_honest"] = {
            str(k): {key: v[key] for key in ("mean", "ci95", "wall_s", "scores")} for k, v in priv.items()
        }

    # Determinism
    d1 = _run_policy(seeds[0], a.per, min(8, eps), scripted_policy)
    d2 = _run_policy(seeds[0], a.per, min(8, eps), scripted_policy)
    report["gates"]["determinism_scripted"] = {
        "match": d1["scores"] == d2["scores"],
        "scores": d1["scores"],
    }
    print(f"determinism match={report['gates']['determinism_scripted']['match']}", flush=True)

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
