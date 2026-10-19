"""Calibration harness for v0.2.0 anti-scripting release gate.

Reports junk / baseline / obs-only scripted / privileged reference across seeds,
with per-difficulty breakdowns. Writes CALIBRATION.md.

    python tools/calibrate.py --seeds 10 --per 4 --onnx-seeds 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tools"))

from nist_sim.conditions import SLOT_SPECS, sample_episodes  # noqa: E402
from nist_sim.env import OBS_DIM, Episode  # noqa: E402
from nist_sim.policies import NearestFirstBaseline, ObsOnlyScripted  # noqa: E402
from nist_sim.reserve import reserved_from_secret  # noqa: E402
from nist_sim.scoring import episode_score  # noqa: E402
import nist_sim.conditions as conditions  # noqa: E402
from reference_solver import solve_episode  # noqa: E402


def _clearance_bin(c: float) -> str:
    mm = c * 1e3
    if mm < 0.40:
        return "tight(<0.40mm)"
    if mm < 0.60:
        return "mid(0.40-0.60mm)"
    return "loose(>=0.60mm)"


def _pert_bin(c) -> str:
    bits = []
    if c.act_delay > 0:
        bits.append("delay")
    if c.disturbance:
        bits.append(c.disturbance_target.split(":")[0])
    if c.backlash > 0.0002:
        bits.append("backlash")
    return "+".join(bits) if bits else "none"


def _start_mix(c) -> str:
    n_lying = sum(1 for p in c.pegs if p.start_type == "lying")
    n_up = sum(1 for p in c.pegs if p.start_type in ("loose", "near_hole", "grasped"))
    if n_lying == len(c.pegs):
        return "all_lying"
    if n_lying > 0:
        return "mixed_lying"
    if n_up == len(c.pegs):
        return "all_upright"
    return "has_tray"


def _run_episode(seed: int, i: int, c, kind: str) -> dict:
    ss = np.random.SeedSequence([seed, 0x5245, i])
    r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
    e = Episode(c, r1)
    if kind == "zero":
        act_fn = lambda o: np.zeros(5)
    elif kind == "random":
        rng_a = np.random.default_rng(seed * 10007 + i)

        def act_fn(o, rng_a=rng_a):
            return rng_a.uniform(-1, 1, 5)

    elif kind == "constant":
        act_fn = lambda o: np.array([0.0, 0.0, -1.0, 0.0, -1.0])
    elif kind == "baseline":
        pol = NearestFirstBaseline()

        def act_fn(o, pol=pol):
            return pol.act(o)

    elif kind == "obs_scripted":
        pol = ObsOnlyScripted()

        def act_fn(o, pol=pol):
            return pol.act(o)

    elif kind == "reference":
        r = solve_episode(e)
        return {
            "score": r["score"],
            "peg_frac": r["n_seated"] / max(1, r["n_pegs"]),
            "n_seated": r["n_seated"],
            "n_pegs": r["n_pegs"],
            "terminal": e.terminal_reason,
        }
    else:
        raise ValueError(kind)

    while not e.done:
        o = e.observe(r2)
        a = np.clip(act_fn(o), -1, 1)
        e.step(a)
    seated = e.peg_seated_at_end()
    first = e.peg_first_seat_steps()
    msa = e.action_l2 / max(1, e.steps)
    sc = episode_score(seated, first, c.max_steps, msa)
    return {
        "score": sc,
        "peg_frac": sum(seated) / max(1, len(seated)),
        "n_seated": int(sum(seated)),
        "n_pegs": len(seated),
        "terminal": e.terminal_reason,
    }


def _run_python_policy(seed: int, per: int, kind: str, split: str = "live") -> dict:
    _, conds = sample_episodes(seed, per, split)
    scores = []
    seated_fracs = []
    by_k = defaultdict(list)
    by_clear = defaultdict(list)
    by_clutter = defaultdict(list)
    by_start = defaultdict(list)
    by_pert = defaultdict(list)
    t0 = time.monotonic()
    for i, c in enumerate(conds):
        row = _run_episode(seed, i, c, kind)
        sc = row["score"]
        scores.append(sc)
        seated_fracs.append(row["peg_frac"])
        by_k[c.n_pegs].append(sc)
        by_clear[_clearance_bin(c.clearance)].append(sc)
        by_clutter[c.clutter_bucket].append(sc)
        by_start[_start_mix(c)].append(sc)
        by_pert[_pert_bin(c)].append(sc)
    wall = time.monotonic() - t0
    return {
        "seed": seed,
        "kind": kind,
        "mean": float(np.mean(scores)),
        "peg_frac": float(np.mean(seated_fracs)),
        "wall_s": wall,
        "n": len(scores),
        "by_k": {str(k): float(np.mean(v)) for k, v in sorted(by_k.items())},
        "by_clear": {k: float(np.mean(v)) for k, v in sorted(by_clear.items())},
        "by_clutter": {k: float(np.mean(v)) for k, v in sorted(by_clutter.items())},
        "by_start": {k: float(np.mean(v)) for k, v in sorted(by_start.items())},
        "by_pert": {k: float(np.mean(v)) for k, v in sorted(by_pert.items())},
    }


def _run_onnx(seed: int, policy: str, out: Path) -> dict:
    t0 = time.monotonic()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/local_run.py"),
            "--policy",
            str(ROOT / policy),
            "--seed",
            str(seed),
            "--out",
            str(out),
        ],
        check=True,
    )
    wall = time.monotonic() - t0
    res = json.loads((out / "result.json").read_text())
    return {
        "seed": seed,
        "raw": res["raw_scores"][0],
        "wall_s": wall,
        "terminal": res["terminal_reason"],
    }


def _agg(rows: list[dict], key: str = "mean") -> dict:
    vals = [r[key] for r in rows]
    return {
        "mean": statistics.mean(vals),
        "sd": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
    }


def _merge_breakdown(rows: list[dict], field: str) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for k, v in r.get(field, {}).items():
            acc[k].append(v)
    return {k: statistics.mean(v) for k, v in sorted(acc.items())}


def _fmt_breakdown(d: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.3f}" for k, v in d.items())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--onnx-seeds", type=int, default=3)
    ap.add_argument("--per", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "CALIBRATION.md"))
    a = ap.parse_args()

    seeds = list(range(a.seeds))
    lines = [
        "# NIST-Lite v0.2.0 calibration (anti-scripting)",
        "",
        f"Measured on this box (MuJoCo 3.16). `OBS_DIM={OBS_DIM}`.",
        f"Episodes/round: `{8 * a.per}` (8 strata × {a.per}).",
        "",
        "## Why RL is required",
        "",
        "v0.1.5's round-4 king was a hand-written controller compiled to ONNX",
        "(~5.8k elementwise ops, no weight matrices). v0.2 defeats that class:",
        "",
        "- **Pose-estimate bias**: every peg/slot channel carries a fixed",
        "  per-episode bias of several mm and degrees (plus white noise + dropout).",
        "  Exact hole/peg poses are never in the observation.",
        "- **Wrist force/torque**: contact-driven search is observable; open-loop",
        "  geometry following is not enough.",
        "- **Hidden dynamics**: friction, mass, gain, mild backlash, chamfer,",
        "  act-delay, mid-episode shove/drop.",
        "- **Held-out shapes**: hex and D-flat keyed pegs appear only on the",
        "  reserved split (never in live rounds).",
        "",
        "## Difficulty (live ranges)",
        "",
        f"- clearance: `{conditions.RANGES['clearance'][0]*1e3:.2f}–"
        f"{conditions.RANGES['clearance'][1]*1e3:.2f} mm`",
        f"- pose_bias_xy mag: `{conditions.RANGES['pose_bias_xy'][0]*1e3:.1f}–"
        f"{conditions.RANGES['pose_bias_xy'][1]*1e3:.1f} mm`",
        f"- STEPS_PER_PEG: `{conditions.STEPS_PER_PEG}`",
        f"- held-out shapes: `{conditions.HELD_OUT_SHAPES}`",
        "",
    ]

    lines += ["## Junk policies (must score ~0)", ""]
    for kind in ("zero", "random", "constant"):
        rows = [_run_python_policy(s, a.per, kind) for s in seeds[:5]]
        agg = _agg(rows)
        lines.append(
            f"- **{kind}**: mean={agg['mean']:.4f} sd={agg['sd']:.4f} "
            f"range=[{agg['min']:.4f}, {agg['max']:.4f}] over {agg['n']} seeds"
        )
    lines.append("")

    lines += ["## Anti-scripting and privileged reference", ""]
    base_rows = [_run_python_policy(s, a.per, "baseline") for s in seeds]
    obs_rows = [_run_python_policy(s, a.per, "obs_scripted") for s in seeds]
    ref_rows = [_run_python_policy(s, a.per, "reference") for s in seeds]
    for label, rows in (
        ("baseline (nearest-first, obs)", base_rows),
        ("obs-only scripted (strongest hand-coded on obs)", obs_rows),
        ("privileged reference (true state; tools/ only)", ref_rows),
    ):
        agg = _agg(rows)
        peg = _agg(rows, "peg_frac")
        wall = _agg(rows, "wall_s")
        lines += [
            f"### {label}",
            f"- raw score: mean={agg['mean']:.4f} sd={agg['sd']:.4f} "
            f"min={agg['min']:.4f} max={agg['max']:.4f} (n={agg['n']})",
            f"- peg seat fraction: mean={peg['mean']:.4f}",
            f"- in-process wall: mean={wall['mean']:.1f}s "
            f"range=[{wall['min']:.1f}, {wall['max']:.1f}]",
            f"- per-K: {_fmt_breakdown(_merge_breakdown(rows, 'by_k'))}",
            f"- by clearance: {_fmt_breakdown(_merge_breakdown(rows, 'by_clear'))}",
            f"- by clutter: {_fmt_breakdown(_merge_breakdown(rows, 'by_clutter'))}",
            f"- by start mix: {_fmt_breakdown(_merge_breakdown(rows, 'by_start'))}",
            f"- by perturbation: {_fmt_breakdown(_merge_breakdown(rows, 'by_pert'))}",
            "",
        ]

    lines += ["## ONNX baseline full eval (2 workers, HTTP gym_v1)", ""]
    onnx_rows = []
    for s in seeds[: a.onnx_seeds]:
        try:
            onnx_rows.append(
                _run_onnx(s, "baselines/baseline.onnx", ROOT / "out" / f"cal_onnx_{s}")
            )
        except Exception as ex:  # noqa: BLE001
            lines.append(f"- ONNX seed {s} failed: `{ex}`")
    if onnx_rows:
        agg = _agg([{**r, "mean": r["raw"]} for r in onnx_rows])
        wall = {
            "mean": statistics.mean(r["wall_s"] for r in onnx_rows),
            "min": min(r["wall_s"] for r in onnx_rows),
            "max": max(r["wall_s"] for r in onnx_rows),
        }
        lines += [
            f"- ONNX raw: mean={agg['mean']:.4f} sd={agg['sd']:.4f} "
            f"range=[{agg['min']:.4f}, {agg['max']:.4f}] over {len(onnx_rows)} seeds",
            f"- wall time (2 workers): mean={wall['mean']:.1f}s "
            f"range=[{wall['min']:.1f}, {wall['max']:.1f}]",
            "",
        ]

    lines += ["## Held-out split (dev reserve from throwaway secret)", ""]
    secret = "calibration-throwaway-secret-v020"
    spec = reserved_from_secret(secret)
    reserved_path = ROOT / "out" / "cal_reserved.json"
    reserved_path.parent.mkdir(parents=True, exist_ok=True)
    reserved_path.write_text(json.dumps(spec))
    conditions.RESERVED_PATH = reserved_path
    live = [_run_python_policy(s, a.per, "reference", "live") for s in seeds[:5]]
    held = [_run_python_policy(s, a.per, "reference", "reserved") for s in seeds[:5]]
    lines += [
        f"- reserved axes: `{spec['ranges']}`",
        f"- reference on live: mean={_agg(live)['mean']:.4f}",
        f"- reference on reserved (incl. held-out shapes): mean={_agg(held)['mean']:.4f}",
        f"- drop: {_agg(live)['mean'] - _agg(held)['mean']:+.4f}",
        "",
    ]

    ref_mean = _agg(ref_rows)["mean"]
    obs_mean = _agg(obs_rows)["mean"]
    base_mean = _agg(base_rows)["mean"]
    lines += [
        "## Targets (release gate)",
        f"- privileged reference ≈ 0.50–0.75 → **{'met' if 0.45 <= ref_mean <= 0.80 else 'MISSED'}** "
        f"(mean={ref_mean:.3f})",
        f"- obs-only scripted ≤ ~0.25 → **{'met' if obs_mean <= 0.25 else 'MISSED'}** "
        f"(mean={obs_mean:.3f})",
        f"- baseline ≤ ~0.15 → **{'met' if base_mean <= 0.15 else 'MISSED'}** "
        f"(mean={base_mean:.3f})",
        "- junk ~0 → see above",
        "- eval ≤ ~400 s with 2 workers → see ONNX wall above",
        "",
        "## Privileged reference solver",
        "`tools/reference_solver.py` — not shipped in the player. Reads true MuJoCo",
        "state; proves the task is physically solvable with headroom below 1.0.",
        "",
        "## Obs-only scripted controller",
        "`ObsOnlyScripted` in `referee/nist_sim/policies.py` — strongest hand-coded",
        "policy on the biased observation (spiral + force threshold). Stays far",
        "below the privileged solver, so a pure script cannot win.",
        "",
    ]

    Path(a.out).write_text("\n".join(lines) + "\n")
    print("wrote", a.out)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
