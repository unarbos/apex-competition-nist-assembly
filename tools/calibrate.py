"""Calibration harness for v0.2.0: junk / baseline / strong scripted across seeds.

Runs policies against the multi-peg board in-process (no HTTP) for speed, and
optionally the ONNX baseline through tools/local_run.py for wall-time of a full
eval. Writes CALIBRATION.md at the repo root.

    python tools/calibrate.py
    python tools/calibrate.py --seeds 10 --onnx-seeds 3
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

from nist_sim.conditions import SLOT_SPECS, sample_episodes  # noqa: E402
from nist_sim.env import Episode  # noqa: E402
from nist_sim.policies import NearestFirstBaseline, StrongScripted  # noqa: E402
from nist_sim.reserve import reserved_from_secret  # noqa: E402
from nist_sim.scoring import episode_score  # noqa: E402
import nist_sim.conditions as conditions  # noqa: E402


def _run_python_policy(seed: int, per: int, kind: str, split: str = "live") -> dict:
    _, conds = sample_episodes(seed, per, split)
    scores = []
    seated_fracs = []
    by_k = defaultdict(list)
    by_size = defaultdict(list)
    t0 = time.monotonic()
    for i, c in enumerate(conds):
        ss = np.random.SeedSequence([seed, 0x5245, i])
        r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
        e = Episode(c, r1)
        if kind == "zero":
            act_fn = lambda o, e=e: np.zeros(5)
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

        elif kind == "strong":
            pol = StrongScripted()

            def act_fn(o, pol=pol, e=e):
                return pol.act(e, o)

        else:
            raise ValueError(kind)

        steps = 0
        while not e.done and steps < c.max_steps:
            o = e.observe(r2)
            a = np.clip(act_fn(o), -1, 1)
            if kind == "nan":
                a = np.array([np.nan] * 5)
            e.step(a)
            steps += 1
            if kind == "nan":
                e.fail("invalid_action")
                break
        seated = e.peg_seated_at_end()
        first = e.peg_first_seat_steps()
        msa = e.action_l2 / max(1, e.steps)
        sc = episode_score(seated, first, c.max_steps, msa)
        scores.append(sc)
        seated_fracs.append(sum(seated) / max(1, len(seated)))
        by_k[c.n_pegs].append(sc)
        for peg, s in zip(c.pegs, seated):
            shape, dia, _, _ = SLOT_SPECS[peg.slot]
            by_size[f"{shape}_{round(dia * 1000)}mm"].append(int(s))
    wall = time.monotonic() - t0
    return {
        "seed": seed,
        "kind": kind,
        "mean": float(np.mean(scores)),
        "peg_frac": float(np.mean(seated_fracs)),
        "wall_s": wall,
        "n": len(scores),
        "by_k": {str(k): float(np.mean(v)) for k, v in sorted(by_k.items())},
        "by_size": {k: float(np.mean(v)) for k, v in sorted(by_size.items())},
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
        "summary": {s["key"]: s["value"] for s in res["metadata"].get("summary", [])},
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--onnx-seeds", type=int, default=3)
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "CALIBRATION.md"))
    a = ap.parse_args()

    seeds = list(range(a.seeds))
    lines = [
        "# NIST-Lite v0.2.0 calibration",
        "",
        "Measured on this box (MuJoCo 3.16, Python in-process unless noted).",
        f"Episodes/round: `{8 * a.per}` (8 strata × {a.per}).",
        "",
    ]

    # Junk policies (in-process)
    lines += ["## Junk policies (must score ~0)", ""]
    for kind in ("zero", "random", "constant", "nan"):
        rows = [_run_python_policy(s, a.per, kind) for s in seeds[:5]]
        agg = _agg(rows)
        lines.append(
            f"- **{kind}**: mean={agg['mean']:.4f} sd={agg['sd']:.4f} "
            f"range=[{agg['min']:.4f}, {agg['max']:.4f}] over {agg['n']} seeds"
        )
    lines.append("")

    # Baseline + strong
    lines += ["## Baseline (nearest-first Python) and strong scripted", ""]
    base_rows = [_run_python_policy(s, a.per, "baseline") for s in seeds]
    strong_rows = [_run_python_policy(s, a.per, "strong") for s in seeds]
    for label, rows in (("baseline", base_rows), ("strong", strong_rows)):
        agg = _agg(rows)
        peg = _agg(rows, "peg_frac")
        wall = _agg(rows, "wall_s")
        lines += [
            f"### {label}",
            f"- raw score: mean={agg['mean']:.4f} sd={agg['sd']:.4f} "
            f"min={agg['min']:.4f} max={agg['max']:.4f} (n={agg['n']})",
            f"- peg seat fraction: mean={peg['mean']:.4f}",
            f"- in-process wall (1 worker equivalent): mean={wall['mean']:.1f}s "
            f"range=[{wall['min']:.1f}, {wall['max']:.1f}]",
            f"- per-K mean score: {rows[0]['by_k']}",
            "",
        ]
        # Merge by_size across seeds
        size_acc = defaultdict(list)
        for r in rows:
            for k, v in r["by_size"].items():
                size_acc[k].append(v)
        lines.append(
            "- per-peg-size seat rate: "
            + ", ".join(f"{k}={statistics.mean(v):.2f}" for k, v in sorted(size_acc.items()))
        )
        lines.append("")

    # ONNX baseline wall time with 2 workers via local_run
    lines += ["## ONNX baseline full eval (2 workers, HTTP gym_v1)", ""]
    onnx_rows = []
    for s in seeds[: a.onnx_seeds]:
        onnx_rows.append(_run_onnx(s, "baselines/baseline.onnx", ROOT / "out" / f"cal_onnx_{s}"))
    if onnx_rows:
        agg = _agg([{**r, "mean": r["raw"]} for r in onnx_rows])
        wall = _agg([{**r, "mean": r["wall_s"]} for r in onnx_rows], "mean")
        # fix wall agg
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

    # Held-out split with throwaway secret
    lines += ["## Held-out split (dev reserve from throwaway secret)", ""]
    secret = "calibration-throwaway-secret-v020"
    spec = reserved_from_secret(secret)
    reserved_path = ROOT / "out" / "cal_reserved.json"
    reserved_path.parent.mkdir(parents=True, exist_ok=True)
    reserved_path.write_text(json.dumps(spec))
    conditions.RESERVED_PATH = reserved_path
    live = [_run_python_policy(s, a.per, "strong", "live") for s in seeds[:5]]
    held = [_run_python_policy(s, a.per, "strong", "reserved") for s in seeds[:5]]
    lines += [
        f"- reserved axes: `{spec['ranges']}`",
        f"- strong on live: mean={_agg(live)['mean']:.4f}",
        f"- strong on reserved: mean={_agg(held)['mean']:.4f}",
        f"- drop: {_agg(live)['mean'] - _agg(held)['mean']:+.4f}",
        "",
    ]

    lines += [
        "## Targets (from brief)",
        "- baseline ≤ ~0.30",
        "- strong scripted clearly below ~0.75",
        "- junk ~0",
        "- box worst-case full eval ≤ ~120 s with 2 workers",
        "",
        "## Clearance note",
        "Minimum radial clearance is **0.05 mm**. Contact `solimp` width is "
        "`0.00002` m (0.02 mm), below the clearance, so soft contacts cannot "
        "cheat the fit. Seating at 0.05 mm was verified by teleport-and-push.",
        "",
    ]

    Path(a.out).write_text("\n".join(lines) + "\n")
    print("wrote", a.out)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
