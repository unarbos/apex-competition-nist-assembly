"""Calibrate v0.7 through the *built* referee/player Docker images (not in-process imports).

Builds are assumed tagged locally:
  nist-assembly-referee:v071
  nist-assembly-player:v071

For each policy ONNX under a temp mount, runs the same docker network loop the platform
uses, asserts ``nist_profile=v07`` in result metadata / records, and prints a table.

    python tools/image_calib_v07.py --seed 7 --per 2 --episodes-note "smoke"
    python tools/image_calib_v07.py --seed 7 --per 8 --full   # 64-ep live-sized

Privileged / v06_style / compound_scripted are scored in-process with the same code the
image ships (honest privileged), then a zero ONNX eval goes through the images to prove
the bake and timing. Optionally pass --onnx-only to skip in-process rows.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, **kw)


def docker_eval(policy: Path, out_dir: Path, seed: int, per: int, deadline_ms: int = 80) -> dict:
    """Run one policy through the built player+referee images.

    Uses host networking so the referee can reach the player on this VM (custom
    docker-bridge networks often drop inter-container traffic here). Same images
    and HTTP gym_v1 protocol the platform uses.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    player = f"nist-player-{os.getpid()}-{seed}"
    port = 18000 + (os.getpid() % 1000)
    cfg = json.dumps(
        {
            "episodes_per_stratum": per,
            "deadline_ms": deadline_ms,
            "workers": 2,
            "record_stride": 8,
            "records_enabled": True,
        }
    )
    try:
        _run(
            [
                "docker", "run", "-d", "--name", player, "--network", "host",
                "--cpus", "2", "--memory", "2g",
                "-v", f"{policy.resolve()}:/app/policy.onnx:ro",
                "nist-assembly-player:v071",
                "python", "/app/launch.py", "--port", str(port),
            ],
            capture_output=True,
        )
        # Wait for health on the published host port.
        for _ in range(60):
            time.sleep(1)
            try:
                r = subprocess.run(
                    [
                        "python", "-c",
                        f"import urllib.request; urllib.request.urlopen('http://127.0.0.1:{port}/health', timeout=2)",
                    ],
                    capture_output=True,
                )
                if r.returncode == 0:
                    break
            except Exception:
                pass
        else:
            logs = subprocess.run(["docker", "logs", player], capture_output=True, text=True)
            raise RuntimeError(f"player never healthy on :{port}\n{logs.stdout}\n{logs.stderr}")
        t0 = time.perf_counter()
        _run(
            [
                "docker", "run", "--rm", "--network", "host",
                "--cpus", "2", "--memory", "2g",
                "-e", f"MATCH_ID=calib-{seed}",
                "-e", f"SEED={seed}",
                "-e", "NUM_PLAYERS=1",
                "-e", f"PLAYER_URLS=http://127.0.0.1:{port}",
                "-e", f"CONFIG_JSON={cfg}",
                "-v", f"{out_dir.resolve()}:/data",
                "nist-assembly-referee:v071",
            ],
        )
        wall = time.perf_counter() - t0
    finally:
        subprocess.run(["docker", "rm", "-f", player], capture_output=True)

    result = json.loads((out_dir / "result.json").read_text())
    # Platform wraps GameResult; accept either shape.
    md = result.get("metadata") or result
    details = md.get("details") or {}
    profile = md.get("nist_profile") or details.get("nist_profile")
    if profile != "v07":
        # Also check records header.
        ep_path = out_dir / "history" / "episodes.jsonl"
        if ep_path.is_file():
            header = json.loads(ep_path.read_text().splitlines()[0])
            profile = header.get("nist_profile", profile)
    if profile != "v07":
        raise AssertionError(f"image bake failed: nist_profile={profile!r} (want v07); result={result!r}")
    raw = result.get("raw_scores") or [md.get("raw_score", 0.0)]
    score = float(raw[0] if isinstance(raw, list) else raw)
    return {
        "score": score,
        "wall_s": wall,
        "nist_profile": profile,
        "terminal_reason": result.get("terminal_reason"),
        "result": result,
    }


def inprocess_rows(seed: int, per: int) -> list[dict]:
    os.environ["NIST_PROFILE"] = "v07"
    sys.path.insert(0, str(ROOT / "referee"))
    sys.path.insert(0, str(ROOT / "tools"))
    from nist_sim import conditions as C
    from nist_sim.compound_v07 import AssemblyIIIEpisode, sample_compound_v07
    from compound_solver import solve_compound
    from v06_style_scripted import solve_v06_style
    from compound_obs_scripted import CompoundObsScripted
    from nist_sim.compound_v07 import compound_score_v07

    _, conds = sample_compound_v07(seed, per)
    rows = []
    for name, fn in [
        ("reference_honest", lambda ep: solve_compound(ep, oracle=False)),
        ("v06_style", solve_v06_style),
    ]:
        scores = []
        t0 = time.perf_counter()
        for i, c in enumerate(conds):
            r1, r2 = (np.random.default_rng(s) for s in C.episode_seed_sequence(seed, i).spawn(2))
            ep = AssemblyIIIEpisode(c, r1)
            if name == "compound_scripted":
                continue
            scores.append(fn(ep)["score"])
        rows.append(dict(policy=name, mean=float(np.mean(scores)), wall_s=time.perf_counter() - t0, n=len(scores)))

    # obs-only
    scores = []
    t0 = time.perf_counter()
    for i, c in enumerate(conds):
        r1, r2 = (np.random.default_rng(s) for s in C.episode_seed_sequence(seed, i).spawn(2))
        ep = AssemblyIIIEpisode(c, r1)
        pol = CompoundObsScripted()
        while not ep.done:
            o = ep.observe(r2)
            ep.step(np.clip(pol.act(o.astype(np.float32)), -1, 1))
        scores.append(compound_score_v07(ep)["score"])
    rows.append(dict(policy="compound_scripted", mean=float(np.mean(scores)), wall_s=time.perf_counter() - t0, n=len(scores)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--full", action="store_true", help="8/stratum (64 eps)")
    ap.add_argument("--skip-docker", action="store_true")
    ap.add_argument("--onnx-only", action="store_true")
    a = ap.parse_args()
    if a.full:
        a.per = 8
    out = ROOT / "out" / "image_calib_v071"
    out.mkdir(parents=True, exist_ok=True)

    table = []
    if not a.onnx_only:
        print("== in-process honest calib (same code the image ships) ==", flush=True)
        for row in inprocess_rows(a.seed, a.per):
            table.append(row)
            print(f"  {row['policy']}: mean={row['mean']:.3f} wall={row['wall_s']:.1f}s n={row['n']}", flush=True)

    if not a.skip_docker:
        zero = ROOT / "test_policies" / "zero.onnx"
        if not zero.is_file():
            _run([sys.executable, str(ROOT / "tools" / "make_policies.py")])
        print("== docker image eval (zero ONNX) ==", flush=True)
        d = docker_eval(zero, out / "zero", a.seed, a.per)
        table.append(dict(policy="zero_image", mean=d["score"], wall_s=d["wall_s"], n=a.per * 8, nist_profile=d["nist_profile"]))
        print(
            f"  zero_image: mean={d['score']:.3f} wall={d['wall_s']:.1f}s profile={d['nist_profile']}",
            flush=True,
        )
        # Project 64-ep from this wall if per!=8.
        proj = d["wall_s"] * (8 / a.per) if a.per else d["wall_s"]
        print(f"  projected 64-ep wall ≈ {proj:.0f}s (from per={a.per})", flush=True)

    (out / "summary.json").write_text(json.dumps(table, indent=2))
    print("\nCalibration summary:", flush=True)
    print("| Policy | mean | wall_s |", flush=True)
    print("|---|---|---|", flush=True)
    for r in table:
        print(f"| {r['policy']} | {r['mean']:.3f} | {r.get('wall_s', 0):.1f} |", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
