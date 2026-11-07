#!/usr/bin/env python3
"""Evaluate any ONNX policy through the *built* referee + player Docker images.

Does **not** ship or require any particular ONNX file — pass paths on the CLI.
Asserts the referee image bakes ``nist_profile=v07`` and prints score + wall time.

    # Build images first:
    DOCKER_BUILDKIT=1 docker build -f referee/Dockerfile -t nist-assembly-referee:v071 .
    DOCKER_BUILDKIT=1 docker build -f player/Dockerfile -t nist-assembly-player:v071 .

    # 64-ep live-sized eval (8 per stratum):
    python tools/image_eval_onnx.py --seed 7 --per 8 /path/to/policy.onnx

    # Several policies:
    python tools/image_eval_onnx.py --seed 7 --full --out /tmp/img_eval \\
        /tmp/kings/onnx/5CfGwv.onnx /tmp/kings/onnx/5GHgWv.onnx

Never commit miner / king ONNX files into this repo.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, **kw)


def docker_eval(
    policy: Path,
    out_dir: Path,
    seed: int,
    per: int,
    *,
    referee_image: str = "nist-assembly-referee:v072",
    player_image: str = "nist-assembly-player:v072",
    deadline_ms: int = 80,
    workers: int = 2,
) -> dict:
    """Run one ONNX through built images (host networking)."""
    policy = policy.resolve()
    if not policy.is_file():
        raise FileNotFoundError(policy)
    out_dir.mkdir(parents=True, exist_ok=True)
    player = f"nist-onnx-{os.getpid()}-{abs(hash(str(policy))) % 10000}"
    port = 19000 + (os.getpid() % 700) + (abs(hash(str(policy))) % 200)
    cfg = json.dumps(
        {
            "episodes_per_stratum": per,
            "deadline_ms": deadline_ms,
            "workers": workers,
            "record_stride": 8,
            "records_enabled": True,
        }
    )
    try:
        _run(
            [
                "docker", "run", "-d", "--name", player, "--network", "host",
                "--cpus", "2", "--memory", "2g",
                "-v", f"{policy}:/app/policy.onnx:ro",
                player_image,
                "python", "/app/launch.py", "--port", str(port),
            ],
            capture_output=True,
        )
        for _ in range(90):
            time.sleep(1)
            r = subprocess.run(
                [
                    "python", "-c",
                    f"import urllib.request; urllib.request.urlopen('http://127.0.0.1:{port}/health', timeout=2)",
                ],
                capture_output=True,
            )
            if r.returncode == 0:
                break
        else:
            logs = subprocess.run(["docker", "logs", player], capture_output=True, text=True)
            raise RuntimeError(f"player never healthy on :{port}\n{logs.stdout}\n{logs.stderr}")
        t0 = time.perf_counter()
        _run(
            [
                "docker", "run", "--rm", "--network", "host",
                "--cpus", "2", "--memory", "2g",
                "-e", f"MATCH_ID=onnx-{seed}",
                "-e", f"SEED={seed}",
                "-e", "NUM_PLAYERS=1",
                "-e", f"PLAYER_URLS=http://127.0.0.1:{port}",
                "-e", f"CONFIG_JSON={cfg}",
                "-v", f"{out_dir.resolve()}:/data",
                referee_image,
            ],
        )
        wall = time.perf_counter() - t0
    finally:
        subprocess.run(["docker", "rm", "-f", player], capture_output=True)

    result = json.loads((out_dir / "result.json").read_text())
    md = result.get("metadata") or result
    details = md.get("details") or {}
    profile = md.get("nist_profile") or details.get("nist_profile")
    if profile != "v07":
        ep_path = out_dir / "history" / "episodes.jsonl"
        if ep_path.is_file():
            header = json.loads(ep_path.read_text().splitlines()[0])
            profile = header.get("nist_profile", profile)
    if profile != "v07":
        raise AssertionError(f"image bake failed: nist_profile={profile!r}")
    raw = result.get("raw_scores") or [md.get("raw_score", 0.0)]
    score = float(raw[0] if isinstance(raw, list) else raw)
    return {
        "policy": str(policy),
        "label": policy.stem,
        "score": score,
        "wall_s": wall,
        "nist_profile": profile,
        "n_episodes": per * 8,
        "terminal_reason": result.get("terminal_reason"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", nargs="+", type=Path, help="ONNX policy path(s)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per", type=int, default=2, help="episodes per stratum (8 strata)")
    ap.add_argument("--full", action="store_true", help="8/stratum = 64 episodes")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "image_eval_onnx")
    ap.add_argument("--referee-image", default="nist-assembly-referee:v072")
    ap.add_argument("--player-image", default="nist-assembly-player:v072")
    ap.add_argument("--twice", action="store_true", help="run each policy twice (determinism check)")
    a = ap.parse_args()
    if a.full:
        a.per = 8
    a.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for onnx in a.onnx:
        runs = 2 if a.twice else 1
        for run_i in range(runs):
            sub = a.out / f"{onnx.stem}_r{run_i}"
            print(f"\n== {onnx} (run {run_i + 1}/{runs}) ==", flush=True)
            d = docker_eval(
                onnx, sub, a.seed, a.per,
                referee_image=a.referee_image,
                player_image=a.player_image,
            )
            d["run"] = run_i
            rows.append(d)
            print(
                f"  {d['label']}: score={d['score']:.4f} wall={d['wall_s']:.1f}s "
                f"profile={d['nist_profile']} n={d['n_episodes']}",
                flush=True,
            )

    (a.out / "summary.json").write_text(json.dumps(rows, indent=2))
    print("\n| Policy | run | score | wall_s | profile |", flush=True)
    print("|---|---|---|---|---|", flush=True)
    for r in rows:
        print(
            f"| {r['label']} | {r['run']} | {r['score']:.4f} | {r['wall_s']:.1f} | {r['nist_profile']} |",
            flush=True,
        )
    if a.twice:
        by = {}
        for r in rows:
            by.setdefault(r["label"], []).append(r["score"])
        print("\nDeterminism:", flush=True)
        for label, scores in by.items():
            ok = abs(scores[0] - scores[1]) < 1e-9
            print(f"  {label}: {scores[0]:.6f} vs {scores[1]:.6f} → {'OK' if ok else 'DIFF'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
