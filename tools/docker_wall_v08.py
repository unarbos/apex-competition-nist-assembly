"""Measure a real 64-episode round with the BUILT referee image on two CPUs.

Run on a Docker-capable host after building ``referee/Dockerfile``. The ONNX
player runs on the host; the referee runs in the newly built image and talks to
it over the real HTTP contract. No in-process simulation shortcut is used.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="nist-assembly-referee:ci")
    ap.add_argument("--policy", default=str(ROOT / "baselines" / "baseline.onnx"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpus", default="0,1")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    port = free_port()
    env = dict(os.environ)
    env.update(NIST_POLICY_PATH=str(Path(args.policy).resolve()), PYTHONPATH=str(ROOT / "player"))
    player = subprocess.Popen(
        [sys.executable, str(ROOT / "player" / "launch.py"), "--port", str(port)],
        env=env, stdout=(out / "player.log").open("w"), stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    config = {"episodes_per_stratum": 8, "workers": 2, "record_stride": 8, "records_enabled": True}
    cmd = [
        "docker", "run", "--rm", f"--cpuset-cpus={args.cpus}",
        "--add-host=host.docker.internal:host-gateway",
        "-e", "MATCH_ID=local-image-wall", "-e", f"SEED={args.seed}", "-e", "NUM_PLAYERS=1",
        "-e", f"PLAYER_URLS=http://host.docker.internal:{port}",
        "-e", f"CONFIG_JSON={json.dumps(config)}", "-e", "NIST_HISTORY_DIR=/data/history",
        "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1",
        "-v", f"{out}:/data", args.image,
    ]
    t0 = time.monotonic()
    ready_deadline = t0 + min(60.0, args.timeout / 4)
    try:
        while time.monotonic() < ready_deadline:
            if player.poll() is not None:
                raise RuntimeError(f"player exited with code {player.returncode}")
            try:
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=1.0) as response:
                    if response.status == 200 and json.loads(response.read()).get("ready") is True:
                        break
            except (OSError, ValueError):
                time.sleep(0.1)
        else:
            raise RuntimeError("player did not become ready")
        with (out / "referee.log").open("w") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
        wall = time.monotonic() - t0
        result = json.loads((out / "result.json").read_text())
        episodes = next(row["value"] for row in result["metadata"]["summary"] if row["key"] == "episodes")
        fault = result["metadata"].get("details", {}).get("player_fault")
        report = {
            "built_image": args.image,
            "docker_returncode": proc.returncode,
            "wall_s": wall,
            "episodes": episodes,
            "raw_score": result["raw_scores"][0],
            "terminal_reason": result["terminal_reason"],
            "player_fault": fault,
            "cpuset": args.cpus,
            "under_600s": wall < 600.0,
        }
        (out / "wall.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
        if proc.returncode != 0 or episodes != 64 or fault is not None or wall >= 600.0:
            return 1
        return 0
    finally:
        if not (out / "wall.json").exists():
            (out / "wall.json").write_text(json.dumps({
                "built_image": args.image,
                "wall_s": time.monotonic() - t0,
                "cpuset": args.cpus,
                "under_600s": False,
                "error": "player/referee did not produce a complete result; inspect logs",
            }, indent=2))
        if player.poll() is None:
            os.killpg(player.pid, signal.SIGTERM)
            try:
                player.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(player.pid, signal.SIGKILL)
                player.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
