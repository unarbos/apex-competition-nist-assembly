"""Read nist_lite evaluation records (downloaded artifacts or a local run's out/history/).

    python tools/read_records.py out/history            # per-stratum table
    python tools/read_records.py out/history --episode 32   # one trajectory
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

SUPPORTED_MAJOR = 2


def _check(rec: dict) -> None:
    name, _, major = rec.get("format", "/").partition("/")
    if name != "nist_lite_record" or int(major or 0) > SUPPORTED_MAJOR:
        raise SystemExit(f"unsupported record format: {rec.get('format')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("history_dir")
    ap.add_argument("--episode", type=int)
    a = ap.parse_args()
    d = Path(a.history_dir)
    if a.episode is not None:
        t = json.loads((d / f"traj_{a.episode:05d}.json").read_text())
        _check(t)
        for s in t["steps"]:
            print(s["t"], "seated", s.get("seated"), "hand", s["hand"], "a", s["action"])
        return
    lines = [json.loads(x) for x in (d / "episodes.jsonl").read_text().splitlines() if x.strip()]
    for r in lines:
        _check(r)
    eps = [r for r in lines if "episode" in r]
    agg = defaultdict(lambda: [0.0, 0, 0, 0])  # score_sum, n, pegs_seated, pegs_total
    reasons = defaultdict(int)
    for r in eps:
        c = r["conditions"]
        k = (c["n_pegs"], c["clutter_bucket"])
        agg[k][0] += r["score"]
        agg[k][1] += 1
        agg[k][2] += r.get("pegs_seated", 0)
        agg[k][3] += r.get("n_pegs", c["n_pegs"])
        reasons[r["terminal_reason"]] += 1
    mean_score = sum(r["score"] for r in eps) / max(1, len(eps))
    peg_frac = sum(r.get("pegs_seated", 0) for r in eps) / max(
        1, sum(r.get("n_pegs", 0) for r in eps)
    )
    print(f"episodes: {len(eps)}  mean_score: {mean_score:.4f}  peg_frac: {peg_frac:.4f}")
    print("terminal reasons:", dict(reasons))
    for k in sorted(agg):
        sc, n, ps, pt = agg[k]
        print(
            f"  K={k[0]} clutter={k[1]:4s}  score {sc / n:.3f}  "
            f"pegs {ps}/{pt} ({ps / max(1, pt):.2f})  n={n}"
        )


if __name__ == "__main__":
    main()
