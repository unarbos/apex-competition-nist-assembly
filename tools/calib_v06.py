"""v0.6 calibration harness: paired per-episode comparison, epsilon-copy reroll test, eval-time projection.

    python tools/calib_v06.py --profile v06 --seeds 1 2 --per 4 \
        --policies reference obs_scripted path/to/policy.onnx path/to/policy.onnx+eps=1e-4 \
        --ref obs_scripted --out /tmp/calib_v06

Policy specs: ``zero``, ``baseline``, ``obs_scripted``, ``refport`` (obs-only reference port),
``reference`` (privileged), or an ONNX path. Append ``+eps=X`` to run an *epsilon copy*: the same
policy with a constant ``X`` added to every observation entry. A copy that behaves the same should
score the same; the fraction of episodes whose score changes measures how much of a round is
sampling noise rather than skill (the "reroll" share).

Every policy runs on the same episodes (same SEED draw and per-episode RNG streams as the referee).
Outputs ``rows.jsonl`` (one row per policy x episode, with wall time), ``summary.json`` and
``CALIB_V06.md`` (tables + release bars) in ``--out``.

``--profile`` sets ``NIST_PROFILE`` for the worker processes (``v05`` = live task, ``v06`` = dev).
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Release bars (roadmap v0.6). Scores are live-mix means.
BARS = {
    "privileged_min": 0.60,
    "scripted_max": 0.35,
    "reroll_identical_min": 0.95,
    "projected_live_s_max": 400.0,
}
SCRIPTED = {"baseline", "obs_scripted", "refport"}


def parse_spec(spec: str) -> tuple[str, float]:
    base, eps = spec, 0.0
    if "+eps=" in spec:
        base, e = spec.split("+eps=", 1)
        eps = float(e)
    return base, eps


def _worker_init(root: str, profile: str) -> None:
    os.environ["NIST_PROFILE"] = profile
    sys.path.insert(0, os.path.join(root, "referee"))
    sys.path.insert(0, os.path.join(root, "tools"))


def _make_act(base: str):
    if base.endswith(".onnx"):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(base, so, providers=["CPUExecutionProvider"])
        nm = sess.get_inputs()[0].name
        return lambda o: np.asarray(sess.run(None, {nm: o.astype(np.float32)[None]})[0]).reshape(-1)[:5]
    if base == "zero":
        return lambda o: np.zeros(5)
    from nist_sim import policies as P

    pol = {"baseline": P.NearestFirstBaseline, "obs_scripted": P.ObsOnlyScripted}[base]()
    return lambda o: pol.act(o.astype(np.float32))


def run_one(job: tuple) -> dict:
    spec, seed, per, i, split = job
    import nist_sim.conditions as C
    from nist_sim import profile
    from nist_sim.conditions import sample_episodes
    from nist_sim.env import Episode
    from nist_sim.scoring import episode_score

    base, eps = parse_spec(spec)
    _, conds = sample_episodes(seed, per, split)
    c = conds[i]
    r1, r2 = (np.random.default_rng(s) for s in C.episode_seed_sequence(seed, i).spawn(2))
    t0 = time.perf_counter()
    e = Episode(c, r1)
    if base == "reference":
        from reference_solver import solve_episode

        sc = float(solve_episode(e)["score"])
    elif base == "refport":
        from obs_refport import solve_obs_episode

        sc = float(solve_obs_episode(e, r2)["score"])
    else:
        act = _make_act(base)
        while not e.done:
            o = np.round(e.observe(r2), 6) + eps
            a = np.asarray(act(o), dtype=float)
            if not np.all(np.isfinite(a)):
                a = np.zeros(5)
            e.step(np.clip(a, -1, 1))
        sc = float(episode_score(e.peg_seated_at_end(), e.peg_first_seat_steps(), c.max_steps,
                                 e.action_l2 / max(1, e.steps)))
    wall = time.perf_counter() - t0
    return dict(policy=spec, seed=seed, i=i, score=sc, wall_s=round(wall, 3), steps=int(e.steps),
                max_steps=int(c.max_steps), k=int(c.n_pegs), tier=c.difficulty_tier,
                term=e.terminal_reason, profile=profile.ACTIVE.name)


def paired(a: list[float], b: list[float]) -> dict:
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = statistics.fmean(d) if n else 0.0
    se = (statistics.stdev(d) / math.sqrt(n)) if n > 1 else float("nan")
    same = sum(1 for x in d if abs(x) < 1e-9)
    return dict(n=n, delta=mean, se=se, z=(mean / se) if se and se > 0 else float("nan"),
                identical_frac=same / n if n else float("nan"))


def project_live_s(rows: list[dict], live_eps: int, cpus: int, prod_factor: float) -> float:
    """Projected live eval wall time: mean episode cost x live episodes / CPUs x prod slowdown."""
    if not rows:
        return float("nan")
    return statistics.fmean(r["wall_s"] for r in rows) * live_eps / cpus * prod_factor


def summarize(rows: list[dict], specs: list[str], ref: str | None, live_eps: int, cpus: int,
              prod_factor: float) -> dict:
    by = defaultdict(dict)
    for r in rows:
        by[r["policy"]][(r["seed"], r["i"])] = r
    keys = sorted(set.intersection(*(set(v) for v in by.values()))) if by else []
    out: dict = {"episodes": len(keys), "policies": {}, "paired": {}, "reroll": {}}
    for s in specs:
        rs = [by[s][k] for k in keys]
        tiers = defaultdict(list)
        for r in rs:
            tiers[r["tier"]].append(r["score"])
        out["policies"][s] = dict(
            mean=statistics.fmean(r["score"] for r in rs) if rs else float("nan"),
            tier={t: round(statistics.fmean(v), 3) for t, v in sorted(tiers.items())},
            mean_wall_s=round(statistics.fmean(r["wall_s"] for r in rs), 3) if rs else float("nan"),
            projected_live_s=round(project_live_s(rs, live_eps, cpus, prod_factor), 1),
        )
    if ref and ref in by:
        for s in specs:
            if s != ref:
                out["paired"][s] = paired([by[s][k]["score"] for k in keys], [by[ref][k]["score"] for k in keys])
    for s in specs:
        base, eps = parse_spec(s)
        if eps and base in by:
            out["reroll"][s] = paired([by[s][k]["score"] for k in keys], [by[base][k]["score"] for k in keys])
    return out


def bars_table(summ: dict) -> list[tuple[str, str, str, str]]:
    P = summ["policies"]
    rows = []
    if "reference" in P:
        v = P["reference"]["mean"]
        rows.append(("privileged reference", f">= {BARS['privileged_min']}", f"{v:.3f}", "yes" if v >= BARS["privileged_min"] else "NO"))
    scr = [(s, P[s]["mean"]) for s in P if parse_spec(s)[0] in SCRIPTED or "scripted" in s]
    if scr:
        s, v = max(scr, key=lambda t: t[1])
        rows.append((f"best scripted ({s})", f"<= {BARS['scripted_max']}", f"{v:.3f}", "yes" if v <= BARS["scripted_max"] else "NO"))
    for s, r in summ["reroll"].items():
        f = r["identical_frac"]
        rows.append((f"epsilon copy {s}: identical episodes", f">= {BARS['reroll_identical_min']}", f"{f:.2f}",
                     "yes" if f >= BARS["reroll_identical_min"] else "NO"))
    worst = max((p["projected_live_s"] for p in P.values()), default=float("nan"))
    rows.append(("projected live eval time (slowest policy)", f"<= {BARS['projected_live_s_max']:.0f} s", f"{worst:.0f} s",
                 "yes" if worst <= BARS["projected_live_s_max"] else "NO"))
    return rows


def write_md(path: Path, summ: dict, args) -> None:
    L = [f"# v0.6 calibration run (profile `{args.profile}`)", "",
         f"Seeds {args.seeds}, {args.per}/stratum, split `{args.split}`, {summ['episodes']} paired episodes. "
         f"Projection: {args.live_eps} live episodes on {args.cpus} CPU x prod factor {args.prod_factor}.", "",
         "| Policy | mean | by tier | mean s/episode | projected live s |", "|---|---|---|---|---|"]
    for s, p in summ["policies"].items():
        tiers = " / ".join(f"{k[0]} {v:.2f}" for k, v in p["tier"].items())
        L.append(f"| {s} | **{p['mean']:.3f}** | {tiers} | {p['mean_wall_s']:.2f} | {p['projected_live_s']:.0f} |")
    if summ["paired"]:
        L += ["", f"Paired vs `{args.ref}` (same episodes):", "", "| Policy | delta | paired SE | z | identical episodes |", "|---|---|---|---|---|"]
        for s, r in summ["paired"].items():
            L.append(f"| {s} | {r['delta']:+.4f} | {r['se']:.4f} | {r['z']:.2f} | {r['identical_frac']:.2f} |")
    if summ["reroll"]:
        L += ["", "Epsilon-copy reroll test (copy vs original, same episodes):", "", "| Copy | delta | paired SE | identical episodes |", "|---|---|---|---|"]
        for s, r in summ["reroll"].items():
            L.append(f"| {s} | {r['delta']:+.4f} | {r['se']:.4f} | {r['identical_frac']:.2f} |")
    L += ["", "Release bars:", "", "| Bar | Target | Measured | Met |", "|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in bars_table(summ)]
    path.write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="v05", choices=["v05", "v06"])
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--ref", default=None, help="policy spec for paired deltas")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--per", type=int, default=4)
    ap.add_argument("--split", default="live")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--live-eps", type=int, default=128)
    ap.add_argument("--cpus", type=int, default=2)
    ap.add_argument("--prod-factor", type=float, default=2.5,
                    help="prod/box slowdown (v0.5.1: 233 s podman smoke vs ~400-660 s MC staging)")
    ap.add_argument("--out", default="calib_v06_out")
    a = ap.parse_args()
    specs = [os.path.abspath(s.split("+eps=")[0]) + (s[len(s.split("+eps=")[0]):]) if s.split("+eps=")[0].endswith(".onnx") else s
             for s in a.policies]
    if a.ref and a.ref.split("+eps=")[0].endswith(".onnx"):
        a.ref = os.path.abspath(a.ref)
    jobs = [(s, seed, a.per, i, a.split) for s in specs for seed in a.seeds for i in range(8 * a.per)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(a.workers, mp_context=ctx, initializer=_worker_init, initargs=(str(ROOT), a.profile)) as ex:
        rows = list(ex.map(run_one, jobs, chunksize=1))
    with (out / "rows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summ = summarize(rows, specs, a.ref, a.live_eps, a.cpus, a.prod_factor)
    summ["wall_s"] = round(time.time() - t, 1)
    summ["args"] = vars(a)
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    write_md(out / "CALIB_V06.md", summ, a)
    print((out / "CALIB_V06.md").read_text())


if __name__ == "__main__":
    main()
