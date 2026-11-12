"""v0.6 calibration harness: paired per-episode comparison, epsilon-copy reroll test, eval-time projection.

    python tools/calib_v06.py --profile v06 --seeds 1 2 --per 4 \
        --policies reference obs_scripted path/to/policy.onnx path/to/policy.onnx+eps=1e-4 \
        --ref obs_scripted --out /tmp/calib_v06

Policy specs: ``zero``, ``baseline``, ``obs_scripted``, ``compound_scripted`` (288-d compound
obs-only scripted), ``refport`` (obs-only reference port), ``reference`` (privileged), or an
ONNX path. Append ``+eps=X`` to run an *epsilon copy*: the same
policy with a constant ``X`` added to every observation entry, or ``+aeps=X`` to add a fixed
pseudo-random +-X to every action component (closest to a re-exported copy). A copy that behaves the same should
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
    "privileged_min": 0.60,  # v05 task
    "privileged_min_compound": 0.30,  # v06: high ceiling, privileged only has to prove solvability
    "c4_privileged_max": 0.70,  # hardest tier must leave headroom even for the privileged solver
    "scripted_ratio_max": 0.50,  # best non-privileged policy vs privileged
    "scripted_max": 0.35,
    "reroll_identical_min": 0.95,
    "projected_live_s_max": 400.0,
}
SCRIPTED = {"baseline", "obs_scripted", "compound_scripted", "v06_style", "refport"}
# v0.7 Assembly III bars (success criteria from the v0.7-dev brief).
BARS_V07 = {
    # Honest privileged (no step_done / collar_run / _key_hold snaps). Image-calib bars.
    "privileged_min_compound": 0.40,
    "c4_privileged_max": 0.70,
    "scripted_max": 0.25,  # v0.6-style / revealed-king ceiling on v0.7.1
    "obs_scripted_max": 0.15,
    "scripted_ratio_max": 0.55,
    "reroll_identical_min": 0.95,
    "projected_live_s_max": 600.0,  # target; hard ceiling remains platform 1800 s
}


def parse_spec(spec: str) -> tuple[str, float]:
    """``base``, obs offset (``+eps=X``) or action perturbation (``+aeps=X``, returned negative)."""
    base, eps = spec, 0.0
    if "+aeps=" in spec:
        base, e = spec.split("+aeps=", 1)
        eps = -abs(float(e))
    elif "+eps=" in spec:
        base, e = spec.split("+eps=", 1)
        eps = float(e)
    return base, eps


def _base(spec: str) -> str:
    return spec.split("+aeps=")[0].split("+eps=")[0]


def _worker_init(root: str, profile: str) -> None:
    os.environ["NIST_PROFILE"] = profile
    sys.path.insert(0, os.path.join(root, "referee"))
    sys.path.insert(0, os.path.join(root, "tools"))


# Per-worker ONNX session cache (spawn workers; avoid reloading a multi-MB model every episode).
_ONNX_CACHE: dict[str, tuple] = {}


def _make_act(base: str):
    if base.endswith(".onnx"):
        import onnxruntime as ort

        hit = _ONNX_CACHE.get(base)
        if hit is None:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            sess = ort.InferenceSession(base, so, providers=["CPUExecutionProvider"])
            nm = sess.get_inputs()[0].name
            shp = sess.get_inputs()[0].shape
            _ONNX_CACHE[base] = (sess, nm, shp)
            hit = _ONNX_CACHE[base]
        sess, nm, _shp = hit
        return lambda o: np.asarray(sess.run(None, {nm: o.astype(np.float32)[None]})[0]).reshape(-1)[:5]
    if base == "zero":
        return lambda o: np.zeros(5)
    if base == "compound_scripted":
        from compound_obs_scripted import CompoundObsScripted

        pol = CompoundObsScripted()
        return lambda o: pol.act(o.astype(np.float32))
    if base == "v06_style":
        # Handled inside run_compound (privileged-state v0.6 insert/collar/twist port).
        return lambda o: np.zeros(5)
    from nist_sim import policies as P

    pol = {"baseline": P.NearestFirstBaseline, "obs_scripted": P.ObsOnlyScripted}[base]()
    return lambda o: pol.act(o.astype(np.float32))


def run_compound(job: tuple) -> dict:
    """One compound episode (v0.6 or v0.7). 184-input policies see the v0.5-compatible obs prefix."""
    spec, seed, per, i, split = job[:5]
    import nist_sim.conditions as C
    from nist_sim import profile
    from nist_sim.compound import OBS_DIM_V06, CompoundEpisode, compound_score, sample_compound
    from nist_sim.compound_v07 import (
        OBS_DIM_V07,
        AssemblyIIIEpisode,
        compound_score_v07,
        sample_compound_v07,
    )
    from nist_sim.env import OBS_DIM

    base, eps = parse_spec(spec)
    v07 = profile.ACTIVE.name == "v07"
    sample = sample_compound_v07 if v07 else sample_compound
    make_ep = AssemblyIIIEpisode if v07 else CompoundEpisode
    score_fn = compound_score_v07 if v07 else compound_score
    obs_full = OBS_DIM_V07 if v07 else OBS_DIM_V06
    _, conds = sample(seed, per, split)
    c = conds[i]
    r1, r2 = (np.random.default_rng(s) for s in C.episode_seed_sequence(seed, i).spawn(2))
    t0 = time.perf_counter()
    e = make_ep(c, r1)
    if base == "reference":
        from compound_solver import solve_compound

        # v0.7 calibration bar must be honest (no oracle step_done / hold snaps).
        # Pass oracle=True only via the legacy ``reference_oracle`` spec.
        r = solve_compound(e, oracle=False if v07 else True)
    elif base == "reference_oracle":
        from compound_solver import solve_compound

        r = solve_compound(e, oracle=True)
    elif base == "v06_style":
        from v06_style_scripted import solve_v06_style

        r = solve_v06_style(e)
    else:
        act = _make_act(base)
        width = OBS_DIM
        if base == "compound_scripted":
            width = obs_full
        elif base.endswith(".onnx"):
            _sess, _nm, shp = _ONNX_CACHE[base]
            width = obs_full if (len(shp) > 1 and shp[-1] in (OBS_DIM_V06, OBS_DIM_V07)) else OBS_DIM
        obs_eps = eps if eps > 0 else 0.0
        a_rng = np.random.default_rng([seed, i, 0xAE])
        while not e.done:
            o = np.round(e.observe(r2), 6)[:width] + obs_eps
            a = np.asarray(act(o), dtype=float)
            if eps < 0:
                a = a + (-eps) * a_rng.choice([-1.0, 1.0], size=5)
            if not np.all(np.isfinite(a)):
                a = np.zeros(5)
            e.step(np.clip(a, -1, 1))
        r = score_fn(e)
    wall = time.perf_counter() - t0
    return dict(policy=spec, seed=seed, i=i, score=float(r["score"]), wall_s=round(wall, 3), steps=int(e.steps),
                max_steps=int(c.max_steps), k=int(c.n_pegs), tier=c.compound.tier, n_steps=r["n_steps"],
                n_done=r["n_done"], n_in_order=r["n_in_order"], per_kind=r["per_kind"], events=r["events"],
                lost=int(e.parts_lost), term=e.terminal_reason, profile=profile.ACTIVE.name,
                action_quant=profile.ACTIVE.action_quant)


def run_one(job: tuple) -> dict:
    if len(job) > 5 and job[5] == "compound":
        return run_compound(job)
    spec, seed, per, i, split = job[:5]
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
        obs_eps = eps if eps > 0 else 0.0
        a_rng = np.random.default_rng([seed, i, 0xAE])
        while not e.done:
            o = np.round(e.observe(r2), 6) + obs_eps
            a = np.asarray(act(o), dtype=float)
            if eps < 0:
                a = a + (-eps) * a_rng.choice([-1.0, 1.0], size=5)
            if not np.all(np.isfinite(a)):
                a = np.zeros(5)
            e.step(np.clip(a, -1, 1))
        sc = float(episode_score(e.peg_seated_at_end(), e.peg_first_seat_steps(), c.max_steps,
                                 e.action_l2 / max(1, e.steps)))
    wall = time.perf_counter() - t0
    return dict(policy=spec, seed=seed, i=i, score=sc, wall_s=round(wall, 3), steps=int(e.steps),
                max_steps=int(c.max_steps), k=int(c.n_pegs), tier=c.difficulty_tier,
                term=e.terminal_reason, profile=profile.ACTIVE.name,
                action_quant=profile.ACTIVE.action_quant)


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
        if rs and "n_steps" in rs[0]:
            kinds = defaultdict(list)
            for r in rs:
                for kd, v in (r.get("per_kind") or {}).items():
                    if v is not None:
                        kinds[kd].append(v)
            out["policies"][s]["steps_done_frac"] = round(sum(r["n_done"] for r in rs) / max(1, sum(r["n_steps"] for r in rs)), 3)
            out["policies"][s]["in_order_frac"] = round(sum(r["n_in_order"] for r in rs) / max(1, sum(r["n_steps"] for r in rs)), 3)
            out["policies"][s]["kind_credit"] = {kd: round(statistics.fmean(v), 3) for kd, v in sorted(kinds.items())}
            out["policies"][s]["mean_steps"] = round(statistics.fmean(r["steps"] for r in rs), 1)
    if ref and ref in by:
        for s in specs:
            if s != ref:
                out["paired"][s] = paired([by[s][k]["score"] for k in keys], [by[ref][k]["score"] for k in keys])
    for s in specs:
        base, eps = parse_spec(s)
        if eps and base in by:  # obs (+eps) or action (+aeps) copy vs its original
            out["reroll"][s] = paired([by[s][k]["score"] for k in keys], [by[base][k]["score"] for k in keys])
    return out


def bars_table(summ: dict, profile: str = "v06") -> list[tuple[str, str, str, str]]:
    P = summ["policies"]
    rows = []
    compound = any("steps_done_frac" in p for p in P.values())
    bars = BARS_V07 if profile == "v07" else BARS
    if "reference" in P:
        v = P["reference"]["mean"]
        lo = bars.get("privileged_min_compound", bars.get("privileged_min", 0.3)) if compound else bars.get("privileged_min", 0.6)
        rows.append(("privileged reference", f">= {lo}", f"{v:.3f}", "yes" if v >= lo else "NO"))
        if compound:
            top = P["reference"]["tier"].get("c4")
            if top is not None:
                rows.append(("privileged on hardest tier c4 (ceiling headroom)", f"<= {bars['c4_privileged_max']}", f"{top:.3f}",
                             "yes" if top <= bars["c4_privileged_max"] else "NO"))
    scr = [(s, P[s]["mean"]) for s in P if s != "reference" and s != "zero" and not parse_spec(s)[1]]
    if scr:
        s, v = max(scr, key=lambda t: t[1])
        if profile == "v07":
            if "v06_style" in P:
                v06s = P["v06_style"]["mean"]
                rows.append(("v0.6-style scripted (insert/collar/twist port)", f"<= {bars['scripted_max']}",
                             f"{v06s:.3f}", "yes" if v06s <= bars["scripted_max"] else "NO"))
            obs_names = [n for n in ("compound_scripted", "obs_scripted") if n in P]
            if obs_names:
                on, ov = max(((n, P[n]["mean"]) for n in obs_names), key=lambda t: t[1])
                lim = bars.get("obs_scripted_max", 0.15)
                rows.append((f"obs-only scripted ({on})", f"<= {lim}", f"{ov:.3f}",
                             "yes" if ov <= lim else "NO"))
            rows.append((f"best non-privileged ({os.path.basename(s)})", f"<= {bars['scripted_max']}",
                         f"{v:.3f}", "yes" if v <= bars["scripted_max"] else "NO"))
            if "reference" in P and P["reference"]["mean"] > 0:
                rows.append(("scripted below privileged", f"< reference {P['reference']['mean']:.3f}",
                             f"{v:.3f}", "yes" if v < P["reference"]["mean"] - 0.15 else "NO"))
        elif compound and "reference" in P:
            lim = bars["scripted_ratio_max"] * P["reference"]["mean"]
            rows.append((f"best non-privileged ({os.path.basename(s)})", f"<= {bars['scripted_ratio_max']} x privileged = {lim:.3f}",
                         f"{v:.3f}", "yes" if v <= lim else "NO"))
        else:
            rows.append((f"best scripted ({os.path.basename(s)})", f"<= {bars['scripted_max']}", f"{v:.3f}", "yes" if v <= bars["scripted_max"] else "NO"))
    for s, r in summ["reroll"].items():
        f = r["identical_frac"]
        rows.append((f"epsilon copy {s}: identical episodes", f">= {bars['reroll_identical_min']}", f"{f:.2f}",
                     "yes" if f >= bars["reroll_identical_min"] else "NO"))
    worst = max((p["projected_live_s"] for p in P.values()), default=float("nan"))
    rows.append(("projected live eval time (slowest policy)", f"<= {bars['projected_live_s_max']:.0f} s", f"{worst:.0f} s",
                 "yes" if worst <= bars["projected_live_s_max"] else "NO"))
    return rows


def write_md(path: Path, summ: dict, args) -> None:
    L = [f"# v0.{args.profile[1:]} calibration run (profile `{args.profile}`, action quant {os.environ.get('NIST_ACTION_QUANT') or 'profile default'}, physics dt {os.environ.get('NIST_PHYSICS_DT') or 'profile default'})", "",
         f"Seeds {args.seeds}, {args.per}/stratum, split `{args.split}`, {summ['episodes']} paired episodes. "
         f"Projection: {args.live_eps} live episodes on {args.cpus} CPU x prod factor {args.prod_factor}.", "",
         "| Policy | mean | by tier | mean s/episode | projected live s |", "|---|---|---|---|---|"]
    for s, p in summ["policies"].items():
        tiers = " / ".join(f"{k if k.startswith('c') else k[0]} {v:.2f}" for k, v in p["tier"].items())
        L.append(f"| {s} | **{p['mean']:.3f}** | {tiers} | {p['mean_wall_s']:.2f} | {p['projected_live_s']:.0f} |")
    comp = [(s, p) for s, p in summ["policies"].items() if "steps_done_frac" in p]
    if comp:
        L += ["", "Compound detail:", "", "| Policy | steps done | in order | credit by kind | mean control steps |", "|---|---|---|---|---|"]
        for s, p in comp:
            kc = ", ".join(f"{k} {v:.2f}" for k, v in p["kind_credit"].items())
            L.append(f"| {s} | {p['steps_done_frac']:.2f} | {p['in_order_frac']:.2f} | {kc} | {p['mean_steps']:.0f} |")
    if summ["paired"]:
        L += ["", f"Paired vs `{args.ref}` (same episodes):", "", "| Policy | delta | paired SE | z | identical episodes |", "|---|---|---|---|---|"]
        for s, r in summ["paired"].items():
            L.append(f"| {s} | {r['delta']:+.4f} | {r['se']:.4f} | {r['z']:.2f} | {r['identical_frac']:.2f} |")
    if summ["reroll"]:
        L += ["", "Epsilon-copy reroll test (copy vs original, same episodes):", "", "| Copy | delta | paired SE | identical episodes |", "|---|---|---|---|"]
        for s, r in summ["reroll"].items():
            L.append(f"| {s} | {r['delta']:+.4f} | {r['se']:.4f} | {r['identical_frac']:.2f} |")
    L += ["", "Release bars:", "", "| Bar | Target | Measured | Met |", "|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in bars_table(summ, args.profile)]
    path.write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="v05", choices=["v05", "v06", "v07"])
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--task", choices=["v05", "compound"], default=None, help="default: compound for v06/v07, else v05")
    ap.add_argument("--episodes", type=int, default=0, help="only the first N episodes of each seed (0 = all)")
    ap.add_argument("--ref", default=None, help="policy spec for paired deltas")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--per", type=int, default=4)
    ap.add_argument("--split", default="live")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--live-eps", type=int, default=128)
    ap.add_argument("--cpus", type=int, default=2)
    ap.add_argument("--prod-factor", type=float, default=0.55,
                    help="live seconds per box second. Box s/episode is measured with all workers busy and B=1 ONNX "
                         "calls; v0.5.x king-class policies: ~17 s/episode here vs ~400-660 s per 128-episode live "
                         "round on 2 CPU -> ~0.55. Recalibrate after each smoke run.")
    ap.add_argument("--action-quant", type=float, default=None, help="dev override of the profile's action quantization")
    ap.add_argument("--physics-dt", type=float, default=None, help="dev override of the profile's physics timestep")
    ap.add_argument("--out", default="calib_v06_out")
    a = ap.parse_args()
    def _abs(s: str) -> str:
        b = _base(s)
        return os.path.abspath(b) + s[len(b):] if b.endswith(".onnx") else s

    specs = [_abs(s) for s in a.policies]
    if a.ref:
        a.ref = _abs(a.ref)
    if a.action_quant is not None:
        os.environ["NIST_ACTION_QUANT"] = str(a.action_quant)
    if a.physics_dt is not None:
        os.environ["NIST_PHYSICS_DT"] = str(a.physics_dt)
    task = a.task or ("compound" if a.profile in ("v06", "v07") else "v05")
    jobs = [(s, seed, a.per, i, a.split, task) for s in specs for seed in a.seeds for i in range(8 * a.per)]
    if a.episodes:
        jobs = [j for j in jobs if j[3] < a.episodes]
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
    md_name = "CALIB_V07.md" if a.profile == "v07" else "CALIB_V06.md"
    write_md(out / md_name, summ, a)
    print((out / md_name).read_text())


if __name__ == "__main__":
    main()
