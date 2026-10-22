"""Fast in-process evaluation over the live mix (no HTTP / Docker; episodes run in parallel).

    python tools/eval_policies.py --policy baselines/baseline.onnx --seeds 1 2 --per 8
    python tools/eval_policies.py --policy refport --seeds 1     # obs-only reference port

``--policy`` is an ONNX path (memoryless, ``[B,184] -> [B,5]``, called with B=1) or one of
``zero``, ``baseline``, ``obs_scripted``, ``refport`` (tools/obs_refport.py) and
``reference`` (privileged; tools/reference_solver.py). Uses the same episode draw and
per-episode RNG streams as the referee (dev build). Prints one JSON line per run.
"""
import argparse, json, os, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import numpy as np

def _init(root):
    sys.path.insert(0, os.path.join(root, "referee")); sys.path.insert(0, os.path.join(root, "tools"))

def run_one(args):
    root, policy, seed, per, i, split = args
    _init(root)
    from nist_sim.conditions import sample_episodes
    import nist_sim.conditions as C
    from nist_sim.env import Episode
    from nist_sim.scoring import episode_score
    _, conds = sample_episodes(seed, per, split)
    c = conds[i]
    ss = C.episode_seed_sequence(seed, i) if hasattr(C, "episode_seed_sequence") else np.random.SeedSequence([seed, 0x5245, i])
    r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
    e = Episode(c, r1)
    if policy == "reference":
        from reference_solver import solve_episode
        r = solve_episode(e)
        sc = r["score"]
    elif policy == "refport":
        from obs_refport import solve_obs_episode
        sc = solve_obs_episode(e, r2)["score"]
    else:
        if policy.endswith(".onnx"):
            import onnxruntime as ort
            so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
            sess = ort.InferenceSession(policy, so, providers=["CPUExecutionProvider"])
            nm = sess.get_inputs()[0].name
            act = lambda o: np.asarray(sess.run(None, {nm: np.round(o, 6).astype(np.float32)[None]})[0]).reshape(-1)[:5]
        elif policy == "zero":
            act = lambda o: np.zeros(5)
        else:
            from nist_sim import policies as P
            pol = {"baseline": P.NearestFirstBaseline, "obs_scripted": P.ObsOnlyScripted}[policy]()
            act = lambda o: pol.act(np.round(o, 6).astype(np.float32))
        while not e.done:
            o = e.observe(r2)
            a = np.asarray(act(o), dtype=float)
            if not np.all(np.isfinite(a)): a = np.zeros(5)
            e.step(np.clip(a, -1, 1))
        seated = e.peg_seated_at_end()
        sc = episode_score(seated, e.peg_first_seat_steps(), c.max_steps, e.action_l2 / max(1, e.steps))
    shapes = [p.shape() for p in c.pegs]
    return dict(seed=seed, i=i, score=float(sc), tier=c.difficulty_tier, k=c.n_pegs,
                seated=[bool(x) for x in e.peg_seated_at_end()], shapes=shapes, term=e.terminal_reason,
                starts=[str(p.start_type) for p in c.pegs])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--policy", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--split", default="live")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    pol = os.path.abspath(a.policy) if a.policy.endswith(".onnx") else a.policy
    jobs = [(a.root, pol, s, a.per, i, a.split) for s in a.seeds for i in range(8 * a.per)]
    t = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(run_one, jobs, chunksize=1))
    per_seed = defaultdict(list); tier = defaultdict(list); shp = defaultdict(list)
    for r in rows:
        per_seed[r["seed"]].append(r["score"]); tier[r["tier"]].append(r["score"])
        for s, ok in zip(r["shapes"], r["seated"]): shp[s].append(ok)
    out = dict(policy=os.path.basename(a.policy), mean=float(np.mean([r["score"] for r in rows])),
               per_seed={s: round(float(np.mean(v)), 4) for s, v in per_seed.items()},
               tier={k: round(float(np.mean(v)), 3) for k, v in tier.items()},
               seat_by_shape={k: round(float(np.mean(v)), 3) for k, v in shp.items()},
               n=len(rows), wall=round(time.time() - t, 1))
    print(json.dumps(out))
    if a.json:
        with open(a.json, "a") as f: f.write(json.dumps({**out, "rows": rows}) + "\n")

if __name__ == "__main__":
    main()
