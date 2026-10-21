"""nist_lite REFEREE: multi-peg board simulation and scoring (v0.3.0).

Solo competition (NUM_PLAYERS=1). Every episode of the round is derived from the
platform SEED plus a private build salt (release builds only). Episodes are stepped in lockstep at 25 Hz; at each control step the
player gets one batched observation for the episodes still running and returns one
action per episode.

raw_score = mean of per-episode scores (fixed denominator). Each episode score is
the mean per-peg partial credit (see nist_sim.scoring). A fault (NaN / wrong shape /
late) ends that episode: remaining unseated pegs score 0; already-seated pegs keep
their credit if still seated when the episode stops.
"""

from __future__ import annotations

import http.client
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from gym_v1.client import PlayerClient, PlayerError
from gym_v1.referee import GameResult, Referee, RefereeContext

from nist_sim.conditions import SLOT_SPECS, episode_seed_sequence, sample_episodes
from nist_sim.env import ACT_DIM, CONTROL_HZ, OBS_DIM, Episode
from nist_sim.scoring import episode_score, raw_score_from_episodes

RECORD_FORMAT = "nist_lite_record/3"
HISTORY_DIR = Path(os.environ.get("NIST_HISTORY_DIR", "/data/history"))
DEFAULTS = {
    "episodes_per_stratum": 8,
    "deadline_ms": 500,
    "workers": 2,
    "record_stride": 8,
    "records_enabled": True,
}


def _worker(conn, seed: int, per_stratum: int, indices: list[int], stride: int, split: str):
    _, conds = sample_episodes(seed, per_stratum, split)
    eps = {}
    noise = {}
    traj = {}
    for i in indices:
        ss = episode_seed_sequence(seed, i)
        r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
        eps[i] = Episode(conds[i], r1)
        noise[i] = r2
        if stride and i % stride == 0:
            traj[i] = []
    while True:
        msg = conn.recv()
        if msg[0] == "obs":
            ids = [i for i in indices if not eps[i].done]
            obs = (
                np.stack([eps[i].observe(noise[i]) for i in ids])
                if ids
                else np.zeros((0, OBS_DIM), np.float32)
            )
            conn.send((ids, obs))
        elif msg[0] == "act":
            acts: dict[int, Any] = msg[1]
            for i, a in acts.items():
                e = eps[i]
                if a is None:
                    e.fail("invalid_action")
                else:
                    e.step(a)
                if i in traj:
                    pegs = []
                    for pi in range(e.n_pegs):
                        p, q = e.peg_pose(pi)
                        pegs.append({"pos": np.round(p, 5).tolist(), "quat": np.round(q, 4).tolist()})
                    traj[i].append({
                        "t": e.steps,
                        "pegs": pegs,
                        "hand": np.round(e.data.qpos[0:4], 5).tolist(),
                        "action": None if a is None else np.round(np.clip(a, -1, 1), 3).tolist(),
                        "seated": list(e.seated_now),
                    })
            conn.send(True)
        elif msg[0] == "fail_all":
            for i in indices:
                if not eps[i].done:
                    eps[i].fail(msg[1])
            conn.send(True)
        elif msg[0] == "summary":
            rows = []
            for i in indices:
                e = eps[i]
                seated = e.peg_seated_at_end()
                first = e.peg_first_seat_steps()
                msa = e.action_l2 / max(1, e.steps)
                score = episode_score(seated, first, e.c.max_steps, msa)
                rows.append({
                    "format": RECORD_FORMAT,
                    "episode": i,
                    "conditions": conds[i].public(),
                    "success": bool(e.success),
                    "success_step": e.success_step,
                    "steps": e.steps,
                    "terminal_reason": e.terminal_reason,
                    "mean_sq_action": round(msa, 5),
                    "peg_seated_at_end": seated,
                    "peg_first_seat_step": first,
                    "pegs_seated": int(sum(seated)),
                    "n_pegs": e.n_pegs,
                    "score": score,
                })
            conn.send((rows, traj))
            return


def _validate_actions(raw: Any, n: int) -> list[np.ndarray | None] | None:
    """Returns one action per row, None for an unusable row; None overall if the batch is unusable."""
    if not isinstance(raw, list) or len(raw) != n:
        return None
    out: list[np.ndarray | None] = []
    for row in raw:
        if (
            not isinstance(row, list)
            or len(row) != ACT_DIM
            or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in row)
        ):
            out.append(None)
            continue
        a = np.asarray(row, dtype=np.float64)
        out.append(a if np.all(np.isfinite(a)) else None)
    return out


class NistLiteReferee(Referee):
    def play_game(self, ctx: RefereeContext, players: list[PlayerClient]) -> GameResult:
        cfg = {**DEFAULTS, **(ctx.config or {})}
        per = int(cfg["episodes_per_stratum"])
        deadline_ms = int(cfg["deadline_ms"])
        stride = int(cfg["record_stride"]) if cfg["records_enabled"] else 0
        split = os.environ.get("NIST_SPLIT", "live")
        round_ranges, conds = sample_episodes(ctx.seed, per, split)
        n = len(conds)
        max_steps = max(c.max_steps for c in conds)
        workers = max(1, int(cfg["workers"]))
        slices = [list(range(w, n, workers)) for w in range(workers)]

        ctx_mp = mp.get_context("fork")
        conns, procs = [], []
        for sl in slices:
            a, b = ctx_mp.Pipe()
            p = ctx_mp.Process(
                target=_worker, args=(b, ctx.seed, per, sl, stride, split), daemon=True
            )
            p.start()
            conns.append(a)
            procs.append(p)

        player = players[0]
        player_fault = None
        try:
            player.reset(
                match_id=ctx.match_id,
                player_index=0,
                seed=0,
                config={"obs_dim": OBS_DIM, "act_dim": ACT_DIM, "control_hz": CONTROL_HZ},
            )
        except (PlayerError, OSError, http.client.HTTPException) as e:
            player_fault = f"reset failed: {type(e).__name__}"

        calls = 0
        for _ in range(max_steps):
            if player_fault:
                break
            for c in conns:
                c.send(("obs",))
            parts = [c.recv() for c in conns]
            ids = [i for p_ids, _ in parts for i in p_ids]
            if not ids:
                break
            obs = np.concatenate([o for _, o in parts if len(o)], axis=0)
            try:
                raw = player.act(observation=np.round(obs, 6).tolist(), deadline_ms=deadline_ms)
                calls += 1
            except (PlayerError, OSError, http.client.HTTPException) as e:
                player_fault = f"act failed: {type(e).__name__}"
                break
            except (ValueError, json.JSONDecodeError):
                player_fault = "act returned an unreadable response"
                break
            acts = _validate_actions(raw, len(ids))
            if acts is None:
                player_fault = "act returned a batch of the wrong shape"
                break
            by_id = dict(zip(ids, acts))
            for c, (p_ids, _) in zip(conns, parts):
                c.send(("act", {i: by_id[i] for i in p_ids}))
            for c in conns:
                c.recv()

        reason = "player_fault" if player_fault else "time_limit"
        for c in conns:
            c.send(("fail_all", reason))
        for c in conns:
            c.recv()
        rows, traj = [], {}
        for c in conns:
            c.send(("summary",))
            r, t = c.recv()
            rows += r
            traj.update(t)
        for p in procs:
            p.join(timeout=30)
        rows.sort(key=lambda r: r["episode"])

        successes = sum(1 for r in rows if r["success"])
        success_rate = successes / n
        pegs_seated = sum(r["pegs_seated"] for r in rows)
        pegs_total = sum(r["n_pegs"] for r in rows)
        peg_fraction = pegs_seated / max(1, pegs_total)
        raw_score = raw_score_from_episodes(rows)
        self._write_records(rows, traj, round_ranges)

        by_k: dict[str, list[float]] = {}
        by_size: dict[str, list[int]] = {}
        for r in rows:
            c = r["conditions"]
            by_k.setdefault(f'k{c["n_pegs"]}', []).append(float(r["score"]))
            for peg, seated in zip(c["pegs"], r["peg_seated_at_end"]):
                shape, dia, _, _ = SLOT_SPECS[peg["slot"]]
                key = f"{shape}_{round(dia * 1000)}mm"
                by_size.setdefault(key, []).append(int(seated))

        summary = [
            {"key": "mean_episode_score", "label": "Mean episode score", "value": raw_score, "format": "score"},
            {"key": "success_rate", "label": "Full-board success rate", "value": success_rate, "format": "percent"},
            {"key": "peg_seat_fraction", "label": "Fraction of pegs seated", "value": peg_fraction, "format": "percent"},
            {"key": "episodes", "label": "Episodes", "value": n, "format": "count"},
        ] + [
            {
                "key": f"score_{k}",
                "label": f"Score: {k}",
                "value": sum(v) / len(v),
                "format": "score",
            }
            for k, v in sorted(by_k.items())
        ] + [
            {
                "key": f"seat_{k}",
                "label": f"Seat rate: {k.replace('_', ' ')}",
                "value": sum(v) / len(v),
                "format": "percent",
            }
            for k, v in sorted(by_size.items())
        ]

        metadata = {
            "schema_version": 1,
            "summary": summary,
            "units": [
                {
                    "id": f"ep-{r['episode']}",
                    "type": "instance",
                    "index": r["episode"],
                    "label": f'K={r["conditions"]["n_pegs"]} / {r["conditions"]["clutter_bucket"]} clutter',
                    "category": f'k{r["conditions"]["n_pegs"]}',
                    "steps": r["steps"],
                    "outcome": "assembled" if r["success"] else r["terminal_reason"],
                }
                for r in rows
            ],
            "metrics": [
                {
                    "key": "episode_score",
                    "label": "Episode score",
                    "value": r["score"],
                    "format": "score",
                    "unit_id": f"ep-{r['episode']}",
                }
                for r in rows
            ]
            + [
                {
                    "key": "pegs_seated",
                    "label": "Pegs seated",
                    "value": r["pegs_seated"],
                    "format": "count",
                    "unit_id": f"ep-{r['episode']}",
                }
                for r in rows
            ],
            "capabilities": {
                "outcome_breakdown": True,
                "score_distribution": True,
                "category_breakdown": True,
            },
            "details": {
                "format": RECORD_FORMAT,
                "round_condition_ranges": round_ranges,
                "player_calls": calls,
                "player_fault": player_fault,
                "peg_seat_fraction": peg_fraction,
                "terminal_reasons": {
                    k: sum(1 for r in rows if r["terminal_reason"] == k)
                    for k in sorted({r["terminal_reason"] for r in rows})
                },
            },
        }
        return GameResult(
            raw_scores=[raw_score],
            winner=0 if successes else -1,
            terminal_reason=reason if player_fault else "scored",
            steps=calls,
            metadata=metadata,
        )

    def run(self) -> None:
        """Like Referee.run, but a player that never becomes ready is scored (0), not a referee crash."""
        import dataclasses

        from gym_v1 import referee as _r

        ctx = RefereeContext.from_env()
        players = [PlayerClient(url) for url in ctx.player_urls]
        try:
            for p in players:
                p.wait_until_ready(self.readiness_timeout_s)
        except PlayerError:
            result = GameResult(
                raw_scores=[0.0],
                winner=-1,
                terminal_reason="player_not_ready",
                steps=0,
                metadata={"details": {"player_fault": "policy server never became ready"}},
            )
        else:
            result = self.play_game(ctx, players)
        _r.RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _r.RESULT_PATH.write_text(json.dumps(dataclasses.asdict(result)))

    def _write_records(self, rows, traj, round_ranges) -> None:
        try:
            HISTORY_DIR.mkdir(parents=True, exist_ok=True)
            with (HISTORY_DIR / "episodes.jsonl").open("w") as f:
                f.write(
                    json.dumps(
                        {
                            "format": RECORD_FORMAT,
                            "kind": "round",
                            "round_condition_ranges": round_ranges,
                        }
                    )
                    + "\n"
                )
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            for i, steps in traj.items():
                (HISTORY_DIR / f"traj_{i:05d}.json").write_text(
                    json.dumps(
                        {"format": RECORD_FORMAT, "kind": "trajectory", "episode": i, "steps": steps}
                    )
                )
        except Exception as e:
            print(f"record write failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    NistLiteReferee().run()
