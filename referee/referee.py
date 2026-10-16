"""nist_lite REFEREE: runs the NIST-lite assembly simulation and scores the player's policy.

Solo competition (NUM_PLAYERS=1). Every episode of the round is derived from the platform
SEED. All episodes are stepped in lockstep at 25 Hz; at each control step the player gets one
batched observation for the episodes still running and returns one action per episode.

raw_score = assembled episodes / total episodes (fixed denominator).
"""

from __future__ import annotations

import http.client
import json
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from gym_v1.client import PlayerClient, PlayerError
from gym_v1.referee import GameResult, Referee, RefereeContext

from nist_sim.conditions import sample_episodes
from nist_sim.env import ACT_DIM, CONTROL_HZ, OBS_DIM, Episode

RECORD_FORMAT = "nist_lite_record/1"
HISTORY_DIR = Path(os.environ.get("NIST_HISTORY_DIR", "/data/history"))
DEFAULTS = {
    "episodes_per_stratum": 16,
    "deadline_ms": 500,
    "workers": 2,
    "record_stride": 16,
    "records_enabled": True,
}


# ---- worker processes: each owns a fixed slice of episodes ---------------------------------
def _worker(conn, seed: int, per_stratum: int, indices: list[int], stride: int, split: str):
    _, conds = sample_episodes(seed, per_stratum, split)
    eps = {}
    noise = {}
    traj = {}
    for i in indices:
        ss = np.random.SeedSequence([seed, 0x5245, i])
        r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
        eps[i] = Episode(conds[i], r1)
        noise[i] = r2
        if stride and i % stride == 0:
            traj[i] = []
    while True:
        msg = conn.recv()
        if msg[0] == "obs":
            ids = [i for i in indices if not eps[i].done]
            obs = np.stack([eps[i].observe(noise[i]) for i in ids]) if ids else np.zeros((0, OBS_DIM), np.float32)
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
                    p, q = e.part_pose()
                    traj[i].append({
                        "t": e.steps, "part_pos": np.round(p, 5).tolist(), "part_quat": np.round(q, 4).tolist(),
                        "hand": np.round(e.data.qpos[0:4], 5).tolist(),
                        "action": None if a is None else np.round(np.clip(a, -1, 1), 3).tolist(),
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
                rows.append({
                    "format": RECORD_FORMAT, "episode": i, "conditions": conds[i].public(),
                    "success": bool(e.success), "success_step": e.success_step, "steps": e.steps,
                    "terminal_reason": e.terminal_reason,
                    "mean_sq_action": round(e.action_l2 / max(1, e.steps), 5),
                    "score": 1.0 if e.success else 0.0,
                })
            conn.send((rows, traj))
            return


def _validate_actions(raw: Any, n: int) -> list[np.ndarray | None] | None:
    """Returns one action per row, None for an unusable row; None overall if the batch is unusable."""
    if not isinstance(raw, list) or len(raw) != n:
        return None
    out: list[np.ndarray | None] = []
    for row in raw:
        if (not isinstance(row, list) or len(row) != ACT_DIM
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in row)):
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
            p = ctx_mp.Process(target=_worker, args=(b, ctx.seed, per, sl, stride, split), daemon=True)
            p.start()
            conns.append(a)
            procs.append(p)

        player = players[0]
        player_fault = None
        try:
            player.reset(match_id=ctx.match_id, player_index=0, seed=0,
                         config={"obs_dim": OBS_DIM, "act_dim": ACT_DIM, "control_hz": CONTROL_HZ})
        except (PlayerError, OSError, http.client.HTTPException) as e:
            # The vendored client only wraps URLError/TimeoutError; a player that dies mid-response
            # (RemoteDisconnected, IncompleteRead, connection reset) is still the player's fault.
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
        raw_score = successes / n  # fixed denominator: every episode counts
        self._write_records(rows, traj, round_ranges)

        by_type: dict[str, list[int]] = {}
        by_task: dict[str, list[int]] = {}
        for r in rows:
            c = r["conditions"]
            by_type.setdefault(c["reset_type"], []).append(int(r["success"]))
            by_task.setdefault(f'{c["shape"]}_{round(c["diameter"]*1000)}mm', []).append(int(r["success"]))
        # Envelope follows the platform's StandardEvalMetadata: formats from
        # {number, percent, duration_s, score, count, text}, unit types from
        # {match, game, task, subset, instance, completion, other}, and every `metrics` entry
        # names the unit it belongs to.
        summary = [
            {"key": "success_rate", "label": "Assembly success rate", "value": raw_score, "format": "percent"},
            {"key": "episodes", "label": "Episodes", "value": n, "format": "count"},
        ] + [
            {"key": f"success_{k}", "label": f"Success: {k}", "value": sum(v) / len(v), "format": "percent"}
            for k, v in sorted(by_type.items())
        ] + [
            {"key": f"success_{k}", "label": f"Success: {k.replace('_', ' ')}", "value": sum(v) / len(v),
             "format": "percent"}
            for k, v in sorted(by_task.items())
        ]

        def _task(r):
            return f'{r["conditions"]["shape"]}_{round(r["conditions"]["diameter"]*1000)}mm'

        metadata = {
            "schema_version": 1,
            "summary": summary,
            "units": [
                {"id": f"ep-{r['episode']}", "type": "instance", "index": r["episode"],
                 "label": f'{r["conditions"]["shape"]} {round(r["conditions"]["diameter"]*1000)}mm / {r["conditions"]["reset_type"]}',
                 "category": _task(r), "steps": r["steps"],
                 "outcome": "assembled" if r["success"] else r["terminal_reason"]}
                for r in rows
            ],
            "metrics": [
                {"key": "assembled", "label": "Assembled", "value": r["score"], "format": "score",
                 "unit_id": f"ep-{r['episode']}"}
                for r in rows
            ],
            "capabilities": {"outcome_breakdown": True, "score_distribution": True, "category_breakdown": True},
            "details": {
                "format": RECORD_FORMAT,
                "round_condition_ranges": round_ranges,
                "player_calls": calls,
                "player_fault": player_fault,
                "terminal_reasons": {k: sum(1 for r in rows if r["terminal_reason"] == k)
                                     for k in sorted({r["terminal_reason"] for r in rows})},
            },
        }
        return GameResult(raw_scores=[raw_score], winner=0 if successes else -1,
                          terminal_reason=reason if player_fault else "scored", steps=calls, metadata=metadata)

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
            result = GameResult(raw_scores=[0.0], winner=-1, terminal_reason="player_not_ready", steps=0,
                                metadata={"details": {"player_fault": "policy server never became ready"}})
        else:
            result = self.play_game(ctx, players)  # unexpected exceptions propagate -> referee failure
        _r.RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _r.RESULT_PATH.write_text(json.dumps(dataclasses.asdict(result)))

    def _write_records(self, rows, traj, round_ranges) -> None:
        try:
            HISTORY_DIR.mkdir(parents=True, exist_ok=True)
            with (HISTORY_DIR / "episodes.jsonl").open("w") as f:
                f.write(json.dumps({"format": RECORD_FORMAT, "kind": "round", "round_condition_ranges": round_ranges}) + "\n")
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            for i, steps in traj.items():
                (HISTORY_DIR / f"traj_{i:05d}.json").write_text(
                    json.dumps({"format": RECORD_FORMAT, "kind": "trajectory", "episode": i, "steps": steps}))
        except Exception as e:  # records are best-effort and never affect the score
            print(f"record write failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    NistLiteReferee().run()
