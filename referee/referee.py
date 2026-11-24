"""nist_lite REFEREE: compound multi-step assembly simulation and scoring (v0.8 Assembly IV).

Solo competition (NUM_PLAYERS=1). Every episode of the round is derived from the
platform SEED plus a private build salt (release builds only). Episodes are stepped in lockstep at 25 Hz; at each control step the
player gets one batched observation for the episodes still running and returns one
action per episode.

raw_score = mean of per-episode scores (fixed denominator). Live default is Assembly IV
(``NIST_PROFILE=v08`` → nist_sim.compound_v08: long procedural blueprints, equal-weight
end-state + perturbation scoring, no answer-leaking key cues). ``NIST_PROFILE=v07`` /
``v06`` remain selectable for A/B. A fault (NaN / wrong shape / late) ends that episode;
credit requires physically verified end states. Observation is 320-d (288-d Assembly III
layout + 32-d blueprint encoding). Records use ``nist_lite_record/4``.

Cost is bounded by sim horizon only (no wall-clock round/episode caps). Same policy +
same seed must score the same on any machine load.
"""

from __future__ import annotations

import http.client
import json
import multiprocessing as mp
import multiprocessing.connection as mp_connection
import os
import select
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Live task is Assembly IV (v0.8). Set before nist_sim is imported (env.py reads physics_dt).
# NIST_PROFILE=v07 / v06 still select prior compound tasks for A/B and regression.
os.environ.setdefault("NIST_PROFILE", "v08")

from gym_v1.client import PlayerClient, PlayerError
from gym_v1.referee import GameResult, Referee, RefereeContext

from nist_sim import profile as _profile
from nist_sim.compound import (
    OBS_DIM_V06,
    STEP_KINDS,
    CompoundEpisode,
    compound_score as compound_score_v06,
    record_events,
    sample_compound,
    step_results,
)
from nist_sim.compound_v07 import (
    OBS_DIM_V07,
    STEP_KINDS_V07,
    AssemblyIIIEpisode,
    compound_score_v07,
    sample_compound_v07,
)
from nist_sim.compound_v08 import (
    OBS_DIM_V08,
    STEP_KINDS_V08,
    AssemblyIVEpisode,
    compound_score_v08,
    compound_record_events_v08,
    compound_record_task_v08,
    sample_compound_v08,
    step_results_v08,
)
from nist_sim.conditions import episode_seed_sequence, sample_episodes
from nist_sim.env import ACT_DIM, CONTROL_HZ, OBS_DIM, Episode
from nist_sim.scoring import episode_score

# Read the env (not profile.ACTIVE): ACTIVE is fixed at first import of nist_sim.profile and
# stays stale across NIST_PROFILE changes / importlib.reload in tests.
_PROFILE_NAME = (os.environ.get("NIST_PROFILE") or "v08").strip().lower()
COMPOUND = _PROFILE_NAME in ("v06", "v07", "v08")
ASSEMBLY_IV = _PROFILE_NAME == "v08"
ASSEMBLY_III = _PROFILE_NAME == "v07"
ASSEMBLY_LABEL = "IV" if ASSEMBLY_IV else ("III" if ASSEMBLY_III else ("II" if COMPOUND else "I"))
if ASSEMBLY_IV:
    OBS_W = OBS_DIM_V08
    compound_score = compound_score_v08
elif ASSEMBLY_III:
    OBS_W = OBS_DIM_V07
    compound_score = compound_score_v07
elif COMPOUND:
    OBS_W = OBS_DIM_V06
    compound_score = compound_score_v06
else:
    OBS_W = OBS_DIM
    compound_score = None  # type: ignore[assignment]
# /4 = compound records with collars + full task + step_results (v0.6.1+).
# /2 = v0.5 single-stage records. (/3 was the brief v0.6.0 layout; superseded.)
RECORD_FORMAT = "nist_lite_record/4" if COMPOUND else "nist_lite_record/2"
HISTORY_DIR = Path(os.environ.get("NIST_HISTORY_DIR", "/data/history"))
DEFAULTS = {
    # Live v0.7: 8/stratum → 64 episodes. Spec timeout is 1800 s; we stay ≪ that.
    "episodes_per_stratum": 8,
    # Per-batch policy budget. Kings are ~40–60 ms/call; 80 ms leaves headroom but
    # stops a runaway ONNX from burning the platform timeout.
    "deadline_ms": 80,
    # No wall-clock round cap: cost is bounded by sim horizon (max_steps) only so
    # scores stay load-invariant. Platform input schema has no wall-clock override.
    "workers": 2,
    "record_stride": 8,
    "records_enabled": True,
}


def _sample(seed: int, per_stratum: int, split: str):
    if ASSEMBLY_IV:
        return sample_compound_v08(seed, per_stratum, split)
    if ASSEMBLY_III:
        return sample_compound_v07(seed, per_stratum, split)
    if COMPOUND:
        return sample_compound(seed, per_stratum, split)
    return sample_episodes(seed, per_stratum, split)


def _make_episode(cond, rng):
    if ASSEMBLY_IV:
        return AssemblyIVEpisode(cond, rng)
    if ASSEMBLY_III:
        return AssemblyIIIEpisode(cond, rng)
    if COMPOUND:
        return CompoundEpisode(cond, rng)
    return Episode(cond, rng)


def _failed_episode_row(cond, episode: int, terminal_reason: str) -> dict:
    """Deterministic zero record; never include exception text or hidden state."""
    public = cond.public()
    if COMPOUND:
        public = dict(public)
        public["collars"] = cond.compound.record_collars()
    n_steps = len(cond.compound.steps) if COMPOUND else 0
    steps = cond.compound.steps if COMPOUND else []
    if ASSEMBLY_IV:
        kinds = STEP_KINDS_V08
    elif COMPOUND:
        # v0.7 falls back to the v0.6 three-kind report for legacy blueprints.
        kinds = (
            STEP_KINDS_V07
            if {s.kind for s in steps} - set(STEP_KINDS)
            else STEP_KINDS
        )
    else:
        kinds = ()
    row = {
        "format": RECORD_FORMAT,
        "episode": episode,
        "conditions": public,
        "success": False,
        "success_step": None,
        "steps": 0,
        "terminal_reason": terminal_reason,
        "failure_reason": terminal_reason,
        "mean_sq_action": 0.0,
        "peg_seated_at_end": [False] * len(cond.pegs),
        "peg_first_seat_step": [None] * len(cond.pegs),
        "pegs_seated": 0,
        "n_pegs": len(cond.pegs),
        "score": 0.0,
    }
    if COMPOUND:
        row.update({
            "task": compound_record_task_v08(cond.compound) if ASSEMBLY_IV else cond.compound.record_task(),
            "tier": cond.compound.tier,
            "steps_done": 0,
            "steps_in_order": 0,
            "n_steps": n_steps,
            "credit_by_kind": {k: (0.0 if any(s.kind == k for s in steps) else None) for k in kinds},
            "step_results": [
                {
                    "done": False,
                    "done_at": None,
                    "in_order": False,
                    "credit": 0.0,
                    **({
                        "precision": 0.0,
                        "stable": False,
                        "partial": {
                            "lift": False, "transport": False, "align": False,
                            "stable": False, "insert": False,
                        },
                        "lift_fraction": 0.0,
                        "transport_fraction": 0.0,
                        "insert_fraction": 0.0,
                    } if ASSEMBLY_IV else {}),
                }
                for _ in steps
            ],
            "events": [],
            "events_fired": 0,
            "parts_lost": 0,
            "nist_profile": _PROFILE_NAME,
        })
    return row


def _valid_observation_reply(reply, indices: list[int]) -> bool:
    if not isinstance(reply, tuple) or len(reply) != 2:
        return False
    ids, obs = reply
    return (
        isinstance(ids, list)
        and all(type(i) is int for i in ids)
        and ids == [i for i in indices if i in ids]
        and isinstance(obs, np.ndarray)
        and obs.dtype.kind == "f"
        and obs.shape == (len(ids), OBS_W)
        and bool(np.all(np.isfinite(obs)))
    )


def _is_record_int(value) -> bool:
    return type(value) is int


def _is_record_number(value) -> bool:
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return bool(np.isfinite(value))
    except (TypeError, ValueError, OverflowError):
        # NumPy rejects Python integers too large for its numeric conversion.
        return False


def _valid_summary_row(row, cond, episode: int) -> bool:
    """Validate everything aggregation/reporting reads before trusting a child."""
    if not isinstance(row, dict) or type(row.get("episode")) is not int or row["episode"] != episode:
        return False
    required = {
        "format": str, "conditions": dict, "success": bool, "steps": int,
        "terminal_reason": str, "mean_sq_action": (int, float),
        "peg_seated_at_end": list, "peg_first_seat_step": list,
        "pegs_seated": int, "n_pegs": int, "score": (int, float),
    }
    if any(key not in row or not isinstance(row[key], typ) for key, typ in required.items()):
        return False
    if (
        row["format"] != RECORD_FORMAT
        or type(row["success"]) is not bool
        or any(type(row[k]) is not int for k in ("steps", "pegs_seated", "n_pegs"))
        or not _is_record_number(row["score"])
        or not _is_record_number(row["mean_sq_action"])
        or not 0 <= row["steps"] <= cond.max_steps
        or row["n_pegs"] != len(cond.pegs)
        or not 0 <= row["pegs_seated"] <= row["n_pegs"]
        or len(row["peg_seated_at_end"]) != row["n_pegs"]
        or len(row["peg_first_seat_step"]) != row["n_pegs"]
        or any(type(v) is not bool for v in row["peg_seated_at_end"])
        or any(v is not None and (not _is_record_int(v) or not 0 <= v <= row["steps"]) for v in row["peg_first_seat_step"])
        or "success_step" not in row
        or row["success_step"] is not None and (
            not _is_record_int(row["success_step"]) or not 0 <= row["success_step"] <= row["steps"]
        )
        or not isinstance(row["conditions"].get("pegs"), list)
        or type(row["conditions"].get("n_pegs")) is not int
        or row["conditions"]["n_pegs"] != row["n_pegs"]
        or len(row["conditions"]["pegs"]) != row["n_pegs"]
        or any(
            not isinstance(p, dict)
            or not isinstance(p.get("shape"), str)
            or not _is_record_number(p.get("size"))
            for p in row["conditions"]["pegs"]
        )
        or not COMPOUND and not isinstance(row["conditions"].get("clutter_bucket"), str)
    ):
        return False
    if not COMPOUND:
        return True
    if "failure_reason" in row and not isinstance(row["failure_reason"], str):
        return False
    compound_keys = {
        "task": dict, "tier": str, "steps_done": int, "steps_in_order": int,
        "n_steps": int, "credit_by_kind": dict, "step_results": list,
        "events": list, "events_fired": int, "parts_lost": int, "nist_profile": str,
    }
    if any(key not in row or not isinstance(row[key], typ) for key, typ in compound_keys.items()):
        return False
    steps = cond.compound.steps
    if ASSEMBLY_IV:
        kinds = STEP_KINDS_V08
    else:
        kinds = STEP_KINDS_V07 if {s.kind for s in steps} - set(STEP_KINDS) else STEP_KINDS
    if (
        any(type(row[k]) is not int for k in ("steps_done", "steps_in_order", "n_steps", "events_fired", "parts_lost"))
        or row["n_steps"] != len(steps)
        or not 0 <= row["steps_done"] <= row["n_steps"]
        or not 0 <= row["steps_in_order"] <= row["n_steps"]
        or row["events_fired"] != len(row["events"])
        or row["events_fired"] < 0
        or row["parts_lost"] < 0
        or len(row["step_results"]) != row["n_steps"]
        or not isinstance(row["task"].get("steps"), list)
        or len(row["task"]["steps"]) != row["n_steps"]
        or set(row["credit_by_kind"]) != set(kinds)
        or any(
            not _is_record_number(row["credit_by_kind"][k])
            for k in kinds if any(s.kind == k for s in steps)
        )
        or any(
            row["credit_by_kind"][k] is not None
            for k in kinds if not any(s.kind == k for s in steps)
        )
    ):
        return False
    for k, sr in enumerate(row["step_results"]):
        if not isinstance(sr, dict):
            return False
        keys = {"done": bool, "in_order": bool, "credit": (int, float)}
        if ASSEMBLY_IV:
            keys.update({
                "precision": (int, float), "stable": bool, "partial": dict,
                "lift_fraction": (int, float), "transport_fraction": (int, float),
                "insert_fraction": (int, float),
            })
        if any(k not in sr or not isinstance(sr[k], typ) for k, typ in keys.items()):
            return False
        if "done_at" not in sr:
            return False
        done_at = sr["done_at"]
        if done_at is not None and not _is_record_int(done_at):
            return False
        if done_at == -1:
            # Only the real Assembly IV misseat path produces this sentinel:
            # completion was undone, and its public event names the same peg.
            if not ASSEMBLY_IV or sr["done"] or steps[k].kind not in ("insert", "rework") or not any(
                isinstance(event, dict)
                and event.get("kind") == "misseat"
                and type(event.get("step")) is int
                and 0 <= event["step"] <= row["steps"]
                and type(event.get("peg")) is int
                and event["peg"] == steps[k].peg
                for event in row["events"]
            ):
                return False
        elif done_at is not None and not 0 <= done_at <= row["steps"]:
            return False
        if (
            any(type(sr[k]) is not bool for k in ("done", "in_order"))
            or any(not _is_record_number(sr[k]) for k, typ in keys.items() if typ == (int, float))
        ):
            return False
        if ASSEMBLY_IV and (
            type(sr["stable"]) is not bool
            or set(sr["partial"]) != {"lift", "transport", "align", "stable", "insert"}
            or any(type(v) is not bool for v in sr["partial"].values())
        ):
            return False
    return True


def _valid_worker_summary(response, conds, indices: list[int], reset_failures=frozenset()) -> bool:
    if not isinstance(response, tuple) or len(response) != 2:
        return False
    rows, traj = response
    if not (
        isinstance(rows, list)
        and len(rows) == len(indices)
        and isinstance(traj, dict)
        and all(type(i) is int and i in indices and isinstance(frames, list) for i, frames in traj.items())
        and all(_valid_summary_row(row, conds[i], i) for row, i in zip(rows, indices))
    ):
        return False
    try:
        # Accepted summaries must be writable as public JSON records, not merely
        # well-shaped enough to aggregate before failing during serialization.
        json.dumps((rows, traj), allow_nan=False)
        expected = {
            i: _failed_episode_row(conds[i], i, "reset_failure")
            for i in indices if i in reset_failures
        }
        return all(
            json.dumps(row, sort_keys=True, allow_nan=False)
            == json.dumps(expected[row["episode"]], sort_keys=True, allow_nan=False)
            for row in rows if row["episode"] in expected
        )
    except (TypeError, ValueError, OverflowError):
        # This also contains array-valued fields whose equality would otherwise
        # raise before an invalid worker summary can become deterministic zeroes.
        return False


def _contain_invalid_summary_rows(response, conds, indices: list[int], reset_failures=frozenset()):
    """Replace only invalid rows when a summary still has a trustworthy envelope.

    Entirely missing or duplicate episode identities make a row-wise mapping unsafe;
    those summaries retain the worker-level failure path. Valid siblings are never
    rewritten, and a malformed reset-failure row becomes a worker_failure zero.
    """
    if not isinstance(response, tuple) or len(response) != 2:
        return None
    rows, traj = response
    if not (
        isinstance(rows, list)
        and len(rows) == len(indices)
        and isinstance(traj, dict)
        and all(type(i) is int and i in indices and isinstance(frames, list) for i, frames in traj.items())
    ):
        return None
    try:
        if not all(isinstance(row, dict) and type(row.get("episode")) is int for row in rows):
            return None
        if [row["episode"] for row in rows] != indices:
            return None
        contained = []
        for row, i in zip(rows, indices):
            expected = (
                _failed_episode_row(conds[i], i, "reset_failure")
                if i in reset_failures else None
            )
            try:
                valid = _valid_summary_row(row, conds[i], i)
                if valid and expected is not None:
                    # The reset-failure handshake identifies this row. Compare it
                    # through JSON, not raw equality, so array values fail closed.
                    valid = json.dumps(row, sort_keys=True, allow_nan=False) == json.dumps(
                        expected, sort_keys=True, allow_nan=False
                    )
                if valid:
                    # An extra non-JSON field is still an invalid episode row;
                    # it must not poison otherwise valid siblings.
                    json.dumps(row, allow_nan=False)
            except (TypeError, ValueError, OverflowError):
                valid = False
            contained.append(
                row if valid else _failed_episode_row(conds[i], i, "worker_failure")
            )
        # The episode identities above remain the only accepted trajectory keys;
        # their frame payloads must be writable with the preserved records.
        json.dumps(traj, allow_nan=False)
        return contained, traj
    except (TypeError, ValueError, OverflowError):
        return None


def _worker_entry(
    conn,
    inherited,
    ready,
    seed: int,
    per_stratum: int,
    indices: list[int],
    stride: int,
    split: str,
):
    # A later fork must not retain a parent's duplex endpoints: such copies can
    # suppress EOF when an earlier worker dies before sending a response. The
    # parent reads this readiness byte before forking again, so all inherited
    # copies are closed first. Episode construction still proceeds in parallel.
    for other in inherited:
        other.close()
    # Drop the inherited parent handle as soon as it is closed; otherwise the
    # local argument list retains an inert descriptor object until this returns.
    inherited.clear()
    try:
        ready.send_bytes(b"\0")
    finally:
        ready.close()
    return _worker(conn, seed, per_stratum, indices, stride, split)


def _worker(conn, seed: int, per_stratum: int, indices: list[int], stride: int, split: str):
    try:
        _, conds = _sample(seed, per_stratum, split)
        eps = {}
        noise = {}
        traj = {}
        reset_failures = set()
        for i in indices:
            ss = episode_seed_sequence(seed, i)
            r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
            try:
                eps[i] = _make_episode(conds[i], r1)
            except Exception:
                # Reset construction is not a player fault. Keep this episode out of
                # the action batch, but include its deterministic zero in the round.
                eps[i] = None
                reset_failures.add(i)
            noise[i] = r2
            if stride and i % stride == 0:
                traj[i] = []
        # Publish contained construction failures before any step or transport
        # failure so a later worker death cannot erase their classification.
        conn.send(("reset_failures", sorted(reset_failures)))
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                return
            if msg[0] == "obs":
                ids = [i for i in indices if eps[i] is not None and not eps[i].done]
                obs = (
                    np.stack([eps[i].observe(noise[i]) for i in ids])
                    if ids
                    else np.zeros((0, OBS_W), np.float32)
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
                        frame = {
                            "t": e.steps,
                            "pegs": pegs,
                            "hand": np.round(e.data.qpos[0:4], 5).tolist(),
                            "action": None if a is None else np.round(np.clip(a, -1, 1), 3).tolist(),
                            "seated": list(e.seated_now),
                        }
                        if COMPOUND:
                            collars = []
                            for cj in range(e.n_collars):
                                cp, cq = e.collar_pose(cj)
                                collars.append(
                                    {"pos": np.round(cp, 5).tolist(), "quat": np.round(cq, 4).tolist()}
                                )
                            frame["collars"] = collars
                            frame["parked"] = list(e.peg_parked)
                            frame["collar_parked"] = list(e.collar_parked)
                        traj[i].append(frame)
                conn.send(True)
            elif msg[0] == "fail_all":
                for i in indices:
                    if eps[i] is not None and not eps[i].done:
                        eps[i].fail(msg[1])
                conn.send(True)
            elif msg[0] == "summary":
                rows = []
                for i in indices:
                    e = eps[i]
                    if e is None:
                        assert i in reset_failures
                        rows.append(_failed_episode_row(conds[i], i, "reset_failure"))
                        continue
                    seated = e.peg_seated_at_end()
                    first = e.peg_first_seat_steps()
                    msa = e.action_l2 / max(1, e.steps)
                    cinfo = compound_score(e) if COMPOUND else None
                    score = cinfo["score"] if COMPOUND else episode_score(seated, first, e.c.max_steps, msa)
                    row = {
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
                    }
                    if COMPOUND:
                        pub = dict(row["conditions"])
                        pub["collars"] = conds[i].compound.record_collars()
                        row["conditions"] = pub
                        row["task"] = compound_record_task_v08(conds[i].compound) if ASSEMBLY_IV else conds[i].compound.record_task()
                        row["tier"] = conds[i].compound.tier
                        row["steps_done"] = cinfo["n_done"]
                        row["steps_in_order"] = cinfo["n_in_order"]
                        row["n_steps"] = cinfo["n_steps"]
                        row["credit_by_kind"] = cinfo["per_kind"]
                        row["step_results"] = step_results_v08(e) if ASSEMBLY_IV else step_results(e)
                        row["events"] = compound_record_events_v08(e.events_fired) if ASSEMBLY_IV else record_events(e)
                        row["events_fired"] = cinfo["events"]
                        row["parts_lost"] = int(e.parts_lost)
                        row["nist_profile"] = _PROFILE_NAME
                    rows.append(row)
                conn.send((rows, traj))
                return
    finally:
        conn.close()


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
    # Bound only an already-stalled kernel pipe. There is no new deadline while
    # a live worker is constructing or simulating before writing its next frame.
    TRANSPORT_STALL_TIMEOUT_S = 5.0

    def play_game(self, ctx: RefereeContext, players: list[PlayerClient]) -> GameResult:
        cfg = {**DEFAULTS, **(ctx.config or {})}
        per = int(cfg["episodes_per_stratum"])
        deadline_ms = int(cfg["deadline_ms"])
        stride = int(cfg["record_stride"]) if cfg["records_enabled"] else 0
        split = os.environ.get("NIST_SPLIT", "live")
        round_ranges, conds = _sample(ctx.seed, per, split)
        n = len(conds)
        max_steps = max(c.max_steps for c in conds)
        workers = max(1, int(cfg["workers"]))
        slices = [list(range(w, n, workers)) for w in range(workers)]

        ctx_mp = mp.get_context("fork")
        worker_info = []
        try:
            for sl in slices:
                parent_conn, child_conn = ctx_mp.Pipe()
                p = None
                try:
                    ready_reader, ready_writer = ctx_mp.Pipe(duplex=False)
                except Exception:
                    child_conn.close()
                    parent_conn.close()
                    raise
                try:
                    try:
                        inherited = [info["conn"] for info in worker_info]
                        p = ctx_mp.Process(
                            target=_worker_entry,
                            args=(
                                child_conn,
                                inherited,
                                ready_writer,
                                ctx.seed,
                                per,
                                sl,
                                stride,
                                split,
                            ),
                            daemon=True,
                        )
                        p.start()
                    finally:
                        # Only the child may retain these endpoints. A retained
                        # readiness writer would itself hide an early worker death.
                        child_conn.close()
                        ready_writer.close()
                    try:
                        # A one-way pipe cannot poll forever after its only writer
                        # exits. Do not impose a deadline on healthy setup.
                        ready = ready_reader.recv_bytes(1)
                    except EOFError:
                        ready = b""
                    finally:
                        ready_reader.close()
                    if ready != b"\0":
                        raise RuntimeError("referee worker exited before readiness")
                    worker_info.append({"conn": parent_conn, "proc": p, "indices": sl})
                    p = None
                except Exception:
                    parent_conn.close()
                    if p is not None and p.pid is not None:
                        if p.is_alive():
                            p.terminate()
                        p.join(timeout=10)
                    raise
        except Exception:
            for info in worker_info:
                info["conn"].close()
                if info["proc"].is_alive():
                    info["proc"].terminate()
            for info in worker_info:
                info["proc"].join(timeout=10)
            raise

        player = players[0]
        player_fault = None
        rows, traj = [], {}
        dead = set()
        idle = set()
        contained_reset_failures = set()
        try:
            # A pipe becoming readable/writable says nothing about completion of
            # its next framed message. Keep incremental buffers and never call
            # the stdlib's blocking recv()/send() on a live worker socket.
            for info in worker_info:
                info["sock"] = socket.socket(fileno=info["conn"].fileno())
                info["sock"].setblocking(False)
                info["recv_buffer"] = bytearray()
                info["recv_expected"] = None
                info["recv_stalled_since"] = None
                info["send_buffer"] = bytearray()
                info["send_stalled_since"] = None

            def close_parent_conn(info) -> None:
                # The socket wrapper only configures nonblocking IO. Relinquish
                # it before Connection closes the descriptor it exclusively owns.
                sock = info.pop("sock", None)
                if sock is not None:
                    sock.detach()
                info["conn"].close()

            def fail_worker(w: int) -> None:
                dead.add(w)
                info = worker_info[w]
                close_parent_conn(info)
                if info["proc"].is_alive():
                    info["proc"].terminate()
                info["proc"].join(timeout=1.0)

            def _take_frame(info):
                """Return a complete pickled frame, False if incomplete, or None if invalid."""
                buf, expected = info["recv_buffer"], info["recv_expected"]
                if expected is None:
                    if len(buf) < 4:
                        return False
                    expected = struct.unpack("!i", buf[:4])[0]
                    del buf[:4]
                    if expected < -1:
                        return None
                    info["recv_expected"] = expected
                if expected == -1:
                    if len(buf) < 8:
                        return False
                    expected = struct.unpack("!Q", buf[:8])[0]
                    del buf[:8]
                    info["recv_expected"] = expected
                if len(buf) < expected:
                    return False
                payload = bytes(buf[:expected])
                del buf[:expected]
                info["recv_expected"] = None
                try:
                    return mp_connection._ForkingPickler.loads(payload)
                except Exception:
                    return None

            def recv_worker(w: int, wait: bool = True):
                """Receive complete frames, draining a dead child's pipe to EOF."""
                info = worker_info[w]
                conn, proc = info["conn"], info["proc"]
                draining = False
                while w not in dead:
                    try:
                        # Always retry a partial frame before polling or reading:
                        # the previous read may have completed it in the buffer.
                        frame = _take_frame(info)
                        if frame is None:
                            break
                        if frame is not False:
                            return frame
                        if not draining and (
                            info["recv_buffer"] or info["recv_expected"] is not None
                        ):
                            if info["recv_stalled_since"] is None:
                                info["recv_stalled_since"] = time.monotonic()
                            if time.monotonic() - info["recv_stalled_since"] >= self.TRANSPORT_STALL_TIMEOUT_S:
                                break
                        chunk = os.read(conn.fileno(), 65536)
                        if not chunk:
                            break
                        info["recv_buffer"].extend(chunk)
                        info["recv_stalled_since"] = None
                        continue
                    except BlockingIOError:
                        pass
                    except (EOFError, OSError):
                        break
                    if not proc.is_alive():
                        # The child can exit after send() while its final frame is
                        # still buffered. Drain until EOF, not a live-writer timeout.
                        # Empty userspace bytes are not EOF: the complete frame may
                        # still be queued in the kernel. Only a zero-byte read is EOF.
                        # A truncated final frame then fails closed.
                        draining = True
                    elif not wait:
                        return False
                    try:
                        select.select([conn.fileno()], [], [], 0.1)
                    except (OSError, ValueError):
                        break
                fail_worker(w)
                return None

            def send_worker(w: int, msg) -> bool:
                if w in dead or (w in idle and msg[0] != "summary"):
                    return False
                info = worker_info[w]
                proc, fd = info["proc"], info["conn"].fileno()
                if not info["send_buffer"]:
                    payload = mp_connection._ForkingPickler.dumps(msg)
                    header = (
                        struct.pack("!iQ", -1, len(payload))
                        if len(payload) > 0x7fffffff
                        else struct.pack("!i", len(payload))
                    )
                    info["send_buffer"].extend(header + payload)
                while w not in dead:
                    try:
                        if info["send_buffer"]:
                            written = os.write(fd, info["send_buffer"])
                            del info["send_buffer"][:written]
                            info["send_stalled_since"] = None
                            if info["send_buffer"]:
                                info["send_stalled_since"] = time.monotonic()
                                raise BlockingIOError
                        return True
                    except BlockingIOError:
                        pass
                    except (BrokenPipeError, EOFError, OSError):
                        break
                    if not proc.is_alive():
                        break
                    if info["send_stalled_since"] is None:
                        info["send_stalled_since"] = time.monotonic()
                    if time.monotonic() - info["send_stalled_since"] >= self.TRANSPORT_STALL_TIMEOUT_S:
                        break
                    try:
                        select.select([], [fd], [], 0.1)
                    except (OSError, ValueError):
                        break
                fail_worker(w)
                return False

            awaiting_startup = set(range(len(worker_info)))
            while awaiting_startup:
                # Wait for readable startup pipes without letting one slow
                # construction hide another worker's death or busy-spinning.
                pending = [worker_info[w]["conn"] for w in sorted(awaiting_startup)]
                try:
                    readable = set(mp_connection.wait(pending, timeout=0.1))
                except (EOFError, OSError):
                    readable = set()
                progress = False
                for w in sorted(awaiting_startup):
                    if worker_info[w]["conn"] not in readable and worker_info[w]["proc"].is_alive():
                        continue
                    initial = recv_worker(w, wait=False)
                    if initial is False:
                        continue
                    awaiting_startup.remove(w)
                    progress = True
                    if (
                        not isinstance(initial, tuple)
                        or len(initial) != 2
                        or initial[0] != "reset_failures"
                        or not isinstance(initial[1], list)
                        or any(type(i) is not int or i not in worker_info[w]["indices"] for i in initial[1])
                        or len(set(initial[1])) != len(initial[1])
                    ):
                        if w not in dead:
                            fail_worker(w)
                        continue
                    contained_reset_failures.update(initial[1])
                if not readable and not progress:
                    # Recheck all liveness even if wait() failed on a closed pipe.
                    for w in sorted(awaiting_startup):
                        if not worker_info[w]["proc"].is_alive():
                            recv_worker(w)

            try:
                player.reset(
                    match_id=ctx.match_id,
                    player_index=0,
                    seed=0,
                    config={"obs_dim": OBS_W, "obs_dim_v05_prefix": OBS_DIM, "act_dim": ACT_DIM, "control_hz": CONTROL_HZ},
                )
            except (PlayerError, OSError, http.client.HTTPException) as e:
                player_fault = f"reset failed: {type(e).__name__}"

            calls = 0
            steps_used = 0
            pending_actions = None
            delivered_workers = set()
            while steps_used < max_steps or (pending_actions is not None and not player_fault):
                if player_fault:
                    break
                if pending_actions is None:
                    # Every live worker starts observing before the parent waits for
                    # any one of them. Preserve worker order in the concatenated batch.
                    observation_workers = [
                        w for w in range(len(worker_info)) if send_worker(w, ("obs",))
                    ]
                    parts = []
                    for w in observation_workers:
                        if (part := recv_worker(w)) is None:
                            continue
                        if not _valid_observation_reply(part, worker_info[w]["indices"]):
                            fail_worker(w)
                            continue
                        parts.append((w, part))
                    ids = [i for _, (p_ids, _) in parts for i in p_ids]
                    if not ids:
                        # An all-reset or already-complete slice has no fail_all
                        # acknowledgment to consume; its next reply is its summary.
                        idle.update(w for w, (p_ids, _) in parts if not p_ids)
                        break
                    obs = np.concatenate([o for _, (_, o) in parts], axis=0)
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
                    pending_actions = {
                        w: {i: by_id[i] for i in p_ids} for w, (p_ids, _) in parts if p_ids
                    }
                    idle.update(w for w, (p_ids, _) in parts if not p_ids)
                    steps_used += 1
                # Send every outstanding action before waiting for acknowledgements.
                # Workers simulate in parallel while the player sees one unchanged,
                # deterministically ordered batch at the next control boundary.
                for w, action_map in pending_actions.items():
                    if w not in delivered_workers:
                        send_worker(w, ("act", action_map))
                for w in pending_actions:
                    if w in delivered_workers or w in dead:
                        continue
                    ack = recv_worker(w)
                    if ack is None:
                        continue
                    if ack is not True:
                        fail_worker(w)
                        continue
                    delivered_workers.add(w)
                if all(w in dead or w in delivered_workers for w in pending_actions):
                    pending_actions = None
                    delivered_workers.clear()
                # A failure leaves only unacknowledged actions pending. Continue
                # those same deliveries before collecting any new observations;
                # this keeps every survivor on one logical control-step boundary.

            reason = "player_fault" if player_fault else "time_limit"
            finalizing_workers = [
                w for w in range(len(worker_info)) if send_worker(w, ("fail_all", reason))
            ]
            for w in finalizing_workers:
                recv_worker(w)
            summary_workers = [
                w for w in range(len(worker_info)) if send_worker(w, ("summary",))
            ]
            for w in summary_workers:
                response = recv_worker(w)
                if response is not None:
                    summary_valid = _valid_worker_summary(
                        response, conds, worker_info[w]["indices"], contained_reset_failures
                    )
                    if not summary_valid:
                        # A malformed episode record is contained to that episode.
                        # An invalid envelope still fails the worker: there is no safe
                        # one-to-one mapping for missing, duplicate, or unreadable rows.
                        response = _contain_invalid_summary_rows(
                            response, conds, worker_info[w]["indices"], contained_reset_failures
                        )
                    if response is None:
                        fail_worker(w)
                        continue
                    r, t = response
                    rows += r
                    traj.update(t)
            for w in sorted(dead):
                rows.extend(
                    _failed_episode_row(
                        conds[i], i,
                        "reset_failure" if i in contained_reset_failures else "worker_failure",
                    )
                    for i in worker_info[w]["indices"]
                )
        finally:
            for info in worker_info:
                close_parent_conn(info)
                if info["proc"].is_alive():
                    info["proc"].terminate()
            for info in worker_info:
                info["proc"].join(timeout=10)
        rows.sort(key=lambda r: r["episode"])
        if len(rows) != n or {r["episode"] for r in rows} != set(range(n)):
            raise RuntimeError("incomplete or duplicate episode records")

        successes = sum(1 for r in rows if r["success"])
        success_rate = successes / n
        pegs_seated = sum(r["pegs_seated"] for r in rows)
        pegs_total = sum(r["n_pegs"] for r in rows)
        peg_fraction = pegs_seated / max(1, pegs_total)
        raw_score = sum(float(r["score"]) for r in rows) / n
        self._write_records(rows, traj, round_ranges)

        by_k: dict[str, list[float]] = {}
        by_size: dict[str, list[int]] = {}
        for r in rows:
            c = r["conditions"]
            by_k.setdefault(r["tier"] if COMPOUND else f'k{c["n_pegs"]}', []).append(float(r["score"]))
            for peg, seated in zip(c["pegs"], r["peg_seated_at_end"]):
                dia = float(peg["size"])
                size_class = "small" if dia < 0.008 else "medium" if dia < 0.012 else "large"
                key = f"{peg['shape']}_{size_class}"
                by_size.setdefault(key, []).append(int(seated))

        summary = [
            {"key": "mean_episode_score", "label": "Mean episode score", "value": raw_score, "format": "score"},
            {"key": "success_rate", "label": "Full-board success rate", "value": success_rate, "format": "percent"},
            {"key": "peg_seat_fraction", "label": "Fraction of pegs seated", "value": peg_fraction, "format": "percent"},
            {"key": "episodes", "label": "Episodes", "value": n, "format": "count"},
        ] + ([
            {"key": "step_done_fraction", "label": "Fraction of task steps done",
             "value": sum(r["steps_done"] for r in rows) / max(1, sum(r["n_steps"] for r in rows)),
             "format": "percent"},
        ] if COMPOUND else []) + [
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
                    "label": (f'{r["tier"]} / {r["n_steps"]} steps / K={r["conditions"]["n_pegs"]}' if COMPOUND
                              else f'K={r["conditions"]["n_pegs"]} / {r["conditions"]["clutter_bucket"]} clutter'),
                    "category": r["tier"] if COMPOUND else f'k{r["conditions"]["n_pegs"]}',
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
                "nist_profile": _PROFILE_NAME,
                "assembly": ASSEMBLY_LABEL,
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
        # Always print the active profile so image/preview logs prove the bake.
        if ASSEMBLY_IV:
            _asm = "IV"
        elif ASSEMBLY_III:
            _asm = "III"
        elif COMPOUND:
            _asm = "II"
        else:
            _asm = "I"
        print(
            f"nist_lite profile={_PROFILE_NAME} assembly={_asm} "
            f"episodes={n} raw_score={raw_score:.4f}",
            flush=True,
        )
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
                            "nist_profile": _PROFILE_NAME,
                            "assembly": ASSEMBLY_LABEL,
                            "round_condition_ranges": round_ranges,
                        }
                    )
                    + "\n"
                )
                for r in rows:
                    row = dict(r)
                    row["nist_profile"] = _PROFILE_NAME
                    f.write(json.dumps(row) + "\n")
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
