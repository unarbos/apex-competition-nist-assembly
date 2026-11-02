"""v0.6.1 ``nist_lite_record/4`` shape and privacy predicates.

    pytest -q tests/test_v06_records.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))

# Force the live compound profile for this module regardless of prior imports.
os.environ["NIST_PROFILE"] = "v06"

from nist_sim.compound import (  # noqa: E402
    COLLAR_H,
    PARTIAL_CAP,
    CompoundEpisode,
    compound_score,
    record_events,
    sample_compound,
    step_credit,
    step_results,
)
from nist_sim.conditions import episode_seed_sequence  # noqa: E402

ALLOWED_COLLAR = {"peg", "r_in", "r_out", "height", "wave"}
ALLOWED_STEP_KEYS_COMMON = {"kind", "part", "peg", "prereq"}
ALLOWED_STEP_RESULT = {"done", "done_at", "in_order", "credit"}


def _all_floats(obj):
    if isinstance(obj, float):
        yield obj
    elif isinstance(obj, (int, np.integer)) and not isinstance(obj, bool):
        return
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _all_floats(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _all_floats(v)


def test_record_format_is_slash_four():
    # Import referee only after NIST_PROFILE=v06 is set above.
    import referee as ref

    assert ref.COMPOUND
    assert ref.RECORD_FORMAT == "nist_lite_record/4"


def test_compound_public_collars_and_task_shape():
    _, conds = sample_compound(7, 4)
    for c in conds:
        cols = c.compound.record_collars()
        assert all(set(col) == ALLOWED_COLLAR for col in cols)
        for col, raw in zip(cols, c.compound.collars):
            assert col["peg"] == raw.peg
            assert col["height"] == COLLAR_H
            assert col["wave"] == raw.wave
        task = c.compound.record_task()
        assert set(task) >= {"tier", "n_waves", "wave_time", "steps"}
        assert task["tier"] == c.compound.tier
        assert len(task["wave_time"]) == task["n_waves"]
        assert len(task["steps"]) == len(c.compound.steps)
        for row, s in zip(task["steps"], c.compound.steps):
            assert set(ALLOWED_STEP_KEYS_COMMON) <= set(row)
            assert row["kind"] == s.kind
            if s.kind == "twist":
                assert "twist_target" in row and "twist_dir" in row


def test_step_results_align_with_task_and_credit():
    _, conds = sample_compound(3, 2)
    ss = episode_seed_sequence(3, 0)
    r1, _ = (np.random.default_rng(s) for s in ss.spawn(2))
    e = CompoundEpisode(conds[0], r1)
    # Idle a few steps so progress / bookkeeping is defined.
    for _ in range(5):
        if e.done:
            break
        e.step(np.array([0.0, 0.0, 0.0, 0.0, 1.0]))
    rows = step_results(e)
    assert len(rows) == len(e.cs.steps)
    for row, k in zip(rows, range(len(rows))):
        assert set(row) == ALLOWED_STEP_RESULT
        expect = step_credit(
            e.step_done[k], e.step_in_order[k], e.step_done_at[k], e.c.max_steps, e.step_progress[k]
        )
        assert row["credit"] == round(float(expect), 5)
    assert compound_score(e)["n_steps"] == len(rows)


def test_record_events_use_t_not_seed():
    _, conds = sample_compound(5, 4)
    # Find an episode that may fire events (c3/c4 denser); still ok if empty.
    idx = next(i for i, c in enumerate(conds) if c.compound.tier in ("c3", "c4"))
    ss = episode_seed_sequence(5, idx)
    r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
    e = CompoundEpisode(conds[idx], r1)
    for _ in range(40):
        if e.done:
            break
        e.observe(r2)
        e.step(np.array([0.0, 0.0, 0.2, 0.0, 1.0]))
    evs = record_events(e)
    assert isinstance(evs, list)
    for ev in evs:
        assert "t" in ev and "kind" in ev
        assert "step" not in ev or ev.get("t") is not None


def test_v06_public_bundle_hides_biases_and_salt(tmp_path, monkeypatch):
    """conditions.public() + collars + task must not leak biases or reserved ranges."""
    from nist_sim import conditions
    from nist_sim.reserve import reserved_from_secret

    reserved = reserved_from_secret("v06-record-privacy")
    p = tmp_path / "reserved.json"
    p.write_text(json.dumps(reserved))
    monkeypatch.setattr(conditions, "RESERVED_PATH", p)

    _, conds = sample_compound(9, 2)
    for c in conds:
        pub = c.public()
        pub["collars"] = c.compound.record_collars()
        task = c.compound.record_task()
        blob = json.dumps({"conditions": pub, "task": task})
        assert str(reserved["salt"]) not in blob
        hidden = [
            *c.pose_bias_xy, c.pose_bias_yaw, *c.peg_bias_delta_xy,
            c.clearance, c.friction, c.peg_mass, c.gain_scale, c.obs_noise,
            c.dropout_prob, c.bias_drift, c.backlash, c.chamfer,
        ]
        hidden += [x for b in c.slot_bias_list for x in b]
        hidden += [x for b in c.peg_bias_delta_list for x in b]
        hidden += [x for peg in c.pegs for x in (*peg.offset_xy, peg.height, peg.yaw_err)]
        hidden = [h for h in hidden if abs(h) > 1e-9]
        for v in _all_floats({"conditions": pub, "task": task}):
            assert not any(abs(v - h) < 1e-7 for h in hidden), v
        for a, b in reserved["ranges"].values():
            assert not any(abs(v - a) < 1e-9 or abs(v - b) < 1e-9 for v in _all_floats(pub))


def test_referee_worker_writes_record_four_fields():
    """End-to-end worker summary: format /4, collars, step_results, traj collars/parked."""
    # Drive a tiny zero-action summary without a full play_game loop: call worker after
    # stepping via obs/act would be heavy; instead build rows the same way as summary
    # for a fresh unstepped episode (valid empty traj / zero progress).
    os.environ["NIST_PROFILE"] = "v06"
    import importlib

    import referee as ref

    importlib.reload(ref)
    assert ref.RECORD_FORMAT == "nist_lite_record/4"

    seed, per, idx = 2, 1, 0
    ctx_conn, worker_conn = mp.Pipe()
    proc = mp.Process(target=ref._worker, args=(worker_conn, seed, per, [idx], 1, "live"))
    proc.start()
    try:
        # One obs + one zero act so traj has a frame, then summary.
        ctx_conn.send(("obs",))
        ids, obs = ctx_conn.recv()
        assert ids == [idx]
        assert obs.shape[1] == ref.OBS_W
        act = {idx: np.zeros(5, dtype=np.float64)}
        ctx_conn.send(("act", act))
        assert ctx_conn.recv() is True
        ctx_conn.send(("summary",))
        rows, traj = ctx_conn.recv()
    finally:
        proc.join(timeout=180)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)

    assert len(rows) == 1
    row = rows[0]
    assert row["format"] == "nist_lite_record/4"
    assert "collars" in row["conditions"]
    for col in row["conditions"]["collars"]:
        assert set(col) == ALLOWED_COLLAR
    assert "steps" in row["task"] and "wave_time" in row["task"]
    assert len(row["step_results"]) == row["n_steps"] == len(row["task"]["steps"])
    for sr in row["step_results"]:
        assert set(sr) == ALLOWED_STEP_RESULT
        assert 0.0 <= sr["credit"] <= 1.0
    assert isinstance(row["events"], list)
    assert "events_fired" in row and "parts_lost" in row
    assert idx in traj and len(traj[idx]) >= 1
    frame = traj[idx][0]
    assert "collars" in frame and "parked" in frame and "collar_parked" in frame
    assert len(frame["parked"]) == row["n_pegs"]
    assert len(frame["collar_parked"]) == len(row["conditions"]["collars"])
    assert len(frame["collars"]) == len(row["conditions"]["collars"])


def test_out_of_order_credit_at_least_partial_cap():
    full = step_credit(True, True, 0, 100, 0.0)
    ooo = step_credit(True, False, 0, 100, 0.0)
    assert ooo >= PARTIAL_CAP
    assert ooo == max(PARTIAL_CAP, 0.15 * full)
    assert step_credit(False, False, None, 100, 0.9) == PARTIAL_CAP


def test_extra_precedence_respects_wave_order():
    _, conds = sample_compound(11, 8)
    for c in conds:
        cs = c.compound
        for k, s in enumerate(cs.steps):
            wa = cs.collars[s.part].wave if s.kind == "collar" else cs.peg_wave[s.peg]
            for p in s.prereq:
                pre = cs.steps[p]
                wp = cs.collars[pre.part].wave if pre.kind == "collar" else cs.peg_wave[pre.peg]
                assert wp <= wa, (c.compound.tier, p, k, wp, wa)
