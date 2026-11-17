"""v0.8 Assembly IV: interface, anti-aligned keys, no answer cues, scoring seals."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT / "tools"))

os.environ["NIST_PROFILE"] = "v08"

from nist_sim.compound_v08 import (  # noqa: E402
    KEY_ANTI_ALIGN,
    OBS_DIM_V08,
    AssemblyIVEpisode,
    compound_score_v08,
    sample_compound_v08,
    sample_key_yaw_v08,
    wrap,
)
from nist_sim.conditions import episode_seed_sequence  # noqa: E402


@pytest.fixture(scope="module")
def conds():
    _, c = sample_compound_v08(7, 8, "live")
    return c


def test_blueprint_decoder_preserves_peg_id():
    sys.path.insert(0, str(ROOT / "tools"))
    from obs_scripted_v08 import blueprint_step

    obs = np.zeros(320)
    obs[288 + 4] = 0.1
    obs[288 + 5] = 3 + 0.5  # peg 3, blocked, not done
    assert blueprint_step(obs, 0) == ("insert", 3, True, False)
    obs[288 + 5] = 2 + 0.25  # peg 2, unblocked, done
    assert blueprint_step(obs, 0) == ("insert", 2, False, True)
    for peg in range(8):
        for blocked in (False, True):
            for done in (False, True):
                obs[288 + 5] = peg + 0.5 * blocked + 0.25 * done
                assert blueprint_step(obs, 0) == ("insert", peg, blocked, done)


def test_obs_blueprint_round_trip(conds):
    from obs_scripted_v08 import blueprint_step

    ep = AssemblyIVEpisode(conds[0], np.random.default_rng(8))
    obs = ep.observe(np.random.default_rng(9))
    for k, step in enumerate(ep.cs.steps[:12]):
        assert blueprint_step(obs, k) == (
            step.kind, step.peg, ep.step_blocked(k), ep.step_done[k]
        )


def test_strong_controller_cannot_read_privileged_episode_state(conds):
    """The strong script must run through a fail-closed observation boundary."""
    from obs_strong_v08 import V08ObsEpisode, solve_obs_episode
    from strong_residual_v08 import StrongResidualController
    from obs_boundary_v08 import ObservationBoundary

    ss = episode_seed_sequence(7, 0)
    r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
    boundary = ObservationBoundary(AssemblyIVEpisode(conds[0], r1))
    obs = boundary.observe(r2)
    view = V08ObsEpisode(boundary, r2, initial_obs=obs)
    for name in ("model", "qpos", "board_xy", "peg_body", "peg_parked", "_set_free"):
        with pytest.raises(AttributeError):
            getattr(view, name)
    controller = StrongResidualController(boundary, r2, initial_obs=obs)
    for _ in range(20):
        action = controller.act()
        assert action.shape == (5,)
        assert np.all(np.isfinite(action))
        boundary.step(action)
        if not boundary.done:
            controller.post_step(False, boundary.observe(r2), action=action)
    r1, r2 = (np.random.default_rng(s) for s in episode_seed_sequence(7, 0).spawn(2))
    boundary = ObservationBoundary(AssemblyIVEpisode(conds[0], r1))
    solve_obs_episode(boundary, r2)
    assert boundary.done


def test_strong_script_matches_zero_residual_wrapper(conds):
    from obs_boundary_v08 import ObservationBoundary
    from obs_strong_v08 import ObservedPegView, V08ObsEpisode, _choices, rs
    from strong_residual_v08 import StrongResidualController

    ss = episode_seed_sequence(7, 0)
    r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
    boundary = ObservationBoundary(AssemblyIVEpisode(conds[0], r1))
    obs = boundary.observe(r2)
    view = V08ObsEpisode(boundary, r2, initial_obs=obs)
    actions = []
    while not boundary.done:
        choices = _choices(view, view.attempted)
        if not choices:
            action = np.array([0.0, 0.0, 0.15, 0.0, 1.0])
            actions.append(action)
            boundary.step(action)
            if not boundary.done:
                view._update(obs=boundary.observe(r2))
            continue
        i = choices[0][1]
        view.attempted.add(i)
        script = rs.ScriptedPeg(ObservedPegView(view, i), search_amp=0.006, search_steps=140, swing_grip=-1.0)
        for action in script.pick_upright():
            actions.append(action)
            boundary.step(action)
            if boundary.done:
                break
            view._update(obs=boundary.observe(r2))
        if boundary.done:
            break
        if view.peg_pose(i)[0][2] < 0.043:
            phases = (script.goto([None, None, 0.095], grip=1.0, max_n=24),)
        else:
            script = rs.ScriptedPeg(ObservedPegView(view, i), search_amp=0.006, search_steps=160, swing_grip=-1.0)
            phases = (script.insert(), script.goto([None, None, 0.095], grip=1.0, max_n=25))
        for phase in phases:
            for action in phase:
                actions.append(action)
                boundary.step(action)
                if boundary.done:
                    break
                view._update(obs=boundary.observe(r2))
            if boundary.done:
                break
    r1, r2 = (np.random.default_rng(s) for s in episode_seed_sequence(7, 0).spawn(2))
    boundary = ObservationBoundary(AssemblyIVEpisode(conds[0], r1))
    obs = boundary.observe(r2)
    controller = StrongResidualController(boundary, r2, initial_obs=obs)
    learned_actions = []
    while not boundary.done:
        action = controller.act()
        learned_actions.append(action)
        boundary.step(action)
        if not boundary.done:
            controller.post_step(False, boundary.observe(r2), action=action)
    assert len(learned_actions) == len(actions)
    np.testing.assert_array_equal(np.asarray(learned_actions), np.asarray(actions))


def test_shared_strong_solver_matches_controller(conds):
    from obs_boundary_v08 import ObservationBoundary
    from obs_strong_v08 import solve_obs_episode
    from strong_residual_v08 import StrongResidualController

    ss = episode_seed_sequence(7, 0)
    r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
    boundary = ObservationBoundary(AssemblyIVEpisode(conds[0], r1))
    obs = boundary.observe(r2)
    controller = StrongResidualController(boundary, r2, initial_obs=obs)
    actions = []
    while not boundary.done:
        action = controller.act()
        actions.append(action)
        boundary.step(action)
        if boundary.done:
            break
        obs = boundary.observe(r2)
        controller.post_step(False, obs, action=action)
    r1, r2 = (np.random.default_rng(s) for s in episode_seed_sequence(7, 0).spawn(2))
    boundary = ObservationBoundary(AssemblyIVEpisode(conds[0], r1))
    recorded = []
    original_step = boundary.step

    def record_action(action):
        recorded.append(np.asarray(action).copy())
        original_step(action)

    boundary.step = record_action
    solve_obs_episode(boundary, r2)
    assert len(recorded) == len(actions)
    np.testing.assert_array_equal(np.asarray(recorded), np.asarray(actions))


def test_contact_cache_matches_direct_scan(conds):
    ep = AssemblyIVEpisode(conds[0], np.random.default_rng(10))
    for _ in range(4):
        ep.step(np.array([0.0, 0.0, -0.4, 0.0, -1.0]))
    m, d = ep.model, ep.data
    expected = set()
    for i in range(d.ncon):
        con = d.contact[i]
        b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
        if b1 in ep.grip_bodies or b2 in ep.grip_bodies:
            expected.update((int(b1), int(b2)))
    ep._touching_bodies_cache = None
    assert {b for b in range(m.nbody) if ep._gripper_touching(b)} == expected
    assert ep._touching_bodies_cache is not None
    ep._set_free(ep.peg_qadr[0], ep.peg_vadr[0], [0.0, 0.0, 0.1], [1.0, 0, 0, 0])
    assert ep._touching_bodies_cache is None


def test_obs_dim_is_320(conds):
    ss = episode_seed_sequence(7, 0)
    r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
    ep = AssemblyIVEpisode(conds[0], r1)
    o = ep.observe(r2)
    assert o.shape == (OBS_DIM_V08,)
    assert OBS_DIM_V08 == 320


def test_v08_conditions_keep_easy_drift_invariant(conds):
    """Compound-specific v08 noise scaling must not break the raw v05 invariant."""
    from nist_sim.conditions import RANGES

    for c in conds:
        if c.difficulty_tier == "easy":
            assert c.bias_drift <= 0.5 * RANGES["bias_drift"][1] + 1e-12


def test_blueprint_longer_than_v07(conds):
    mean = sum(len(c.compound.steps) for c in conds) / len(conds)
    assert mean >= 8.0
    assert max(len(c.compound.steps) for c in conds) >= 12


def test_key_yaw_anti_aligned(conds):
    for c in conds:
        for s in c.compound.steps:
            if s.kind != "key":
                continue
            peg = c.pegs[s.peg]
            slot_yaw = float(peg.spec.yaw + c.board_yaw)
            sym = int(peg.spec.symmetry())
            n = max(1, sym) if sym > 0 else 1
            forbidden = [slot_yaw + (2.0 * np.pi * k) / n for k in range(n)] if sym > 0 else [slot_yaw]
            forbidden = list(forbidden) + [0.0]
            dmin = min(abs(wrap(s.key_yaw - f)) for f in forbidden)
            assert dmin >= KEY_ANTI_ALIGN - 1e-6, (s.key_yaw, dmin, forbidden)


def test_no_closeness_or_detent_in_obs(conds):
    """Key task feats must not publish closeness / retarget pulse / key_tol."""
    ss = episode_seed_sequence(7, 1)
    r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
    ep = AssemblyIVEpisode(conds[1], r1)
    # Force probe+touch on a key peg if present so v0.7 would have published cues.
    key_pegs = [s.peg for s in ep.cs.steps if s.kind == "key"]
    if not key_pegs:
        pytest.skip("no key in episode")
    peg = key_pegs[0]
    ep._peg_touched[peg] = True
    ep._peg_probe_frames[peg] = 99
    ep._key_index_armed[peg] = True
    ep._key_retarget_pulse[peg] = 20
    o = ep.observe(r2)
    task0 = 184
    b = task0 + peg * 6
    # feat2 may be present-mark 0.5; feat3 must stay 0 (no closeness); feat5 not key_tol.
    assert float(o[b + 3]) == 0.0
    assert float(o[b + 5]) in (0.0, 1.0, 2.0)  # collar flag or empty — not tol*10 (~0.4)


def test_equal_weight_scoring_caps_insert_only(conds):
    """Finishing only inserts cannot dominate when the blueprint has many hard steps."""
    ss = episode_seed_sequence(7, 0)
    r1, _ = (np.random.default_rng(s) for s in ss.spawn(2))
    ep = AssemblyIVEpisode(conds[0], r1)
    n = len(ep.cs.steps)
    # Pretend all inserts done in order; nothing else.
    for k, s in enumerate(ep.cs.steps):
        if s.kind == "insert":
            ep.step_done[k] = True
            ep.step_in_order[k] = True
            ep.step_done_at[k] = 10
            ep._stability_ok[k] = True
            ep._precision[k] = 1.0
        else:
            ep.step_done[k] = False
            ep.step_progress[k] = 0.0
    ep._finalized = True
    ep.done = True
    sc = compound_score_v08(ep)["score"]
    n_ins = sum(1 for s in ep.cs.steps if s.kind == "insert")
    # Upper bound: inserts get ≤1.0 each, mean ≤ n_ins/n
    assert sc <= (n_ins / n) + 1e-6
    assert sc < 0.55  # blueprints are dense enough that inserts alone stay mid/low


def test_zero_policy_exactly_zero_on_sampled_round():
    _, sampled = sample_compound_v08(11, 1, "live")
    for i, c in enumerate(sampled):
        ss = episode_seed_sequence(11, i)
        r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
        ep = AssemblyIVEpisode(c, r1)
        assert all(p.start_type not in ("grasped", "near_hole") for p in c.pegs)
        assert not any(ep.step_done)
        assert compound_score_v08(ep)["score"] == 0.0
        while not ep.done:
            ep.step(np.zeros(5))
        assert compound_score_v08(ep)["score"] == 0.0, (i, ep.terminal_reason)


def test_random_and_invalid_actions_cannot_earn_credit(conds):
    for i, action in enumerate((np.ones(5), np.full(5, np.nan), np.full(5, 1e100))):
        ss = episode_seed_sequence(7, i)
        r1, _ = (np.random.default_rng(s) for s in ss.spawn(2))
        ep = AssemblyIVEpisode(conds[i], r1)
        while not ep.done:
            ep.step(action)
        assert compound_score_v08(ep)["score"] == 0.0, (i, ep.terminal_reason)


def test_reset_and_passive_pose_do_not_complete_or_partially_credit():
    _, sampled = sample_compound_v08(23, 1, "live")
    ep = AssemblyIVEpisode(sampled[0], np.random.default_rng(3))
    k = next(k for k, s in enumerate(ep.cs.steps) if s.kind == "insert")
    ep.step_done[k] = True
    ep.step_in_order[k] = True
    ep.step_progress[k] = 1.0  # inherited array must not be a reward backdoor
    ep._finalized = True
    ep.done = True
    assert compound_score_v08(ep)["score"] == 0.0


def test_partial_requires_prerequisites_and_never_uses_legacy_progress():
    from nist_sim.compound_v08 import (
        ALIGN_CREDIT_V08,
        INSERT_CREDIT_V08,
        LIFT_CREDIT_V08,
        PARTIAL_CAP_V08,
        STABLE_CREDIT_V08,
        TRANSPORT_CREDIT_V08,
        step_credit_v08,
    )

    assert step_credit_v08(False, False, None, 100, 1.0) == 0.0
    assert step_credit_v08(False, False, None, 100, {"lift": True}) == LIFT_CREDIT_V08
    assert step_credit_v08(
        False, False, None, 100, {"lift": True, "align": True, "stable": True}
    ) == LIFT_CREDIT_V08 + ALIGN_CREDIT_V08 + STABLE_CREDIT_V08
    assert step_credit_v08(False, False, None, 100, {"transport": True, "transport_fraction": 1.0}) == TRANSPORT_CREDIT_V08
    assert step_credit_v08(False, False, None, 100, {"insert": True, "insert_fraction": 1.0}) == INSERT_CREDIT_V08
    assert step_credit_v08(
        False,
        False,
        None,
        100,
        {
            "lift": True,
            "transport_fraction": 1.0,
            "align": True,
            "stable": True,
            "insert_fraction": 1.0,
        },
    ) == PARTIAL_CAP_V08
    assert step_credit_v08(False, False, None, 100, {"lift_fraction": 0.5}) == 0.5 * LIFT_CREDIT_V08
    assert step_credit_v08(False, False, None, 100, {"lift": True, "lift_fraction": 0.5}) == LIFT_CREDIT_V08
    assert step_credit_v08(False, False, None, 100, {"lift_fraction": 100.0}) == LIFT_CREDIT_V08
    assert step_credit_v08(
        False, False, None, 100, {"lift_fraction": 1.0, "align": True, "stable": True}
    ) == LIFT_CREDIT_V08 + ALIGN_CREDIT_V08 + STABLE_CREDIT_V08
    assert step_credit_v08(True, False, 10, 100, None) == 0.03


def test_dense_lift_requires_held_part_and_settled_reference(conds):
    ep = AssemblyIVEpisode(conds[0], np.random.default_rng(4))
    k = next(k for k, s in enumerate(ep.cs.steps) if s.kind == "insert" and not s.prereq)
    i = ep.cs.steps[k].peg
    body = ep.peg_body[i]
    ep.peg_parked[i] = False
    ep._refresh_part_support_v08()
    reference = ep._part_reference_v08[k]
    ep._gripper_touching = lambda b: b == body  # physical-contact fixture
    ep.data.qpos[4] = ep.data.qpos[5] = 0.002
    ep._action_earned_v08 = True
    ep._update_milestones_v08()
    assert ep._milestones_v08[k].get("lift_fraction", 0.0) == 0.0
    # A vertical wrist/part fixture at spawn height earns nothing. A small
    # measured rise above settled height earns only a bounded partial fraction.
    ep.data.qpos[ep.peg_qadr[i] + 2] = reference + 0.00425
    ep._update_milestones_v08()
    fraction = ep._milestones_v08[k]["lift_fraction"]
    assert 0.0 < fraction == pytest.approx(0.10)
    ep.data.qpos[ep.peg_qadr[i] + 2] = reference
    ep._update_milestones_v08()
    assert ep._milestones_v08[k]["lift_fraction"] == fraction  # best-so-far, not oscillation farm
    ep._gripper_touching = lambda _b: False
    ep.data.qpos[ep.peg_qadr[i] + 2] = reference + 0.00425
    ep._update_milestones_v08()
    assert ep._milestones_v08[k]["lift_fraction"] == fraction
    ep._gripper_touching = lambda b: b == body
    for prereq in ep.cs.steps[k].prereq:
        ep.step_done[prereq] = False
    if ep.cs.steps[k].prereq:
        ep._milestones_v08[k] = dict(lift=False, align=False, stable=False)
        ep._update_milestones_v08()
        assert "lift_fraction" not in ep._milestones_v08[k]
        for prereq in ep.cs.steps[k].prereq:
            ep.step_done[prereq] = True
    ep._action_earned_v08 = False
    ep._milestones_v08[k] = dict(lift=False, align=False, stable=False)
    ep._update_milestones_v08()
    assert compound_score_v08(ep)["score"] == 0.0


def test_dense_lift_is_bounded_and_prerequisite_gated(conds):
    from nist_sim.compound_v08 import LIFT_CREDIT_V08, PARTIAL_CAP_V08, step_results_v08

    ep = AssemblyIVEpisode(conds[0], np.random.default_rng(4))
    ep._action_earned_v08 = True
    k = next(k for k, s in enumerate(ep.cs.steps) if s.kind == "insert")
    ep._milestones_v08[k] = dict(lift=False, align=False, stable=False, lift_fraction=0.5)
    if ep.cs.steps[k].prereq:
        assert step_results_v08(ep)[k]["credit"] == 0.0
        for p in ep.cs.steps[k].prereq:
            ep.step_done[p] = True
    ep.done = True
    ep._finalize_stability()
    row = step_results_v08(ep)[k]
    assert row["lift_fraction"] == 0.5
    assert row["credit"] == 0.5 * LIFT_CREDIT_V08
    assert compound_score_v08(ep)["score"] <= PARTIAL_CAP_V08 / len(ep.cs.steps)


def test_bounded_physical_milestones(conds):
    from nist_sim.compound_v08 import (
        ALIGN_CREDIT_V08,
        LIFT_CREDIT_V08,
        PARTIAL_CAP_V08,
        STABLE_CREDIT_V08,
        step_results_v08,
    )

    ep = AssemblyIVEpisode(conds[0], np.random.default_rng(4))
    ep._action_earned_v08 = True
    k = next(k for k, s in enumerate(ep.cs.steps) if s.kind == "insert" and not s.prereq)
    ep._milestones_v08[k] = dict(lift=True, align=True, stable=True)
    ep.done = True
    ep._finalize_stability()
    result = compound_score_v08(ep)
    assert 0.0 < result["score"] <= PARTIAL_CAP_V08 / len(ep.cs.steps)
    row = step_results_v08(ep)[k]
    assert row["partial"] == {"lift": True, "transport": False, "align": True, "stable": True, "insert": False}
    assert row["lift_fraction"] == 0.0
    assert row["credit"] == LIFT_CREDIT_V08 + ALIGN_CREDIT_V08 + STABLE_CREDIT_V08
    assert row["transport_fraction"] == 0.0
    assert row["insert_fraction"] == 0.0


def test_transport_and_insert_require_physical_predecessors(conds):
    ep = AssemblyIVEpisode(conds[0], np.random.default_rng(5))
    k = next(k for k, s in enumerate(ep.cs.steps) if s.kind == "insert" and not s.prereq)
    i = ep.cs.steps[k].peg
    ep.peg_parked[i] = False
    ep._refresh_part_support_v08()
    p, q = ep.peg_pose(i)
    ep._action_earned_v08 = True
    ep._gripper_touching = lambda b: b == ep.peg_body[i]
    ep.data.qpos[4] = ep.data.qpos[5] = 0.002
    # Wrist/part at reset height cannot earn transport or insertion without a lift.
    ep._update_milestones_v08()
    assert ep._milestones_v08[k].get("transport_fraction", 0.0) == 0.0
    assert ep._milestones_v08[k].get("insert_fraction", 0.0) == 0.0
    ep._milestones_v08[k]["lift"] = True
    ep._update_milestones_v08()
    assert ep._milestones_v08[k].get("insert_fraction", 0.0) == 0.0
    # Transport toward the real hole is bounded and best-so-far. Moving the
    # wrist/part back cannot farm repeated transport or create insertion credit.
    hx, hy = ep.slot_xy[i]
    ep.data.qpos[ep.peg_qadr[i]] = hx - 0.030
    ep.data.qpos[ep.peg_qadr[i] + 1] = hy
    ep.data.qpos[ep.peg_qadr[i] + 2] = 0.045
    ep._update_milestones_v08()
    transport = ep._milestones_v08[k]["transport_fraction"]
    assert 0.0 < transport < 1.0
    ep.data.qpos[ep.peg_qadr[i]] = hx + 0.030
    ep._update_milestones_v08()
    assert ep._milestones_v08[k]["transport_fraction"] == transport
    assert ep._milestones_v08[k].get("insert_fraction", 0.0) == 0.0
    # Released/falling parts cannot receive insertion evidence.
    ep._gripper_touching = lambda _b: False
    ep.data.qpos[ep.peg_qadr[i]] = hx
    ep.data.qpos[ep.peg_qadr[i] + 2] = 0.0
    ep._update_milestones_v08()
    assert ep._milestones_v08[k].get("insert_fraction", 0.0) == 0.0


def test_evolution_zero_residual_and_disjoint_selection():
    sys.path.insert(0, str(ROOT / "tools"))
    from evolve_residual_v08 import Evaluator

    train = Evaluator([101, 102], 2, 2, 0.35, 1)
    validation = Evaluator([201, 202], 2, 2, 0.35, 1)
    test = Evaluator([7], 2, 2, 0.35, 1)
    assert set(train.seed).isdisjoint(validation.seed)
    assert set(train.seed).isdisjoint(test.seed)
    assert set(validation.seed).isdisjoint(test.seed)
    assert train.zero.shape == (2 * OBS_DIM_V08 * 5,)
    assert np.count_nonzero(train.zero) == 0
    assert len(train.conditions) == 4
    assert len(test.conditions) == 2


def test_sample_key_yaw_helper_reachable():
    rng = np.random.default_rng(0)
    _, conds = sample_compound_v08(3, 2, "live")
    for c in conds[:4]:
        for i in range(c.n_pegs):
            yaw = sample_key_yaw_v08(rng, c, i)
            assert abs(yaw) <= 2.95 + 1e-6
