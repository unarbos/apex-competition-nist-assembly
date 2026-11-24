"""v0.8 Assembly IV: interface, anti-aligned keys, no answer cues, scoring seals."""

from __future__ import annotations

import json
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


def test_bc_targets_are_normalized_exactly_once(tmp_path):
    """Collectors save (teacher - script) / scale; training must not divide again."""
    from nist_sim.env import ACT_DIM
    from train_ppo_v08 import load_bc_demos

    X = np.zeros((2, OBS_DIM_V08), np.float32)
    Y = np.full((2, ACT_DIM), 0.35, np.float32)
    path = tmp_path / "public_demos.npz"
    np.savez(path, X=X, Y=Y)
    loaded_X, loaded_Y = load_bc_demos(path)
    np.testing.assert_array_equal(loaded_X, X)
    np.testing.assert_array_equal(loaded_Y, Y)
    bad_path = tmp_path / "bad_demos.npz"
    np.savez(bad_path, X=X, Y=Y + 1.0)
    with pytest.raises(ValueError, match="normalized residuals"):
        load_bc_demos(bad_path)


def test_cem_requires_explicit_selected_parameters(tmp_path, monkeypatch):
    import eval_ladder_v08

    with pytest.raises(ValueError, match="requires --model"):
        eval_ladder_v08.run(7, 8, 1, "cem")
    theta = tmp_path / "theta.json"
    theta.write_text(json.dumps({"theta": [-0.002, 0.028, 0.105]}))
    report = eval_ladder_v08.make_report("cem", [7], 1, 0, [eval_ladder_v08.summary([0.0, 0.0])], str(theta))
    assert report["model_sha256"] is not None


def test_matched_round_resume_refuses_different_simulator(tmp_path, monkeypatch):
    import eval_ladder_v08

    report = eval_ladder_v08.make_report("zero", [7], 1, 0, [])
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps(report))
    monkeypatch.setattr(sys, "argv", [
        "eval_ladder_v08.py", "--kind", "zero", "--seeds", "7", "--episodes", "1",
        "--out", str(tmp_path / "out.json"), "--checkpoint", str(checkpoint),
    ])
    report["simulator_sha256"] = "different-scorer"
    checkpoint.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="different scorer"):
        eval_ladder_v08.main()


def test_matched_round_checkpoint_resumes_and_compares(tmp_path, monkeypatch):
    import eval_ladder_v08

    calls = []

    def fake_round(seed, per, episodes, kind, *args):
        calls.append(seed)
        if seed == 19 and not getattr(fake_round, "allow_last", False):
            raise RuntimeError("interrupted after completed round")
        scores = {7: [0.1, 0.2], 19: [0.3, 0.4]}[seed]
        return {
            **eval_ladder_v08.summary(scores), "seed": seed,
            "episodes": [{"episode": i, "score": s} for i, s in enumerate(scores)],
        }

    monkeypatch.setattr(eval_ladder_v08, "run", fake_round)
    checkpoint = tmp_path / "checkpoint.json"
    out = tmp_path / "report.json"
    argv = ["eval_ladder_v08.py", "--kind", "zero", "--seeds", "7,19", "--episodes", "2", "--out", str(out), "--checkpoint", str(checkpoint)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="interrupted"):
        eval_ladder_v08.main()
    saved = json.loads(checkpoint.read_text())
    assert saved["complete_rounds"] == 1
    assert [r["seed"] for r in saved["rounds"]] == [7]
    assert calls == [7, 19]

    baseline = eval_ladder_v08.make_report("script", [7, 19], 2, 0, [
        {**eval_ladder_v08.summary([0.0, 0.0]), "seed": seed,
             "episodes": [{"episode": i, "score": 0.0} for i in range(2)]}
        for seed in (7, 19)
    ])
    compare = tmp_path / "baseline.json"
    compare.write_text(json.dumps(baseline))
    fake_round.allow_last = True
    monkeypatch.setattr(sys, "argv", argv + ["--compare", str(compare)])
    eval_ladder_v08.main()
    assert calls == [7, 19, 19]
    final = json.loads(out.read_text())
    assert final["complete_rounds"] == 2
    assert final["matched_improvement"]["mean"] == pytest.approx(0.25)
    assert final["matched_improvement"]["ci95"] == pytest.approx([0.15, 0.35])
    eval_ladder_v08.main()
    assert calls == [7, 19, 19]




def test_matched_guards_reject_empty_chunks_layout_and_optimized_mode(tmp_path, monkeypatch):
    import eval_ladder_v08

    with pytest.raises(ValueError, match="positive episode count"):
        eval_ladder_v08.run(7, 8, 0, "zero")
    with pytest.raises(ValueError, match="only 0 conditions"):
        eval_ladder_v08.run(7, 8, 1, "zero", start_episode=10000)

    def fake_run(seed, per, episodes, kind, *args):
        return {**eval_ladder_v08.summary([0.0]), "seed": seed,
                "episodes": [{"episode": 0, "score": 0.0}]}
    monkeypatch.setattr(eval_ladder_v08, "run", fake_run)
    checkpoint = tmp_path / "checkpoint.json"
    out = tmp_path / "report.json"
    argv = ["eval_ladder_v08.py", "--kind", "zero", "--seeds", "7", "--episodes", "1", "--out", str(out), "--checkpoint", str(checkpoint)]
    monkeypatch.setattr(sys, "argv", argv)
    eval_ladder_v08.main()
    monkeypatch.setattr(sys, "argv", argv + ["--per", "4"])
    with pytest.raises(ValueError, match="conditions-per-seed"):
        eval_ladder_v08.main()
    baseline = json.loads(out.read_text())
    baseline["per"] = 4
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps(baseline))
    monkeypatch.setattr(sys, "argv", argv + ["--compare", str(comparison)])
    with pytest.raises(ValueError, match="same conditions per seed"):
        eval_ladder_v08.main()
    import subprocess
    probe = (
        "import sys;sys.path.insert(0,'tools');"
        "from eval_ladder_v08 import require;"
        "require(False,'optimized-mode guard survived')"
    )
    proc = subprocess.run([sys.executable, "-O", "-I", "-c", probe], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "optimized-mode guard survived" in proc.stderr


def test_insert_teacher_matches_observation_script_search(conds):
    from collect_privileged_demos_v08 import INSERT_SEARCH_STEPS_V08, matched_insert_teacher
    from compound_solver import PegView

    ep = AssemblyIVEpisode(conds[0], np.random.default_rng(17))
    teacher = matched_insert_teacher(ep, 0)
    assert isinstance(teacher.e, PegView)
    assert teacher.search_steps == INSERT_SEARCH_STEPS_V08 == 160
    assert teacher.search_amp == 0.006
    assert teacher.swing_grip == -1.0


def test_matched_random_is_uniform_and_observation_only(monkeypatch):
    from eval_ladder_v08 import run

    class GuardedBoundary:
        def __init__(self, ep):
            self.ep = ep
            self.done = False
            self.actions = []

        def observe(self, rng):
            return self.ep.observe(rng).astype(np.float64)

        def step(self, action):
            self.actions.append(np.asarray(action).copy())
            self.ep.step(action)
            self.done = self.ep.done

        def __getattr__(self, name):
            raise AssertionError(f"privileged random read: {name}")

    import eval_ladder_v08
    original = eval_ladder_v08.ObservationBoundary
    boundaries = []
    def make_boundary(ep):
        boundary = GuardedBoundary(ep)
        boundaries.append(boundary)
        return boundary
    eval_ladder_v08.ObservationBoundary = make_boundary
    try:
        result = run(7, 8, 1, "random", start_episode=51)
    finally:
        eval_ladder_v08.ObservationBoundary = original
    assert result["n"] == 1
    assert result["episodes"][0]["episode"] == 51
    assert boundaries and len(boundaries[0].actions) > 1
    actions = np.asarray(boundaries[0].actions)
    assert np.all(actions >= -1.0) and np.all(actions <= 1.0)
    assert not np.array_equal(actions[0], actions[1])
    assert np.all(np.isfinite(actions))


def test_cem_matched_evaluator_uses_observation_boundary(tmp_path):
    theta = tmp_path / "theta.json"
    theta.write_text(json.dumps({"theta": [-0.002, 0.028, 0.105]}))
    from eval_ladder_v08 import run

    class GuardedBoundary:
        def __init__(self, ep):
            self.ep = ep
            self.done = False

        def observe(self, rng):
            return self.ep.observe(rng).astype(np.float64)

        def step(self, action):
            self.ep.step(action)
            self.done = self.ep.done

        def __getattr__(self, name):
            raise AssertionError(f"privileged CEM read: {name}")

    import eval_ladder_v08
    original = eval_ladder_v08.ObservationBoundary
    eval_ladder_v08.ObservationBoundary = GuardedBoundary
    try:
        result = run(7, 8, 1, "cem", str(theta))
    finally:
        eval_ladder_v08.ObservationBoundary = original
    assert result["n"] == 1
    assert result["episodes"][0]["score"] >= 0.0


def test_ppo_matched_evaluator_uses_observation_boundary(monkeypatch, tmp_path):
    from eval_ladder_v08 import run

    class GuardedBoundary:
        def __init__(self, ep):
            self.ep = ep
            self.done = False

        def observe(self, rng):
            return self.ep.observe(rng).astype(np.float64)

        def step(self, action):
            self.ep.step(action)
            self.done = self.ep.done

        def __getattr__(self, name):
            raise AssertionError(f"privileged PPO read: {name}")

    import eval_ladder_v08
    import types
    # The boundary contract does not require the optional PPO dependency.
    import importlib.util
    if importlib.util.find_spec("stable_baselines3") is None:
        pytest.skip("optional Stable-Baselines3 is not installed in referee CI")
    from stable_baselines3 import PPO

    model = type("ZeroPPO", (), {
        "predict": lambda self, obs, deterministic: (np.zeros(5), None),
    })()
    monkeypatch.setattr(PPO, "load", lambda *args, **kwargs: model)
    original = eval_ladder_v08.ObservationBoundary
    eval_ladder_v08.ObservationBoundary = GuardedBoundary
    try:
        result = run(7, 8, 1, "ppo", str(tmp_path / "synthetic-ppo.zip"))
    finally:
        eval_ladder_v08.ObservationBoundary = original
    assert result["n"] == 1
    assert result["episodes"][0]["score"] >= 0.0


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


def test_v08_never_requests_legacy_key_detent_and_preserves_genuine_torque(conds, monkeypatch):
    ep = AssemblyIVEpisode(conds[0], np.random.default_rng(31))
    called = []

    def no_detent(action, *, key_detent=False):
        called.append(key_detent)
        # Contact-torque fixture at the legacy bookkeeping boundary: v0.8 must
        # neither inject the random detent nor revert a jump greater than 0.10.
        ep._wrist_torque[2] = 0.22

    monkeypatch.setattr(ep, "_step_without_key_detent", no_detent)
    ep._wrist_torque[2] = 0.0
    ep.step(np.zeros(5))
    assert called == [False]
    assert ep._wrist_torque[2] == pytest.approx(0.22)


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


def test_key_shared_collar_and_stack_flags_are_preserved(conds):
    ep = AssemblyIVEpisode(conds[0], np.random.default_rng(17))
    shared = {}
    for step in ep.cs.steps:
        if step.kind in ("collar", "stack") and any(
            key.kind == "key" and key.peg == step.peg for key in ep.cs.steps
        ):
            shared[step.kind] = step.peg
    assert set(shared) == {"collar", "stack"}
    for kind, peg in shared.items():
        ep._active_key = next(
            k for k, step in enumerate(ep.cs.steps) if step.kind == "key" and step.peg == peg
        )
        ep._peg_touched[peg] = True
        ep._peg_probe_frames[peg] = 99
        obs = ep.observe(np.random.default_rng(23))
        assert obs[184 + peg * 6 + 5] == (1.0 if kind == "collar" else 2.0)


def test_generator_step_count_is_bounded_by_twenty(conds):
    # Every peg has one insert/rework and at most one of key/twist/screw.
    # There are at most four ring operations across all eight pegs.
    for cond in conds:
        steps = cond.compound.steps
        peg_ops = [s for s in steps if s.kind in ("insert", "rework", "key", "twist", "screw")]
        rings = [s for s in steps if s.kind in ("collar", "stack")]
        assert len({s.peg for s in peg_ops if s.kind in ("insert", "rework")}) <= 8
        assert len({s.peg for s in peg_ops if s.kind in ("key", "twist", "screw")}) <= 8
        assert len(rings) <= 4
        assert len(steps) <= 20


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


@pytest.mark.parametrize("episode_index,n_keys", [(1, 2), (0, 3)])
def test_all_completed_keys_survive_wrist_retarget_and_finalization(conds, episode_index, n_keys):
    """Each completed key is evidence about its own peg, not the final wrist yaw."""
    from nist_sim.compound_v07 import KEY_HOLD, KEY_PROBE_FRAMES

    ep = AssemblyIVEpisode(conds[episode_index], np.random.default_rng(12))
    keys = [k for k, s in enumerate(ep.cs.steps) if s.kind == "key"]
    assert len(keys) == n_keys
    yaws = [ep.cs.steps[k].key_yaw for k in keys]
    assert all(abs(wrap(a - b)) > ep.cs.steps[keys[0]].key_tol for a in yaws for b in yaws if a != b)
    ep._action_earned_v08 = True
    for k, s in enumerate(ep.cs.steps):
        if s.kind != "key":
            ep.step_done[k] = True
            ep.step_in_order[k] = True
    for k in keys:
        s = ep.cs.steps[k]
        ep._peg_probe_frames[s.peg] = KEY_PROBE_FRAMES
        ep._key_index_armed[s.peg] = True
        ep._key_hold[s.peg] = KEY_HOLD
        ep.step_done[k] = True
        ep.step_in_order[k] = True
        ep.step_done_at[k] = 50 + k
        ep._key_precision_v08[k] = 0.9
        hx, hy = ep.slot_xy[s.peg]
        from nist_sim.env import PEG_LEN
        ep._set_free(ep.peg_qadr[s.peg], ep.peg_vadr[s.peg], [hx, hy, PEG_LEN / 2], [1, 0, 0, 0])
        ep.seated_now[s.peg] = True
    # The last key's window cannot retroactively invalidate the first two.
    ep.data.qpos[3] = float(ep.cs.steps[keys[-1]].key_yaw)
    assert all(not ep._key_aligned(ep.cs.steps[k]) for k in keys[:-1])
    assert ep._key_aligned(ep.cs.steps[keys[-1]])
    ep.done = True
    ep._finalize_stability()
    assert all(ep.step_done[k] for k in keys)
    assert all(ep._physical_step_ok(k)[0] for k in keys)
    assert all(ep._stability_ok[k] for k in keys)
    # Completion evidence cannot preserve a key whose peg was pulled out.
    removed = keys[0]
    i = ep.cs.steps[removed].peg
    ep._set_free(ep.peg_qadr[i], ep.peg_vadr[i], [0.15, 0.15, 0.05], [1, 0, 0, 0])
    ep.seated_now[i] = False
    ep._physical_step_ok(removed)
    assert not ep._physical_step_ok(removed)[0]
    ep.step_done[removed] = False
    assert all(ep.step_done[k] for k in keys[1:])
    for k in keys:
        ep.step_done[k] = False
        ep._key_hold[ep.cs.steps[k].peg] = 0
    assert not any(ep._physical_step_ok(k)[0] for k in keys)


def test_dependent_step_partials_do_not_reuse_insert_peg_motion(conds):
    from nist_sim.compound_v08 import _partial_credit_v08

    ep = AssemblyIVEpisode(conds[1], np.random.default_rng(5))
    ep._action_earned_v08 = True
    ks = [k for k, s in enumerate(ep.cs.steps) if s.kind in ("key", "twist")]
    ks = [k for k in ks if not ep.peg_parked[ep.cs.steps[k].peg]]
    assert ks
    screw_ep = AssemblyIVEpisode(conds[1], np.random.default_rng(5))
    screw_ep._action_earned_v08 = True
    for dependent_ep, k in [(ep, k) for k in ks] + [
        (screw_ep, next(k for k, s in enumerate(screw_ep.cs.steps) if s.kind == "screw"))
    ]:
        for p in dependent_ep.cs.steps[k].prereq:
            dependent_ep.step_done[p] = True
        i = dependent_ep.cs.steps[k].peg
        assert not dependent_ep.peg_parked[i]
        # Test-only physical fixture: insert the peg, then re-lift it. Generic
        # insert evidence must not become dependent key/twist/screw progress.
        from nist_sim.env import PEG_LEN
        hx, hy = dependent_ep.slot_xy[i]
        dependent_ep._set_free(dependent_ep.peg_qadr[i], dependent_ep.peg_vadr[i], [hx, hy, PEG_LEN / 2], [1, 0, 0, 0])
        dependent_ep.seated_now[i] = True
        dependent_ep._gripper_touching = lambda b, ep=dependent_ep, i=i: b == ep.peg_body[i]
        dependent_ep.data.qpos[4] = dependent_ep.data.qpos[5] = 0.002
        dependent_ep.data.qpos[dependent_ep.peg_qadr[i] + 2] += 0.012
        dependent_ep._snapshot_parts_v08()
        dependent_ep._part_prev_z_v08[k] -= 0.001
        dependent_ep._update_milestones_v08()
        assert not any(dependent_ep._milestones_v08[k].values())
        # Fail closed if an old checkpoint carries stale generic peg evidence.
        dependent_ep._milestones_v08[k] = dict(
            lift=True, transport=True, transport_fraction=1.0,
            insert=True, insert_fraction=1.0, align=True, stable=True,
        )
        assert _partial_credit_v08(dependent_ep, k) == 0.0


def test_collar_and_stack_stable_and_displaced_finalization(conds, monkeypatch):
    """Stable and displaced collar/ring fixtures verify final physical credit.

    These are test-only finalization fixtures, not controller demonstrations.
    A stable collar is checked without an impulse; the mated stack is checked
    under the default kick. Displaced parts must still lose credit.
    """
    from nist_sim.compound import COLLAR_H
    from nist_sim.env import MAX_OPEN, PEG_LEN
    import nist_sim.compound_v08 as v08

    for kind in ("collar", "stack"):
        monkeypatch.setattr(v08, "PERTURB_XY", 0.0018 if kind == "stack" else 0.0)
        ep = AssemblyIVEpisode(conds[1 if kind == "stack" else 8], np.random.default_rng(23))
        k = next(k for k, s in enumerate(ep.cs.steps) if s.kind == kind and not ep.peg_parked[s.peg])
        j, i = ep.cs.steps[k].part, ep.cs.steps[k].peg
        assert not ep.peg_parked[i]
        hx, hy = ep.slot_xy[i]
        ep._set_free(ep.peg_qadr[i], ep.peg_vadr[i], [hx, hy, PEG_LEN / 2], [1, 0, 0, 0])
        ep.seated_now[i] = True
        ep.collar_parked[j] = False
        ep._set_body_active(ep.collar_body[j], True)
        if kind == "stack":
            ep._pin_stack_on_peg(j)
        else:
            ep._set_free(ep.collar_qadr[j], ep.collar_vadr[j], [hx, hy, 0.020 + COLLAR_H / 2], [1, 0, 0, 0])
        ep.collar_run[j] = 0
        ep._action_earned_v08 = True
        ep.step_done[k] = True
        ep.step_in_order[k] = True
        ep.step_done_at[k] = 1
        ep.done = True
        ep.data.qpos[2] = 0.16
        ep.grip = MAX_OPEN
        ep._apply_ctrl()
        ep._finalize_stability()
        assert ep.step_done[k]
        assert ep._physical_step_ok(k)[0]
        if kind == "collar":
            assert ep.collar_run[j] > 0
        else:
            assert ep._stack_mated[j]

        # A displaced collar/ring cannot retain credit from stale evidence.
        monkeypatch.setattr(v08, "PERTURB_XY", 0.0018)
        ep._set_free(ep.collar_qadr[j], ep.collar_vadr[j], [hx + 0.05, hy, 0.004], [1, 0, 0, 0])
        ep.collar_run[j] = 0
        assert not ep._physical_step_ok(k)[0]
        ep.step_done[k] = True
        ep._finalized = False
        ep._finalize_stability()
        assert not ep.step_done[k]


def test_collar_finalization_uses_current_physical_seat_not_stale_counter(conds, monkeypatch):
    from nist_sim.compound import COLLAR_H, SEATED_HOLD
    from nist_sim.env import MAX_OPEN, PEG_LEN
    import nist_sim.compound_v08 as v08
    monkeypatch.setattr(v08, "PERTURB_XY", 0.0)

    ep = AssemblyIVEpisode(conds[1], np.random.default_rng(17))
    k = next(k for k, s in enumerate(ep.cs.steps) if s.kind == "collar")
    j, i = ep.cs.steps[k].part, ep.cs.steps[k].peg
    hx, hy = ep.slot_xy[i]
    ep.collar_parked[j] = False
    ep._set_body_active(ep.collar_body[j], True)
    ep._set_free(ep.peg_qadr[i], ep.peg_vadr[i], [hx, hy, PEG_LEN / 2], [1, 0, 0, 0])
    ep._set_free(ep.collar_qadr[j], ep.collar_vadr[j], [hx, hy, 0.020 + COLLAR_H / 2], [1, 0, 0, 0])
    ep.seated_now[i] = True
    ep.data.qpos[2] = 0.16
    ep.grip = MAX_OPEN
    ep._apply_ctrl()
    ep.collar_run[j] = 0  # deliberately stale, as in the old settle path
    ep._action_earned_v08 = True
    ep.step_done[k] = True
    ep.step_in_order[k] = True
    ep.step_done_at[k] = 1
    ep.done = True
    ep._finalize_stability()
    assert ep.step_done[k]
    assert ep._physical_step_ok(k)[0]
    assert ep.collar_run[j] >= SEATED_HOLD


def _stack_finalization_fixture(cond, k, rng=23, *, mated=True):
    """Place one earned stack at its measured slot yaw for finalization."""
    import copy
    from nist_sim.env import MAX_OPEN, PEG_LEN, _yaw_quat

    # Reset retries adjust offsets in the condition object. Keep this fixture
    # independent of other boards and kick-direction trials.
    ep = AssemblyIVEpisode(copy.deepcopy(cond), np.random.default_rng(rng))
    s = ep.cs.steps[k]
    assert s.kind == "stack" and not ep.peg_parked[s.peg]
    j, i = s.part, s.peg
    hx, hy = ep.slot_xy[i]
    ep._set_free(
        ep.peg_qadr[i], ep.peg_vadr[i], [hx, hy, PEG_LEN / 2],
        _yaw_quat(float(ep.slot_yaw[i])),
    )
    ep.seated_now[i] = True
    ep.collar_parked[j] = False
    ep._set_body_active(ep.collar_body[j], True)
    if mated:
        ep._pin_stack_on_peg(j)
    ep._action_earned_v08 = True
    ep.step_done[k] = True
    ep.step_in_order[k] = True
    ep.step_done_at[k] = 1
    ep.done = True
    ep.data.qpos[2] = 0.16
    ep.grip = MAX_OPEN
    ep._apply_ctrl()
    return ep, j, i


def test_all_unparked_stacks_survive_slot_yaw_finalization(conds, monkeypatch):
    """Every shape bucket retains >=95% of valid mates over several kick angles."""
    from collections import Counter
    import copy
    import nist_sim.compound_v08 as v08

    count, kept = Counter(), Counter()
    failures = []
    explicit_cases = {}
    for board, cond in enumerate(conds):
        probe = AssemblyIVEpisode(copy.deepcopy(cond), np.random.default_rng(23))
        stacks = [
            k for k, s in enumerate(probe.cs.steps)
            if s.kind == "stack" and not probe.peg_parked[s.peg]
        ]
        for k in stacks:
            shape = probe.c.pegs[probe.cs.steps[k].peg].spec.shape
            for kick, rng in [(False, 23)] + [(True, seed) for seed in range(23, 27)]:
                monkeypatch.setattr(v08, "PERTURB_XY", 0.0018 if kick else 0.0)
                ep, j, i = _stack_finalization_fixture(cond, k, rng)
                pins = []
                original_pin = ep._pin_stack_on_peg

                def reject_repin(ring):
                    pins.append(ring)
                    return original_pin(ring)

                ep._pin_stack_on_peg = reject_repin
                ep._finalize_stability()
                if not kick:
                    explicit_cases[(board, k)] = ep
                    assert ep.step_done[k], (board, k, shape)
                    assert ep._physical_step_ok(k)[0]
                elif ep.step_done[k]:
                    count[shape] += 1
                    kept[shape] += 1
                else:
                    count[shape] += 1
                    failures.append((board, k, shape, rng, ep.collar_pose(j)[0].tolist()))
                assert not pins, "settling must not restore the mate"
    assert sum(count.values()) == 69 * 4
    for shape in sorted(count):
        assert kept[shape] / count[shape] >= 0.95, (shape, kept[shape], count[shape], failures)
    # Explicit support regressions and the former board-42 clutter projectile.
    for case in ((38, 13), (52, 10), (54, 12), (42, 12)):
        assert explicit_cases[case].step_done[case[1]], case


@pytest.mark.parametrize("board", [1, 41, 42])
def test_forced_fallback_validates_all_active_parts_and_preserves_rework(conds, monkeypatch, board):
    """Fallback cells include active rings and respect parked parts and rework."""
    import copy
    import mujoco
    from nist_sim.env import SPAWN_BOARD_MARGIN, SPAWN_MAX_PENETRATION
    from nist_sim.conditions import slot_world_xy

    ep = AssemblyIVEpisode(copy.deepcopy(conds[board]), np.random.default_rng(26))
    assert ep._rework_wrong
    assert (any(ep.peg_parked) and any(ep.collar_parked)) or board == 41
    # The sampled reset already constructed a valid board. Run the fallback
    # directly so its all-active-part invariant is exercised deterministically.
    ep._teleport_safe_grid()
    assert ep._spawn_ok()
    cells = ep._fallback_grid()
    expected = sum(
        not ep.peg_parked[i] and p.start_type != "grasped" and i not in ep._rework_wrong
        for i, p in enumerate(ep.c.pegs)
    ) + sum(not parked for parked in ep.collar_parked) + ep.n_clutter
    assert len(cells) == expected
    for i, p in enumerate(ep.c.pegs):
        pos, _ = ep.peg_pose(i)
        if ep.peg_parked[i]:
            assert pos[2] > 0.4
        else:
            assert np.all(np.isfinite(pos))
            assert max(abs(pos[0]), abs(pos[1])) <= SPAWN_BOARD_MARGIN
    for j, parked in enumerate(ep.collar_parked):
        pos, _ = ep.collar_pose(j)
        assert pos[2] > 0.4 if parked else max(abs(pos[0]), abs(pos[1])) <= SPAWN_BOARD_MARGIN
        if not parked:
            expected_xy = np.array(ep._grid_fallback_collar_cells[j])
            np.testing.assert_allclose(pos[:2], expected_xy, atol=0.002)
    for i in range(ep.n_clutter):
        pos, _ = ep.clutter_pose(i)
        assert np.all(np.isfinite(pos))
        assert max(abs(pos[0]), abs(pos[1])) <= SPAWN_BOARD_MARGIN
        assert ep._clutter_spawn_ok(i)
    part_bodies = set(ep.peg_body) | set(ep.collar_body) | set(ep.clutter_body)
    for con in ep.data.contact[:ep.data.ncon]:
        b1 = ep.model.geom_bodyid[con.geom1]
        b2 = ep.model.geom_bodyid[con.geom2]
        if con.dist < -SPAWN_MAX_PENETRATION and (b1 in part_bodies or b2 in part_bodies):
            # Only the intentional wrong-slot peg/hole pair can remain.
            assert ep._rework_contact_exempt(con)
    for i, wrong_slot in ep._rework_wrong.items():
        hx, hy = slot_world_xy(ep.c.board_xy, ep.c.board_yaw, ep.c.slots[wrong_slot])
        np.testing.assert_allclose(ep.peg_pose(i)[0][:2], [hx, hy], atol=1e-6)
    mujoco.mj_forward(ep.model, ep.data)


def test_valid_stack_preserves_unrelated_kicks_and_angle_stream(conds, monkeypatch):
    """The claimed carrier is exempt; another seated peg still gets its drawn kick."""
    import copy
    from nist_sim.env import PEG_LEN, _yaw_quat

    k = next(
        k for k, s in enumerate(conds[1].compound.steps)
        if s.kind == "stack" and conds[1].compound.peg_wave[s.peg] == 0
    )
    ep, j, i = _stack_finalization_fixture(conds[1], k)
    other = next(p for p in range(ep.n_pegs) if p != i and not ep.peg_parked[p])
    hx, hy = ep.slot_xy[other]
    ep._set_free(
        ep.peg_qadr[other], ep.peg_vadr[other], [hx, hy, PEG_LEN / 2],
        _yaw_quat(float(ep.slot_yaw[other])),
    )
    ep.seated_now[other] = True
    peg_order = [
        p for p in range(ep.n_pegs)
        if not ep.peg_parked[p] and (ep.seated_now[p] or ep._peg_in_hole(p))
    ]
    assert set(peg_order) == {i, other}
    expected_rng = np.random.default_rng(23)
    expected_rng.bit_generator.state = ep.rng.bit_generator.state
    angles = expected_rng.uniform(0, 2 * np.pi, len(peg_order) + sum(not p for p in ep.collar_parked))
    other_angle = angles[peg_order.index(other)]
    observed_angles = []
    original_rng = ep.rng

    class AngleRecorder:
        def __getattr__(self, name):
            return getattr(original_rng, name)

        def uniform(self, low, high, *args, **kwargs):
            value = original_rng.uniform(low, high, *args, **kwargs)
            if low == 0 and high == 2 * np.pi:
                observed_angles.append(float(value))
            return value

    ep.rng = AngleRecorder()
    checked = []

    import nist_sim.compound_v08 as v08
    import mujoco
    original_step = mujoco.mj_step

    def check_impulses_before_settle(model, data):
        if not checked:
            checked.append(True)
            np.testing.assert_allclose(data.qvel[ep.peg_vadr[i] : ep.peg_vadr[i] + 2], [0.0, 0.0])
            np.testing.assert_allclose(
                data.qvel[ep.peg_vadr[other] : ep.peg_vadr[other] + 2],
                0.9 * np.array([np.cos(other_angle), np.sin(other_angle)]),
            )
            np.testing.assert_allclose(data.qvel[ep.collar_vadr[j] : ep.collar_vadr[j] + 2], [0.0, 0.0])
        original_step(model, data)

    monkeypatch.setattr(v08.mujoco, "mj_step", check_impulses_before_settle)
    ep._finalize_stability()
    assert checked
    np.testing.assert_allclose(observed_angles, angles)
    assert ep.step_done[k]
    np.testing.assert_allclose(ep.collar_pose(j)[0][:2], ep.slot_xy[i], atol=0.003)


def test_non_upright_in_hole_stack_carrier_is_kicked_without_changing_angles(conds, monkeypatch):
    """An in-hole predicate alone must not exempt a claimed stack's carrier."""
    k = next(
        k for k, s in enumerate(conds[1].compound.steps)
        if s.kind == "stack" and conds[1].compound.peg_wave[s.peg] == 0
    )
    ep, _j, i = _stack_finalization_fixture(conds[1], k)
    ep.seated_now[i] = False
    ep._peg_in_hole = lambda peg: peg == i
    assert not ep.seated_now[i] and ep._peg_in_hole(i)
    peg_order = [
        p for p in range(ep.n_pegs)
        if not ep.peg_parked[p] and (ep.seated_now[p] or ep._peg_in_hole(p))
    ]
    assert peg_order == [i]
    expected_rng = np.random.default_rng(23)
    expected_rng.bit_generator.state = ep.rng.bit_generator.state
    angles = expected_rng.uniform(0, 2 * np.pi, 1 + sum(not p for p in ep.collar_parked))
    observed_angles = []
    original_rng = ep.rng

    class AngleRecorder:
        def __getattr__(self, name):
            return getattr(original_rng, name)

        def uniform(self, low, high, *args, **kwargs):
            value = original_rng.uniform(low, high, *args, **kwargs)
            if low == 0 and high == 2 * np.pi:
                observed_angles.append(float(value))
            return value

    ep.rng = AngleRecorder()
    monkeypatch.setattr("nist_sim.compound_v08.mujoco.mj_step", lambda model, data: None)
    ep._finalize_stability()
    np.testing.assert_allclose(observed_angles, angles)
    np.testing.assert_allclose(
        ep.data.qvel[ep.peg_vadr[i] : ep.peg_vadr[i] + 2],
        0.9 * np.array([np.cos(angles[0]), np.sin(angles[0])]),
    )


def test_invalid_stack_claim_cannot_exempt_carrier_peg(conds, monkeypatch):
    """False, displaced and gripped claims fail; only valid claimed stacks skip a peg kick."""
    from nist_sim.compound import COLLAR_H
    import nist_sim.compound_v08 as v08
    from nist_sim.env import _yaw_quat

    board, k = 1, next(
        k for k, s in enumerate(conds[1].compound.steps)
        if s.kind == "stack" and conds[1].compound.peg_wave[s.peg] == 0
    )
    for attack in ("never_mated", "displaced", "gripped"):
        ep, j, i = _stack_finalization_fixture(conds[board], k, mated=attack != "never_mated")
        if attack == "never_mated":
            hx, hy = ep.slot_xy[i]
            ep._set_free(ep.collar_qadr[j], ep.collar_vadr[j], [hx, hy, 0.054], [1, 0, 0, 0])
            ep._stack_mated[j] = False
        elif attack == "displaced":
            p, _ = ep.collar_pose(j)
            ep._set_free(ep.collar_qadr[j], ep.collar_vadr[j], [p[0] + 0.05, p[1], COLLAR_H / 2], [1, 0, 0, 0])
        else:
            ep._gripper_touching = lambda body: body == ep.collar_body[j]
        other = next(p for p in range(ep.n_pegs) if p != i and not ep.peg_parked[p])
        peg_order = [
            p for p in range(ep.n_pegs)
            if not ep.peg_parked[p] and (ep.seated_now[p] or ep._peg_in_hole(p))
        ]
        assert i in peg_order and other not in peg_order
        hx, hy = ep.slot_xy[other]
        ep._set_free(
            ep.peg_qadr[other], ep.peg_vadr[other], [hx, hy, 0.025],
            _yaw_quat(float(ep.slot_yaw[other])),
        )
        ep.seated_now[other] = True
        peg_order = [
            p for p in range(ep.n_pegs)
            if not ep.peg_parked[p] and (ep.seated_now[p] or ep._peg_in_hole(p))
        ]
        assert set(peg_order) == {i, other}
        monkeypatch.setattr(v08.mujoco, "mj_step", lambda model, data: None)
        before = ep.data.qvel.copy()
        angles = []
        original_rng = ep.rng
        expected_rng = np.random.default_rng(23)
        expected_rng.bit_generator.state = original_rng.bit_generator.state

        class AngleRecorder:
            def __getattr__(self, name):
                return getattr(original_rng, name)

            def uniform(self, low, high, *args, **kwargs):
                value = original_rng.uniform(low, high, *args, **kwargs)
                if low == 0 and high == 2 * np.pi:
                    angles.append(float(value))
                return value

        ep.rng = AngleRecorder()
        active_collars = sum(not parked for parked in ep.collar_parked)
        ep._finalize_stability()
        expected_angles = expected_rng.uniform(0, 2 * np.pi, len(peg_order) + active_collars)
        np.testing.assert_allclose(angles, expected_angles)
        for peg in (i, other):
            angle = expected_angles[peg_order.index(peg)]
            np.testing.assert_allclose(
                ep.data.qvel[ep.peg_vadr[peg] : ep.peg_vadr[peg] + 2],
                before[ep.peg_vadr[peg] : ep.peg_vadr[peg] + 2] + 0.9 * np.array([np.cos(angle), np.sin(angle)]),
            )
        assert not ep.step_done[k], attack
        assert compound_score_v08(ep)["score"] == 0.0, attack
        if attack == "never_mated":
            np.testing.assert_allclose(
                ep.data.qvel[ep.collar_vadr[j] : ep.collar_vadr[j] + 2],
                before[ep.collar_vadr[j] : ep.collar_vadr[j] + 2] + 0.63 * np.array([np.cos(angles[len(peg_order)]), np.sin(angles[len(peg_order)])]),
            )


def test_active_collar_cannot_start_embedded_without_rework(conds):
    """A rework-free episode also rejects a collar embedded in the board."""
    import copy
    import mujoco

    ep = AssemblyIVEpisode(copy.deepcopy(conds[49]), np.random.default_rng(23))
    assert not ep._rework_wrong
    j = next(j for j, parked in enumerate(ep.collar_parked) if not parked)
    p, q = ep.collar_pose(j)
    ep._set_free(ep.collar_qadr[j], ep.collar_vadr[j], [p[0], p[1], p[2] - 0.004], q)
    mujoco.mj_forward(ep.model, ep.data)
    assert any(
        ep.collar_body[j] in (
            ep.model.geom_bodyid[con.geom1], ep.model.geom_bodyid[con.geom2]
        ) and con.dist < -0.0015
        for con in ep.data.contact[:ep.data.ncon]
    )
    assert not ep._spawn_ok()
    p, q = ep.collar_pose(j)
    ep._set_free(ep.collar_qadr[j], ep.collar_vadr[j], [p[0] + 0.050, p[1], p[2]], q)
    ep.data.qvel[ep.collar_vadr[j] : ep.collar_vadr[j] + 3] = 0.0
    assert not ep._collar_spawn_ok(j)


def test_rework_spawn_validation_also_checks_clutter(conds, monkeypatch):
    """A rework contact exemption must not admit a frozen clutter projectile."""
    import mujoco
    import nist_sim.compound_v08 as v08
    from nist_sim.env import SPAWN_MAX_PENETRATION

    monkeypatch.setattr(v08, "PERTURB_XY", 0.0)
    ep = AssemblyIVEpisode(conds[42], np.random.default_rng(23))
    assert ep._rework_wrong and ep.n_clutter
    for i in range(ep.n_clutter):
        p, _ = ep.clutter_pose(i)
        assert np.all(np.isfinite(p)) and np.linalg.norm(p) < 0.3
    # Isolate the original failure: a wrong-hole peg remains exempt while a
    # deeply embedded block must invalidate the same spawn candidate.
    ep._place_clutter(0)
    p, q = ep.clutter_pose(0)
    rework_peg = min(ep._rework_wrong)
    spec = ep.c.slots[ep._rework_wrong[rework_peg]]
    from nist_sim.conditions import slot_world_xy
    hx, hy = slot_world_xy(ep.c.board_xy, ep.c.board_yaw, spec)
    ep._set_free(ep.clutter_qadr[0], ep.clutter_vadr[0], [hx, hy, p[2]], q)
    ep._seat_in_wrong_slot(rework_peg)
    mujoco.mj_forward(ep.model, ep.data)
    deep = min(
        (float(con.dist) for con in ep.data.contact[:ep.data.ncon]
         if ep.clutter_body[0] in (
             ep.model.geom_bodyid[con.geom1], ep.model.geom_bodyid[con.geom2],
         )),
        default=0.0,
    )
    assert deep < -SPAWN_MAX_PENETRATION
    assert not ep._clutter_spawn_ok(0)
    assert not ep._spawn_ok()


def test_mated_stack_still_requires_measured_ring_pose(conds, monkeypatch):
    from nist_sim.env import PEG_LEN

    # Isolate measured-pose verification from the separate perturbation check.
    import nist_sim.compound_v08 as v08
    monkeypatch.setattr(v08, "PERTURB_XY", 0.0018)
    ep = AssemblyIVEpisode(conds[1], np.random.default_rng(17))
    k = next(k for k, s in enumerate(ep.cs.steps) if s.kind == "stack" and not ep.peg_parked[s.peg])
    j, i = ep.cs.steps[k].part, ep.cs.steps[k].peg
    assert not ep.peg_parked[i]
    ep.collar_parked[j] = False
    ep._set_body_active(ep.collar_body[j], True)
    hx, hy = ep.slot_xy[i]
    from nist_sim.env import _yaw_quat
    ep._set_free(
        ep.peg_qadr[i], ep.peg_vadr[i], [hx, hy, PEG_LEN / 2],
        _yaw_quat(float(ep.slot_yaw[i])),
    )
    ep.seated_now[i] = True
    ep.data.qpos[2] = 0.16
    ep._pin_stack_on_peg(j)
    ep._action_earned_v08 = True
    ep.step_done[k] = True
    ep.step_in_order[k] = True
    ep.step_done_at[k] = 1
    ep.done = True
    initial_impulses = []
    real_step = v08.mujoco.mj_step
    def capture_impulse(model, data):
        if not initial_impulses:
            initial_impulses.append(float(np.linalg.norm(data.qvel[adr:adr + 2])))
        return real_step(model, data)
    adr = ep.peg_vadr[i]
    monkeypatch.setattr(v08.mujoco, "mj_step", capture_impulse)
    ep._finalize_stability()
    assert ep.step_done[k]
    assert ep._physical_step_ok(k)[0]
    assert ep._stack_mated[j]
    # Neither the claimed mated ring nor its carrier peg is directly kicked.
    # An unmated ring and unrelated pegs must still receive their impulses.
    assert initial_impulses == [0.0]

    displaced = AssemblyIVEpisode(conds[0], np.random.default_rng(17))
    monkeypatch.setattr(v08, "PERTURB_XY", 0.0018)
    displaced.collar_parked[j] = False
    displaced._set_body_active(displaced.collar_body[j], True)
    displaced._set_free(displaced.peg_qadr[i], displaced.peg_vadr[i], [hx, hy, PEG_LEN / 2], [1, 0, 0, 0])
    displaced.seated_now[i] = True
    displaced.data.qpos[2] = 0.16
    displaced._pin_stack_on_peg(j)
    displaced._action_earned_v08 = True
    displaced.step_done[k] = True
    displaced.step_in_order[k] = True
    displaced.step_done_at[k] = 1
    displaced.done = True
    pins = []
    original_pin = displaced._pin_stack_on_peg
    def count_pin(ring):
        pins.append(ring)
        return original_pin(ring)
    displaced._pin_stack_on_peg = count_pin
    displaced._finalize_stability()
    assert not pins, "finalization must not restore a ring after the perturbation"
    if displaced.step_done[k]:
        # If this fixture's kick did not dislodge the ring, independently test
        # a displaced state rather than claiming a forced successful rejection.
        displaced._set_free(displaced.collar_qadr[j], displaced.collar_vadr[j], [hx + 0.05, hy, 0.004], [1, 0, 0, 0])
    assert not displaced._physical_step_ok(k)[0]

    # A stale mate latch cannot hide a ring that physically left the peg.
    ep._set_free(ep.collar_qadr[j], ep.collar_vadr[j], [hx + 0.05, hy, 0.004], [1, 0, 0, 0])
    assert not ep._physical_step_ok(k)[0]

    unmated = AssemblyIVEpisode(conds[0], np.random.default_rng(17))
    unmated.collar_parked[j] = False
    unmated._set_body_active(unmated.collar_body[j], True)
    unmated._set_free(unmated.peg_qadr[i], unmated.peg_vadr[i], [hx, hy, PEG_LEN / 2], [1, 0, 0, 0])
    unmated.seated_now[i] = True
    unmated.data.qpos[2] = 0.16
    unmated._set_free(unmated.collar_qadr[j], unmated.collar_vadr[j], [hx, hy, 0.054], [1, 0, 0, 0])
    unmated._stack_mated[j] = False
    unmated._action_earned_v08 = True
    unmated.step_done[k] = True
    unmated.step_done_at[k] = 1
    unmated.done = True
    unmated._finalize_stability()
    speed = float(np.linalg.norm(unmated.data.qvel[unmated.collar_vadr[j]:unmated.collar_vadr[j] + 2]))
    assert speed > 0.0, "an unmated loose ring must still be kicked directly"


def test_false_or_barely_mated_ring_cannot_dodge_final_verification(conds):
    from nist_sim.env import MAX_OPEN, PEG_LEN

    for xy_offset, z_offset in ((0.05, 0.0), (0.012, 0.0), (0.0, 0.012), (0.0, -0.020)):
        ep = AssemblyIVEpisode(conds[1], np.random.default_rng(31))
        k = next(k for k, s in enumerate(ep.cs.steps) if s.kind == "stack" and not ep.peg_parked[s.peg])
        j, i = ep.cs.steps[k].part, ep.cs.steps[k].peg
        hx, hy = ep.slot_xy[i]
        ep.collar_parked[j] = False
        ep._set_body_active(ep.collar_body[j], True)
        ep._set_free(ep.peg_qadr[i], ep.peg_vadr[i], [hx, hy, PEG_LEN / 2], [1, 0, 0, 0])
        ep.seated_now[i] = True
        ep.data.qpos[2] = 0.16
        ep.grip = MAX_OPEN
        ep._apply_ctrl()
        ep._action_earned_v08 = True
        ep.step_done[k] = True
        ep.step_in_order[k] = True
        ep.done = True
        ep._pin_stack_on_peg(j)
        assert ep._physical_step_ok(k)[0]
        correct, _ = ep.collar_pose(j)

        # A stale or falsely asserted mate flag cannot preserve credit after a
        # post-mate strike or a merely near-mate pose. Finalization never pins
        # the ring back onto the peg to make this check pass.
        ep._set_free(
            ep.collar_qadr[j], ep.collar_vadr[j],
            [correct[0] + xy_offset, correct[1], correct[2] + z_offset], [1, 0, 0, 0],
        )
        assert not ep._physical_step_ok(k)[0], (xy_offset, z_offset)
        ep._finalized = False
        ep._finalize_stability()
        assert not ep.step_done[k], (xy_offset, z_offset)
        assert compound_score_v08(ep)["score"] == 0.0


def test_key_hold_and_completion_precision_cannot_be_farmed_by_yaw_sweeps(conds):
    from nist_sim.compound_v07 import KEY_HOLD, KEY_PROBE_FRAMES
    from nist_sim.compound_v08 import _partial_credit_v08

    ep = AssemblyIVEpisode(conds[0], np.random.default_rng(12))
    keys = [k for k, s in enumerate(ep.cs.steps) if s.kind == "key"]
    assert len(keys) >= 2
    ep._action_earned_v08 = True
    for k in keys:
        s = ep.cs.steps[k]
        i = s.peg
        for p in s.prereq:
            ep.step_done[p] = True
        ep._peg_probe_frames[i] = KEY_PROBE_FRAMES
        ep._key_index_armed[i] = True
        ep._active_key = k
        ep._key_is_active = lambda _k: True
        ep._pressing_on_peg = lambda _i: True
        hx, hy = ep.slot_xy[i]
        from nist_sim.env import PEG_LEN
        ep._set_free(ep.peg_qadr[i], ep.peg_vadr[i], [hx, hy, PEG_LEN / 2], [1, 0, 0, 0])
        ep.seated_now[i] = True
        ep._gripper_touching = lambda _body: False

        # Brief visits below KEY_HOLD cannot complete or create precision. A
        # later sweep cannot turn that transient into completion-latched credit.
        for yaw in (s.key_yaw + 2 * s.key_tol, s.key_yaw, s.key_yaw + 2 * s.key_tol):
            ep.data.qpos[3] = yaw
            ep._step_state(k)
        assert ep._key_hold[i] < KEY_HOLD
        assert not ep.step_done[k]
        assert not np.isfinite(ep._key_precision_v08[k])
        assert _partial_credit_v08(ep, k) == 0.0

        # A proper uninterrupted hold, rather than a transient sweep, is the
        # only path to completion. No early latch or precision is produced by
        # crossing a window below KEY_HOLD.
        ep.data.qpos[3] = s.key_yaw
        for _ in range(KEY_HOLD):
            done, _ = ep._step_state(k)
        assert done
        assert ep._key_hold[i] >= KEY_HOLD
        assert not ep.step_done[k]  # parent bookkeeping has not run in this isolated fixture
        assert not np.isfinite(ep._key_precision_v08[k])

    # Completion precision is recorded once and cannot be improved by sweeping
    # to the centre of a different key window later. The two/three-key physical
    # finalization tests above exercise the actual seated-peg completion latch.
    latched = AssemblyIVEpisode(conds[0], np.random.default_rng(12))
    key = next(k for k, s in enumerate(latched.cs.steps) if s.kind == "key")
    latched.step_done[key] = True
    latched._key_precision_v08[key] = 0.37
    latched.data.qpos[3] = latched.cs.steps[key].key_yaw
    latched._update_milestones_v08()
    assert latched._key_precision_v08[key] == pytest.approx(0.37)
    other = next(k for k, s in enumerate(latched.cs.steps) if s.kind == "key" and k != key)
    latched.data.qpos[3] = latched.cs.steps[other].key_yaw
    latched._update_milestones_v08()
    assert latched._key_precision_v08[key] == pytest.approx(0.37)


def test_finalization_requires_upright_seated_peg(conds):
    from nist_sim.env import PEG_LEN

    for kind in ("insert", "key", "twist", "screw"):
        ep = next(
            ep
            for c in conds for seed in range(50)
            for ep in [AssemblyIVEpisode(c, np.random.default_rng(seed))]
            if any(s.kind == kind and not ep.peg_parked[s.peg] for s in c.compound.steps)
        )
        k = next(k for k, s in enumerate(ep.cs.steps) if s.kind == kind and not ep.peg_parked[s.peg])
        i = ep.cs.steps[k].peg
        assert not ep.peg_parked[i]
        hx, hy = ep.slot_xy[i]
        ep._set_free(ep.peg_qadr[i], ep.peg_vadr[i], [hx, hy, PEG_LEN / 2], [1, 0, 0, 0])
        ep.seated_now[i] = True
        ep._action_earned_v08 = True
        ep.step_done[k] = True
        ep.step_done_at[k] = 1
        ep.done = True
        # Test-only finalization fixture: retain depth but rotate the peg onto
        # its side. A depth-only hole predicate must not preserve completion.
        ep._set_free(ep.peg_qadr[i], ep.peg_vadr[i], [hx, hy, PEG_LEN / 2], [0.70710678, 0.70710678, 0, 0])
        assert ep._peg_in_hole(i)
        ep._finalize_stability()
        assert not ep.step_done[k], kind
        assert not ep._physical_step_ok(k)[0], kind


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


def test_record_task_and_event_sanitizers_hide_key_windows(conds):
    from nist_sim.compound_v08 import compound_record_events_v08, compound_record_task_v08

    ep = AssemblyIVEpisode(conds[0], np.random.default_rng(123))
    task = compound_record_task_v08(ep.cs)
    assert all("key_yaw" not in row and "key_tol" not in row for row in task["steps"])
    events = compound_record_events_v08([
        {"kind": "retarget", "step": 17, "key_step": 1, "old_yaw": 0.8, "new_yaw": -0.4},
        {"kind": "gain", "step": 22, "scale": 1.7, "friction": 1.3},
    ])
    assert "old_yaw" not in events[0] and "new_yaw" not in events[0]
    assert events[0]["kind"] == "retarget" and events[0]["step"] == 17
    assert events[1] == {"kind": "gain", "step": 22, "scale": 1.7}


def test_observation_has_no_seed_key_window_or_private_numeric_leak(conds, tmp_path, monkeypatch):
    from nist_sim import conditions
    from nist_sim.reserve import reserved_from_secret

    reserved = reserved_from_secret("v081-redteam-privacy")
    path = tmp_path / "reserved.json"
    path.write_text(json.dumps(reserved))
    monkeypatch.setattr(conditions, "RESERVED_PATH", path)
    _, sampled = sample_compound_v08(43, 1, "live")
    ep = AssemblyIVEpisode(sampled[0], np.random.default_rng(123))
    keys = [(k, s) for k, s in enumerate(ep.cs.steps) if s.kind == "key"]
    assert keys
    for k, s in keys:
        ep._active_key = k
        ep._peg_touched[s.peg] = True
        ep._peg_probe_frames[s.peg] = 99
        ep._key_index_armed[s.peg] = True
        ep._key_retarget_pulse[s.peg] = 20
    frames = np.asarray([ep.observe(np.random.default_rng(500 + i)) for i in range(12)])
    assert frames.shape == (12, 320)
    assert np.all(np.isfinite(frames))

    # These are referee-private numeric values, not values derivable from the
    # public board/blueprint. No observation channel may reveal them verbatim.
    private = [
        float(s.key_yaw) for _, s in keys
    ] + [
        float(s.key_tol) for _, s in keys
    ] + [
        float(x) for pair in ep.c.slot_bias_list for x in pair
    ] + [
        float(x) for pair in ep.c.peg_bias_delta_list for x in pair
    ] + [
        ep.c.friction, ep.c.peg_mass, ep.c.gain_scale,
        ep.c.obs_noise, ep.c.obs_noise_rad, ep.c.dropout_prob, ep.c.bias_drift,
    ]
    for value in private:
        if abs(value) < 1e-9:
            continue
        assert not np.any(np.isclose(frames, value, rtol=0.0, atol=1e-12)), value
        assert not np.any(np.isclose(frames, 10 * value, rtol=0.0, atol=1e-12)), 10 * value
    for lo, hi in reserved["ranges"].values():
        for value in (lo, hi):
            assert not np.any(np.isclose(frames, value, rtol=0.0, atol=1e-12))
    task0 = 184
    for _, s in keys:
        base = task0 + s.peg * 6
        assert np.all(frames[:, base + 3] == 0.0)  # no closeness/search gradient
        assert np.all(np.isin(frames[:, base + 5], (0.0, 1.0, 2.0)))
        # A present mark is permitted, but it cannot vary with the hidden yaw.
        yaw = ep.data.qpos[3]
        for candidate in (s.key_yaw - 2 * s.key_tol, s.key_yaw, s.key_yaw + 2 * s.key_tol):
            ep.data.qpos[3] = candidate
            obs = ep.observe(np.random.default_rng(900))
            assert obs[base + 3] == 0.0
            assert obs[base + 5] in (0.0, 1.0, 2.0)
        ep.data.qpos[3] = yaw


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


def test_zero_actions_cannot_seat_board_top_part():
    """A resting part over a recess is not an inserted part (seed 31, episode 58)."""
    from nist_sim.env import PEG_LEN

    fixture = Path(os.environ.get("NIST_V081_ARTIFACTS", "/home/bb/nist-v080-artifacts")) / "v080-free-start-condition.pkl"
    if not fixture.exists():
        pytest.skip("preserved v0.8.0 free-start fixture is unavailable")
    import pickle
    cond = pickle.loads(fixture.read_bytes())
    r1, _ = (np.random.default_rng(s) for s in episode_seed_sequence(31, 58).spawn(2))
    ep = AssemblyIVEpisode(cond, r1)
    for _ in range(6):
        ep.step(np.zeros(5))
    p, _ = ep.peg_pose(0)
    assert abs(p[2] - PEG_LEN / 2) < 0.001
    assert not ep.seated_now[0]
    assert not ep.step_done[0]
    assert not ep._peg_in_hole(0)
    while not ep.done:
        ep.step(np.zeros(5))
    assert compound_score_v08(ep)["score"] == 0.0


def test_v08_recessed_seat_remains_valid(conds):
    from nist_sim.env import PEG_LEN

    ep = AssemblyIVEpisode(conds[0], np.random.default_rng(17))
    i = next(s.peg for s in ep.cs.steps if s.kind == "insert")
    hx, hy = ep.slot_xy[i]
    ep._set_free(ep.peg_qadr[i], ep.peg_vadr[i], [hx, hy, PEG_LEN / 2], [1, 0, 0, 0])
    assert ep._peg_in_hole(i)
    ep._set_free(ep.peg_qadr[i], ep.peg_vadr[i], [hx, hy, PEG_LEN / 2 + 0.010], [1, 0, 0, 0])
    assert not ep._peg_in_hole(i)


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


def test_physically_reachable_partial_credit_cap_is_0312():
    """Producer gains, not the looser numeric clamp, bound transport/insert."""
    from nist_sim.compound_v08 import (
        ALIGN_CREDIT_V08,
        INSERT_CREDIT_V08,
        INSERT_GAIN_V08,
        LIFT_CREDIT_V08,
        PARTIAL_CAP_V08,
        STABLE_CREDIT_V08,
        TRANSPORT_CREDIT_V08,
        TRANSPORT_GAIN_V08,
        step_credit_v08,
    )

    reachable = (
        LIFT_CREDIT_V08
        + TRANSPORT_CREDIT_V08 * TRANSPORT_GAIN_V08
        + INSERT_CREDIT_V08 * INSERT_GAIN_V08
        + ALIGN_CREDIT_V08
        + STABLE_CREDIT_V08
    )
    assert reachable == pytest.approx(0.312)
    producer_maximum = {
        "lift": True,
        "transport": True,
        "transport_fraction": TRANSPORT_GAIN_V08,
        "align": True,
        "stable": True,
        "insert": True,
        "insert_fraction": INSERT_GAIN_V08,
    }
    assert step_credit_v08(False, False, None, 920, producer_maximum) == pytest.approx(0.312)
    assert PARTIAL_CAP_V08 == pytest.approx(0.36)  # nominal clamp; unchanged scoring
    assert step_credit_v08(False, False, None, 100, {"lift_fraction": 0.5}) == 0.5 * LIFT_CREDIT_V08
    assert step_credit_v08(False, False, None, 100, {"lift": True, "lift_fraction": 0.5}) == LIFT_CREDIT_V08
    assert step_credit_v08(False, False, None, 100, {"lift_fraction": 100.0}) == LIFT_CREDIT_V08
    assert step_credit_v08(
        False, False, None, 100, {"lift_fraction": 1.0, "align": True, "stable": True}
    ) == LIFT_CREDIT_V08 + ALIGN_CREDIT_V08 + STABLE_CREDIT_V08
    assert step_credit_v08(True, False, 10, 100, None) == 0.03


def test_dense_lift_requires_bilateral_grasp_and_sustained_real_rise(conds, monkeypatch):
    from nist_sim.compound_v08 import DENSE_LIFT_HOLD_FRAMES_V08, DENSE_LIFT_START_V08

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

    # The seed-31 episode-11 failure mode: sub-mm generic contact motion is not
    # a grasp-and-raise and cannot earn a fractional lift, even if repeated.
    monkeypatch.setattr(ep, "_bilateral_grasp_v08", lambda _body: False)
    ep.data.qpos[ep.peg_qadr[i] + 2] = reference + 0.0008
    for _ in range(2 * DENSE_LIFT_HOLD_FRAMES_V08):
        ep._update_milestones_v08()
    assert ep._milestones_v08[k].get("lift_fraction", 0.0) == 0.0

    # A real grasp must sustain a rise above 4 mm. A brief or one-frame dip
    # below the threshold resets the hold and cannot accumulate lift credit.
    monkeypatch.setattr(ep, "_bilateral_grasp_v08", lambda _body: True)
    ep.data.qpos[ep.peg_qadr[i] + 2] = reference + 0.00425
    for _ in range(DENSE_LIFT_HOLD_FRAMES_V08 - 1):
        ep._update_milestones_v08()
    assert ep._milestones_v08[k].get("lift_fraction", 0.0) == 0.0
    ep.data.qpos[ep.peg_qadr[i] + 2] = reference + DENSE_LIFT_START_V08 - 0.0001
    ep._update_milestones_v08()
    ep.data.qpos[ep.peg_qadr[i] + 2] = reference + 0.00425
    for _ in range(DENSE_LIFT_HOLD_FRAMES_V08 - 1):
        ep._update_milestones_v08()
    assert ep._milestones_v08[k].get("lift_fraction", 0.0) == 0.0
    ep._update_milestones_v08()
    fraction = ep._milestones_v08[k]["lift_fraction"]
    assert 0.0 < fraction == pytest.approx(0.0125)

    # Only bilateral contact counts: a palm push or a one-finger grasp cannot
    # start the hold even with a large apparent part rise.
    ep._milestones_v08[k] = dict(lift=False, transport=False, align=False, stable=False, insert=False)
    monkeypatch.setattr(ep, "_bilateral_grasp_v08", lambda _body: False)
    ep.data.qpos[ep.peg_qadr[i] + 2] = reference + 0.020
    for _ in range(2 * DENSE_LIFT_HOLD_FRAMES_V08):
        ep._update_milestones_v08()
    assert ep._milestones_v08[k].get("lift_fraction", 0.0) == 0.0

    ep._gripper_touching = lambda _b: False
    ep.data.qpos[ep.peg_qadr[i] + 2] = reference + 0.00425
    ep._update_milestones_v08()
    assert ep._milestones_v08[k].get("lift_fraction", 0.0) == 0.0
    for prereq in ep.cs.steps[k].prereq:
        ep.step_done[prereq] = False
    ep._action_earned_v08 = False
    ep._update_milestones_v08()
    assert compound_score_v08(ep)["score"] == 0.0


def test_bilateral_grasp_requires_both_finger_bodies(conds):
    ep = AssemblyIVEpisode(conds[0], np.random.default_rng(4))
    k = next(k for k, s in enumerate(ep.cs.steps) if s.kind == "insert" and not s.prereq)
    body = ep.peg_body[ep.cs.steps[k].peg]
    fl = ep.model.body("fl").id
    fr = ep.model.body("fr").id
    palm = ep.model.body("hand").id
    other = ep.peg_body[(ep.cs.steps[k].peg + 1) % ep.n_pegs]
    contacts = []
    from types import SimpleNamespace

    def fixture(pairs):
        rows = [SimpleNamespace(geom1=ep.model.body(a).geomadr[0], geom2=ep.model.body(b).geomadr[0]) for a, b in pairs]
        data = SimpleNamespace(ncon=len(rows), contact=rows)
        episode = AssemblyIVEpisode.__new__(AssemblyIVEpisode)
        episode.model = ep.model
        episode.data = data
        return episode

    assert not fixture([(fl, body)])._bilateral_grasp_v08(body)
    assert fixture([(fl, body), (fr, body)])._bilateral_grasp_v08(body)
    assert not fixture([(palm, body), (fr, body)])._bilateral_grasp_v08(body)
    assert not fixture([(fl, other), (fr, body)])._bilateral_grasp_v08(body)


def test_dense_lift_is_bounded_and_prerequisite_gated(conds):
    from nist_sim.compound_v08 import LIFT_CREDIT_V08, PARTIAL_CAP_V08, step_results_v08

    ep = AssemblyIVEpisode(conds[0], np.random.default_rng(4))
    ep._action_earned_v08 = True
    k = next(k for k, s in enumerate(ep.cs.steps) if s.kind == "insert" and not s.prereq)
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
