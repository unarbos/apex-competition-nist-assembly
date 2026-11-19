"""Integration tests: baseline scores > 0; junk policies stay low; scoring math;
records match; faults leave a result.

    python tools/make_policies.py && pytest -q tests/
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
from nist_sim.scoring import episode_score, peg_score  # noqa: E402

FLOOR = 0.10
# 8 strata × 1 = 8 episodes. Enough for smoke; keeps v0.6 compound evals inside CI time.
SMALL = '{"episodes_per_stratum": 1, "workers": 2}'


def run(policy: str, out: Path, seed: int = 3, override: str = SMALL) -> dict:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/local_run.py"),
            "--policy",
            str(ROOT / policy),
            "--seed",
            str(seed),
            "--out",
            str(out),
            "--override",
            override,
        ],
        check=True,
    )
    return json.loads((out / "result.json").read_text())


def test_peg_score_math():
    assert peg_score(False, 10, 800) == 0.0
    # Perfect ASAP with zero effort → approaches 1.0 but effort share keeps it < 1
    s = peg_score(True, 1, 800, mean_sq_action=0.0)
    assert 0.75 < s <= 1.0
    # Late seat → near the floor
    late = peg_score(True, 800, 800, mean_sq_action=0.0)
    assert abs(late - 0.75) < 0.05
    # Knocked-out / not seated at end
    assert peg_score(False, 10, 800) == 0.0


def test_episode_score_averages_pegs():
    # 2/3 seated late → below 0.75 mean
    s = episode_score([True, True, False], [400, 400, None], 800)
    assert 0.4 < s < 0.6


def test_baseline_scores(tmp_path):
    r = run("baselines/baseline.onnx", tmp_path)
    assert r["raw_scores"][0] >= 0.0  # may be low on tiny draw; must not crash


@pytest.mark.parametrize("adv", ["zero", "constant_push", "random", "huge", "nan", "wrong_shape", "slow"])
def test_policy_set_below_floor(tmp_path, adv):
    try:
        r = run(f"test_policies/{adv}.onnx", tmp_path)
    except subprocess.CalledProcessError:
        return  # player never became ready -> platform scores the submission 0
    assert r["raw_scores"][0] <= FLOOR


def test_records_match_and_deterministic(tmp_path):
    r1 = run("baselines/baseline.onnx", tmp_path / "a")
    r2 = run("baselines/baseline.onnx", tmp_path / "b")
    assert r1["raw_scores"] == r2["raw_scores"]
    rows = [json.loads(x) for x in (tmp_path / "a/history/episodes.jsonl").read_text().splitlines()][1:]
    assert len(rows) == next(s["value"] for s in r1["metadata"]["summary"] if s["key"] == "episodes")
    mean_score = sum(x["score"] for x in rows) / len(rows)
    assert abs(mean_score - r1["raw_scores"][0]) < 1e-9


def test_player_dying_mid_response_is_a_player_fault(tmp_path):
    r = run("test_policies/const_batch2.onnx", tmp_path)
    assert r["raw_scores"][0] == 0.0
    assert r["terminal_reason"] == "player_fault"
    assert r["metadata"]["details"]["player_fault"]


FORMATS = {"number", "percent", "duration_s", "score", "count", "text"}
UNIT_TYPES = {"match", "game", "task", "subset", "instance", "completion", "other"}


def test_validate_actions_rejects_nonfinite_bad_shape_and_non_numeric():
    from referee import _validate_actions

    assert _validate_actions([[0.0] * 5], 1)[0].shape == (5,)
    for raw in (
        [[float("nan")] * 5],
        [[float("inf"), 0.0, 0.0, 0.0, 0.0]],
        [[-float("inf"), 0.0, 0.0, 0.0, 0.0]],
        [[True, 0.0, 0.0, 0.0, 0.0]],
        [["0", 0.0, 0.0, 0.0, 0.0]],
        [[0.0] * 4],
    ):
        assert _validate_actions(raw, 1) == [None]
    assert _validate_actions([], 1) is None
    assert _validate_actions([0.0] * 5, 1) is None
    assert _validate_actions([[1e100, -1e100, 1e100, -1e100, 1e100]], 1)[0].tolist() == [1e100, -1e100, 1e100, -1e100, 1e100]


def test_result_metadata_matches_platform_envelope(tmp_path):
    result = run("baselines/baseline.onnx", tmp_path)
    md = result["metadata"]
    assert set(md) == {"schema_version", "summary", "units", "metrics", "capabilities", "details"}
    assert md["details"]["nist_profile"] == "v08"
    assert md["details"]["assembly"] == "IV"
    assert all("nist_profile" not in u for u in md["units"])
    records = [json.loads(x) for x in (tmp_path / "history/episodes.jsonl").read_text().splitlines()]
    assert records[0]["assembly"] == "IV"
    assert all(row["nist_profile"] == "v08" for row in records[1:])
    blob = json.dumps({"metadata": md, "records": records})
    forbidden = (
        "key_yaw", "key_tol", "seed_sequence", "build_salt", "reserved", "salt",
        "pose_bias", "slot_bias", "peg_bias", "friction", "peg_mass", "gain_scale",
        "obs_noise", "dropout_prob", "bias_drift", "backlash", "search_pitch",
        "yaw_err", "hidden_range",
    )
    # The round-condition ranges are public training-draw axes, not reserved
    # endpoints. Public board/clutter heights and offsets are allowed as well.
    allowed_round_ranges = md["details"]["round_condition_ranges"]
    assert allowed_round_ranges == records[0]["round_condition_ranges"]
    public_round_blob = json.dumps(allowed_round_ranges)
    assert all(token not in public_round_blob for token in ("reserved", "salt", "hidden_range"))
    private_blob = json.dumps({"metadata_without_public_ranges": {
        k: v for k, v in md["details"].items() if k != "round_condition_ranges"
    }, "records": [row for row in records[1:] if True]})
    assert all(token not in private_blob for token in forbidden)
    unit_ids = {u["id"] for u in md["units"]}
    assert all(u["type"] in UNIT_TYPES for u in md["units"])
    assert all(m.get("format", "number") in FORMATS for m in md["summary"] + md["metrics"])
    assert not any(m.get("unit_id") for m in md["summary"])
    assert md["metrics"] and all(m.get("unit_id") in unit_ids for m in md["metrics"])
    caps = md["capabilities"]
    assert not caps.get("category_breakdown") or all(u.get("category") for u in md["units"])
    assert not caps.get("outcome_breakdown") or all(u.get("outcome") for u in md["units"])
