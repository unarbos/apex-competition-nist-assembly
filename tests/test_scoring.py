"""Integration tests: baseline scores > 0; every test-set policy scores <= the declared floor;
records match result.json; an unwritable record path leaves the score unchanged.

    python tools/make_policies.py && pytest -q tests/
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Honest-bad-answer ceiling for the test set: every degenerate policy must stay at or below it.
FLOOR = 0.10
SMALL = '{"episodes_per_stratum": 2}'


def run(policy: str, out: Path, seed: int = 3, override: str = SMALL) -> dict:
    subprocess.run([sys.executable, str(ROOT / "tools/local_run.py"), "--policy", str(ROOT / policy),
                    "--seed", str(seed), "--out", str(out), "--override", override], check=True)
    return json.loads((out / "result.json").read_text())


def test_baseline_scores(tmp_path):
    r = run("baselines/baseline.onnx", tmp_path)
    assert r["raw_scores"][0] > FLOOR


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
    assert len(rows) == r1["metadata"]["summary"][1]["value"]
    assert sum(x["success"] for x in rows) / len(rows) == r1["raw_scores"][0]


def test_player_dying_mid_response_is_a_player_fault(tmp_path):
    # Must produce a result.json (check=True): a crash here would be scored as a referee error.
    r = run("test_policies/const_batch2.onnx", tmp_path)
    assert r["raw_scores"][0] == 0.0
    assert r["terminal_reason"] == "player_fault"
    assert r["metadata"]["details"]["player_fault"]


# The platform's StandardEvalMetadata (apex-mvp shared/common eval_metadata.py). A payload that
# fails it is wrapped into `details` and the dashboard loses the per-episode breakdown.
FORMATS = {"number", "percent", "duration_s", "score", "count", "text"}
UNIT_TYPES = {"match", "game", "task", "subset", "instance", "completion", "other"}


def test_result_metadata_matches_platform_envelope(tmp_path):
    md = run("baselines/baseline.onnx", tmp_path)["metadata"]
    unit_ids = {u["id"] for u in md["units"]}
    assert all(u["type"] in UNIT_TYPES for u in md["units"])
    assert all(m.get("format", "number") in FORMATS for m in md["summary"] + md["metrics"])
    assert not any(m.get("unit_id") for m in md["summary"])
    assert md["metrics"] and all(m.get("unit_id") in unit_ids for m in md["metrics"])
    caps = md["capabilities"]
    assert not caps.get("category_breakdown") or all(u.get("category") for u in md["units"])
    assert not caps.get("outcome_breakdown") or all(u.get("outcome") for u in md["units"])
