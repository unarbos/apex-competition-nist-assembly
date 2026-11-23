"""v0.8.3 reset and worker-failure guards, using the real fork/pipe path."""

from __future__ import annotations

import json
import atexit
import importlib
import multiprocessing as mp
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
os.environ["NIST_PROFILE"] = "v08"

import referee as ref  # noqa: E402
ref = importlib.reload(ref)
from gym_v1.referee import RefereeContext  # noqa: E402


@pytest.fixture(autouse=True)
def restore_v08_profile(monkeypatch):
    """Legacy record tests may reload the shared module under an older profile."""
    monkeypatch.setenv("NIST_PROFILE", "v08")
    importlib.reload(ref)
    yield


class ZeroPlayer:
    def __init__(self):
        self.observations = []

    def reset(self, **kwargs):
        assert kwargs["config"]["obs_dim"] == ref.OBS_W

    def act(self, observation, deadline_ms):
        self.observations.append(observation)
        return [[0.0] * ref.ACT_DIM for _ in observation]


def _play(tmp_path, monkeypatch, label, player=None, workers=2, max_steps=None, stride=1):
    assert ref.ASSEMBLY_IV
    monkeypatch.setattr(ref, "HISTORY_DIR", tmp_path / label)
    config = {"episodes_per_stratum": 1, "workers": workers, "record_stride": stride}
    if max_steps is not None:
        original = ref._sample

        def short_round(seed, per_stratum, split):
            ranges, conds = original(seed, per_stratum, split)
            for cond in conds:
                cond.max_steps = max_steps
            return ranges, conds

        monkeypatch.setattr(ref, "_sample", short_round)
    ctx = RefereeContext(
        match_id="test-v083",
        seed=7,
        config=config,
        player_urls=["http://unused.test"],
        num_players=1,
    )
    started = time.monotonic()
    result = ref.NistLiteReferee().play_game(ctx, [player or ZeroPlayer()])
    elapsed = time.monotonic() - started
    assert elapsed < 60.0
    rows = [json.loads(line) for line in (tmp_path / label / "episodes.jsonl").read_text().splitlines()][1:]
    assert result.metadata["details"]["player_fault"] is None
    assert len(rows) == 8
    assert [r["episode"] for r in rows] == list(range(8))
    assert len(result.metadata["units"]) == 8
    return result, rows


@pytest.fixture
def referee_watchdog():
    watchdog = mp.get_context("fork").Process(
        target=_terminate_referee_workers, args=(os.getpid(),)
    )
    watchdog.start()
    yield
    if watchdog.is_alive():
        watchdog.terminate()
    watchdog.join(timeout=2)


def _assert_failure_rows(rows, reason, failed):
    assert {r["episode"] for r in rows if r["terminal_reason"] == reason} == set(failed)
    for row in rows:
        if row["episode"] not in failed:
            continue
        assert row["score"] == 0.0
        assert row["failure_reason"] == reason
        assert row["success"] is False
        assert row["steps"] == 0
        assert row["events_fired"] == 0
        assert row["events"] == []
        assert len(row["step_results"]) == row["n_steps"] == len(row["task"]["steps"])
        assert all(sr["credit"] == 0.0 for sr in row["step_results"])
        assert all(not any(sr["partial"].values()) for sr in row["step_results"])
        assert all(v == 0.0 for v in row["credit_by_kind"].values() if v is not None)
        assert set(row["credit_by_kind"]) == set(ref.STEP_KINDS_V08)
        blob = json.dumps(row)
        assert "exception" not in blob
        assert all(token not in blob for token in ("seed_sequence", "build_salt", "reserved", "salt"))


def test_forced_reset_failures_are_deterministic_zero_without_hang(
    tmp_path, monkeypatch, referee_watchdog
):
    original = ref._make_episode
    worker_calls = [0]

    def fail_selected(cond, rng):
        position = worker_calls[0]
        worker_calls[0] += 1
        # Worker-local positions 0,1,2 in slices [0,2,4,6] / [1,3,5,7].
        if position in {0, 1, 2}:
            raise RuntimeError("injected construction failure; this text must not reach a record")
        return original(cond, rng)

    monkeypatch.setattr(ref, "_make_episode", fail_selected)
    first, first_rows = _play(tmp_path, monkeypatch, "first")
    failed = [r["episode"] for r in first_rows if r["terminal_reason"] == "reset_failure"]
    assert failed == [0, 1, 2, 3, 4, 5]
    _assert_failure_rows(first_rows, "reset_failure", failed)
    assert all(r["steps"] == 920 for r in first_rows if r["episode"] in (6, 7))
    assert first.metadata["details"]["terminal_reasons"]["reset_failure"] == 6
    assert sum(r["score"] for r in first_rows) / 8 == first.raw_scores[0]

    monkeypatch.setattr(ref, "_make_episode", fail_selected)
    second, second_rows = _play(tmp_path, monkeypatch, "second")
    _assert_failure_rows(second_rows, "reset_failure", failed)
    assert second_rows == first_rows
    assert second.metadata == first.metadata
    assert second.raw_scores == first.raw_scores


def test_all_reset_worker_preserves_reset_and_surviving_rows(
    tmp_path, monkeypatch, referee_watchdog
):
    original_worker = ref._worker

    def all_reset_first_worker(conn, seed, per_stratum, indices, stride, split):
        if indices[0] == 0:
            original_make = ref._make_episode

            def fail_every_construction(cond, rng):
                raise RuntimeError("injected all-reset worker")

            ref._make_episode = fail_every_construction
            try:
                return original_worker(conn, seed, per_stratum, indices, stride, split)
            finally:
                ref._make_episode = original_make
        return original_worker(conn, seed, per_stratum, indices, stride, split)

    monkeypatch.setattr(ref, "_worker", all_reset_first_worker)
    result, rows = _play(tmp_path, monkeypatch, "all-reset-worker", max_steps=3)
    _assert_failure_rows(rows, "reset_failure", [0, 2, 4, 6])
    assert "worker_failure" not in result.metadata["details"]["terminal_reasons"]
    assert all(row["steps"] == 3 for row in rows if row["episode"] % 2)
    assert result.metadata["details"]["player_calls"] == 3


def test_normal_two_worker_round_preserves_full_lockstep(tmp_path, monkeypatch):
    player = ZeroPlayer()
    result, rows = _play(tmp_path, monkeypatch, "normal-lockstep", player)
    assert all(len(batch) == 8 for batch in player.observations)
    assert len(player.observations) == 920
    assert all(row["steps"] == 920 for row in rows)
    assert not result.metadata["details"]["player_fault"]


def test_dead_worker_returns_zero_rows_and_does_not_wait_for_platform_timeout(tmp_path, monkeypatch):
    original = ref._worker

    def die_before_first_observation(conn, seed, per_stratum, indices, stride, split):
        if indices[0] == 0:
            conn.close()
            os._exit(23)
        return original(conn, seed, per_stratum, indices, stride, split)

    monkeypatch.setattr(ref, "_worker", die_before_first_observation)
    result, rows = _play(tmp_path, monkeypatch, "dead-worker")
    failed = list(range(0, 8, 2))
    _assert_failure_rows(rows, "worker_failure", failed)
    assert result.metadata["details"]["terminal_reasons"]["worker_failure"] == 4
    assert result.raw_scores == [0.0]


def test_early_worker_death_during_peer_construction_does_not_hang(tmp_path, monkeypatch):
    original_make_episode = ref._make_episode
    original_entry = ref._worker_entry
    original_sample = ref._sample
    original_short_sample = None

    def short_sample(seed, per_stratum, split):
        ranges, conds = original_sample(seed, per_stratum, split)
        for cond in conds:
            cond.max_steps = 3
        return ranges, conds

    def install_short_sample():
        nonlocal original_short_sample
        original_short_sample = ref._sample
        ref._sample = short_sample

    ctx = mp.get_context("fork")
    construction_parent, construction_child = ctx.Pipe(duplex=False)
    watchdog = ctx.Process(target=_terminate_referee_workers, args=(os.getpid(),))
    watchdog.start()
    worker_processes = []
    original_process = ctx.Process

    def record_process(*args, **kwargs):
        proc = original_process(*args, **kwargs)
        worker_processes.append(proc)
        return proc

    delayed = False

    def pause_first_worker(cond, rng):
        nonlocal delayed
        if not delayed:
            delayed = True
            construction_child.send_bytes(b"\0")
            construction_child.close()
            # This delay is confined to construction: no parent scheduling or
            # peer-side event is needed for the surviving worker to finish.
            time.sleep(1)
        return original_make_episode(cond, rng)

    def die_after_readiness(conn, inherited, ready, seed, per_stratum, indices, stride, split):
        if indices[0] == 0:
            # Install the first worker's delay before the launch handshake.
            ref._make_episode = pause_first_worker
            install_short_sample()
            return original_entry(conn, inherited, ready, seed, per_stratum, indices, stride, split)
        construction_child.close()
        # Synchronize on the peer entering construction. Production _worker_entry
        # must close every inherited endpoint before this fixture acknowledges.
        if not construction_parent.poll(30):
            os._exit(25)
        construction_parent.recv_bytes(1)
        construction_parent.close()
        ref._worker = lambda *args: os._exit(23)
        return original_entry(conn, inherited, ready, seed, per_stratum, indices, stride, split)

    monkeypatch.setattr(ref, "_worker_entry", die_after_readiness)
    monkeypatch.setattr(ctx, "Process", record_process)
    try:
        result, rows = _play(tmp_path, monkeypatch, "early-death-during-construction", max_steps=3)
    finally:
        construction_parent.close()
        construction_child.close()
        if watchdog.is_alive():
            watchdog.terminate()
        watchdog.join(timeout=2)
        for proc in worker_processes:
            if proc.pid is not None and proc.is_alive():
                proc.terminate()
        for proc in worker_processes:
            if proc.pid is not None:
                proc.join(timeout=2)
        if original_short_sample is not None:
            ref._sample = original_short_sample
    _assert_failure_rows(rows, "worker_failure", [1, 3, 5, 7])
    assert result.metadata["details"]["player_calls"] == 3
    assert "reset_failure" not in result.metadata["details"]["terminal_reasons"]
    assert all(row["steps"] == 3 for row in rows if row["episode"] % 2 == 0)
    assert all(proc.exitcode is not None for proc in worker_processes)


def test_readiness_failure_terminates_and_reaps_every_started_worker(tmp_path, monkeypatch):
    original_entry = ref._worker_entry
    ctx = mp.get_context("fork")
    worker_processes = []
    original_process = ctx.Process

    def record_process(*args, **kwargs):
        proc = original_process(*args, **kwargs)
        worker_processes.append(proc)
        return proc

    def die_before_readiness(conn, inherited, ready, seed, per_stratum, indices, stride, split):
        if indices[0] == 0:
            return original_entry(conn, inherited, ready, seed, per_stratum, indices, stride, split)
        for other in inherited:
            other.close()
        ready.close()
        conn.close()
        os._exit(23)

    monkeypatch.setattr(ref, "_worker_entry", die_before_readiness)
    monkeypatch.setattr(ctx, "Process", record_process)
    monkeypatch.setattr(ref, "HISTORY_DIR", tmp_path / "readiness-failure")
    game = RefereeContext(
        match_id="test-v083",
        seed=7,
        config={"episodes_per_stratum": 1, "workers": 2, "record_stride": 0},
        player_urls=["http://unused.test"],
        num_players=1,
    )
    with pytest.raises(RuntimeError, match="before readiness"):
        ref.NistLiteReferee().play_game(game, [ZeroPlayer()])
    assert len(worker_processes) == 2
    assert all(proc.exitcode is not None for proc in worker_processes)
    assert not worker_processes[0].is_alive()
    assert worker_processes[1].exitcode == 23


def _terminate_referee_workers(parent_pid):
    """Last-resort fixture watchdog; normal completion terminates this helper."""
    import signal

    time.sleep(55)
    try:
        children = Path(f"/proc/{parent_pid}/task/{parent_pid}/children").read_text().split()
        for child in children:
            os.kill(int(child), signal.SIGTERM)
        os.kill(parent_pid, signal.SIGTERM)
    except (FileNotFoundError, ProcessLookupError):
        pass


def test_malformed_worker_frame_is_recovered_without_hang(tmp_path, monkeypatch):
    def malformed_frame(conn, seed, per_stratum, indices, stride, split):
        if indices[0] == 0:
            conn._send(struct.pack("!i", 1) + b"X")
        conn.close()
        os._exit(23)

    monkeypatch.setattr(ref, "_worker", malformed_frame)
    result, rows = _play(tmp_path, monkeypatch, "malformed-frame")
    _assert_failure_rows(rows, "worker_failure", range(8))
    assert result.raw_scores == [0.0]


def test_truncated_worker_frame_is_recovered_without_hang(tmp_path, monkeypatch):
    def truncated_frame(conn, seed, per_stratum, indices, stride, split):
        if indices[0] == 0:
            # Complete the header but leave the announced payload unfinished.
            conn._send(struct.pack("!i", 100) + b"abc")
        conn.close()
        os._exit(23)

    monkeypatch.setattr(ref, "_worker", truncated_frame)
    result, rows = _play(tmp_path, monkeypatch, "truncated-frame")
    _assert_failure_rows(rows, "worker_failure", range(8))
    assert result.raw_scores == [0.0]


def test_live_worker_partial_frame_does_not_block_receive(tmp_path, monkeypatch):
    def partial_live_frame(conn, seed, per_stratum, indices, stride, split):
        if indices[0] != 0:
            return original_worker(conn, seed, per_stratum, indices, stride, split)
        conn.send(("reset_failures", []))
        assert conn.recv()[0] == "obs"
        # Keep the descriptor open and the process alive after writing only a
        # complete header and a fragment of its announced payload.
        conn._send(struct.pack("!i", 100) + b"abc")
        time.sleep(60)

    original_worker = ref._worker
    monkeypatch.setattr(ref, "_worker", partial_live_frame)
    result, rows = _play(tmp_path, monkeypatch, "live-partial-frame", max_steps=3)
    _assert_failure_rows(rows, "worker_failure", [0, 2, 4, 6])
    assert all(row["steps"] == 3 for row in rows if row["episode"] % 2)
    assert result.metadata["details"]["player_calls"] == 3


@pytest.mark.parametrize("initially_blocked", [False, True])
def test_live_worker_not_reading_does_not_block_send(tmp_path, monkeypatch, initially_blocked):
    send_buffer_remaining = []
    first_action_write_blocked = []
    original_entry = ref._worker_entry

    def stop_reading(conn, inherited, ready, seed, per_stratum, indices, stride, split):
        if indices[0] != 0:
            return original_entry(conn, inherited, ready, seed, per_stratum, indices, stride, split)
        # A duplex multiprocessing pipe is a Unix socket here. Shrink this
        # child's receive buffer before the launch handshake completes.
        sock = original_socket(fileno=conn.fileno())
        sock.setsockopt(ref.socket.SOL_SOCKET, ref.socket.SO_RCVBUF, 8192)
        sock.detach()

        def stalled_worker(*args):
            conn.send(("reset_failures", []))
            assert conn.recv()[0] == "obs"
            conn.send((indices, np.zeros((len(indices), ref.OBS_W), np.float32)))
            time.sleep(60)

        ref._worker = stalled_worker
        return original_entry(conn, inherited, ready, seed, per_stratum, indices, stride, split)

    original_os_write = ref.os.write
    prefilled = False
    parent_fds = []

    def record_backpressure(fd, data):
        try:
            written = original_os_write(fd, data)
        except BlockingIOError:
            if len(data) > 100_000 and not first_action_write_blocked:
                first_action_write_blocked.append(True)
            raise
        remaining = len(data) - written
        # Only the parent's large action frame can have an unsent payload this large.
        if remaining > 100_000:
            if not first_action_write_blocked:
                first_action_write_blocked.append(False)
            send_buffer_remaining.append(remaining)
        return written

    monkeypatch.setattr(ref.os, "write", record_backpressure)
    monkeypatch.setattr(ref, "_worker_entry", stop_reading)
    original_dumps = ref.mp_connection._ForkingPickler.dumps
    original_socket = ref.socket.socket

    def capture_parent_fd(*args, **kwargs):
        sock = original_socket(*args, **kwargs)
        if "fileno" in kwargs:
            parent_fds.append(sock.fileno())
        return sock

    monkeypatch.setattr(ref.socket, "socket", capture_parent_fd)

    def pad_action_frame(obj, protocol=None):
        nonlocal prefilled
        payload = bytes(original_dumps(obj, protocol))
        if isinstance(obj, tuple) and obj and obj[0] == "act":
            # Preserve the valid player actions; only this serialized test frame
            # receives trailing bytes, which the non-reading peer never decodes.
            payload += b"x" * 8_000_000
            if initially_blocked and 0 in obj[1] and not prefilled:
                prefilled = True
                parent_fd = parent_fds[0]
                sock = original_socket(fileno=parent_fd)
                sock.setsockopt(ref.socket.SOL_SOCKET, ref.socket.SO_SNDBUF, 4096)
                sock.detach()
                padding = b"x" * 65536
                while True:
                    try:
                        original_os_write(parent_fd, padding)
                    except BlockingIOError:
                        break
        return payload

    monkeypatch.setattr(ref.mp_connection._ForkingPickler, "dumps", pad_action_frame)
    result, rows = _play(
        tmp_path, monkeypatch,
        f"live-blocked-send-{'initial' if initially_blocked else 'partial'}",
        max_steps=3,
    )
    if initially_blocked:
        assert prefilled and first_action_write_blocked == [True]
    else:
        assert send_buffer_remaining and send_buffer_remaining[0] > 0
        assert first_action_write_blocked == [False]
    _assert_failure_rows(rows, "worker_failure", [0, 2, 4, 6])
    assert result.raw_scores == [0.0]


def test_worker_death_during_action_delivery_preserves_lockstep(tmp_path, monkeypatch):
    original = ref._worker
    original_entry = ref._worker_entry
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe()
    indices = list(range(0, 8, 2))
    # Launch this non-daemon helper in the test process, not inside the daemon
    # referee worker. It exits only after acknowledging its first action.
    proc = ctx.Process(target=original, args=(child, 7, 1, indices, 1, "live"))
    proc.start()
    child.close()
    assert parent.recv() == ("reset_failures", [])
    parent.send(("reset_failures", []))

    def reap_helper():
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=2)

    atexit.register(reap_helper)
    started = False

    def die_after_first_action(conn, inherited, ready, seed, per_stratum, indices_, stride, split):
        nonlocal started
        if indices_[0] != 0:
            return original_entry(conn, inherited, ready, seed, per_stratum, indices_, stride, split)
        assert not started
        started = True
        try:
            ready.send_bytes(b"\0")
        finally:
            ready.close()
        conn.send(("reset_failures", []))
        try:
            msg = conn.recv()
            assert msg[0] == "obs"
            parent.send(msg)
            conn.send(parent.recv())
            msg = conn.recv()
            assert msg[0] == "act"
            parent.send(msg)
            parent.recv()
        finally:
            parent.close()
            conn.close()
        os._exit(23)

    monkeypatch.setattr(ref, "_worker_entry", die_after_first_action)
    player = ZeroPlayer()
    result, rows = _play(tmp_path, monkeypatch, "action-delivery-death", player)
    reap_helper()
    atexit.unregister(reap_helper)
    assert not proc.is_alive()
    _assert_failure_rows(rows, "worker_failure", range(0, 8, 2))
    assert len(player.observations) > 2
    # No first observation is skipped, and no second tick is silently omitted
    # while the surviving worker's first action acknowledgement is outstanding.
    assert len(player.observations[0]) == 8
    assert len(player.observations[1]) == 4
    assert all(len(batch) == 4 for batch in player.observations[1:])
    assert all(row["steps"] == 920 for row in rows if row["episode"] % 2)
    assert not result.metadata["details"]["player_fault"]


def test_three_worker_middle_death_after_peer_ack_preserves_lockstep(tmp_path, monkeypatch):
    original_entry = ref._worker_entry

    def die_after_second_ack(conn, inherited, ready, seed, per_stratum, indices, stride, split):
        if indices[0] != 1:
            return original_entry(conn, inherited, ready, seed, per_stratum, indices, stride, split)
        try:
            ready.send_bytes(b"\0")
        finally:
            ready.close()
        conn.send(("reset_failures", []))
        try:
            for _ in range(2):
                assert conn.recv()[0] == "obs"
                conn.send((indices, np.zeros((len(indices), ref.OBS_W), np.float32)))
                assert conn.recv()[0] == "act"
                conn.send(True)
        finally:
            conn.close()
        os._exit(23)

    monkeypatch.setattr(ref, "_worker_entry", die_after_second_ack)
    player = ZeroPlayer()
    result, rows = _play(
        tmp_path, monkeypatch, "three-worker-action-death", player, workers=3, max_steps=3
    )
    failed = {1, 4, 7}
    _assert_failure_rows(rows, "worker_failure", failed)
    assert all(row["steps"] == 3 for row in rows if row["episode"] not in failed)
    assert len(player.observations) == 3
    assert all(len(batch) == 8 for batch in player.observations[:2])
    assert len(player.observations[2]) == 5
    assert result.metadata["details"]["player_calls"] == 3


def test_worker_death_during_observation_does_not_consume_control_step(tmp_path, monkeypatch):
    original = ref._worker

    def die_before_second_observation(conn, seed, per_stratum, indices, stride, split):
        if indices[0] != 1:
            return original(conn, seed, per_stratum, indices, stride, split)
        conn.send(("reset_failures", []))
        try:
            assert conn.recv()[0] == "obs"
            conn.send((indices, np.zeros((len(indices), ref.OBS_W), np.float32)))
            assert conn.recv()[0] == "act"
            conn.send(True)
        finally:
            conn.close()
        os._exit(23)

    monkeypatch.setattr(ref, "_worker", die_before_second_observation)
    player = ZeroPlayer()
    result, rows = _play(
        tmp_path, monkeypatch, "observation-death", player, workers=3, max_steps=3
    )
    failed = {1, 4, 7}
    _assert_failure_rows(rows, "worker_failure", failed)
    assert all(row["steps"] == 3 for row in rows if row["episode"] not in failed)
    assert len(player.observations) == 3
    assert [len(batch) for batch in player.observations] == [8, 5, 5]
    assert result.metadata["details"]["player_calls"] == 3


def test_decoded_malformed_observation_is_recovered_without_hang(tmp_path, monkeypatch):
    original = ref._worker

    def malformed_observation(conn, seed, per_stratum, indices, stride, split):
        if indices[0] == 1:
            conn.send(("reset_failures", []))
            try:
                assert conn.recv()[0] == "obs"
                conn.send(True)
            finally:
                conn.close()
            os._exit(23)
        # Exercise the healthy path's strict summary validation as well.
        return original(conn, seed, per_stratum, indices, stride, split)

    monkeypatch.setattr(ref, "_worker", malformed_observation)
    result, rows = _play(tmp_path, monkeypatch, "malformed-observation")
    _assert_failure_rows(rows, "worker_failure", [1, 3, 5, 7])
    assert all(row["steps"] == 920 for row in rows if row["episode"] % 2 == 0)


def test_incomplete_summary_rows_become_worker_failure(tmp_path, monkeypatch):
    original = ref._worker

    def incomplete_summary(conn, seed, per_stratum, indices, stride, split):
        if indices[0] != 0:
            return original(conn, seed, per_stratum, indices, stride, split)
        conn.send(("reset_failures", []))
        try:
            while True:
                msg = conn.recv()
                if msg[0] == "obs":
                    conn.send((indices, np.zeros((len(indices), ref.OBS_W), np.float32)))
                elif msg[0] == "summary":
                    conn.send(([{"episode": i} for i in indices], {}))
                    return
                else:
                    conn.send(True)
        finally:
            conn.close()

    monkeypatch.setattr(ref, "_worker", incomplete_summary)
    result, rows = _play(tmp_path, monkeypatch, "incomplete-summary")
    _assert_failure_rows(rows, "worker_failure", [0, 2, 4, 6])
    assert all(row["steps"] == 920 for row in rows if row["episode"] % 2)


def test_non_finite_observation_becomes_worker_failure(tmp_path, monkeypatch):
    original = ref._worker

    def non_finite_observation(conn, seed, per_stratum, indices, stride, split):
        if indices[0] != 1:
            return original(conn, seed, per_stratum, indices, stride, split)
        conn.send(("reset_failures", []))
        try:
            assert conn.recv()[0] == "obs"
            conn.send((indices, np.full((len(indices), ref.OBS_W), np.nan, np.float32)))
        finally:
            conn.close()
        os._exit(23)

    monkeypatch.setattr(ref, "_worker", non_finite_observation)
    result, rows = _play(tmp_path, monkeypatch, "non-finite-observation", max_steps=3)
    _assert_failure_rows(rows, "worker_failure", [1, 3, 5, 7])
    assert all(row["steps"] == 3 for row in rows if row["episode"] % 2 == 0)
    assert result.metadata["details"]["player_calls"] == 3


@pytest.mark.parametrize("field,bad_value", [
    ("success_step", -1),
    ("peg_first_seat_step", 1),
    ("done_at", -1),
])
def test_out_of_range_summary_timestamp_becomes_worker_failure(
    tmp_path, monkeypatch, field, bad_value
):
    original = ref._worker

    def invalid_timestamp(conn, seed, per_stratum, indices, stride, split):
        if indices[0] != 0:
            return original(conn, seed, per_stratum, indices, stride, split)
        ctx = mp.get_context("fork")
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=original, args=(child, seed, per_stratum, indices, stride, split))
        # The fixture's non-daemon helper is owned and reaped by the test process.
        proc.daemon = False
        proc.start()
        child.close()
        try:
            conn.send(parent.recv())
            while True:
                msg = conn.recv()
                if msg[0] == "obs":
                    parent.send(msg)
                    conn.send(parent.recv())
                elif msg[0] == "summary":
                    parent.send(msg)
                    rows, traj = parent.recv()
                    for row in rows:
                        if field == "done_at":
                            row["step_results"][0][field] = bad_value
                        elif field == "peg_first_seat_step":
                            row[field] = [bad_value] * row["n_pegs"]
                        else:
                            row[field] = bad_value
                    conn.send((rows, traj))
                    return
                else:
                    parent.send(msg)
                    conn.send(parent.recv())
        finally:
            conn.close()
            parent.close()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)

    monkeypatch.setattr(ref, "_worker", invalid_timestamp)
    result, rows = _play(tmp_path, monkeypatch, f"invalid-{field}", max_steps=3)
    _assert_failure_rows(rows, "worker_failure", [0, 2, 4, 6])
    assert all(row["steps"] == 3 for row in rows if row["episode"] % 2)


def _summary_mutation_worker(mutation, announced_reset_failures=()):
    """Run the real worker in a helper and mutate only its public summary."""
    original = ref._worker

    def mutate_summary(conn, seed, per_stratum, indices, stride, split):
        if indices[0] != 0:
            return original(conn, seed, per_stratum, indices, stride, split)
        ctx = mp.get_context("fork")
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=original, args=(child, seed, per_stratum, indices, stride, split))
        proc.daemon = False
        proc.start()
        child.close()
        try:
            startup = parent.recv()
            assert startup[0] == "reset_failures"
            conn.send(("reset_failures", sorted(set(startup[1]) | set(announced_reset_failures))))
            while True:
                msg = conn.recv()
                if msg[0] == "summary":
                    parent.send(msg)
                    rows, traj = parent.recv()
                    mutation(rows, traj)
                    conn.send((rows, traj))
                    return
                parent.send(msg)
                conn.send(parent.recv())
        finally:
            conn.close()
            parent.close()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)

    return mutate_summary


def test_oversized_summary_number_becomes_worker_failure(tmp_path, monkeypatch):
    def oversized_score(rows, traj):
        rows[0]["score"] = 10**100

    monkeypatch.setattr(ref, "_worker", _summary_mutation_worker(oversized_score))
    result, rows = _play(
        tmp_path, monkeypatch, "oversized-summary-number", max_steps=3, stride=0
    )
    _assert_failure_rows(rows, "worker_failure", [0, 2, 4, 6])
    assert all(row["steps"] == 3 for row in rows if row["episode"] % 2)
    assert result.raw_scores == [0.0]


def test_non_json_summary_payload_becomes_worker_failure(tmp_path, monkeypatch):
    def numpy_task_step(rows, traj):
        rows[0]["task"]["steps"][0]["injected"] = np.float32(0.0)

    monkeypatch.setattr(ref, "_worker", _summary_mutation_worker(numpy_task_step))
    result, rows = _play(
        tmp_path, monkeypatch, "non-json-summary-payload", max_steps=3, stride=0
    )
    _assert_failure_rows(rows, "worker_failure", [0, 2, 4, 6])
    assert all(row["steps"] == 3 for row in rows if row["episode"] % 2)
    assert result.raw_scores == [0.0]


def test_array_valued_reset_summary_is_contained_as_worker_failure(tmp_path, monkeypatch):
    def array_valued_step(rows, traj):
        # Preserve the dictionary shape so raw equality would compare this value.
        rows[0]["task"]["steps"][0]["kind"] = np.zeros(2)

    monkeypatch.setattr(
        ref, "_worker", _summary_mutation_worker(array_valued_step, announced_reset_failures=[0])
    )
    result, rows = _play(
        tmp_path, monkeypatch, "array-reset-summary", max_steps=3, stride=0
    )
    _assert_failure_rows(rows, "worker_failure", [0, 2, 4, 6])
    assert all(row["steps"] == 3 for row in rows if row["episode"] % 2)
    assert result.raw_scores == [0.0]


def test_announced_reset_failure_cannot_return_nonzero_summary(tmp_path, monkeypatch):
    def nonzero_reset_score(rows, traj):
        rows[0]["terminal_reason"] = "reset_failure"
        rows[0]["failure_reason"] = "reset_failure"
        rows[0]["score"] = 0.5

    monkeypatch.setattr(
        ref, "_worker", _summary_mutation_worker(nonzero_reset_score, announced_reset_failures=[0])
    )
    result, rows = _play(
        tmp_path, monkeypatch, "nonzero-reset-summary", max_steps=3, stride=0
    )
    _assert_failure_rows(rows, "worker_failure", [0, 2, 4, 6])
    assert all(row["steps"] == 3 for row in rows if row["episode"] % 2)
    assert result.raw_scores == [0.0]


def test_later_worker_death_preserves_contained_reset_failures(tmp_path, monkeypatch):
    original = ref._worker

    def fail_construction_then_worker(conn, seed, per_stratum, indices, stride, split):
        if indices[0] != 0:
            return original(conn, seed, per_stratum, indices, stride, split)
        # Episode 0 fails construction; episode 2 is constructed successfully.
        ref._sample(seed, per_stratum, split)
        conn.send(("reset_failures", [0]))
        try:
            assert conn.recv()[0] == "obs"
            conn.send(([2], np.zeros((1, ref.OBS_W), np.float32)))
            assert conn.recv()[0] == "act"
            conn.send(True)
        finally:
            conn.close()
        os._exit(23)

    monkeypatch.setattr(ref, "_worker", fail_construction_then_worker)
    result, rows = _play(
        tmp_path, monkeypatch, "reset-then-worker-death", max_steps=3, stride=0
    )
    _assert_failure_rows(rows, "reset_failure", [0])
    _assert_failure_rows(rows, "worker_failure", [2, 4, 6])
    assert all(row["steps"] == 3 for row in rows if row["episode"] % 2)
    assert result.metadata["details"]["terminal_reasons"]["reset_failure"] == 1
    assert result.raw_scores == [0.0]


def test_failed_record_has_normal_compound_schema():
    _, conds = ref._sample(7, 1, "live")
    row = ref._failed_episode_row(conds[0], 0, "reset_failure")
    _assert_failure_rows([row], "reset_failure", [0])
    assert row["format"] == ref.RECORD_FORMAT
    assert np.isfinite(row["score"])
