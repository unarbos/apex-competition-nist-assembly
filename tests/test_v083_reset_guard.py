"""v0.8.3/v0.8.4 reset, worker-transport, and summary-containment guards."""

from __future__ import annotations

import json
import atexit
import pickle
import ctypes
import importlib
import multiprocessing as mp
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import resource
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
    reap_test_helper = getattr(ref._worker, "reap_test_helper", None)
    if reap_test_helper is not None:
        reap_test_helper()
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


def test_observation_and_action_requests_reach_all_workers_before_receives(tmp_path, monkeypatch):
    original = ref._worker
    ctx = mp.get_context("fork")
    peer_received = {phase: ctx.Event() for phase in ("obs", "act")}

    def gated_worker(conn, seed, per_stratum, indices, stride, split):
        if indices[0] != 0:
            return original(conn, seed, per_stratum, indices, stride, split)
        conn.send(("reset_failures", []))
        try:
            for phase in ("obs", "act"):
                assert conn.recv()[0] == phase
                # A sequential parent is now waiting for this reply and cannot
                # send the peer's matching request. Parallel sends release us.
                if not peer_received[phase].wait(10):
                    os._exit(24)
                conn.send(
                    (indices, np.zeros((len(indices), ref.OBS_W), np.float32))
                    if phase == "obs" else True
                )
            assert conn.recv()[0] == "fail_all"
            assert conn.recv()[0] == "summary"
            _, conds = ref._sample(seed, per_stratum, split)
            rows = [ref._failed_episode_row(conds[i], i, "time_limit") for i in indices]
            conn.send((rows, {}))
        finally:
            conn.close()

    def peer_worker(conn, seed, per_stratum, indices, stride, split):
        if indices[0] != 1:
            return original(conn, seed, per_stratum, indices, stride, split)
        conn.send(("reset_failures", []))
        for expected in ("obs", "act"):
            assert conn.recv()[0] == expected
            peer_received[expected].set()
            conn.send(
                (indices, np.zeros((len(indices), ref.OBS_W), np.float32))
                if expected == "obs" else True
            )
        assert conn.recv()[0] == "fail_all"
        assert conn.recv()[0] == "summary"
        _, conds = ref._sample(seed, per_stratum, split)
        rows = [ref._failed_episode_row(conds[i], i, "time_limit") for i in indices]
        conn.send((rows, {}))
        conn.close()

    def phased_worker(conn, seed, per_stratum, indices, stride, split):
        return (gated_worker if indices[0] == 0 else peer_worker)(
            conn, seed, per_stratum, indices, stride, split
        )

    monkeypatch.setattr(ref, "_worker", phased_worker)
    result, rows = _play(tmp_path, monkeypatch, "parallel-request-phases", max_steps=1)
    assert all(event.is_set() for event in peer_received.values())
    assert result.metadata["details"]["player_fault"] is None


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


def test_exited_worker_final_frame_is_drained_before_eof(tmp_path, monkeypatch):
    original = ref._worker

    def send_final_frame_then_exit(conn, seed, per_stratum, indices, stride, split):
        if indices[0] != 0:
            return original(conn, seed, per_stratum, indices, stride, split)
        conn.send(("reset_failures", []))
        conn.close()
        os._exit(0)

    monkeypatch.setattr(ref, "_worker", send_final_frame_then_exit)
    result, rows = _play(tmp_path, monkeypatch, "final-frame-at-exit", max_steps=3)
    _assert_failure_rows(rows, "worker_failure", [0, 2, 4, 6])
    # A valid final frame, not process liveness, completes startup. The next
    # request finds the empty slice idle; healthy siblings retain their score path.
    assert result.metadata["details"]["player_fault"] is None
    assert all(row["steps"] == 3 for row in rows if row["episode"] % 2)


def test_exited_worker_truncated_final_frame_still_fails(tmp_path, monkeypatch):
    original = ref._worker

    def truncated_final_frame(conn, seed, per_stratum, indices, stride, split):
        if indices[0] != 0:
            return original(conn, seed, per_stratum, indices, stride, split)
        conn._send(struct.pack("!i", 100) + b"abc")
        conn.close()
        os._exit(0)

    monkeypatch.setattr(ref, "_worker", truncated_final_frame)
    result, rows = _play(tmp_path, monkeypatch, "truncated-final-frame", max_steps=3)
    _assert_failure_rows(rows, "worker_failure", [0, 2, 4, 6])
    assert all(row["steps"] == 3 for row in rows if row["episode"] % 2)


def test_exited_worker_final_summary_is_preserved(tmp_path, monkeypatch):
    """A complete summary sent immediately before exit is not worker death."""
    original = ref._worker

    def final_summary_then_exit(conn, seed, per_stratum, indices, stride, split):
        if indices[0] != 0:
            return original(conn, seed, per_stratum, indices, stride, split)
        _, conds = ref._sample(seed, per_stratum, split)
        for cond in conds:
            # The worker below consumes exactly three action frames, so its
            # time_limit rows must use the same three-step public horizon.
            cond.max_steps = 3
        conn.send(("reset_failures", []))
        try:
            for _ in range(3):
                assert conn.recv()[0] == "obs"
                conn.send((indices, np.zeros((len(indices), ref.OBS_W), np.float32)))
                assert conn.recv()[0] == "act"
                conn.send(True)
            assert conn.recv()[0] == "fail_all"
            # Queue the final response and close without ever waiting for its
            # request. EOF draining must preserve this complete final frame.
            rows = []
            for i in indices:
                row = ref._failed_episode_row(conds[i], i, "time_limit")
                row["steps"] = 3
                del row["failure_reason"]
                rows.append(row)
            conn.send((rows, {}))
        finally:
            conn.close()
        os._exit(0)

    monkeypatch.setattr(ref, "_worker", final_summary_then_exit)
    result, rows = _play(tmp_path, monkeypatch, "final-summary-at-exit", max_steps=3)
    assert not any(row["terminal_reason"] == "worker_failure" for row in rows)
    assert [row["terminal_reason"] for row in rows] == ["time_limit"] * 8
    assert all(row["score"] == 0.0 for row in rows)


def test_already_complete_worker_finalizes_without_ack_round_trip(tmp_path, monkeypatch):
    """An idle worker consumes ordered finalization and sends only its summary."""
    original = ref._worker

    def already_complete(conn, seed, per_stratum, indices, stride, split):
        if indices[0] != 0:
            return original(conn, seed, per_stratum, indices, stride, split)
        real_make = ref._make_episode

        def completed_episode(cond, rng):
            ep = real_make(cond, rng)
            ep.fail("time_limit")
            return ep

        ref._make_episode = completed_episode
        return original(conn, seed, per_stratum, indices, stride, split)

    monkeypatch.setattr(ref, "_worker", already_complete)
    result, rows = _play(tmp_path, monkeypatch, "already-complete-summary", max_steps=3)
    assert result.metadata["details"]["player_calls"] == 3
    assert result.metadata["details"]["terminal_reasons"] == {"time_limit": 8}
    assert all(row["score"] == 0.0 for row in rows)


def test_half_open_final_summary_is_preserved(tmp_path, monkeypatch):
    """An observed request failure cannot invalidate an in-flight valid summary."""
    original = ref._worker
    original_os_write = ref.os.write
    original_dumps = ref.mp_connection._ForkingPickler.dumps
    request_error_seen = []
    write_event = mp.get_context("fork").Event()
    fail_all_delivered = mp.get_context("fork").Event()
    original_socket = ref.socket.socket

    def half_open_summary(conn, seed, per_stratum, indices, stride, split):
        if indices[0] != 0:
            return original(conn, seed, per_stratum, indices, stride, split)
        # Shrink the response receive buffer before forking the second worker,
        # which closes every other child descriptor retained by its peer.
        sock = original_socket(fileno=conn.fileno())
        sock.setsockopt(ref.socket.SOL_SOCKET, ref.socket.SO_RCVBUF, 4096)
        sock.detach()
        _, conds = ref._sample(seed, per_stratum, split)
        conn.send(("reset_failures", []))
        for _ in range(3):
            assert conn.recv()[0] == "obs"
            conn.send((indices, np.zeros((len(indices), ref.OBS_W), np.float32)))
            assert conn.recv()[0] == "act"
            conn.send(True)
        assert conn.recv()[0] == "fail_all"
        rows = []
        for i in indices:
            row = ref._failed_episode_row(conds[i], i, "time_limit")
            row["steps"] = 3
            del row["failure_reason"]
            rows.append(row)
        # The worker queues a response prefix after fail_all, without waiting for
        # the summary command. Its request direction then half-closes. Write the
        # actual frame bytes: Connection._send_bytes() itself adds a frame header.
        def write_all(data):
            view = memoryview(data)
            while view:
                written = original_os_write(conn.fileno(), view)
                assert written > 0
                view = view[written:]

        payload = ref.mp_connection._ForkingPickler.dumps((rows, {}))
        write_all(struct.pack("!i", len(payload)) + payload[:60])
        sock = original_socket(fileno=conn.fileno())
        sock.shutdown(ref.socket.SHUT_RD)
        sock.detach()
        # Only the parent's failing summary write releases the response remainder.
        fail_all_delivered.set()
        if not write_event.wait(5):
            os._exit(23)
        write_all(payload[60:])
        time.sleep(60)

    fail_all_payload = bytes(original_dumps(("fail_all", "time_limit")))
    fail_all_frame = struct.pack("!i", len(fail_all_payload)) + fail_all_payload

    def record_write_error(fd, data):
        data = bytes(data)
        delivered = data == fail_all_frame
        try:
            written = original_os_write(fd, data)
            if delivered:
                # This small complete frame fits atomically. Hold only its return
                # until the peer half-closes; the next summary write observes EPIPE.
                assert written == len(data)
                assert fail_all_delivered.wait(5)
            return written
        except (BrokenPipeError, OSError) as error:
            if not request_error_seen:
                request_error_seen.append(type(error).__name__)
                write_event.set()
            raise

    monkeypatch.setattr(ref.os, "write", record_write_error)
    monkeypatch.setattr(ref, "_worker", half_open_summary)
    result, rows = _play(tmp_path, monkeypatch, "half-open-summary", max_steps=3)
    assert request_error_seen == ["BrokenPipeError"]
    assert result.metadata["details"]["terminal_reasons"] == {"time_limit": 8}
    assert all(row["score"] == 0.0 for row in rows)


def test_false_action_ack_is_rejected_without_waiting_for_another_frame(tmp_path, monkeypatch):
    """A valid pickle containing False is a reply, not an incomplete frame."""
    original = ref._worker

    def false_ack(conn, seed, per_stratum, indices, stride, split):
        if indices[0] != 0:
            return original(conn, seed, per_stratum, indices, stride, split)
        conn.send(("reset_failures", []))
        assert conn.recv()[0] == "obs"
        conn.send((indices, np.zeros((len(indices), ref.OBS_W), np.float32)))
        assert conn.recv()[0] == "act"
        conn.send(False)
        time.sleep(60)

    monkeypatch.setattr(ref, "_worker", false_ack)
    result, rows = _play(tmp_path, monkeypatch, "false-action-ack", max_steps=3)
    _assert_failure_rows(rows, "worker_failure", [0, 2, 4, 6])
    assert all(row["steps"] == 3 for row in rows if row["episode"] % 2)
    assert result.metadata["details"]["player_calls"] == 3


def test_complete_malformed_frame_fails_open_worker_immediately(tmp_path, monkeypatch):
    """An invalid complete pickle cannot hide behind an indefinitely open peer."""
    original = ref._worker

    def malformed_summary(conn, seed, per_stratum, indices, stride, split):
        if indices[0] != 0:
            return original(conn, seed, per_stratum, indices, stride, split)
        conn.send(("reset_failures", []))
        for _ in range(3):
            assert conn.recv()[0] == "obs"
            conn.send((indices, np.zeros((len(indices), ref.OBS_W), np.float32)))
            assert conn.recv()[0] == "act"
            conn.send(True)
        assert conn.recv()[0] == "fail_all"
        assert conn.recv()[0] == "summary"
        conn._send(struct.pack("!i", 1) + b"X")
        time.sleep(60)

    monkeypatch.setattr(ref, "_worker", malformed_summary)
    monkeypatch.setattr(ref.NistLiteReferee, "TRANSPORT_STALL_TIMEOUT_S", 60.0)
    result, rows = _play(tmp_path, monkeypatch, "open-malformed-summary", max_steps=3)
    _assert_failure_rows(rows, "worker_failure", [0, 2, 4, 6])
    assert all(row["steps"] == 3 for row in rows if row["episode"] % 2)


def test_high_numbered_worker_descriptors_do_not_fail_siblings(tmp_path, monkeypatch):
    """Shared readiness uses poll(), not select()'s fixed-size descriptor set."""
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    if min(soft_limit, hard_limit) < 1200:
        pytest.skip("requires at least 1200 open file descriptors")
    fds = [os.open("/dev/null", os.O_RDONLY) for _ in range(1100)]
    original_pipe = ref.mp.get_context("fork").Pipe
    checked = []

    def checked_pipe(duplex=True):
        parent, child = original_pipe(duplex)
        if duplex:
            checked.append(parent.fileno())
            assert parent.fileno() >= 1024
        return parent, child

    monkeypatch.setattr(ref.mp.get_context("fork"), "Pipe", checked_pipe)
    try:
        result, rows = _play(tmp_path, monkeypatch, "high-worker-fd", max_steps=3)
    finally:
        for fd in fds:
            os.close(fd)
    assert checked
    assert result.metadata["details"]["terminal_reasons"] == {"time_limit": 8}
    assert all(row["steps"] == 3 for row in rows)


def test_failed_request_without_response_has_bounded_drain(tmp_path, monkeypatch):
    """A half-open peer that never replies cannot suspend finalization forever."""
    original = ref._worker
    original_socket = ref.socket.socket
    original_write = ref.os.write
    original_poll = ref.select.poll
    request_errors = []
    waits = []

    class RecordPoll:
        def __init__(self):
            self.poller = original_poll()

        def register(self, *args, **kwargs):
            return self.poller.register(*args, **kwargs)

        def poll(self, timeout=None):
            waits.append(timeout)
            return self.poller.poll(timeout)

    def half_open_silence(conn, seed, per_stratum, indices, stride, split):
        if indices[0] != 0:
            return original(conn, seed, per_stratum, indices, stride, split)
        sock = original_socket(fileno=conn.fileno())
        sock.setsockopt(ref.socket.SOL_SOCKET, ref.socket.SO_RCVBUF, 4096)
        sock.shutdown(ref.socket.SHUT_RD)
        sock.detach()
        conn.send(("reset_failures", []))
        time.sleep(60)

    def record_request_error(fd, data):
        try:
            return original_write(fd, data)
        except BrokenPipeError as error:
            request_errors.append(type(error).__name__)
            raise

    monkeypatch.setattr(ref.os, "write", record_request_error)
    monkeypatch.setattr(ref.select, "poll", RecordPoll)
    monkeypatch.setattr(ref, "_worker", half_open_silence)
    monkeypatch.setattr(ref.NistLiteReferee, "TRANSPORT_STALL_TIMEOUT_S", 0.2)
    result, rows = _play(tmp_path, monkeypatch, "half-open-silence", max_steps=3)
    assert "BrokenPipeError" in request_errors
    assert any(value is not None and value <= 200 for value in waits)
    assert not any(row["terminal_reason"] == "worker_failure" for row in rows if row["episode"] % 2)
    _assert_failure_rows(rows, "worker_failure", [0, 2, 4, 6])
    assert all(row["steps"] == 3 for row in rows if row["episode"] % 2)


def test_continuous_partial_frame_does_not_starve_siblings(tmp_path, monkeypatch):
    """A legal-sized incomplete frame yields to its sibling between read budgets."""
    original_socket = ref.socket.socket
    original_entry = ref._worker_entry
    original_read = ref.os.read
    ctx = mp.get_context("fork")
    sibling_sent = ctx.Value(ctypes.c_bool, False, lock=False)
    streaming_started = ctx.Value(ctypes.c_bool, False, lock=False)
    release = ctx.Value(ctypes.c_bool, False, lock=False)
    observed = set()
    fds = []
    busy_reads = 0

    def stream_incomplete_frame(conn, inherited, ready, seed, per_stratum, indices, stride, split):
        if indices[0] != 0:
            return original_entry(conn, inherited, ready, seed, per_stratum, indices, stride, split)
        for other in inherited:
            other.close()
        inherited.clear()
        try:
            ready.send_bytes(b"\0")
        finally:
            ready.close()
        if indices[0] != 0:
            conn.send(("reset_failures", []))
            sibling_sent.value = True
            time.sleep(60)
            return
        # _send() writes these bytes unchanged; _send_bytes() would frame them
        # as a complete nested message instead of exposing an incomplete header.
        conn._send(struct.pack("!i", (1 << 20) - 1))
        streaming_started.value = True
        while not release.value:
            try:
                conn._send(b"x" * 65536)
            except (BrokenPipeError, EOFError, OSError):
                return
            time.sleep(0.001)
        conn.close()

    def record_read(fd, count):
        nonlocal busy_reads
        data = original_read(fd, count)
        if fd in fds:
            observed.add(fd)
            if streaming_started.value and fd == fds[0]:
                busy_reads += 1
            # Require a second complete read budget and sibling IO before EOF.
            if busy_reads > 16 and len(observed) == len(fds) == 2:
                release.value = True
        return data

    def capture_fd(*args, **kwargs):
        sock = original_socket(*args, **kwargs)
        if "fileno" in kwargs:
            fds.append(sock.fileno())
        return sock

    monkeypatch.setattr(ref, "_worker_entry", stream_incomplete_frame)
    monkeypatch.setattr(ref.socket, "socket", capture_fd)
    monkeypatch.setattr(ref.os, "read", record_read)
    monkeypatch.setattr(ref.NistLiteReferee, "TRANSPORT_STALL_TIMEOUT_S", 60.0)
    result, rows = _play(tmp_path, monkeypatch, "continuous-partial-frame", max_steps=0)
    assert busy_reads > 16
    assert len(fds) == 2
    assert observed == set(fds)
    _assert_failure_rows(rows, "worker_failure", [0, 2, 4, 6])
    assert all(row["terminal_reason"] == "time_limit" for row in rows if row["episode"] % 2)


def test_read_budget_boundary_starts_finite_stall_timer(tmp_path, monkeypatch):
    """A stopped partial frame cannot return from its read budget to poll(None)."""
    original = ref._worker
    original_read = ref.os.read
    original_poll = ref.select.poll
    reads = []
    waits = []

    class RecordPoll:
        def __init__(self):
            self.poller = original_poll()

        def register(self, *args, **kwargs):
            return self.poller.register(*args, **kwargs)

        def poll(self, timeout=None):
            waits.append(timeout)
            return self.poller.poll(timeout)

    def stopped_frame(conn, seed, per_stratum, indices, stride, split):
        if indices[0] != 0:
            return original(conn, seed, per_stratum, indices, stride, split)
        conn._send(struct.pack("!i", (1 << 20) - 1))
        conn._send(b"x" * (16 * 65536 - 4))
        time.sleep(60)

    def record_read(fd, count):
        data = original_read(fd, count)
        reads.append((fd, len(data)))
        return data

    monkeypatch.setattr(ref, "_worker", stopped_frame)
    monkeypatch.setattr(ref.os, "read", record_read)
    monkeypatch.setattr(ref.select, "poll", RecordPoll)
    monkeypatch.setattr(ref.NistLiteReferee, "TRANSPORT_STALL_TIMEOUT_S", 0.2)
    result, rows = _play(tmp_path, monkeypatch, "read-budget-stall", max_steps=0)
    assert any(size == 65536 for _, size in reads)
    assert any(value is not None and value <= 200 for value in waits)
    assert any(value is not None and value <= 200 for value in waits[:-1])
    _assert_failure_rows(rows, "worker_failure", [0, 2, 4, 6])
    assert all(row["terminal_reason"] == "time_limit" for row in rows if row["episode"] % 2)


def test_mixed_request_and_receive_stalls_do_not_crash_round(tmp_path, monkeypatch):
    """Each timeout source is checked independently without a None subtraction."""
    original = ref._worker
    original_socket = ref.socket.socket
    original_write = ref.os.write
    request_errors = []

    def mixed_stalls(conn, seed, per_stratum, indices, stride, split):
        if indices[0] == 0:
            sock = original_socket(fileno=conn.fileno())
            sock.setsockopt(ref.socket.SOL_SOCKET, ref.socket.SO_RCVBUF, 4096)
            sock.shutdown(ref.socket.SHUT_RD)
            sock.detach()
            conn.send(("reset_failures", []))
            time.sleep(60)
            return
        conn.send(("reset_failures", []))
        conn._send(struct.pack("!i", 100) + b"x")
        time.sleep(60)

    def record_request_error(fd, data):
        try:
            return original_write(fd, data)
        except BrokenPipeError as error:
            request_errors.append(type(error).__name__)
            raise

    monkeypatch.setattr(ref, "_worker", mixed_stalls)
    monkeypatch.setattr(ref.os, "write", record_request_error)
    monkeypatch.setattr(ref.NistLiteReferee, "TRANSPORT_STALL_TIMEOUT_S", 0.2)
    result, rows = _play(tmp_path, monkeypatch, "mixed-stall-sources", max_steps=0)
    assert "BrokenPipeError" in request_errors
    assert all(row["terminal_reason"] == "worker_failure" for row in rows)
    assert all(row["score"] == 0.0 for row in rows)


@pytest.mark.parametrize("size", [4096, 1 << 40])
def test_oversized_bytearray_pickle_is_rejected_before_expansion(
    tmp_path, monkeypatch, size
):
    """Both above-limit and truncated below-limit declarations fail closed."""
    original = ref._worker

    def oversized_scalar(conn, seed, per_stratum, indices, stride, split):
        if indices[0] != 0:
            return original(conn, seed, per_stratum, indices, stride, split)
        conn.send(("reset_failures", []))
        payload = pickle.BYTEARRAY8 + size.to_bytes(8, "little") + pickle.STOP
        conn._send_bytes(payload)
        time.sleep(60)

    monkeypatch.setattr(ref, "_worker", oversized_scalar)
    monkeypatch.setattr(ref.NistLiteReferee, "TRANSPORT_STALL_TIMEOUT_S", 60.0)
    result, rows = _play(tmp_path, monkeypatch, f"oversized-bytearray-{size}", max_steps=0)
    _assert_failure_rows(rows, "worker_failure", [0, 2, 4, 6])
    assert all(row["terminal_reason"] == "time_limit" for row in rows if row["episode"] % 2)


@pytest.mark.parametrize("protocol", [4, 5])
def test_bounded_pickle_preserves_valid_bytearray_frames(protocol):
    value = bytearray(b"x" * 65536)
    assert ref._limited_pickle_loads(pickle.dumps(value, protocol=protocol)) == value


def test_schema_maximum_recording_and_worker_frame_bounds():
    _, conds = ref._sample(7, 64, "live")
    slices = [list(range(w, len(conds), 4)) for w in range(4)]
    limits = ref._frame_limits(conds, slices, 1, True)
    assert sum(limits) > 256 * 1024 * 1024
    assert ref._frame_limits(conds, [list(range(len(conds)))], 1, True)[0] > 256 * 1024 * 1024
    for w, indices in enumerate(slices):
        rows = []
        for i in indices:
            c = conds[i]
            row = ref._failed_episode_row(c, i, "worker_failure")
            event = {
                "kind": "reorder", "step": c.max_steps,
                "edges": [[a, b] for a in range(len(c.compound.steps))
                          for b in range(a + 1, len(c.compound.steps))],
                "validation_forced": True,
            }
            row["events"] = [event] * (1 + 2 * len(c.compound.steps))
            row["events_fired"] = len(row["events"])
            rows.append(row)
        assert ref._valid_worker_summary((rows, {}), conds, indices)
        limit_without_recording = ref._frame_limits(
            conds, [indices], 0, False
        )[0]
        assert len(pickle.dumps((rows, {}), protocol=4)) <= limit_without_recording
        assert len(pickle.dumps((rows, {}), protocol=4)) <= limits[w]


def test_frame_limits_are_worker_local():
    _, conds = ref._sample(7, 1, "live")
    limits = ref._frame_limits(conds, [[0], [1]], 1, True)
    assert limits[0] > limits[1] >= 1048576


def test_legacy_v05_frame_limits_do_not_require_compound_fields(monkeypatch):
    monkeypatch.setenv("NIST_PROFILE", "v05")
    importlib.reload(ref)
    assert not ref.COMPOUND
    _, conds = ref._sample(7, 1, "live")
    assert not hasattr(conds[0], "compound")
    limits = ref._frame_limits(conds, [[0], [1]], 1, True)
    assert limits[0] >= limits[1] >= 1048576


def test_deserializer_system_exit_is_contained_to_worker(tmp_path, monkeypatch):
    original = ref._worker

    def system_exit_frame(conn, seed, per_stratum, indices, stride, split):
        if indices[0] != 0:
            return original(conn, seed, per_stratum, indices, stride, split)
        conn.send(("reset_failures", []))

        class Exit:
            def __reduce__(self):
                return (sys.exit, (23,))

        conn.send(Exit())
        time.sleep(60)

    monkeypatch.setattr(ref, "_worker", system_exit_frame)
    result, rows = _play(tmp_path, monkeypatch, "deserializer-system-exit", max_steps=0)
    _assert_failure_rows(rows, "worker_failure", [0, 2, 4, 6])
    assert all(row["terminal_reason"] == "time_limit" for row in rows if row["episode"] % 2)


@pytest.mark.parametrize("extended", [False, True])
def test_announced_frame_size_is_bounded_before_payload_allocation(
    tmp_path, monkeypatch, extended
):
    original = ref._worker

    def huge_frame(conn, seed, per_stratum, indices, stride, split):
        if indices[0] != 0:
            return original(conn, seed, per_stratum, indices, stride, split)
        conn.send(("reset_failures", []))
        size = (1 << 40) if extended else (1 << 30)
        conn._send_bytes(
            struct.pack("!iQ", -1, size) if extended else struct.pack("!i", size)
        )
        while True:
            try:
                conn._send_bytes(b"x" * 65536)
            except (BrokenPipeError, EOFError, OSError):
                return

    monkeypatch.setattr(ref, "_worker", huge_frame)
    monkeypatch.setattr(ref.NistLiteReferee, "TRANSPORT_STALL_TIMEOUT_S", 60.0)
    result, rows = _play(
        tmp_path, monkeypatch, f"oversized-frame-{extended}", max_steps=0
    )
    _assert_failure_rows(rows, "worker_failure", [0, 2, 4, 6])


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
                elif msg[0] == "act":
                    conn.send(True)
                elif msg[0] == "summary":
                    conn.send(([{"episode": i} for i in indices], {}))
                    return
                # fail_all is ordered before summary and has no acknowledgement.
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
    ("peg_first_seat_step", 4),
    ("done_at", -2),
    ("done_at", -1.0),
    ("done_at", np.array([-1])),
])
def test_out_of_range_summary_timestamp_replaces_only_that_episode(
    tmp_path, monkeypatch, field, bad_value
):
    original = ref._worker

    def mutate_timestamp(rows, traj):
        row = rows[0]
        assert row["episode"] == 0
        if field == "done_at":
            row["step_results"][0][field] = bad_value
        elif field == "peg_first_seat_step":
            row[field] = [bad_value] * row["n_pegs"]
        else:
            row[field] = bad_value

    # Install this fixture's horizon before building the shared real-worker
    # helper. Its fork inherits the same three-step limit as the parent.
    original_sample = ref._sample

    def short_sample(seed, per_stratum, split):
        ranges, conds = original_sample(seed, per_stratum, split)
        for cond in conds:
            cond.max_steps = 3
        return ranges, conds

    monkeypatch.setattr(ref, "_sample", short_sample)
    monkeypatch.setattr(ref, "_worker", _summary_mutation_worker(mutate_timestamp))
    result, rows = _play(tmp_path, monkeypatch, f"invalid-{field}")
    _assert_failure_rows(rows, "worker_failure", [0])
    assert all(row["steps"] == 3 for row in rows if row["episode"] != 0)

    assert result.metadata["details"]["terminal_reasons"]["worker_failure"] == 1


@pytest.mark.parametrize("mutation", ["done", "missing_event", "other_peg"])
def test_done_at_sentinel_requires_real_misseat_row(tmp_path, monkeypatch, mutation):
    """The timestamp alone must not turn a forged row into a valid summary."""

    def invalidate_sentinel(rows, traj):
        row = rows[0]
        target = next(k for k, step in enumerate(row["task"]["steps"]) if step["kind"] in ("insert", "rework"))
        row["step_results"][target]["done_at"] = -1
        row["step_results"][target]["done"] = mutation == "done"
        if mutation == "done":
            row["events"] = [{"kind": "misseat", "step": 1, "peg": row["task"]["steps"][target]["peg"]}]
            row["events_fired"] = 1
        elif mutation == "other_peg":
            other = next(
                step for step in row["task"]["steps"]
                if step["kind"] in ("insert", "rework") and step["peg"] != row["task"]["steps"][target]["peg"]
            )
            row["events"] = [{"kind": "misseat", "step": 1, "peg": other["peg"]}]
            row["events_fired"] = 1

    monkeypatch.setattr(ref, "_worker", _summary_mutation_worker(invalidate_sentinel))
    result, rows = _play(tmp_path, monkeypatch, f"invalid-misseat-{mutation}", max_steps=3)
    _assert_failure_rows(rows, "worker_failure", [0])
    assert all(row["steps"] == 3 for row in rows if row["episode"] % 2)
    assert result.metadata["details"]["terminal_reasons"]["worker_failure"] == 1


def test_misseat_done_at_sentinel_preserves_normal_score(monkeypatch):
    """The real event leaves a legitimate sentinel and keeps independent credit."""
    from nist_sim.compound_v08 import (
        AssemblyIVEpisode,
        compound_score_v08,
        step_results_v08,
    )
    from nist_sim.env import MAX_OPEN, PEG_LEN, _yaw_quat

    _, conds = ref._sample(7, 1, "live")
    conds[0].max_steps = 3
    ep = AssemblyIVEpisode(conds[0], np.random.default_rng(9))
    ep.step(np.zeros(5))
    target_k = next(k for k, s in enumerate(ep.cs.steps) if s.kind in ("insert", "rework"))
    survivor = next(
        other for other, step in enumerate(ep.cs.steps)
        if other != target_k and step.kind in ("insert", "rework")
        and not ep.peg_parked[step.peg]
        and all(ep.cs.steps[p].kind == "insert" for p in step.prereq)
    )

    def seat(k):
        i = ep.cs.steps[k].peg
        hx, hy = ep.slot_xy[i]
        ep._set_free(
            ep.peg_qadr[i], ep.peg_vadr[i],
            [hx, hy, PEG_LEN / 2], _yaw_quat(float(ep.slot_yaw[i])),
        )
        ep.seated_now[i] = True
        ep.step_done[k] = True
        ep.step_in_order[k] = all(ep.step_done[p] for p in ep.cs.steps[k].prereq)
        ep.step_done_at[k] = ep.steps
        ep.data.qpos[2] = 0.16
        ep.grip = MAX_OPEN
        ep._apply_ctrl()

    for prereq in ep.cs.steps[survivor].prereq:
        seat(prereq)
    seat(survivor)
    assert ep._physical_step_ok(survivor)[0]
    for other, step in enumerate(ep.cs.steps):
        if step.kind in ("insert", "rework"):
            ep.seated_now[step.peg] = other == survivor
    seat(target_k)
    original_rng = ep.rng
    original_integers = original_rng.integers

    class ChooseTarget:
        def integers(self, low, high=None):
            # _fire_event chooses from the physically seated candidate list.
            # Keep its real selection logic and its one draw, but choose the
            # intended candidate rather than the independent survivor.
            assert low == 0
            assert original_integers(0, 1) == 0
            seated = [i for i in range(ep.n_pegs) if ep.seated_now[i] and not ep.peg_parked[i]]
            return seated.index(ep.cs.steps[target_k].peg)

        def __getattr__(self, name):
            return getattr(original_rng, name)

    monkeypatch.setattr(ep, "rng", ChooseTarget())
    ep._fire_event({"kind": "misseat", "lift": 0.007})
    assert ep.step_done_at[target_k] == -1
    assert ep.events_fired[-1]["kind"] == "misseat"
    assert ep.events_fired[-1]["peg"] == ep.cs.steps[target_k].peg
    ep.fail("time_limit")

    results = step_results_v08(ep)
    assert results[target_k]["done_at"] == -1
    assert results[target_k]["done"] is False
    assert results[survivor]["done"] is True
    score = compound_score_v08(ep)
    assert score["n_done"] > 0
    assert score["score"] > 0.0

    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=ref._worker, args=(child, 7, 1, [0], 0, "live"))
    monkeypatch.setattr(ref, "_make_episode", lambda cond, rng: ep)
    original_sample = ref._sample

    def short_sample(seed, per_stratum, split):
        ranges, sampled = original_sample(seed, per_stratum, split)
        for cond in sampled:
            cond.max_steps = 3
        return ranges, sampled

    monkeypatch.setattr(ref, "_sample", short_sample)
    proc.start()
    child.close()
    try:
        assert parent.recv() == ("reset_failures", [])
        parent.send(("fail_all", "time_limit"))
        parent.send(("summary",))
        response = parent.recv()
        assert ref._valid_worker_summary(response, conds, [0])
        rows, _ = response
        assert rows[0]["terminal_reason"] == "time_limit"
        assert rows[0]["steps_done"] > 0
        assert rows[0]["score"] > 0.0
        assert any(event["kind"] == "misseat" for event in rows[0]["events"])
        assert rows[0]["step_results"][target_k]["done_at"] == -1
        assert rows[0]["step_results"][target_k]["done"] is False
        assert rows[0]["step_results"][survivor]["done"] is True
    finally:
        parent.close()
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=2)


def test_invalid_summary_row_preserves_positive_sibling_scores(tmp_path, monkeypatch):
    original = ref._worker

    def invalid_one_positive_sibling(conn, seed, per_stratum, indices, stride, split):
        if indices[0] != 0:
            return original(conn, seed, per_stratum, indices, stride, split)
        conn.send(("reset_failures", []))
        try:
            while True:
                msg = conn.recv()
                if msg[0] == "obs":
                    conn.send((indices, np.zeros((len(indices), ref.OBS_W), np.float32)))
                elif msg[0] == "summary":
                    _, conds = ref._sample(seed, per_stratum, split)
                    rows = []
                    for i in indices:
                        row = ref._failed_episode_row(conds[i], i, "time_limit")
                        row["steps"] = 3
                        row["mean_sq_action"] = 0.0
                        row["terminal_reason"] = "time_limit"
                        del row["failure_reason"]
                        if i != 0:
                            row["score"] = 0.125
                            row["steps_done"] = 1
                            row["steps_in_order"] = 1
                            row["step_results"][0]["done"] = True
                            row["step_results"][0]["in_order"] = True
                            row["step_results"][0]["done_at"] = 1
                            row["step_results"][0]["credit"] = 0.9
                            row["credit_by_kind"][row["task"]["steps"][0]["kind"]] = 0.9
                        else:
                            row["score"] = float("nan")
                        rows.append(row)
                    conn.send((rows, {}))
                    return
                elif msg[0] == "act":
                    conn.send(True)
        finally:
            conn.close()

    monkeypatch.setattr(ref, "_worker", invalid_one_positive_sibling)
    result, rows = _play(tmp_path, monkeypatch, "positive-summary-siblings", max_steps=3)
    _assert_failure_rows(rows, "worker_failure", [0])
    assert [r["score"] for r in rows if r["episode"] in (2, 4, 6)] == [0.125] * 3
    assert all(r["steps_done"] == 1 for r in rows if r["episode"] in (2, 4, 6))
    assert result.raw_scores == [pytest.approx(3 * 0.125 / 8)]
    assert result.metadata["details"]["terminal_reasons"]["worker_failure"] == 1


def _summary_mutation_worker(mutation, announced_reset_failures=()):
    """Run the real worker in a non-daemon test-process helper and mutate its summary."""
    original = ref._worker
    original_sample = ref._sample

    def short_sample(seed, per_stratum, split):
        ranges, conds = original_sample(seed, per_stratum, split)
        for cond in conds:
            cond.max_steps = 3
        return ranges, conds

    # Fork in the test process, but only after installing the same short horizon
    # that _play installs in the referee. Otherwise this helper's valid siblings
    # and the parent's conditions describe different episodes.
    ref._sample = short_sample
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe()
    indices = list(range(0, 8, 2))
    proc = ctx.Process(target=original, args=(child, 7, 1, indices, 0, "live"))
    try:
        proc.start()
    finally:
        ref._sample = original_sample
        child.close()

    def reap_helper():
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=2)

    atexit.register(reap_helper)
    try:
        startup = parent.recv()
        assert startup[0] == "reset_failures"
        startup_message = ("reset_failures", sorted(set(startup[1]) | set(announced_reset_failures)))
        parent.send(startup_message)

        def mutate_summary(conn, seed, per_stratum, actual_indices, stride, split):
            if actual_indices[0] != 0:
                return original(conn, seed, per_stratum, actual_indices, stride, split)
            conn.send(startup_message)
            try:
                while True:
                    msg = conn.recv()
                    if msg[0] == "summary":
                        parent.send(msg)
                        rows, traj = parent.recv()
                        mutation(rows, traj)
                        conn.send((rows, traj))
                        return
                    if msg[0] == "obs":
                        conn.send((actual_indices, np.zeros((len(actual_indices), ref.OBS_W), np.float32)))
                    else:
                        # Preserve real action stepping and fail_all bookkeeping so
                        # valid siblings have exactly the referee's expected horizon.
                        parent.send(msg)
                        if msg[0] != "fail_all":
                            conn.send(parent.recv())
            finally:
                conn.close()

        def finish():
            reap_helper()
            parent.close()
            atexit.unregister(reap_helper)

        mutate_summary.reap_test_helper = finish
        return mutate_summary
    except BaseException:
        parent.close()
        reap_helper()
        atexit.unregister(reap_helper)
        raise


def test_oversized_summary_number_becomes_worker_failure(tmp_path, monkeypatch):
    def oversized_score(rows, traj):
        rows[0]["score"] = 10**100

    monkeypatch.setattr(ref, "_worker", _summary_mutation_worker(oversized_score))
    result, rows = _play(
        tmp_path, monkeypatch, "oversized-summary-number", max_steps=3, stride=0
    )
    _assert_failure_rows(rows, "worker_failure", [0])
    assert all(row["steps"] == 3 for row in rows if row["episode"] != 0)
    assert result.raw_scores == [0.0]


def test_non_json_summary_payload_becomes_worker_failure(tmp_path, monkeypatch):
    def numpy_task_step(rows, traj):
        rows[0]["task"]["steps"][0]["injected"] = np.float32(0.0)

    monkeypatch.setattr(ref, "_worker", _summary_mutation_worker(numpy_task_step))
    result, rows = _play(
        tmp_path, monkeypatch, "non-json-summary-payload", max_steps=3, stride=0
    )
    _assert_failure_rows(rows, "worker_failure", [0])
    assert all(row["steps"] == 3 for row in rows if row["episode"] != 0)
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
    _assert_failure_rows(rows, "worker_failure", [0])
    assert all(row["steps"] == 3 for row in rows if row["episode"] != 0)
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
    _assert_failure_rows(rows, "worker_failure", [0])
    assert all(row["steps"] == 3 for row in rows if row["episode"] != 0)
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
