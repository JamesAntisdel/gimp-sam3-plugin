"""Tests for the job queue: states, supersession, progress and long-polling.

No HTTP, no engine, no torch -- jobs are plain callables here, which is exactly
how ``server.py`` uses them.  Everything that ``API.md`` §10 and §11 promise is
asserted directly rather than through the wire format.
"""

from __future__ import annotations

import threading
import time

import pytest

from sam3gimpd.jobs import (
    PROGRESS_EPSILON,
    EngineResult,
    Job,
    JobManager,
    JobResult,
    Stage,
    new_job_id,
)
from sam3gimpd.types import (
    ApiError,
    BBox,
    Engine,
    ErrorCode,
    JobState,
    Limits,
    MaskInstance,
    ResultHeader,
    Size,
    CanvasTransform,
    pack_result,
)


@pytest.fixture
def manager():
    """A started manager, stopped on teardown even if a test fails."""
    mgr = JobManager().start()
    try:
        yield mgr
    finally:
        mgr.stop(timeout=2.0)


def _wait_for(predicate, timeout=5.0, interval=0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _noop(job, progress):
    return None


# --------------------------------------------------------------------------- #
# ids
# --------------------------------------------------------------------------- #
def test_job_ids_are_unique_and_shaped_like_the_contract():
    ids = [new_job_id() for _ in range(200)]
    assert len(set(ids)) == 200
    for job_id in ids:
        head, counter, suffix = job_id.split("-")
        assert head == "j"
        assert len(counter) == 6 and counter.isdigit()
        assert len(suffix) == 4 and int(suffix, 16) >= 0


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #
def test_a_job_runs_and_reaches_done(manager):
    ran = threading.Event()

    def fn(job, progress):
        progress(0.5, Stage.DECODING)
        ran.set()
        return None

    job, superseded = manager.submit(Engine.PCS, "img", fn, request_id="r-1")
    assert superseded == []
    assert ran.wait(5.0)
    assert _wait_for(lambda: manager.get(job.job_id).state == JobState.DONE)

    status = manager.status(job.job_id)
    assert status.state == JobState.DONE
    assert status.progress == 1.0
    assert status.stage == Stage.DONE
    assert status.request_id == "r-1"
    assert status.engine == Engine.PCS
    assert status.started_at is not None and status.finished_at is not None
    assert status.elapsed_ms >= 0.0
    assert status.queue_position is None
    assert status.error is None


def test_job_result_is_retained_and_exposed(manager):
    header = ResultHeader(job_id="x", request_id="r", image_id="i", engine=Engine.PCS,
                          image=Size(10, 10), model_canvas=Size(20, 20),
                          canvas_from_image=CanvasTransform(2.0, 2.0))
    frame = pack_result(header, [])
    job, _ = manager.submit(Engine.PCS, "img",
                            lambda j, p: JobResult(frame=frame, header=header))
    assert _wait_for(lambda: manager.get(job.job_id).state == JobState.DONE)
    status = manager.status(job.job_id, include_result=True)
    assert status.masks_available is True
    assert status.result is header
    assert manager.get(job.job_id).result.frame == frame


def test_jobs_run_one_at_a_time_in_fifo_order():
    """API.md section 10: exactly one inference at a time, accepted order."""
    order = []
    overlap = []
    running = threading.Lock()
    mgr = JobManager().start()
    try:
        def fn(tag):
            def run(job, progress):
                if not running.acquire(blocking=False):
                    overlap.append(tag)
                    return None
                try:
                    time.sleep(0.02)
                    order.append(tag)
                finally:
                    running.release()
                return None
            return run

        # Different image ids so nothing is superseded.
        jobs = [mgr.submit(Engine.PCS, "img-%d" % i, fn(i))[0] for i in range(5)]
        assert _wait_for(lambda: all(mgr.get(j.job_id).is_terminal for j in jobs))
        assert order == [0, 1, 2, 3, 4]
        assert overlap == []
    finally:
        mgr.stop(timeout=2.0)


# --------------------------------------------------------------------------- #
# supersession (API.md section 10)
# --------------------------------------------------------------------------- #
def test_queued_jobs_for_the_same_image_are_superseded():
    mgr = JobManager()          # not started: nothing drains the queue
    first, _ = mgr.submit(Engine.PCS, "img", _noop, request_id="r-1")
    second, sup2 = mgr.submit(Engine.PVS, "img", _noop, request_id="r-2")
    assert sup2 == [first.job_id]

    status = mgr.status(first.job_id)
    assert status.state == JobState.SUPERSEDED
    assert status.superseded_by == second.job_id
    # Supersession crosses engines: a PVS prompt displaces a queued PCS one.
    assert mgr.status(second.job_id).state == JobState.QUEUED
    assert mgr.counts()["queued"] == 1


def test_jobs_for_other_images_are_never_superseded():
    mgr = JobManager()
    other, _ = mgr.submit(Engine.PCS, "img-a", _noop)
    _, superseded = mgr.submit(Engine.PCS, "img-b", _noop)
    assert superseded == []
    assert mgr.status(other.job_id).state == JobState.QUEUED
    assert mgr.counts()["queued"] == 2


def test_the_encode_job_is_never_superseded():
    """Prompts depend on it, so section 10 exempts it explicitly."""
    mgr = JobManager()
    encode, _ = mgr.submit(Engine.ENCODE, "img", _noop, supersedable=False)
    prompt, superseded = mgr.submit(Engine.PCS, "img", _noop, request_id="r-1")
    assert superseded == []
    assert mgr.status(encode.job_id).state == JobState.QUEUED

    _, superseded = mgr.submit(Engine.PCS, "img", _noop, request_id="r-2")
    assert superseded == [prompt.job_id]
    assert mgr.status(encode.job_id).state == JobState.QUEUED
    assert mgr.status(prompt.job_id).state == JobState.SUPERSEDED


def test_a_running_job_is_never_superseded():
    started = threading.Event()
    release = threading.Event()
    mgr = JobManager().start()
    try:
        def slow(job, progress):
            started.set()
            release.wait(5.0)
            return None

        running, _ = mgr.submit(Engine.PCS, "img", slow, request_id="r-1")
        assert started.wait(5.0)
        newer, superseded = mgr.submit(Engine.PCS, "img", _noop, request_id="r-2")
        assert superseded == []                       # the running job is untouched
        assert mgr.status(running.job_id).state == JobState.RUNNING
        release.set()
        assert _wait_for(lambda: mgr.get(newer.job_id).state == JobState.DONE)
        assert mgr.status(running.job_id).state == JobState.DONE
    finally:
        release.set()
        mgr.stop(timeout=2.0)


def test_queue_position_counts_from_zero():
    mgr = JobManager()
    a, _ = mgr.submit(Engine.PCS, "img-a", _noop)
    b, _ = mgr.submit(Engine.PCS, "img-b", _noop)
    assert mgr.status(a.job_id).queue_position == 0
    assert mgr.status(b.job_id).queue_position == 1


def test_deleting_an_image_cancels_its_queued_jobs():
    mgr = JobManager()
    doomed, _ = mgr.submit(Engine.PCS, "img", _noop)
    survivor, _ = mgr.submit(Engine.PCS, "other", _noop)
    cancelled = mgr.cancel_queued_for_image("img")
    assert cancelled == [doomed.job_id]
    assert mgr.status(doomed.job_id).state == JobState.CANCELLED
    assert mgr.status(survivor.job_id).state == JobState.QUEUED


def test_queue_full_raises_the_documented_error():
    mgr = JobManager(max_queue=2)
    mgr.submit(Engine.PCS, "a", _noop)
    mgr.submit(Engine.PCS, "b", _noop)
    with pytest.raises(ApiError) as excinfo:
        mgr.submit(Engine.PCS, "c", _noop)
    assert excinfo.value.code == ErrorCode.QUEUE_FULL
    assert excinfo.value.status == 503


def test_submitting_after_stop_is_refused():
    mgr = JobManager().start()
    mgr.stop(timeout=2.0)
    with pytest.raises(ApiError) as excinfo:
        mgr.submit(Engine.PCS, "a", _noop)
    assert excinfo.value.code == ErrorCode.SHUTTING_DOWN


def test_check_capacity_raises_what_submit_would():
    mgr = JobManager(max_queue=1)
    mgr.check_capacity()                          # room: no error
    mgr.submit(Engine.PCS, "a", _noop)
    with pytest.raises(ApiError) as excinfo:
        mgr.check_capacity()
    assert excinfo.value.code == ErrorCode.QUEUE_FULL
    mgr.stop(timeout=0.0)
    with pytest.raises(ApiError) as excinfo:
        mgr.check_capacity()
    assert excinfo.value.code == ErrorCode.SHUTTING_DOWN


# --------------------------------------------------------------------------- #
# shutdown (API.md section 6.8)
# --------------------------------------------------------------------------- #
def test_stop_cancels_queued_jobs_and_none_of_them_ever_runs():
    """Nothing queued may start once shutdown begins -- not least because the
    engine is about to be torn down underneath it."""
    started = threading.Event()
    release = threading.Event()
    ran = []
    cleaned = []
    mgr = JobManager().start()
    try:
        def slow(job, progress):
            started.set()
            release.wait(5.0)
            return None

        def record(tag):
            def fn(job, progress):
                ran.append(tag)
            return fn

        running, _ = mgr.submit(Engine.PCS, "img-0", slow)
        assert started.wait(5.0)
        queued = [mgr.submit(Engine.PCS, "img-%d" % i, record(i),
                             cleanup=lambda j: cleaned.append(j.job_id))[0]
                  for i in range(1, 4)]

        assert mgr.stop(timeout=0.1) is False          # the running job holds it
        for job in queued:
            status = mgr.status(job.job_id)
            assert status.state == JobState.CANCELLED
            assert status.error.code == ErrorCode.SHUTTING_DOWN
            assert mgr.get(job.job_id).fn is None
        assert sorted(cleaned) == sorted(j.job_id for j in queued)
        assert mgr.counts()["queued"] == 0
        assert mgr.status(running.job_id).state == JobState.RUNNING   # not interrupted

        release.set()
        assert _wait_for(lambda: not mgr._thread.is_alive())
        assert mgr.status(running.job_id).state == JobState.DONE
        assert ran == []
    finally:
        release.set()
        mgr.stop(timeout=2.0)


def test_stop_returns_true_once_the_worker_has_exited():
    mgr = JobManager().start()
    assert mgr.stop(timeout=2.0) is True
    assert not mgr._thread.is_alive()
    assert JobManager().stop(timeout=0.0) is True     # never started


def test_the_worker_exits_even_if_work_is_queued_behind_the_running_job():
    """The worker returns once stopping is set, not once the queue is empty."""
    started = threading.Event()
    release = threading.Event()
    ran = []
    mgr = JobManager().start()
    try:
        def slow(job, progress):
            started.set()
            release.wait(5.0)

        mgr.submit(Engine.PCS, "a", slow)
        assert started.wait(5.0)
        with mgr._cv:
            # A job slipping into the queue after stop() drained it must still
            # never start.
            mgr._stopping = True
            late = Job(job_id="j-late", engine=Engine.PCS, image_id="b",
                       fn=lambda j, p: ran.append(1))
            mgr._jobs[late.job_id] = late
            mgr._queue.append(late)
        release.set()
        assert _wait_for(lambda: not mgr._thread.is_alive())
        assert ran == []
    finally:
        release.set()


# --------------------------------------------------------------------------- #
# job lifetime: fn is dropped, cleanup runs exactly once
# --------------------------------------------------------------------------- #
def test_fn_is_dropped_when_a_job_settles(manager):
    """A retained job must not keep its closure -- and the image and pixels the
    closure holds -- alive for the retention window."""
    done, _ = manager.submit(Engine.PCS, "a", _noop)
    failed, _ = manager.submit(Engine.PCS, "b", lambda j, p: 1 / 0)
    assert _wait_for(lambda: manager.get(done.job_id).is_terminal
                     and manager.get(failed.job_id).is_terminal)
    assert manager.get(done.job_id).fn is None
    assert manager.get(failed.job_id).fn is None

    idle = JobManager()                       # not started: jobs stay queued
    first, _ = idle.submit(Engine.PCS, "img", _noop)
    idle.submit(Engine.PCS, "img", _noop)     # supersedes the first
    assert idle.get(first.job_id).state == JobState.SUPERSEDED
    assert idle.get(first.job_id).fn is None
    doomed, _ = idle.submit(Engine.PCS, "other", _noop)
    idle.cancel_queued_for_image("other")
    assert idle.get(doomed.job_id).fn is None


def test_cleanup_runs_once_for_every_way_a_job_can_settle():
    calls = []

    def cleanup(job):
        calls.append((job.job_id, job.state))

    mgr = JobManager()
    superseded, _ = mgr.submit(Engine.PCS, "img", _noop, cleanup=cleanup)
    survivor, _ = mgr.submit(Engine.PCS, "img", _noop, cleanup=cleanup)
    cancelled, _ = mgr.submit(Engine.PCS, "gone", _noop, cleanup=cleanup)
    mgr.cancel_queued_for_image("gone")
    assert calls == [(superseded.job_id, JobState.SUPERSEDED),
                     (cancelled.job_id, JobState.CANCELLED)]

    mgr.start()
    try:
        failing, _ = mgr.submit(Engine.PCS, "bad", lambda j, p: 1 / 0, cleanup=cleanup)
        assert _wait_for(lambda: len(calls) == 4)
        assert (survivor.job_id, JobState.DONE) in calls
        assert (failing.job_id, JobState.FAILED) in calls
        assert mgr.get(survivor.job_id).cleanup is None
    finally:
        mgr.stop(timeout=2.0)
    assert len(calls) == 4                    # nothing ran twice


def test_a_raising_cleanup_does_not_break_the_queue(manager):
    def bad_cleanup(job):
        raise RuntimeError("cleanup blew up")

    first, _ = manager.submit(Engine.PCS, "a", _noop, cleanup=bad_cleanup)
    second, _ = manager.submit(Engine.PCS, "b", _noop)
    assert _wait_for(lambda: manager.get(second.job_id).state == JobState.DONE)
    assert manager.get(first.job_id).state == JobState.DONE


# --------------------------------------------------------------------------- #
# failures
# --------------------------------------------------------------------------- #
def test_an_api_error_inside_a_job_becomes_a_failed_state(manager):
    def boom(job, progress):
        raise ApiError(ErrorCode.MODEL_LOAD_FAILED, "no checkpoint", {"path": "/x"})

    job, _ = manager.submit(Engine.PCS, "img", boom, request_id="r-1")
    assert _wait_for(lambda: manager.get(job.job_id).state == JobState.FAILED)
    status = manager.status(job.job_id)
    assert status.error.code == ErrorCode.MODEL_LOAD_FAILED
    assert status.error.detail == {"path": "/x"}
    assert status.stage == Stage.FAILED
    assert status.masks_available is False


def test_an_unexpected_exception_becomes_inference_failed(manager):
    def boom(job, progress):
        raise ZeroDivisionError("nope")

    job, _ = manager.submit(Engine.PCS, "img", boom)
    assert _wait_for(lambda: manager.get(job.job_id).state == JobState.FAILED)
    error = manager.status(job.job_id).error
    assert error.code == ErrorCode.INFERENCE_FAILED
    assert "ZeroDivisionError" in error.message
    assert "trace_id" in error.detail


def test_a_failing_job_does_not_kill_the_worker(manager):
    manager.submit(Engine.PCS, "a", lambda j, p: 1 / 0)
    job, _ = manager.submit(Engine.PCS, "b", _noop)
    assert _wait_for(lambda: manager.get(job.job_id).state == JobState.DONE)


def test_on_finished_hook_sees_every_terminal_job():
    seen = []
    mgr = JobManager(on_finished=seen.append).start()
    try:
        good, _ = mgr.submit(Engine.PCS, "a", _noop)
        bad, _ = mgr.submit(Engine.PCS, "b", lambda j, p: 1 / 0)
        assert _wait_for(lambda: len(seen) == 2)
        assert {j.job_id for j in seen} == {good.job_id, bad.job_id}
    finally:
        mgr.stop(timeout=2.0)


# --------------------------------------------------------------------------- #
# progress and long-polling (API.md section 11)
# --------------------------------------------------------------------------- #
def test_progress_is_clamped_and_monotonic(manager):
    seen = []
    release = threading.Event()

    def fn(job, progress):
        # A prompt job starts at the floor of its stage band (0.60 for
        # "prompting"), so anything below that is already behind us.
        seen.append(manager.get(job.job_id).progress)
        progress(0.7, Stage.DECODING)
        seen.append(manager.get(job.job_id).progress)
        progress(0.2)                       # backwards: ignored
        seen.append(manager.get(job.job_id).progress)
        progress(9.0)                       # clamped to 1.0
        seen.append(manager.get(job.job_id).progress)
        release.set()
        return None

    job, _ = manager.submit(Engine.PCS, "img", fn)
    assert release.wait(5.0)
    assert seen == [0.60, 0.7, 0.7, 1.0]


def test_a_stage_change_lifts_progress_to_its_floor(manager):
    release = threading.Event()
    observed = {}

    def fn(job, progress):
        progress(stage=Stage.DECODING)
        observed["progress"] = manager.get(job.job_id).progress
        observed["stage"] = manager.get(job.job_id).stage
        release.set()
        return None

    manager.submit(Engine.PCS, "img", fn)
    assert release.wait(5.0)
    assert observed["stage"] == Stage.DECODING
    assert observed["progress"] == pytest.approx(0.80)


def test_wait_returns_immediately_for_a_terminal_job(manager):
    job, _ = manager.submit(Engine.PCS, "img", _noop)
    assert _wait_for(lambda: manager.get(job.job_id).is_terminal)
    started = time.monotonic()
    status = manager.wait(job.job_id, timeout=5.0)
    assert time.monotonic() - started < 0.5
    assert status.state == JobState.DONE


def _poll_across(mgr, job, release, timeout=5.0):
    """Start a long-poll, let the job move, return what the poll saw."""
    result = {}

    def poll():
        result["status"] = mgr.wait(job.job_id, timeout=timeout)

    thread = threading.Thread(target=poll)
    thread.start()
    time.sleep(0.05)
    release.set()
    thread.join(5.0)
    assert not thread.is_alive()
    return result["status"]


def test_wait_returns_on_a_material_progress_change():
    """Progress alone -- same stage, above the stage's floor -- wakes a poller.

    The value has to clear the 0.60 a prompt job starts at, or it is ignored
    as backwards and nothing but the stage could have woken the poll.
    """
    release = threading.Event()
    mgr = JobManager().start()
    try:
        def fn(job, progress):
            release.wait(5.0)
            progress(0.60 + 5 * PROGRESS_EPSILON)      # no stage change
            time.sleep(0.5)
            return None

        job, _ = mgr.submit(Engine.PCS, "img", fn)
        assert _wait_for(lambda: mgr.get(job.job_id).state == JobState.RUNNING)
        status = _poll_across(mgr, job, release)
        # Woken by the progress step, not by the job finishing 0.5 s later.
        assert status.state == JobState.RUNNING
        assert status.stage == Stage.PROMPTING
        assert status.progress == pytest.approx(0.60 + 5 * PROGRESS_EPSILON)
    finally:
        release.set()
        mgr.stop(timeout=2.0)


def test_wait_returns_on_a_stage_change():
    """A stage change wakes a poller even when progress does not move."""
    release = threading.Event()
    mgr = JobManager().start()
    try:
        def fn(job, progress):
            release.wait(5.0)
            progress(0.42, Stage.DECODING)   # below 0.60: the value is ignored
            time.sleep(0.5)
            return None

        job, _ = mgr.submit(Engine.PCS, "img", fn)
        assert _wait_for(lambda: mgr.get(job.job_id).state == JobState.RUNNING)
        status = _poll_across(mgr, job, release)
        assert status.state == JobState.RUNNING
        assert status.stage == Stage.DECODING
        assert status.progress == pytest.approx(0.60)
    finally:
        release.set()
        mgr.stop(timeout=2.0)


def test_wait_times_out_without_being_an_error():
    """A long-poll timeout is a normal status, not a failure (section 11.3)."""
    mgr = JobManager()                       # not started: the job stays queued
    job, _ = mgr.submit(Engine.PCS, "img", _noop)
    started = time.monotonic()
    status = mgr.wait(job.job_id, timeout=0.25)
    elapsed = time.monotonic() - started
    assert 0.2 <= elapsed < 3.0
    assert status.state == JobState.QUEUED
    assert status.progress == 0.0


def test_a_negative_wait_is_clamped_to_zero():
    mgr = JobManager()
    job, _ = mgr.submit(Engine.PCS, "img", _noop)
    started = time.monotonic()
    mgr.wait(job.job_id, timeout=-5.0)       # negative clamps to 0
    assert time.monotonic() - started < 0.5


def test_wait_is_clamped_to_the_documented_maximum(monkeypatch):
    """``?wait=`` is capped at ``Limits.MAX_LONG_POLL_SECONDS`` (§11).

    The ceiling is shrunk so the test does not sit out the real 30 s: a
    10 s request on a job that never moves must come back at the ceiling.
    """
    monkeypatch.setattr(Limits, "MAX_LONG_POLL_SECONDS", 0.2)
    mgr = JobManager()                       # not started: the job stays queued
    job, _ = mgr.submit(Engine.PCS, "img", _noop)
    started = time.monotonic()
    status = mgr.wait(job.job_id, timeout=10.0)
    assert time.monotonic() - started < 2.0
    assert status.state == JobState.QUEUED


def test_wait_ignores_a_sub_epsilon_progress_change():
    """A real progress change smaller than the epsilon must not wake a poller.

    Everything happens above the prompt stage's 0.60 floor, so the nudge does
    move ``progress`` (and notifies) -- only the epsilon rule keeps the poll
    asleep.
    """
    release = threading.Event()
    mgr = JobManager().start()
    try:
        def fn(job, progress):
            progress(0.70, Stage.DECODING)
            release.wait(5.0)
            return None

        job, _ = mgr.submit(Engine.PCS, "img", fn)
        assert _wait_for(lambda: mgr.get(job.job_id).progress >= 0.70)

        def nudge():
            time.sleep(0.05)
            mgr._make_progress_fn(mgr.get(job.job_id))(0.70 + PROGRESS_EPSILON / 4.0)

        threading.Thread(target=nudge).start()
        started = time.monotonic()
        status = mgr.wait(job.job_id, timeout=0.4)
        assert time.monotonic() - started >= 0.35
        assert status.state == JobState.RUNNING
        # The nudge landed; it just was not material.
        assert status.progress == pytest.approx(0.70 + PROGRESS_EPSILON / 4.0)
    finally:
        release.set()
        mgr.stop(timeout=2.0)


def test_wait_for_an_unknown_job_is_none(manager):
    assert manager.wait("j-nope", timeout=0.0) is None
    assert manager.status("j-nope") is None


# --------------------------------------------------------------------------- #
# retention (API.md section 15)
# --------------------------------------------------------------------------- #
def test_old_finished_jobs_are_pruned_but_recent_ones_are_kept():
    mgr = JobManager(retain_seconds=0.0, retain_min_jobs=3)
    for i in range(6):
        job, _ = mgr.submit(Engine.PCS, "img-%d" % i, _noop)
        job.state = JobState.DONE
        job.finished_at = time.time() - (10 - i)   # oldest first, all in the past
        mgr._queue.remove(job)
    mgr._prune_locked()
    assert mgr.counts()["retained"] == 3


def test_pinned_jobs_survive_pruning():
    mgr = JobManager(retain_seconds=0.0, retain_min_jobs=1)
    pinned, _ = mgr.submit(Engine.ENCODE, "img", _noop, pinned=True)
    others = []
    for i in range(5):
        job, _ = mgr.submit(Engine.PCS, "img-%d" % i, _noop)
        others.append(job)
    for job in [pinned] + others:
        job.state = JobState.DONE
        job.finished_at = time.time()
        if job in mgr._queue:
            mgr._queue.remove(job)
    mgr._prune_locked()
    assert mgr.get(pinned.job_id) is not None

    mgr.unpin(pinned.job_id)
    mgr._prune_locked()
    assert mgr.get(pinned.job_id) is None


# --------------------------------------------------------------------------- #
# EngineResult.coerce -- the adapter for loosely typed engines
# --------------------------------------------------------------------------- #
def _instance(width=4, height=3, score=0.5):
    return MaskInstance(instance_id=0, score=score, bbox=BBox(0, 0, width, height),
                        mask_width=width, mask_height=height,
                        blob_offset=0, blob_length=0)


def test_coerce_accepts_none_and_empty():
    assert EngineResult.coerce(None).instances == []
    assert EngineResult.coerce([]).instances == []


def test_coerce_accepts_a_tuple_of_instances_and_blobs():
    result = EngineResult.coerce(([_instance()], [bytes(12)]))
    assert len(result.instances) == 1
    assert result.blobs[0] == bytes(12)
    assert result.truncated is False
    assert EngineResult.coerce(([_instance()], [bytes(12)], True)).truncated is True


def test_coerce_accepts_a_dict():
    result = EngineResult.coerce({"instances": [_instance()], "blobs": [bytes(12)],
                                  "truncated": True})
    assert result.truncated is True
    assert len(result.blobs) == 1


def test_coerce_accepts_plain_mappings_with_masks():
    result = EngineResult.coerce([
        {"score": 0.9, "bbox": [0, 0, 2, 2], "mask": bytes(4), "label": "cat"},
        {"score": 0.4, "bbox": [1, 1, 3, 4], "mask": bytes(6)},
    ])
    assert [i.instance_id for i in result.instances] == [0, 1]
    assert result.instances[0].label == "cat"
    assert result.instances[1].mask_width == 2 and result.instances[1].mask_height == 3
    assert [len(b) for b in result.blobs] == [4, 6]


def test_coerce_rejects_a_blob_of_the_wrong_length():
    with pytest.raises(ValueError):
        EngineResult.coerce(([_instance()], [bytes(11)]))


def test_coerce_rejects_nonsense():
    with pytest.raises(TypeError):
        EngineResult.coerce(42)


# --------------------------------------------------------------------------- #
# Job bookkeeping
# --------------------------------------------------------------------------- #
def test_elapsed_ms_is_zero_before_a_job_starts():
    job = Job(job_id="j-1", engine=Engine.PCS, image_id="i")
    assert job.elapsed_ms() == 0.0
    job.started_at = time.time() - 0.5
    assert job.elapsed_ms() >= 400.0
    job.finished_at = job.started_at + 0.25
    assert job.elapsed_ms() == pytest.approx(250.0, abs=1.0)


def test_status_has_no_queue_position_unless_queued():
    job = Job(job_id="j-1", engine=Engine.PCS, image_id="i", state=JobState.RUNNING)
    assert job.to_status(queue_position=3).queue_position is None
    job.state = JobState.QUEUED
    assert job.to_status(queue_position=3).queue_position == 3
