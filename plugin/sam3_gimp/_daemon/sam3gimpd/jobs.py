"""Job queue, request ids, supersession and progress for the sam3gimpd daemon.

This module owns everything in ``API.md`` sections 10 and 11 and nothing else:
it knows about *jobs*, not about HTTP and not about segmentation.  It imports
no torch, no numpy and no transformers -- ever -- and is fully exercised by
``tests/daemon/test_jobs.py`` with plain callables in place of engines.

The shape of the thing
----------------------

* One :class:`JobManager` owns one **worker thread**.  ``API.md`` §10: *the
  daemon runs exactly one inference at a time*.  HTTP concurrency lives in
  ``server.py``; here everything is strictly serial.
* A job is a callable plus bookkeeping.  ``server.py`` supplies
  ``fn(job, progress) -> Optional[JobResult]``; the manager runs it, records
  timings, catches failures into the error envelope, and wakes long-pollers.
* **Supersession** (§10): when a *supersedable* job is submitted for image *I*,
  every job for *I* that is queued-but-not-started becomes ``superseded``.  The
  encode job created by ``POST /images`` is submitted with
  ``supersedable=False`` and is therefore never displaced -- prompts depend on
  it.
* **Long-poll** (§11): :meth:`JobManager.wait` returns on a terminal state, on a
  material progress change (``>= 0.01`` or a stage change), or on timeout.  A
  timeout is not an error.
* **Lifetime.**  A job's ``fn`` is dropped the moment the job settles -- run,
  superseded or cancelled -- because the closure holds the image and its pixels,
  and a retained job must not keep them alive.  Its ``cleanup`` runs exactly once
  at that moment, outside the lock, whether or not the job ever ran; that is
  where the server lets go of the image the job had pinned.
* **Shutdown** (§6.8): :meth:`JobManager.stop` cancels everything still queued
  and the worker exits after the job it is running, if any.

The engine interface
--------------------

``server.py`` calls the injected engine object; the manager never does.  The
contract, restated here so this module can be read on its own:

.. code-block:: python

    engine.encode_image(image, progress=cb) -> embedding        # any object
    engine.prompt_text(image, prompt, progress=cb)   -> EngineResult
    engine.prompt_points(image, prompt, progress=cb) -> EngineResult

``image`` is a duck-typed record with ``image_id``, ``width``, ``height``,
``pixels`` (raw RGB bytes, §7), ``model_canvas``, ``canvas_from_image`` and
``embedding``.  ``prompt`` is :class:`sam3gimpd.types.TextPrompt` /
:class:`sam3gimpd.types.PointPrompt`.  ``progress`` is optional -- an engine that
declares no ``progress`` parameter is called without it.

The result may be an :class:`EngineResult`, a ``(instances, blobs)`` pair, a
dict, or a plain sequence of per-instance mappings; :meth:`EngineResult.coerce`
normalises all of them, so a loosely written engine degrades into an adapter,
not a crash.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .types import (
    ApiError,
    BBox,
    Engine as EngineKind,
    ErrorCode,
    ErrorInfo,
    JobState,
    JobStatus,
    Limits,
    MaskInstance,
    ResultHeader,
)

__all__ = [
    "PROGRESS_EPSILON",
    "Stage",
    "STAGE_PROGRESS",
    "ProgressFn",
    "EngineResult",
    "JobResult",
    "Job",
    "JobManager",
    "new_job_id",
]

LOG = logging.getLogger("sam3gimpd.jobs")

#: A progress change smaller than this does not wake a long-poller (§11).
PROGRESS_EPSILON = 0.01

#: Default retention for finished jobs: at least this long, and at least
#: :data:`RETAIN_MIN_JOBS` of them (``API.md`` §15).
RETAIN_SECONDS = 120.0
RETAIN_MIN_JOBS = 64
#: Hard ceiling so a very long session cannot grow the table without bound.
RETAIN_MAX_JOBS = 512


class Stage:
    """Free-form but stable stage names (``API.md`` §11)."""

    QUEUED = "queued"
    ENCODING = "encoding"
    PROMPTING = "prompting"
    DECODING = "decoding"
    PACKING = "packing"
    DONE = "done"
    FAILED = "failed"


#: Reference milestones so a progress bar behaves the same across engines.
STAGE_PROGRESS = {
    Stage.QUEUED: 0.0,
    Stage.ENCODING: 0.05,
    Stage.PROMPTING: 0.60,
    Stage.DECODING: 0.80,
    Stage.PACKING: 0.95,
    Stage.DONE: 1.0,
    Stage.FAILED: 1.0,
}

#: ``progress(value, stage=None)`` -- what an engine is handed.
ProgressFn = Callable[..., None]

_JOB_COUNTER_LOCK = threading.Lock()
_JOB_COUNTER = 0


def new_job_id() -> str:
    """``j-000001-a91f``: a monotonic counter plus 16 random bits.

    The counter makes ids sort by creation order in a log; the random suffix
    keeps ids from colliding across daemon restarts, which matters because a
    client may still be polling an id from a daemon that died.
    """
    global _JOB_COUNTER
    with _JOB_COUNTER_LOCK:
        _JOB_COUNTER += 1
        n = _JOB_COUNTER
    return "j-%06d-%04x" % (n % 1000000, secrets.randbelow(0x10000))


# --------------------------------------------------------------------------- #
# engine output
# --------------------------------------------------------------------------- #
@dataclass
class EngineResult:
    """What an engine hands back: instances plus their raw mask bytes.

    ``blobs[i]`` belongs to ``instances[i]`` and is ``mask_width *
    mask_height`` bytes of soft uint8 mask (§8.3).  ``blob_offset`` and
    ``blob_length`` on the instances are recomputed by
    :func:`sam3gimpd.types.pack_result`, so an engine may leave them at zero.
    """

    instances: List[MaskInstance] = field(default_factory=list)
    blobs: List[bytes] = field(default_factory=list)
    truncated: bool = False

    def validate(self) -> None:
        if len(self.instances) != len(self.blobs):
            raise ValueError("engine returned %d instances but %d blobs"
                             % (len(self.instances), len(self.blobs)))
        for inst, blob in zip(self.instances, self.blobs):
            expected = int(inst.mask_width) * int(inst.mask_height)
            if len(blob) != expected:
                raise ValueError(
                    "instance %s: %d mask bytes, expected %d (%dx%d)"
                    % (inst.instance_id, len(blob), expected,
                       inst.mask_width, inst.mask_height))

    # -- normalisation ------------------------------------------------------ #
    @classmethod
    def coerce(cls, obj: Any) -> "EngineResult":
        """Accept anything reasonable an engine might return.

        Supported forms, in the order they are tried:

        ``EngineResult``                      returned as-is
        ``(instances, blobs)`` / ``(..., truncated)``
        ``{"instances": [...], "blobs": [...], "truncated": bool}``
        ``[{"score":…, "bbox":[…], "mask": b"…", "label": "…"}, …]``
        ``[obj_with_.score/.bbox/.mask, …]``
        ``None``                              an empty result

        A zero-instance result is legal (§8.1) and is what an honest engine
        returns when nothing matched.
        """
        if obj is None:
            return cls()
        if isinstance(obj, EngineResult):
            obj.validate()
            return obj
        if isinstance(obj, dict):
            if "instances" in obj:
                blobs = obj.get("blobs")
                if blobs is None:
                    blobs = obj.get("masks")
                if blobs is None:
                    return cls._from_items(obj["instances"], bool(obj.get("truncated", False)))
                res = cls(
                    instances=[cls._as_instance(i, n) for n, i in enumerate(obj["instances"])],
                    blobs=[bytes(b) for b in blobs],
                    truncated=bool(obj.get("truncated", False)),
                )
                res.validate()
                return res
            raise TypeError("engine returned a dict without an 'instances' key")
        if isinstance(obj, tuple) and 2 <= len(obj) <= 3:
            instances, blobs = obj[0], obj[1]
            truncated = bool(obj[2]) if len(obj) == 3 else False
            res = cls(
                instances=[cls._as_instance(i, n) for n, i in enumerate(instances)],
                blobs=[bytes(b) for b in blobs],
                truncated=truncated,
            )
            res.validate()
            return res
        if isinstance(obj, (list, tuple)):
            return cls._from_items(obj, False)
        raise TypeError("cannot interpret engine result of type %s" % type(obj).__name__)

    @classmethod
    def _from_items(cls, items: Sequence[Any], truncated: bool) -> "EngineResult":
        instances: List[MaskInstance] = []
        blobs: List[bytes] = []
        for n, item in enumerate(items):
            inst, blob = cls._split_item(item, n)
            instances.append(inst)
            blobs.append(blob)
        res = cls(instances=instances, blobs=blobs, truncated=truncated)
        res.validate()
        return res

    @staticmethod
    def _get(item: Any, name: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(name, default)
        return getattr(item, name, default)

    @classmethod
    def _split_item(cls, item: Any, index: int) -> Tuple[MaskInstance, bytes]:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            return cls._as_instance(item[0], index), bytes(item[1])
        mask = cls._get(item, "mask")
        if mask is None:
            mask = cls._get(item, "blob")
        if mask is None:
            raise TypeError("engine instance %d carries no mask bytes" % index)
        return cls._as_instance(item, index), bytes(mask)

    @classmethod
    def _as_instance(cls, item: Any, index: int) -> MaskInstance:
        if isinstance(item, MaskInstance):
            if not item.mask_width or not item.mask_height:
                item.mask_width = item.bbox.width
                item.mask_height = item.bbox.height
            return item
        bbox = cls._get(item, "bbox")
        if bbox is None:
            raise TypeError("engine instance %d has no bbox" % index)
        if not isinstance(bbox, BBox):
            bbox = BBox.from_list(list(bbox))
        width = cls._get(item, "mask_width")
        height = cls._get(item, "mask_height")
        raw_id = cls._get(item, "instance_id", None)
        return MaskInstance(
            instance_id=int(raw_id) if raw_id is not None else index,
            score=float(cls._get(item, "score", 1.0)),
            bbox=bbox,
            mask_width=int(width) if width else bbox.width,
            mask_height=int(height) if height else bbox.height,
            blob_offset=0,
            blob_length=0,
            label=str(cls._get(item, "label", "") or ""),
        )


@dataclass
class JobResult:
    """A finished job's payload: the packed binary frame plus its parsed header.

    Keeping both means ``GET /jobs/{id}?meta=1`` can answer without unpacking
    the frame, and the frame itself is served as one ``sendall``.
    """

    frame: bytes
    header: ResultHeader


# --------------------------------------------------------------------------- #
# a job
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    """One unit of serialised work.  Mutated only under the manager's lock."""

    job_id: str
    engine: str
    image_id: str
    request_id: str = ""
    fn: Optional[Callable[["Job", ProgressFn], Any]] = None
    #: False for the encode job created by ``POST /images``: §10 forbids
    #: superseding it, because every prompt for the image depends on it.
    supersedable: bool = True
    #: Pinned jobs survive retention pruning (the server pins encode jobs for
    #: as long as their image is cached, so ``cached: true`` can keep naming a
    #: real, still-fetchable job id).
    pinned: bool = False
    #: Called once, outside the manager's lock, when the job settles for any
    #: reason -- done, failed, superseded or cancelled.  Cleared after the call.
    cleanup: Optional[Callable[["Job"], None]] = None

    state: str = JobState.QUEUED
    progress: float = 0.0
    stage: str = Stage.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[ErrorInfo] = None
    result: Optional[JobResult] = None
    superseded_by: Optional[str] = None
    seq: int = 0

    @property
    def is_terminal(self) -> bool:
        return JobState.is_terminal(self.state)

    def elapsed_ms(self, now: Optional[float] = None) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else (now or time.time())
        return max(0.0, (end - self.started_at) * 1000.0)

    def to_status(self, queue_position: Optional[int] = None,
                  include_result: bool = False,
                  now: Optional[float] = None) -> JobStatus:
        return JobStatus(
            job_id=self.job_id,
            state=self.state,
            engine=self.engine,
            image_id=self.image_id,
            request_id=self.request_id,
            progress=float(self.progress),
            stage=self.stage,
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            elapsed_ms=self.elapsed_ms(now),
            queue_position=queue_position if self.state == JobState.QUEUED else None,
            superseded_by=self.superseded_by,
            masks_available=bool(self.state == JobState.DONE and self.result is not None),
            error=self.error,
            result=(self.result.header
                    if include_result and self.result is not None else None),
        )


# --------------------------------------------------------------------------- #
# the manager
# --------------------------------------------------------------------------- #
class JobManager:
    """FIFO queue + one worker thread + long-poll notification.

    Thread-safe.  Every mutation happens under ``self._cv``'s lock and ends with
    ``notify_all`` so that pollers blocked in :meth:`wait` wake up.
    """

    def __init__(self,
                 max_queue: int = 16,
                 retain_seconds: float = RETAIN_SECONDS,
                 retain_min_jobs: int = RETAIN_MIN_JOBS,
                 on_finished: Optional[Callable[[Job], None]] = None,
                 name: str = "sam3gimpd-worker") -> None:
        self.max_queue = int(max_queue)
        self.retain_seconds = float(retain_seconds)
        self.retain_min_jobs = int(retain_min_jobs)
        self._on_finished = on_finished
        self._cv = threading.Condition(threading.RLock())
        self._jobs: Dict[str, Job] = {}
        self._queue: List[Job] = []
        self._running: Optional[Job] = None
        self._stopping = False
        self._seq = 0
        self._total = 0
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._started = False

    # -- lifecycle ---------------------------------------------------------- #
    def start(self) -> "JobManager":
        with self._cv:
            if not self._started:
                self._started = True
                self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> bool:
        """Cancel every queued job, then let the worker finish and exit.

        Queued jobs go to ``cancelled`` with a ``shutting_down`` error at once
        -- none of them may start after this, and none may reach an engine
        that is about to be torn down.  A *running* job is never interrupted
        (§10); ``timeout`` bounds how long we wait for it before the caller
        gives up on it.  Safe to call again.  Returns True when the worker has
        exited (or never started).
        """
        with self._cv:
            self._stopping = True
            error = ErrorInfo(ErrorCode.SHUTTING_DOWN,
                              "the daemon shut down before this job started")
            settled = [self._settle_queued_locked(job, JobState.CANCELLED, error=error)
                       for job in list(self._queue)]
            self._cv.notify_all()
        if settled:
            LOG.info("cancelled %d queued job(s) at shutdown", len(settled))
        self._run_cleanups(settled)
        if self._started and self._thread.is_alive():
            self._thread.join(max(0.0, float(timeout)))
            return not self._thread.is_alive()
        return True

    @property
    def stopping(self) -> bool:
        with self._cv:
            return self._stopping

    # -- submission --------------------------------------------------------- #
    def check_capacity(self) -> None:
        """Raise what :meth:`submit` would raise right now, without submitting.

        Lets a caller refuse early, before doing anything a refused submit
        would have to undo.
        """
        with self._cv:
            self._check_accepting_locked()

    def _check_accepting_locked(self) -> None:
        if self._stopping:
            raise ApiError(ErrorCode.SHUTTING_DOWN,
                           "the daemon is shutting down and takes no new work")
        if len(self._queue) >= self.max_queue:
            raise ApiError(ErrorCode.QUEUE_FULL,
                           "job queue is full (%d pending)" % len(self._queue),
                           {"queue_depth": len(self._queue), "limit": self.max_queue})

    def submit(self,
               engine: str,
               image_id: str,
               fn: Callable[[Job, ProgressFn], Any],
               request_id: str = "",
               supersedable: bool = True,
               pinned: bool = False,
               job_id: Optional[str] = None,
               cleanup: Optional[Callable[[Job], None]] = None) -> Tuple[Job, List[str]]:
        """Enqueue a job, superseding this image's queued-not-started work.

        Returns ``(job, superseded_job_ids)``.  Raises ``ApiError(queue_full)``
        when the queue is at capacity and ``ApiError(shutting_down)`` once
        :meth:`stop` has been called; ``cleanup`` is not called in that case.
        """
        with self._cv:
            self._check_accepting_locked()
            self._seq += 1
            self._total += 1
            job = Job(
                job_id=job_id or new_job_id(),
                engine=engine,
                image_id=image_id,
                request_id=request_id,
                fn=fn,
                supersedable=supersedable,
                pinned=pinned,
                cleanup=cleanup,
                seq=self._seq,
            )
            settled: List[Tuple[Job, Optional[Callable[[Job], None]]]] = []
            if supersedable:
                # §10: queued-but-not-started jobs for the SAME image, whatever
                # their engine.  Running jobs are left alone; jobs for other
                # images are untouched.
                for other in list(self._queue):
                    if other.image_id == image_id and other.supersedable:
                        settled.append(self._settle_queued_locked(
                            other, JobState.SUPERSEDED, superseded_by=job.job_id))
            self._jobs[job.job_id] = job
            self._queue.append(job)
            self._prune_locked()
            self._cv.notify_all()
        superseded = [other.job_id for other, _ in settled]
        LOG.debug("job %s queued (engine=%s image=%s request=%s) superseding %s",
                  job.job_id, engine, image_id, request_id, superseded)
        self._run_cleanups(settled)
        return job, superseded

    def cancel_queued_for_image(self, image_id: str) -> List[str]:
        """``DELETE /images/{id}``: queued jobs go to ``cancelled``.

        A running job is *not* interrupted -- it finishes and its result stays
        fetchable (§6.6).
        """
        with self._cv:
            settled = [self._settle_queued_locked(other, JobState.CANCELLED)
                       for other in list(self._queue) if other.image_id == image_id]
            if settled:
                self._cv.notify_all()
        self._run_cleanups(settled)
        return [job.job_id for job, _ in settled]

    def _settle_queued_locked(self, job: Job, state: str,
                              superseded_by: Optional[str] = None,
                              error: Optional[ErrorInfo] = None
                              ) -> Tuple[Job, Optional[Callable[[Job], None]]]:
        """Take a job that never started out of the queue, terminally.

        Returns the job with its cleanup, which the caller runs once the lock
        is released.
        """
        self._queue.remove(job)
        job.state = state
        job.superseded_by = superseded_by
        job.error = error
        job.finished_at = time.time()
        job.fn = None
        cleanup, job.cleanup = job.cleanup, None
        return job, cleanup

    @staticmethod
    def _run_cleanups(settled: Sequence[Tuple[Job, Optional[Callable[[Job], None]]]]) -> None:
        for job, cleanup in settled:
            if cleanup is None:
                continue
            try:
                cleanup(job)
            except Exception:  # noqa: BLE001 -- a cleanup must never break the queue
                LOG.exception("cleanup for job %s raised", job.job_id)

    # -- inspection --------------------------------------------------------- #
    def get(self, job_id: str) -> Optional[Job]:
        with self._cv:
            return self._jobs.get(job_id)

    def status(self, job_id: str, include_result: bool = False) -> Optional[JobStatus]:
        with self._cv:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return job.to_status(self._queue_position_locked(job), include_result)

    def _queue_position_locked(self, job: Job) -> Optional[int]:
        if job.state != JobState.QUEUED:
            return None
        try:
            return self._queue.index(job)
        except ValueError:
            return None

    def counts(self) -> Dict[str, int]:
        with self._cv:
            return {
                "queued": len(self._queue),
                "running": 1 if self._running is not None else 0,
                "total": self._total,
                "retained": len(self._jobs),
            }

    def running_job(self) -> Optional[Job]:
        with self._cv:
            return self._running

    # -- long poll ---------------------------------------------------------- #
    def wait(self, job_id: str, timeout: float = 0.0,
             include_result: bool = False) -> Optional[JobStatus]:
        """``GET /jobs/{id}?wait=N``, exactly as §11 defines it.

        Returns when the job is terminal, when progress moved by
        :data:`PROGRESS_EPSILON` or the stage changed, or when ``timeout``
        expires -- a timeout is a normal ``200``, not an error.
        """
        timeout = max(0.0, min(float(timeout), Limits.MAX_LONG_POLL_SECONDS))
        deadline = time.monotonic() + timeout
        with self._cv:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            base_progress = job.progress
            base_stage = job.stage
            base_state = job.state
            while True:
                if job.is_terminal:
                    break
                if job.state != base_state:
                    break
                if job.stage != base_stage:
                    break
                if job.progress - base_progress >= PROGRESS_EPSILON:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cv.wait(remaining)
            return job.to_status(self._queue_position_locked(job), include_result)

    # -- progress ----------------------------------------------------------- #
    def _make_progress_fn(self, job: Job) -> ProgressFn:
        def progress(value: Optional[float] = None, stage: Optional[str] = None) -> None:
            with self._cv:
                if job.state != JobState.RUNNING:
                    return
                changed = False
                if stage is not None and stage != job.stage:
                    job.stage = str(stage)
                    changed = True
                    if value is None:
                        floor = STAGE_PROGRESS.get(job.stage)
                        if floor is not None and floor > job.progress:
                            job.progress = floor
                if value is not None:
                    v = max(0.0, min(1.0, float(value)))
                    # Monotonic within a job (§11); a lower value is ignored
                    # rather than rejected, so a sloppy engine cannot make a
                    # progress bar jump backwards.
                    if v > job.progress:
                        job.progress = v
                        changed = True
                if changed:
                    self._cv.notify_all()
        return progress

    # -- worker ------------------------------------------------------------- #
    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._stopping:
                    self._cv.wait()
                if self._stopping:
                    # stop() already cancelled whatever was queued; nothing
                    # starts after it, even if the job just finished raced it.
                    return
                job = self._queue.pop(0)
                job.state = JobState.RUNNING
                job.started_at = time.time()
                job.stage = (Stage.ENCODING if job.engine == EngineKind.ENCODE
                             else Stage.PROMPTING)
                job.progress = max(job.progress, STAGE_PROGRESS.get(job.stage, 0.05))
                self._running = job
                self._cv.notify_all()
            self._execute(job)

    def _execute(self, job: Job) -> None:
        progress = self._make_progress_fn(job)
        result: Optional[JobResult] = None
        error: Optional[ErrorInfo] = None
        try:
            raw = job.fn(job, progress) if job.fn is not None else None
            if raw is not None and not isinstance(raw, JobResult):
                raise TypeError("job function returned %s, expected JobResult or None"
                                % type(raw).__name__)
            result = raw
        except ApiError as exc:
            error = ErrorInfo(exc.code, exc.message, dict(exc.detail))
            LOG.warning("job %s failed: %s: %s", job.job_id, exc.code, exc.message)
        except Exception as exc:  # noqa: BLE001 -- a job must never kill the worker
            trace_id = secrets.token_hex(6)
            LOG.error("job %s crashed [trace %s]\n%s", job.job_id, trace_id,
                      traceback.format_exc())
            error = ErrorInfo(
                ErrorCode.INFERENCE_FAILED,
                "%s: %s" % (type(exc).__name__, exc),
                {"trace_id": trace_id},
            )
        with self._cv:
            job.finished_at = time.time()
            self._running = None
            if error is not None:
                job.state = JobState.FAILED
                job.error = error
                job.stage = Stage.FAILED
            else:
                job.state = JobState.DONE
                job.result = result
                job.stage = Stage.DONE
                job.progress = 1.0
            job.fn = None
            cleanup, job.cleanup = job.cleanup, None
            self._prune_locked()
            self._cv.notify_all()
        self._run_cleanups([(job, cleanup)])
        if self._on_finished is not None:
            try:
                self._on_finished(job)
            except Exception:  # noqa: BLE001
                LOG.exception("on_finished hook raised for job %s", job.job_id)

    # -- retention ---------------------------------------------------------- #
    def _prune_locked(self, now: Optional[float] = None) -> None:
        """Drop old finished jobs: keep >= ``retain_min_jobs`` and >= 120 s."""
        now = now or time.time()
        finished = [j for j in self._jobs.values()
                    if j.is_terminal and not j.pinned]
        if len(finished) <= self.retain_min_jobs:
            return
        finished.sort(key=lambda j: (j.finished_at or j.created_at))
        droppable = len(finished) - self.retain_min_jobs
        for job in finished[:droppable]:
            age = now - (job.finished_at or job.created_at)
            if age >= self.retain_seconds or len(self._jobs) > RETAIN_MAX_JOBS:
                self._jobs.pop(job.job_id, None)

    def unpin(self, job_id: str) -> None:
        with self._cv:
            job = self._jobs.get(job_id)
            if job is not None:
                job.pinned = False
            self._prune_locked()
