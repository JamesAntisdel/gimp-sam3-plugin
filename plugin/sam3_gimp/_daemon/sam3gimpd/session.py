"""The embedding cache: one encoded image, and an LRU of them.

Encoding an image through SAM 3's vision backbone costs seconds; prompting
against a cached embedding costs milliseconds (DESIGN.md §1).  That single fact
is why this daemon is persistent at all, and this module is where the saving
lives.  Everything else in ``sam3gimpd`` is plumbing around it.

Design notes worth reading before changing anything here:

**Keyed by content, not by client.**  The id is ``blake2b`` of the *uploaded
pixel bytes* plus their dimensions, so re-uploading the same picture -- a second
dialog session on an unchanged image, two GIMP windows, a retry after an
eviction -- hits the cache instead of paying the encoder again.  Clients treat
the id as opaque (API.md §6.2); they never compute it.

**Thread-safe, because it must be.**  The HTTP server is multi-threaded by
contract (API.md §1) while exactly one worker thread runs inference.  Handler
threads read and touch entries; the worker mutates them.  Every public method
here takes one lock and no method calls out to user code while holding it.

**Release is injected, never imported.**  Evicting an entry must drop GPU
memory, but this module must stay importable with no torch present, so the
owner passes a ``release`` callable.  It is invoked **outside** the lock -- a
release hook that synchronises a CUDA stream would otherwise block every
handler thread -- and exactly once per entry, even when a delete races an
eviction.

**Pinning.**  An image with work pending must not be evicted out from under
it: evicting an image whose encode is still queued would leave that encode to
run for nothing, and evicting one a queued prompt targets would fail the
prompt.  So the server pins an entry when it accepts a job for it -- atomically
with the lookup, via ``get(pin=True)`` / ``get_or_create(pin=True)`` -- and
unpins when the job settles.  Eviction skips pinned entries and, if everything
is pinned, lets the cache run over capacity rather than corrupt a running
inference; the last unpin prunes it back down.

Nothing here imports torch, transformers or numpy.  ``embedding`` is an opaque
object the engine owns; this module only stores it, hands it back, and passes it
to ``release``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import blake2b
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from .types import CacheEntry, CanvasTransform, ErrorInfo, JobState, Size

__all__ = [
    "DEFAULT_CAPACITY",
    "IMAGE_ID_HEX_CHARS",
    "SessionState",
    "compute_image_id",
    "ImageSession",
    "SessionCache",
]

_log = logging.getLogger("sam3gimpd.session")

#: LRU capacity, matching ``--cache-size`` and API.md §15.  Three is enough for
#: the realistic pattern (the image being worked on, the one before it, and one
#: spare) and bounded VRAM matters more than a deeper history.
DEFAULT_CAPACITY = 3

#: ``image_id`` is a 128-bit digest rendered as 32 lowercase hex characters.
IMAGE_ID_HEX_CHARS = 32

#: Domain separator: hashing the raw pixels alone would collide across
#: representations, and prefixing a version string means a future change to the
#: id scheme cannot be mistaken for the old one.
_ID_DOMAIN = b"sam3gimpd-image-v1"


class SessionState:
    """Lifecycle of one cached image, mirroring the encode job's state.

    ``queued`` and ``running`` are the encode pass; ``done`` means the embedding
    is resident and prompts may run; ``failed`` means the encode raised and any
    prompt against this image must answer ``409 image_not_ready`` (API.md §4).
    """

    QUEUED = JobState.QUEUED
    RUNNING = JobState.RUNNING
    DONE = JobState.DONE
    FAILED = JobState.FAILED

    ALL = (QUEUED, RUNNING, DONE, FAILED)
    TERMINAL = (DONE, FAILED)


def compute_image_id(pixels: bytes, width: int, height: int) -> str:
    """Content address for an uploaded image: 32 lowercase hex characters.

    The dimensions are hashed alongside the bytes so that two images with the
    same byte count but different shapes (e.g. 32x48 and 48x32) cannot collide,
    and a domain prefix keeps the digest from being confused with any other
    hash in the project.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive, got %dx%d" % (width, height))
    h = blake2b(digest_size=IMAGE_ID_HEX_CHARS // 2)
    h.update(_ID_DOMAIN)
    h.update(b"%d:%d:" % (int(width), int(height)))
    h.update(pixels)
    return h.hexdigest()


@dataclass
class ImageSession:
    """One cached image: its pixels, its embedding, and the geometry a result
    needs to be mapped back onto the client's image.

    The geometry lives here rather than being recomputed per prompt because it
    is reported twice -- once on ``POST /images``, once in every result header
    (API.md §5) -- and the two must agree exactly.

    ``embedding`` is whatever the engine cached: a tensor, a tuple of tensors, a
    dict of intermediates.  This module never inspects it.
    """

    image_id: str
    image: Size
    model_canvas: Size
    canvas_from_image: CanvasTransform
    pixels: Optional[bytes] = None
    created_at: float = 0.0
    last_used_at: float = 0.0
    state: str = SessionState.QUEUED
    error: Optional[ErrorInfo] = None
    embedding: Any = None
    embedding_bytes: int = 0
    encode_job_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    _settled: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)
    _released: bool = field(default=False, repr=False, compare=False)

    # -- lifecycle ---------------------------------------------------------- #
    def mark_running(self) -> None:
        with self._lock:
            if self.state in SessionState.TERMINAL:
                return
            self.state = SessionState.RUNNING

    def mark_ready(self, embedding: Any = None, embedding_bytes: int = 0) -> None:
        """The encode pass finished.  Wakes every :meth:`wait_ready`."""
        with self._lock:
            self.embedding = embedding
            self.embedding_bytes = int(embedding_bytes)
            self.error = None
            self.state = SessionState.DONE
        self._settled.set()

    def mark_failed(self, error: Any) -> None:
        """The encode pass raised.  ``error`` is an :class:`ErrorInfo`, an
        envelope dict, or any exception (rendered as ``internal_error``)."""
        info = _coerce_error(error)
        with self._lock:
            self.state = SessionState.FAILED
            self.error = info
            self.embedding = None
            self.embedding_bytes = 0
        self._settled.set()

    def reset(self) -> None:
        """Return a failed entry to ``queued`` so an encode can be retried.

        The recovery path a client is told to take after ``409
        image_not_ready`` is to re-``POST /images``; that must actually produce
        a new attempt rather than replaying the old failure forever.
        """
        with self._lock:
            self.state = SessionState.QUEUED
            self.error = None
            self.embedding = None
            self.embedding_bytes = 0
            self._released = False
        self._settled.clear()

    def wait_ready(self, timeout: Optional[float] = None) -> str:
        """Block until the encode settles; return the resulting state.

        For callers *outside* the inference worker (tools, tests).  The worker
        must never call it: only the worker can settle an encode, so it would
        be waiting on itself.  Nor is it called from an HTTP handler on the
        request path: prompts may be issued immediately and *queue* (API.md
        §6.2); the connection does not block.
        """
        self._settled.wait(timeout)
        with self._lock:
            return self.state

    # -- accessors ---------------------------------------------------------- #
    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self.state == SessionState.DONE

    @property
    def is_failed(self) -> bool:
        with self._lock:
            return self.state == SessionState.FAILED

    @property
    def bytes_estimate(self) -> int:
        with self._lock:
            return (len(self.pixels) if self.pixels else 0) + int(self.embedding_bytes)

    def touch(self, now: Optional[float] = None) -> float:
        with self._lock:
            self.last_used_at = time.time() if now is None else float(now)
            return self.last_used_at

    def to_cache_entry(self) -> CacheEntry:
        """The ``/status`` view of this entry."""
        with self._lock:
            return CacheEntry(
                image_id=self.image_id,
                width=self.image.width,
                height=self.image.height,
                created_at=self.created_at,
                last_used_at=self.last_used_at,
                state=self.state,
                bytes_estimate=self.bytes_estimate,
            )

    def describe(self) -> Dict[str, Any]:
        return self.to_cache_entry().to_dict()


def _coerce_error(error: Any) -> ErrorInfo:
    if isinstance(error, ErrorInfo):
        return error
    if isinstance(error, dict):
        if "error" in error and isinstance(error["error"], dict):
            return ErrorInfo.from_envelope(error)
        return ErrorInfo.from_dict(error)
    to_env = getattr(error, "to_envelope", None)
    if callable(to_env):
        try:
            return ErrorInfo.from_envelope(to_env())
        except Exception:  # pragma: no cover - defensive
            pass
    if isinstance(error, BaseException):
        return ErrorInfo(code="internal_error", message=str(error) or type(error).__name__)
    return ErrorInfo(code="internal_error", message=str(error))


class SessionCache:
    """Thread-safe LRU of :class:`ImageSession`.

    ``release(session)`` is called once for every entry that leaves the cache --
    evicted, deleted or cleared -- and is where the owner drops GPU memory
    (``del session.embedding; torch.cuda.empty_cache()``).  It runs outside the
    cache lock and its exceptions are logged and swallowed: a broken release
    hook must not wedge the daemon.

    ``clock`` is injected so tests can advance time deterministically instead of
    sleeping.
    """

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        release: Optional[Callable[[ImageSession], None]] = None,
        clock: Callable[[], float] = time.time,
        keep_pixels: bool = True,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1, got %d" % capacity)
        self._capacity = int(capacity)
        self._release = release
        self._clock = clock
        self._keep_pixels = bool(keep_pixels)
        self._lock = threading.RLock()
        self._entries: "OrderedDict[str, ImageSession]" = OrderedDict()
        self._pins: Dict[str, int] = {}
        # Entries deleted while pinned: removed from the map at once (the id
        # stops resolving immediately, as DELETE promises) but released only
        # when the last job that pinned the id lets go.  A list, because the
        # same pixels can be deleted, re-uploaded and deleted again while an
        # older job still holds the id.
        self._deferred: Dict[str, List[ImageSession]] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    # -- configuration ------------------------------------------------------ #
    @property
    def capacity(self) -> int:
        with self._lock:
            return self._capacity

    def set_capacity(self, capacity: int) -> List[ImageSession]:
        """Change the LRU bound, evicting immediately if it shrank."""
        if capacity < 1:
            raise ValueError("capacity must be at least 1, got %d" % capacity)
        with self._lock:
            self._capacity = int(capacity)
            victims = self._prune_locked()
        return self._release_all(victims)

    # -- lookup ------------------------------------------------------------- #
    def compute_id(self, pixels: bytes, width: int, height: int) -> str:
        return compute_image_id(pixels, width, height)

    def get(self, image_id: str, touch: bool = True,
            pin: bool = False) -> Optional[ImageSession]:
        """Fetch and (by default) mark most-recently-used.

        ``pin=True`` also pins a found entry, under the same lock as the
        lookup, so no eviction can come between the two; the caller owes one
        :meth:`unpin`.  Nothing is pinned when ``None`` is returned.
        """
        with self._lock:
            session = self._entries.get(image_id)
            if session is None:
                self._misses += 1
                return None
            self._hits += 1
            if touch:
                self._entries.move_to_end(image_id)
                session.touch(self._clock())
            if pin:
                self._pin_locked(image_id)
            return session

    def peek(self, image_id: str) -> Optional[ImageSession]:
        """Fetch without disturbing LRU order or the hit/miss counters."""
        with self._lock:
            return self._entries.get(image_id)

    def touch(self, image_id: str) -> bool:
        with self._lock:
            session = self._entries.get(image_id)
            if session is None:
                return False
            self._entries.move_to_end(image_id)
            session.touch(self._clock())
            return True

    def __contains__(self, image_id: object) -> bool:
        with self._lock:
            return image_id in self._entries

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def image_ids(self) -> List[str]:
        """Most-recently-used first."""
        with self._lock:
            return list(reversed(self._entries.keys()))

    def sessions(self) -> List[ImageSession]:
        with self._lock:
            return [self._entries[k] for k in reversed(self._entries.keys())]

    def entries(self) -> List[CacheEntry]:
        """``/status``'s ``images`` array, most-recently-used first."""
        return [s.to_cache_entry() for s in self.sessions()]

    # -- insertion ---------------------------------------------------------- #
    def get_or_create(
        self,
        pixels: bytes,
        width: int,
        height: int,
        model_canvas: Optional[Size] = None,
        canvas_from_image: Optional[CanvasTransform] = None,
        image_id: Optional[str] = None,
        pin: bool = False,
    ) -> Tuple[ImageSession, bool]:
        """Return ``(session, created)`` for these pixels.

        ``created`` is the inverse of the ``cached`` field on ``POST /images``:
        ``True`` means the caller must enqueue an encode job.  A previously
        *failed* entry is reset and reported as created, so re-uploading is the
        documented way to retry a failed encode.

        ``pin=True`` pins the entry when ``created`` is True, before anything
        can evict it -- the pin the encode job holds until it settles.  The
        caller owes one :meth:`unpin` in that case, and none otherwise.

        Geometry defaults to the reference policy (the canvas is the uploaded
        image and the transform the identity; see
        :func:`sam3gimpd.masks.model_canvas_size`); the caller may pass its
        own, which is why the daemon can change processors without touching
        this module.
        """
        # Imported here rather than at module scope: masks.py imports nothing
        # heavy either, but keeping the cache free of a hard dependency on the
        # geometry policy makes both easier to test in isolation.
        from . import masks as _masks

        size = Size(int(width), int(height))
        canvas = model_canvas or _masks.model_canvas_size(size)
        transform = canvas_from_image or _masks.canvas_transform(size, canvas)
        iid = image_id or compute_image_id(pixels, size.width, size.height)
        now = self._clock()

        with self._lock:
            existing = self._entries.get(iid)
            if existing is not None:
                self._entries.move_to_end(iid)
                existing.touch(now)
                if existing.state == SessionState.FAILED:
                    existing.reset()
                    if self._keep_pixels and existing.pixels is None:
                        existing.pixels = bytes(pixels)
                    self._hits += 1
                    if pin:
                        self._pin_locked(iid)
                    return existing, True
                self._hits += 1
                return existing, False

            self._misses += 1
            session = ImageSession(
                image_id=iid,
                image=size,
                model_canvas=canvas,
                canvas_from_image=transform,
                pixels=bytes(pixels) if self._keep_pixels else None,
                created_at=now,
                last_used_at=now,
                state=SessionState.QUEUED,
            )
            self._entries[iid] = session
            if pin:
                self._pin_locked(iid)
            victims = self._prune_locked()

        self._release_all(victims)
        return session, True

    def put(self, session: ImageSession) -> List[ImageSession]:
        """Insert a pre-built session (engines' tests, warm-start paths).

        Re-inserting a session that was previously released makes it live
        again, so its *next* removal releases it again: the "release exactly
        once" guarantee is per residency in the cache, not per object lifetime.
        """
        with session._lock:
            session._released = False
        with self._lock:
            self._entries.pop(session.image_id, None)
            self._entries[session.image_id] = session
            session.touch(self._clock())
            victims = self._prune_locked()
        return self._release_all(victims)

    # -- removal ------------------------------------------------------------ #
    def delete(self, image_id: str) -> bool:
        """Drop one entry now (``DELETE /images/{id}``).  ``True`` if it existed.

        A pinned entry is still removed from the map -- the contract says the
        embedding is freed immediately and a *running* job is not interrupted
        (API.md §6.6) -- but its release is deferred to the unpin so the worker
        keeps a valid embedding until it finishes.
        """
        with self._lock:
            session = self._entries.pop(image_id, None)
            if session is None:
                return False
            if self._pins.get(image_id):
                self._deferred.setdefault(image_id, []).append(session)
                return True
        self._release_all([session])
        return True

    def clear(self) -> List[ImageSession]:
        """Drop everything (shutdown, idle unload, Doctor's "free memory")."""
        with self._lock:
            victims = list(self._entries.values())
            self._entries.clear()
        return self._release_all(victims)

    def evict_lru(self) -> Optional[ImageSession]:
        """Force out the least-recently-used unpinned entry."""
        with self._lock:
            victim = self._pick_victim_locked()
            if victim is None:
                return None
            self._entries.pop(victim.image_id, None)
            self._evictions += 1
        self._release_all([victim])
        return victim

    # -- pinning ------------------------------------------------------------ #
    def pin(self, image_id: str) -> None:
        with self._lock:
            self._pin_locked(image_id)

    def _pin_locked(self, image_id: str) -> None:
        self._pins[image_id] = self._pins.get(image_id, 0) + 1

    def unpin(self, image_id: str) -> None:
        """Drop one pin.  The last one releases anything deleted meanwhile and
        prunes a cache that pins had pushed over capacity."""
        victims: List[ImageSession] = []
        with self._lock:
            n = self._pins.get(image_id, 0)
            if n <= 1:
                self._pins.pop(image_id, None)
                victims.extend(self._deferred.pop(image_id, None) or ())
                victims.extend(self._prune_locked())
            else:
                self._pins[image_id] = n - 1
        self._release_all(victims)

    @contextmanager
    def use(self, image_id: str) -> Iterator[Optional[ImageSession]]:
        """Touch, pin, yield, unpin: hold an entry for the length of a block.

        Yields ``None`` when the image is not cached, so a caller can answer
        ``404 image_not_found`` without a second lookup and without a race
        against a concurrent eviction.
        """
        session = self.get(image_id, pin=True)
        if session is None:
            yield None
            return
        try:
            yield session
        finally:
            self.unpin(image_id)

    def is_pinned(self, image_id: str) -> bool:
        with self._lock:
            return self._pins.get(image_id, 0) > 0

    # -- statistics --------------------------------------------------------- #
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "size": len(self._entries),
                "capacity": self._capacity,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "pinned": sum(1 for v in self._pins.values() if v > 0),
                "bytes_estimate": sum(s.bytes_estimate for s in self._entries.values()),
            }

    # -- internals ---------------------------------------------------------- #
    def _pick_victim_locked(self, exclude_mru: bool = False) -> Optional[ImageSession]:
        """Oldest unpinned entry, or ``None``.

        ``exclude_mru`` protects the most-recently-used entry, which during a
        prune is the one just inserted.  Without it a cache whose older entries
        are all pinned would evict the brand-new image it was asked to hold --
        the exact opposite of least-recently-used.
        """
        keys = list(self._entries.keys())  # oldest first
        if exclude_mru:
            keys = keys[:-1]
        for image_id in keys:
            if not self._pins.get(image_id):
                return self._entries[image_id]
        return None

    def _prune_locked(self) -> List[ImageSession]:
        victims: List[ImageSession] = []
        while len(self._entries) > self._capacity:
            victim = self._pick_victim_locked(exclude_mru=True)
            if victim is None:
                # Every evictable entry has work pending: run over capacity
                # rather than drop an image a queued or running job needs.  The
                # last unpin prunes back down.
                _log.info("session cache over capacity (%d/%d) until pending work settles",
                          len(self._entries), self._capacity)
                break
            self._entries.pop(victim.image_id, None)
            self._evictions += 1
            victims.append(victim)
        return victims

    def _release_all(self, victims: List[ImageSession]) -> List[ImageSession]:
        """Run the release hook once per victim, outside the lock."""
        if not victims:
            return victims
        for session in victims:
            with session._lock:
                if session._released:
                    continue
                session._released = True
            if self._release is None:
                session.embedding = None
                continue
            try:
                self._release(session)
            except Exception:  # pragma: no cover - hook is user code
                _log.exception("release hook failed for image %s", session.image_id)
            finally:
                session.embedding = None
        return victims
