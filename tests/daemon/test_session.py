"""Tests for ``sam3gimpd.session`` -- the LRU embedding cache.

The cache is small (three entries by default) but it is touched from every
thread in the daemon: HTTP handlers look entries up while the single inference
worker mutates them, and eviction has to hand a GPU allocation back without
ever pulling it out from under a running forward pass.  So the tests here are
about *concurrency and lifetime*, not about clever data structures:

* the id is content-derived, so re-uploading an unchanged image is free;
* eviction is strictly least-recently-used and calls the injected release hook
  exactly once per entry, outside the lock;
* a pinned entry (one a job is using) is never evicted, and deleting it defers
  the release rather than freeing memory mid-inference;
* nothing here imports torch, which is checked in a subprocess.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time

import pytest

from sam3gimpd import masks
from sam3gimpd.session import (
    DEFAULT_CAPACITY,
    IMAGE_ID_HEX_CHARS,
    ImageSession,
    SessionCache,
    SessionState,
    compute_image_id,
)
from sam3gimpd.types import CacheEntry, ErrorInfo, JobState, Size


def pixels(width=16, height=16, seed=0):
    """Deterministic RGB payload in the ``POST /images`` layout."""
    return bytes(((x * 7 + y * 13 + seed * 31 + c * 5) & 0xFF)
                 for y in range(height) for x in range(width) for c in range(3))


class Recorder:
    """A release hook that records what it was handed."""

    def __init__(self, fail=False):
        self.released = []
        self.objects = []          # strong refs, so id() stays unique
        self.fail = fail
        self.lock = threading.Lock()

    def __call__(self, session):
        with self.lock:
            self.released.append(session.image_id)
            self.objects.append(session)
        if self.fail:
            raise RuntimeError("release hook blew up")

    @property
    def ids(self):
        with self.lock:
            return list(self.released)

    @property
    def sessions(self):
        with self.lock:
            return list(self.objects)


# --------------------------------------------------------------------------- #
# the content address
# --------------------------------------------------------------------------- #
def test_image_id_is_32_lowercase_hex_and_deterministic():
    data = pixels()
    a = compute_image_id(data, 16, 16)
    b = compute_image_id(data, 16, 16)
    assert a == b
    assert len(a) == IMAGE_ID_HEX_CHARS == 32
    assert a == a.lower()
    assert all(c in "0123456789abcdef" for c in a)


def test_image_id_depends_on_pixels_and_on_dimensions():
    a = compute_image_id(pixels(16, 16, 0), 16, 16)
    b = compute_image_id(pixels(16, 16, 1), 16, 16)
    assert a != b
    # same byte count, different shape: must not collide
    flat = pixels(1, 256)
    assert compute_image_id(flat, 1, 256) != compute_image_id(flat, 256, 1)


def test_image_id_rejects_degenerate_dimensions():
    with pytest.raises(ValueError):
        compute_image_id(b"", 0, 10)


def test_uploading_identical_pixels_hits_the_cache():
    """API.md §6.2: the same pixels always yield the same id, and the second
    upload reports ``cached``."""
    cache = SessionCache()
    data = pixels()
    first, created_first = cache.get_or_create(data, 16, 16)
    second, created_second = cache.get_or_create(bytes(data), 16, 16)
    assert created_first is True
    assert created_second is False
    assert first is second
    assert len(cache) == 1


# --------------------------------------------------------------------------- #
# geometry recorded on the session
# --------------------------------------------------------------------------- #
def test_session_records_the_geometry_a_result_needs():
    """Masks come back in uploaded-image space, so the canvas is the image.

    This asserted a square 1008x1008 canvas, which described a squash the
    processor does not perform: SAM 3 resizes preserving aspect ratio and pads.
    Asking post_process for target_sizes equal to the uploaded image makes it
    undo both, so the transform is the identity.
    """
    cache = SessionCache()
    session, _ = cache.get_or_create(pixels(64, 48), 64, 48)
    assert session.image == Size(64, 48)
    assert session.model_canvas == Size(64, 48)
    expected = masks.canvas_transform(Size(64, 48), Size(64, 48))
    assert session.canvas_from_image.to_dict() == expected.to_dict()
    # the mapping is exact in both directions
    cx, cy = session.canvas_from_image.image_to_canvas(32.0, 24.0)
    assert session.canvas_from_image.canvas_to_image(cx, cy) == pytest.approx((32.0, 24.0))


@pytest.mark.parametrize("w,h", [(1008, 672), (672, 1008), (800, 600),
                                 (1008, 1008), (1, 1000), (1000, 1)])
def test_the_transform_is_never_anisotropic(w, h):
    """Regression: unequal x/y scales silently distorted every mask.

    A 1008x672 upload was mapped back as though it had been stretched 1.5x
    vertically, so selections landed off the object. On a square image the two
    scales agreed and nothing looked wrong, which is why this survived.
    """
    canvas = masks.model_canvas_size(Size(w, h))
    transform = masks.canvas_transform(Size(w, h), canvas)
    assert transform.scale_x == pytest.approx(transform.scale_y)
    assert transform.scale_x == pytest.approx(1.0)
    assert (transform.offset_x, transform.offset_y) == (0.0, 0.0)


@pytest.mark.parametrize("w,h", [(1008, 672), (640, 480), (333, 999)])
def test_a_corner_maps_to_itself(w, h):
    """The far corner is where an aspect error shows up worst."""
    canvas = masks.model_canvas_size(Size(w, h))
    transform = masks.canvas_transform(Size(w, h), canvas)
    assert transform.image_to_canvas(float(w), float(h)) == pytest.approx((w, h))


def test_caller_supplied_geometry_wins():
    """A future processor that letterboxes must be able to say so."""
    cache = SessionCache()
    canvas = Size(512, 512)
    transform = masks.canvas_transform(Size(64, 48), canvas)
    session, _ = cache.get_or_create(pixels(64, 48), 64, 48,
                                     model_canvas=canvas, canvas_from_image=transform)
    assert session.model_canvas == canvas
    assert session.canvas_from_image.scale_x == pytest.approx(8.0)


def test_keep_pixels_false_drops_the_upload_but_keeps_the_id():
    cache = SessionCache(keep_pixels=False)
    data = pixels()
    session, _ = cache.get_or_create(data, 16, 16)
    assert session.pixels is None
    assert session.image_id == compute_image_id(data, 16, 16)
    assert session.bytes_estimate == 0


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #
def test_session_lifecycle_queued_running_done():
    cache = SessionCache()
    session, _ = cache.get_or_create(pixels(), 16, 16)
    assert session.state == SessionState.QUEUED == JobState.QUEUED
    assert session.is_ready is False

    session.mark_running()
    assert session.state == SessionState.RUNNING

    session.mark_ready(embedding={"fake": "tensor"}, embedding_bytes=4096)
    assert session.is_ready is True
    assert session.embedding == {"fake": "tensor"}
    assert session.bytes_estimate == len(pixels()) + 4096
    assert session.wait_ready(timeout=0.0) == SessionState.DONE


def test_failed_session_records_the_error_and_drops_the_embedding():
    cache = SessionCache()
    session, _ = cache.get_or_create(pixels(), 16, 16)
    session.mark_ready(embedding=object(), embedding_bytes=10)
    session.mark_failed(ErrorInfo(code="model_load_failed", message="no weights"))
    assert session.is_failed is True
    assert session.embedding is None
    assert session.error.code == "model_load_failed"
    assert session.wait_ready(timeout=0.0) == SessionState.FAILED


def test_mark_failed_accepts_exceptions_and_envelopes():
    session = ImageSession(image_id="x", image=Size(16, 16), model_canvas=Size(32, 32),
                           canvas_from_image=masks.canvas_transform(Size(16, 16), Size(32, 32)))
    session.mark_failed(RuntimeError("boom"))
    assert session.error.code == "internal_error"
    assert "boom" in session.error.message

    session.reset()
    session.mark_failed({"error": {"code": "inference_failed", "message": "nan"}})
    assert session.error.code == "inference_failed"


def test_wait_ready_blocks_until_the_encode_settles():
    cache = SessionCache()
    session, _ = cache.get_or_create(pixels(), 16, 16)

    def finish():
        time.sleep(0.05)
        session.mark_ready(embedding=1, embedding_bytes=1)

    t = threading.Thread(target=finish)
    t.start()
    try:
        assert session.wait_ready(timeout=5.0) == SessionState.DONE
    finally:
        t.join()


def test_wait_ready_times_out_without_raising():
    cache = SessionCache()
    session, _ = cache.get_or_create(pixels(), 16, 16)
    start = time.monotonic()
    assert session.wait_ready(timeout=0.05) == SessionState.QUEUED
    assert time.monotonic() - start < 2.0


def test_reuploading_a_failed_image_retries_the_encode():
    """The documented recovery path after ``409 image_not_ready``."""
    cache = SessionCache()
    data = pixels()
    session, created = cache.get_or_create(data, 16, 16)
    assert created is True
    session.mark_failed(ErrorInfo(code="inference_failed", message="nope"))

    again, created_again = cache.get_or_create(data, 16, 16)
    assert again is session
    assert created_again is True          # the caller must enqueue a new encode
    assert again.state == SessionState.QUEUED
    assert again.error is None


def test_to_cache_entry_matches_the_status_shape():
    cache = SessionCache(clock=lambda: 1000.0)
    session, _ = cache.get_or_create(pixels(64, 48), 64, 48)
    session.mark_ready(embedding=None, embedding_bytes=2048)
    entry = session.to_cache_entry()
    assert isinstance(entry, CacheEntry)
    d = entry.to_dict()
    assert d["image_id"] == session.image_id
    assert (d["width"], d["height"]) == (64, 48)
    assert d["state"] == "done"
    assert d["created_at"] == 1000.0 and d["last_used_at"] == 1000.0
    assert d["bytes_estimate"] == 64 * 48 * 3 + 2048
    assert set(d) == {"image_id", "width", "height", "created_at", "last_used_at",
                      "state", "bytes_estimate"}


# --------------------------------------------------------------------------- #
# LRU behaviour
# --------------------------------------------------------------------------- #
def test_default_capacity_is_three():
    assert DEFAULT_CAPACITY == 3
    assert SessionCache().capacity == 3


def test_evicts_least_recently_used_and_releases_it():
    rec = Recorder()
    cache = SessionCache(capacity=2, release=rec)
    a, _ = cache.get_or_create(pixels(seed=1), 16, 16)
    b, _ = cache.get_or_create(pixels(seed=2), 16, 16)
    a.mark_ready(embedding="A")
    b.mark_ready(embedding="B")

    cache.get(a.image_id)                       # a is now the most recent
    c, _ = cache.get_or_create(pixels(seed=3), 16, 16)

    assert rec.ids == [b.image_id]              # b was the least recently used
    assert cache.get(b.image_id) is None
    assert cache.get(a.image_id) is a
    assert cache.get(c.image_id) is c
    assert b.embedding is None                  # the reference is dropped
    assert cache.stats()["evictions"] == 1


def test_get_or_create_touches_and_reorders():
    cache = SessionCache(capacity=2)
    a, _ = cache.get_or_create(pixels(seed=1), 16, 16)
    b, _ = cache.get_or_create(pixels(seed=2), 16, 16)
    cache.get_or_create(pixels(seed=1), 16, 16)  # re-upload a: makes it recent
    cache.get_or_create(pixels(seed=3), 16, 16)
    assert cache.get(a.image_id, touch=False) is a
    assert cache.get(b.image_id, touch=False) is None


def test_peek_does_not_reorder():
    cache = SessionCache(capacity=2)
    a, _ = cache.get_or_create(pixels(seed=1), 16, 16)
    cache.get_or_create(pixels(seed=2), 16, 16)
    assert cache.peek(a.image_id) is a
    cache.get_or_create(pixels(seed=3), 16, 16)
    assert cache.peek(a.image_id) is None        # peek did not save it


def test_entries_are_most_recently_used_first():
    cache = SessionCache(capacity=3)
    a, _ = cache.get_or_create(pixels(seed=1), 16, 16)
    b, _ = cache.get_or_create(pixels(seed=2), 16, 16)
    c, _ = cache.get_or_create(pixels(seed=3), 16, 16)
    cache.get(a.image_id)
    assert [e.image_id for e in cache.entries()] == [a.image_id, c.image_id, b.image_id]
    assert cache.image_ids()[0] == a.image_id


def test_shrinking_capacity_evicts_immediately():
    rec = Recorder()
    cache = SessionCache(capacity=3, release=rec)
    ids = [cache.get_or_create(pixels(seed=i), 16, 16)[0].image_id for i in range(3)]
    cache.set_capacity(1)
    assert len(cache) == 1
    assert sorted(rec.ids) == sorted(ids[:2])
    with pytest.raises(ValueError):
        cache.set_capacity(0)


def test_delete_removes_and_releases():
    rec = Recorder()
    cache = SessionCache(release=rec)
    session, _ = cache.get_or_create(pixels(), 16, 16)
    assert cache.delete(session.image_id) is True
    assert rec.ids == [session.image_id]
    assert cache.get(session.image_id) is None
    assert cache.delete(session.image_id) is False   # 404 image_not_found
    assert cache.delete("nope") is False


def test_clear_releases_everything_once():
    rec = Recorder()
    cache = SessionCache(capacity=5, release=rec)
    ids = [cache.get_or_create(pixels(seed=i), 16, 16)[0].image_id for i in range(4)]
    cache.clear()
    assert sorted(rec.ids) == sorted(ids)
    assert len(cache) == 0
    cache.clear()
    assert len(rec.ids) == 4                        # not released twice


def test_release_is_called_once_per_residency():
    """An entry's GPU memory is handed back exactly once while it is cached.

    Deleting an already-evicted id releases nothing more (there is nothing left
    to free, and the client sees ``404``); re-inserting the session makes it
    live again, so its next removal does release again.
    """
    rec = Recorder()
    cache = SessionCache(capacity=1, release=rec)
    a, _ = cache.get_or_create(pixels(seed=1), 16, 16)
    cache.get_or_create(pixels(seed=2), 16, 16)      # evicts a
    assert rec.ids == [a.image_id]

    assert cache.delete(a.image_id) is False         # already gone
    assert rec.ids.count(a.image_id) == 1

    cache.put(a)                                     # live again
    assert cache.delete(a.image_id) is True
    assert rec.ids.count(a.image_id) == 2
    assert len(cache) == 0


def test_a_broken_release_hook_does_not_break_the_cache():
    rec = Recorder(fail=True)
    cache = SessionCache(capacity=1, release=rec)
    a, _ = cache.get_or_create(pixels(seed=1), 16, 16)
    b, _ = cache.get_or_create(pixels(seed=2), 16, 16)   # release() raises
    assert rec.ids == [a.image_id]
    assert cache.get(b.image_id) is b
    assert len(cache) == 1


def test_release_runs_outside_the_lock():
    """A hook that synchronises a CUDA stream must not block HTTP handlers."""
    entered = threading.Event()
    proceed = threading.Event()
    other_thread_saw = []

    def hook(session):
        entered.set()
        proceed.wait(5.0)

    cache = SessionCache(capacity=1, release=hook)
    cache.get_or_create(pixels(seed=1), 16, 16)

    def evict():
        cache.get_or_create(pixels(seed=2), 16, 16)

    t = threading.Thread(target=evict)
    t.start()
    try:
        assert entered.wait(5.0), "release hook was never called"
        # the cache must still answer while the hook is running
        reader = threading.Thread(target=lambda: other_thread_saw.append(cache.stats()))
        reader.start()
        reader.join(timeout=5.0)
        assert not reader.is_alive(), "cache was locked while release ran"
        assert other_thread_saw and other_thread_saw[0]["capacity"] == 1
    finally:
        proceed.set()
        t.join(timeout=5.0)


# --------------------------------------------------------------------------- #
# pinning
# --------------------------------------------------------------------------- #
def test_a_pinned_entry_is_never_evicted():
    rec = Recorder()
    cache = SessionCache(capacity=1, release=rec)
    a, _ = cache.get_or_create(pixels(seed=1), 16, 16)
    cache.pin(a.image_id)
    b, _ = cache.get_or_create(pixels(seed=2), 16, 16)

    assert rec.ids == []                       # nothing could be evicted
    assert len(cache) == 2                     # deliberately over capacity
    assert cache.get(a.image_id) is a

    cache.unpin(a.image_id)                    # the last unpin prunes back down
    assert len(cache) == 1
    assert rec.ids == [b.image_id]             # the get() above made b the LRU
    assert cache.peek(a.image_id) is a


def test_the_last_unpin_prunes_but_not_while_pins_remain():
    rec = Recorder()
    cache = SessionCache(capacity=1, release=rec)
    a, _ = cache.get_or_create(pixels(seed=1), 16, 16, pin=True)
    b, _ = cache.get_or_create(pixels(seed=2), 16, 16, pin=True)
    assert len(cache) == 2 and rec.ids == []
    cache.pin(a.image_id)
    cache.unpin(a.image_id)                    # one of two pins: still held
    assert len(cache) == 2 and rec.ids == []
    cache.unpin(a.image_id)
    assert rec.ids == [a.image_id]             # the older, now unpinned, goes
    assert cache.peek(b.image_id) is b
    cache.unpin(b.image_id)
    assert len(cache) == 1


def test_get_or_create_can_pin_what_it_creates():
    """Atomic with the insertion, so no concurrent upload can evict it first."""
    rec = Recorder()
    cache = SessionCache(capacity=1, release=rec)
    a, created = cache.get_or_create(pixels(seed=1), 16, 16, pin=True)
    assert created and cache.is_pinned(a.image_id)
    # A cache hit creates nothing and so pins nothing.
    again, created = cache.get_or_create(pixels(seed=1), 16, 16, pin=True)
    assert again is a and not created
    cache.unpin(a.image_id)
    assert not cache.is_pinned(a.image_id)
    # A failed entry is reset, reported as created -- and pinned for its retry.
    a.mark_failed(ErrorInfo("model_load_failed", "no checkpoint"))
    retry, created = cache.get_or_create(pixels(seed=1), 16, 16, pin=True)
    assert retry is a and created and cache.is_pinned(a.image_id)


def test_get_can_pin_what_it_finds():
    cache = SessionCache(capacity=1)
    a, _ = cache.get_or_create(pixels(seed=1), 16, 16)
    assert cache.get(a.image_id, pin=True) is a
    assert cache.is_pinned(a.image_id)
    assert cache.get("0" * 32, pin=True) is None
    assert not cache.is_pinned("0" * 32)       # nothing found, nothing pinned
    cache.get_or_create(pixels(seed=2), 16, 16)
    assert cache.peek(a.image_id) is a         # the pin kept it


def test_every_deletion_of_a_pinned_id_is_released_eventually():
    """Delete, re-upload and delete again while an older job still pins the id:
    both sessions are released when the pins go, neither is forgotten."""
    rec = Recorder()
    cache = SessionCache(release=rec)
    first, _ = cache.get_or_create(pixels(seed=1), 16, 16, pin=True)
    cache.delete(first.image_id)
    second, created = cache.get_or_create(pixels(seed=1), 16, 16, pin=True)
    assert created and second is not first
    cache.delete(second.image_id)
    assert rec.ids == []
    cache.unpin(first.image_id)
    assert rec.ids == []                       # the id is still pinned once
    cache.unpin(first.image_id)
    assert {id(s) for s in rec.sessions} == {id(first), id(second)}


def test_use_pins_for_the_duration_of_a_job():
    rec = Recorder()
    cache = SessionCache(capacity=1, release=rec)
    a, _ = cache.get_or_create(pixels(seed=1), 16, 16)
    with cache.use(a.image_id) as session:
        assert session is a
        assert cache.is_pinned(a.image_id)
        cache.get_or_create(pixels(seed=2), 16, 16)
        assert rec.ids == []
    assert not cache.is_pinned(a.image_id)


def test_use_yields_none_for_an_unknown_image():
    cache = SessionCache()
    with cache.use("deadbeef") as session:
        assert session is None
    assert not cache.is_pinned("deadbeef")


def test_delete_while_pinned_defers_the_release_but_not_the_removal():
    """API.md §6.6: the entry is gone immediately; the running job is not."""
    rec = Recorder()
    cache = SessionCache(release=rec)
    a, _ = cache.get_or_create(pixels(seed=1), 16, 16)
    a.mark_ready(embedding="weights")

    cache.pin(a.image_id)
    assert cache.delete(a.image_id) is True
    assert cache.get(a.image_id) is None       # a later prompt gets 404
    assert rec.ids == []                       # but the memory is still live
    assert a.embedding == "weights"

    cache.unpin(a.image_id)
    assert rec.ids == [a.image_id]
    assert a.embedding is None


def test_nested_pins_are_counted():
    rec = Recorder()
    cache = SessionCache(capacity=1, release=rec)
    a, _ = cache.get_or_create(pixels(seed=1), 16, 16)
    cache.pin(a.image_id)
    cache.pin(a.image_id)
    cache.unpin(a.image_id)
    assert cache.is_pinned(a.image_id)
    cache.get_or_create(pixels(seed=2), 16, 16)
    assert rec.ids == []
    cache.unpin(a.image_id)
    assert not cache.is_pinned(a.image_id)


def test_evict_lru_skips_pinned_entries():
    cache = SessionCache(capacity=4)
    a, _ = cache.get_or_create(pixels(seed=1), 16, 16)
    b, _ = cache.get_or_create(pixels(seed=2), 16, 16)
    cache.pin(a.image_id)
    victim = cache.evict_lru()
    assert victim is b
    cache.unpin(a.image_id)
    assert cache.evict_lru() is a
    assert cache.evict_lru() is None


# --------------------------------------------------------------------------- #
# bookkeeping
# --------------------------------------------------------------------------- #
def test_stats_track_hits_misses_and_evictions():
    cache = SessionCache(capacity=1)
    a, _ = cache.get_or_create(pixels(seed=1), 16, 16)
    cache.get(a.image_id)
    cache.get("missing")
    cache.get_or_create(pixels(seed=2), 16, 16)
    s = cache.stats()
    assert s["capacity"] == 1 and s["size"] == 1
    assert s["hits"] >= 1 and s["misses"] >= 1
    assert s["evictions"] == 1
    assert s["bytes_estimate"] == 16 * 16 * 3


def test_injected_clock_drives_the_timestamps():
    now = [500.0]
    cache = SessionCache(clock=lambda: now[0])
    session, _ = cache.get_or_create(pixels(), 16, 16)
    assert session.created_at == 500.0 and session.last_used_at == 500.0
    now[0] = 900.0
    cache.touch(session.image_id)
    assert session.last_used_at == 900.0
    assert cache.touch("missing") is False


def test_describe_and_sessions_views():
    cache = SessionCache()
    session, _ = cache.get_or_create(pixels(32, 24), 32, 24)
    assert session.describe()["image_id"] == session.image_id
    assert [s.image_id for s in cache.sessions()] == [session.image_id]


def test_contains_and_len():
    cache = SessionCache()
    session, _ = cache.get_or_create(pixels(), 16, 16)
    assert session.image_id in cache
    assert "nope" not in cache
    assert len(cache) == 1


# --------------------------------------------------------------------------- #
# concurrency
# --------------------------------------------------------------------------- #
def test_cache_survives_concurrent_hammering():
    """Handler threads read while the worker mutates: no exception, no
    double-release, and the capacity bound is respected throughout."""
    rec = Recorder()
    cache = SessionCache(capacity=3, release=rec)
    errors = []
    stop = threading.Event()
    payloads = [pixels(16, 16, seed=i) for i in range(12)]

    def churn(worker):
        try:
            for i in range(150):
                data = payloads[(worker * 7 + i) % len(payloads)]
                session, _ = cache.get_or_create(data, 16, 16)
                cache.get(session.image_id)
                if i % 5 == 0:
                    with cache.use(session.image_id) as s:
                        if s is not None:
                            s.mark_ready(embedding=object(), embedding_bytes=64)
                if i % 11 == 0:
                    cache.delete(session.image_id)
                if i % 17 == 0:
                    cache.entries()
                    cache.stats()
                assert len(cache) <= 8   # capacity 3 + at most a few pins
        except Exception as exc:  # pragma: no cover - the assertion below reports it
            errors.append(exc)
        finally:
            stop.set()

    threads = [threading.Thread(target=churn, args=(w,)) for w in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
        assert not t.is_alive(), "a cache thread deadlocked"

    assert not errors, errors
    assert len(cache) <= cache.capacity
    # Image ids repeat legitimately (the same pixels are re-created after a
    # delete), but no session *object* may be released twice...
    released = rec.sessions
    assert len({id(s) for s in released}) == len(released)
    # ...every released one had its embedding dropped...
    assert all(s.embedding is None for s in released)
    # ...and nothing still in the cache was released out from under it.
    cached = {id(s) for s in cache.sessions()}
    assert not cached & {id(s) for s in released}
    assert not any(cache.is_pinned(s.image_id) for s in cache.sessions())


def test_no_session_is_released_twice_under_contention():
    rec = Recorder()
    cache = SessionCache(capacity=2, release=rec)
    sessions = [cache.get_or_create(pixels(seed=i), 16, 16)[0] for i in range(2)]

    def delete_all():
        for s in sessions:
            cache.delete(s.image_id)

    threads = [threading.Thread(target=delete_all) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive()

    assert sorted(rec.ids) == sorted(s.image_id for s in sessions)


# --------------------------------------------------------------------------- #
# import hygiene
# --------------------------------------------------------------------------- #
def test_importing_the_cache_does_not_import_torch(repo_root):
    """The base install has no torch and ``--stub`` must never load it.

    Checked in a subprocess because another test in this session may already
    have imported something heavy.
    """
    code = (
        "import sys; import sam3gimpd.session, sam3gimpd.masks; "
        "bad = [m for m in ('torch', 'transformers') if m in sys.modules]; "
        "print(','.join(bad))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root / "plugin" / "sam3_gimp" / "_daemon"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", "session/masks pulled in %s" % proc.stdout.strip()
