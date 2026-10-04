"""``sam3gimpd`` HTTP client for the GIMP plug-in.

**Zero third-party dependencies.**  This module runs inside GIMP 3's *embedded*
Python, which has no pip and no site-packages we control, so it imports only the
standard library (``http.client``, ``json``, ``struct``, ``threading``, ...).
``gi`` is imported **lazily and optionally**, purely to marshal callbacks onto
the GTK main loop; without it the module still works headlessly, which is what
makes it testable on a machine with no GIMP.

It is a hand-written mirror of ``_daemon/sam3gimpd/types.py`` -- see
``_daemon/API.md`` §16.10.  It must never import ``sam3gimpd``: the daemon lives in a
different interpreter.

What lives here
---------------

``Sam3Client``
    One method per endpoint in ``API.md`` §6, plus the composed helpers
    ``wait_for_job`` / ``run_text`` / ``run_points``.  Connections are
    keep-alive and **checked out per request** from a small pool, so the same
    client object can be shared by the GTK thread and any number of worker
    threads, and a thread that exits leaves no socket behind.

``Sam3Client.hello``
    Sends a fresh nonce and checks the daemon's HMAC over it: a process that
    merely squats on the port recorded in a stale ``runtime.json`` cannot
    answer, so it is never handed an image (:class:`DaemonIdentityError`).

``Dispatcher`` / ``Sam3Client.submit``
    The worker-thread API.  **The GTK main loop must never block on HTTP**
    (``DESIGN.md`` §4).  ``submit(fn, on_done)`` runs ``fn`` on a worker thread
    and marshals ``on_done(result)`` back through ``GLib.idle_add`` when ``gi``
    is importable, or calls it directly when it is not.

``parse_result_frame``
    The §8 binary frame -> :class:`Result` with per-instance soft-mask crops.
    Every documented invariant is checked; a violation is a
    :class:`ProtocolError`, never a silently mis-sliced buffer.  So are the
    sizes: a response body is refused before it is read if it could not be a
    legal frame, and every mask must land on the uploaded image, so a
    misbehaving daemon cannot make GIMP allocate an arbitrarily large layer.

Request ids and supersession (``API.md`` §10)
---------------------------------------------
The client owns a monotonic request-id counter and remembers the **latest**
request id issued per ``image_id``.  Any result whose ``request_id`` is not the
latest is **discarded** -- ``wait_for_job`` returns ``None`` rather than handing
stale masks to the canvas.  That single rule is what makes typing a new prompt
mid-inference feel instant.

Coordinate spaces (``API.md`` §5)
---------------------------------
Everything this client *sends* (points, boxes) is in **uploaded-image** pixels.
Everything it *receives* (instance bboxes) is in **model-canvas** pixels.  The
mapping between them is the daemon-reported ``canvas_from_image`` affine and is
never hardcoded here; :func:`place_instance` implements §9 exactly.
"""

from __future__ import annotations

import hashlib
import hmac
import http.client
import ipaddress
import json
import math
import os
import pathlib
import secrets
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    # constants
    "API_VERSION",
    "RESULT_MAGIC",
    "RESULT_PREFIX_SIZE",
    "RESULT_CONTENT_TYPE",
    "MASK_ENCODING_U8_SOFT",
    "DEFAULT_MASK_THRESHOLD",
    "DEFAULT_SCORE_THRESHOLD",
    "DEFAULT_STALL_TIMEOUT",
    "NONCE_HEADER",
    "NONCE_PROOF_KEY",
    "NONCE_PROOF_PREFIX",
    "MAX_JSON_RESPONSE_BYTES",
    "MAX_RESULT_HEADER_BYTES",
    "MAX_RESULT_FRAME_BYTES",
    "MAX_CANVAS_SIDE",
    "Limits",
    "ErrorCode",
    "JobState",
    "Engine",
    # exceptions
    "Sam3ClientError",
    "TransportError",
    "DaemonUnavailable",
    "DaemonIdentityError",
    "DaemonDied",
    "RequestTimeout",
    "ConnectTimeout",
    "ProtocolError",
    "ApiError",
    "AuthError",
    "VersionMismatch",
    "JobFailed",
    "JobTimeout",
    # value types
    "Size",
    "CanvasTransform",
    "BBox",
    "Point",
    "Placement",
    "Hello",
    "ImageAccepted",
    "JobAccepted",
    "JobStatus",
    "MaskInstance",
    "Result",
    # functions
    "api_major",
    "versions_compatible",
    "nonce_proof",
    "parse_result_frame",
    "place_instance",
    "default_runtime_file",
    "read_runtime_json",
    # client / threading
    "Sam3Client",
    "Dispatcher",
    "Call",
    "glib_idle_add",
]

# --------------------------------------------------------------------------- #
# protocol constants -- mirrored by hand from _daemon/sam3gimpd/types.py
# --------------------------------------------------------------------------- #

#: Version of the HTTP contract this client implements.  Only the MAJOR
#: component participates in the compatibility check (``API.md`` §6.1).
#: 1.1 added the ``/hello`` identity proof below.
API_VERSION = "1.1"

#: ``GET /hello`` request header carrying a fresh random nonce.
NONCE_HEADER = "X-Sam3-Nonce"
#: ``/hello`` response field holding the daemon's answer to that nonce.
NONCE_PROOF_KEY = "nonce_proof"
#: Domain-separation prefix of the proof, so the HMAC cannot be replayed as
#: anything else keyed by the same token.
NONCE_PROOF_PREFIX = b"sam3gimpd-hello:"

RESULT_MAGIC = b"SAM3RES\x00"
RESULT_PREFIX_SIZE = 12  # len(magic) + uint32 header length
RESULT_CONTENT_TYPE = "application/vnd.sam3.result+binary"
JSON_CONTENT_TYPE = "application/json; charset=utf-8"
OCTET_CONTENT_TYPE = "application/octet-stream"

MASK_ENCODING_U8_SOFT = "u8_soft"

#: 128 == sigmoid(0) == the model's own binarisation point (``API.md`` §8.3).
DEFAULT_MASK_THRESHOLD = 128

#: PCS returns every candidate at or above this; the slider filters locally.
#: The floor *requested from the daemon*, not the user-facing filter.
#: post_process_instance_segmentation drops everything below this before
#: the client ever sees it, so it is a hard ceiling on what the score
#: slider can reveal.  At 0.1 the slider's bottom tenth was dead and weak
#: matches were unreachable: a prompt like "guitar" kept only the
#: headstock, and "guitar strap" -- whose instances all score lower --
#: returned nothing at all and looked like a model failure.
DEFAULT_SCORE_THRESHOLD = 0.02

#: A prompt wait gives up only when the job's status (state, stage, progress,
#: queue position) has not moved for this long.  There is no wall-clock limit
#: by default: a CPU-only machine legitimately takes minutes per prompt, and
#: a job that is still advancing is never abandoned just for being slow.
#: Ten minutes, because some waits report nothing until they end: a cold
#: model load, one CPU inference stage, or a place in the queue behind
#: another client's slow job.
DEFAULT_STALL_TIMEOUT = 600.0

_U32 = struct.Struct("<I")


class Limits:
    """Mirror of ``sam3gimpd.types.Limits`` / ``API.md`` §15."""

    MAX_IMAGE_SIDE = 1008
    MIN_IMAGE_SIDE = 16
    MAX_UPLOAD_BYTES = 1008 * 1008 * 3
    MAX_JSON_BYTES = 256 * 1024
    MAX_TEXT_CHARS = 512
    MAX_POINTS = 64
    MAX_BOXES = 16
    MAX_INSTANCES = 256
    MAX_REQUEST_ID_CHARS = 64
    MAX_LONG_POLL_SECONDS = 30.0


#: Largest model canvas side a result may declare.  The reference policy makes
#: the canvas the uploaded image (<= 1008); this leaves room for a processor
#: with a larger working resolution without trusting an arbitrary number.
MAX_CANVAS_SIDE = 4096
#: Largest JSON header a result frame may carry: 256 instances whose labels
#: are 512-character prompts, JSON-escaped, plus the echoed prompt.
MAX_RESULT_HEADER_BYTES = 4 * 1024 * 1024
#: Largest legal result frame (``API.md`` §8, §15): every instance's crop can
#: be at most one full 1008 x 1008 canvas.  About 252 MiB -- a bound, not an
#: expectation; real frames are tens of kilobytes.
MAX_RESULT_FRAME_BYTES = (
    Limits.MAX_INSTANCES * Limits.MAX_IMAGE_SIDE * Limits.MAX_IMAGE_SIDE
    + MAX_RESULT_HEADER_BYTES + RESULT_PREFIX_SIZE
)
#: Largest JSON response body.  ``/status`` is the biggest one the daemon
#: sends and is a few kilobytes.
MAX_JSON_RESPONSE_BYTES = 8 * 1024 * 1024


class JobState:
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SUPERSEDED = "superseded"
    CANCELLED = "cancelled"

    TERMINAL = (DONE, FAILED, SUPERSEDED, CANCELLED)

    @staticmethod
    def is_terminal(state: str) -> bool:
        return state in JobState.TERMINAL


class Engine:
    ENCODE = "encode"
    PCS = "pcs"
    PVS = "pvs"


class ErrorCode:
    """The closed list of machine-readable error codes (``API.md`` §4)."""

    BAD_REQUEST = "bad_request"
    INVALID_JSON = "invalid_json"
    MISSING_HEADER = "missing_header"
    BAD_DIMENSIONS = "bad_dimensions"
    PAYLOAD_SIZE_MISMATCH = "payload_size_mismatch"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN_HOST = "forbidden_host"
    NOT_FOUND = "not_found"
    IMAGE_NOT_FOUND = "image_not_found"
    JOB_NOT_FOUND = "job_not_found"
    IMAGE_NOT_READY = "image_not_ready"
    METHOD_NOT_ALLOWED = "method_not_allowed"
    VERSION_MISMATCH = "version_mismatch"
    QUEUE_FULL = "queue_full"
    ENGINE_UNAVAILABLE = "engine_unavailable"
    MODEL_LOAD_FAILED = "model_load_failed"
    INFERENCE_FAILED = "inference_failed"
    SHUTTING_DOWN = "shutting_down"
    INTERNAL_ERROR = "internal_error"


# --------------------------------------------------------------------------- #
# exceptions
# --------------------------------------------------------------------------- #
class Sam3ClientError(Exception):
    """Base class for everything this module raises deliberately."""


class TransportError(Sam3ClientError):
    """Something went wrong below HTTP: sockets, DNS, connection lifetime."""


class DaemonUnavailable(TransportError):
    """Could not establish a connection at all (typically ECONNREFUSED).

    The UI treats it as "the daemon is not running".  The launcher, which can
    check the daemon's process, decides per subclass: a refusal means "respawn
    me", while :class:`ConnectTimeout` from a process that is verifiably ours
    means "busy, wait", and :class:`DaemonIdentityError` means "not ours".
    """


class DaemonIdentityError(DaemonUnavailable):
    """Something answered ``GET /hello`` but could not prove it holds the token.

    ``API.md`` §6.1: the daemon answers the client's nonce with an HMAC keyed
    by the bearer token.  A process that merely listens on the port a stale
    ``runtime.json`` names -- another user's server, or anything that took the
    port after the daemon died -- cannot, and neither can a daemon that
    predates API 1.1.  Either way it must not be sent an image.

    ``hello`` is the payload it sent, or ``None`` when that was not even a
    well-formed hello.  ``reason`` is :attr:`MISSING` (no proof at all -- what
    a daemon older than API 1.1 sends), :attr:`WRONG` (a proof that does not
    verify: whoever sent it accepted a token it does not hold) or
    :attr:`MALFORMED`.  Only ``MISSING`` from a hello claiming API 1.0 can be
    an older build of our own daemon; the launcher never acts on the others.
    A subclass of :class:`DaemonUnavailable` because to every caller it means
    the same thing: the daemon this client was pointed at is not there.
    """

    MISSING = "missing"
    WRONG = "wrong"
    MALFORMED = "malformed"

    def __init__(self, message: str, hello: Optional["Hello"] = None,
                 reason: str = "missing") -> None:
        self.hello = hello
        self.reason = reason
        DaemonUnavailable.__init__(self, message)

    @property
    def predates_proof(self) -> bool:
        """No proof, from a hello that claims API 1.0 -- the one version that
        had none.  Evidence of an older build of our daemon, not proof of it."""
        return (self.reason == self.MISSING and self.hello is not None
                and str(self.hello.api_version).strip().split(".")[:2] == ["1", "0"])


class DaemonDied(TransportError):
    """The connection was established and then broke mid-exchange.

    Distinguished from :class:`DaemonUnavailable` because it usually means the
    daemon *crashed while working*, so the Doctor panel should show
    ``logs/crash.log`` rather than simply respawning silently.
    """


class ConnectTimeout(DaemonUnavailable):
    """Nothing accepted the connection within the timeout.

    Usually nothing is listening: Windows retries a connection to a closed
    local port for about two seconds instead of refusing it, so there this,
    not a refusal, is what an absent daemon looks like.  But a daemon whose
    accept loop is stalled (a frozen process, a full listen queue) also drops
    connections, so a caller that knows the daemon's process is alive and its
    own -- the launcher -- treats this as busy rather than gone.
    """


class RequestTimeout(TransportError):
    """The daemon accepted the connection but did not answer in time."""


class ProtocolError(Sam3ClientError):
    """The daemon answered, but not in a shape ``API.md`` permits."""


class ApiError(Sam3ClientError):
    """A non-2xx response carrying the ``{"error": {...}}`` envelope (§4).

    ``code`` is the machine-readable :class:`ErrorCode` value and is what
    callers switch on -- e.g. ``exc.code == ErrorCode.IMAGE_NOT_FOUND`` means
    the daemon no longer has the image (evicted, deleted, or a fresh daemon)
    and the fix is to upload it again.  ``status`` is the HTTP status,
    ``message`` the daemon's sentence, ``detail`` its free-form object.
    """

    def __init__(
        self,
        code: str,
        message: str = "",
        detail: Optional[Dict[str, Any]] = None,
        status: Optional[int] = None,
    ) -> None:
        self.code = code
        self.message = message or code
        self.detail = detail if isinstance(detail, dict) else {}
        self.status = status
        Exception.__init__(self, self._render())

    def _render(self) -> str:
        """Include the nested cause, when the daemon sent one.

        Several §4 errors are *consequences*: ``image_not_ready`` means the
        encode pass failed, and the reason it failed -- out of memory, a
        checkpoint that will not load, no torch -- travels in
        ``detail["error"]``.  Printing only the outer message told the user
        "the encode pass for this image failed" and left the actual cause
        sitting unread in the payload.
        """
        head = "%s%s: %s" % (
            self.code,
            "" if self.status is None else " (HTTP %d)" % self.status,
            self.message,
        )
        cause = self.detail.get("error") if isinstance(self.detail, dict) else None
        if isinstance(cause, dict):
            inner = cause.get("message") or cause.get("code")
            if inner:
                head += "\n\nCaused by: %s" % inner
                if cause.get("code") and cause.get("message"):
                    head += " [%s]" % cause["code"]
                extra = cause.get("detail")
                if isinstance(extra, dict):
                    bits = ", ".join(
                        "%s=%s" % (k, v) for k, v in sorted(extra.items())
                        if k not in ("image_id",) and v not in (None, "")
                    )
                    if bits:
                        head += "\n%s" % bits
        return head


class AuthError(ApiError):
    """``401 unauthorized`` -- the bearer token is missing, stale or wrong.

    In practice this means ``runtime.json`` belongs to a *previous* daemon; the
    launcher's answer is to delete it and respawn.
    """


class VersionMismatch(ApiError):
    """The daemon speaks a different API major than this client.

    Raised either locally (from :meth:`Sam3Client.hello`) or from a
    ``400 version_mismatch`` response.
    """

    def __init__(
        self,
        message: str = "",
        client_api: str = API_VERSION,
        server_api: str = "",
        detail: Optional[Dict[str, Any]] = None,
        status: Optional[int] = None,
    ) -> None:
        self.client_api = client_api
        self.server_api = server_api
        ApiError.__init__(
            self,
            ErrorCode.VERSION_MISMATCH,
            message or ("client API %s is incompatible with daemon API %s" % (client_api, server_api)),
            detail,
            status,
        )


class JobFailed(Sam3ClientError):
    """A job reached ``state == "failed"``.

    Note this is **not** an HTTP error: ``GET /jobs/{id}`` answered ``200``
    (``API.md`` §4).  ``code``/``message``/``detail`` come from the job envelope.
    """

    def __init__(self, job_id: str, code: str, message: str = "", detail: Optional[Dict[str, Any]] = None) -> None:
        self.job_id = job_id
        self.code = code
        self.message = message or code
        self.detail = detail if isinstance(detail, dict) else {}
        Exception.__init__(self, "job %s failed: %s: %s" % (job_id, code, self.message))


class JobTimeout(Sam3ClientError):
    """The client gave up waiting for a job that had not finished.

    ``stalled`` is True when the job simply stopped moving -- its status did
    not change for the stall window, which ``waited`` then holds -- and False
    when a caller-imposed overall limit (``waited``) ran out while it was still
    advancing.  The job itself may still finish on the daemon; ``job_id`` lets
    a caller wait for it again.
    """

    def __init__(self, job_id: str, waited: float, last_status: Optional["JobStatus"] = None,
                 stalled: bool = False) -> None:
        self.job_id = job_id
        self.waited = waited
        self.last_status = last_status
        self.stalled = bool(stalled)
        if stalled:
            where = ""
            if last_status is not None:
                where = " (stuck in %s at %d%%)" % (
                    last_status.stage or last_status.state, int(round(100 * last_status.progress)))
            text = "job %s made no progress for %.0fs%s" % (job_id, waited, where)
        else:
            text = "job %s did not finish within %.1fs" % (job_id, waited)
        Exception.__init__(self, text)


# --------------------------------------------------------------------------- #
# value types
# --------------------------------------------------------------------------- #
@dataclass
class Size:
    width: int
    height: int

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Size":
        return cls(int(d["width"]), int(d["height"]))

    def to_dict(self) -> Dict[str, int]:
        return {"width": int(self.width), "height": int(self.height)}

    @property
    def area(self) -> int:
        return int(self.width) * int(self.height)


@dataclass
class CanvasTransform:
    """``canvas = image * scale + offset`` (``API.md`` §5).

    Always taken from the daemon's reported values; never assumed.
    """

    scale_x: float
    scale_y: float
    offset_x: float = 0.0
    offset_y: float = 0.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CanvasTransform":
        return cls(
            float(d["scale_x"]),
            float(d["scale_y"]),
            float(d.get("offset_x", 0.0)),
            float(d.get("offset_y", 0.0)),
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "scale_x": float(self.scale_x),
            "scale_y": float(self.scale_y),
            "offset_x": float(self.offset_x),
            "offset_y": float(self.offset_y),
        }

    def image_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        return (x * self.scale_x + self.offset_x, y * self.scale_y + self.offset_y)

    def canvas_to_image(self, x: float, y: float) -> Tuple[float, float]:
        return ((x - self.offset_x) / self.scale_x, (y - self.offset_y) / self.scale_y)


@dataclass
class BBox:
    """Half-open integer rectangle, **model-canvas** pixels in a result header."""

    x0: int
    y0: int
    x1: int
    y1: int

    @classmethod
    def from_list(cls, v: Sequence[Any]) -> "BBox":
        if len(v) != 4:
            raise ProtocolError("bbox must have 4 elements, got %d" % len(v))
        return cls(int(v[0]), int(v[1]), int(v[2]), int(v[3]))

    def to_list(self) -> List[int]:
        return [int(self.x0), int(self.y0), int(self.x1), int(self.y1)]

    @property
    def width(self) -> int:
        return int(self.x1) - int(self.x0)

    @property
    def height(self) -> int:
        return int(self.y1) - int(self.y0)


@dataclass
class Point:
    """A PVS click in **uploaded-image** pixels.  ``label`` 1=include, 0=exclude."""

    x: float
    y: float
    label: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {"x": float(self.x), "y": float(self.y), "label": int(self.label)}


@dataclass
class Placement:
    """Where an instance's mask belongs on the *original* image (``API.md`` §9).

    ``x, y`` is the top-left corner; ``width, height`` is the size the soft crop
    must be scaled to before thresholding.
    """

    x: int
    y: int
    width: int
    height: int

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


# --------------------------------------------------------------------------- #
# response envelopes
# --------------------------------------------------------------------------- #
@dataclass
class Hello:
    """``GET /hello`` (``API.md`` §6.1)."""

    api_version: str
    sam3d_version: str
    engine_mode: str = ""
    device: str = ""
    dtype: str = ""
    capabilities: List[str] = field(default_factory=list)
    torch_available: bool = False
    weights_available: bool = False
    model_canvas: Optional[Size] = None
    pid: int = 0
    started_at: float = 0.0
    uptime_s: float = 0.0
    limits: Dict[str, Any] = field(default_factory=dict)
    #: Source hash of the running daemon (``API.md`` §6.1 ``build``); "" from
    #: a daemon that predates the field.
    build: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Hello":
        canvas = d.get("model_canvas")
        return cls(
            api_version=str(d.get("api_version", "")),
            sam3d_version=str(d.get("sam3d_version", "")),
            build=str(d.get("build", "") or ""),
            engine_mode=str(d.get("engine_mode", "")),
            device=str(d.get("device", "")),
            dtype=str(d.get("dtype", "")),
            capabilities=[str(c) for c in d.get("capabilities", [])],
            torch_available=bool(d.get("torch_available", False)),
            weights_available=bool(d.get("weights_available", False)),
            model_canvas=Size.from_dict(canvas) if isinstance(canvas, dict) else None,
            pid=int(d.get("pid", 0) or 0),
            started_at=float(d.get("started_at", 0.0) or 0.0),
            uptime_s=float(d.get("uptime_s", 0.0) or 0.0),
            limits=dict(d.get("limits") or {}),
            raw=d,
        )

    def has_capability(self, name: str) -> bool:
        """Feature-check rather than assume (``API.md`` §6.1)."""
        return name in self.capabilities

    @property
    def is_stub(self) -> bool:
        return self.engine_mode == "stub"


@dataclass
class ImageAccepted:
    """``202`` from ``POST /images`` (``API.md`` §6.2)."""

    image_id: str
    job_id: str
    cached: bool
    state: str
    image: Size
    model_canvas: Size
    canvas_from_image: CanvasTransform
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ImageAccepted":
        try:
            return cls(
                image_id=str(d["image_id"]),
                job_id=str(d["job_id"]),
                cached=bool(d.get("cached", False)),
                state=str(d.get("state", JobState.QUEUED)),
                image=Size.from_dict(d["image"]),
                model_canvas=Size.from_dict(d["model_canvas"]),
                canvas_from_image=CanvasTransform.from_dict(d["canvas_from_image"]),
                raw=d,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("malformed POST /images response: %s" % (exc,))


@dataclass
class JobAccepted:
    """``202`` from a prompt endpoint (``API.md`` §6.3 / §6.4)."""

    job_id: str
    request_id: str
    image_id: str
    engine: str
    state: str
    superseded_job_ids: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JobAccepted":
        try:
            return cls(
                job_id=str(d["job_id"]),
                request_id=str(d.get("request_id", "")),
                image_id=str(d.get("image_id", "")),
                engine=str(d.get("engine", "")),
                state=str(d.get("state", JobState.QUEUED)),
                superseded_job_ids=[str(j) for j in d.get("superseded_job_ids", [])],
                raw=d,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("malformed prompt response: %s" % (exc,))


@dataclass
class JobStatus:
    """``JobStatus`` JSON from ``GET /jobs/{id}`` (``API.md`` §6.5)."""

    job_id: str
    state: str
    engine: str = ""
    image_id: str = ""
    request_id: str = ""
    progress: float = 0.0
    stage: str = ""
    created_at: float = 0.0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    elapsed_ms: float = 0.0
    queue_position: Optional[int] = None
    superseded_by: Optional[str] = None
    masks_available: bool = False
    error: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JobStatus":
        if not isinstance(d, dict) or "job_id" not in d or "state" not in d:
            raise ProtocolError("malformed JobStatus: %r" % (d,))

        def _optfloat(key: str) -> Optional[float]:
            v = d.get(key)
            return None if v is None else float(v)

        qp = d.get("queue_position")
        return cls(
            job_id=str(d["job_id"]),
            state=str(d["state"]),
            engine=str(d.get("engine", "")),
            image_id=str(d.get("image_id", "") or ""),
            request_id=str(d.get("request_id", "") or ""),
            progress=float(d.get("progress", 0.0) or 0.0),
            stage=str(d.get("stage", "") or ""),
            created_at=float(d.get("created_at", 0.0) or 0.0),
            started_at=_optfloat("started_at"),
            finished_at=_optfloat("finished_at"),
            elapsed_ms=float(d.get("elapsed_ms", 0.0) or 0.0),
            queue_position=None if qp is None else int(qp),
            superseded_by=d.get("superseded_by"),
            masks_available=bool(d.get("masks_available", False)),
            error=d.get("error"),
            result=d.get("result"),
            raw=d,
        )

    @property
    def is_terminal(self) -> bool:
        return JobState.is_terminal(self.state)

    @property
    def is_done(self) -> bool:
        return self.state == JobState.DONE

    def error_tuple(self) -> Tuple[str, str, Dict[str, Any]]:
        err = self.error if isinstance(self.error, dict) else {}
        detail = err.get("detail")
        return (
            str(err.get("code", ErrorCode.INTERNAL_ERROR)),
            str(err.get("message", "")),
            detail if isinstance(detail, dict) else {},
        )


@dataclass
class MaskInstance:
    """One segmented instance plus its **soft** uint8 mask crop (``API.md`` §8).

    ``mask`` is ``mask_width * mask_height`` bytes, row-major top-to-bottom,
    value ``round(255 * sigmoid(logit))``.  ``128`` is the model's own
    binarisation point and therefore the default client threshold; the crop is
    thresholded locally with a byte comparison and **no round trip**.
    """

    instance_id: int
    score: float
    label: str
    bbox: BBox
    mask_width: int
    mask_height: int
    blob_offset: int
    blob_length: int
    mask: bytes = b""

    @property
    def size(self) -> Size:
        return Size(self.mask_width, self.mask_height)

    def value_at(self, x: int, y: int) -> int:
        """Soft mask value at a coordinate **inside the crop** (0 outside it)."""
        if x < 0 or y < 0 or x >= self.mask_width or y >= self.mask_height:
            return 0
        return self.mask[y * self.mask_width + x]

    def row(self, y: int) -> bytes:
        """One row of the crop, ``mask_width`` bytes."""
        if y < 0 or y >= self.mask_height:
            raise IndexError("row %d out of range" % y)
        start = y * self.mask_width
        return self.mask[start:start + self.mask_width]

    def count_above(self, threshold: int = DEFAULT_MASK_THRESHOLD) -> int:
        """Pixels that a local threshold would call "inside"."""
        t = int(threshold)
        return sum(1 for v in self.mask if v >= t)

    def is_soft(self) -> bool:
        """True when the crop actually contains intermediate values.

        A constant 0/255 mask hides client bugs, so the stub engine promises
        genuine gradients (``API.md`` §14) and the canvas tests assert this.
        """
        return any(0 < v < 255 for v in self.mask)


@dataclass
class Result:
    """A decoded ``application/vnd.sam3.result+binary`` frame (``API.md`` §8)."""

    api_version: str
    job_id: str
    request_id: str
    image_id: str
    engine: str
    state: str
    prompt: Dict[str, Any]
    image: Size
    model_canvas: Size
    canvas_from_image: CanvasTransform
    mask_encoding: str
    elapsed_ms: float
    truncated: bool
    instances: List[MaskInstance]
    header: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.instances)

    def __iter__(self):
        return iter(self.instances)

    def place(self, instance: MaskInstance, source_width: int, source_height: int) -> Placement:
        """Where this instance goes on the original ``source_width x source_height`` image."""
        return place_instance(
            instance,
            self.canvas_from_image,
            uploaded=self.image,
            source=Size(int(source_width), int(source_height)),
        )


# --------------------------------------------------------------------------- #
# version handling
# --------------------------------------------------------------------------- #
def api_major(version: str) -> Optional[int]:
    """Major component of a ``"MAJOR.MINOR"`` string, or ``None`` if unparseable."""
    if not version:
        return None
    head = str(version).split(".", 1)[0].strip()
    try:
        return int(head)
    except ValueError:
        return None


def versions_compatible(client_api: str, server_api: str) -> bool:
    """Majors must be equal; that is the *only* compatibility check (§6.1)."""
    a, b = api_major(client_api), api_major(server_api)
    return a is not None and a == b


def nonce_proof(token: str, nonce: str) -> str:
    """The ``nonce_proof`` a genuine daemon returns from ``GET /hello`` (§6.1).

    ``HMAC-SHA256(key=token, msg=b"sam3gimpd-hello:" + nonce)``, hex.  Only a
    process that knows the bearer token -- the daemon that wrote
    ``runtime.json`` -- can compute it for a nonce it has never seen.
    """
    return hmac.new(str(token).encode("utf-8"), NONCE_PROOF_PREFIX + str(nonce).encode("ascii"),
                    hashlib.sha256).hexdigest()


# --------------------------------------------------------------------------- #
# binary frame decoding (API.md section 8)
# --------------------------------------------------------------------------- #
def parse_result_frame(body: bytes) -> Result:
    """Decode a §8 result frame, validating every documented invariant.

    Layout::

        0        8   magic  b"SAM3RES\\x00"
        8        4   uint32 LE  H = JSON header length
        12       H   JSON header, UTF-8
        12+H     B   blob region, masks tightly packed in instance order

    ``blob_offset`` is relative to the **blob region**, i.e. absolute position
    is ``12 + H + blob_offset``.  Getting that wrong is the single easiest way
    to mis-slice every mask, so it is asserted here rather than trusted.

    The geometry is bounded as well as typed: every bbox must lie inside the
    model canvas (§8.2) and map back onto the uploaded image through the
    reported transform.  Downstream code scales each crop to its placement on
    the original image, so an unchecked bbox or transform from a misbehaving
    daemon would otherwise become an arbitrarily large allocation inside GIMP.
    """
    if not isinstance(body, (bytes, bytearray, memoryview)):
        raise ProtocolError("result frame must be bytes, got %s" % type(body).__name__)
    body = bytes(body)
    if len(body) < RESULT_PREFIX_SIZE:
        raise ProtocolError("result frame shorter than its %d-byte prefix" % RESULT_PREFIX_SIZE)
    if body[:8] != RESULT_MAGIC:
        raise ProtocolError("bad result magic %r (expected %r)" % (body[:8], RESULT_MAGIC))

    (hlen,) = _U32.unpack_from(body, 8)
    if hlen > MAX_RESULT_HEADER_BYTES:
        raise ProtocolError(
            "declared header length %d exceeds the %d-byte limit" % (hlen, MAX_RESULT_HEADER_BYTES)
        )
    blob_start = RESULT_PREFIX_SIZE + hlen
    if blob_start > len(body):
        raise ProtocolError(
            "declared header length %d overruns the %d-byte body" % (hlen, len(body))
        )
    try:
        header = json.loads(body[RESULT_PREFIX_SIZE:blob_start].decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ProtocolError("result header is not valid UTF-8 JSON: %s" % (exc,))
    if not isinstance(header, dict):
        raise ProtocolError("result header must be a JSON object")

    # One view of the blob region; each mask is copied out of it exactly once.
    blob = memoryview(body)[blob_start:]
    declared_blob = header.get("blob_length")
    if declared_blob is not None:
        try:
            declared_blob = int(declared_blob)
        except (TypeError, ValueError):
            raise ProtocolError("header blob_length %r is not an integer" % (declared_blob,))
        if declared_blob != len(blob):
            raise ProtocolError(
                "header blob_length %d != actual blob region %d bytes" % (declared_blob, len(blob))
            )

    try:
        image = Size.from_dict(header["image"])
        canvas = Size.from_dict(header["model_canvas"])
        transform = CanvasTransform.from_dict(header["canvas_from_image"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("result header missing geometry: %s" % (exc,))
    _check_geometry(image, canvas, transform)

    raw_instances = header.get("instances", [])
    if not isinstance(raw_instances, list):
        raise ProtocolError("result header 'instances' must be a list")
    if len(raw_instances) > Limits.MAX_INSTANCES:
        raise ProtocolError(
            "result has %d instances; the limit is %d" % (len(raw_instances), Limits.MAX_INSTANCES)
        )

    instances: List[MaskInstance] = []
    expected_offset = 0
    for raw in raw_instances:
        try:
            bbox = BBox.from_list(raw["bbox"])
            mw = int(raw["mask_width"])
            mh = int(raw["mask_height"])
            off = int(raw["blob_offset"])
            blen = int(raw["blob_length"])
            instance_id = int(raw.get("instance_id", len(instances)))
            score = float(raw.get("score", 0.0))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ProtocolError("malformed instance entry: %s" % (exc,))

        _check_bbox(instance_id, bbox, image, canvas, transform)
        # API.md section 16 non-negotiables 2 and 4.
        if mw != bbox.width or mh != bbox.height:
            raise ProtocolError(
                "instance %s: mask %dx%d != bbox %dx%d"
                % (raw.get("instance_id"), mw, mh, bbox.width, bbox.height)
            )
        if blen != mw * mh:
            raise ProtocolError(
                "instance %s: blob_length %d != mask_width*mask_height %d"
                % (raw.get("instance_id"), blen, mw * mh)
            )
        if off != expected_offset:
            raise ProtocolError(
                "instance %s: blob_offset %d, expected tightly packed %d"
                % (raw.get("instance_id"), off, expected_offset)
            )
        if off + blen > len(blob):
            raise ProtocolError(
                "instance %s: mask [%d, %d) overruns the %d-byte blob region"
                % (raw.get("instance_id"), off, off + blen, len(blob))
            )
        expected_offset = off + blen

        instances.append(
            MaskInstance(
                instance_id=instance_id,
                score=score,
                label=str(raw.get("label", "") or ""),
                bbox=bbox,
                mask_width=mw,
                mask_height=mh,
                blob_offset=off,
                blob_length=blen,
                mask=bytes(blob[off:off + blen]),
            )
        )

    if expected_offset != len(blob):
        raise ProtocolError(
            "blob region has %d trailing bytes after the last instance" % (len(blob) - expected_offset)
        )

    encoding = str(header.get("mask_encoding", MASK_ENCODING_U8_SOFT))
    if encoding != MASK_ENCODING_U8_SOFT:
        raise ProtocolError("unsupported mask_encoding %r" % encoding)

    return Result(
        api_version=str(header.get("api_version", "")),
        job_id=str(header.get("job_id", "")),
        request_id=str(header.get("request_id", "") or ""),
        image_id=str(header.get("image_id", "") or ""),
        engine=str(header.get("engine", "")),
        state=str(header.get("state", JobState.DONE)),
        prompt=dict(header.get("prompt") or {}),
        image=image,
        model_canvas=canvas,
        canvas_from_image=transform,
        mask_encoding=encoding,
        elapsed_ms=float(header.get("elapsed_ms", 0.0) or 0.0),
        truncated=bool(header.get("truncated", False)),
        instances=instances,
        header=header,
    )


def _check_geometry(image: Size, canvas: Size, transform: CanvasTransform) -> None:
    """Sizes in range and an invertible, finite ``canvas_from_image``."""
    if not (1 <= image.width <= Limits.MAX_IMAGE_SIDE and 1 <= image.height <= Limits.MAX_IMAGE_SIDE):
        raise ProtocolError(
            "result image %dx%d is outside [1, %d]" % (image.width, image.height, Limits.MAX_IMAGE_SIDE)
        )
    if not (1 <= canvas.width <= MAX_CANVAS_SIDE and 1 <= canvas.height <= MAX_CANVAS_SIDE):
        raise ProtocolError(
            "result model canvas %dx%d is outside [1, %d]" % (canvas.width, canvas.height, MAX_CANVAS_SIDE)
        )
    values = (transform.scale_x, transform.scale_y, transform.offset_x, transform.offset_y)
    if not all(math.isfinite(v) for v in values) or transform.scale_x <= 0 or transform.scale_y <= 0:
        raise ProtocolError("canvas_from_image %r is not a finite positive scaling" % (transform.to_dict(),))


def _check_bbox(instance_id: int, bbox: BBox, image: Size, canvas: Size,
                transform: CanvasTransform) -> None:
    """``0 <= x0 < x1 <= canvas.width`` (likewise y), and on the uploaded image.

    The second half is what bounds every placement computed from this result:
    mapped back through the reported transform, the box must fall on the
    uploaded image, give or take the pixel or two of padding §8.3 allows and
    one percent for rounding.
    """
    if not (0 <= bbox.x0 < bbox.x1 <= canvas.width and 0 <= bbox.y0 < bbox.y1 <= canvas.height):
        raise ProtocolError(
            "instance %s: bbox %r is not inside the %dx%d model canvas"
            % (instance_id, bbox.to_list(), canvas.width, canvas.height)
        )
    ix0, iy0 = transform.canvas_to_image(float(bbox.x0), float(bbox.y0))
    ix1, iy1 = transform.canvas_to_image(float(bbox.x1), float(bbox.y1))
    tol_x = 4.0 + 0.01 * image.width
    tol_y = 4.0 + 0.01 * image.height
    if ix0 < -tol_x or iy0 < -tol_y or ix1 > image.width + tol_x or iy1 > image.height + tol_y:
        raise ProtocolError(
            "instance %s: bbox %r maps to (%.1f, %.1f)-(%.1f, %.1f), off the %dx%d uploaded image"
            % (instance_id, bbox.to_list(), ix0, iy0, ix1, iy1, image.width, image.height)
        )


def place_instance(
    instance: MaskInstance,
    transform: CanvasTransform,
    uploaded: Size,
    source: Size,
) -> Placement:
    """Model-canvas bbox -> original-image rectangle, exactly as ``API.md`` §9.

    Step 1 inverts the daemon-reported affine (canvas -> uploaded-image);
    step 2 applies the client's own downscale (uploaded -> original);
    step 3 rounds **each edge independently** and only then derives the size, so
    adjacent instances tile without gaps.
    """
    b = instance.bbox
    ix0, iy0 = transform.canvas_to_image(float(b.x0), float(b.y0))
    ix1, iy1 = transform.canvas_to_image(float(b.x1), float(b.y1))

    u = float(source.width) / float(uploaded.width)
    v = float(source.height) / float(uploaded.height)

    x0 = int(round(ix0 * u))
    x1 = int(round(ix1 * u))
    y0 = int(round(iy0 * v))
    y1 = int(round(iy1 * v))
    return Placement(x=x0, y=y0, width=max(1, x1 - x0), height=max(1, y1 - y0))


# --------------------------------------------------------------------------- #
# runtime.json
# --------------------------------------------------------------------------- #
_RUNTIME_REQUIRED = ("port", "token", "pid", "version", "started_at")


def default_runtime_file() -> str:
    """Path of ``runtime.json`` (``API.md`` §3.1).

    Delegates to ``launcher`` -- which owns the full platform layout -- and
    falls back to an environment-only resolution when the client is used
    standalone (e.g. from ``tools/canvas_harness.py``) without the launcher on
    the path.
    """
    try:
        try:
            from . import launcher as _launcher  # type: ignore[attr-defined]
        except ImportError:
            import launcher as _launcher  # type: ignore[no-redef]
        return str(_launcher.runtime_file())
    except Exception:
        pass
    override = os.environ.get("SAM3D_RUNTIME_FILE")
    if override:
        return _spelled_path(override)
    home = os.environ.get("SAM3_GIMP_HOME")
    if home:
        return _spelled_path(home, "runtime.json")
    if os.name == "nt" or sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if not base:
            base = os.path.join(os.path.expanduser("~"), "AppData", "Local")
        return _spelled_path(base, "sam3-gimp", "runtime.json")
    if sys.platform == "darwin":
        return _spelled_path("~", "Library", "Application Support", "sam3-gimp", "runtime.json")
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg and os.path.isabs(xdg):
        return _spelled_path(xdg, "sam3-gimp", "runtime.json")
    return _spelled_path("~", ".local", "share", "sam3-gimp", "runtime.json")


def _spelled_path(first: str, *rest: str) -> str:
    """``first`` (``~`` expanded) joined with ``rest``, spelled the way
    ``pathlib`` spells it -- as ``launcher`` and the daemon do."""
    try:
        path = pathlib.Path(first).expanduser()
    except (RuntimeError, KeyError):  # no such ~user, or no home: kept as written
        path = pathlib.Path(first)
    return str(path.joinpath(*rest))


def read_runtime_json(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Read and *validate* ``runtime.json``; ``None`` means "treat as stale".

    Missing, empty, unparseable, not-an-object or missing any of the five
    required fields all collapse to ``None`` (``API.md`` §3.3 step 1).  Unknown
    keys are preserved and ignored, as the contract requires.
    """
    target = path or default_runtime_file()
    try:
        with open(target, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return None
    if not text.strip():
        return None
    try:
        info = json.loads(text)
    except ValueError:
        return None
    if not isinstance(info, dict):
        return None
    for key in _RUNTIME_REQUIRED:
        if key not in info:
            return None
    try:
        info["port"] = int(info["port"])
        info["pid"] = int(info["pid"])
        info["started_at"] = float(info["started_at"])
        info["token"] = str(info["token"])
        info["version"] = str(info["version"])
    except (TypeError, ValueError):
        return None
    if not (0 < info["port"] < 65536) or not info["token"]:
        return None
    return info


# --------------------------------------------------------------------------- #
# GTK marshalling (gi imported lazily and optionally)
# --------------------------------------------------------------------------- #
_GLIB_SENTINEL = object()
_glib_idle = _GLIB_SENTINEL
_glib_lock = threading.Lock()


def glib_idle_add() -> Optional[Callable[..., Any]]:
    """Return ``GLib.idle_add`` if ``gi`` is importable, else ``None``.

    Imported lazily and cached: this module must stay importable (and testable)
    on a machine with no GIMP and no GTK.
    """
    global _glib_idle
    if _glib_idle is _GLIB_SENTINEL:
        with _glib_lock:
            if _glib_idle is _GLIB_SENTINEL:
                fn = None
                try:
                    import gi  # noqa: PLC0415

                    try:
                        gi.require_version("GLib", "2.0")
                    except (ValueError, AttributeError):
                        pass
                    from gi.repository import GLib  # noqa: PLC0415

                    fn = GLib.idle_add
                except Exception:
                    fn = None
                _glib_idle = fn
    return _glib_idle


def _reset_glib_cache() -> None:
    """Test hook: forget the cached ``gi`` probe."""
    global _glib_idle
    _glib_idle = _GLIB_SENTINEL


class Call:
    """Handle for one :meth:`Dispatcher.submit`.

    Cheaper and more predictable than ``concurrent.futures``: cancellation is
    best-effort (a call that has not started never runs; a call in flight keeps
    running but its callbacks are suppressed), which is exactly the semantic a
    dialog that is being closed needs.
    """

    __slots__ = ("_fn", "_on_done", "_on_error", "_dispatcher", "_event",
                 "_result", "_error", "_state", "_lock")

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"

    def __init__(self, dispatcher: "Dispatcher", fn, on_done=None, on_error=None) -> None:
        self._dispatcher = dispatcher
        self._fn = fn
        self._on_done = on_done
        self._on_error = on_error
        self._event = threading.Event()
        self._result: Any = None
        self._error: Optional[BaseException] = None
        self._state = Call.PENDING
        self._lock = threading.Lock()

    # -- state ------------------------------------------------------------- #
    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def cancelled(self) -> bool:
        return self.state == Call.CANCELLED

    @property
    def done(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> bool:
        """Suppress this call.  Returns ``True`` if it had not started yet."""
        with self._lock:
            if self._state != Call.PENDING:
                fresh = False
            else:
                self._state = Call.CANCELLED
                fresh = True
        if fresh:
            self._event.set()
        else:
            with self._lock:
                self._on_done = None
                self._on_error = None
        return fresh

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._event.wait(timeout)

    def result(self, timeout: Optional[float] = None) -> Any:
        """Block for the outcome; re-raise the worker's exception here."""
        if not self._event.wait(timeout):
            raise JobTimeout("<call>", timeout or 0.0)
        if self._error is not None:
            raise self._error
        return self._result

    # -- worker side ------------------------------------------------------- #
    def _run(self) -> None:
        with self._lock:
            if self._state != Call.PENDING:
                return
            self._state = Call.RUNNING
        try:
            value = self._fn()
        except BaseException as exc:  # noqa: BLE001 -- reported to on_error
            with self._lock:
                self._error = exc
                if self._state != Call.CANCELLED:
                    self._state = Call.ERROR
                cb = self._on_error
            self._event.set()
            if cb is not None and self.state != Call.CANCELLED:
                self._dispatcher.marshal(cb, exc)
            return
        with self._lock:
            self._result = value
            if self._state != Call.CANCELLED:
                self._state = Call.DONE
            cb = self._on_done
        self._event.set()
        if cb is not None and self.state != Call.CANCELLED:
            self._dispatcher.marshal(cb, value)


class Dispatcher:
    """A tiny worker pool whose callbacks land on the GTK main loop.

    ``DESIGN.md`` §4: *the GTK main loop must never block on HTTP*.  Every
    daemon call goes through here; results are marshalled with
    ``GLib.idle_add`` when ``gi`` is present and invoked directly when it is
    not, so the same code is exercised headlessly by the test-suite.

    ``marshal`` is ``"auto"`` (GLib if importable), ``"glib"`` (require it) or
    ``"direct"`` (never touch gi -- what the tests use).
    """

    def __init__(self, workers: int = 1, name: str = "sam3gimpd-client", marshal: str = "auto") -> None:
        if workers < 1:
            raise ValueError("workers must be >= 1")
        if marshal not in ("auto", "glib", "direct"):
            raise ValueError("marshal must be 'auto', 'glib' or 'direct'")
        self._workers = int(workers)
        self._name = name
        self._marshal_mode = marshal
        self._queue: "List[Optional[Call]]" = []
        self._cond = threading.Condition()
        self._threads: List[threading.Thread] = []
        self._shutdown = False

    # -- callback marshalling --------------------------------------------- #
    def marshal(self, callback: Callable[..., Any], *args: Any) -> None:
        """Invoke ``callback(*args)`` on the UI thread (or right here)."""
        if callback is None:
            return
        if self._marshal_mode != "direct":
            idle_add = glib_idle_add()
            if idle_add is None and self._marshal_mode == "glib":
                raise RuntimeError("marshal='glib' but gi/GLib is not importable")
            if idle_add is not None:
                def _once(*_ignored: Any) -> bool:
                    try:
                        callback(*args)
                    finally:
                        pass
                    return False  # GLib: run once, then remove the source

                idle_add(_once)
                return
        callback(*args)

    def wrap(self, callback: Callable[..., Any]) -> Callable[..., None]:
        """Wrap a UI callback so worker threads can call it safely.

        Use for progress callbacks::

            report = dispatcher.wrap(self._on_progress)
            client.submit(lambda: client.run_text(iid, txt, on_progress=report))
        """

        def _wrapped(*args: Any) -> None:
            self.marshal(callback, *args)

        return _wrapped

    # -- submission -------------------------------------------------------- #
    def submit(self, fn: Callable[[], Any], on_done=None, on_error=None) -> Call:
        """Run ``fn()`` on a worker thread; deliver its outcome to the UI thread."""
        if self._shutdown:
            raise RuntimeError("dispatcher has been shut down")
        call = Call(self, fn, on_done=on_done, on_error=on_error)
        with self._cond:
            self._ensure_threads_locked()
            self._queue.append(call)
            self._cond.notify()
        return call

    def _ensure_threads_locked(self) -> None:
        # Threads are created on first use so that merely constructing a client
        # inside a GIMP plug-in costs nothing.
        while len(self._threads) < self._workers:
            t = threading.Thread(
                target=self._worker_loop,
                name="%s-%d" % (self._name, len(self._threads)),
                daemon=True,  # GIMP plug-in processes are short-lived; never block exit
            )
            self._threads.append(t)
            t.start()

    def _worker_loop(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._shutdown:
                    self._cond.wait()
                if self._queue:
                    item = self._queue.pop(0)
                elif self._shutdown:
                    return
                else:
                    continue
            if item is None:
                return
            item._run()

    def shutdown(self, wait: bool = True, timeout: float = 5.0) -> None:
        """Stop the pool.  Pending (unstarted) calls are dropped."""
        with self._cond:
            if self._shutdown:
                return
            self._shutdown = True
            pending = list(self._queue)
            self._queue = []
            self._cond.notify_all()
        for call in pending:
            if call is not None:
                call.cancel()
        if wait:
            deadline = time.monotonic() + timeout
            for t in self._threads:
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    t.join(remaining)
        self._threads = []


# --------------------------------------------------------------------------- #
# the client
# --------------------------------------------------------------------------- #
class _ConnectTimeoutError(OSError):
    """Raised by :class:`_Connection` when ``connect()`` times out; reported
    as :class:`ConnectTimeout`."""


def _abort_socket(sock: Any) -> None:
    """Make a read blocked on ``sock`` in another thread return now.

    ``shutdown()`` does that on Linux and macOS.  Windows leaves a pending read
    waiting after a shutdown, and a plain ``close()`` is deferred while a
    response still reads through ``sock.makefile()``; closing the OS handle
    itself is what aborts the call there.  The socket object is detached first,
    so later calls through it fail cleanly instead of reaching a handle number
    Windows may already have handed to another socket.
    """
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    if os.name != "nt":
        return
    try:
        handle = sock.detach()
    except OSError:
        return
    if handle is not None and handle != -1:
        try:
            socket.close(handle)
        except OSError:
            pass


class _Connection(http.client.HTTPConnection):
    """An ``HTTPConnection`` that remembers the socket it connected.

    ``http.client`` hands the socket to the response (and forgets it) when
    the server says ``Connection: close``; :meth:`Sam3Client.close` still has
    to be able to shut it down under a reader blocked on it.
    """

    last_sock: Any = None

    def connect(self) -> None:
        try:
            http.client.HTTPConnection.connect(self)
        except socket.timeout as exc:
            raise _ConnectTimeoutError("connect timed out") from exc
        self.last_sock = self.sock


def _connect_host(host: Optional[str]) -> str:
    """The address to *connect* to for a daemon that reported ``host``.

    ``runtime.json`` records the bind address.  A daemon started by hand with
    ``--host 0.0.0.0`` (or ``::``) listens everywhere, but "everywhere" is not
    an address a client can dial: use the matching loopback instead.  Brackets
    around an IPv6 literal are dropped; ``http.client`` wants the bare form.
    """
    text = (host or "").strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return "127.0.0.1"
    try:
        addr = ipaddress.ip_address(text.split("%", 1)[0])
    except ValueError:
        return text
    if addr.is_unspecified:
        return "::1" if addr.version == 6 else "127.0.0.1"
    return text


class Sam3Client:
    """Typed client for every endpoint in ``API.md``.

    Thread-safe.  Each request checks a keep-alive
    ``http.client.HTTPConnection`` out of a small pool and returns it when the
    response has been read (``API.md`` §1 -- the client long-polls a job on
    one connection while POSTing a new prompt on another, which is exactly why
    the daemon must be multi-threaded).  Connections belong to requests, not
    to threads: the dialog runs every call on a fresh thread, and a
    per-thread connection outlived its thread -- one open socket here and one
    parked handler thread in the daemon per call, for minutes.
    """

    #: Idle keep-alive connections kept for reuse; any beyond this are closed
    #: as they are returned.  Concurrency is not limited by it.
    MAX_IDLE_CONNECTIONS = 4

    def __init__(
        self,
        host: str,
        port: int,
        token: str,
        *,
        connect_timeout: float = 5.0,
        read_timeout: float = 30.0,
        api_version: str = API_VERSION,
        pid: Optional[int] = None,
        version: str = "",
        dispatcher: Optional[Dispatcher] = None,
    ) -> None:
        self.host = _connect_host(host)
        self.port = int(port)
        self.token = str(token)
        self.connect_timeout = float(connect_timeout)
        self.read_timeout = float(read_timeout)
        self.api_version = api_version
        self.daemon_pid = pid
        self.daemon_version = version

        self._conn_lock = threading.Lock()
        self._idle: List[http.client.HTTPConnection] = []
        self._busy: List[http.client.HTTPConnection] = []
        self._closed = False

        self._dispatcher = dispatcher
        self._dispatcher_owned = dispatcher is None

        self._rid_lock = threading.Lock()
        self._rid_counter = 0
        self._latest: Dict[str, str] = {}
        #: image_id -> uploaded (width, height), to check result frames against.
        self._uploaded: Dict[str, Tuple[int, int]] = {}

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_runtime_info(cls, info: Dict[str, Any], **kwargs: Any) -> "Sam3Client":
        """Build from a parsed ``runtime.json`` object."""
        return cls(
            host=str(info.get("host", "127.0.0.1")),
            port=int(info["port"]),
            token=str(info["token"]),
            pid=int(info["pid"]) if info.get("pid") is not None else None,
            version=str(info.get("version", "")),
            **kwargs
        )

    @classmethod
    def from_runtime_file(cls, path: Optional[str] = None, **kwargs: Any) -> "Sam3Client":
        """Build from ``runtime.json`` on disk.

        Raises :class:`DaemonUnavailable` when the file is missing or stale --
        the launcher's cue to spawn.
        """
        info = read_runtime_json(path)
        if info is None:
            raise DaemonUnavailable(
                "no usable runtime.json at %s" % (path or default_runtime_file())
            )
        return cls.from_runtime_info(info, **kwargs)

    def __enter__(self) -> "Sam3Client":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    @property
    def _authority(self) -> str:
        """``host:port`` as it appears in a URL or a ``Host`` header --
        bracketed for an IPv6 literal (RFC 3986), which is also the only
        spelling of ``::1`` the daemon's DNS-rebinding guard accepts."""
        if ":" in self.host:
            return "[%s]:%d" % (self.host, self.port)
        return "%s:%d" % (self.host, self.port)

    @property
    def base_url(self) -> str:
        return "http://" + self._authority

    # -- connection management -------------------------------------------- #
    def _checkout(self, timeout: float, fresh: bool = False) -> Tuple[http.client.HTTPConnection, bool]:
        """An idle keep-alive connection, or a new one.  ``(conn, reused)``.

        ``fresh`` skips the idle pool and drops it: used for the retry after a
        reused socket turned out dead, when the rest of the pool (the daemon
        restarted, or timed them all out) almost certainly is too.

        Raises :class:`DaemonUnavailable` once :meth:`close` has been called --
        checked under the same lock that ``close`` takes, so a request either
        started before ``close`` (and is shut down by it) or never starts.
        """
        stale: List[http.client.HTTPConnection] = []
        with self._conn_lock:
            if self._closed:
                raise DaemonUnavailable("client closed")
            if fresh:
                stale, self._idle = self._idle, []
            conn = self._idle.pop() if self._idle else None
            if conn is None:
                conn = _Connection(self.host, self.port, timeout=timeout)
            self._busy.append(conn)
        for old in stale:
            try:
                old.close()
            except Exception:
                pass
        conn.timeout = timeout
        sock = getattr(conn, "sock", None)
        if sock is not None:
            try:
                sock.settimeout(timeout)
            except OSError:
                pass
        return conn, sock is not None

    def _checkin(self, conn: http.client.HTTPConnection, reusable: bool) -> None:
        """Return a connection after its response was read in full, or close it."""
        with self._conn_lock:
            try:
                self._busy.remove(conn)
            except ValueError:
                pass
            if (reusable and not self._closed and getattr(conn, "sock", None) is not None
                    and len(self._idle) < self.MAX_IDLE_CONNECTIONS):
                self._idle.append(conn)
                return
        try:
            conn.close()
        except Exception:
            pass

    def open_connections(self) -> int:
        """How many sockets this client holds (idle plus in flight)."""
        with self._conn_lock:
            return sum(1 for c in self._idle + self._busy if getattr(c, "sock", None) is not None)

    @property
    def closed(self) -> bool:
        with self._conn_lock:
            return self._closed

    def close(self) -> None:
        """End this client for good; idempotent and safe from any thread.

        Every connection is shut down, including one another thread is
        blocked on in a long-poll (see :func:`_abort_socket`): a bare
        ``close()`` from another thread does not wake a pending read.  From then on
        every call -- the interrupted one included -- raises
        :class:`DaemonUnavailable` ("client closed") instead of reconnecting,
        so a ``wait_for_job`` on a worker thread stops rather than carrying on
        against a daemon the owner has let go of.  The owned dispatcher is
        stopped too.
        """
        with self._conn_lock:
            self._closed = True
            conns = self._idle + self._busy
            self._idle, self._busy = [], []
        for conn in conns:
            for sock in {id(s): s for s in (getattr(conn, "sock", None),
                                             getattr(conn, "last_sock", None))
                         if s is not None}.values():
                _abort_socket(sock)
            try:
                conn.close()
            except Exception:
                pass
        if self._dispatcher is not None and self._dispatcher_owned:
            self._dispatcher.shutdown(wait=False)
            self._dispatcher = None

    # -- request plumbing -------------------------------------------------- #
    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = {
            "Authorization": "Bearer " + self.token,
            "Host": self._authority,  # loopback literal: passes the §2 rebind guard
            "Accept": "application/json, " + RESULT_CONTENT_TYPE,
            "X-Sam3-Api": self.api_version,
            "Connection": "keep-alive",
            "User-Agent": "gimp-sam3-plugin/%s" % self.api_version,
        }
        if extra:
            headers.update(extra)
        return headers

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        max_bytes: int = MAX_JSON_RESPONSE_BYTES,
    ) -> Tuple[int, Dict[str, str], bytes]:
        """One HTTP exchange, with a single retry on a dead keep-alive socket.

        The body is read only when its declared ``Content-Length`` is at most
        ``max_bytes``; a chunked or unsized body is refused outright (§1: every
        response carries ``Content-Length``).  Whatever is on the other end of
        the socket, it cannot make this process buffer more than that.
        """
        t = float(self.read_timeout if timeout is None else timeout)
        hdrs = self._headers(headers)
        hdrs["Content-Length"] = str(len(body) if body is not None else 0)

        last_exc: Optional[BaseException] = None
        for attempt in (0, 1):
            conn, reused = self._checkout(t, fresh=attempt > 0)
            keep = False
            resp = None
            try:
                conn.request(method, path, body=body, headers=hdrs)
                resp = conn.getresponse()
                data = self._read_body(resp, max_bytes, method, path)
                status = resp.status
                resp_headers = {k.lower(): v for k, v in resp.getheaders()}
                keep = not resp.will_close
                return status, resp_headers, data
            except (http.client.HTTPException, OSError) as exc:
                if self.closed:
                    # close() shut this socket down under us; that is the
                    # answer, not a daemon failure to retry.
                    raise DaemonUnavailable("client closed") from exc
                if isinstance(exc, _ConnectTimeoutError):
                    raise ConnectTimeout(
                        "nothing accepted a connection at %s within %.1fs" % (self.base_url, t)
                    ) from exc
                if isinstance(exc, socket.timeout):
                    raise RequestTimeout("%s %s timed out after %.1fs" % (method, path, t)) from exc
                if isinstance(exc, ConnectionRefusedError):
                    raise DaemonUnavailable(
                        "connection refused at %s -- the daemon is not listening" % self.base_url
                    ) from exc
                # A reused keep-alive connection can be closed by the server
                # between requests; that is not a daemon failure, so retry once
                # on a fresh socket before giving up.
                last_exc = exc
                if attempt == 0 and reused:
                    continue
                if isinstance(exc, socket.gaierror):
                    raise DaemonUnavailable("cannot resolve %s" % self.host) from exc
                raise DaemonDied(
                    "connection to %s failed during %s %s: %s" % (self.base_url, method, path, exc)
                ) from exc
            finally:
                if resp is not None and not keep:
                    # With "Connection: close" the response owns the socket;
                    # closing the connection alone would leave it open.
                    resp.close()
                self._checkin(conn, keep)
        raise DaemonDied("request failed: %s" % (last_exc,))  # pragma: no cover - unreachable

    @staticmethod
    def _read_body(resp: http.client.HTTPResponse, max_bytes: int, method: str, path: str) -> bytes:
        problem = ""
        length = resp.length
        if getattr(resp, "chunked", False):
            problem = "the daemon sent a chunked response; API.md §1 requires Content-Length"
        elif length is None:
            problem = "the response has no Content-Length"
        elif length > max_bytes:
            problem = "a %d-byte response exceeds the %d-byte limit" % (length, max_bytes)
        if problem:
            raise ProtocolError("%s %s: %s" % (method, path, problem))
        return resp.read(length) if length else b""

    @staticmethod
    def _decode_json(data: bytes, what: str) -> Any:
        try:
            return json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ProtocolError("%s: response is not valid JSON: %s" % (what, exc))

    def _raise_for_status(self, status: int, data: bytes, what: str) -> None:
        """Turn a non-2xx §4 error envelope into the right typed exception."""
        if 200 <= status < 300:
            return
        code, message, detail = ErrorCode.INTERNAL_ERROR, "", {}
        try:
            payload = json.loads(data.decode("utf-8"))
            err = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(err, dict):
                code = str(err.get("code", code))
                message = str(err.get("message", ""))
                d = err.get("detail")
                detail = d if isinstance(d, dict) else {}
        except (ValueError, UnicodeDecodeError, AttributeError):
            message = data[:200].decode("utf-8", "replace")
        # The token is a capability (§2): whatever the other end chose to echo
        # back must not carry it into an error dialog or a log line.
        if len(self.token) >= 8 and self.token in message:
            message = message.replace(self.token, "<token>")
        if status == 401 or code == ErrorCode.UNAUTHORIZED:
            raise AuthError(ErrorCode.UNAUTHORIZED, message or "bad bearer token", detail, status)
        if code == ErrorCode.VERSION_MISMATCH:
            raise VersionMismatch(
                message,
                client_api=self.api_version,
                server_api=str(detail.get("api_version", "")),
                detail=detail,
                status=status,
            )
        raise ApiError(code, message or what, detail, status)

    def _json_request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        body = None
        headers = dict(headers or {})
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            if len(body) > Limits.MAX_JSON_BYTES:
                raise ValueError("JSON body of %d bytes exceeds the 256 KiB limit" % len(body))
            headers["Content-Type"] = JSON_CONTENT_TYPE
        status, _hdrs, data = self._request(method, path, body, headers or None, timeout)
        self._raise_for_status(status, data, "%s %s" % (method, path))
        parsed = self._decode_json(data, "%s %s" % (method, path))
        if not isinstance(parsed, dict):
            raise ProtocolError("%s %s: expected a JSON object" % (method, path))
        return parsed

    # -- request ids and supersession (API.md section 10) ------------------- #
    def next_request_id(self, image_id: Optional[str] = None) -> str:
        """Allocate the next request id and record it as the latest for ``image_id``.

        Monotonic across the whole client so ids are unique even when several
        images are in play; the *latest* map is per image because supersession
        is per image.
        """
        with self._rid_lock:
            self._rid_counter += 1
            rid = "r-%06d" % self._rid_counter
            self._latest[image_id or ""] = rid
        return rid

    def _claim_request_id(self, image_id: Optional[str], request_id: Optional[str]) -> str:
        """The id a prompt goes out under, recorded as the latest for its image.

        A caller-chosen id is validated and recorded exactly like one this
        client allocated; otherwise a caller that names its own ids would have
        its results judged against whatever this client allocated last.
        """
        if not request_id:
            return self.next_request_id(image_id)
        rid = str(request_id)
        _validate_request_id(rid)
        with self._rid_lock:
            self._latest[image_id or ""] = rid
        return rid

    def latest_request_id(self, image_id: Optional[str] = None) -> Optional[str]:
        with self._rid_lock:
            return self._latest.get(image_id or "")

    def is_latest(self, image_id: Optional[str], request_id: str) -> bool:
        """The §10 rule: a result is usable only if its id is still the latest."""
        with self._rid_lock:
            latest = self._latest.get(image_id or "")
        return latest is None or latest == request_id

    def is_stale(self, image_id: Optional[str], request_id: str) -> bool:
        return not self.is_latest(image_id, request_id)

    def forget_image(self, image_id: str) -> None:
        with self._rid_lock:
            self._latest.pop(image_id or "", None)
            self._uploaded.pop(image_id or "", None)

    # -- endpoints --------------------------------------------------------- #
    def hello(self, *, check: bool = True, timeout: Optional[float] = None) -> Hello:
        """``GET /hello``.  Cheap; never loads a model.

        Every call sends a fresh nonce (``X-Sam3-Nonce``) and requires the
        daemon's ``nonce_proof`` -- an HMAC of it keyed by the bearer token --
        in the answer.  Anything that cannot produce it raises
        :class:`DaemonIdentityError` and must not be used: it is not the
        daemon that wrote ``runtime.json`` (or it predates API 1.1).

        With ``check=True`` (the default) an incompatible API major raises
        :class:`VersionMismatch` -- step 4 of the find-or-spawn handshake.
        """
        nonce = secrets.token_urlsafe(24)
        payload = self._json_request(
            "GET", "/hello", timeout=self.connect_timeout if timeout is None else timeout,
            headers={NONCE_HEADER: nonce},
        )
        reason = ""
        try:
            hello = Hello.from_dict(payload)
        except (TypeError, ValueError, AttributeError) as exc:
            hello = None
            reason, problem = DaemonIdentityError.MALFORMED, "a malformed /hello (%s)" % (exc,)
        else:
            proof = payload.get(NONCE_PROOF_KEY)
            if proof is None or proof == "":
                reason, problem = DaemonIdentityError.MISSING, "no %s" % NONCE_PROOF_KEY
            elif not isinstance(proof, str) or not hmac.compare_digest(
                    proof.encode("utf-8", "replace"), nonce_proof(self.token, nonce).encode("ascii")):
                reason, problem = DaemonIdentityError.WRONG, "a wrong %s" % NONCE_PROOF_KEY
        if reason:
            raise DaemonIdentityError(
                "the process answering at %s sent %s in its /hello: it cannot prove it "
                "holds this daemon's token, so it is not the sam3gimpd that wrote "
                "runtime.json (or it is older than API %s)" % (self.base_url, problem, API_VERSION),
                hello=hello, reason=reason,
            )
        if check and not versions_compatible(self.api_version, hello.api_version):
            raise VersionMismatch(
                "daemon speaks API %s, this plug-in speaks %s"
                % (hello.api_version or "?", self.api_version),
                client_api=self.api_version,
                server_api=hello.api_version,
            )
        return hello

    def status(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """``GET /status`` -- the raw Doctor-panel payload (§6.7)."""
        return self._json_request("GET", "/status", timeout=timeout)

    def upload_image(
        self,
        pixels: bytes,
        width: int,
        height: int,
        *,
        source_width: Optional[int] = None,
        source_height: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> ImageAccepted:
        """``POST /images`` with raw RGB (§7).

        ``pixels`` must be exactly ``width * height * 3`` bytes: uint8, channel
        order R-G-B, rows top-to-bottom, stride ``width * 3``, no padding and no
        alpha -- precisely what ``Gegl.Buffer.get(..., "R'G'B' u8", ...)``
        returns.  The size rules are checked here so a 3 MB body is never sent
        just to be rejected.
        """
        width = int(width)
        height = int(height)
        for name, value in (("width", width), ("height", height)):
            if not (Limits.MIN_IMAGE_SIDE <= value <= Limits.MAX_IMAGE_SIDE):
                raise ValueError(
                    "%s %d is outside the legal range [%d, %d]; downscale before uploading"
                    % (name, value, Limits.MIN_IMAGE_SIDE, Limits.MAX_IMAGE_SIDE)
                )
        expected = width * height * 3
        body = bytes(pixels)
        if len(body) != expected:
            raise ValueError(
                "pixel buffer is %d bytes, expected width*height*3 = %d" % (len(body), expected)
            )

        headers = {
            "Content-Type": OCTET_CONTENT_TYPE,
            "X-Width": str(width),
            "X-Height": str(height),
        }
        if source_width is not None:
            headers["X-Source-Width"] = str(int(source_width))
        if source_height is not None:
            headers["X-Source-Height"] = str(int(source_height))

        status, _hdrs, data = self._request(
            "POST",
            "/images",
            body,
            headers,
            timeout=self.read_timeout if timeout is None else timeout,
        )
        self._raise_for_status(status, data, "POST /images")
        accepted = ImageAccepted.from_dict(self._decode_json(data, "POST /images"))
        with self._rid_lock:
            self._uploaded[accepted.image_id] = (width, height)
        return accepted

    def prompt_text(
        self,
        image_id: str,
        text: str,
        *,
        request_id: Optional[str] = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        max_instances: int = 64,
        boxes: Optional[Sequence[Any]] = None,
        timeout: Optional[float] = None,
    ) -> JobAccepted:
        """``POST /images/{id}/text`` -- PCS: one noun phrase, every instance.

        SAM 3 wants a *simple noun phrase* (``"red car"``), not a relational
        description; exclusion is expressed with negative boxes/points, not
        with words.  The daemon does not rewrite the text and neither do we.

        ``request_id`` defaults to a fresh one; either way it becomes the
        latest for ``image_id`` (§10).
        """
        text = _check_text(text)
        norm_boxes = _normalise_boxes(boxes)
        rid = self._claim_request_id(image_id, request_id)
        return self._post_text(image_id, text, rid, score_threshold, max_instances,
                               norm_boxes, timeout)

    def _post_text(self, image_id: str, text: str, rid: str, score_threshold: float,
                   max_instances: int, norm_boxes: List[Dict[str, Any]],
                   timeout: Optional[float]) -> JobAccepted:
        payload: Dict[str, Any] = {
            "request_id": rid,
            "text": text,
            "score_threshold": float(score_threshold),
            "max_instances": int(max_instances),
        }
        if norm_boxes:
            payload["boxes"] = norm_boxes
        data = self._json_request(
            "POST", "/images/%s/text" % _quote(image_id), payload, timeout=timeout
        )
        return JobAccepted.from_dict(data)

    def prompt_points(
        self,
        image_id: str,
        points: Optional[Iterable[Any]] = None,
        *,
        box: Optional[Sequence[float]] = None,
        request_id: Optional[str] = None,
        multimask: bool = True,
        max_instances: int = 3,
        timeout: Optional[float] = None,
    ) -> JobAccepted:
        """``POST /images/{id}/points`` -- PVS: points/box -> one instance.

        Coordinates are **uploaded-image** pixels (§5).  The API is stateless
        per request: send the *complete* point set every time.  ``request_id``
        is handled as in :meth:`prompt_text`.
        """
        pts, norm_box = _check_points(points, box)
        rid = self._claim_request_id(image_id, request_id)
        return self._post_points(image_id, pts, norm_box, rid, multimask, max_instances, timeout)

    def _post_points(self, image_id: str, pts: List[Dict[str, Any]], norm_box: Optional[List[float]],
                     rid: str, multimask: bool, max_instances: int,
                     timeout: Optional[float]) -> JobAccepted:
        payload: Dict[str, Any] = {
            "request_id": rid,
            "points": pts,
            "multimask": bool(multimask),
            "max_instances": int(max_instances),
        }
        if norm_box is not None:
            payload["box"] = norm_box
        data = self._json_request(
            "POST", "/images/%s/points" % _quote(image_id), payload, timeout=timeout
        )
        return JobAccepted.from_dict(data)

    def job_status(self, job_id: str, wait: float = 0.0) -> JobStatus:
        """``GET /jobs/{id}?wait=N&meta=1`` -- always the JSON form.

        The socket timeout is stretched past the long-poll budget so that a
        server honouring ``wait`` is never mistaken for a hung one.
        """
        w = max(0.0, min(float(wait), Limits.MAX_LONG_POLL_SECONDS))
        path = "/jobs/%s?wait=%s&meta=1" % (_quote(job_id), _fmt_float(w))
        data = self._json_request("GET", path, timeout=self.read_timeout + w)
        return JobStatus.from_dict(data)

    def job_result_frame(self, job_id: str, timeout: Optional[float] = None) -> bytes:
        """``GET /jobs/{id}`` returning the **raw** §8 frame, undecoded.

        ``ui/canvas.py`` has its own decoder (``set_result_frame``) so that the
        preview can be driven from a socket, a file or the offline harness with
        no client object at all; this is how it gets the bytes.  Everything
        except the decode is shared with :meth:`job_result`, including the
        ``X-Sam3-Header-Length`` cross-check.
        """
        status, headers, data = self._request(
            "GET", "/jobs/%s" % _quote(job_id), timeout=timeout, max_bytes=MAX_RESULT_FRAME_BYTES
        )
        self._raise_for_status(status, data, "GET /jobs/%s" % job_id)
        ctype = headers.get("content-type", "")
        if not ctype.startswith(RESULT_CONTENT_TYPE):
            state = headers.get("x-sam3-job-state", "")
            if not state:
                try:
                    state = str(json.loads(data.decode("utf-8")).get("state", ""))
                except Exception:
                    state = "?"
            raise ProtocolError(
                "job %s is not done (state=%r, content-type=%r)" % (job_id, state, ctype)
            )
        declared = headers.get("x-sam3-header-length")
        if declared is not None:
            try:
                if int(declared) != _U32.unpack_from(data, 8)[0]:
                    raise ProtocolError(
                        "X-Sam3-Header-Length %s disagrees with the in-band value" % declared
                    )
            except ValueError:
                pass  # unparseable hint header; the in-band value is authoritative
            except struct.error:
                raise ProtocolError("result frame is shorter than its 12 byte prefix")
        return data

    def job_result(self, job_id: str, timeout: Optional[float] = None) -> Result:
        """``GET /jobs/{id}`` expecting the binary frame; decode it (§8).

        Raises :class:`ProtocolError` if the job is not ``done`` (the daemon
        answered with JSON instead of a frame), if the frame names a different
        job, or if it describes a different image size than the one this
        client uploaded under that ``image_id``.
        """
        result = parse_result_frame(self.job_result_frame(job_id, timeout=timeout))
        if result.job_id and result.job_id != str(job_id):
            raise ProtocolError("asked for job %s, got the result of job %s" % (job_id, result.job_id))
        with self._rid_lock:
            uploaded = self._uploaded.get(result.image_id)
        if uploaded is not None and uploaded != (result.image.width, result.image.height):
            raise ProtocolError(
                "result for image %s describes a %dx%d upload; this client uploaded %dx%d"
                % (result.image_id, result.image.width, result.image.height, uploaded[0], uploaded[1])
            )
        return result

    def delete_image(self, image_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """``DELETE /images/{id}`` -- free the embedding, cancel *queued* jobs."""
        data = self._json_request("DELETE", "/images/%s" % _quote(image_id), timeout=timeout)
        self.forget_image(image_id)
        return data

    def shutdown_daemon(self, grace_ms: int = 500, timeout: Optional[float] = None) -> Dict[str, Any]:
        """``POST /shutdown``.  Used by the Doctor panel and by the launcher when
        it finds a daemon speaking an incompatible API major."""
        return self._json_request("POST", "/shutdown", {"grace_ms": int(grace_ms)}, timeout=timeout)

    # -- composed helpers -------------------------------------------------- #
    def _long_poll(
        self,
        job_id: str,
        timeout: Optional[float],
        stall_timeout: float,
        poll_wait: float,
        on_progress: Optional[Callable[[JobStatus], Any]],
        finished: Callable[[JobStatus], bool],
    ) -> JobStatus:
        """Long-poll ``job_id`` until ``finished(status)``; return that status.

        Gives up with :class:`JobTimeout` only when the job stops moving --
        state, stage, progress and queue position all unchanged for
        ``stall_timeout`` seconds -- or, if the caller set one, when the overall
        ``timeout`` runs out.  A slow machine that keeps advancing is waited
        for; a wedged daemon is not.
        """
        start = time.monotonic()
        cap = None if timeout is None else start + max(0.0, float(timeout))
        stall = max(0.0, float(stall_timeout))
        last: Optional[JobStatus] = None
        mark: Any = None
        moved_at = start
        while True:
            now = time.monotonic()
            if cap is not None and now >= cap:
                raise JobTimeout(job_id, float(timeout or 0.0), last)
            if now - moved_at >= stall:
                raise JobTimeout(job_id, now - moved_at, last, stalled=True)
            budget = min(float(poll_wait), moved_at + stall - now)
            if cap is not None:
                budget = min(budget, cap - now)
            status = self.job_status(job_id, wait=max(0.0, budget))
            if on_progress is not None and _progress_changed(last, status):
                on_progress(status)
            last = status
            current = (status.state, status.stage, status.progress, status.queue_position)
            if current != mark:
                mark = current
                moved_at = time.monotonic()
            if finished(status):
                return status

    def wait_for_job(
        self,
        job_id: str,
        *,
        timeout: Optional[float] = None,
        stall_timeout: float = DEFAULT_STALL_TIMEOUT,
        poll_wait: float = 10.0,
        on_progress: Optional[Callable[[JobStatus], Any]] = None,
        image_id: Optional[str] = None,
        request_id: Optional[str] = None,
        raise_on_failure: bool = True,
    ) -> Optional[Result]:
        """Long-poll a job to a terminal state and fetch its frame.

        Returns the decoded :class:`Result`, or ``None`` when the result is not
        wanted any more: the job was ``superseded``/``cancelled``, or its
        ``request_id`` is no longer the latest for that image (``API.md`` §10 --
        *drop any result whose request_id is not the latest*).

        Raises :class:`JobFailed` on ``state == "failed"`` (unless
        ``raise_on_failure=False``, which returns ``None`` instead) and
        :class:`JobTimeout` when the job stalls for ``stall_timeout`` seconds or
        an explicit overall ``timeout`` runs out (there is none by default).
        """

        def _finished(status: JobStatus) -> bool:
            # A different request id means the id was reused or we are looking
            # at someone else's job: stop, and treat it as stale below.
            foreign = bool(request_id and status.request_id and status.request_id != request_id)
            return foreign or status.is_terminal

        status = self._long_poll(job_id, timeout, stall_timeout, poll_wait, on_progress, _finished)
        if request_id and status.request_id and status.request_id != request_id:
            return None
        _raise_if_daemon_left(status)
        if status.state in (JobState.SUPERSEDED, JobState.CANCELLED):
            return None
        if status.state == JobState.FAILED:
            if not raise_on_failure:
                return None
            code, message, detail = status.error_tuple()
            raise JobFailed(job_id, code, message, detail)
        rid = request_id or status.request_id
        key = image_id if image_id is not None else (status.image_id or None)
        if rid and self.is_stale(key, rid):
            return None
        result = self.job_result(job_id)
        if result.request_id and self.is_stale(key, result.request_id):
            return None
        return result

    def run_text(
        self,
        image_id: str,
        text: str,
        *,
        request_id: Optional[str] = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        max_instances: int = 64,
        boxes: Optional[Sequence[Any]] = None,
        timeout: Optional[float] = None,
        stall_timeout: float = DEFAULT_STALL_TIMEOUT,
        poll_wait: float = 10.0,
        on_progress: Optional[Callable[[JobStatus], Any]] = None,
    ) -> Optional[Result]:
        """PCS prompt end to end.  ``None`` means "superseded, show nothing".

        Waits as :meth:`wait_for_job` does: until the job stalls, or until
        ``timeout`` if the caller gives one.
        """
        text = _check_text(text)
        norm_boxes = _normalise_boxes(boxes)
        rid = self._claim_request_id(image_id, request_id)
        accepted = self._post_text(image_id, text, rid, score_threshold, max_instances,
                                   norm_boxes, None)
        return self.wait_for_job(
            accepted.job_id,
            timeout=timeout,
            stall_timeout=stall_timeout,
            poll_wait=poll_wait,
            on_progress=on_progress,
            image_id=image_id,
            request_id=rid,
        )

    def run_points(
        self,
        image_id: str,
        points: Optional[Iterable[Any]] = None,
        *,
        box: Optional[Sequence[float]] = None,
        request_id: Optional[str] = None,
        multimask: bool = True,
        max_instances: int = 3,
        timeout: Optional[float] = None,
        stall_timeout: float = DEFAULT_STALL_TIMEOUT,
        poll_wait: float = 10.0,
        on_progress: Optional[Callable[[JobStatus], Any]] = None,
    ) -> Optional[Result]:
        """PVS prompt end to end.  ``None`` means "superseded, show nothing".

        Waits as :meth:`wait_for_job` does.
        """
        pts, norm_box = _check_points(points, box)
        rid = self._claim_request_id(image_id, request_id)
        accepted = self._post_points(image_id, pts, norm_box, rid, multimask, max_instances, None)
        return self.wait_for_job(
            accepted.job_id,
            timeout=timeout,
            stall_timeout=stall_timeout,
            poll_wait=poll_wait,
            on_progress=on_progress,
            image_id=image_id,
            request_id=rid,
        )

    def wait_for_image(
        self,
        accepted: ImageAccepted,
        *,
        timeout: Optional[float] = None,
        stall_timeout: float = DEFAULT_STALL_TIMEOUT,
        poll_wait: float = 10.0,
        on_progress: Optional[Callable[[JobStatus], Any]] = None,
    ) -> JobStatus:
        """Block until the encode job finishes.

        Callers *may* prompt without waiting -- the queue guarantees ordering
        (§6.2) -- but the Doctor panel and the tests want the encode outcome.
        Gives up on the same terms as :meth:`wait_for_job`.
        """
        status = self._long_poll(accepted.job_id, timeout, stall_timeout, poll_wait,
                                 on_progress, lambda s: s.is_terminal)
        _raise_if_daemon_left(status)
        if status.state == JobState.FAILED:
            code, message, detail = status.error_tuple()
            raise JobFailed(accepted.job_id, code, message, detail)
        return status

    # -- worker-thread API -------------------------------------------------- #
    @property
    def dispatcher(self) -> Dispatcher:
        """The lazily created worker pool backing :meth:`submit`."""
        if self._dispatcher is None:
            self._dispatcher = Dispatcher(workers=1, name="sam3gimpd-http")
            self._dispatcher_owned = True
        return self._dispatcher

    def submit(self, fn: Callable[[], Any], on_done=None, on_error=None) -> Call:
        """Run ``fn()`` off the GTK main loop; deliver the outcome back onto it.

        ``fn`` is a zero-argument callable -- typically a lambda closing over
        this client::

            client.submit(lambda: client.run_text(image_id, "red car"),
                          on_done=self._show_masks,
                          on_error=self._show_error)

        ``on_done`` / ``on_error`` are marshalled with ``GLib.idle_add`` when
        ``gi`` is importable, and called directly otherwise.  **Never call a
        blocking client method from the GTK thread.**
        """
        return self.dispatcher.submit(fn, on_done=on_done, on_error=on_error)

    def wrap(self, callback: Callable[..., Any]) -> Callable[..., None]:
        """Wrap a UI callback (e.g. a progress bar update) for worker threads."""
        return self.dispatcher.wrap(callback)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
_RID_ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:-")


def _validate_request_id(rid: str) -> None:
    if not rid or len(rid) > Limits.MAX_REQUEST_ID_CHARS:
        raise ValueError("request_id must be 1..%d characters" % Limits.MAX_REQUEST_ID_CHARS)
    bad = set(rid) - _RID_ALLOWED
    if bad:
        raise ValueError("request_id contains illegal characters: %s" % "".join(sorted(bad)))


def _raise_if_daemon_left(status: JobStatus) -> None:
    """A job the daemon cancelled because it is shutting down did not lose to
    a newer prompt; the daemon is going away.  Say so as a transport error --
    what a caller reconnects on -- instead of the silent ``None`` that means
    "superseded"."""
    if status.state == JobState.CANCELLED and status.error_tuple()[0] == ErrorCode.SHUTTING_DOWN:
        raise DaemonUnavailable(
            "the daemon is shutting down; job %s was cancelled" % status.job_id)


def _check_text(text: Any) -> str:
    text = str(text)
    if not text.strip():
        raise ValueError("text prompt must not be empty")
    if len(text) > Limits.MAX_TEXT_CHARS:
        raise ValueError("text prompt exceeds %d characters" % Limits.MAX_TEXT_CHARS)
    return text


def _check_points(points: Optional[Iterable[Any]],
                  box: Optional[Sequence[float]]) -> Tuple[List[Dict[str, Any]], Optional[List[float]]]:
    pts = _normalise_points(points)
    if not pts and box is None:
        raise ValueError("at least one point or a box is required")
    if len(pts) > Limits.MAX_POINTS:
        raise ValueError("at most %d points are allowed" % Limits.MAX_POINTS)
    return pts, (_normalise_box(box) if box is not None else None)


def _quote(value: str) -> str:
    """Percent-encode a path segment.  Ids are opaque, so never trust them."""
    out = []
    for ch in str(value):
        if ch.isalnum() or ch in "-._~":
            out.append(ch)
        else:
            out.extend("%%%02X" % b for b in ch.encode("utf-8"))
    return "".join(out)


def _fmt_float(v: float) -> str:
    return ("%.3f" % float(v)).rstrip("0").rstrip(".") or "0"


def _normalise_points(points: Optional[Iterable[Any]]) -> List[Dict[str, Any]]:
    """Accept ``Point``s, ``{"x","y","label"}`` dicts or ``(x, y[, label])`` tuples."""
    out: List[Dict[str, Any]] = []
    for p in points or []:
        if isinstance(p, Point):
            out.append(p.to_dict())
        elif isinstance(p, dict):
            out.append(
                {"x": float(p["x"]), "y": float(p["y"]), "label": int(p.get("label", 1))}
            )
        elif isinstance(p, (tuple, list)) and len(p) in (2, 3):
            label = int(p[2]) if len(p) == 3 else 1
            out.append({"x": float(p[0]), "y": float(p[1]), "label": label})
        else:
            raise ValueError("cannot interpret %r as a point" % (p,))
    for p in out:
        if p["label"] not in (0, 1):
            raise ValueError("point label must be 0 (exclude) or 1 (include)")
    return out


def _normalise_box(box: Sequence[float]) -> List[float]:
    if len(box) != 4:
        raise ValueError("box must be [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (float(v) for v in box)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("box must satisfy x1 > x0 and y1 > y0")
    return [x0, y0, x1, y1]


def _normalise_boxes(boxes: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
    """Exemplar boxes for PCS: ``{"box": [...], "label": 1|0}`` entries."""
    out: List[Dict[str, Any]] = []
    for b in boxes or []:
        if isinstance(b, dict):
            out.append({"box": _normalise_box(b["box"]), "label": int(b.get("label", 1))})
        elif isinstance(b, (tuple, list)) and len(b) == 4:
            out.append({"box": _normalise_box(b), "label": 1})
        else:
            raise ValueError("cannot interpret %r as an exemplar box" % (b,))
    if len(out) > Limits.MAX_BOXES:
        raise ValueError("at most %d exemplar boxes are allowed" % Limits.MAX_BOXES)
    for b in out:
        if b["label"] not in (0, 1):
            raise ValueError("box label must be 0 (negative) or 1 (positive)")
    return out


def _progress_changed(prev: Optional[JobStatus], cur: JobStatus) -> bool:
    """Mirror of the daemon's long-poll wake condition (``API.md`` §11)."""
    if prev is None:
        return True
    if prev.state != cur.state or prev.stage != cur.stage:
        return True
    return (cur.progress - prev.progress) >= 0.01
