"""Wire types for the sam3gimpd HTTP API.

Every JSON shape in ``API.md`` has a dataclass here, plus the pack/unpack
routines for the binary result frame.  Pure standard library: no torch, no
numpy, no third-party imports, importable anywhere.

The GIMP plug-in **does not import this module** -- it runs inside GIMP's
embedded Python and is restricted to the stdlib plus ``gi``.  Plug-in authors
mirror these shapes by hand, which is why everything here is deliberately dumb:
flat dataclasses, ``to_dict``/``from_dict``, no inheritance tricks, no
validation frameworks.  If you find yourself adding cleverness, put it in the
server instead.

Conventions shared by every ``from_dict``:

* unknown keys are ignored (forward compatibility -- an older client must be
  able to parse a newer daemon's payload);
* missing optional keys fall back to the documented default;
* missing required keys raise ``KeyError``.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "API_VERSION",
    "RESULT_MAGIC",
    "RESULT_PREFIX_SIZE",
    "MASK_ENCODING_U8_SOFT",
    "DEFAULT_MASK_THRESHOLD",
    "DEFAULT_SCORE_THRESHOLD",
    "NONCE_HEADER",
    "NONCE_PROOF_PREFIX",
    "NONCE_MIN_CHARS",
    "NONCE_MAX_CHARS",
    "is_valid_nonce",
    "nonce_proof",
    "JobState",
    "Engine",
    "PromptKind",
    "ErrorCode",
    "Limits",
    "ErrorInfo",
    "ApiError",
    "Size",
    "CanvasTransform",
    "BBox",
    "Point",
    "RuntimeInfo",
    "HelloResponse",
    "ImageAccepted",
    "JobAccepted",
    "TextPrompt",
    "PointPrompt",
    "MaskInstance",
    "ResultHeader",
    "JobStatus",
    "CacheEntry",
    "StatusResponse",
    "pack_result",
    "unpack_result",
]

#: Version of the HTTP contract implemented by this package.  Clients compare
#: the *major* component only; a minor bump is additive and backwards
#: compatible.  1.1 added the ``/hello`` identity proof (``nonce_proof``).
API_VERSION = "1.1"

#: First 8 bytes of a binary result frame.
RESULT_MAGIC = b"SAM3RES\x00"

#: ``len(RESULT_MAGIC) + 4`` -- magic plus the little-endian uint32 header length.
RESULT_PREFIX_SIZE = 12

_U32 = struct.Struct("<I")

#: The only mask encoding in API 1.x: uint8 soft masks, 0..255, row-major.
MASK_ENCODING_U8_SOFT = "u8_soft"

#: 128 corresponds to a raw logit of 0, i.e. the model's own binarisation point.
DEFAULT_MASK_THRESHOLD = 128

#: PCS returns every candidate at or above this score; filtering happens client
#: side so the user can move the slider with no round trip.  It is a hard floor
#: on what that slider can reveal, so it sits well under any UI default.
DEFAULT_SCORE_THRESHOLD = 0.02

#: Request header carrying the client's nonce on ``GET /hello`` (``API.md`` §2).
NONCE_HEADER = "X-Sam3-Nonce"
#: Domain separator for the proof, so it can never be mistaken for another MAC.
NONCE_PROOF_PREFIX = b"sam3gimpd-hello:"
#: A nonce is URL-safe base64 (``secrets.token_urlsafe``) of this many characters.
NONCE_MIN_CHARS = 16
NONCE_MAX_CHARS = 128

_NONCE_RE = re.compile(r"^[A-Za-z0-9_-]{%d,%d}$" % (NONCE_MIN_CHARS, NONCE_MAX_CHARS))


def is_valid_nonce(nonce: Any) -> bool:
    """Is this a well-formed ``X-Sam3-Nonce`` value?"""
    return isinstance(nonce, str) and bool(_NONCE_RE.match(nonce))


def nonce_proof(token: str, nonce: str) -> str:
    """``HMAC-SHA256(token, "sam3gimpd-hello:" + nonce)`` as lowercase hex.

    Only the process that generated ``token`` can answer a fresh nonce, so a
    client that checks this knows it reached the daemon ``runtime.json`` names
    and not whatever else is listening on that port now (``API.md`` §2).
    """
    return hmac.new(token.encode("utf-8"), NONCE_PROOF_PREFIX + nonce.encode("ascii"),
                    hashlib.sha256).hexdigest()


class JobState:
    """Terminal states are ``done``, ``failed``, ``superseded`` and ``cancelled``."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SUPERSEDED = "superseded"
    CANCELLED = "cancelled"

    ALL = (QUEUED, RUNNING, DONE, FAILED, SUPERSEDED, CANCELLED)
    TERMINAL = (DONE, FAILED, SUPERSEDED, CANCELLED)

    @staticmethod
    def is_terminal(state: str) -> bool:
        return state in JobState.TERMINAL


class Engine:
    ENCODE = "encode"  # the image-embedding job created by POST /images
    PCS = "pcs"        # text -> every matching instance
    PVS = "pvs"        # points/box -> one instance (+ multimask candidates)

    ALL = (ENCODE, PCS, PVS)


class PromptKind:
    TEXT = "text"
    POINTS = "points"
    ENCODE = "encode"


class ErrorCode:
    """Machine-readable codes carried in every error envelope.

    See ``API.md`` for the HTTP status each one is paired with.  The list is
    closed for API 1.x: clients may switch on these strings.
    """

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

    ALL = (
        BAD_REQUEST, INVALID_JSON, MISSING_HEADER, BAD_DIMENSIONS,
        PAYLOAD_SIZE_MISMATCH, PAYLOAD_TOO_LARGE, UNSUPPORTED_MEDIA_TYPE,
        UNAUTHORIZED, FORBIDDEN_HOST, NOT_FOUND, IMAGE_NOT_FOUND,
        JOB_NOT_FOUND, IMAGE_NOT_READY, METHOD_NOT_ALLOWED, VERSION_MISMATCH,
        QUEUE_FULL, ENGINE_UNAVAILABLE, MODEL_LOAD_FAILED, INFERENCE_FAILED,
        SHUTTING_DOWN, INTERNAL_ERROR,
    )

    #: Default HTTP status for each code.  Handlers may not deviate.
    STATUS = {
        BAD_REQUEST: 400,
        INVALID_JSON: 400,
        MISSING_HEADER: 400,
        BAD_DIMENSIONS: 400,
        PAYLOAD_SIZE_MISMATCH: 400,
        VERSION_MISMATCH: 400,
        UNAUTHORIZED: 401,
        FORBIDDEN_HOST: 403,
        NOT_FOUND: 404,
        IMAGE_NOT_FOUND: 404,
        JOB_NOT_FOUND: 404,
        METHOD_NOT_ALLOWED: 405,
        IMAGE_NOT_READY: 409,
        PAYLOAD_TOO_LARGE: 413,
        UNSUPPORTED_MEDIA_TYPE: 415,
        MODEL_LOAD_FAILED: 500,
        INFERENCE_FAILED: 500,
        INTERNAL_ERROR: 500,
        ENGINE_UNAVAILABLE: 503,
        QUEUE_FULL: 503,
        SHUTTING_DOWN: 503,
    }

    @staticmethod
    def status_for(code: str) -> int:
        return ErrorCode.STATUS.get(code, 500)


class Limits:
    """Hard protocol limits.  The server enforces them; clients must respect them."""

    #: Longest side accepted by POST /images.  SAM 3 works at 1008px; a larger
    #: upload buys nothing and would just be downscaled again.
    MAX_IMAGE_SIDE = 1008
    MIN_IMAGE_SIDE = 16
    #: 1008 * 1008 * 3, the largest legal pixel payload.
    MAX_UPLOAD_BYTES = 1008 * 1008 * 3
    MAX_JSON_BYTES = 256 * 1024
    MAX_TEXT_CHARS = 512
    MAX_POINTS = 64
    MAX_BOXES = 16
    MAX_INSTANCES = 256
    MAX_REQUEST_ID_CHARS = 64
    #: Upper bound accepted for ``?wait=`` on GET /jobs/{id}, in seconds.
    MAX_LONG_POLL_SECONDS = 30.0


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
@dataclass
class ErrorInfo:
    """Body of every error response: ``{"error": {code, message, detail}}``."""

    code: str
    message: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, "detail": dict(self.detail)}

    def to_envelope(self) -> Dict[str, Any]:
        return {"error": self.to_dict()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ErrorInfo":
        detail = d.get("detail") or {}
        if not isinstance(detail, dict):
            detail = {"value": detail}
        return cls(code=d["code"], message=d.get("message", ""), detail=detail)

    @classmethod
    def from_envelope(cls, d: Dict[str, Any]) -> "ErrorInfo":
        return cls.from_dict(d["error"])

    @property
    def http_status(self) -> int:
        return ErrorCode.status_for(self.code)


class ApiError(Exception):
    """Raise to abort a request with a specific error envelope."""

    def __init__(self, code: str, message: str, detail: Optional[Dict[str, Any]] = None,
                 status: Optional[int] = None) -> None:
        super().__init__("%s: %s" % (code, message))
        self.info = ErrorInfo(code=code, message=message, detail=dict(detail or {}))
        self.status = status if status is not None else ErrorCode.status_for(code)

    @property
    def code(self) -> str:
        return self.info.code

    @property
    def message(self) -> str:
        return self.info.message

    @property
    def detail(self) -> Dict[str, Any]:
        return self.info.detail

    def to_envelope(self) -> Dict[str, Any]:
        return self.info.to_envelope()


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
@dataclass
class Size:
    width: int
    height: int

    def to_dict(self) -> Dict[str, int]:
        return {"width": int(self.width), "height": int(self.height)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Size":
        return cls(width=int(d["width"]), height=int(d["height"]))

    @property
    def area(self) -> int:
        return int(self.width) * int(self.height)


@dataclass
class CanvasTransform:
    """Affine map from *uploaded-image* pixels to *model-canvas* pixels.

    ``canvas_x = image_x * scale_x + offset_x``

    The daemon always reports the real transform for the image in question.
    Clients MUST use the reported values and MUST NOT assume any policy: the
    reference engines report the identity (masks are post-processed back to
    the uploaded image), and a hardcoded client breaks the day that changes.
    """

    scale_x: float
    scale_y: float
    offset_x: float = 0.0
    offset_y: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "scale_x": float(self.scale_x),
            "scale_y": float(self.scale_y),
            "offset_x": float(self.offset_x),
            "offset_y": float(self.offset_y),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CanvasTransform":
        return cls(
            scale_x=float(d["scale_x"]),
            scale_y=float(d["scale_y"]),
            offset_x=float(d.get("offset_x", 0.0)),
            offset_y=float(d.get("offset_y", 0.0)),
        )

    @classmethod
    def fit(cls, image: Size, canvas: Size) -> "CanvasTransform":
        """Map ``image`` onto the whole of ``canvas`` (the identity when equal)."""
        return cls(scale_x=canvas.width / float(image.width),
                   scale_y=canvas.height / float(image.height))

    def image_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        return (x * self.scale_x + self.offset_x, y * self.scale_y + self.offset_y)

    def canvas_to_image(self, x: float, y: float) -> Tuple[float, float]:
        return ((x - self.offset_x) / self.scale_x, (y - self.offset_y) / self.scale_y)


@dataclass
class BBox:
    """Half-open integer rectangle ``[x0, x1) x [y0, y1)``, origin top-left.

    In a result header these are **model-canvas** pixel coordinates.
    """

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return int(self.x1) - int(self.x0)

    @property
    def height(self) -> int:
        return int(self.y1) - int(self.y0)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_list(self) -> List[int]:
        return [int(self.x0), int(self.y0), int(self.x1), int(self.y1)]

    # JSON carries bboxes as 4-element arrays, never as objects.
    to_dict = to_list

    @classmethod
    def from_list(cls, v: Sequence[Any]) -> "BBox":
        if len(v) != 4:
            raise ValueError("bbox must have exactly 4 elements, got %d" % len(v))
        return cls(int(v[0]), int(v[1]), int(v[2]), int(v[3]))

    from_dict = from_list

    def is_valid(self, canvas: Optional[Size] = None) -> bool:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            return False
        if self.x0 < 0 or self.y0 < 0:
            return False
        if canvas is not None and (self.x1 > canvas.width or self.y1 > canvas.height):
            return False
        return True


@dataclass
class Point:
    """A PVS click.  ``label`` is 1 for include, 0 for exclude.

    Coordinates are **uploaded-image** pixels (floats allowed).  Clients never
    convert to canvas space for input; the daemon does that.
    """

    x: float
    y: float
    label: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {"x": float(self.x), "y": float(self.y), "label": int(self.label)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Point":
        return cls(x=float(d["x"]), y=float(d["y"]), label=int(d.get("label", 1)))


# --------------------------------------------------------------------------- #
# runtime.json
# --------------------------------------------------------------------------- #
@dataclass
class RuntimeInfo:
    """Contents of ``runtime.json``.

    The five required fields are frozen for API 1.x.  Optional fields may be
    added; readers must ignore keys they do not know.
    """

    port: int
    token: str
    pid: int
    version: str
    started_at: float
    api_version: str = API_VERSION
    host: str = "127.0.0.1"
    log_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "port": int(self.port),
            "token": self.token,
            "pid": int(self.pid),
            "version": self.version,
            "started_at": float(self.started_at),
            "api_version": self.api_version,
            "host": self.host,
        }
        if self.log_path is not None:
            d["log_path"] = self.log_path
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RuntimeInfo":
        return cls(
            port=int(d["port"]),
            token=str(d["token"]),
            pid=int(d["pid"]),
            version=str(d["version"]),
            started_at=float(d["started_at"]),
            api_version=str(d.get("api_version", API_VERSION)),
            host=str(d.get("host", "127.0.0.1")),
            log_path=d.get("log_path"),
        )

    @property
    def base_url(self) -> str:
        return "http://%s:%d" % (self.host, self.port)

    @property
    def auth_header(self) -> str:
        return "Bearer " + self.token


# --------------------------------------------------------------------------- #
# GET /hello, GET /status
# --------------------------------------------------------------------------- #
@dataclass
class HelloResponse:
    """Cheap version/capability handshake.  Never loads a model."""

    api_version: str
    sam3d_version: str
    engine_mode: str            # "stub" | "torch"
    device: str                 # "cpu" | "cuda" | "cuda:0" | "mps" | "stub"
    dtype: str                  # "float32" | "bfloat16" | "float16" | "none"
    capabilities: List[str] = field(default_factory=list)
    torch_available: bool = False
    weights_available: bool = False
    model_canvas: Size = field(default_factory=lambda: Size(1008, 1008))
    pid: int = 0
    started_at: float = 0.0
    uptime_s: float = 0.0
    limits: Dict[str, Any] = field(default_factory=dict)
    #: :func:`nonce_proof` of the request's ``X-Sam3-Nonce``; absent without one.
    nonce_proof: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "api_version": self.api_version,
            "sam3d_version": self.sam3d_version,
            "engine_mode": self.engine_mode,
            "device": self.device,
            "dtype": self.dtype,
            "capabilities": list(self.capabilities),
            "torch_available": bool(self.torch_available),
            "weights_available": bool(self.weights_available),
            "model_canvas": self.model_canvas.to_dict(),
            "pid": int(self.pid),
            "started_at": float(self.started_at),
            "uptime_s": float(self.uptime_s),
            "limits": dict(self.limits),
        }
        if self.nonce_proof is not None:
            d["nonce_proof"] = self.nonce_proof
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HelloResponse":
        known = {
            "api_version", "sam3d_version", "engine_mode", "device", "dtype",
            "capabilities", "torch_available", "weights_available",
            "model_canvas", "pid", "started_at", "uptime_s", "limits", "nonce_proof",
        }
        canvas = d.get("model_canvas") or {"width": 1008, "height": 1008}
        return cls(
            api_version=str(d["api_version"]),
            sam3d_version=str(d.get("sam3d_version", "")),
            engine_mode=str(d.get("engine_mode", "stub")),
            device=str(d.get("device", "cpu")),
            dtype=str(d.get("dtype", "float32")),
            capabilities=list(d.get("capabilities", [])),
            torch_available=bool(d.get("torch_available", False)),
            weights_available=bool(d.get("weights_available", False)),
            model_canvas=Size.from_dict(canvas),
            pid=int(d.get("pid", 0)),
            started_at=float(d.get("started_at", 0.0)),
            uptime_s=float(d.get("uptime_s", 0.0)),
            limits=dict(d.get("limits", {})),
            nonce_proof=d.get("nonce_proof"),
            extra={k: v for k, v in d.items() if k not in known},
        )

    @property
    def api_major(self) -> int:
        return int(str(self.api_version).split(".", 1)[0])

    def is_compatible_with(self, client_api_version: str = API_VERSION) -> bool:
        """Major-version match is the whole compatibility rule."""
        try:
            return self.api_major == int(str(client_api_version).split(".", 1)[0])
        except (TypeError, ValueError):
            return False


@dataclass
class CacheEntry:
    """One cached image embedding, as reported by GET /status."""

    image_id: str
    width: int
    height: int
    created_at: float
    last_used_at: float
    state: str = JobState.DONE
    bytes_estimate: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "width": int(self.width),
            "height": int(self.height),
            "created_at": float(self.created_at),
            "last_used_at": float(self.last_used_at),
            "state": self.state,
            "bytes_estimate": int(self.bytes_estimate),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CacheEntry":
        return cls(
            image_id=str(d["image_id"]),
            width=int(d["width"]),
            height=int(d["height"]),
            created_at=float(d.get("created_at", 0.0)),
            last_used_at=float(d.get("last_used_at", 0.0)),
            state=str(d.get("state", JobState.DONE)),
            bytes_estimate=int(d.get("bytes_estimate", 0)),
        )


@dataclass
class StatusResponse:
    """Everything the Doctor panel shows.  A superset of /hello."""

    hello: HelloResponse
    images: List[CacheEntry] = field(default_factory=list)
    jobs_running: int = 0
    jobs_queued: int = 0
    jobs_total: int = 0
    cache_limit: int = 3
    idle_ttl_s: float = 1800.0
    idle_seconds: float = 0.0
    parent_pid: Optional[int] = None
    models_loaded: List[str] = field(default_factory=list)
    last_error: Optional[Dict[str, Any]] = None
    paths: Dict[str, str] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.hello.to_dict())
        d.update({
            "images": [e.to_dict() for e in self.images],
            "jobs_running": int(self.jobs_running),
            "jobs_queued": int(self.jobs_queued),
            "jobs_total": int(self.jobs_total),
            "cache_limit": int(self.cache_limit),
            "idle_ttl_s": float(self.idle_ttl_s),
            "idle_seconds": float(self.idle_seconds),
            "parent_pid": self.parent_pid,
            "models_loaded": list(self.models_loaded),
            "last_error": self.last_error,
            "paths": dict(self.paths),
            "memory": dict(self.memory),
        })
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StatusResponse":
        return cls(
            hello=HelloResponse.from_dict(d),
            images=[CacheEntry.from_dict(x) for x in d.get("images", [])],
            jobs_running=int(d.get("jobs_running", 0)),
            jobs_queued=int(d.get("jobs_queued", 0)),
            jobs_total=int(d.get("jobs_total", 0)),
            cache_limit=int(d.get("cache_limit", 3)),
            idle_ttl_s=float(d.get("idle_ttl_s", 1800.0)),
            idle_seconds=float(d.get("idle_seconds", 0.0)),
            parent_pid=d.get("parent_pid"),
            models_loaded=list(d.get("models_loaded", [])),
            last_error=d.get("last_error"),
            paths=dict(d.get("paths", {})),
            memory=dict(d.get("memory", {})),
        )


# --------------------------------------------------------------------------- #
# POST /images and prompt acceptance
# --------------------------------------------------------------------------- #
@dataclass
class ImageAccepted:
    """202 body of POST /images."""

    image_id: str
    job_id: str
    cached: bool
    image: Size
    model_canvas: Size
    canvas_from_image: CanvasTransform
    state: str = JobState.QUEUED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "job_id": self.job_id,
            "cached": bool(self.cached),
            "image": self.image.to_dict(),
            "model_canvas": self.model_canvas.to_dict(),
            "canvas_from_image": self.canvas_from_image.to_dict(),
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ImageAccepted":
        return cls(
            image_id=str(d["image_id"]),
            job_id=str(d["job_id"]),
            cached=bool(d.get("cached", False)),
            image=Size.from_dict(d["image"]),
            model_canvas=Size.from_dict(d["model_canvas"]),
            canvas_from_image=CanvasTransform.from_dict(d["canvas_from_image"]),
            state=str(d.get("state", JobState.QUEUED)),
        )


@dataclass
class JobAccepted:
    """202 body of POST /images/{id}/text and POST /images/{id}/points."""

    job_id: str
    request_id: str
    image_id: str
    engine: str
    state: str = JobState.QUEUED
    superseded_job_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "request_id": self.request_id,
            "image_id": self.image_id,
            "engine": self.engine,
            "state": self.state,
            "superseded_job_ids": list(self.superseded_job_ids),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JobAccepted":
        return cls(
            job_id=str(d["job_id"]),
            request_id=str(d["request_id"]),
            image_id=str(d["image_id"]),
            engine=str(d["engine"]),
            state=str(d.get("state", JobState.QUEUED)),
            superseded_job_ids=list(d.get("superseded_job_ids", [])),
        )


@dataclass
class TextPrompt:
    """Request body of POST /images/{id}/text (PCS)."""

    request_id: str
    text: str
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    max_instances: int = 64
    boxes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "text": self.text,
            "score_threshold": float(self.score_threshold),
            "max_instances": int(self.max_instances),
            "boxes": list(self.boxes),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TextPrompt":
        return cls(
            request_id=str(d["request_id"]),
            text=str(d["text"]),
            score_threshold=float(d.get("score_threshold", DEFAULT_SCORE_THRESHOLD)),
            max_instances=int(d.get("max_instances", 64)),
            boxes=list(d.get("boxes", [])),
        )


@dataclass
class PointPrompt:
    """Request body of POST /images/{id}/points (PVS).

    ``points`` are uploaded-image pixels; ``box`` is ``[x0, y0, x1, y1]`` in the
    same space (floats, half-open).  Either may be empty but not both.
    """

    request_id: str
    points: List[Point] = field(default_factory=list)
    box: Optional[List[float]] = None
    multimask: bool = True
    max_instances: int = 3

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "request_id": self.request_id,
            "points": [p.to_dict() for p in self.points],
            "multimask": bool(self.multimask),
            "max_instances": int(self.max_instances),
        }
        if self.box is not None:
            d["box"] = [float(v) for v in self.box]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PointPrompt":
        box = d.get("box")
        return cls(
            request_id=str(d["request_id"]),
            points=[Point.from_dict(p) for p in d.get("points", [])],
            box=[float(v) for v in box] if box else None,
            multimask=bool(d.get("multimask", True)),
            max_instances=int(d.get("max_instances", 3)),
        )


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
@dataclass
class MaskInstance:
    """One returned mask.

    ``bbox`` is in model-canvas pixels; ``mask_width``/``mask_height`` always
    equal the bbox width/height.  ``blob_offset`` is measured from the start of
    the blob region (``RESULT_PREFIX_SIZE + header_length``), never from the
    start of the body.  ``blob_length == mask_width * mask_height``.
    """

    instance_id: int
    score: float
    bbox: BBox
    mask_width: int
    mask_height: int
    blob_offset: int
    blob_length: int
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": int(self.instance_id),
            "score": float(self.score),
            "label": self.label,
            "bbox": self.bbox.to_list(),
            "mask_width": int(self.mask_width),
            "mask_height": int(self.mask_height),
            "blob_offset": int(self.blob_offset),
            "blob_length": int(self.blob_length),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MaskInstance":
        return cls(
            instance_id=int(d["instance_id"]),
            score=float(d["score"]),
            bbox=BBox.from_list(d["bbox"]),
            mask_width=int(d["mask_width"]),
            mask_height=int(d["mask_height"]),
            blob_offset=int(d["blob_offset"]),
            blob_length=int(d["blob_length"]),
            label=str(d.get("label", "")),
        )

    def validate(self) -> None:
        if self.mask_width != self.bbox.width or self.mask_height != self.bbox.height:
            raise ValueError(
                "instance %d: mask %dx%d does not match bbox %dx%d"
                % (self.instance_id, self.mask_width, self.mask_height,
                   self.bbox.width, self.bbox.height))
        if self.blob_length != self.mask_width * self.mask_height:
            raise ValueError(
                "instance %d: blob_length %d != %d*%d"
                % (self.instance_id, self.blob_length, self.mask_width, self.mask_height))
        if self.blob_offset < 0:
            raise ValueError("instance %d: negative blob_offset" % self.instance_id)

    def mask_bytes(self, blob: bytes) -> bytes:
        """Slice this instance's pixels out of the blob region."""
        end = self.blob_offset + self.blob_length
        if end > len(blob):
            raise ValueError("instance %d: blob slice %d:%d exceeds blob of %d bytes"
                             % (self.instance_id, self.blob_offset, end, len(blob)))
        return blob[self.blob_offset:end]

    def image_rect(self, transform: "CanvasTransform") -> Tuple[float, float, float, float]:
        """Map the canvas crop back to uploaded-image coordinates (floats)."""
        x0, y0 = transform.canvas_to_image(self.bbox.x0, self.bbox.y0)
        x1, y1 = transform.canvas_to_image(self.bbox.x1, self.bbox.y1)
        return (x0, y0, x1, y1)


@dataclass
class ResultHeader:
    """JSON header of a binary result frame, and the ``result`` object embedded
    in the JSON form of a finished job."""

    job_id: str
    request_id: str
    image_id: str
    engine: str
    image: Size
    model_canvas: Size
    canvas_from_image: CanvasTransform
    instances: List[MaskInstance] = field(default_factory=list)
    prompt: Dict[str, Any] = field(default_factory=dict)
    blob_length: int = 0
    elapsed_ms: float = 0.0
    api_version: str = API_VERSION
    mask_encoding: str = MASK_ENCODING_U8_SOFT
    state: str = JobState.DONE
    truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_version": self.api_version,
            "job_id": self.job_id,
            "request_id": self.request_id,
            "image_id": self.image_id,
            "engine": self.engine,
            "state": self.state,
            "prompt": dict(self.prompt),
            "image": self.image.to_dict(),
            "model_canvas": self.model_canvas.to_dict(),
            "canvas_from_image": self.canvas_from_image.to_dict(),
            "mask_encoding": self.mask_encoding,
            "elapsed_ms": float(self.elapsed_ms),
            "truncated": bool(self.truncated),
            "instances": [i.to_dict() for i in self.instances],
            "blob_length": int(self.blob_length),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResultHeader":
        return cls(
            job_id=str(d["job_id"]),
            request_id=str(d.get("request_id", "")),
            image_id=str(d["image_id"]),
            engine=str(d["engine"]),
            image=Size.from_dict(d["image"]),
            model_canvas=Size.from_dict(d["model_canvas"]),
            canvas_from_image=CanvasTransform.from_dict(d["canvas_from_image"]),
            instances=[MaskInstance.from_dict(x) for x in d.get("instances", [])],
            prompt=dict(d.get("prompt", {})),
            blob_length=int(d.get("blob_length", 0)),
            elapsed_ms=float(d.get("elapsed_ms", 0.0)),
            api_version=str(d.get("api_version", API_VERSION)),
            mask_encoding=str(d.get("mask_encoding", MASK_ENCODING_U8_SOFT)),
            state=str(d.get("state", JobState.DONE)),
            truncated=bool(d.get("truncated", False)),
        )

    def validate(self) -> None:
        total = 0
        for inst in self.instances:
            inst.validate()
            if inst.blob_offset != total:
                raise ValueError(
                    "instance %d: blob_offset %d, expected %d (masks are tightly "
                    "packed in instance order)" % (inst.instance_id, inst.blob_offset, total))
            if not inst.bbox.is_valid(self.model_canvas):
                raise ValueError("instance %d: bbox %s outside canvas %s"
                                 % (inst.instance_id, inst.bbox.to_list(),
                                    self.model_canvas.to_dict()))
            total += inst.blob_length
        if self.blob_length != total:
            raise ValueError("blob_length %d != sum of instance lengths %d"
                             % (self.blob_length, total))


@dataclass
class JobStatus:
    """JSON body of GET /jobs/{id} in every state.

    ``result`` is present only when ``state == "done"`` and the client asked for
    the JSON form (``?meta=1``); it holds the same header the binary frame
    carries, so ``blob_offset``/``blob_length`` are meaningful only against a
    subsequent binary fetch.
    """

    job_id: str
    state: str
    engine: str
    image_id: str
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
    error: Optional[ErrorInfo] = None
    result: Optional[ResultHeader] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "job_id": self.job_id,
            "state": self.state,
            "engine": self.engine,
            "image_id": self.image_id,
            "request_id": self.request_id,
            "progress": float(self.progress),
            "stage": self.stage,
            "created_at": float(self.created_at),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_ms": float(self.elapsed_ms),
            "queue_position": self.queue_position,
            "superseded_by": self.superseded_by,
            "masks_available": bool(self.masks_available),
        }
        d["error"] = self.error.to_dict() if self.error is not None else None
        if self.result is not None:
            d["result"] = self.result.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JobStatus":
        err = d.get("error")
        res = d.get("result")
        return cls(
            job_id=str(d["job_id"]),
            state=str(d["state"]),
            engine=str(d.get("engine", "")),
            image_id=str(d.get("image_id", "")),
            request_id=str(d.get("request_id", "")),
            progress=float(d.get("progress", 0.0)),
            stage=str(d.get("stage", "")),
            created_at=float(d.get("created_at", 0.0)),
            started_at=d.get("started_at"),
            finished_at=d.get("finished_at"),
            elapsed_ms=float(d.get("elapsed_ms", 0.0)),
            queue_position=d.get("queue_position"),
            superseded_by=d.get("superseded_by"),
            masks_available=bool(d.get("masks_available", False)),
            error=ErrorInfo.from_dict(err) if err else None,
            result=ResultHeader.from_dict(res) if res else None,
        )

    @property
    def is_terminal(self) -> bool:
        return JobState.is_terminal(self.state)


# --------------------------------------------------------------------------- #
# binary result frame
# --------------------------------------------------------------------------- #
def pack_result(header: "ResultHeader", blobs: Sequence[bytes]) -> bytes:
    """Serialise a finished job into the ``application/vnd.sam3.result+binary``
    body.

    ``blobs[i]`` belongs to ``header.instances[i]``.  ``blob_offset``,
    ``blob_length`` and ``header.blob_length`` are (re)computed here so a caller
    cannot get them wrong; the instances are packed tightly, in order, with no
    alignment padding.
    """
    import json as _json

    if len(blobs) != len(header.instances):
        raise ValueError("got %d blobs for %d instances" % (len(blobs), len(header.instances)))
    offset = 0
    for inst, blob in zip(header.instances, blobs):
        expected = inst.mask_width * inst.mask_height
        if len(blob) != expected:
            raise ValueError("instance %d: %d mask bytes, expected %d (%dx%d)"
                             % (inst.instance_id, len(blob), expected,
                                inst.mask_width, inst.mask_height))
        inst.blob_offset = offset
        inst.blob_length = expected
        offset += expected
    header.blob_length = offset
    header.validate()

    head = _json.dumps(header.to_dict(), separators=(",", ":")).encode("utf-8")
    out = bytearray()
    out += RESULT_MAGIC
    out += _U32.pack(len(head))
    out += head
    for blob in blobs:
        out += blob
    return bytes(out)


def unpack_result(payload: bytes) -> Tuple["ResultHeader", bytes]:
    """Inverse of :func:`pack_result`.

    Returns ``(header, blob_region)``.  Slice a mask with
    ``instance.mask_bytes(blob_region)``.
    """
    import json as _json

    if len(payload) < RESULT_PREFIX_SIZE:
        raise ValueError("result frame truncated: %d bytes" % len(payload))
    if payload[:len(RESULT_MAGIC)] != RESULT_MAGIC:
        raise ValueError("bad result magic %r" % payload[:len(RESULT_MAGIC)])
    (head_len,) = _U32.unpack_from(payload, len(RESULT_MAGIC))
    head_end = RESULT_PREFIX_SIZE + head_len
    if len(payload) < head_end:
        raise ValueError("result header truncated: need %d bytes, have %d"
                         % (head_end, len(payload)))
    header = ResultHeader.from_dict(_json.loads(payload[RESULT_PREFIX_SIZE:head_end].decode("utf-8")))
    blob = payload[head_end:]
    if len(blob) != header.blob_length:
        raise ValueError("blob region is %d bytes, header says %d"
                         % (len(blob), header.blob_length))
    header.validate()
    return header, blob
