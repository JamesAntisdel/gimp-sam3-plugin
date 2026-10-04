"""The engine interface the sam3gimpd server codes against.

An *engine* is the only thing in the daemon that knows how a mask is produced.
The server (``server.py``, ``jobs.py``, ``session.py``) knows nothing about
torch, transformers, SAM 3 or blob geometry: it hands an engine pixels, gets an
:class:`EncodedImage` back, hands it prompts, and gets a :class:`PromptResult`
which it turns into the binary frame of ``API.md`` §8.

Three implementations exist:

``stub.StubEngine``
    Synthesises deterministic soft blobs.  Imports **no** torch, ever.  This is
    a product feature (``sam3gimpd serve --stub``, ``API.md`` §14), not a fixture.
``pcs.PcsEngine`` / ``pvs.PvsEngine``
    The real halves: ``Sam3Model``/``Sam3Processor`` for text, and
    ``Sam3TrackerModel``/``Sam3TrackerProcessor`` for points.  Both are wrapped
    by ``engines.TorchEngine``, which is what the server actually holds.

**This module must import with no torch, no transformers and no numpy.**  Every
heavy import in the package is inside a function.

--------------------------------------------------------------------------
Contract summary for the server
--------------------------------------------------------------------------

``engine.describe() -> EngineInfo``
    Cheap.  Never loads a model and never waits for one to load.  Feeds
    ``GET /hello`` and ``GET /status``.  ``EngineInfo.capabilities`` covers
    only what the *engine* provides (``"pcs"``, ``"pvs"``, maybe
    ``"exemplar_boxes"``); the server adds any capabilities of its own.

``engine.canvas_for(image) -> (Size, CanvasTransform)``
    Cheap and pure.  The server needs it to answer ``POST /images`` with
    ``model_canvas`` and ``canvas_from_image`` *before* the encode job runs.

``engine.encode_image(image_data, progress) -> EncodedImage``
    The expensive step.  Runs on the single inference worker as the ``encode``
    job.  Reports progress in the ``encoding`` band (0.05 -> 0.60).

``engine.prompt_text(encoded, TextPrompt, progress) -> PromptResult``
``engine.prompt_points(encoded, PointPrompt, progress) -> PromptResult``
    Cheap, because they reuse ``encoded``.  Report progress through the
    ``prompting`` -> ``decoding`` -> ``packing`` bands.  ``prompt_points`` may
    lazily encode the image for the PVS half on its first call (``DESIGN.md``
    §7: the tracker is only loaded on the first click), in which case it reports
    the ``encoding`` band first.

``engine.unload(which=None) -> List[str]``
    Free model weights.  Returns the engine names actually unloaded.  Safe to
    call at any time from the idle-TTL timer, and safe to call twice.

``engine.close()``
    Final teardown, idempotent.

Errors: engines raise :class:`~sam3gimpd.types.ApiError` with a code from
``ErrorCode``.  ``engine_unavailable`` when torch/transformers/weights are
missing, ``model_load_failed`` when the checkpoint will not load, and
``inference_failed`` when a forward pass raises.  The server turns a raised
``ApiError`` inside a job into ``state: "failed"`` with that envelope
(``API.md`` §4), not into an HTTP error.

Coordinate spaces (``API.md`` §5), because this is the classic integration bug:

* everything in a :class:`~sam3gimpd.types.TextPrompt` / :class:`PointPrompt` that
  a client sent -- points, boxes -- is in **uploaded-image** pixels, which is
  the space the processors take them in;
* every ``bbox`` an engine puts in a :class:`RawInstance` is in **model-canvas**
  pixels, half-open, integer, inside the canvas.
"""

from __future__ import annotations

import abc
import math
import struct
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..types import (
    API_VERSION,
    ApiError,
    BBox,
    CanvasTransform,
    ErrorCode,
    JobState,
    Limits,
    MaskInstance,
    PointPrompt,
    PromptKind,
    ResultHeader,
    Size,
    TextPrompt,
    pack_result,
)

__all__ = [
    "ENGINE_LOGGER",
    "hf_instance_scores",
    "hf_keep_indices",
    "forward_kwargs_for",
    "DEFAULT_CANVAS_SIDE",
    "DEFAULT_CROP_EPSILON",
    "DEFAULT_CROP_PAD",
    "Stage",
    "STAGE_PROGRESS",
    "ProgressFn",
    "ProgressReporter",
    "ImageData",
    "EncodedImage",
    "RawInstance",
    "CanvasMask",
    "PromptResult",
    "EngineInfo",
    "BaseEngine",
    "default_canvas_for",
    "build_instances",
    "finalize_instances",
    "rgb_array",
    "logits_to_canvas_masks",
    "masks_to_canvas_masks",
    "call_with_supported_kwargs",
    "accepts_kwarg",
    "move_inputs",
    "require_finite",
    "clamp_score",
    "engine_unavailable",
    "model_load_failed",
    "inference_failed",
]

#: SAM 3 works at 1008px.  This is the *nominal* working resolution reported as
#: ``model_canvas`` in ``/hello``; the canvas masks are returned in is the
#: uploaded image itself (:func:`default_canvas_for`), and clients always use
#: the reported transform, never this constant.
DEFAULT_CANVAS_SIDE = 1008

#: Logger the torch engines report through when none is injected.  Their
#: warnings -- embedding reuse that failed, masks that came back binarised --
#: are the only trace of a silent quality or speed regression.
ENGINE_LOGGER = "sam3gimpd.engines"

#: Soft-mask value below which a pixel is considered outside the crop.  The
#: crop is "tight but not guaranteed minimal" (``API.md`` §8.3), so a small
#: epsilon plus a pad keeps the anti-aliased falloff intact.
DEFAULT_CROP_EPSILON = 8

#: Extra pixels kept around the tight bounds so soft edges are not clipped.
DEFAULT_CROP_PAD = 2


class Stage:
    """Stage names from ``API.md`` §11.  Free-form on the wire but stable."""

    QUEUED = "queued"
    #: Reported before the first encode in a process: a cold load of the 0.9B
    #: checkpoint plus CUDA init can take a minute, and a client that only ever
    #: saw "encoding 10%" for that long reads it as a freeze.
    LOADING_MODEL = "loading model"
    ENCODING = "encoding"
    PROMPTING = "prompting"
    DECODING = "decoding"
    PACKING = "packing"
    DONE = "done"
    FAILED = "failed"

    ALL = (QUEUED, LOADING_MODEL, ENCODING, PROMPTING, DECODING, PACKING, DONE, FAILED)


#: ``stage -> (progress_low, progress_high)``, the reference milestones of
#: ``API.md`` §11.  Engines report absolute progress from this table so that a
#: progress bar behaves identically across the stub and the real engines.
STAGE_PROGRESS: Dict[str, Tuple[float, float]] = {
    Stage.QUEUED: (0.0, 0.05),
    Stage.ENCODING: (0.05, 0.60),
    Stage.PROMPTING: (0.60, 0.80),
    Stage.DECODING: (0.80, 0.95),
    Stage.PACKING: (0.95, 1.0),
    Stage.DONE: (1.0, 1.0),
    Stage.FAILED: (1.0, 1.0),
}

#: ``progress(fraction, stage)``.  Never raises; engines call it freely.
ProgressFn = Callable[[float, str], None]


class ProgressReporter:
    """Clamping, monotonic wrapper around a raw progress callback.

    ``API.md`` §11 requires progress to be in ``[0, 1]`` and non-decreasing
    within a job.  Engines report *stage-relative* fractions and this class maps
    them onto the absolute bands in :data:`STAGE_PROGRESS`, so an engine never
    has to remember that "encoding" means 0.05..0.60.

    A ``None`` callback is legal and makes every call a no-op, which is what
    lets engines be driven straight from a test with no job machinery.
    """

    __slots__ = ("_fn", "_last")

    def __init__(self, fn: Optional[ProgressFn] = None) -> None:
        self._fn = fn
        self._last = 0.0

    @property
    def value(self) -> float:
        return self._last

    def stage(self, stage: str, t: float = 0.0) -> None:
        """Report ``t`` in ``[0, 1]`` *within* ``stage``'s band."""
        lo, hi = STAGE_PROGRESS.get(stage, (0.0, 1.0))
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        self.absolute(lo + (hi - lo) * t, stage)

    def absolute(self, fraction: float, stage: str) -> None:
        """Report an absolute progress value, clamped and made monotonic."""
        if fraction < 0.0:
            fraction = 0.0
        elif fraction > 1.0:
            fraction = 1.0
        if fraction < self._last:
            fraction = self._last
        self._last = fraction
        if self._fn is not None:
            self._fn(fraction, stage)


# --------------------------------------------------------------------------- #
# data carried between the server and an engine
# --------------------------------------------------------------------------- #
@dataclass
class ImageData:
    """Raw pixels as they arrived on ``POST /images`` (``API.md`` §7).

    ``pixels`` is ``width * height * 3`` bytes, R-G-B, top-to-bottom, stride
    ``width * 3``, no padding and no alpha.  ``image_id`` is the server's
    content hash; engines treat it as opaque but *do* use it as a determinism
    seed (the stub) and as a cache key label.
    """

    image_id: str
    width: int
    height: int
    pixels: bytes

    @property
    def size(self) -> Size:
        return Size(self.width, self.height)

    @property
    def expected_bytes(self) -> int:
        return int(self.width) * int(self.height) * 3

    def validate(self) -> None:
        if self.width < Limits.MIN_IMAGE_SIDE or self.width > Limits.MAX_IMAGE_SIDE:
            raise ApiError(ErrorCode.BAD_DIMENSIONS,
                           "width %d outside [%d, %d]"
                           % (self.width, Limits.MIN_IMAGE_SIDE, Limits.MAX_IMAGE_SIDE),
                           {"width": self.width})
        if self.height < Limits.MIN_IMAGE_SIDE or self.height > Limits.MAX_IMAGE_SIDE:
            raise ApiError(ErrorCode.BAD_DIMENSIONS,
                           "height %d outside [%d, %d]"
                           % (self.height, Limits.MIN_IMAGE_SIDE, Limits.MAX_IMAGE_SIDE),
                           {"height": self.height})
        if len(self.pixels) != self.expected_bytes:
            raise ApiError(ErrorCode.PAYLOAD_SIZE_MISMATCH,
                           "body is %d bytes, expected %d"
                           % (len(self.pixels), self.expected_bytes),
                           {"expected": self.expected_bytes, "got": len(self.pixels)})


@dataclass
class EncodedImage:
    """One entry of the embedding cache.

    The server owns the LRU (``session.py``); the engine owns ``parts``.

    ``parts`` maps an engine half (``"pcs"``, ``"pvs"``) to whatever that half
    needs to prompt without re-encoding -- a tensor, a tuple of tensors, or for
    the stub a plain dict.  It starts with only the half that ``encode_image``
    filled; ``prompt_points`` adds ``"pvs"`` on the first click, which is how
    ``DESIGN.md`` §7's "the tracker is only loaded on the first click-refine"
    is implemented.

    ``pixels`` is deliberately retained for exactly that reason: the second half
    needs the image again, and re-uploading it would defeat the cache.  Three
    cached 1008x1008 images cost ~9 MB of RAM, which is noise next to the model.
    """

    image_id: str
    image: Size
    canvas: Size
    transform: CanvasTransform
    pixels: bytes
    parts: Dict[str, Any] = field(default_factory=dict)
    #: Rough resident size for ``/status``'s ``bytes_estimate``: the sum of
    #: what :meth:`put` was told about the payloads currently held.
    bytes_estimate: int = 0
    _part_bytes: Dict[str, int] = field(default_factory=dict, repr=False)

    def has(self, part: str) -> bool:
        return part in self.parts

    def put(self, part: str, payload: Any, bytes_estimate: int = 0) -> None:
        """Store (or replace) one half's payload."""
        self.bytes_estimate -= self._part_bytes.pop(part, 0)
        self.parts[part] = payload
        self._part_bytes[part] = int(bytes_estimate)
        self.bytes_estimate += int(bytes_estimate)

    def get(self, part: str) -> Any:
        return self.parts.get(part)

    def drop(self, part: Optional[str] = None) -> None:
        """Release payloads.  Called on cache eviction and on engine unload."""
        names = list(self.parts) if part is None else [part]
        for name in names:
            self.parts.pop(name, None)
            self.bytes_estimate -= self._part_bytes.pop(name, 0)

    def describe(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "width": self.image.width,
            "height": self.image.height,
            "canvas": self.canvas.to_dict(),
            "parts": sorted(self.parts.keys()),
            "bytes_estimate": int(self.bytes_estimate) + len(self.pixels),
        }


@dataclass
class RawInstance:
    """One mask on its way to the wire.

    ``bbox`` is **model-canvas** pixels, half-open and integer.  ``mask`` is
    exactly ``bbox.width * bbox.height`` bytes: soft uint8, row-major,
    top-to-bottom, ``255 * sigmoid(logit)`` so that 128 is the model's own
    binarisation point (``API.md`` §8.3).
    """

    score: float
    bbox: BBox
    mask: bytes
    label: str = ""

    def validate(self, canvas: Optional[Size] = None) -> None:
        if not self.bbox.is_valid(canvas):
            raise ValueError("bbox %s is not valid for canvas %s"
                             % (self.bbox.to_list(), canvas.to_dict() if canvas else None))
        expected = self.bbox.width * self.bbox.height
        if len(self.mask) != expected:
            raise ValueError("mask is %d bytes, bbox %s wants %d"
                             % (len(self.mask), self.bbox.to_list(), expected))
        if not (0.0 <= float(self.score) <= 1.0):
            raise ValueError("score %r outside [0, 1]" % (self.score,))

    def to_mask_instance(self, instance_id: int) -> MaskInstance:
        """Wire shape.  ``blob_offset``/``blob_length`` are recomputed by
        :func:`~sam3gimpd.types.pack_result`, so the placeholders here are safe."""
        return MaskInstance(
            instance_id=int(instance_id),
            score=float(self.score),
            bbox=self.bbox,
            mask_width=self.bbox.width,
            mask_height=self.bbox.height,
            blob_offset=0,
            blob_length=self.bbox.width * self.bbox.height,
            label=self.label,
        )


@dataclass
class CanvasMask:
    """A *full-canvas* soft mask on its way into :func:`build_instances`.

    ``mask`` is either ``canvas.width * canvas.height`` bytes of uint8, or a 2-D
    numpy ``uint8`` array of that shape (the real engines produce the latter;
    numpy is only ever touched inside :func:`build_instances`).
    """

    score: float
    mask: Any
    label: str = ""


@dataclass
class PromptResult:
    """Everything the server needs to build the binary frame of ``API.md`` §8.

    The engine has already done the hard parts -- canvas geometry, cropping,
    quantisation to soft uint8, score ordering, truncation.  The server adds
    only the job/request/image identifiers it owns.
    """

    engine: str
    instances: List[RawInstance]
    image: Size
    canvas: Size
    transform: CanvasTransform
    prompt: Dict[str, Any] = field(default_factory=dict)
    truncated: bool = False
    elapsed_ms: float = 0.0

    def validate(self) -> None:
        for inst in self.instances:
            inst.validate(self.canvas)
        scores = [i.score for i in self.instances]
        if scores != sorted(scores, reverse=True):
            raise ValueError("instances must be sorted by descending score, got %r" % (scores,))

    def header_and_blobs(self, job_id: str, request_id: str,
                         image_id: str) -> Tuple[ResultHeader, List[bytes]]:
        """Split into the JSON header and the per-instance mask bytes."""
        header = ResultHeader(
            job_id=job_id,
            request_id=request_id,
            image_id=image_id,
            engine=self.engine,
            image=self.image,
            model_canvas=self.canvas,
            canvas_from_image=self.transform,
            instances=[inst.to_mask_instance(i) for i, inst in enumerate(self.instances)],
            prompt=dict(self.prompt),
            elapsed_ms=float(self.elapsed_ms),
            api_version=API_VERSION,
            state=JobState.DONE,
            truncated=bool(self.truncated),
        )
        return header, [inst.mask for inst in self.instances]

    def to_frame(self, job_id: str, request_id: str, image_id: str,
                 elapsed_ms: Optional[float] = None) -> bytes:
        """Serialise straight to the ``application/vnd.sam3.result+binary`` body.

        Offsets, lengths and every §16 invariant are enforced by
        :func:`~sam3gimpd.types.pack_result`, so a server that calls this cannot
        emit a malformed frame.
        """
        if elapsed_ms is not None:
            self.elapsed_ms = float(elapsed_ms)
        header, blobs = self.header_and_blobs(job_id, request_id, image_id)
        return pack_result(header, blobs)


@dataclass
class EngineInfo:
    """Cheap self-description.  Never triggers a model load or a network call."""

    mode: str                       # "stub" | "torch"
    device: str                     # "stub" | "cpu" | "cuda" | "cuda:0" | "mps"
    dtype: str                      # "float32" | "bfloat16" | "float16" | "none"
    capabilities: List[str] = field(default_factory=list)
    torch_available: bool = False
    weights_available: bool = False
    model_canvas: Size = field(default_factory=lambda: Size(DEFAULT_CANVAS_SIDE,
                                                            DEFAULT_CANVAS_SIDE))
    models_loaded: List[str] = field(default_factory=list)
    #: Free-form diagnostics for ``/status``: torch version, cuda capability,
    #: why the engine is unavailable, whether fp32 fallback fired, and so on.
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_mode": self.mode,
            "device": self.device,
            "dtype": self.dtype,
            "capabilities": list(self.capabilities),
            "torch_available": bool(self.torch_available),
            "weights_available": bool(self.weights_available),
            "model_canvas": self.model_canvas.to_dict(),
            "models_loaded": list(self.models_loaded),
            "detail": dict(self.detail),
        }


# --------------------------------------------------------------------------- #
# helpers shared by every engine
# --------------------------------------------------------------------------- #
def default_canvas_for(image: Size, side: int = DEFAULT_CANVAS_SIDE
                       ) -> Tuple[Size, CanvasTransform]:
    """Masks are returned in uploaded-image space, so the canvas is the image.

    SAM 3 resizes preserving aspect ratio and pads, so squashing the image onto
    a ``side`` square -- a 900x600 upload at scale (1.12, 1.68) -- would bring
    every mask back stretched, beside its object rather than on it (square
    images hide this, because there the two scales agree).  Rather than model
    the letterbox, the engines ask ``post_process_instance_segmentation`` for
    ``target_sizes`` equal to the uploaded image -- exactly the reference
    example's ``original_sizes`` -- so it undoes resize and padding itself.
    The canvas is then the image and this transform is the identity.

    ``side`` is kept in the signature for callers reporting a nominal working
    resolution; it does not determine the geometry.
    """
    canvas = Size(int(image.width), int(image.height))
    return canvas, CanvasTransform.fit(image, canvas)


def engine_unavailable(message: str, detail: Optional[Dict[str, Any]] = None) -> ApiError:
    return ApiError(ErrorCode.ENGINE_UNAVAILABLE, message, detail or {})


def model_load_failed(message: str, detail: Optional[Dict[str, Any]] = None) -> ApiError:
    return ApiError(ErrorCode.MODEL_LOAD_FAILED, message, detail or {})


def inference_failed(message: str, detail: Optional[Dict[str, Any]] = None) -> ApiError:
    return ApiError(ErrorCode.INFERENCE_FAILED, message, detail or {})


# translation table mapping "value >= epsilon" to 1 and everything else to 0.
def _threshold_table(epsilon: int) -> bytes:
    return bytes(1 if v >= epsilon else 0 for v in range(256))


def _bbox_py(mask: bytes, canvas: Size, epsilon: int) -> Optional[Tuple[int, int, int, int]]:
    """Tight bounds of ``mask >= epsilon``, in pure Python but at C speed.

    ``bytes.translate`` + ``find``/``rfind`` do the per-row scan inside CPython,
    so a 1008x1008 canvas costs ~1008 cheap C calls rather than a million
    interpreted comparisons.
    """
    w, h = int(canvas.width), int(canvas.height)
    table = _threshold_table(epsilon)
    view = memoryview(mask)
    y0 = -1
    y1 = -1
    x0 = w
    x1 = -1
    for y in range(h):
        row = bytes(view[y * w:(y + 1) * w]).translate(table)
        left = row.find(b"\x01")
        if left < 0:
            continue
        right = row.rfind(b"\x01")
        if y0 < 0:
            y0 = y
        y1 = y
        if left < x0:
            x0 = left
        if right > x1:
            x1 = right
    if y1 < 0:
        return None
    return (x0, y0, x1 + 1, y1 + 1)


def _crop_py(mask: bytes, canvas: Size, bbox: BBox) -> bytes:
    w = int(canvas.width)
    view = memoryview(mask)
    out = bytearray()
    for y in range(bbox.y0, bbox.y1):
        start = y * w + bbox.x0
        out += view[start:start + bbox.width]
    return bytes(out)


def _pad_clip(bounds: Tuple[int, int, int, int], canvas: Size, pad: int) -> BBox:
    x0, y0, x1, y1 = bounds
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(int(canvas.width), x1 + pad)
    y1 = min(int(canvas.height), y1 + pad)
    if x1 <= x0:
        x1 = min(int(canvas.width), x0 + 1)
        x0 = x1 - 1
    if y1 <= y0:
        y1 = min(int(canvas.height), y0 + 1)
        y0 = y1 - 1
    return BBox(x0, y0, x1, y1)


def build_instances(canvas: Size,
                    masks: Sequence[CanvasMask],
                    *,
                    max_instances: Optional[int] = None,
                    crop_epsilon: int = DEFAULT_CROP_EPSILON,
                    pad: int = DEFAULT_CROP_PAD,
                    score_threshold: float = 0.0,
                    ) -> Tuple[List[RawInstance], bool]:
    """Turn full-canvas soft masks into cropped :class:`RawInstance` objects.

    This is where every §16 invariant is established exactly once, for both the
    stub and the real engines:

    * instances are sorted by **descending score**;
    * ``max_instances`` clips the list and the second return value says whether
      anything was dropped (``truncated``);
    * the bbox is the tight bounds of ``mask >= crop_epsilon``, padded by ``pad``
      and clipped to the canvas, so ``mask_width == bbox.width``,
      ``mask_height == bbox.height`` and ``blob_length == w * h`` hold by
      construction;
    * a mask with no pixel at or above ``crop_epsilon`` is dropped.  Nothing
      of it could be selected at any threshold a client can set, so it would
      be an entry in the instance list that does nothing -- and an instance
      with a zero-area bbox would be rejected by ``ResultHeader.validate``.

    numpy is used when it is importable (the real engines have it) and skipped
    entirely otherwise (the stub path), which keeps ``--stub`` numpy-free.
    """
    try:
        import numpy as _np  # noqa: PLC0415  (deliberately lazy, optional)
    except Exception:
        _np = None

    kept = [m for m in masks if float(m.score) >= float(score_threshold)]
    kept.sort(key=lambda m: float(m.score), reverse=True)
    truncated = False
    if max_instances is not None and len(kept) > int(max_instances):
        kept = kept[:int(max_instances)]
        truncated = True

    out: List[RawInstance] = []
    for cm in kept:
        mask = cm.mask
        bounds: Optional[Tuple[int, int, int, int]] = None
        arr = None
        if _np is not None and isinstance(mask, _np.ndarray):
            arr = _np.ascontiguousarray(mask.reshape(int(canvas.height), int(canvas.width)),
                                        dtype=_np.uint8)
            hits = arr >= crop_epsilon
            rows = _np.flatnonzero(hits.any(axis=1))
            cols = _np.flatnonzero(hits.any(axis=0))
            if rows.size and cols.size:
                bounds = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
        else:
            if not isinstance(mask, (bytes, bytearray, memoryview)):
                raise TypeError("mask must be bytes or a numpy array, got %r" % type(mask))
            mask = bytes(mask)
            if len(mask) != canvas.area:
                raise ValueError("canvas mask is %d bytes, canvas %s wants %d"
                                 % (len(mask), canvas.to_dict(), canvas.area))
            bounds = _bbox_py(mask, canvas, crop_epsilon)

        if bounds is None:
            continue
        bbox = _pad_clip(bounds, canvas, pad)
        if arr is not None:
            blob = arr[bbox.y0:bbox.y1, bbox.x0:bbox.x1].tobytes()
        else:
            blob = _crop_py(mask, canvas, bbox)

        inst = RawInstance(score=float(cm.score), bbox=bbox, mask=blob, label=cm.label)
        inst.validate(canvas)
        out.append(inst)
    return out, truncated


# --------------------------------------------------------------------------- #
# the interface itself
# --------------------------------------------------------------------------- #
class BaseEngine(abc.ABC):
    """Abstract engine.  See the module docstring for the full contract."""

    #: ``"stub"`` or ``"torch"``; surfaces as ``engine_mode`` in ``/hello``.
    MODE = "base"

    # -- introspection ----------------------------------------------------- #
    @abc.abstractmethod
    def describe(self) -> EngineInfo:
        """Cheap capability report.  MUST NOT load a model or touch the network."""

    def canvas_for(self, image: Size) -> Tuple[Size, CanvasTransform]:
        """Canvas size and image->canvas transform for an image of this size.

        Cheap and pure: the server answers ``POST /images`` with it before the
        encode job has run.  The default is the reference policy: the canvas
        is the uploaded image and the transform is the identity.
        """
        return default_canvas_for(image, self.canvas_side)

    @property
    def canvas_side(self) -> int:
        return DEFAULT_CANVAS_SIDE

    # -- the work ---------------------------------------------------------- #
    @abc.abstractmethod
    def encode_image(self, image: ImageData,
                     progress: Optional[ProgressFn] = None) -> EncodedImage:
        """Run the vision encoder and return a cacheable :class:`EncodedImage`.

        Progress is reported in the ``encoding`` band.  This is the only slow
        call in the daemon; everything downstream reuses its result.
        """

    @abc.abstractmethod
    def prompt_text(self, encoded: EncodedImage, prompt: TextPrompt,
                    progress: Optional[ProgressFn] = None) -> PromptResult:
        """PCS: a noun phrase -> every matching instance, scores descending.

        Returns **all** candidates at or above ``prompt.score_threshold`` so the
        client's score slider filters locally with no round trip
        (``API.md`` §6.3).
        """

    @abc.abstractmethod
    def prompt_points(self, encoded: EncodedImage, prompt: PointPrompt,
                      progress: Optional[ProgressFn] = None) -> PromptResult:
        """PVS: points and/or a box -> one instance, best candidate first.

        ``prompt.points`` and ``prompt.box`` are in **uploaded-image** pixels.
        May lazily encode the image for the tracker half on its first call.
        """

    # -- lifecycle --------------------------------------------------------- #
    def unload(self, which: Optional[str] = None) -> List[str]:
        """Release model weights.  Returns the halves actually unloaded."""
        return []

    def close(self) -> None:
        """Final teardown.  Idempotent; the server calls it on shutdown."""
        self.unload()

    # -- convenience ------------------------------------------------------- #
    def prompt(self, encoded: EncodedImage, prompt: Any,
               progress: Optional[ProgressFn] = None) -> PromptResult:
        """Dispatch on the prompt type, so a caller with a heterogeneous queue
        does not have to switch."""
        if isinstance(prompt, TextPrompt):
            return self.prompt_text(encoded, prompt, progress)
        if isinstance(prompt, PointPrompt):
            return self.prompt_points(encoded, prompt, progress)
        raise TypeError("unsupported prompt type %r" % type(prompt))

    @staticmethod
    def text_prompt_dict(prompt: TextPrompt, ignored: Optional[Sequence[str]] = None
                         ) -> Dict[str, Any]:
        """The ``prompt`` object of a result header for a text prompt."""
        d: Dict[str, Any] = {
            "kind": PromptKind.TEXT,
            "text": prompt.text,
            "score_threshold": float(prompt.score_threshold),
        }
        if ignored:
            d["ignored"] = list(ignored)
        return d

    @staticmethod
    def point_prompt_dict(prompt: PointPrompt, ignored: Optional[Sequence[str]] = None
                          ) -> Dict[str, Any]:
        """The ``prompt`` object of a result header for a point prompt."""
        d: Dict[str, Any] = {
            "kind": PromptKind.POINTS,
            "points": [p.to_dict() for p in prompt.points],
            "multimask": bool(prompt.multimask),
        }
        if prompt.box is not None:
            d["box"] = [float(v) for v in prompt.box]
        if ignored:
            d["ignored"] = list(ignored)
        return d

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<%s mode=%s>" % (type(self).__name__, self.MODE)


# --------------------------------------------------------------------------- #
# helpers for the real (torch) engines
#
# Everything below imports numpy/torch *inside* the function.  ``stub.py`` never
# reaches this section, which is what keeps ``--stub`` free of both.
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# HF post-processing, mirrored -- pure Python, torch-free, unit-tested
# --------------------------------------------------------------------------- #
def _sigmoid(x: float) -> float:
    x = float(x)
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def hf_instance_scores(pred_logits: Sequence[float],
                       presence_logit: Optional[float] = None) -> List[float]:
    """The score ``Sam3ImageProcessor.post_process_instance_segmentation`` uses.

    Verbatim from transformers 5.16.1::

        batch_scores = pred_logits.sigmoid()
        if presence_logits is not None:
            batch_scores = batch_scores * presence_logits.sigmoid()

    The presence head matters in practice: it is a per-image "is this concept
    here at all" gate, and a low presence scales *every* instance down.  That is
    what makes a phrase the text encoder barely knows return nothing even at a
    floor of 0.02 -- the model is saying "absent", not "found but weak".
    """
    p = _sigmoid(presence_logit) if presence_logit is not None else 1.0
    return [_sigmoid(v) * p for v in pred_logits]


def _as_dtype(value: float, dtype: Optional[str]) -> float:
    """``value`` rounded the way torch rounds a Python scalar into ``dtype``.

    A comparison between a tensor and a Python float runs in the *tensor's*
    dtype: torch narrows the scalar to float32 and from there to the half
    format, rounding to nearest-even at each step.  ``None`` and any other
    name leave ``value`` alone.
    """
    name = (dtype or "").rsplit(".", 1)[-1].lower()
    if name not in ("float32", "float", "float16", "half", "bfloat16"):
        return float(value)
    f32 = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    if name in ("float32", "float"):
        return f32
    if name in ("float16", "half"):
        return struct.unpack("<e", struct.pack("<e", f32))[0]
    bits = struct.unpack("<I", struct.pack("<f", f32))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def hf_keep_indices(scores: Sequence[float], threshold: float,
                    dtype: Optional[str] = None) -> List[int]:
    """``keep = scores > threshold`` -- strict, and query order preserved.

    Boolean indexing drops rows but never reorders, so the surviving instances
    are in ascending query index, which is exactly what ``masks[keep]``,
    ``scores[keep]`` and ``boxes[keep]`` share.  Reproducing this is what lets
    the *soft* logits be selected with the same rows the processor kept.

    ``dtype`` is the scores tensor's dtype.  The processor compares in it, so
    under bfloat16 a threshold of 0.02 is really 0.0200195...; ``scores`` are
    that tensor's values (exact in a Python float), and the threshold is
    rounded the same way, or a score sitting on the rounded threshold would be
    kept here and dropped there.
    """
    t = _as_dtype(threshold, dtype)
    return [i for i, v in enumerate(scores) if float(v) > t]


def forward_kwargs_for(inputs: Dict[str, Any], *,
                       vision_embeds: Any = None,
                       name: str = "vision_embeds") -> Dict[str, Any]:
    """Arguments for ``Sam3Model.forward`` with exactly one vision source.

    ``forward`` raises ``ValueError("You must specify exactly one of
    pixel_values or vision_embeds")``, so the cached encoding must never be
    injected on top of a processor call that was given ``images=``.  With a
    cached encoding the processor is called text-only and the encoding
    supplied here; without one, ``pixel_values`` from the processor is the
    single source.

    ``name`` is the keyword the model takes the encoding under --
    ``vision_embeds`` for ``Sam3Model``, ``image_embeddings`` for the tracker,
    whose SAM-lineage forward raises "Only one of pixel_values and
    image_embeddings can be provided" for the same mistake.
    """
    out = dict(inputs)
    if vision_embeds is not None:
        out.pop("pixel_values", None)
        if name == "vision_embeds":
            out.pop("original_sizes", None)
        out[name] = vision_embeds
    elif "pixel_values" not in out:
        raise ValueError("forward needs pixel_values or %s" % name)
    assert not ("pixel_values" in out and name in out)
    return out

def rgb_array(pixels: bytes, width: int, height: int) -> Any:
    """Raw ``POST /images`` bytes -> ``(H, W, 3)`` uint8 numpy array.

    The upload layout of ``API.md`` §7 -- R-G-B, top-to-bottom, stride
    ``W * 3``, no padding -- is exactly numpy's default C order, so this is a
    zero-copy reshape of the buffer rather than a conversion.
    """
    import numpy as np  # noqa: PLC0415  (torch engines only)

    expected = int(width) * int(height) * 3
    if len(pixels) != expected:
        raise ValueError("expected %d pixel bytes, got %d" % (expected, len(pixels)))
    return np.frombuffer(pixels, dtype=np.uint8).reshape(int(height), int(width), 3)


def logits_to_canvas_masks(logits: Any, canvas: Size, *,
                           already_probabilities: bool = False) -> List[Any]:
    """Mask logits -> a list of full-canvas uint8 soft masks.

    ``logits`` is a torch tensor shaped ``(N, H, W)``, ``(N, 1, H, W)`` or
    ``(1, N, H, W)``.  It is bilinearly resampled to the model canvas *before*
    the sigmoid (upsampling the soft field, then quantising, is what gives the
    smooth edges ``DESIGN.md`` §4 is after) and quantised as
    ``round(255 * sigmoid(logit))``, so 128 is exactly logit 0 -- the model's
    own binarisation point and the client's default threshold (``API.md`` §8.3).

    Raises :class:`~sam3gimpd.modelmgr.NumericalFailure` on a non-finite tensor so
    that the manager's fp32 retry can fire instead of a silent NaN mask -- which
    it only can when this runs *inside* the function handed to
    ``ModelManager.run``.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    from ..modelmgr import NumericalFailure  # noqa: PLC0415  (circular at module scope)

    t = logits
    if t.dim() == 2:
        t = t[None, None]
    elif t.dim() == 3:
        t = t[:, None]
    elif t.dim() == 4:
        if t.shape[1] != 1 and t.shape[0] == 1:
            t = t[0][:, None]
    else:
        raise ValueError("cannot interpret mask tensor of shape %r" % (tuple(t.shape),))

    t = t.detach().to(torch.float32)
    if not bool(torch.isfinite(t).all()):
        raise NumericalFailure("mask tensor contains NaN or Inf")
    if (int(t.shape[-2]), int(t.shape[-1])) != (int(canvas.height), int(canvas.width)):
        t = F.interpolate(t, size=(int(canvas.height), int(canvas.width)),
                          mode="bilinear", align_corners=False)
    probs = t if already_probabilities else torch.sigmoid(t)
    probs = probs.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
    arr = probs[:, 0].cpu().numpy()
    return [arr[i] for i in range(arr.shape[0])]


def masks_to_canvas_masks(masks: Any, canvas: Size) -> Tuple[List[Any], bool]:
    """Whatever a processor's post-processing returned -> uint8 canvas masks.

    Returns ``(masks, soft)``.  ``soft`` is ``False`` when the input was already
    binarised, which the engine surfaces in ``EngineInfo.detail`` -- a hard mask
    still satisfies the contract (0/255 with 128 as the threshold) but loses the
    edge quality the format exists for, so it must not pass unnoticed.
    """
    import torch  # noqa: PLC0415

    if not isinstance(masks, torch.Tensor):
        masks = torch.as_tensor(masks)
    if masks.numel() == 0:
        return [], True
    if masks.is_floating_point():
        lo = float(masks.min())
        hi = float(masks.max())
        if lo >= 0.0 and hi <= 1.0:
            return logits_to_canvas_masks(masks, canvas, already_probabilities=True), True
        return logits_to_canvas_masks(masks, canvas), True
    hard = masks.to(torch.float32)
    return logits_to_canvas_masks(hard, canvas, already_probabilities=True), False


def call_with_supported_kwargs(fn: Callable[..., Any], kwargs: Dict[str, Any],
                               *args: Any) -> Any:
    """Call ``fn`` with only the keyword arguments its signature accepts.

    ``transformers`` v5 is young and SAM 3's signatures are still moving
    (``DESIGN.md`` §9 lists the churn as a Medium risk).  Filtering by
    signature means a renamed or added argument degrades to "not passed"
    instead of a ``TypeError`` in the middle of a user's prompt.  A callable
    that takes ``**kwargs`` gets everything.
    """
    import inspect  # noqa: PLC0415

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    allowed = {k: v for k, v in kwargs.items() if k in params}
    return fn(*args, **allowed)


def accepts_kwarg(fn: Callable[..., Any], name: str, *, explicit: bool = False) -> bool:
    """True when ``fn`` names ``name`` in its signature.  ``**kwargs`` counts
    too, unless ``explicit`` -- for an argument that must not be silently
    swallowed."""
    import inspect  # noqa: PLC0415

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    if name in sig.parameters:
        return True
    if explicit:
        return False
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def require_finite(values: Sequence[float], what: str) -> None:
    """Raise :class:`~sam3gimpd.modelmgr.NumericalFailure` if any value is NaN
    or infinite.

    For scores, which otherwise pass through ``min``/``max`` clamping and come
    out as certainty: ``min(1.0, nan)`` is ``1.0``.  Like
    :func:`logits_to_canvas_masks`, call it inside the function handed to
    ``ModelManager.run``, so that half precision gets its float32 retry.
    """
    from ..modelmgr import NumericalFailure  # noqa: PLC0415  (circular at module scope)

    for v in values:
        if not math.isfinite(float(v)):
            raise NumericalFailure("%s contain NaN or Inf" % (what,))


def clamp_score(value: float) -> float:
    """A score clamped into ``[0, 1]``, with anything non-finite as ``0.0``."""
    v = float(value)
    if not math.isfinite(v):
        return 0.0
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def move_inputs(inputs: Any, device: str, dtype: Any = None) -> Dict[str, Any]:
    """Move a processor's ``BatchFeature`` onto the device.

    Floating tensors are cast to the model dtype; integer tensors (input ids,
    point labels, attention masks) are left alone, because casting them to
    bf16 is a classic and very confusing source of garbage output.
    """
    import torch  # noqa: PLC0415

    out: Dict[str, Any] = {}
    items = inputs.items() if hasattr(inputs, "items") else dict(inputs).items()
    for key, value in items:
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            if dtype is not None and value.is_floating_point():
                value = value.to(dtype)
        out[key] = value
    return out


def finalize_instances(raws: Sequence[RawInstance], canvas: Size, *,
                       score_threshold: float = 0.0,
                       max_instances: Optional[int] = None
                       ) -> Tuple[List[RawInstance], bool]:
    """Sort descending, drop sub-threshold, clip, validate.

    The same rules :func:`build_instances` applies, for instances an engine
    cropped itself (the stub does).  Returns ``(instances, truncated)``.
    """
    kept = [r for r in raws if float(r.score) >= float(score_threshold)]
    kept.sort(key=lambda r: float(r.score), reverse=True)
    truncated = False
    if max_instances is not None and len(kept) > int(max_instances):
        kept = kept[:int(max_instances)]
        truncated = True
    for inst in kept:
        inst.validate(canvas)
    return kept, truncated
