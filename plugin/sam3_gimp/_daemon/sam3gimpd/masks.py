"""Mask geometry, quantisation and the binary wire format, as a reference.

``API.md`` §8 and §9 are binding, and every off-by-one against them shows up as
a mask that sits a pixel or two off the object -- exactly the bug a user notices
and cannot describe.  This module is an executable statement of that
arithmetic, and ``tests/daemon/test_masks.py`` is its property test.

What the daemon serves does not all come through here.  The live pieces are the
canvas geometry -- :func:`model_canvas_size` and :func:`canvas_transform`, which
``session.py`` reports on ``POST /images`` -- and :func:`sigmoid`, the
definition of the ``u8_soft`` encoding.  The masks themselves are quantised by
the engines (``engines.base.logits_to_canvas_masks``), cropped by
``engines.base.build_instances`` with its own cutoff and padding, and framed by
``types.pack_result``; :func:`encode_result` and :func:`decode_result` here
frame through that same ``pack_result``, which is what lets the tests
round-trip real frames.  :func:`plan_downscale` and
:func:`source_rect_for_bbox` are the client side of §9, mirrored in stdlib by
the plug-in.

What lives here
---------------

**Quantisation.**  Engines hand us mask logits (floats), probabilities, or
already-quantised bytes.  :func:`quantise_u8` turns any of those into the wire
representation: one ``uint8`` per pixel, ``round(255 * sigmoid(logit))``, so
that **128 is logit 0** -- the model's own binarisation point and the client's
default threshold.  Soft, never binary: the client re-thresholds locally with a
byte comparison and never pays a round trip for a slider drag.

**Cropping.**  :func:`tight_bbox` finds the smallest rectangle containing every
pixel at or above a low cutoff, optionally padded by a pixel or two so the soft
falloff is not clipped (API.md §8.3 explicitly permits, and this module uses,
that padding).  Values outside the crop are *defined* to be 0.

**Framing.**  :func:`encode_result` builds a :class:`~sam3gimpd.types.ResultHeader`
plus the tightly packed blob region and hands both to
:func:`sam3gimpd.types.pack_result`; :func:`decode_result` is its exact inverse, so
tests round-trip real frames rather than asserting against a golden blob.

**The downscale.**  SAM 3 resizes internally to 1008 px, so an image is
downscaled to a longest side of 1008 *before* transfer and encoding.
:func:`plan_downscale` computes that target size and -- critically -- records
the exact per-axis scale, so a mask produced at model resolution can be placed
back onto the full-resolution image with :func:`source_rect_for_bbox`.  The two
are inverses to well under a pixel and the property tests prove it over random
sizes and aspect ratios.

Coordinate conventions
----------------------

Three spaces, named as in API.md §5:

``source``
    the client's original image (e.g. 4000x3000).  The daemon normally never
    sees it; :func:`plan_downscale` exists because the *client* needs this
    computation and because the daemon's own CLI and test harness feed it real
    files.
``image`` (a.k.a. uploaded-image)
    what ``POST /images`` carried, longest side <= 1008.  **Every coordinate a
    client sends is in this space.**
``canvas`` (a.k.a. model-canvas)
    the space masks are returned in -- under the reference policy the uploaded
    image itself, so ``canvas_from_image`` is the identity.  **Every coordinate
    a client receives is in this space**, and a client maps it back with the
    reported transform whatever that is.

All mapping functions here work in **continuous edge coordinates**: integer
``x`` is the *left edge* of column ``x``, and the centre of pixel ``x`` is
``x + 0.5``.  Rectangles are half-open, ``[x0, x1) x [y0, y1)``, matching
:class:`~sam3gimpd.types.BBox`.  Mixing edge and centre conventions is the classic
half-pixel bug; it is avoided by never using centres except where a docstring
says so explicitly.

Dependencies
------------

Standard library only, **plus numpy if -- and only if -- it happens to be
importable**.  The base ``pip install sam3gimpd`` has zero runtime dependencies and
``--stub`` mode must never import anything heavy, so every numpy fast path has
a pure-Python fallback that produces byte-identical output.  The tests exercise
both by monkeypatching the module-level handle to ``None``.  torch is never
imported: a torch tensor is accepted by duck-typing (``.detach().cpu().numpy()``)
and converted by its own methods.
"""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .types import (
    DEFAULT_MASK_THRESHOLD,
    MASK_ENCODING_U8_SOFT,
    BBox,
    CanvasTransform,
    JobState,
    Limits,
    MaskInstance,
    ResultHeader,
    Size,
    pack_result,
    unpack_result,
)

__all__ = [
    "HAVE_NUMPY",
    "DEFAULT_CROP_CUTOFF",
    "DEFAULT_CROP_PAD",
    "DEFAULT_MASK_THRESHOLD",
    "DEFAULT_MODEL_CANVAS_SIDE",
    "MASK_ENCODING_U8_SOFT",
    "sigmoid",
    "SoftInstance",
    "EncodedMask",
    "DecodedInstance",
    "DownscalePlan",
    "normalise_mask",
    "quantise_u8",
    "tight_bbox",
    "crop_u8",
    "encode_mask",
    "encode_result",
    "build_result",
    "decode_result",
    "expand_to_canvas",
    "threshold_u8",
    "model_canvas_size",
    "canvas_transform",
    "canvas_rect_for_image_rect",
    "image_rect_for_bbox",
    "source_rect_for_bbox",
    "plan_downscale",
    "downscale_rgb",
    "resample_rgb",
    "resample_gray",
]


# --------------------------------------------------------------------------- #
# optional numpy
# --------------------------------------------------------------------------- #
#: Sentinel for "numpy has not been tried yet", so that ``None`` keeps its one
#: meaning: *do not use numpy*.
_NUMPY_UNTRIED = object()

#: The numpy module, ``None`` (unavailable or disabled), or the sentinel above.
#: Tests set ``masks._numpy_module = None`` to force the pure-Python fallback,
#: which still works because ``None`` is never re-probed.
_numpy_module: Any = _NUMPY_UNTRIED


def _np():
    """Return the numpy module or ``None``.

    The import is **lazy and cached**.  ``API.md`` §14 promises that a
    ``--stub`` daemon imports no numpy on any code path, and the stub never
    reaches a fast path that wants one; a module-scope ``import numpy`` would
    break that promise (and cost ~150 ms of start-up) on any machine that
    happens to have numpy installed alongside -- which every ``[cuda]`` venv
    does.  Every fast path goes through this single accessor, so a test can
    still disable numpy globally with one monkeypatch and exercise real code.
    """
    global _numpy_module
    if _numpy_module is _NUMPY_UNTRIED:
        try:
            import numpy as _numpy  # noqa: PLC0415  (deliberately lazy, optional)
        except Exception:  # pragma: no cover - environment-dependent
            _numpy_module = None
        else:
            _numpy_module = _numpy
    return _numpy_module


def __getattr__(name: str):
    """``masks.HAVE_NUMPY`` -- resolved on access, not at import.

    PEP 562 module ``__getattr__``; keeping the name out of the module
    namespace is what makes it lazy.
    """
    if name == "HAVE_NUMPY":
        return _np() is not None
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
#: Bytes below this are treated as "outside" when computing the tight crop.
#: 26/255 ~= 0.10 probability, i.e. logit ~= -2.2.  Deliberately *low*: the crop
#: exists to save bandwidth, not to threshold.  Thresholding is the client's job
#: (API.md §8.3) and happens against the full soft byte range.
DEFAULT_CROP_CUTOFF = 26

#: Extra pixels kept around the tight bbox so the soft falloff is not clipped.
#: API.md §8.3: "the crop is tight but not guaranteed minimal".
DEFAULT_CROP_PAD = 1

#: Reference model canvas side.  SAM 3 works at 1008 px.
DEFAULT_MODEL_CANVAS_SIDE = 1008

_SIGMOID_CLAMP = 40.0  # exp(40) is finite; beyond this sigmoid is 0 or 1 in f64


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #
def sigmoid(x: float) -> float:
    """Numerically stable logistic function.

    ``math.exp`` overflows around 710, and mask logits from a half-precision
    forward pass can be large; both tails are therefore computed from the side
    that cannot overflow, and the argument is clamped so that +-inf and huge
    finite values return exactly 1.0 / 0.0 instead of raising.
    """
    if x != x:  # NaN -> "outside", the safe direction for a mask
        return 0.0
    if x > _SIGMOID_CLAMP:
        return 1.0
    if x < -_SIGMOID_CLAMP:
        return 0.0
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _round_half_up(v: float) -> int:
    """Round halves away from zero.

    Python's built-in ``round`` is banker's rounding, which makes geometry
    depend on the parity of a coordinate -- fine statistically, maddening when
    two adjacent rectangles must tile.  API.md §9 says "round"; every rounding
    in this project means this one.
    """
    if v != v:
        raise ValueError("cannot round NaN")
    if v >= 0.0:
        return int(math.floor(v + 0.5))
    return -int(math.floor(-v + 0.5))


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else (hi if v > hi else v)


def _as_size(value: Any) -> Size:
    """Accept ``Size``, ``(w, h)`` or ``{"width":..,"height":..}``."""
    if isinstance(value, Size):
        return value
    if isinstance(value, dict):
        return Size.from_dict(value)
    w, h = value
    return Size(int(w), int(h))


# --------------------------------------------------------------------------- #
# mask normalisation
# --------------------------------------------------------------------------- #
def _detach_tensor(mask: Any) -> Any:
    """Convert a torch tensor to numpy **without importing torch**.

    Duck-typing on ``detach``/``cpu``/``numpy`` is enough: only tensors have all
    three, and calling them is the tensor's own business.  This keeps the daemon
    importable in a base install while still accepting real engine output.
    """
    if hasattr(mask, "detach") and hasattr(mask, "cpu") and hasattr(mask, "numpy"):
        try:
            t = mask.detach().cpu()
            try:
                t = t.float()
            except Exception:
                pass
            return t.numpy()
        except Exception:
            return mask
    return mask


def normalise_mask(
    mask: Any,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Tuple[Any, int, int]:
    """Coerce any mask representation to ``(values, width, height)``.

    Accepted inputs:

    * a numpy 2-D array (any dtype), a 1-D array plus explicit dimensions, or a
      3-D array with a leading or trailing singleton axis (``(1, H, W)`` and
      ``(H, W, 1)`` both come out of real models);
    * a torch tensor of the same shapes (converted via its own methods);
    * ``bytes`` / ``bytearray`` / ``memoryview`` / ``array.array`` flat buffers,
      which require explicit ``width`` and ``height``;
    * a flat Python sequence plus explicit dimensions;
    * a nested sequence of rows, whose dimensions are inferred.

    ``values`` is either a 2-D numpy array (numpy path) or a flat sequence
    (fallback path); callers must not care which, and every consumer in this
    module handles both.
    """
    mask = _detach_tensor(mask)
    np = _np()

    if np is not None and isinstance(mask, np.ndarray):
        arr = mask
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]
            else:
                raise ValueError("3-D mask must have a singleton axis, got %r" % (arr.shape,))
        if arr.ndim == 1:
            if width is None or height is None:
                raise ValueError("a flat mask requires explicit width and height")
            if arr.size != int(width) * int(height):
                raise ValueError("flat mask has %d values, expected %d*%d"
                                 % (arr.size, width, height))
            arr = arr.reshape(int(height), int(width))
        elif arr.ndim != 2:
            raise ValueError("mask must be 1-D or 2-D, got %dd" % arr.ndim)
        h, w = int(arr.shape[0]), int(arr.shape[1])
        _check_dims(w, h, width, height)
        return arr, w, h

    if isinstance(mask, (bytes, bytearray, memoryview)):
        buf = bytes(mask)
        if width is None or height is None:
            raise ValueError("a raw buffer mask requires explicit width and height")
        w, h = int(width), int(height)
        if len(buf) != w * h:
            raise ValueError("buffer has %d bytes, expected %d*%d" % (len(buf), w, h))
        return buf, w, h

    if isinstance(mask, array):
        if width is None or height is None:
            raise ValueError("an array.array mask requires explicit width and height")
        w, h = int(width), int(height)
        if len(mask) != w * h:
            raise ValueError("array has %d values, expected %d*%d" % (len(mask), w, h))
        return mask, w, h

    if not isinstance(mask, Sequence):
        raise TypeError("unsupported mask type %r" % type(mask).__name__)

    if len(mask) and _is_row(mask[0]):
        rows = mask
        h = len(rows)
        w = len(rows[0])
        flat: List[float] = []
        for row in rows:
            if len(row) != w:
                raise ValueError("ragged mask: row lengths %d and %d" % (w, len(row)))
            flat.extend(row)
        _check_dims(w, h, width, height)
        return flat, w, h

    if width is None or height is None:
        raise ValueError("a flat mask requires explicit width and height")
    w, h = int(width), int(height)
    if len(mask) != w * h:
        raise ValueError("flat mask has %d values, expected %d*%d" % (len(mask), w, h))
    return mask, w, h


def _is_row(item: Any) -> bool:
    np = _np()
    if np is not None and isinstance(item, np.ndarray):
        return True
    return isinstance(item, (list, tuple, bytes, bytearray, memoryview, array))


def _check_dims(w: int, h: int, width: Optional[int], height: Optional[int]) -> None:
    if width is not None and int(width) != w:
        raise ValueError("mask width %d does not match declared %d" % (w, int(width)))
    if height is not None and int(height) != h:
        raise ValueError("mask height %d does not match declared %d" % (h, int(height)))


# --------------------------------------------------------------------------- #
# quantisation
# --------------------------------------------------------------------------- #
def _detect_kind_seq(values: Sequence[Any]) -> str:
    """Infer whether a flat sequence holds logits, probabilities or bytes.

    Rules, in order (documented so that ``kind="auto"`` is predictable rather
    than magical):

    1. all-integer values within ``[0, 255]`` whose maximum exceeds 1 -> already
       quantised bytes;
    2. anything outside ``[0, 1]`` -> logits;
    3. otherwise -> probabilities.

    An all-zero or 0/1 mask is ambiguous between (1) and (3) and quantises
    identically either way, so the ambiguity is harmless.
    """
    vmin = None
    vmax = None
    all_int = True
    for v in values:
        if isinstance(v, bool) or not isinstance(v, int):
            all_int = False
        fv = float(v)
        if fv != fv:
            continue
        if vmin is None or fv < vmin:
            vmin = fv
        if vmax is None or fv > vmax:
            vmax = fv
    if vmin is None:
        return "prob"
    if all_int and vmin >= 0.0 and vmax <= 255.0 and vmax > 1.0:
        return "u8"
    if vmin < 0.0 or vmax > 1.0:
        return "logit"
    return "prob"


def _detect_kind_np(arr: Any) -> str:
    np = _np()
    if arr.dtype == np.bool_:
        return "prob"
    if arr.dtype.kind in ("u", "i"):
        if arr.size == 0:
            return "u8"
        lo = int(arr.min())
        hi = int(arr.max())
        if 0 <= lo and hi <= 255 and hi > 1:
            return "u8"
        if lo < 0 or hi > 1:
            return "logit"
        return "prob"
    if arr.size == 0:
        return "prob"
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return "prob"
    if float(finite.min()) < 0.0 or float(finite.max()) > 1.0:
        return "logit"
    return "prob"


def quantise_u8(
    mask: Any,
    width: Optional[int] = None,
    height: Optional[int] = None,
    kind: str = "auto",
) -> Tuple[bytes, int, int]:
    """Quantise a mask to the wire representation.

    Returns ``(data, width, height)`` where ``data`` is ``width * height`` bytes,
    row-major, top-to-bottom, left-to-right -- exactly the layout of API.md §8.3
    and exactly what ``Gegl.Buffer.set()`` wants for ``"Y' u8"``.

    ``kind`` selects the interpretation of the input:

    ``"logit"``
        ``value = round(255 * sigmoid(logit))``.  This is what the model emits
        and what the contract specifies, which is why **128 means logit 0**.
    ``"prob"``
        ``value = round(255 * clamp(p, 0, 1))``.
    ``"u8"``
        already 0-255; rounded and clamped, otherwise passed through.
    ``"auto"``
        detected per :func:`_detect_kind_seq` / :func:`_detect_kind_np`.

    NaN is mapped to 0 ("outside"), which is the safe direction: a NaN pixel
    disappears from the mask rather than punching a hole of certainty into it.
    """
    values, w, h = normalise_mask(mask, width, height)
    np = _np()

    if np is not None and isinstance(values, np.ndarray):
        arr = values
        use = _detect_kind_np(arr) if kind == "auto" else kind
        if use == "u8" and arr.dtype == np.uint8:
            return np.ascontiguousarray(arr).tobytes(), w, h
        f = arr.astype(np.float64, copy=True)
        f = np.nan_to_num(f, nan=(-_SIGMOID_CLAMP if use == "logit" else 0.0),
                          posinf=_SIGMOID_CLAMP, neginf=-_SIGMOID_CLAMP)
        if use == "logit":
            f = 1.0 / (1.0 + np.exp(-np.clip(f, -_SIGMOID_CLAMP, _SIGMOID_CLAMP)))
            f = f * 255.0
        elif use == "prob":
            f = np.clip(f, 0.0, 1.0) * 255.0
        elif use == "u8":
            f = np.clip(f, 0.0, 255.0)
        else:
            raise ValueError("unknown mask kind %r" % (kind,))
        out = np.floor(f + 0.5)
        out = np.clip(out, 0.0, 255.0).astype(np.uint8)
        return np.ascontiguousarray(out).tobytes(), w, h

    if isinstance(values, (bytes, bytearray)) and kind in ("auto", "u8"):
        return bytes(values), w, h

    use = _detect_kind_seq(values) if kind == "auto" else kind
    out = bytearray(w * h)
    if use == "logit":
        _sig = sigmoid
        for i, v in enumerate(values):
            out[i] = _clamp_int(int(math.floor(_sig(float(v)) * 255.0 + 0.5)), 0, 255)
    elif use == "prob":
        for i, v in enumerate(values):
            fv = float(v)
            if fv != fv:
                fv = 0.0
            if fv <= 0.0:
                out[i] = 0
            elif fv >= 1.0:
                out[i] = 255
            else:
                out[i] = _clamp_int(int(math.floor(fv * 255.0 + 0.5)), 0, 255)
    elif use == "u8":
        for i, v in enumerate(values):
            fv = float(v)
            if fv != fv:
                fv = 0.0
            out[i] = _clamp_int(int(math.floor(fv + 0.5)), 0, 255)
    else:
        raise ValueError("unknown mask kind %r" % (kind,))
    return bytes(out), w, h


def threshold_u8(data: bytes, threshold: int = DEFAULT_MASK_THRESHOLD,
                 inside: int = 255, outside: int = 0) -> bytes:
    """Binarise soft bytes: ``inside`` where ``value >= threshold``.

    Present so that tests (and the CLI) can do exactly what the plug-in does
    with a byte comparison, and so the semantics of ``>=`` live in one place.
    """
    t = _clamp_int(int(threshold), 0, 256)
    table = bytes((inside if v >= t else outside) for v in range(256))
    return bytes(data).translate(table)


# --------------------------------------------------------------------------- #
# cropping
# --------------------------------------------------------------------------- #
def tight_bbox(
    data: bytes,
    width: int,
    height: int,
    cutoff: int = DEFAULT_CROP_CUTOFF,
    pad: int = DEFAULT_CROP_PAD,
    bounds: Optional[Size] = None,
) -> Optional[BBox]:
    """Smallest half-open rectangle containing every pixel ``>= cutoff``.

    Returns ``None`` when no pixel reaches the cutoff -- an empty instance the
    caller should drop, since a zero-area bbox is not representable on the wire
    (``x1 > x0`` is required).

    ``pad`` grows the rectangle on every side and is clipped to the mask (or to
    ``bounds`` when the mask is a view of a larger canvas).  The padding is why
    API.md §8.3 says the crop is "tight but not guaranteed minimal": soft
    falloff below the cutoff is worth a couple of bytes per edge, because
    clipping it produces a visibly hard mask boundary when the client
    thresholds low.
    """
    w, h = int(width), int(height)
    if w <= 0 or h <= 0:
        return None
    if len(data) != w * h:
        raise ValueError("mask has %d bytes, expected %d*%d" % (len(data), w, h))
    cut = _clamp_int(int(cutoff), 0, 256)

    np = _np()
    if np is not None:
        arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w)
        hot = arr >= cut
        rows = np.flatnonzero(hot.any(axis=1))
        if rows.size == 0:
            return None
        cols = np.flatnonzero(hot.any(axis=0))
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
    else:
        # translate() does the comparison in C: 0 below the cutoff, 1 at or
        # above it.  find/rfind then locate the first and last hot column of a
        # row without a Python-level loop over pixels.
        table = bytes((0 if v < cut else 1) for v in range(256))
        flags = bytes(data).translate(table)
        y0 = -1
        y1 = 0
        x0 = w
        x1 = 0
        for y in range(h):
            row = flags[y * w:(y + 1) * w]
            first = row.find(1)
            if first < 0:
                continue
            last = row.rfind(1) + 1
            if y0 < 0:
                y0 = y
            y1 = y + 1
            if first < x0:
                x0 = first
            if last > x1:
                x1 = last
        if y0 < 0:
            return None

    p = max(0, int(pad))
    lim = bounds if bounds is not None else Size(w, h)
    x0 = _clamp_int(x0 - p, 0, lim.width)
    y0 = _clamp_int(y0 - p, 0, lim.height)
    x1 = _clamp_int(x1 + p, 0, lim.width)
    y1 = _clamp_int(y1 + p, 0, lim.height)
    if x1 <= x0 or y1 <= y0:
        return None
    return BBox(x0, y0, x1, y1)


def crop_u8(data: bytes, width: int, height: int, bbox: BBox) -> bytes:
    """Extract ``bbox`` from a full-canvas soft mask, row-major, no padding."""
    w, h = int(width), int(height)
    if len(data) != w * h:
        raise ValueError("mask has %d bytes, expected %d*%d" % (len(data), w, h))
    if not bbox.is_valid(Size(w, h)):
        raise ValueError("bbox %s is not inside %dx%d" % (bbox.to_list(), w, h))
    if bbox.x0 == 0 and bbox.x1 == w and bbox.y0 == 0 and bbox.y1 == h:
        return bytes(data)

    np = _np()
    if np is not None:
        arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w)
        return np.ascontiguousarray(arr[bbox.y0:bbox.y1, bbox.x0:bbox.x1]).tobytes()

    out = bytearray(bbox.width * bbox.height)
    src = bytes(data)
    o = 0
    span = bbox.width
    for y in range(bbox.y0, bbox.y1):
        base = y * w + bbox.x0
        out[o:o + span] = src[base:base + span]
        o += span
    return bytes(out)


# --------------------------------------------------------------------------- #
# one encoded mask
# --------------------------------------------------------------------------- #
@dataclass
class EncodedMask:
    """A cropped soft mask plus the canvas rectangle it belongs at.

    ``data`` is ``bbox.width * bbox.height`` bytes; every pixel outside ``bbox``
    is defined to be 0.  This is precisely one wire instance minus its identity
    (id, score, label), which is why :meth:`to_instance` takes those.
    """

    bbox: BBox
    data: bytes

    @property
    def width(self) -> int:
        return self.bbox.width

    @property
    def height(self) -> int:
        return self.bbox.height

    def __post_init__(self) -> None:
        expected = self.bbox.width * self.bbox.height
        if len(self.data) != expected:
            raise ValueError("crop has %d bytes, bbox %s needs %d"
                             % (len(self.data), self.bbox.to_list(), expected))

    def to_instance(self, instance_id: int, score: float, label: str = "",
                    blob_offset: int = 0) -> MaskInstance:
        """Build the wire metadata.  ``pack_result`` recomputes the offsets, so
        ``blob_offset`` here is only a convenience for callers that lay out the
        blob themselves."""
        return MaskInstance(
            instance_id=int(instance_id),
            score=float(score),
            bbox=self.bbox,
            mask_width=self.bbox.width,
            mask_height=self.bbox.height,
            blob_offset=int(blob_offset),
            blob_length=len(self.data),
            label=str(label),
        )

    def value_at(self, x: int, y: int) -> int:
        """Soft value at **canvas** coordinates; 0 outside the crop."""
        if not (self.bbox.x0 <= x < self.bbox.x1 and self.bbox.y0 <= y < self.bbox.y1):
            return 0
        return self.data[(y - self.bbox.y0) * self.bbox.width + (x - self.bbox.x0)]

    def to_canvas(self, canvas: Any, fill: int = 0) -> bytes:
        """Paste the crop back into a full ``canvas``-sized buffer.

        The daemon never sends this -- it is the whole point of cropping that it
        does not -- but tests use it to assert an exact round-trip, and the
        canvas harness uses it to draw overlays.
        """
        size = _as_size(canvas)
        return _paste(self.data, self.bbox, size, fill)


def _paste(crop: bytes, bbox: BBox, canvas: Size, fill: int = 0) -> bytes:
    if not bbox.is_valid(canvas):
        raise ValueError("bbox %s is not inside canvas %s"
                         % (bbox.to_list(), canvas.to_dict()))
    np = _np()
    if np is not None:
        out = np.full((canvas.height, canvas.width), fill, dtype=np.uint8)
        out[bbox.y0:bbox.y1, bbox.x0:bbox.x1] = (
            np.frombuffer(crop, dtype=np.uint8).reshape(bbox.height, bbox.width))
        return out.tobytes()
    out_b = bytearray(bytes((fill,)) * (canvas.width * canvas.height))
    span = bbox.width
    for row in range(bbox.height):
        src = row * span
        dst = (bbox.y0 + row) * canvas.width + bbox.x0
        out_b[dst:dst + span] = crop[src:src + span]
    return bytes(out_b)


def encode_mask(
    mask: Any,
    width: Optional[int] = None,
    height: Optional[int] = None,
    kind: str = "auto",
    cutoff: int = DEFAULT_CROP_CUTOFF,
    pad: int = DEFAULT_CROP_PAD,
    canvas: Optional[Any] = None,
) -> Optional[EncodedMask]:
    """Quantise, find the tight bbox, crop.  ``None`` if the mask is empty.

    When ``canvas`` is given and the mask is not already that size, the **soft**
    mask is resampled to the canvas first.  SAM's decoder emits low-resolution
    logits (256x256 in the reference model) that are upsampled to the working
    resolution; doing that upsample on the soft values and letting the client
    threshold afterwards is what keeps mask edges smooth instead of stepped
    (DESIGN.md §1, API.md §8).  Resampling *before* the crop also means the
    reported bbox is in canvas pixels, as the contract requires.
    """
    if width is None and height is None and canvas is not None and isinstance(
            mask, (bytes, bytearray, memoryview, array)):
        # A flat buffer carries no shape.  Canvas-sized is the overwhelmingly
        # common case (an engine that already upsampled its logits), and it is
        # the only guess that cannot silently mean something else.
        _c = _as_size(canvas)
        width, height = _c.width, _c.height
    data, w, h = quantise_u8(mask, width, height, kind=kind)
    if canvas is not None:
        bounds = _as_size(canvas)
        if bounds.width != w or bounds.height != h:
            data = resample_gray(data, w, h, bounds.width, bounds.height)
            w, h = bounds.width, bounds.height
    bbox = tight_bbox(data, w, h, cutoff=cutoff, pad=pad, bounds=Size(w, h))
    if bbox is None:
        return None
    return EncodedMask(bbox=bbox, data=crop_u8(data, w, h, bbox))


# --------------------------------------------------------------------------- #
# a full result frame
# --------------------------------------------------------------------------- #
@dataclass
class SoftInstance:
    """One engine output on its way to the wire.

    Engines produce these; :func:`encode_result` sorts, truncates, quantises,
    crops and packs them.  ``mask`` is anything :func:`normalise_mask` accepts.
    """

    mask: Any
    score: float = 1.0
    label: str = ""
    width: Optional[int] = None
    height: Optional[int] = None


def _coerce_soft(item: Any) -> SoftInstance:
    if isinstance(item, SoftInstance):
        return item
    if isinstance(item, dict):
        return SoftInstance(
            mask=item["mask"],
            score=float(item.get("score", 1.0)),
            label=str(item.get("label", "")),
            width=item.get("width"),
            height=item.get("height"),
        )
    if isinstance(item, tuple):
        if len(item) == 2:
            return SoftInstance(mask=item[0], score=float(item[1]))
        if len(item) == 3:
            return SoftInstance(mask=item[0], score=float(item[1]), label=str(item[2]))
        raise ValueError("mask tuple must be (mask, score[, label])")
    return SoftInstance(mask=item)


def encode_result(
    job_id: str,
    request_id: str,
    image_id: str,
    engine: str,
    image: Any,
    model_canvas: Any,
    masks: Iterable[Any],
    canvas_from_image: Optional[CanvasTransform] = None,
    prompt: Optional[Dict[str, Any]] = None,
    elapsed_ms: float = 0.0,
    max_instances: Optional[int] = None,
    cutoff: int = DEFAULT_CROP_CUTOFF,
    pad: int = DEFAULT_CROP_PAD,
    kind: str = "auto",
    truncated: bool = False,
    state: str = JobState.DONE,
) -> Tuple[ResultHeader, bytes]:
    """Build ``(header, frame_bytes)`` for a finished job.

    Instances are sorted by score **descending** (API.md §8.2), then clipped to
    ``max_instances`` -- setting ``truncated`` when that drops anything -- then
    quantised and cropped.  A mask with nothing above ``cutoff`` is dropped
    entirely rather than sent as a degenerate rectangle, and ``instance_id`` is
    assigned after that so ids are always ``0..N-1`` in returned order.

    The header is returned alongside the bytes because ``GET /jobs/{id}?meta=1``
    serves exactly this object as JSON; the server must not rebuild it.
    """
    img = _as_size(image)
    canvas = _as_size(model_canvas)
    transform = canvas_from_image or canvas_transform(img, canvas)

    items = [_coerce_soft(m) for m in masks]
    items.sort(key=lambda s: float(s.score), reverse=True)
    if max_instances is not None:
        limit = _clamp_int(int(max_instances), 1, Limits.MAX_INSTANCES)
        if len(items) > limit:
            items = items[:limit]
            truncated = True

    instances: List[MaskInstance] = []
    blobs: List[bytes] = []
    for item in items:
        enc = encode_mask(item.mask, item.width, item.height, kind=kind,
                          cutoff=cutoff, pad=pad, canvas=canvas)
        if enc is None:
            continue
        instances.append(enc.to_instance(len(instances), item.score, item.label))
        blobs.append(enc.data)

    header = ResultHeader(
        job_id=str(job_id),
        request_id=str(request_id),
        image_id=str(image_id),
        engine=str(engine),
        image=img,
        model_canvas=canvas,
        canvas_from_image=transform,
        instances=instances,
        prompt=dict(prompt or {}),
        elapsed_ms=float(elapsed_ms),
        mask_encoding=MASK_ENCODING_U8_SOFT,
        state=state,
        truncated=bool(truncated),
    )
    payload = pack_result(header, blobs)
    return header, payload


def build_result(*args: Any, **kwargs: Any) -> bytes:
    """:func:`encode_result` when only the bytes are wanted."""
    return encode_result(*args, **kwargs)[1]


@dataclass
class DecodedInstance:
    """One instance recovered from a frame: metadata plus its crop bytes."""

    instance: MaskInstance
    data: bytes

    @property
    def bbox(self) -> BBox:
        return self.instance.bbox

    @property
    def score(self) -> float:
        return self.instance.score

    @property
    def label(self) -> str:
        return self.instance.label

    @property
    def width(self) -> int:
        return self.instance.mask_width

    @property
    def height(self) -> int:
        return self.instance.mask_height

    def as_encoded(self) -> EncodedMask:
        return EncodedMask(bbox=self.bbox, data=self.data)

    def value_at(self, x: int, y: int) -> int:
        return self.as_encoded().value_at(x, y)

    def to_canvas(self, canvas: Optional[Any] = None, fill: int = 0) -> bytes:
        if canvas is None:
            raise ValueError("to_canvas needs the canvas size")
        return _paste(self.data, self.bbox, _as_size(canvas), fill)

    def binary(self, threshold: int = DEFAULT_MASK_THRESHOLD) -> bytes:
        return threshold_u8(self.data, threshold)


def decode_result(payload: bytes) -> Tuple[ResultHeader, List[DecodedInstance]]:
    """Exact inverse of :func:`encode_result`.

    Validates the frame through :func:`sam3gimpd.types.unpack_result` (magic, header
    length, blob length, tight packing, bbox sanity) and slices each instance
    out of the blob region.  Tests round-trip through this rather than against a
    golden blob, so a change to the layout fails loudly on both sides.
    """
    header, blob = unpack_result(payload)
    out = [DecodedInstance(instance=inst, data=inst.mask_bytes(blob))
           for inst in header.instances]
    return header, out


def expand_to_canvas(decoded: Sequence[DecodedInstance], canvas: Any,
                     fill: int = 0) -> List[bytes]:
    """Full-canvas buffers for every decoded instance (tests, previews)."""
    size = _as_size(canvas)
    return [d.to_canvas(size, fill) for d in decoded]


# --------------------------------------------------------------------------- #
# geometry: image <-> canvas <-> source
# --------------------------------------------------------------------------- #
def model_canvas_size(image: Any, side: int = DEFAULT_MODEL_CANVAS_SIDE) -> Size:
    """The space returned masks live in: the uploaded image itself.

    SAM 3 resizes preserving aspect ratio and pads, so a square ``side x side``
    canvas would map a 1008x672 upload back as though it had been stretched
    1.5x vertically, and every selection on a non-square image would land off
    its object.  Rather than model the letterbox, the engines avoid it: they
    ask ``post_process_instance_segmentation`` for ``target_sizes`` equal to
    the uploaded image -- exactly what the reference example passes as
    ``original_sizes`` -- so it undoes the resize and padding itself and
    returns masks in image space.  The canvas is then the image, and
    ``canvas_from_image`` is the identity.

    ``side`` is retained for callers that report a nominal working resolution.
    """
    img = _as_size(image)
    return Size(int(img.width), int(img.height))


def canvas_transform(image: Any, canvas: Any) -> CanvasTransform:
    """``canvas = image * scale + offset``.

    With masks post-processed straight to the uploaded image size this is the
    identity.  It is computed rather than hardcoded so that a canvas which
    genuinely differs still maps correctly, per axis.
    """
    img = _as_size(image)
    cvs = _as_size(canvas)
    if img.width <= 0 or img.height <= 0:
        raise ValueError("image size must be positive, got %s" % (img.to_dict(),))
    return CanvasTransform(
        scale_x=cvs.width / float(img.width),
        scale_y=cvs.height / float(img.height),
        offset_x=0.0,
        offset_y=0.0,
    )


def canvas_rect_for_image_rect(rect: Sequence[float],
                               transform: CanvasTransform) -> Tuple[float, float, float, float]:
    """Forward map an uploaded-image rectangle into canvas floats."""
    x0, y0, x1, y1 = (float(v) for v in rect)
    cx0, cy0 = transform.image_to_canvas(x0, y0)
    cx1, cy1 = transform.image_to_canvas(x1, y1)
    return (cx0, cy0, cx1, cy1)


def image_rect_for_bbox(bbox: BBox,
                        transform: CanvasTransform) -> Tuple[float, float, float, float]:
    """API.md §9 step 1: canvas bbox -> uploaded-image floats."""
    x0, y0 = transform.canvas_to_image(float(bbox.x0), float(bbox.y0))
    x1, y1 = transform.canvas_to_image(float(bbox.x1), float(bbox.y1))
    return (x0, y0, x1, y1)


def source_rect_for_bbox(
    bbox: BBox,
    transform: CanvasTransform,
    image: Any,
    source: Optional[Any] = None,
) -> Tuple[int, int, int, int]:
    """API.md §9 steps 1-3: canvas bbox -> ``(X0, Y0, DW, DH)`` on the original.

    ``image`` is the uploaded size, ``source`` the client's original size
    (``None`` means "no client-side downscale", i.e. source == image).

    Each edge is rounded **independently** and the size derived from the rounded
    edges, exactly as the contract specifies.  Doing it the other way -- round
    the origin, round the size -- makes adjacent instances overlap or leave a
    one-pixel seam, and that seam is visible when two masks are applied as
    neighbouring channels.

    The plug-in mirrors this function in stdlib; keep the two in step.
    """
    img = _as_size(image)
    src = _as_size(source) if source is not None else img
    ix0, iy0, ix1, iy1 = image_rect_for_bbox(bbox, transform)
    u = src.width / float(img.width)
    v = src.height / float(img.height)
    x0 = _round_half_up(ix0 * u)
    x1 = _round_half_up(ix1 * u)
    y0 = _round_half_up(iy0 * v)
    y1 = _round_half_up(iy1 * v)
    x0 = _clamp_int(x0, 0, max(0, src.width - 1))
    y0 = _clamp_int(y0, 0, max(0, src.height - 1))
    x1 = _clamp_int(x1, x0 + 1, src.width)
    y1 = _clamp_int(y1, y0 + 1, src.height)
    return (x0, y0, x1 - x0, y1 - y0)


# --------------------------------------------------------------------------- #
# the pre-encode downscale
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DownscalePlan:
    """How an original image maps onto the uploaded image.

    SAM 3 resizes to 1008 px internally (DESIGN.md §1), so sending more pixels
    buys nothing and the daemon would only throw them away; the API rejects
    anything larger outright rather than silently downscaling, because a silent
    downscale would desynchronise the client's own coordinate bookkeeping
    (API.md §7).  This plan is that bookkeeping.

    Scales are **per axis**.  They are normally equal, but the ``[16, 1008]``
    side limits force anisotropy for extreme aspect ratios (a 1x1000 strip
    cannot be uploaded at all unless its width is grown to 16), and recording
    both is what keeps the mapping exact in those cases.
    """

    source: Size
    target: Size

    @property
    def scale_x(self) -> float:
        return self.target.width / float(self.source.width)

    @property
    def scale_y(self) -> float:
        return self.target.height / float(self.source.height)

    @property
    def needs_resample(self) -> bool:
        return (self.target.width != self.source.width
                or self.target.height != self.source.height)

    @property
    def transform(self) -> CanvasTransform:
        """``uploaded = source * scale``, in the same shape the API uses for
        ``canvas_from_image`` -- so the two compose cleanly."""
        return CanvasTransform(scale_x=self.scale_x, scale_y=self.scale_y)

    def source_to_target(self, x: float, y: float) -> Tuple[float, float]:
        return (x * self.scale_x, y * self.scale_y)

    def target_to_source(self, x: float, y: float) -> Tuple[float, float]:
        return (x / self.scale_x, y / self.scale_y)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
        }


def plan_downscale(
    width: int,
    height: int,
    max_side: int = Limits.MAX_IMAGE_SIDE,
    min_side: int = Limits.MIN_IMAGE_SIDE,
) -> DownscalePlan:
    """Choose the uploaded size for a ``width x height`` original.

    Policy:

    1. scale so the longest side is at most ``max_side``; never upscale for the
       sake of it (``scale <= 1``);
    2. round each axis to the nearest integer, then clamp into
       ``[min_side, max_side]``.

    Step 2's clamp is the only source of anisotropy, and it is unavoidable: the
    API requires both sides in ``[16, 1008]``, so a 1x1000 strip must be widened
    to 16.  The resulting distortion is recorded exactly in the plan's per-axis
    scales, so masks still land where they should.
    """
    w, h = int(width), int(height)
    if w <= 0 or h <= 0:
        raise ValueError("image size must be positive, got %dx%d" % (w, h))
    if min_side > max_side:
        raise ValueError("min_side %d exceeds max_side %d" % (min_side, max_side))

    scale = min(1.0, float(max_side) / float(max(w, h)))
    tw = _clamp_int(_round_half_up(w * scale), 1, max_side)
    th = _clamp_int(_round_half_up(h * scale), 1, max_side)
    tw = _clamp_int(max(tw, int(min_side)), int(min_side), int(max_side))
    th = _clamp_int(max(th, int(min_side)), int(min_side), int(max_side))
    return DownscalePlan(source=Size(w, h), target=Size(tw, th))


def downscale_rgb(
    pixels: bytes,
    width: int,
    height: int,
    max_side: int = Limits.MAX_IMAGE_SIDE,
    min_side: int = Limits.MIN_IMAGE_SIDE,
) -> Tuple[bytes, DownscalePlan]:
    """Resize raw RGB to the uploadable size.  Returns ``(pixels, plan)``.

    Input and output are both the ``POST /images`` body layout: ``uint8``, R-G-B
    interleaved, row-major top-to-bottom, stride ``width * 3``, no padding.
    """
    plan = plan_downscale(width, height, max_side=max_side, min_side=min_side)
    if not plan.needs_resample:
        return bytes(pixels), plan
    out = resample_rgb(pixels, width, height, plan.target.width, plan.target.height)
    return out, plan


def resample_rgb(pixels: bytes, sw: int, sh: int, dw: int, dh: int) -> bytes:
    """Area-average (down) / bilinear (up) resample of interleaved RGB."""
    return _resample_planar(pixels, int(sw), int(sh), int(dw), int(dh), 3)


def resample_gray(data: bytes, sw: int, sh: int, dw: int, dh: int) -> bytes:
    """Same, for a single-channel ``uint8`` buffer (a soft mask)."""
    return _resample_planar(data, int(sw), int(sh), int(dw), int(dh), 1)


def _axis_weights(src_n: int, dst_n: int) -> List[Tuple[int, List[float]]]:
    """Per-destination ``(first_source_index, weights)``.

    Downscaling uses a **box filter**: destination pixel ``i`` covers the source
    interval ``[i*src/dst, (i+1)*src/dst)`` and averages it with the exact
    fractional overlap.  That is the right filter for shrinking photographs --
    it uses every source pixel exactly once and does not alias.

    Upscaling uses **bilinear interpolation** about pixel centres
    (``p = (i + 0.5) * src/dst - 0.5``); a box filter degenerates to nearest
    neighbour when magnifying, which would visibly block up a mask.

    Both agree on the mapping ``source = dest / scale`` in edge coordinates,
    which is exactly the mapping :class:`DownscalePlan` reports, so the pixels
    and the geometry cannot drift apart.
    """
    if src_n <= 0 or dst_n <= 0:
        raise ValueError("resample dimensions must be positive")
    out: List[Tuple[int, List[float]]] = []
    ratio = src_n / float(dst_n)

    if dst_n <= src_n:
        for i in range(dst_n):
            a = i * ratio
            b = (i + 1) * ratio
            i0 = int(math.floor(a))
            i1 = int(math.ceil(b - 1e-12))
            i0 = _clamp_int(i0, 0, src_n - 1)
            i1 = _clamp_int(i1, i0 + 1, src_n)
            ws = []
            total = 0.0
            for s in range(i0, i1):
                w = min(b, s + 1.0) - max(a, float(s))
                if w < 0.0:
                    w = 0.0
                ws.append(w)
                total += w
            if total <= 0.0:
                ws = [1.0]
                i1 = i0 + 1
                total = 1.0
            out.append((i0, [w / total for w in ws]))
        return out

    for i in range(dst_n):
        p = (i + 0.5) * ratio - 0.5
        if p <= 0.0:
            out.append((0, [1.0]))
            continue
        if p >= src_n - 1:
            out.append((src_n - 1, [1.0]))
            continue
        i0 = int(math.floor(p))
        frac = p - i0
        out.append((i0, [1.0 - frac, frac]))
    return out


def _weight_matrix(src_n: int, dst_n: int):
    np = _np()
    m = np.zeros((dst_n, src_n), dtype=np.float64)
    for i, (i0, ws) in enumerate(_axis_weights(src_n, dst_n)):
        m[i, i0:i0 + len(ws)] = ws
    return m


def _resample_planar(buf: bytes, sw: int, sh: int, dw: int, dh: int, ch: int) -> bytes:
    if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
        raise ValueError("resample dimensions must be positive")
    if len(buf) != sw * sh * ch:
        raise ValueError("buffer has %d bytes, expected %d*%d*%d" % (len(buf), sw, sh, ch))
    if sw == dw and sh == dh:
        return bytes(buf)

    np = _np()
    if np is not None:
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(sh, sw, ch).astype(np.float64)
        if dw != sw:
            arr = np.einsum("yxc,dx->ydc", arr, _weight_matrix(sw, dw))
        if dh != sh:
            arr = np.einsum("yxc,dy->dxc", arr, _weight_matrix(sh, dh))
        out = np.clip(np.floor(arr + 0.5), 0.0, 255.0).astype(np.uint8)
        return np.ascontiguousarray(out).tobytes()

    src = bytes(buf)
    # Horizontal pass into a float scratch buffer, then vertical.  Separable, so
    # the cost is O((sw*dh + dw*sh) * taps) instead of O(dw*dh*taps^2).
    wx = _axis_weights(sw, dw)
    mid = array("d", bytes(8 * dw * sh * ch))
    for y in range(sh):
        row_base = y * sw * ch
        out_base = y * dw * ch
        for i, (i0, ws) in enumerate(wx):
            o = out_base + i * ch
            p0 = row_base + i0 * ch
            for c in range(ch):
                acc = 0.0
                p = p0 + c
                for w in ws:
                    acc += src[p] * w
                    p += ch
                mid[o + c] = acc

    wy = _axis_weights(sh, dh)
    out_b = bytearray(dw * dh * ch)
    stride = dw * ch
    for j, (j0, ws) in enumerate(wy):
        out_base = j * stride
        for x in range(dw):
            o = out_base + x * ch
            p0 = j0 * stride + x * ch
            for c in range(ch):
                acc = 0.0
                p = p0 + c
                for w in ws:
                    acc += mid[p] * w
                    p += stride
                v = int(math.floor(acc + 0.5))
                out_b[o + c] = 0 if v < 0 else (255 if v > 255 else v)
    return bytes(out_b)
