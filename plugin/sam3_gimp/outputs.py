"""Turn SAM 3 masks into things a GIMP user can actually edit.

This module is the *output plumbing* half of the plug-in: it takes the decoded
result of a ``sam3gimpd`` job -- cropped uint8 soft masks in **model-canvas**
coordinates, as specified in ``_daemon/API.md`` sections 8 and 9 -- and
materialises it inside a ``Gimp.Image`` as a selection, saved channels, layer
masks, a layer group, or vector paths.

Constraints it lives under
--------------------------

* **Zero third-party dependencies.**  Standard library plus ``gi`` only; this
  code runs inside GIMP's embedded Python, which cannot be counted on to have
  numpy.
* **Importable without GIMP.**  ``gi.repository.Gimp`` is imported lazily, on
  first use, so the module imports (and most of it tests) without GIMP
  installed.  See :func:`set_gimp_modules` for the test/harness hook.
* **No pure-Python resampling.**  Scaling a mask from model-canvas resolution up
  to image resolution is done by GEGL, in C, by scaling a throwaway GRAY
  ``Gimp.Image`` (``API.md`` section 9, step 2; see :func:`_scale_gray`).

The three-stage pipeline
------------------------

Every output mode runs the same pipeline, which is why they behave
consistently:

1. **Byte-level post-ops**, applied to the cropped soft mask *before* it ever
   reaches GIMP: threshold remap, hole fill, minimum-area rejection.  These are
   ``bytes.translate``/``bytes.count`` operations, i.e. C speed, plus one
   scanline flood fill.
2. **Placement**, ``API.md`` section 9: invert the reported
   ``canvas_from_image`` affine, apply the client's own downscale factor, round
   each edge independently, then let GEGL scale the crop into that rectangle.
   The transform is *always* taken from the result header; the 1008x1008 squash
   is never hardcoded.
3. **GIMP-level post-ops**, applied through the selection because that is where
   GIMP keeps its morphology in C: grow, shrink, smooth, feather.  The result is
   read back out with ``Gimp.Selection.save`` when a channel is wanted.

Thresholding deserves a note.  ``API.md`` section 8.3 says a client thresholds
with ``value >= T`` and section 9 says to threshold *after* scaling, because
binarising first and upsampling second gives stair-stepped edges.  Both are
honoured here without any fragile PDB colour op: :func:`threshold_table` builds
a 256-entry lookup that maps the soft mask to a narrow linear ramp *centred on
T* rather than a hard step.  The mask stays soft across the scale (so the edge
quality argument holds), the user's threshold is respected exactly, and the
whole thing is one ``bytes.translate`` call.  ``PostOps.edge_softness = 0``
restores hard binarisation for anyone who wants it.

Undo
----

:func:`apply_result` wraps everything, *including the error path*, in
``image.undo_group_start()`` / ``undo_group_end()``.  One Ctrl+Z reverts the
entire Apply -- not twenty separate steps.  ``Gimp.context_push()`` /
``context_pop()`` are paired the same way.

Inside the group, everything that is only a means to an end runs with undo
*frozen*: the hidden scratch channels (inserted, renamed and removed) and
every use of the selection as a work surface.  What the user asked for is
recorded thawed -- the new channels, layers, masks and paths, and a changed
selection as exactly one ``select_item`` from a finished scratch channel.
The two never mix, so nothing on the undo stack refers to a scratch channel
that has already gone: GIMP's ``Gimp.Selection.save`` inserts its channel
with undo recorded, and a scratch channel added that way but removed frozen
made Undo fail an assertion and Redo put the channel back in the dock.

An Apply that fails halfway puts the selection back and takes out the
outputs it had already added, so the image is left as it was.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "OutputError",
    "OutputMode",
    "SelectionOp",
    "Space",
    "CanvasTransform",
    "Instance",
    "MaskResult",
    "PostOps",
    "OutputOptions",
    "AppliedResult",
    "threshold_table",
    "apply_threshold",
    "mask_area",
    "fill_holes",
    "instance_name",
    "unique_name",
    "apply_result",
    "set_gimp_modules",
    "reset_gimp_modules",
    "rebind_gi",
]


# --------------------------------------------------------------------------- #
# lazy gi access
# --------------------------------------------------------------------------- #
#: babl format of the mask bytes described in API.md 8.3 -- one uint8 per pixel,
#: row-major, top to bottom, no padding.  Exactly what Gegl.Buffer.set() wants.
GRAY_FORMAT = "Y' u8"

_GIMP = None
_GEGL = None


def set_gimp_modules(gimp: Any = None, gegl: Any = None) -> None:
    """Install module objects to use instead of importing ``gi.repository``.

    The unit tests and ``tools/canvas_harness.py`` use this to run the output
    logic against stubs.  Passing ``None`` for either argument leaves that one
    to be imported normally.
    """
    global _GIMP, _GEGL
    _GIMP = gimp
    _GEGL = gegl


def reset_gimp_modules() -> None:
    """Forget anything :func:`set_gimp_modules` or a previous import cached."""
    global _GIMP, _GEGL
    _GIMP = None
    _GEGL = None


def rebind_gi() -> None:
    """Re-resolve ``gi.repository`` on next use.

    ``tests/fake_gimp`` calls this on every client module it knows about after
    swapping the stubs into ``sys.modules``; providing it means this module is
    never ``importlib.reload``ed out from under a running test.
    """
    reset_gimp_modules()


def _require(name: str):
    cached = sys.modules.get("gi.repository." + name)
    if cached is not None:
        return cached
    import gi  # noqa: PLC0415  (deliberately lazy: importable without GIMP)

    version = {"Gimp": "3.0", "Gegl": "0.4", "Babl": "0.1"}.get(name)
    if version is not None:
        try:
            gi.require_version(name, version)
        except (ValueError, AttributeError):
            # Already required at another version, or a stub gi without the
            # function.  The import below is what actually matters.
            pass
    from gi import repository  # noqa: PLC0415

    return getattr(repository, name)


def _gimp():
    global _GIMP
    if _GIMP is None:
        _GIMP = _require("Gimp")
    return _GIMP


def _gegl():
    global _GEGL
    if _GEGL is None:
        _GEGL = _require("Gegl")
    return _GEGL


class OutputError(RuntimeError):
    """Something in the GIMP-side plumbing failed.

    Raised with a message meant to be shown to the user, not a traceback.
    """


# --------------------------------------------------------------------------- #
# enumerations (plain strings so Gimp.ProcedureConfig can persist them)
# --------------------------------------------------------------------------- #
class OutputMode:
    SELECTION = "selection"
    CHANNELS = "channels"
    LAYER_MASKS = "layer-masks"
    LAYER_GROUPS = "layer-groups"
    PATHS = "paths"

    ALL = (SELECTION, CHANNELS, LAYER_MASKS, LAYER_GROUPS, PATHS)

    #: Labels for the mode combo in the dialog.
    LABELS = {
        SELECTION: "Selection",
        CHANNELS: "Channels",
        LAYER_MASKS: "Layer mask",
        LAYER_GROUPS: "Layer group",
        PATHS: "Paths",
    }


class SelectionOp:
    REPLACE = "replace"
    ADD = "add"
    SUBTRACT = "subtract"
    INTERSECT = "intersect"

    ALL = (REPLACE, ADD, SUBTRACT, INTERSECT)


class Space:
    """Coordinate space of geometry handed to :func:`apply_result`.

    Named after ``API.md`` section 5.  Contours arriving from the daemon are
    ``CANVAS`` unless the daemon says otherwise; ``SOURCE`` is the user's own
    GIMP image, which is the only space this module writes into.
    """

    CANVAS = "canvas"
    UPLOADED = "uploaded"
    SOURCE = "source"

    ALL = (CANVAS, UPLOADED, SOURCE)


_CHANNEL_OP_NAMES = {
    SelectionOp.REPLACE: "REPLACE",
    SelectionOp.ADD: "ADD",
    SelectionOp.SUBTRACT: "SUBTRACT",
    SelectionOp.INTERSECT: "INTERSECT",
}


# --------------------------------------------------------------------------- #
# wire-shaped data, mirrored from _daemon/sam3gimpd/types.py (never imported)
# --------------------------------------------------------------------------- #
@dataclass
class CanvasTransform:
    """``canvas = image * scale + offset``  (API.md section 5).

    Clients must use the transform the daemon reports and must never assume the
    1008x1008 squash; a processor that letterboxed would set non-zero offsets.
    """

    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CanvasTransform":
        return cls(
            scale_x=float(d.get("scale_x", 1.0)),
            scale_y=float(d.get("scale_y", 1.0)),
            offset_x=float(d.get("offset_x", 0.0)),
            offset_y=float(d.get("offset_y", 0.0)),
        )

    def canvas_to_image(self, x: float, y: float) -> Tuple[float, float]:
        sx = self.scale_x or 1.0
        sy = self.scale_y or 1.0
        return ((x - self.offset_x) / sx, (y - self.offset_y) / sy)

    def image_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        return (x * self.scale_x + self.offset_x, y * self.scale_y + self.offset_y)


@dataclass
class Instance:
    """One returned mask: a cropped uint8 soft mask plus its canvas bbox."""

    instance_id: int
    score: float
    label: str
    bbox: Tuple[int, int, int, int]
    mask_width: int
    mask_height: int
    mask: bytes

    def __post_init__(self) -> None:
        self.bbox = tuple(int(v) for v in self.bbox)  # type: ignore[assignment]
        if len(self.bbox) != 4:
            raise ValueError("bbox must be [x0, y0, x1, y1]")
        self.mask_width = int(self.mask_width)
        self.mask_height = int(self.mask_height)
        if not isinstance(self.mask, (bytes, bytearray, memoryview)):
            raise TypeError("mask must be bytes-like")
        self.mask = bytes(self.mask)
        expected = self.mask_width * self.mask_height
        if len(self.mask) != expected:
            raise ValueError(
                "instance %s: mask is %d bytes, expected %d (%dx%d)"
                % (self.instance_id, len(self.mask), expected,
                   self.mask_width, self.mask_height))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], blob: Optional[bytes] = None) -> "Instance":
        """Build from one entry of a result header's ``instances`` array.

        ``blob`` is the frame's blob region (``API.md`` section 8.1); the slice
        is taken with ``blob_offset`` measured **from the start of that region**,
        never from the start of the body.
        """
        mask = d.get("mask")
        if mask is None:
            if blob is None:
                raise ValueError("instance %s: no mask bytes and no blob given"
                                 % d.get("instance_id"))
            start = int(d["blob_offset"])
            end = start + int(d["blob_length"])
            if end > len(blob):
                raise ValueError(
                    "instance %s: blob slice %d:%d exceeds the %d byte blob region"
                    % (d.get("instance_id"), start, end, len(blob)))
            mask = blob[start:end]
        return cls(
            instance_id=int(d.get("instance_id", 0)),
            score=float(d.get("score", 0.0)),
            label=str(d.get("label", "")),
            bbox=tuple(int(v) for v in d["bbox"]),  # type: ignore[arg-type]
            mask_width=int(d["mask_width"]),
            mask_height=int(d["mask_height"]),
            mask=mask,
        )


def _coerce_instance(obj: Any) -> Instance:
    """Accept an :class:`Instance`, a header dict, or any duck-typed object.

    ``client.py`` decodes the binary frame; rather than couple to its exact
    class this module takes anything carrying the wire field names.
    """
    if isinstance(obj, Instance):
        return obj
    if isinstance(obj, Mapping):
        return Instance.from_dict(obj)
    try:
        return Instance(
            instance_id=int(getattr(obj, "instance_id", 0)),
            score=float(getattr(obj, "score", 0.0)),
            label=str(getattr(obj, "label", "") or ""),
            bbox=tuple(int(v) for v in _bbox_of(obj)),  # type: ignore[arg-type]
            mask_width=int(getattr(obj, "mask_width")),
            mask_height=int(getattr(obj, "mask_height")),
            mask=getattr(obj, "mask"),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("cannot read %r as a mask instance: %s" % (type(obj).__name__, exc))


def _bbox_of(obj: Any) -> Sequence[int]:
    bbox = getattr(obj, "bbox")
    if hasattr(bbox, "to_list"):
        return bbox.to_list()
    if isinstance(bbox, Mapping):
        return [bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]]
    return list(bbox)


@dataclass
class MaskResult:
    """A decoded result frame plus the client-side bookkeeping section 9 needs.

    ``uploaded`` is the size actually POSTed to ``/images``; ``source`` is the
    user's GIMP image size.  The daemon never sees ``source`` -- mapping back to
    it is the client's business, and this class is where that happens.
    """

    instances: List[Instance] = field(default_factory=list)
    canvas_from_image: CanvasTransform = field(default_factory=CanvasTransform)
    uploaded: Tuple[int, int] = (0, 0)
    source: Tuple[int, int] = (0, 0)
    prompt_text: str = ""
    request_id: str = ""

    @classmethod
    def from_frame(
        cls,
        header: Mapping[str, Any],
        blob: bytes,
        source: Tuple[int, int],
        prompt_text: Optional[str] = None,
    ) -> "MaskResult":
        """Build from the JSON header + blob region of a binary result frame."""
        image = header.get("image") or {}
        uploaded = (int(image.get("width", 0)), int(image.get("height", 0)))
        if prompt_text is None:
            prompt = header.get("prompt") or {}
            prompt_text = str(prompt.get("text", "") or "")
        return cls(
            instances=[Instance.from_dict(d, blob) for d in header.get("instances", [])],
            canvas_from_image=CanvasTransform.from_dict(header.get("canvas_from_image") or {}),
            uploaded=uploaded,
            source=(int(source[0]), int(source[1])),
            prompt_text=prompt_text,
            request_id=str(header.get("request_id", "")),
        )

    # -- geometry ---------------------------------------------------------- #
    @property
    def _upscale(self) -> Tuple[float, float]:
        """The client's own downscale factor, inverted (API.md section 9 step 2)."""
        uw, uh = self.uploaded
        sw, sh = self.source
        u = (float(sw) / float(uw)) if uw and sw else 1.0
        v = (float(sh) / float(uh)) if uh and sh else 1.0
        return (u, v)

    def canvas_to_source(self, x: float, y: float) -> Tuple[float, float]:
        """model-canvas -> original-image, as floats (steps 1 and 2)."""
        ix, iy = self.canvas_from_image.canvas_to_image(x, y)
        u, v = self._upscale
        return (ix * u, iy * v)

    def point_to_source(self, x: float, y: float, space: str = Space.CANVAS) -> Tuple[float, float]:
        """Map a point in ``space`` into original-image coordinates."""
        if space == Space.SOURCE:
            return (float(x), float(y))
        if space == Space.UPLOADED:
            u, v = self._upscale
            return (float(x) * u, float(y) * v)
        if space == Space.CANVAS:
            return self.canvas_to_source(x, y)
        raise ValueError("unknown coordinate space %r" % (space,))

    def rect_for(self, inst: Instance) -> Tuple[int, int, int, int]:
        """Destination rectangle in original-image pixels: ``(x, y, w, h)``.

        Implements ``API.md`` section 9 step 3 exactly: round each edge
        independently and *then* derive the size, so adjacent instances tile
        without gaps.
        """
        x0, y0, x1, y1 = inst.bbox
        dx0, dy0 = self.canvas_to_source(x0, y0)
        dx1, dy1 = self.canvas_to_source(x1, y1)
        X0 = _round_half_up(dx0)
        X1 = _round_half_up(dx1)
        Y0 = _round_half_up(dy0)
        Y1 = _round_half_up(dy1)
        return (X0, Y0, max(1, X1 - X0), max(1, Y1 - Y0))


def _round_half_up(v: float) -> int:
    """Round half away from zero.

    Deliberately not :func:`round`: banker's rounding would make an edge landing
    exactly on ``.5`` depend on its parity, which is a miserable thing to debug
    from a screenshot.  This value never crosses the wire, so the daemon does
    not have to agree.
    """
    return int(math.floor(v + 0.5)) if v >= 0 else -int(math.floor(-v + 0.5))


# --------------------------------------------------------------------------- #
# options
# --------------------------------------------------------------------------- #
@dataclass
class PostOps:
    """Mask post-processing, applied before the output is materialised.

    ``threshold`` and ``edge_softness`` are byte-level (see the module
    docstring); ``hole_fill`` and ``min_area`` are byte-level too.  ``grow``,
    ``shrink``, ``smooth`` and ``feather`` are GIMP selection operations, so
    they run in C and behave exactly like the menu items of the same name.
    """

    #: ``inside = value >= threshold``; 128 is a raw logit of 0, the model's own
    #: binarisation point and therefore the default (API.md section 8.3).
    threshold: int = 128
    #: Width of the linear ramp around ``threshold``.  0 = hard binarisation.
    edge_softness: int = 32
    #: Fill interior holes in each mask before it is scaled.
    hole_fill: bool = False
    #: Drop instances whose area in *original-image* pixels is below this.
    min_area: float = 0.0
    #: Selection grow, in pixels.
    grow: int = 0
    #: Selection shrink, in pixels.
    shrink: int = 0
    #: Rounds corners and eats speckle: feather by this radius, then sharpen.
    smooth: float = 0.0
    #: Final feather radius, applied last so nothing sharpens it away again.
    feather: float = 0.0

    def normalised(self) -> "PostOps":
        return PostOps(
            threshold=max(1, min(255, int(self.threshold))),
            edge_softness=max(0, int(self.edge_softness)),
            hole_fill=bool(self.hole_fill),
            min_area=max(0.0, float(self.min_area)),
            grow=max(0, int(self.grow)),
            shrink=max(0, int(self.shrink)),
            smooth=max(0.0, float(self.smooth)),
            feather=max(0.0, float(self.feather)),
        )

    @property
    def has_selection_ops(self) -> bool:
        return bool(self.grow or self.shrink or self.smooth or self.feather)


@dataclass
class OutputOptions:
    mode: str = OutputMode.SELECTION
    selection_op: str = SelectionOp.REPLACE
    post: PostOps = field(default_factory=PostOps)
    #: Prepended to every generated channel/layer name, e.g. ``"sam3 "``.
    name_prefix: str = ""
    #: Layer masks / groups: work on a copy rather than the user's own layer.
    duplicate_layer: bool = True
    #: Layer group name; defaults to the prompt text.
    group_name: str = ""
    #: Put the user's selection back afterwards (modes other than SELECTION).
    restore_selection: bool = True
    channel_opacity: float = 50.0
    path_space: str = Space.CANVAS
    flush_displays: bool = True

    def validated(self) -> "OutputOptions":
        if self.mode not in OutputMode.ALL:
            raise ValueError("unknown output mode %r" % (self.mode,))
        if self.selection_op not in SelectionOp.ALL:
            raise ValueError("unknown selection operation %r" % (self.selection_op,))
        if self.path_space not in Space.ALL:
            raise ValueError("unknown coordinate space %r" % (self.path_space,))
        return OutputOptions(
            mode=self.mode,
            selection_op=self.selection_op,
            post=self.post.normalised(),
            name_prefix=self.name_prefix,
            duplicate_layer=bool(self.duplicate_layer),
            group_name=self.group_name,
            restore_selection=bool(self.restore_selection),
            channel_opacity=float(self.channel_opacity),
            path_space=self.path_space,
            flush_displays=bool(self.flush_displays),
        )


@dataclass
class AppliedResult:
    """What one Apply produced.  Everything here is inside a single undo step."""

    mode: str = OutputMode.SELECTION
    channels: List[Any] = field(default_factory=list)
    layers: List[Any] = field(default_factory=list)
    group: Any = None
    paths: List[Any] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    #: instance ids rejected by ``PostOps.min_area``
    dropped: List[int] = field(default_factory=list)
    selection_changed: bool = False
    #: Something the user should know about how the result was combined,
    #: e.g. that Intersect met an empty selection and was treated as Replace.
    note: str = ""

    def summary(self) -> str:
        n = len(self.names)
        noun = {
            OutputMode.SELECTION: "instance" if n != 1 else "instance",
            OutputMode.CHANNELS: "channel",
            OutputMode.LAYER_MASKS: "layer",
            OutputMode.LAYER_GROUPS: "layer",
            OutputMode.PATHS: "path",
        }.get(self.mode, "item")
        text = "%d %s%s" % (n, noun, "" if n == 1 else "s")
        if self.dropped:
            text += " (%d below minimum area)" % len(self.dropped)
        return text


# --------------------------------------------------------------------------- #
# byte-level mask operations  (pure Python, no GIMP, fully testable here)
# --------------------------------------------------------------------------- #
def threshold_table(threshold: int = 128, softness: int = 32) -> bytes:
    """A 256-entry ``bytes.translate`` table implementing the mask threshold.

    ``softness`` is the width of the linear ramp centred on ``threshold``:

    * ``softness == 0`` -> a hard step, ``value >= threshold`` becomes 255.
    * ``softness > 0``  -> values below ``threshold - softness/2`` become 0,
      above ``threshold + softness/2`` become 255, and the band between is a
      linear ramp.  The mask therefore stays *soft* going into GEGL's scaler,
      which is what keeps upsampled edges from stair-stepping (API.md 9 step 4),
      while still honouring the user's threshold exactly at the midpoint.
    """
    t = max(1, min(255, int(threshold)))
    s = max(0, int(softness))
    out = bytearray(256)
    if s == 0:
        for i in range(t, 256):
            out[i] = 255
        return bytes(out)
    lo = t - s / 2.0
    hi = t + s / 2.0
    span = hi - lo
    for i in range(256):
        if i <= lo:
            out[i] = 0
        elif i >= hi:
            out[i] = 255
        else:
            out[i] = int(round(255.0 * (i - lo) / span))
    return bytes(out)


def _binary_table(threshold: int) -> bytes:
    t = max(1, min(255, int(threshold)))
    return bytes(0 if i < t else 1 for i in range(256))


def apply_threshold(data: bytes, threshold: int = 128, softness: int = 32) -> bytes:
    """Remap a soft mask through :func:`threshold_table`.  One C-level call."""
    return bytes(data).translate(threshold_table(threshold, softness))


def mask_area(data: bytes, threshold: int = 128) -> int:
    """Number of pixels at or above ``threshold``.

    ``translate`` + ``count`` keeps this in C even for a 1008x1008 crop.
    """
    return bytes(data).translate(_binary_table(threshold)).count(1)


def fill_holes(data: bytes, width: int, height: int, threshold: int = 128) -> bytes:
    """Fill interior holes: below-threshold pixels unreachable from the border.

    Scanline flood fill over the crop, which is at model-canvas resolution and
    therefore at most about a megapixel.  Holes are set to 255 rather than to
    the local maximum, because a hole in a segmentation mask is a hole, not a
    gradient.
    """
    width = int(width)
    height = int(height)
    n = width * height
    if width <= 0 or height <= 0 or n == 0:
        return bytes(data)
    if len(data) != n:
        raise ValueError("fill_holes: %d bytes for a %dx%d mask" % (len(data), width, height))
    t = max(1, min(255, int(threshold)))
    buf = bytes(data)
    binary = buf.translate(_binary_table(t))
    background = binary.count(0)
    if background == 0:
        return buf  # solid mask, nothing to fill

    outside = bytearray(n)
    stack: List[Tuple[int, int]] = []

    def seed(x: int, y: int) -> None:
        i = y * width + x
        if not outside[i] and binary[i] == 0:
            stack.append((x, y))

    for x in range(width):
        seed(x, 0)
        if height > 1:
            seed(x, height - 1)
    for y in range(height):
        seed(0, y)
        if width > 1:
            seed(width - 1, y)

    while stack:
        x, y = stack.pop()
        row = y * width
        if outside[row + x] or binary[row + x]:
            continue
        xl = x
        while xl > 0 and not binary[row + xl - 1] and not outside[row + xl - 1]:
            xl -= 1
        xr = x
        while xr < width - 1 and not binary[row + xr + 1] and not outside[row + xr + 1]:
            xr += 1
        for xi in range(xl, xr + 1):
            outside[row + xi] = 1
        for ny in (y - 1, y + 1):
            if 0 <= ny < height:
                nrow = ny * width
                xi = xl
                while xi <= xr:
                    if not binary[nrow + xi] and not outside[nrow + xi]:
                        stack.append((xi, ny))
                        while xi <= xr and not binary[nrow + xi]:
                            xi += 1
                    xi += 1

    reached = outside.count(1)
    if reached == background:
        return buf  # every background pixel touches the border: no holes

    out = bytearray(buf)
    for i in range(n):
        if not binary[i] and not outside[i]:
            out[i] = 255
    return bytes(out)


# --------------------------------------------------------------------------- #
# naming
# --------------------------------------------------------------------------- #
def instance_name(
    inst: Instance,
    fallback: str = "",
    prefix: str = "",
) -> str:
    """A name a human would have typed: ``"car (0.93)"``.

    PCS instances carry the prompt text as their label.  PVS instances have an
    empty label (API.md section 8.2), so ``fallback`` -- normally the prompt
    text, otherwise ``"selection"`` -- is used.  Two candidates that come out
    with the same name are told apart by :func:`unique_name`.
    """
    label = (inst.label or "").strip()
    if not label:
        label = (fallback or "").strip() or "selection"
    name = "%s (%.2f)" % (label, max(0.0, min(1.0, float(inst.score))))
    if prefix:
        name = prefix + name
    return name


def unique_name(name: str, taken: Iterable[str]) -> str:
    """Disambiguate a duplicate with ``" #2"``, ``" #3"``, ...

    GIMP tolerates duplicate item names, but a Channels dock with three rows
    reading ``car (0.93)`` is useless to click on.
    """
    existing = set(taken)
    if name not in existing:
        return name
    n = 2
    while "%s #%d" % (name, n) in existing:
        n += 1
    return "%s #%d" % (name, n)


# --------------------------------------------------------------------------- #
# thin wrappers over the GIMP API
#
# Every call whose Python binding shape has varied between GIMP and GEGL
# builds goes through one of these, so there is exactly one place to fix it.
# --------------------------------------------------------------------------- #
def _channel_op(op: str):
    G = _gimp()
    return getattr(G.ChannelOps, _CHANNEL_OP_NAMES[op])


def _black():
    """A black ``Gegl.Color`` for ``Gimp.Channel.new``.

    GIMP 3.0 takes a ``GeglColor``; 2.x-era stubs and early 3.0 RCs took a
    ``Gimp.RGB``.  Neither is worth failing an Apply over, so fall back to None.
    """
    try:
        Gegl = _gegl()
        color = Gegl.Color.new("black")
        if color is not None:
            return color
    except Exception:  # noqa: BLE001 -- any binding shape at all is acceptable
        pass
    G = _gimp()
    rgb = getattr(G, "RGB", None)
    if rgb is not None:
        try:
            return rgb(0.0, 0.0, 0.0, 1.0)
        except Exception:  # noqa: BLE001
            try:
                return rgb()
            except Exception:  # noqa: BLE001
                pass
    return None


def _new_channel(image, name: str, width: int, height: int, opacity: float = 100.0):
    G = _gimp()
    channel = G.Channel.new(image, name, int(width), int(height), float(opacity), _black())
    if channel is None:
        raise OutputError("GIMP refused to create a %dx%d channel" % (width, height))
    return channel


def _write_gray(drawable, width: int, height: int, data: bytes,
                x: int = 0, y: int = 0) -> None:
    """Push ``width*height`` uint8 gray bytes into a drawable at ``(x, y)``.

    The byte layout in ``API.md`` section 8.3 -- row-major, top to bottom,
    ``width`` bytes per row, no padding -- is precisely what ``Gegl.Buffer.set``
    expects for the ``"Y' u8"`` format, so no repacking is needed.

    The introspected arity of ``gegl_buffer_set`` is not the same in every
    GEGL build, so the three plausible spellings are tried in order and the
    first that type-checks wins.
    """
    Gegl = _gegl()
    payload = bytes(data)
    buf = drawable.get_buffer()
    if buf is None:
        raise OutputError("could not get a GEGL buffer for %r" % (drawable,))
    rect = Gegl.Rectangle.new(int(x), int(y), int(width), int(height))
    attempts = (
        lambda: buf.set(rect, GRAY_FORMAT, payload),
        lambda: buf.set(rect, 0, GRAY_FORMAT, payload, 0),
        lambda: buf.set(rect, 0, GRAY_FORMAT, payload, int(width)),
    )
    last: Optional[BaseException] = None
    for attempt in attempts:
        try:
            attempt()
            break
        except TypeError as exc:
            last = exc
    else:
        raise OutputError(
            "Gegl.Buffer.set rejected every known signature (%s). The GEGL "
            "introspection binding is not what this build expects." % (last,))
    for method, args in (("flush", ()),
                         ("update", (int(x), int(y), int(width), int(height)))):
        fn = getattr(buf if method == "flush" else drawable, method, None)
        if callable(fn):
            try:
                fn(*args)
            except TypeError:
                pass


def _as_bytes(value: Any) -> bytes:
    """Whatever ``Gegl.Buffer.get`` handed back, as ``bytes``."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    get_data = getattr(value, "get_data", None)  # GLib.Bytes
    if callable(get_data):
        return bytes(get_data())
    if isinstance(value, (list, tuple)):
        return bytes(value)
    raise OutputError("Gegl.Buffer.get returned %r, not bytes" % (type(value).__name__,))


def _read_gray(drawable, width: int, height: int, x: int = 0, y: int = 0) -> bytes:
    """``width*height`` uint8 gray bytes from a drawable at ``(x, y)`` --
    the ``Gegl.Buffer.get`` call shape ``gimpbridge`` pins down for uploads."""
    Gegl = _gegl()
    buf = drawable.get_buffer()
    rect = Gegl.Rectangle.new(int(x), int(y), int(width), int(height))
    abyss = getattr(getattr(Gegl, "AbyssPolicy", None), "CLAMP", None)
    attempts = (
        lambda: buf.get(rect, 1.0, GRAY_FORMAT, abyss),
        lambda: buf.get(rect, 1.0, GRAY_FORMAT),
    )
    last: Optional[BaseException] = None
    for attempt in attempts:
        try:
            out = _as_bytes(attempt())
            break
        except TypeError as exc:
            last = exc
    else:
        raise OutputError("Gegl.Buffer.get rejected every known signature (%s)" % (last,))
    if len(out) != int(width) * int(height):
        raise OutputError("read %d bytes for a %dx%d mask" % (len(out), width, height))
    return out


def _scale_gray(data: bytes, width: int, height: int, dst_width: int, dst_height: int) -> bytes:
    """Resample a soft mask with GIMP's own scaler, never in Python.

    The crop is loaded into a throwaway GRAY image, ``Gimp.Image.scale`` does
    the work in C (with the interpolation ``_context_push`` set), and the
    result is read back.  This replaced ``Gimp.Item.transform_scale`` on a
    scratch *channel*: GIMP's ``gimp_channel_get_clip`` returns
    ``GIMP_TRANSFORM_RESIZE_CLIP`` unconditionally and ``gimp_channel_scale``
    pins a channel's offset to (0, 0), so a small channel "scaled into" a
    rectangle elsewhere in the image was clipped back to its own original box
    at the origin -- and every Apply selected nothing.  Layers are not
    clipped, and a whole-image scale is the one operation that cannot be.
    """
    if (int(dst_width), int(dst_height)) == (int(width), int(height)):
        return bytes(data)
    G = _gimp()
    temp = G.Image.new(int(width), int(height), G.ImageBaseType.GRAY)
    if temp is None:
        raise OutputError("GIMP refused to create a %dx%d scratch image" % (width, height))
    try:
        layer = G.Layer.new(temp, "sam3-mask", int(width), int(height),
                            G.ImageType.GRAY_IMAGE, 100.0, G.LayerMode.NORMAL)
        temp.insert_layer(layer, None, 0)
        _write_gray(layer, width, height, data)
        temp.scale(int(dst_width), int(dst_height))
        layers = temp.get_layers()
        if not layers:
            raise OutputError("the scratch image lost its layer while scaling")
        return _read_gray(layers[0], dst_width, dst_height)
    finally:
        delete = getattr(temp, "delete", None)
        if callable(delete):
            try:
                delete()
            except Exception:  # noqa: BLE001 -- cleanup must never mask the real error
                pass


def _ensure_zeroed(channel, width: int, height: int) -> None:
    """Make sure a fresh channel reads as 0 (unselected) everywhere.

    GEGL zero-fills unwritten tiles, so a new channel should already be
    empty; four one-pixel reads confirm it, and only a build that answers
    otherwise pays for an explicit zero write (row-chunked, so a huge image
    never needs a whole-channel buffer in memory at once).
    """
    Gegl = _gegl()
    buf = channel.get_buffer()
    abyss = getattr(getattr(Gegl, "AbyssPolicy", None), "CLAMP", None)
    corners = ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1),
               (width // 2, height // 2))
    dirty = False
    for cx, cy in corners:
        rect = Gegl.Rectangle.new(int(cx), int(cy), 1, 1)
        try:
            try:
                sample = _as_bytes(buf.get(rect, 1.0, GRAY_FORMAT, abyss))
            except TypeError:
                sample = _as_bytes(buf.get(rect, 1.0, GRAY_FORMAT))
        except Exception:  # noqa: BLE001 -- cannot read: zero it to be safe
            dirty = True
            break
        if any(sample):
            dirty = True
            break
    if not dirty:
        return
    rows = max(1, 4 * 1024 * 1024 // max(1, width))  # ~4 MiB per write
    zeros = bytes(width * rows)
    for y0 in range(0, height, rows):
        h = min(rows, height - y0)
        _write_gray(channel, width, h, zeros if h == rows else bytes(width * h), 0, y0)


def _clip_rows(data: bytes, width: int, height: int, x: int, y: int,
               bound_w: int, bound_h: int) -> Optional[Tuple[bytes, int, int, int, int]]:
    """The part of a ``width*height`` block placed at ``(x, y)`` that falls
    inside ``bound_w*bound_h``: ``(payload, px, py, w, h)``, or ``None``.
    Row slices only; no per-pixel Python."""
    crop_x = max(0, -x)
    crop_y = max(0, -y)
    crop_w = min(width - crop_x, bound_w - max(x, 0))
    crop_h = min(height - crop_y, bound_h - max(y, 0))
    if crop_w <= 0 or crop_h <= 0:
        return None
    if (crop_x, crop_y, crop_w, crop_h) == (0, 0, width, height):
        return (bytes(data), x, y, width, height)
    out = bytearray(crop_w * crop_h)
    for row in range(crop_h):
        src = (crop_y + row) * width + crop_x
        out[row * crop_w:(row + 1) * crop_w] = data[src:src + crop_w]
    return (bytes(out), max(x, 0), max(y, 0), crop_w, crop_h)


def _set_offsets(item, x: int, y: int) -> None:
    fn = getattr(item, "set_offsets", None)
    if callable(fn):
        fn(int(x), int(y))


def _get_offsets(item) -> Tuple[int, int]:
    fn = getattr(item, "get_offsets", None)
    if not callable(fn):
        return (0, 0)
    res = fn()
    if isinstance(res, tuple):
        if len(res) == 3:  # (success, x, y) -- the GI out-parameter shape
            return (int(res[1]), int(res[2]))
        if len(res) == 2:
            return (int(res[0]), int(res[1]))
    return (0, 0)


def _selection_select(image, op: str, item) -> None:
    """Combine ``item`` into the selection with ``op``.

    ``gimp_image_select_item`` dispatches on item type: a layer contributes its
    alpha, a channel its contents (grey values and all, so a feathered channel
    stays feathered), a path its outline.
    """
    G = _gimp()
    select_item = getattr(image, "select_item", None)
    if callable(select_item):
        select_item(_channel_op(op), item)
        return
    if op == SelectionOp.REPLACE:
        G.Selection.load(item)
        return
    raise OutputError("this GIMP build has no Gimp.Image.select_item(); "
                      "cannot combine a selection with %r" % (op,))


def _selection_save(image):
    """Snapshot the selection into a new channel, already inserted in the image.

    ``gimp-selection-save`` inserts it with undo recorded, so a snapshot that
    is only scratch must be taken with undo frozen (:meth:`_Scratch.save_selection`).
    """
    G = _gimp()
    channel = G.Selection.save(image)
    if channel is None:
        raise OutputError("Gimp.Selection.save() returned nothing")
    return channel


def _selection_load(image, channel) -> None:
    _selection_select(image, SelectionOp.REPLACE, channel)


def _selection_is_empty(image) -> Optional[bool]:
    """``Gimp.Selection.is_empty``, or ``None`` when the binding lacks it."""
    G = _gimp()
    fn = getattr(getattr(G, "Selection", None), "is_empty", None)
    if not callable(fn):
        return None
    try:
        return bool(fn(image))
    except Exception:  # noqa: BLE001
        return None


def _selection_bounds_text(image) -> str:
    """``"empty"`` or ``"x0,y0-x1,y1"`` for a log line; never raises."""
    G = _gimp()
    fn = getattr(getattr(G, "Selection", None), "bounds", None)
    if not callable(fn):
        return "?"
    try:
        res = tuple(fn(image))
    except Exception as exc:  # noqa: BLE001
        return "? (%s)" % exc
    # GIMP 3 returns (ok, non_empty, x1, y1, x2, y2); older bindings drop ok.
    if len(res) == 6:
        res = res[1:]
    if len(res) == 5:
        return "%d,%d-%d,%d" % tuple(int(v) for v in res[1:]) if res[0] else "empty"
    return repr(res)


def _remember_selected_items(image) -> Tuple[List[Any], List[Any]]:
    """``(layers, channels)`` selected in the dialogs before Apply touches them."""
    out: List[List[Any]] = []
    for name in ("get_selected_layers", "get_selected_channels"):
        fn = getattr(image, name, None)
        try:
            out.append(list(fn()) if callable(fn) else [])
        except Exception:  # noqa: BLE001
            out.append([])
    return (out[0], out[1])


def _restore_selected_items(image, remembered: Tuple[List[Any], List[Any]], fallback) -> None:
    """Put the Layers/Channels dialog selection back after the scratch channel is gone.

    This is not cosmetic.  ``gimp_selection_boundary`` draws marching ants
    only when a channel is selected or at least one layer is: while the
    scratch channel was inserted it *was* the selected drawable (so the ants
    appeared), and removing it left the user's image with nothing selected,
    so GIMP drew no outline at all -- "the selection unselects at the end",
    with ``Gimp.Selection.bounds`` insisting the mask was still there.

    GIMP keeps selected layers and selected channels mutually exclusive
    (selecting either deselects the other), so at most one of the two lists
    is put back; callers that want new layers selected pass an empty channel
    list.
    """
    layers, channels = remembered
    layers = [l for l in layers if _still_attached(l)]
    channels = [c for c in channels if _still_attached(c)]
    if not layers and not channels:
        if fallback is not None and _still_attached(fallback):
            layers = [fallback]
        else:
            default = _default_layer(image)
            layers = [default] if default is not None else []
    set_layers = getattr(image, "set_selected_layers", None)
    set_channels = getattr(image, "set_selected_channels", None)
    try:
        if layers and callable(set_layers):
            set_layers(layers)
        if channels and callable(set_channels):
            set_channels(channels)
    except Exception:  # noqa: BLE001 -- a stale item is not worth failing an Apply
        pass


def _still_attached(item) -> bool:
    fn = getattr(item, "is_valid", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:  # noqa: BLE001
            return True
    return True


def _apply_selection_post_ops(image, post: PostOps) -> None:
    """grow -> shrink -> smooth -> feather, on the current selection.

    Order is deliberate.  grow-then-shrink is a morphological close, which is
    what a user who typed both actually wants.  ``smooth`` is GIMP's classic
    feather-then-sharpen trick: it rounds corners and eats speckle, and doing it
    before ``feather`` means the sharpen does not undo the feather.
    """
    G = _gimp()
    if post.grow:
        G.Selection.grow(image, int(post.grow))
    if post.shrink:
        G.Selection.shrink(image, int(post.shrink))
    if post.smooth:
        G.Selection.feather(image, float(post.smooth))
        G.Selection.sharpen(image)
    if post.feather:
        G.Selection.feather(image, float(post.feather))


# --------------------------------------------------------------------------- #
# instance -> channel
# --------------------------------------------------------------------------- #
def _source_area(inst: Instance, rect: Tuple[int, int, int, int], post: PostOps) -> float:
    """Mask area measured in *original-image* pixels.

    The crop is measured at model-canvas resolution and rescaled by the area
    ratio of the destination rectangle, which is exact enough to filter
    speckle and costs one ``translate``/``count`` pass.
    """
    canvas_area = mask_area(inst.mask, post.threshold)
    crop = inst.mask_width * inst.mask_height
    if crop <= 0:
        return 0.0
    _, _, dw, dh = rect
    return canvas_area * (float(dw * dh) / float(crop))


class _undo_frozen:
    """``Gimp.Image.undo_freeze`` / ``undo_thaw`` around scratch-only work.

    Every insertion and removal of an image-sized scratch channel otherwise
    pushes an undo step holding a copy of it; on a 12-megapixel photo with
    sixty instances that was gigabytes of undo memory for channels the user
    never sees.  Frozen, those steps are skipped -- and so is every change
    made while the selection is only a work surface, because the work is
    finished by putting the user's selection back.

    The rule every caller keeps: a frozen window may leave behind only
    scratch channels (which a later frozen window removes) and a selection
    that a later frozen window restores.  Anything thawed in between -- new
    layers, masks, channels, the one ``select_item`` that commits a new
    selection -- is recorded against state that is still true when it is
    undone, which is what GIMP asks of freeze/thaw.  A GIMP without the calls
    simply runs unfrozen.
    """

    def __init__(self, image):
        self.image = image
        self._active = False

    def __enter__(self):
        freeze = getattr(self.image, "undo_freeze", None)
        if callable(freeze):
            try:
                freeze()
                self._active = True
            except Exception:  # noqa: BLE001
                self._active = False
        return self

    def __exit__(self, *_exc):
        if self._active:
            try:
                self.image.undo_thaw()
            except Exception:  # noqa: BLE001
                pass
        return False


class _Scratch:
    """What one Apply makes along the way, and how to take it out again.

    ``items`` are the hidden work channels.  They live in the image only
    long enough to be written and loaded into the selection, and are
    inserted, named and removed with undo frozen, so none of that reaches
    the undo stack.

    ``original`` is the user's selection, saved the first time a mode
    borrows the selection as a work surface (:meth:`borrow_selection`) and
    loaded back by :meth:`put_back_selection`.

    ``created`` lists ``(kind, item)`` for every output already added, so an
    Apply that fails halfway can remove them again (:func:`_roll_back`).
    """

    def __init__(self, image):
        self.image = image
        self.items: List[Any] = []
        self.created: List[Tuple[str, Any]] = []
        self.original = None

    def add(self, channel):
        self.items.append(channel)
        return channel

    def save_selection(self, name: str):
        """The selection as a hidden scratch channel, saved with undo frozen."""
        with _undo_frozen(self.image):
            channel = self.add(_selection_save(self.image))
            channel.set_name(name)
            _set_visible(channel, False)
        return channel

    def borrow_selection(self) -> None:
        """Save the user's selection before the selection is first used as a
        work surface; every later call is free."""
        if self.original is None:
            self.original = self.save_selection("sam3-previous-selection")

    def put_back_selection(self) -> None:
        """Load the user's selection back, frozen: after a borrow, nothing
        about the selection has changed as far as the undo stack knows."""
        if self.original is not None:
            with _undo_frozen(self.image):
                _selection_load(self.image, self.original)

    def discard(self) -> None:
        with _undo_frozen(self.image):
            for channel in reversed(self.items):
                try:
                    self.image.remove_channel(channel)
                except Exception:  # noqa: BLE001 -- cleanup must never mask the real error
                    pass
        self.items = []


def _placed_instance(image, result: MaskResult, inst: Instance, post: PostOps
                     ) -> Optional[Tuple[bytes, int, int, int, int]]:
    """One instance as ``(bytes, x, y, w, h)`` in image space, clipped to the
    image, or ``None`` when nothing of it is inside.

    ``API.md`` section 9, in order: hole-fill the crop, scale the *soft* mask
    to the destination rectangle in GIMP, threshold at destination
    resolution.
    """
    x, y, w, h = result.rect_for(inst)
    soft = inst.mask
    if post.hole_fill:
        soft = fill_holes(soft, inst.mask_width, inst.mask_height, post.threshold)
    soft = _scale_gray(soft, inst.mask_width, inst.mask_height, w, h)
    data = apply_threshold(soft, post.threshold, post.edge_softness)
    image_w, image_h = int(image.get_width()), int(image.get_height())
    return _clip_rows(data, w, h, x, y, image_w, image_h)


def _max_bytes(a: bytes, b: bytes) -> bytes:
    """Byte-wise ``max`` of two equal-length byte strings, without numpy.

    ``bytes(map(max, a, b))`` costs about 0.7 s per 9 megapixels, on the GTK
    thread.  Here both strings are read as one big integer each and compared
    eight bits at a time, all lanes at once: every line below is a single
    C-level pass over the integers, and the same 9 megapixels take about
    0.13 s.  The subtraction never borrows across a lane, because it only
    ever takes the low seven bits of ``b`` from ``a`` with bit 7 forced on;
    bit 7 of each lane then says whether ``a``'s low bits are the larger,
    and the top bits settle the rest.
    """
    n = len(a)
    if n != len(b):
        raise ValueError("cannot combine %d bytes with %d" % (n, len(b)))
    if not n or a == b:
        return bytes(a)
    x = int.from_bytes(a, "little")
    y = int.from_bytes(b, "little")
    top = int.from_bytes(b"\x80" * n, "little")
    low = top - (top >> 7)                              # 0x7f in every lane
    low_ge = ((x | top) - (y & low)) & top              # a's low 7 bits >= b's
    a_ge = (x & ~y & top) | (low_ge & ~(x ^ y))         # a >= b, per lane
    take_a = (a_ge >> 7) * 0xFF
    return (y ^ ((x ^ y) & take_a)).to_bytes(n, "little")


def _intersection(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
                  ) -> Optional[Tuple[int, int, int, int]]:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1 = min(a[0] + a[2], b[0] + b[2])
    y1 = min(a[1] + a[3], b[1] + b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def _merge_with_written(channel, payload: bytes, rect: Tuple[int, int, int, int],
                        written: Sequence[Tuple[int, int, int, int]]) -> bytes:
    """``payload`` for ``rect``, maxed with what the channel already holds.

    Only the bounding box of ``rect``'s overlaps with the rectangles written
    so far is read back and combined: everywhere else inside ``rect`` the
    channel is still zero, and ``max(0, v)`` is ``v``.
    """
    overlaps = [box for box in (_intersection(rect, done) for done in written) if box]
    if not overlaps:
        return payload
    bx0 = min(o[0] for o in overlaps)
    by0 = min(o[1] for o in overlaps)
    bx1 = max(o[0] + o[2] for o in overlaps)
    by1 = max(o[1] + o[3] for o in overlaps)
    bw, bh = bx1 - bx0, by1 - by0
    existing = _read_gray(channel, bw, bh, bx0, by0)
    if existing.count(0) == len(existing):
        return payload
    px, py, cw, _ch = rect
    ox, oy = bx0 - px, by0 - py
    if (ox, oy, bw) == (0, 0, cw) and bh * cw == len(payload):
        return _max_bytes(existing, payload)
    block = bytearray(bw * bh)
    for row in range(bh):
        src = (oy + row) * cw + ox
        block[row * bw:(row + 1) * bw] = payload[src:src + bw]
    merged = _max_bytes(existing, bytes(block))
    out = bytearray(payload)
    for row in range(bh):
        dst = (oy + row) * cw + ox
        out[dst:dst + bw] = merged[row * bw:(row + 1) * bw]
    return bytes(out)


def _union_channel(image, result: MaskResult, instances: Sequence[Instance],
                   post: PostOps, scratch: _Scratch):
    """The union of ``instances`` as **one** image-sized scratch channel.

    One channel, not one per instance: every inserted channel is image-sized
    and every ``select_item`` pushes an undo copy of the selection, so N
    instances on a large photo cost N x (channel + mask copy) of undo memory
    -- gigabytes on a large photo -- and 6N PDB round trips.
    Here the masks are combined in Python where their rectangles overlap
    (byte-wise max, so a neighbour's zero edge never erases a mask already
    written) and GIMP is asked to select once.

    The channel is image-sized on purpose: GIMP clips every transform of a
    channel to the channel's own bounds and pins its offset at the origin
    (``gimp_channel_get_clip`` / ``gimp_channel_scale``), so a crop-sized
    scratch channel can never be moved or grown into place.
    """
    channel = _scratch_channel(image, scratch)
    with _undo_frozen(image):
        written: List[Tuple[int, int, int, int]] = []
        for inst in instances:
            placed = _placed_instance(image, result, inst, post)
            if placed is None:
                continue
            payload, px, py, cw, ch = placed
            rect = (px, py, cw, ch)
            payload = _merge_with_written(channel, payload, rect, written)
            _write_gray(channel, cw, ch, payload, px, py)
            written.append(rect)
    return channel


def _scratch_channel(image, scratch: _Scratch):
    """One hidden, zeroed, image-sized scratch channel, inserted frozen."""
    image_w, image_h = int(image.get_width()), int(image.get_height())
    with _undo_frozen(image):
        channel = _new_channel(image, "sam3-scratch", image_w, image_h)
        image.insert_channel(channel, None, 0)
        scratch.add(channel)
        _set_visible(channel, False)
        # Never Gimp.Drawable.fill(FillType.TRANSPARENT) here: GIMP fills with
        # the context *background colour* and only drops alpha when the
        # drawable has some.  A channel has none, the default background is
        # white, and so the whole channel read 255 -- every Apply selected the
        # entire image.
        _ensure_zeroed(channel, image_w, image_h)
    return channel


def _each_instance_selected(image, result: MaskResult, instances: Sequence[Instance],
                            post: PostOps, scratch: _Scratch):
    """Yield each instance with the selection set to it, reusing ONE channel.

    Channels and layer-group modes need the selection per instance.  Doing
    that through ``_selection_from_instances`` cost a fresh image-sized
    scratch channel per instance; here the previous instance's rectangle is
    zeroed and the next one written into the same channel, and GIMP is asked
    to select once per instance.  The selection is borrowed (see
    :class:`_Scratch`) and set with undo frozen; whatever the caller builds
    from it between yields is recorded as usual.  Instances with nothing
    inside the image are skipped.
    """
    channel = None
    previous: Optional[Tuple[int, int, int, int]] = None
    for inst in instances:
        placed = _placed_instance(image, result, inst, post)
        if placed is None:
            continue
        payload, px, py, cw, ch = placed
        if channel is None:
            scratch.borrow_selection()
            channel = _scratch_channel(image, scratch)
        with _undo_frozen(image):
            if previous is not None:
                _write_gray(channel, previous[2], previous[3],
                            bytes(previous[2] * previous[3]), previous[0], previous[1])
            _write_gray(channel, cw, ch, payload, px, py)
            _selection_select(image, SelectionOp.REPLACE, channel)
            _apply_selection_post_ops(image, post)
        previous = (px, py, cw, ch)
        yield inst


def _set_visible(item, visible: bool) -> None:
    fn = getattr(item, "set_visible", None)
    if callable(fn):
        try:
            fn(bool(visible))
        except TypeError:
            pass


def _selection_from_instances(
    image,
    result: MaskResult,
    instances: Sequence[Instance],
    post: PostOps,
    scratch: _Scratch,
) -> None:
    """Set the selection to the union of ``instances``, then run selection ops.

    Scratch work: the selection is borrowed first and changed only with undo
    frozen, so the caller either puts it back or commits the result.
    """
    scratch.borrow_selection()
    channel = _union_channel(image, result, instances, post, scratch)
    with _undo_frozen(image):
        _selection_select(image, SelectionOp.REPLACE, channel)
        _apply_selection_post_ops(image, post)


# --------------------------------------------------------------------------- #
# output modes
# --------------------------------------------------------------------------- #
def _mode_selection(image, result, instances, options, scratch, applied) -> None:
    """Combine the union of the instances with the user's existing selection.

    All of the combining happens on scratch -- the selection borrowed as a
    work surface with undo frozen, the finished result saved into a scratch
    channel, the user's selection put back -- and then one thawed
    ``select_item`` loads the result.  That call is the only selection change
    on the undo stack, so Ctrl+Z brings back exactly the selection the user
    had.  With Replace and no selection post-ops there is nothing to combine
    and the union channel is loaded directly.
    """
    G = _gimp()
    post = options.post
    op = options.selection_op

    everything = False
    if op in (SelectionOp.INTERSECT, SelectionOp.SUBTRACT) and _selection_is_empty(image):
        # GIMP's convention: no selection means the whole image is fair game.
        # Taken literally, Intersect with an empty selection is empty and
        # Subtract from it is empty -- the user watched every instance get
        # selected and then "they all unselect at the end".  Follow the
        # convention instead: intersecting with everything is the new
        # selection; subtracting from everything is its inverse.
        if op == SelectionOp.INTERSECT:
            op = SelectionOp.REPLACE
            applied.note = "there was no selection to intersect with, so the result replaces it"
        elif callable(getattr(getattr(G, "Selection", None), "all", None)):
            everything = True
            applied.note = "there was no selection to subtract from, so everything else is selected"

    if not instances:
        # Nothing survived filtering.  REPLACE means "clear" and subtracting
        # nothing from everything is everything; any other op against an
        # empty operand leaves the user's selection as it was.
        if op == SelectionOp.REPLACE:
            G.Selection.none(image)
        elif everything:
            G.Selection.all(image)
        applied.selection_changed = True
        return

    result_channel = _union_channel(image, result, instances, post, scratch)
    if op != SelectionOp.REPLACE or post.has_selection_ops:
        scratch.borrow_selection()
        with _undo_frozen(image):
            _selection_select(image, SelectionOp.REPLACE, result_channel)
            _apply_selection_post_ops(image, post)
            if op != SelectionOp.REPLACE:
                union = scratch.save_selection("sam3-union")
                if everything:
                    G.Selection.all(image)
                else:
                    _selection_load(image, scratch.original)
                _selection_select(image, op, union)
            result_channel = scratch.save_selection("sam3-result")
        scratch.put_back_selection()
    # The one selection change undo records, made while the user's own
    # selection is current: that is what the undo step stores.
    _selection_select(image, SelectionOp.REPLACE, result_channel)

    applied.selection_changed = True
    applied.names = [instance_name(i, result.prompt_text, options.name_prefix)
                     for i in instances]


def _mode_channels(image, result, instances, options, scratch, applied) -> None:
    """One saved channel per instance, named ``"car (0.93)"`` and top of the dock.

    The mask bytes go straight into each output channel.  Only when a
    selection operation (grow, shrink, smooth, feather) is asked for does an
    instance take the selection round trip, because those are GIMP selection
    operations.  The old route -- scratch channel, select, ``Selection.save``
    -- made two image-sized channels and an undo copy per instance.
    """
    post = options.post
    image_w, image_h = int(image.get_width()), int(image.get_height())

    def finish(channel, inst: Instance) -> None:
        name = unique_name(
            instance_name(inst, result.prompt_text, options.name_prefix),
            applied.names,
        )
        channel.set_name(name)
        try:
            channel.set_opacity(float(options.channel_opacity))
        except (AttributeError, TypeError):
            pass
        _set_visible(channel, False)
        applied.channels.append(channel)
        applied.names.append(name)

    if post.has_selection_ops:
        for inst in _each_instance_selected(image, result, instances, post, scratch):
            channel = _selection_save(image)
            scratch.created.append(("channel", channel))
            finish(channel, inst)
        return

    for inst in instances:
        placed = _placed_instance(image, result, inst, post)
        if placed is None:
            continue
        payload, px, py, cw, ch = placed
        channel = _new_channel(image, "sam3", image_w, image_h)
        image.insert_channel(channel, None, 0)
        scratch.created.append(("channel", channel))
        _ensure_zeroed(channel, image_w, image_h)
        _write_gray(channel, cw, ch, payload, px, py)
        finish(channel, inst)


def _duplicate_for_mask(image, layer, name: str, scratch: _Scratch, parent=None,
                        position: int = -1):
    """Copy ``layer``, keep its offsets, insert it, and give it an alpha channel.

    Offsets matter: a layer that sits at (120, 40) in the image must keep
    sitting there, and its mask is in layer space, which is why the mask is
    built from the selection (GIMP does the image-space -> layer-space
    translation itself).
    """
    dup = layer.copy()
    if dup is None:
        raise OutputError("could not duplicate layer %r" % (layer,))
    dup.set_name(name)
    off_x, off_y = _get_offsets(layer)
    _set_offsets(dup, off_x, off_y)
    add_alpha = getattr(dup, "add_alpha", None)
    if callable(add_alpha):
        add_alpha()
    if position < 0:
        position = _item_position(image, layer)
    image.insert_layer(dup, parent, position)
    scratch.created.append(("layer", dup))
    _set_offsets(dup, off_x, off_y)
    return dup


def _item_position(image, item) -> int:
    fn = getattr(image, "get_item_position", None)
    if callable(fn):
        try:
            return int(fn(item))
        except Exception:  # noqa: BLE001
            return 0
    return 0


def _mask_from_selection(layer) -> None:
    """Add a layer mask taken from the current selection.

    ``AddMaskType.SELECTION`` is what handles layer offsets for us: the mask is
    layer-sized and layer-positioned, and GIMP clips the image-space selection
    into it.  Doing this by hand with buffer writes would mean reimplementing
    that translation, and getting it wrong on any layer not at (0, 0).

    The new mask is made before a mask the layer already has is discarded,
    so a failure to make one leaves the old mask where it was.
    """
    G = _gimp()
    mask = layer.create_mask(G.AddMaskType.SELECTION)
    if mask is None:
        raise OutputError("could not create a layer mask for %r" % (layer,))
    existing = None
    get_mask = getattr(layer, "get_mask", None)
    if callable(get_mask):
        existing = get_mask()
    if existing is not None:
        layer.remove_mask(G.MaskApplyMode.DISCARD)
    layer.add_mask(mask)


def _mode_layer_masks(image, result, instances, options, scratch, applied) -> None:
    """The union of the instances, as a mask on one duplicate of the source layer."""
    layer = _require_layer(image, options)
    post = options.post
    if not instances:
        return
    _selection_from_instances(image, result, instances, post, scratch)
    label = _union_label(instances, result)
    name = unique_name(
        "%s%s" % (options.name_prefix, label),
        [_name_of(l) for l in _all_layers(image)],
    )
    parent = layer.get_parent() if hasattr(layer, "get_parent") else None
    if options.duplicate_layer:
        target = _duplicate_for_mask(image, layer, name, scratch, parent=parent)
        _mask_from_selection(target)
    else:
        target = layer
        _mask_from_selection(target)
        scratch.created.append(("mask", target))
    applied.layers.append(target)
    applied.names.append(name)


def _mode_layer_groups(image, result, instances, options, scratch, applied) -> None:
    """One masked copy of the source layer per instance, inside a named group."""
    G = _gimp()
    layer = _require_layer(image, options)
    post = options.post
    group_name = options.group_name or ("%s%s" % (
        options.name_prefix, _union_label(instances, result)))
    group = _new_group(G, image, group_name)
    parent = layer.get_parent() if hasattr(layer, "get_parent") else None
    image.insert_layer(group, parent, _item_position(image, layer))
    scratch.created.append(("layer", group))
    applied.group = group
    for inst in _each_instance_selected(image, result, instances, post, scratch):
        name = unique_name(
            instance_name(inst, result.prompt_text, options.name_prefix),
            applied.names,
        )
        dup = _duplicate_for_mask(image, layer, name, scratch, parent=group,
                                  position=len(applied.layers))
        _mask_from_selection(dup)
        applied.layers.append(dup)
        applied.names.append(name)
    if not applied.layers:
        try:
            image.remove_layer(group)
            scratch.created.remove(("layer", group))
        except Exception:  # noqa: BLE001
            pass
        applied.group = None


def _new_group(G, image, name: str):
    """``Gimp.GroupLayer.new`` gained a name argument during the 3.0 cycle."""
    try:
        group = G.GroupLayer.new(image, name)
    except TypeError:
        group = G.GroupLayer.new(image)
        group.set_name(name)
    if group is None:
        raise OutputError("could not create the layer group %r" % (name,))
    if group.get_name() != name:
        group.set_name(name)
    return group


def _union_label(instances: Sequence[Instance], result: MaskResult) -> str:
    """A name for a group or a union: the prompt, else the shared label."""
    text = (result.prompt_text or "").strip()
    if text:
        return text
    labels = {(i.label or "").strip() for i in instances}
    labels.discard("")
    if len(labels) == 1:
        return labels.pop()
    return "sam3 selection"


def _name_of(item) -> str:
    fn = getattr(item, "get_name", None)
    return str(fn()) if callable(fn) else ""


def _all_layers(image) -> List[Any]:
    fn = getattr(image, "get_layers", None)
    return list(fn()) if callable(fn) else []


def _require_layer(image, options):
    """The source layer :func:`apply_result` resolved (see :func:`_as_layer`)."""
    layer = getattr(options, "_layer", None)
    if layer is None:
        raise OutputError("output mode %r needs a source layer" % (options.mode,))
    return layer


# --------------------------------------------------------------------------- #
# vector paths
# --------------------------------------------------------------------------- #
def _path_class(G):
    """GIMP 3.0 renamed ``Gimp.Vectors`` to ``Gimp.Path`` late in the cycle."""
    cls = getattr(G, "Path", None) or getattr(G, "Vectors", None)
    if cls is None:
        raise OutputError("this GIMP build exposes neither Gimp.Path nor Gimp.Vectors")
    return cls


def _stroke_type(G):
    for name in ("PathStrokeType", "VectorsStrokeType"):
        enum = getattr(G, name, None)
        if enum is not None and hasattr(enum, "BEZIER"):
            return enum.BEZIER
    return 0


def _insert_path(image, path) -> None:
    for name in ("insert_path", "insert_vectors"):
        fn = getattr(image, name, None)
        if callable(fn):
            fn(path, None, 0)
            return
    raise OutputError("this GIMP build has no Gimp.Image.insert_path()")


def _remove_path(image, path) -> None:
    for name in ("remove_path", "remove_vectors"):
        fn = getattr(image, name, None)
        if callable(fn):
            fn(path)
            return


def normalise_stroke(stroke: Any) -> Tuple[List[float], bool]:
    """Coerce one contour into ``(control_points, closed)``.

    ``_daemon/API.md`` v1.0 does not pin a wire shape for contours (the
    ``"contours"`` capability exists, but no endpoint returns them yet), so this
    accepts the plausible spellings rather than betting on one:

    * ``{"control_points": [lx, ly, ax, ay, rx, ry, ...], "closed": bool}`` --
      the bezier form, groups of three points per anchor, exactly what
      ``gimp_path_stroke_new_from_points`` wants;
    * ``{"points": [x, y, ...]}`` or ``{"polyline": [(x, y), ...]}`` -- a plain
      polygon, expanded here into degenerate triples (both handles sitting on
      the anchor), which draws as straight segments;
    * a bare flat sequence, read as control points when its length is a multiple
      of six and as a polyline otherwise;
    * a bare sequence of ``(x, y)`` pairs.
    """
    closed = True
    data: Any = stroke
    if isinstance(stroke, Mapping):
        closed = bool(stroke.get("closed", True))
        for key in ("control_points", "controlpoints", "bezier"):
            if key in stroke:
                return ([float(v) for v in stroke[key]], closed)
        for key in ("points", "polyline", "polygon", "contour"):
            if key in stroke:
                data = stroke[key]
                break
        else:
            raise ValueError("stroke dict has no point data: %r" % (sorted(stroke),))

    flat = _flatten_points(data)
    if len(flat) % 2:
        raise ValueError("stroke has an odd number of coordinates (%d)" % len(flat))
    if isinstance(stroke, Mapping) or len(flat) % 6:
        return (_polyline_to_control_points(flat), closed)
    # A bare flat sequence whose length divides by six is taken as the bezier
    # control-point triples the contours module emits.
    return (flat, closed)


def _flatten_points(data: Any) -> List[float]:
    out: List[float] = []
    for item in data:
        if isinstance(item, (int, float)):
            out.append(float(item))
        else:
            for v in item:
                out.append(float(v))
    return out


def _polyline_to_control_points(flat: Sequence[float]) -> List[float]:
    out: List[float] = []
    for i in range(0, len(flat), 2):
        x, y = float(flat[i]), float(flat[i + 1])
        out.extend((x, y, x, y, x, y))
    return out


def build_path(image, name: str, strokes: Iterable[Any], result: MaskResult,
               space: str = Space.CANVAS):
    """Create one ``Gimp.Path`` from a list of contours and insert it."""
    G = _gimp()
    path = _path_class(G).new(image, name)
    if path is None:
        raise OutputError("could not create the path %r" % (name,))
    stroke_type = _stroke_type(G)
    count = 0
    for stroke in strokes:
        points, closed = normalise_stroke(stroke)
        if len(points) < 12:
            # Fewer than two anchors is not a stroke; GIMP would take it, and
            # the user would get an invisible path item they have to delete.
            continue
        mapped: List[float] = []
        for i in range(0, len(points), 2):
            sx, sy = result.point_to_source(points[i], points[i + 1], space)
            mapped.extend((sx, sy))
        path.stroke_new_from_points(stroke_type, mapped, bool(closed))
        count += 1
    if count == 0:
        return None
    _insert_path(image, path)
    # A new path is hidden by default and lives only in the Paths dialog, so
    # an Apply that made twenty of them looked like it did nothing at all.
    # Shown, it draws its outline on the canvas in the path colour.
    _set_visible(path, True)
    return path


def _select_paths(image, paths: Sequence[Any]) -> None:
    """Make the new paths the selected ones in the Paths dialog (GIMP 3 API,
    with the 2.10-era spelling as a fallback).  Best effort."""
    if not paths:
        return
    for name in ("set_selected_paths", "set_selected_vectors"):
        fn = getattr(image, name, None)
        if callable(fn):
            try:
                fn(list(paths))
            except Exception:  # noqa: BLE001
                pass
            return


def _mode_paths(image, result, instances, options, scratch, applied) -> None:
    contours = getattr(options, "_contours", None) or {}
    for index, inst in enumerate(instances):
        strokes = _contours_for(contours, inst, index)
        if not strokes:
            continue
        name = unique_name(
            instance_name(inst, result.prompt_text, options.name_prefix),
            applied.names,
        )
        path = build_path(image, name, strokes, result, options.path_space)
        if path is None:
            continue
        scratch.created.append(("path", path))
        applied.paths.append(path)
        applied.names.append(name)


def _contours_for(contours: Any, inst: Instance, index: int) -> List[Any]:
    """Look contours up by instance id, then by position; accept either shape."""
    if isinstance(contours, Mapping):
        for key in (inst.instance_id, str(inst.instance_id)):
            if key in contours:
                return list(contours[key])
        return []
    try:
        return list(contours[index])
    except (IndexError, KeyError, TypeError):
        return []


#: Modes that build their output *through* the selection, and therefore have to
#: put the user's own selection back when they are done.  PATHS never touches it.
_SELECTION_TOUCHING = (OutputMode.CHANNELS, OutputMode.LAYER_MASKS, OutputMode.LAYER_GROUPS)

_MODES = {
    OutputMode.SELECTION: _mode_selection,
    OutputMode.CHANNELS: _mode_channels,
    OutputMode.LAYER_MASKS: _mode_layer_masks,
    OutputMode.LAYER_GROUPS: _mode_layer_groups,
    OutputMode.PATHS: _mode_paths,
}


# --------------------------------------------------------------------------- #
# the entry point
# --------------------------------------------------------------------------- #
def filter_instances(
    result: MaskResult,
    post: PostOps,
    selected: Optional[Iterable[int]] = None,
) -> Tuple[List[Instance], List[int]]:
    """Apply the instance-level filters: user selection, then minimum area.

    Returns ``(kept, dropped_instance_ids)``.  Separated out because the dialog
    wants to show the count before the user commits to an Apply.
    """
    # client.py decodes the wire frame; accept its objects, plain dicts, or our
    # own Instance rather than coupling to one decoder's class.
    instances = [_coerce_instance(i) for i in result.instances]
    if selected is not None:
        wanted = {int(i) for i in selected}
        instances = [i for i in instances if int(i.instance_id) in wanted]
    if post.min_area <= 0:
        return (instances, [])
    kept: List[Instance] = []
    dropped: List[int] = []
    for inst in instances:
        if _source_area(inst, result.rect_for(inst), post) >= post.min_area:
            kept.append(inst)
        else:
            dropped.append(int(inst.instance_id))
    return (kept, dropped)


def apply_result(
    image,
    result: MaskResult,
    options: Optional[OutputOptions] = None,
    layer: Any = None,
    selected: Optional[Iterable[int]] = None,
    contours: Any = None,
    on_log: Optional[Callable[[str], None]] = None,
) -> AppliedResult:
    """Materialise ``result`` inside ``image``.  One undo step, always.

    ``on_log`` receives one line per stage with the selection bounds after it,
    so a report of "the ants vanished at the end" can say which step did it.

    :param image: the ``Gimp.Image`` being edited.
    :param result: a :class:`MaskResult`, normally from :meth:`MaskResult.from_frame`.
    :param options: :class:`OutputOptions`; defaults to a plain replace-selection.
    :param layer: source layer for the layer-mask and layer-group modes.
        Defaults to the image's first selected layer.  A plug-in's drawable
        may be a layer mask or a channel; a mask stands for its own layer
        and a channel for the default one (:func:`_as_layer`).
    :param selected: instance ids the user ticked; ``None`` means all of them.
    :param contours: for :data:`OutputMode.PATHS` -- ``{instance_id: [stroke, ...]}``
        or a sequence parallel to the instances.

    The undo group, the context push and the scratch-channel cleanup are all in
    ``finally`` blocks: an exception halfway through still leaves the image with
    one undoable step and no leftover ``sam3-scratch`` channels in the dock.  A
    failed Apply also puts the selection back and removes whatever outputs it
    had already added (:func:`_roll_back`) before the exception propagates.
    """
    G = _gimp()
    options = (options or OutputOptions()).validated()
    post = options.post

    layer = _as_layer(image, layer)
    if options.mode in (OutputMode.LAYER_MASKS, OutputMode.LAYER_GROUPS) and layer is None:
        raise OutputError("output mode %r needs a layer, and the image has none"
                          % (options.mode,))
    setattr(options, "_layer", layer)
    setattr(options, "_contours", contours)

    instances, dropped = filter_instances(result, post, selected)
    applied = AppliedResult(mode=options.mode, dropped=dropped)

    def log(stage: str) -> None:
        if on_log is not None:
            try:
                on_log("apply %s/%s: %s -> selection %s"
                       % (options.mode, options.selection_op, stage,
                          _selection_bounds_text(image)))
            except Exception:  # noqa: BLE001 -- logging must never break an Apply
                pass

    log("start (%d instance(s), %d dropped)" % (len(instances), len(dropped)))
    remembered = _remember_selected_items(image)
    image.undo_group_start()
    try:
        _context_push(G)
        try:
            scratch = _Scratch(image)
            try:
                try:
                    _MODES[options.mode](image, result, instances, options, scratch, applied)
                    log("mode done")
                    if options.mode in _SELECTION_TOUCHING and _settle_selection(
                            image, options, scratch, applied):
                        log("previous selection restored" if options.restore_selection
                            else "selection kept")
                except BaseException:
                    _roll_back(image, scratch, applied)
                    log("failed; selection and outputs rolled back")
                    raise
            finally:
                scratch.discard()
                log("scratch channels removed")
                # The new layers a mode created are a better thing to leave
                # selected than the source layer; otherwise the user's own
                # selection of items comes back.  Layers and channels cannot
                # both be selected, so new layers mean no channels.
                created = list(applied.layers) if applied.layers else []
                _restore_selected_items(
                    image, (created, []) if created else remembered, layer)
                _select_paths(image, applied.paths)
                log("item selection restored")
        finally:
            _context_pop(G)
    finally:
        image.undo_group_end()
    log("undo group closed")

    if options.flush_displays:
        flush = getattr(G, "displays_flush", None)
        if callable(flush):
            flush()
    return applied


def _context_push(G) -> None:
    """Set interpolation for the upscale, without disturbing the user's context."""
    push = getattr(G, "context_push", None)
    if callable(push):
        push()
    setter = getattr(G, "context_set_interpolation", None)
    interp = getattr(G, "InterpolationType", None)
    if callable(setter) and interp is not None:
        mode = getattr(interp, "CUBIC", None)
        if mode is not None:
            try:
                setter(mode)
            except TypeError:
                pass
    resize = getattr(G, "context_set_transform_resize", None)
    resize_enum = getattr(G, "TransformResize", None)
    if callable(resize) and resize_enum is not None:
        mode = getattr(resize_enum, "ADJUST", None)
        if mode is not None:
            try:
                resize(mode)
            except TypeError:
                pass


def _context_pop(G) -> None:
    pop = getattr(G, "context_pop", None)
    if callable(pop):
        pop()


def _settle_selection(image, options: OutputOptions, scratch: _Scratch,
                      applied: AppliedResult) -> bool:
    """End a mode that borrowed the selection; False when it never did.

    With ``restore_selection`` (the default) the user's selection is put
    back, frozen, so undo records no selection change at all.  Without it
    the mode's last selection is kept, committed as one recorded change.
    """
    if scratch.original is None:
        return False
    if options.restore_selection:
        scratch.put_back_selection()
        return True
    final = scratch.save_selection("sam3-result")
    scratch.put_back_selection()
    _selection_select(image, SelectionOp.REPLACE, final)
    applied.selection_changed = True
    return True


def _roll_back(image, scratch: _Scratch, applied: AppliedResult) -> None:
    """Leave the image as it was before a failed Apply, as far as possible.

    The selection goes back to the saved original (frozen, like every other
    scratch change), and the outputs already added are removed again, last
    first.  Their additions and removals are both inside the Apply's undo
    group, so they cancel out instead of leaving a half-built step behind.
    Best effort: the image is in whatever state made the Apply fail.
    """
    try:
        scratch.put_back_selection()
    except Exception:  # noqa: BLE001
        pass
    G = _gimp()
    for kind, item in reversed(scratch.created):
        try:
            if kind == "layer":
                image.remove_layer(item)
            elif kind == "channel":
                image.remove_channel(item)
            elif kind == "path":
                _remove_path(image, item)
            elif kind == "mask":
                item.remove_mask(G.MaskApplyMode.DISCARD)
        except Exception:  # noqa: BLE001
            pass
    scratch.created = []
    applied.channels, applied.layers, applied.paths, applied.names = [], [], [], []
    applied.group = None


def _gi_class(G, name: str):
    cls = getattr(G, name, None)
    return cls if isinstance(cls, type) else None


def _as_layer(image, item):
    """The layer ``item`` stands for, or the image's default layer.

    GIMP hands a plug-in its *selected drawables*, and those are not always
    layers (``gimp_image_get_selected_drawables``): while a layer's mask is
    being edited -- which ``add_mask`` switches on, so right after a Layer
    mask Apply -- the drawable is the ``Gimp.LayerMask``, and whenever a
    channel is selected it is the channel.  Copying either gives a channel,
    which ``insert_layer`` refuses.  A mask maps back to its own layer; any
    other channel, or nothing at all, falls back to :func:`_default_layer`.
    Anything else is taken to be a layer already.
    """
    G = _gimp()
    mask_cls = _gi_class(G, "LayerMask")
    if item is not None and mask_cls is not None and isinstance(item, mask_cls):
        from_mask = getattr(getattr(G, "Layer", None), "from_mask", None)
        layer = None
        if callable(from_mask):
            try:
                layer = from_mask(item)
            except Exception:  # noqa: BLE001
                layer = None
        return layer if layer is not None else _default_layer(image)
    channel_cls = _gi_class(G, "Channel")
    if item is None or (channel_cls is not None and isinstance(item, channel_cls)):
        return _default_layer(image)
    return item


def _default_layer(image):
    """The first selected layer, else the top layer, else ``None``.

    Never ``get_selected_drawables``: that answers the selected channels, or
    the mask being edited, instead of a layer.
    """
    fn = getattr(image, "get_selected_layers", None)
    if callable(fn):
        items = fn()
        if items:
            return items[0]
    layers = _all_layers(image)
    return layers[0] if layers else None
