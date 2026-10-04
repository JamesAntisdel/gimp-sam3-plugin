"""Pixel I/O between GIMP and the ``sam3gimpd`` daemon.

Two directions, and nothing else:

**In** -- read what the user sees (the image *projection*, or optionally just
the active layer), downscale it to the daemon's upload limit **using GIMP's own
scaler**, and hand back raw ``R'G'B' u8`` bytes plus the exact geometry needed
to map results home.  See ``_daemon/API.md`` section 7.

**Out** -- take a cropped soft mask (uint8, model-canvas resolution, section 8
of the contract) and push it into a GIMP channel at *image* resolution, again
letting **GIMP's own scaler** do the resampling.  See ``_daemon/API.md``
section 9.

Design rules this module exists to enforce
------------------------------------------
* **Zero third-party dependencies.**  Standard library plus ``gi`` only: this
  code runs inside GIMP's embedded Python, which has no numpy, no pillow, no
  requests.
* **Never resample in pure Python.**  Every scale goes through a temporary
  ``Gimp.Image`` and ``Gimp.Image.scale()``, i.e. GEGL in C with a real
  interpolator.  The only bulk byte work done here is slicing (row crops) and
  ``bytes.translate`` (thresholding), both of which are C-speed and lossless.
* **Work in image space.**  Everything the caller sees is in *original-image*
  pixels.  Layer offsets are applied in exactly one place: writing a mask into
  a drawable that is not the full canvas (a layer mask).
* **Never hardcode the model canvas.**  The daemon reports
  ``canvas_from_image``; :func:`place_instance` inverts whatever it is given.
  A future letterboxing processor would set non-zero offsets and this code
  keeps working.
* **Importable without GIMP.**  The pure-geometry half of this module
  (:func:`compute_upload_size`, :class:`UploadGeometry`, :func:`place_instance`,
  :func:`threshold_bytes`, :func:`crop_rows`) has no ``gi`` dependency at all
  and is unit-tested without GIMP.  The GIMP half raises
  :class:`GimpUnavailableError` when the bindings are missing.

GIMP 3 API SURFACE USED -- VERIFY ON A REAL GIMP
------------------------------------------------
The unit tests run these against ``tests/fake_gimp``, not a real GIMP.  Each
is a documented libgimp-3.0 / GEGL entry point, but the *Python binding
shape* is the risk.  Check these first when the
plug-in is run under a real GIMP:

1.  ``gi.require_version("Gimp", "3.0")`` / ``("Gegl", "0.4")``.
2.  ``Gimp.Image``: ``new``, ``get_width``, ``get_height``, ``get_layers``,
    ``get_selected_layers``, ``insert_layer``, ``insert_channel``,
    ``get_selection``, ``duplicate``, ``flatten``, ``scale``, ``delete``.
    -- ``flatten()`` is assumed to return the resulting ``Gimp.Layer``; the code
    falls back to ``get_layers()[0]`` if it returns ``None``/``True``.
3.  ``Gimp.Layer.new_from_visible(image, dest_image, name)`` -- the cheap
    projection grab.  If it is missing or its signature differs, the code falls
    back to ``image.duplicate()`` + ``flatten()``, which is strictly more
    expensive but equivalent.  **Verify which path runs.**
4.  ``Gimp.Layer.new_from_drawable(drawable, dest_image)`` and
    ``Gimp.Layer.new(image, name, w, h, type, opacity, mode)``.
5.  ``Gimp.Channel.new(image, name, w, h, opacity, color)`` where *color* is a
    ``Gegl.Color``.  The code does **not** rely on a new channel being
    zero-filled: it writes zeros explicitly (``zero_fill=True``).
6.  ``Gimp.Drawable.get_offsets()`` -- assumed to return ``(ok, x, y)``;
    :func:`drawable_offsets` also accepts a bare ``(x, y)``.
7.  ``Gimp.Drawable.get_buffer()`` -> ``Gegl.Buffer``, plus ``buffer.flush()``
    and ``Gimp.Drawable.update(x, y, w, h)`` to make writes visible.
8.  ``Gegl.Buffer.get(rect, scale, format_string, Gegl.AbyssPolicy.CLAMP)``
    returning bytes -- **the single most important call to verify.**  The
    contract (``_daemon/API.md`` section 7) pins this shape.  :func:`buffer_get`
    tries that first and then ``get(rect, scale, format)``, and normalises a
    ``GLib.Bytes`` return.
9.  ``Gegl.Buffer.set(rect, format_string, data)`` -- second most important.
    :func:`buffer_set` tries that, then the full C shape
    ``set(rect, level, format, data, rowstride)``.
10. ``Gimp.context_push`` / ``context_pop`` /
    ``context_set_background(Gegl.Color)`` / ``context_set_interpolation``.
    The background colour is only used when flattening; the default is white,
    where the linear-vs-perceptual question for ``Gegl.Color.set_rgba`` cannot
    change the result.
11. ``Gimp.InterpolationType.NOHALO`` (downscale) and ``.CUBIC`` (upscale);
    both are resolved with ``getattr`` and degrade to ``LINEAR``/``CUBIC``.
12. ``Gimp.Selection.bounds(image)`` -> ``(ok, non_empty, x1, y1, x2, y2)`` on GIMP 3
    (seen on 3.2); ``(non_empty, x1, y1, x2, y2)`` on older bindings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

__all__ = [
    "MAX_UPLOAD_SIDE",
    "MIN_UPLOAD_SIDE",
    "DEFAULT_MASK_THRESHOLD",
    "FORMAT_RGB",
    "FORMAT_GRAY",
    "SOURCE_PROJECTION",
    "SOURCE_LAYER",
    "BridgeError",
    "GimpUnavailableError",
    "GimpApiError",
    "UploadGeometry",
    "UploadedImage",
    "MaskPlacement",
    "compute_upload_size",
    "place_instance",
    "threshold_bytes",
    "crop_rows",
    "gimp_available",
    "rebind_gi",
    "read_upload_pixels",
    "scale_soft_mask",
    "build_mask_channel",
    "instance_to_channel",
    "write_soft_mask",
    "drawable_offsets",
    "selection_box",
    "source_layer",
]

# --------------------------------------------------------------------------- #
# constants (mirrors of _daemon/API.md; the plug-in must not import sam3gimpd)
# --------------------------------------------------------------------------- #
MAX_UPLOAD_SIDE = 1008
MIN_UPLOAD_SIDE = 16
DEFAULT_MASK_THRESHOLD = 128

FORMAT_RGB = "R'G'B' u8"
FORMAT_GRAY = "Y' u8"

SOURCE_PROJECTION = "projection"
SOURCE_LAYER = "layer"


class BridgeError(Exception):
    """Base class for every failure raised by this module."""


class GimpUnavailableError(BridgeError):
    """Raised when a GIMP-touching helper is called outside GIMP."""


class GimpApiError(BridgeError):
    """Raised when a libgimp/GEGL call exists but not in a shape we understand."""


# --------------------------------------------------------------------------- #
# gi binding, resolved lazily enough to stay importable without GIMP
# --------------------------------------------------------------------------- #
Gimp = None
Gegl = None


def rebind_gi() -> bool:
    """(Re-)resolve the ``Gimp`` and ``Gegl`` module globals.

    Called once at import.  ``tests/fake_gimp`` calls it again after installing
    its stubs, which is why the bindings are module globals rather than
    from-imports: every function below looks them up at call time.
    """
    global Gimp, Gegl
    try:
        import gi

        gi.require_version("Gimp", "3.0")
        gi.require_version("Gegl", "0.4")
        from gi.repository import Gegl as _Gegl
        from gi.repository import Gimp as _Gimp
    except (ImportError, ValueError, AttributeError):
        Gimp = None
        Gegl = None
        return False
    Gimp = _Gimp
    Gegl = _Gegl
    return True


rebind_gi()


def gimp_available() -> bool:
    """True when ``Gimp`` and ``Gegl`` are bound (inside GIMP, or under stubs)."""
    return Gimp is not None and Gegl is not None


def _require_gimp() -> None:
    if not gimp_available():
        raise GimpUnavailableError(
            "gi.repository.Gimp / Gegl are unavailable; this helper only runs "
            "inside GIMP (or against tests/fake_gimp)"
        )


# --------------------------------------------------------------------------- #
# pure geometry -- no GIMP needed, fully unit-tested without it
# --------------------------------------------------------------------------- #
def _round_half_up(value: float) -> int:
    """Deterministic rounding.

    ``round()`` is banker's rounding, which would place two adjacent instance
    edges inconsistently at exact ``.5``.  ``_daemon/API.md`` section 9 says
    "round each edge independently"; half-up is the intent.
    """
    return int(math.floor(float(value) + 0.5))


def compute_upload_size(
    width: int,
    height: int,
    max_side: int = MAX_UPLOAD_SIDE,
    min_side: int = MIN_UPLOAD_SIDE,
) -> Tuple[int, int]:
    """Uploaded-image size for a source of ``width`` x ``height``.

    Aspect ratio is preserved by a uniform scale onto ``max_side``.  Two
    clamps can break that ratio, and both are deliberate:

    * a side below ``min_side`` is raised (the daemon rejects ``< 16``);
    * a side above ``max_side`` is lowered (the daemon rejects ``> 1008``
      rather than silently downscaling, precisely so the client's own mapping
      cannot desynchronise).

    Extreme aspect ratios (4000x10) hit both clamps at once and *must*
    distort; that is why :class:`UploadGeometry` records ``scale_x`` and
    ``scale_y`` separately instead of one factor.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive, got %dx%d" % (width, height))
    if max_side < min_side:
        raise ValueError("max_side must be >= min_side")

    scale = min(1.0, float(max_side) / float(max(width, height)))
    upload_w = max(1, _round_half_up(width * scale))
    upload_h = max(1, _round_half_up(height * scale))

    upload_w = min(max(upload_w, min_side), max_side)
    upload_h = min(max(upload_h, min_side), max_side)
    return upload_w, upload_h


@dataclass(frozen=True)
class UploadGeometry:
    """The client-owned half of the coordinate story.

    ``_daemon/API.md`` section 5 names three spaces.  This object owns the
    mapping between the first two:

    * **original-image** -- ``source_width`` x ``source_height``, the user's
      GIMP image;
    * **uploaded-image** -- ``width`` x ``height``, what was POSTed.

    The third (model-canvas) belongs to the daemon and arrives as
    ``canvas_from_image`` in every result header.
    """

    source_width: int
    source_height: int
    width: int
    height: int

    @property
    def scale_x(self) -> float:
        """uploaded_x = original_x * scale_x"""
        return float(self.width) / float(self.source_width)

    @property
    def scale_y(self) -> float:
        return float(self.height) / float(self.source_height)

    def to_upload(self, x: float, y: float) -> Tuple[float, float]:
        """original-image -> uploaded-image (what every prompt coordinate needs)."""
        return (float(x) * self.scale_x, float(y) * self.scale_y)

    def from_upload(self, x: float, y: float) -> Tuple[float, float]:
        """uploaded-image -> original-image."""
        return (float(x) / self.scale_x, float(y) / self.scale_y)

    def box_to_upload(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> Tuple[float, float, float, float]:
        ux0, uy0 = self.to_upload(x0, y0)
        ux1, uy1 = self.to_upload(x1, y1)
        return (ux0, uy0, ux1, uy1)

    def upload_headers(self) -> Dict[str, str]:
        """The ``X-*`` headers ``POST /images`` wants (contract section 6.2)."""
        return {
            "X-Width": str(self.width),
            "X-Height": str(self.height),
            "X-Source-Width": str(self.source_width),
            "X-Source-Height": str(self.source_height),
        }

    def expected_payload_size(self) -> int:
        return self.width * self.height * 3


@dataclass(frozen=True)
class UploadedImage:
    """Raw ``R'G'B' u8`` bytes plus the geometry that produced them."""

    pixels: bytes
    geometry: UploadGeometry
    source: str = SOURCE_PROJECTION

    @property
    def width(self) -> int:
        return self.geometry.width

    @property
    def height(self) -> int:
        return self.geometry.height


@dataclass(frozen=True)
class MaskPlacement:
    """Where a scaled instance mask belongs, in **original-image** pixels."""

    x: int
    y: int
    width: int
    height: int

    @property
    def size(self) -> Tuple[int, int]:
        return (self.width, self.height)


def _affine(canvas_from_image: Any) -> Tuple[float, float, float, float]:
    """Read ``{scale_x, scale_y, offset_x, offset_y}`` from a dict or an object."""
    if isinstance(canvas_from_image, Mapping):
        get = canvas_from_image.__getitem__

        def read(name, default=None):
            try:
                return get(name)
            except KeyError:
                if default is None:
                    raise KeyError("canvas_from_image is missing %r" % name)
                return default

    else:

        def read(name, default=None):
            value = getattr(canvas_from_image, name, None)
            if value is None:
                if default is None:
                    raise KeyError("canvas_from_image is missing %r" % name)
                return default
            return value

    scale_x = float(read("scale_x"))
    scale_y = float(read("scale_y"))
    offset_x = float(read("offset_x", 0.0))
    offset_y = float(read("offset_y", 0.0))
    if scale_x == 0.0 or scale_y == 0.0:
        raise ValueError("canvas_from_image has a zero scale; cannot invert")
    return scale_x, scale_y, offset_x, offset_y


def place_instance(
    bbox: Sequence[int],
    canvas_from_image: Any,
    geometry: UploadGeometry,
) -> MaskPlacement:
    """Map a model-canvas bbox onto the original image.

    This is ``_daemon/API.md`` section 9, steps 1-3, verbatim:

    1. invert the *reported* affine (canvas -> uploaded-image);
    2. undo the client's own downscale (uploaded-image -> original-image);
    3. round each edge independently, then derive the size, so adjacent
       instances tile without gaps.

    The result is clamped into the image, and never smaller than 1x1.
    """
    if len(bbox) != 4:
        raise ValueError("bbox must be [x0, y0, x1, y1], got %r" % (bbox,))
    scale_x, scale_y, offset_x, offset_y = _affine(canvas_from_image)
    x0, y0, x1, y1 = (float(v) for v in bbox)

    # canvas -> uploaded-image
    ix0 = (x0 - offset_x) / scale_x
    ix1 = (x1 - offset_x) / scale_x
    iy0 = (y0 - offset_y) / scale_y
    iy1 = (y1 - offset_y) / scale_y

    # uploaded-image -> original-image
    u = float(geometry.source_width) / float(geometry.width)
    v = float(geometry.source_height) / float(geometry.height)

    big_x0 = _round_half_up(ix0 * u)
    big_x1 = _round_half_up(ix1 * u)
    big_y0 = _round_half_up(iy0 * v)
    big_y1 = _round_half_up(iy1 * v)

    max_w = geometry.source_width
    max_h = geometry.source_height
    big_x0 = min(max(big_x0, 0), max(0, max_w - 1))
    big_y0 = min(max(big_y0, 0), max(0, max_h - 1))
    big_x1 = min(max(big_x1, big_x0 + 1), max_w)
    big_y1 = min(max(big_y1, big_y0 + 1), max_h)

    return MaskPlacement(big_x0, big_y0, big_x1 - big_x0, big_y1 - big_y0)


_THRESHOLD_TABLES: Dict[int, bytes] = {}


def threshold_bytes(data: bytes, threshold: int = DEFAULT_MASK_THRESHOLD) -> bytes:
    """Binarise a soft mask: ``value >= threshold`` -> 255, else 0.

    ``bytes.translate`` does this at C speed with a 256-entry table, so a
    6-megapixel mask costs a memcpy rather than a Python loop.  This is the
    whole point of the u8-soft mask format: the threshold slider re-thresholds
    bytes already in memory, with no HTTP round trip (contract section 8.3).
    """
    if not 1 <= int(threshold) <= 255:
        raise ValueError("threshold must be in [1, 255], got %r" % (threshold,))
    threshold = int(threshold)
    table = _THRESHOLD_TABLES.get(threshold)
    if table is None:
        table = bytes(0 if i < threshold else 255 for i in range(256))
        _THRESHOLD_TABLES[threshold] = table
    return bytes(data).translate(table)


def crop_rows(
    data: bytes,
    width: int,
    height: int,
    x: int,
    y: int,
    crop_width: int,
    crop_height: int,
    components: int = 1,
) -> bytes:
    """Cut a sub-rectangle out of packed row-major pixel bytes.

    One Python-level iteration per *row* (a C-level slice each), never per
    pixel.  Used to clip a scaled mask against a drawable's bounds.
    """
    if crop_width <= 0 or crop_height <= 0:
        return b""
    if (x, y, crop_width, crop_height) == (0, 0, width, height):
        return bytes(data)
    stride = width * components
    out = bytearray(crop_width * crop_height * components)
    row_bytes = crop_width * components
    for row in range(crop_height):
        src = (y + row) * stride + x * components
        out[row * row_bytes : (row + 1) * row_bytes] = data[src : src + row_bytes]
    return bytes(out)


# --------------------------------------------------------------------------- #
# thin GEGL / libgimp compatibility helpers
# --------------------------------------------------------------------------- #
def _rect(x: int, y: int, width: int, height: int):
    return Gegl.Rectangle.new(int(x), int(y), int(width), int(height))


def _as_bytes(value: Any) -> bytes:
    """Normalise whatever ``Gegl.Buffer.get`` handed back into ``bytes``."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    get_data = getattr(value, "get_data", None)  # GLib.Bytes
    if callable(get_data):
        return bytes(get_data())
    if isinstance(value, (list, tuple)):
        return bytes(value)
    raise GimpApiError(
        "Gegl.Buffer.get returned %r, which this bridge cannot turn into bytes"
        % (type(value).__name__,)
    )


def buffer_get(buffer, rect, fmt: str, scale: float = 1.0) -> bytes:
    """``Gegl.Buffer.get`` with the call shape pinned by the contract.

    See VERIFY item 8 in the module docstring.
    """
    attempts = (
        (rect, scale, fmt, Gegl.AbyssPolicy.CLAMP),
        (rect, scale, fmt),
    )
    last: Optional[TypeError] = None
    for args in attempts:
        try:
            return _as_bytes(buffer.get(*args))
        except TypeError as exc:
            last = exc
    raise GimpApiError(
        "Gegl.Buffer.get did not accept any known signature (%s)" % (last,)
    )


def buffer_set(buffer, rect, fmt: str, data: bytes) -> None:
    """``Gegl.Buffer.set`` with the call shape pinned by the contract.

    See VERIFY item 9 in the module docstring.
    """
    payload = bytes(data)
    attempts = (
        (rect, fmt, payload),
        (rect, 0, fmt, payload, 0),
    )
    last: Optional[TypeError] = None
    for args in attempts:
        try:
            buffer.set(*args)
            return
        except TypeError as exc:
            last = exc
    raise GimpApiError(
        "Gegl.Buffer.set did not accept any known signature (%s)" % (last,)
    )


def drawable_offsets(drawable) -> Tuple[int, int]:
    """``(offset_x, offset_y)`` from a GIMP 3 ``(ok, x, y)`` or a bare ``(x, y)``."""
    offsets = drawable.get_offsets()
    if offsets is None:
        return (0, 0)
    values = tuple(offsets)
    if len(values) == 3:
        return (int(values[1]), int(values[2]))
    if len(values) == 2:
        return (int(values[0]), int(values[1]))
    raise GimpApiError("Gimp.Drawable.get_offsets returned %r" % (offsets,))


def _color(r: int, g: int, b: int, a: float = 1.0):
    color = Gegl.Color.new("black")
    color.set_rgba(r / 255.0, g / 255.0, b / 255.0, float(a))
    return color


def _interpolation(name: str, *fallbacks: str):
    enum = Gimp.InterpolationType
    for candidate in (name,) + fallbacks:
        value = getattr(enum, candidate, None)
        if value is not None:
            return value
    raise GimpApiError("Gimp.InterpolationType has none of %r" % ((name,) + fallbacks,))


def _sole_layer(image):
    layers = image.get_layers()
    if not layers:
        raise GimpApiError("expected a flattened image to have exactly one layer")
    return layers[0]


# --------------------------------------------------------------------------- #
# which layer
# --------------------------------------------------------------------------- #
def _gi_class(name: str):
    cls = getattr(Gimp, name, None)
    return cls if isinstance(cls, type) else None


def source_layer(image, drawable=None):
    """The ``Gimp.Layer`` a plug-in's drawable stands for.

    GIMP hands a plug-in its *selected drawables*, and those are not always
    layers (``gimp_image_get_selected_drawables``): while a layer's mask is
    being edited -- and adding a mask switches that on, so this is the state
    right after a Layer mask Apply -- the drawable is the ``Gimp.LayerMask``;
    whenever a channel is selected it is the channel.  Read as "the layer",
    a mask uploads its grey pixels and copies into a channel.

    A mask maps back to its own layer (``Gimp.Layer.from_mask``); a channel,
    or no drawable at all, falls back to the first selected layer, then the
    top layer.  ``None`` only when the image has no layer.
    """
    _require_gimp()
    mask_cls = _gi_class("LayerMask")
    if drawable is not None and mask_cls is not None and isinstance(drawable, mask_cls):
        from_mask = getattr(Gimp.Layer, "from_mask", None)
        layer = from_mask(drawable) if callable(from_mask) else None
        if layer is not None:
            return layer
    else:
        channel_cls = _gi_class("Channel")
        if drawable is not None and not (channel_cls is not None
                                         and isinstance(drawable, channel_cls)):
            return drawable
    selected = image.get_selected_layers()
    if selected:
        return selected[0]
    layers = image.get_layers()
    return layers[0] if layers else None


# --------------------------------------------------------------------------- #
# IN: GIMP -> raw RGB bytes
# --------------------------------------------------------------------------- #
def _projection_work_image(image, width: int, height: int):
    """A fresh single-layer image holding the projection of ``image``.

    Deleted again if anything fails before it is handed back, so a failed
    read never leaves an invisible image behind in GIMP.
    """
    work = Gimp.Image.new(width, height, Gimp.ImageBaseType.RGB)
    try:
        new_from_visible = getattr(Gimp.Layer, "new_from_visible", None)
        if new_from_visible is not None:
            try:
                layer = new_from_visible(image, work, "sam3-source")
            except TypeError:
                layer = None
            if layer is not None:
                work.insert_layer(layer, None, 0)
                return work
    except BaseException:
        work.delete()
        raise
    # Fallback (VERIFY item 3): duplicate the whole stack and flatten it.
    work.delete()
    return image.duplicate()


def _layer_work_image(image, drawable, width: int, height: int):
    """A fresh single-layer image holding ``drawable`` at its image-space offsets.

    Deleted again if anything fails before it is handed back.
    """
    work = Gimp.Image.new(width, height, Gimp.ImageBaseType.RGB)
    try:
        copy = Gimp.Layer.new_from_drawable(drawable, work)
        work.insert_layer(copy, None, 0)
        offset_x, offset_y = drawable_offsets(drawable)
        copy.set_offsets(offset_x, offset_y)
    except BaseException:
        work.delete()
        raise
    return work


def read_upload_pixels(
    image,
    source: str = SOURCE_PROJECTION,
    drawable=None,
    max_side: int = MAX_UPLOAD_SIDE,
    min_side: int = MIN_UPLOAD_SIDE,
    background: Tuple[int, int, int] = (255, 255, 255),
) -> UploadedImage:
    """Read pixels for ``POST /images``.

    ``source`` is :data:`SOURCE_PROJECTION` (default -- what the user actually
    sees, all visible layers composited) or :data:`SOURCE_LAYER` (just the
    layer ``drawable`` stands for -- see :func:`source_layer` -- positioned
    by its own offsets on an otherwise empty canvas).

    The whole pipeline happens inside a throwaway ``Gimp.Image`` that is
    deleted before returning, so the user's image, its selection and its undo
    stack are never touched:

    1. build the work image (projection or single layer);
    2. ``flatten()`` it onto ``background`` -- the daemon takes no alpha
       (contract section 7), and babl's alpha *drop* would leak whatever colour
       hid under transparent pixels;
    3. ``scale()`` it to the upload size with GEGL's NoHalo/cubic sampler --
       **this is the "do not hand-roll resampling" rule**;
    4. read the single remaining layer as ``R'G'B' u8``.

    Returns an :class:`UploadedImage` whose ``pixels`` is exactly
    ``width * height * 3`` bytes, ready to be a request body verbatim.
    """
    _require_gimp()
    if source not in (SOURCE_PROJECTION, SOURCE_LAYER):
        raise ValueError("source must be %r or %r" % (SOURCE_PROJECTION, SOURCE_LAYER))

    source_width = int(image.get_width())
    source_height = int(image.get_height())
    upload_w, upload_h = compute_upload_size(
        source_width, source_height, max_side, min_side
    )
    geometry = UploadGeometry(source_width, source_height, upload_w, upload_h)

    if source == SOURCE_LAYER:
        drawable = source_layer(image, drawable)
        if drawable is None:
            raise BridgeError("the image has no layer; cannot read SOURCE_LAYER")

    Gimp.context_push()
    work = None
    try:
        Gimp.context_set_background(_color(*background))
        Gimp.context_set_interpolation(
            _interpolation("NOHALO", "CUBIC", "LINEAR")
            if (upload_w, upload_h) != (source_width, source_height)
            else _interpolation("CUBIC", "LINEAR")
        )
        if source == SOURCE_PROJECTION:
            work = _projection_work_image(image, source_width, source_height)
        else:
            work = _layer_work_image(image, drawable, source_width, source_height)

        flattened = work.flatten()
        layer = flattened if hasattr(flattened, "get_buffer") else _sole_layer(work)

        if (upload_w, upload_h) != (source_width, source_height):
            work.scale(upload_w, upload_h)
            layer = _sole_layer(work)

        pixels = buffer_get(
            layer.get_buffer(), _rect(0, 0, upload_w, upload_h), FORMAT_RGB
        )
    finally:
        if work is not None:
            work.delete()
        Gimp.context_pop()

    expected = geometry.expected_payload_size()
    if len(pixels) != expected:
        raise GimpApiError(
            "projection read returned %d bytes, expected %d (%dx%d R'G'B' u8)"
            % (len(pixels), expected, upload_w, upload_h)
        )
    return UploadedImage(pixels=pixels, geometry=geometry, source=source)


def selection_box(image, geometry: Optional[UploadGeometry] = None):
    """The current selection's bounds, or ``None`` when there is no selection.

    Returns ``(x0, y0, x1, y1)`` in **uploaded-image** pixels when ``geometry``
    is given (ready to be a ``box`` prompt, contract section 5), otherwise in
    original-image pixels.  This is the "Rectangle Select as a box prompt"
    fallback path from ``DESIGN.md`` section 5.
    """
    _require_gimp()
    bounds = Gimp.Selection.bounds(image)
    values = tuple(bounds)
    # GIMP 3 returns (ok, non_empty, x1, y1, x2, y2): the PDB success flag
    # comes first and the "is there a selection" flag second.  Older
    # bindings drop the leading flag.
    if len(values) == 6:
        values = values[1:]
    if len(values) != 5:
        raise GimpApiError("Gimp.Selection.bounds returned %r" % (bounds,))
    non_empty, x0, y0, x1, y1 = values
    if not non_empty or x1 <= x0 or y1 <= y0:
        return None
    if geometry is None:
        return (float(x0), float(y0), float(x1), float(y1))
    return geometry.box_to_upload(x0, y0, x1, y1)


# --------------------------------------------------------------------------- #
# OUT: soft mask -> GIMP
# --------------------------------------------------------------------------- #
def scale_soft_mask(
    mask: bytes,
    mask_width: int,
    mask_height: int,
    dst_width: int,
    dst_height: int,
    interpolation=None,
) -> bytes:
    """Resample a cropped soft mask with **GIMP's own scaler**.

    ``mask`` is the raw ``uint8`` crop from the result frame: row-major,
    top-to-bottom, ``mask_width`` bytes per row (contract section 8.3) -- which
    is byte-for-byte what ``Gegl.Buffer.set`` wants for ``"Y' u8"``.

    The mask is loaded into a temporary GRAY ``Gimp.Image``, scaled by
    ``Gimp.Image.scale()`` (GEGL, in C, cubic by default) and read back.  The
    plug-in has no numpy and must never interpolate in Python; this function is
    the only place the project resamples a mask, and it does so in C.

    Note the ordering rule from ``_daemon/API.md`` section 9 step 4: scale the
    *soft* mask, then threshold.  Thresholding first gives stair-stepped edges.
    """
    if len(mask) != mask_width * mask_height:
        raise ValueError(
            "soft mask is %d bytes, expected %d (%dx%d)"
            % (len(mask), mask_width * mask_height, mask_width, mask_height)
        )
    if dst_width <= 0 or dst_height <= 0:
        raise ValueError("destination size must be positive")
    if (dst_width, dst_height) == (mask_width, mask_height):
        return bytes(mask)

    _require_gimp()
    Gimp.context_push()
    temp = None
    try:
        if interpolation is None:
            upscaling = dst_width * dst_height >= mask_width * mask_height
            interpolation = (
                _interpolation("CUBIC", "LINEAR")
                if upscaling
                else _interpolation("NOHALO", "CUBIC", "LINEAR")
            )
        Gimp.context_set_interpolation(interpolation)

        temp = Gimp.Image.new(mask_width, mask_height, Gimp.ImageBaseType.GRAY)
        layer = Gimp.Layer.new(
            temp,
            "sam3-mask",
            mask_width,
            mask_height,
            Gimp.ImageType.GRAY_IMAGE,
            100.0,
            Gimp.LayerMode.NORMAL,
        )
        temp.insert_layer(layer, None, 0)

        buffer = layer.get_buffer()
        buffer_set(buffer, _rect(0, 0, mask_width, mask_height), FORMAT_GRAY, mask)
        buffer.flush()
        layer.update(0, 0, mask_width, mask_height)

        temp.scale(dst_width, dst_height)
        scaled_layer = _sole_layer(temp)
        out = buffer_get(
            scaled_layer.get_buffer(), _rect(0, 0, dst_width, dst_height), FORMAT_GRAY
        )
    finally:
        if temp is not None:
            temp.delete()
        Gimp.context_pop()

    if len(out) != dst_width * dst_height:
        raise GimpApiError(
            "scaled mask is %d bytes, expected %d"
            % (len(out), dst_width * dst_height)
        )
    return out


def write_soft_mask(
    drawable,
    mask: bytes,
    mask_width: int,
    mask_height: int,
    x: int,
    y: int,
    use_offsets: bool = True,
) -> bool:
    """Write ``Y' u8`` bytes into ``drawable`` at image-space ``(x, y)``.

    Everything else in this module works in image space; this is the single
    place where layer offsets are undone.  ``use_offsets=True`` (the default)
    subtracts the drawable's own offsets, which is exactly what a **layer
    mask** needs: the mask drawable is the size of its layer and sits at the
    layer's position, not the canvas origin.  Pass ``use_offsets=False`` for a
    channel or the selection, which are always canvas-aligned.

    Anything falling outside the drawable is clipped away with row slices (no
    per-pixel Python).  Returns ``False`` when nothing was inside.
    """
    _require_gimp()
    if len(mask) != mask_width * mask_height:
        raise ValueError(
            "mask is %d bytes, expected %d" % (len(mask), mask_width * mask_height)
        )

    offset_x, offset_y = drawable_offsets(drawable) if use_offsets else (0, 0)
    dest_x = int(x) - offset_x
    dest_y = int(y) - offset_y

    bound_w = int(drawable.get_width())
    bound_h = int(drawable.get_height())

    crop_x = max(0, -dest_x)
    crop_y = max(0, -dest_y)
    crop_w = min(mask_width - crop_x, bound_w - max(dest_x, 0))
    crop_h = min(mask_height - crop_y, bound_h - max(dest_y, 0))
    if crop_w <= 0 or crop_h <= 0:
        return False

    payload = crop_rows(mask, mask_width, mask_height, crop_x, crop_y, crop_w, crop_h)
    put_x = max(dest_x, 0)
    put_y = max(dest_y, 0)

    buffer = drawable.get_buffer()
    buffer_set(buffer, _rect(put_x, put_y, crop_w, crop_h), FORMAT_GRAY, payload)
    buffer.flush()
    drawable.update(put_x, put_y, crop_w, crop_h)
    return True


def build_mask_channel(
    image,
    mask: bytes,
    mask_width: int,
    mask_height: int,
    placement: MaskPlacement,
    name: str = "SAM 3 mask",
    threshold: Optional[int] = None,
    insert: bool = True,
    zero_fill: bool = True,
    visible: bool = False,
):
    """Turn one instance's soft mask into a full-canvas ``Gimp.Channel``.

    This is the core trick from ``DESIGN.md`` section 4 and ``API.md``
    section 9:

    1. the daemon returns a small soft crop at model-canvas resolution;
    2. :func:`scale_soft_mask` upsamples it to ``placement`` size **in GIMP**;
    3. it is composited into an image-sized channel that is ``0`` everywhere
       else;
    4. thresholding is optional and happens *after* the scale.

    Leave ``threshold=None`` to keep the channel soft -- that is the good
    default, because a soft channel loaded as a selection gives antialiased
    edges and lets the user re-threshold without another round trip.

    The channel is inserted into ``image`` by default (an item must be in the
    image before ``Gimp.Image.select_item`` will take it).  Callers that want a
    scratch channel should pass ``insert=False`` or remove it afterwards; wrap
    the whole apply in ``image.undo_group_start()/end()`` so it is one Ctrl+Z.
    """
    _require_gimp()
    width = int(image.get_width())
    height = int(image.get_height())

    channel = Gimp.Channel.new(image, name, width, height, 100.0, _color(0, 0, 0))
    try:
        channel.set_visible(bool(visible))
    except AttributeError:  # pragma: no cover - older bindings
        pass

    buffer = channel.get_buffer()
    if zero_fill:
        # Do not assume gimp_channel_new zero-fills (VERIFY item 5).
        buffer_set(buffer, _rect(0, 0, width, height), FORMAT_GRAY, b"\x00" * (width * height))

    scaled = scale_soft_mask(
        mask, mask_width, mask_height, placement.width, placement.height
    )
    if threshold is not None:
        scaled = threshold_bytes(scaled, threshold)

    write_soft_mask(
        channel,
        scaled,
        placement.width,
        placement.height,
        placement.x,
        placement.y,
        use_offsets=False,
    )
    buffer.flush()
    channel.update(0, 0, width, height)

    if insert:
        image.insert_channel(channel, None, 0)
    return channel


def instance_to_channel(
    image,
    instance: Mapping[str, Any],
    blob: bytes,
    canvas_from_image: Any,
    geometry: UploadGeometry,
    name: Optional[str] = None,
    threshold: Optional[int] = None,
    insert: bool = True,
):
    """Convenience: one decoded result-frame instance -> one ``Gimp.Channel``.

    ``instance`` is a single element of the result header's ``instances`` array
    (contract section 8.2) and ``blob`` is exactly its
    ``blob_offset``/``blob_length`` slice of the frame's blob region.  Naming
    follows ``DESIGN.md`` section 6: label plus score.
    """
    mask_width = int(instance["mask_width"])
    mask_height = int(instance["mask_height"])
    if len(blob) != mask_width * mask_height:
        raise ValueError(
            "blob is %d bytes but the instance declares %dx%d"
            % (len(blob), mask_width, mask_height)
        )
    placement = place_instance(instance["bbox"], canvas_from_image, geometry)
    if name is None:
        label = str(instance.get("label") or "mask").strip() or "mask"
        name = "%s %.0f%%" % (label, float(instance.get("score", 0.0)) * 100.0)
    return build_mask_channel(
        image,
        blob,
        mask_width,
        mask_height,
        placement,
        name=name,
        threshold=threshold,
        insert=insert,
    )
