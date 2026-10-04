"""Interactive GTK3 preview canvas for the sam3-gimp plug-in.

GIMP plug-ins **cannot receive canvas click events** -- there is no such API --
so the plug-in brings its own canvas.  :class:`Sam3Canvas` is a reusable
``Gtk.DrawingArea`` that knows nothing about dialogs, HTTP or GIMP; it renders
pixels and emits GObject signals.

What it does
------------
* Draws an **uploaded-image** RGB buffer (the ``POST /images`` body, see
  ``_daemon/API.md`` §7) with instance masks composited on top as translucent
  coloured overlays, each instance in a distinct, stable colour.
* Per-instance **visibility** (driven by the dialog's list) and **selection**
  (driven by clicking the canvas), plus hover highlighting.
* **Point prompts**: left click drops a positive point, right click or
  shift-click a negative one; ``Ctrl``-drag draws a box.  Coordinates are
  emitted in **uploaded-image** space, which is exactly what
  ``POST /images/{id}/points`` wants (API §5).
* **Zoom** (scroll wheel, centred on the pointer) and **pan** (middle-drag or
  space-drag), with widget<->image transforms factored into small pure
  functions at the top of this module so they can be tested directly.
* **Client-side thresholding.**  Masks arrive as *soft* uint8 (API §8.3,
  ``128`` == logit 0).  Moving the mask-threshold or score slider re-thresholds
  and re-filters **locally, with no network round trip** -- that is the entire
  reason the wire format is soft.  The re-threshold is one
  :meth:`bytes.translate` per instance into a ``cairo.FORMAT_A8`` surface, so it
  is genuinely instant even for full-canvas masks; surfaces are cached per
  ``(instance, threshold)`` and only rebuilt when they actually change.
* A **marching-ants** outline for the active mask (Moore-neighbour contour
  tracing + Douglas-Peucker, both pure functions) and an overlay-opacity
  control.

Coordinate spaces (names taken from ``_daemon/API.md`` §5)
---------------------------------------------------------
``widget``
    device pixels inside the ``Gtk.DrawingArea``.
``image``
    **uploaded-image** pixels -- the buffer handed to :meth:`Sam3Canvas.set_image`
    and the space of *every* coordinate the client sends to the daemon.
``canvas``
    **model-canvas** pixels -- the space of *every* geometry the daemon sends
    back (instance ``bbox``, and therefore the mask crops).

``canvas = image * scale + offset``, with the affine reported by the daemon as
``canvas_from_image``.  It is **never hardcoded**: :class:`Affine` is built from
whatever the result header says (API §5 is emphatic about this).

Dependencies
------------
Standard library, ``gi`` and ``cairo`` only -- see ``ui/__init__.py`` for why
pycairo does not violate the zero-dependency rule.  The GTK import is guarded so
that the pure functions in this module remain importable (and testable) without
GTK at all.
"""

from __future__ import annotations

import json
import math
import struct
import sys as _sys
from collections import namedtuple

try:  # pragma: no cover - exercised implicitly by whichever branch runs
    import gi

    gi.require_version("Gtk", "3.0")
    gi.require_version("Gdk", "3.0")
    from gi.repository import Gdk, GLib, GObject, Gtk

    import cairo

    GTK_AVAILABLE = True
except Exception:  # pragma: no cover - only without GTK/pycairo
    gi = None
    Gdk = GLib = GObject = Gtk = None
    try:
        import cairo  # pycairo can exist without PyGObject
    except Exception:
        cairo = None
    GTK_AVAILABLE = False


__all__ = [
    # wire format
    "RESULT_MAGIC",
    "RESULT_PREFIX_SIZE",
    "decode_frame",
    "encode_frame",
    # geometry
    "View",
    "Affine",
    "image_to_widget",
    "widget_to_image",
    "view_fit",
    "view_zoom_at",
    "view_pan",
    "view_clamp",
    # masks
    "Instance",
    "alpha_table",
    "binarize",
    "mask_sample",
    "instance_sample",
    "filter_by_score",
    "hit_test",
    "instance_color",
    # contours
    "trace_contours",
    "simplify_polyline",
    "contour_polygons",
    # surfaces
    "surface_from_rgb",
    "alpha_surface_from_mask",
    # widget
    "GTK_AVAILABLE",
    "Sam3Canvas",
    "DEFAULT_MASK_THRESHOLD",
]


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #

#: ``_daemon/API.md`` §8.1.  Mirrored by hand: the plug-in must not import
#: ``sam3gimpd.types`` (different interpreter, zero-dependency rule).
RESULT_MAGIC = b"SAM3RES\x00"
RESULT_PREFIX_SIZE = 12  # len(RESULT_MAGIC) + 4 bytes of uint32 header length

#: API §8.3: value 128 corresponds to a raw logit of 0, i.e. the model's own
#: binarisation point, and is therefore the client's default threshold.
DEFAULT_MASK_THRESHOLD = 128

#: Width, in mask units, of the soft ramp used when turning a soft mask into an
#: overlay alpha channel.  Zero would give hard, aliased overlay edges; the ramp
#: is what makes the preview look like the eventual GEGL-scaled result.
DEFAULT_ALPHA_SOFTNESS = 10

MIN_ZOOM = 0.02
MAX_ZOOM = 64.0

_U32 = struct.Struct("<I")

#: cairo's 32-bit formats are native-endian ints, so the in-memory channel order
#: is B,G,R,X on little-endian machines and X,R,G,B on big-endian ones.
_LITTLE_ENDIAN = _sys.byteorder == "little"


# --------------------------------------------------------------------------- #
# result frame (API §8) -- stdlib only, so the harness and tests are self-sufficient
# --------------------------------------------------------------------------- #
def decode_frame(payload):
    """Split a ``application/vnd.sam3.result+binary`` body into header and blob.

    Returns ``(header_dict, blob_bytes)`` where ``blob_bytes`` is the blob
    *region*, i.e. exactly what :meth:`Sam3Canvas.set_result` expects, and
    instance ``blob_offset`` values index into it directly (API §8.1: the offset
    is relative to the blob region, never to the start of the body).

    Raises :exc:`ValueError` on any malformed frame.
    """
    payload = bytes(payload)
    if len(payload) < RESULT_PREFIX_SIZE:
        raise ValueError("result frame shorter than its %d byte prefix" % RESULT_PREFIX_SIZE)
    if payload[: len(RESULT_MAGIC)] != RESULT_MAGIC:
        raise ValueError("bad result magic %r" % (payload[: len(RESULT_MAGIC)],))
    (head_len,) = _U32.unpack_from(payload, len(RESULT_MAGIC))
    head_end = RESULT_PREFIX_SIZE + head_len
    if head_end > len(payload):
        raise ValueError("declared header length %d overruns the frame" % head_len)
    try:
        header = json.loads(payload[RESULT_PREFIX_SIZE:head_end].decode("utf-8"))
    except Exception as exc:
        raise ValueError("result header is not valid JSON: %s" % (exc,))
    if not isinstance(header, dict):
        raise ValueError("result header is not a JSON object")
    blob = payload[head_end:]
    declared = header.get("blob_length")
    if isinstance(declared, int) and declared != len(blob):
        raise ValueError("blob_length %d != actual %d" % (declared, len(blob)))
    return header, blob


def encode_frame(header, blobs):
    """Inverse of :func:`decode_frame`; used by the offline harness and tests.

    ``blobs`` is a sequence of mask byte strings in instance order.  The
    ``blob_offset`` / ``blob_length`` fields and the header total in ``header`` are
    recomputed so the frame is always self-consistent (API §16 rule 2 and 4).
    """
    header = dict(header)
    instances = [dict(i) for i in header.get("instances", [])]
    if len(instances) != len(blobs):
        raise ValueError("%d instances but %d blobs" % (len(instances), len(blobs)))
    offset = 0
    for inst, blob in zip(instances, blobs):
        inst["blob_offset"] = offset
        inst["blob_length"] = len(blob)
        offset += len(blob)
    header["instances"] = instances
    header["blob_length"] = offset
    raw = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    out = bytearray()
    out += RESULT_MAGIC
    out += _U32.pack(len(raw))
    out += raw
    for blob in blobs:
        out += blob
    return bytes(out)


# --------------------------------------------------------------------------- #
# geometry -- pure functions, no GTK, no state
# --------------------------------------------------------------------------- #

#: The widget's view of the image: ``widget = image * zoom + off``.
#:
#: Keeping the origin offset (rather than a centre point) makes zoom-at-pointer
#: a two-line calculation and makes every transform invertible in closed form.
View = namedtuple("View", "zoom off_x off_y")


def image_to_widget(view, ix, iy):
    """image (uploaded-image px) -> widget px."""
    return (ix * view.zoom + view.off_x, iy * view.zoom + view.off_y)


def widget_to_image(view, wx, wy):
    """widget px -> image (uploaded-image px).  Exact inverse of the above."""
    z = view.zoom if view.zoom else 1.0
    return ((wx - view.off_x) / z, (wy - view.off_y) / z)


def view_fit(image_w, image_h, area_w, area_h, margin=0.0, max_zoom=1.0):
    """The View that centres ``image_w x image_h`` inside ``area_w x area_h``.

    ``max_zoom`` defaults to ``1.0`` so a small image is shown at 100 % rather
    than blown up to fill the widget.
    """
    if image_w <= 0 or image_h <= 0 or area_w <= 0 or area_h <= 0:
        return View(1.0, 0.0, 0.0)
    avail_w = max(1.0, area_w - 2.0 * margin)
    avail_h = max(1.0, area_h - 2.0 * margin)
    zoom = min(avail_w / float(image_w), avail_h / float(image_h))
    zoom = max(MIN_ZOOM, min(float(max_zoom), zoom))
    off_x = (area_w - image_w * zoom) * 0.5
    off_y = (area_h - image_h * zoom) * 0.5
    return View(zoom, off_x, off_y)


def view_zoom_at(view, factor, wx, wy, min_zoom=MIN_ZOOM, max_zoom=MAX_ZOOM):
    """Scale by ``factor`` keeping the image point under ``(wx, wy)`` pinned.

    This is the invariant the tests assert: ``widget_to_image`` of the pointer
    is identical before and after, which is what makes wheel-zoom feel right.
    """
    old = view.zoom if view.zoom else 1.0
    new = max(min_zoom, min(max_zoom, old * float(factor)))
    if new == old:
        return view
    ix, iy = widget_to_image(view, wx, wy)
    return View(new, wx - ix * new, wy - iy * new)


def view_pan(view, dx, dy):
    """Translate the view by a widget-space delta."""
    return View(view.zoom, view.off_x + dx, view.off_y + dy)


def view_clamp(view, image_w, image_h, area_w, area_h):
    """Keep the image sensibly placed: centred when smaller, edge-locked when larger."""
    if image_w <= 0 or image_h <= 0:
        return view

    def _axis(off, extent, area):
        scaled = extent * view.zoom
        if scaled <= area:
            return (area - scaled) * 0.5
        return max(area - scaled, min(0.0, off))

    return View(view.zoom, _axis(view.off_x, image_w, area_w), _axis(view.off_y, image_h, area_h))


class Affine(namedtuple("Affine", "scale_x scale_y offset_x offset_y")):
    """The daemon's ``canvas_from_image`` transform (API §5).

    ``canvas = image * scale + offset``.  Built from whatever the daemon
    reports; the reference processor squashes anisotropically onto 1008x1008,
    but a letterboxing processor would set non-zero offsets and a client that
    assumed otherwise would misplace every mask.
    """

    __slots__ = ()

    @classmethod
    def identity(cls):
        return cls(1.0, 1.0, 0.0, 0.0)

    @classmethod
    def from_dict(cls, d):
        """Tolerant constructor: missing/zero scales fall back to identity."""
        if not d:
            return cls.identity()
        sx = float(d.get("scale_x", 1.0) or 1.0)
        sy = float(d.get("scale_y", 1.0) or 1.0)
        return cls(sx, sy, float(d.get("offset_x", 0.0)), float(d.get("offset_y", 0.0)))

    def to_canvas(self, ix, iy):
        return (ix * self.scale_x + self.offset_x, iy * self.scale_y + self.offset_y)

    def to_image(self, cx, cy):
        sx = self.scale_x or 1.0
        sy = self.scale_y or 1.0
        return ((cx - self.offset_x) / sx, (cy - self.offset_y) / sy)

    def rect_to_image(self, bbox):
        """model-canvas ``[x0,y0,x1,y1]`` -> uploaded-image floats (API §9 step 1)."""
        x0, y0 = self.to_image(bbox[0], bbox[1])
        x1, y1 = self.to_image(bbox[2], bbox[3])
        return (x0, y0, x1, y1)


# --------------------------------------------------------------------------- #
# instances and client-side thresholding
# --------------------------------------------------------------------------- #
class Instance:
    """One returned mask: a soft uint8 crop plus its model-canvas placement.

    Mirrors the per-instance object of ``_daemon/API.md`` §8.2.  The invariants
    the daemon guarantees (``mask_width == x1 - x0``, ``mask_height == y1 - y0``,
    ``len(mask) == mask_width * mask_height``) are *checked*, not assumed: a
    violated one is a wire bug and silently drawing garbage would hide it.
    """

    __slots__ = (
        "instance_id",
        "score",
        "label",
        "bbox",
        "mask_width",
        "mask_height",
        "mask",
        "visible",
        "selected",
        "color",
    )

    def __init__(self, instance_id, score, label, bbox, mask_width, mask_height, mask,
                 visible=True, selected=False, color=None):
        bbox = tuple(int(v) for v in bbox)
        if len(bbox) != 4:
            raise ValueError("bbox must have 4 elements")
        mask_width = int(mask_width)
        mask_height = int(mask_height)
        if mask_width != bbox[2] - bbox[0] or mask_height != bbox[3] - bbox[1]:
            raise ValueError(
                "mask %dx%d does not match bbox %r (API 8.2 invariant)"
                % (mask_width, mask_height, bbox)
            )
        if len(mask) != mask_width * mask_height:
            raise ValueError(
                "mask blob is %d bytes, expected %d" % (len(mask), mask_width * mask_height)
            )
        self.instance_id = int(instance_id)
        self.score = float(score)
        self.label = label or ""
        self.bbox = bbox
        self.mask_width = mask_width
        self.mask_height = mask_height
        self.mask = bytes(mask)
        self.visible = bool(visible)
        self.selected = bool(selected)
        self.color = color if color is not None else instance_color(self.instance_id)

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_header(cls, entry, blob):
        """Build from one ``header["instances"]`` entry and the frame's blob region."""
        off = int(entry["blob_offset"])
        length = int(entry["blob_length"])
        if off < 0 or off + length > len(blob):
            raise ValueError(
                "instance blob [%d, %d) overruns the %d byte blob region"
                % (off, off + length, len(blob))
            )
        return cls(
            instance_id=entry.get("instance_id", 0),
            score=entry.get("score", 0.0),
            label=entry.get("label", ""),
            bbox=entry["bbox"],
            mask_width=entry["mask_width"],
            mask_height=entry["mask_height"],
            mask=blob[off:off + length],
        )

    # -- geometry ---------------------------------------------------------- #
    @property
    def area(self):
        """Bounding-box area in model-canvas pixels (the hit-test tiebreak)."""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

    def __repr__(self):
        return "Instance(id=%d, score=%.3f, label=%r, bbox=%r)" % (
            self.instance_id, self.score, self.label, self.bbox,
        )


def instance_color(index):
    """A distinct, *stable* RGB colour for instance ``index``.

    The golden-ratio hue walk gives maximally spread hues for any prefix length,
    so instance 3 is the same colour whether the result had 4 instances or 40 --
    which matters because the dialog's list and the canvas must agree.
    """
    hue = (int(index) * 0.6180339887498949) % 1.0
    sat, val = 0.72, 1.0
    i = int(hue * 6.0)
    f = hue * 6.0 - i
    p = val * (1.0 - sat)
    q = val * (1.0 - sat * f)
    t = val * (1.0 - sat * (1.0 - f))
    return ((val, t, p), (q, val, p), (p, val, t), (p, q, val), (t, p, val), (val, p, q))[i % 6]


def alpha_table(threshold, softness=DEFAULT_ALPHA_SOFTNESS):
    """A 256-byte translation table mapping soft mask values to overlay alpha.

    This *is* the client-side threshold.  Applying it with
    :meth:`bytes.translate` re-thresholds a whole mask in one C-level pass, which
    is what makes the threshold slider instant with no network round trip
    (``_daemon/API.md`` §8, ``DESIGN.md`` §4).

    ``softness`` widens a linear ramp around the threshold so overlay edges are
    anti-aliased rather than jagged; ``softness=0`` gives a hard cut.
    """
    threshold = max(1, min(255, int(threshold)))
    softness = max(0, int(softness))
    if softness == 0:
        return bytes(0 if v < threshold else 255 for v in range(256))
    lo = threshold - softness
    hi = threshold + softness
    span = float(hi - lo)
    out = bytearray(256)
    for v in range(256):
        if v <= lo:
            out[v] = 0
        elif v >= hi:
            out[v] = 255
        else:
            out[v] = int(round(255.0 * (v - lo) / span))
    return bytes(out)


_BINARY_TABLES = {}


def binarize(mask, threshold):
    """Soft mask -> ``b"\\x00"``/``b"\\x01"`` bytes using ``inside = value >= threshold``.

    API §8.3 defines the comparison as a plain byte test; this is that test,
    vectorised through :meth:`bytes.translate`.
    """
    threshold = max(1, min(255, int(threshold)))
    table = _BINARY_TABLES.get(threshold)
    if table is None:
        table = bytes(0 if v < threshold else 1 for v in range(256))
        _BINARY_TABLES[threshold] = table
    return bytes(mask).translate(table)


def mask_sample(mask, mask_width, mask_height, mx, my):
    """Soft value at mask-local integer ``(mx, my)``; ``0`` outside the crop.

    API §8.3: "values outside the crop are defined to be 0".
    """
    mx = int(mx)
    my = int(my)
    if mx < 0 or my < 0 or mx >= mask_width or my >= mask_height:
        return 0
    return mask[my * mask_width + mx]


def instance_sample(inst, canvas_x, canvas_y):
    """Soft value of ``inst`` at a **model-canvas** point."""
    return mask_sample(
        inst.mask,
        inst.mask_width,
        inst.mask_height,
        math.floor(canvas_x) - inst.bbox[0],
        math.floor(canvas_y) - inst.bbox[1],
    )


def filter_by_score(instances, score_threshold):
    """The score slider, applied locally.

    PCS deliberately returns every candidate with score >= 0.1 (API §6.3) so the
    slider never costs a round trip.
    """
    t = float(score_threshold)
    return [i for i in instances if i.score >= t]


def hit_test(instances, canvas_x, canvas_y, mask_threshold=DEFAULT_MASK_THRESHOLD,
             score_threshold=0.0, require_visible=True):
    """Topmost instance whose thresholded mask covers a model-canvas point.

    Ties are broken by **smallest bounding box first**, then higher score, then
    lower id.  Smallest-first is deliberate: it is what lets a user click a small
    instance that sits inside a large one, which is the common case for
    "select every car" style prompts.  Returns ``None`` for a miss.
    """
    best = None
    for inst in instances:
        if require_visible and not inst.visible:
            continue
        if inst.score < score_threshold:
            continue
        if instance_sample(inst, canvas_x, canvas_y) < mask_threshold:
            continue
        key = (inst.area, -inst.score, inst.instance_id)
        if best is None or key < best[0]:
            best = (key, inst)
    return None if best is None else best[1]


# --------------------------------------------------------------------------- #
# contours -- the marching-ants outline
# --------------------------------------------------------------------------- #

#: Moore neighbourhood, clockwise starting east.  Index arithmetic on this list
#: is the whole of the boundary tracer.
_MOORE = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
_MOORE_INDEX = {d: i for i, d in enumerate(_MOORE)}


def trace_contours(binary, width, height, max_contours=48, max_points=20000):
    """Ordered closed boundary loops of a binary mask (Moore-neighbour tracing).

    ``binary`` is the output of :func:`binarize`: one byte per pixel, non-zero
    meaning inside.  Returns a list of lists of ``(x, y)`` **integer pixel
    coordinates**, each an ordered closed loop, outer boundaries and hole
    boundaries alike.

    Why tracing and not marching squares: a dash pattern needs an *ordered*
    path, otherwise cairo restarts the dash phase at every segment and the ants
    stop marching.

    Seed pixels are found by run-scanning each row with :meth:`bytes.find`, so
    the cost is proportional to the number of runs plus the total boundary
    length -- not to the mask area.  Both ``max_*`` arguments are belt-and-braces
    caps for pathological (noisy) masks.
    """
    if width <= 0 or height <= 0 or len(binary) < width * height:
        return []
    contours = []
    visited = set()
    total_points = 0
    step_cap = 4 * width * height + 16
    for y in range(height):
        base = y * width
        end = base + width
        pos = base
        while pos < end:
            i = binary.find(1, pos, end)
            if i < 0:
                break
            x = i - base
            if (x, y) not in visited:
                contour = _trace_one(binary, width, height, x, y, step_cap)
                for pt in contour:
                    visited.add(pt)
                if len(contour) >= 3:
                    contours.append(contour)
                    total_points += len(contour)
                    if len(contours) >= max_contours or total_points >= max_points:
                        return contours
            j = binary.find(0, i, end)
            if j < 0:
                break
            pos = j
    return contours


def _trace_one(binary, width, height, sx, sy, step_cap):
    """One clockwise loop starting at ``(sx, sy)``, whose left neighbour is outside.

    Termination is by *state repetition*: the walk is a deterministic function
    of ``(pixel, backtrack)``, so the first time a state recurs the loop is
    closed, whatever direction it happens to re-enter the start pixel from.
    Jacob's criterion -- stop on the start pixel with the start backtrack --
    is the special case where the cycle passes through the initial state,
    and on speckled masks it routinely does not: the walk then circled the
    same boundary for the whole ``step_cap`` (four times the pixel count,
    three million steps on a 1008x756 mask) and the window sat "Not
    Responding" for minutes on "person" while "guitar" traced in a
    millisecond.  Points before the first recurrence are a lead-in that is
    not part of the cycle and are dropped.
    """
    contour = []
    seen = {}
    px, py = sx, sy
    bx, by = sx - 1, sy
    for _ in range(step_cap):
        state = (px, py, bx, by)
        first = seen.get(state)
        if first is not None:
            return contour[first:]
        seen[state] = len(contour)
        contour.append((px, py))
        d = _MOORE_INDEX[(bx - px, by - py)]
        moved = False
        for k in range(1, 9):
            idx = (d + k) & 7
            dx, dy = _MOORE[idx]
            nx, ny = px + dx, py + dy
            if 0 <= nx < width and 0 <= ny < height and binary[ny * width + nx]:
                pdx, pdy = _MOORE[(idx - 1) & 7]
                bx, by = px + pdx, py + pdy
                px, py = nx, ny
                moved = True
                break
        if not moved:
            break  # isolated pixel
    return contour


def simplify_polyline(points, epsilon):
    """Iterative Douglas-Peucker.

    Iterative rather than recursive because a 1008-px contour can be thousands
    of points deep and CPython's recursion limit is not a place to find that
    out.  Endpoints are always kept.
    """
    n = len(points)
    if n < 3 or epsilon <= 0:
        return list(points)
    keep = [False] * n
    keep[0] = keep[n - 1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        ax, ay = points[i]
        bx, by = points[j]
        dx = bx - ax
        dy = by - ay
        norm = math.hypot(dx, dy)
        best = -1.0
        best_k = -1
        if norm == 0.0:
            for k in range(i + 1, j):
                d = math.hypot(points[k][0] - ax, points[k][1] - ay)
                if d > best:
                    best, best_k = d, k
        else:
            for k in range(i + 1, j):
                px, py = points[k]
                d = abs(dy * (px - ax) - dx * (py - ay)) / norm
                if d > best:
                    best, best_k = d, k
        if best > epsilon and best_k > 0:
            keep[best_k] = True
            stack.append((i, best_k))
            stack.append((best_k, j))
    return [points[k] for k in range(n) if keep[k]]


def contour_polygons(mask, mask_width, mask_height, threshold, epsilon=0.75):
    """Simplified outline polygons of a soft mask, in **mask-local** coordinates.

    Points sit at pixel centres (``x + 0.5``) so the stroke runs down the middle
    of the boundary pixels instead of half a pixel off.
    """
    binary = binarize(mask, threshold)
    polys = []
    for contour in trace_contours(binary, mask_width, mask_height):
        pts = [(x + 0.5, y + 0.5) for (x, y) in contour]
        pts = simplify_polyline(pts, epsilon)
        if len(pts) >= 3:
            polys.append(pts)
    return polys


# --------------------------------------------------------------------------- #
# cairo surfaces
# --------------------------------------------------------------------------- #
def surface_from_rgb(data, width, height):
    """Packed RGB u8 (the ``POST /images`` layout, API §7) -> ``cairo`` RGB24 surface.

    The interleave is four strided slice assignments rather than a Python loop,
    so a full 1008x1008 upload converts in milliseconds without numpy.
    """
    if cairo is None:  # pragma: no cover
        raise RuntimeError("pycairo is unavailable")
    width = int(width)
    height = int(height)
    expected = width * height * 3
    if width <= 0 or height <= 0:
        raise ValueError("image must be at least 1x1")
    if len(data) != expected:
        raise ValueError("expected %d RGB bytes for %dx%d, got %d" % (expected, width, height, len(data)))
    data = bytes(data)
    npx = width * height
    packed = bytearray(npx * 4)
    if _LITTLE_ENDIAN:
        packed[0::4] = data[2::3]  # B
        packed[1::4] = data[1::3]  # G
        packed[2::4] = data[0::3]  # R
        packed[3::4] = b"\xff" * npx
    else:  # pragma: no cover - no big-endian box to test on
        packed[0::4] = b"\xff" * npx
        packed[1::4] = data[0::3]
        packed[2::4] = data[1::3]
        packed[3::4] = data[2::3]

    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, width, height)
    surface.flush()
    stride = surface.get_stride()
    row = width * 4
    buf = surface.get_data()
    if stride == row:
        buf[: npx * 4] = packed
    else:
        for y in range(height):
            buf[y * stride:y * stride + row] = packed[y * row:(y + 1) * row]
    surface.mark_dirty()
    return surface


def alpha_surface_from_mask(mask, mask_width, mask_height, table):
    """Soft mask + a 256-byte :func:`alpha_table` -> ``cairo`` A8 surface.

    A8 is the trick that makes re-thresholding free: the mask bytes *are* the
    alpha channel, so the whole operation is one :meth:`bytes.translate` plus a
    memcpy per row.  Painting is then
    ``ctx.set_source_rgba(colour); ctx.mask_surface(a8)`` -- cairo does the
    (smooth, filtered) upscale, which is the preview-side analogue of letting
    GEGL scale the soft mask at apply time (API §9 step 4).
    """
    if cairo is None:  # pragma: no cover
        raise RuntimeError("pycairo is unavailable")
    mask_width = int(mask_width)
    mask_height = int(mask_height)
    if mask_width <= 0 or mask_height <= 0:
        return None
    alpha = bytes(mask).translate(table)
    surface = cairo.ImageSurface(cairo.FORMAT_A8, mask_width, mask_height)
    surface.flush()
    stride = surface.get_stride()
    buf = surface.get_data()
    if stride == mask_width:
        buf[: mask_width * mask_height] = alpha
    else:
        for y in range(mask_height):
            buf[y * stride:y * stride + mask_width] = alpha[y * mask_width:(y + 1) * mask_width]
    surface.mark_dirty()
    return surface


# --------------------------------------------------------------------------- #
# the widget
# --------------------------------------------------------------------------- #
if GTK_AVAILABLE:

    _MODE_SELECT = "select"
    _MODE_POINTS = "points"

    _CLICK_SLOP_PX = 4.0  # a press/release further apart than this is a drag, not a click

    # PyGObject's Gdk.Event override shadows the union member names (``event.button``
    # yields the GdkEventButton *struct*, not the button number) on a generic
    # Gdk.Event.  The ``get_*`` accessors work on both a generic event and the
    # concrete one GTK hands a vfunc, so the widget uses those exclusively.  That
    # also makes synthesised events -- how the tests drive input -- behave
    # identically to real ones.
    # The accessors' return shape differs between PyGObject builds: most return
    # ``(ok, value...)``, but the PyGObject shipped with GIMP 3.2 on Windows
    # drops the boolean and returns the values alone -- ``get_coords()`` gave
    # ``(x, y)`` and every mouse movement over the canvas raised
    # "not enough values to unpack (expected 3, got 2)".  ``_ev_unpack``
    # accepts both.
    def _ev_unpack(result, n, default):
        """``(ok, v1..vn)`` or ``(v1..vn)`` or a bare value -> ``(v1..vn)``."""
        if not isinstance(result, (tuple, list)):
            return (result,) if n == 1 else default
        values = tuple(result)
        if len(values) == n + 1 and isinstance(values[0], bool):
            return values[1:] if values[0] else default
        if len(values) == n:
            return values
        return default

    def _ev_coords(event):
        return tuple(float(v) for v in _ev_unpack(event.get_coords(), 2, (0.0, 0.0)))

    def _ev_button(event):
        return int(_ev_unpack(event.get_button(), 1, (0,))[0])

    def _ev_state(event):
        return _ev_unpack(event.get_state(), 1, (Gdk.ModifierType(0),))[0]

    def _ev_keyval(event):
        return int(_ev_unpack(event.get_keyval(), 1, (0,))[0])

    class Sam3Canvas(Gtk.DrawingArea):
        """Interactive preview: image + instance overlays + point prompts.

        The widget owns *rendering and input*, nothing else.  It never calls the
        dialog, never touches HTTP and never imports GIMP; it emits signals and
        exposes setters.  That keeps it reusable (the standalone
        ``tools/canvas_harness.py`` drives the very same class) and keeps the
        threading rule of ``DESIGN.md`` §4 easy to honour: the dialog's worker
        thread marshals results here with ``GLib.idle_add``.

        Signals
        -------
        ``point-added(x: float, y: float, label: int)``
            A prompt point was placed, in **uploaded-image** coordinates --
            exactly the space ``POST /images/{id}/points`` expects.  ``label`` is
            ``1`` for include, ``0`` for exclude.
        ``points-changed()``
            The point list changed for any reason (added, undone, cleared).
        ``instance-toggled(instance_id: int, selected: bool)``
            ``set_instance_selected`` changed the optional highlight border.
            Clicks no longer drive it; they change the tick below.
        ``instance-visibility-changed(instance_id: int, visible: bool)``
            A click (left or right) in ``select`` mode ticked or unticked an
            instance: ``visible`` is the flag the dialog's checkbox shows and
            Apply uses, and an unticked instance is drawn no more.
        ``instance-activated(instance_id: int)``
            The active instance (the one wearing the marching ants) changed;
            ``-1`` when nothing is active.
        ``instance-hovered(instance_id: int)``
            Hover moved; ``-1`` when the pointer is over no instance.
        ``threshold-changed(mask_threshold: int, score_threshold: float)``
            Thresholds changed *from inside the widget* (keyboard shortcuts).
            Both are applied locally with no round trip.
        ``opacity-changed(opacity: float)``
        ``view-changed(zoom: float, off_x: float, off_y: float)``
        ``box-drawn(x0: float, y0: float, x1: float, y1: float)``
            Ctrl-drag finished; **uploaded-image** coordinates, normalised so
            ``x1 > x0`` and ``y1 > y0`` and clipped to the image.  A drag that
            leaves no area inside the image emits nothing.
        """

        __gtype_name__ = "Sam3Canvas"

        __gsignals__ = {
            "point-added": (GObject.SignalFlags.RUN_FIRST, None, (float, float, int)),
            "points-changed": (GObject.SignalFlags.RUN_FIRST, None, ()),
            "instance-toggled": (GObject.SignalFlags.RUN_FIRST, None, (int, bool)),
            "instance-visibility-changed": (GObject.SignalFlags.RUN_FIRST, None, (int, bool)),
            "instance-activated": (GObject.SignalFlags.RUN_FIRST, None, (int,)),
            "instance-hovered": (GObject.SignalFlags.RUN_FIRST, None, (int,)),
            "threshold-changed": (GObject.SignalFlags.RUN_FIRST, None, (int, float)),
            "opacity-changed": (GObject.SignalFlags.RUN_FIRST, None, (float,)),
            "view-changed": (GObject.SignalFlags.RUN_FIRST, None, (float, float, float)),
            "box-drawn": (GObject.SignalFlags.RUN_FIRST, None, (float, float, float, float)),
        }

        # ------------------------------------------------------------------ #
        # construction
        # ------------------------------------------------------------------ #
        def __init__(self, mask_threshold=DEFAULT_MASK_THRESHOLD, score_threshold=0.1,
                     overlay_opacity=0.55, mode=_MODE_SELECT):
            Gtk.DrawingArea.__init__(self)

            # image (uploaded-image space)
            self._image_surface = None
            self._image_w = 0
            self._image_h = 0

            # results
            self._instances = []
            self._affine = Affine.identity()
            self._header = {}
            self._model_canvas = (0, 0)

            # local filtering -- never a round trip
            self._mask_threshold = int(mask_threshold)
            self._score_threshold = float(score_threshold)
            self._overlay_opacity = float(overlay_opacity)
            self._alpha_softness = DEFAULT_ALPHA_SOFTNESS
            self._alpha_table = alpha_table(self._mask_threshold, self._alpha_softness)
            self._alpha_cache = {}     # instance_id -> (threshold, softness, A8 surface)
            self._overlay_cache = None  # (key, widget-sized ARGB surface of every visible mask)
            self._contour_cache = {}   # instance_id -> (threshold, [polygons])

            # view
            self._view = View(1.0, 0.0, 0.0)
            self._need_fit = True

            # interaction
            self._mode = mode
            self._points = []          # [(x, y, label)] in image space
            self._hover_id = -1
            self._active_id = -1
            self._pan_anchor = None    # (widget_x, widget_y, View)
            self._press = None         # (button, wx, wy, state)
            self._box_drag = None      # (ix0, iy0, ix1, iy1) image space
            self._boxes = []           # kept boxes: (ix0, iy0, ix1, iy1, label), image space
            self._space_down = False

            # chrome
            self._busy = False
            self._progress = 0.0
            self._stage = ""
            self._status_text = ""
            self._show_ants = True
            self._ants_phase = 0.0
            self._ants_source = 0

            self.set_can_focus(True)
            self.set_size_request(320, 240)
            self.set_has_tooltip(False)
            self.add_events(
                Gdk.EventMask.BUTTON_PRESS_MASK
                | Gdk.EventMask.BUTTON_RELEASE_MASK
                | Gdk.EventMask.POINTER_MOTION_MASK
                | Gdk.EventMask.SCROLL_MASK
                | Gdk.EventMask.SMOOTH_SCROLL_MASK
                | Gdk.EventMask.LEAVE_NOTIFY_MASK
                | Gdk.EventMask.ENTER_NOTIFY_MASK
                | Gdk.EventMask.KEY_PRESS_MASK
                | Gdk.EventMask.KEY_RELEASE_MASK
                | Gdk.EventMask.FOCUS_CHANGE_MASK
            )
            self.connect("destroy", lambda *_a: self._stop_ants())

        # ------------------------------------------------------------------ #
        # image
        # ------------------------------------------------------------------ #
        def set_image(self, data, width, height, keep_view=False):
            """Set the displayed buffer from packed RGB u8 (API §7 layout)."""
            self._image_surface = surface_from_rgb(data, width, height)
            self._image_w = int(width)
            self._image_h = int(height)
            if not keep_view:
                self._need_fit = True
            self.clear_result()
            self.queue_draw()

        def set_image_surface(self, surface, width, height, keep_view=False):
            """Set the displayed buffer from an existing cairo surface."""
            self._image_surface = surface
            self._image_w = int(width)
            self._image_h = int(height)
            if not keep_view:
                self._need_fit = True
            self.clear_result()
            self.queue_draw()

        @property
        def image_size(self):
            return (self._image_w, self._image_h)

        @property
        def has_image(self):
            return self._image_surface is not None

        # ------------------------------------------------------------------ #
        # results
        # ------------------------------------------------------------------ #
        def set_result(self, header, blob, keep_view=True):
            """Adopt a decoded result frame (API §8): JSON header + blob region.

            Staleness is *not* judged here -- the dialog owns the "drop any
            result whose ``request_id`` is not the latest" rule (API §10) because
            only it knows what it last issued.  :attr:`request_id` is exposed so
            it can assert after the fact.

            Malformed instances are skipped rather than fatal: one bad mask must
            not blank the preview.
            """
            header = dict(header or {})
            self._header = header
            self._affine = Affine.from_dict(header.get("canvas_from_image"))
            mc = header.get("model_canvas") or {}
            self._model_canvas = (int(mc.get("width", 0)), int(mc.get("height", 0)))

            instances = []
            for entry in header.get("instances", []):
                try:
                    instances.append(Instance.from_header(entry, blob))
                except (KeyError, ValueError, TypeError):
                    continue
            for pos, inst in enumerate(instances):
                inst.color = instance_color(inst.instance_id if inst.instance_id >= 0 else pos)
            self._instances = instances
            self._alpha_cache.clear()
            self._overlay_cache = None
            self._contour_cache.clear()
            self._set_active(instances[0].instance_id if instances else -1, notify=True)
            self._hover_id = -1
            if not keep_view:
                self._need_fit = True
            self._sync_ants()
            self.queue_draw()

        def set_result_frame(self, payload, keep_view=True):
            """Convenience: decode a raw frame body and adopt it."""
            header, blob = decode_frame(payload)
            self.set_result(header, blob, keep_view=keep_view)
            return header

        def clear_result(self):
            self._instances = []
            self._header = {}
            self._alpha_cache.clear()
            self._overlay_cache = None
            self._contour_cache.clear()
            self._hover_id = -1
            self._set_active(-1, notify=False)
            self._sync_ants()
            self.queue_draw()

        @property
        def instances(self):
            """All instances, unfiltered.  Do not mutate the list itself."""
            return list(self._instances)

        @property
        def visible_instances(self):
            """Instances that pass the score slider *and* are not hidden."""
            return [i for i in self._instances if i.visible and i.score >= self._score_threshold]

        @property
        def request_id(self):
            return self._header.get("request_id")

        @property
        def canvas_from_image(self):
            return self._affine

        def get_instance(self, instance_id):
            for inst in self._instances:
                if inst.instance_id == instance_id:
                    return inst
            return None

        def selected_instance_ids(self):
            return [i.instance_id for i in self._instances if i.selected]

        def set_instance_selected(self, instance_id, selected, notify=True):
            inst = self.get_instance(instance_id)
            if inst is None or inst.selected == bool(selected):
                return
            inst.selected = bool(selected)
            if notify:
                self.emit("instance-toggled", inst.instance_id, inst.selected)
            self.queue_draw()

        def set_instance_visible(self, instance_id, visible, notify=True):
            inst = self.get_instance(instance_id)
            if inst is None or inst.visible == bool(visible):
                return
            inst.visible = bool(visible)
            if notify:
                self.emit("instance-visibility-changed", inst.instance_id, inst.visible)
            self.queue_draw()

        def set_all_visible(self, visible):
            for inst in self._instances:
                inst.visible = bool(visible)
            self.queue_draw()

        @property
        def active_instance_id(self):
            return self._active_id

        def set_active_instance(self, instance_id):
            self._set_active(instance_id, notify=True)
            self.queue_draw()

        def _set_active(self, instance_id, notify):
            instance_id = int(instance_id)
            if instance_id == self._active_id:
                return
            self._active_id = instance_id
            self._sync_ants()
            if notify:
                self.emit("instance-activated", instance_id)

        @property
        def hovered_instance_id(self):
            return self._hover_id

        # ------------------------------------------------------------------ #
        # local thresholds -- the whole point of the soft mask wire format
        # ------------------------------------------------------------------ #
        @property
        def mask_threshold(self):
            return self._mask_threshold

        def set_mask_threshold(self, value, notify=False):
            """Re-threshold every mask locally.  No HTTP, no re-inference.

            Only the A8 alpha surfaces and the active contour are invalidated;
            the base image surface, the view and the point list are untouched,
            so a slider drag repaints at whatever rate GTK can composite.
            """
            value = max(1, min(255, int(value)))
            if value == self._mask_threshold:
                return
            self._mask_threshold = value
            self._alpha_table = alpha_table(value, self._alpha_softness)
            self._alpha_cache.clear()
            self._overlay_cache = None
            self._contour_cache.clear()
            if notify:
                self.emit("threshold-changed", self._mask_threshold, self._score_threshold)
            self.queue_draw()

        @property
        def score_threshold(self):
            return self._score_threshold

        def set_score_threshold(self, value, notify=False):
            """Re-filter instances locally (PCS returns everything >= 0.1, API §6.3)."""
            value = max(0.0, min(1.0, float(value)))
            if value == self._score_threshold:
                return
            self._score_threshold = value
            if self._active_id >= 0:
                active = self.get_instance(self._active_id)
                if active is not None and active.score < value:
                    self._set_active(-1, notify=True)
            if notify:
                self.emit("threshold-changed", self._mask_threshold, self._score_threshold)
            self.queue_draw()

        @property
        def overlay_opacity(self):
            return self._overlay_opacity

        def set_overlay_opacity(self, value, notify=False):
            value = max(0.0, min(1.0, float(value)))
            if value == self._overlay_opacity:
                return
            self._overlay_opacity = value
            if notify:
                self.emit("opacity-changed", value)
            self.queue_draw()

        @property
        def alpha_softness(self):
            return self._alpha_softness

        def set_alpha_softness(self, value):
            value = max(0, min(64, int(value)))
            if value == self._alpha_softness:
                return
            self._alpha_softness = value
            self._alpha_table = alpha_table(self._mask_threshold, value)
            self._alpha_cache.clear()
            self._overlay_cache = None
            self.queue_draw()

        @property
        def show_ants(self):
            return self._show_ants

        def set_show_ants(self, enabled):
            self._show_ants = bool(enabled)
            self._sync_ants()
            self.queue_draw()

        # ------------------------------------------------------------------ #
        # point prompts
        # ------------------------------------------------------------------ #
        @property
        def points(self):
            """``[(x, y, label)]`` in **uploaded-image** coordinates (API §5)."""
            return list(self._points)

        def add_point(self, x, y, label=1, notify=True):
            self._points.append((float(x), float(y), 1 if label else 0))
            if notify:
                self.emit("point-added", float(x), float(y), 1 if label else 0)
                self.emit("points-changed")
            self.queue_draw()

        def remove_last_point(self):
            if not self._points:
                return False
            self._points.pop()
            self.emit("points-changed")
            self.queue_draw()
            return True

        def clear_points(self):
            if not self._points:
                return
            self._points = []
            self.emit("points-changed")
            self.queue_draw()

        # ------------------------------------------------------------------ #
        # interaction mode
        # ------------------------------------------------------------------ #
        @property
        def interaction_mode(self):
            return self._mode

        def set_interaction_mode(self, mode):
            """``"select"`` browses PCS instances; ``"points"`` places PVS prompts.

            Two modes exist because the two gestures genuinely collide: clicking
            an instance to include it and clicking a pixel to say "this one" are
            the same click.  The dialog switches modes when the user moves
            between the text prompt and the click-refine controls.
            """
            if mode not in (_MODE_SELECT, _MODE_POINTS):
                raise ValueError("mode must be 'select' or 'points'")
            if mode == self._mode:
                return
            self._mode = mode
            self._update_cursor()
            self.queue_draw()

        # ------------------------------------------------------------------ #
        # view
        # ------------------------------------------------------------------ #
        @property
        def view(self):
            return self._view

        def set_view(self, view, clamp=True, notify=True):
            alloc = self.get_allocation()
            if clamp:
                view = view_clamp(view, self._image_w, self._image_h, alloc.width, alloc.height)
            if view == self._view:
                return
            self._view = view
            if notify:
                self.emit("view-changed", view.zoom, view.off_x, view.off_y)
            self.queue_draw()

        def zoom_fit(self):
            alloc = self.get_allocation()
            self._need_fit = False
            self.set_view(
                view_fit(self._image_w, self._image_h, alloc.width, alloc.height, margin=6.0),
                clamp=False,
            )

        def zoom_to(self, zoom, center=None):
            """Set an absolute zoom, keeping ``center`` (widget px) pinned."""
            alloc = self.get_allocation()
            if center is None:
                center = (alloc.width * 0.5, alloc.height * 0.5)
            current = self._view.zoom or 1.0
            self._need_fit = False
            self.set_view(view_zoom_at(self._view, float(zoom) / current, center[0], center[1]))

        def zoom_by(self, factor, center=None):
            alloc = self.get_allocation()
            if center is None:
                center = (alloc.width * 0.5, alloc.height * 0.5)
            self._need_fit = False
            self.set_view(view_zoom_at(self._view, factor, center[0], center[1]))

        # ------------------------------------------------------------------ #
        # chrome
        # ------------------------------------------------------------------ #
        def set_busy(self, busy, stage="", progress=0.0):
            self._busy = bool(busy)
            self._stage = stage or ""
            self._progress = max(0.0, min(1.0, float(progress)))
            self.queue_draw()

        def set_progress(self, progress, stage=None):
            self._progress = max(0.0, min(1.0, float(progress)))
            if stage is not None:
                self._stage = stage
            self.queue_draw()

        def set_status_text(self, text):
            self._status_text = text or ""
            self.queue_draw()

        # ------------------------------------------------------------------ #
        # coordinate helpers (widget <-> image <-> model-canvas)
        # ------------------------------------------------------------------ #
        def widget_to_image_point(self, wx, wy):
            return widget_to_image(self._view, wx, wy)

        def image_to_widget_point(self, ix, iy):
            return image_to_widget(self._view, ix, iy)

        def widget_to_canvas_point(self, wx, wy):
            ix, iy = widget_to_image(self._view, wx, wy)
            return self._affine.to_canvas(ix, iy)

        def canvas_to_widget_point(self, cx, cy):
            ix, iy = self._affine.to_image(cx, cy)
            return image_to_widget(self._view, ix, iy)

        def instance_at_widget(self, wx, wy):
            """Hit-test a widget point against the *currently visible* instances."""
            cx, cy = self.widget_to_canvas_point(wx, wy)
            return hit_test(
                self._instances,
                cx,
                cy,
                mask_threshold=self._mask_threshold,
                score_threshold=self._score_threshold,
                require_visible=True,
            )

        # ------------------------------------------------------------------ #
        # cached derived data
        # ------------------------------------------------------------------ #
        def _alpha_surface_for(self, inst):
            """The instance's A8 alpha surface at the current threshold.

            Cached per ``(instance, threshold, softness)``: panning, zooming and
            opacity changes reuse it, and only a threshold move rebuilds -- which
            is one ``bytes.translate`` plus a row memcpy anyway.
            """
            key = inst.instance_id
            cached = self._alpha_cache.get(key)
            if cached is not None and cached[0] == self._mask_threshold and cached[1] == self._alpha_softness:
                return cached[2]
            surface = alpha_surface_from_mask(
                inst.mask, inst.mask_width, inst.mask_height, self._alpha_table
            )
            self._alpha_cache[key] = (self._mask_threshold, self._alpha_softness, surface)
            return surface

        def _contours_for(self, inst):
            """Simplified mask-local outline polygons for the marching ants."""
            key = inst.instance_id
            cached = self._contour_cache.get(key)
            if cached is not None and cached[0] == self._mask_threshold:
                return cached[1]
            polys = contour_polygons(
                inst.mask, inst.mask_width, inst.mask_height, self._mask_threshold
            )
            self._contour_cache[key] = (self._mask_threshold, polys)
            return polys

        # ------------------------------------------------------------------ #
        # drawing
        # ------------------------------------------------------------------ #
        def _ensure_view(self, area_w, area_h):
            """Fit on the first real allocation, then keep the view in bounds.

            Done here rather than only in ``do_draw`` so that the transforms are
            correct the instant the widget has a size -- a click that arrives
            before the first frame must not be interpreted against a stale view.
            """
            if not self._image_w or area_w <= 1 or area_h <= 1:
                return
            if self._need_fit:
                self._need_fit = False
                self._view = view_fit(self._image_w, self._image_h, area_w, area_h, margin=6.0)
            else:
                self._view = view_clamp(self._view, self._image_w, self._image_h, area_w, area_h)

        def do_draw(self, cr):
            alloc = self.get_allocation()
            width, height = alloc.width, alloc.height

            cr.set_source_rgb(0.14, 0.14, 0.15)
            cr.paint()

            if self._image_surface is None:
                self._draw_placeholder(cr, width, height)
                self._draw_chrome(cr, width, height)
                return False

            self._ensure_view(width, height)
            view = self._view

            # 1. the image itself
            cr.save()
            cr.translate(view.off_x, view.off_y)
            cr.scale(view.zoom, view.zoom)
            cr.set_source_surface(self._image_surface, 0, 0)
            pattern = cr.get_source()
            # Above 2x, show honest pixels rather than a blur; below, filter well.
            pattern.set_filter(cairo.FILTER_NEAREST if view.zoom >= 2.0 else cairo.FILTER_GOOD)
            cr.rectangle(0, 0, self._image_w, self._image_h)
            cr.fill()
            cr.restore()

            # 2. mask overlays, filtered locally by the score slider.  Every
            #    visible mask is composited once into a widget-sized surface
            #    and that surface is painted per frame (see _overlay_for); the
            #    active and hovered instances get their extra emphasis on top.
            drawn = self.visible_instances
            overlay = self._overlay_for(width, height, drawn)
            if overlay is not None:
                cr.save()
                cr.set_source_surface(overlay, 0, 0)
                cr.paint()
                cr.restore()
            for inst in drawn:
                if inst.instance_id == self._hover_id:
                    self._draw_overlay(cr, inst, min(1.0, self._overlay_opacity * 0.55))
                elif inst.instance_id == self._active_id:
                    self._draw_overlay(cr, inst, min(1.0, self._overlay_opacity * 0.25))

            # 3. selection ticks: a solid border on every instance the user included
            hovered = None
            for inst in drawn:
                if inst.selected:
                    self._draw_bbox(cr, inst, dash=None, alpha=0.95, line_width=2.0)
                elif inst.instance_id == self._hover_id:
                    self._draw_bbox(cr, inst, dash=[3.0, 3.0], alpha=0.8, line_width=1.0)
                if inst.instance_id == self._hover_id:
                    hovered = inst
            if hovered is not None:
                # The list has scores; the preview had none, so relating a
                # mask to its row meant reading sixty rows.
                self._draw_badge(cr, hovered)

            # 4. marching ants on the active instance
            if self._show_ants and self._active_id >= 0:
                active = self.get_instance(self._active_id)
                if active is not None and active.visible and active.score >= self._score_threshold:
                    self._draw_ants(cr, active)

            # 5. prompt points and the in-progress box
            for box in self._boxes:
                self._draw_kept_box(cr, box)
            self._draw_points(cr)
            if self._box_drag is not None:
                self._draw_drag_box(cr, self._box_drag)

            # 6. image border, so the extent is unambiguous when zoomed out
            x0, y0 = image_to_widget(view, 0, 0)
            x1, y1 = image_to_widget(view, self._image_w, self._image_h)
            cr.set_source_rgba(1, 1, 1, 0.18)
            cr.set_line_width(1.0)
            cr.rectangle(x0 + 0.5, y0 + 0.5, max(1.0, x1 - x0 - 1.0), max(1.0, y1 - y0 - 1.0))
            cr.stroke()

            self._draw_chrome(cr, width, height)
            return False

        def _canvas_matrix(self, cr, inst):
            """Set the CTM so user units are the instance's mask-local pixels.

            widget <- image <- model-canvas <- mask-local, i.e. the view
            transform, then the inverse of the daemon's reported
            ``canvas_from_image`` (never a hardcoded 1008 squash), then the
            instance's bbox origin.
            """
            view = self._view
            aff = self._affine
            cr.translate(view.off_x, view.off_y)
            cr.scale(view.zoom, view.zoom)
            cr.scale(1.0 / (aff.scale_x or 1.0), 1.0 / (aff.scale_y or 1.0))
            cr.translate(-aff.offset_x, -aff.offset_y)
            cr.translate(inst.bbox[0], inst.bbox[1])

        def _overlay_for(self, width, height, drawn):
            """All visible masks composited once, at widget resolution.

            The marching ants redraw the whole widget eleven times a second,
            and before this every tick re-composited every visible mask through
            cairo's scaler: 2 ms for one guitar, 144 ms for 64 person-sized
            masks -- longer than the tick, so the main loop never drained and
            the window read as frozen.  Now a tick is one surface paint.

            Keyed on everything the composite depends on -- view, thresholds,
            opacity and the ordered set of visible ids -- and compared per
            frame, so no mutation path has to remember to invalidate it; the
            explicit resets next to ``_alpha_cache.clear()`` only cover a new
            result reusing the same ids.
            """
            scale = self._device_scale()
            key = (int(width), int(height), scale, self._view, self._score_threshold,
                   self._mask_threshold, self._alpha_softness, self._overlay_opacity,
                   tuple(inst.instance_id for inst in drawn))
            cached = self._overlay_cache
            if cached is not None and cached[0] == key:
                return cached[1]
            if not drawn or width <= 0 or height <= 0:
                self._overlay_cache = (key, None)
                return None
            # Rendered at the screen's own resolution: a logical-size surface
            # painted on a HiDPI display is scaled up, and every mask edge
            # comes out soft.
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,
                                         int(width) * scale, int(height) * scale)
            if scale != 1:
                surface.set_device_scale(scale, scale)
            ctx = cairo.Context(surface)
            for inst in drawn:
                self._draw_overlay(ctx, inst, self._overlay_opacity)
            surface.flush()
            self._overlay_cache = (key, surface)
            return surface

        def _device_scale(self):
            """The widget's scale factor (2 on most HiDPI screens), or 1 when
            this pycairo cannot give a surface a device scale."""
            try:
                scale = int(self.get_scale_factor())
            except Exception:  # pragma: no cover - no window yet
                return 1
            if scale <= 1 or not hasattr(cairo.ImageSurface, "set_device_scale"):
                return 1
            return scale

        def _draw_overlay(self, cr, inst, alpha):
            surface = self._alpha_surface_for(inst)
            if surface is None or alpha <= 0.0:
                return
            r, g, b = inst.color
            cr.save()
            self._canvas_matrix(cr, inst)
            cr.set_source_rgba(r, g, b, alpha)
            cr.mask_surface(surface, 0, 0)
            cr.restore()

        def _draw_bbox(self, cr, inst, dash, alpha, line_width):
            ix0, iy0, ix1, iy1 = self._affine.rect_to_image(inst.bbox)
            x0, y0 = image_to_widget(self._view, ix0, iy0)
            x1, y1 = image_to_widget(self._view, ix1, iy1)
            r, g, b = inst.color
            cr.save()
            cr.set_line_width(line_width)
            cr.set_dash(dash or [], 0)
            cr.set_source_rgba(r, g, b, alpha)
            cr.rectangle(x0 + 0.5, y0 + 0.5, max(1.0, x1 - x0), max(1.0, y1 - y0))
            cr.stroke()
            cr.restore()

        def _draw_badge(self, cr, inst):
            """``label 0.93`` in a small box at the hovered instance's top-left."""
            ix0, iy0, _ix1, _iy1 = self._affine.rect_to_image(inst.bbox)
            x0, y0 = image_to_widget(self._view, ix0, iy0)
            text = "%s %.2f" % (inst.label or "instance %d" % inst.instance_id, inst.score)
            cr.save()
            cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            cr.set_font_size(11)
            ext = cr.text_extents(text)
            pad = 4.0
            bw, bh = ext.width + 2 * pad, ext.height + 2 * pad
            bx = max(0.0, x0)
            by = max(0.0, y0 - bh - 2.0)
            r, g, b = inst.color
            cr.set_source_rgba(r * 0.6, g * 0.6, b * 0.6, 0.9)
            cr.rectangle(bx, by, bw, bh)
            cr.fill()
            cr.set_source_rgba(1, 1, 1, 0.95)
            cr.move_to(bx + pad - ext.x_bearing, by + pad - ext.y_bearing)
            cr.show_text(text)
            cr.restore()

        def _draw_ants(self, cr, inst):
            """Black/white dashed outline, phase-animated: classic marching ants.

            The polygons are transformed to widget space in Python rather than by
            the CTM, because the canvas transform is *anisotropic* -- letting
            cairo scale the stroke would give a 1 px line horizontally and a
            1.5 px line vertically, and would stretch the dash pattern too.
            """
            polys = self._contours_for(inst)
            if not polys:
                return
            view = self._view
            aff = self._affine
            bx, by = inst.bbox[0], inst.bbox[1]
            cr.save()
            cr.set_line_width(1.0)
            cr.new_path()
            for poly in polys:
                first = True
                for (mx, my) in poly:
                    ix, iy = aff.to_image(bx + mx, by + my)
                    wx, wy = image_to_widget(view, ix, iy)
                    if first:
                        cr.move_to(wx, wy)
                        first = False
                    else:
                        cr.line_to(wx, wy)
                cr.close_path()
            cr.set_dash([], 0)
            cr.set_source_rgba(0, 0, 0, 0.9)
            cr.stroke_preserve()
            cr.set_dash([4.0, 4.0], self._ants_phase)
            cr.set_source_rgba(1, 1, 1, 0.95)
            cr.stroke()
            cr.restore()

        def _draw_points(self, cr):
            view = self._view
            cr.save()
            for (ix, iy, label) in self._points:
                wx, wy = image_to_widget(view, ix, iy)
                if label:
                    fill = (0.16, 0.85, 0.36)
                else:
                    fill = (0.95, 0.26, 0.26)
                cr.set_source_rgba(0, 0, 0, 0.55)
                cr.arc(wx, wy, 7.0, 0, 2 * math.pi)
                cr.fill()
                cr.set_source_rgb(*fill)
                cr.arc(wx, wy, 5.0, 0, 2 * math.pi)
                cr.fill()
                cr.set_source_rgb(1, 1, 1)
                cr.set_line_width(1.4)
                if label:
                    cr.move_to(wx - 2.6, wy)
                    cr.line_to(wx + 2.6, wy)
                    cr.move_to(wx, wy - 2.6)
                    cr.line_to(wx, wy + 2.6)
                else:
                    cr.move_to(wx - 2.6, wy)
                    cr.line_to(wx + 2.6, wy)
                cr.stroke()
            cr.restore()

        @property
        def boxes(self):
            """Kept boxes as ``(x0, y0, x1, y1, label)`` in image space."""
            return list(self._boxes)

        def set_boxes(self, boxes):
            """Boxes to keep drawing: exemplar boxes for a text prompt, or the
            box of a point prompt.  ``(x0, y0, x1, y1[, label])`` each."""
            out = []
            for b in boxes or []:
                x0, y0, x1, y1 = (float(v) for v in b[:4])
                label = int(b[4]) if len(b) > 4 else 1
                out.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), label))
            self._boxes = out
            self.queue_draw()

        def clear_boxes(self):
            self.set_boxes([])

        def _draw_kept_box(self, cr, box):
            x0, y0 = image_to_widget(self._view, box[0], box[1])
            x1, y1 = image_to_widget(self._view, box[2], box[3])
            positive = len(box) < 5 or box[4]
            cr.save()
            cr.set_line_width(1.5)
            cr.set_source_rgba(0, 0, 0, 0.55)
            cr.rectangle(x0 + 0.5, y0 + 0.5, max(1.0, x1 - x0), max(1.0, y1 - y0))
            cr.stroke()
            cr.set_line_width(1.0)
            cr.set_source_rgba(*((0.16, 0.85, 0.36, 0.95) if positive else (0.95, 0.26, 0.26, 0.95)))
            cr.rectangle(x0 + 0.5, y0 + 0.5, max(1.0, x1 - x0), max(1.0, y1 - y0))
            cr.stroke()
            cr.restore()

        def _draw_drag_box(self, cr, box):
            x0, y0 = image_to_widget(self._view, box[0], box[1])
            x1, y1 = image_to_widget(self._view, box[2], box[3])
            cr.save()
            cr.set_line_width(1.0)
            cr.set_dash([5.0, 4.0], 0)
            cr.set_source_rgba(1.0, 0.9, 0.3, 0.95)
            cr.rectangle(min(x0, x1) + 0.5, min(y0, y1) + 0.5, abs(x1 - x0), abs(y1 - y0))
            cr.stroke()
            cr.restore()

        def _draw_placeholder(self, cr, width, height):
            cr.save()
            cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            cr.set_font_size(13)
            msg = "no image loaded"
            ext = cr.text_extents(msg)
            cr.set_source_rgba(1, 1, 1, 0.35)
            cr.move_to((width - ext.width) * 0.5, height * 0.5)
            cr.show_text(msg)
            cr.restore()

        def _draw_chrome(self, cr, width, height):
            if self._busy:
                cr.save()
                cr.set_source_rgba(1, 1, 1, 0.12)
                cr.rectangle(0, 0, width, 3)
                cr.fill()
                cr.set_source_rgba(0.35, 0.72, 1.0, 0.95)
                cr.rectangle(0, 0, width * self._progress, 3)
                cr.fill()
                cr.restore()

            text = self._status_text
            if self._busy and self._stage:
                text = "%s  %d%%%s" % (
                    self._stage,
                    int(self._progress * 100),
                    ("  -  " + text) if text else "",
                )
            if not text:
                return
            cr.save()
            cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            cr.set_font_size(11)
            ext = cr.text_extents(text)
            pad = 6.0
            bw = ext.width + 2 * pad
            bh = ext.height + 2 * pad
            bx = 8.0
            by = height - bh - 8.0
            cr.set_source_rgba(0, 0, 0, 0.62)
            cr.rectangle(bx, by, bw, bh)
            cr.fill()
            cr.set_source_rgba(1, 1, 1, 0.92)
            cr.move_to(bx + pad - ext.x_bearing, by + pad - ext.y_bearing)
            cr.show_text(text)
            cr.restore()

        # ------------------------------------------------------------------ #
        # marching-ants animation
        # ------------------------------------------------------------------ #
        def _sync_ants(self):
            want = bool(self._show_ants and self._active_id >= 0 and self._instances)
            if want and not self._ants_source:
                self._ants_source = GLib.timeout_add(90, self._tick_ants)
            elif not want and self._ants_source:
                self._stop_ants()

        def _stop_ants(self):
            if self._ants_source:
                GLib.source_remove(self._ants_source)
                self._ants_source = 0

        def _tick_ants(self):
            self._ants_phase = (self._ants_phase + 1.0) % 8.0
            self.queue_draw()
            return True

        # ------------------------------------------------------------------ #
        # input
        # ------------------------------------------------------------------ #
        def do_size_allocate(self, allocation):
            Gtk.DrawingArea.do_size_allocate(self, allocation)
            self._ensure_view(allocation.width, allocation.height)

        def do_button_press_event(self, event):
            self.grab_focus()
            state = _ev_state(event)
            button = _ev_button(event)
            wx, wy = _ev_coords(event)
            ctrl = bool(state & Gdk.ModifierType.CONTROL_MASK)
            self._press = (button, wx, wy, state)

            if button == 2 or (button == 1 and self._space_down):
                self._pan_anchor = (wx, wy, self._view)
                self._set_cursor("grabbing")
                return True
            if button == 1 and ctrl and self.has_image:
                ix, iy = self._clamp_to_image(*widget_to_image(self._view, wx, wy))
                self._box_drag = (ix, iy, ix, iy)
                return True
            return True

        def do_motion_notify_event(self, event):
            wx, wy = _ev_coords(event)
            if self._pan_anchor is not None:
                ax, ay, base = self._pan_anchor
                self.set_view(view_pan(base, wx - ax, wy - ay))
                return True
            if self._box_drag is not None:
                ix, iy = self._clamp_to_image(*widget_to_image(self._view, wx, wy))
                self._box_drag = (self._box_drag[0], self._box_drag[1], ix, iy)
                self.queue_draw()
                return True
            if self._mode == _MODE_SELECT:
                inst = self.instance_at_widget(wx, wy)
                new_id = -1 if inst is None else inst.instance_id
                if new_id != self._hover_id:
                    self._hover_id = new_id
                    self.emit("instance-hovered", new_id)
                    self.queue_draw()
            return True

        def do_button_release_event(self, event):
            press = self._press
            self._press = None
            wx, wy = _ev_coords(event)

            if self._pan_anchor is not None:
                self._pan_anchor = None
                self._update_cursor()
                return True

            if self._box_drag is not None:
                x0, y0, x1, y1 = self._box_drag
                self._box_drag = None
                self.queue_draw()
                # Clipped to the image: a box prompt reaching past the edge
                # names pixels the daemon does not have.
                x0, y0 = self._clamp_to_image(x0, y0)
                x1, y1 = self._clamp_to_image(x1, y1)
                bx0, bx1 = (x0, x1) if x0 <= x1 else (x1, x0)
                by0, by1 = (y0, y1) if y0 <= y1 else (y1, y0)
                if (bx1 - bx0) * self._view.zoom >= 3.0 and (by1 - by0) * self._view.zoom >= 3.0:
                    self.emit("box-drawn", bx0, by0, bx1, by1)
                return True

            if press is None:
                return True
            _button, px, py, state = press
            if math.hypot(wx - px, wy - py) > _CLICK_SLOP_PX:
                return True  # it was a drag, not a click
            self._handle_click(_ev_button(event), wx, wy, state)
            return True

        def _clamp_to_image(self, ix, iy):
            return (min(max(float(ix), 0.0), float(self._image_w)),
                    min(max(float(iy), 0.0), float(self._image_h)))

        def _handle_click(self, button, wx, wy, state):
            if not self.has_image:
                return
            shift = bool(state & Gdk.ModifierType.SHIFT_MASK)
            ix, iy = widget_to_image(self._view, wx, wy)

            if self._mode == _MODE_POINTS:
                # Points outside the uploaded image are meaningless to the daemon.
                if not (0.0 <= ix < self._image_w and 0.0 <= iy < self._image_h):
                    return
                label = 0 if (button == 3 or shift) else 1
                self.add_point(ix, iy, label)
                return

            inst = self.instance_at_widget(wx, wy)
            if button == 3:
                if inst is not None:
                    self.set_instance_visible(inst.instance_id, not inst.visible)
                return
            if inst is None:
                self._set_active(-1, notify=True)
                self.queue_draw()
                return
            # A left click toggles the instance's *tick* -- the same flag the
            # dialog's checkbox shows and Apply uses.  It used to flip a
            # separate "selected" highlight that nothing else read, so
            # clicking an object on the canvas changed a border and not the
            # list, and the two never agreed.
            self._set_active(inst.instance_id, notify=True)
            self.set_instance_visible(inst.instance_id, not inst.visible, notify=True)
            self.queue_draw()

        def do_scroll_event(self, event):
            direction = _ev_unpack(event.get_scroll_direction(), 1, (None,))[0]
            wx, wy = _ev_coords(event)
            factor = 1.0
            if direction == Gdk.ScrollDirection.UP:
                factor = 1.15
            elif direction == Gdk.ScrollDirection.DOWN:
                factor = 1.0 / 1.15
            else:
                _dx, dy = _ev_unpack(event.get_scroll_deltas(), 2, (0.0, 0.0))
                if not dy:
                    return False
                factor = math.pow(1.15, -float(dy))
            self._need_fit = False
            self.set_view(view_zoom_at(self._view, factor, wx, wy))
            return True

        def do_leave_notify_event(self, event):
            if self._hover_id != -1:
                self._hover_id = -1
                self.emit("instance-hovered", -1)
                self.queue_draw()
            return False

        def do_key_press_event(self, event):
            key = _ev_keyval(event)
            if key == Gdk.KEY_space:
                if not self._space_down:
                    self._space_down = True
                    self._set_cursor("grab")
                return True
            if key in (Gdk.KEY_plus, Gdk.KEY_equal, Gdk.KEY_KP_Add):
                self.zoom_by(1.25)
                return True
            if key in (Gdk.KEY_minus, Gdk.KEY_KP_Subtract):
                self.zoom_by(1.0 / 1.25)
                return True
            if key in (Gdk.KEY_0, Gdk.KEY_KP_0, Gdk.KEY_f):
                self.zoom_fit()
                return True
            if key in (Gdk.KEY_1, Gdk.KEY_KP_1):
                self.zoom_to(1.0)
                return True
            if key == Gdk.KEY_bracketleft:
                self.set_mask_threshold(self._mask_threshold - 8, notify=True)
                return True
            if key == Gdk.KEY_bracketright:
                self.set_mask_threshold(self._mask_threshold + 8, notify=True)
                return True
            if key in (Gdk.KEY_BackSpace, Gdk.KEY_Delete):
                return self.remove_last_point()
            if key == Gdk.KEY_Escape:
                self.clear_points()
                return True
            if key == Gdk.KEY_a:
                self.set_show_ants(not self._show_ants)
                return True
            return False

        def do_key_release_event(self, event):
            if _ev_keyval(event) == Gdk.KEY_space:
                self._space_down = False
                self._update_cursor()
                return True
            return False

        def do_focus_out_event(self, event):
            """A Space released while another window has focus never reaches
            this widget, so losing focus ends the Space-held pan mode; left
            held, every later left-drag would pan instead of click or draw."""
            if self._space_down:
                self._space_down = False
                self._update_cursor()
            return False

        # ------------------------------------------------------------------ #
        # cursor
        # ------------------------------------------------------------------ #
        def _set_cursor(self, name):
            window = self.get_window()
            if window is None:
                return
            try:
                cursor = Gdk.Cursor.new_from_name(window.get_display(), name)
            except Exception:
                cursor = None
            window.set_cursor(cursor)

        def _update_cursor(self):
            if self._space_down:
                self._set_cursor("grab")
            elif self._mode == _MODE_POINTS:
                self._set_cursor("crosshair")
            else:
                self._set_cursor("default")

else:  # pragma: no cover - no GTK on this interpreter

    Sam3Canvas = None
