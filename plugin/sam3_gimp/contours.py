"""``contours`` -- turn a segmentation mask into vector paths.

This module is deliberately the most boring one in the package: **pure standard
library**, no torch, no transformers, no GIMP, and numpy only as an optional
duck-typed fast path for reading array input.  It imports nothing that could
fail on a machine with no GPU and no weights, so it is fully unit-testable.

It lives on the *plug-in* side on purpose.  Masks arrive as soft uint8 and the
user re-thresholds them locally (``DESIGN.md`` §4), so tracing here keeps the
Paths output consistent with whatever the threshold slider currently says --
no round trip, and no second implementation on the daemon.

The pipeline, in order:

1. **Marching squares** (:func:`extract_contours`) walks the *cracks* between
   foreground and background pixels and emits closed, axis-aligned polygons on
   the pixel-corner lattice.  Multiple disconnected components and interior
   holes (and islands inside holes, to any depth) all come out of the same
   walk, each as its own contour, with nesting recorded in
   :attr:`Contour.parent`.
2. **A minimum-area filter** drops speckle -- and, because a dropped contour
   takes its descendants with it, dropping a tiny island also drops the
   pinholes inside it.
3. **Douglas--Peucker** (:func:`simplify_polygon`) removes the staircase.
4. **An optional Schneider cubic fit** (:func:`fit_closed_bezier`) turns the
   simplified polyline into smooth beziers.  With ``bezier=False`` you get the
   polyline back in the same control-point shape, handles coincident with their
   anchors, which is exactly how GIMP draws a polygon with a bezier stroke.

Coordinates and winding
-----------------------
Contour coordinates live on the **pixel-corner lattice** of the mask: integer
``x`` in ``[0, width]`` and ``y`` in ``[0, height]``, with ``y`` increasing
downward as in every image format.  A polygon around the single pixel
``(0, 0)`` is therefore ``(0,0) (1,0) (1,1) (0,1)`` and has area exactly ``1``.

Winding is fixed and matters: contours are walked so that **foreground is
always on the right**.  With the shoelace formula evaluated in that y-down
frame this means

* an **outer** contour has **positive** signed area, and
* a **hole** has **negative** signed area,

so the two are opposite and a nonzero-winding fill subtracts holes correctly.
:attr:`Contour.is_hole` is just ``area < 0``.  Nesting deeper than one level
falls out for free: an island inside a hole is again positive.

Output for GIMP
---------------
``Gimp.Path.stroke_new_from_points()`` wants a flat list of floats, six per
anchor -- ``[c1x, c1y, px, py, c2x, c2y]`` -- where ``c1`` is the handle
*preceding* the anchor and ``c2`` the one *following* it.  :class:`Stroke`
carries exactly that list, and :attr:`PathResult.polygons` carries the plain
polygon form beside it.

Tolerances (``simplify_tolerance``, ``bezier_tolerance``) and ``min_area`` are
all expressed in **mask pixels**, before ``origin``/``scale`` are applied; the
affine transform is applied last so that the caller can hand back coordinates
in canvas or image space without changing what the tolerances mean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_THRESHOLD",
    "DEFAULT_CORNER_ANGLE_DEG",
    "Contour",
    "Stroke",
    "Shape",
    "PathResult",
    "signed_area",
    "polygon_area",
    "point_in_polygon",
    "extract_contours",
    "simplify_polygon",
    "simplify_contours",
    "fit_closed_bezier",
    "polygon_control_points",
    "bezier_control_points",
    "contour_to_stroke",
    "mask_to_paths",
    "bbox_origin",
    "instance_strokes",
]

#: Soft masks come back from the daemon as ``round(255 * sigmoid(logit))``, so
#: 128 is the logit-zero decision boundary and the natural default cut.
DEFAULT_THRESHOLD = 128
#: Component cap for extract_contours(); see its docstring.
DEFAULT_MAX_CONTOURS = 1000

#: A turn sharper than this many degrees is treated as a corner by the bezier
#: fitter, and gets independent handles on either side instead of a smooth
#: joint.  90 deg square corners are well above it; a simplified circle's ~15
#: deg turns are well below.
DEFAULT_CORNER_ANGLE_DEG = 60.0

#: Guard against pathological Schneider recursion; past this depth the fitter
#: gives up and emits straight segments.
_MAX_FIT_DEPTH = 24

Pt = Tuple[float, float]

_EAST = (1, 0)
_SOUTH = (0, 1)
_WEST = (-1, 0)
_NORTH = (0, -1)


# ---------------------------------------------------------------------------
# small vector helpers
# ---------------------------------------------------------------------------

def _sub(a: Pt, b: Pt) -> Pt:
    return (a[0] - b[0], a[1] - b[1])


def _unit(v: Pt) -> Pt:
    n = math.hypot(v[0], v[1])
    if n <= 0.0:
        # Degenerate input; any finite direction keeps the fitter well-defined.
        return (1.0, 0.0)
    return (v[0] / n, v[1] / n)


def _neg(v: Pt) -> Pt:
    return (-v[0], -v[1])


def _dist(a: Pt, b: Pt) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _sqdist(a: Pt, b: Pt) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def signed_area(points: Sequence[Pt]) -> float:
    """Shoelace area of a closed polygon, y-down.

    Positive for an outer contour under this module's winding convention,
    negative for a hole.  The polygon is implicitly closed; do not repeat the
    first point.
    """
    n = len(points)
    if n < 3:
        return 0.0
    total = 0.0
    x0, y0 = points[-1]
    for x1, y1 in points:
        total += x0 * y1 - x1 * y0
        x0, y0 = x1, y1
    return 0.5 * total


def polygon_area(points: Sequence[Pt]) -> float:
    """Unsigned area of a closed polygon."""
    return abs(signed_area(points))


def point_in_polygon(pt: Pt, points: Sequence[Pt]) -> bool:
    """Crossing-number test.  Behaviour exactly on the boundary is undefined;
    callers use the strictly-interior sample points computed at trace time."""
    x, y = pt
    inside = False
    n = len(points)
    if n < 3:
        return False
    xj, yj = points[-1]
    for i in range(n):
        xi, yi = points[i]
        if (yi > y) != (yj > y):
            xc = xj + (y - yj) * (xi - xj) / (yi - yj)
            if x < xc:
                inside = not inside
        xj, yj = xi, yi
    return inside


def _bbox(points: Sequence[Pt]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


# ---------------------------------------------------------------------------
# data shapes
# ---------------------------------------------------------------------------

@dataclass
class Contour:
    """One closed polygon.

    ``points`` never repeats the first vertex.  ``area`` is signed (see the
    module docstring): positive outer, negative hole.  ``parent`` indexes the
    smallest contour that encloses this one within the same result list, or
    ``-1`` at top level; ``sample`` is a point strictly inside this contour's
    enclosed region, kept from the raw lattice geometry so that nesting stays
    stable across simplification.
    """

    points: List[Pt]
    area: float
    parent: int = -1
    sample: Pt = (0.0, 0.0)

    @property
    def is_hole(self) -> bool:
        return self.area < 0.0

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.points)

    def flat(self) -> List[float]:
        """``[x0, y0, x1, y1, ...]``."""
        out: List[float] = []
        for x, y in self.points:
            out.append(float(x))
            out.append(float(y))
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "points": self.flat(),
            "area": float(self.area),
            "is_hole": bool(self.is_hole),
            "parent": int(self.parent),
        }


@dataclass
class Stroke:
    """A GIMP path stroke.

    :attr:`control_points` is the flat six-floats-per-anchor list that
    ``Gimp.Path.stroke_new_from_points()`` expects::

        [c1x, c1y, px, py, c2x, c2y,  c1x, c1y, px, py, c2x, c2y,  ...]

    ``c1`` precedes its anchor, ``c2`` follows it.  Strokes from this module
    are always closed, so the last anchor's ``c2`` pairs with the first
    anchor's ``c1``.
    """

    control_points: List[float]
    closed: bool = True
    is_hole: bool = False
    parent: int = -1

    @property
    def anchor_count(self) -> int:
        return len(self.control_points) // 6

    def anchors(self) -> List[Tuple[Pt, Pt, Pt]]:
        """``[(c1, p, c2), ...]`` -- the same data, unflattened."""
        cp = self.control_points
        out = []
        for i in range(0, len(cp), 6):
            out.append(((cp[i], cp[i + 1]), (cp[i + 2], cp[i + 3]),
                        (cp[i + 4], cp[i + 5])))
        return out

    def anchor_points(self) -> List[Pt]:
        cp = self.control_points
        return [(cp[i + 2], cp[i + 3]) for i in range(0, len(cp), 6)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "control_points": [float(v) for v in self.control_points],
            "closed": bool(self.closed),
            "is_hole": bool(self.is_hole),
            "parent": int(self.parent),
        }


@dataclass
class Shape:
    """An outer contour together with the holes immediately inside it.

    Indices refer to :attr:`PathResult.contours` / :attr:`PathResult.strokes`,
    which are index-parallel.  An island inside a hole is its own ``Shape``.
    """

    outer: int
    holes: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"outer": int(self.outer), "holes": [int(h) for h in self.holes]}


@dataclass
class PathResult:
    """Everything :func:`mask_to_paths` produces.

    :attr:`contours` and :attr:`strokes` are index-parallel: ``strokes[i]`` is
    the GIMP form of ``contours[i]``.  :attr:`shapes` groups them.
    """

    contours: List[Contour] = field(default_factory=list)
    strokes: List[Stroke] = field(default_factory=list)
    shapes: List[Shape] = field(default_factory=list)
    width: int = 0
    height: int = 0

    @property
    def polygons(self) -> List[List[Pt]]:
        """The plain polygon form: a list of closed point lists."""
        return [c.points for c in self.contours]

    @property
    def outer_count(self) -> int:
        return sum(1 for c in self.contours if not c.is_hole)

    @property
    def hole_count(self) -> int:
        return sum(1 for c in self.contours if c.is_hole)

    def total_area(self) -> float:
        """Outer areas minus hole areas -- the enclosed area of the whole
        result, which should track the mask's foreground pixel count."""
        return sum(c.area for c in self.contours)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "contours": [c.to_dict() for c in self.contours],
            "strokes": [s.to_dict() for s in self.strokes],
            "shapes": [s.to_dict() for s in self.shapes],
        }


# ---------------------------------------------------------------------------
# mask input normalisation
# ---------------------------------------------------------------------------

def _threshold_table(threshold: int) -> bytes:
    t = max(0, min(256, int(threshold)))
    return bytes(1 if i >= t else 0 for i in range(256))


def _row_from_iterable(row: Iterable[Any], threshold: int) -> bytearray:
    out = bytearray()
    for v in row:
        if v is True or v is False:
            out.append(1 if v else 0)
        else:
            out.append(1 if v >= threshold else 0)
    return out


def _binary_rows(mask: Any, width: Optional[int], height: Optional[int],
                 threshold: int) -> Tuple[List[bytearray], int, int]:
    """Normalise any accepted mask form into rows of 0/1 bytes.

    Accepted: a flat ``bytes``/``bytearray``/``memoryview`` plus ``width`` and
    ``height``; a 2-D numpy-like array (duck-typed via ``.shape``, never
    imported); a sequence of row sequences; a flat sequence of numbers plus
    ``width``/``height``.
    """
    # --- flat byte buffer -------------------------------------------------
    if isinstance(mask, (bytes, bytearray, memoryview)):
        buf = bytes(mask)
        if width is None or height is None:
            raise ValueError("width and height are required for a flat mask buffer")
        w, h = int(width), int(height)
        if w < 0 or h < 0:
            raise ValueError("negative mask dimensions")
        if len(buf) != w * h:
            raise ValueError("mask buffer is %d bytes, expected %d (%dx%d)"
                             % (len(buf), w * h, w, h))
        bits = buf.translate(_threshold_table(threshold))
        rows = [bytearray(bits[y * w:(y + 1) * w]) for y in range(h)]
        return rows, w, h

    # --- numpy-like 2-D array (duck-typed, numpy is never imported) -------
    shape = getattr(mask, "shape", None)
    if shape is not None and not isinstance(mask, (list, tuple)):
        if len(shape) != 2:
            raise ValueError("array mask must be 2-D, got shape %r" % (tuple(shape),))
        h, w = int(shape[0]), int(shape[1])
        dtype = getattr(mask, "dtype", None)
        tobytes = getattr(mask, "tobytes", None)
        if dtype is not None and str(dtype) == "uint8" and callable(tobytes):
            try:
                return _binary_rows(tobytes(), w, h, threshold)
            except ValueError:
                pass  # non-contiguous or odd layout: fall through to tolist()
        tolist = getattr(mask, "tolist", None)
        listed = tolist() if callable(tolist) else [list(r) for r in mask]
        rows = [_row_from_iterable(r, threshold) for r in listed]
        return rows, w, h

    # --- sequences --------------------------------------------------------
    seq = list(mask)
    if not seq:
        return [], int(width or 0), 0
    first = seq[0]
    if isinstance(first, (list, tuple, bytes, bytearray, memoryview)) or (
            getattr(first, "__len__", None) is not None
            and not isinstance(first, (int, float, bool))):
        rows = [_row_from_iterable(r, threshold) for r in seq]
        w = len(rows[0])
        for r in rows:
            if len(r) != w:
                raise ValueError("ragged mask rows")
        return rows, w, len(rows)

    # flat sequence of numbers
    if width is None or height is None:
        raise ValueError("width and height are required for a flat mask sequence")
    w, h = int(width), int(height)
    if len(seq) != w * h:
        raise ValueError("mask sequence has %d values, expected %d (%dx%d)"
                         % (len(seq), w * h, w, h))
    rows = [_row_from_iterable(seq[y * w:(y + 1) * w], threshold) for y in range(h)]
    return rows, w, h


# ---------------------------------------------------------------------------
# marching squares
# ---------------------------------------------------------------------------

def _collapse_collinear(pts: List[Tuple[int, int]]) -> List[Pt]:
    """Drop vertices that continue in a straight line.

    The raw walk emits one vertex per unit lattice step, so this is where the
    long straight runs of a rectangle collapse to their two endpoints.  A
    vertex that the walk visits twice (the pinch of an 8-connected diagonal)
    turns on both visits and is therefore kept twice, which is correct.
    """
    n = len(pts)
    if n < 3:
        return [(float(x), float(y)) for x, y in pts]
    out: List[Pt] = []
    px, py = pts[-1]
    for i in range(n):
        cx, cy = pts[i]
        nx, ny = pts[(i + 1) % n]
        # cross product of (c - p) and (n - c); zero means no turn
        if (cx - px) * (ny - cy) != (cy - py) * (nx - cx):
            out.append((float(cx), float(cy)))
        px, py = cx, cy
    return out


def _interior_sample(points: Sequence[Pt], area: float) -> Pt:
    """A point strictly inside the region this contour encloses.

    Uses the midpoint of a *horizontal* lattice edge nudged a quarter pixel
    towards the enclosed side.  Because every vertex has an integer ``y``, the
    resulting ``y`` (integer +/- 0.25) can never lie on another contour's
    horizontal edge, which keeps the crossing test in
    :func:`point_in_polygon` exact.
    """
    n = len(points)
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        if y0 == y1 and x0 != x1:
            side = 1.0 if x1 > x0 else -1.0
            if area < 0.0:
                side = -side
            return ((x0 + x1) * 0.5, y0 + 0.25 * side)
    # Unreachable for a closed lattice loop (it must contain a horizontal
    # edge), but stay defined rather than raise.
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (sum(xs) / n, sum(ys) / n)


def _trace_loops(rows: List[bytearray], w: int, h: int,
                 connectivity: int) -> List[List[Tuple[int, int]]]:
    """Walk every boundary crack, foreground on the right.

    ``connectivity`` selects how the two ambiguous saddle configurations (two
    diagonally opposite foreground cells) are resolved: ``8`` joins them into
    one component (the trace turns left through the pinch), ``4`` keeps them
    apart (it turns right and hugs each cell).
    """
    if w <= 0 or h <= 0:
        return []
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8, got %r" % (connectivity,))
    turn_left = connectivity == 8

    empty = bytearray(w)

    def row(y: int) -> bytearray:
        if 0 <= y < h:
            return rows[y]
        return empty

    def cell(x: int, y: int) -> int:
        if x < 0 or x >= w or y < 0 or y >= h:
            return 0
        return rows[y][x]

    def outgoing(x: int, y: int) -> List[Tuple[int, int]]:
        a = cell(x - 1, y - 1)
        b = cell(x, y - 1)
        c = cell(x - 1, y)
        d = cell(x, y)
        outs = []
        if d and not b:
            outs.append(_EAST)
        if a and not c:
            outs.append(_WEST)
        if c and not d:
            outs.append(_SOUTH)
        if b and not a:
            outs.append(_NORTH)
        return outs

    visited = set()
    loops: List[List[Tuple[int, int]]] = []

    def walk(sx: int, sy: int, sdx: int, sdy: int) -> List[Tuple[int, int]]:
        pts: List[Tuple[int, int]] = []
        x, y, dx, dy = sx, sy, sdx, sdy
        while (x, y, dx, dy) not in visited:
            visited.add((x, y, dx, dy))
            pts.append((x, y))
            x += dx
            y += dy
            outs = outgoing(x, y)
            if not outs:
                break  # cannot happen for a consistent lattice
            if len(outs) == 1:
                dx, dy = outs[0]
            else:  # saddle
                cand = (dy, -dx) if turn_left else (-dy, dx)
                dx, dy = cand if cand in outs else outs[0]
        return pts

    # Seeds.  Every closed loop contains at least one horizontal edge; at the
    # loop's topmost horizontal edge the foreground is either below it (walked
    # east) or above it (walked west).  Both cases are exactly a vertical
    # change between row y-1 and row y, so rows identical to their predecessor
    # cannot contain a seed at all -- which is what makes this scan cheap on
    # large masks.
    for y in range(h + 1):
        cur = row(y)
        prev = row(y - 1)
        if cur == prev:
            continue
        for x in range(w):
            below = cur[x]
            above = prev[x]
            if below == above:
                continue
            if below:
                key = (x, y, _EAST[0], _EAST[1])
                if key not in visited:
                    pts = walk(x, y, _EAST[0], _EAST[1])
                    if pts:
                        loops.append(pts)
            else:
                key = (x + 1, y, _WEST[0], _WEST[1])
                if key not in visited:
                    pts = walk(x + 1, y, _WEST[0], _WEST[1])
                    if pts:
                        loops.append(pts)
    return loops


_GRID_CELL = 32.0


def _assign_parents(contours: List[Contour]) -> None:
    """Set ``parent`` to the smallest contour enclosing each one (-1 if none).

    Candidates come from a bbox grid rather than the whole list: a contour
    can only be enclosed by one whose bbox covers its sample point, so only
    contours registered in that point's cell are considered.  Speckle then
    costs one cell lookup each instead of a pass over every other speck.
    """
    n = len(contours)
    boxes = [_bbox(c.points) for c in contours]
    grid: Dict[Tuple[int, int], List[int]] = {}
    for j, (x0, y0, x1, y1) in enumerate(boxes):
        for gy in range(int(y0 // _GRID_CELL), int(y1 // _GRID_CELL) + 1):
            for gx in range(int(x0 // _GRID_CELL), int(x1 // _GRID_CELL) + 1):
                grid.setdefault((gx, gy), []).append(j)
    for i in range(n):
        ci = contours[i]
        best = -1
        best_area = None
        ai = abs(ci.area)
        sx, sy = ci.sample
        for j in grid.get((int(sx // _GRID_CELL), int(sy // _GRID_CELL)), ()):
            if i == j:
                continue
            aj = abs(contours[j].area)
            if aj <= ai:
                continue  # a container is strictly larger
            if best_area is not None and aj >= best_area:
                continue  # already have a smaller container
            x0, y0, x1, y1 = boxes[j]
            if sx < x0 or sx > x1 or sy < y0 or sy > y1:
                continue
            if point_in_polygon(ci.sample, contours[j].points):
                best = j
                best_area = aj
        ci.parent = best


def _prune(contours: List[Contour], keep: List[bool]) -> List[Contour]:
    """Drop the contours marked False, plus every descendant of a dropped one,
    then renumber ``parent`` into the surviving list."""
    changed = True
    while changed:
        changed = False
        for i, c in enumerate(contours):
            if keep[i] and c.parent >= 0 and not keep[c.parent]:
                keep[i] = False
                changed = True
    remap: Dict[int, int] = {}
    out: List[Contour] = []
    for i, c in enumerate(contours):
        if keep[i]:
            remap[i] = len(out)
            out.append(c)
    for c in out:
        c.parent = remap.get(c.parent, -1) if c.parent >= 0 else -1
    return out


def extract_contours(mask: Any, width: Optional[int] = None,
                     height: Optional[int] = None, *,
                     threshold: int = DEFAULT_THRESHOLD,
                     connectivity: int = 8,
                     min_area: float = 0.0,
                     min_hole_area: Optional[float] = None,
                     fill_holes: bool = False,
                     max_contours: int = DEFAULT_MAX_CONTOURS) -> List[Contour]:
    """Marching-squares contour extraction.

    Returns closed lattice polygons: outers wound positive, holes negative,
    nesting recorded in :attr:`Contour.parent`.  ``min_area`` (in mask pixels
    squared) drops speckle; dropping a contour drops everything nested inside
    it, so a filtered-out island cannot leave orphaned pinholes behind.
    ``min_hole_area`` defaults to ``min_area``.  ``fill_holes`` discards every
    hole (and anything nested inside one).

    ``max_contours`` bounds the component count: when a mask yields more
    (a speckled, low-confidence mask easily yields tens of thousands), only
    the largest by area are kept.  Nesting is assigned afterwards, so the
    survivors are classified among themselves.  This is not merely a speed
    cap -- GIMP adds every component as a separate stroke through the PDB,
    and a path with thirty thousand strokes is unusable -- but it is also
    what keeps this linear: parent assignment is quadratic in the number of
    contours, and 31,595 of them took three minutes.
    """
    rows, w, h = _binary_rows(mask, width, height, threshold)
    loops = _trace_loops(rows, w, h, connectivity)

    contours: List[Contour] = []
    for loop in loops:
        pts = _collapse_collinear(loop)
        if len(pts) < 3:
            continue
        area = signed_area(pts)
        if area == 0.0:
            continue
        contours.append(Contour(points=pts, area=area,
                                sample=_interior_sample(pts, area)))

    if not contours:
        return []

    if max_contours and len(contours) > int(max_contours):
        contours.sort(key=lambda c: abs(c.area), reverse=True)
        del contours[int(max_contours):]

    _assign_parents(contours)

    hole_min = min_area if min_hole_area is None else min_hole_area
    keep = []
    for c in contours:
        if fill_holes and c.is_hole:
            keep.append(False)
            continue
        limit = hole_min if c.is_hole else min_area
        keep.append(abs(c.area) >= limit)
    contours = _prune(contours, keep)

    # A hole whose container was filtered away is not a hole any more.
    keep = [not (c.is_hole and c.parent < 0) for c in contours]
    if not all(keep):
        contours = _prune(contours, keep)
    return contours


# ---------------------------------------------------------------------------
# Douglas-Peucker
# ---------------------------------------------------------------------------

def _point_segment_distance(p: Pt, a: Pt, b: Pt) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx = bx - ax
    dy = by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _dp_open(points: Sequence[Pt], tolerance: float) -> List[Pt]:
    """Douglas-Peucker on an open polyline.  Iterative: a 60k-point contour
    must not blow the interpreter stack."""
    n = len(points)
    if n < 3:
        return list(points)
    keep = [False] * n
    keep[0] = True
    keep[n - 1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        pa = points[a]
        pb = points[b]
        dmax = -1.0
        imax = -1
        for i in range(a + 1, b):
            d = _point_segment_distance(points[i], pa, pb)
            if d > dmax:
                dmax = d
                imax = i
        if dmax > tolerance and imax > a:
            keep[imax] = True
            stack.append((a, imax))
            stack.append((imax, b))
    return [points[i] for i in range(n) if keep[i]]


def simplify_polygon(points: Sequence[Pt], tolerance: float,
                     closed: bool = True) -> List[Pt]:
    """Douglas-Peucker simplification.

    For a closed ring there is no natural pair of fixed endpoints, so the ring
    is cut at two points that are guaranteed to survive -- the lexicographic
    minimum vertex (deterministic regardless of where the trace started) and
    the vertex farthest from it -- and each half is simplified as an open
    polyline.
    """
    pts = list(points)
    if tolerance <= 0.0 or len(pts) < 4:
        return pts
    if not closed:
        return _dp_open(pts, tolerance)

    start = min(range(len(pts)), key=lambda i: (pts[i][1], pts[i][0]))
    if start:
        pts = pts[start:] + pts[:start]
    far = max(range(1, len(pts)), key=lambda i: _sqdist(pts[i], pts[0]))
    head = _dp_open(pts[:far + 1], tolerance)
    tail = _dp_open(pts[far:] + [pts[0]], tolerance)
    return head[:-1] + tail[:-1]


def simplify_contours(contours: Sequence[Contour],
                      tolerance: float) -> List[Contour]:
    """Simplify every contour, keeping the list index-parallel with the input
    so that ``parent`` links stay valid.  Contours that collapse below three
    points are marked by an empty ``points`` list for the caller to prune."""
    out: List[Contour] = []
    for c in contours:
        pts = simplify_polygon(c.points, tolerance, closed=True)
        if len(pts) < 3 or signed_area(pts) == 0.0:
            # Simplification flattened the ring: a one-pixel line, a hair-thin
            # strap, anything narrower than the tolerance.  Those are real
            # objects (they are exactly what click-to-segment finds that a
            # phrase cannot), and dropping them made "Paths" trace nothing for
            # them.  Keep the unsimplified ring instead; it is small anyway.
            pts = list(c.points)
            if len(pts) < 3 or signed_area(pts) == 0.0:
                out.append(Contour(points=[], area=c.area, parent=c.parent,
                                   sample=c.sample))
                continue
        area = signed_area(pts)
        # Keep the original orientation even if simplification somehow flipped
        # the sign of a near-degenerate ring.
        if (area < 0.0) != (c.area < 0.0):
            pts.reverse()
            area = -area
        out.append(Contour(points=pts, area=area, parent=c.parent,
                           sample=c.sample))
    return out


# ---------------------------------------------------------------------------
# Schneider cubic bezier fitting
# ---------------------------------------------------------------------------

def _bezier_point(bez: Sequence[Pt], t: float) -> Pt:
    mt = 1.0 - t
    a = mt * mt * mt
    b = 3.0 * t * mt * mt
    c = 3.0 * t * t * mt
    d = t * t * t
    return (a * bez[0][0] + b * bez[1][0] + c * bez[2][0] + d * bez[3][0],
            a * bez[0][1] + b * bez[1][1] + c * bez[2][1] + d * bez[3][1])


def _chord_length_parameterize(pts: Sequence[Pt]) -> List[float]:
    u = [0.0]
    for i in range(1, len(pts)):
        u.append(u[i - 1] + _dist(pts[i], pts[i - 1]))
    total = u[-1]
    if total <= 0.0:
        n = len(pts) - 1
        return [i / n for i in range(len(pts))] if n else [0.0]
    return [v / total for v in u]


def _generate_bezier(pts: Sequence[Pt], u: Sequence[float],
                     t1: Pt, t2: Pt) -> Tuple[Pt, Pt, Pt, Pt]:
    """Least-squares fit of one cubic to ``pts`` with the tangent directions
    fixed at both ends (Schneider 1990, 'An Algorithm for Automatically
    Fitting Digitized Curves')."""
    p0 = pts[0]
    p3 = pts[-1]
    c00 = c01 = c11 = 0.0
    x0 = x1 = 0.0
    for i in range(len(pts)):
        ui = u[i]
        mt = 1.0 - ui
        b0 = mt * mt * mt
        b1 = 3.0 * ui * mt * mt
        b2 = 3.0 * ui * ui * mt
        b3 = ui * ui * ui
        a0 = (t1[0] * b1, t1[1] * b1)
        a1 = (t2[0] * b2, t2[1] * b2)
        c00 += a0[0] * a0[0] + a0[1] * a0[1]
        c01 += a0[0] * a1[0] + a0[1] * a1[1]
        c11 += a1[0] * a1[0] + a1[1] * a1[1]
        tx = pts[i][0] - (p0[0] * (b0 + b1) + p3[0] * (b2 + b3))
        ty = pts[i][1] - (p0[1] * (b0 + b1) + p3[1] * (b2 + b3))
        x0 += a0[0] * tx + a0[1] * ty
        x1 += a1[0] * tx + a1[1] * ty

    det_c = c00 * c11 - c01 * c01
    if det_c != 0.0:
        alpha_l = (x0 * c11 - x1 * c01) / det_c
        alpha_r = (c00 * x1 - c01 * x0) / det_c
    else:
        alpha_l = alpha_r = 0.0

    seg_len = _dist(p0, p3)
    eps = 1.0e-6 * seg_len
    if alpha_l < eps or alpha_r < eps or not (
            math.isfinite(alpha_l) and math.isfinite(alpha_r)):
        # Wu/Barsky fallback: handles a third of the way along the chord.
        # For a closed ring p0 == p3 makes that zero, and the caller's split
        # logic recovers on the next recursion.
        alpha_l = alpha_r = seg_len / 3.0
    return (p0,
            (p0[0] + t1[0] * alpha_l, p0[1] + t1[1] * alpha_l),
            (p3[0] + t2[0] * alpha_r, p3[1] + t2[1] * alpha_r),
            p3)


def _max_error(pts: Sequence[Pt], bez: Sequence[Pt],
               u: Sequence[float]) -> Tuple[float, int]:
    n = len(pts)
    split = n // 2
    worst = 0.0
    for i in range(1, n - 1):
        d = _sqdist(_bezier_point(bez, u[i]), pts[i])
        if d > worst:
            worst = d
            split = i
    return worst, split


def _reparameterize(pts: Sequence[Pt], u: Sequence[float],
                    bez: Sequence[Pt]) -> List[float]:
    q1 = [(3.0 * (bez[i + 1][0] - bez[i][0]),
           3.0 * (bez[i + 1][1] - bez[i][1])) for i in range(3)]
    q2 = [(2.0 * (q1[i + 1][0] - q1[i][0]),
           2.0 * (q1[i + 1][1] - q1[i][1])) for i in range(2)]
    out = []
    for i, ui in enumerate(u):
        mt = 1.0 - ui
        qx, qy = _bezier_point(bez, ui)
        d1x = mt * mt * q1[0][0] + 2.0 * mt * ui * q1[1][0] + ui * ui * q1[2][0]
        d1y = mt * mt * q1[0][1] + 2.0 * mt * ui * q1[1][1] + ui * ui * q1[2][1]
        d2x = mt * q2[0][0] + ui * q2[1][0]
        d2y = mt * q2[0][1] + ui * q2[1][1]
        dx = qx - pts[i][0]
        dy = qy - pts[i][1]
        num = dx * d1x + dy * d1y
        den = d1x * d1x + d1y * d1y + dx * d2x + dy * d2y
        v = ui if den == 0.0 else ui - num / den
        if not math.isfinite(v):
            v = ui
        out.append(min(1.0, max(0.0, v)))
    return out


def _straight_segments(pts: Sequence[Pt]) -> List[Tuple[Pt, Pt, Pt, Pt]]:
    segs = []
    for i in range(len(pts) - 1):
        a = pts[i]
        b = pts[i + 1]
        c1 = (a[0] + (b[0] - a[0]) / 3.0, a[1] + (b[1] - a[1]) / 3.0)
        c2 = (a[0] + 2.0 * (b[0] - a[0]) / 3.0, a[1] + 2.0 * (b[1] - a[1]) / 3.0)
        segs.append((a, c1, c2, b))
    return segs


def _fit_cubic(pts: Sequence[Pt], t1: Pt, t2: Pt, tol2: float,
               depth: int) -> List[Tuple[Pt, Pt, Pt, Pt]]:
    n = len(pts)
    if n < 2:
        return []
    if n == 2:
        d = _dist(pts[0], pts[1]) / 3.0
        return [(pts[0],
                 (pts[0][0] + t1[0] * d, pts[0][1] + t1[1] * d),
                 (pts[1][0] + t2[0] * d, pts[1][1] + t2[1] * d),
                 pts[1])]
    if depth > _MAX_FIT_DEPTH:
        return _straight_segments(pts)

    u = _chord_length_parameterize(pts)
    bez = _generate_bezier(pts, u, t1, t2)
    err, split = _max_error(pts, bez, u)
    if err <= tol2:
        return [bez]

    if err <= tol2 * 4.0:
        for _ in range(4):
            u = _reparameterize(pts, u, bez)
            bez = _generate_bezier(pts, u, t1, t2)
            err, split = _max_error(pts, bez, u)
            if err <= tol2:
                return [bez]

    if split <= 0 or split >= n - 1:
        split = n // 2
    centre = _unit(_sub(pts[split - 1], pts[split + 1]))
    left = _fit_cubic(pts[:split + 1], t1, centre, tol2, depth + 1)
    right = _fit_cubic(pts[split:], _neg(centre), t2, tol2, depth + 1)
    return left + right


def _densify(points: Sequence[Pt], max_edge: float,
             max_per_edge: int = 64) -> List[Pt]:
    """Insert collinear points so that no edge is longer than ``max_edge``.

    Schneider's error metric only looks at the input samples, so a fit through
    widely spaced vertices can bulge far outside the tolerance *between* them.
    Splitting every edge down to roughly the tolerance turns "close at the
    vertices" into "close everywhere", which is the guarantee callers actually
    want.  The inserted points are exactly on the source polyline, so the
    polygon being approximated is unchanged.
    """
    if max_edge <= 0.0 or len(points) < 2:
        return list(points)
    out: List[Pt] = [points[0]]
    for i in range(1, len(points)):
        a = points[i - 1]
        b = points[i]
        steps = int(math.ceil(_dist(a, b) / max_edge))
        if steps > max_per_edge:
            steps = max_per_edge
        for k in range(1, steps):
            t = k / steps
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
        out.append(b)
    return out


def _corner_indices(points: Sequence[Pt], corner_angle_deg: float) -> List[int]:
    n = len(points)
    if n < 3:
        return []
    limit = math.cos(math.radians(max(0.0, min(180.0, corner_angle_deg))))
    corners = []
    for i in range(n):
        din = _unit(_sub(points[i], points[i - 1]))
        dout = _unit(_sub(points[(i + 1) % n], points[i]))
        if din[0] * dout[0] + din[1] * dout[1] < limit:
            corners.append(i)
    return corners


def fit_closed_bezier(points: Sequence[Pt], tolerance: float,
                      corner_angle_deg: float = DEFAULT_CORNER_ANGLE_DEG
                      ) -> List[Tuple[Pt, Pt, Pt, Pt]]:
    """Fit a closed chain of cubic beziers through ``points``.

    The ring is cut at its corners -- vertices whose turn exceeds
    ``corner_angle_deg`` -- and each arc between two corners is fitted
    independently, so square corners stay square.  A ring with no corners (a
    blob, a circle) is cut at two antipodal vertices instead and joined
    smoothly there, so no seam is visible.  Returns segments
    ``(p0, c1, c2, p3)`` where each segment's ``p3`` is the next one's ``p0``
    and the last one's ``p3`` is the first one's ``p0``.
    """
    pts = list(points)
    n = len(pts)
    if n < 3:
        return []
    tol2 = max(tolerance, 0.0) ** 2

    corner_set = set(_corner_indices(pts, corner_angle_deg))
    splits = sorted(corner_set)
    if len(splits) < 2:
        base = splits[0] if splits else 0
        other = (base + max(1, n // 2)) % n
        splits = sorted({base, other})
    if len(splits) < 2:  # n == 1 is already excluded; belt and braces
        return _straight_segments(pts + [pts[0]])

    # Tangents at each cut: one-sided at a real corner, symmetric (and hence
    # G1-continuous) at a smooth cut.
    tan_out: Dict[int, Pt] = {}
    tan_in: Dict[int, Pt] = {}
    for k in splits:
        if k in corner_set:
            tan_out[k] = _unit(_sub(pts[(k + 1) % n], pts[k]))
            tan_in[k] = _unit(_sub(pts[k - 1], pts[k]))
        else:
            t = _unit(_sub(pts[(k + 1) % n], pts[k - 1]))
            tan_out[k] = t
            tan_in[k] = _neg(t)

    segments: List[Tuple[Pt, Pt, Pt, Pt]] = []
    for idx, a in enumerate(splits):
        b = splits[(idx + 1) % len(splits)]
        count = (b - a) % n
        if count == 0:
            count = n
        arc = [pts[(a + step) % n] for step in range(count + 1)]
        arc = _densify(arc, max(tolerance, 0.0))
        segments.extend(_fit_cubic(arc, tan_out[a], tan_in[b], tol2, 0))
    return segments


# ---------------------------------------------------------------------------
# control-point emission
# ---------------------------------------------------------------------------

def polygon_control_points(points: Sequence[Pt]) -> List[float]:
    """The GIMP flat form for a plain polygon: both handles sit on the anchor,
    which is how a bezier stroke draws straight segments."""
    out: List[float] = []
    for x, y in points:
        fx = float(x)
        fy = float(y)
        out.extend((fx, fy, fx, fy, fx, fy))
    return out


def bezier_control_points(segments: Sequence[Tuple[Pt, Pt, Pt, Pt]]) -> List[float]:
    """Flatten a closed chain of cubic segments into GIMP's six-floats-per-
    anchor form.  Anchor *k* takes its incoming handle from segment *k-1*'s
    second control point and its outgoing handle from segment *k*'s first."""
    m = len(segments)
    if m == 0:
        return []
    out: List[float] = []
    for k in range(m):
        prev_c2 = segments[k - 1][2]
        p = segments[k][0]
        c1 = segments[k][1]
        out.extend((float(prev_c2[0]), float(prev_c2[1]),
                    float(p[0]), float(p[1]),
                    float(c1[0]), float(c1[1])))
    return out


def contour_to_stroke(contour: Contour, *, bezier: bool = True,
                      bezier_tolerance: float = 1.0,
                      corner_angle_deg: float = DEFAULT_CORNER_ANGLE_DEG
                      ) -> Stroke:
    """Convert one contour into a GIMP stroke."""
    if bezier:
        segments = fit_closed_bezier(contour.points, bezier_tolerance,
                                     corner_angle_deg)
        cp = bezier_control_points(segments)
        if not cp:
            cp = polygon_control_points(contour.points)
    else:
        cp = polygon_control_points(contour.points)
    return Stroke(control_points=cp, closed=True, is_hole=contour.is_hole,
                  parent=contour.parent)


# ---------------------------------------------------------------------------
# the whole pipeline
# ---------------------------------------------------------------------------

def _normalise_scale(scale: Any) -> Tuple[float, float]:
    if isinstance(scale, (tuple, list)):
        if len(scale) != 2:
            raise ValueError("scale must be a number or a (sx, sy) pair")
        return float(scale[0]), float(scale[1])
    return float(scale), float(scale)


def _transform_contours(contours: List[Contour], ox: float, oy: float,
                        sx: float, sy: float) -> None:
    flip = (sx * sy) < 0.0
    for c in contours:
        pts = [(ox + x * sx, oy + y * sy) for x, y in c.points]
        if flip:
            # A mirroring transform reverses winding; undo that so outers stay
            # positive and holes stay negative.
            pts.reverse()
        c.points = pts
        c.area = signed_area(pts)
        c.sample = (ox + c.sample[0] * sx, oy + c.sample[1] * sy)


def _build_shapes(contours: Sequence[Contour]) -> List[Shape]:
    shapes: List[Shape] = []
    index: Dict[int, int] = {}
    for i, c in enumerate(contours):
        if not c.is_hole:
            index[i] = len(shapes)
            shapes.append(Shape(outer=i, holes=[]))
    for i, c in enumerate(contours):
        if c.is_hole and c.parent >= 0 and c.parent in index:
            shapes[index[c.parent]].holes.append(i)
    return shapes


def mask_to_paths(mask: Any, width: Optional[int] = None,
                  height: Optional[int] = None, *,
                  threshold: int = DEFAULT_THRESHOLD,
                  connectivity: int = 8,
                  min_area: float = 0.0,
                  min_hole_area: Optional[float] = None,
                  fill_holes: bool = False,
                  simplify_tolerance: float = 1.0,
                  bezier: bool = True,
                  bezier_tolerance: Optional[float] = None,
                  corner_angle_deg: float = DEFAULT_CORNER_ANGLE_DEG,
                  origin: Tuple[float, float] = (0.0, 0.0),
                  scale: Any = 1.0) -> PathResult:
    """Mask in, GIMP-ready paths out.

    ``mask`` may be a flat ``bytes``-like buffer (with ``width``/``height``), a
    sequence of row sequences, or a 2-D numpy-like array.  Values at or above
    ``threshold`` are foreground; booleans are taken as-is.

    ``simplify_tolerance`` and ``bezier_tolerance`` (default: the simplify
    tolerance, floored at 1.0) and ``min_area`` are all in **mask pixels**.
    ``origin`` and ``scale`` apply the affine ``out = origin + p * scale``
    *last*, so the caller can shift a cropped mask back into canvas space --
    ``origin=(bbox.x0, bbox.y0)`` -- or on into image space, without changing
    what the tolerances mean.  ``scale`` is a number or an ``(sx, sy)`` pair.

    With ``bezier=False`` the strokes are plain polylines in the same
    control-point form (handles coincident with anchors).
    """
    contours = extract_contours(mask, width, height, threshold=threshold,
                                connectivity=connectivity, min_area=min_area,
                                min_hole_area=min_hole_area,
                                fill_holes=fill_holes)
    rows_w, rows_h = _mask_size(mask, width, height)

    if simplify_tolerance > 0.0 and contours:
        contours = simplify_contours(contours, simplify_tolerance)
        keep = [bool(c.points) for c in contours]
        if not all(keep):
            contours = _prune(contours, keep)
        keep = [not (c.is_hole and c.parent < 0) for c in contours]
        if not all(keep):
            contours = _prune(contours, keep)

    ox, oy = float(origin[0]), float(origin[1])
    sx, sy = _normalise_scale(scale)
    if (ox, oy, sx, sy) != (0.0, 0.0, 1.0, 1.0):
        _transform_contours(contours, ox, oy, sx, sy)

    if bezier_tolerance is None:
        bezier_tolerance = max(simplify_tolerance, 1.0)
    # The fit runs in output space, so scale the tolerance with the transform.
    fit_tol = bezier_tolerance * math.sqrt(abs(sx * sy)) if bezier else 0.0

    strokes = [contour_to_stroke(c, bezier=bezier, bezier_tolerance=fit_tol,
                                 corner_angle_deg=corner_angle_deg)
               for c in contours]
    return PathResult(contours=contours, strokes=strokes,
                      shapes=_build_shapes(contours),
                      width=rows_w, height=rows_h)


def _mask_size(mask: Any, width: Optional[int],
               height: Optional[int]) -> Tuple[int, int]:
    """Report the mask dimensions without re-thresholding the pixels."""
    if width is not None and height is not None:
        return int(width), int(height)
    shape = getattr(mask, "shape", None)
    if shape is not None and not isinstance(mask, (list, tuple)) and len(shape) == 2:
        return int(shape[1]), int(shape[0])
    try:
        rows = list(mask)
    except TypeError:  # pragma: no cover - defensive
        return 0, 0
    if not rows:
        return 0, 0
    first = rows[0]
    if isinstance(first, (list, tuple, bytes, bytearray, memoryview)):
        return len(first), len(rows)
    return int(width or 0), int(height or 0)


def bbox_origin(bbox: Any) -> Tuple[float, float]:
    """Top-left of a bbox, whichever of the two shapes it arrives in.

    The plug-in carries two instance families that both call their rectangle
    ``bbox``: ``ui.canvas.Instance`` and ``outputs.Instance`` use a plain
    ``(x0, y0, x1, y1)`` tuple, while ``client.MaskInstance`` uses the
    :class:`client.BBox` dataclass, which has no ``__getitem__``.
    ``MainDialog._instances`` can hold *either* -- it falls back to the client
    objects whenever the canvas is unavailable or ``set_result`` raises -- so
    subscripting blindly turns a canvas failure into a second, stranger crash
    on Apply.  Read attributes first, fall back to indexing.
    """
    x0 = getattr(bbox, "x0", None)
    if x0 is not None:
        return (float(x0), float(getattr(bbox, "y0")))
    return (float(bbox[0]), float(bbox[1]))


def instance_strokes(mask: Any, mask_width: int, mask_height: int,
                     origin: Tuple[float, float] = (0.0, 0.0), *,
                     threshold: int = DEFAULT_THRESHOLD,
                     **kwargs: Any) -> List[Dict[str, Any]]:
    """Strokes for one daemon instance, in the shape ``outputs.build_path`` wants.

    The daemon returns each instance as a soft mask cropped to its bounding box
    (``API.md`` §8), so ``origin`` is that bbox's top-left and the strokes come
    back in **model-canvas** coordinates -- which is what makes
    ``path_space=Space.CANVAS`` correct downstream.

    The explicit ``control_points`` spelling is deliberate:
    ``outputs.normalise_stroke`` reads a *bare* flat sequence whose length
    divides by six as bezier triples, so naming the key removes the ambiguity.
    """
    result = mask_to_paths(mask, mask_width, mask_height, threshold=threshold,
                           origin=bbox_origin(origin), **kwargs)
    return [{"control_points": list(s.control_points), "closed": bool(s.closed)}
            for s in result.strokes]
