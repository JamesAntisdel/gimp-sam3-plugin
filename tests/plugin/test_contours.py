"""Tests for ``contours`` -- mask to vector paths.

This is one of the few modules that is fully exercisable without GIMP (no
GPU, no torch, no weights either), so it gets tested hard: synthetic shapes
with known areas, exact-area assertions on the raw lattice contours, a
winding-number re-rasterisation that proves holes subtract, and a dense
sampling check that the fitted beziers stay near the polygon they came from.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import pytest

import contours as C


# ---------------------------------------------------------------------------
# synthetic shape helpers
# ---------------------------------------------------------------------------

def make_mask(width, height, predicate, on=255, off=0):
    """Rasterise ``predicate(x, y) -> bool`` into a flat u8 buffer."""
    buf = bytearray(width * height)
    for y in range(height):
        row = y * width
        for x in range(width):
            if predicate(x, y):
                buf[row + x] = on
            else:
                buf[row + x] = off
    return bytes(buf)


def rect_mask(width, height, rects, on=255):
    buf = bytearray(width * height)
    for (x0, y0, x1, y1) in rects:
        for y in range(y0, y1):
            for x in range(x0, x1):
                buf[y * width + x] = on
    return bytes(buf)


def disc(cx, cy, r):
    rr = r * r
    return lambda x, y: (x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 <= rr


def annulus(cx, cy, r_in, r_out):
    def pred(x, y):
        d2 = (x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2
        return r_in * r_in <= d2 <= r_out * r_out
    return pred


def count_fg(mask, threshold=128):
    return sum(1 for v in mask if v >= threshold)


def winding_number(pt, polygon):
    """Non-zero winding number of a closed polygon around a point."""
    x, y = pt
    wn = 0
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        if y0 <= y:
            if y1 > y:
                # upward crossing; left-of test
                if (x1 - x0) * (y - y0) - (x - x0) * (y1 - y0) > 0:
                    wn += 1
        else:
            if y1 <= y:
                if (x1 - x0) * (y - y0) - (x - x0) * (y1 - y0) < 0:
                    wn -= 1
    return wn


def rasterize(result_contours, width, height):
    """Fill the contours with the non-zero winding rule, back into a mask.

    This is the real proof that winding is right: if outers and holes were
    wound the same way, holes would fill in and this would not match.
    """
    buf = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            total = 0
            for c in result_contours:
                total += winding_number((x + 0.5, y + 0.5), c.points)
            if total != 0:
                buf[y * width + x] = 255
    return bytes(buf)


def point_to_polygon_distance(pt, polygon):
    n = len(polygon)
    return min(C._point_segment_distance(pt, polygon[i], polygon[(i + 1) % n])
               for i in range(n))


def max_bezier_deviation(polygon, segments, samples=40):
    """Largest distance from any point on the fitted curve to the polygon."""
    worst = 0.0
    for seg in segments:
        for k in range(samples + 1):
            p = C._bezier_point(seg, k / samples)
            d = point_to_polygon_distance(p, polygon)
            if d > worst:
                worst = d
    return worst


def depth_of(contours, i):
    d = 0
    while contours[i].parent >= 0:
        i = contours[i].parent
        d += 1
    return d


# ---------------------------------------------------------------------------
# extraction: degenerate and trivial cases
# ---------------------------------------------------------------------------

def test_empty_mask_yields_nothing():
    assert C.extract_contours(bytes(10 * 10), 10, 10) == []
    res = C.mask_to_paths(bytes(10 * 10), 10, 10)
    assert res.contours == [] and res.strokes == [] and res.shapes == []


def test_zero_sized_mask():
    assert C.extract_contours(b"", 0, 0) == []


def test_full_mask_is_one_contour_covering_everything():
    mask = b"\xff" * (7 * 5)
    cs = C.extract_contours(mask, 7, 5)
    assert len(cs) == 1
    assert cs[0].points == [(0.0, 0.0), (7.0, 0.0), (7.0, 5.0), (0.0, 5.0)]
    assert cs[0].area == 35.0
    assert cs[0].is_hole is False


def test_single_pixel_polygon_area_and_winding():
    cs = C.extract_contours(rect_mask(5, 5, [(1, 1, 2, 2)]), 5, 5)
    assert len(cs) == 1
    c = cs[0]
    assert c.points == [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)]
    # positive == outer, per the module's winding convention
    assert c.area == 1.0
    assert c.is_hole is False
    assert c.parent == -1


def test_rectangle_is_exact():
    cs = C.extract_contours(rect_mask(20, 12, [(3, 2, 15, 9)]), 20, 12)
    assert len(cs) == 1
    assert len(cs[0].points) == 4
    assert cs[0].area == float(12 * 7)


def test_mask_touching_the_border_still_closes():
    cs = C.extract_contours(rect_mask(8, 8, [(0, 0, 3, 8)]), 8, 8)
    assert len(cs) == 1
    assert cs[0].area == 24.0
    assert cs[0].points == [(0.0, 0.0), (3.0, 0.0), (3.0, 8.0), (0.0, 8.0)]


# ---------------------------------------------------------------------------
# components, holes, nesting
# ---------------------------------------------------------------------------

def test_circle_is_one_component_with_the_right_area():
    w = h = 101
    mask = make_mask(w, h, disc(50, 50, 40))
    cs = C.extract_contours(mask, w, h)
    assert len(cs) == 1
    assert not cs[0].is_hole
    # the lattice polygon's area is exactly the foreground pixel count
    assert cs[0].area == float(count_fg(mask))
    assert cs[0].area == pytest.approx(math.pi * 40 * 40, rel=0.01)


def test_two_disjoint_blobs_give_two_outer_contours():
    mask = rect_mask(30, 14, [(1, 1, 6, 6), (18, 3, 27, 11)])
    cs = C.extract_contours(mask, 30, 14)
    assert len(cs) == 2
    assert all(not c.is_hole for c in cs)
    assert all(c.parent == -1 for c in cs)
    assert sorted(c.area for c in cs) == [25.0, 72.0]


def test_annulus_has_one_outer_and_one_hole():
    w = h = 61
    mask = make_mask(w, h, annulus(30, 30, 12, 26))
    cs = C.extract_contours(mask, w, h)
    assert len(cs) == 2
    outers = [c for c in cs if not c.is_hole]
    holes = [c for c in cs if c.is_hole]
    assert len(outers) == 1 and len(holes) == 1
    # opposite winding: outer positive, hole negative
    assert outers[0].area > 0 and holes[0].area < 0
    # the hole knows which outer it belongs to
    assert holes[0].parent == cs.index(outers[0])
    assert outers[0].parent == -1
    # outer minus hole is exactly the foreground pixel count
    assert sum(c.area for c in cs) == float(count_fg(mask))


def test_annulus_hole_is_not_merely_a_second_component():
    w = h = 41
    mask = make_mask(w, h, annulus(20, 20, 8, 17))
    cs = C.extract_contours(mask, w, h)
    hole = [c for c in cs if c.is_hole][0]
    outer = [c for c in cs if not c.is_hole][0]
    # the hole is geometrically inside the outer
    assert C.point_in_polygon(hole.sample, outer.points)
    assert not C.point_in_polygon(outer.sample, hole.points)


def test_island_inside_a_hole_alternates_winding():
    # concentric squares: solid ring, empty ring, solid core
    w = h = 31

    def pred(x, y):
        d = max(abs(x - 15), abs(y - 15))
        return d <= 14 and not (5 <= d <= 9)

    mask = make_mask(w, h, pred)
    cs = C.extract_contours(mask, w, h)
    assert len(cs) == 3
    signs = {depth_of(cs, i): cs[i].area > 0 for i in range(3)}
    assert signs == {0: True, 1: False, 2: True}
    # nesting chain: island -> hole -> outer
    island = [i for i in range(3) if depth_of(cs, i) == 2][0]
    hole = cs[island].parent
    assert cs[hole].is_hole
    assert cs[cs[hole].parent].parent == -1
    assert sum(c.area for c in cs) == float(count_fg(mask))


def test_two_holes_in_one_blob():
    w = h = 21
    mask = bytearray(rect_mask(w, h, [(2, 2, 19, 19)]))
    for (hx, hy) in ((5, 5), (13, 12)):
        for y in range(hy, hy + 3):
            for x in range(hx, hx + 3):
                mask[y * w + x] = 0
    cs = C.extract_contours(bytes(mask), w, h)
    assert len(cs) == 3
    holes = [c for c in cs if c.is_hole]
    assert len(holes) == 2
    assert all(h.area == -9.0 for h in holes)
    assert len({h.parent for h in holes}) == 1


# ---------------------------------------------------------------------------
# winding correctness proved by re-rasterisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,w,h,pred", [
    ("annulus", 41, 41, annulus(20, 20, 8, 17)),
    ("disc", 33, 33, disc(16, 16, 13)),
])
def test_nonzero_fill_of_contours_reproduces_the_mask(name, w, h, pred):
    mask = make_mask(w, h, pred)
    cs = C.extract_contours(mask, w, h)
    assert rasterize(cs, w, h) == mask


def test_nonzero_fill_reproduces_nested_shape():
    w = h = 27

    def pred(x, y):
        d = max(abs(x - 13), abs(y - 13))
        return d <= 12 and not (4 <= d <= 8)

    mask = make_mask(w, h, pred)
    cs = C.extract_contours(mask, w, h)
    # if the hole were wound the same way as the outer, the hole would fill in
    assert rasterize(cs, w, h) == mask


def test_nonzero_fill_reproduces_two_blobs():
    mask = rect_mask(24, 12, [(2, 2, 8, 9), (14, 1, 22, 10)])
    cs = C.extract_contours(mask, 24, 12)
    assert rasterize(cs, 24, 12) == mask


# ---------------------------------------------------------------------------
# minimum-area filter
# ---------------------------------------------------------------------------

def test_speckle_survives_without_a_filter_and_dies_with_one():
    mask = rect_mask(40, 40, [(5, 5, 25, 25), (32, 3, 33, 4), (36, 30, 37, 31)])
    assert len(C.extract_contours(mask, 40, 40)) == 3
    cs = C.extract_contours(mask, 40, 40, min_area=2.0)
    assert len(cs) == 1
    assert cs[0].area == 400.0


def test_min_area_keeps_shapes_at_exactly_the_limit():
    mask = rect_mask(20, 20, [(2, 2, 5, 5), (10, 10, 11, 11)])
    cs = C.extract_contours(mask, 20, 20, min_area=9.0)
    assert [c.area for c in cs] == [9.0]


def test_dropping_a_blob_drops_the_hole_inside_it():
    # a small ring (area 8) plus a large square; min_area removes the ring and
    # must not leave its hole behind as an orphan
    w = h = 40
    mask = bytearray(rect_mask(w, h, [(20, 20, 38, 38), (2, 2, 5, 5)]))
    mask[3 * w + 3] = 0  # 1px hole inside the 3x3 blob
    cs = C.extract_contours(bytes(mask), w, h, min_area=50.0)
    assert len(cs) == 1
    assert cs[0].area == float(18 * 18)
    assert not cs[0].is_hole


def test_min_hole_area_fills_pinholes_but_keeps_the_blob():
    w = h = 30
    mask = bytearray(rect_mask(w, h, [(4, 4, 26, 26)]))
    mask[10 * w + 10] = 0
    cs = C.extract_contours(bytes(mask), w, h)
    assert len(cs) == 2
    cs = C.extract_contours(bytes(mask), w, h, min_hole_area=4.0)
    assert len(cs) == 1
    assert not cs[0].is_hole


def test_fill_holes_drops_holes_and_their_islands():
    w = h = 31

    def pred(x, y):
        d = max(abs(x - 15), abs(y - 15))
        return d <= 14 and not (5 <= d <= 9)

    mask = make_mask(w, h, pred)
    assert len(C.extract_contours(mask, w, h)) == 3
    cs = C.extract_contours(mask, w, h, fill_holes=True)
    assert len(cs) == 1
    assert cs[0].area == float(29 * 29)


# ---------------------------------------------------------------------------
# connectivity
# ---------------------------------------------------------------------------

def test_diagonal_pixels_are_one_component_at_8_and_two_at_4():
    mask = rect_mask(6, 6, [(1, 1, 2, 2), (2, 2, 3, 3)])
    eight = C.extract_contours(mask, 6, 6, connectivity=8)
    four = C.extract_contours(mask, 6, 6, connectivity=4)
    assert len(eight) == 1
    assert eight[0].area == 2.0
    assert len(four) == 2
    assert [c.area for c in four] == [1.0, 1.0]


def test_bad_connectivity_is_rejected():
    with pytest.raises(ValueError):
        C.extract_contours(rect_mask(4, 4, [(1, 1, 3, 3)]), 4, 4, connectivity=6)


# ---------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------

class FakeArray:
    """Duck-types just enough of numpy for ``_binary_rows``; the module must
    never import numpy to read it."""

    def __init__(self, rows, dtype="uint8"):
        self._rows = [list(r) for r in rows]
        self.shape = (len(self._rows), len(self._rows[0]))
        self.dtype = dtype

    def tolist(self):
        return [list(r) for r in self._rows]

    def tobytes(self):
        out = bytearray()
        for r in self._rows:
            out.extend(bytes(r))
        return bytes(out)


def test_every_input_form_agrees():
    w, h = 9, 7
    flat = rect_mask(w, h, [(2, 1, 7, 5)])
    rows = [list(flat[y * w:(y + 1) * w]) for y in range(h)]
    forms = [
        C.extract_contours(flat, w, h),
        C.extract_contours(bytearray(flat), w, h),
        C.extract_contours(memoryview(flat), w, h),
        C.extract_contours(rows),
        C.extract_contours(list(flat), w, h),
        C.extract_contours(FakeArray(rows)),
        C.extract_contours(FakeArray(rows, dtype="float32")),
        C.extract_contours([[bool(v) for v in r] for r in rows]),
    ]
    first = forms[0][0].points
    assert first == [(2.0, 1.0), (7.0, 1.0), (7.0, 5.0), (2.0, 5.0)]
    for got in forms[1:]:
        assert len(got) == 1
        assert got[0].points == first


def test_flat_buffer_requires_dimensions():
    with pytest.raises(ValueError):
        C.extract_contours(b"\x00" * 16)
    with pytest.raises(ValueError):
        C.extract_contours(b"\x00" * 15, 4, 4)


def test_ragged_rows_are_rejected():
    with pytest.raises(ValueError):
        C.extract_contours([[0, 0, 0], [0, 0]])


def test_threshold_selects_the_level_set():
    w = h = 9
    # a soft ramp: value grows with x
    mask = bytes(bytearray(min(255, x * 32) for y in range(h) for x in range(w)))
    lo = C.extract_contours(mask, w, h, threshold=32)
    hi = C.extract_contours(mask, w, h, threshold=200)
    assert lo[0].area > hi[0].area
    # threshold 32 keeps x >= 1, threshold 200 keeps x >= 7
    assert lo[0].area == float(8 * h)
    assert hi[0].area == float(2 * h)


def test_default_threshold_is_the_soft_mask_midpoint():
    assert C.DEFAULT_THRESHOLD == 128
    mask = bytes([127, 128, 127, 127])
    cs = C.extract_contours(mask, 2, 2)
    assert len(cs) == 1 and cs[0].area == 1.0


# ---------------------------------------------------------------------------
# Douglas-Peucker
# ---------------------------------------------------------------------------

def test_simplify_removes_the_staircase_and_preserves_area():
    w = h = 121
    mask = make_mask(w, h, disc(60, 60, 50))
    raw = C.extract_contours(mask, w, h)[0]
    simple = C.simplify_polygon(raw.points, 1.0)
    assert len(simple) < len(raw.points) / 4
    assert C.signed_area(simple) > 0  # winding survived
    assert C.signed_area(simple) == pytest.approx(raw.area, rel=0.01)


def test_simplify_area_error_grows_with_tolerance_but_stays_bounded():
    w = h = 121
    mask = make_mask(w, h, disc(60, 60, 50))
    raw = C.extract_contours(mask, w, h)[0]
    perimeter = sum(C._dist(raw.points[i], raw.points[(i + 1) % len(raw.points)])
                    for i in range(len(raw.points)))
    losses = []
    for tol, rel in ((0.75, 0.01), (2.0, 0.04), (4.0, 0.10)):
        simple = C.simplify_polygon(raw.points, tol)
        assert len(simple) >= 3
        loss = abs(raw.area - C.signed_area(simple))
        losses.append(loss)
        # DP moves the outline by at most `tol`, so the area it can gain or
        # lose is bounded by the band that sweeps out along the perimeter.
        assert loss <= perimeter * tol * 0.5
        assert C.signed_area(simple) == pytest.approx(raw.area, rel=rel)
    assert losses == sorted(losses)


def test_simplify_keeps_a_square_square():
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert C.simplify_polygon(square, 3.0) == square


def test_simplify_is_identity_at_zero_tolerance():
    ring = [(0.0, 0.0), (4.0, 0.0), (4.0, 1.0), (4.0, 4.0), (0.0, 4.0)]
    assert C.simplify_polygon(ring, 0.0) == ring


def test_simplify_open_polyline():
    line = [(0.0, 0.0), (1.0, 0.05), (2.0, -0.05), (3.0, 0.0), (4.0, 3.0)]
    out = C.simplify_polygon(line, 0.5, closed=False)
    assert out[0] == line[0] and out[-1] == line[-1]
    assert out == [(0.0, 0.0), (3.0, 0.0), (4.0, 3.0)]


def test_simplify_is_rotation_independent():
    w = h = 61
    mask = make_mask(w, h, disc(30, 30, 25))
    raw = C.extract_contours(mask, w, h)[0].points
    a = C.simplify_polygon(raw, 1.5)
    rotated = raw[17:] + raw[:17]
    b = C.simplify_polygon(rotated, 1.5)
    assert set(a) == set(b)


def test_simplify_contours_is_index_parallel():
    w = h = 41
    mask = make_mask(w, h, annulus(20, 20, 8, 17))
    cs = C.extract_contours(mask, w, h)
    simplified = C.simplify_contours(cs, 1.0)
    assert len(simplified) == len(cs)
    assert [c.parent for c in simplified] == [c.parent for c in cs]
    assert [c.is_hole for c in simplified] == [c.is_hole for c in cs]


def test_simplify_preserves_hole_winding():
    w = h = 61
    mask = make_mask(w, h, annulus(30, 30, 12, 26))
    res = C.mask_to_paths(mask, w, h, simplify_tolerance=1.5, bezier=False)
    outer = [c for c in res.contours if not c.is_hole][0]
    hole = [c for c in res.contours if c.is_hole][0]
    assert outer.area > 0 > hole.area
    assert abs(hole.area) < abs(outer.area)


# ---------------------------------------------------------------------------
# stroke / control-point output shape
# ---------------------------------------------------------------------------

def test_polyline_mode_puts_handles_on_the_anchors():
    mask = rect_mask(12, 12, [(2, 2, 9, 8)])
    res = C.mask_to_paths(mask, 12, 12, bezier=False)
    stroke = res.strokes[0]
    assert stroke.closed is True
    assert stroke.is_hole is False
    assert len(stroke.control_points) == 6 * len(res.contours[0].points)
    assert len(stroke.control_points) % 6 == 0
    for c1, p, c2 in stroke.anchors():
        assert c1 == p == c2
    assert stroke.anchor_points() == res.contours[0].points


def test_bezier_mode_emits_six_floats_per_anchor():
    w = h = 81
    mask = make_mask(w, h, disc(40, 40, 30))
    res = C.mask_to_paths(mask, w, h, simplify_tolerance=1.0)
    stroke = res.strokes[0]
    assert len(stroke.control_points) % 6 == 0
    assert stroke.anchor_count >= 2
    assert all(isinstance(v, float) for v in stroke.control_points)
    # handles are genuinely off the anchors for a curved shape
    assert any(c1 != p for c1, p, c2 in stroke.anchors())


def test_bezier_keeps_square_corners_exact():
    mask = rect_mask(20, 20, [(4, 4, 16, 16)])
    res = C.mask_to_paths(mask, 20, 20)
    stroke = res.strokes[0]
    assert stroke.anchor_count == 4
    assert set(stroke.anchor_points()) == {(4.0, 4.0), (16.0, 4.0),
                                           (16.0, 16.0), (4.0, 16.0)}
    # each anchor's handles lie on the incident edges, so the corner stays sharp
    for c1, p, c2 in stroke.anchors():
        assert (c1[0] == p[0]) != (c1[1] == p[1])
        assert (c2[0] == p[0]) != (c2[1] == p[1])


def test_hole_strokes_are_flagged_and_parented():
    w = h = 61
    mask = make_mask(w, h, annulus(30, 30, 12, 26))
    res = C.mask_to_paths(mask, w, h, simplify_tolerance=1.0)
    holes = [s for s in res.strokes if s.is_hole]
    assert len(holes) == 1
    assert holes[0].parent >= 0
    assert not res.strokes[holes[0].parent].is_hole


# ---------------------------------------------------------------------------
# bezier accuracy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tol", [0.75, 1.5, 3.0])
def test_fitted_bezier_stays_within_tolerance_of_a_circle(tol):
    w = h = 141
    mask = make_mask(w, h, disc(70, 70, 60))
    res = C.mask_to_paths(mask, w, h, simplify_tolerance=tol,
                          bezier_tolerance=tol)
    poly = res.contours[0].points
    segs = C.fit_closed_bezier(poly, tol)
    assert max_bezier_deviation(poly, segs) <= tol + 1e-6


def test_fitted_bezier_stays_within_tolerance_of_a_wavy_blob():
    w = h = 201
    tol = 1.5

    def pred(x, y):
        dx, dy = x - 100, y - 100
        r = math.hypot(dx, dy)
        a = math.atan2(dy, dx)
        return r <= 62 + 14 * math.sin(3 * a) + 8 * math.cos(5 * a + 1.0)

    mask = make_mask(w, h, pred)
    res = C.mask_to_paths(mask, w, h, simplify_tolerance=tol,
                          bezier_tolerance=tol)
    assert len(res.contours) == 1
    poly = res.contours[0].points
    segs = C.fit_closed_bezier(poly, tol)
    assert max_bezier_deviation(poly, segs) <= tol + 1e-6


def test_fitted_bezier_stays_within_tolerance_on_hole_contours():
    w = h = 121
    tol = 1.25
    mask = make_mask(w, h, annulus(60, 60, 22, 52))
    res = C.mask_to_paths(mask, w, h, simplify_tolerance=tol,
                          bezier_tolerance=tol)
    assert len(res.contours) == 2
    for c in res.contours:
        segs = C.fit_closed_bezier(c.points, tol)
        assert max_bezier_deviation(c.points, segs) <= tol + 1e-6


def test_bezier_fit_of_a_polygon_with_corners_is_exact():
    poly = [(0.0, 0.0), (30.0, 0.0), (30.0, 20.0), (10.0, 20.0),
            (10.0, 40.0), (0.0, 40.0)]
    segs = C.fit_closed_bezier(poly, 1.0)
    assert len(segs) == 6
    assert max_bezier_deviation(poly, segs) < 1e-9


def test_bezier_seam_on_a_smooth_ring_is_smooth():
    # a circle has no corners, so the fitter cuts it at two antipodal points
    # and must join them G1-continuously
    w = h = 121
    mask = make_mask(w, h, disc(60, 60, 50))
    res = C.mask_to_paths(mask, w, h, simplify_tolerance=1.0,
                          bezier_tolerance=1.0)
    stroke = res.strokes[0]
    for c1, p, c2 in stroke.anchors():
        din = (p[0] - c1[0], p[1] - c1[1])
        dout = (c2[0] - p[0], c2[1] - p[1])
        na = math.hypot(*din)
        nb = math.hypot(*dout)
        assert na > 0 and nb > 0
        cos = (din[0] * dout[0] + din[1] * dout[1]) / (na * nb)
        assert cos > 0.98  # handles are near-collinear: no visible kink


def test_bezier_reduces_the_anchor_count_versus_the_polygon():
    w = h = 141
    mask = make_mask(w, h, disc(70, 70, 60))
    res = C.mask_to_paths(mask, w, h, simplify_tolerance=1.0,
                          bezier_tolerance=1.0)
    poly = C.mask_to_paths(mask, w, h, simplify_tolerance=1.0, bezier=False)
    assert res.strokes[0].anchor_count < poly.strokes[0].anchor_count


def test_fit_closed_bezier_rejects_degenerate_input():
    assert C.fit_closed_bezier([(0.0, 0.0), (1.0, 1.0)], 1.0) == []


def test_bezier_segments_chain_end_to_end():
    w = h = 81
    mask = make_mask(w, h, disc(40, 40, 33))
    poly = C.mask_to_paths(mask, w, h, simplify_tolerance=1.0,
                           bezier=False).contours[0].points
    segs = C.fit_closed_bezier(poly, 1.0)
    for i in range(len(segs)):
        assert segs[i][3] == pytest.approx(segs[(i + 1) % len(segs)][0])


# ---------------------------------------------------------------------------
# shapes, transform, serialisation
# ---------------------------------------------------------------------------

def test_shapes_group_holes_under_their_outer():
    w = h = 61
    mask = bytearray(rect_mask(w, h, [(4, 4, 56, 56)]))
    for (hx, hy) in ((10, 10), (36, 34)):
        for y in range(hy, hy + 6):
            for x in range(hx, hx + 6):
                mask[y * w + x] = 0
    res = C.mask_to_paths(bytes(mask), w, h, simplify_tolerance=1.0)
    assert len(res.shapes) == 1
    assert len(res.shapes[0].holes) == 2
    assert res.outer_count == 1 and res.hole_count == 2
    for h_idx in res.shapes[0].holes:
        assert res.contours[h_idx].is_hole
        assert res.contours[h_idx].parent == res.shapes[0].outer


def test_shapes_for_two_blobs_and_an_island():
    w = h = 41

    def pred(x, y):
        if x < 20:
            d = max(abs(x - 9), abs(y - 20))
            return d <= 8 and not (3 <= d <= 5)
        return 25 <= x <= 35 and 10 <= y <= 30

    mask = make_mask(w, h, pred)
    res = C.mask_to_paths(mask, w, h, simplify_tolerance=1.0)
    # left blob: outer + hole + island; right blob: outer.  3 shapes total.
    assert len(res.shapes) == 3
    assert sum(len(s.holes) for s in res.shapes) == 1


def test_origin_and_scale_are_applied_last():
    mask = rect_mask(12, 12, [(2, 2, 8, 6)])
    plain = C.mask_to_paths(mask, 12, 12, bezier=False)
    moved = C.mask_to_paths(mask, 12, 12, bezier=False,
                            origin=(100.0, 50.0), scale=2.0)
    assert moved.contours[0].points == [
        (100.0 + 2 * x, 50.0 + 2 * y) for (x, y) in plain.contours[0].points]
    assert moved.contours[0].area == pytest.approx(plain.contours[0].area * 4)
    assert moved.strokes[0].control_points[2:4] == [
        100.0 + 2 * plain.strokes[0].control_points[2],
        50.0 + 2 * plain.strokes[0].control_points[3]]


def test_non_uniform_scale():
    mask = rect_mask(10, 10, [(1, 1, 5, 3)])
    res = C.mask_to_paths(mask, 10, 10, bezier=False, scale=(3.0, 0.5))
    assert res.contours[0].points == [(3.0, 0.5), (15.0, 0.5),
                                      (15.0, 1.5), (3.0, 1.5)]
    assert res.contours[0].area == pytest.approx(4 * 3.0 * 2 * 0.5)


def test_mirroring_scale_preserves_winding():
    w = h = 41
    mask = make_mask(w, h, annulus(20, 20, 8, 17))
    res = C.mask_to_paths(mask, w, h, bezier=False, scale=(-1.0, 1.0))
    assert res.outer_count == 1 and res.hole_count == 1
    outer = [c for c in res.contours if not c.is_hole][0]
    hole = [c for c in res.contours if c.is_hole][0]
    assert outer.area > 0 > hole.area


def test_result_is_json_serialisable():
    w = h = 41
    mask = make_mask(w, h, annulus(20, 20, 8, 17))
    res = C.mask_to_paths(mask, w, h, simplify_tolerance=1.0)
    blob = json.dumps(res.to_dict())
    back = json.loads(blob)
    assert back["width"] == w and back["height"] == h
    assert len(back["contours"]) == len(res.contours)
    assert len(back["strokes"][0]["control_points"]) % 6 == 0
    assert len(back["contours"][0]["points"]) % 2 == 0
    assert back["shapes"][0]["outer"] == res.shapes[0].outer


def test_result_reports_mask_size_for_every_input_form():
    rows = [[255] * 6 for _ in range(4)]
    assert (C.mask_to_paths(rows).width, C.mask_to_paths(rows).height) == (6, 4)
    flat = bytes([255] * 24)
    res = C.mask_to_paths(flat, 6, 4)
    assert (res.width, res.height) == (6, 4)


def test_total_area_tracks_the_pixel_count():
    w = h = 81
    mask = make_mask(w, h, annulus(40, 40, 14, 34))
    exact = C.mask_to_paths(mask, w, h, simplify_tolerance=0.0, bezier=False)
    assert exact.total_area() == float(count_fg(mask))
    smooth = C.mask_to_paths(mask, w, h, simplify_tolerance=1.0)
    assert smooth.total_area() == pytest.approx(float(count_fg(mask)), rel=0.02)


# ---------------------------------------------------------------------------
# misc invariants
# ---------------------------------------------------------------------------

def test_signed_area_and_helpers():
    square = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    assert C.signed_area(square) == 4.0
    assert C.signed_area(list(reversed(square))) == -4.0
    assert C.polygon_area(list(reversed(square))) == 4.0
    assert C.signed_area([(0.0, 0.0), (1.0, 1.0)]) == 0.0
    assert C.point_in_polygon((1.0, 1.0), square)
    assert not C.point_in_polygon((3.0, 1.0), square)
    assert not C.point_in_polygon((1.0, 1.0), square[:2])


def test_pipeline_is_deterministic():
    w = h = 61
    mask = make_mask(w, h, annulus(30, 30, 11, 26))
    a = C.mask_to_paths(mask, w, h, simplify_tolerance=1.0).to_dict()
    b = C.mask_to_paths(mask, w, h, simplify_tolerance=1.0).to_dict()
    assert a == b


def test_large_mask_is_handled_without_recursion_trouble():
    # a full-canvas-sized mask: the DP and the tracer must both be iterative
    # enough to survive a 1008px contour
    w = h = 1008
    mask = make_mask(w, h, disc(504, 504, 480))
    res = C.mask_to_paths(mask, w, h, simplify_tolerance=1.0, min_area=4.0)
    assert len(res.contours) == 1
    assert res.contours[0].area == pytest.approx(math.pi * 480 * 480, rel=0.01)
    assert res.strokes[0].anchor_count >= 8


def test_module_imports_without_torch_or_numpy(plugin_dir):
    """Stdlib only -- nothing about this module may drag in a heavy import.

    ``gi`` is on the list because the scriptable (non-interactive) Paths mode
    routes through here precisely so that it does *not* have to import GTK.
    """
    code = ("import sys, contours as c; "
            "assert 'torch' not in sys.modules, 'torch imported'; "
            "assert 'numpy' not in sys.modules, 'numpy imported'; "
            "assert 'transformers' not in sys.modules, 'transformers imported'; "
            "assert 'gi' not in sys.modules, 'gi imported'; "
            "assert 'cairo' not in sys.modules, 'cairo imported'; "
            "print('clean')")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(plugin_dir)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=False, env=env)
    assert out.returncode == 0, out.stderr
    assert "clean" in out.stdout


# --------------------------------------------------------------------------- #
# bbox shape tolerance -- regression for a real interface-drift bug
# --------------------------------------------------------------------------- #
class TestBBoxOrigin:
    """``instance_strokes`` is handed instances from two unrelated families.

    ``ui.canvas.Instance`` and ``outputs.Instance`` carry a plain
    ``(x0, y0, x1, y1)`` tuple; ``client.MaskInstance`` carries a
    ``client.BBox`` dataclass with no ``__getitem__``.  ``MainDialog``
    genuinely holds either -- it falls back to the client objects whenever the
    canvas is missing or ``set_result`` raises -- so subscripting the bbox
    turned a canvas failure into a second crash on Apply in Paths mode.
    """

    @staticmethod
    def _square_mask(w=24, h=24, lo=6, hi=18):
        m = bytearray(w * h)
        for y in range(lo, hi):
            for x in range(lo, hi):
                m[y * w + x] = 255
        return bytes(m), w, h

    def test_accepts_a_plain_tuple(self):
        mask, w, h = self._square_mask()
        assert C.instance_strokes(mask, w, h, (100, 50, 124, 74))

    def test_accepts_a_client_bbox_dataclass(self):
        """The shape that used to raise ``'BBox' object is not subscriptable``."""
        import client

        mask, w, h = self._square_mask()
        strokes = C.instance_strokes(mask, w, h, client.BBox(100, 50, 124, 74))
        assert strokes
        anchors_x = strokes[0]["control_points"][2::6]
        assert min(anchors_x) >= 100.0      # the origin really was applied

    def test_both_shapes_agree(self):
        import client

        mask, w, h = self._square_mask()
        as_tuple = C.instance_strokes(mask, w, h, (7, 11, 31, 35))
        as_bbox = C.instance_strokes(mask, w, h, client.BBox(7, 11, 31, 35))
        assert as_tuple == as_bbox

    def test_a_real_client_mask_instance_round_trips(self):
        """End of the drift: the exact object ``client`` hands back."""
        import client

        mask, w, h = self._square_mask()
        inst = client.MaskInstance(
            instance_id=0, score=0.9, label="car",
            bbox=client.BBox(40, 60, 40 + w, 60 + h),
            mask_width=w, mask_height=h,
            blob_offset=0, blob_length=len(mask), mask=mask,
        )
        strokes = C.instance_strokes(inst.mask, inst.mask_width,
                                     inst.mask_height, inst.bbox)
        assert strokes
        cp = strokes[0]["control_points"]
        assert len(cp) % 6 == 0
        assert min(cp[2::6]) >= 40.0 and min(cp[3::6]) >= 60.0


def test_component_cap_keeps_the_largest_and_stays_fast():
    """A speckled mask yields thousands of components; the cap keeps the big
    body and the largest specks, and parent assignment stays linear-ish.
    Uncapped, 31,595 components took three minutes on a real-sized mask."""
    import random
    import time

    rng = random.Random(5)
    w, h = 240, 180
    mask = bytearray(w * h)
    for y in range(h):
        for x in range(w):
            body = 60 <= x < 180 and 40 <= y < 140
            speck = rng.random() < 0.08
            mask[y * w + x] = 255 if body != speck else 0

    start = time.monotonic()
    everything = C.extract_contours(bytes(mask), w, h, max_contours=0)
    capped = C.extract_contours(bytes(mask), w, h, max_contours=50)
    elapsed = time.monotonic() - start
    assert len(everything) > 500
    assert len(capped) <= 50
    biggest = max(everything, key=lambda c: abs(c.area))
    assert max(abs(c.area) for c in capped) == abs(biggest.area)
    # every survivor is at least as large as every dropped component
    floor = min(abs(c.area) for c in capped)
    kept_ids = {id(c) for c in capped}
    assert all(abs(c.area) <= floor for c in everything if abs(c.area) < floor)
    assert elapsed < 5.0, "%.1fs" % elapsed


def test_default_cap_is_applied():
    import random

    rng = random.Random(9)
    w, h = 200, 200
    mask = bytes(255 if rng.random() < 0.3 else 0 for _ in range(w * h))
    out = C.extract_contours(mask, w, h)
    assert 0 < len(out) <= C.DEFAULT_MAX_CONTOURS



class TestThinFeaturesSurviveSimplification:
    """A one-pixel line is a real object -- a cable, a strap -- and the kind
    of thing click-to-segment finds.  Douglas-Peucker at the default
    tolerance flattened its ring to two points and Paths traced nothing."""

    @staticmethod
    def _line(w, h, y, x0, x1):
        m = bytearray(w * h)
        for x in range(x0, x1):
            m[y * w + x] = 255
        return bytes(m)

    def test_a_one_pixel_line_still_traces(self):
        strokes = C.instance_strokes(self._line(64, 64, 30, 5, 60), 64, 64, (0, 0, 64, 64),
                                      threshold=128)
        assert len(strokes) == 1 and strokes[0]["closed"]
        pts = strokes[0]["control_points"]
        xs, ys = pts[0::2], pts[1::2]
        assert 4 <= min(xs) <= 6 and 59 <= max(xs) <= 61
        assert 29 <= min(ys) and max(ys) <= 32

    def test_a_thin_ring_keeps_both_edges(self):
        w = h = 64
        m = bytearray(w * h)
        for y in range(h):
            for x in range(w):
                d2 = (x - 32) ** 2 + (y - 32) ** 2
                if 23 * 23 <= d2 <= 25 * 25:
                    m[y * w + x] = 255
        strokes = C.instance_strokes(bytes(m), w, h, (0, 0, w, h), threshold=128)
        assert len(strokes) == 2

    def test_a_wide_shape_is_still_simplified(self):
        """The fallback only fires for rings that would vanish; a disc's
        staircase outline is still reduced as before."""
        m = bytearray(64 * 64)
        for y in range(64):
            for x in range(64):
                if (x - 32) ** 2 + (y - 32) ** 2 <= 20 * 20:
                    m[y * 64 + x] = 255
        contours = C.extract_contours(bytes(m), 64, 64, threshold=128)
        simplified = C.simplify_contours(contours, 1.0)
        assert 3 <= len(simplified[0].points) < len(contours[0].points)
