"""Tests for the interactive preview canvas (``plugin/sam3_gimp/ui/canvas.py``).

Two layers, deliberately:

1. **Pure functions** -- coordinate transforms, the client-side threshold, score
   filtering, hit-testing, contour tracing, the result-frame codec.  These need
   neither GTK nor a display, so they run everywhere and are where the real
   coverage lives.  Coordinate maths is where a widget like this normally
   breaks, so it is tested directly rather than through the widget.
2. **The widget itself** -- constructed for real, given a real allocation, and
   drawn through its real ``do_draw`` onto a ``cairo.ImageSurface``, with input
   delivered as synthesised ``Gdk`` events.  Marked ``needs_gtk`` so a run
   without GTK skips instead of failing.

Nothing here needs GIMP, torch, a GPU, weights or a running daemon.
"""

from __future__ import annotations

import json
import math
import struct

import pytest

from ui import canvas as cv


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def soft_disc(canvas_w, canvas_h, cx, cy, radius, feather=0.25):
    """A cropped soft uint8 mask for a disc, exactly as the daemon would send it.

    Value ``128`` lands on the nominal boundary, matching API §8.3 (``128`` ==
    logit 0), so a threshold above 128 shrinks the shape and below 128 grows it.
    Returns ``(bbox, mask_bytes)`` in **model-canvas** space.
    """
    pad = radius * (1.0 + feather) + 1.0
    x0 = max(0, int(math.floor(cx - pad)))
    y0 = max(0, int(math.floor(cy - pad)))
    x1 = min(canvas_w, int(math.ceil(cx + pad)))
    y1 = min(canvas_h, int(math.ceil(cy + pad)))
    mw, mh = x1 - x0, y1 - y0
    mask = bytearray(mw * mh)
    i = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            d = math.hypot(x + 0.5 - cx, y + 0.5 - cy) / radius
            t = (1.0 - d) / feather
            mask[i] = 0 if t <= -1.0 else (255 if t >= 1.0 else int(round(127.5 + 127.5 * t)))
            i += 1
    return (x0, y0, x1, y1), bytes(mask)


def make_frame(discs, image_wh=(640, 480), canvas_wh=(1008, 1008), request_id="r-1",
               engine="pcs", label="thing"):
    """Build a conforming ``SAM3RES`` frame from ``[(cx, cy, r, score)]``."""
    iw, ih = image_wh
    cw, ch = canvas_wh
    entries, blobs = [], []
    for n, (cx, cy, radius, score) in enumerate(discs):
        bbox, mask = soft_disc(cw, ch, cx, cy, radius)
        entries.append(
            {
                "instance_id": n,
                "score": score,
                "label": label,
                "bbox": list(bbox),
                "mask_width": bbox[2] - bbox[0],
                "mask_height": bbox[3] - bbox[1],
                "blob_offset": 0,
                "blob_length": 0,
            }
        )
        blobs.append(mask)
    header = {
        "api_version": "1.0",
        "job_id": "j-test",
        "request_id": request_id,
        "image_id": "test",
        "engine": engine,
        "state": "done",
        "prompt": {"kind": "text", "text": label, "score_threshold": 0.1},
        "image": {"width": iw, "height": ih},
        "model_canvas": {"width": cw, "height": ch},
        "canvas_from_image": {
            "scale_x": cw / float(iw),
            "scale_y": ch / float(ih),
            "offset_x": 0.0,
            "offset_y": 0.0,
        },
        "mask_encoding": "u8_soft",
        "elapsed_ms": 1.0,
        "truncated": False,
        "instances": entries,
        "blob_length": 0,
    }
    return cv.encode_frame(header, blobs)


def rgb_gradient(width, height):
    buf = bytearray(width * height * 3)
    i = 0
    for y in range(height):
        for x in range(width):
            buf[i] = (x * 255) // max(1, width - 1)
            buf[i + 1] = (y * 255) // max(1, height - 1)
            buf[i + 2] = 128
            i += 3
    return bytes(buf)


# --------------------------------------------------------------------------- #
# result frame codec (API §8)
# --------------------------------------------------------------------------- #
class TestFrameCodec:
    def test_round_trip(self):
        payload = make_frame([(300.0, 300.0, 60.0, 0.9), (700.0, 500.0, 40.0, 0.5)])
        header, blob = cv.decode_frame(payload)
        assert payload[:8] == cv.RESULT_MAGIC
        assert header["engine"] == "pcs"
        assert len(header["instances"]) == 2
        assert len(blob) == header["blob_length"]

    def test_blob_offsets_are_relative_to_the_blob_region(self):
        """API §16 rule 2: offsets index the blob region, not the whole body."""
        payload = make_frame([(300.0, 300.0, 60.0, 0.9), (700.0, 500.0, 40.0, 0.5)])
        header, blob = cv.decode_frame(payload)
        first, second = header["instances"]
        assert first["blob_offset"] == 0
        assert second["blob_offset"] == first["blob_length"]
        assert first["blob_length"] + second["blob_length"] == len(blob)

    def test_invariants_hold_for_every_instance(self):
        payload = make_frame([(300.0, 300.0, 60.0, 0.9)])
        header, _blob = cv.decode_frame(payload)
        for inst in header["instances"]:
            x0, y0, x1, y1 = inst["bbox"]
            assert inst["mask_width"] == x1 - x0
            assert inst["mask_height"] == y1 - y0
            assert inst["blob_length"] == inst["mask_width"] * inst["mask_height"]

    def test_bad_magic_rejected(self):
        payload = bytearray(make_frame([(300.0, 300.0, 60.0, 0.9)]))
        payload[0:1] = b"X"
        with pytest.raises(ValueError, match="magic"):
            cv.decode_frame(bytes(payload))

    def test_short_frame_rejected(self):
        with pytest.raises(ValueError):
            cv.decode_frame(b"SAM3")

    def test_overrunning_header_rejected(self):
        payload = bytearray(make_frame([(300.0, 300.0, 60.0, 0.9)]))
        struct.pack_into("<I", payload, 8, 10 ** 7)
        with pytest.raises(ValueError, match="overruns"):
            cv.decode_frame(bytes(payload))

    def test_blob_length_mismatch_rejected(self):
        payload = make_frame([(300.0, 300.0, 60.0, 0.9)])
        header, blob = cv.decode_frame(payload)
        header["blob_length"] += 1
        raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
        bad = cv.RESULT_MAGIC + struct.pack("<I", len(raw)) + raw + blob
        with pytest.raises(ValueError, match="blob_length"):
            cv.decode_frame(bad)

    def test_zero_instance_frame_is_legal(self):
        payload = make_frame([])
        header, blob = cv.decode_frame(payload)
        assert header["instances"] == []
        assert blob == b""


# --------------------------------------------------------------------------- #
# view transforms
# --------------------------------------------------------------------------- #
class TestViewTransforms:
    @pytest.mark.parametrize("view", [
        cv.View(1.0, 0.0, 0.0),
        cv.View(0.37, -122.5, 48.25),
        cv.View(7.5, 900.0, -1200.0),
    ])
    @pytest.mark.parametrize("point", [(0.0, 0.0), (13.5, 7.25), (1007.0, 671.0)])
    def test_round_trip(self, view, point):
        wx, wy = cv.image_to_widget(view, *point)
        ix, iy = cv.widget_to_image(view, wx, wy)
        assert ix == pytest.approx(point[0])
        assert iy == pytest.approx(point[1])

    def test_fit_centres_a_small_image_without_upscaling(self):
        view = cv.view_fit(64, 48, 400, 300)
        assert view.zoom == 1.0                       # max_zoom defaults to 100 %
        assert view.off_x == pytest.approx((400 - 64) / 2.0)
        assert view.off_y == pytest.approx((300 - 48) / 2.0)

    def test_fit_scales_a_large_image_down_and_preserves_aspect(self):
        view = cv.view_fit(1008, 672, 500, 500, margin=0.0)
        assert view.zoom == pytest.approx(500.0 / 1008.0)
        # the shorter axis gets the leftover space
        assert view.off_x == pytest.approx(0.0, abs=1e-9)
        assert view.off_y == pytest.approx((500 - 672 * view.zoom) / 2.0)

    def test_fit_of_a_degenerate_area_is_safe(self):
        assert cv.view_fit(0, 0, 100, 100) == cv.View(1.0, 0.0, 0.0)
        assert cv.view_fit(100, 100, 0, 0) == cv.View(1.0, 0.0, 0.0)

    @pytest.mark.parametrize("pointer", [(0.0, 0.0), (321.0, 77.0), (799.5, 599.5)])
    @pytest.mark.parametrize("factor", [1.15, 1.0 / 1.15, 4.0, 0.25])
    def test_zoom_pins_the_point_under_the_pointer(self, pointer, factor):
        """The invariant that makes wheel-zoom feel right."""
        view = cv.View(0.83, 41.0, -17.0)
        before = cv.widget_to_image(view, *pointer)
        zoomed = cv.view_zoom_at(view, factor, *pointer)
        after = cv.widget_to_image(zoomed, *pointer)
        assert after[0] == pytest.approx(before[0])
        assert after[1] == pytest.approx(before[1])
        assert zoomed.zoom == pytest.approx(view.zoom * factor)

    def test_zoom_clamps_and_is_idempotent_at_the_limit(self):
        view = cv.View(cv.MAX_ZOOM, 0.0, 0.0)
        assert cv.view_zoom_at(view, 2.0, 10.0, 10.0) is view
        low = cv.view_zoom_at(cv.View(cv.MIN_ZOOM, 0.0, 0.0), 0.5, 10.0, 10.0)
        assert low.zoom >= cv.MIN_ZOOM

    def test_pan_is_a_pure_widget_space_translation(self):
        view = cv.view_pan(cv.View(2.0, 10.0, 20.0), -5.0, 7.5)
        assert view == cv.View(2.0, 5.0, 27.5)

    def test_clamp_centres_when_the_image_is_smaller_than_the_widget(self):
        clamped = cv.view_clamp(cv.View(1.0, -500.0, 900.0), 100, 100, 400, 300)
        assert clamped.off_x == pytest.approx(150.0)
        assert clamped.off_y == pytest.approx(100.0)

    def test_clamp_locks_edges_when_the_image_is_larger(self):
        # 1000 px at zoom 1 inside a 400 px widget: offset must stay in [-600, 0]
        assert cv.view_clamp(cv.View(1.0, 50.0, 0.0), 1000, 1000, 400, 400).off_x == 0.0
        assert cv.view_clamp(cv.View(1.0, -900.0, 0.0), 1000, 1000, 400, 400).off_x == -600.0
        assert cv.view_clamp(cv.View(1.0, -300.0, 0.0), 1000, 1000, 400, 400).off_x == -300.0


# --------------------------------------------------------------------------- #
# canvas_from_image (API §5)
# --------------------------------------------------------------------------- #
class TestAffine:
    def test_round_trip_with_anisotropic_squash(self):
        aff = cv.Affine(1008 / 640.0, 1008 / 480.0, 0.0, 0.0)
        cx, cy = aff.to_canvas(320.0, 240.0)
        assert (cx, cy) == pytest.approx((504.0, 504.0))
        assert aff.to_image(cx, cy) == pytest.approx((320.0, 240.0))

    def test_letterboxing_transform_is_honoured_not_hardcoded(self):
        """API §5: a processor that letterboxes sets non-zero offsets.

        A client that assumed the 1008 squash would misplace every mask, so the
        offsets must actually be used.
        """
        aff = cv.Affine.from_dict(
            {"scale_x": 0.5, "scale_y": 0.5, "offset_x": 100.0, "offset_y": 254.0}
        )
        assert aff.to_canvas(0.0, 0.0) == (100.0, 254.0)
        assert aff.to_image(100.0, 254.0) == (0.0, 0.0)
        assert aff.rect_to_image((100, 254, 200, 354)) == pytest.approx((0.0, 0.0, 200.0, 200.0))

    def test_from_dict_is_tolerant(self):
        assert cv.Affine.from_dict(None) == cv.Affine.identity()
        assert cv.Affine.from_dict({}) == cv.Affine.identity()
        assert cv.Affine.from_dict({"scale_x": 0}).scale_x == 1.0  # never divide by zero


# --------------------------------------------------------------------------- #
# instances
# --------------------------------------------------------------------------- #
class TestInstance:
    def test_from_header_slices_the_right_bytes(self):
        payload = make_frame([(300.0, 300.0, 50.0, 0.9), (700.0, 400.0, 30.0, 0.4)])
        header, blob = cv.decode_frame(payload)
        insts = [cv.Instance.from_header(e, blob) for e in header["instances"]]
        assert len(insts) == 2
        for inst, entry in zip(insts, header["instances"]):
            assert len(inst.mask) == entry["blob_length"]
            assert inst.mask == blob[entry["blob_offset"]:
                                     entry["blob_offset"] + entry["blob_length"]]

    def test_bbox_mask_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="invariant"):
            cv.Instance(0, 0.5, "x", (0, 0, 10, 10), 9, 10, bytes(90))

    def test_blob_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="expected"):
            cv.Instance(0, 0.5, "x", (0, 0, 10, 10), 10, 10, bytes(99))

    def test_out_of_range_blob_offset_is_rejected(self):
        entry = {
            "instance_id": 0, "score": 0.5, "label": "", "bbox": [0, 0, 4, 4],
            "mask_width": 4, "mask_height": 4, "blob_offset": 8, "blob_length": 16,
        }
        with pytest.raises(ValueError, match="overruns"):
            cv.Instance.from_header(entry, bytes(16))

    def test_area(self):
        inst = cv.Instance(0, 1.0, "", (10, 20, 40, 60), 30, 40, bytes(1200))
        assert inst.area == 1200

    def test_colours_are_distinct_and_stable(self):
        colors = [cv.instance_color(i) for i in range(12)]
        assert colors[3] == cv.instance_color(3)          # stable
        assert len(set(colors)) == len(colors)            # distinct
        for r, g, b in colors:
            assert 0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0
            assert max(r, g, b) > 0.5                     # always legible on a photo


# --------------------------------------------------------------------------- #
# client-side thresholding -- the reason the wire format is soft
# --------------------------------------------------------------------------- #
class TestThresholding:
    def test_alpha_table_is_a_ramp_around_the_threshold(self):
        table = cv.alpha_table(128, softness=10)
        assert len(table) == 256
        assert table[0] == 0 and table[100] == 0
        assert table[255] == 255 and table[200] == 255
        assert table[128] == pytest.approx(127, abs=2)
        assert all(table[i] <= table[i + 1] for i in range(255))   # monotonic

    def test_alpha_table_with_zero_softness_is_a_hard_cut(self):
        table = cv.alpha_table(200, softness=0)
        assert table[199] == 0
        assert table[200] == 255

    def test_alpha_table_clamps_the_threshold(self):
        assert cv.alpha_table(0, 0)[0] == 0        # threshold floors at 1
        assert cv.alpha_table(9999, 0)[255] == 255

    def test_binarize_matches_the_documented_comparison(self):
        """API §8.3: ``inside = value >= T``."""
        mask = bytes(range(256))
        out = cv.binarize(mask, 128)
        assert all(out[v] == (1 if v >= 128 else 0) for v in range(256))

    def test_mask_sample_is_zero_outside_the_crop(self):
        mask = bytes([10, 20, 30, 40])
        assert cv.mask_sample(mask, 2, 2, 1, 1) == 40
        assert cv.mask_sample(mask, 2, 2, -1, 0) == 0
        assert cv.mask_sample(mask, 2, 2, 2, 0) == 0
        assert cv.mask_sample(mask, 2, 2, 0, 2) == 0

    def test_instance_sample_uses_model_canvas_coordinates(self):
        bbox, mask = soft_disc(1008, 1008, 400.0, 400.0, 50.0)
        inst = cv.Instance(0, 0.9, "", bbox, bbox[2] - bbox[0], bbox[3] - bbox[1], mask)
        assert cv.instance_sample(inst, 400, 400) == 255      # centre
        assert cv.instance_sample(inst, 400, 400 - 50) == pytest.approx(128, abs=6)
        assert cv.instance_sample(inst, 900, 900) == 0        # far outside the crop

    def test_score_filter(self):
        insts = [cv.Instance(n, s, "", (0, 0, 2, 2), 2, 2, bytes(4))
                 for n, s in enumerate((0.9, 0.5, 0.15))]
        assert [i.instance_id for i in cv.filter_by_score(insts, 0.4)] == [0, 1]
        assert cv.filter_by_score(insts, 0.95) == []
        assert len(cv.filter_by_score(insts, 0.0)) == 3


# --------------------------------------------------------------------------- #
# hit-testing
# --------------------------------------------------------------------------- #
class TestHitTest:
    @staticmethod
    def _disc(instance_id, cx, cy, r, score=0.9):
        bbox, mask = soft_disc(1008, 1008, cx, cy, r)
        return cv.Instance(instance_id, score, "", bbox,
                           bbox[2] - bbox[0], bbox[3] - bbox[1], mask)

    def test_hit_and_miss(self):
        inst = self._disc(0, 400.0, 400.0, 60.0)
        assert cv.hit_test([inst], 400, 400) is inst
        assert cv.hit_test([inst], 10, 10) is None

    def test_threshold_shrinks_the_hittable_area(self):
        """Raising the threshold must exclude the soft falloff -- no round trip."""
        inst = self._disc(0, 400.0, 400.0, 60.0)
        edge = (400, 400 - 66)                                 # inside the falloff
        assert 0 < cv.instance_sample(inst, *edge) < 255
        assert cv.hit_test([inst], *edge, mask_threshold=8) is inst
        assert cv.hit_test([inst], *edge, mask_threshold=250) is None

    def test_smallest_instance_wins_when_nested(self):
        """Clicking a small instance inside a big one must select the small one."""
        big = self._disc(0, 400.0, 400.0, 200.0, score=0.99)
        small = self._disc(1, 400.0, 400.0, 30.0, score=0.40)
        assert cv.hit_test([big, small], 400, 400).instance_id == 1
        assert cv.hit_test([small, big], 400, 400).instance_id == 1   # order-independent
        assert cv.hit_test([big, small], 400, 300).instance_id == 0   # outside the small one

    def test_score_filter_applies_to_hit_testing(self):
        inst = self._disc(0, 400.0, 400.0, 60.0, score=0.2)
        assert cv.hit_test([inst], 400, 400, score_threshold=0.5) is None
        assert cv.hit_test([inst], 400, 400, score_threshold=0.1) is inst

    def test_hidden_instances_are_not_hittable(self):
        inst = self._disc(0, 400.0, 400.0, 60.0)
        inst.visible = False
        assert cv.hit_test([inst], 400, 400) is None
        assert cv.hit_test([inst], 400, 400, require_visible=False) is inst


# --------------------------------------------------------------------------- #
# contour tracing -- the marching-ants outline
# --------------------------------------------------------------------------- #
def raster(width, height, predicate):
    return bytes(1 if predicate(x, y) else 0 for y in range(height) for x in range(width))


class TestContours:
    def test_filled_rectangle_gives_one_loop_with_the_right_extent(self):
        binary = raster(20, 20, lambda x, y: 5 <= x < 15 and 4 <= y < 14)
        contours = cv.trace_contours(binary, 20, 20)
        assert len(contours) == 1
        loop = contours[0]
        xs = [p[0] for p in loop]
        ys = [p[1] for p in loop]
        assert (min(xs), max(xs)) == (5, 14)
        assert (min(ys), max(ys)) == (4, 13)
        # every boundary pixel exactly once: 2*(w + h) - 4
        assert len(set(loop)) == 2 * (10 + 10) - 4

    def test_two_disjoint_blobs_give_two_loops(self):
        binary = raster(
            40, 20,
            lambda x, y: (2 <= x < 8 and 2 <= y < 8) or (25 <= x < 33 and 10 <= y < 17),
        )
        contours = cv.trace_contours(binary, 40, 20)
        assert len(contours) == 2

    def test_a_hole_produces_its_own_loop(self):
        """Ants must run around holes too, so the inner boundary is traced."""
        def ring(x, y):
            d = math.hypot(x - 20.0, y - 20.0)
            return 6.0 < d < 15.0

        binary = raster(40, 40, ring)
        contours = cv.trace_contours(binary, 40, 40)
        assert len(contours) == 2
        outer, inner = sorted(contours, key=len, reverse=True)
        assert len(outer) > len(inner)

    def test_isolated_pixels_are_discarded(self):
        binary = raster(10, 10, lambda x, y: (x, y) == (5, 5))
        assert cv.trace_contours(binary, 10, 10) == []

    def test_empty_and_degenerate_inputs(self):
        assert cv.trace_contours(b"", 0, 0) == []
        assert cv.trace_contours(bytes(9), 3, 3) == []
        assert cv.trace_contours(bytes(4), 10, 10) == []      # buffer too small

    def test_full_frame_mask_is_traced(self):
        binary = raster(12, 12, lambda x, y: True)
        contours = cv.trace_contours(binary, 12, 12)
        assert len(contours) == 1
        assert len(contours[0]) == 2 * (12 + 12) - 4

    def test_speckled_mask_terminates_in_bounded_steps(self):
        """Regression: Jacob's criterion never fired on speckled masks and the
        tracer ran to its step cap (3 M steps on a real image), which is what
        froze the window on "person".  A state can recur at most once, so no
        loop may be longer than eight visits per distinct pixel."""
        import random
        import time

        rng = random.Random(3)
        w, h = 160, 120

        def speckled(x, y):
            body = 40 <= x < 120 and 20 <= y < 100
            return body != (rng.random() < 0.08)

        binary = raster(w, h, speckled)
        foreground = binary.count(1)
        start = time.monotonic()
        contours = cv.trace_contours(binary, w, h, max_contours=10 ** 6, max_points=10 ** 9)
        elapsed = time.monotonic() - start
        assert contours
        for loop in contours:
            assert len(loop) <= 8 * len(set(loop))
        assert sum(map(len, contours)) <= 8 * foreground
        assert elapsed < 1.0, "tracing took %.2fs" % elapsed

    def test_a_loop_that_reenters_its_start_from_another_side_still_closes(self):
        """Two blobs touching at one diagonal: the walk returns to the seed
        pixel with a different backtrack than it started with."""
        binary = raster(12, 12, lambda x, y: (2 <= x < 6 and 2 <= y < 6) or (6 <= x < 10 and 6 <= y < 10))
        contours = cv.trace_contours(binary, 12, 12)
        total = sum(map(len, contours))
        assert 0 < total <= 8 * binary.count(1)
        for loop in contours:
            assert len(loop) <= 8 * len(set(loop))

    def test_simplify_collapses_a_straight_line(self):
        line = [(float(x), 0.0) for x in range(50)]
        assert cv.simplify_polyline(line, 0.5) == [(0.0, 0.0), (49.0, 0.0)]

    def test_simplify_keeps_corners(self):
        square = ([(float(x), 0.0) for x in range(21)]
                  + [(20.0, float(y)) for y in range(1, 21)])
        out = cv.simplify_polyline(square, 0.75)
        assert (20.0, 0.0) in out
        assert len(out) == 3
        assert out[0] == (0.0, 0.0) and out[-1] == (20.0, 20.0)

    def test_simplify_is_a_no_op_below_three_points(self):
        assert cv.simplify_polyline([(0.0, 0.0), (1.0, 1.0)], 1.0) == [(0.0, 0.0), (1.0, 1.0)]

    def test_polygons_follow_the_threshold(self):
        """A higher threshold must enclose a strictly smaller shape."""
        bbox, mask = soft_disc(1008, 1008, 300.0, 300.0, 80.0, feather=0.4)
        mw, mh = bbox[2] - bbox[0], bbox[3] - bbox[1]

        def extent(threshold):
            polys = cv.contour_polygons(mask, mw, mh, threshold, epsilon=0.4)
            assert polys, "threshold %d produced no contour" % threshold
            xs = [p[0] for poly in polys for p in poly]
            return max(xs) - min(xs)

        wide = extent(40)
        nominal = extent(128)
        tight = extent(230)
        assert wide > nominal > tight

    def test_polygons_are_simplified(self):
        bbox, mask = soft_disc(1008, 1008, 300.0, 300.0, 90.0)
        mw, mh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        raw = cv.trace_contours(cv.binarize(mask, 128), mw, mh)
        simplified = cv.contour_polygons(mask, mw, mh, 128)
        assert sum(len(p) for p in simplified) < sum(len(c) for c in raw)


# --------------------------------------------------------------------------- #
# cairo surfaces
# --------------------------------------------------------------------------- #
try:
    import cairo
except ImportError:  # pragma: no cover - pycairo ships with PyGObject
    cairo = None

needs_cairo = pytest.mark.skipif(cairo is None, reason="pycairo is unavailable")


@needs_cairo
class TestSurfaces:
    def test_rgb_surface_geometry_and_pixels(self):
        width, height = 7, 5
        data = bytearray(width * height * 3)
        # a single known pixel at (3, 2): pure orange
        idx = (2 * width + 3) * 3
        data[idx:idx + 3] = bytes((255, 128, 0))
        surface = cv.surface_from_rgb(bytes(data), width, height)
        assert surface.get_width() == width
        assert surface.get_height() == height
        assert surface.get_format() == cairo.FORMAT_RGB24
        surface.flush()
        stride = surface.get_stride()
        buf = bytes(surface.get_data())
        off = 2 * stride + 3 * 4
        b, g, r = buf[off], buf[off + 1], buf[off + 2]
        assert (r, g, b) == (255, 128, 0)

    def test_rgb_surface_rejects_a_wrong_sized_buffer(self):
        with pytest.raises(ValueError, match="expected"):
            cv.surface_from_rgb(bytes(10), 4, 4)

    def test_rgb_surface_rejects_degenerate_dimensions(self):
        with pytest.raises(ValueError):
            cv.surface_from_rgb(b"", 0, 4)

    def test_alpha_surface_carries_the_thresholded_mask(self):
        mask = bytes([0, 64, 128, 200, 255, 10])
        table = cv.alpha_table(128, softness=0)
        surface = cv.alpha_surface_from_mask(mask, 3, 2, table)
        assert surface.get_format() == cairo.FORMAT_A8
        assert (surface.get_width(), surface.get_height()) == (3, 2)
        surface.flush()
        stride = surface.get_stride()
        buf = bytes(surface.get_data())
        row0 = [buf[0], buf[1], buf[2]]
        row1 = [buf[stride], buf[stride + 1], buf[stride + 2]]
        assert row0 == [0, 0, 255]      # 0, 64 below 128; 128 is inside
        assert row1 == [255, 255, 0]

    def test_alpha_surface_of_an_empty_crop_is_none(self):
        assert cv.alpha_surface_from_mask(b"", 0, 0, cv.alpha_table(128)) is None


# --------------------------------------------------------------------------- #
# the widget itself
# --------------------------------------------------------------------------- #
pytest_gtk = pytest.mark.needs_gtk

IMAGE_W, IMAGE_H = 320, 240
AREA_W, AREA_H = 640, 480


@pytest.fixture
def gdk():
    pytest.importorskip("gi")
    import gi

    gi.require_version("Gtk", "3.0")
    gi.require_version("Gdk", "3.0")
    from gi.repository import Gdk

    return Gdk


@pytest.fixture
def widget(gdk):
    """A real ``Sam3Canvas`` with a real allocation, image and result.

    GTK ignores ``size_allocate`` on an invisible widget, so the widget is shown
    first.  Nothing is realized and no window is mapped -- ``do_draw`` only needs
    an allocation, which is what makes this runnable in CI.
    """
    if not cv.GTK_AVAILABLE:
        pytest.skip("GTK is unavailable")
    canvas = cv.Sam3Canvas()
    canvas.set_image(rgb_gradient(IMAGE_W, IMAGE_H), IMAGE_W, IMAGE_H)
    canvas.show()
    alloc = gdk.Rectangle()
    alloc.x = alloc.y = 0
    alloc.width, alloc.height = AREA_W, AREA_H
    canvas.size_allocate(alloc)
    try:
        yield canvas
    finally:
        canvas.destroy()


def render(canvas, width=AREA_W, height=AREA_H):
    """Drive the widget's real ``do_draw`` and return the resulting pixels."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    canvas.do_draw(cairo.Context(surface))
    surface.flush()
    return bytes(surface.get_data())


def press(gdk, canvas, x, y, button=1, state=0):
    ev = gdk.Event.new(gdk.EventType.BUTTON_PRESS)
    ev.button = button
    ev.x, ev.y = float(x), float(y)
    ev.state = state
    canvas.do_button_press_event(ev)


def release(gdk, canvas, x, y, button=1, state=0):
    ev = gdk.Event.new(gdk.EventType.BUTTON_RELEASE)
    ev.button = button
    ev.x, ev.y = float(x), float(y)
    ev.state = state
    canvas.do_button_release_event(ev)


def click(gdk, canvas, x, y, button=1, state=0):
    press(gdk, canvas, x, y, button, state)
    release(gdk, canvas, x, y, button, state)


def motion(gdk, canvas, x, y, state=0):
    ev = gdk.Event.new(gdk.EventType.MOTION_NOTIFY)
    ev.x, ev.y = float(x), float(y)
    ev.state = state
    canvas.do_motion_notify_event(ev)


def scroll(gdk, canvas, x, y, direction):
    ev = gdk.Event.new(gdk.EventType.SCROLL)
    ev.direction = direction
    ev.x, ev.y = float(x), float(y)
    canvas.do_scroll_event(ev)


def two_disc_frame(request_id="r-1"):
    """One big low-scoring disc and one small high-scoring disc inside it."""
    return make_frame(
        [(500.0, 500.0, 260.0, 0.35), (500.0, 500.0, 60.0, 0.92)],
        image_wh=(IMAGE_W, IMAGE_H),
        request_id=request_id,
    )


@pytest_gtk
@needs_cairo
class TestWidgetRendering:
    def test_fits_the_image_on_allocation(self, widget):
        # a small image is shown at 100 %, never blown up to fill the widget
        assert widget.view.zoom == pytest.approx(1.0)
        # ...and centred
        x0, _ = widget.image_to_widget_point(0, 0)
        x1, _ = widget.image_to_widget_point(IMAGE_W, 0)
        assert x0 + x1 == pytest.approx(AREA_W)

    def test_a_large_image_is_scaled_down_to_fit(self, gdk):
        canvas = cv.Sam3Canvas()
        canvas.set_image(rgb_gradient(1008, 672), 1008, 672)
        canvas.show()
        alloc = gdk.Rectangle()
        alloc.x = alloc.y = 0
        alloc.width, alloc.height = AREA_W, AREA_H
        canvas.size_allocate(alloc)
        try:
            assert canvas.view.zoom == pytest.approx(
                min((AREA_W - 12) / 1008.0, (AREA_H - 12) / 672.0)
            )
        finally:
            canvas.destroy()

    def test_draw_paints_the_image(self, widget):
        blank = cv.Sam3Canvas()
        blank.show()
        try:
            painted = render(widget)
            assert len(set(painted[::4093])) > 1, "the render is a flat fill"
        finally:
            blank.destroy()

    def test_masks_change_the_render(self, widget):
        before = render(widget)
        widget.set_result_frame(two_disc_frame())
        after = render(widget)
        assert before != after
        assert len(widget.instances) == 2

    def test_mask_threshold_re_renders_locally(self, widget):
        """The whole point of the soft wire format: no network, visible change."""
        widget.set_result_frame(two_disc_frame())
        widget.set_mask_threshold(30)
        loose = render(widget)
        widget.set_mask_threshold(240)
        tight = render(widget)
        assert loose != tight
        assert widget.mask_threshold == 240

    def test_alpha_surfaces_are_cached_until_the_threshold_moves(self, widget):
        widget.set_result_frame(two_disc_frame())
        render(widget)
        first = widget._alpha_cache[0][2]
        render(widget)
        assert widget._alpha_cache[0][2] is first, "surface rebuilt with no change"
        widget.set_mask_threshold(200)
        assert widget._alpha_cache == {}
        render(widget)
        assert widget._alpha_cache[0][2] is not first

    def test_score_threshold_filters_locally(self, widget):
        widget.set_result_frame(two_disc_frame())
        assert len(widget.visible_instances) == 2
        widget.set_score_threshold(0.5)
        assert [i.instance_id for i in widget.visible_instances] == [1]
        widget.set_score_threshold(0.99)
        assert widget.visible_instances == []
        render(widget)  # must not raise with nothing to draw

    def test_overlay_opacity_changes_the_render(self, widget):
        widget.set_result_frame(two_disc_frame())
        widget.set_overlay_opacity(0.1)
        faint = render(widget)
        widget.set_overlay_opacity(1.0)
        solid = render(widget)
        assert faint != solid

    def test_hiding_an_instance_changes_the_render(self, widget):
        widget.set_result_frame(two_disc_frame())
        shown = render(widget)
        widget.set_instance_visible(0, False)
        widget.set_instance_visible(1, False)
        hidden = render(widget)
        assert shown != hidden

    def test_clear_result_returns_to_the_bare_image(self, widget):
        bare = render(widget)
        widget.set_result_frame(two_disc_frame())
        assert render(widget) != bare
        widget.clear_result()
        assert widget.instances == []
        assert widget.active_instance_id == -1

    def test_malformed_instances_are_skipped_not_fatal(self, widget):
        payload = two_disc_frame()
        header, blob = cv.decode_frame(payload)
        header["instances"][0]["mask_width"] += 3      # violates the API §8.2 invariant
        widget.set_result(header, blob)
        assert [i.instance_id for i in widget.instances] == [1]
        render(widget)

    def test_render_survives_a_letterboxing_transform(self, widget):
        """No hardcoded squash anywhere in the draw path (API §5)."""
        header, blob = cv.decode_frame(two_disc_frame())
        header["canvas_from_image"] = {
            "scale_x": 1.5, "scale_y": 1.5, "offset_x": 60.0, "offset_y": 12.0,
        }
        widget.set_result(header, blob)
        assert widget.canvas_from_image.offset_x == 60.0
        cx, cy = widget.widget_to_canvas_point(*widget.image_to_widget_point(10.0, 20.0))
        assert (cx, cy) == pytest.approx((10.0 * 1.5 + 60.0, 20.0 * 1.5 + 12.0))
        render(widget)

    def test_busy_chrome_renders(self, widget):
        idle = render(widget)
        widget.set_busy(True, "encoding", 0.42)
        assert render(widget) != idle
        widget.set_status_text("hello")
        widget.set_busy(False)
        render(widget)


@pytest_gtk
@needs_cairo
class TestWidgetInteraction:
    def test_left_click_places_a_positive_point_in_image_coordinates(self, widget, gdk):
        widget.set_interaction_mode("points")
        seen = []
        widget.connect("point-added", lambda _w, x, y, label: seen.append((x, y, label)))
        wx, wy = widget.image_to_widget_point(100.0, 60.0)
        click(gdk, widget, wx, wy, button=1)
        assert len(seen) == 1
        assert seen[0][0] == pytest.approx(100.0)
        assert seen[0][1] == pytest.approx(60.0)
        assert seen[0][2] == 1
        assert widget.points == [(pytest.approx(100.0), pytest.approx(60.0), 1)]

    def test_right_click_and_shift_click_place_negative_points(self, widget, gdk):
        widget.set_interaction_mode("points")
        wx, wy = widget.image_to_widget_point(50.0, 50.0)
        click(gdk, widget, wx, wy, button=3)
        click(gdk, widget, wx, wy, button=1, state=gdk.ModifierType.SHIFT_MASK)
        assert [p[2] for p in widget.points] == [0, 0]

    def test_clicks_outside_the_image_are_ignored(self, widget, gdk):
        widget.set_interaction_mode("points")
        click(gdk, widget, 2.0, 2.0)                    # in the margin, not the image
        assert widget.points == []

    def test_a_drag_is_not_a_click(self, widget, gdk):
        widget.set_interaction_mode("points")
        wx, wy = widget.image_to_widget_point(100.0, 60.0)
        press(gdk, widget, wx, wy)
        release(gdk, widget, wx + 40, wy + 5)
        assert widget.points == []

    def test_points_can_be_undone_and_cleared(self, widget, gdk):
        widget.set_interaction_mode("points")
        changes = []
        widget.connect("points-changed", lambda _w: changes.append(1))
        wx, wy = widget.image_to_widget_point(100.0, 60.0)
        click(gdk, widget, wx, wy)
        click(gdk, widget, wx + 10, wy)
        assert len(widget.points) == 2
        assert widget.remove_last_point() is True
        assert len(widget.points) == 1
        widget.clear_points()
        assert widget.points == []
        assert widget.remove_last_point() is False
        assert len(changes) == 4

    def test_clicking_an_instance_unticks_it(self, widget, gdk):
        """A left click changes the *tick* -- the flag the dialog's checkbox
        shows and Apply uses -- not a private highlight the list never saw."""
        widget.set_result_frame(two_disc_frame())
        seen = []
        widget.connect("instance-visibility-changed", lambda _w, i, v: seen.append((i, v)))
        toggles = []
        widget.connect("instance-toggled", lambda _w, i, s: toggles.append((i, s)))
        wx, wy = widget.canvas_to_widget_point(500.0, 500.0)
        click(gdk, widget, wx, wy)
        assert widget.get_instance(1).visible is False       # the small disc wins
        assert widget.active_instance_id == 1
        assert [i.instance_id for i in widget.visible_instances] == [0]
        # The small disc is hidden now, so the same spot hits the big one.
        click(gdk, widget, wx, wy)
        assert widget.get_instance(0).visible is False
        assert seen == [(1, False), (0, False)]
        assert toggles == [], "clicks do not touch the highlight flag any more"
        assert widget.selected_instance_ids() == []

    def test_right_click_toggles_visibility_in_select_mode(self, widget, gdk):
        widget.set_result_frame(two_disc_frame())
        seen = []
        widget.connect("instance-visibility-changed", lambda _w, i, v: seen.append((i, v)))
        wx, wy = widget.canvas_to_widget_point(500.0, 500.0)
        click(gdk, widget, wx, wy, button=3)
        assert widget.get_instance(1).visible is False
        assert seen == [(1, False)]

    def test_hover_emits_and_clears(self, widget, gdk):
        widget.set_result_frame(two_disc_frame())
        seen = []
        widget.connect("instance-hovered", lambda _w, i: seen.append(i))
        wx, wy = widget.canvas_to_widget_point(500.0, 500.0)
        motion(gdk, widget, wx, wy)
        assert widget.hovered_instance_id == 1
        far_x, far_y = widget.canvas_to_widget_point(20.0, 20.0)
        motion(gdk, widget, far_x, far_y)
        assert widget.hovered_instance_id == -1
        assert seen == [1, -1]

    def test_clicking_empty_space_deactivates(self, widget, gdk):
        widget.set_result_frame(two_disc_frame())
        activated = []
        widget.connect("instance-activated", lambda _w, i: activated.append(i))
        far_x, far_y = widget.canvas_to_widget_point(20.0, 20.0)
        click(gdk, widget, far_x, far_y)
        assert widget.active_instance_id == -1
        assert activated == [-1]

    def test_scroll_zooms_around_the_pointer(self, widget, gdk):
        # zoom in first: while the whole image fits, the view is deliberately
        # kept centred (view_clamp), which would mask the pinning behaviour.
        widget.zoom_to(4.0)
        pointer = (200.0, 150.0)
        before_zoom = widget.view.zoom
        anchor = widget.widget_to_image_point(*pointer)
        scroll(gdk, widget, pointer[0], pointer[1], gdk.ScrollDirection.UP)
        assert widget.view.zoom > before_zoom
        after = widget.widget_to_image_point(*pointer)
        assert after == pytest.approx(anchor)
        scroll(gdk, widget, pointer[0], pointer[1], gdk.ScrollDirection.DOWN)
        assert widget.view.zoom == pytest.approx(before_zoom)

    def test_middle_drag_pans(self, widget, gdk):
        widget.zoom_to(4.0)                        # zoom in so panning is not clamped away
        before = widget.view
        press(gdk, widget, 300.0, 300.0, button=2)
        motion(gdk, widget, 260.0, 285.0)
        release(gdk, widget, 260.0, 285.0, button=2)
        assert widget.view.off_x == pytest.approx(before.off_x - 40.0)
        assert widget.view.off_y == pytest.approx(before.off_y - 15.0)

    def test_ctrl_drag_emits_a_box_in_image_coordinates(self, widget, gdk):
        boxes = []
        widget.connect("box-drawn", lambda _w, *b: boxes.append(b))
        ax, ay = widget.image_to_widget_point(200.0, 150.0)
        bx, by = widget.image_to_widget_point(40.0, 30.0)
        press(gdk, widget, ax, ay, button=1, state=gdk.ModifierType.CONTROL_MASK)
        motion(gdk, widget, bx, by)
        release(gdk, widget, bx, by, button=1, state=gdk.ModifierType.CONTROL_MASK)
        assert len(boxes) == 1
        assert boxes[0] == pytest.approx((40.0, 30.0, 200.0, 150.0))   # normalised

    def test_a_box_dragged_past_the_edge_is_clipped_to_the_image(self, widget, gdk):
        """A box prompt is in uploaded-image pixels; the part hanging off the
        image names pixels the daemon never received."""
        boxes = []
        widget.connect("box-drawn", lambda _w, *b: boxes.append(b))
        ax, ay = widget.image_to_widget_point(20.0, 30.0)
        bx, by = widget.image_to_widget_point(-40.0, IMAGE_H + 50.0)
        press(gdk, widget, ax, ay, button=1, state=gdk.ModifierType.CONTROL_MASK)
        motion(gdk, widget, bx, by)
        release(gdk, widget, bx, by, button=1, state=gdk.ModifierType.CONTROL_MASK)
        assert boxes == [pytest.approx((0.0, 30.0, 20.0, float(IMAGE_H)))]

    def test_a_box_with_no_area_inside_the_image_is_dropped(self, widget, gdk):
        boxes = []
        widget.connect("box-drawn", lambda _w, *b: boxes.append(b))
        ax, ay = widget.image_to_widget_point(-60.0, -60.0)
        bx, by = widget.image_to_widget_point(-10.0, 100.0)
        press(gdk, widget, ax, ay, button=1, state=gdk.ModifierType.CONTROL_MASK)
        motion(gdk, widget, bx, by)
        release(gdk, widget, bx, by, button=1, state=gdk.ModifierType.CONTROL_MASK)
        assert boxes == []

    def test_losing_focus_lets_go_of_space(self, widget, gdk):
        """Space released in another window never reaches the canvas; held
        "down" for ever, every left-drag afterwards would pan."""
        ev = gdk.Event.new(gdk.EventType.KEY_PRESS)
        ev.keyval = gdk.KEY_space
        widget.do_key_press_event(ev)
        assert widget._space_down
        widget.do_focus_out_event(gdk.Event.new(gdk.EventType.FOCUS_CHANGE))
        assert not widget._space_down
        before = widget.view
        points = []
        widget.connect("point-added", lambda _w, *p: points.append(p))
        widget.set_interaction_mode("points")
        x, y = widget.image_to_widget_point(50.0, 50.0)
        press(gdk, widget, x, y, button=1)
        release(gdk, widget, x, y, button=1)
        assert widget.view == before and len(points) == 1, "a click, not a pan"

    def test_keyboard_zoom_and_threshold(self, widget, gdk):
        thresholds = []
        widget.connect("threshold-changed", lambda _w, m, s: thresholds.append(m))

        def key(keyval):
            ev = gdk.Event.new(gdk.EventType.KEY_PRESS)
            ev.keyval = keyval
            return widget.do_key_press_event(ev)

        fitted = widget.view.zoom
        assert key(gdk.KEY_1) is True
        assert widget.view.zoom == pytest.approx(1.0)
        assert key(gdk.KEY_f) is True
        assert widget.view.zoom == pytest.approx(fitted)
        assert key(gdk.KEY_bracketright) is True
        assert widget.mask_threshold == cv.DEFAULT_MASK_THRESHOLD + 8
        assert key(gdk.KEY_bracketleft) is True
        assert thresholds == [136, 128]

    def test_interaction_mode_validation(self, widget):
        widget.set_interaction_mode("points")
        assert widget.interaction_mode == "points"
        with pytest.raises(ValueError):
            widget.set_interaction_mode("nonsense")

    def test_request_id_is_exposed_for_the_staleness_rule(self, widget):
        """API §10: the dialog drops results whose request_id is not the latest."""
        widget.set_result_frame(two_disc_frame(request_id="r-000042"))
        assert widget.request_id == "r-000042"


@pytest_gtk
@needs_cairo
class TestAntsLifecycle:
    def test_animation_timer_runs_only_while_something_is_active(self, widget):
        """The ants timer must not tick when there is nothing to animate."""
        assert widget._ants_source == 0
        widget.set_result_frame(two_disc_frame())
        assert widget._ants_source != 0              # instance 0 becomes active
        widget.set_show_ants(False)
        assert widget._ants_source == 0
        widget.set_show_ants(True)
        assert widget._ants_source != 0
        widget.clear_result()
        assert widget._ants_source == 0

    def test_phase_advances_and_wraps(self, widget):
        widget.set_result_frame(two_disc_frame())
        phases = []
        for _ in range(10):
            widget._tick_ants()
            phases.append(widget._ants_phase)
        assert phases[0] == 1.0
        assert max(phases) < 8.0
        assert 0.0 in phases                          # wrapped


@needs_cairo
class TestThresholdPerformance:
    """Backs the claim that re-thresholding is instant enough to be sliderable."""

    def test_full_canvas_mask_rethresholds_quickly(self):
        import time

        side = 1008
        mask = bytes((x * 7 + 13) & 0xFF for x in range(side * side))
        start = time.monotonic()
        for threshold in (60, 100, 128, 180, 220):
            surface = cv.alpha_surface_from_mask(
                mask, side, side, cv.alpha_table(threshold)
            )
            assert surface is not None
        elapsed = time.monotonic() - start
        # Five full 1008x1008 re-thresholds; a pure-Python per-pixel loop would
        # take tens of seconds, which is exactly why translate() + A8 is used.
        assert elapsed < 1.0, "re-thresholding took %.2fs" % elapsed


@pytest_gtk
@needs_cairo
class TestOverlayCache:
    """The composited-overlay cache behind a responsive many-instance window.

    A frame used to re-composite every visible mask; with 64 person-sized
    masks that was slower than the marching-ants tick and the window froze.
    """

    @staticmethod
    def big_frame(count=64):
        # Person-sized: each disc spans ~60 % of the shorter side, and they overlap.
        r = 0.3 * min(IMAGE_W, IMAGE_H)
        discs = [(IMAGE_W * (0.15 + (i % 8) * 0.1), IMAGE_H * (0.2 + (i // 8) * 0.08),
                  r, 0.95 - i * 0.005) for i in range(count)]
        return make_frame(discs, image_wh=(IMAGE_W, IMAGE_H), canvas_wh=(IMAGE_W, IMAGE_H))

    def test_consecutive_frames_reuse_one_surface(self, widget):
        widget.set_result_frame(two_disc_frame())
        render(widget)
        first = widget._overlay_cache
        render(widget)
        assert widget._overlay_cache is first
        assert first[1] is not None

    def test_every_input_of_the_composite_invalidates_it(self, widget):
        widget.set_result_frame(two_disc_frame())
        render(widget)
        seen = [widget._overlay_cache[1]]

        def changed(what):
            render(widget)
            surface = widget._overlay_cache[1]
            assert surface is not None, "%s left nothing to composite" % what
            assert all(surface is not s for s in seen), "%s did not rebuild the overlay" % what
            seen.append(surface)

        # Hide the low-scoring disc first so the high-scoring one survives the
        # score step below: the composite of an empty set is None, not a surface.
        low = min(widget.instances, key=lambda i: i.score)
        widget.set_instance_visible(low.instance_id, False, notify=False); changed("visibility")
        widget.set_score_threshold(0.5); changed("score threshold")
        widget.set_mask_threshold(160); changed("mask threshold")
        widget.set_overlay_opacity(0.2); changed("opacity")
        widget.set_view(cv.view_zoom_at(widget.view, 2.0, AREA_W / 2, AREA_H / 2)); changed("zoom")
        widget.set_result_frame(two_disc_frame("r-2")); changed("new result")

    def test_hover_and_active_do_not_rebuild_it(self, widget):
        widget.set_result_frame(two_disc_frame())
        render(widget)
        surface = widget._overlay_cache[1]
        ids = [i.instance_id for i in widget.instances]
        widget.set_active_instance(ids[1])
        render(widget)
        assert widget._overlay_cache[1] is surface

    def test_hiding_an_instance_changes_the_pixels(self, widget):
        widget.set_result_frame(two_disc_frame())
        widget.set_score_threshold(0.0)
        before = render(widget)
        ids = [i.instance_id for i in widget.instances]
        widget.set_instance_visible(ids[0], False, notify=False)
        assert render(widget) != before

    def test_the_overlay_is_rendered_at_the_screens_resolution(self, widget, monkeypatch):
        """At logical size the composite is scaled up on a HiDPI screen and
        every mask edge goes soft."""
        widget.set_result_frame(two_disc_frame())
        drawn = widget.visible_instances
        monkeypatch.setattr(widget, "get_scale_factor", lambda: 2)
        surface = widget._overlay_for(AREA_W, AREA_H, drawn)
        assert (surface.get_width(), surface.get_height()) == (AREA_W * 2, AREA_H * 2)
        assert surface.get_device_scale() == (2.0, 2.0)
        monkeypatch.setattr(widget, "get_scale_factor", lambda: 1)
        surface = widget._overlay_for(AREA_W, AREA_H, drawn)
        assert (surface.get_width(), surface.get_height()) == (AREA_W, AREA_H)
        render(widget)

    def test_sixty_four_large_masks_draw_within_the_ants_tick(self, widget):
        import time

        widget.set_result_frame(self.big_frame())
        widget.set_score_threshold(0.0)
        render(widget)                      # builds the composite once
        start = time.monotonic()
        for _ in range(10):
            render(widget)
        per_frame = (time.monotonic() - start) / 10.0
        # Uncached this measured ~144 ms per frame; the ants tick is 90 ms.
        # Cached it is a single surface paint plus the highlights.
        assert per_frame < 0.06, "%.0f ms per frame" % (per_frame * 1000)


class TestHarness:
    """``tools/canvas_harness.py`` is a deliverable, so it gets smoke-tested too."""

    @staticmethod
    def _load():
        import importlib.util
        import pathlib

        path = (pathlib.Path(__file__).resolve().parents[2] / "tools" / "canvas_harness.py")
        pytest.importorskip("gi")
        if not cv.GTK_AVAILABLE:
            pytest.skip("GTK is unavailable")
        spec = importlib.util.spec_from_file_location("canvas_harness", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @pytest.mark.needs_gtk
    def test_offline_engine_emits_conforming_frames(self):
        harness = self._load()
        engine = harness.OfflineEngine(640, 480)
        payload = engine.text("yellow school bus", "r-000001")
        header, blob = cv.decode_frame(payload)

        assert header["engine"] == "pcs"
        assert header["mask_encoding"] == "u8_soft"
        assert header["request_id"] == "r-000001"
        assert 1 <= len(header["instances"]) <= 8
        offset = 0
        scores = []
        for entry in header["instances"]:
            x0, y0, x1, y1 = entry["bbox"]
            assert 0 <= x0 < x1 <= header["model_canvas"]["width"]
            assert 0 <= y0 < y1 <= header["model_canvas"]["height"]
            assert entry["mask_width"] == x1 - x0
            assert entry["mask_height"] == y1 - y0
            assert entry["blob_length"] == entry["mask_width"] * entry["mask_height"]
            assert entry["blob_offset"] == offset      # tightly packed, in order
            offset += entry["blob_length"]
            scores.append(entry["score"])
        assert offset == len(blob) == header["blob_length"]
        assert scores == sorted(scores, reverse=True)
        assert all(0.1 <= s <= 1.0 for s in scores)

    @pytest.mark.needs_gtk
    def test_offline_masks_are_genuinely_soft(self):
        """A flat 255 rectangle would hide every client-side threshold bug."""
        harness = self._load()
        engine = harness.OfflineEngine(640, 480)
        _header, blob = cv.decode_frame(engine.text("red car", "r-1"))
        values = set(blob)
        assert 0 in values and 255 in values
        assert len([v for v in values if 0 < v < 255]) > 8

    @pytest.mark.needs_gtk
    def test_offline_engine_is_deterministic_and_prompt_sensitive(self):
        harness = self._load()
        a = harness.OfflineEngine(640, 480).text("red car", "r-1")
        b = harness.OfflineEngine(640, 480).text("red car", "r-1")
        c = harness.OfflineEngine(640, 480).text("blue bicycle", "r-1")
        assert a == b
        assert a != c

    @pytest.mark.needs_gtk
    def test_offline_point_prompt_uses_uploaded_image_coordinates(self):
        harness = self._load()
        engine = harness.OfflineEngine(640, 480)
        header, _blob = cv.decode_frame(
            engine.points([(320.0, 240.0, 1)], "r-2", multimask=True)
        )
        assert header["engine"] == "pvs"
        assert len(header["instances"]) == 3            # multimask candidates
        aff = cv.Affine.from_dict(header["canvas_from_image"])
        cx, cy = aff.to_canvas(320.0, 240.0)
        best = header["instances"][0]["bbox"]
        assert best[0] < cx < best[2]
        assert best[1] < cy < best[3]

    @pytest.mark.needs_gtk
    def test_runtime_file_path_honours_the_documented_overrides(self, monkeypatch, tmp_path):
        harness = self._load()
        monkeypatch.setenv("SAM3D_RUNTIME_FILE", str(tmp_path / "custom.json"))
        assert harness.runtime_file_path() == str(tmp_path / "custom.json")
        monkeypatch.delenv("SAM3D_RUNTIME_FILE")
        monkeypatch.setenv("SAM3_GIMP_HOME", str(tmp_path))
        assert harness.runtime_file_path() == str(tmp_path / "runtime.json")

    @pytest.mark.needs_gtk
    def test_pixbuf_to_rgb_produces_the_upload_layout(self):
        harness = self._load()
        import gi

        gi.require_version("GdkPixbuf", "2.0")
        from gi.repository import GdkPixbuf, GLib

        width, height = 5, 3
        rgba = bytearray()
        for _ in range(width * height):
            rgba += bytes((200, 100, 50, 255))
        pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
            GLib.Bytes.new(bytes(rgba)), GdkPixbuf.Colorspace.RGB, True, 8,
            width, height, width * 4,
        )
        rgb = harness.pixbuf_to_rgb(pixbuf)
        assert len(rgb) == width * height * 3
        assert rgb[:3] == bytes((200, 100, 50))

    @pytest.mark.needs_gtk
    def test_synthetic_image_is_a_valid_upload_body(self):
        harness = self._load()
        rgb, width, height = harness.synthetic_image()
        assert len(rgb) == width * height * 3
        assert 16 <= width <= harness.MAX_SIDE and 16 <= height <= harness.MAX_SIDE



def test_dialog_teardown_is_bounded_and_guarded():
    """Source-level: release is joined with a bound, and late idle callbacks
    bail once the window has closed."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "plugin" / "sam3_gimp" / "ui"
           / "main_dialog.py").read_text(encoding="utf-8")
    assert "thread.join(1.5)" in src
    for name in ("_on_prompt_done", "_on_progress", "_on_session_ready"):
        # Anchor on "(self": _progress_callback holds a *closure* also named
        # _on_progress, and a looser split lands in it first.
        body = src.split("    def %s(self" % name)[1].split("\n    def ")[0]
        assert 'getattr(self, "_closed", False)' in body, name


class TestEventAccessorShapes:
    """PyGObject builds disagree on whether ``Gdk.Event.get_*`` returns
    ``(ok, value...)`` or the values alone.  The PyGObject inside GIMP 3.2 on
    Windows returns ``(x, y)`` from ``get_coords()``; the canvas unpacked three
    and every mouse movement raised.  Both shapes must be accepted."""

    @pytest.fixture(autouse=True)
    def _need_gtk(self):
        if not hasattr(cv, "_ev_unpack"):
            pytest.skip("canvas GTK half unavailable")

    def test_ok_prefixed_tuple(self):
        assert cv._ev_unpack((True, 3.0, 4.0), 2, (0.0, 0.0)) == (3.0, 4.0)
        assert cv._ev_unpack((True, 2), 1, (0,)) == (2,)

    def test_bare_values(self):
        assert cv._ev_unpack((3.0, 4.0), 2, (0.0, 0.0)) == (3.0, 4.0)
        assert cv._ev_unpack((2,), 1, (0,)) == (2,)
        assert cv._ev_unpack(2, 1, (0,)) == (2,)

    def test_failure_and_garbage_fall_back(self):
        assert cv._ev_unpack((False, 3.0, 4.0), 2, (0.0, 0.0)) == (0.0, 0.0)
        assert cv._ev_unpack((1.0, 2.0, 3.0, 4.0), 2, (0.0, 0.0)) == (0.0, 0.0)
        assert cv._ev_unpack(None, 2, (0.0, 0.0)) == (0.0, 0.0)

    def test_accessors_accept_both_shapes(self):
        class Ev:
            def __init__(self, shape):
                self.shape = shape

            def get_coords(self):
                return (True, 5.0, 6.0) if self.shape == 3 else (5.0, 6.0)

            def get_button(self):
                return (True, 1) if self.shape == 3 else 1

            def get_keyval(self):
                return (True, 65) if self.shape == 3 else 65

        for shape in (2, 3):
            assert cv._ev_coords(Ev(shape)) == (5.0, 6.0)
            assert cv._ev_button(Ev(shape)) == 1
            assert cv._ev_keyval(Ev(shape)) == 65



class TestKeptBoxes:
    @pytest.fixture(autouse=True)
    def _need_gtk(self):
        if getattr(cv, "Sam3Canvas", None) is None:
            pytest.skip("canvas GTK half unavailable")

    def test_boxes_are_normalised_and_cleared(self):
        widget = cv.Sam3Canvas()
        widget.set_boxes([(30, 40, 10, 20), (1, 2, 3, 4, 0)])
        assert widget.boxes == [(10.0, 20.0, 30.0, 40.0, 1), (1.0, 2.0, 3.0, 4.0, 0)]
        widget.clear_boxes()
        assert widget.boxes == []
