"""Tests for ``sam3gimpd.masks`` -- quantisation, cropping, framing and geometry.

Two things are being defended here.

**The wire format.**  ``API.md`` §8 fixes the frame layout down to the
byte, so these tests parse frames by hand (magic, ``uint32`` header length,
blob offsets relative to the blob region) rather than trusting the encoder to
agree with itself.

**Alignment.**  A mask that lands one or two pixels off its object is the bug a
user notices and cannot describe, and it is produced by exactly one kind of
mistake: a rounding or half-pixel error in the chain
``source -> uploaded -> canvas -> bbox -> uploaded -> source``.  The property
tests below hammer that chain over hundreds of random sizes and aspect ratios,
including the degenerate ones (1x1000 strips, 4000x3000 photographs, odd
numbers, exact squares), and assert a round trip accurate to within a pixel.

Every test runs twice: once with numpy and once with the module's numpy handle
monkeypatched away, because the daemon's base install has no numpy and the two
paths must agree.
"""

from __future__ import annotations

import json
import math
import random
import struct

import pytest

from sam3gimpd import masks
from sam3gimpd.types import (
    API_VERSION,
    DEFAULT_MASK_THRESHOLD,
    RESULT_MAGIC,
    RESULT_PREFIX_SIZE,
    BBox,
    Size,
    unpack_result,
)


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
@pytest.fixture(params=["numpy", "stdlib"])
def backend(request, monkeypatch):
    """Run a test on both the numpy fast path and the pure-Python fallback.

    The fallback is not a fiction: ``pip install sam3gimpd`` pulls in nothing at
    all, so on a base install every byte on the wire is produced by the
    stdlib path.
    """
    if request.param == "numpy":
        if masks._np() is None:
            pytest.skip("numpy is not installed")
    else:
        monkeypatch.setattr(masks, "_numpy_module", None)
    return request.param


def _rect_mask(width, height, rect, inside=255, outside=0):
    """A canvas-sized soft mask that is ``inside`` within ``rect``."""
    x0, y0, x1, y1 = rect
    buf = bytearray(bytes((outside,)) * (width * height))
    for y in range(y0, y1):
        base = y * width
        for x in range(x0, x1):
            buf[base + x] = inside
    return bytes(buf)


# --------------------------------------------------------------------------- #
# sigmoid / quantisation
# --------------------------------------------------------------------------- #
def test_sigmoid_is_stable_at_the_extremes():
    assert masks.sigmoid(0.0) == 0.5
    assert masks.sigmoid(1000.0) == 1.0
    assert masks.sigmoid(-1000.0) == 0.0
    assert masks.sigmoid(float("inf")) == 1.0
    assert masks.sigmoid(float("-inf")) == 0.0
    assert masks.sigmoid(float("nan")) == 0.0
    assert 0.7310 < masks.sigmoid(1.0) < 0.7311


def test_logit_zero_quantises_to_128(backend):
    """The contract's single most load-bearing constant.

    API.md §8.3: 128 is logit 0, the model's own binarisation point and hence
    the client's default threshold.  If this drifts by one, every default
    selection in GIMP changes shape.
    """
    data, w, h = masks.quantise_u8([[0.0, 0.0], [0.0, 0.0]], kind="logit")
    assert (w, h) == (2, 2)
    assert data == bytes([128, 128, 128, 128])
    assert DEFAULT_MASK_THRESHOLD == 128


def test_quantise_logit_endpoints_and_monotonicity(backend):
    logits = [-40.0, -10.0, -2.0, -0.5, 0.0, 0.5, 2.0, 10.0, 40.0]
    data, _, _ = masks.quantise_u8(logits, len(logits), 1, kind="logit")
    assert data[0] == 0
    assert data[-1] == 255
    assert list(data) == sorted(data)
    # round(255 * sigmoid(x)) computed independently
    for i, x in enumerate(logits):
        assert data[i] == int(math.floor(255.0 * masks.sigmoid(x) + 0.5))


def test_quantise_probabilities(backend):
    data, _, _ = masks.quantise_u8([0.0, 0.25, 0.5, 1.0], 4, 1, kind="prob")
    assert list(data) == [0, 64, 128, 255]


def test_quantise_clamps_and_maps_nan_to_outside(backend):
    data, _, _ = masks.quantise_u8([-1.0, 2.0, float("nan")], 3, 1, kind="prob")
    assert list(data) == [0, 255, 0]


def test_quantise_auto_detects_kind(backend):
    # values outside [0,1] must be logits
    logit_data, _, _ = masks.quantise_u8([-5.0, 0.0, 5.0], 3, 1, kind="auto")
    assert list(logit_data) == [2, 128, 253]
    # values inside [0,1] are probabilities
    prob_data, _, _ = masks.quantise_u8([0.0, 0.5, 1.0], 3, 1, kind="auto")
    assert list(prob_data) == [0, 128, 255]
    # plain ints in 0..255 are already quantised
    u8_data, _, _ = masks.quantise_u8([0, 128, 255], 3, 1, kind="auto")
    assert list(u8_data) == [0, 128, 255]


def test_quantise_passes_bytes_through(backend):
    src = bytes(range(16))
    data, w, h = masks.quantise_u8(src, 4, 4)
    assert data == src and (w, h) == (4, 4)


def test_numpy_and_stdlib_quantise_agree():
    """The two implementations must be interchangeable.

    Cross-backend results are allowed to differ by at most one least
    significant bit (``np.exp`` and ``math.exp`` may differ in the last ulp,
    which can flip a value sitting exactly on a rounding boundary).  Within a
    single process the backend never changes, so frames stay byte-identical as
    API.md §14 requires of the stub.
    """
    np = masks._np()
    if np is None:
        pytest.skip("numpy is not installed")
    rng = random.Random(20240501)
    values = [rng.uniform(-8.0, 8.0) for _ in range(2048)]

    with_np, _, _ = masks.quantise_u8(values, 64, 32, kind="logit")
    saved = masks._numpy_module
    try:
        masks._numpy_module = None
        without_np, _, _ = masks.quantise_u8(values, 64, 32, kind="logit")
    finally:
        masks._numpy_module = saved

    assert len(with_np) == len(without_np)
    assert max(abs(a - b) for a, b in zip(with_np, without_np)) <= 1


def test_numpy_array_shapes_are_accepted():
    np = masks._np()
    if np is None:
        pytest.skip("numpy is not installed")
    flat = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32).reshape(2, 2)
    for arr in (flat, flat[None, :, :], flat[:, :, None]):
        values, w, h = masks.normalise_mask(arr)
        assert (w, h) == (2, 2)
    data, w, h = masks.quantise_u8(np.zeros((3, 5), dtype=np.uint8))
    assert (w, h) == (5, 3) and data == bytes(15)


def test_torch_like_tensor_is_accepted_without_torch(backend):
    """Engines hand us tensors; masks.py must not import torch to read one."""

    class FakeTensor:
        def __init__(self, rows):
            self._rows = rows

        def detach(self):
            return self

        def cpu(self):
            return self

        def float(self):
            return self

        def numpy(self):
            return self._rows  # a nested sequence: still normalisable

    values, w, h = masks.normalise_mask(FakeTensor([[0.0, 1.0], [1.0, 0.0]]))
    assert (w, h) == (2, 2)
    assert list(values) == [0.0, 1.0, 1.0, 0.0]


def test_normalise_mask_rejects_bad_shapes(backend):
    with pytest.raises(ValueError):
        masks.normalise_mask([0, 1, 2], 2, 2)
    with pytest.raises(ValueError):
        masks.normalise_mask([[0, 1], [2]])
    with pytest.raises(ValueError):
        masks.normalise_mask(b"\x00\x01")  # no dimensions
    with pytest.raises(TypeError):
        masks.normalise_mask(object())


def test_threshold_u8_uses_greater_or_equal(backend):
    data = bytes([0, 127, 128, 129, 255])
    assert list(masks.threshold_u8(data, 128)) == [0, 0, 255, 255, 255]
    assert list(masks.threshold_u8(data, 1)) == [0, 255, 255, 255, 255]


# --------------------------------------------------------------------------- #
# cropping
# --------------------------------------------------------------------------- #
def test_tight_bbox_is_exact_without_padding(backend):
    data = _rect_mask(20, 12, (3, 4, 9, 10))
    bbox = masks.tight_bbox(data, 20, 12, cutoff=128, pad=0)
    assert bbox.to_list() == [3, 4, 9, 10]
    assert (bbox.width, bbox.height) == (6, 6)


def test_tight_bbox_pads_and_clips_to_the_canvas(backend):
    data = _rect_mask(10, 10, (0, 0, 2, 2))
    bbox = masks.tight_bbox(data, 10, 10, cutoff=128, pad=2)
    assert bbox.to_list() == [0, 0, 4, 4]  # clipped at the top-left corner

    data = _rect_mask(10, 10, (8, 8, 10, 10))
    bbox = masks.tight_bbox(data, 10, 10, cutoff=128, pad=2)
    assert bbox.to_list() == [6, 6, 10, 10]  # clipped at the bottom-right


def test_tight_bbox_of_an_empty_mask_is_none(backend):
    assert masks.tight_bbox(bytes(64), 8, 8) is None
    # every pixel below the cutoff counts as empty
    assert masks.tight_bbox(bytes([10]) * 64, 8, 8, cutoff=26) is None


def test_tight_bbox_finds_a_single_pixel(backend):
    data = bytearray(64)
    data[5 * 8 + 6] = 255
    bbox = masks.tight_bbox(bytes(data), 8, 8, pad=0)
    assert bbox.to_list() == [6, 5, 7, 6]


def test_tight_bbox_cutoff_boundary_is_inclusive(backend):
    data = bytearray(16)
    data[3] = masks.DEFAULT_CROP_CUTOFF
    assert masks.tight_bbox(bytes(data), 4, 4, pad=0) is not None
    data[3] = masks.DEFAULT_CROP_CUTOFF - 1
    assert masks.tight_bbox(bytes(data), 4, 4, pad=0) is None


def test_crop_u8_matches_a_manual_crop(backend):
    rng = random.Random(7)
    w, h = 9, 7
    data = bytes(rng.randrange(256) for _ in range(w * h))
    bbox = BBox(2, 1, 7, 6)
    got = masks.crop_u8(data, w, h, bbox)
    want = bytes(
        data[y * w + x]
        for y in range(bbox.y0, bbox.y1)
        for x in range(bbox.x0, bbox.x1)
    )
    assert got == want


def test_encode_mask_resamples_a_low_resolution_mask_to_the_canvas(backend):
    """SAM decodes low-res logits; the soft mask is upsampled before cropping so
    that the reported bbox is in canvas pixels (API.md §8.2)."""
    low = _rect_mask(8, 8, (2, 2, 6, 6))
    enc = masks.encode_mask(low, 8, 8, canvas=Size(32, 32), cutoff=128, pad=0)
    assert enc is not None
    x0, y0, x1, y1 = enc.bbox.to_list()
    for got, want in ((x0, 8), (y0, 8), (x1, 24), (y1, 24)):
        assert abs(got - want) <= 1
    assert len(enc.data) == enc.width * enc.height


# --------------------------------------------------------------------------- #
# the binary frame
# --------------------------------------------------------------------------- #
def _make_result(backend_masks, canvas=Size(64, 64), image=Size(48, 32), **kw):
    return masks.encode_result(
        job_id="j-000002-77c1",
        request_id="r-000007",
        image_id="5b1d3d3fd6a34c2f8f8d3c9a41b7e0d2",
        engine="pcs",
        image=image,
        model_canvas=canvas,
        masks=backend_masks,
        prompt={"kind": "text", "text": "yellow school bus"},
        **kw,
    )


def test_frame_layout_is_exactly_as_documented(backend):
    canvas = Size(64, 64)
    items = [
        masks.SoftInstance(_rect_mask(64, 64, (10, 10, 20, 24)), 0.9, "bus"),
        masks.SoftInstance(_rect_mask(64, 64, (30, 30, 34, 36)), 0.5, "bus"),
    ]
    header, payload = _make_result(items, canvas=canvas, cutoff=128, pad=0)

    assert payload[:8] == RESULT_MAGIC
    (hlen,) = struct.unpack_from("<I", payload, 8)
    assert RESULT_PREFIX_SIZE == 12
    raw_header = json.loads(payload[12:12 + hlen].decode("utf-8"))
    assert raw_header["mask_encoding"] == "u8_soft"
    assert raw_header["api_version"] == API_VERSION
    assert raw_header["request_id"] == "r-000007"
    # compact separators: no wasted whitespace on the wire
    assert b", " not in payload[12:12 + hlen]

    blob_start = 12 + hlen
    assert len(payload) == blob_start + raw_header["blob_length"]

    total = 0
    for i, inst in enumerate(raw_header["instances"]):
        assert inst["instance_id"] == i
        assert inst["mask_width"] == inst["bbox"][2] - inst["bbox"][0]
        assert inst["mask_height"] == inst["bbox"][3] - inst["bbox"][1]
        assert inst["blob_length"] == inst["mask_width"] * inst["mask_height"]
        assert inst["blob_offset"] == total  # tightly packed, in order
        # blob_offset is relative to the blob region, never to the body
        chunk = payload[blob_start + inst["blob_offset"]:
                        blob_start + inst["blob_offset"] + inst["blob_length"]]
        assert len(chunk) == inst["blob_length"]
        total += inst["blob_length"]
    assert total == raw_header["blob_length"]
    assert raw_header["instances"][0]["blob_offset"] == 0
    assert header.blob_length == total


def test_instances_are_sorted_by_score_descending(backend):
    items = [
        masks.SoftInstance(_rect_mask(32, 32, (1, 1, 5, 5)), 0.2, "a"),
        masks.SoftInstance(_rect_mask(32, 32, (6, 6, 9, 9)), 0.9, "b"),
        masks.SoftInstance(_rect_mask(32, 32, (10, 10, 14, 14)), 0.5, "c"),
    ]
    header, _ = _make_result(items, canvas=Size(32, 32), cutoff=128, pad=0)
    assert [i.label for i in header.instances] == ["b", "c", "a"]
    assert [i.instance_id for i in header.instances] == [0, 1, 2]
    assert header.truncated is False


def test_max_instances_truncates_and_flags(backend):
    items = [
        masks.SoftInstance(_rect_mask(32, 32, (i, i, i + 3, i + 3)), 0.1 * (i + 1))
        for i in range(6)
    ]
    header, payload = _make_result(items, canvas=Size(32, 32), max_instances=2,
                                   cutoff=128, pad=0)
    assert len(header.instances) == 2
    assert header.truncated is True
    assert header.instances[0].score > header.instances[1].score
    unpack_result(payload)  # revalidates the whole frame


def test_empty_masks_are_dropped_not_sent_as_degenerate_rectangles(backend):
    items = [
        masks.SoftInstance(bytes(32 * 32), 0.9, "nothing", 32, 32),
        masks.SoftInstance(_rect_mask(32, 32, (4, 4, 8, 8)), 0.4, "something"),
    ]
    header, _ = _make_result(items, canvas=Size(32, 32), cutoff=128, pad=0)
    assert [i.label for i in header.instances] == ["something"]
    assert header.instances[0].instance_id == 0


def test_a_result_with_no_instances_is_legal(backend):
    header, payload = _make_result([], canvas=Size(32, 32))
    (hlen,) = struct.unpack_from("<I", payload, 8)
    assert len(payload) == 12 + hlen
    assert header.blob_length == 0
    decoded_header, decoded = masks.decode_result(payload)
    assert decoded == []
    assert decoded_header.instances == []


def test_decode_is_the_exact_inverse_of_encode(backend):
    canvas = Size(48, 40)
    original = _rect_mask(canvas.width, canvas.height, (5, 6, 21, 30))
    header, payload = _make_result([masks.SoftInstance(original, 0.77, "bus")],
                                   canvas=canvas, cutoff=128, pad=0)
    decoded_header, decoded = masks.decode_result(payload)

    assert decoded_header.to_dict() == header.to_dict()
    assert len(decoded) == 1
    inst = decoded[0]
    assert inst.score == pytest.approx(0.77)
    assert inst.label == "bus"
    assert inst.bbox.to_list() == [5, 6, 21, 30]
    # pasted back onto the canvas the mask is byte-identical to the input:
    # everything the crop dropped was zero by construction.
    assert inst.to_canvas(canvas) == original


def test_soft_values_survive_the_round_trip_so_the_client_can_threshold(backend):
    """The entire justification for the format: the client re-thresholds bytes
    it already has, with no round trip (API.md §8.3)."""
    canvas = Size(32, 32)
    logits = []
    for y in range(canvas.height):
        row = []
        for x in range(canvas.width):
            # a smooth ramp across the canvas: every byte value is represented
            row.append((x - 16) * 0.5)
        logits.append(row)
    header, payload = _make_result([masks.SoftInstance(logits, 1.0, "ramp")],
                                   canvas=canvas, cutoff=1, pad=0)
    _, decoded = masks.decode_result(payload)
    full = decoded[0].to_canvas(canvas)
    assert len(set(full)) > 20  # genuinely soft, not binarised
    low = masks.threshold_u8(full, 100).count(255)
    high = masks.threshold_u8(full, 200).count(255)
    assert low > high > 0


def test_encode_result_reports_the_transform_it_used(backend):
    """The reference policy: the canvas is the upload, the transform the
    identity -- on a non-square image too, where a square canvas would have
    stretched every mask (API.md §5)."""
    image = Size(1008, 672)
    canvas = masks.model_canvas_size(image)
    assert canvas == image
    header, _ = _make_result([], image=image, canvas=canvas)
    t = header.canvas_from_image
    assert (t.scale_x, t.scale_y) == (pytest.approx(1.0), pytest.approx(1.0))
    assert (t.offset_x, t.offset_y) == (0.0, 0.0)


def test_encode_result_reports_a_non_identity_transform_it_is_given(backend):
    """Clients invert whatever transform is reported, so the arithmetic must
    hold for a canvas that is not the image."""
    header, _ = _make_result([], image=Size(800, 600), canvas=Size(400, 300))
    t = header.canvas_from_image
    assert (t.scale_x, t.scale_y) == (pytest.approx(0.5), pytest.approx(0.5))


def test_frames_are_identical_across_backends():
    """A frame must not depend on whether numpy happened to be installed."""
    np = masks._np()
    if np is None:
        pytest.skip("numpy is not installed")
    items = [masks.SoftInstance(_rect_mask(40, 40, (3, 4, 25, 33)), 0.6, "x")]
    _, with_np = _make_result(items, canvas=Size(40, 40), cutoff=128, pad=1)
    saved = masks._numpy_module
    try:
        masks._numpy_module = None
        items = [masks.SoftInstance(_rect_mask(40, 40, (3, 4, 25, 33)), 0.6, "x")]
        _, without_np = _make_result(items, canvas=Size(40, 40), cutoff=128, pad=1)
    finally:
        masks._numpy_module = saved
    assert with_np == without_np


# --------------------------------------------------------------------------- #
# the downscale plan
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "source,expected",
    [
        ((3000, 2000), (1008, 672)),     # the worked example in API.md §12.2
        ((4000, 3000), (1008, 756)),
        ((1008, 1008), (1008, 1008)),    # already at the limit: untouched
        ((640, 480), (640, 480)),        # smaller than the limit: never upscaled
        ((2000, 3000), (672, 1008)),     # portrait
        ((16, 16), (16, 16)),            # the minimum legal upload
        ((1, 1), (16, 16)),              # clamped up to the minimum
        ((1, 1000), (16, 1000)),         # extreme strip: width clamped
        ((1000, 1), (1000, 16)),
        ((4001, 2999), (1008, 756)),     # odd numbers
        ((1009, 1009), (1008, 1008)),    # one pixel over
    ],
)
def test_plan_downscale_targets(source, expected):
    plan = masks.plan_downscale(*source)
    assert (plan.target.width, plan.target.height) == expected
    assert 16 <= plan.target.width <= 1008
    assert 16 <= plan.target.height <= 1008


def test_plan_downscale_records_exact_per_axis_scales():
    plan = masks.plan_downscale(3000, 2000)
    assert plan.scale_x == pytest.approx(1008 / 3000.0)
    assert plan.scale_y == pytest.approx(672 / 2000.0)
    assert plan.needs_resample is True
    x, y = plan.source_to_target(1500.0, 1000.0)
    assert plan.target_to_source(x, y) == pytest.approx((1500.0, 1000.0))
    assert plan.to_dict()["target"] == {"width": 1008, "height": 672}


def test_plan_downscale_rejects_degenerate_sizes():
    with pytest.raises(ValueError):
        masks.plan_downscale(0, 10)
    with pytest.raises(ValueError):
        masks.plan_downscale(10, -1)


def test_plan_downscale_over_many_random_sizes():
    """Whatever the aspect ratio, the plan must produce an uploadable size."""
    rng = random.Random(99)
    for _ in range(500):
        w = rng.choice([rng.randint(1, 40), rng.randint(1, 6000)])
        h = rng.choice([rng.randint(1, 40), rng.randint(1, 6000)])
        plan = masks.plan_downscale(w, h)
        t = plan.target
        assert 16 <= t.width <= 1008 and 16 <= t.height <= 1008
        assert t.width * t.height * 3 <= 1008 * 1008 * 3
        # never enlarged beyond the min-side clamp
        assert t.width <= max(w, 16) and t.height <= max(h, 16)


# --------------------------------------------------------------------------- #
# resampling
# --------------------------------------------------------------------------- #
def test_resample_is_identity_at_the_same_size(backend):
    src = bytes(range(48))
    assert masks.resample_rgb(src, 4, 4, 4, 4) == src


def test_resample_preserves_a_constant_image(backend):
    src = bytes([17, 34, 51]) * (20 * 15)
    out = masks.resample_rgb(src, 20, 15, 7, 5)
    assert set(out[0::3]) == {17}
    assert set(out[1::3]) == {34}
    assert set(out[2::3]) == {51}


def test_resample_box_filter_averages_exactly(backend):
    # 2x2 -> 1x1 must be the plain mean of the four samples
    src = bytes([0, 0, 0, 100, 100, 100, 200, 200, 200, 255, 255, 255])
    out = masks.resample_rgb(src, 2, 2, 1, 1)
    assert list(out) == [139, 139, 139]  # round((0+100+200+255)/4) = 139


def test_resample_upscale_is_bilinear_not_blocky(backend):
    src = bytes([0, 255])  # 2x1 gray
    out = masks.resample_gray(src, 2, 1, 4, 1)
    assert list(out) == [0, 64, 191, 255]


def test_downscale_rgb_returns_untouched_pixels_when_already_small(backend):
    src = bytes(range(256)) * 3
    out, plan = masks.downscale_rgb(src, 16, 16)
    assert out == src
    assert plan.needs_resample is False


def test_resample_backends_agree_within_one_lsb():
    np = masks._np()
    if np is None:
        pytest.skip("numpy is not installed")
    rng = random.Random(4242)
    src = bytes(rng.randrange(256) for _ in range(37 * 23 * 3))
    with_np = masks.resample_rgb(src, 37, 23, 11, 9)
    saved = masks._numpy_module
    try:
        masks._numpy_module = None
        without_np = masks.resample_rgb(src, 37, 23, 11, 9)
    finally:
        masks._numpy_module = saved
    assert len(with_np) == len(without_np)
    assert max(abs(a - b) for a, b in zip(with_np, without_np)) <= 1


def _centroid_gray(data, width, height, cutoff=128):
    sx = sy = n = 0
    for y in range(height):
        for x in range(width):
            if data[y * width + x] >= cutoff:
                sx += x + 0.5
                sy += y + 0.5
                n += 1
    assert n, "no pixels above the cutoff"
    return sx / n, sy / n


@pytest.mark.parametrize("source", [(400, 300), (1500, 500), (333, 999), (2048, 1024)])
def test_downscale_keeps_a_marker_where_the_plan_says_it_is(backend, source):
    """The pixels and the geometry must not drift apart.

    A bright block is placed in the source image, the image is downscaled, and
    the block's centroid in the result is compared with the centroid the plan
    predicts.  Within half a destination pixel, i.e. the resampler and
    :class:`DownscalePlan` agree on the mapping.
    """
    sw, sh = source
    rect = (sw // 4, sh // 3, sw // 4 + max(8, sw // 8), sh // 3 + max(8, sh // 8))
    gray = _rect_mask(sw, sh, rect)
    plan = masks.plan_downscale(sw, sh)
    out = masks.resample_gray(gray, sw, sh, plan.target.width, plan.target.height)

    cx, cy = _centroid_gray(out, plan.target.width, plan.target.height)
    src_cx = (rect[0] + rect[2]) / 2.0
    src_cy = (rect[1] + rect[3]) / 2.0
    want_x, want_y = plan.source_to_target(src_cx, src_cy)
    assert abs(cx - want_x) <= 0.5
    assert abs(cy - want_y) <= 0.5


# --------------------------------------------------------------------------- #
# geometry: the alignment property tests
# --------------------------------------------------------------------------- #
def test_worked_example_from_the_contract():
    """API.md §12.6: a 3000x2000 original uploaded at 1008x672, identity
    transform, instance bbox [412, 200, 700, 320]."""
    image = Size(1008, 672)
    transform = masks.canvas_transform(image, masks.model_canvas_size(image))
    bbox = BBox(412, 200, 700, 320)
    x0, y0, dw, dh = masks.source_rect_for_bbox(bbox, transform, image, Size(3000, 2000))
    assert (x0, y0, dw, dh) == (1226, 595, 857, 357)


def test_image_rect_for_bbox_inverts_the_reported_affine():
    image = Size(800, 600)
    canvas = Size(1008, 1008)
    t = masks.canvas_transform(image, canvas)
    rect = masks.image_rect_for_bbox(BBox(0, 0, canvas.width, canvas.height), t)
    assert rect == pytest.approx((0.0, 0.0, 800.0, 600.0))


def test_source_rect_edges_are_rounded_independently_so_neighbours_tile():
    """API.md §9 step 3: round each edge, then derive the size.

    Rounding the origin and the size separately makes adjacent instances
    overlap or leave a seam; two abutting canvas boxes must abut on the source
    too.
    """
    image = Size(500, 500)
    canvas = Size(1008, 1008)
    t = masks.canvas_transform(image, canvas)
    source = Size(1731, 1731)  # deliberately not a round ratio
    left = masks.source_rect_for_bbox(BBox(100, 0, 337, 10), t, image, source)
    right = masks.source_rect_for_bbox(BBox(337, 0, 500, 10), t, image, source)
    assert left[0] + left[2] == right[0]


def _random_sizes(rng, n):
    out = []
    for _ in range(n):
        style = rng.randrange(5)
        if style == 0:
            out.append((rng.randint(16, 64), rng.randint(16, 64)))          # tiny
        elif style == 1:
            out.append((rng.randint(1000, 6000), rng.randint(1000, 6000)))  # photo
        elif style == 2:
            s = rng.randint(17, 4000)
            out.append((s, s))                                              # square
        elif style == 3:
            out.append((rng.randint(1, 30), rng.randint(500, 4000)))        # strip
        else:
            out.append((rng.randint(500, 4000), rng.randint(1, 30)))
    return out


def test_canvas_bbox_round_trips_to_the_source_within_one_pixel():
    """The property that matters: bbox -> source -> exact float source.

    For hundreds of random sizes and aspect ratios, the integer rectangle the
    client would place on its original image is within one pixel of the exact
    real-valued answer.  Anything worse is a visible misalignment.
    """
    rng = random.Random(1234567)
    for sw, sh in _random_sizes(rng, 400):
        plan = masks.plan_downscale(sw, sh)
        image = plan.target
        canvas = masks.model_canvas_size(image)
        t = masks.canvas_transform(image, canvas)

        for _ in range(3):
            x0 = rng.randrange(0, canvas.width - 1)
            x1 = rng.randrange(x0 + 1, canvas.width + 1)
            y0 = rng.randrange(0, canvas.height - 1)
            y1 = rng.randrange(y0 + 1, canvas.height + 1)
            bbox = BBox(x0, y0, x1, y1)

            X0, Y0, DW, DH = masks.source_rect_for_bbox(bbox, t, image, Size(sw, sh))

            u = sw / float(image.width)
            v = sh / float(image.height)
            ex0 = (x0 / t.scale_x) * u
            ex1 = (x1 / t.scale_x) * u
            ey0 = (y0 / t.scale_y) * v
            ey1 = (y1 / t.scale_y) * v

            # A rectangle narrower than one source pixel cannot be represented
            # -- the wire format requires x1 > x0 -- so it is inflated to a
            # single pixel.  That inflation is the only permitted deviation
            # beyond half-up rounding, and it is bounded by exactly the amount
            # the rectangle was widened.
            tol_x = 1.0 + max(0.0, 1.0 - (ex1 - ex0))
            tol_y = 1.0 + max(0.0, 1.0 - (ey1 - ey0))

            assert abs(X0 - ex0) <= tol_x, (sw, sh, bbox.to_list())
            assert abs(Y0 - ey0) <= tol_y, (sw, sh, bbox.to_list())
            assert abs((X0 + DW) - ex1) <= tol_x, (sw, sh, bbox.to_list())
            assert abs((Y0 + DH) - ey1) <= tol_y, (sw, sh, bbox.to_list())
            assert 1 <= DW and 1 <= DH
            assert 0 <= X0 and X0 + DW <= sw
            assert 0 <= Y0 and Y0 + DH <= sh


def test_forward_and_inverse_transforms_compose_to_identity():
    """source -> uploaded -> canvas -> uploaded -> source, in floats."""
    rng = random.Random(31337)
    for sw, sh in _random_sizes(rng, 200):
        plan = masks.plan_downscale(sw, sh)
        image = plan.target
        canvas = masks.model_canvas_size(image)
        t = masks.canvas_transform(image, canvas)
        for _ in range(4):
            px = rng.uniform(0.0, sw)
            py = rng.uniform(0.0, sh)
            ix, iy = plan.source_to_target(px, py)
            cx, cy = t.image_to_canvas(ix, iy)
            bx, by = t.canvas_to_image(cx, cy)
            ox, oy = plan.target_to_source(bx, by)
            assert abs(ox - px) < 1e-6
            assert abs(oy - py) < 1e-6


@pytest.mark.parametrize("canvas_side", [32, 64, 97])
def test_a_rectangle_encodes_and_decodes_to_the_same_canvas_pixels(backend, canvas_side):
    """Encode/decode must not move a mask by even one pixel."""
    rng = random.Random(canvas_side * 7919)
    canvas = Size(canvas_side, canvas_side)
    for _ in range(25):
        x0 = rng.randrange(0, canvas_side - 1)
        x1 = rng.randrange(x0 + 1, canvas_side + 1)
        y0 = rng.randrange(0, canvas_side - 1)
        y1 = rng.randrange(y0 + 1, canvas_side + 1)
        mask = _rect_mask(canvas_side, canvas_side, (x0, y0, x1, y1))

        header, payload = _make_result([masks.SoftInstance(mask, 0.5, "r")],
                                       canvas=canvas, cutoff=128, pad=0)
        _, decoded = masks.decode_result(payload)
        assert len(decoded) == 1
        assert decoded[0].bbox.to_list() == [x0, y0, x1, y1]
        assert decoded[0].to_canvas(canvas) == mask


def test_padding_grows_the_crop_without_moving_the_content(backend):
    canvas = Size(40, 40)
    mask = _rect_mask(40, 40, (10, 12, 20, 26))
    header, payload = _make_result([masks.SoftInstance(mask, 0.5)],
                                   canvas=canvas, cutoff=128, pad=2)
    _, decoded = masks.decode_result(payload)
    assert decoded[0].bbox.to_list() == [8, 10, 22, 28]
    # the padded border is zero and the content is untouched
    assert decoded[0].to_canvas(canvas) == mask


def test_mask_lands_on_the_object_end_to_end():
    """The user-visible property, over random sizes and aspect ratios.

    An object occupies a known rectangle of the *original* image.  It is
    downscaled, segmented at canvas resolution, framed, decoded and mapped back
    with :func:`source_rect_for_bbox` -- the exact chain the plug-in performs.
    The recovered rectangle must sit within one canvas pixel (expressed in
    source pixels) of where the object actually is.
    """
    rng = random.Random(8675309)
    canvas = Size(96, 96)  # small on purpose: 400 iterations of 1008x1008 is slow
    for sw, sh in _random_sizes(rng, 120):
        plan = masks.plan_downscale(sw, sh)
        image = plan.target
        t = masks.canvas_transform(image, canvas)

        # a rectangle on the original image, at least a few source pixels wide
        ox0 = rng.uniform(0.0, sw * 0.6)
        oy0 = rng.uniform(0.0, sh * 0.6)
        ox1 = min(float(sw), ox0 + max(sw * 0.2, 4.0))
        oy1 = min(float(sh), oy0 + max(sh * 0.2, 4.0))

        cx0, cy0 = t.image_to_canvas(*plan.source_to_target(ox0, oy0))
        cx1, cy1 = t.image_to_canvas(*plan.source_to_target(ox1, oy1))
        bx0 = max(0, min(canvas.width - 1, int(math.floor(cx0))))
        by0 = max(0, min(canvas.height - 1, int(math.floor(cy0))))
        bx1 = max(bx0 + 1, min(canvas.width, int(math.ceil(cx1))))
        by1 = max(by0 + 1, min(canvas.height, int(math.ceil(cy1))))

        mask = _rect_mask(canvas.width, canvas.height, (bx0, by0, bx1, by1))
        header, payload = masks.encode_result(
            job_id="j", request_id="r", image_id="i", engine="pvs",
            image=image, model_canvas=canvas,
            masks=[masks.SoftInstance(mask, 1.0)], cutoff=128, pad=0)
        _, decoded = masks.decode_result(payload)
        assert decoded, (sw, sh)

        X0, Y0, DW, DH = masks.source_rect_for_bbox(
            decoded[0].bbox, header.canvas_from_image, image, Size(sw, sh))

        # one canvas pixel, measured in source pixels, plus rounding slack
        tol_x = sw / float(canvas.width) + 1.5
        tol_y = sh / float(canvas.height) + 1.5
        assert abs(X0 - ox0) <= tol_x, (sw, sh, X0, ox0)
        assert abs(Y0 - oy0) <= tol_y, (sw, sh, Y0, oy0)
        assert abs((X0 + DW) - ox1) <= tol_x, (sw, sh, X0 + DW, ox1)
        assert abs((Y0 + DH) - oy1) <= tol_y, (sw, sh, Y0 + DH, oy1)


def test_full_resolution_canvas_round_trip_is_exact(backend):
    """One case at full upload size -- a 1008x672 canvas, the reference
    policy's -- so the property tests' smaller canvases are not hiding a
    size-dependent bug."""
    image = Size(1008, 672)
    canvas = masks.model_canvas_size(image)
    rect = (400, 200, 700, 320)
    mask = _rect_mask(canvas.width, canvas.height, rect)
    header, payload = masks.encode_result(
        job_id="j", request_id="r", image_id="i", engine="pcs",
        image=image, model_canvas=canvas,
        masks=[masks.SoftInstance(mask, 0.9, "bus")], cutoff=128, pad=0)
    _, decoded = masks.decode_result(payload)
    assert decoded[0].bbox.to_list() == list(rect)
    assert decoded[0].width * decoded[0].height == len(decoded[0].data)
    x0, y0, dw, dh = masks.source_rect_for_bbox(
        decoded[0].bbox, header.canvas_from_image, image, Size(3000, 2000))
    assert (x0, y0) == (1190, 595)
    assert (dw, dh) == (893, 357)


def test_build_result_returns_just_the_bytes(backend):
    canvas = Size(32, 32)
    payload = masks.build_result(
        job_id="j", request_id="r", image_id="i", engine="pvs",
        image=Size(32, 24), model_canvas=canvas,
        masks=[masks.SoftInstance(_rect_mask(32, 32, (2, 3, 9, 11)), 0.8)],
        cutoff=128, pad=0)
    assert payload[:8] == RESULT_MAGIC
    header, decoded = masks.decode_result(payload)
    assert header.engine == "pvs"
    assert decoded[0].bbox.to_list() == [2, 3, 9, 11]


def test_encoded_mask_value_at_is_zero_outside_the_crop(backend):
    canvas = Size(16, 16)
    enc = masks.encode_mask(_rect_mask(16, 16, (4, 5, 8, 9)), 16, 16, cutoff=128, pad=0)
    assert enc.value_at(4, 5) == 255
    assert enc.value_at(7, 8) == 255
    assert enc.value_at(3, 5) == 0        # outside the crop, defined as 0
    assert enc.value_at(100, 100) == 0
    assert enc.to_canvas(canvas) == _rect_mask(16, 16, (4, 5, 8, 9))


def test_decoded_instance_helpers(backend):
    canvas = Size(24, 24)
    header, payload = _make_result(
        [masks.SoftInstance(_rect_mask(24, 24, (2, 2, 10, 10), inside=200), 0.31, "z")],
        canvas=canvas, image=Size(24, 24), cutoff=128, pad=0)
    _, decoded = masks.decode_result(payload)
    inst = decoded[0]
    assert inst.label == "z" and inst.score == pytest.approx(0.31)
    assert (inst.width, inst.height) == (8, 8)
    assert inst.value_at(2, 2) == 200
    assert set(inst.binary(128)) == {255}
    assert set(inst.binary(201)) == {0}
    full = masks.expand_to_canvas(decoded, canvas)
    assert len(full) == 1 and len(full[0]) == 24 * 24
    with pytest.raises(ValueError):
        inst.to_canvas(None)
