"""Tests for ``plugin/sam3_gimp/gimpbridge.py``.

Two halves, matching the module:

* **pure geometry** -- runs with no GIMP at all, and is checked against the
  worked example in ``_daemon/API.md`` section 12.6 so a drift in the contract
  breaks a test rather than a user's masks;
* **pixel I/O** -- runs against ``tests/fake_gimp``, which models real
  byte-backed GEGL buffers, so the assertions are on actual pixel values.

Remember the harness caveat: the fake scaler is nearest-neighbour, so tests
assert on constant / blocky mask content only, never on interpolated values.
"""

from __future__ import annotations

import pytest

import gimpbridge
from fake_gimp import fake_gimp, fake_image  # noqa: F401  (pytest fixtures)

# The affine the reference processor reports for a 1008x672 upload: an
# anisotropic squash onto a 1008x1008 canvas, no letterboxing.
CFI_SQUASH = {"scale_x": 1.0, "scale_y": 1.5, "offset_x": 0.0, "offset_y": 0.0}


# --------------------------------------------------------------------------- #
# compute_upload_size
# --------------------------------------------------------------------------- #
class TestComputeUploadSize:
    def test_landscape_is_scaled_to_the_long_side(self):
        assert gimpbridge.compute_upload_size(3000, 2000) == (1008, 672)

    def test_portrait_is_scaled_to_the_long_side(self):
        assert gimpbridge.compute_upload_size(2000, 3000) == (672, 1008)

    def test_square(self):
        assert gimpbridge.compute_upload_size(4096, 4096) == (1008, 1008)

    def test_small_images_are_never_upscaled_for_no_reason(self):
        assert gimpbridge.compute_upload_size(640, 480) == (640, 480)
        assert gimpbridge.compute_upload_size(1008, 1008) == (1008, 1008)

    def test_a_side_below_the_daemon_minimum_is_raised(self):
        # The daemon rejects anything under 16 px (API.md section 6.2).
        assert gimpbridge.compute_upload_size(200, 4) == (200, 16)
        assert gimpbridge.compute_upload_size(8, 8) == (16, 16)

    def test_extreme_aspect_ratio_hits_both_clamps(self):
        # 4000x10 cannot satisfy "long side <= 1008" and "short side >= 16"
        # while preserving the ratio.  Distorting is correct; the geometry
        # object records the two axes separately so nothing desynchronises.
        width, height = gimpbridge.compute_upload_size(4000, 10)
        assert width == 1008
        assert height == 16

    def test_result_is_always_inside_the_daemons_accepted_range(self):
        for size in [(1, 1), (17, 4001), (10000, 10000), (1009, 15), (16, 1008)]:
            width, height = gimpbridge.compute_upload_size(*size)
            assert 16 <= width <= 1008
            assert 16 <= height <= 1008

    def test_rejects_degenerate_input(self):
        with pytest.raises(ValueError):
            gimpbridge.compute_upload_size(0, 100)
        with pytest.raises(ValueError):
            gimpbridge.compute_upload_size(100, -1)


# --------------------------------------------------------------------------- #
# UploadGeometry
# --------------------------------------------------------------------------- #
class TestUploadGeometry:
    def setup_method(self):
        self.geometry = gimpbridge.UploadGeometry(3000, 2000, 1008, 672)

    def test_scales(self):
        assert self.geometry.scale_x == pytest.approx(1008.0 / 3000.0)
        assert self.geometry.scale_y == pytest.approx(672.0 / 2000.0)

    def test_click_maps_to_the_contracts_worked_example(self):
        # API.md section 12.7: a canvas click at original (1524, 893) is sent
        # as uploaded-image (512.06, 300.05).
        x, y = self.geometry.to_upload(1524, 893)
        assert x == pytest.approx(512.064, abs=1e-3)
        assert y == pytest.approx(300.048, abs=1e-3)

    def test_round_trip(self):
        for point in [(0, 0), (1524, 893), (2999, 1999)]:
            back = self.geometry.from_upload(*self.geometry.to_upload(*point))
            assert back[0] == pytest.approx(point[0])
            assert back[1] == pytest.approx(point[1])

    def test_box_to_upload(self):
        box = self.geometry.box_to_upload(0, 0, 3000, 2000)
        assert box == pytest.approx((0.0, 0.0, 1008.0, 672.0))

    def test_upload_headers_match_the_contract(self):
        headers = self.geometry.upload_headers()
        assert headers["X-Width"] == "1008"
        assert headers["X-Height"] == "672"
        assert headers["X-Source-Width"] == "3000"
        assert headers["X-Source-Height"] == "2000"

    def test_expected_payload_size_is_w_h_3(self):
        assert self.geometry.expected_payload_size() == 1008 * 672 * 3


# --------------------------------------------------------------------------- #
# place_instance -- the mask-mapping arithmetic
# --------------------------------------------------------------------------- #
class TestPlaceInstance:
    def test_matches_the_contract_worked_example(self):
        # _daemon/API.md section 12.6, verbatim.
        geometry = gimpbridge.UploadGeometry(3000, 2000, 1008, 672)
        placement = gimpbridge.place_instance([412, 300, 700, 480], CFI_SQUASH, geometry)
        assert (placement.x, placement.y) == (1226, 595)
        assert (placement.width, placement.height) == (857, 357)

    def test_uses_the_reported_offsets_instead_of_assuming_zero(self):
        # A letterboxing processor would report non-zero offsets.  A client
        # that hardcoded the squash would misplace every mask (API.md s5).
        geometry = gimpbridge.UploadGeometry(1000, 1000, 500, 500)
        letterbox = {"scale_x": 0.5, "scale_y": 0.5, "offset_x": 100.0, "offset_y": 50.0}
        placement = gimpbridge.place_instance([100, 50, 200, 150], letterbox, geometry)
        # canvas 100 -> uploaded 0 -> original 0 ; canvas 200 -> uploaded 200 -> original 400
        assert (placement.x, placement.y) == (0, 0)
        assert (placement.width, placement.height) == (400, 400)

    def test_identity_transform_is_a_straight_upscale(self):
        geometry = gimpbridge.UploadGeometry(2016, 1008, 1008, 504)
        identity = {"scale_x": 1.0, "scale_y": 1.0, "offset_x": 0.0, "offset_y": 0.0}
        placement = gimpbridge.place_instance([10, 20, 30, 40], identity, geometry)
        assert (placement.x, placement.y, placement.width, placement.height) == (
            20,
            40,
            40,
            40,
        )

    def test_accepts_an_object_as_well_as_a_mapping(self):
        class Affine:
            scale_x = 1.0
            scale_y = 1.5
            offset_x = 0.0
            offset_y = 0.0

        geometry = gimpbridge.UploadGeometry(3000, 2000, 1008, 672)
        assert gimpbridge.place_instance(
            [412, 300, 700, 480], Affine(), geometry
        ) == gimpbridge.place_instance([412, 300, 700, 480], CFI_SQUASH, geometry)

    def test_clamps_into_the_image_and_never_degenerates(self):
        geometry = gimpbridge.UploadGeometry(100, 100, 100, 100)
        identity = {"scale_x": 1.0, "scale_y": 1.0, "offset_x": 0.0, "offset_y": 0.0}
        placement = gimpbridge.place_instance([90, 90, 400, 400], identity, geometry)
        assert placement.x + placement.width <= 100
        assert placement.y + placement.height <= 100
        assert placement.width >= 1 and placement.height >= 1

    def test_adjacent_instances_tile_without_gaps(self):
        # Rounding each edge independently (rather than rounding origin+size)
        # is what makes this true.
        geometry = gimpbridge.UploadGeometry(1000, 1000, 333, 333)
        identity = {"scale_x": 1.0, "scale_y": 1.0, "offset_x": 0.0, "offset_y": 0.0}
        left = gimpbridge.place_instance([0, 0, 111, 333], identity, geometry)
        right = gimpbridge.place_instance([111, 0, 222, 333], identity, geometry)
        assert left.x + left.width == right.x

    def test_rejects_a_zero_scale(self):
        geometry = gimpbridge.UploadGeometry(100, 100, 100, 100)
        with pytest.raises(ValueError):
            gimpbridge.place_instance(
                [0, 0, 10, 10],
                {"scale_x": 0.0, "scale_y": 1.0, "offset_x": 0, "offset_y": 0},
                geometry,
            )

    def test_rejects_a_malformed_bbox(self):
        geometry = gimpbridge.UploadGeometry(100, 100, 100, 100)
        with pytest.raises(ValueError):
            gimpbridge.place_instance([0, 0, 10], CFI_SQUASH, geometry)


# --------------------------------------------------------------------------- #
# threshold_bytes / crop_rows
# --------------------------------------------------------------------------- #
class TestPureByteHelpers:
    def test_threshold_default_is_the_models_own_binarisation_point(self):
        # API.md section 8.3: 128 == logit 0 == the default client threshold.
        assert gimpbridge.DEFAULT_MASK_THRESHOLD == 128
        data = bytes([0, 1, 127, 128, 129, 255])
        assert gimpbridge.threshold_bytes(data) == bytes([0, 0, 0, 255, 255, 255])

    def test_threshold_is_inclusive_at_the_cut(self):
        assert gimpbridge.threshold_bytes(bytes([200]), 200) == b"\xff"
        assert gimpbridge.threshold_bytes(bytes([199]), 200) == b"\x00"

    def test_threshold_preserves_length_for_a_large_buffer(self):
        data = bytes(range(256)) * 400
        out = gimpbridge.threshold_bytes(data, 64)
        assert len(out) == len(data)
        assert set(out) <= {0, 255}

    def test_threshold_range_is_validated(self):
        for bad in (0, 256, -1):
            with pytest.raises(ValueError):
                gimpbridge.threshold_bytes(b"\x00", bad)

    def test_crop_rows_extracts_a_sub_rectangle(self):
        data = bytes(range(16))  # 4x4 gray
        assert gimpbridge.crop_rows(data, 4, 4, 1, 1, 2, 2) == bytes([5, 6, 9, 10])

    def test_crop_rows_full_rect_is_identity(self):
        data = bytes(range(16))
        assert gimpbridge.crop_rows(data, 4, 4, 0, 0, 4, 4) == data

    def test_crop_rows_empty(self):
        assert gimpbridge.crop_rows(b"\x00" * 16, 4, 4, 0, 0, 0, 3) == b""

    def test_crop_rows_handles_multi_component_pixels(self):
        data = bytes(
            [
                0, 0, 0, 1, 1, 1,
                2, 2, 2, 3, 3, 3,
            ]
        )  # 2x2 RGB
        assert gimpbridge.crop_rows(data, 2, 2, 1, 0, 1, 2, components=3) == bytes(
            [1, 1, 1, 3, 3, 3]
        )


# --------------------------------------------------------------------------- #
# guards when GIMP is absent
# --------------------------------------------------------------------------- #
class TestWithoutGimp:
    def test_module_imports_and_reports_no_gimp(self):
        # This process has no Gimp typelib; the module must still import so the
        # pure helpers above can be tested (and so a bootstrap/doctor path can
        # import it to explain the problem).
        assert gimpbridge.gimp_available() is False

    def test_pixel_helpers_refuse_politely(self):
        with pytest.raises(gimpbridge.GimpUnavailableError):
            gimpbridge.read_upload_pixels(object())
        with pytest.raises(gimpbridge.GimpUnavailableError):
            gimpbridge.scale_soft_mask(b"\x00" * 4, 2, 2, 4, 4)
        with pytest.raises(gimpbridge.GimpUnavailableError):
            gimpbridge.selection_box(object())

    def test_a_no_op_scale_needs_no_gimp_at_all(self):
        data = bytes([1, 2, 3, 4])
        assert gimpbridge.scale_soft_mask(data, 2, 2, 2, 2) == data


# --------------------------------------------------------------------------- #
# the harness itself
# --------------------------------------------------------------------------- #
class TestFakeGimpHarness:
    def test_stubs_are_installed_and_removed_again(self, fake_gimp):
        import fake_gimp as harness

        assert harness.is_installed()
        assert gimpbridge.gimp_available()
        assert gimpbridge.Gimp is fake_gimp.Gimp

    def test_stubs_are_gone_after_the_fixture(self):
        import fake_gimp as harness

        assert not harness.is_installed()
        assert gimpbridge.gimp_available() is False

    def test_buffer_converts_formats_like_babl(self, fake_gimp):
        rect = fake_gimp.Rectangle.new(0, 0, 2, 1)
        buffer = fake_gimp.Buffer(0, 0, 2, 1, "R'G'B'A u8")
        buffer.set(rect, "R'G'B'A u8", bytes([255, 0, 0, 255, 0, 0, 255, 128]))
        # alpha dropped, exactly as "ask babl for R'G'B' u8" does
        assert buffer.get(rect, 1.0, "R'G'B' u8", 1) == bytes([255, 0, 0, 0, 0, 255])

    def test_fake_image_projection_differs_from_either_layer(self, fake_image):
        assert fake_image.get_width() == 24
        assert fake_image.get_height() == 16
        assert [l.get_name() for l in fake_image.get_layers()] == ["patch", "backdrop"]


# --------------------------------------------------------------------------- #
# IN: reading pixels
# --------------------------------------------------------------------------- #
def _pixel(pixels, width, x, y):
    off = (y * width + x) * 3
    return tuple(pixels[off : off + 3])


class TestReadUploadPixels:
    def test_projection_is_what_the_user_sees(self, fake_gimp, fake_image):
        result = gimpbridge.read_upload_pixels(fake_image)

        assert isinstance(result, gimpbridge.UploadedImage)
        assert result.source == gimpbridge.SOURCE_PROJECTION
        assert (result.width, result.height) == (24, 16)
        assert len(result.pixels) == 24 * 16 * 3 == result.geometry.expected_payload_size()

        # backdrop shows through outside the patch...
        assert _pixel(result.pixels, 24, 0, 0) == (64, 64, 64)
        assert _pixel(result.pixels, 24, 23, 15) == (64, 64, 64)
        # ...and the patch (6x4 at 4,3) covers its own rectangle.
        assert _pixel(result.pixels, 24, 4, 3) == (255, 0, 0)
        assert _pixel(result.pixels, 24, 9, 6) == (255, 0, 0)
        assert _pixel(result.pixels, 24, 10, 6) == (64, 64, 64)
        assert _pixel(result.pixels, 24, 4, 2) == (64, 64, 64)

    def test_geometry_records_the_identity_when_nothing_is_scaled(
        self, fake_gimp, fake_image
    ):
        geometry = gimpbridge.read_upload_pixels(fake_image).geometry
        assert geometry.source_width == 24 and geometry.source_height == 16
        assert geometry.scale_x == 1.0 and geometry.scale_y == 1.0

    def test_a_hidden_layer_is_not_in_the_projection(self, fake_gimp, fake_image):
        fake_image.get_layers()[0].set_visible(False)
        result = gimpbridge.read_upload_pixels(fake_image)
        assert _pixel(result.pixels, 24, 5, 4) == (64, 64, 64)

    def test_the_temporary_work_image_is_always_deleted(self, fake_gimp, fake_image):
        before = fake_gimp.live_images()
        gimpbridge.read_upload_pixels(fake_image)
        assert fake_gimp.live_images() == before

    def test_the_users_image_is_untouched(self, fake_gimp, fake_image):
        layers_before = [l.get_name() for l in fake_image.get_layers()]
        gimpbridge.read_upload_pixels(fake_image)
        assert [l.get_name() for l in fake_image.get_layers()] == layers_before
        assert fake_image.undo_depth == 0

    def test_layer_only_source_honours_layer_offsets_and_background(
        self, fake_gimp, fake_image
    ):
        patch = fake_image.get_layers()[0]
        result = gimpbridge.read_upload_pixels(
            fake_image,
            source=gimpbridge.SOURCE_LAYER,
            drawable=patch,
            background=(255, 255, 255),
        )
        assert result.source == gimpbridge.SOURCE_LAYER
        # the layer keeps its image-space position...
        assert _pixel(result.pixels, 24, 4, 3) == (255, 0, 0)
        assert _pixel(result.pixels, 24, 9, 6) == (255, 0, 0)
        # ...and everything else is the chosen background, not the backdrop.
        assert _pixel(result.pixels, 24, 0, 0) == (255, 255, 255)
        assert _pixel(result.pixels, 24, 3, 3) == (255, 255, 255)

    def test_layer_only_defaults_to_the_selected_layer(self, fake_gimp, fake_image):
        result = gimpbridge.read_upload_pixels(
            fake_image, source=gimpbridge.SOURCE_LAYER, background=(0, 0, 0)
        )
        assert _pixel(result.pixels, 24, 5, 4) == (255, 0, 0)
        assert _pixel(result.pixels, 24, 0, 0) == (0, 0, 0)

    def test_background_choice_is_respected(self, fake_gimp, fake_image):
        patch = fake_image.get_layers()[0]
        result = gimpbridge.read_upload_pixels(
            fake_image,
            source=gimpbridge.SOURCE_LAYER,
            drawable=patch,
            background=(0, 128, 0),
        )
        assert _pixel(result.pixels, 24, 0, 0) == (0, 128, 0)

    def test_projection_uses_the_cheap_new_from_visible_path(
        self, fake_gimp, fake_image
    ):
        del fake_gimp.Gimp.calls[:]
        gimpbridge.read_upload_pixels(fake_image)
        names = [name for name, _ in fake_gimp.Gimp.calls]
        assert "Layer.new_from_visible" in names
        assert "Image.duplicate" not in names

    def test_projection_falls_back_to_duplicate_and_flatten(
        self, fake_gimp, fake_image, monkeypatch
    ):
        # VERIFY item 3 in gimpbridge's docstring: if Gimp.Layer has no
        # new_from_visible (or its signature differs), the bridge must still
        # produce the same pixels via image.duplicate() + flatten().
        monkeypatch.delattr(fake_gimp.Gimp.Layer, "new_from_visible")
        del fake_gimp.Gimp.calls[:]
        before = fake_gimp.live_images()

        result = gimpbridge.read_upload_pixels(fake_image)

        names = [name for name, _ in fake_gimp.Gimp.calls]
        assert "Image.duplicate" in names
        assert _pixel(result.pixels, 24, 0, 0) == (64, 64, 64)
        assert _pixel(result.pixels, 24, 5, 4) == (255, 0, 0)
        assert fake_gimp.live_images() == before

    def test_rejects_an_unknown_source(self, fake_gimp, fake_image):
        with pytest.raises(ValueError):
            gimpbridge.read_upload_pixels(fake_image, source="drawable")

    def test_a_layer_mask_reads_as_its_layer_not_its_grey_pixels(self, fake_gimp, fake_image):
        """With a mask in edit mode GIMP hands the plug-in the *mask* as its
        drawable; read as "the layer", the upload was the mask's grey."""
        stubs = fake_gimp
        patch = fake_image.get_layers()[0]
        mask = patch.create_mask(stubs.AddMaskType.WHITE)
        patch.add_mask(mask)
        drawable = fake_image.get_selected_drawables()[0]
        assert isinstance(drawable, stubs.Gimp.LayerMask), "edit-mask is on after add_mask"
        result = gimpbridge.read_upload_pixels(
            fake_image, source=gimpbridge.SOURCE_LAYER, drawable=drawable,
            background=(0, 0, 0))
        assert _pixel(result.pixels, 24, 5, 4) == (255, 0, 0)

    def test_a_selected_channel_falls_back_to_the_top_layer(self, fake_gimp, fake_image):
        stubs = fake_gimp
        channel = stubs.Channel.new(fake_image, "saved", 24, 16, 100.0, stubs.Color.new("black"))
        fake_image.insert_channel(channel, None, 0)
        drawable = fake_image.get_selected_drawables()[0]
        assert drawable is channel and fake_image.get_selected_layers() == []
        assert gimpbridge.source_layer(fake_image, drawable) is fake_image.get_layers()[0]
        result = gimpbridge.read_upload_pixels(
            fake_image, source=gimpbridge.SOURCE_LAYER, drawable=drawable,
            background=(0, 0, 0))
        assert _pixel(result.pixels, 24, 5, 4) == (255, 0, 0)

    def test_source_layer_keeps_a_layer_and_maps_a_mask(self, fake_gimp, fake_image):
        stubs = fake_gimp
        backdrop = fake_image.get_layers()[1]
        assert gimpbridge.source_layer(fake_image, backdrop) is backdrop
        mask = backdrop.create_mask(stubs.AddMaskType.WHITE)
        backdrop.add_mask(mask)
        assert gimpbridge.source_layer(fake_image, mask) is backdrop
        assert gimpbridge.source_layer(fake_image, None) is fake_image.get_selected_layers()[0]

    @pytest.mark.parametrize("source", [gimpbridge.SOURCE_PROJECTION, gimpbridge.SOURCE_LAYER])
    def test_a_failed_read_leaves_no_work_image_behind(
            self, fake_gimp, fake_image, monkeypatch, source):
        stubs = fake_gimp

        def refuse(*_a, **_k):
            raise RuntimeError("the PDB call failed")

        name = "new_from_visible" if source == gimpbridge.SOURCE_PROJECTION else "new_from_drawable"
        monkeypatch.setattr(stubs.Gimp.Layer, name, staticmethod(refuse))
        before = stubs.live_images()
        with pytest.raises(RuntimeError):
            gimpbridge.read_upload_pixels(fake_image, source=source)
        assert stubs.live_images() == before

    def test_downscaling_goes_through_gimps_scaler(self, fake_gimp):
        stubs = fake_gimp
        image = stubs.Image.new(2016, 1008, stubs.ImageBaseType.RGB)
        layer = stubs.Layer.new(
            image, "flat", 2016, 1008, stubs.ImageType.RGB_IMAGE, 100.0,
            stubs.LayerMode.NORMAL,
        )
        layer.fill_bytes(bytes((10, 20, 30)) * (2016 * 1008))
        image.insert_layer(layer, None, 0)
        del stubs.Gimp.calls[:]

        result = gimpbridge.read_upload_pixels(image)

        assert (result.width, result.height) == (1008, 504)
        assert len(result.pixels) == 1008 * 504 * 3
        assert result.geometry.scale_x == 0.5 and result.geometry.scale_y == 0.5
        # The resampling must be GIMP's, never a Python loop in the plug-in.
        assert ("Image.scale", (1008, 504)) in stubs.Gimp.calls
        assert any(name == "context_set_interpolation" for name, _ in stubs.Gimp.calls)
        assert _pixel(result.pixels, 1008, 500, 250) == (10, 20, 30)


# --------------------------------------------------------------------------- #
# selection as a box prompt
# --------------------------------------------------------------------------- #
class TestSelectionBox:
    def test_no_selection_is_none(self, fake_gimp, fake_image):
        assert gimpbridge.selection_box(fake_image) is None

    def test_bounds_in_image_space(self, fake_gimp, fake_image):
        fake_image.set_selection_rect(4, 3, 6, 4)
        assert gimpbridge.selection_box(fake_image) == (4.0, 3.0, 10.0, 7.0)

    def test_bounds_converted_to_uploaded_image_space(self, fake_gimp, fake_image):
        fake_image.set_selection_rect(4, 3, 6, 4)
        geometry = gimpbridge.UploadGeometry(24, 16, 12, 8)
        assert gimpbridge.selection_box(fake_image, geometry) == (2.0, 1.5, 5.0, 3.5)


# --------------------------------------------------------------------------- #
# OUT: masks back into GIMP
# --------------------------------------------------------------------------- #
class TestScaleSoftMask:
    def test_a_constant_mask_survives_upscaling(self, fake_gimp):
        mask = b"\xc8" * 16  # 4x4 of 200
        out = gimpbridge.scale_soft_mask(mask, 4, 4, 8, 8)
        assert len(out) == 64
        assert set(out) == {200}

    def test_scaling_is_delegated_to_gimp(self, fake_gimp):
        del fake_gimp.Gimp.calls[:]
        gimpbridge.scale_soft_mask(b"\x80" * 9, 3, 3, 12, 12)
        assert ("Image.scale", (12, 12)) in fake_gimp.Gimp.calls

    def test_the_temporary_image_is_deleted(self, fake_gimp):
        before = fake_gimp.live_images()
        gimpbridge.scale_soft_mask(b"\x80" * 9, 3, 3, 12, 12)
        assert fake_gimp.live_images() == before

    def test_downscaling_a_blocky_mask(self, fake_gimp):
        # 4x4, left half 0 and right half 255 -> halving keeps the split.
        mask = bytes([0, 0, 255, 255] * 4)
        out = gimpbridge.scale_soft_mask(mask, 4, 4, 2, 2)
        assert out == bytes([0, 255, 0, 255])

    def test_length_is_validated(self, fake_gimp):
        with pytest.raises(ValueError):
            gimpbridge.scale_soft_mask(b"\x00" * 5, 2, 2, 4, 4)

    def test_destination_size_is_validated(self, fake_gimp):
        with pytest.raises(ValueError):
            gimpbridge.scale_soft_mask(b"\x00" * 4, 2, 2, 0, 4)


class TestBuildMaskChannel:
    def _channel_pixel(self, channel, x, y):
        return channel.pixel(x, y)[0]

    def test_mask_lands_at_the_placement_and_nowhere_else(self, fake_gimp, fake_image):
        placement = gimpbridge.MaskPlacement(4, 3, 6, 4)
        channel = gimpbridge.build_mask_channel(
            fake_image, b"\xff" * 6, 3, 2, placement, name="bus"
        )

        assert channel.get_width() == 24 and channel.get_height() == 16
        assert self._channel_pixel(channel, 4, 3) == 255
        assert self._channel_pixel(channel, 9, 6) == 255
        assert self._channel_pixel(channel, 3, 3) == 0
        assert self._channel_pixel(channel, 10, 6) == 0
        assert self._channel_pixel(channel, 0, 0) == 0
        assert self._channel_pixel(channel, 23, 15) == 0

    def test_channel_is_inserted_by_default(self, fake_gimp, fake_image):
        channel = gimpbridge.build_mask_channel(
            fake_image, b"\xff" * 4, 2, 2, gimpbridge.MaskPlacement(0, 0, 4, 4)
        )
        assert channel in fake_image.get_channels()
        assert channel.get_name() == "SAM 3 mask"

    def test_insert_can_be_declined(self, fake_gimp, fake_image):
        channel = gimpbridge.build_mask_channel(
            fake_image,
            b"\xff" * 4,
            2,
            2,
            gimpbridge.MaskPlacement(0, 0, 4, 4),
            insert=False,
        )
        assert channel not in fake_image.get_channels()

    def test_soft_values_are_preserved_when_no_threshold_is_given(
        self, fake_gimp, fake_image
    ):
        channel = gimpbridge.build_mask_channel(
            fake_image, bytes([90] * 4), 2, 2, gimpbridge.MaskPlacement(2, 2, 4, 4)
        )
        assert self._channel_pixel(channel, 3, 3) == 90

    def test_threshold_is_applied_after_scaling(self, fake_gimp, fake_image):
        mask = bytes([100, 200, 100, 200])  # 2x2
        channel = gimpbridge.build_mask_channel(
            fake_image,
            mask,
            2,
            2,
            gimpbridge.MaskPlacement(0, 0, 4, 4),
            threshold=128,
        )
        assert self._channel_pixel(channel, 0, 0) == 0
        assert self._channel_pixel(channel, 3, 0) == 255

    def test_a_placement_touching_the_edge_is_clipped_not_wrapped(
        self, fake_gimp, fake_image
    ):
        placement = gimpbridge.MaskPlacement(22, 14, 2, 2)
        channel = gimpbridge.build_mask_channel(
            fake_image, b"\xff" * 4, 2, 2, placement, insert=False
        )
        assert self._channel_pixel(channel, 23, 15) == 255
        assert self._channel_pixel(channel, 21, 15) == 0

    def test_no_temporary_images_leak(self, fake_gimp, fake_image):
        before = fake_gimp.live_images()
        gimpbridge.build_mask_channel(
            fake_image, b"\xff" * 6, 3, 2, gimpbridge.MaskPlacement(4, 3, 6, 4)
        )
        assert fake_gimp.live_images() == before


class TestWriteSoftMask:
    def test_writes_into_a_canvas_aligned_channel(self, fake_gimp, fake_image):
        channel = fake_gimp.Channel.new(
            fake_image, "c", 24, 16, 100.0, fake_gimp.Color.new("black")
        )
        assert gimpbridge.write_soft_mask(
            channel, b"\xff" * 4, 2, 2, 5, 5, use_offsets=False
        )
        assert channel.pixel(5, 5)[0] == 255
        assert channel.pixel(4, 5)[0] == 0

    def test_layer_mask_offsets_are_undone(self, fake_gimp, fake_image):
        # The one place layer offsets matter: a layer mask is layer-sized and
        # sits at the layer's position, so image space must be translated.
        patch = fake_image.get_layers()[0]
        mask = patch.create_mask(fake_gimp.AddMaskType.BLACK)
        patch.add_mask(mask)
        assert gimpbridge.drawable_offsets(mask) == (4, 3)

        assert gimpbridge.write_soft_mask(mask, b"\xff" * 4, 2, 2, 5, 4)

        # image (5,4) is layer-local (1,1)
        assert mask.pixel(1, 1)[0] == 255
        assert mask.pixel(2, 2)[0] == 255
        assert mask.pixel(0, 0)[0] == 0

    def test_clips_against_the_drawable_bounds(self, fake_gimp, fake_image):
        channel = fake_gimp.Channel.new(
            fake_image, "c", 24, 16, 100.0, fake_gimp.Color.new("black")
        )
        assert gimpbridge.write_soft_mask(
            channel, b"\xff" * 36, 6, 6, 22, 14, use_offsets=False
        )
        assert channel.pixel(23, 15)[0] == 255
        assert channel.pixel(21, 15)[0] == 0

    def test_clips_negative_origins(self, fake_gimp, fake_image):
        channel = fake_gimp.Channel.new(
            fake_image, "c", 24, 16, 100.0, fake_gimp.Color.new("black")
        )
        assert gimpbridge.write_soft_mask(
            channel, b"\xff" * 16, 4, 4, -2, -2, use_offsets=False
        )
        assert channel.pixel(0, 0)[0] == 255
        assert channel.pixel(1, 1)[0] == 255
        assert channel.pixel(2, 2)[0] == 0

    def test_fully_outside_is_a_no_op(self, fake_gimp, fake_image):
        channel = fake_gimp.Channel.new(
            fake_image, "c", 24, 16, 100.0, fake_gimp.Color.new("black")
        )
        assert not gimpbridge.write_soft_mask(
            channel, b"\xff" * 4, 2, 2, 100, 100, use_offsets=False
        )
        assert set(channel.get_buffer().data) == {0}

    def test_length_is_validated(self, fake_gimp, fake_image):
        channel = fake_gimp.Channel.new(
            fake_image, "c", 24, 16, 100.0, fake_gimp.Color.new("black")
        )
        with pytest.raises(ValueError):
            gimpbridge.write_soft_mask(channel, b"\xff" * 3, 2, 2, 0, 0)


# --------------------------------------------------------------------------- #
# end to end: upload geometry -> daemon result frame -> channel
# --------------------------------------------------------------------------- #
class TestRoundTrip:
    def test_a_result_frame_instance_becomes_a_correctly_placed_channel(
        self, fake_gimp
    ):
        stubs = fake_gimp
        # A 480x320 image uploads unscaled; pretend the daemon squashed it onto
        # a 1008x1008 canvas, as the reference processor does.
        image = stubs.Image.new(480, 320, stubs.ImageBaseType.RGB)
        layer = stubs.Layer.new(
            image, "flat", 480, 320, stubs.ImageType.RGB_IMAGE, 100.0,
            stubs.LayerMode.NORMAL,
        )
        layer.fill_bytes(bytes((5, 5, 5)) * (480 * 320))
        image.insert_layer(layer, None, 0)

        uploaded = gimpbridge.read_upload_pixels(image)
        assert (uploaded.width, uploaded.height) == (480, 320)

        canvas_from_image = {
            "scale_x": 1008.0 / 480.0,
            "scale_y": 1008.0 / 320.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        }
        # A quarter-image box: uploaded (120,80)-(240,160) in canvas units.
        instance = {
            "instance_id": 0,
            "score": 0.87,
            "label": "yellow school bus",
            "bbox": [252, 252, 504, 504],
            "mask_width": 4,
            "mask_height": 4,
            "blob_offset": 0,
            "blob_length": 16,
        }
        channel = gimpbridge.instance_to_channel(
            image,
            instance,
            b"\xff" * 16,
            canvas_from_image,
            uploaded.geometry,
        )

        assert channel.get_name() == "yellow school bus 87%"
        assert channel in image.get_channels()
        # canvas 252 -> uploaded x 120 / y 80 ; canvas 504 -> 240 / 160
        assert channel.pixel(120, 80)[0] == 255
        assert channel.pixel(239, 159)[0] == 255
        assert channel.pixel(119, 80)[0] == 0
        assert channel.pixel(240, 160)[0] == 0

    def test_blob_length_mismatch_is_rejected(self, fake_gimp, fake_image):
        instance = {
            "bbox": [0, 0, 10, 10],
            "mask_width": 4,
            "mask_height": 4,
            "score": 0.5,
            "label": "x",
        }
        with pytest.raises(ValueError):
            gimpbridge.instance_to_channel(
                fake_image, instance, b"\xff" * 15, CFI_SQUASH,
                gimpbridge.UploadGeometry(24, 16, 24, 16),
            )
