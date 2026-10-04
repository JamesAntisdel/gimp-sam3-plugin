"""Tests for the mask -> GIMP-object output plumbing.

These run without GIMP: ``outputs.py`` is driven against a fake ``Gimp`` /
``Gegl`` pair defined below and installed with ``outputs.set_gimp_modules()``.

The fake is deliberately *not* a mock-with-assertions: its images, layers,
channels and selection hold real bytes, ``select_item`` really composites at the
item's offsets, and ``transform_scale`` really resamples.  That means placement
and layer-offset handling can be checked numerically rather than by asserting
that some method was called with some number -- which is exactly the class of
bug (a mask 40 px off because a layer sits at (120, 40)) these tests exist to
catch.  Call *ordering* is checked separately through ``image.log``.

The undo stack is modelled too, with the same :class:`UndoModel` the shared
``tests/fake_gimp`` stubs use: what is recorded while undo is thawed can be
undone and redone, and a recording that cannot be replayed -- a scratch
channel added with undo and removed without -- shows up in
``image.criticals`` the way it shows up as a CRITICAL in GIMP.

The fake here is local and output-shaped; ``TestSharedStubInterop`` checks the
seam with the shared stubs.  Nothing in ``outputs.py`` depends on either -- it
takes its GIMP module from ``set_gimp_modules()``.
"""

from __future__ import annotations

import itertools
import types

import pytest

import outputs
from fake_gimp.gimp_stubs import UndoModel
from outputs import (
    CanvasTransform,
    Instance,
    MaskResult,
    OutputError,
    OutputMode,
    OutputOptions,
    PostOps,
    SelectionOp,
    Space,
    apply_result,
    apply_threshold,
    fill_holes,
    filter_instances,
    instance_name,
    mask_area,
    threshold_table,
    unique_name,
)


# =========================================================================== #
# the fake GIMP
# =========================================================================== #
class FakeBuffer:
    """A Gegl.Buffer that accepts the 3-argument ``set(rect, format, data)``."""

    ARITIES = (2,)

    def __init__(self, drawable):
        self.drawable = drawable
        self.flushed = 0

    def set(self, rect, *args):
        if len(args) not in self.ARITIES:
            raise TypeError("Gegl.Buffer.set() takes %s extra arguments, got %d"
                            % (self.ARITIES, len(args)))
        if len(args) == 2:
            fmt, data = args
        else:  # (level, format, data, rowstride)
            _level, fmt, data, _stride = args
        assert fmt == "Y' u8", "masks must be written as %r, got %r" % ("Y' u8", fmt)
        self.drawable.write_rect(rect, bytes(data))

    def get(self, rect, scale=1.0, fmt=None, abyss=None):
        assert fmt in (None, "Y' u8"), fmt
        return self.drawable.read_rect(rect)

    def flush(self):
        self.flushed += 1


class FiveArgBuffer(FakeBuffer):
    """A build whose ``set`` only takes ``(rect, level, format, data, rowstride)``."""

    ARITIES = (4,)


class Rectangle:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.width, self.height = int(x), int(y), int(w), int(h)

    @staticmethod
    def new(x, y, w, h):
        return Rectangle(x, y, w, h)


class Color:
    def __init__(self, name):
        self.name = name

    @staticmethod
    def new(name):
        return Color(name)


_ids = itertools.count(1)


class FakeItem:
    buffer_class = FakeBuffer

    def __init__(self, image, name, width, height):
        self.image = image
        self.name = name
        self.width = int(width)
        self.height = int(height)
        self.offx = 0
        self.offy = 0
        self.visible = True
        self.parent = None
        self.pixels = bytearray(self.width * self.height)
        self.id = next(_ids)

    # -- item -------------------------------------------------------------- #
    def get_name(self):
        return self.name

    def set_name(self, name):
        """``gimp-item-set-name``: renaming an item in the image records an
        ``ITEM_RENAME`` step, which later replays through the item tree."""
        if name != self.name and self.image._is_attached(self):
            self.image._record_rename(self, self.image._is_attached, self.name, name,
                                      lambda value: setattr(self, "name", value))
        self.name = name

    def get_offsets(self):
        return (True, self.offx, self.offy)

    def set_offsets(self, x, y):
        self.offx, self.offy = int(x), int(y)

    def set_visible(self, visible):
        self.visible = bool(visible)

    def get_parent(self):
        return self.parent

    # -- drawable ---------------------------------------------------------- #
    def get_buffer(self):
        return self.buffer_class(self)

    def write_rect(self, rect, data):
        """Sub-rectangle writes in drawable coordinates, clipped like GEGL."""
        assert len(data) == rect.width * rect.height, "wrong byte count"
        assert 0 <= rect.x and 0 <= rect.y, "the plug-in must clip before writing"
        assert rect.x + rect.width <= self.width and rect.y + rect.height <= self.height, (
            "write outside the drawable: %r into %dx%d" % (
                (rect.x, rect.y, rect.width, rect.height), self.width, self.height))
        for row in range(rect.height):
            dst = (rect.y + row) * self.width + rect.x
            self.pixels[dst:dst + rect.width] = data[row * rect.width:(row + 1) * rect.width]
        self.image.log.append(("write", self.name, rect.x, rect.y, rect.width, rect.height))

    def fill(self, fill_type):
        """``gimp_drawable_fill``: TRANSPARENT means *the context background*
        with alpha dropped only where the drawable has alpha.  A channel has
        none and the default background is white, so this fills with 255 --
        the very thing that once turned every Apply into Select All."""
        self.pixels = bytearray(b"\xff" * (self.width * self.height))
        self.image.log.append(("fill", self.name, fill_type))

    def read_rect(self, rect):
        self.image.log.append(("read", self.name, rect.x, rect.y, rect.width, rect.height))
        out = bytearray(rect.width * rect.height)
        for row in range(rect.height):
            src = (rect.y + row) * self.width + rect.x
            out[row * rect.width:(row + 1) * rect.width] = self.pixels[src:src + rect.width]
        return bytes(out)

    def update(self, x, y, w, h):
        self.image.log.append(("update", self.name, x, y, w, h))

    def _resampled(self, nw, nh):
        out = bytearray(nw * nh)
        for yy in range(nh):
            sy = min(self.height - 1, int(yy * self.height / nh))
            for xx in range(nw):
                sx = min(self.width - 1, int(xx * self.width / nw))
                out[yy * nw + xx] = self.pixels[sy * self.width + sx]
        return out

    # -- transform --------------------------------------------------------- #
    def transform_scale(self, x0, y0, x1, y1):
        nw = max(1, int(round(x1 - x0)))
        nh = max(1, int(round(y1 - y0)))
        out = self._resampled(nw, nh)
        self.image.log.append(("transform_scale", self.name, int(round(x0)), int(round(y0)), nw, nh))
        if isinstance(self, FakeChannel):
            # Real GIMP: channels always clip a transform to their own bounds
            # and keep offset (0, 0) -- gimp_channel_get_clip() returns
            # GIMP_TRANSFORM_RESIZE_CLIP unconditionally.  The plug-in once
            # scaled a crop-sized scratch channel "into place" this way and
            # every Apply selected nothing; the fake now says so too.
            ox, oy = int(round(x0)), int(round(y0))
            kept = bytearray(self.width * self.height)
            for yy in range(self.height):
                sy = yy - oy
                if not (0 <= sy < nh):
                    continue
                for xx in range(self.width):
                    sx = xx - ox
                    if 0 <= sx < nw:
                        kept[yy * self.width + xx] = out[sy * nw + sx]
            self.pixels = kept
            return self
        self.pixels = out
        self.width, self.height = nw, nh
        self.offx, self.offy = int(round(x0)), int(round(y0))
        return self


class FakeChannel(FakeItem):
    def __init__(self, image, name, width, height, opacity=100.0):
        super().__init__(image, name, width, height)
        self.opacity = opacity

    @staticmethod
    def new(image, name, width, height, opacity, color):
        assert color is not None, "a channel colour must be supplied"
        return FakeChannel(image, name, width, height, opacity)

    def set_opacity(self, value):
        self.opacity = float(value)

    def copy(self):
        dup = FakeChannel(self.image, self.name + " copy", self.width, self.height,
                          self.opacity)
        dup.pixels = bytearray(self.pixels)
        return dup


class FakeLayerMask(FakeChannel):
    """``Gimp.LayerMask``: a channel owned by one layer.  Copying it gives a
    plain channel -- which is what "duplicate the layer" produced when the
    plug-in's drawable was a mask."""

    def __init__(self, image, name, width, height, layer=None):
        super().__init__(image, name, width, height)
        self.layer = layer


class FakeLayer(FakeItem):
    def __init__(self, image, name, width, height):
        super().__init__(image, name, width, height)
        self.mask = None
        self.edit_mask = False
        self.has_alpha = False

    @staticmethod
    def new(image, name, width, height, image_type, opacity, mode):
        assert image_type == ImageType.GRAY_IMAGE
        return FakeLayer(image, name, width, height)

    @staticmethod
    def from_mask(mask):
        """``gimp_layer_from_mask``."""
        return mask.layer

    def copy(self):
        dup = FakeLayer(self.image, self.name + " copy", self.width, self.height)
        dup.pixels = bytearray(self.pixels)
        dup.offx, dup.offy = self.offx, self.offy
        dup.has_alpha = self.has_alpha
        self.image.log.append(("copy", self.name))
        return dup

    def add_alpha(self):
        self.has_alpha = True

    def get_mask(self):
        return self.mask

    def remove_mask(self, mode):
        self.image.log.append(("remove_mask", self.name))
        mask, edit = self.mask, self.edit_mask
        self.mask = None
        self.edit_mask = False
        if mask is not None and self.image._is_attached(self):
            def undo():
                self.mask, self.edit_mask = mask, edit

            def redo():
                self.mask, self.edit_mask = None, False

            self.image._undo_record("layer-mask-remove", undo, redo)

    def create_mask(self, mask_type):
        assert mask_type == AddMaskType.SELECTION, "only SELECTION masks are used"
        mask = FakeLayerMask(self.image, self.name + " mask", self.width, self.height, self)
        sel = self.image.selection
        for y in range(self.height):
            iy = y + self.offy
            if not (0 <= iy < self.image.height):
                continue
            for x in range(self.width):
                ix = x + self.offx
                if 0 <= ix < self.image.width:
                    mask.pixels[y * self.width + x] = sel[iy * self.image.width + ix]
        self.image.log.append(("create_mask", self.name))
        return mask

    def add_mask(self, mask):
        """``gimp-layer-add-mask`` attaches the mask with edit-mask ON, so from
        now on the image's selected drawable is the mask, not the layer."""
        self.mask = mask
        mask.layer = self
        self.edit_mask = True
        self.image.log.append(("add_mask", self.name))
        if self.image._is_attached(self):
            def undo():
                self.mask, self.edit_mask = None, False

            def redo():
                self.mask, self.edit_mask = mask, True

            self.image._undo_record("layer-mask-add", undo, redo)


class FakeGroupLayer(FakeLayer):
    def __init__(self, image, name):
        super().__init__(image, name, image.width, image.height)
        self.children = []


class FakePath:
    def __init__(self, image, name):
        self.image = image
        self.name = name
        self.strokes = []
        self.visible = False   # gimp_path_new: hidden until shown

    def set_visible(self, visible):
        self.visible = bool(visible)

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def stroke_new_from_points(self, stroke_type, points, closed):
        self.strokes.append((stroke_type, list(points), bool(closed)))
        self.image.log.append(("stroke", self.name, len(points), closed))
        return len(self.strokes)


#: Every ``Gimp.Image.scale`` the fake performs, as ``(width, height)``; the
#: scratch image ``outputs._scale_gray`` uses is not the user's image, so its
#: own log is not reachable from a test.
SCALE_LOG = []


class FakeImage(UndoModel):
    def __init__(self, width=200, height=120):
        self.width = int(width)
        self.height = int(height)
        self.layers = []
        self.channels = []
        self.paths = []
        self.selection = bytearray(self.width * self.height)
        self.selected_layers = []
        self.selected_channels = []
        self.selected_paths = []
        self.undo_depth = 0
        self.undo_max_depth = 0
        self.undo_starts = 0
        self.undo_ends = 0
        self.log = []
        self._undo_setup()

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def scale(self, width, height):
        """``gimp_image_scale`` -- nearest neighbour on every drawable."""
        SCALE_LOG.append((int(width), int(height)))
        sx = float(width) / self.width
        sy = float(height) / self.height
        for item in list(self.layers) + list(self.channels):
            nw = max(1, int(round(item.width * sx)))
            nh = max(1, int(round(item.height * sy)))
            item.pixels = item._resampled(nw, nh)
            item.offx, item.offy = int(round(item.offx * sx)), int(round(item.offy * sy))
            item.width, item.height = nw, nh
        self.width, self.height = int(width), int(height)
        self.selection = bytearray(self.width * self.height)
        self.log.append(("scale", int(width), int(height)))

    def delete(self):
        self.log.append(("delete",))

    # -- undo -------------------------------------------------------------- #
    def undo_freeze(self):
        """gimp_image_undo_freeze: nothing pushed while frozen."""
        self.frozen = getattr(self, "frozen", 0) + 1
        self.log.append(("undo_freeze",))

    def undo_thaw(self):
        assert getattr(self, "frozen", 0) > 0, "undo_thaw without a matching freeze"
        self.frozen -= 1
        self.log.append(("undo_thaw",))

    def undo_group_start(self):
        self.undo_depth += 1
        self.undo_starts += 1
        self.undo_max_depth = max(self.undo_max_depth, self.undo_depth)
        self.log.append(("undo_group_start",))
        self._undo_group_opened()

    def undo_group_end(self):
        self.undo_depth -= 1
        self.undo_ends += 1
        self.log.append(("undo_group_end",))
        assert self.undo_depth >= 0, "undo_group_end without a matching start"
        self._undo_group_closed()

    def _undo_is_frozen(self):
        return getattr(self, "frozen", 0) > 0

    def _record_selection(self):
        """Every change to the selection mask pushes ``GIMP_UNDO_MASK``."""
        self._record_mask(lambda: self.selection,
                          lambda data: setattr(self, "selection", bytearray(data)))

    def _is_attached(self, item):
        if item in self.channels or item in self.layers:
            return True
        groups = [l for l in self.layers if isinstance(l, FakeGroupLayer)]
        if any(item in g.children for g in groups):
            return True
        owner = getattr(item, "layer", None)
        return owner is not None and owner.mask is item and self._is_attached(owner)

    def undo_state(self):
        """Everything an undo step may change, for before/after comparisons."""
        def layer_state(layer):
            children = [layer_state(c) for c in getattr(layer, "children", [])]
            return (layer.name, layer.mask is not None, tuple(children))

        return (bytes(self.selection),
                tuple(c.name for c in self.channels),
                tuple(layer_state(l) for l in self.layers),
                tuple(p.name for p in self.paths))

    # -- containers -------------------------------------------------------- #
    def _list_for(self, parent):
        return parent.children if isinstance(parent, FakeGroupLayer) else self.layers

    def insert_layer(self, layer, parent, position):
        target = self._list_for(parent)
        position = len(target) if position < 0 else min(position, len(target))
        target.insert(position, layer)
        layer.parent = parent
        self.log.append(("insert_layer", layer.get_name(),
                         parent.get_name() if parent else None, position))
        self._record_item_added("layer-add", lambda: target, layer, position,
                                self._is_attached)

    def remove_layer(self, layer):
        target = self._list_for(layer.parent)
        position = target.index(layer)
        target.remove(layer)
        self.log.append(("remove_layer", layer.get_name()))
        self._record_item_removed("layer-remove", lambda: target, layer, position,
                                  self._is_attached)
        if layer in self.selected_layers:
            self.selected_layers = [l for l in self.selected_layers if l is not layer]

    def _add_channel(self, channel, position):
        """``gimp_image_add_channel (..., push_undo=TRUE)``: recorded unless
        frozen, and the new channel becomes the selected item, which
        deselects every layer."""
        position = len(self.channels) if position < 0 else min(position, len(self.channels))
        self.channels.insert(position, channel)
        self._record_item_added("channel-add", lambda: self.channels, channel, position,
                                self._is_attached)
        self.selected_channels = [channel]
        self.selected_layers = []
        return position

    def insert_channel(self, channel, parent, position):
        position = self._add_channel(channel, position)
        self.log.append(("insert_channel", channel.get_name(), position))

    def remove_channel(self, channel):
        position = self.channels.index(channel)
        self.channels.remove(channel)
        self.log.append(("remove_channel", channel.get_name()))
        self._record_item_removed("channel-remove", lambda: self.channels, channel,
                                  position, self._is_attached)
        if channel in self.selected_channels:
            # gimp_image_unset_selected_channels restores the layer stack --
            # when there is one.  On the user's image there was not, so this
            # models the observed state: nothing selected, hence no ants.
            self.selected_channels = []

    def get_selected_channels(self):
        return list(self.selected_channels)

    def set_selected_channels(self, channels):
        """Selecting channels deselects every layer (``gimpimage.c``,
        ``gimp_image_selected_channels_notify``)..."""
        self.selected_channels = list(channels)
        if self.selected_channels:
            self.selected_layers = []
        self.log.append(("set_selected_channels", [c.get_name() for c in channels]))

    def set_selected_layers(self, layers):
        """...and selecting layers deselects every channel."""
        self.selected_layers = list(layers)
        if self.selected_layers:
            self.selected_channels = []
        self.log.append(("set_selected_layers", [l.get_name() for l in layers]))

    def get_selected_drawables(self):
        """``gimp_image_get_selected_drawables``: the selected channels if
        any; else the selected layers, with a lone layer whose mask is being
        edited answered as that mask."""
        if self.selected_channels:
            return list(self.selected_channels)
        drawables = list(self.selected_layers)
        if len(drawables) == 1 and drawables[0].mask is not None and drawables[0].edit_mask:
            drawables = [drawables[0].mask]
        return drawables

    def set_selected_paths(self, paths):
        self.selected_paths = list(paths)
        self.log.append(("set_selected_paths", [p.get_name() for p in paths]))

    def ants_visible(self):
        """gimp_selection_boundary draws only with a selected channel or layer."""
        return bool(self.selected_channels or self.selected_layers) and any(self.selection)

    def insert_path(self, path, parent, position):
        position = 0 if position < 0 else position
        self.paths.insert(position, path)
        self.log.append(("insert_path", path.get_name()))
        self._record_item_added("path-add", lambda: self.paths, path, position,
                                lambda item: item in self.paths)

    def remove_path(self, path):
        position = self.paths.index(path)
        self.paths.remove(path)
        self.log.append(("remove_path", path.get_name()))
        self._record_item_removed("path-remove", lambda: self.paths, path, position,
                                  lambda item: item in self.paths)

    def get_layers(self):
        return list(self.layers)

    def get_selected_layers(self):
        return list(self.selected_layers)

    def get_item_position(self, item):
        target = self._list_for(item.parent)
        return target.index(item) if item in target else 0

    # -- selection --------------------------------------------------------- #
    def get_selection(self):
        return self.selection

    def select_item(self, op, item):
        self._record_selection()
        placed = bytearray(self.width * self.height)
        for y in range(item.height):
            iy = y + item.offy
            if not (0 <= iy < self.height):
                continue
            for x in range(item.width):
                ix = x + item.offx
                if 0 <= ix < self.width:
                    placed[iy * self.width + ix] = item.pixels[y * item.width + x]
        sel = self.selection
        if op == ChannelOps.REPLACE:
            self.selection = placed
        elif op == ChannelOps.ADD:
            self.selection = bytearray(max(a, b) for a, b in zip(sel, placed))
        elif op == ChannelOps.SUBTRACT:
            self.selection = bytearray(max(0, a - b) for a, b in zip(sel, placed))
        elif op == ChannelOps.INTERSECT:
            self.selection = bytearray(min(a, b) for a, b in zip(sel, placed))
        else:  # pragma: no cover
            raise AssertionError("unknown channel op %r" % (op,))
        self.log.append(("select_item", op, item.get_name()))
        self.select_frozen = getattr(self, "select_frozen", []) + [getattr(self, "frozen", 0)]


# -- module-level fakes ------------------------------------------------------ #
class ChannelOps:
    REPLACE, ADD, SUBTRACT, INTERSECT = "replace", "add", "subtract", "intersect"


class AddMaskType:
    SELECTION = "selection"
    WHITE = "white"


class MaskApplyMode:
    DISCARD = "discard"


class InterpolationType:
    CUBIC = "cubic"


class ImageBaseType:
    RGB, GRAY, INDEXED = "rgb", "gray", "indexed"


class ImageType:
    GRAY_IMAGE = "gray-image"
    RGBA_IMAGE = "rgba-image"


class LayerMode:
    NORMAL = "normal"


class FillType:
    FOREGROUND, BACKGROUND, WHITE, TRANSPARENT, PATTERN = range(5)


class TransformResize:
    ADJUST = "adjust"


class PathStrokeType:
    BEZIER = "bezier"


def _make_gimp(*, group_takes_name=True):
    """Build the fake ``Gimp`` module object."""
    calls = []

    class Selection:
        @staticmethod
        def none(image):
            image._record_selection()
            image.selection = bytearray(image.width * image.height)
            image.log.append(("selection_none",))

        @staticmethod
        def save(image):
            """``gimp-selection-save``: a new hidden channel, added to the
            image *with undo recorded* (``selection-cmds.c``)."""
            ch = FakeChannel(image, "Selection Mask copy", image.width, image.height)
            ch.pixels = bytearray(image.selection)
            ch.visible = False
            image._add_channel(ch, 0)
            image.log.append(("selection_save",))
            return ch

        @staticmethod
        def load(channel):
            channel.image.select_item(ChannelOps.REPLACE, channel)

        @staticmethod
        def grow(image, steps):
            image._record_selection()
            image.log.append(("grow", steps))

        @staticmethod
        def shrink(image, steps):
            image._record_selection()
            image.log.append(("shrink", steps))

        @staticmethod
        def feather(image, radius):
            image._record_selection()
            image.log.append(("feather", radius))

        @staticmethod
        def sharpen(image):
            image._record_selection()
            image.log.append(("sharpen",))
            image.selection = bytearray(0 if v < 128 else 255 for v in image.selection)

        @staticmethod
        def all(image):
            image._record_selection()
            image.selection = bytearray(b"\xff" * (image.width * image.height))
            image.log.append(("selection_all",))

        @staticmethod
        def is_empty(image):
            return not any(image.selection)

        @staticmethod
        def bounds(image):
            box = selection_bbox(image)
            return (False, 0, 0, 0, 0) if box is None else (True,) + tuple(box)

    class Image:
        @staticmethod
        def new(width, height, base_type):
            assert base_type == ImageBaseType.GRAY, "scratch images are GRAY"
            return FakeImage(width, height)

    class GroupLayer:
        @staticmethod
        def new(image, name=None):
            if name is None:
                return FakeGroupLayer(image, "Layer Group")
            if not group_takes_name:
                raise TypeError("GroupLayer.new() takes 1 argument")
            return FakeGroupLayer(image, name)

    class Path:
        @staticmethod
        def new(image, name):
            return FakePath(image, name)

    mod = types.SimpleNamespace(
        Selection=Selection,
        Channel=FakeChannel,
        LayerMask=FakeLayerMask,
        Image=Image,
        Layer=FakeLayer,
        GroupLayer=GroupLayer,
        Path=Path,
        ChannelOps=ChannelOps,
        AddMaskType=AddMaskType,
        MaskApplyMode=MaskApplyMode,
        ImageBaseType=ImageBaseType,
        ImageType=ImageType,
        LayerMode=LayerMode,
        FillType=FillType,
        InterpolationType=InterpolationType,
        TransformResize=TransformResize,
        PathStrokeType=PathStrokeType,
        context_push=lambda: calls.append("context_push"),
        context_pop=lambda: calls.append("context_pop"),
        context_set_interpolation=lambda v: calls.append(("interp", v)),
        context_set_transform_resize=lambda v: calls.append(("resize", v)),
        displays_flush=lambda: calls.append("flush"),
        calls=calls,
    )
    return mod


def _make_gegl():
    return types.SimpleNamespace(Rectangle=Rectangle, Color=Color)


# =========================================================================== #
# fixtures
# =========================================================================== #
@pytest.fixture
def gimp():
    mod = _make_gimp()
    outputs.set_gimp_modules(mod, _make_gegl())
    try:
        yield mod
    finally:
        outputs.reset_gimp_modules()


@pytest.fixture
def image(gimp):
    img = FakeImage(200, 120)
    layer = FakeLayer(img, "Background", 200, 120)
    img.insert_layer(layer, None, 0)
    img.selected_layers = [layer]   # a freshly opened image has its layer selected
    img.log.clear()
    return img


def solid_mask(w, h, value=255):
    return bytes([value]) * (w * h)


def make_result(instances=None, *, uploaded=(100, 60), source=(200, 120),
                transform=None, prompt="car"):
    """A 100x60 upload squashed onto a 200x120 canvas, restored to a 200x120 image.

    scale is 2x on both axes, and the source is 2x the upload, so canvas pixels
    and image pixels happen to map 1:1 -- which keeps the expected rectangles in
    these tests readable.  Anisotropy and offsets get their own tests.
    """
    if instances is None:
        instances = [Instance(0, 0.93, "car", (10, 20, 30, 40), 20, 20, solid_mask(20, 20))]
    return MaskResult(
        instances=list(instances),
        canvas_from_image=transform or CanvasTransform(2.0, 2.0, 0.0, 0.0),
        uploaded=uploaded,
        source=source,
        prompt_text=prompt,
    )


def selection_bbox(image):
    """Bounding box of the non-zero selection, or None."""
    xs, ys = [], []
    for y in range(image.height):
        row = y * image.width
        for x in range(image.width):
            if image.selection[row + x]:
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs) + 1, max(ys) + 1)


# =========================================================================== #
# byte-level mask operations (no GIMP involved)
# =========================================================================== #
class TestThreshold:
    def test_hard_table_is_a_step_at_the_threshold(self):
        table = threshold_table(128, softness=0)
        assert table[127] == 0
        assert table[128] == 255
        assert table[255] == 255
        assert len(table) == 256

    def test_soft_table_ramps_around_the_threshold(self):
        table = threshold_table(128, softness=32)
        assert table[111] == 0
        assert table[145] == 255
        assert 100 < table[128] < 155, "the threshold itself must land mid-ramp"
        assert all(table[i] <= table[i + 1] for i in range(255)), "must be monotonic"

    def test_threshold_is_honoured_not_ignored(self):
        data = bytes([100, 150, 200])
        assert apply_threshold(data, 128, 0) == bytes([0, 255, 255])
        assert apply_threshold(data, 180, 0) == bytes([0, 0, 255])

    def test_out_of_range_thresholds_are_clamped(self):
        assert threshold_table(0, 0)[1] == 255
        assert threshold_table(999, 0)[254] == 0

    def test_soft_edges_survive_the_remap(self):
        """The point of the ramp: values near T stay grey, so GEGL upsamples a
        soft mask and the edge does not stair-step (API.md 9 step 4)."""
        remapped = apply_threshold(bytes(range(256)), 128, 64)
        greys = [v for v in remapped if 0 < v < 255]
        assert len(greys) > 30


class TestArea:
    def test_counts_pixels_at_or_above_threshold(self):
        assert mask_area(bytes([0, 127, 128, 255]), 128) == 2
        assert mask_area(bytes([0, 127, 128, 255]), 100) == 3

    def test_empty_mask(self):
        assert mask_area(bytes(64), 128) == 0


class TestFillHoles:
    def test_fills_an_interior_hole(self):
        w = h = 7
        data = bytearray([255] * (w * h))
        data[3 * w + 3] = 0
        filled = fill_holes(bytes(data), w, h, 128)
        assert filled[3 * w + 3] == 255

    def test_leaves_background_connected_to_the_border_alone(self):
        w = h = 7
        data = bytearray([0] * (w * h))
        for y in range(2, 5):
            for x in range(2, 5):
                data[y * w + x] = 255
        assert fill_holes(bytes(data), w, h, 128) == bytes(data)

    def test_a_bay_open_to_the_edge_is_not_a_hole(self):
        w = h = 7
        data = bytearray([255] * (w * h))
        for y in range(0, 4):
            data[y * w + 3] = 0  # a channel cut in from the top edge
        assert fill_holes(bytes(data), w, h, 128) == bytes(data)

    def test_solid_mask_is_returned_unchanged(self):
        data = solid_mask(5, 5)
        assert fill_holes(data, 5, 5, 128) == data

    def test_rejects_a_byte_count_that_does_not_match(self):
        with pytest.raises(ValueError):
            fill_holes(bytes(10), 4, 4, 128)


# =========================================================================== #
# placement -- API.md section 9
# =========================================================================== #
class TestPlacement:
    def test_inverts_the_reported_affine_and_the_client_downscale(self):
        result = MaskResult(
            instances=[],
            canvas_from_image=CanvasTransform(1008 / 504.0, 1008 / 336.0, 0.0, 0.0),
            uploaded=(504, 336),
            source=(2016, 1344),
        )
        inst = Instance(0, 0.5, "x", (504, 504, 1008, 1008), 504, 504, solid_mask(504, 504))
        # canvas 504 -> image 252 -> source 1008 ; canvas 504y -> image 168 -> source 672
        assert result.rect_for(inst) == (1008, 672, 1008, 672)

    def test_letterbox_offsets_are_honoured_not_hardcoded(self):
        """A processor that letterboxed would report non-zero offsets; a client
        that assumed the 1008x1008 squash would misplace every mask."""
        result = MaskResult(
            instances=[],
            canvas_from_image=CanvasTransform(1.0, 1.0, 100.0, 50.0),
            uploaded=(200, 200),
            source=(200, 200),
        )
        inst = Instance(0, 0.5, "x", (110, 60, 130, 80), 20, 20, solid_mask(20, 20))
        assert result.rect_for(inst) == (10, 10, 20, 20)

    def test_edges_round_independently_so_neighbours_tile(self):
        """Section 9 step 3 exists so adjacent instances do not leave a seam."""
        result = MaskResult(
            instances=[],
            canvas_from_image=CanvasTransform(1.0, 1.0, 0.0, 0.0),
            uploaded=(100, 100),
            source=(333, 100),
        )
        left = Instance(0, 1.0, "a", (0, 0, 50, 10), 50, 10, solid_mask(50, 10))
        right = Instance(1, 1.0, "b", (50, 0, 100, 10), 50, 10, solid_mask(50, 10))
        lx, _, lw, _ = result.rect_for(left)
        rx, _, rw, _ = result.rect_for(right)
        assert lx + lw == rx, "adjacent crops must tile without a gap"
        assert rx + rw == 333

    def test_degenerate_rect_is_at_least_one_pixel(self):
        result = MaskResult(
            instances=[],
            canvas_from_image=CanvasTransform(100.0, 100.0, 0.0, 0.0),
            uploaded=(10, 10),
            source=(10, 10),
        )
        inst = Instance(0, 1.0, "x", (0, 0, 1, 1), 1, 1, b"\xff")
        _, _, w, h = result.rect_for(inst)
        assert (w, h) == (1, 1)

    def test_point_mapping_matches_rect_mapping(self):
        result = make_result()
        assert result.point_to_source(20, 40, Space.CANVAS) == (20.0, 40.0)
        assert result.point_to_source(20, 40, Space.SOURCE) == (20.0, 40.0)
        assert result.point_to_source(20, 40, Space.UPLOADED) == (40.0, 80.0)
        with pytest.raises(ValueError):
            result.point_to_source(0, 0, "moon")


# =========================================================================== #
# wire decoding
# =========================================================================== #
class TestInstanceDecoding:
    def test_from_frame_slices_the_blob_region_in_order(self):
        blob = b"\x10" * 4 + b"\x20" * 9
        header = {
            "image": {"width": 100, "height": 60},
            "canvas_from_image": {"scale_x": 2.0, "scale_y": 2.0,
                                  "offset_x": 0.0, "offset_y": 0.0},
            "prompt": {"kind": "text", "text": "car"},
            "request_id": "r-1",
            "instances": [
                {"instance_id": 0, "score": 0.9, "label": "car", "bbox": [0, 0, 2, 2],
                 "mask_width": 2, "mask_height": 2, "blob_offset": 0, "blob_length": 4},
                {"instance_id": 1, "score": 0.5, "label": "car", "bbox": [2, 2, 5, 5],
                 "mask_width": 3, "mask_height": 3, "blob_offset": 4, "blob_length": 9},
            ],
        }
        result = MaskResult.from_frame(header, blob, source=(200, 120))
        assert [i.instance_id for i in result.instances] == [0, 1]
        assert result.instances[0].mask == b"\x10" * 4
        assert result.instances[1].mask == b"\x20" * 9
        assert result.prompt_text == "car"
        assert result.uploaded == (100, 60)

    def test_a_slice_past_the_end_of_the_blob_is_an_error(self):
        header = {
            "image": {"width": 10, "height": 10},
            "canvas_from_image": {},
            "instances": [{"instance_id": 0, "score": 1.0, "bbox": [0, 0, 4, 4],
                           "mask_width": 4, "mask_height": 4,
                           "blob_offset": 0, "blob_length": 16}],
        }
        with pytest.raises(ValueError):
            MaskResult.from_frame(header, b"\x00" * 4, source=(10, 10))

    def test_mask_byte_count_must_match_the_bbox(self):
        with pytest.raises(ValueError):
            Instance(0, 1.0, "x", (0, 0, 4, 4), 4, 4, b"\x00" * 5)


# =========================================================================== #
# naming
# =========================================================================== #
class TestNaming:
    def test_reads_like_something_a_person_typed(self):
        inst = Instance(0, 0.9342, "car", (0, 0, 2, 2), 2, 2, solid_mask(2, 2))
        assert instance_name(inst) == "car (0.93)"

    def test_pvs_instances_fall_back_to_the_prompt(self):
        inst = Instance(0, 0.87, "", (0, 0, 2, 2), 2, 2, solid_mask(2, 2))
        assert instance_name(inst, "click point") == "click point (0.87)"
        assert instance_name(inst, "") == "selection (0.87)"

    def test_prefix(self):
        inst = Instance(0, 0.5, "dog", (0, 0, 2, 2), 2, 2, solid_mask(2, 2))
        assert instance_name(inst, prefix="sam3 ") == "sam3 dog (0.50)"

    def test_duplicates_get_a_suffix(self):
        assert unique_name("car (0.93)", []) == "car (0.93)"
        assert unique_name("car (0.93)", ["car (0.93)"]) == "car (0.93) #2"
        assert unique_name("car (0.93)", ["car (0.93)", "car (0.93) #2"]) == "car (0.93) #3"


# =========================================================================== #
# undo grouping -- the thing that makes the plug-in bearable to use
# =========================================================================== #
class TestUndoGrouping:
    def test_exactly_one_undo_group_wraps_everything(self, image):
        result = make_result()
        apply_result(image, result, OutputOptions(mode=OutputMode.CHANNELS))
        assert image.undo_starts == 1
        assert image.undo_ends == 1
        assert image.undo_max_depth == 1
        assert image.undo_depth == 0

    def test_layer_groups_reuse_one_scratch_channel(self, image):
        result = make_result(instances=[
            Instance(0, 0.9, "car", (10, 20, 30, 40), 20, 20, solid_mask(20, 20)),
            Instance(1, 0.7, "car", (50, 20, 70, 40), 20, 20, solid_mask(20, 20)),
            Instance(2, 0.6, "car", (90, 20, 110, 40), 20, 20, solid_mask(20, 20)),
        ])
        applied = apply_result(image, result, OutputOptions(mode=OutputMode.LAYER_GROUPS))
        assert len(applied.layers) == 3
        assert [e for e in image.log if e[0] == "insert_channel"] == [
            ("insert_channel", "sam3-scratch", 0)]
        # each mask carries only its own instance: the previous rectangle was zeroed
        for inst, layer in zip(result.instances, applied.layers):
            x, y, w, h = result.rect_for(inst)
            assert layer.mask.pixels[(y + h // 2) * image.width + x + w // 2] == 255
            others = [o for o in result.instances if o is not inst]
            for other in others:
                ox, oy, ow, oh = result.rect_for(other)
                assert layer.mask.pixels[(oy + oh // 2) * image.width + ox + ow // 2] == 0
        assert image.channels == []

    def test_every_mutation_happens_inside_the_group(self, image):
        result = make_result(instances=[
            Instance(0, 0.9, "car", (10, 20, 30, 40), 20, 20, solid_mask(20, 20)),
            Instance(1, 0.7, "car", (50, 20, 70, 40), 20, 20, solid_mask(20, 20)),
        ])
        apply_result(image, result, OutputOptions(mode=OutputMode.LAYER_GROUPS))
        names = [entry[0] for entry in image.log]
        first, last = names.index("undo_group_start"), names.index("undo_group_end")
        mutations = {"insert_layer", "insert_channel", "remove_channel", "select_item",
                     "add_mask", "create_mask", "selection_save", "transform_scale"}
        for i, name in enumerate(names):
            if name in mutations:
                assert first < i < last, "%r at %d escaped the undo group" % (name, i)

    def test_the_group_closes_even_when_the_apply_blows_up(self, image, gimp, monkeypatch):
        def boom(op, item):
            raise RuntimeError("GEGL had a bad day")

        monkeypatch.setattr(image, "select_item", boom)
        with pytest.raises(RuntimeError):
            apply_result(image, make_result(), OutputOptions(mode=OutputMode.SELECTION))
        assert image.undo_starts == 1 and image.undo_ends == 1
        assert image.undo_depth == 0, "a leaked undo group breaks every later Ctrl+Z"

    def test_the_context_is_popped_even_when_the_apply_blows_up(self, image, gimp, monkeypatch):
        monkeypatch.setattr(
            image, "select_item",
            lambda op, item: (_ for _ in ()).throw(RuntimeError("nope")))
        with pytest.raises(RuntimeError):
            apply_result(image, make_result(), OutputOptions(mode=OutputMode.SELECTION))
        assert gimp.calls.count("context_push") == gimp.calls.count("context_pop") == 1

    def test_scratch_channels_are_removed_even_when_the_apply_blows_up(self, image, gimp,
                                                                      monkeypatch):
        original = gimp.Selection.save
        state = {"n": 0}

        def flaky(image_):
            state["n"] += 1
            if state["n"] > 1:
                raise RuntimeError("out of memory")
            return original(image_)

        monkeypatch.setattr(gimp.Selection, "save", staticmethod(flaky))
        with pytest.raises(RuntimeError):
            # feather forces channels mode through the selection route, which
            # saves the selection per instance -- the second save blows up
            apply_result(image, make_result(), OutputOptions(
                mode=OutputMode.CHANNELS, post=PostOps(feather=1.0)))
        leftovers = [c.get_name() for c in image.channels if "sam3" in c.get_name()]
        assert leftovers == [], "temp channels leaked into the Channels dock: %r" % leftovers


# =========================================================================== #
# selection mode
# =========================================================================== #
class TestSelectionMode:
    def test_replace_places_the_mask_at_the_mapped_rectangle(self, image):
        apply_result(image, make_result(), OutputOptions(mode=OutputMode.SELECTION))
        assert selection_bbox(image) == (10, 20, 30, 40)

    def test_selection_is_the_union_of_several_instances(self, image):
        result = make_result(instances=[
            Instance(0, 0.9, "car", (10, 20, 30, 40), 20, 20, solid_mask(20, 20)),
            Instance(1, 0.8, "car", (100, 20, 120, 40), 20, 20, solid_mask(20, 20)),
        ])
        apply_result(image, result, OutputOptions(mode=OutputMode.SELECTION))
        assert selection_bbox(image) == (10, 20, 120, 40)
        assert image.selection[25 * image.width + 15] == 255
        assert image.selection[25 * image.width + 105] == 255
        assert image.selection[25 * image.width + 60] == 0

    def test_add_keeps_what_was_already_selected(self, image):
        for y in range(0, 10):
            for x in range(0, 10):
                image.selection[y * image.width + x] = 255
        apply_result(image, make_result(),
                     OutputOptions(mode=OutputMode.SELECTION, selection_op=SelectionOp.ADD))
        assert image.selection[5 * image.width + 5] == 255, "prior selection was lost"
        assert image.selection[25 * image.width + 15] == 255

    def test_subtract_takes_the_masks_out_of_the_prior_selection(self, image):
        for y in range(0, 60):
            for x in range(0, 60):
                image.selection[y * image.width + x] = 255
        apply_result(image, make_result(),
                     OutputOptions(mode=OutputMode.SELECTION, selection_op=SelectionOp.SUBTRACT))
        assert image.selection[25 * image.width + 15] == 0, "mask was not subtracted"
        assert image.selection[5 * image.width + 5] == 255, "the rest was not kept"

    def test_intersect_keeps_only_the_overlap(self, image):
        for y in range(0, 30):
            for x in range(0, 200):
                image.selection[y * image.width + x] = 255
        apply_result(image, make_result(),
                     OutputOptions(mode=OutputMode.SELECTION, selection_op=SelectionOp.INTERSECT))
        assert selection_bbox(image) == (10, 20, 30, 30)

    def test_replace_with_nothing_selected_clears(self, image):
        for i in range(len(image.selection)):
            image.selection[i] = 255
        applied = apply_result(image, make_result(instances=[]),
                               OutputOptions(mode=OutputMode.SELECTION))
        assert selection_bbox(image) is None
        assert applied.names == []

    def test_add_with_nothing_selected_leaves_the_user_alone(self, image):
        for y in range(0, 10):
            for x in range(0, 10):
                image.selection[y * image.width + x] = 255
        apply_result(image, make_result(instances=[]),
                     OutputOptions(mode=OutputMode.SELECTION, selection_op=SelectionOp.ADD))
        assert selection_bbox(image) == (0, 0, 10, 10)

    def test_no_channels_are_left_behind(self, image):
        apply_result(image, make_result(),
                     OutputOptions(mode=OutputMode.SELECTION, selection_op=SelectionOp.ADD))
        assert image.channels == []

    def test_the_mask_is_scaled_by_gegl_not_by_python(self, image):
        """A 20x20 canvas crop landing on a 40x40 image rect must go through
        Gimp.Image.scale on a scratch image -- the plug-in has no resampler of
        its own, and it must never be Item.transform_scale on a channel."""
        del SCALE_LOG[:]
        result = make_result(uploaded=(100, 60), source=(400, 240))
        apply_result(image, result, OutputOptions(mode=OutputMode.SELECTION))
        assert (40, 40) in SCALE_LOG, SCALE_LOG
        assert not [e for e in image.log if e[0] == "transform_scale"]

    def test_a_scaled_mask_selects_where_the_rectangle_says(self, image):
        """A scaled mask must land where its rectangle says, or the layer
        mask comes out blank.

        The crop (20x20 at canvas (10, 20)) lands on image rect (40, 80, 40, 40)
        after the x4 upscale.  Scaling a crop-sized scratch *channel* into
        place cannot work: GIMP clips channel transforms to the channel's own
        bounds at the origin, so nothing would land at (40, 80) and the
        selection -- and every mask, channel and layer built from it -- would
        be empty.  The fake channel clips the same way, so this is a real
        regression test, not a tautology.
        """
        result = make_result(uploaded=(100, 60), source=(400, 240))
        image = FakeImage(400, 240)
        image.layers.append(FakeLayer(image, "Background", 400, 240))
        x, y, w, h = result.rect_for(result.instances[0])
        assert (w, h) == (40, 40) and (x, y) != (0, 0), (x, y, w, h)
        inside = (y + h // 2) * 400 + (x + w // 2)
        for mode in (OutputMode.SELECTION, OutputMode.LAYER_MASKS):
            apply_result(image, result, OutputOptions(mode=mode))
            if mode == OutputMode.SELECTION:
                sel = image.selection
                assert sel[inside] == 255, "inside the rectangle must be selected"
                assert sel[5 * 400 + 5] == 0, "the origin must not be"
                assert selection_bbox(image) == (x, y, x + w, y + h), selection_bbox(image)
            else:
                mask = image.layers[0].mask
                assert mask is not None and mask.pixels[inside] == 255, (
                    "the layer mask must carry the selection")

    def test_apply_never_fills_a_channel_with_the_background_colour(self, image):
        """No solid *white* layer mask and marching ants around the whole
        image: Gimp.Drawable.fill(TRANSPARENT) on a channel is a fill with the
        (white) background colour."""
        apply_result(image, make_result(), OutputOptions(mode=OutputMode.SELECTION))
        assert not [e for e in image.log if e[0] == "fill"], "no Drawable.fill on scratch channels"
        x, y, w, h = make_result().rect_for(make_result().instances[0])
        assert selection_bbox(image) == (x, y, x + w, y + h), selection_bbox(image)
        assert image.selection[0] == 0 and image.selection[-1] == 0

    def test_a_channel_that_does_not_start_zeroed_is_zeroed(self, image):
        """GEGL zero-fills new buffers; a build that did not would otherwise
        select the whole image.  The check samples, then clears only if needed."""
        original_new = FakeChannel.__init__

        def dirty_new(self_, *a, **k):
            original_new(self_, *a, **k)
            self_.pixels = bytearray(b"\x7f" * (self_.width * self_.height))

        FakeChannel.__init__ = dirty_new
        try:
            apply_result(image, make_result(), OutputOptions(mode=OutputMode.SELECTION))
        finally:
            FakeChannel.__init__ = original_new
        x, y, w, h = make_result().rect_for(make_result().instances[0])
        assert selection_bbox(image) == (x, y, x + w, y + h), selection_bbox(image)

    def test_intersect_with_no_selection_keeps_the_result(self, image):
        """Intersect with no selection must not end with nothing selected.
        Intersect with an empty selection is empty by arithmetic; GIMP's
        convention is that no selection means everything, so the result must
        simply replace it."""
        applied = apply_result(image, make_result(), OutputOptions(
            mode=OutputMode.SELECTION, selection_op=SelectionOp.INTERSECT))
        x, y, w, h = make_result().rect_for(make_result().instances[0])
        assert selection_bbox(image) == (x, y, x + w, y + h), selection_bbox(image)
        assert "no selection" in applied.note

    def test_subtract_from_no_selection_selects_everything_else(self, image):
        applied = apply_result(image, make_result(), OutputOptions(
            mode=OutputMode.SELECTION, selection_op=SelectionOp.SUBTRACT))
        x, y, w, h = make_result().rect_for(make_result().instances[0])
        assert image.selection[0] == 255 and image.selection[-1] == 255
        assert image.selection[(y + h // 2) * image.width + x + w // 2] == 0
        assert "everything else" in applied.note

    def test_intersect_with_a_real_selection_still_intersects(self, image):
        x, y, w, h = make_result().rect_for(make_result().instances[0])
        for yy in range(y, y + h // 2):
            for xx in range(0, image.width):
                image.selection[yy * image.width + xx] = 255
        applied = apply_result(image, make_result(), OutputOptions(
            mode=OutputMode.SELECTION, selection_op=SelectionOp.INTERSECT))
        assert selection_bbox(image) == (x, y, x + w, y + h // 2), selection_bbox(image)
        assert applied.note == ""

    def test_every_stage_is_logged_with_the_selection_bounds(self, image):
        lines = []
        apply_result(image, make_result(), OutputOptions(mode=OutputMode.SELECTION),
                     on_log=lines.append)
        assert [l.split(": ")[1].split(" ->")[0] for l in lines] == [
            "start (1 instance(s), 0 dropped)", "mode done",
            "scratch channels removed", "item selection restored", "undo group closed"]
        x, y, w, h = make_result().rect_for(make_result().instances[0])
        assert lines[-1].endswith("selection %d,%d-%d,%d" % (x, y, x + w, y + h)), lines[-1]
        assert lines[0].endswith("selection empty")

    def test_instances_share_one_scratch_channel_and_one_select(self, image):
        """N image-sized channels and N select_items cost N x (channel + undo
        copy of the selection); on a big photo that was gigabytes."""
        result = make_result(instances=[
            Instance(0, 0.9, "car", (10, 20, 30, 40), 20, 20, solid_mask(20, 20)),
            Instance(1, 0.7, "car", (50, 20, 70, 40), 20, 20, solid_mask(20, 20)),
            Instance(2, 0.6, "car", (90, 20, 110, 40), 20, 20, solid_mask(20, 20)),
        ])
        apply_result(image, result, OutputOptions(mode=OutputMode.SELECTION))
        assert [e for e in image.log if e[0] == "insert_channel"] == [
            ("insert_channel", "sam3-scratch", 0)]
        assert [e for e in image.log if e[0] == "select_item"] == [
            ("select_item", ChannelOps.REPLACE, "sam3-scratch")]
        for inst in result.instances:
            x, y, w, h = result.rect_for(inst)
            assert image.selection[(y + h // 2) * image.width + x + w // 2] == 255
        assert image.channels == []

    def test_only_the_overlap_is_read_back(self, image):
        """Merging costs a read of the overlap only, not of each whole rectangle."""
        result = make_result(instances=[
            Instance(0, 0.9, "car", (10, 20, 30, 40), 20, 20, solid_mask(20, 20)),
            Instance(1, 0.8, "car", (25, 30, 45, 50), 20, 20, solid_mask(20, 20)),
            Instance(2, 0.7, "car", (100, 20, 120, 40), 20, 20, solid_mask(20, 20)),
        ])
        apply_result(image, result, OutputOptions(mode=OutputMode.SELECTION))
        reads = [e[2:] for e in image.log
                 if e[0] == "read" and e[1] == "sam3-scratch" and e[4] * e[5] > 1]
        assert reads == [(25, 30, 5, 10)], reads
        assert selection_bbox(image) == (10, 20, 120, 50)

    def test_max_bytes_is_a_bytewise_max(self):
        pairs = [(a, b) for a in range(256) for b in range(256)]
        left = bytes(a for a, _ in pairs)
        right = bytes(b for _, b in pairs)
        assert outputs._max_bytes(left, right) == bytes(map(max, left, right))
        assert outputs._max_bytes(b"", b"") == b""
        with pytest.raises(ValueError):
            outputs._max_bytes(b"\x00", b"")

    def test_overlapping_instances_union_rather_than_overwrite(self, image):
        """Adjacent objects have overlapping boxes.  The second mask's zero
        margin must not erase the first mask where the boxes overlap."""
        left = bytearray(20 * 20)
        for yy in range(20):
            left[yy * 20:yy * 20 + 10] = b"\xff" * 10         # left half solid
        right = bytearray(20 * 20)
        for yy in range(20):
            right[yy * 20 + 10:yy * 20 + 20] = b"\xff" * 10   # right half solid
        result = make_result(instances=[
            Instance(0, 0.9, "car", (10, 20, 30, 40), 20, 20, bytes(left)),
            Instance(1, 0.8, "car", (10, 20, 30, 40), 20, 20, bytes(right)),
        ])
        apply_result(image, result, OutputOptions(mode=OutputMode.SELECTION))
        x, y, w, h = result.rect_for(result.instances[0])
        row = (y + h // 2) * image.width
        assert image.selection[row + x + 2] == 255, "first instance survived"
        assert image.selection[row + x + w - 3] == 255, "second instance landed"
        assert selection_bbox(image) == (x, y, x + w, y + h)

    def test_the_marching_ants_survive_the_scratch_channel_removal(self, image):
        """The marching ants must still show after Apply, even though the
        mask itself is intact either way.  gimp_selection_boundary() draws an
        outline only when a channel or a layer is selected; inserting the
        scratch channel selects *it* (ants appear), removing it leaves nothing
        selected (ants gone).  The Layers/Channels selection from before Apply
        must come back."""
        before = list(image.get_selected_layers())
        assert before, "the fixture image starts with its layer selected"
        apply_result(image, make_result(), OutputOptions(mode=OutputMode.SELECTION))
        assert any(image.selection), "the mask itself was never the problem"
        assert image.get_selected_layers() == before
        assert image.ants_visible(), "no selected drawable means no outline in GIMP"

    def test_nothing_selected_before_apply_falls_back_to_the_source_layer(self, image):
        image.selected_layers = []
        apply_result(image, make_result(), OutputOptions(mode=OutputMode.SELECTION),
                     layer=image.layers[0])
        assert image.get_selected_layers() == [image.layers[0]]
        assert image.ants_visible()

    def test_a_selected_channel_is_restored_too(self, image):
        keep = FakeChannel(image, "user channel", image.width, image.height)
        image.channels.append(keep)
        image.selected_channels = [keep]
        image.selected_layers = []
        apply_result(image, make_result(), OutputOptions(mode=OutputMode.SELECTION))
        assert image.get_selected_channels() == [keep]
        assert image.ants_visible()

    def test_created_layers_end_up_selected(self, image):
        applied = apply_result(image, make_result(), OutputOptions(mode=OutputMode.LAYER_MASKS))
        assert applied.layers and image.get_selected_layers() == applied.layers

    @pytest.mark.parametrize("mode,op,post", [
        (OutputMode.SELECTION, SelectionOp.REPLACE, PostOps()),
        (OutputMode.SELECTION, SelectionOp.REPLACE, PostOps(grow=1)),
        (OutputMode.SELECTION, SelectionOp.ADD, PostOps()),
        (OutputMode.SELECTION, SelectionOp.SUBTRACT, PostOps()),
        (OutputMode.SELECTION, SelectionOp.INTERSECT, PostOps(feather=1.0)),
        (OutputMode.CHANNELS, SelectionOp.REPLACE, PostOps(feather=1.0)),
        (OutputMode.LAYER_MASKS, SelectionOp.REPLACE, PostOps()),
        (OutputMode.LAYER_GROUPS, SelectionOp.REPLACE, PostOps(grow=1)),
    ])
    def test_scratch_work_is_undo_frozen_and_the_selection_is_not(self, image, mode, op, post):
        """Inserting and removing the image-sized scratch channels must push no
        undo (that was gigabytes on a big photo), and neither must the
        selection while it is only a work surface.  Selection mode changes the
        user's selection in exactly one thawed select_item, its last; the
        modes that only borrow the selection change it in none."""
        for y in range(0, 30):
            for x in range(0, 60):
                image.selection[y * image.width + x] = 255
        result = make_result(instances=[
            Instance(0, 0.9, "car", (10, 20, 30, 40), 20, 20, solid_mask(20, 20)),
            Instance(1, 0.7, "car", (50, 20, 70, 40), 20, 20, solid_mask(20, 20)),
        ])
        apply_result(image, result, OutputOptions(mode=mode, selection_op=op, post=post))
        names = [e[0] for e in image.log]
        assert getattr(image, "frozen", 0) == 0, "freeze/thaw must balance"
        assert names.count("undo_freeze") == names.count("undo_thaw") >= 2
        thawed = [i for i, depth in enumerate(image.select_frozen) if depth == 0]
        if mode == OutputMode.SELECTION:
            assert thawed == [len(image.select_frozen) - 1], image.select_frozen
        else:
            assert thawed == [], image.select_frozen
        # every scratch channel is inserted and removed inside a frozen window
        for i, entry in enumerate(image.log):
            if entry[0] in ("insert_channel", "remove_channel") and entry[1].startswith("sam3-"):
                depth = names[:i].count("undo_freeze") - names[:i].count("undo_thaw")
                assert depth >= 1, (entry, depth)
        assert "sam3" not in " ".join(image.recorded()), image.recorded()

    def test_a_rectangle_hanging_off_the_image_is_clipped_not_refused(self, image):
        """Rounding can push a rect past the image edge; the visible part must
        still be written and GEGL must never be asked to write outside."""
        inst = Instance(0, 0.9, "car", (190, 110, 210, 130), 20, 20, solid_mask(20, 20))
        result = make_result(instances=[inst], uploaded=(100, 60), source=(400, 240))
        image = FakeImage(400, 240)
        apply_result(image, result, OutputOptions(mode=OutputMode.SELECTION))
        x, y, w, h = result.rect_for(inst)
        assert x + w > 400 and y + h > 240, "the test needs a rect past both edges"
        assert selection_bbox(image) == (x, y, 400, 240), selection_bbox(image)


# =========================================================================== #
# undo and redo -- what GIMP replays
# =========================================================================== #
_EVERY_APPLY = [
    (OutputMode.SELECTION, SelectionOp.REPLACE, PostOps()),
    (OutputMode.SELECTION, SelectionOp.REPLACE, PostOps(grow=1, feather=1.0)),
    (OutputMode.SELECTION, SelectionOp.ADD, PostOps()),
    (OutputMode.SELECTION, SelectionOp.SUBTRACT, PostOps()),
    (OutputMode.SELECTION, SelectionOp.INTERSECT, PostOps(smooth=1.0)),
    (OutputMode.CHANNELS, SelectionOp.REPLACE, PostOps()),
    (OutputMode.CHANNELS, SelectionOp.REPLACE, PostOps(feather=1.0)),
    (OutputMode.LAYER_MASKS, SelectionOp.REPLACE, PostOps()),
    (OutputMode.LAYER_GROUPS, SelectionOp.REPLACE, PostOps()),
    (OutputMode.PATHS, SelectionOp.REPLACE, PostOps()),
]


def _apply_id(case):
    mode, op, post = case
    return "%s-%s%s" % (mode, op, "-postops" if post.has_selection_ops else "")


class TestUndoAndRedo:
    """Edit > Undo, then Edit > Redo, must replay an Apply cleanly.

    ``Gimp.Selection.save`` adds its channel with undo recorded.  Scratch
    snapshots taken that way and removed with undo frozen left a recorded
    "channel added" step behind: Undo failed GIMP's ``gimp_item_is_attached``
    assertion trying to remove a channel that was already gone, and Redo put
    ``sam3-previous-selection`` and ``sam3-union`` back into the Channels dock.
    """

    def _prepare(self, image):
        for y in range(0, 30):
            for x in range(0, 60):
                image.selection[y * image.width + x] = 255
        return make_result(instances=[
            Instance(0, 0.9, "car", (10, 20, 30, 40), 20, 20, solid_mask(20, 20)),
            Instance(1, 0.7, "car", (50, 20, 70, 40), 20, 20, solid_mask(20, 20)),
        ])

    def _apply(self, image, result, mode, op, post, **kwargs):
        return apply_result(
            image, result, OutputOptions(mode=mode, selection_op=op, post=post, **kwargs),
            contours={0: [{"points": [0.0, 0.0, 4.0, 0.0, 4.0, 4.0]}],
                      1: [{"points": [50.0, 20.0, 54.0, 20.0, 54.0, 24.0]}]})

    @pytest.mark.parametrize("mode,op,post", _EVERY_APPLY, ids=[_apply_id(c) for c in _EVERY_APPLY])
    def test_undo_restores_and_redo_reapplies(self, image, mode, op, post):
        result = self._prepare(image)
        before = image.undo_state()
        steps = len(image.undo_steps)
        self._apply(image, result, mode, op, post)
        after = image.undo_state()
        assert after != before, "the Apply must have changed something"
        assert len(image.undo_steps) == steps + 1, "one Apply, one undo step"

        image.undo()
        assert image.undo_state() == before, "Undo must bring back the image as it was"
        image.redo()
        assert image.undo_state() == after, "Redo must bring back the Apply, and nothing else"
        assert image.criticals == [], image.criticals
        assert not [c.name for c in image.channels if c.name.startswith("sam3-")]

    @pytest.mark.parametrize("mode,op,post", _EVERY_APPLY, ids=[_apply_id(c) for c in _EVERY_APPLY])
    def test_a_changed_selection_is_recorded_exactly_once(self, image, mode, op, post):
        result = self._prepare(image)
        applied = self._apply(image, result, mode, op, post)
        recorded = image.recorded()
        expected = 1 if mode == OutputMode.SELECTION else 0
        assert recorded.count("mask") == expected, recorded
        assert recorded.count("channel-add") == len(applied.channels), recorded
        assert "channel-remove" not in recorded, recorded

    def test_a_kept_selection_is_one_recorded_change(self, image):
        """``restore_selection=False`` leaves the modes' last selection in
        place; that is then the one recorded selection change."""
        result = self._prepare(image)
        before = image.undo_state()
        self._apply(image, result, OutputMode.LAYER_MASKS, SelectionOp.REPLACE, PostOps(),
                    restore_selection=False)
        assert image.recorded().count("mask") == 1, image.recorded()
        x, y, w, h = result.rect_for(result.instances[0])
        assert image.selection[(y + h // 2) * image.width + x + w // 2] == 255
        assert image.selection[5 * image.width + 5] == 0, "the user's selection was replaced"
        image.undo()
        assert image.undo_state() == before
        image.redo()
        assert image.criticals == []


class TestTheDrawableIsNotAlwaysALayer:
    """GIMP hands a plug-in its selected *drawables*: the mask while a mask is
    being edited (and ``add_mask`` turns that on, so right after a Layer mask
    Apply), and the channels whenever a channel is selected."""

    @pytest.mark.parametrize("mode", [OutputMode.LAYER_MASKS, OutputMode.LAYER_GROUPS])
    def test_the_run_after_a_layer_mask_apply_works(self, image, mode):
        first = apply_result(image, make_result(), OutputOptions(mode=OutputMode.LAYER_MASKS))
        drawable = image.get_selected_drawables()[0]
        assert isinstance(drawable, FakeLayerMask), "the next run is handed the mask"
        applied = apply_result(image, make_result(), OutputOptions(mode=mode), layer=drawable)
        assert applied.layers
        assert all(type(layer) is FakeLayer for layer in applied.layers)
        assert ("copy", first.layers[0].name) in image.log, "copied from the mask's layer"

    def test_a_selected_channel_falls_back_to_a_layer(self, image):
        user = FakeChannel(image, "user channel", image.width, image.height)
        image.insert_channel(user, None, 0)
        drawable = image.get_selected_drawables()[0]
        assert drawable is user and image.get_selected_layers() == []
        applied = apply_result(image, make_result(), OutputOptions(mode=OutputMode.LAYER_MASKS),
                               layer=drawable)
        assert ("copy", "Background") in image.log
        assert type(applied.layers[0]) is FakeLayer and applied.layers[0].get_mask() is not None

    def test_selection_mode_restores_items_through_the_layer_not_the_mask(self, image):
        layer = image.layers[0]
        mask = layer.create_mask(AddMaskType.SELECTION)
        layer.add_mask(mask)
        image.selected_layers = []
        apply_result(image, make_result(), OutputOptions(mode=OutputMode.SELECTION), layer=mask)
        assert image.get_selected_layers() == [layer]

    def test_created_layers_win_over_a_remembered_channel(self, image):
        """Layers and channels cannot both be selected: restoring the old
        channel after selecting the new layers deselected the new layers."""
        user = FakeChannel(image, "user channel", image.width, image.height)
        image.insert_channel(user, None, 0)
        applied = apply_result(image, make_result(), OutputOptions(mode=OutputMode.LAYER_MASKS))
        assert image.get_selected_layers() == applied.layers
        assert image.get_selected_channels() == []


class TestAFailedApplyLeavesTheImageAsItWas:
    """A failure halfway put the selection back only on the success path, and
    left the outputs already made in the image."""

    def _user_selection(self, image):
        for y in range(0, 30):
            for x in range(0, 60):
                image.selection[y * image.width + x] = 255

    def _check(self, image, before):
        assert image.undo_state() == before
        assert not [c.name for c in image.channels if "sam3" in c.name]
        # what is left on the undo stack still replays cleanly
        while image.undo_steps:
            image.undo()
        while image.redo_steps:
            image.redo()
        assert image.criticals == []

    def test_layer_mask(self, image, monkeypatch):
        self._user_selection(image)
        before = image.undo_state()

        def boom(self_, mask_type):
            raise RuntimeError("no mask today")

        monkeypatch.setattr(FakeLayer, "create_mask", boom)
        with pytest.raises(RuntimeError):
            apply_result(image, make_result(), OutputOptions(mode=OutputMode.LAYER_MASKS))
        self._check(image, before)
        assert image.get_selected_layers() == [image.layers[0]]

    def test_layer_group_failing_on_the_second_instance(self, image, monkeypatch):
        self._user_selection(image)
        before = image.undo_state()
        real = FakeLayer.create_mask
        calls = []

        def second_fails(self_, mask_type):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("out of memory")
            return real(self_, mask_type)

        monkeypatch.setattr(FakeLayer, "create_mask", second_fails)
        result = make_result(instances=[
            Instance(0, 0.9, "car", (10, 20, 30, 40), 20, 20, solid_mask(20, 20)),
            Instance(1, 0.7, "car", (50, 20, 70, 40), 20, 20, solid_mask(20, 20)),
        ])
        with pytest.raises(RuntimeError):
            apply_result(image, result, OutputOptions(mode=OutputMode.LAYER_GROUPS))
        self._check(image, before)

    def test_channels_failing_on_the_second_instance(self, image, gimp, monkeypatch):
        self._user_selection(image)
        before = image.undo_state()
        real = gimp.Selection.save
        calls = []

        def third_fails(image_):
            calls.append(1)
            if len(calls) == 3:   # the user's selection, instance 0, instance 1
                raise RuntimeError("out of memory")
            return real(image_)

        monkeypatch.setattr(gimp.Selection, "save", staticmethod(third_fails))
        result = make_result(instances=[
            Instance(0, 0.9, "car", (10, 20, 30, 40), 20, 20, solid_mask(20, 20)),
            Instance(1, 0.7, "car", (50, 20, 70, 40), 20, 20, solid_mask(20, 20)),
        ])
        with pytest.raises(RuntimeError):
            apply_result(image, result, OutputOptions(mode=OutputMode.CHANNELS,
                                                      post=PostOps(feather=1.0)))
        self._check(image, before)

    def test_selection_combining_fails(self, image, monkeypatch):
        self._user_selection(image)
        before = image.undo_state()

        def boom(image_, post):
            raise RuntimeError("feather failed")

        monkeypatch.setattr(outputs, "_apply_selection_post_ops", boom)
        with pytest.raises(RuntimeError):
            apply_result(image, make_result(), OutputOptions(
                mode=OutputMode.SELECTION, selection_op=SelectionOp.ADD))
        self._check(image, before)


# =========================================================================== #
# post-ops
# =========================================================================== #
class TestPostOps:
    def test_selection_ops_run_in_the_documented_order(self, image):
        post = PostOps(grow=3, shrink=2, smooth=1.5, feather=4.0)
        apply_result(image, make_result(),
                     OutputOptions(mode=OutputMode.SELECTION, post=post))
        ops = [e for e in image.log if e[0] in ("grow", "shrink", "feather", "sharpen")]
        assert ops == [("grow", 3), ("shrink", 2), ("feather", 1.5), ("sharpen",),
                       ("feather", 4.0)]

    def test_nothing_runs_when_nothing_is_asked_for(self, image):
        apply_result(image, make_result(), OutputOptions(mode=OutputMode.SELECTION))
        assert [e for e in image.log if e[0] in ("grow", "shrink", "feather", "sharpen")] == []

    def test_post_ops_apply_once_to_the_union_in_selection_mode(self, image):
        result = make_result(instances=[
            Instance(0, 0.9, "car", (10, 20, 30, 40), 20, 20, solid_mask(20, 20)),
            Instance(1, 0.8, "car", (100, 20, 120, 40), 20, 20, solid_mask(20, 20)),
        ])
        apply_result(image, result,
                     OutputOptions(mode=OutputMode.SELECTION, post=PostOps(grow=2)))
        assert [e for e in image.log if e[0] == "grow"] == [("grow", 2)]

    def test_post_ops_apply_per_instance_in_channels_mode(self, image):
        result = make_result(instances=[
            Instance(0, 0.9, "car", (10, 20, 30, 40), 20, 20, solid_mask(20, 20)),
            Instance(1, 0.8, "car", (100, 20, 120, 40), 20, 20, solid_mask(20, 20)),
        ])
        apply_result(image, result,
                     OutputOptions(mode=OutputMode.CHANNELS, post=PostOps(grow=2)))
        assert [e for e in image.log if e[0] == "grow"] == [("grow", 2), ("grow", 2)]

    def test_threshold_is_applied_to_the_bytes_before_they_reach_gimp(self, image):
        soft = bytes([200] * 200 + [60] * 200)
        result = make_result(instances=[
            Instance(0, 0.9, "car", (10, 20, 30, 40), 20, 20, soft)])
        post = PostOps(threshold=128, edge_softness=0)
        apply_result(image, result, OutputOptions(mode=OutputMode.SELECTION, post=post))
        assert selection_bbox(image) == (10, 20, 30, 30), "the 60-valued half leaked in"

    def test_raising_the_threshold_shrinks_the_mask(self, image):
        soft = bytes([200] * 200 + [150] * 200)
        result = make_result(instances=[
            Instance(0, 0.9, "car", (10, 20, 30, 40), 20, 20, soft)])
        apply_result(image, result, OutputOptions(
            mode=OutputMode.SELECTION, post=PostOps(threshold=180, edge_softness=0)))
        assert selection_bbox(image) == (10, 20, 30, 30)

    def test_hole_fill_reaches_the_selection(self, image):
        data = bytearray([255] * 400)
        for y in range(8, 12):
            for x in range(8, 12):
                data[y * 20 + x] = 0
        result = make_result(instances=[
            Instance(0, 0.9, "car", (10, 20, 30, 40), 20, 20, bytes(data))])
        opts = OutputOptions(mode=OutputMode.SELECTION, post=PostOps(hole_fill=False))
        apply_result(image, result, opts)
        assert image.selection[(20 + 9) * image.width + (10 + 9)] == 0

        image.selection = bytearray(len(image.selection))
        opts = OutputOptions(mode=OutputMode.SELECTION, post=PostOps(hole_fill=True))
        apply_result(image, result, opts)
        assert image.selection[(20 + 9) * image.width + (10 + 9)] == 255

    def test_min_area_drops_specks_and_reports_them(self, image):
        big = Instance(0, 0.9, "car", (10, 20, 30, 40), 20, 20, solid_mask(20, 20))
        speck = Instance(1, 0.8, "car", (100, 20, 103, 23), 3, 3, solid_mask(3, 3))
        result = make_result(instances=[big, speck])
        applied = apply_result(image, result, OutputOptions(
            mode=OutputMode.CHANNELS, post=PostOps(min_area=100.0)))
        assert applied.dropped == [1]
        assert [c.get_name() for c in applied.channels] == ["car (0.90)"]

    def test_min_area_is_measured_in_original_image_pixels(self):
        """400 canvas px at a 4x upscale is 6400 source px, not 400."""
        inst = Instance(0, 0.9, "car", (10, 20, 30, 40), 20, 20, solid_mask(20, 20))
        result = make_result(instances=[inst], uploaded=(100, 60), source=(400, 240))
        kept, dropped = filter_instances(result, PostOps(min_area=1000.0).normalised())
        assert kept and not dropped
        kept, dropped = filter_instances(result, PostOps(min_area=10000.0).normalised())
        assert not kept and dropped == [0]

    def test_selected_instance_ids_filter_the_list(self):
        result = make_result(instances=[
            Instance(0, 0.9, "car", (10, 20, 30, 40), 20, 20, solid_mask(20, 20)),
            Instance(7, 0.8, "car", (50, 20, 70, 40), 20, 20, solid_mask(20, 20)),
        ])
        kept, _ = filter_instances(result, PostOps().normalised(), selected=[7])
        assert [i.instance_id for i in kept] == [7]


# =========================================================================== #
# channels
# =========================================================================== #
class TestChannelsMode:
    def test_one_named_channel_per_instance(self, image):
        result = make_result(instances=[
            Instance(0, 0.934, "car", (10, 20, 30, 40), 20, 20, solid_mask(20, 20)),
            Instance(1, 0.61, "car", (100, 20, 120, 40), 20, 20, solid_mask(20, 20)),
        ])
        applied = apply_result(image, result, OutputOptions(mode=OutputMode.CHANNELS))
        assert applied.names == ["car (0.93)", "car (0.61)"]
        assert [c.get_name() for c in image.channels] == ["car (0.61)", "car (0.93)"]

    def test_identical_names_are_disambiguated(self, image):
        result = make_result(instances=[
            Instance(0, 0.9, "car", (10, 20, 30, 40), 20, 20, solid_mask(20, 20)),
            Instance(1, 0.9, "car", (100, 20, 120, 40), 20, 20, solid_mask(20, 20)),
        ])
        applied = apply_result(image, result, OutputOptions(mode=OutputMode.CHANNELS))
        assert applied.names == ["car (0.90)", "car (0.90) #2"]

    def test_channel_content_is_the_placed_mask(self, image):
        applied = apply_result(image, make_result(), OutputOptions(mode=OutputMode.CHANNELS))
        channel = applied.channels[0]
        assert (channel.width, channel.height) == (image.width, image.height)
        assert channel.pixels[25 * image.width + 15] == 255
        assert channel.pixels[5 * image.width + 5] == 0

    def test_the_users_selection_survives(self, image):
        for y in range(0, 10):
            for x in range(0, 10):
                image.selection[y * image.width + x] = 255
        apply_result(image, make_result(), OutputOptions(mode=OutputMode.CHANNELS))
        assert selection_bbox(image) == (0, 0, 10, 10), "the Apply ate the user's selection"

    def test_channels_are_written_directly_without_a_selection_round_trip(self, image):
        """Two image-sized channels and an undo copy per instance was the
        3 GB status bar; the mask bytes now go straight into each channel."""
        result = make_result(instances=[
            Instance(0, 0.9, "car", (10, 20, 30, 40), 20, 20, solid_mask(20, 20)),
            Instance(1, 0.7, "car", (50, 20, 70, 40), 20, 20, solid_mask(20, 20)),
        ])
        applied = apply_result(image, result, OutputOptions(mode=OutputMode.CHANNELS))
        assert [c.get_name() for c in applied.channels] == ["car (0.90)", "car (0.70)"]
        # the selection is never touched, so it is neither saved nor restored
        assert [e for e in image.log if e[0] in ("select_item", "selection_save")] == []
        assert len(image.channels) == 2
        for inst, channel in zip(result.instances, applied.channels):
            x, y, w, h = result.rect_for(inst)
            assert channel.pixels[(y + h // 2) * image.width + x + w // 2] == 255
            assert channel.pixels[0] == 0 and channel.opacity == 50.0 and not channel.visible

    def test_channels_take_the_selection_route_only_for_selection_ops(self, image):
        result = make_result(instances=[
            Instance(0, 0.9, "car", (10, 20, 30, 40), 20, 20, solid_mask(20, 20)),
            Instance(1, 0.7, "car", (50, 20, 70, 40), 20, 20, solid_mask(20, 20)),
        ])
        applied = apply_result(image, result, OutputOptions(
            mode=OutputMode.CHANNELS, post=PostOps(feather=2.0)))
        assert len(applied.channels) == 2
        assert [e for e in image.log if e[0] == "feather"] == [("feather", 2.0)] * 2
        # one scratch channel for all instances, not one per instance
        assert [e for e in image.log if e[0] == "insert_channel" and e[1] == "sam3-scratch"] == [
            ("insert_channel", "sam3-scratch", 0)]
        assert all("scratch" not in c.get_name() for c in image.channels)

    def test_no_scratch_channels_remain(self, image):
        apply_result(image, make_result(), OutputOptions(mode=OutputMode.CHANNELS))
        assert all("scratch" not in c.get_name() for c in image.channels)
        assert all("previous" not in c.get_name() for c in image.channels)
        assert len(image.channels) == 1


# =========================================================================== #
# layer masks and groups -- offsets are the whole game here
# =========================================================================== #
class TestLayerModes:
    def test_layer_mask_is_applied_to_a_duplicate_not_the_original(self, image):
        source = image.layers[0]
        applied = apply_result(image, make_result(),
                               OutputOptions(mode=OutputMode.LAYER_MASKS))
        assert source.get_mask() is None, "the user's own layer was modified"
        assert len(applied.layers) == 1
        assert applied.layers[0] is not source
        assert applied.layers[0].get_mask() is not None
        assert applied.layers[0].has_alpha

    def test_the_duplicate_is_named_after_the_prompt(self, image):
        applied = apply_result(image, make_result(),
                               OutputOptions(mode=OutputMode.LAYER_MASKS))
        assert applied.names == ["car"]

    def test_layer_offsets_are_respected(self, image):
        """The mask lives in layer space.  A layer at (60, 30) must get the
        image-space mask translated by exactly that much."""
        layer = FakeLayer(image, "Offset layer", 80, 60)
        layer.set_offsets(60, 30)
        image.insert_layer(layer, None, 0)
        applied = apply_result(image, make_result(),
                               OutputOptions(mode=OutputMode.LAYER_MASKS), layer=layer)
        dup = applied.layers[0]
        assert dup.get_offsets() == (True, 60, 30), "the duplicate moved"
        mask = dup.get_mask()
        assert (mask.width, mask.height) == (80, 60)
        # Mask covers image x 10..30, y 20..40 -> layer x -50..-30 (clipped away),
        # so nothing of it should land in a layer that starts at x=60.
        assert max(mask.pixels) == 0

        mask_result = make_result(instances=[
            Instance(0, 0.9, "car", (70, 40, 90, 60), 20, 20, solid_mask(20, 20))])
        applied = apply_result(image, mask_result,
                               OutputOptions(mode=OutputMode.LAYER_MASKS), layer=layer)
        mask = applied.layers[0].get_mask()
        assert mask.pixels[(45 - 30) * 80 + (75 - 60)] == 255
        assert mask.pixels[0] == 0

    def test_apply_to_the_original_when_duplication_is_off(self, image):
        source = image.layers[0]
        apply_result(image, make_result(),
                     OutputOptions(mode=OutputMode.LAYER_MASKS, duplicate_layer=False))
        assert source.get_mask() is not None
        assert len(image.layers) == 1

    def test_layer_group_holds_one_masked_layer_per_instance(self, image):
        result = make_result(instances=[
            Instance(0, 0.9, "car", (10, 20, 30, 40), 20, 20, solid_mask(20, 20)),
            Instance(1, 0.7, "car", (100, 20, 120, 40), 20, 20, solid_mask(20, 20)),
        ])
        applied = apply_result(image, result, OutputOptions(mode=OutputMode.LAYER_GROUPS))
        assert applied.group is not None
        assert applied.group.get_name() == "car"
        assert [l.get_name() for l in applied.group.children] == ["car (0.90)", "car (0.70)"]
        for child in applied.group.children:
            assert child.get_mask() is not None
            assert child.get_parent() is applied.group

    def test_group_masks_differ_per_instance(self, image):
        result = make_result(instances=[
            Instance(0, 0.9, "car", (10, 20, 30, 40), 20, 20, solid_mask(20, 20)),
            Instance(1, 0.7, "car", (100, 20, 120, 40), 20, 20, solid_mask(20, 20)),
        ])
        applied = apply_result(image, result, OutputOptions(mode=OutputMode.LAYER_GROUPS))
        first, second = (l.get_mask() for l in applied.group.children)
        assert first.pixels[25 * image.width + 15] == 255
        assert first.pixels[25 * image.width + 105] == 0
        assert second.pixels[25 * image.width + 105] == 255
        assert second.pixels[25 * image.width + 15] == 0

    def test_an_empty_group_is_not_left_behind(self, image):
        applied = apply_result(image, make_result(instances=[]),
                               OutputOptions(mode=OutputMode.LAYER_GROUPS))
        assert applied.group is None
        assert len(image.layers) == 1

    def test_group_name_can_be_overridden(self, image):
        applied = apply_result(image, make_result(),
                               OutputOptions(mode=OutputMode.LAYER_GROUPS,
                                             group_name="Cutouts"))
        assert applied.group.get_name() == "Cutouts"

    def test_old_group_layer_new_signature_still_works(self, image, gimp, monkeypatch):
        """Early GIMP 3.0 builds had GroupLayer.new(image) with no name."""
        outputs.set_gimp_modules(_make_gimp(group_takes_name=False), _make_gegl())
        img = FakeImage(200, 120)
        img.insert_layer(FakeLayer(img, "Background", 200, 120), None, 0)
        applied = apply_result(img, make_result(), OutputOptions(mode=OutputMode.LAYER_GROUPS))
        assert applied.group.get_name() == "car"

    def test_layer_mode_without_a_layer_is_a_clear_error(self, gimp):
        empty = FakeImage(50, 50)
        with pytest.raises(OutputError):
            apply_result(empty, make_result(), OutputOptions(mode=OutputMode.LAYER_MASKS))


# =========================================================================== #
# paths
# =========================================================================== #
class TestPaths:
    def test_control_point_triples_are_passed_through_and_mapped(self, image):
        # one anchor at canvas (20, 40) with handles either side
        stroke = {"control_points": [10.0, 40.0, 20.0, 40.0, 30.0, 40.0,
                                     10.0, 80.0, 20.0, 80.0, 30.0, 80.0],
                  "closed": True}
        applied = apply_result(image, make_result(),
                               OutputOptions(mode=OutputMode.PATHS),
                               contours={0: [stroke]})
        assert len(applied.paths) == 1
        path = applied.paths[0]
        assert path.get_name() == "car (0.93)"
        stroke_type, points, closed = path.strokes[0]
        assert stroke_type == PathStrokeType.BEZIER
        assert closed is True
        assert points == [10.0, 40.0, 20.0, 40.0, 30.0, 40.0,
                          10.0, 80.0, 20.0, 80.0, 30.0, 80.0]
        assert image.paths == [path]

    def test_points_are_mapped_out_of_canvas_space(self, image):
        result = make_result(uploaded=(100, 60), source=(400, 240))
        stroke = {"control_points": [0.0, 0.0, 10.0, 20.0, 0.0, 0.0,
                                     0.0, 0.0, 30.0, 5.0, 0.0, 0.0]}
        applied = apply_result(image, result, OutputOptions(mode=OutputMode.PATHS),
                               contours={0: [stroke]})
        _, points, _ = applied.paths[0].strokes[0]
        assert points == [0.0, 0.0, 20.0, 40.0, 0.0, 0.0,
                          0.0, 0.0, 60.0, 10.0, 0.0, 0.0]

    def test_a_polyline_is_expanded_into_degenerate_triples(self, image):
        stroke = {"points": [0.0, 0.0, 10.0, 0.0, 10.0, 10.0], "closed": False}
        applied = apply_result(image, make_result(), OutputOptions(mode=OutputMode.PATHS),
                               contours={0: [stroke]})
        _, points, closed = applied.paths[0].strokes[0]
        assert closed is False
        assert len(points) == 18
        assert points[0:6] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert points[6:12] == [10.0, 0.0, 10.0, 0.0, 10.0, 0.0]

    def test_a_bare_flat_sequence_of_triples_is_accepted(self, image):
        flat = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0]
        applied = apply_result(image, make_result(), OutputOptions(mode=OutputMode.PATHS),
                               contours={0: [flat]})
        _, points, _ = applied.paths[0].strokes[0]
        assert points == flat

    def test_pairs_are_accepted(self, image):
        pairs = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]
        applied = apply_result(image, make_result(), OutputOptions(mode=OutputMode.PATHS),
                               contours={0: [pairs]})
        _, points, _ = applied.paths[0].strokes[0]
        assert len(points) == 24

    def test_contours_may_be_a_parallel_sequence(self, image):
        result = make_result(instances=[
            Instance(0, 0.9, "car", (10, 20, 30, 40), 20, 20, solid_mask(20, 20)),
            Instance(1, 0.7, "car", (50, 20, 70, 40), 20, 20, solid_mask(20, 20)),
        ])
        contours = [[{"points": [0.0, 0.0, 4.0, 0.0, 4.0, 4.0]}],
                    [{"points": [0.0, 0.0, 8.0, 0.0, 8.0, 8.0]}]]
        applied = apply_result(image, result, OutputOptions(mode=OutputMode.PATHS),
                               contours=contours)
        assert [p.get_name() for p in applied.paths] == ["car (0.90)", "car (0.70)"]

    def test_new_paths_are_visible_and_selected(self, image):
        """A path is hidden by default and lives in a closed dock, so an
        Apply that made only hidden paths looks like it did nothing.  Shown,
        it draws on the canvas."""
        applied = apply_result(image, make_result(), OutputOptions(mode=OutputMode.PATHS),
                               contours={0: [{"points": [0.0, 0.0, 4.0, 0.0, 4.0, 4.0]}]})
        assert applied.paths and all(p.visible for p in applied.paths)
        assert image.selected_paths == applied.paths
        assert image.get_selected_layers(), "the layer stays selected so the ants still draw"

    def test_several_strokes_land_in_one_path(self, image):
        contours = {0: [{"points": [0.0, 0.0, 4.0, 0.0, 4.0, 4.0]},
                        {"points": [8.0, 8.0, 9.0, 8.0, 9.0, 9.0]}]}
        applied = apply_result(image, make_result(), OutputOptions(mode=OutputMode.PATHS),
                               contours=contours)
        assert len(applied.paths[0].strokes) == 2

    def test_instances_without_contours_produce_no_path(self, image):
        applied = apply_result(image, make_result(), OutputOptions(mode=OutputMode.PATHS),
                               contours={})
        assert applied.paths == []
        assert image.paths == []

    def test_a_stroke_with_fewer_than_two_anchors_is_skipped(self, image):
        """One anchor is not a stroke; emitting it would leave the user an
        invisible path item to delete."""
        applied = apply_result(image, make_result(), OutputOptions(mode=OutputMode.PATHS),
                               contours={0: [{"points": [1.0, 1.0]}]})
        assert applied.paths == []
        assert image.paths == []

    def test_a_stroke_dict_with_no_points_is_rejected(self):
        with pytest.raises(ValueError):
            outputs.normalise_stroke({"closed": True})

    def test_an_odd_coordinate_count_is_rejected(self):
        with pytest.raises(ValueError):
            outputs.normalise_stroke({"points": [1.0, 2.0, 3.0]})


# =========================================================================== #
# odds and ends
# =========================================================================== #
class TestPlumbing:
    def test_gegl_buffer_set_falls_back_to_the_five_argument_signature(self, image, gimp):
        """The introspected arity of Gegl.Buffer.set differs between GEGL
        builds, so outputs.py tries each plausible one.  A build exposing only
        the long form must work."""
        FakeItem.buffer_class = FiveArgBuffer
        try:
            applied = apply_result(image, make_result(),
                                   OutputOptions(mode=OutputMode.CHANNELS))
        finally:
            FakeItem.buffer_class = FakeBuffer
        assert applied.channels[0].pixels[25 * image.width + 15] == 255

    def test_an_unknown_mode_is_rejected_before_anything_is_touched(self, image):
        with pytest.raises(ValueError):
            apply_result(image, make_result(), OutputOptions(mode="teleport"))
        assert image.undo_starts == 0

    def test_an_unknown_selection_op_is_rejected(self, image):
        with pytest.raises(ValueError):
            apply_result(image, make_result(), OutputOptions(selection_op="xor"))

    def test_displays_are_flushed_once(self, image, gimp):
        apply_result(image, make_result(), OutputOptions(mode=OutputMode.SELECTION))
        assert gimp.calls.count("flush") == 1

    def test_interpolation_is_set_for_the_upscale(self, image, gimp):
        apply_result(image, make_result(), OutputOptions(mode=OutputMode.SELECTION))
        assert ("interp", InterpolationType.CUBIC) in gimp.calls
        assert ("resize", TransformResize.ADJUST) in gimp.calls

    def test_summary_reads_like_a_status_bar_message(self):
        from outputs import AppliedResult

        applied = AppliedResult(mode=OutputMode.CHANNELS, names=["a", "b"], dropped=[3])
        assert applied.summary() == "2 channels (1 below minimum area)"
        assert AppliedResult(mode=OutputMode.PATHS, names=["a"]).summary() == "1 path"

    def test_the_module_imports_without_gimp(self):
        """Importing outputs must not need GIMP."""
        import importlib

        outputs_module = importlib.import_module("outputs")
        assert outputs_module.OutputMode.SELECTION == "selection"

    def test_post_op_values_are_clamped(self):
        post = PostOps(threshold=-5, edge_softness=-1, grow=-3, min_area=-1.0).normalised()
        assert (post.threshold, post.edge_softness, post.grow, post.min_area) == (1, 0, 0, 0.0)


# =========================================================================== #
# interop with the shared tests/fake_gimp stubs
#
# Those stubs exist first for gimpbridge (projection reading) and model pixels
# rather than the output PDB surface in detail -- no Gimp.Path, a nearest-
# neighbour scaler -- so the suite above uses the local fake.  What is worth
# checking against the shared stubs is the *seam*: that outputs.py picks them
# up when they are installed, that writing mask bytes into a real stub channel
# through Gegl.Buffer.set has the layout API.md section 8.3 specifies, and that
# an Apply undoes and redoes cleanly under their undo model as well.
# =========================================================================== #
class TestSharedStubInterop:
    def test_outputs_resolves_the_shared_stubs_and_writes_mask_bytes(self):
        shared = pytest.importorskip("fake_gimp")
        outputs.reset_gimp_modules()
        try:
            manager = shared.installed()
        except Exception as exc:  # pragma: no cover - the shared stubs failed to install
            pytest.skip("fake_gimp.install() failed: %s" % (exc,))
        with manager as stubs:
            assert outputs._gimp() is stubs.Gimp
            assert outputs._gegl() is stubs.Gegl

            img = stubs.Gimp.Image.new(16, 12, stubs.Gimp.ImageBaseType.RGB)
            channel = outputs._new_channel(img, "sam3-scratch", 4, 3)
            data = bytes([0, 40, 200, 255,
                          10, 50, 210, 250,
                          20, 60, 220, 245])
            outputs._write_gray(channel, 4, 3, data)
            # row-major, top to bottom, width bytes per row, no padding
            assert bytes(channel.get_buffer().data) == data
            img.delete()
        outputs.reset_gimp_modules()

    @pytest.mark.parametrize("mode,op", [
        (OutputMode.SELECTION, SelectionOp.REPLACE),
        (OutputMode.SELECTION, SelectionOp.ADD),
        (OutputMode.CHANNELS, SelectionOp.REPLACE),
        (OutputMode.LAYER_MASKS, SelectionOp.REPLACE),
        (OutputMode.LAYER_GROUPS, SelectionOp.REPLACE),
    ])
    def test_an_apply_undoes_and_redoes_on_the_shared_stubs(self, mode, op):
        shared = pytest.importorskip("fake_gimp")
        outputs.reset_gimp_modules()
        with shared.installed() as stubs:
            img = stubs.Image(40, 30)
            layer = stubs.Layer.new(img, "bg", 40, 30, stubs.ImageType.RGB_IMAGE, 100.0,
                                    stubs.LayerMode.NORMAL)
            img.insert_layer(layer, None, 0)
            img.set_selection_rect(0, 0, 5, 5)

            def state():
                return (bytes(img.get_selection().get_buffer().data),
                        [c.get_name() for c in img.get_channels()],
                        [l.get_name() for l in img.get_layers()])

            before = state()
            inst = Instance(0, 0.9, "car", (2, 2, 12, 12), 10, 10, solid_mask(10, 10))
            res = MaskResult(instances=[inst], canvas_from_image=CanvasTransform(1, 1, 0, 0),
                             uploaded=(40, 30), source=(40, 30), prompt_text="car")
            apply_result(img, res, OutputOptions(mode=mode, selection_op=op), layer=layer)
            after = state()
            img.undo()
            assert state() == before
            img.redo()
            assert state() == after
            assert img.criticals == []
            assert not [n for n in after[1] if n.startswith("sam3-")]
            img.delete()
        outputs.reset_gimp_modules()


class TestSelectionIsNotDisturbed:
    """Only the selection mode is allowed to change the selection."""

    @pytest.mark.parametrize("mode", [OutputMode.CHANNELS, OutputMode.LAYER_MASKS,
                                      OutputMode.LAYER_GROUPS, OutputMode.PATHS])
    def test_the_users_selection_is_put_back(self, image, mode):
        for y in range(0, 10):
            for x in range(0, 10):
                image.selection[y * image.width + x] = 255
        before = bytes(image.selection)
        apply_result(image, make_result(), OutputOptions(mode=mode),
                     contours={0: [{"points": [0.0, 0.0, 4.0, 0.0, 4.0, 4.0]}]})
        assert bytes(image.selection) == before

    def test_paths_mode_does_not_bother_snapshotting_the_selection(self, image):
        apply_result(image, make_result(), OutputOptions(mode=OutputMode.PATHS),
                     contours={0: [{"points": [0.0, 0.0, 4.0, 0.0, 4.0, 4.0]}]})
        assert [e for e in image.log if e[0] == "selection_save"] == []


class TestDuckTypedInstances:
    """client.py owns the frame decoder; outputs.py must not care which class
    it hands over, only that the wire field names are there."""

    def test_a_foreign_object_with_the_wire_fields_works(self, image):
        class ForeignInstance:
            instance_id = 0
            score = 0.75
            label = "cat"
            bbox = (10, 20, 30, 40)
            mask_width = 20
            mask_height = 20
            mask = solid_mask(20, 20)

        result = make_result(instances=[ForeignInstance()])
        applied = apply_result(image, result, OutputOptions(mode=OutputMode.CHANNELS))
        assert applied.names == ["cat (0.75)"]
        assert applied.channels[0].pixels[25 * image.width + 15] == 255

    def test_a_plain_dict_works(self, image):
        raw = {"instance_id": 3, "score": 0.5, "label": "dog", "bbox": [10, 20, 30, 40],
               "mask_width": 20, "mask_height": 20, "mask": solid_mask(20, 20)}
        applied = apply_result(image, make_result(instances=[raw]),
                               OutputOptions(mode=OutputMode.CHANNELS))
        assert applied.names == ["dog (0.50)"]

    def test_something_unusable_is_a_clear_type_error(self, image):
        with pytest.raises(TypeError):
            apply_result(image, make_result(instances=[object()]),
                         OutputOptions(mode=OutputMode.CHANNELS))
