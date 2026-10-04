"""In-memory stand-ins for ``gi.repository.Gimp``, ``gi.repository.Gegl`` and
``gi.repository.Babl``.

Why this exists
---------------
Outside GIMP there are **no Gimp/Gegl typelibs** (``gi`` may be installed, but
``gi.require_version("Gimp", "3.0")`` raises ``ValueError``).  The plug-in
modules that touch pixels -- ``gimpbridge.py`` and ``outputs.py`` -- must
therefore be exercised against a fake.  These stubs model a *real* small
image: layers with offsets, channels, a selection, and byte-backed GEGL buffers,
so tests can assert on **actual pixel values**, not on call spies alone.

What is modelled faithfully
---------------------------
* ``Gegl.Buffer`` holds a ``bytearray`` in one babl format and converts on
  ``get()`` / ``set()``, exactly like babl does for
  ``"R'G'B' u8"`` / ``"R'G'B'A u8"`` / ``"Y' u8"`` / ``"Y'A u8"``.
* Layer offsets, layer visibility, opacity and normal-mode alpha compositing.
* ``Gimp.Image.duplicate()``, ``flatten()``, ``scale()``, ``delete()`` and the
  channel list -- enough that a leaked temporary image is a detectable bug
  (see :func:`live_images`).
* ``Gimp.Drawable.get_offsets()`` returns the GIMP 3 shape ``(True, x, y)``.
* ``Gimp.Selection.bounds()`` returns ``(ok, non_empty, x1, y1, x2, y2)``, the GIMP 3 shape.
* The undo stack, as far as a plug-in can see it (:class:`UndoModel`): item
  additions and removals, selection changes, renames and layer masks are
  recorded unless undo is frozen, and ``image.undo()`` / ``image.redo()``
  replay them, so a recording that cannot be replayed is caught.
* Which items are selected: layers and channels exclude each other, a new
  layer or channel becomes the selected one, ``add_mask`` turns edit-mask on,
  and ``get_selected_drawables()`` answers the mask or the channels exactly
  where GIMP does.

What is deliberately *not* faithful  (do not assert on these)
-------------------------------------------------------------
* **The scaler is nearest-neighbour.**  Real GIMP scales with GEGL (cubic /
  NoHalo) in C.  Tests must only assert on constant or blocky patterns, never
  on interpolated edge values.
* Colour management: ``"R'G'B' u8"`` and ``"RGB u8"`` are treated as the same
  bytes.  Gray<->colour uses Rec.601 luma on the primed values.
* Layer modes other than NORMAL, indexed images, and precisions other than u8.
* ``Gimp.Image.flatten()`` composites onto the context background colour, which
  is what GIMP does, but the stub does no gamma-correct blending.

Recording
---------
Every module-level ``Gimp.*`` function call is appended to :data:`Gimp.calls`
as ``(name, args)``.  ``Gimp.reset()`` clears the recording *and* the live-image
registry.  :func:`fake_gimp.install` calls it for you.
"""

from __future__ import annotations

import types

__all__ = [
    "Gimp",
    "Gegl",
    "Babl",
    "MODULES",
    "reset",
    "live_images",
    "format_components",
    "make_rgb_bytes",
    "UndoModel",
]


# --------------------------------------------------------------------------- #
# babl-ish format handling
# --------------------------------------------------------------------------- #
# name -> (components, has_alpha, is_gray)
_FORMAT_INFO = {
    "R'G'B' u8": (3, False, False),
    "RGB u8": (3, False, False),
    "R'G'B'A u8": (4, True, False),
    "RGBA u8": (4, True, False),
    "Y' u8": (1, False, True),
    "Y u8": (1, False, True),
    "Y'A u8": (2, True, True),
    "YA u8": (2, True, True),
}


def _info(fmt):
    if fmt is None:
        raise ValueError("a babl format string is required by the stubs")
    name = fmt if isinstance(fmt, str) else getattr(fmt, "name", None)
    try:
        return _FORMAT_INFO[name]
    except KeyError:
        raise ValueError(
            "fake_gimp does not model the babl format %r; supported: %s"
            % (name, ", ".join(sorted(_FORMAT_INFO)))
        )


def format_components(fmt) -> int:
    """Bytes per pixel for a format string the stubs understand."""
    return _info(fmt)[0]


def _same_layout(a, b) -> bool:
    return _info(a) == _info(b)


def _luma(rgba: bytes, n: int) -> bytes:
    out = bytearray(n)
    for i in range(n):
        j = i * 4
        out[i] = (rgba[j] * 299 + rgba[j + 1] * 587 + rgba[j + 2] * 114 + 500) // 1000
    return bytes(out)


def _to_rgba(data, fmt, n: int) -> bytearray:
    comps, alpha, gray = _info(fmt)
    out = bytearray(n * 4)
    if gray:
        y = bytes(data[0::comps])
        out[0::4] = y
        out[1::4] = y
        out[2::4] = y
    else:
        out[0::4] = bytes(data[0::comps])
        out[1::4] = bytes(data[1::comps])
        out[2::4] = bytes(data[2::comps])
    out[3::4] = bytes(data[comps - 1 :: comps]) if alpha else b"\xff" * n
    return out


def _from_rgba(rgba, fmt, n: int) -> bytes:
    comps, alpha, gray = _info(fmt)
    out = bytearray(n * comps)
    if gray:
        out[0::comps] = _luma(rgba, n)
    else:
        out[0::comps] = bytes(rgba[0::4])
        out[1::comps] = bytes(rgba[1::4])
        out[2::comps] = bytes(rgba[2::4])
    if alpha:
        out[comps - 1 :: comps] = bytes(rgba[3::4])
    return bytes(out)


def _convert(data, src_fmt, dst_fmt, n: int) -> bytes:
    """Convert ``n`` packed pixels from ``src_fmt`` to ``dst_fmt``."""
    if _same_layout(src_fmt, dst_fmt):
        return bytes(data)
    return _from_rgba(_to_rgba(data, src_fmt, n), dst_fmt, n)


# --------------------------------------------------------------------------- #
# nearest-neighbour resampling (stand-in for GEGL's scaler)
# --------------------------------------------------------------------------- #
def _resample_nearest(data, comps, sw, sh, dw, dh) -> bytes:
    if (sw, sh) == (dw, dh):
        return bytes(data)
    out = bytearray(dw * dh * comps)
    drow = dw * comps
    srow = sw * comps
    integer_x = dw and sw % dw == 0
    integer_y = dh and sh % dh == 0
    fx = sw // dw if integer_x else 0
    fy = sh // dh if integer_y else 0
    for dy in range(dh):
        sy = dy * fy if integer_y else (dy * sh) // dh
        row = data[sy * srow : (sy + 1) * srow]
        base = dy * drow
        if integer_x:
            step = comps * fx
            for c in range(comps):
                out[base + c : base + drow : comps] = row[c::step][:dw]
        else:
            for dx in range(dw):
                sx = (dx * sw) // dw
                out[base + dx * comps : base + (dx + 1) * comps] = row[
                    sx * comps : (sx + 1) * comps
                ]
    return bytes(out)


# --------------------------------------------------------------------------- #
# Gegl
# --------------------------------------------------------------------------- #
class Rectangle(object):
    """``Gegl.Rectangle`` -- a plain (x, y, width, height) box."""

    def __init__(self, x=0, y=0, width=0, height=0):
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)

    @staticmethod
    def new(x, y, width, height):
        return Rectangle(x, y, width, height)

    def __repr__(self):
        return "Rectangle(%d, %d, %d, %d)" % (self.x, self.y, self.width, self.height)

    def __eq__(self, other):
        return (
            isinstance(other, Rectangle)
            and (self.x, self.y, self.width, self.height)
            == (other.x, other.y, other.width, other.height)
        )


class AbyssPolicy(object):
    NONE = 0
    CLAMP = 1
    LOOP = 2
    BLACK = 3
    WHITE = 4


class Color(object):
    """``Gegl.Color`` -- only ``new`` / ``set_rgba`` / ``get_rgba`` are modelled."""

    _NAMED = {
        "black": (0.0, 0.0, 0.0, 1.0),
        "white": (1.0, 1.0, 1.0, 1.0),
        "transparent": (0.0, 0.0, 0.0, 0.0),
    }

    def __init__(self, name="black"):
        self._rgba = list(self._NAMED.get(name, (0.0, 0.0, 0.0, 1.0)))

    @staticmethod
    def new(name="black"):
        return Color(name)

    def set_rgba(self, r, g, b, a):
        self._rgba = [float(r), float(g), float(b), float(a)]

    def get_rgba(self):
        return tuple(self._rgba)

    def as_bytes(self):
        return tuple(max(0, min(255, int(round(c * 255.0)))) for c in self._rgba)

    def __repr__(self):
        return "Color(%.3f, %.3f, %.3f, %.3f)" % tuple(self._rgba)


class Buffer(object):
    """``Gegl.Buffer`` backed by a ``bytearray``.

    The buffer has an extent (``x``, ``y``, ``width``, ``height``) and one
    native babl format.  ``get()`` and ``set()`` convert to/from any other
    format the stubs understand, which is exactly the babl service the plug-in
    relies on ("ask for ``R'G'B' u8`` and let babl deal with gray / indexed /
    alpha").
    """

    def __init__(self, x, y, width, height, fmt="R'G'B'A u8", data=None):
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)
        self.format = fmt
        comps = format_components(fmt)
        if data is None:
            self.data = bytearray(self.width * self.height * comps)
        else:
            if len(data) != self.width * self.height * comps:
                raise ValueError("initial data has the wrong length")
            self.data = bytearray(data)
        self.flush_count = 0

    # -- construction ------------------------------------------------------ #
    @staticmethod
    def new(fmt, x, y, width, height):
        return Buffer(x, y, width, height, fmt)

    # -- introspection ----------------------------------------------------- #
    def get_extent(self):
        return Rectangle(self.x, self.y, self.width, self.height)

    def get_format(self):
        return self.format

    def flush(self):
        self.flush_count += 1

    # -- pixel access ------------------------------------------------------ #
    def _extract(self, rx, ry, rw, rh):
        """Native-format bytes for a rect in buffer-local coords, CLAMP abyss."""
        comps = format_components(self.format)
        if rx >= 0 and ry >= 0 and rx + rw <= self.width and ry + rh <= self.height:
            if rw == self.width and rx == 0:
                start = ry * self.width * comps
                return bytes(self.data[start : start + rw * rh * comps])
            out = bytearray(rw * rh * comps)
            step = rw * comps
            for row in range(rh):
                src = ((ry + row) * self.width + rx) * comps
                out[row * step : (row + 1) * step] = self.data[src : src + step]
            return bytes(out)
        out = bytearray(rw * rh * comps)
        for row in range(rh):
            sy = min(max(ry + row, 0), self.height - 1)
            for col in range(rw):
                sx = min(max(rx + col, 0), self.width - 1)
                src = (sy * self.width + sx) * comps
                dst = (row * rw + col) * comps
                out[dst : dst + comps] = self.data[src : src + comps]
        return bytes(out)

    def get(self, rect, scale=1.0, fmt=None, repeat_mode=AbyssPolicy.CLAMP):
        """``buffer.get(rect, scale, "R'G'B' u8", Gegl.AbyssPolicy.CLAMP) -> bytes``

        This is the call shape the API contract (_daemon/API.md section 7)
        pins down for the plug-in.
        """
        fmt = fmt or self.format
        rw, rh = int(rect.width), int(rect.height)
        native = self._extract(int(rect.x) - self.x, int(rect.y) - self.y, rw, rh)
        comps = format_components(self.format)
        if scale and float(scale) != 1.0:
            dw = max(1, int(round(rw * float(scale))))
            dh = max(1, int(round(rh * float(scale))))
            native = _resample_nearest(native, comps, rw, rh, dw, dh)
            rw, rh = dw, dh
        return _convert(native, self.format, fmt, rw * rh)

    def set(self, rect, fmt, data):
        """``buffer.set(rect, "Y' u8", data)`` -- writes, clipping to the extent."""
        rw, rh = int(rect.width), int(rect.height)
        comps_src = format_components(fmt)
        if len(data) != rw * rh * comps_src:
            raise ValueError(
                "fake Gegl.Buffer.set: expected %d bytes for %dx%d %s, got %d"
                % (rw * rh * comps_src, rw, rh, fmt, len(data))
            )
        native = _convert(data, fmt, self.format, rw * rh)
        comps = format_components(self.format)
        rx = int(rect.x) - self.x
        ry = int(rect.y) - self.y
        for row in range(rh):
            dy = ry + row
            if dy < 0 or dy >= self.height:
                continue
            x0 = max(rx, 0)
            x1 = min(rx + rw, self.width)
            if x1 <= x0:
                continue
            src = (row * rw + (x0 - rx)) * comps
            dst = (dy * self.width + x0) * comps
            n = (x1 - x0) * comps
            self.data[dst : dst + n] = native[src : src + n]


def _gegl_init(argv=None):
    Gegl.calls.append(("init", (argv,)))
    return argv


# --------------------------------------------------------------------------- #
# Gimp enums
# --------------------------------------------------------------------------- #
class ImageBaseType(object):
    RGB = 0
    GRAY = 1
    INDEXED = 2


class ImageType(object):
    RGB_IMAGE = 0
    RGBA_IMAGE = 1
    GRAY_IMAGE = 2
    GRAYA_IMAGE = 3
    INDEXED_IMAGE = 4
    INDEXEDA_IMAGE = 5


_TYPE_FORMAT = {
    ImageType.RGB_IMAGE: "R'G'B' u8",
    ImageType.RGBA_IMAGE: "R'G'B'A u8",
    ImageType.GRAY_IMAGE: "Y' u8",
    ImageType.GRAYA_IMAGE: "Y'A u8",
}


class LayerMode(object):
    NORMAL = 28
    NORMAL_LEGACY = 0


class InterpolationType(object):
    NONE = 0
    LINEAR = 1
    CUBIC = 2
    NOHALO = 3
    LOHALO = 4


class ChannelOps(object):
    ADD = 0
    SUBTRACT = 1
    REPLACE = 2
    INTERSECT = 3


class MergeType(object):
    EXPAND_AS_NECESSARY = 0
    CLIP_TO_IMAGE = 1
    CLIP_TO_BOTTOM_LAYER = 2
    FLATTEN_IMAGE = 3


class FillType(object):
    FOREGROUND = 0
    BACKGROUND = 1
    WHITE = 2
    TRANSPARENT = 3
    PATTERN = 4


class AddMaskType(object):
    WHITE = 0
    BLACK = 1
    ALPHA = 2
    ALPHA_TRANSFER = 3
    SELECTION = 4
    COPY = 5
    CHANNEL = 6


# --------------------------------------------------------------------------- #
# undo
# --------------------------------------------------------------------------- #
class UndoModel(object):
    """What a plug-in can observe of GIMP's undo stack.

    A change made while the image's undo is *not* frozen is recorded, as
    ``gimp_image_undo_push`` records it, into the open undo group or as a step
    of its own; a change made while frozen is not recorded at all
    (``gimp_image_undo_push`` returns before creating the undo).  :meth:`undo`
    and :meth:`redo` replay one whole step.

    A replay that finds an item in the wrong state is appended to
    :attr:`criticals` instead of being performed.  That is the fake's version
    of the ``gimp_item_is_attached`` assertion GIMP fails on Undo when a
    channel was added with undo recorded and then removed with undo frozen:
    the recorded "channel added" step tries to remove a channel that is no
    longer there, and Redo puts it back into the Channels dock.

    Subclasses call :meth:`_undo_setup` from ``__init__``, route their group
    start/end through :meth:`_undo_group_opened` / :meth:`_undo_group_closed`
    and answer :meth:`_undo_is_frozen`.
    """

    def _undo_setup(self):
        #: Closed undo steps, oldest first; each is ``[(label, undo, redo), ...]``.
        self.undo_steps = []
        self.redo_steps = []
        self.criticals = []
        self._undo_open = None
        self._undo_nesting = 0

    def _undo_is_frozen(self):
        return False

    def _undo_group_opened(self):
        if self._undo_is_frozen():
            return  # gimp_image_undo_group_start returns FALSE while frozen
        self._undo_nesting += 1
        if self._undo_nesting == 1:
            self._undo_open = []

    def _undo_group_closed(self):
        if self._undo_is_frozen() or self._undo_nesting == 0:
            return
        self._undo_nesting -= 1
        if self._undo_nesting == 0:
            if self._undo_open:
                self.undo_steps.append(self._undo_open)
                del self.redo_steps[:]
            self._undo_open = None

    def _undo_record(self, label, undo, redo):
        if self._undo_is_frozen():
            return
        if self._undo_open is not None:
            self._undo_open.append((label, undo, redo))
        else:
            self.undo_steps.append([(label, undo, redo)])
            del self.redo_steps[:]

    def _undo_critical(self, text):
        self.criticals.append(text)

    def recorded(self, step=-1):
        """Labels of one recorded undo step (the latest by default)."""
        return [label for label, _undo, _redo in self.undo_steps[step]]

    def undo(self):
        """Edit > Undo: replay the latest step backwards."""
        step = self.undo_steps.pop()
        for _label, undo, _redo in reversed(step):
            undo()
        self.redo_steps.append(step)

    def redo(self):
        """Edit > Redo: replay the step that was just undone."""
        step = self.redo_steps.pop()
        for _label, _undo, redo in step:
            redo()
        self.undo_steps.append(step)

    # -- the recordings GIMP makes for the calls outputs.py uses ------------- #
    def _record_mask(self, get, put):
        """``GIMP_UNDO_MASK``: a copy of the selection before it changed."""
        cell = [bytearray(get())]

        def swap():
            before = bytearray(get())
            put(cell[0])
            cell[0] = before

        self._undo_record("mask", swap, swap)

    def _record_item_added(self, label, items, item, position, attached):
        """``GIMP_UNDO_*_ADD``: Undo removes the item, Redo puts it back."""

        def undo():
            if not attached(item):
                self._undo_critical("%s undo: %r is not attached" % (label, _name(item)))
                return
            items().remove(item)

        def redo():
            if attached(item):
                self._undo_critical("%s redo: %r is already attached" % (label, _name(item)))
                return
            target = items()
            target.insert(min(position, len(target)), item)

        self._undo_record(label, undo, redo)

    def _record_item_removed(self, label, items, item, position, attached):
        """``GIMP_UNDO_*_REMOVE``: Undo puts the item back, Redo removes it."""

        def undo():
            if attached(item):
                self._undo_critical("%s undo: %r is already attached" % (label, _name(item)))
                return
            target = items()
            target.insert(min(position, len(target)), item)

        def redo():
            if not attached(item):
                self._undo_critical("%s redo: %r is not attached" % (label, _name(item)))
                return
            items().remove(item)

        self._undo_record(label, undo, redo)

    def _record_rename(self, item, attached, old, new, setter):
        """``GIMP_UNDO_ITEM_RENAME`` goes through the item tree, which a
        detached item no longer has."""

        def rename(value):
            def run():
                if not attached(item):
                    self._undo_critical("rename: %r is not attached" % (_name(item),))
                    return
                setter(value)
            return run

        self._undo_record("item-rename", rename(old), rename(new))


def _name(item):
    fn = getattr(item, "get_name", None)
    return fn() if callable(fn) else repr(item)


# --------------------------------------------------------------------------- #
# Gimp items
# --------------------------------------------------------------------------- #
class _Item(object):
    _next_id = 1

    def __init__(self, image, name):
        self.id = _Item._next_id
        _Item._next_id += 1
        self._image = image
        self._name = name
        self._valid = True

    def get_id(self):
        return self.id

    def get_name(self):
        return self._name

    def set_name(self, name):
        image = self._image
        if isinstance(image, Image) and image._is_attached(self) and name != self._name:
            image._record_rename(self, image._is_attached, self._name, name,
                                 lambda value: setattr(self, "_name", value))
        self._name = name
        return True

    def get_image(self):
        return self._image

    def is_valid(self):
        return self._valid

    def is_group(self):
        return False


class Drawable(_Item):
    """Base for ``Gimp.Layer`` / ``Gimp.Channel`` -- owns one :class:`Buffer`."""

    def __init__(self, image, name, width, height, fmt, offsets=(0, 0)):
        _Item.__init__(self, image, name)
        self._buffer = Buffer(0, 0, width, height, fmt)
        self._offsets = [int(offsets[0]), int(offsets[1])]
        self.visible = True
        self.opacity = 100.0
        self.updates = []

    # -- geometry ---------------------------------------------------------- #
    def get_width(self):
        return self._buffer.width

    def get_height(self):
        return self._buffer.height

    def get_offsets(self):
        """GIMP 3 shape: ``(success, offset_x, offset_y)``."""
        return (True, self._offsets[0], self._offsets[1])

    def set_offsets(self, x, y):
        self._offsets = [int(x), int(y)]
        return True

    # -- pixels ------------------------------------------------------------ #
    def get_buffer(self):
        return self._buffer

    def get_shadow_buffer(self):
        return self._buffer

    def merge_shadow(self, undo=True):
        return True

    def update(self, x, y, width, height):
        self.updates.append((int(x), int(y), int(width), int(height)))
        return True

    def fill(self, fill_type):
        """``gimp_drawable_fill``.  TRANSPARENT is *the context background*
        (white by default) with alpha zeroed only if the drawable has alpha --
        on a channel that is a white fill, i.e. Select All."""
        Gimp.calls.append(("Drawable.fill", (self.get_name(), fill_type)))
        comps = format_components(self._buffer.format)
        has_alpha = _info(self._buffer.format)[1]
        if fill_type == FillType.TRANSPARENT and has_alpha:
            self._buffer.data[:] = bytearray(len(self._buffer.data))
        else:
            bg = _CONTEXT[-1]["background"].as_bytes()
            px = bytes(bg[:comps]) if comps <= len(bg) else bytes(bg) + b"\xff" * (comps - len(bg))
            if comps == 1:
                px = bytes([max(bg[:3])])
            self._buffer.data[:] = px * (self.get_width() * self.get_height())
        return True

    def get_format(self):
        return self._buffer.format

    def has_alpha(self):
        return _info(self._buffer.format)[1]

    def get_visible(self):
        return self.visible

    def set_visible(self, visible):
        self.visible = bool(visible)
        return True

    def get_opacity(self):
        return self.opacity

    def set_opacity(self, opacity):
        self.opacity = float(opacity)
        return True

    # -- test conveniences (not part of the real API) ----------------------- #
    def fill_bytes(self, data):
        """Overwrite the whole buffer with native-format bytes."""
        if len(data) != len(self._buffer.data):
            raise ValueError("wrong length for fill_bytes")
        self._buffer.data[:] = data

    def pixel(self, x, y):
        """Native-format tuple at (x, y), in *drawable* coordinates."""
        comps = format_components(self._buffer.format)
        off = (y * self._buffer.width + x) * comps
        return tuple(self._buffer.data[off : off + comps])


class Layer(Drawable):
    def __init__(self, image, name, width, height, image_type, offsets=(0, 0)):
        Drawable.__init__(
            self, image, name, width, height, _TYPE_FORMAT[image_type], offsets
        )
        self.image_type = image_type
        self.mode = LayerMode.NORMAL
        self.mask = None
        self.edit_mask = False

    # -- constructors ------------------------------------------------------ #
    @staticmethod
    def new(image, name, width, height, image_type, opacity, mode):
        Gimp.calls.append(("Layer.new", (name, width, height, image_type)))
        layer = Layer(image, name, width, height, image_type)
        layer.opacity = float(opacity)
        layer.mode = mode
        return layer

    @staticmethod
    def new_from_drawable(drawable, dest_image):
        """``gimp_layer_new_from_drawable`` -- copy pixels into another image."""
        Gimp.calls.append(("Layer.new_from_drawable", (drawable.get_name(),)))
        fmt = drawable.get_format()
        image_type = ImageType.RGBA_IMAGE
        for t, f in _TYPE_FORMAT.items():
            if f == fmt:
                image_type = t
                break
        ok, ox, oy = drawable.get_offsets()
        layer = Layer(
            dest_image,
            drawable.get_name(),
            drawable.get_width(),
            drawable.get_height(),
            image_type,
            (ox, oy),
        )
        layer._buffer.data[:] = drawable.get_buffer().data
        layer.opacity = drawable.get_opacity()
        return layer

    @staticmethod
    def new_from_visible(image, dest_image, name):
        """``gimp_layer_new_from_visible`` -- the projection, as a layer.

        This is the cheap way to get "what the user sees": one composite pass,
        no duplication of the layer stack.
        """
        Gimp.calls.append(("Layer.new_from_visible", (name,)))
        rgba = _project(image, background=None)
        layer = Layer(
            dest_image, name, image.get_width(), image.get_height(), ImageType.RGBA_IMAGE
        )
        layer._buffer.data[:] = rgba
        return layer

    # -- duplication ------------------------------------------------------- #
    def copy(self):
        """``gimp_layer_copy`` -- a detached copy, not yet in any image.

        ``outputs._duplicate_for_mask`` copies the source layer before masking
        it so the user's original is never touched; the copy keeps the offsets
        and the caller inserts it.
        """
        Gimp.calls.append(("Layer.copy", (self._name,)))
        dup = Layer(
            self._image, self._name, self.get_width(), self.get_height(),
            self.image_type, tuple(self._offsets),
        )
        dup._buffer.data[:] = self._buffer.data
        dup.opacity = self.opacity
        dup.mode = self.mode
        dup.visible = self.visible
        return dup

    def add_alpha(self):
        """``gimp_layer_add_alpha`` -- promote RGB to RGBA (opaque)."""
        Gimp.calls.append(("Layer.add_alpha", (self._name,)))
        if self.image_type == ImageType.RGB_IMAGE:
            new_type = ImageType.RGBA_IMAGE
        elif self.image_type == ImageType.GRAY_IMAGE:
            new_type = ImageType.GRAYA_IMAGE
        else:
            return True
        old_fmt = self.get_format()
        n = self.get_width() * self.get_height()
        data = _convert(self._buffer.data, old_fmt, _TYPE_FORMAT[new_type], n)
        self.image_type = new_type
        self._buffer = Buffer(
            0, 0, self.get_width(), self.get_height(), _TYPE_FORMAT[new_type], data
        )
        return True

    # -- masks ------------------------------------------------------------- #
    def create_mask(self, mask_type):
        """``gimp_layer_create_mask``.

        ``AddMaskType.SELECTION`` is modelled properly because it is the whole
        mechanism ``outputs._mask_from_selection`` relies on: the mask is
        layer-sized and layer-positioned, and GIMP -- not the plug-in -- does the
        image-space to layer-space translation.  Getting that wrong is exactly
        the bug the layer-mask output mode would ship.
        """
        Gimp.calls.append(("Layer.create_mask", (mask_type,)))
        mask = LayerMask(
            self._image, self._name + " mask", self.get_width(), self.get_height()
        )
        if mask_type == AddMaskType.WHITE:
            mask._buffer.data[:] = b"\xff" * len(mask._buffer.data)
        elif mask_type == AddMaskType.SELECTION and self._image is not None:
            sel = self._image.get_selection().get_buffer()
            iw, ih = sel.width, sel.height
            ox, oy = self._offsets
            w, h = self.get_width(), self.get_height()
            out = mask._buffer.data
            for y in range(h):
                sy = y + oy
                if not (0 <= sy < ih):
                    continue
                for x in range(w):
                    sx = x + ox
                    if 0 <= sx < iw:
                        out[y * w + x] = sel.data[sy * iw + sx]
        return mask

    def add_mask(self, mask):
        """``gimp-layer-add-mask``: attaches the mask *and* turns on
        edit-mask (``gimp_layer_add_mask (layer, mask, TRUE, TRUE, ...)``),
        so the image's selected drawable becomes the mask from here on."""
        self.mask = mask
        mask._layer = self
        mask._offsets = list(self._offsets)
        self.edit_mask = True
        image = self._image
        if isinstance(image, Image) and image._is_attached(self):
            def undo():
                self.mask, self.edit_mask = None, False

            def redo():
                self.mask, self.edit_mask = mask, True

            image._undo_record("layer-mask-add", undo, redo)
        return True

    def get_mask(self):
        return self.mask

    def get_edit_mask(self):
        return self.edit_mask

    def set_edit_mask(self, edit):
        self.edit_mask = bool(edit) and self.mask is not None
        return True

    def remove_mask(self, mode=0):
        mask = self.mask
        self.mask = None
        self.edit_mask = False
        image = self._image
        if mask is not None and isinstance(image, Image) and image._is_attached(self):
            def undo():
                self.mask = mask

            def redo():
                self.mask = None

            image._undo_record("layer-mask-remove", undo, redo)
        return True

    @staticmethod
    def from_mask(mask):
        """``gimp_layer_from_mask`` -- the layer a mask belongs to."""
        Gimp.calls.append(("Layer.from_mask", (mask.get_name(),)))
        return getattr(mask, "_layer", None)


def _drawable_transform_scale(self, x0, y0, x1, y1):
    """``gimp_item_transform_scale`` -- resize *and* place in one call.

    ``outputs._scale_item_into`` leans on both halves: the soft mask crop is
    created at model resolution and this is what brings it to the rectangle
    §9 computed, in image space.  Nearest-neighbour here (see the module
    docstring); assert on placement and on constant regions, never on edges.
    """
    Gimp.calls.append(
        ("Item.transform_scale", (self.get_name(), float(x0), float(y0), float(x1), float(y1)))
    )
    new_w = max(1, int(round(float(x1) - float(x0))))
    new_h = max(1, int(round(float(y1) - float(y0))))
    comps = format_components(self.get_format())
    data = _resample_nearest(
        self._buffer.data, comps, self.get_width(), self.get_height(), new_w, new_h
    )
    if isinstance(self, Channel):
        # GIMP: gimp_channel_get_clip() returns GIMP_TRANSFORM_RESIZE_CLIP
        # unconditionally and gimp_channel_scale() pins the offset at (0, 0),
        # so a channel keeps its bounds and only the part of the scaled result
        # that lands inside them survives.  Modelled faithfully because the
        # plug-in once relied on the opposite and every Apply came out blank.
        old_w, old_h = self.get_width(), self.get_height()
        off_x, off_y = int(round(float(x0))), int(round(float(y0)))
        kept = bytearray(old_w * old_h * comps)
        for yy in range(old_h):
            sy = yy - off_y
            if not (0 <= sy < new_h):
                continue
            for xx in range(old_w):
                sx = xx - off_x
                if 0 <= sx < new_w:
                    src = (sy * new_w + sx) * comps
                    dst = (yy * old_w + xx) * comps
                    kept[dst:dst + comps] = data[src:src + comps]
        self._buffer = Buffer(0, 0, old_w, old_h, self.get_format(), kept)
        return self
    self._buffer = Buffer(0, 0, new_w, new_h, self.get_format(), data)
    self.set_offsets(int(round(float(x0))), int(round(float(y0))))
    return self


Drawable.transform_scale = _drawable_transform_scale


class Channel(Drawable):
    def __init__(self, image, name, width, height, offsets=(0, 0)):
        Drawable.__init__(self, image, name, width, height, "Y' u8", offsets)
        self.show_masked = False
        self.color = Color("black")

    @staticmethod
    def new(image, name, width, height, opacity, color):
        """``gimp_channel_new`` -- a new channel is filled with 0."""
        Gimp.calls.append(("Channel.new", (name, width, height)))
        ch = Channel(image, name, width, height)
        ch.opacity = float(opacity)
        ch.color = color
        ch.visible = False
        return ch


class LayerMask(Channel):
    """``Gimp.LayerMask`` -- a channel, not a layer, owned by one layer.

    ``copy()`` gives a plain channel, as ``gimp_channel_copy`` does, which is
    why a plug-in handed a mask as its drawable must map it back to its layer
    (``Gimp.Layer.from_mask``) before copying "the layer".
    """

    def __init__(self, image, name, width, height, offsets=(0, 0)):
        Channel.__init__(self, image, name, width, height, offsets)
        self._layer = None

    def copy(self):
        Gimp.calls.append(("Channel.copy", (self._name,)))
        dup = Channel(self._image, self._name, self.get_width(), self.get_height(),
                      tuple(self._offsets))
        dup._buffer.data[:] = self._buffer.data
        return dup


class GroupLayer(Layer):
    """A layer group.  ``get_children()`` returns its members, top first."""

    def __init__(self, image, name):
        Layer.__init__(
            self, image, name, image.get_width(), image.get_height(), ImageType.RGBA_IMAGE
        )
        self.children = []

    @staticmethod
    def new(image, name=None):
        """``gimp_group_layer_new`` -- it gained the ``name`` argument during the
        GIMP 3.0 cycle, so both arities exist in the wild and ``outputs._new_group``
        tries them in order.  Modelling both is what makes that fallback testable.
        """
        Gimp.calls.append(("GroupLayer.new", (name,)))
        return GroupLayer(image, name if name is not None else "Layer Group")

    def is_group(self):
        return True

    def get_children(self):
        return list(self.children)


# --------------------------------------------------------------------------- #
# compositing
# --------------------------------------------------------------------------- #
def _visible_layers_bottom_up(layers):
    out = []
    for layer in reversed(list(layers)):
        if not layer.visible:
            continue
        if layer.is_group():
            out.extend(_visible_layers_bottom_up(layer.get_children()))
        else:
            out.append(layer)
    return out


def _project(image, background=None):
    """Composite the image's visible layers.

    ``background`` is ``None`` (transparent base -> RGBA result) or an
    ``(r, g, b)`` byte triple (opaque base).  Always returns ``R'G'B'A u8``
    bytes of ``image`` size; the caller converts.
    """
    width, height = image.get_width(), image.get_height()
    npix = width * height
    layers = _visible_layers_bottom_up(image.get_layers())

    if background is None:
        base = bytearray(npix * 4)
    else:
        r, g, b = background
        base = bytearray(bytes((r, g, b, 255)) * npix)

    # Fast path: a single full-canvas opaque layer replaces the base outright.
    if len(layers) == 1:
        only = layers[0]
        ok, ox, oy = only.get_offsets()
        if (
            (ox, oy) == (0, 0)
            and only.get_width() == width
            and only.get_height() == height
            and float(only.opacity) >= 100.0
        ):
            rgba = _convert(
                only.get_buffer().data, only.get_format(), "R'G'B'A u8", npix
            )
            if background is None or rgba[3::4] == b"\xff" * npix:
                return bytearray(rgba)

    for layer in layers:
        ok, ox, oy = layer.get_offsets()
        lw, lh = layer.get_width(), layer.get_height()
        rgba = _convert(layer.get_buffer().data, layer.get_format(), "R'G'B'A u8", lw * lh)
        mask = layer.get_mask()
        mask_data = mask.get_buffer().data if mask is not None else None
        opacity = float(layer.opacity) / 100.0
        for ly in range(lh):
            iy = oy + ly
            if iy < 0 or iy >= height:
                continue
            for lx in range(lw):
                ix = ox + lx
                if ix < 0 or ix >= width:
                    continue
                s = (ly * lw + lx) * 4
                a = rgba[s + 3] / 255.0 * opacity
                if mask_data is not None:
                    a *= mask_data[ly * lw + lx] / 255.0
                if a <= 0.0:
                    continue
                d = (iy * width + ix) * 4
                da = base[d + 3] / 255.0
                out_a = a + da * (1.0 - a)
                for c in range(3):
                    src = rgba[s + c] / 255.0
                    dst = base[d + c] / 255.0
                    value = (src * a + dst * da * (1.0 - a)) / out_a if out_a else 0.0
                    base[d + c] = max(0, min(255, int(round(value * 255.0))))
                base[d + 3] = max(0, min(255, int(round(out_a * 255.0))))
    return base


# --------------------------------------------------------------------------- #
# Gimp.Image
# --------------------------------------------------------------------------- #
class Image(UndoModel):
    def __init__(self, width, height, base_type=ImageBaseType.RGB):
        self.id = _Item._next_id
        _Item._next_id += 1
        self._width = int(width)
        self._height = int(height)
        self._base_type = base_type
        self._layers = []
        self._channels = []
        self._selected = []
        self._selected_channels = []
        self._valid = True
        self.undo_depth = 0
        self.undo_groups = []
        self._undo_setup()
        self._selection = Channel(self, "Selection", self._width, self._height)
        _REGISTRY.append(self)

    # -- construction ------------------------------------------------------ #
    @staticmethod
    def new(width, height, base_type):
        Gimp.calls.append(("Image.new", (width, height, base_type)))
        return Image(width, height, base_type)

    # -- basics ------------------------------------------------------------ #
    def get_id(self):
        return self.id

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height

    def get_base_type(self):
        return self._base_type

    def is_valid(self):
        return self._valid

    def get_precision(self):
        return 100  # Gimp.Precision.U8_NON_LINEAR, value irrelevant to the stubs

    # -- layers / channels -------------------------------------------------- #
    def get_layers(self):
        return list(self._layers)

    def get_selected_layers(self):
        return list(self._selected)

    def set_selected_layers(self, layers):
        """Selecting layers deselects every channel (``gimpimage.c``,
        ``gimp_image_selected_layers_notify``)."""
        self._selected = [l for l in layers if self._is_attached(l)]
        if self._selected:
            self._selected_channels = []
        return True

    def get_selected_channels(self):
        return list(self._selected_channels)

    def set_selected_channels(self, channels):
        """...and selecting channels deselects every layer."""
        self._selected_channels = list(channels)
        if self._selected_channels:
            self._selected = []
        return True

    def get_selected_drawables(self):
        """``gimp_image_get_selected_drawables``: the selected channels when
        there are any; otherwise the selected layers, except that a lone
        layer whose mask is being edited is reported as that *mask*."""
        if self._selected_channels:
            return list(self._selected_channels)
        drawables = list(self._selected)
        if len(drawables) == 1:
            mask = drawables[0].get_mask()
            if mask is not None and getattr(drawables[0], "edit_mask", False):
                drawables = [mask]
        return drawables

    def _layer_list_for(self, layer):
        for group in self._all_groups():
            if layer in group.children:
                return group.children
        return self._layers

    def _all_groups(self):
        out = []
        pending = list(self._layers)
        while pending:
            item = pending.pop()
            if item.is_group():
                out.append(item)
                pending.extend(item.children)
        return out

    def _is_attached(self, item):
        if item in self._channels or item in self._layers:
            return True
        if any(item in group.children for group in self._all_groups()):
            return True
        owner = getattr(item, "_layer", None)
        return owner is not None and owner.get_mask() is item and self._is_attached(owner)

    def insert_layer(self, layer, parent=None, position=0):
        """``gimp_image_add_layer``: the new layer becomes the selected one."""
        Gimp.calls.append(("Image.insert_layer", (layer.get_name(), position)))
        layer._image = self
        target = parent.children if parent is not None and parent.is_group() else self._layers
        position = len(target) if position < 0 else min(position, len(target))
        target.insert(position, layer)
        self._record_item_added("layer-add", lambda: target, layer, position,
                                self._is_attached)
        self.set_selected_layers([layer])
        return True

    def remove_layer(self, layer):
        target = self._layer_list_for(layer)
        if layer in target:
            position = target.index(layer)
            target.remove(layer)
            self._record_item_removed("layer-remove", lambda: target, layer, position,
                                      self._is_attached)
        if layer in self._selected:
            self._selected = [l for l in self._selected if l is not layer]
        return True

    def get_channels(self):
        return list(self._channels)

    def insert_channel(self, channel, parent=None, position=0):
        """``gimp_image_add_channel``: the new channel becomes the selected
        drawable, which deselects every layer."""
        Gimp.calls.append(("Image.insert_channel", (channel.get_name(), position)))
        channel._image = self
        position = len(self._channels) if position < 0 else min(position, len(self._channels))
        self._channels.insert(position, channel)
        self._record_item_added("channel-add", lambda: self._channels, channel, position,
                                self._is_attached)
        self.set_selected_channels([channel])
        return True

    def remove_channel(self, channel):
        if channel not in self._channels:
            return False  # the PDB refuses an item that is not in the image
        position = self._channels.index(channel)
        self._channels.remove(channel)
        self._record_item_removed("channel-remove", lambda: self._channels, channel,
                                  position, self._is_attached)
        if channel in self._selected_channels:
            self._selected_channels = [c for c in self._selected_channels if c is not channel]
        return True

    # -- selection ---------------------------------------------------------- #
    def get_selection(self):
        return self._selection

    def select_rectangle(self, operation, x, y, width, height):
        Gimp.calls.append(("Image.select_rectangle", (operation, x, y, width, height)))
        self.set_selection_rect(x, y, width, height)
        return True

    def _record_selection(self):
        """Every change to the selection mask pushes ``GIMP_UNDO_MASK``."""
        buf = self._selection._buffer

        def put(data):
            buf.data[:] = data

        self._record_mask(lambda: buf.data, put)

    def select_item(self, operation, item):
        """``gimp_image_select_item`` -- combine an item's mask with the selection.

        The operation is honoured rather than ignored: ``outputs.apply_result``
        unions N instance channels with ``ADD`` and its whole per-instance path
        would look correct while producing only the last mask if this replaced
        every time.
        """
        Gimp.calls.append(("Image.select_item", (operation, item.get_name())))
        self._record_selection()
        target = self._selection._buffer.data
        source = item.get_buffer().data
        # An item smaller than the image (a channel is image-sized here, but a
        # layer need not be) contributes at its own offsets.
        ox, oy = item.get_offsets()[1:3] if len(item.get_offsets()) == 3 else (0, 0)
        iw, ih = item.get_width(), item.get_height()
        if operation == ChannelOps.REPLACE:
            target[:] = bytearray(len(target))
        for y in range(ih):
            ty = y + oy
            if not (0 <= ty < self._height):
                continue
            for x in range(iw):
                tx = x + ox
                if not (0 <= tx < self._width):
                    continue
                si = source[y * iw + x]
                ti = ty * self._width + tx
                if operation in (ChannelOps.REPLACE, ChannelOps.ADD):
                    if si > target[ti]:
                        target[ti] = si
                elif operation == ChannelOps.SUBTRACT:
                    target[ti] = max(0, target[ti] - si)
                elif operation == ChannelOps.INTERSECT:
                    target[ti] = min(target[ti], si)
                else:
                    raise ValueError("unknown ChannelOps %r" % (operation,))
        if operation == ChannelOps.INTERSECT:
            # Pixels outside a smaller item are outside the intersection.
            for y in range(self._height):
                for x in range(self._width):
                    if not (ox <= x < ox + iw and oy <= y < oy + ih):
                        target[y * self._width + x] = 0
        return True

    # test convenience (not real API)
    def set_selection_rect(self, x, y, width, height):
        buf = self._selection._buffer
        buf.data[:] = bytearray(len(buf.data))
        row = b"\xff" * width
        for yy in range(y, y + height):
            if 0 <= yy < self._height:
                off = yy * self._width + x
                buf.data[off : off + width] = row

    # -- whole-image operations --------------------------------------------- #
    def duplicate(self):
        Gimp.calls.append(("Image.duplicate", (self.id,)))
        dup = Image(self._width, self._height, self._base_type)
        for layer in self._layers:
            copy = Layer.new_from_drawable(layer, dup)
            copy.visible = layer.visible
            copy.opacity = layer.opacity
            dup._layers.append(copy)
        for channel in self._channels:
            copy = Channel(dup, channel.get_name(), channel.get_width(), channel.get_height())
            copy._buffer.data[:] = channel._buffer.data
            dup._channels.append(copy)
        dup._selection._buffer.data[:] = self._selection._buffer.data
        return dup

    def flatten(self):
        """``gimp_image_flatten`` -- composite onto the context background."""
        Gimp.calls.append(("Image.flatten", (self.id,)))
        bg = _CONTEXT[-1]["background"].as_bytes()[:3]
        rgba = _project(self, background=bg)
        rgb = _convert(rgba, "R'G'B'A u8", "R'G'B' u8", self._width * self._height)
        layer = Layer(self, "Background", self._width, self._height, ImageType.RGB_IMAGE)
        layer._buffer.data[:] = rgb
        self._layers = [layer]
        self._selected = [layer]
        self._base_type = ImageBaseType.RGB
        return layer

    def merge_visible_layers(self, merge_type):
        Gimp.calls.append(("Image.merge_visible_layers", (merge_type,)))
        return self.flatten()

    def scale(self, width, height):
        """``gimp_image_scale`` -- nearest-neighbour in the stub (see module docstring)."""
        Gimp.calls.append(("Image.scale", (width, height)))
        sx = float(width) / float(self._width)
        sy = float(height) / float(self._height)
        for drawable in list(self._layers) + list(self._channels) + [self._selection]:
            comps = format_components(drawable.get_format())
            new_w = max(1, int(round(drawable.get_width() * sx)))
            new_h = max(1, int(round(drawable.get_height() * sy)))
            data = _resample_nearest(
                drawable._buffer.data,
                comps,
                drawable.get_width(),
                drawable.get_height(),
                new_w,
                new_h,
            )
            drawable._buffer = Buffer(0, 0, new_w, new_h, drawable.get_format(), data)
            ok, ox, oy = drawable.get_offsets()
            drawable.set_offsets(int(round(ox * sx)), int(round(oy * sy)))
        self._width = int(width)
        self._height = int(height)
        return True

    def resize(self, width, height, offx, offy):
        Gimp.calls.append(("Image.resize", (width, height, offx, offy)))
        self._width = int(width)
        self._height = int(height)
        return True

    def delete(self):
        Gimp.calls.append(("Image.delete", (self.id,)))
        self._valid = False
        if self in _REGISTRY:
            _REGISTRY.remove(self)
        return True

    # -- undo ---------------------------------------------------------------- #
    def undo_group_start(self):
        self.undo_depth += 1
        self.undo_groups.append("start")
        self._undo_group_opened()
        return True

    def undo_group_end(self):
        self.undo_depth -= 1
        self.undo_groups.append("end")
        self._undo_group_closed()
        return True

    def _undo_is_frozen(self):
        return getattr(self, "undo_freeze_count", 0) > 0

    def undo_freeze(self):
        """``gimp_image_undo_freeze``: steps pushed while frozen are dropped."""
        self.undo_freeze_count = getattr(self, "undo_freeze_count", 0) + 1
        Gimp.calls.append(("Image.undo_freeze", (self.get_id(),)))
        return True

    def undo_thaw(self):
        assert getattr(self, "undo_freeze_count", 0) > 0, "thaw without freeze"
        self.undo_freeze_count -= 1
        Gimp.calls.append(("Image.undo_thaw", (self.get_id(),)))
        return True

    def undo_disable(self):
        return True

    def undo_enable(self):
        return True

    def clean_all(self):
        return True

    def __repr__(self):
        return "<fake Gimp.Image #%d %dx%d>" % (self.id, self._width, self._height)


# --------------------------------------------------------------------------- #
# Gimp.Selection (namespace of static helpers, as in libgimp)
# --------------------------------------------------------------------------- #
class Selection(object):
    @staticmethod
    def bounds(image):
        """``(ok, non_empty, x1, y1, x2, y2)`` -- the GIMP 3 shape, with the
        PDB success flag first (observed on 3.2); half-open on the far edge."""
        buf = image.get_selection().get_buffer()
        data = buf.data
        w, h = buf.width, buf.height
        x0, y0, x1, y1 = w, h, 0, 0
        found = False
        for y in range(h):
            row = data[y * w : (y + 1) * w]
            if not any(row):
                continue
            found = True
            first = next(i for i, v in enumerate(row) if v)
            last = w - 1 - next(i for i, v in enumerate(reversed(row)) if v)
            x0 = min(x0, first)
            x1 = max(x1, last + 1)
            y0 = min(y0, y)
            y1 = max(y1, y + 1)
        if not found:
            return (True, False, 0, 0, 0, 0)
        return (True, True, x0, y0, x1, y1)

    @staticmethod
    def none(image):
        Gimp.calls.append(("Selection.none", (image.get_id(),)))
        image._record_selection()
        buf = image.get_selection().get_buffer()
        buf.data[:] = bytearray(len(buf.data))
        return True

    @staticmethod
    def all(image):
        Gimp.calls.append(("Selection.all", (image.get_id(),)))
        image._record_selection()
        buf = image.get_selection().get_buffer()
        buf.data[:] = b"\xff" * len(buf.data)
        return True

    @staticmethod
    def is_empty(image):
        return not Selection.bounds(image)[1]

    # -- the helpers outputs.py drives ------------------------------------- #
    @staticmethod
    def save(image):
        """``gimp_selection_save`` -- snapshot the selection into a channel.

        The real call inserts the channel into the image, which is why
        ``outputs`` can hand the result straight back to ``select_item`` and
        why its scratch bookkeeping has to remove it again.
        """
        Gimp.calls.append(("Selection.save", (image.get_id(),)))
        channel = Channel(image, "Selection Mask", image.get_width(), image.get_height())
        channel.visible = False
        channel._buffer.data[:] = image.get_selection().get_buffer().data
        # gimp_image_add_channel (image, channel, ..., push_undo=TRUE)
        image.insert_channel(channel, None, 0)
        return channel

    @staticmethod
    def load(item):
        """``gimp_selection_load`` -- replace the selection with a channel."""
        image = item.get_image()
        Gimp.calls.append(("Selection.load", (item.get_name(),)))
        image._record_selection()
        image.get_selection().get_buffer().data[:] = item.get_buffer().data
        return True

    @staticmethod
    def grow(image, steps):
        """``gimp_selection_grow``.

        A square (Chebyshev) dilation, not GIMP's circular kernel: the stub is
        for checking that ``outputs`` *delegates* morphology to GIMP and in what
        order, not for matching GIMP's edge pixels.  Tests that care about the
        exact kernel belong on a real GIMP.
        """
        Gimp.calls.append(("Selection.grow", (image.get_id(), int(steps))))
        image._record_selection()
        _morph(image, int(steps), grow=True)
        return True

    @staticmethod
    def shrink(image, steps):
        """``gimp_selection_shrink`` -- the erosion counterpart of :meth:`grow`."""
        Gimp.calls.append(("Selection.shrink", (image.get_id(), int(steps))))
        image._record_selection()
        _morph(image, int(steps), grow=False)
        return True

    @staticmethod
    def feather(image, radius):
        """``gimp_selection_feather`` -- a separable box blur of the mask."""
        Gimp.calls.append(("Selection.feather", (image.get_id(), float(radius))))
        image._record_selection()
        _box_blur(image, float(radius))
        return True

    @staticmethod
    def sharpen(image):
        """``gimp_selection_sharpen`` -- binarise the (possibly feathered) mask."""
        Gimp.calls.append(("Selection.sharpen", (image.get_id(),)))
        image._record_selection()
        buf = image.get_selection().get_buffer()
        buf.data[:] = bytes(255 if v >= 128 else 0 for v in buf.data)
        return True


def _morph(image, steps, grow):
    """Chebyshev dilation/erosion of the selection mask, ``steps`` pixels."""
    if steps <= 0:
        return
    buf = image.get_selection().get_buffer()
    w, h = buf.width, buf.height
    src = bytearray(buf.data)
    pick = max if grow else min
    outside = 0 if grow else 255   # erosion must not eat the border from nothing
    for _ in range(steps):
        dst = bytearray(len(src))
        for y in range(h):
            for x in range(w):
                best = src[y * w + x]
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        v = src[ny * w + nx] if (0 <= ny < h and 0 <= nx < w) else outside
                        best = pick(best, v)
                dst[y * w + x] = best
        src = dst
    buf.data[:] = src


def _box_blur(image, radius):
    """Separable box blur of the selection mask; ``radius`` in pixels.

    A box blur, not GIMP's gaussian-ish feather: enough to turn a hard edge into
    a ramp so that a test can tell "feathered" from "not feathered", and no more.
    """
    r = int(round(radius))
    if r <= 0:
        return
    buf = image.get_selection().get_buffer()
    w, h = buf.width, buf.height
    src = list(buf.data)

    # horizontal pass
    mid = [0] * (w * h)
    for y in range(h):
        base = y * w
        for x in range(w):
            lo, hi = max(0, x - r), min(w - 1, x + r)
            mid[base + x] = sum(src[base + lo:base + hi + 1]) // (hi - lo + 1)

    # vertical pass
    dst = [0] * (w * h)
    for x in range(w):
        for y in range(h):
            lo, hi = max(0, y - r), min(h - 1, y + r)
            total = 0
            for yy in range(lo, hi + 1):
                total += mid[yy * w + x]
            dst[y * w + x] = total // (hi - lo + 1)

    buf.data[:] = bytes(dst)


# --------------------------------------------------------------------------- #
# module-level Gimp functions (context, displays, progress)
# --------------------------------------------------------------------------- #
_REGISTRY = []
_CONTEXT = [
    {
        "background": Color("white"),
        "foreground": Color("black"),
        "interpolation": InterpolationType.CUBIC,
    }
]


def live_images():
    """Images created through the stubs and not yet ``delete()``d.

    A test that drives ``gimpbridge`` should assert this list is back to its
    starting contents -- temporary work images must never leak.
    """
    return list(_REGISTRY)


def _context_push():
    Gimp.calls.append(("context_push", ()))
    _CONTEXT.append(dict(_CONTEXT[-1]))
    return True


def _context_pop():
    Gimp.calls.append(("context_pop", ()))
    if len(_CONTEXT) > 1:
        _CONTEXT.pop()
    return True


def _context_set_background(color):
    Gimp.calls.append(("context_set_background", (color,)))
    _CONTEXT[-1]["background"] = color
    return True


def _context_set_foreground(color):
    Gimp.calls.append(("context_set_foreground", (color,)))
    _CONTEXT[-1]["foreground"] = color
    return True


def _context_get_background():
    return _CONTEXT[-1]["background"]


def _context_set_interpolation(interpolation):
    Gimp.calls.append(("context_set_interpolation", (interpolation,)))
    _CONTEXT[-1]["interpolation"] = interpolation
    return True


def _context_get_interpolation():
    return _CONTEXT[-1]["interpolation"]


def _displays_flush():
    Gimp.calls.append(("displays_flush", ()))
    return True


def _progress_init(text):
    Gimp.calls.append(("progress_init", (text,)))
    return True


def _progress_update(fraction):
    Gimp.calls.append(("progress_update", (fraction,)))
    return True


def _progress_end():
    Gimp.calls.append(("progress_end", ()))
    return True


def _message(text):
    Gimp.calls.append(("message", (text,)))
    return True


def reset():
    """Clear recorded calls, the live-image registry and the context stack."""
    del Gimp.calls[:]
    del Gegl.calls[:]
    del _REGISTRY[:]
    del _CONTEXT[1:]
    _CONTEXT[0] = {
        "background": Color("white"),
        "foreground": Color("black"),
        "interpolation": InterpolationType.CUBIC,
    }


# --------------------------------------------------------------------------- #
# test-data helpers
# --------------------------------------------------------------------------- #
def make_rgb_bytes(width, height, fn=None):
    """Deterministic ``R'G'B' u8`` bytes; ``fn(x, y) -> (r, g, b)``."""
    if fn is None:

        def fn(x, y):
            return (x & 0xFF, y & 0xFF, (x + y) & 0xFF)

    out = bytearray(width * height * 3)
    i = 0
    for y in range(height):
        for x in range(width):
            r, g, b = fn(x, y)
            out[i] = r & 0xFF
            out[i + 1] = g & 0xFF
            out[i + 2] = b & 0xFF
            i += 3
    return bytes(out)


# --------------------------------------------------------------------------- #
# assemble the fake gi.repository modules
# --------------------------------------------------------------------------- #
def _module(name, attrs):
    mod = types.ModuleType(name)
    mod.__fake_gimp__ = True
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


Gegl = _module(
    "gi.repository.Gegl",
    {
        "Rectangle": Rectangle,
        "Buffer": Buffer,
        "Color": Color,
        "AbyssPolicy": AbyssPolicy,
        "init": _gegl_init,
        "exit": lambda: True,
        "calls": [],
    },
)

Babl = _module(
    "gi.repository.Babl",
    {
        "format": lambda name: name,
        "init": lambda: True,
        "component_count": format_components,
    },
)

Gimp = _module(
    "gi.repository.Gimp",
    {
        "Image": Image,
        "Layer": Layer,
        "GroupLayer": GroupLayer,
        "Channel": Channel,
        "LayerMask": LayerMask,
        "Drawable": Drawable,
        "Item": _Item,
        "Selection": Selection,
        "ImageBaseType": ImageBaseType,
        "ImageType": ImageType,
        "LayerMode": LayerMode,
        "InterpolationType": InterpolationType,
        "ChannelOps": ChannelOps,
        "MergeType": MergeType,
        "FillType": FillType,
        "AddMaskType": AddMaskType,
        "context_push": _context_push,
        "context_pop": _context_pop,
        "context_set_background": _context_set_background,
        "context_get_background": _context_get_background,
        "context_set_foreground": _context_set_foreground,
        "context_set_interpolation": _context_set_interpolation,
        "context_get_interpolation": _context_get_interpolation,
        "displays_flush": _displays_flush,
        "progress_init": _progress_init,
        "progress_update": _progress_update,
        "progress_end": _progress_end,
        "message": _message,
        "calls": [],
        "reset": reset,
        "live_images": live_images,
    },
)

#: name -> fake module, as installed into ``sys.modules``.
MODULES = {
    "gi.repository.Gimp": Gimp,
    "gi.repository.Gegl": Gegl,
    "gi.repository.Babl": Babl,
}
