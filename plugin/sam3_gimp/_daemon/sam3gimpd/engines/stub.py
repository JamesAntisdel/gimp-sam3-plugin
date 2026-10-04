"""The stub engine: synthetic but structurally valid masks, with no torch.

``sam3gimpd serve --stub`` is a **product feature**, not a test fixture.  SAM 3's
weights are gated and a GPU is not always at hand, so the stub is how the GIMP
plug-in, the GTK canvas, the job queue, supersession, progress bars and the
whole HTTP contract get developed and exercised on an ordinary laptop.  A stub
daemon is a fully conforming implementation of ``API.md``; see §14 there for the
guarantees this module keeps:

* it imports **no torch, no transformers, no numpy**, on any code path;
* results are deterministic -- the same ``(image_id, prompt)`` produces
  byte-identical frames, because all geometry comes from a BLAKE2b digest of
  those inputs rather than from an unseeded RNG;
* masks are genuine **soft** uint8 gradients with an anti-aliased falloff, so a
  client's threshold slider visibly does something.  A constant-255 rectangle
  would hide client bugs, which is the whole reason this is not a fixture;
* the value at the nominal blob boundary is 128 -- the same "logit 0" point the
  real engine produces (``API.md`` §8.3) -- so a client that defaults to 128
  sees the shape it asked for;
* scores lie in ``[0.1, 1.0]`` and descend; PCS returns 1-8 instances chosen by
  the prompt hash, PVS returns up to ``max_instances`` candidates centred on
  the click;
* stages advance in the same order and the same progress bands as the real
  engine, with simulated latency of a few hundred milliseconds.

Rendering performance matters here: this runs in the daemon's single inference
worker and must not take longer than a real GPU would.  Two tricks keep pure
Python fast enough:

* the elliptical falloff is a 1024-entry lookup table indexed by the *squared*
  normalised radius, so no ``sqrt`` runs per pixel, and only the thin feather
  band is evaluated per pixel at all -- the interior is a slice fill;
* the rectangular falloff is separable, so each row is either a copy of a
  precomputed column profile or that profile passed through a 256-byte
  ``bytes.translate`` table, both of which run in C.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from ..types import (
    ApiError,
    BBox,
    ErrorCode,
    PointPrompt,
    Size,
    TextPrompt,
)
from .base import (
    DEFAULT_CANVAS_SIDE,
    BaseEngine,
    EncodedImage,
    EngineInfo,
    ImageData,
    ProgressFn,
    ProgressReporter,
    PromptResult,
    RawInstance,
    Stage,
    finalize_instances as _finalize,
)

__all__ = ["StubEngine", "render_ellipse", "render_rect", "stub_seed"]

#: Relative width of the anti-aliased band, as a fraction of the radius.
_FEATHER = 0.13

#: Resolution of the falloff lookup table.
_RAMP_STEPS = 1024

#: Nominal per-prompt latency reported in the result header, in milliseconds.
#: It is a *constant*, not a measurement: ``API.md`` §14 promises that the same
#: ``(image_id, prompt)`` yields a **byte-identical** frame, and a wall-clock
#: ``elapsed_ms`` in the JSON header would break that for no benefit.  The
#: simulated sleeps still make the job take real time; only the reported number
#: is fixed.
_NOMINAL_TEXT_MS = 120.0
_NOMINAL_POINTS_MS = 85.0
#: Added per returned instance, so the number moves with the work as a real
#: engine's would.
_NOMINAL_PER_INSTANCE_MS = 4.0

_SHAPE_ELLIPSE = 0
_SHAPE_RECT = 1


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def stub_seed(*parts: Any) -> int:
    """A stable 64-bit seed from the given parts.

    ``hash()`` is deliberately not used: it is salted per process
    (``PYTHONHASHSEED``), which would break the byte-identical-results guarantee
    across daemon restarts.
    """
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(repr(part).encode("utf-8"))
        h.update(b"\x1f")
    return int.from_bytes(h.digest(), "little")


class _Rng:
    """A tiny deterministic PRNG (SplitMix64).

    ``random.Random`` would work, but its stream is only guaranteed stable
    within a CPython release line.  A dozen lines of SplitMix64 make the "same
    prompt, same bytes" guarantee independent of the interpreter version, which
    matters when a Windows GIMP and a Linux daemon must agree on what the stub
    drew.
    """

    __slots__ = ("_s",)

    _MASK = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        self._s = int(seed) & self._MASK

    def next_u64(self) -> int:
        self._s = (self._s + 0x9E3779B97F4A7C15) & self._MASK
        z = self._s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self._MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self._MASK
        return z ^ (z >> 31)

    def random(self) -> float:
        """Uniform in ``[0, 1)`` with 53 bits of resolution."""
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.random()

    def randint(self, lo: int, hi: int) -> int:
        """Inclusive on both ends."""
        span = int(hi) - int(lo) + 1
        return int(lo) + int(self.next_u64() % span)


# --------------------------------------------------------------------------- #
# soft blob rendering
# --------------------------------------------------------------------------- #
_RAMP_CACHE: Dict[Tuple[int, int], bytes] = {}


def _ramp_table(feather: float) -> bytes:
    """``d2 -> uint8`` falloff, sampled over ``[inner^2, outer^2]``.

    The mapped value is ``255 * clamp(0.5 + (1 - d) / (2 * feather), 0, 1)``,
    a linear ramp that is 255 at ``d = 1 - feather``, exactly **128 at d = 1**
    (the nominal boundary, matching a raw logit of 0) and 0 at
    ``d = 1 + feather``.
    """
    key = (int(round(feather * 10000)), _RAMP_STEPS)
    table = _RAMP_CACHE.get(key)
    if table is not None:
        return table
    inner = 1.0 - feather
    outer = 1.0 + feather
    inner2 = inner * inner
    outer2 = outer * outer
    step = (outer2 - inner2) / _RAMP_STEPS
    buf = bytearray(_RAMP_STEPS)
    for i in range(_RAMP_STEPS):
        d = math.sqrt(inner2 + (i + 0.5) * step)
        v = 0.5 + (1.0 - d) / (2.0 * feather)
        if v <= 0.0:
            buf[i] = 0
        elif v >= 1.0:
            buf[i] = 255
        else:
            buf[i] = int(round(255.0 * v))
    table = bytes(buf)
    _RAMP_CACHE[key] = table
    return table


def render_ellipse(canvas: Size, cx: float, cy: float, rx: float, ry: float,
                   feather: float = _FEATHER
                   ) -> Optional[Tuple[BBox, bytearray]]:
    """Render one soft ellipse, cropped to its own bounding box.

    Returns ``(bbox, mask)`` in **model-canvas** coordinates with
    ``len(mask) == bbox.width * bbox.height``, or ``None`` when the ellipse
    falls entirely outside the canvas.
    """
    rx = max(1.0, float(rx))
    ry = max(1.0, float(ry))
    inner = 1.0 - feather
    outer = 1.0 + feather
    inner2 = inner * inner
    outer2 = outer * outer
    table = _ramp_table(feather)
    scale = _RAMP_STEPS / (outer2 - inner2)

    x0 = max(0, int(math.floor(cx - rx * outer)))
    x1 = min(int(canvas.width), int(math.ceil(cx + rx * outer)) + 1)
    y0 = max(0, int(math.floor(cy - ry * outer)))
    y1 = min(int(canvas.height), int(math.ceil(cy + ry * outer)) + 1)
    if x1 <= x0 or y1 <= y0:
        return None

    cw = x1 - x0
    mask = bytearray(cw * (y1 - y0))
    for y in range(y0, y1):
        fy = (y + 0.5 - cy) / ry
        ty = fy * fy
        if ty >= outer2:
            continue
        base = (y - y0) * cw
        half_out = rx * math.sqrt(outer2 - ty)
        px_lo = max(x0, int(math.floor(cx - half_out - 0.5)))
        px_hi = min(x1, int(math.ceil(cx + half_out + 0.5)))
        if px_hi <= px_lo:
            continue
        if ty < inner2:
            half_in = rx * math.sqrt(inner2 - ty)
            in_lo = max(px_lo, int(math.ceil(cx - half_in - 0.5)))
            in_hi = min(px_hi, int(math.floor(cx + half_in - 0.5)) + 1)
            if in_hi < in_lo:
                in_lo = in_hi = px_lo
        else:
            in_lo = in_hi = px_lo
        if in_hi > in_lo:
            mask[base + in_lo - x0:base + in_hi - x0] = b"\xff" * (in_hi - in_lo)
        for x in range(px_lo, in_lo):
            fx = (x + 0.5 - cx) / rx
            idx = int((ty + fx * fx - inner2) * scale)
            if idx < 0:
                mask[base + x - x0] = 255
            elif idx < _RAMP_STEPS:
                mask[base + x - x0] = table[idx]
        for x in range(in_hi, px_hi):
            fx = (x + 0.5 - cx) / rx
            idx = int((ty + fx * fx - inner2) * scale)
            if idx < 0:
                mask[base + x - x0] = 255
            elif idx < _RAMP_STEPS:
                mask[base + x - x0] = table[idx]
    return BBox(x0, y0, x1, y1), mask


_MIN_TABLE_CACHE: Dict[int, bytes] = {}


def _min_table(k: int) -> bytes:
    """256-byte translation table implementing ``v -> min(v, k)``."""
    t = _MIN_TABLE_CACHE.get(k)
    if t is None:
        t = bytes(min(v, k) for v in range(256))
        _MIN_TABLE_CACHE[k] = t
    return t


def _edge_byte(distance: float, feather_px: float) -> int:
    """Signed distance to an edge (positive inside) -> soft uint8, 128 at 0."""
    v = 0.5 + distance / (2.0 * feather_px)
    if v <= 0.0:
        return 0
    if v >= 1.0:
        return 255
    return int(round(255.0 * v))


def render_rect(canvas: Size, cx: float, cy: float, rx: float, ry: float,
                feather: float = _FEATHER
                ) -> Optional[Tuple[BBox, bytearray]]:
    """Render one soft-edged rectangle, cropped to its own bounding box.

    The falloff is separable -- ``v = min(profile_x, profile_y)`` -- and the
    ``min`` of two monotonically mapped signed distances equals the mapping of
    their ``min``, so each row is either a straight copy of the column profile
    or that profile passed through one ``bytes.translate`` table.
    """
    rx = max(1.0, float(rx))
    ry = max(1.0, float(ry))
    fpx = max(1.0, feather * min(rx, ry))

    x0 = max(0, int(math.floor(cx - rx - fpx)))
    x1 = min(int(canvas.width), int(math.ceil(cx + rx + fpx)) + 1)
    y0 = max(0, int(math.floor(cy - ry - fpx)))
    y1 = min(int(canvas.height), int(math.ceil(cy + ry + fpx)) + 1)
    if x1 <= x0 or y1 <= y0:
        return None

    cw = x1 - x0
    left, right = cx - rx, cx + rx
    top, bottom = cy - ry, cy + ry
    col = bytes(_edge_byte(min((x + 0.5) - left, right - (x + 0.5)), fpx)
                for x in range(x0, x1))
    mask = bytearray(cw * (y1 - y0))
    for y in range(y0, y1):
        rowv = _edge_byte(min((y + 0.5) - top, bottom - (y + 0.5)), fpx)
        if rowv == 0:
            continue
        base = (y - y0) * cw
        if rowv >= 255:
            mask[base:base + cw] = col
        else:
            mask[base:base + cw] = col.translate(_min_table(rowv))
    return BBox(x0, y0, x1, y1), mask


def _subtract_disc(bbox: BBox, mask: bytearray, cx: float, cy: float,
                   radius: float, feather: float = _FEATHER) -> None:
    """Carve a soft disc out of ``mask`` in place.

    This is what makes a negative PVS point visibly do something in stub mode:
    ``v <- min(v, 255 - disc(v))``.
    """
    hole = render_ellipse(Size(bbox.x1, bbox.y1), cx, cy, radius, radius, feather)
    if hole is None:
        return
    hbox, hmask = hole
    hw = hbox.width
    cw = bbox.width
    for y in range(max(hbox.y0, bbox.y0), min(hbox.y1, bbox.y1)):
        hbase = (y - hbox.y0) * hw
        mbase = (y - bbox.y0) * cw
        for x in range(max(hbox.x0, bbox.x0), min(hbox.x1, bbox.x1)):
            hv = hmask[hbase + x - hbox.x0]
            if hv:
                i = mbase + x - bbox.x0
                keep = 255 - hv
                if mask[i] > keep:
                    mask[i] = keep


# --------------------------------------------------------------------------- #
# the engine
# --------------------------------------------------------------------------- #
class StubEngine(BaseEngine):
    """A fully conforming fake engine.  Imports nothing heavier than ``math``.

    ``latency_scale`` multiplies the simulated stage timings.  The default of
    ``1.0`` gives roughly the few-hundred-millisecond total that ``API.md`` §14
    promises, which is what makes progress bars and supersession exercisable;
    tests pass ``0.0`` to remove the sleeps entirely.
    """

    MODE = "stub"

    #: Engine-level capabilities.  Exemplar boxes are accepted and reported in
    #: ``prompt.ignored``, so ``"exemplar_boxes"`` is deliberately absent.
    CAPABILITIES = ("pcs", "pvs")

    def __init__(self, canvas_side: int = DEFAULT_CANVAS_SIDE,
                 latency_scale: float = 1.0,
                 logger: Any = None) -> None:
        self._canvas_side = int(canvas_side)
        self.latency_scale = float(latency_scale)
        self._log = logger
        self._loaded: List[str] = []

    # -- introspection ----------------------------------------------------- #
    @property
    def canvas_side(self) -> int:
        return self._canvas_side

    def describe(self) -> EngineInfo:
        return EngineInfo(
            mode=self.MODE,
            device="stub",
            dtype="float32",
            capabilities=list(self.CAPABILITIES),
            torch_available=False,
            weights_available=False,
            model_canvas=Size(self._canvas_side, self._canvas_side),
            models_loaded=list(self._loaded),
            detail={
                "synthetic": True,
                "latency_scale": self.latency_scale,
                "note": "deterministic synthetic masks; no model is loaded",
            },
        )

    # -- lifecycle --------------------------------------------------------- #
    def unload(self, which: Optional[str] = None) -> List[str]:
        if which is None:
            gone, self._loaded = list(self._loaded), []
        else:
            gone = [which] if which in self._loaded else []
            self._loaded = [k for k in self._loaded if k != which]
        return gone

    # -- work -------------------------------------------------------------- #
    def _sleep(self, seconds: float) -> None:
        if self.latency_scale > 0.0 and seconds > 0.0:
            time.sleep(seconds * self.latency_scale)

    def encode_image(self, image: ImageData,
                     progress: Optional[ProgressFn] = None) -> EncodedImage:
        image.validate()
        rep = ProgressReporter(progress)
        canvas, transform = self.canvas_for(image.size)
        enc = EncodedImage(
            image_id=image.image_id,
            image=image.size,
            canvas=canvas,
            transform=transform,
            pixels=image.pixels,
        )
        # Simulated encoder pass.  Five steps so a long-polling client sees
        # several material progress changes, as it would with a real backbone.
        for i in range(5):
            rep.stage(Stage.ENCODING, i / 5.0)
            self._sleep(0.05)
        rep.stage(Stage.ENCODING, 1.0)
        if "pcs" not in self._loaded:
            self._loaded.append("pcs")
        # Only the PCS half is populated here; the PVS half is filled lazily on
        # the first point prompt, mirroring the real engine's lazy tracker load.
        enc.put("pcs", {"kind": "stub", "image_id": image.image_id}, 0)
        return enc

    def _ensure_pvs(self, encoded: EncodedImage, rep: ProgressReporter) -> None:
        if encoded.has("pvs"):
            return
        for i in range(3):
            rep.stage(Stage.ENCODING, i / 3.0)
            self._sleep(0.03)
        rep.stage(Stage.ENCODING, 1.0)
        if "pvs" not in self._loaded:
            self._loaded.append("pvs")
        encoded.put("pvs", {"kind": "stub", "image_id": encoded.image_id}, 0)

    # -- PCS --------------------------------------------------------------- #
    def prompt_text(self, encoded: EncodedImage, prompt: TextPrompt,
                    progress: Optional[ProgressFn] = None) -> PromptResult:
        rep = ProgressReporter(progress)
        text = (prompt.text or "").strip()
        if not text:
            raise ApiError(ErrorCode.BAD_REQUEST, "text prompt is empty", {})

        rep.stage(Stage.PROMPTING, 0.0)
        self._sleep(0.06)
        rep.stage(Stage.PROMPTING, 1.0)

        canvas = encoded.canvas
        seed = stub_seed("pcs", encoded.image_id, text)
        rng = _Rng(seed)
        count = 1 + (seed >> 7) % 8          # API.md §14: 1-8 instances

        rep.stage(Stage.DECODING, 0.0)
        raws: List[RawInstance] = []
        score = 0.55 + (seed >> 17 & 0xFFFF) / 65535.0 * 0.44
        for i in range(count):
            shape = _SHAPE_RECT if rng.next_u64() & 1 else _SHAPE_ELLIPSE
            # Each radius is bounded by its own axis and then clamped to a
            # quarter of it, so a blob always fits.  Deriving ry from
            # canvas.width made a wide, short canvas (64x32, once the canvas
            # became the image rather than a 1008 square) produce blobs taller
            # than the canvas, an inverted range for cy, and finally a bbox the
            # frame packer rejected.
            rx = min(rng.uniform(0.055, 0.165) * canvas.width, canvas.width / 4.0)
            ry = min(rx * rng.uniform(0.6, 1.7), canvas.height / 4.0)
            rx = max(1.0, rx)
            ry = max(1.0, ry)
            cx = rng.uniform(rx, max(rx, canvas.width - rx))
            cy = rng.uniform(ry, max(ry, canvas.height - ry))
            drawn = (render_rect if shape == _SHAPE_RECT else render_ellipse)(
                canvas, cx, cy, rx, ry)
            rep.stage(Stage.DECODING, (i + 1) / float(count))
            if drawn is None:
                continue
            bbox, mask = drawn
            raws.append(RawInstance(score=round(max(0.1, min(1.0, score)), 4),
                                    bbox=bbox, mask=bytes(mask), label=text))
            score *= rng.uniform(0.70, 0.92)
        self._sleep(0.04)

        rep.stage(Stage.PACKING, 0.0)
        instances, truncated = _finalize(raws, canvas,
                                         score_threshold=prompt.score_threshold,
                                         max_instances=prompt.max_instances)
        rep.stage(Stage.PACKING, 1.0)

        ignored = ["boxes"] if prompt.boxes else None
        return PromptResult(
            engine="pcs",
            instances=instances,
            image=encoded.image,
            canvas=canvas,
            transform=encoded.transform,
            prompt=self.text_prompt_dict(prompt, ignored),
            truncated=truncated,
            elapsed_ms=_NOMINAL_TEXT_MS + _NOMINAL_PER_INSTANCE_MS * len(instances),
        )

    # -- PVS --------------------------------------------------------------- #
    def prompt_points(self, encoded: EncodedImage, prompt: PointPrompt,
                      progress: Optional[ProgressFn] = None) -> PromptResult:
        rep = ProgressReporter(progress)
        if not prompt.points and not prompt.box:
            raise ApiError(ErrorCode.BAD_REQUEST,
                           "a point prompt needs at least one point or a box", {})
        self._ensure_pvs(encoded, rep)

        rep.stage(Stage.PROMPTING, 0.0)
        self._sleep(0.04)
        rep.stage(Stage.PROMPTING, 1.0)

        canvas = encoded.canvas
        tf = encoded.transform
        pos = [tf.image_to_canvas(p.x, p.y) for p in prompt.points if int(p.label) == 1]
        neg = [tf.image_to_canvas(p.x, p.y) for p in prompt.points if int(p.label) != 1]
        box_canvas: Optional[Tuple[float, float, float, float]] = None
        if prompt.box:
            bx0, by0 = tf.image_to_canvas(prompt.box[0], prompt.box[1])
            bx1, by1 = tf.image_to_canvas(prompt.box[2], prompt.box[3])
            box_canvas = (min(bx0, bx1), min(by0, by1), max(bx0, bx1), max(by0, by1))

        if box_canvas is not None:
            cx = (box_canvas[0] + box_canvas[2]) * 0.5
            cy = (box_canvas[1] + box_canvas[3]) * 0.5
            rx = max(4.0, (box_canvas[2] - box_canvas[0]) * 0.5)
            ry = max(4.0, (box_canvas[3] - box_canvas[1]) * 0.5)
        else:
            anchors = pos or neg
            cx = sum(p[0] for p in anchors) / len(anchors)
            cy = sum(p[1] for p in anchors) / len(anchors)
            spread_x = max((abs(p[0] - cx) for p in anchors), default=0.0)
            spread_y = max((abs(p[1] - cy) for p in anchors), default=0.0)
            base = canvas.width * 0.085
            rx = max(base, spread_x * 1.45)
            ry = max(base, spread_y * 1.45)

        seed = stub_seed("pvs", encoded.image_id,
                         [(round(p[0], 3), round(p[1], 3)) for p in pos],
                         [(round(p[0], 3), round(p[1], 3)) for p in neg],
                         None if box_canvas is None else [round(v, 3) for v in box_canvas])
        rng = _Rng(seed)

        n = 1
        if prompt.multimask:
            n = max(1, min(int(prompt.max_instances), 3))
        # Candidate scales, best first: the model's own guess, a tighter part
        # and a looser whole -- the classic SAM multimask triple.
        scales = (1.0, 0.62, 1.42)[:n]
        base_score = 0.82 + rng.random() * 0.16

        rep.stage(Stage.DECODING, 0.0)
        raws: List[RawInstance] = []
        for i, scale in enumerate(scales):
            drawn = render_ellipse(canvas, cx, cy, rx * scale, ry * scale)
            rep.stage(Stage.DECODING, (i + 1) / float(len(scales)))
            if drawn is None:
                continue
            bbox, mask = drawn
            for nx, ny in neg:
                if bbox.x0 <= nx < bbox.x1 and bbox.y0 <= ny < bbox.y1:
                    _subtract_disc(bbox, mask, nx, ny,
                                   max(6.0, 0.42 * min(rx, ry) * scale))
            score = base_score * (1.0 - 0.22 * i)
            raws.append(RawInstance(score=round(max(0.1, min(1.0, score)), 4),
                                    bbox=bbox, mask=bytes(mask), label=""))
        self._sleep(0.03)

        rep.stage(Stage.PACKING, 0.0)
        instances, truncated = _finalize(raws, canvas, score_threshold=0.0,
                                         max_instances=prompt.max_instances)
        rep.stage(Stage.PACKING, 1.0)

        return PromptResult(
            engine="pvs",
            instances=instances,
            image=encoded.image,
            canvas=canvas,
            transform=encoded.transform,
            prompt=self.point_prompt_dict(prompt),
            truncated=truncated,
            elapsed_ms=_NOMINAL_POINTS_MS + _NOMINAL_PER_INSTANCE_MS * len(instances),
        )

