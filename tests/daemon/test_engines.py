"""Tests for the model manager and the three engines.

Everything here passes on a machine with **no GPU, no torch, no transformers
and no SAM 3 weights** -- and, just as much, on one that has them: a test about
the no-torch path pins that state itself instead of assuming it.  The stub
engine is production code (``API.md`` §14) and is tested as such: geometry,
determinism, soft edges, score ordering, truncation and wire invariants.  The
torch engines' control flow -- embedding reuse, the float32 fallback, soft-mask
selection -- is driven through a small numpy-backed stand-in for torch
(:func:`_make_fake_torch`) and fake models with SAM 3's calling conventions.
Anything that would need real weights is marked ``needs_torch`` and skips.

The two properties worth stating up front, because they are what the rest of
the stack is built on:

* a stub frame is **byte-identical** for the same ``(image_id, prompt)``, so a
  client-side regression is visible as a diff rather than as noise;
* stub masks are genuinely **soft**, with 128 at the nominal boundary, so a
  client whose threshold slider is broken fails a test here rather than looking
  plausible in the canvas.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.abc
import importlib.util
import logging
import math
import pathlib
import struct
import sys
import threading
import time
import types

import pytest

from sam3gimpd import modelmgr
from sam3gimpd.engines import (
    StubEngine,
    TorchEngine,
    build_instances,
    create_engine,
    default_canvas_for,
    finalize_instances,
)
from sam3gimpd.engines import base as engine_base
from sam3gimpd.engines.base import (
    BBox,
    CanvasMask,
    EncodedImage,
    ImageData,
    ProgressReporter,
    RawInstance,
    Stage,
)
from sam3gimpd.engines.pcs import VISION_CACHE_KWARG, PcsEngine
from sam3gimpd.engines.pvs import PvsEngine, nest_box, nest_points
from sam3gimpd.engines.stub import (
    _Rng,
    render_ellipse,
    render_rect,
    stub_seed,
)
from sam3gimpd.types import (
    DEFAULT_MASK_THRESHOLD,
    ApiError,
    ErrorCode,
    Limits,
    Point,
    PointPrompt,
    Size,
    TextPrompt,
    unpack_result,
)

IMAGE_W, IMAGE_H = 64, 48


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def stub() -> StubEngine:
    """A stub engine with the simulated latency removed."""
    return StubEngine(latency_scale=0.0)


@pytest.fixture
def image_data(make_rgb) -> ImageData:
    return ImageData(image_id="a" * 32, width=IMAGE_W, height=IMAGE_H,
                     pixels=make_rgb(IMAGE_W, IMAGE_H, seed=1))


@pytest.fixture
def encoded(stub: StubEngine, image_data: ImageData) -> EncodedImage:
    return stub.encode_image(image_data)


def _frame_instances(result, job="j-1", req="r-1", image_id="a" * 32):
    """Round-trip a PromptResult through the real wire format."""
    frame = result.to_frame(job, req, image_id)
    header, blob = unpack_result(frame)
    return header, blob, frame


@pytest.fixture
def no_torch(monkeypatch):
    """Pin the probe to "torch is not installed", whatever this interpreter has."""
    probe = modelmgr.TorchProbe(available=False,
                                error="ModuleNotFoundError: No module named 'torch'")
    monkeypatch.setattr(modelmgr, "_PROBE", probe)
    return probe


@pytest.fixture
def no_weights(sam3_home, monkeypatch):
    """No checkpoint anywhere the manager looks."""
    for var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
                modelmgr.ENV_WEIGHTS_DIR):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(sam3_home))
    return sam3_home


def _encoded(image_data):
    canvas, tf = default_canvas_for(image_data.size)
    return EncodedImage(image_data.image_id, image_data.size, canvas, tf, image_data.pixels)


# --------------------------------------------------------------------------- #
# a stand-in for torch, just big enough for the torch engines' code paths
# --------------------------------------------------------------------------- #
def _bf16(np, values):
    """float32 values rounded to bfloat16 precision, nearest-even."""
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    bits = arr.view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32).reshape(arr.shape)


class _FakeDtype:
    def __init__(self, name, storage, floating, itemsize):
        self.name = name
        self.storage = storage
        self.is_floating_point = floating
        self.itemsize = itemsize

    def __repr__(self):
        return "torch." + self.name


def _make_fake_torch(np):
    """A module that answers to ``import torch`` for the engines' purposes.

    Tensors wrap numpy arrays and keep a dtype of their own; bfloat16 is stored
    as float32 rounded to bfloat16 after every operation, and comparing a
    tensor with a Python number narrows the number into the tensor's dtype
    first, as torch does.  Returns ``(torch, {module name: module})``.
    """
    torch = types.ModuleType("torch")
    torch.__version__ = "0.0-fake"
    for name, storage, floating, size in (("float32", "float32", True, 4),
                                          ("float16", "float16", True, 2),
                                          ("bfloat16", "float32", True, 2),
                                          ("uint8", "uint8", False, 1),
                                          ("int64", "int64", False, 8),
                                          ("bool", "bool", False, 1)):
        setattr(torch, name, _FakeDtype(name, storage, floating, size))
    torch.long = torch.int64

    def _cast(values, dtype):
        out = np.asarray(values).astype(dtype.storage)
        return _bf16(np, out) if dtype is torch.bfloat16 else out

    def _dtype_of(arr):
        if arr.dtype == np.float16:
            return torch.float16
        if arr.dtype.kind == "f":
            return torch.float32
        if arr.dtype == np.uint8:
            return torch.uint8
        if arr.dtype == np.bool_:
            return torch.bool
        return torch.int64

    class Tensor:
        def __init__(self, data, dtype=None, device="cpu"):
            arr = np.asarray(data)
            self.dtype = dtype if dtype is not None else _dtype_of(arr)
            self.data = _cast(arr, self.dtype)
            self.device = device

        @property
        def shape(self):
            return tuple(self.data.shape)

        def dim(self):
            return self.data.ndim

        def numel(self):
            return int(self.data.size)

        def element_size(self):
            return self.dtype.itemsize

        def is_floating_point(self):
            return self.dtype.is_floating_point

        def detach(self):
            return self

        def cpu(self):
            return Tensor(self.data, self.dtype, "cpu")

        def numpy(self):
            return self.data

        def tolist(self):
            return self.data.tolist()

        def float(self):
            return self.to(torch.float32)

        def to(self, arg):
            if isinstance(arg, _FakeDtype):
                return Tensor(self.data, arg, self.device)
            return Tensor(self.data, self.dtype, str(arg))

        def sigmoid(self):
            with np.errstate(over="ignore", invalid="ignore"):
                out = 1.0 / (1.0 + np.exp(-self.data.astype(np.float64)))
            return Tensor(out, self.dtype, self.device)

        def reshape(self, *shape):
            if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                shape = tuple(shape[0])
            return Tensor(self.data.reshape(tuple(int(v) for v in shape)),
                          self.dtype, self.device)

        def __getitem__(self, index):
            if isinstance(index, Tensor):
                index = index.data
            return Tensor(self.data[index], self.dtype, self.device)

        def __mul__(self, other):
            value = other.data if isinstance(other, Tensor) else other
            return Tensor(self.data * value, self.dtype, self.device)

        __rmul__ = __mul__

        def __gt__(self, other):
            if isinstance(other, Tensor):
                value = other.data
            else:
                value = _cast(np.float32(other), self.dtype)
            return Tensor(self.data > value, torch.bool, self.device)

        def clamp_(self, lo, hi):
            self.data = np.clip(self.data, lo, hi).astype(self.data.dtype)
            return self

        def mul_(self, value):
            self.data = (self.data * value).astype(self.data.dtype)
            return self

        def round_(self):
            self.data = np.round(self.data)
            return self

        def min(self):
            return Tensor(self.data.min(), self.dtype)

        def max(self):
            return Tensor(self.data.max(), self.dtype)

        def all(self):
            return Tensor(np.all(self.data), torch.bool)

        def __bool__(self):
            return bool(self.data)

        def __float__(self):
            return float(self.data)

        def __len__(self):
            return len(self.data)

    def as_tensor(values, dtype=None, device=None):
        if isinstance(values, Tensor):
            t = values if dtype is None else values.to(dtype)
        else:
            t = Tensor(np.asarray(values), dtype)
        return t if device is None else Tensor(t.data, t.dtype, str(device))

    def interpolate(t, size=None, mode="bilinear", align_corners=False):
        _n, _c, h, w = t.shape
        out_h, out_w = int(size[0]), int(size[1])

        def _axis(out_n, in_n):
            src = np.clip((np.arange(out_n) + 0.5) * (in_n / float(out_n)) - 0.5, 0, None)
            lo = np.minimum(np.floor(src).astype(int), in_n - 1)
            hi = np.minimum(lo + 1, in_n - 1)
            return lo, hi, src - lo

        y0, y1, fy = _axis(out_h, h)
        x0, x1, fx = _axis(out_w, w)
        d = t.data.astype(np.float64)
        rows = (d[:, :, y0, :] * (1 - fy)[None, None, :, None]
                + d[:, :, y1, :] * fy[None, None, :, None])
        out = rows[:, :, :, x0] * (1 - fx) + rows[:, :, :, x1] * fx
        return Tensor(out.astype(np.float32), t.dtype, t.device)

    torch.Tensor = Tensor
    torch.as_tensor = as_tensor
    torch.isfinite = lambda t: Tensor(np.isfinite(t.data), torch.bool, t.device)
    torch.sigmoid = lambda t: t.sigmoid()
    torch.inference_mode = contextlib.nullcontext
    nn = types.ModuleType("torch.nn")
    functional = types.ModuleType("torch.nn.functional")
    functional.interpolate = interpolate
    nn.functional = functional
    torch.nn = nn
    return torch, {"torch": torch, "torch.nn": nn, "torch.nn.functional": functional}


@pytest.fixture
def fake_torch(monkeypatch):
    """Install the stand-in as ``torch``, with a probe that says it is there."""
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(modelmgr, "_PROBE",
                        modelmgr.TorchProbe(available=True, version="0.0-fake",
                                            transformers_available=True))
    torch, modules = _make_fake_torch(np)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    torch.np = np
    return torch


class _FakeTracker:
    """``Sam3TrackerModel``'s calling convention, returning a square blob.

    ``state`` is shared with the test: ``nan`` makes a half-precision model
    produce NaN masks, ``nan_iou`` a NaN IoU score, and ``reject_embeddings``
    refuses a cached embedding.  Like torch, it refuses a floating input whose
    dtype differs from its own.
    """

    def __init__(self, torch, dtype, state):
        self.torch, self.np, self.dtype, self.state = torch, torch.np, dtype, state
        self.embed_calls = 0
        self.forward_calls = []

    def forward(self, pixel_values=None, image_embeddings=None, input_points=None,
                input_labels=None, input_boxes=None, original_sizes=None,
                multimask_output=True):
        """Signature only: the engine inspects it."""

    def _check(self, tensor):
        if tensor is not None and tensor.is_floating_point() and tensor.dtype is not self.dtype:
            raise RuntimeError("expected scalar type %s but found %s"
                               % (self.dtype.name, tensor.dtype.name))

    def get_image_embeddings(self, pixel_values=None):
        self._check(pixel_values)
        self.embed_calls += 1
        return self.torch.as_tensor(self.np.ones((1, 4, 4, 4), self.np.float32),
                                    dtype=self.dtype)

    def __call__(self, **kw):
        np = self.np
        self.forward_calls.append(dict(kw))
        embeddings, pixels = kw.get("image_embeddings"), kw.get("pixel_values")
        if embeddings is not None and pixels is not None:
            raise ValueError("Only one of pixel_values and image_embeddings can be provided")
        if embeddings is not None and self.state.get("reject_embeddings"):
            raise ValueError("image_embeddings are not accepted here")
        self._check(embeddings)
        self._check(pixels)
        half = self.dtype is not self.torch.float32
        logits = np.full((1, 1, 3, 8, 8), -8.0, np.float32)
        logits[..., 2:6, 2:6] = 8.0
        if half and self.state.get("nan"):
            logits[...] = np.nan
        iou = np.array([[[0.9, 0.6, 0.3]]], np.float32)
        if self.state.get("nan_iou") and (half or self.state.get("nan_iou") == "always"):
            iou[..., 0] = np.nan
        t = self.torch
        return types.SimpleNamespace(
            pred_masks=t.as_tensor(logits, dtype=self.dtype),
            iou_scores=t.as_tensor(iou, dtype=self.dtype),
            object_score_logits=t.as_tensor(np.array([[[4.0]]], np.float32), dtype=self.dtype))


class _FakeTrackerProcessor:
    def __init__(self, torch):
        self.torch, self.np = torch, torch.np
        self.calls = []

    def __call__(self, images=None, input_points=None, input_labels=None,
                 input_boxes=None, original_sizes=None, return_tensors=None):
        np, t = self.np, self.torch
        self.calls.append({"images": images is not None,
                           "original_sizes": original_sizes is not None,
                           "input_points": input_points})
        out = {}
        if images is not None:
            out["pixel_values"] = t.as_tensor(np.zeros((1, 3, 8, 8), np.float32))
            out["original_sizes"] = t.as_tensor(np.array([list(images.shape[:2])], np.int64))
        elif original_sizes is not None:
            out["original_sizes"] = t.as_tensor(original_sizes)
        if input_points is not None:
            out["input_points"] = t.as_tensor(np.array(input_points, np.float32))
            out["input_labels"] = t.as_tensor(np.array(input_labels, np.int64))
        return out


class _FakeSam3:
    """``Sam3Model``'s calling convention: four queries, each a soft blob.

    Query 3's logit sits on the score threshold of 0.02 as bfloat16 rounds it,
    which is where a mirror comparing in float64 disagrees with the processor.
    ``state``: ``nan`` (half-precision masks are NaN), ``reject_vision_embeds``.
    """

    #: sigmoid(-3.8918) == 0.02; in bfloat16 it rounds to exactly bf16(0.02).
    LOGITS = (4.0, -6.0, 2.0, -3.8918203)

    def __init__(self, torch, dtype, state):
        self.torch, self.np, self.dtype, self.state = torch, torch.np, dtype, state
        self.vision_calls = 0
        self.forward_calls = []

    def forward(self, pixel_values=None, vision_embeds=None, input_ids=None,
                attention_mask=None, input_boxes=None, input_boxes_labels=None):
        """Signature only."""

    def _check(self, tensor):
        if tensor is not None and tensor.is_floating_point() and tensor.dtype is not self.dtype:
            raise RuntimeError("expected scalar type %s but found %s"
                               % (self.dtype.name, tensor.dtype.name))

    def get_vision_features(self, pixel_values=None):
        self._check(pixel_values)
        self.vision_calls += 1
        return self.torch.as_tensor(self.np.ones((1, 4, 4, 4), self.np.float32),
                                    dtype=self.dtype)

    def __call__(self, **kw):
        np, t = self.np, self.torch
        self.forward_calls.append(dict(kw))
        pixels, vision = kw.get("pixel_values"), kw.get("vision_embeds")
        if (pixels is None) == (vision is None):
            raise ValueError("You must specify exactly one of pixel_values or vision_embeds")
        if vision is not None and self.state.get("reject_vision_embeds"):
            raise TypeError("forward() got an unexpected keyword argument 'vision_embeds'")
        self._check(pixels)
        self._check(vision)
        masks = np.full((1, 4, 8, 8), -6.0, np.float32)
        for q in range(4):
            # A ramp, so the soft values are genuinely soft.
            masks[0, q, 1 + q:5 + q // 2, 1:7] = np.linspace(-2.0, 6.0, 6, dtype=np.float32)
        if self.dtype is not t.float32 and self.state.get("nan"):
            masks[...] = np.nan
        return types.SimpleNamespace(
            pred_logits=t.as_tensor(np.array([self.LOGITS], np.float32), dtype=self.dtype),
            pred_masks=t.as_tensor(masks, dtype=self.dtype),
            presence_logits=None)


class _FakeSam3Processor:
    """``Sam3Processor``, including ``post_process_instance_segmentation`` done
    the way transformers does it: scores and ``keep`` in the model's dtype."""

    def __init__(self, torch):
        self.torch, self.np = torch, torch.np
        self.image_processor = types.SimpleNamespace(size={"height": 1008, "width": 1008})
        self.calls = []

    def __call__(self, images=None, text=None, input_boxes=None, input_boxes_labels=None,
                 return_tensors=None):
        return self._inputs(images=images, text=text, input_boxes=input_boxes,
                            input_boxes_labels=input_boxes_labels)

    def _inputs(self, **kw):
        np, t = self.np, self.torch
        self.calls.append(kw)
        out = {}
        if kw.get("images") is not None:
            out["pixel_values"] = t.as_tensor(np.zeros((1, 3, 8, 8), np.float32))
        if kw.get("text") is not None:
            out["input_ids"] = t.as_tensor(np.array([[1, 2, 3]], np.int64))
            out["attention_mask"] = t.as_tensor(np.array([[1, 1, 1]], np.int64))
        return out

    def post_process_instance_segmentation(self, outputs, threshold=0.5, mask_threshold=0.5,
                                           target_sizes=None):
        t = self.torch
        scores = outputs.pred_logits.sigmoid()
        if outputs.presence_logits is not None:
            scores = scores * outputs.presence_logits.sigmoid()
        masks = outputs.pred_masks.sigmoid()
        keep = scores[0] > threshold
        kept_scores, kept = scores[0][keep], masks[0][keep]
        h, w = target_sizes[0]
        kept = sys.modules["torch.nn.functional"].interpolate(
            kept.float()[None], size=(h, w))[0]
        return [{"scores": kept_scores, "masks": (kept > mask_threshold).to(t.long),
                 "boxes": None}]


class _NoLabelsSam3Processor(_FakeSam3Processor):
    def __call__(self, images=None, text=None, input_boxes=None, return_tensors=None):
        return self._inputs(images=images, text=text, input_boxes=input_boxes)


class _NoBoxesSam3Processor(_FakeSam3Processor):
    def __call__(self, images=None, text=None, return_tensors=None):
        return self._inputs(images=images, text=text)


def _torch_engine(torch, dtype="float16", state=None, pcs_processor=_FakeSam3Processor,
                  logger=None):
    """A real ``TorchEngine`` whose manager loads the fakes above.

    Returns ``(engine, loads, models)``: every ``(half, dtype)`` loaded, and
    every model built, in order.
    """
    state = {} if state is None else state
    mgr = modelmgr.ModelManager(device="cuda", dtype=dtype, configure_hf=False)
    mgr.ensure_available = lambda: None
    loads, models = [], []

    def _load(half, dtype_name):
        loads.append((half, dtype_name))
        dt = getattr(torch, dtype_name)
        if half == "pvs":
            model, processor = _FakeTracker(torch, dt, state), _FakeTrackerProcessor(torch)
        else:
            model, processor = _FakeSam3(torch, dt, state), pcs_processor(torch)
        models.append(model)
        return modelmgr._Loaded(half, model, processor, dtype_name, "cuda")

    mgr._load_locked = _load
    return TorchEngine(manager=mgr, logger=logger), loads, models


# --------------------------------------------------------------------------- #
# import hygiene -- the rule the whole dev workflow rests on
# --------------------------------------------------------------------------- #
class TestImportHygiene:
    def test_engine_imports_pull_in_no_heavy_dependency(self):
        for name in ("sam3gimpd.modelmgr", "sam3gimpd.engines", "sam3gimpd.engines.base",
                     "sam3gimpd.engines.stub", "sam3gimpd.engines.pcs", "sam3gimpd.engines.pvs"):
            importlib.import_module(name)
        loaded = {m.split(".", 1)[0] for m in sys.modules}
        assert "torch" not in loaded
        assert "transformers" not in loaded

    def test_stub_engine_never_imports_numpy(self, stub, image_data, monkeypatch):
        """Poison the import so a stray ``import numpy`` fails loudly.

        ``sys.modules[name] = None`` makes ``import name`` raise; monkeypatch
        restores the real module afterwards.  Popping numpy instead would let a
        later test trigger a module reload, which numpy warns about.
        """
        monkeypatch.setitem(sys.modules, "numpy", None)
        with pytest.raises(ImportError):
            importlib.import_module("numpy")
        enc = stub.encode_image(image_data)
        stub.prompt_text(enc, TextPrompt(request_id="r", text="a red car"))
        stub.prompt_points(enc, PointPrompt(request_id="r2",
                                            points=[Point(20.0, 20.0, 1)]))

    def test_stub_module_has_no_numpy_or_torch_reference(self):
        source = pathlib.Path(engine_base.__file__).with_name("stub.py").read_text(encoding="utf-8")
        for banned in ("import numpy", "import torch", "import transformers"):
            assert banned not in source, banned

    def test_creating_a_torch_engine_does_not_import_torch(self, monkeypatch):
        """Construction is cheap and torch-free: the daemon binds its socket
        and publishes runtime.json before anything imports torch.

        With torch absent, ``"torch" not in sys.modules`` would hold whatever
        the code did, so ``torch`` is replaced by a finder that records every
        import of it -- and with torch installed, the real one is set aside for
        the duration.  ``describe()`` must not import it either: that is what
        answers ``/hello``.
        """
        imported = []

        class _RecordingTorch(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            def find_spec(self, name, path=None, target=None):
                if name == "torch" or name.startswith("torch."):
                    return importlib.util.spec_from_loader(name, self)
                return None

            def create_module(self, spec):
                return None

            def exec_module(self, module):
                imported.append(module.__name__)

        # setitem-then-del records "torch" either way, so the recorder's module
        # is removed afterwards even where there was no torch to restore.
        monkeypatch.setitem(sys.modules, "torch", None)
        del sys.modules["torch"]
        for name in [m for m in sys.modules if m.startswith("torch.")]:
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.setattr(sys, "meta_path", [_RecordingTorch()] + sys.meta_path)
        monkeypatch.setattr(modelmgr, "_PROBE", None)

        engine = TorchEngine()
        assert imported == [], "constructing the engine imported torch"
        info = engine.describe()
        assert imported == [], "describe() imported torch"
        assert info.detail["probed"] is False
        assert importlib.import_module("torch") is not None
        assert imported == ["torch"], "the recording finder is not in the way"


# --------------------------------------------------------------------------- #
# geometry and the coordinate contract
# --------------------------------------------------------------------------- #
class TestGeometry:
    def test_default_canvas_is_the_uploaded_image(self):
        """Masks come back in image space, so the transform is the identity.

        Not a 1008x1008 canvas at scale (1.0, 1.5): that anisotropic squash is
        something the processor never performs.  SAM 3 resizes preserving
        aspect ratio and pads, so under it every mask on a non-square image
        would be stretched and land beside the object.
        """
        canvas, tf = default_canvas_for(Size(1008, 672))
        assert canvas.to_dict() == {"width": 1008, "height": 672}
        assert tf.scale_x == pytest.approx(1.0)
        assert tf.scale_y == pytest.approx(1.0)
        assert (tf.offset_x, tf.offset_y) == (0.0, 0.0)

    def test_transform_round_trips(self):
        _canvas, tf = default_canvas_for(Size(IMAGE_W, IMAGE_H))
        cx, cy = tf.image_to_canvas(10.0, 20.0)
        back = tf.canvas_to_image(cx, cy)
        assert back == pytest.approx((10.0, 20.0))

    def test_encode_reports_canvas_before_any_prompt(self, stub, image_data):
        enc = stub.encode_image(image_data)
        assert enc.canvas.to_dict() == {"width": IMAGE_W, "height": IMAGE_H}
        assert enc.image.to_dict() == {"width": IMAGE_W, "height": IMAGE_H}
        assert enc.transform.scale_x == pytest.approx(1.0)
        assert enc.transform.scale_y == pytest.approx(1.0)

    def test_canvas_for_is_cheap_and_matches_encode(self, stub, image_data):
        canvas, tf = stub.canvas_for(Size(IMAGE_W, IMAGE_H))
        enc = stub.encode_image(image_data)
        assert canvas == enc.canvas
        assert tf.to_dict() == enc.transform.to_dict()

    def test_oversized_upload_is_rejected(self, stub):
        bad = ImageData("x" * 32, 2000, 10, bytes(2000 * 10 * 3))
        with pytest.raises(ApiError) as exc:
            stub.encode_image(bad)
        assert exc.value.code == ErrorCode.BAD_DIMENSIONS

    def test_wrong_payload_length_is_rejected(self, stub):
        bad = ImageData("x" * 32, 64, 48, bytes(10))
        with pytest.raises(ApiError) as exc:
            stub.encode_image(bad)
        assert exc.value.code == ErrorCode.PAYLOAD_SIZE_MISMATCH


# --------------------------------------------------------------------------- #
# the stub: PCS
# --------------------------------------------------------------------------- #
class TestStubText:
    def test_returns_between_one_and_eight_instances(self, stub, encoded):
        for text in ("cat", "yellow school bus", "a", "tree", "person", "wheel"):
            result = stub.prompt_text(encoded, TextPrompt(request_id="r", text=text))
            assert 1 <= len(result.instances) <= 8, text

    def test_scores_descend_and_are_in_range(self, stub, encoded):
        result = stub.prompt_text(encoded,
                                  TextPrompt(request_id="r", text="yellow school bus"))
        scores = [i.score for i in result.instances]
        assert scores == sorted(scores, reverse=True)
        assert all(0.1 <= s <= 1.0 for s in scores)

    def test_label_is_the_prompt_text(self, stub, encoded):
        result = stub.prompt_text(encoded, TextPrompt(request_id="r", text="red car"))
        assert {i.label for i in result.instances} == {"red car"}

    def test_deterministic_frames(self, stub, encoded, image_data):
        prompt = TextPrompt(request_id="r", text="yellow school bus")
        a = stub.prompt_text(encoded, prompt).to_frame("j", "r", image_data.image_id)
        # A completely fresh engine and a fresh encode must agree byte for byte,
        # including the JSON header -- which is why the stub reports a nominal
        # elapsed_ms rather than a wall-clock one (API.md §14).
        other = StubEngine(latency_scale=0.0)
        enc2 = other.encode_image(image_data)
        b = other.prompt_text(enc2, prompt).to_frame("j", "r", image_data.image_id)
        assert a == b

    def test_different_prompts_give_different_geometry(self, stub, encoded):
        a = stub.prompt_text(encoded, TextPrompt(request_id="r", text="cat"))
        b = stub.prompt_text(encoded, TextPrompt(request_id="r", text="dog"))
        assert [i.bbox.to_list() for i in a.instances] != \
               [i.bbox.to_list() for i in b.instances]

    def test_different_images_give_different_geometry(self, stub, make_rgb):
        one = stub.encode_image(ImageData("1" * 32, IMAGE_W, IMAGE_H,
                                          make_rgb(IMAGE_W, IMAGE_H, 1)))
        two = stub.encode_image(ImageData("2" * 32, IMAGE_W, IMAGE_H,
                                          make_rgb(IMAGE_W, IMAGE_H, 2)))
        prompt = TextPrompt(request_id="r", text="cat")
        assert [i.bbox.to_list() for i in stub.prompt_text(one, prompt).instances] != \
               [i.bbox.to_list() for i in stub.prompt_text(two, prompt).instances]

    def test_max_instances_truncates_and_flags(self, stub, encoded):
        full = stub.prompt_text(encoded, TextPrompt(request_id="r", text="wheel"))
        if len(full.instances) < 2:
            pytest.skip("this prompt hash yields a single instance")
        clipped = stub.prompt_text(
            encoded, TextPrompt(request_id="r", text="wheel", max_instances=1))
        assert len(clipped.instances) == 1
        assert clipped.truncated is True
        assert full.truncated is False
        # truncation keeps the best, not the first drawn
        assert clipped.instances[0].score == max(i.score for i in full.instances)

    def test_score_threshold_filters(self, stub, encoded):
        low = stub.prompt_text(encoded, TextPrompt(request_id="r", text="tree",
                                                   score_threshold=0.1))
        high = stub.prompt_text(encoded, TextPrompt(request_id="r", text="tree",
                                                    score_threshold=0.9))
        assert len(high.instances) <= len(low.instances)
        assert all(i.score >= 0.9 for i in high.instances)

    def test_empty_text_is_a_bad_request(self, stub, encoded):
        with pytest.raises(ApiError) as exc:
            stub.prompt_text(encoded, TextPrompt(request_id="r", text="   "))
        assert exc.value.code == ErrorCode.BAD_REQUEST

    def test_exemplar_boxes_are_reported_as_ignored(self, stub, encoded):
        prompt = TextPrompt(request_id="r", text="car",
                            boxes=[{"box": [1.0, 2.0, 3.0, 4.0], "label": 1}])
        result = stub.prompt_text(encoded, prompt)
        assert result.prompt["ignored"] == ["boxes"]
        assert "exemplar_boxes" not in stub.describe().capabilities


# --------------------------------------------------------------------------- #
# the stub: PVS
# --------------------------------------------------------------------------- #
class TestStubPoints:
    def test_blob_is_centred_on_the_click(self, stub, encoded):
        click = Point(32.0, 24.0, 1)
        result = stub.prompt_points(encoded,
                                    PointPrompt(request_id="r", points=[click]))
        expected = encoded.transform.image_to_canvas(click.x, click.y)
        best = result.instances[0].bbox
        assert (best.x0 + best.x1) / 2 == pytest.approx(expected[0], abs=3)
        assert (best.y0 + best.y1) / 2 == pytest.approx(expected[1], abs=3)

    def test_clicking_elsewhere_moves_the_blob(self, stub, encoded):
        a = stub.prompt_points(encoded, PointPrompt(request_id="r",
                                                    points=[Point(10.0, 10.0, 1)]))
        b = stub.prompt_points(encoded, PointPrompt(request_id="r",
                                                    points=[Point(50.0, 40.0, 1)]))
        assert a.instances[0].bbox.x0 < b.instances[0].bbox.x0
        assert a.instances[0].bbox.y0 < b.instances[0].bbox.y0

    def test_box_prompt_defines_the_blob(self, stub, encoded):
        box = [10.0, 8.0, 40.0, 32.0]
        result = stub.prompt_points(encoded, PointPrompt(request_id="r", box=box,
                                                         multimask=False))
        assert len(result.instances) == 1
        cx = encoded.transform.image_to_canvas((box[0] + box[2]) / 2,
                                               (box[1] + box[3]) / 2)[0]
        bbox = result.instances[0].bbox
        assert (bbox.x0 + bbox.x1) / 2 == pytest.approx(cx, abs=3)

    def test_multimask_returns_candidates_best_first(self, stub, encoded):
        result = stub.prompt_points(
            encoded, PointPrompt(request_id="r", points=[Point(32.0, 24.0, 1)],
                                 multimask=True, max_instances=3))
        assert len(result.instances) == 3
        scores = [i.score for i in result.instances]
        assert scores == sorted(scores, reverse=True)

    def test_multimask_off_returns_one(self, stub, encoded):
        result = stub.prompt_points(
            encoded, PointPrompt(request_id="r", points=[Point(32.0, 24.0, 1)],
                                 multimask=False))
        assert len(result.instances) == 1

    def test_negative_point_carves_the_mask(self, stub, encoded):
        click = Point(32.0, 24.0, 1)
        plain = stub.prompt_points(encoded, PointPrompt(request_id="r", points=[click],
                                                        multimask=False))
        carved = stub.prompt_points(
            encoded, PointPrompt(request_id="r",
                                 points=[click, Point(34.0, 25.0, 0)],
                                 multimask=False))
        assert sum(plain.instances[0].mask) > sum(carved.instances[0].mask)

    def test_points_are_deterministic(self, stub, encoded, image_data):
        prompt = PointPrompt(request_id="r", points=[Point(11.0, 12.0, 1),
                                                     Point(30.0, 31.0, 0)])
        a = stub.prompt_points(encoded, prompt).to_frame("j", "r", image_data.image_id)
        other = StubEngine(latency_scale=0.0)
        enc2 = other.encode_image(image_data)
        b = other.prompt_points(enc2, prompt).to_frame("j", "r", image_data.image_id)
        assert a == b

    def test_pvs_label_is_empty(self, stub, encoded):
        result = stub.prompt_points(encoded, PointPrompt(request_id="r",
                                                         points=[Point(32.0, 24.0, 1)]))
        assert {i.label for i in result.instances} == {""}

    def test_no_points_and_no_box_is_a_bad_request(self, stub, encoded):
        with pytest.raises(ApiError) as exc:
            stub.prompt_points(encoded, PointPrompt(request_id="r"))
        assert exc.value.code == ErrorCode.BAD_REQUEST

    def test_tracker_half_is_encoded_lazily(self, stub, image_data):
        enc = stub.encode_image(image_data)
        assert enc.has("pcs") and not enc.has("pvs")
        stub.prompt_text(enc, TextPrompt(request_id="r", text="cat"))
        assert not enc.has("pvs"), "a text prompt must not load the tracker"
        stub.prompt_points(enc, PointPrompt(request_id="r",
                                            points=[Point(32.0, 24.0, 1)]))
        assert enc.has("pvs")


# --------------------------------------------------------------------------- #
# soft edges -- the reason the format exists
# --------------------------------------------------------------------------- #
class TestSoftMasks:
    def test_masks_are_soft_not_binary(self, stub, encoded):
        result = stub.prompt_text(encoded, TextPrompt(request_id="r", text="cat"))
        values = set()
        for inst in result.instances:
            values.update(inst.mask)
        midtones = {v for v in values if 0 < v < 255}
        assert len(midtones) > 16, "a client's threshold slider would do nothing"

    def test_boundary_value_is_the_binarisation_point(self):
        """128 == sigmoid(0): the model's own boundary and the client default."""
        canvas = Size(400, 400)
        drawn = render_ellipse(canvas, 200.0, 200.0, 60.0, 60.0)
        assert drawn is not None
        bbox, mask = drawn
        # walk the horizontal centre line and find where it crosses 128
        row = 200 - bbox.y0
        line = mask[row * bbox.width:(row + 1) * bbox.width]
        crossing = next(x for x in range(len(line) - 1)
                        if line[x] >= DEFAULT_MASK_THRESHOLD > line[x + 1])
        crossing_x = bbox.x0 + crossing
        assert abs((crossing_x - 200) - 60) <= 2

    def test_falloff_is_monotonic_from_the_centre(self):
        canvas = Size(400, 400)
        bbox, mask = render_ellipse(canvas, 200.0, 200.0, 70.0, 50.0)
        row = 200 - bbox.y0
        line = list(mask[row * bbox.width:(row + 1) * bbox.width])
        centre = 200 - bbox.x0
        right = line[centre:]
        assert right[0] == 255
        assert right[-1] == 0
        assert all(a >= b for a, b in zip(right, right[1:]))

    def test_rect_has_soft_edges_too(self):
        canvas = Size(400, 400)
        bbox, mask = render_rect(canvas, 200.0, 200.0, 60.0, 40.0)
        values = set(mask)
        assert 255 in values
        assert 0 in values
        assert len({v for v in values if 0 < v < 255}) > 4
        row = 200 - bbox.y0
        line = list(mask[row * bbox.width:(row + 1) * bbox.width])
        centre = 200 - bbox.x0
        right = line[centre:]
        assert all(a >= b for a, b in zip(right, right[1:]))

    def test_render_clips_to_the_canvas(self):
        canvas = Size(100, 100)
        bbox, mask = render_ellipse(canvas, 5.0, 5.0, 40.0, 40.0)
        assert bbox.x0 >= 0 and bbox.y0 >= 0
        assert bbox.x1 <= 100 and bbox.y1 <= 100
        assert len(mask) == bbox.width * bbox.height

    def test_render_off_canvas_returns_none(self):
        assert render_ellipse(Size(100, 100), -400.0, -400.0, 10.0, 10.0) is None
        assert render_rect(Size(100, 100), 500.0, 500.0, 10.0, 10.0) is None


# --------------------------------------------------------------------------- #
# wire invariants (API.md §16)
# --------------------------------------------------------------------------- #
class TestWireInvariants:
    @pytest.mark.parametrize("kind", ["text", "points"])
    def test_frame_round_trips(self, stub, encoded, image_data, kind):
        if kind == "text":
            result = stub.prompt_text(encoded, TextPrompt(request_id="r-7", text="cat"))
        else:
            result = stub.prompt_points(
                encoded, PointPrompt(request_id="r-7", points=[Point(32.0, 24.0, 1)]))
        header, blob, frame = _frame_instances(result, req="r-7",
                                               image_id=image_data.image_id)
        assert header.request_id == "r-7"
        assert header.image_id == image_data.image_id
        assert header.engine == ("pcs" if kind == "text" else "pvs")
        assert header.mask_encoding == "u8_soft"
        assert header.state == "done"
        assert frame[:8] == b"SAM3RES\x00"
        assert header.blob_length == len(blob)

    def test_blob_offsets_are_tight_and_ordered(self, stub, encoded):
        result = stub.prompt_text(encoded, TextPrompt(request_id="r", text="person"))
        header, blob, _ = _frame_instances(result)
        offset = 0
        for i, inst in enumerate(header.instances):
            assert inst.instance_id == i
            assert inst.blob_offset == offset
            assert inst.blob_length == inst.mask_width * inst.mask_height
            assert inst.mask_width == inst.bbox.width
            assert inst.mask_height == inst.bbox.height
            assert len(inst.mask_bytes(blob)) == inst.blob_length
            offset += inst.blob_length
        assert offset == header.blob_length == len(blob)

    def test_bboxes_are_inside_the_canvas(self, stub, encoded):
        result = stub.prompt_text(encoded, TextPrompt(request_id="r", text="person"))
        header, _blob, _ = _frame_instances(result)
        for inst in header.instances:
            assert inst.bbox.is_valid(header.model_canvas)

    def test_header_carries_the_transform_clients_must_invert(self, stub, encoded):
        result = stub.prompt_text(encoded, TextPrompt(request_id="r", text="cat"))
        header, _blob, _ = _frame_instances(result)
        assert header.canvas_from_image.to_dict() == encoded.transform.to_dict()
        assert header.image.to_dict() == {"width": IMAGE_W, "height": IMAGE_H}

    def test_prompt_object_describes_the_request(self, stub, encoded):
        text = stub.prompt_text(encoded, TextPrompt(request_id="r", text="a red car",
                                                    score_threshold=0.25))
        assert text.prompt == {"kind": "text", "text": "a red car",
                               "score_threshold": 0.25}
        pts = stub.prompt_points(
            encoded, PointPrompt(request_id="r", points=[Point(1.0, 2.0, 0)],
                                 box=[1.0, 2.0, 30.0, 40.0], multimask=False))
        assert pts.prompt["kind"] == "points"
        assert pts.prompt["points"] == [{"x": 1.0, "y": 2.0, "label": 0}]
        assert pts.prompt["box"] == [1.0, 2.0, 30.0, 40.0]

    def test_zero_instance_result_is_legal(self, stub, encoded):
        result = stub.prompt_text(encoded, TextPrompt(request_id="r", text="cat",
                                                      score_threshold=1.0))
        assert result.instances == []
        header, blob, frame = _frame_instances(result)
        assert header.instances == []
        assert header.blob_length == 0
        assert len(blob) == 0
        assert len(frame) > 12

    def test_result_validate_catches_bad_ordering(self, stub, encoded):
        result = stub.prompt_text(encoded, TextPrompt(request_id="r", text="wheel"))
        if len(result.instances) < 2:
            pytest.skip("single instance, nothing to reorder")
        result.instances.reverse()
        with pytest.raises(ValueError):
            result.validate()


# --------------------------------------------------------------------------- #
# progress reporting (API.md §11)
# --------------------------------------------------------------------------- #
class TestProgress:
    def test_reporter_clamps_and_stays_monotonic(self):
        seen = []
        rep = ProgressReporter(lambda p, s: seen.append((p, s)))
        rep.absolute(0.5, Stage.ENCODING)
        rep.absolute(0.2, Stage.ENCODING)      # a regression is swallowed
        rep.absolute(9.0, Stage.PACKING)       # out of range is clamped
        assert [p for p, _ in seen] == [0.5, 0.5, 1.0]

    def test_stage_bands_match_the_reference_milestones(self):
        seen = []
        rep = ProgressReporter(lambda p, s: seen.append((p, s)))
        rep.stage(Stage.ENCODING, 0.0)
        rep.stage(Stage.ENCODING, 1.0)
        rep.stage(Stage.PROMPTING, 0.0)
        rep.stage(Stage.PACKING, 1.0)
        assert seen == [(0.05, "encoding"), (0.60, "encoding"),
                        (0.60, "prompting"), (1.0, "packing")]

    def test_encode_progress_advances_through_the_encoding_band(self, stub, image_data):
        seen = []
        stub.encode_image(image_data, progress=lambda p, s: seen.append((p, s)))
        assert seen, "encode reported no progress at all"
        assert all(s == Stage.ENCODING for _p, s in seen)
        assert [p for p, _ in seen] == sorted(p for p, _ in seen)
        assert seen[0][0] >= 0.05
        assert seen[-1][0] == pytest.approx(0.60)

    def test_prompt_progress_ends_at_one(self, stub, encoded):
        seen = []
        stub.prompt_text(encoded, TextPrompt(request_id="r", text="cat"),
                         progress=lambda p, s: seen.append((p, s)))
        stages = [s for _p, s in seen]
        assert stages[0] == Stage.PROMPTING
        assert stages[-1] == Stage.PACKING
        assert seen[-1][0] == pytest.approx(1.0)
        assert [p for p, _ in seen] == sorted(p for p, _ in seen)

    def test_lazy_tracker_encode_reports_the_encoding_band_first(self, stub, image_data):
        enc = stub.encode_image(image_data)
        seen = []
        stub.prompt_points(enc, PointPrompt(request_id="r",
                                            points=[Point(32.0, 24.0, 1)]),
                           progress=lambda p, s: seen.append((p, s)))
        assert seen[0][1] == Stage.ENCODING
        assert seen[-1][0] == pytest.approx(1.0)

    def test_progress_callback_is_optional(self, stub, image_data):
        enc = stub.encode_image(image_data, progress=None)
        assert stub.prompt_text(enc, TextPrompt(request_id="r", text="cat"))


# --------------------------------------------------------------------------- #
# build_instances / finalize_instances -- the shared invariant enforcer
# --------------------------------------------------------------------------- #
class TestBuildInstances:
    @staticmethod
    def _canvas_mask(canvas: Size, x0, y0, x1, y1, value=200):
        buf = bytearray(canvas.area)
        for y in range(y0, y1):
            buf[y * canvas.width + x0:y * canvas.width + x1] = bytes([value]) * (x1 - x0)
        return bytes(buf)

    def test_crops_tightly_and_keeps_the_invariants(self):
        canvas = Size(64, 48)
        mask = self._canvas_mask(canvas, 10, 12, 30, 20)
        got, truncated = build_instances(
            canvas, [CanvasMask(score=0.9, mask=mask, label="x")], pad=0)
        assert truncated is False
        inst = got[0]
        assert inst.bbox.to_list() == [10, 12, 30, 20]
        assert len(inst.mask) == inst.bbox.width * inst.bbox.height
        assert set(inst.mask) == {200}
        inst.validate(canvas)

    def test_padding_expands_and_clips(self):
        canvas = Size(64, 48)
        mask = self._canvas_mask(canvas, 0, 0, 5, 5)
        got, _ = build_instances(canvas, [CanvasMask(score=0.5, mask=mask)], pad=3)
        assert got[0].bbox.to_list() == [0, 0, 8, 8]

    def test_sorts_descending_and_truncates(self):
        canvas = Size(32, 32)
        masks = [CanvasMask(score=s, mask=self._canvas_mask(canvas, 1, 1, 8, 8))
                 for s in (0.2, 0.9, 0.5)]
        got, truncated = build_instances(canvas, masks, max_instances=2)
        assert [i.score for i in got] == [0.9, 0.5]
        assert truncated is True

    def test_score_threshold_drops_candidates(self):
        canvas = Size(32, 32)
        masks = [CanvasMask(score=s, mask=self._canvas_mask(canvas, 1, 1, 8, 8))
                 for s in (0.05, 0.4)]
        got, _ = build_instances(canvas, masks, score_threshold=0.1)
        assert [i.score for i in got] == [0.4]

    def test_empty_mask_is_dropped(self):
        canvas = Size(32, 32)
        got, _ = build_instances(canvas, [CanvasMask(score=0.9, mask=bytes(canvas.area))])
        assert got == []

    def test_a_mask_below_the_crop_epsilon_everywhere_is_dropped(self):
        """Such an instance shows nothing at any threshold a client can set
        (1..255 on bytes that are all < 8), so it is not sent at all -- a
        confident score alone does not make a selectable instance."""
        canvas = Size(32, 32)
        faint = bytes([engine_base.DEFAULT_CROP_EPSILON - 1]) * canvas.area
        got, truncated = build_instances(canvas, [CanvasMask(score=0.99, mask=faint)])
        assert got == [] and truncated is False
        visible = bytes([engine_base.DEFAULT_CROP_EPSILON]) * canvas.area
        got, _ = build_instances(canvas, [CanvasMask(score=0.99, mask=visible)])
        assert len(got) == 1

    def test_wrong_sized_mask_is_rejected(self):
        canvas = Size(32, 32)
        with pytest.raises(ValueError):
            build_instances(canvas, [CanvasMask(score=0.5, mask=bytes(10))])

    def test_finalize_sorts_and_validates(self):
        canvas = Size(32, 32)
        raws = [RawInstance(score=s, bbox=BBox(0, 0, 4, 4), mask=bytes(16))
                for s in (0.3, 0.8)]
        got, truncated = finalize_instances(raws, canvas)
        assert [i.score for i in got] == [0.8, 0.3]
        assert truncated is False

    def test_finalize_rejects_a_mask_that_does_not_match_its_bbox(self):
        canvas = Size(32, 32)
        bad = RawInstance(score=0.5, bbox=BBox(0, 0, 4, 4), mask=bytes(15))
        with pytest.raises(ValueError):
            finalize_instances([bad], canvas)


# --------------------------------------------------------------------------- #
# determinism primitives
# --------------------------------------------------------------------------- #
class TestDeterminism:
    def test_seed_is_stable_and_independent_of_PYTHONHASHSEED(self):
        assert stub_seed("pcs", "abc", "cat") == stub_seed("pcs", "abc", "cat")
        assert stub_seed("pcs", "abc", "cat") != stub_seed("pcs", "abc", "dog")
        # a literal value pins the algorithm: changing it changes every frame
        assert stub_seed("pcs", "abc", "cat") == 3956134231772386661

    def test_rng_stream_is_reproducible(self):
        a = [_Rng(42).next_u64() for _ in range(3)]
        b = [_Rng(42).next_u64() for _ in range(3)]
        assert a == b
        assert a != [_Rng(43).next_u64() for _ in range(3)]

    def test_rng_random_is_in_range(self):
        rng = _Rng(7)
        values = [rng.random() for _ in range(500)]
        assert all(0.0 <= v < 1.0 for v in values)
        assert len(set(values)) > 400

    def test_rng_randint_covers_its_range(self):
        rng = _Rng(9)
        seen = {rng.randint(1, 8) for _ in range(500)}
        assert seen == set(range(1, 9))


# --------------------------------------------------------------------------- #
# PVS nesting -- "getting that wrong is the classic bug here"
# --------------------------------------------------------------------------- #
class TestPvsNesting:
    def test_points_are_four_levels_deep(self):
        points, labels = nest_points([Point(500.0, 375.0, 1)])
        assert points == [[[[500.0, 375.0]]]]
        assert labels == [[[1]]]
        # batch, objects, points, coords
        assert len(points) == 1 and len(points[0]) == 1
        assert len(points[0][0]) == 1 and len(points[0][0][0]) == 2

    def test_labels_are_three_levels_deep(self):
        points, labels = nest_points([Point(1.0, 2.0, 1), Point(3.0, 4.0, 0)])
        assert points == [[[[1.0, 2.0], [3.0, 4.0]]]]
        assert labels == [[[1, 0]]]
        assert len(labels[0][0]) == len(points[0][0])

    def test_non_one_labels_become_zero(self):
        _points, labels = nest_points([Point(1.0, 2.0, 7)])
        assert labels == [[[0]]]

    def test_empty_points_are_none(self):
        assert nest_points([]) == (None, None)

    def test_box_is_three_levels_deep_and_normalised(self):
        assert nest_box([30.0, 40.0, 1.0, 2.0]) == [[[1.0, 2.0, 30.0, 40.0]]]
        assert nest_box(None) is None

    def test_degenerate_box_is_rejected(self):
        with pytest.raises(ApiError) as exc:
            nest_box([5.0, 5.0, 5.2, 40.0])
        assert exc.value.code == ErrorCode.BAD_REQUEST

    def test_wrong_length_box_is_rejected(self):
        with pytest.raises(ApiError):
            nest_box([1.0, 2.0, 3.0])


# --------------------------------------------------------------------------- #
# model manager -- must be honest, not explosive, without torch
# --------------------------------------------------------------------------- #
class TestModelManager:
    def test_probe_reports_missing_torch_without_raising(self, monkeypatch):
        monkeypatch.setattr(modelmgr, "_PROBE", modelmgr._PROBE)   # restored afterwards
        # ``sys.modules[name] = None`` makes ``import name`` raise ImportError,
        # which is what an interpreter without the package does.
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "transformers", None)
        probe = modelmgr.probe_torch(refresh=True)
        assert probe.available is False
        assert probe.error and "torch" in probe.error.lower()
        assert probe.to_dict()["torch_available"] is False
        assert probe.transformers_available is False

    def test_manager_constructs_and_describes_without_torch(self, no_torch):
        mgr = modelmgr.ModelManager()
        assert mgr.device == "cpu"
        assert mgr.dtype == "float32"
        assert mgr.loaded() == []
        info = mgr.describe()
        assert info["usable"] is False
        assert info["probed"] is True
        assert "torch is not installed" in info["unavailable_reason"]
        assert info["model_id"] == "facebook/sam3"

    def test_available_is_false_and_ensure_raises_the_documented_code(self, no_torch):
        mgr = modelmgr.ModelManager()
        ok, reason = mgr.available()
        assert ok is False and reason
        with pytest.raises(ApiError) as exc:
            mgr.ensure_available()
        assert exc.value.code == ErrorCode.ENGINE_UNAVAILABLE
        assert exc.value.status == 503

    def test_unload_is_safe_when_nothing_is_loaded(self):
        mgr = modelmgr.ModelManager()
        assert mgr.unload() == []
        assert mgr.unload("pcs") == []
        mgr.close()

    def test_touch_resets_the_idle_clock(self):
        mgr = modelmgr.ModelManager()
        assert mgr.idle_seconds() < 1.0
        mgr._last_used -= 100.0
        assert mgr.idle_seconds() > 99.0
        mgr.touch()
        assert mgr.idle_seconds() < 1.0

    def test_load_of_an_unknown_half_is_a_programming_error(self):
        with pytest.raises(ValueError):
            modelmgr.ModelManager().load("nope")

    def test_memory_reports_rss_and_null_vram(self, no_torch):
        mem = modelmgr.ModelManager().memory()
        assert mem["vram_total"] is None and mem["vram_free"] is None
        assert mem["rss_bytes"] is None or mem["rss_bytes"] > 0

    @pytest.mark.skipif(not sys.platform.startswith("win"), reason="Windows API")
    def test_windows_reports_the_working_set(self):
        """Windows has no ``resource`` module; the working set stands in."""
        assert modelmgr._windows_working_set() > 1 << 20
        assert modelmgr._rss_bytes() > 1 << 20

    def test_hf_home_is_pinned_under_the_project_base(self, sam3_home, monkeypatch):
        monkeypatch.delenv("HF_HOME", raising=False)
        applied = modelmgr.configure_hf_env()
        assert applied["HF_HOME"].startswith(str(sam3_home))
        assert applied["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"

    def test_a_user_set_hf_home_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HF_HOME", str(tmp_path / "mine"))
        assert modelmgr.configure_hf_env()["HF_HOME"] == str(tmp_path / "mine")

    def test_weights_are_absent_from_an_empty_home(self, no_weights):
        assert modelmgr.weights_available() is False

    def test_a_local_checkpoint_is_found_and_preferred(self, no_weights, tmp_path,
                                                       monkeypatch):
        ckpt = tmp_path / "sam3-local"
        ckpt.mkdir()
        (ckpt / "config.json").write_text("{}")
        monkeypatch.setenv(modelmgr.ENV_WEIGHTS_DIR, str(ckpt))
        assert modelmgr.ENV_WEIGHTS_DIR == "SAM3_WEIGHTS_DIR"   # docs/DEVELOPING.md
        assert modelmgr.local_checkpoint() == str(ckpt)
        assert modelmgr.weights_available() is True
        assert modelmgr.resolve_model_id() == str(ckpt)

    def test_the_checkpoint_setup_recorded_is_found(self, no_weights, tmp_path):
        """Setup's "Use this folder" and "Convert and use" write the folder to
        ``weights.json`` in the base directory; the daemon must use it even
        though nothing exports ``SAM3_WEIGHTS_DIR`` to it."""
        ckpt = tmp_path / "sam3-converted"
        ckpt.mkdir()
        (ckpt / "config.json").write_text("{}")
        assert modelmgr.local_checkpoint() is None
        (no_weights / "weights.json").write_text(
            '{"local_path": "%s", "source": "local"}' % ckpt.as_posix())
        assert modelmgr.local_checkpoint() == ckpt.as_posix()
        assert modelmgr.weights_available() is True

    def test_a_directory_without_a_config_is_not_a_checkpoint(self, no_weights, tmp_path,
                                                              monkeypatch):
        monkeypatch.setenv(modelmgr.ENV_WEIGHTS_DIR, str(tmp_path))
        assert modelmgr.local_checkpoint() is None

    def test_an_hf_cache_directory_counts_as_weights(self, no_weights, monkeypatch):
        cache = no_weights / "hf" / "hub" / "models--facebook--sam3"
        cache.mkdir(parents=True)
        (cache / "refs").write_text("x")
        monkeypatch.setenv("HF_HOME", str(no_weights / "hf"))
        assert modelmgr.weights_available() is True


class TestDevicePolicy:
    """``DESIGN.md`` §7: cuda -> mps -> cpu; bf16 on Ampere+, fp16 older, fp32 CPU."""

    @staticmethod
    def _probe(**kw):
        return modelmgr.TorchProbe(**kw)

    def test_auto_prefers_cuda_then_mps_then_cpu(self):
        cuda = self._probe(available=True, cuda_available=True)
        mps = self._probe(available=True, mps_available=True)
        bare = self._probe(available=True)
        assert modelmgr.select_device("auto", cuda) == "cuda"
        assert modelmgr.select_device("auto", mps) == "mps"
        assert modelmgr.select_device("auto", bare) == "cpu"

    def test_auto_without_torch_is_cpu(self):
        assert modelmgr.select_device("auto", self._probe()) == "cpu"

    def test_an_explicit_device_is_honoured_verbatim(self):
        probe = self._probe(available=True)
        assert modelmgr.select_device("cuda:1", probe) == "cuda:1"
        assert modelmgr.select_device("MPS", probe) == "mps"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv(modelmgr.ENV_DEVICE, "cuda")
        assert modelmgr.select_device("auto", self._probe(available=True)) == "cuda"

    def test_bf16_on_ampere_and_newer(self):
        probe = self._probe(available=True, cuda_available=True,
                            cuda_capability=(8, 6), bf16_supported=True)
        assert modelmgr.select_dtype("cuda", "auto", probe) == "bfloat16"

    def test_fp16_on_older_cuda(self):
        probe = self._probe(available=True, cuda_available=True,
                            cuda_capability=(7, 5), bf16_supported=False)
        assert modelmgr.select_dtype("cuda", "auto", probe) == "float16"

    def test_rocm_gets_fp16_whatever_the_capability_says(self):
        """A HIP build reports a gfx-derived capability -- (10, 3) on RDNA2,
        (11, 0) on RDNA3 -- which the '>= 8 means Ampere' rule would read as
        bf16-capable.  fp16 is native on every ROCm GPU; bf16 is opt-in."""
        for cap in ((10, 3), (11, 0), (9, 0)):
            probe = self._probe(available=True, cuda_available=True, hip=True,
                                cuda_capability=cap, bf16_supported=True)
            assert modelmgr.select_dtype("cuda", "auto", probe) == "float16", cap
        probe = self._probe(available=True, cuda_available=True, hip=True,
                            cuda_capability=(11, 0), bf16_supported=True)
        assert modelmgr.select_dtype("cuda", "bf16", probe) == "bfloat16"

    def test_the_probe_reports_hip(self):
        probe = self._probe(available=True, hip=True, hip_version="6.4.1")
        d = probe.to_dict()
        assert d["hip"] is True and d["hip_version"] == "6.4.1"

    def test_turing_gets_fp16_even_when_torch_claims_bf16_support(self):
        """Regression: an RTX 2080 Ti must not be given emulated bf16.

        Since torch 2.6 ``torch.cuda.is_bf16_supported()`` defaults to
        ``including_emulation=True`` and answers True on Turing (sm_75), where
        bf16 is emulated rather than native.  Trusting it there costs real
        throughput for no numerical gain, so compute capability is the
        authority and 8.0 (Ampere) is the gate.
        """
        probe = self._probe(available=True, cuda_available=True,
                            cuda_capability=(7, 5), bf16_supported=True,
                            cuda_device_name="NVIDIA GeForce RTX 2080 Ti")
        assert modelmgr.select_dtype("cuda", "auto", probe) == "float16"

    def test_ampere_without_bf16_support_falls_back_to_fp16(self):
        """The probe still gets a veto, for the odd build that reports no bf16."""
        probe = self._probe(available=True, cuda_available=True,
                            cuda_capability=(8, 6), bf16_supported=False)
        assert modelmgr.select_dtype("cuda", "auto", probe) == "float16"

    def test_an_explicit_bf16_request_is_still_honoured_on_turing(self):
        """The gate is the *auto* policy, not a ban -- an explicit ask wins."""
        probe = self._probe(available=True, cuda_available=True,
                            cuda_capability=(7, 5), bf16_supported=True)
        assert modelmgr.select_dtype("cuda", "bf16", probe) == "bfloat16"

    def test_fp32_on_cpu(self):
        probe = self._probe(available=True)
        assert modelmgr.select_dtype("cpu", "auto", probe) == "float32"

    def test_fp16_on_mps(self):
        probe = self._probe(available=True, mps_available=True)
        assert modelmgr.select_dtype("mps", "auto", probe) == "float16"

    @pytest.mark.parametrize("given,expected", [
        ("bf16", "bfloat16"), ("fp16", "float16"), ("half", "float16"),
        ("fp32", "float32"), ("float", "float32"), ("bfloat16", "bfloat16"),
    ])
    def test_dtype_aliases(self, given, expected):
        assert modelmgr.select_dtype("cuda", given, self._probe()) == expected

    def test_unknown_dtype_falls_back_to_the_device_policy(self):
        probe = self._probe(available=True)
        assert modelmgr.select_dtype("cpu", "quantum", probe) == "float32"


def _loading_manager(dtype="bfloat16", fail=None, gate=None):
    """A manager whose loads hand out ``"<half>@<dtype>"`` strings.

    ``fail`` maps a dtype name to the exception its load raises; ``gate`` is an
    event every load waits on.  Returns ``(manager, loads)``.
    """
    mgr = modelmgr.ModelManager(device="cuda", dtype=dtype, configure_hf=False)
    mgr.ensure_available = lambda: None
    loads = []

    def _load(half, dtype_name):
        loads.append((half, dtype_name))
        if gate is not None:
            assert gate.wait(10.0), "test gate never opened"
        if fail and dtype_name in fail:
            raise fail[dtype_name]
        return modelmgr._Loaded(half, "%s@%s" % (half, dtype_name), "processor",
                                dtype_name, "cuda")

    mgr._load_locked = _load
    return mgr, loads


class _OutOfMemoryError(RuntimeError):
    """Stands in for ``torch.OutOfMemoryError``, a ``RuntimeError`` subclass."""


class TestFp32Fallback:
    """The safety net from ``DESIGN.md`` §7."""

    def test_numerical_failure_is_detected_by_type(self):
        assert modelmgr._is_numerical_failure(modelmgr.NumericalFailure("nan"))

    @pytest.mark.parametrize("message", [
        "output contained NaN",
        "probability tensor contains either `inf`, `nan` or element < 0",
        "\"upsample_bilinear2d\" not implemented for 'Half'",
        "\"LayerNormKernelImpl\" not implemented for 'BFloat16'",
        "value cannot be converted to type at::Half without overflow",
        "CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasGemmEx(...)`",
        "CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling `cublasGemmEx(...)`",
    ])
    def test_numerical_failure_is_detected_by_message(self, message):
        assert modelmgr._is_numerical_failure(RuntimeError(message))
        assert modelmgr._is_numerical_failure(NotImplementedError(message))

    @pytest.mark.parametrize("exc", [
        RuntimeError("connection refused"),
        RuntimeError("no such file or directory"),
        RuntimeError("out of memory"),
        # "inf" inside a word is not Inf.
        RuntimeError("Inference tensors cannot be saved for backward. To work around "
                     "you can make a clone to get a normal tensor and use it in autograd."),
        RuntimeError("Inplace update to inference tensor outside InferenceMode is not allowed."),
        RuntimeError("The size of tensor a (256) must match the size of tensor b (252) "
                     "at non-singleton dimension 3"),
        # Out of memory, however spelled: a float32 reload needs twice as much.
        _OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a "
                          "total capacity of 7.79 GiB of which 3.44 MiB is free."),
        RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling "
                     "`cublasCreate(handle)`"),
        # A dtype mismatch is plumbing, not precision.
        RuntimeError("expected scalar type Float but found BFloat16"),
        RuntimeError("expected m1 and m2 to have the same dtype, but got: "
                     "c10::BFloat16 != float"),
        RuntimeError("Input type (torch.cuda.HalfTensor) and weight type "
                     "(torch.cuda.FloatTensor) should be the same"),
        # An engine's own error is already a verdict, whatever its words.
        engine_base.inference_failed("this Sam3Model has no get_vision_features(); "
                                     "embedding reuse is unavailable"),
        engine_base.inference_failed("post-processing produced NaN"),
        KeyError("pred_masks_info"),
        ValueError("nan"),
    ])
    def test_unrelated_failures_are_not_treated_as_numerical(self, exc):
        assert not modelmgr._is_numerical_failure(exc)

    def test_fallback_reloads_in_fp32_and_retries_with_the_new_model(self):
        mgr, loads = _loading_manager("bfloat16")
        seen = []

        def flaky(model, processor):
            seen.append((model, mgr.dtype))
            if len(seen) == 1:
                raise modelmgr.NumericalFailure("mask tensor contains NaN or Inf")
            return "ok"

        assert mgr.run("pcs", flaky) == "ok"
        assert seen == [("pcs@bfloat16", "bfloat16"), ("pcs@float32", "float32")]
        assert loads == [("pcs", "bfloat16"), ("pcs", "float32")]
        assert mgr.dtype == "float32" and mgr.fp32_fallback is True
        assert mgr.loaded() == ["pcs"]
        assert mgr.describe()["fp32_fallback"] is True

    def test_the_other_half_reloads_in_fp32_too(self):
        mgr, loads = _loading_manager("float16")
        mgr.load("pvs")
        calls = []

        def once(model, processor):
            calls.append(model)
            if len(calls) == 1:
                raise RuntimeError("\"upsample_bilinear2d\" not implemented for 'Half'")
            return model

        assert mgr.run("pcs", once) == "pcs@float32"
        assert mgr.loaded() == ["pcs"], "the half-precision tracker must be released"
        assert mgr.load("pvs") == ("pvs@float32", "processor")

    def test_a_failed_fp32_reload_leaves_half_precision_in_force(self):
        """float32 needs twice the memory; if it does not fit, the daemon must
        not be left claiming float32 with nothing loaded."""
        oom = ApiError(ErrorCode.MODEL_LOAD_FAILED,
                       "could not load Sam3Model: CUDA out of memory", {})
        mgr, loads = _loading_manager("bfloat16", fail={"float32": oom})

        def nan(model, processor):
            raise modelmgr.NumericalFailure("mask tensor contains NaN or Inf")

        with pytest.raises(ApiError, match="out of memory"):
            mgr.run("pcs", nan)
        assert mgr.dtype == "bfloat16"
        assert mgr.fp32_fallback is False
        assert mgr.loaded() == []
        assert loads == [("pcs", "bfloat16"), ("pcs", "float32")]
        # The next request starts afresh, in half precision.
        assert mgr.load("pcs") == ("pcs@bfloat16", "processor")

    @pytest.mark.parametrize("exc", [
        engine_base.inference_failed("inference failed: no masks"),
        _OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        RuntimeError("expected scalar type BFloat16 but found Float"),
    ])
    def test_failures_inside_fn_that_are_not_numerical_do_not_reload(self, exc):
        mgr, loads = _loading_manager("bfloat16")

        def boom(model, processor):
            raise exc

        with pytest.raises(type(exc)):
            mgr.run("pcs", boom)
        assert loads == [("pcs", "bfloat16")]
        assert mgr.dtype == "bfloat16" and mgr.fp32_fallback is False
        assert mgr.loaded() == ["pcs"]

    def test_a_non_numerical_failure_propagates_untouched(self):
        mgr, _loads = _loading_manager("float16")

        def boom(model, processor):
            raise RuntimeError("disk full")

        with pytest.raises(RuntimeError, match="disk full"):
            mgr.run("pcs", boom)
        assert mgr.fp32_fallback is False

    def test_fp32_never_retries(self):
        mgr, loads = _loading_manager("float32")
        calls = []

        def boom(model, processor):
            calls.append(1)
            raise modelmgr.NumericalFailure("nan")

        with pytest.raises(modelmgr.NumericalFailure):
            mgr.run("pcs", boom)
        assert calls == [1]
        assert loads == [("pcs", "float32")]


# --------------------------------------------------------------------------- #
# the real engines, without the hardware to run them
# --------------------------------------------------------------------------- #
class TestTorchEnginesWithoutTorch:
    @pytest.fixture(autouse=True)
    def _without_torch(self, no_torch):
        return no_torch

    def test_describe_is_honest_about_being_unusable(self, no_weights):
        info = TorchEngine().describe()
        assert info.mode == "torch"
        assert info.torch_available is False
        assert info.weights_available is False
        assert info.dtype == "none"
        assert info.device == "cpu"
        # What the engine serves once it is usable; every prompt fails with
        # engine_unavailable until then.
        assert set(info.capabilities) == {"pcs", "pvs", "exemplar_boxes"}
        assert info.detail["usable"] is False
        assert info.detail["probed"] is True

    def test_encode_raises_engine_unavailable_not_import_error(self, image_data):
        with pytest.raises(ApiError) as exc:
            TorchEngine().encode_image(image_data)
        assert exc.value.code == ErrorCode.ENGINE_UNAVAILABLE

    def test_text_prompt_raises_engine_unavailable(self, image_data):
        engine = TorchEngine()
        canvas, tf = engine.canvas_for(image_data.size)
        enc = EncodedImage(image_data.image_id, image_data.size, canvas, tf,
                           image_data.pixels)
        with pytest.raises(ApiError) as exc:
            engine.prompt_text(enc, TextPrompt(request_id="r", text="cat"))
        assert exc.value.code == ErrorCode.ENGINE_UNAVAILABLE

    def test_point_prompt_raises_engine_unavailable(self, image_data):
        engine = TorchEngine()
        canvas, tf = engine.canvas_for(image_data.size)
        enc = EncodedImage(image_data.image_id, image_data.size, canvas, tf,
                           image_data.pixels)
        with pytest.raises(ApiError) as exc:
            engine.prompt_points(enc, PointPrompt(request_id="r",
                                                  points=[Point(1.0, 2.0, 1)]))
        assert exc.value.code == ErrorCode.ENGINE_UNAVAILABLE

    def test_each_half_refuses_the_other_half_s_prompt(self, image_data):
        canvas, tf = default_canvas_for(image_data.size)
        enc = EncodedImage(image_data.image_id, image_data.size, canvas, tf,
                           image_data.pixels)
        with pytest.raises(ApiError) as exc:
            PcsEngine().prompt_points(enc, PointPrompt(request_id="r"))
        assert exc.value.code == ErrorCode.ENGINE_UNAVAILABLE
        with pytest.raises(ApiError) as exc:
            PvsEngine().prompt_text(enc, TextPrompt(request_id="r", text="cat"))
        assert exc.value.code == ErrorCode.ENGINE_UNAVAILABLE

    def test_both_halves_share_one_manager(self):
        engine = TorchEngine()
        assert engine.pcs.mgr is engine.mgr
        assert engine.pvs.mgr is engine.mgr

    def test_text_length_limit_is_enforced_before_any_model_work(self, image_data):
        canvas, tf = default_canvas_for(image_data.size)
        enc = EncodedImage(image_data.image_id, image_data.size, canvas, tf,
                           image_data.pixels)
        long_text = "x" * (Limits.MAX_TEXT_CHARS + 1)
        with pytest.raises(ApiError) as exc:
            PcsEngine().prompt_text(enc, TextPrompt(request_id="r", text=long_text))
        assert exc.value.code == ErrorCode.BAD_REQUEST

    def test_too_many_points_is_a_bad_request(self, image_data):
        canvas, tf = default_canvas_for(image_data.size)
        enc = EncodedImage(image_data.image_id, image_data.size, canvas, tf,
                           image_data.pixels)
        points = [Point(float(i), 1.0, 1) for i in range(Limits.MAX_POINTS + 1)]
        with pytest.raises(ApiError) as exc:
            PvsEngine().prompt_points(enc, PointPrompt(request_id="r", points=points))
        assert exc.value.code == ErrorCode.BAD_REQUEST


class TestFactory:
    def test_stub_mode(self):
        assert isinstance(create_engine("stub"), StubEngine)

    def test_torch_mode_constructs_even_with_no_torch(self):
        assert isinstance(create_engine("torch"), TorchEngine)

    def test_auto_picks_the_stub_exactly_when_torch_is_missing(self, no_torch):
        assert isinstance(create_engine("auto", latency_scale=0.0), StubEngine)

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError):
            create_engine("gpu-go-brrr")

    def test_manager_kwargs_reach_the_manager(self, monkeypatch):
        def _no_probe(refresh=False):
            raise AssertionError("an explicit device and dtype need no probe")

        monkeypatch.setattr(modelmgr, "probe_torch", _no_probe)
        engine = create_engine("torch", device="cuda", dtype="float32",
                               model_id="local/thing")
        assert engine.mgr.device == "cuda"
        assert engine.mgr.dtype == "float32"
        assert engine.mgr.model_id == "local/thing"


# --------------------------------------------------------------------------- #
# EncodedImage bookkeeping the server relies on
# --------------------------------------------------------------------------- #
class TestEncodedImage:
    def test_parts_are_tracked_and_droppable(self, stub, image_data):
        enc = stub.encode_image(image_data)
        assert enc.has("pcs")
        enc.put("pvs", {"x": 1}, bytes_estimate=1024)
        assert enc.bytes_estimate == 1024
        described = enc.describe()
        assert described["parts"] == ["pcs", "pvs"]
        assert described["bytes_estimate"] == 1024 + len(image_data.pixels)
        enc.drop("pvs")
        assert not enc.has("pvs")
        enc.drop()
        assert enc.parts == {}

    def test_pixels_are_retained_for_the_second_half(self, stub, image_data):
        enc = stub.encode_image(image_data)
        assert enc.pixels == image_data.pixels


# --------------------------------------------------------------------------- #
# latency simulation (API.md §14: "a few hundred milliseconds, not seconds")
# --------------------------------------------------------------------------- #
class TestStubLatency:
    def test_zero_scale_removes_the_sleeps(self, image_data):
        import time

        engine = StubEngine(latency_scale=0.0)
        started = time.time()
        enc = engine.encode_image(image_data)
        engine.prompt_text(enc, TextPrompt(request_id="r", text="cat"))
        assert time.time() - started < 1.0

    @pytest.mark.slow
    def test_default_latency_is_sub_second(self, image_data):
        import time

        engine = StubEngine()
        started = time.time()
        enc = engine.encode_image(image_data)
        engine.prompt_text(enc, TextPrompt(request_id="r", text="yellow school bus"))
        elapsed = time.time() - started
        assert 0.1 < elapsed < 2.0


# --------------------------------------------------------------------------- #
# real inference -- only runs where the hardware and the gated weights exist
# --------------------------------------------------------------------------- #
@pytest.mark.needs_torch
@pytest.mark.needs_weights
@pytest.mark.slow
class TestRealEngines:
    def test_text_prompt_produces_a_valid_frame(self, image_data):
        engine = TorchEngine()
        enc = engine.encode_image(image_data)
        result = engine.prompt_text(enc, TextPrompt(request_id="r", text="a shape"))
        result.validate()
        header, blob = unpack_result(result.to_frame("j", "r", image_data.image_id))
        assert header.engine == "pcs"
        assert len(blob) == header.blob_length

    def test_a_second_text_prompt_reuses_the_cached_embedding(self, image_data):
        engine = TorchEngine()
        enc = engine.encode_image(image_data)
        engine.prompt_text(enc, TextPrompt(request_id="r1", text="a shape"))
        assert engine.pcs._vision_cache_works is True, \
            "embedding reuse is the entire performance argument for the daemon"

    def test_point_prompt_loads_the_tracker_lazily(self, image_data):
        engine = TorchEngine()
        enc = engine.encode_image(image_data)
        assert "pvs" not in engine.mgr.loaded()
        engine.prompt_points(enc, PointPrompt(request_id="r",
                                              points=[Point(32.0, 24.0, 1)]))
        assert "pvs" in engine.mgr.loaded()
        assert enc.has("pvs")

    def test_masks_are_soft(self, image_data):
        engine = TorchEngine()
        enc = engine.encode_image(image_data)
        result = engine.prompt_points(enc, PointPrompt(request_id="r",
                                                       points=[Point(32.0, 24.0, 1)]))
        values = set()
        for inst in result.instances:
            values.update(inst.mask)
        assert len({v for v in values if 0 < v < 255}) > 4



# --------------------------------------------------------------------------- #
# logic audit: the processor mirror and the forward-kwargs invariant
# --------------------------------------------------------------------------- #
class TestHfPostProcessMirror:
    """Pure-Python mirror of Sam3ImageProcessor.post_process_instance_segmentation.

    The soft masks on the wire are only correct if the rows taken from the raw
    logits are exactly the rows the processor kept, in the same order.
    """

    def test_scores_are_sigmoid_of_logits(self):
        import sam3gimpd.engines.base as B
        s = B.hf_instance_scores([0.0, 100.0, -100.0])
        assert s[0] == pytest.approx(0.5)
        assert s[1] == pytest.approx(1.0) and s[2] == pytest.approx(0.0, abs=1e-9)

    def test_presence_scales_every_instance(self):
        import sam3gimpd.engines.base as B
        without = B.hf_instance_scores([0.0, 2.0])
        with_p = B.hf_instance_scores([0.0, 2.0], presence_logit=0.0)
        assert [w * 0.5 for w in without] == pytest.approx(with_p)

    def test_a_confident_absence_sinks_everything_below_the_floor(self):
        """Why 'guitar strap' can return nothing at a 0.02 floor: presence."""
        import sam3gimpd.engines.base as B
        s = B.hf_instance_scores([3.0, 2.0, 1.0], presence_logit=-6.0)
        assert B.hf_keep_indices(s, 0.02) == []

    def test_keep_is_strict_and_order_preserving(self):
        import sam3gimpd.engines.base as B
        assert B.hf_keep_indices([0.9, 0.02, 0.5, 0.021], 0.02) == [0, 2, 3]

    def test_keep_matches_the_processor_on_a_realistic_row(self):
        import sam3gimpd.engines.base as B
        logits = [-4.0, 1.2, -0.3, 2.5, -2.0, 0.1]
        scores = B.hf_instance_scores(logits, presence_logit=1.0)
        keep = B.hf_keep_indices(scores, 0.3)
        expected = [i for i, l in enumerate(logits)
                    if (1 / (1 + math.exp(-l))) * (1 / (1 + math.exp(-1.0))) > 0.3]
        assert keep == expected


class TestForwardKwargsInvariant:
    """Sam3Model.forward raises unless exactly one of pixel_values /
    vision_embeds is given, so the cached-encoding path must never supply both."""

    def test_cached_encoding_replaces_pixel_values(self):
        import sam3gimpd.engines.base as B
        out = B.forward_kwargs_for({"pixel_values": "px", "original_sizes": "os",
                                    "input_ids": "ids", "attention_mask": "am"},
                                   vision_embeds="enc")
        assert out["vision_embeds"] == "enc"
        assert "pixel_values" not in out and "original_sizes" not in out
        assert out["input_ids"] == "ids" and out["attention_mask"] == "am"

    def test_without_a_cache_pixel_values_is_the_source(self):
        import sam3gimpd.engines.base as B
        out = B.forward_kwargs_for({"pixel_values": "px", "input_ids": "ids"})
        assert out["pixel_values"] == "px" and "vision_embeds" not in out

    def test_never_both(self):
        import sam3gimpd.engines.base as B
        out = B.forward_kwargs_for({"pixel_values": "px", "input_ids": "i"},
                                   vision_embeds="enc")
        assert not ("pixel_values" in out and "vision_embeds" in out)

    def test_neither_is_an_error(self):
        import sam3gimpd.engines.base as B
        with pytest.raises(ValueError):
            B.forward_kwargs_for({"input_ids": "i"})

    def test_prompt_path_calls_the_processor_text_only_when_cached(self, fake_torch,
                                                                   image_data):
        """The reference pattern: processor(text=...) + model(vision_embeds=...),
        and the backbone runs once however many prompts follow."""
        engine, _loads, models = _torch_engine(fake_torch, "float16")
        enc = engine.encode_image(image_data)
        processor = engine.mgr._loaded["pcs"].processor
        for request_id in ("r1", "r2"):
            engine.prompt_text(enc, TextPrompt(request_id=request_id, text="cat",
                                               score_threshold=0.02))
        model = models[0]
        assert model.vision_calls == 1
        for call in model.forward_calls:
            assert VISION_CACHE_KWARG in call and "pixel_values" not in call
        prompt_calls = processor.calls[1:]
        assert [c.get("images") is None for c in prompt_calls] == [True, True]
        assert [c.get("text") for c in prompt_calls] == ["cat", "cat"]
        assert engine.pcs._vision_cache_works is True



class TestTrackerForwardInvariant:
    """The tracker's forward rejects pixel_values together with
    image_embeddings, exactly as Sam3Model does for vision_embeds -- and the
    PVS refine path supplied both."""

    def test_helper_honours_the_trackers_keyword(self):
        import sam3gimpd.engines.base as B
        out = B.forward_kwargs_for({"pixel_values": "px", "original_sizes": "os",
                                    "input_points": "pts"},
                                   vision_embeds="emb", name="image_embeddings")
        assert out["image_embeddings"] == "emb"
        assert "pixel_values" not in out
        # the tracker's processor needs original_sizes to rescale coordinates
        assert out["original_sizes"] == "os"
        assert out["input_points"] == "pts"

    def test_refine_path_never_passes_pixels_with_the_embedding(self, fake_torch,
                                                                image_data):
        engine, _loads, models = _torch_engine(fake_torch, "float16")
        enc = _encoded(image_data)
        for points in ([Point(20.0, 20.0, 1)], [Point(20.0, 20.0, 1), Point(40.0, 30.0, 0)]):
            result = engine.prompt_points(enc, PointPrompt(request_id="r", points=points))
            assert result.instances
        tracker = models[0]
        assert tracker.embed_calls == 1
        for call in tracker.forward_calls:
            assert "image_embeddings" in call and "pixel_values" not in call
            assert "original_sizes" in call
        processor = engine.mgr._loaded["pvs"].processor
        # One encode from pixels, then refines from the recorded original_sizes.
        assert [c["images"] for c in processor.calls] == [True, False, False]
        assert [c["original_sizes"] for c in processor.calls[1:]] == [True, True]
        assert engine.pvs._embedding_reuse is True

    def test_a_refused_embedding_falls_back_to_pixels(self, fake_torch, image_data):
        engine, _loads, models = _torch_engine(fake_torch, "float16",
                                               state={"reject_embeddings": True})
        result = engine.prompt_points(_encoded(image_data),
                                      PointPrompt(request_id="r", points=[Point(20.0, 20.0, 1)]))
        assert result.instances
        calls = models[0].forward_calls
        assert "image_embeddings" in calls[0]
        assert "pixel_values" in calls[1] and "image_embeddings" not in calls[1]
        assert engine.pvs._embedding_reuse is False



class TestColdStartIsVisible:
    """A cold model load must be logged and reported, not silent."""

    def test_model_manager_logs_by_default(self):
        mgr = modelmgr.ModelManager(configure_hf=False)
        assert mgr._log is not None
        assert mgr._log.name == "sam3gimpd.modelmgr"

    def test_loading_model_is_a_known_stage(self):
        from sam3gimpd.engines.base import Stage
        assert Stage.LOADING_MODEL == "loading model"
        assert Stage.LOADING_MODEL in Stage.ALL

    def test_both_engines_report_it_before_a_cold_load(self, fake_torch, image_data,
                                                       make_rgb):
        engine, _loads, _models = _torch_engine(fake_torch, "float16")

        def stages(fn):
            seen = []
            fn(lambda p, s: seen.append(s))
            return seen

        first = stages(lambda cb: engine.encode_image(image_data, cb))
        assert first.index(Stage.LOADING_MODEL) < first.index(Stage.ENCODING)
        other = ImageData("b" * 32, IMAGE_W, IMAGE_H, make_rgb(IMAGE_W, IMAGE_H, seed=2))
        assert Stage.LOADING_MODEL not in stages(lambda cb: engine.encode_image(other, cb))

        enc = _encoded(image_data)
        click = PointPrompt(request_id="r", points=[Point(20.0, 20.0, 1)])
        first = stages(lambda cb: engine.prompt_points(enc, click, cb))
        assert first.index(Stage.LOADING_MODEL) < first.index(Stage.ENCODING)
        assert Stage.LOADING_MODEL not in stages(
            lambda cb: engine.prompt_points(_encoded(other), click, cb))


# --------------------------------------------------------------------------- #
# a load never blocks a report
# --------------------------------------------------------------------------- #
def _in_thread(fn):
    """Start ``fn`` on a daemon thread; returns ``(thread, box)`` where ``box``
    receives the result or the exception."""
    box = {}

    def _target():
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 -- handed to the test
            box["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    return thread, box


def _quickly(fn, limit=1.0):
    started = time.monotonic()
    value = fn()
    elapsed = time.monotonic() - started
    assert elapsed < limit, "took %.2fs" % elapsed
    return value


class TestLoadsNeverBlockReports:
    """A cold load takes about a minute on real hardware; ``/hello`` has a 3 s
    read timeout and the launcher kills a daemon that misses it."""

    def test_reports_answer_while_a_load_is_in_progress(self, fake_torch):
        gate = threading.Event()
        mgr, loads = _loading_manager("bfloat16", gate=gate)
        engine = TorchEngine(manager=mgr)
        thread, box = _in_thread(lambda: mgr.load("pcs"))
        try:
            deadline = time.monotonic() + 5.0
            while not loads and time.monotonic() < deadline:
                time.sleep(0.01)
            assert loads == [("pcs", "bfloat16")], "the load never started"
            assert _quickly(mgr.loaded) == []
            assert _quickly(lambda: mgr.is_loaded("pcs")) is False
            assert _quickly(mgr.describe)["models_loaded"] == []
            info = _quickly(engine.describe)
            assert info.models_loaded == [] and info.detail["dtype"] == "bfloat16"
            assert _quickly(lambda: mgr.unload("pcs")) == []
        finally:
            gate.set()
            thread.join(5.0)
        assert "error" not in box
        assert mgr.loaded() == ["pcs"]
        assert engine.describe().models_loaded == ["pcs"]

    def test_one_half_loads_while_the_other_is_loading(self):
        gate = threading.Event()
        mgr, loads = _loading_manager("bfloat16")
        real = mgr._load_locked

        def _load(half, dtype_name):
            if half == "pcs":
                assert gate.wait(10.0)
            return real(half, dtype_name)

        mgr._load_locked = _load
        thread, _box = _in_thread(lambda: mgr.load("pcs"))
        try:
            assert _quickly(lambda: mgr.load("pvs")) == ("pvs@bfloat16", "processor")
            assert mgr.loaded() == ["pvs"]
        finally:
            gate.set()
            thread.join(5.0)
        assert mgr.loaded() == ["pcs", "pvs"]

    def test_concurrent_loads_of_one_half_load_it_once(self):
        gate = threading.Event()
        mgr, loads = _loading_manager("float16", gate=gate)
        runs = [_in_thread(lambda: mgr.load("pvs")) for _ in range(3)]
        time.sleep(0.1)
        gate.set()
        for thread, _box in runs:
            thread.join(5.0)
        assert loads == [("pvs", "float16")]
        assert [box["result"] for _t, box in runs] == [("pvs@float16", "processor")] * 3

    def test_nothing_loads_after_close(self):
        """A job still running when the daemon shuts down must not pull the
        weights back into memory behind ``close()``."""
        mgr, loads = _loading_manager("float16")
        mgr.load("pcs")
        mgr.close()
        assert mgr.loaded() == []
        with pytest.raises(ApiError) as info:
            mgr.load("pvs")
        assert info.value.code == ErrorCode.SHUTTING_DOWN
        assert loads == [("pcs", "float16")]


# --------------------------------------------------------------------------- #
# the torch probe: on the worker, never at construction or on a report
# --------------------------------------------------------------------------- #
class TestLazyProbe:
    """Importing torch holds the GIL through its dlopen -- seconds on a cold
    start -- so it happens neither before the socket is bound nor on a request
    thread, but on the worker, when a prompt first needs it."""

    @pytest.fixture
    def probe_calls(self, monkeypatch):
        monkeypatch.setattr(modelmgr, "_PROBE", None)
        calls = []
        done = modelmgr.TorchProbe(available=True, version="9.9", cuda_available=True,
                                   cuda_capability=(8, 6), bf16_supported=True,
                                   transformers_available=True)

        def _probe(refresh=False):
            # Cached, as the real one is: ``calls`` counts actual probes.
            if modelmgr._PROBE is None or refresh:
                calls.append(threading.current_thread().name)
                modelmgr._PROBE = done
            return modelmgr._PROBE

        monkeypatch.setattr(modelmgr, "probe_torch", _probe)
        return calls

    def test_construction_probes_nothing(self, probe_calls):
        TorchEngine()
        PcsEngine()
        PvsEngine()
        assert probe_calls == []

    def test_describe_reports_what_it_can_without_probing(self, probe_calls, no_weights):
        info = TorchEngine().describe()
        assert probe_calls == []
        assert info.detail["probed"] is False and info.detail["usable"] is None
        assert (info.device, info.dtype) == ("auto", "auto")
        assert info.torch_available is modelmgr._installed("torch")
        assert info.weights_available is False
        assert set(info.capabilities) == {"pcs", "pvs", "exemplar_boxes"}

    def test_the_first_prompt_probes_and_describe_then_has_the_answer(
            self, probe_calls, no_weights, image_data):
        engine = TorchEngine()
        with pytest.raises(ApiError) as exc:
            engine.encode_image(image_data)
        assert exc.value.code == ErrorCode.ENGINE_UNAVAILABLE   # no weights
        assert len(probe_calls) == 1
        info = engine.describe()
        assert info.detail["probed"] is True
        assert (info.device, info.detail["dtype"]) == ("cuda", "bfloat16")
        assert info.detail["usable"] is False and info.dtype == "none"
        assert len(probe_calls) == 1

    def test_explicit_preferences_need_no_probe(self, probe_calls):
        mgr = modelmgr.ModelManager(device="cuda:1", dtype="fp16", configure_hf=False)
        assert (mgr.device, mgr.dtype) == ("cuda:1", "float16")
        mgr = modelmgr.ModelManager(device="cpu", configure_hf=False)
        assert (mgr.device, mgr.dtype) == ("cpu", "float32")
        info = mgr.describe()
        assert (info["device"], info["dtype"]) == ("cpu", "float32")
        assert probe_calls == []


# --------------------------------------------------------------------------- #
# the float32 fallback, through the engines
# --------------------------------------------------------------------------- #
class TestFp32FallbackThroughTheEngines:
    """NaN masks are found inside ``ModelManager.run``, the retry re-encodes
    with the float32 model, and nothing encoded in half precision survives."""

    CLICK = PointPrompt(request_id="r", points=[Point(20.0, 20.0, 1)])

    def test_nan_tracker_masks_retry_in_fp32_and_succeed(self, fake_torch, image_data):
        engine, loads, models = _torch_engine(fake_torch, "float16", state={"nan": True})
        enc = _encoded(image_data)
        result = engine.prompt_points(enc, self.CLICK)
        assert result.instances and result.instances[0].score > 0.5
        assert loads == [("pvs", "float16"), ("pvs", "float32")]
        assert engine.mgr.dtype == "float32" and engine.mgr.fp32_fallback is True
        assert enc.get("pvs")["dtype"] == "float32"
        fp32 = models[1]
        assert fp32.embed_calls == 1, "the retry must re-encode with the float32 model"
        assert fp32.forward_calls[0]["image_embeddings"].dtype is fake_torch.float32

    def test_an_image_cached_before_the_fallback_is_re_encoded(self, fake_torch, image_data,
                                                               make_rgb):
        state = {}
        engine, _loads, models = _torch_engine(fake_torch, "float16", state=state)
        early = _encoded(image_data)
        engine.prompt_points(early, self.CLICK)
        assert early.get("pvs")["dtype"] == "float16"
        half_bytes = early.bytes_estimate

        state["nan"] = True
        late = _encoded(ImageData("c" * 32, IMAGE_W, IMAGE_H,
                                  make_rgb(IMAGE_W, IMAGE_H, seed=3)))
        engine.prompt_points(late, self.CLICK)
        assert engine.mgr.dtype == "float32"

        result = engine.prompt_points(early, self.CLICK)
        assert result.instances
        assert early.get("pvs")["dtype"] == "float32"
        assert early.bytes_estimate == 2 * half_bytes, "replaced, not added to"

    def test_nan_pcs_masks_retry_in_fp32_instead_of_going_binary(self, fake_torch,
                                                                 image_data):
        engine, loads, models = _torch_engine(fake_torch, "bfloat16", state={"nan": True})
        enc = engine.encode_image(image_data)
        result = engine.prompt_text(enc, TextPrompt(request_id="r", text="cat",
                                                    score_threshold=0.02))
        assert result.instances
        assert loads == [("pcs", "bfloat16"), ("pcs", "float32")]
        assert engine.mgr.fp32_fallback is True
        assert engine.pcs._soft_masks is True
        assert models[1].vision_calls == 1
        assert enc.get("pcs")["dtype"] == "float32"
        values = set()
        for inst in result.instances:
            values.update(inst.mask)
        assert len({v for v in values if 0 < v < 255}) > 4

    def test_a_nan_score_is_a_numerical_failure_not_certainty(self, fake_torch, image_data):
        engine, loads, _models = _torch_engine(fake_torch, "float16", state={"nan_iou": True})
        result = engine.prompt_points(_encoded(image_data), self.CLICK)
        assert loads[-1] == ("pvs", "float32")
        assert all(math.isfinite(i.score) for i in result.instances)

    def test_a_nan_score_in_fp32_fails_the_prompt(self, fake_torch, image_data):
        engine, _loads, _models = _torch_engine(fake_torch, "float32",
                                                state={"nan_iou": "always"})
        with pytest.raises(ApiError) as exc:
            engine.prompt_points(_encoded(image_data), self.CLICK)
        assert exc.value.code == ErrorCode.INFERENCE_FAILED
        assert "non-finite" in exc.value.message


class TestScores:
    def test_clamp_maps_non_finite_to_zero(self):
        clamp = engine_base.clamp_score
        assert clamp(float("nan")) == 0.0
        assert clamp(float("inf")) == 0.0
        assert clamp(-0.5) == 0.0 and clamp(1.5) == 1.0 and clamp(0.25) == 0.25

    def test_require_finite_raises_the_retryable_failure(self):
        engine_base.require_finite([0.1, 0.9], "scores")
        with pytest.raises(modelmgr.NumericalFailure):
            engine_base.require_finite([0.1, float("nan")], "scores")


# --------------------------------------------------------------------------- #
# soft-mask selection mirrors the processor, dtype included
# --------------------------------------------------------------------------- #
class TestSoftMaskSelection:
    @staticmethod
    def _bf16(value):
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
        return struct.unpack("<f", struct.pack("<I", bits))[0]

    def test_the_threshold_is_compared_in_the_scores_dtype(self):
        at_bf16 = self._bf16(0.02)
        assert at_bf16 != 0.02
        assert engine_base.hf_keep_indices([at_bf16], 0.02) == [0]
        assert engine_base.hf_keep_indices([at_bf16], 0.02, "torch.bfloat16") == []
        at_fp16 = struct.unpack("<e", struct.pack("<e", 0.02))[0]
        assert engine_base.hf_keep_indices([at_fp16], 0.02, "float16") == []
        assert engine_base.hf_keep_indices([at_fp16 * 1.01], 0.02, "float16") == [0]

    @pytest.mark.parametrize("dtype,kept", [("bfloat16", 2), ("float32", 3)])
    def test_a_score_on_the_rounded_threshold_keeps_masks_soft(self, fake_torch, image_data,
                                                               dtype, kept):
        """In bfloat16 query 3 scores exactly bf16(0.02): the processor drops
        it, and a mirror comparing against 0.02 itself would keep it, disagree
        on the count and fall back to binarised masks."""
        engine, _loads, _models = _torch_engine(fake_torch, dtype)
        enc = engine.encode_image(image_data)
        result = engine.prompt_text(enc, TextPrompt(request_id="r", text="cat",
                                                    score_threshold=0.02))
        assert engine.pcs._soft_masks is True, "fell back to binarised masks"
        assert len(result.instances) == kept
        values = set()
        for inst in result.instances:
            values.update(inst.mask)
        assert len({v for v in values if 0 < v < 255}) > 4


# --------------------------------------------------------------------------- #
# exemplar boxes
# --------------------------------------------------------------------------- #
class TestExemplarBoxes:
    BOXES = [{"box": [2.0, 3.0, 30.0, 20.0], "label": 1},
             {"box": [40.0, 10.0, 60.0, 40.0], "label": 0}]

    def _prompt(self, engine, image_data):
        enc = engine.encode_image(image_data)
        return engine.prompt_text(enc, TextPrompt(request_id="r", text="cat",
                                                  score_threshold=0.02, boxes=self.BOXES))

    def test_boxes_and_their_labels_reach_the_processor(self, fake_torch, image_data):
        engine, _loads, models = _torch_engine(fake_torch, "float16")
        assert "exemplar_boxes" in engine.describe().capabilities
        result = self._prompt(engine, image_data)
        call = engine.mgr._loaded["pcs"].processor.calls[-1]
        assert call["input_boxes"] == [[[2.0, 3.0, 30.0, 20.0], [40.0, 10.0, 60.0, 40.0]]]
        assert call["input_boxes_labels"] == [[1, 0]]
        assert call["images"] is not None, "boxes go to the processor with the image"
        assert "pixel_values" in models[0].forward_calls[-1]
        assert "ignored" not in result.prompt
        assert "exemplar_boxes" in engine.describe().capabilities

    def test_without_labels_a_negative_box_is_reported_not_sent(self, fake_torch,
                                                                image_data):
        engine, _loads, _models = _torch_engine(fake_torch, "float16",
                                                pcs_processor=_NoLabelsSam3Processor)
        result = self._prompt(engine, image_data)
        call = engine.mgr._loaded["pcs"].processor.calls[-1]
        assert call["input_boxes"] == [[[2.0, 3.0, 30.0, 20.0]]]
        assert "input_boxes_labels" not in call
        assert result.prompt["ignored"] == ["boxes:label=0"]

    def test_a_processor_without_boxes_withdraws_the_capability(self, fake_torch,
                                                                image_data):
        engine, _loads, _models = _torch_engine(fake_torch, "float16",
                                                pcs_processor=_NoBoxesSam3Processor)
        result = self._prompt(engine, image_data)
        assert result.prompt["ignored"] == ["boxes"]
        assert "input_boxes" not in engine.mgr._loaded["pcs"].processor.calls[-1]
        assert "exemplar_boxes" not in engine.describe().capabilities


# --------------------------------------------------------------------------- #
# the engines' warnings reach the log
# --------------------------------------------------------------------------- #
class TestEngineLogging:
    def test_engines_log_without_an_injected_logger(self):
        engine = TorchEngine()
        assert engine.pcs._log.name == engine_base.ENGINE_LOGGER
        assert engine.pvs._log.name == engine_base.ENGINE_LOGGER
        assert PcsEngine()._log.name == engine_base.ENGINE_LOGGER

    def test_a_refused_vision_encoding_is_logged(self, fake_torch, image_data, caplog):
        engine, _loads, _models = _torch_engine(fake_torch, "float16",
                                                state={"reject_vision_embeds": True})
        enc = engine.encode_image(image_data)
        with caplog.at_level(logging.WARNING, logger=engine_base.ENGINE_LOGGER):
            result = engine.prompt_text(enc, TextPrompt(request_id="r", text="cat",
                                                        score_threshold=0.02))
        assert result.instances
        assert engine.pcs._vision_cache_works is False
        assert any("vision-feature reuse failed" in r.getMessage()
                   and r.name == engine_base.ENGINE_LOGGER for r in caplog.records)


# --------------------------------------------------------------------------- #
# EncodedImage's size bookkeeping
# --------------------------------------------------------------------------- #
class TestEncodedImageBytes:
    def test_replacing_or_dropping_a_part_adjusts_the_estimate(self, image_data):
        enc = _encoded(image_data)
        enc.put("pcs", {"v": 1}, 100)
        enc.put("pvs", {"v": 1}, 50)
        enc.put("pcs", {"v": 2}, 200)
        assert enc.bytes_estimate == 250
        enc.drop("pvs")
        assert enc.bytes_estimate == 200
        enc.drop()
        assert enc.bytes_estimate == 0 and enc.parts == {}
