"""PVS -- Promptable Visual Segmentation: ``Sam3TrackerModel`` + ``Sam3TrackerProcessor``.

Points and boxes in, **one** instance out (plus the model's alternative
candidates when ``multimask`` is on).  This is the click-to-segment workflow
users already expect from SAM 1/2, preserved so that adopting SAM 3 loses
nothing; PCS is the new capability, PVS is the one that must not regress.

The reference call sequence (``transformers`` v5)::

    model     = Sam3TrackerModel.from_pretrained("facebook/sam3", device_map="auto")
    processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    input_points = [[[[500, 375]]]]    # (batch, objects, points, coords)
    input_labels = [[[1]]]             # (batch, objects, point_labels)
    inputs  = processor(images=img, input_points=input_points,
                        input_labels=input_labels, return_tensors="pt")
    outputs = model(**inputs)
    masks   = processor.post_process_masks(outputs.pred_masks.cpu(),
                                           inputs["original_sizes"])[0]

and refinement reuses the embedding::

    model(**inputs, input_masks=mask_input, image_embeddings=outputs.image_embeddings)

**The nesting is the classic bug here**, so it is built exactly once, in
:func:`nest_points`, and unit-tested without torch: ``input_points`` is four
levels deep -- batch, objects, points, ``[x, y]`` -- and ``input_labels`` is
three -- batch, objects, labels.  Getting either wrong does not raise; it
silently segments the wrong thing.

Two deliberate departures from the snippet:

* **Masks are produced at model-canvas resolution, not image resolution.**
  ``post_process_masks`` resizes to ``original_sizes`` (the uploaded image),
  but ``API.md`` §5 says every mask geometry the daemon returns is in
  model-canvas pixels.  Resampling to the image and back would cost quality for
  nothing, so the raw logits are interpolated straight onto the canvas and
  quantised as ``round(255 * sigmoid(logit))``.
* **Point coordinates arrive in uploaded-image space** (``API.md`` §5) and are
  handed to the processor in that space, because that is the space the
  processor's own rescaling expects -- the same space the reference snippet's
  ``[[[[500, 375]]]]`` is in.  The daemon never asks a client to compute canvas
  coordinates.

The image embedding is cached in ``EncodedImage.parts["pvs"]`` and passed back
as ``image_embeddings=``, so a refine costs milliseconds rather than seconds --
and the tracker is only loaded on the first point prompt (``DESIGN.md`` §7), so
a text-only session never pays for it.  Each cached embedding records the dtype
it was computed in; one from before a float32 fallback is re-encoded rather
than fed to a float32 model.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..modelmgr import PVS, ModelManager, NumericalFailure
from ..types import ApiError, ErrorCode, Limits, PointPrompt, Size, TextPrompt
from .base import (
    DEFAULT_CANVAS_SIDE,
    ENGINE_LOGGER,
    BaseEngine,
    CanvasMask,
    EncodedImage,
    EngineInfo,
    ImageData,
    ProgressFn,
    ProgressReporter,
    PromptResult,
    Stage,
    accepts_kwarg,
    build_instances,
    call_with_supported_kwargs,
    clamp_score,
    forward_kwargs_for,
    inference_failed,
    logits_to_canvas_masks,
    move_inputs,
    require_finite,
    rgb_array,
)

__all__ = ["PvsEngine", "nest_points", "nest_box", "EMBEDDING_KWARGS"]

#: Candidate keyword names for handing a cached image embedding back to
#: ``Sam3TrackerModel.forward``.  Checked in order against the real signature.
EMBEDDING_KWARGS = ("image_embeddings", "image_embeds", "vision_outputs",
                    "vision_features")

#: Methods that might produce an image embedding without a prompt.
_EMBED_METHODS = ("get_image_embeddings", "get_image_features", "get_vision_features")


# --------------------------------------------------------------------------- #
# prompt nesting -- pure, torch-free, and unit-tested as such
# --------------------------------------------------------------------------- #
def nest_points(points: Sequence[Any]
                ) -> Tuple[Optional[List[Any]], Optional[List[Any]]]:
    """Build ``(input_points, input_labels)`` with SAM's exact nesting.

    ``input_points`` is ``(batch, objects, points, 2)`` -- four levels of list::

        [[[[x0, y0], [x1, y1]]]]

    ``input_labels`` is ``(batch, objects, points)`` -- three levels::

        [[[1, 0]]]

    One batch entry and one object: this API segments a single instance per
    request, and the client sends the **complete** point set every time
    (``API.md`` §6.4), so there is never more than one object to nest.

    Returns ``(None, None)`` for an empty point list, which is legal when a box
    is supplied instead.
    """
    coords: List[List[float]] = []
    labels: List[int] = []
    for p in points:
        coords.append([float(p.x), float(p.y)])
        labels.append(1 if int(p.label) == 1 else 0)
    if not coords:
        return None, None
    return [[coords]], [[labels]]


def nest_box(box: Optional[Sequence[float]]) -> Optional[List[Any]]:
    """Build ``input_boxes`` = ``(batch, objects, 4)``::

        [[[x0, y0, x1, y1]]]

    The box is normalised so ``x1 > x0`` and ``y1 > y0``; a degenerate box is
    rejected rather than quietly segmenting the whole image.
    """
    if not box:
        return None
    if len(box) != 4:
        raise ApiError(ErrorCode.BAD_REQUEST,
                       "box must have 4 values, got %d" % len(box), {})
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        raise ApiError(ErrorCode.BAD_REQUEST,
                       "box is degenerate: %r" % ([x0, y0, x1, y1],), {})
    return [[[x0, y0, x1, y1]]]


# --------------------------------------------------------------------------- #
# the engine
# --------------------------------------------------------------------------- #
class PvsEngine(BaseEngine):
    """The point/box half of SAM 3."""

    MODE = "torch"
    HALF = PVS

    def __init__(self, manager: Optional[ModelManager] = None,
                 canvas_side: int = DEFAULT_CANVAS_SIDE,
                 logger: Any = None,
                 **manager_kwargs: Any) -> None:
        self.mgr = manager if manager is not None else ModelManager(**manager_kwargs)
        self._canvas_side = int(canvas_side)
        self._log = logger if logger is not None else logging.getLogger(ENGINE_LOGGER)
        self._embedding_kwarg: Optional[str] = None
        self._embedding_reuse: Optional[bool] = None
        self._soft_masks: Optional[bool] = None

    # -- introspection ----------------------------------------------------- #
    @property
    def canvas_side(self) -> int:
        return self._canvas_side

    def capabilities(self) -> List[str]:
        return ["pvs"]

    def describe(self) -> EngineInfo:
        detail: Dict[str, Any] = self.mgr.describe()
        detail.update({
            "half": self.HALF,
            "embedding_cache_kwarg": self._embedding_kwarg,
            "embedding_cache_effective": self._embedding_reuse,
            "soft_masks": self._soft_masks,
        })
        return EngineInfo(
            mode=self.MODE,
            device=str(detail["device"]),
            dtype=str(detail["dtype"]) if detail.get("usable") is not False else "none",
            capabilities=self.capabilities(),
            torch_available=bool(detail.get("torch_available")),
            weights_available=bool(detail.get("weights_available")),
            # Nominal working resolution only.  The canvas that matters is
            # per-image and equals the upload; see default_canvas_for.
            model_canvas=Size(self._canvas_side, self._canvas_side),
            models_loaded=list(detail.get("models_loaded") or []),
            detail=detail,
        )

    def unload(self, which: Optional[str] = None) -> List[str]:
        return self.mgr.unload(which if which else self.HALF)

    def close(self) -> None:
        self.mgr.close()

    # -- encode ------------------------------------------------------------ #
    def encode_image(self, image: ImageData,
                     progress: Optional[ProgressFn] = None) -> EncodedImage:
        image.validate()
        rep = ProgressReporter(progress)
        canvas, transform = self.canvas_for(image.size)
        encoded = EncodedImage(
            image_id=image.image_id,
            image=image.size,
            canvas=canvas,
            transform=transform,
            pixels=image.pixels,
        )
        self.ensure_encoded(encoded, rep)
        return encoded

    def _current(self, encoded: EncodedImage) -> Optional[Dict[str, Any]]:
        """``parts["pvs"]`` if it was computed at the precision now in force."""
        part = encoded.get(self.HALF)
        if isinstance(part, dict) and part.get("dtype") == self.mgr.dtype:
            return part
        return None

    def ensure_encoded(self, encoded: EncodedImage,
                       rep: Optional[ProgressReporter] = None) -> Dict[str, Any]:
        """Fill ``parts["pvs"]`` if it is missing or stale, and return it.

        This is where the tracker is loaded for the first time -- called from
        :meth:`prompt_points`, never eagerly, which is the whole of
        ``DESIGN.md`` §7's "``Sam3TrackerModel`` is only loaded on the first
        click-refine".
        """
        part = self._current(encoded)
        if part is not None:
            return part
        rep = rep or ProgressReporter(None)
        self.mgr.ensure_available()
        if not self.mgr.is_loaded(self.HALF):
            rep.stage(Stage.LOADING_MODEL, 0.02)
        try:
            part = self.mgr.run(
                self.HALF, lambda model, processor: self._encode(encoded, model, processor, rep))
        except ApiError:
            raise
        except NumericalFailure as exc:
            raise inference_failed("tracker encoder produced non-finite values: %s" % exc,
                                   {"half": self.HALF})
        except Exception as exc:
            raise inference_failed("tracker image encoding failed: %s" % exc,
                                   {"half": self.HALF, "error": type(exc).__name__})
        rep.stage(Stage.ENCODING, 1.0)
        return part

    def _encode(self, encoded: EncodedImage, model: Any, processor: Any,
                rep: ProgressReporter) -> Dict[str, Any]:
        """Compute the tracker's image embedding and store it in ``encoded``,
        stamped with the dtype it was computed in.  Runs inside
        ``ModelManager.run``."""
        import torch  # noqa: PLC0415

        rep.stage(Stage.ENCODING, 0.1)
        arr = rgb_array(encoded.pixels, encoded.image.width, encoded.image.height)
        moved = move_inputs(processor(images=arr, return_tensors="pt"),
                            self.mgr.device, self._torch_dtype())
        rep.stage(Stage.ENCODING, 0.3)
        embeddings = None
        for name in _EMBED_METHODS:
            getter = getattr(model, name, None)
            if getter is None:
                continue
            with torch.inference_mode():
                embeddings = call_with_supported_kwargs(getter, moved)
            break
        if embeddings is None:
            # No standalone embedding entry point: fall back to a bare
            # forward pass and keep whatever it exposes.  Slower on the
            # first click, still correct, and still cached afterwards.
            with torch.inference_mode():
                outputs = model(**moved)
            embeddings = getattr(outputs, "image_embeddings", None)
        rep.stage(Stage.ENCODING, 0.95)
        part = {"embeddings": embeddings,
                "original_sizes": moved.get("original_sizes"),
                "dtype": self.mgr.dtype}
        encoded.put(self.HALF, part, _tensor_bytes(embeddings))
        return part

    def _encoding_for(self, encoded: EncodedImage, model: Any,
                      processor: Any) -> Dict[str, Any]:
        """The cached embedding, re-computed with ``model`` if it is stale --
        which is what the float32 retry of a prompt finds, and what every
        image cached before a fallback holds."""
        part = self._current(encoded)
        if part is None:
            part = self._encode(encoded, model, processor, ProgressReporter(None))
        return part

    # -- prompt ------------------------------------------------------------ #
    def prompt_text(self, encoded: EncodedImage, prompt: TextPrompt,
                    progress: Optional[ProgressFn] = None) -> PromptResult:
        raise ApiError(ErrorCode.ENGINE_UNAVAILABLE,
                       "this engine serves point and box prompts only; text prompts "
                       "need the PCS half (Sam3Model)",
                       {"half": self.HALF})

    def prompt_points(self, encoded: EncodedImage, prompt: PointPrompt,
                      progress: Optional[ProgressFn] = None) -> PromptResult:
        started = time.time()
        rep = ProgressReporter(progress)
        if not prompt.points and not prompt.box:
            raise ApiError(ErrorCode.BAD_REQUEST,
                           "a point prompt needs at least one point or a box", {})
        if len(prompt.points) > Limits.MAX_POINTS:
            raise ApiError(ErrorCode.BAD_REQUEST,
                           "%d points, limit is %d"
                           % (len(prompt.points), Limits.MAX_POINTS), {})

        self.mgr.ensure_available()
        self.ensure_encoded(encoded, rep)
        rep.stage(Stage.PROMPTING, 0.0)

        import torch  # noqa: PLC0415

        arr = rgb_array(encoded.pixels, encoded.image.width, encoded.image.height)
        canvas = encoded.canvas
        # Coordinates stay in uploaded-image space: that is what the processor
        # rescales from, and what API.md §5 says the client sends.
        input_points, input_labels = nest_points(prompt.points)
        input_boxes = nest_box(prompt.box)
        want = max(1, min(int(prompt.max_instances), 8))
        multimask = bool(prompt.multimask) and want > 1

        def _prompt_kwargs() -> Dict[str, Any]:
            kw: Dict[str, Any] = {"return_tensors": "pt"}
            if input_points is not None:
                kw["input_points"] = input_points
                kw["input_labels"] = input_labels
            if input_boxes is not None:
                kw["input_boxes"] = input_boxes
            return kw

        def _pixel_inputs(processor: Any) -> Dict[str, Any]:
            kw = _prompt_kwargs()
            kw["images"] = arr
            return move_inputs(processor(**kw), self.mgr.device, self._torch_dtype())

        def _cached_inputs(processor: Any, part: Dict[str, Any]) -> Dict[str, Any]:
            # The reference refine call: prompts plus the *recorded*
            # original_sizes, and no image -- the processor still needs the
            # size to rescale point coordinates, but must not produce
            # pixel_values, which the forward pass would reject alongside the
            # cached embedding.
            kw = _prompt_kwargs()
            sizes = part.get("original_sizes")
            if sizes is not None:
                kw["original_sizes"] = sizes.cpu() if hasattr(sizes, "cpu") else sizes
            moved = move_inputs(processor(**kw), self.mgr.device, self._torch_dtype())
            if sizes is not None:
                moved.setdefault("original_sizes", sizes)
            return moved

        def _with_multimask(kw: Dict[str, Any], model: Any) -> Dict[str, Any]:
            if accepts_kwarg(getattr(model, "forward", model), "multimask_output"):
                kw["multimask_output"] = multimask
            return kw

        def _run(model: Any, processor: Any) -> List[CanvasMask]:
            part = self._encoding_for(encoded, model, processor)
            kwarg = self._embedding_kwarg_for(model)
            embeddings = part.get("embeddings")
            use_cache = (kwarg is not None and embeddings is not None
                         and self._embedding_reuse is not False)
            if use_cache:
                moved = _cached_inputs(processor, part)
                forward_kwargs = forward_kwargs_for(moved, vision_embeds=embeddings, name=kwarg)
            else:
                moved = _pixel_inputs(processor)
                forward_kwargs = forward_kwargs_for(moved)
            rep.stage(Stage.PROMPTING, 0.5)
            with torch.inference_mode():
                try:
                    outputs = model(**_with_multimask(forward_kwargs, model))
                    if use_cache:
                        self._embedding_reuse = True
                except (TypeError, ValueError) as exc:
                    if not use_cache:
                        raise
                    # The cached embedding was not accepted this way; re-encode
                    # from pixels for this prompt and stop trying to reuse it.
                    self._warn("embedding reuse failed (%s); re-encoding", exc)
                    self._embedding_kwarg = None
                    self._embedding_reuse = False
                    moved = _pixel_inputs(processor)
                    outputs = model(**_with_multimask(forward_kwargs_for(moved), model))
            rep.stage(Stage.DECODING, 0.0)
            # Inside run(), so that NaN masks or scores get the float32 retry.
            candidates = self._candidates(outputs, canvas)
            rep.stage(Stage.DECODING, 0.5)
            return candidates

        try:
            canvas_masks = self.mgr.run(self.HALF, _run)
        except ApiError:
            raise
        except NumericalFailure as exc:
            raise inference_failed("PVS forward pass produced non-finite values: %s" % exc,
                                   {"half": self.HALF})
        except Exception as exc:
            raise inference_failed("PVS inference failed: %s" % exc,
                                   {"half": self.HALF, "error": type(exc).__name__})

        rep.stage(Stage.PACKING, 0.0)
        instances, truncated = build_instances(canvas, canvas_masks, max_instances=want)
        rep.stage(Stage.PACKING, 1.0)
        self.mgr.touch()

        return PromptResult(
            engine="pvs",
            instances=instances,
            image=encoded.image,
            canvas=canvas,
            transform=encoded.transform,
            prompt=self.point_prompt_dict(prompt),
            truncated=truncated,
            elapsed_ms=(time.time() - started) * 1000.0,
        )

    # -- result extraction ------------------------------------------------- #
    def _candidates(self, outputs: Any, canvas: Size) -> List[CanvasMask]:
        """``pred_masks`` + ``iou_scores`` -> soft canvas masks, best first.

        ``pred_masks`` is ``(batch, objects, candidates, h, w)`` in the SAM
        layout; with one batch entry and one object that flattens to
        ``(candidates, h, w)``.  The logits are interpolated onto the canvas
        *before* the sigmoid, which is what keeps the soft edge that ``API.md``
        §8.3 is built around.

        Raises :class:`NumericalFailure` on a non-finite mask, IoU score or
        object score, so it must run inside ``ModelManager.run``.
        """
        import torch  # noqa: PLC0415

        logits = getattr(outputs, "pred_masks", None)
        if logits is None and isinstance(outputs, dict):
            logits = outputs.get("pred_masks")
        if logits is None:
            raise inference_failed("tracker output has no pred_masks",
                                   {"half": self.HALF})
        logits = logits.detach()
        while logits.dim() > 3 and int(logits.shape[0]) == 1:
            logits = logits[0]
        if logits.dim() > 3:
            logits = logits.reshape(-1, int(logits.shape[-2]), int(logits.shape[-1]))
        if logits.dim() == 2:
            logits = logits[None]

        scores = getattr(outputs, "iou_scores", None)
        if scores is None and isinstance(outputs, dict):
            scores = outputs.get("iou_scores")
        if scores is not None:
            flat = torch.as_tensor(scores).reshape(-1).float().tolist()
            require_finite(flat, "PVS IoU scores")
        else:
            # No IoU head output: keep the model's own ordering and fabricate a
            # descending ladder so the contract's "sorted by score" still holds.
            flat = [max(0.1, 0.95 - 0.15 * i) for i in range(int(logits.shape[0]))]

        # The tracker also says whether it believes an object is under the
        # prompt at all: ``object_score_logits`` (batch, point_batch, 1), a
        # logit that goes negative on a click on empty background.  The IoU
        # score alone stays confident there, so the two are combined the way
        # the text engine combines instance and presence: score = iou *
        # sigmoid(object).  One value per prompt, applied to every candidate.
        presence = getattr(outputs, "object_score_logits", None)
        if presence is None and isinstance(outputs, dict):
            presence = outputs.get("object_score_logits")
        if presence is not None:
            try:
                p: Optional[float] = float(
                    torch.as_tensor(presence).reshape(-1)[0].float().sigmoid())
            except Exception:  # noqa: BLE001 -- an odd shape must not fail the prompt
                p = None
            if p is not None:
                require_finite([p], "PVS object scores")
                flat = [s * p for s in flat]

        soft = logits_to_canvas_masks(logits, canvas)
        self._soft_masks = True
        out: List[CanvasMask] = []
        for i, mask in enumerate(soft):
            score = float(flat[i]) if i < len(flat) else 0.1
            out.append(CanvasMask(score=clamp_score(score), mask=mask, label=""))
        return out

    # -- plumbing ---------------------------------------------------------- #
    def _embedding_kwarg_for(self, model: Any) -> Optional[str]:
        if self._embedding_kwarg is not None:
            return self._embedding_kwarg
        if self._embedding_reuse is False:
            return None
        forward = getattr(model, "forward", model)
        for name in EMBEDDING_KWARGS:
            if accepts_kwarg(forward, name):
                self._embedding_kwarg = name
                return name
        self._embedding_reuse = False
        self._warn("Sam3TrackerModel.forward accepts none of %s; every refine will "
                   "re-encode the image", ", ".join(EMBEDDING_KWARGS))
        return None

    def _torch_dtype(self) -> Any:
        from ..modelmgr import resolve_torch_dtype  # noqa: PLC0415

        return resolve_torch_dtype(self.mgr.dtype)

    def _warn(self, msg: str, *args: Any) -> None:
        if self._log is not None:
            try:
                self._log.warning(msg, *args)
            except Exception:
                pass


def _tensor_bytes(obj: Any) -> int:
    """Best-effort resident size of a cached embedding, for ``/status``."""
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return 0
    total = 0
    stack: List[Any] = [obj]
    seen = 0
    while stack and seen < 256:
        item = stack.pop()
        seen += 1
        if isinstance(item, torch.Tensor):
            total += int(item.numel()) * int(item.element_size())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, dict):
            stack.extend(item.values())
    return total
