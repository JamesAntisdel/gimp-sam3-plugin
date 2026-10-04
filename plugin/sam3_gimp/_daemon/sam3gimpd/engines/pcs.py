"""PCS -- Promptable Concept Segmentation: ``Sam3Model`` + ``Sam3Processor``.

Text in, **every** matching instance out.  This is the capability SAM 1 and 2
structurally did not have and the reason this project exists: "select every
car" is one prompt returning N masks, not N clicks.

The reference call sequence (``transformers`` v5)::

    model     = Sam3Model.from_pretrained("facebook/sam3")
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    inputs    = processor(images=image, text="ear", return_tensors="pt")
    outputs   = model(**inputs)
    results   = processor.post_process_instance_segmentation(
                    outputs, threshold=0.5, mask_threshold=0.5,
                    target_sizes=[(h, w)])
    # results[0] has "masks", "boxes", "scores"

Two departures from that snippet, both deliberate:

**1. The vision features are cached.**  ``Sam3Model.get_vision_features()`` runs
the 0.9B backbone once per image; every later text prompt is then cheap.  That
single fact is the entire performance argument for running a persistent daemon
(``DESIGN.md`` §1), so the caching path here is explicit rather than incidental:
:meth:`PcsEngine.encode_image` fills ``EncodedImage.parts["pcs"]`` and
:meth:`PcsEngine.prompt_text` hands it back to the forward pass as
``vision_embeds``, with the processor called on the text alone -- ``forward``
rejects ``pixel_values`` and ``vision_embeds`` together.  If the model will not
take the encoding, the prompt degrades to a full (correct, just slower)
forward pass from pixels and ``EngineInfo.detail`` says so, so a regression
shows up in ``/status`` instead of only in the latency.  Each cached encoding
records the dtype it was computed in; one from before a float32 fallback is
re-encoded rather than fed to a float32 model.

**2. The score threshold is low and the masks come back soft.**  ``API.md``
§6.3 returns every candidate above ``score_threshold`` so the client's score
slider filters locally, and §8.3 wants ``round(255 * sigmoid(logit))`` rather
than a binarised mask so the threshold slider does too.
``post_process_instance_segmentation`` binarises at ``mask_threshold``; this
module takes the same rows from the raw logits instead (see
:meth:`PcsEngine._soft_mask_tensors`), and otherwise reports
``soft_masks: false``.

Nothing here is imported unless a real inference is actually requested: torch,
transformers and numpy all load inside methods.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from ..modelmgr import PCS, ModelManager, NumericalFailure, finished_probe
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
    hf_keep_indices,
    inference_failed,
    logits_to_canvas_masks,
    masks_to_canvas_masks,
    move_inputs,
    require_finite,
    rgb_array,
)

__all__ = ["PcsEngine", "VISION_CACHE_KWARG"]

#: The keyword ``Sam3Model.forward`` takes a cached vision encoding under, from
#: the reference example::
#:
#:     vision_embeds = model.get_vision_features(pixel_values=...)
#:     outputs = model(vision_embeds=vision_embeds, **text_inputs)
VISION_CACHE_KWARG = "vision_embeds"

#: Attributes on a model output that might hold raw (soft) mask logits.
_MASK_LOGIT_ATTRS = ("pred_masks", "mask_logits", "pred_masks_logits", "masks_logits")


class PcsEngine(BaseEngine):
    """The text half of SAM 3.

    Usable on its own -- ``encode_image`` + ``prompt_text`` are a complete
    engine for a text-only client -- and composed with :class:`~sam3gimpd.engines.
    pvs.PvsEngine` by ``engines.TorchEngine`` for the full API.
    """

    MODE = "torch"
    HALF = PCS

    def __init__(self, manager: Optional[ModelManager] = None,
                 canvas_side: int = DEFAULT_CANVAS_SIDE,
                 logger: Any = None,
                 **manager_kwargs: Any) -> None:
        self.mgr = manager if manager is not None else ModelManager(**manager_kwargs)
        self._canvas_side = int(canvas_side)
        self._log = logger if logger is not None else logging.getLogger(ENGINE_LOGGER)
        self._soft_masks: Optional[bool] = None
        self._vision_cache_works: Optional[bool] = None
        #: Does the processor take ``input_boxes`` / ``input_boxes_labels``?
        #: ``None`` until the processor class or instance has been inspected.
        self._exemplar_boxes: Optional[bool] = None
        self._exemplar_box_labels: Optional[bool] = None

    # -- introspection ----------------------------------------------------- #
    @property
    def canvas_side(self) -> int:
        return self._canvas_side

    def capabilities(self) -> List[str]:
        """``pcs``, plus ``exemplar_boxes`` unless the processor is known not
        to take boxes.

        Unknown counts as supported: the ``Sam3Processor`` that ships with the
        model takes ``input_boxes`` (the reference box-prompt example), and a
        prompt whose boxes turn out not to be accepted still reports them in
        ``prompt.ignored``.
        """
        caps = ["pcs"]
        if self._exemplar_boxes is None:
            self._learn_box_support_from_class()
        if self._exemplar_boxes is not False:
            caps.append("exemplar_boxes")
        return caps

    def describe(self) -> EngineInfo:
        detail: Dict[str, Any] = self.mgr.describe()
        detail.update({
            "half": self.HALF,
            "soft_masks": self._soft_masks,
            "vision_cache_kwarg": VISION_CACHE_KWARG,
            "vision_cache_effective": self._vision_cache_works,
            "exemplar_boxes": self._exemplar_boxes,
            "exemplar_box_labels": self._exemplar_box_labels,
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

    # -- what the processor takes ------------------------------------------ #
    def _learn_processor(self, processor: Any) -> None:
        """Adopt what a loaded processor says about itself.

        Its working resolution becomes the nominal ``model_canvas`` of
        ``/hello`` -- geometry never depends on it, since masks are
        post-processed straight to the uploaded image -- and its signature
        settles whether exemplar boxes, and their labels, can be passed.
        """
        self._learn_box_support(processor)
        size = getattr(getattr(processor, "image_processor", None), "size", None)
        if not isinstance(size, dict):
            return
        side = size.get("longest_edge") or size.get("height") or size.get("shortest_edge")
        try:
            side = int(side)
        except (TypeError, ValueError):
            return
        if side and side != self._canvas_side:
            self._warn("processor reports a %dpx working resolution; adopting it (was %d)",
                       side, self._canvas_side)
            self._canvas_side = side

    def _learn_box_support(self, call: Any) -> None:
        self._exemplar_boxes = accepts_kwarg(call, "input_boxes")
        # Labels must be taken by name: a processor that swallowed them in
        # **kwargs would read a negative box as a positive one.
        self._exemplar_box_labels = accepts_kwarg(call, "input_boxes_labels", explicit=True)

    def _learn_box_support_from_class(self) -> None:
        """Inspect ``Sam3Processor.__call__`` once the probe has imported
        transformers -- by then this costs nothing and loads nothing."""
        probe = finished_probe()
        if probe is None or not probe.sam3_classes_available:
            return
        try:
            import transformers  # noqa: PLC0415  (already imported by the probe)

            self._learn_box_support(transformers.Sam3Processor.__call__)
        except Exception:  # noqa: BLE001 -- stays unknown
            pass

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
        """``parts["pcs"]`` if it was computed at the precision now in force."""
        part = encoded.get(self.HALF)
        if isinstance(part, dict) and part.get("dtype") == self.mgr.dtype:
            return part
        return None

    def ensure_encoded(self, encoded: EncodedImage,
                       rep: Optional[ProgressReporter] = None) -> Dict[str, Any]:
        """Fill ``parts["pcs"]`` if it is missing or stale, and return it."""
        part = self._current(encoded)
        if part is not None:
            return part
        rep = rep or ProgressReporter(None)
        # Raise a clean `engine_unavailable` before the first torch import, so a
        # daemon started without --stub on a machine with no torch answers with
        # the documented envelope instead of a ModuleNotFoundError traceback.
        self.mgr.ensure_available()
        if not self.mgr.is_loaded(self.HALF):
            rep.stage(Stage.LOADING_MODEL, 0.02)
        try:
            part = self.mgr.run(
                self.HALF, lambda model, processor: self._encode(encoded, model, processor, rep))
        except ApiError:
            raise
        except NumericalFailure as exc:
            raise inference_failed("vision encoder produced non-finite values: %s" % exc,
                                   {"half": self.HALF})
        except Exception as exc:
            raise inference_failed("vision encoding failed: %s" % exc,
                                   {"half": self.HALF, "error": type(exc).__name__})
        rep.stage(Stage.ENCODING, 1.0)
        return part

    def _encode(self, encoded: EncodedImage, model: Any, processor: Any,
                rep: ProgressReporter) -> Dict[str, Any]:
        """Run the vision backbone and store the result in ``encoded``,
        stamped with the dtype it was computed in.  Runs inside
        ``ModelManager.run``."""
        import torch  # noqa: PLC0415

        self._learn_processor(processor)
        rep.stage(Stage.ENCODING, 0.1)
        arr = rgb_array(encoded.pixels, encoded.image.width, encoded.image.height)
        moved = move_inputs(processor(images=arr, return_tensors="pt"),
                            self.mgr.device, self._torch_dtype())
        rep.stage(Stage.ENCODING, 0.25)
        getter = getattr(model, "get_vision_features", None)
        if getter is None:
            raise inference_failed(
                "this Sam3Model has no get_vision_features(); embedding reuse -- "
                "the whole reason the daemon is persistent -- is unavailable",
                {"half": self.HALF})
        with torch.inference_mode():
            features = call_with_supported_kwargs(getter, moved)
        rep.stage(Stage.ENCODING, 0.95)
        part = {"vision": features, "dtype": self.mgr.dtype}
        encoded.put(self.HALF, part, _tensor_bytes(features))
        return part

    def _encoding_for(self, encoded: EncodedImage, model: Any,
                      processor: Any) -> Dict[str, Any]:
        """The cached encoding, re-computed with ``model`` if it is stale --
        which is what the float32 retry of a prompt finds, and what every
        image cached before a fallback holds."""
        part = self._current(encoded)
        if part is None:
            part = self._encode(encoded, model, processor, ProgressReporter(None))
        return part

    # -- prompt ------------------------------------------------------------ #
    def prompt_text(self, encoded: EncodedImage, prompt: TextPrompt,
                    progress: Optional[ProgressFn] = None) -> PromptResult:
        started = time.time()
        rep = ProgressReporter(progress)
        text = (prompt.text or "").strip()
        if not text:
            raise ApiError(ErrorCode.BAD_REQUEST, "text prompt is empty", {})
        if len(text) > Limits.MAX_TEXT_CHARS:
            raise ApiError(ErrorCode.BAD_REQUEST,
                           "text prompt is %d characters, limit is %d"
                           % (len(text), Limits.MAX_TEXT_CHARS), {})
        if prompt.boxes and len(prompt.boxes) > Limits.MAX_BOXES:
            raise ApiError(ErrorCode.BAD_REQUEST,
                           "%d exemplar boxes, limit is %d"
                           % (len(prompt.boxes), Limits.MAX_BOXES), {})

        self.mgr.ensure_available()
        self.ensure_encoded(encoded, rep)
        rep.stage(Stage.PROMPTING, 0.0)

        import torch  # noqa: PLC0415

        arr = rgb_array(encoded.pixels, encoded.image.width, encoded.image.height)
        canvas = encoded.canvas
        threshold = float(prompt.score_threshold)

        def _pixel_inputs(processor: Any, boxes: Optional[List[Any]],
                          labels: Optional[List[Any]]) -> Dict[str, Any]:
            proc_kwargs: Dict[str, Any] = {"images": arr, "text": text,
                                           "return_tensors": "pt"}
            if boxes is not None:
                proc_kwargs["input_boxes"] = boxes
                if labels is not None:
                    proc_kwargs["input_boxes_labels"] = labels
            return move_inputs(processor(**proc_kwargs), self.mgr.device,
                               self._torch_dtype())

        def _run(model: Any, processor: Any
                 ) -> Tuple[List[CanvasMask], Optional[List[str]]]:
            self._learn_processor(processor)
            boxes, labels, ignored = self._exemplar_payload(prompt)
            vision = self._encoding_for(encoded, model, processor).get("vision")
            # Reuse the encoding only when it can be the *sole* vision source.
            # Exemplar boxes are handed to the processor together with the
            # image in the reference example, so that path re-encodes.
            use_cache = (vision is not None and boxes is None
                         and self._vision_cache_works is not False)
            if use_cache:
                text_inputs = move_inputs(processor(text=text, return_tensors="pt"),
                                          self.mgr.device, self._torch_dtype())
                forward_kwargs = forward_kwargs_for(text_inputs, vision_embeds=vision,
                                                    name=VISION_CACHE_KWARG)
            else:
                forward_kwargs = forward_kwargs_for(_pixel_inputs(processor, boxes, labels))
            rep.stage(Stage.PROMPTING, 0.5)
            with torch.inference_mode():
                try:
                    outputs = model(**forward_kwargs)
                    if use_cache:
                        self._vision_cache_works = True
                except (TypeError, ValueError) as exc:
                    if not use_cache:
                        raise
                    # The cached encoding was not accepted; re-encode from
                    # pixels for this prompt and stop trying to reuse it.
                    self._warn("vision-feature reuse failed (%s); re-encoding", exc)
                    self._vision_cache_works = False
                    outputs = model(**forward_kwargs_for(_pixel_inputs(processor, boxes,
                                                                       labels)))
            rep.stage(Stage.DECODING, 0.0)
            results = call_with_supported_kwargs(
                processor.post_process_instance_segmentation,
                {"threshold": threshold,
                 "mask_threshold": 0.5,
                 "target_sizes": [(int(canvas.height), int(canvas.width))]},
                outputs)
            rep.stage(Stage.DECODING, 0.5)
            # Inside run(), so that NaN masks or scores get the float32 retry.
            return (self._instances_from_results(outputs, results, canvas, text, threshold),
                    ignored)

        try:
            canvas_masks, ignored = self.mgr.run(self.HALF, _run)
        except ApiError:
            raise
        except NumericalFailure as exc:
            raise inference_failed("PCS forward pass produced non-finite values: %s" % exc,
                                   {"half": self.HALF})
        except Exception as exc:
            raise inference_failed("PCS inference failed: %s" % exc,
                                   {"half": self.HALF, "error": type(exc).__name__})

        rep.stage(Stage.PACKING, 0.0)
        instances, truncated = build_instances(
            canvas, canvas_masks,
            max_instances=min(int(prompt.max_instances), Limits.MAX_INSTANCES),
            score_threshold=threshold)
        rep.stage(Stage.PACKING, 1.0)
        self.mgr.touch()

        return PromptResult(
            engine="pcs",
            instances=instances,
            image=encoded.image,
            canvas=canvas,
            transform=encoded.transform,
            prompt=self.text_prompt_dict(prompt, ignored),
            truncated=truncated,
            elapsed_ms=(time.time() - started) * 1000.0,
        )

    def prompt_points(self, encoded: EncodedImage, prompt: PointPrompt,
                      progress: Optional[ProgressFn] = None) -> PromptResult:
        raise ApiError(ErrorCode.ENGINE_UNAVAILABLE,
                       "this engine serves text prompts only; point prompts need the "
                       "PVS half (Sam3TrackerModel)",
                       {"half": self.HALF})

    # -- result extraction ------------------------------------------------- #
    def _instances_from_results(self, outputs: Any, results: Any, canvas: Size,
                                label: str, threshold: float) -> List[CanvasMask]:
        """``post_process_instance_segmentation`` output -> soft canvas masks.

        Raises :class:`NumericalFailure` on a non-finite score or mask, so it
        must run inside ``ModelManager.run``.
        """
        import torch  # noqa: PLC0415

        first = results[0] if isinstance(results, (list, tuple)) and results else results
        if not isinstance(first, dict):
            first = {k: getattr(first, k) for k in ("masks", "boxes", "scores")
                     if hasattr(first, k)}
        masks = first.get("masks")
        scores = first.get("scores")
        if masks is None or scores is None:
            raise inference_failed(
                "post_process_instance_segmentation returned no masks/scores",
                {"keys": sorted(first.keys())})

        score_list = [float(s) for s in torch.as_tensor(scores).reshape(-1).tolist()]
        require_finite(score_list, "PCS scores")
        if len(score_list) == 0:
            return []

        soft, was_soft = self._soft_mask_tensors(outputs, first, masks, canvas, threshold)
        if self._soft_masks is None or self._soft_masks != was_soft:
            if not was_soft:
                self._warn("post-processing returned binarised masks; edge quality "
                           "will be lower than the format allows")
            self._soft_masks = was_soft

        return [CanvasMask(score=clamp_score(score), mask=soft[i], label=label)
                for i, score in enumerate(score_list) if i < len(soft)]

    def _soft_mask_tensors(self, outputs: Any, result: Dict[str, Any], masks: Any,
                           canvas: Size, threshold: float) -> Tuple[List[Any], bool]:
        """Soft masks aligned with the instances the processor kept.

        ``post_process_instance_segmentation`` returns ``masks`` already
        binarised (``(masks > mask_threshold).to(torch.long)``) and gives no
        index back into ``pred_masks``.  Sending those would make every mask on
        the wire 0/255, the client's mask-threshold slider a no-op, and cut
        away anything the model scored below 0.5 *within* an object -- a
        weakly-detected guitar would come back as its most confident fragment.

        The processor's selection is simple and mirrored here exactly --
        ``sigmoid(pred_logits) * sigmoid(presence)``, ``keep = scores >
        threshold`` compared in the scores' own dtype, query order preserved --
        so the same rows of the raw logits can be taken, sigmoided and
        resampled instead.  If the counts still disagree the binarised masks
        are used and the disagreement is logged, never a misaligned soft mask.
        A non-finite score or logit is not a disagreement but a numerical
        failure, and propagates as one.
        """
        import torch  # noqa: PLC0415

        tensor = masks if isinstance(masks, torch.Tensor) else torch.as_tensor(masks)
        if tensor.is_floating_point():
            return masks_to_canvas_masks(tensor, canvas)
        n_kept = int(tensor.shape[0])

        logits = None
        for attr in _MASK_LOGIT_ATTRS:
            candidate = getattr(outputs, attr, None)
            if candidate is None and isinstance(outputs, dict):
                candidate = outputs.get(attr)
            if candidate is not None:
                logits = candidate
                break
        pred_logits = getattr(outputs, "pred_logits", None)
        presence = getattr(outputs, "presence_logits", None)

        if logits is not None and pred_logits is not None:
            try:
                # Same ops, same dtype, same order as the processor.
                scores = pred_logits.detach().sigmoid()
                if presence is not None:
                    scores = scores * presence.detach().sigmoid()
                values = scores.reshape(-1).float().cpu().tolist()
                require_finite(values, "PCS query scores")
                keep = hf_keep_indices(values, threshold, str(scores.dtype))
                flat = logits.reshape(-1, logits.shape[-2], logits.shape[-1])
                if len(keep) == n_kept and n_kept > 0:
                    idx = torch.as_tensor(keep, device=flat.device, dtype=torch.long)
                    # These are logits by construction, so say so: the generic
                    # helper guesses "probabilities" from a [0, 1] value range,
                    # and a small mask whose logits all happen to sit in that
                    # range would skip the sigmoid and come out uniformly grey.
                    return logits_to_canvas_masks(flat[idx], canvas), True
                self._warn("soft-mask selection kept %d rows but the processor "
                           "kept %d; using binarised masks", len(keep), n_kept)
            except NumericalFailure:
                raise
            except Exception as exc:  # noqa: BLE001 -- never trade a result for softness
                self._warn("soft-mask recovery failed (%s); using binarised masks", exc)
        return masks_to_canvas_masks(tensor, canvas)

    # -- plumbing ---------------------------------------------------------- #
    def _exemplar_payload(self, prompt: TextPrompt
                          ) -> Tuple[Optional[List[Any]], Optional[List[Any]],
                                     Optional[List[str]]]:
        """``(input_boxes, input_boxes_labels, ignored)`` for ``prompt.boxes``.

        ``API.md`` §6.3: boxes are in **uploaded-image** coordinates -- the
        pixels the processor is given -- and whatever the processor cannot
        take is reported in ``prompt.ignored`` rather than silently dropped.
        A negative box is sent only with its label; without one the model
        would read it as positive and select exactly what the user excluded.
        """
        if not prompt.boxes:
            return None, None, None
        if self._exemplar_boxes is False:
            return None, None, ["boxes"]
        with_labels = bool(self._exemplar_box_labels)
        boxes: List[List[float]] = []
        labels: List[int] = []
        negatives_dropped = False
        for item in prompt.boxes:
            box = item.get("box")
            if not box or len(box) != 4:
                continue
            label = 1 if int(item.get("label", 1)) == 1 else 0
            if label == 0 and not with_labels:
                negatives_dropped = True
                continue
            boxes.append([float(v) for v in box])
            labels.append(label)
        if not boxes:
            return None, None, ["boxes"]
        ignored = ["boxes:label=0"] if negatives_dropped else None
        return [boxes], ([labels] if with_labels else None), ignored

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
        elif hasattr(item, "to_tuple"):
            try:
                stack.extend(item.to_tuple())
            except Exception:
                pass
    return total
