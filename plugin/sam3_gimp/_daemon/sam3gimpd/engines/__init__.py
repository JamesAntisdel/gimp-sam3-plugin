"""Engines: the only part of ``sam3gimpd`` that knows how a mask is made.

The server holds exactly one :class:`~sam3gimpd.engines.base.BaseEngine` and never
asks what is behind it:

``StubEngine``
    Deterministic synthetic blobs, no torch on any code path.  ``sam3gimpd serve
    --stub``; see ``API.md`` §14.
``TorchEngine``
    The real thing: :class:`~sam3gimpd.engines.pcs.PcsEngine` (text) and
    :class:`~sam3gimpd.engines.pvs.PvsEngine` (points/boxes) sharing one
    :class:`~sam3gimpd.modelmgr.ModelManager`, so both halves agree on device,
    dtype and the fp32 fallback, and ``/status``'s ``models_loaded`` reports
    them together.

Importing this package stays free: ``pcs`` and ``pvs`` are imported inside
:func:`create_engine`, and even they import torch only when a real inference
runs.  ``from sam3gimpd.engines import create_engine`` costs nothing on a machine
with no GPU.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from ..types import PointPrompt, TextPrompt
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
    RawInstance,
    Stage,
    STAGE_PROGRESS,
    build_instances,
    default_canvas_for,
    finalize_instances,
)
from .stub import StubEngine

__all__ = [
    "BaseEngine",
    "CanvasMask",
    "EncodedImage",
    "EngineInfo",
    "ImageData",
    "ProgressFn",
    "ProgressReporter",
    "PromptResult",
    "RawInstance",
    "Stage",
    "STAGE_PROGRESS",
    "DEFAULT_CANVAS_SIDE",
    "build_instances",
    "default_canvas_for",
    "finalize_instances",
    "StubEngine",
    "TorchEngine",
    "create_engine",
    "ENGINE_MODES",
]

#: Values accepted for ``mode``.  ``"auto"`` is what the CLI passes when the
#: user gave neither ``--stub`` nor an explicit choice.
ENGINE_MODES = ("auto", "stub", "torch")


class TorchEngine(BaseEngine):
    """PCS and PVS behind one interface, sharing one model manager.

    Loading is lazy per half.  ``encode_image`` touches only the PCS backbone;
    the tracker is loaded by the first ``prompt_points`` call, which is what
    lets a text-only session avoid the second copy of the vision backbone
    entirely -- the open VRAM question in ``DESIGN.md`` §7.
    """

    MODE = "torch"

    def __init__(self, manager: Optional[Any] = None,
                 canvas_side: int = DEFAULT_CANVAS_SIDE,
                 logger: Any = None,
                 **manager_kwargs: Any) -> None:
        from ..modelmgr import ModelManager  # noqa: PLC0415
        from .pcs import PcsEngine  # noqa: PLC0415
        from .pvs import PvsEngine  # noqa: PLC0415

        self.mgr = manager if manager is not None else ModelManager(logger=logger,
                                                                    **manager_kwargs)
        # The halves log even when no logger is injected: their warnings --
        # embedding reuse that failed, masks that came back binarised -- are
        # the only sign of a silent speed or quality regression.
        self._log = logger if logger is not None else logging.getLogger(ENGINE_LOGGER)
        self.pcs = PcsEngine(self.mgr, canvas_side=canvas_side, logger=self._log)
        self.pvs = PvsEngine(self.mgr, canvas_side=canvas_side, logger=self._log)

    # -- introspection ----------------------------------------------------- #
    @property
    def canvas_side(self) -> int:
        # The PCS half owns canvas discovery: it is the half that always runs
        # first (POST /images), so its answer is the one clients have seen.
        return self.pcs.canvas_side

    def describe(self) -> EngineInfo:
        info = self.pcs.describe()
        caps: List[str] = list(info.capabilities)
        for cap in self.pvs.capabilities():
            if cap not in caps:
                caps.append(cap)
        info.capabilities = caps
        info.models_loaded = self.mgr.loaded()
        info.detail = dict(info.detail)
        info.detail["pvs"] = {
            "embedding_cache_kwarg": self.pvs._embedding_kwarg,
            "embedding_cache_effective": self.pvs._embedding_reuse,
        }
        return info

    # -- work -------------------------------------------------------------- #
    def encode_image(self, image: ImageData,
                     progress: Optional[ProgressFn] = None) -> EncodedImage:
        return self.pcs.encode_image(image, progress)

    def prompt_text(self, encoded: EncodedImage, prompt: TextPrompt,
                    progress: Optional[ProgressFn] = None) -> PromptResult:
        return self.pcs.prompt_text(encoded, prompt, progress)

    def prompt_points(self, encoded: EncodedImage, prompt: PointPrompt,
                      progress: Optional[ProgressFn] = None) -> PromptResult:
        return self.pvs.prompt_points(encoded, prompt, progress)

    # -- lifecycle --------------------------------------------------------- #
    def unload(self, which: Optional[str] = None) -> List[str]:
        return self.mgr.unload(which)

    def close(self) -> None:
        self.mgr.close()


def create_engine(mode: str = "auto", *,
                  canvas_side: int = DEFAULT_CANVAS_SIDE,
                  latency_scale: float = 1.0,
                  logger: Any = None,
                  **manager_kwargs: Any) -> BaseEngine:
    """Build the engine the daemon will serve with.

    ``mode``:

    ``"stub"``
        Always the stub.  ``--stub``.  Never imports torch.
    ``"torch"``
        Always the real engine.  If torch, transformers or the gated weights
        are missing, construction still succeeds -- ``/hello`` and ``/status``
        must answer, and they are how the user finds out what is wrong -- and
        every *prompt* fails with ``503 engine_unavailable`` instead.
    ``"auto"``
        The real engine when torch is importable, the stub otherwise.  This is
        a convenience for development, not the daemon's default: a user who
        installed CUDA wheels and hits a broken torch should see
        ``engine_unavailable``, not silently receive synthetic blobs.

    ``manager_kwargs`` are forwarded to :class:`~sam3gimpd.modelmgr.ModelManager`
    (``model_id``, ``device``, ``dtype``, ...) and ignored in stub mode.
    """
    choice = (mode or "auto").strip().lower()
    if choice not in ENGINE_MODES:
        raise ValueError("unknown engine mode %r; expected one of %s"
                         % (mode, ", ".join(ENGINE_MODES)))
    if choice == "auto":
        from ..modelmgr import probe_torch  # noqa: PLC0415

        choice = "torch" if probe_torch().available else "stub"
    if choice == "stub":
        return StubEngine(canvas_side=canvas_side, latency_scale=latency_scale,
                          logger=logger)
    return TorchEngine(canvas_side=canvas_side, logger=logger, **manager_kwargs)
