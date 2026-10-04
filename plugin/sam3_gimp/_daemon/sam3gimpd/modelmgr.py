"""Device selection, dtype policy and lazy model loading.

This module is the only place in ``sam3gimpd`` that decides *where* and *in what
precision* SAM 3 runs, and the only place that calls
``from_pretrained``.  The engines ask it for a ``(model, processor)`` pair and
otherwise know nothing about devices, dtypes or the HuggingFace cache.

**It imports cleanly with no torch installed and reports that fact** rather than
raising -- ``GET /hello`` and ``GET /status`` must answer in milliseconds on a
machine that has never seen a GPU, and ``sam3gimpd serve --stub`` must not pay for
an import it will never use.  Every torch and transformers import in this file
is therefore inside a function, and the function that performs them,
:func:`probe_torch`, is kept off both the start-up path and the request path.
Constructing a :class:`ModelManager` probes nothing, so the socket is bound and
``runtime.json`` published without waiting for torch; and
:meth:`ModelManager.describe` never starts a probe, because importing torch
holds the GIL through its ``dlopen`` -- seconds on a cold start -- which would
stall even a request thread that was not waiting for it, while the launcher
gives ``GET /hello`` 3 s and treats a daemon that misses them as dead.  The
probe runs on the inference worker, the first time a prompt needs torch.

Policy, from ``DESIGN.md`` §7:

* **Device:** ``cuda`` -> ``mps`` -> ``cpu``.
* **Dtype:** ``bfloat16`` on Ampere and newer (safest for SAM 3's DETR-style
  decoder), ``float16`` on older CUDA, ``float32`` on CPU.  On a numerical
  failure in half precision, fall back to ``float32`` and log it -- see
  :meth:`ModelManager.run`.
* **Lazy per-engine load.** ``Sam3TrackerModel`` (PVS) is only loaded on the
  first click-refine, so a text-only session never pays for it.  This is also
  what makes the open VRAM question ("do the two engines duplicate the vision
  backbone?") measurable: ``/status``'s ``models_loaded`` says which halves are
  resident, and :meth:`ModelManager.describe` reports VRAM alongside it.
* **A load never blocks a report.**  A cold load -- ``from_pretrained`` plus
  ``.to(device)`` -- takes about a minute on real hardware, and the launcher
  treats a daemon whose ``/hello`` misses its 3 s read timeout as dead.  So a
  load holds only the lock for its own half, and everything that reports
  (``loaded()``, ``is_loaded()``, ``describe()``) reads a snapshot without
  taking a lock at all.
* **Weights location.** ``HF_HOME`` is pinned under the project's base directory
  (``paths.hf_home()``) so the Doctor panel can show and clear a ~3.6 GB cache,
  and a local checkpoint directory is accepted as the escape hatch from the
  gated-download flow.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import paths
from .types import ApiError, ErrorCode

__all__ = [
    "DEFAULT_MODEL_ID",
    "ENV_WEIGHTS_DIR",
    "ENV_DEVICE",
    "ENV_DTYPE",
    "PCS",
    "PVS",
    "TorchProbe",
    "NumericalFailure",
    "probe_torch",
    "finished_probe",
    "torch_available",
    "transformers_available",
    "select_device",
    "select_dtype",
    "configure_hf_env",
    "weights_available",
    "local_checkpoint",
    "resolve_model_id",
    "ModelManager",
]

#: The gated HuggingFace repo holding both engines under one checkpoint.
DEFAULT_MODEL_ID = "facebook/sam3"

#: Point this at a directory of locally downloaded (or converted) weights to
#: bypass the gated hub flow entirely -- the second exit ``DESIGN.md`` §7 asks
#: for.  The plug-in's Setup records the same choice in ``weights.json`` under
#: the base directory, and :func:`local_checkpoint` reads that too.
ENV_WEIGHTS_DIR = "SAM3_WEIGHTS_DIR"
ENV_DEVICE = "SAM3D_DEVICE"
ENV_DTYPE = "SAM3D_DTYPE"

#: File, under the base directory, in which the plug-in's Setup records a
#: user-chosen or converted checkpoint as ``{"local_path": ...}``.
WEIGHTS_CONFIG_NAME = "weights.json"

#: Engine half names, matching ``types.Engine`` and ``/status``'s
#: ``models_loaded``.
PCS = "pcs"
PVS = "pvs"

_HALVES = (PCS, PVS)


class NumericalFailure(RuntimeError):
    """Raised by an engine when half precision produced NaN/Inf or blew up.

    :meth:`ModelManager.run` catches it, drops to ``float32`` and retries once.
    Engines raise it from *inside* the function they hand to ``run``, after
    checking what the model produced, because a silent all-NaN mask is far
    worse than a slow one.
    """


# --------------------------------------------------------------------------- #
# probing
# --------------------------------------------------------------------------- #
@dataclass
class TorchProbe:
    """What the environment can actually do.  Answered once and cached."""

    available: bool = False
    version: Optional[str] = None
    error: Optional[str] = None
    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_device_name: Optional[str] = None
    cuda_capability: Optional[Tuple[int, int]] = None
    bf16_supported: bool = False
    #: torch built against HIP (AMD ROCm).  Such a build reports its GPU as a
    #: "cuda" device with a gfx-derived capability tuple that means nothing
    #: on the NVIDIA scale, so the dtype policy must not read it as one.
    hip: bool = False
    hip_version: Optional[str] = None
    mps_available: bool = False
    transformers_available: bool = False
    transformers_version: Optional[str] = None
    sam3_classes_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "torch_available": self.available,
            "torch_version": self.version,
            "torch_error": self.error,
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "cuda_device_name": self.cuda_device_name,
            "cuda_capability": (list(self.cuda_capability)
                                if self.cuda_capability else None),
            "bf16_supported": self.bf16_supported,
            "hip": self.hip,
            "hip_version": self.hip_version,
            "mps_available": self.mps_available,
            "transformers_available": self.transformers_available,
            "transformers_version": self.transformers_version,
            "sam3_classes_available": self.sam3_classes_available,
        }


#: Held for the whole of a probe, so torch is imported once.
_PROBE_LOCK = threading.Lock()
_PROBE: Optional[TorchProbe] = None


def probe_torch(refresh: bool = False) -> TorchProbe:
    """Inspect torch/transformers without ever raising.  Slow the first time:
    it imports both and initialises CUDA.

    A missing torch is a normal, supported state (``--stub`` mode, a base
    ``pip install sam3gimpd``), so every failure here is recorded in
    ``TorchProbe.error`` and reported through ``/hello`` and ``/status`` rather
    than propagated.  Code on a request path uses :func:`finished_probe`.
    """
    global _PROBE
    with _PROBE_LOCK:
        if _PROBE is not None and not refresh:
            return _PROBE
        probe = TorchProbe()
        try:
            import torch  # noqa: PLC0415  (deliberately lazy)
        except Exception as exc:  # ImportError, or a broken CUDA install
            probe.error = "%s: %s" % (type(exc).__name__, exc)
        else:
            probe.available = True
            probe.version = getattr(torch, "__version__", None)
            try:
                hip = getattr(getattr(torch, "version", None), "hip", None)
                probe.hip = bool(hip)
                probe.hip_version = str(hip) if hip else None
            except Exception:
                probe.hip = False
            try:
                probe.cuda_available = bool(torch.cuda.is_available())
                if probe.cuda_available:
                    probe.cuda_device_count = int(torch.cuda.device_count())
                    probe.cuda_device_name = str(torch.cuda.get_device_name(0))
                    cap = torch.cuda.get_device_capability(0)
                    probe.cuda_capability = (int(cap[0]), int(cap[1]))
                    try:
                        probe.bf16_supported = bool(torch.cuda.is_bf16_supported())
                    except Exception:
                        # Ampere (SM 8.0) is where bf16 became usable.
                        probe.bf16_supported = probe.cuda_capability[0] >= 8
            except Exception as exc:
                probe.error = "cuda probe failed: %s: %s" % (type(exc).__name__, exc)
            try:
                mps = getattr(getattr(torch, "backends", None), "mps", None)
                probe.mps_available = bool(mps is not None and mps.is_available())
            except Exception:
                probe.mps_available = False

        if _installed("transformers"):
            probe.transformers_available = True
            try:
                import transformers  # noqa: PLC0415

                probe.transformers_version = getattr(transformers, "__version__", None)
                probe.sam3_classes_available = all(
                    hasattr(transformers, name)
                    for name in ("Sam3Model", "Sam3Processor",
                                 "Sam3TrackerModel", "Sam3TrackerProcessor")
                )
            except Exception as exc:
                probe.transformers_available = False
                if probe.error is None:
                    probe.error = "transformers import failed: %s: %s" % (
                        type(exc).__name__, exc)
        _PROBE = probe
        return probe


def finished_probe() -> Optional[TorchProbe]:
    """The probe if one has completed, else ``None``.  Never starts or waits."""
    return _PROBE


def _installed(name: str) -> bool:
    """Is ``name`` importable?  Answered without importing it."""
    try:
        import importlib.util  # noqa: PLC0415

        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def torch_available() -> bool:
    return probe_torch().available


def transformers_available() -> bool:
    return probe_torch().transformers_available


# --------------------------------------------------------------------------- #
# device and dtype policy
# --------------------------------------------------------------------------- #
_DTYPE_ALIASES = {"bf16": "bfloat16", "fp16": "float16", "half": "float16",
                  "fp32": "float32", "float": "float32", "full": "float32"}
_DTYPES = ("bfloat16", "float16", "float32")


def _device_preference(preference: Optional[str]) -> str:
    """``--device`` after the ``SAM3D_DEVICE`` override; ``"auto"`` if unset."""
    pref = (preference or "auto").strip().lower()
    env = os.environ.get(ENV_DEVICE)
    if pref in ("", "auto") and env:
        pref = env.strip().lower()
    return pref or "auto"


def _dtype_preference(preference: Optional[str]) -> str:
    """The dtype preference after ``SAM3D_DTYPE`` and aliases."""
    pref = (preference or "auto").strip().lower()
    env = os.environ.get(ENV_DTYPE)
    if pref in ("", "auto") and env:
        pref = env.strip().lower()
    return _DTYPE_ALIASES.get(pref, pref) or "auto"


def select_device(preference: str = "auto", probe: Optional[TorchProbe] = None) -> str:
    """Resolve ``--device`` to a concrete torch device string.

    ``auto`` walks ``cuda -> mps -> cpu``.  An explicit preference is honoured
    verbatim, without probing -- if the user asked for ``cuda`` and there is no
    CUDA, they get a clear failure at load time rather than a silent, mystifying
    slowdown on CPU.  With no torch at all the answer is ``"cpu"``: there is no
    device, and ``TorchProbe.available`` is what says so.
    """
    pref = _device_preference(preference)
    if pref != "auto":
        return pref
    p = probe or probe_torch()
    if not p.available:
        return "cpu"
    if p.cuda_available:
        return "cuda"
    if p.mps_available:
        return "mps"
    return "cpu"


def select_dtype(device: str, preference: str = "auto",
                 probe: Optional[TorchProbe] = None) -> str:
    """Resolve the dtype *name* for a device.  Returns a torch dtype attribute
    name (``"bfloat16"``, ``"float16"``, ``"float32"``), never a torch object,
    so this stays importable and testable without torch.  Only a CUDA device
    needs the probe.

    ``DESIGN.md`` §7: bf16 on Ampere+, fp16 on older CUDA, fp32 on CPU.
    """
    pref = _dtype_preference(preference)
    if pref in _DTYPES:
        return pref
    dev = (device or "cpu").split(":", 1)[0].lower()
    if dev == "cuda":
        p = probe or probe_torch()
        if p.hip:
            # AMD via ROCm.  The capability tuple is gfx-derived (RDNA2 says
            # (10, 3), MI250 says (9, 0)), so the ">= 8 means Ampere" rule below
            # would hand bf16 to cards whose matrix units cannot do it.  fp16 is
            # native on every ROCm-supported GPU; anyone on CDNA/RDNA3 who wants
            # bf16 has SAM3D_DTYPE for it.
            return "float16"
        # Compute capability is the authority, not ``is_bf16_supported()``.
        # Since torch 2.6 that helper defaults to ``including_emulation=True``
        # and answers True on Turing (sm_75, e.g. an RTX 2080 Ti), where bf16 is
        # *emulated* rather than native -- picking it there is a straight
        # performance loss over fp16 for no numerical gain.  Ampere (SM 8.0) is
        # where the hardware actually gained bf16, so gate on that and treat the
        # probe as a veto for the odd build that reports no bf16 at all.
        native_bf16 = (p.cuda_capability is not None
                       and p.cuda_capability[0] >= 8)
        if native_bf16 and p.bf16_supported:
            return "bfloat16"
        return "float16"
    if dev == "mps":
        # fp16 is the practical choice on Apple Silicon; a numerical failure
        # falls back to fp32 through ModelManager.run.
        return "float16"
    return "float32"


def resolve_torch_dtype(name: str) -> Any:
    """Map a dtype name to the torch object.  Requires torch."""
    import torch  # noqa: PLC0415

    dtype = getattr(torch, name, None)
    if dtype is None:
        raise ApiError(ErrorCode.MODEL_LOAD_FAILED,
                       "unknown dtype %r" % (name,), {"dtype": name})
    return dtype


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #
def configure_hf_env(environ: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Pin ``HF_HOME`` under the project base directory, in-process.

    Called before any ``transformers`` import so the cache lands somewhere the
    Doctor panel can show and clear.  An ``HF_HOME`` the user set themselves
    wins; ``paths.daemon_environ`` applies the same rule for the *child*
    process, and this applies it to *this* process.
    """
    env = os.environ if environ is None else environ
    applied: Dict[str, str] = {}
    for key, value in (("HF_HOME", str(paths.hf_home())),
                       ("HF_HUB_DISABLE_PROGRESS_BARS", "1"),
                       ("HF_HUB_DISABLE_TELEMETRY", "1")):
        if not env.get(key):
            env[key] = value
        applied[key] = env[key]
    return applied


def _hf_cache_roots() -> List[str]:
    roots = [str(paths.hf_home())]
    for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(key)
        if value:
            roots.append(value)
    roots.append(os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    seen: List[str] = []
    for root in roots:
        if root and root not in seen:
            seen.append(root)
    return seen


def _recorded_checkpoint() -> Optional[str]:
    """The directory the plug-in's Setup recorded in ``weights.json``, if any."""
    data = paths.read_json(paths.base_dir() / WEIGHTS_CONFIG_NAME)
    if isinstance(data, dict):
        value = data.get("local_path")
        if isinstance(value, str) and value:
            return os.path.expanduser(value)
    return None


def local_checkpoint(model_id: str = DEFAULT_MODEL_ID) -> Optional[str]:
    """A user-supplied local weights directory, or ``None``.

    Looked for at ``$SAM3_WEIGHTS_DIR`` first, then at the directory the
    plug-in's Setup recorded in ``<base>/weights.json`` (a folder the user
    pointed it at, or a converted ``sam3.pt``), then at
    ``<base>/models/<last path component of model_id>``, ``<base>/models/
    <org>--<name>`` and ``<base>/models``.  A directory only counts when it
    holds a ``config.json``, which is the cheapest reliable "this is really a
    checkpoint" test.
    """
    candidates: List[str] = []
    env = os.environ.get(ENV_WEIGHTS_DIR)
    if env:
        candidates.append(os.path.expanduser(env))
    recorded = _recorded_checkpoint()
    if recorded:
        candidates.append(recorded)
    leaf = model_id.rsplit("/", 1)[-1]
    candidates.append(str(paths.models_dir() / leaf))
    candidates.append(str(paths.models_dir() / model_id.replace("/", "--")))
    candidates.append(str(paths.models_dir()))
    for path in candidates:
        try:
            if os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json")):
                return path
        except OSError:
            continue
    return None


def weights_available(model_id: str = DEFAULT_MODEL_ID) -> bool:
    """Is the checkpoint on disk?  **Never touches the network.**

    ``GET /hello`` reports this on the critical path of every plug-in
    invocation, so it must stay a handful of ``stat`` calls.
    """
    if local_checkpoint(model_id):
        return True
    marker = "models--" + model_id.replace("/", "--")
    for root in _hf_cache_roots():
        for sub in (marker, os.path.join("hub", marker)):
            candidate = os.path.join(root, sub)
            try:
                if os.path.isdir(candidate) and os.listdir(candidate):
                    return True
            except OSError:
                continue
    return False


def resolve_model_id(model_id: str = DEFAULT_MODEL_ID) -> str:
    """The string to hand ``from_pretrained``: a local directory when one
    exists, otherwise the hub id."""
    return local_checkpoint(model_id) or model_id


# --------------------------------------------------------------------------- #
# the manager
# --------------------------------------------------------------------------- #
@dataclass
class _Loaded:
    """One resident engine half."""

    name: str
    model: Any
    processor: Any
    dtype: str
    device: str
    loaded_at: float = field(default_factory=time.time)


class ModelManager:
    """Owns the loaded SAM 3 halves, their device and their precision.

    Thread safety.  The daemon runs exactly one inference at a time
    (``API.md`` §10), but the HTTP threads that answer ``/hello`` and
    ``/status`` read this object while the worker is loading a model, so:

    * the map of resident halves is never mutated in place: it is replaced,
      under a lock held only for the swap, and :meth:`loaded`,
      :meth:`is_loaded` and :meth:`describe` read whichever map is current
      without taking any lock;
    * each half has its own load lock, held for the whole load, so two callers
      asking for the same half load it once while a caller asking about
      anything else never waits for it.

    Construction is cheap and torch-free, and so is :meth:`describe`: torch is
    probed and imported by :meth:`load`, :meth:`available` and the
    :attr:`device`/:attr:`dtype` properties, which run on the inference worker.
    """

    def __init__(self,
                 model_id: str = DEFAULT_MODEL_ID,
                 device: str = "auto",
                 dtype: str = "auto",
                 local_files_only: Optional[bool] = None,
                 configure_hf: bool = True,
                 logger: Any = None) -> None:
        if configure_hf:
            configure_hf_env()
        self.model_id = model_id
        self._device_pref = device
        self._dtype_pref = dtype
        #: ``None`` means "decide at load time": offline when the weights are
        #: already cached, online otherwise.  The daemon must never block on a
        #: 3.6 GB download it did not announce.
        self.local_files_only = local_files_only
        # Never silent: without a logger a minute-long cold start leaves no
        # trace at all and looks like a hang.
        self._log = logger if logger is not None else logging.getLogger("sam3gimpd.modelmgr")
        self._state_lock = threading.Lock()
        self._load_locks = {half: threading.Lock() for half in _HALVES}
        self._loaded: Dict[str, _Loaded] = {}
        self._closed = False
        self._last_used = time.time()
        self._fp32_fallback = False
        self._last_error: Optional[str] = None
        #: Resolved lazily: an explicit preference now, ``"auto"`` once probed.
        self._device: Optional[str] = None
        self._dtype: Optional[str] = None
        self._resolve_policy(None)

    # -- policy ------------------------------------------------------------ #
    def _resolve_policy(self, probe: Optional[TorchProbe]) -> None:
        """Fill in whatever of device and dtype can be decided.

        Explicit preferences need no probe, and neither does the dtype of a
        CPU or MPS device; only ``"auto"`` on the device, or ``"auto"`` dtype
        on CUDA, waits for ``probe``.
        """
        if self._device is not None and self._dtype is not None:
            return
        with self._state_lock:
            if self._device is None:
                if _device_preference(self._device_pref) != "auto":
                    self._device = select_device(self._device_pref)
                elif probe is not None:
                    self._device = select_device(self._device_pref, probe)
            if self._dtype is None:
                explicit = _dtype_preference(self._dtype_pref) in _DTYPES
                if explicit:
                    self._dtype = select_dtype("cpu", self._dtype_pref)
                elif self._device is not None:
                    on_cuda = self._device.split(":", 1)[0].lower() == "cuda"
                    if not on_cuda or probe is not None:
                        self._dtype = select_dtype(self._device, self._dtype_pref, probe)

    # -- reporting --------------------------------------------------------- #
    @property
    def device(self) -> str:
        """The torch device in force.  Probes torch if ``auto`` is unresolved."""
        if self._device is None:
            self._resolve_policy(probe_torch())
        return self._device or "cpu"

    @property
    def dtype(self) -> str:
        """The dtype *name* in force (fp32 after a fallback).  Probes torch if
        ``auto`` is unresolved."""
        if self._dtype is None:
            self._resolve_policy(probe_torch())
        return self._dtype or "float32"

    @property
    def fp32_fallback(self) -> bool:
        return self._fp32_fallback

    def loaded(self) -> List[str]:
        return sorted(self._loaded)

    def is_loaded(self, half: str) -> bool:
        return half in self._loaded

    def available(self) -> Tuple[bool, Optional[str]]:
        """``(usable, reason)`` -- can this manager actually run inference?

        Never raises and never imports the model, but does wait for the torch
        probe; a request path uses :meth:`describe` instead.  ``reason`` is a
        one-line human explanation suitable for an ``engine_unavailable``
        message.
        """
        return self._availability(probe_torch())

    def _availability(self, probe: TorchProbe,
                      weights: Optional[bool] = None) -> Tuple[bool, Optional[str]]:
        if not probe.available:
            return False, ("torch is not installed in the daemon's environment"
                           + (" (%s)" % probe.error if probe.error else ""))
        if not probe.transformers_available:
            return False, "transformers is not installed in the daemon's environment"
        if weights is None:
            weights = weights_available(self.model_id)
        if not weights:
            return False, ("the SAM 3 checkpoint is not present locally; it is gated, "
                           "so accept the licence and run `sam3gimpd download`")
        return True, None

    def ensure_available(self) -> None:
        """Raise ``engine_unavailable`` unless inference is actually possible."""
        ok, reason = self.available()
        if not ok:
            raise ApiError(ErrorCode.ENGINE_UNAVAILABLE, reason or "engine unavailable",
                           {"model_id": self.model_id,
                            "probe": probe_torch().to_dict()})

    def describe(self) -> Dict[str, Any]:
        """Diagnostics for ``/hello``, ``/status`` and the Doctor panel.

        Never waits for a model load and never imports torch.  Until the first
        prompt has probed torch, ``probed`` is false, ``usable`` is ``None``,
        ``device`` and ``dtype`` are ``"auto"`` unless the user chose them, and
        ``torch_available`` says only whether torch is *installed* -- all that
        can be known without importing it.
        """
        probe = finished_probe()
        if probe is not None:
            self._resolve_policy(probe)
        loaded = self._loaded
        local = local_checkpoint(self.model_id)
        weights = weights_available(self.model_id)
        if probe is not None:
            ok, reason = self._availability(probe, weights)
        else:
            ok, reason = None, None
        d: Dict[str, Any] = {
            "model_id": self.model_id,
            "resolved_model_id": local or self.model_id,
            "local_checkpoint": local,
            "device": self._device or "auto",
            "device_preference": self._device_pref,
            "dtype": self._dtype or "auto",
            "dtype_preference": self._dtype_pref,
            "probed": probe is not None,
            "fp32_fallback": self._fp32_fallback,
            "models_loaded": sorted(loaded),
            "idle_seconds": self.idle_seconds(),
            "weights_available": weights,
            "usable": ok,
            "unavailable_reason": reason,
            "last_error": self._last_error,
            "hf_home": os.environ.get("HF_HOME", str(paths.hf_home())),
        }
        if probe is not None:
            d.update(probe.to_dict())
            if loaded:
                memory = self.memory()
                for key in ("vram_total", "vram_free", "vram_reserved",
                            "vram_allocated", "vram_error"):
                    if memory.get(key) is not None:
                        d[key] = memory[key]
        else:
            d["torch_available"] = _installed("torch")
            d["transformers_available"] = _installed("transformers")
        return d

    def memory(self) -> Dict[str, Any]:
        """``{rss_bytes, vram_total, vram_free, ...}``.

        VRAM is ``None`` unless a finished probe found CUDA -- this never
        starts a probe or initialises CUDA itself.  :meth:`describe` includes
        it while a model is resident, which is how ``/status`` can answer the
        two-engines-duplicate-the-backbone question of ``DESIGN.md`` §7.
        """
        out: Dict[str, Any] = {"rss_bytes": _rss_bytes(),
                               "vram_total": None, "vram_free": None}
        probe = finished_probe()
        if probe is None or not probe.available or not probe.cuda_available:
            return out
        try:
            import torch  # noqa: PLC0415

            free, total = torch.cuda.mem_get_info()
            out["vram_free"] = int(free)
            out["vram_total"] = int(total)
            out["vram_reserved"] = int(torch.cuda.memory_reserved())
            out["vram_allocated"] = int(torch.cuda.memory_allocated())
        except Exception as exc:
            out["vram_error"] = "%s: %s" % (type(exc).__name__, exc)
        return out

    # -- loading ----------------------------------------------------------- #
    def load(self, half: str) -> Tuple[Any, Any]:
        """Return ``(model, processor)`` for ``"pcs"`` or ``"pvs"``, loading on
        first use.

        The PVS half is only ever loaded from :meth:`load` calls made by a point
        prompt, which is how ``DESIGN.md`` §7's "the tracker is only loaded on
        the first click-refine" is enforced -- there is no eager path.
        """
        entry = self._acquire(half)
        return entry.model, entry.processor

    def _acquire(self, half: str, dtype_name: Optional[str] = None) -> _Loaded:
        """The resident entry for ``half``, loading it in ``dtype_name`` (the
        dtype in force by default) if it is not resident."""
        if half not in _HALVES:
            raise ValueError("unknown engine half %r" % (half,))
        entry = self._loaded.get(half)
        if entry is None:
            with self._load_locks[half]:
                # Whoever held the lock before us may have loaded it already.
                entry = self._loaded.get(half)
                if entry is None:
                    if self._closed:
                        # A job still running after shutdown must not bring
                        # the weights back into memory behind close().
                        raise ApiError(ErrorCode.SHUTTING_DOWN, "the daemon is shutting down")
                    self.ensure_available()
                    entry = self._load_locked(half, dtype_name or self.dtype)
                    with self._state_lock:
                        loaded = dict(self._loaded)
                        loaded[half] = entry
                        self._loaded = loaded
        self._last_used = time.time()
        return entry

    #: ``half -> (model class name, processor class name)`` in ``transformers``.
    CLASSES = {
        PCS: ("Sam3Model", "Sam3Processor"),
        PVS: ("Sam3TrackerModel", "Sam3TrackerProcessor"),
    }

    def _load_locked(self, half: str, dtype_name: str) -> _Loaded:
        """Read and place one half.  Called with that half's load lock held."""
        model_cls_name, proc_cls_name = self.CLASSES[half]
        source = resolve_model_id(self.model_id)
        offline = (self.local_files_only if self.local_files_only is not None
                   else weights_available(self.model_id))
        device = self.device
        self._info("loading %s (%s) from %s on %s/%s%s", half, model_cls_name,
                   source, device, dtype_name,
                   " [local_files_only]" if offline else "")
        started = time.time()
        try:
            import transformers  # noqa: PLC0415

            model_cls = getattr(transformers, model_cls_name)
            proc_cls = getattr(transformers, proc_cls_name)
        except Exception as exc:
            self._last_error = "%s: %s" % (type(exc).__name__, exc)
            raise ApiError(
                ErrorCode.ENGINE_UNAVAILABLE,
                "this transformers build has no %s; SAM 3 needs transformers v5"
                % model_cls_name,
                {"half": half, "error": self._last_error,
                 "transformers_version": probe_torch().transformers_version})

        kwargs: Dict[str, Any] = {"local_files_only": bool(offline)}
        torch_dtype = resolve_torch_dtype(dtype_name)
        try:
            model = model_cls.from_pretrained(source, dtype=torch_dtype, **kwargs)
        except TypeError:
            # transformers <5 spells it `torch_dtype`; keep both spellings
            # working rather than pinning behaviour to one point release.
            model = model_cls.from_pretrained(source, torch_dtype=torch_dtype, **kwargs)
        except Exception as exc:
            self._last_error = "%s: %s" % (type(exc).__name__, exc)
            raise ApiError(ErrorCode.MODEL_LOAD_FAILED,
                           "could not load %s: %s" % (model_cls_name, exc),
                           {"half": half, "source": source,
                            "error": self._last_error})
        try:
            processor = proc_cls.from_pretrained(source, **kwargs)
        except Exception as exc:
            self._last_error = "%s: %s" % (type(exc).__name__, exc)
            raise ApiError(ErrorCode.MODEL_LOAD_FAILED,
                           "could not load %s: %s" % (proc_cls_name, exc),
                           {"half": half, "source": source,
                            "error": self._last_error})
        self._info("read %s weights in %.1f s; moving to %s", half,
                   time.time() - started, device)

        try:
            model = model.to(device)
            model.eval()
        except Exception as exc:
            self._last_error = "%s: %s" % (type(exc).__name__, exc)
            raise ApiError(ErrorCode.MODEL_LOAD_FAILED,
                           "could not move %s to %s: %s"
                           % (model_cls_name, device, exc),
                           {"half": half, "device": device,
                            "error": self._last_error})
        self._info("loaded %s in %.1f s", half, time.time() - started)
        return _Loaded(name=half, model=model, processor=processor,
                       dtype=dtype_name, device=device)

    # -- unloading --------------------------------------------------------- #
    def unload(self, half: Optional[str] = None) -> List[str]:
        """Free one half or all of them.  Returns what was actually released.

        Never waits for a load in progress: a half still loading is not
        resident yet, so there is nothing of it to release.
        """
        with self._state_lock:
            loaded = dict(self._loaded)
            names = [half] if half else sorted(loaded)
            gone = [loaded.pop(name) for name in names if name in loaded]
            self._loaded = loaded
        for entry in gone:
            entry.model = None
            entry.processor = None
        if gone:
            self._free_memory()
            self._info("unloaded %s", ", ".join(entry.name for entry in gone))
        return [entry.name for entry in gone]

    def _free_memory(self) -> None:
        import gc  # noqa: PLC0415

        gc.collect()
        probe = finished_probe()
        if probe is None or not probe.available:
            return
        try:
            import torch  # noqa: PLC0415

            if probe.cuda_available:
                torch.cuda.empty_cache()
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            if mps is not None and probe.mps_available:
                empty = getattr(getattr(torch, "mps", None), "empty_cache", None)
                if empty is not None:
                    empty()
        except Exception:
            pass

    def touch(self) -> None:
        """Record activity; ``describe()`` reports the time since."""
        self._last_used = time.time()

    def idle_seconds(self, now: Optional[float] = None) -> float:
        return max(0.0, (now or time.time()) - self._last_used)

    def close(self) -> None:
        self._closed = True
        self.unload()

    # -- running with an fp32 safety net ----------------------------------- #
    def run(self, half: str, fn: Callable[[Any, Any], Any]) -> Any:
        """Call ``fn(model, processor)``, retrying once in ``float32``.

        ``DESIGN.md`` §7: "On a numerical failure in half precision, fall back to
        fp32 and log it."  :func:`_is_numerical_failure` decides what counts:
        the engine raising :class:`NumericalFailure` after finding NaN or Inf in
        what the model produced -- an all-NaN mask is a *silent* wrong answer,
        which is the failure worth spending a reload on -- or torch reporting
        that an op is unimplemented, overflowed, or failed in cuBLAS/cuDNN for
        the half dtype.

        Two obligations follow for ``fn``.  Everything the engine turns into a
        result -- canvas masks, scores -- must be computed *inside* it, since a
        check made after ``run`` has returned cannot trigger the retry.  And the
        retry calls the same ``fn`` against a float32 model, so ``fn`` must not
        feed it tensors from the failed attempt: the engines stamp each cached
        encoding with :attr:`dtype` and re-encode one that no longer matches.

        float32 is committed only once the float32 model has loaded.  If that
        load fails -- float32 needs twice the memory, so out-of-memory is the
        likely reason -- the manager stays in half precision with nothing
        resident and the next request starts afresh.  Once committed, float32
        stays for the life of the process, so the retry can never loop, and
        ``/status`` reports ``fp32_fallback``.
        """
        model, processor = self.load(half)
        try:
            return fn(model, processor)
        except Exception as exc:  # noqa: BLE001 -- re-raised unless numerical
            failed = self.dtype
            if failed == "float32" or not _is_numerical_failure(exc):
                raise
            self._last_error = "%s: %s" % (type(exc).__name__, exc)
            self._warn("half precision (%s) failed on %s (%s); reloading in float32",
                       failed, half, self._last_error)
        # Outside the except block, so that the failed attempt's traceback --
        # whose frames still reference the half-precision model -- is gone
        # before the unload tries to free that model's memory.
        del model, processor
        self.unload()
        try:
            entry = self._acquire(half, "float32")
        except Exception as exc:
            self._warn("float32 reload of %s failed (%s: %s); staying in %s",
                       half, type(exc).__name__, exc, failed)
            raise
        with self._state_lock:
            self._dtype = "float32"
            self._fp32_fallback = True
        return fn(entry.model, entry.processor)

    # -- logging ----------------------------------------------------------- #
    def _info(self, msg: str, *args: Any) -> None:
        if self._log is not None:
            try:
                self._log.info(msg, *args)
            except Exception:
                pass

    def _warn(self, msg: str, *args: Any) -> None:
        if self._log is not None:
            try:
                self._log.warning(msg, *args)
            except Exception:
                pass


#: Messages torch uses when half precision itself is the problem: an op with
#: no Half/BFloat16 kernel, a value that overflows the format, a cuBLAS/cuDNN
#: routine that cannot run in it, or non-finite values in the output.  Matched
#: as phrases -- ``nan`` and ``inf`` as whole words, so "inference" and "info"
#: do not count.
_NUMERICAL = re.compile(
    r"\bnan\b|\binf\b|\binfinity\b|non-finite|not finite"
    r"|not implemented for '(?:half|bfloat16)'"
    r"|without overflow"
    r"|cublas_status_(?:execution_failed|not_supported|arch_mismatch)"
    r"|cudnn_status_(?:execution_failed|not_supported|arch_mismatch)"
    r"|unsupported (?:dtype|data ?type|scalar ?type)",
    re.IGNORECASE)

#: Out of memory.  Reloading in float32 needs twice the memory, so it is never
#: the answer.
_OUT_OF_MEMORY = re.compile(r"out of memory|alloc_failed|failed to allocate", re.IGNORECASE)

#: A tensor of one precision handed to a model of another.  That is a plumbing
#: bug, and a float32 reload would only paper over it -- permanently, and at
#: twice the VRAM.
_DTYPE_MISMATCH = re.compile(
    r"expected scalar type|same dtype|should be the same|dtype mismatch|found dtype",
    re.IGNORECASE)


def _is_numerical_failure(exc: BaseException) -> bool:
    """Would a float32 reload plausibly fix ``exc``?

    :class:`NumericalFailure` always counts.  Otherwise only a torch
    ``RuntimeError`` (which includes ``NotImplementedError``) whose message
    names a half-precision problem does, and never an ``ApiError``, an
    out-of-memory error, or a dtype mismatch.
    """
    if isinstance(exc, NumericalFailure):
        return True
    if isinstance(exc, ApiError) or not isinstance(exc, RuntimeError):
        return False
    if "outofmemory" in type(exc).__name__.lower():
        return False
    text = str(exc)
    if _OUT_OF_MEMORY.search(text) or _DTYPE_MISMATCH.search(text):
        return False
    return bool(_NUMERICAL.search(text))


def _windows_working_set() -> Optional[int]:
    """This process's working set in bytes, from ``kernel32`` via ``ctypes``."""
    try:
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        class _Counters(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD)
        kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = _Counters()
        counters.cb = ctypes.sizeof(counters)
        if kernel32.K32GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
    except Exception:
        pass
    return None


def _rss_bytes() -> Optional[int]:
    """Resident set size, best effort and dependency-free.

    ``/proc`` on Linux, ``psutil`` if it happens to be installed, the working
    set from ``K32GetProcessMemoryInfo`` on Windows (which has no ``resource``
    module), otherwise ``resource`` (whose units differ between Linux and
    macOS, hence the platform check).  ``None`` when nothing works --
    ``/status`` documents the field as nullable.
    """
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        pass
    try:
        import psutil  # noqa: PLC0415

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    if os.name == "nt":
        return _windows_working_set()
    try:
        import resource  # noqa: PLC0415
        import sys  # noqa: PLC0415

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(ru) if sys.platform == "darwin" else int(ru) * 1024
    except Exception:
        return None
