"""One-click environment bootstrap for the sam3-gimp plug-in.

This module is the brain behind ``ui/setup_dialog.py``.  It contains **no GTK
and no GIMP**: everything here is pure standard library so that it imports (and
is unit-tested) on a bare machine with no torch, no weights and no display.
``setup_dialog.py`` is a thin view over it.

Why it exists
-------------
``gimpsegany``, the plug-in this project succeeds, asks users to clone a repo,
``pip install`` SAM, fetch ``.pth`` checkpoints and hand-configure paths.  That
is where most installs die.  ``DESIGN.md`` §8 therefore makes onboarding a
design pillar, and this module implements it:

1. Detect whether the ``sam3gimpd`` environment already exists at the locations
   ``sam3gimpd.paths`` defines (mirrored here in stdlib -- the plug-in may never
   import the daemon package; see ``API.md`` §16.10).
2. Download ``uv`` -- a single static binary -- into ``<base>/tools``, checked
   against a SHA-256 pinned in this file.
3. Create a venv with a **pinned** Python, install the pinned torch from the
   PyTorch index for the accelerator (``cuda`` on Windows/Linux with an NVIDIA
   GPU, ``rocm`` on Linux with ROCm, ``mps`` on Apple Silicon, ``cpu``
   otherwise), then ``sam3gimpd[runtime]`` from its bundled path and PyPI.
4. Verify the install, then optionally fetch the **gated** SAM 3 checkpoint.

Every step is *idempotent* and *resumable*: each step carries a filesystem probe
and is skipped only while that probe holds, and progress is journalled to
``<base>/install-state.json``, so closing GIMP mid-install and reopening it
resumes rather than restarts.

Injection points (this is what makes it testable without a network)
-------------------------------------------------------------------
* ``CommandRunner``  -- every subprocess goes through one object.  Tests pass a
  fake and assert on the exact argv.
* ``Downloader``     -- every byte off the network goes through one object.
* ``opener``         -- the HTTP calls that validate a HuggingFace token take a
  callable so tests never touch huggingface.co.

Nothing in this module downloads anything at import time, and no function here
has a side effect until it is explicitly called.
"""

from __future__ import annotations

import json
import os
import pathlib
import platform as _platform
import hashlib
import queue
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    # constants
    "APP_NAME",
    "ENV_HOME",
    "ENV_RUNTIME_FILE",
    "ENV_WEIGHTS_DIR",
    "UV_VERSION",
    "UV_SHA256",
    "PINNED_PYTHON",
    "MIN_INTERPRETER",
    "SAM3_REPO_ID",
    "SAM3_MODEL_URL",
    "TORCH_INDEX_URLS",
    "TORCH_REQUIREMENTS",
    "STEP_KEYS",
    # platform
    "platform_key",
    "machine_key",
    "Accelerator",
    "detect_nvidia",
    "detect_accelerator",
    "torch_extra_for",
    "torch_index_url",
    # paths (mirror of sam3gimpd.paths)
    "base_dir",
    "runtime_file",
    "lock_file",
    "log_dir",
    "server_log",
    "crash_log",
    "hf_home",
    "models_dir",
    "venv_root",
    "venv_bin_dir",
    "venv_python",
    "venv_pythonw",
    "sam3d_executable",
    "tools_dir",
    "uv_binary",
    "cache_dir",
    "state_file",
    "weights_config_file",
    "ui_settings_file",
    "describe_paths",
    "ensure_layout",
    # uv
    "uv_asset_name",
    "uv_download_url",
    "uv_checksum_url",
    "uv_expected_sha256",
    "find_uv",
    "fetch_uv",
    "ensure_uv",
    # commands
    "uv_venv_command",
    "torch_install_command",
    "uv_install_command",
    "verify_command",
    "doctor_command",
    "hf_login_command",
    "weights_download_command",
    # plan
    "InstallStep",
    "InstallPlan",
    "build_install_plan",
    "build_weights_plan",
    "plan_fingerprint",
    "installed_torch_extra",
    # state
    "InstallState",
    "load_state",
    "save_state",
    "clear_state",
    # environment
    "EnvironmentReport",
    "inspect_environment",
    "read_runtime",
    "pid_alive",
    "tail_file",
    # running
    "CommandResult",
    "CommandRunner",
    "Downloader",
    "extract_uv_archive",
    "Installer",
    "InstallOutcome",
    # weights / doctor
    "validate_hf_token",
    "check_gated_access",
    "inspect_local_weights",
    "set_local_weights",
    "clear_local_weights",
    "local_weights_path",
    "weights_environ",
    "weights_present",
    "run_doctor",
    "doctor_argv",
    "doctor_view",
    "parse_doctor_output",
    "ACCELERATOR_KINDS",
    "UnsupportedPlatform",
    "detect_rocm",
    "platform_blocker",
    "nvidia_driver_version",
    "cuda_driver_warning",
    "bundled_daemon_version",
    "bundled_daemon_build",
    "daemon_update_command",
    "install_sam3d_command",
    "resolve_daemon_install_command",
]


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
APP_NAME = "sam3-gimp"
ENV_HOME = "SAM3_GIMP_HOME"
ENV_RUNTIME_FILE = "SAM3D_RUNTIME_FILE"

#: Where a user-supplied (already downloaded) checkpoint lives.  ``tests/
#: conftest.py`` already probes this name, so it is a project-wide convention:
#: the launcher exports it to the daemon when the user chose the local-path
#: escape hatch instead of the gated download.
ENV_WEIGHTS_DIR = "SAM3_WEIGHTS_DIR"

#: Pinned so an install is reproducible and a cached binary is reusable.
UV_VERSION = "0.9.7"

#: SHA-256 of every uv ``UV_VERSION`` release asset :func:`uv_asset_name` can
#: name, copied from the ``.sha256`` files published beside them.  uv is an
#: unsigned binary that is downloaded and then executed, so its expected hash
#: has to come from here and not from the same server that serves the archive:
#: a checksum fetched from the release can only catch corruption, never a
#: substituted file.  An asset missing from this table is refused.
#: ``tools/install.ps1`` carries the two Windows entries too.
UV_SHA256 = {
    "uv-x86_64-pc-windows-msvc.zip":
        "5d250c32d3604e28dbe18dc65c668ff628c53e00dde2c642576e831e4a60da64",
    "uv-aarch64-pc-windows-msvc.zip":
        "4482ad2544544e1966b6d933d38e27368ce0307739d293322c10459f698a1629",
    "uv-i686-pc-windows-msvc.zip":
        "02349c4380c26838be6fd9bde646c5c3296adc2ba224509ee8e16ffa5baba13e",
    "uv-x86_64-apple-darwin.tar.gz":
        "41946d87e1576c297d6d3cca88b089b6942b8777a5a25e70de1ef8c57b94b9cf",
    "uv-aarch64-apple-darwin.tar.gz":
        "35572b9619fc14d67fc1cd72582c3cfc5c9c66d97f310192e04f26fb3fe96005",
    "uv-x86_64-unknown-linux-gnu.tar.gz":
        "b26fcc8dfa1c39b5a5613445af3be3eefda45d9a39359bee271eafe34913583e",
    "uv-aarch64-unknown-linux-gnu.tar.gz":
        "8b3d31a154673c6d357727d2083a33525b515589d153fa5b5455e1db9e9e6363",
    "uv-i686-unknown-linux-gnu.tar.gz":
        "164d9901a130a4b1e3ac19b33bd26c3e41bb726555b7543dbffe1d7d7fcb2f3c",
    "uv-armv7-unknown-linux-gnueabihf.tar.gz":
        "5957d4296da0d5f668a4203811c764e4d7f3b1296c7e899e94dfe4753dc6c910",
}

#: The interpreter uv provisions for the daemon venv.  Pinned for the same
#: reason ``transformers`` is pinned in ``_daemon/pyproject.toml``: the SAM 3
#: stack is young and a floating Python is a silent breakage waiting to happen.
PINNED_PYTHON = "3.11"

#: The oldest interpreter the daemon can be installed into, and so the oldest
#: one the "use an environment you already have" flow may accept.  It mirrors
#: ``requires-python`` in ``_daemon/pyproject.toml`` and must not drift from it:
#: this gate said 3.9 while the package said 3.10, so a 3.9 environment passed
#: the dialog's check and then failed at "Install sam3gimpd here" with pip's
#: wording instead of ours.  There is deliberately no maximum -- the daemon is
#: pure standard library, and the environment the user already has is usually
#: the newest one they installed.
MIN_INTERPRETER = (3, 10)

UV_RELEASE_BASE = "https://github.com/astral-sh/uv/releases/download"

SAM3_REPO_ID = "facebook/sam3"
SAM3_MODEL_URL = "https://huggingface.co/facebook/sam3"
HF_TOKENS_URL = "https://huggingface.co/settings/tokens"
HF_API_BASE = "https://huggingface.co/api"

#: The PyTorch wheel index for each accelerator.  torch is installed from this
#: index *alone* (``--index-url``), in a step of its own; see
#: :func:`torch_install_command` for why it is never mixed with PyPI.
TORCH_INDEX_URLS = {
    "cuda": "https://download.pytorch.org/whl/cu128",
    "rocm": "https://download.pytorch.org/whl/rocm6.4",  # AMD, Linux x86_64 only
    "cpu": "https://download.pytorch.org/whl/cpu",
    "mps": None,  # Apple Silicon uses the default PyPI wheels
}

#: torch and torchvision exactly as every accelerator extra in
#: ``_daemon/pyproject.toml`` pins them.  The managed install puts these in
#: first, from the index above, and then the daemon with its ``[runtime]``
#: extra, which names no torch at all.
TORCH_REQUIREMENTS = ("torch==2.9.0", "torchvision==0.24.0")

#: Every accelerator kind the installer knows how to provision, in the order
#: the Setup dialog offers them.  Each is an extra in ``_daemon/pyproject.toml``.
ACCELERATOR_KINDS = ("cuda", "rocm", "mps", "cpu")

#: Ordered install-step identifiers.  ``InstallState.completed`` holds a subset.
STEP_KEYS = ("uv", "venv", "torch", "sam3gimpd", "verify", "weights")

#: Bumped when the on-disk journal format changes; a mismatch discards state.
STATE_VERSION = 1


# --------------------------------------------------------------------------- #
# platform detection
# --------------------------------------------------------------------------- #
def platform_key(name: Optional[str] = None) -> str:
    """``"windows"`` | ``"macos"`` | ``"linux"``.

    Mirrors ``sam3gimpd.paths.platform_key``.  Everything that is not Windows or
    Darwin is treated as Linux/XDG, which is also right for the BSDs.
    """
    if name is not None:
        n = name.lower()
        if n.startswith("win"):
            return "windows"
        if n in ("darwin", "macos", "mac", "osx"):
            return "macos"
        return "linux"
    if os.name == "nt" or sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


_MACHINE_ALIASES = {
    "amd64": "x86_64",
    "x86_64": "x86_64",
    "x64": "x86_64",
    "em64t": "x86_64",
    "arm64": "aarch64",
    "aarch64": "aarch64",
    "armv8": "aarch64",
    "i386": "i686",
    "i486": "i686",
    "i586": "i686",
    "i686": "i686",
    "x86": "i686",
    "armv7l": "armv7",
}


def machine_key(name: Optional[str] = None) -> str:
    """Normalised CPU architecture: ``x86_64``, ``aarch64``, ``i686`` ...

    ``platform.machine()`` says ``AMD64`` on Windows and ``x86_64`` on Linux for
    the same CPU; uv's release assets use the Rust triple spelling.  This
    collapses the spellings so :func:`uv_asset_name` has one thing to switch on.
    """
    raw = (name if name is not None else _platform.machine()) or ""
    return _MACHINE_ALIASES.get(raw.strip().lower(), raw.strip().lower() or "x86_64")


@dataclass
class Accelerator:
    """What the daemon will most likely run on, decided *without* torch.

    ``kind`` is one of ``"cuda"``, ``"rocm"``, ``"mps"``, ``"cpu"``.  It picks
    the install *extra*; the daemon still does its own ``--device auto`` probe
    at runtime and is the authority on what it actually used (``GET /hello``).
    ROCm is "cuda" to torch (``torch.cuda.is_available()`` answers True on HIP),
    so the daemon needs no separate device name for it; the distinction only
    matters here, where it decides which wheel index to pull from.
    """

    kind: str
    reason: str
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_gpu(self) -> bool:
        return self.kind in ("cuda", "rocm", "mps")


def detect_nvidia(
    *,
    runner: Optional["CommandRunner"] = None,
    which: Optional[Callable[[str], Optional[str]]] = None,
    env: Optional[Dict[str, str]] = None,
    exists: Optional[Callable[[str], bool]] = None,
    platform_name: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Is there an NVIDIA GPU with a working driver?

    Deliberately does not import torch (there is no torch in GIMP's Python, and
    on a fresh machine there is no torch anywhere yet).  Order of evidence:

    1. ``SAM3_ASSUME_NVIDIA`` -- an explicit override for headless CI and for
       users on exotic setups.
    2. ``nvidia-smi -L`` -- the only positive proof that a *driver* is loaded.
    3. Filesystem tells: ``%SystemRoot%\\System32\\nvidia-smi.exe`` on Windows,
       ``/proc/driver/nvidia/version`` on Linux.  These say "driver installed"
       when ``nvidia-smi`` is not on ``PATH``.
    """
    env = os.environ if env is None else env
    which = shutil.which if which is None else which
    exists = os.path.exists if exists is None else exists

    override = (env.get("SAM3_ASSUME_NVIDIA") or "").strip().lower()
    if override in ("1", "true", "yes", "on"):
        return True, {"source": "SAM3_ASSUME_NVIDIA"}
    if override in ("0", "false", "no", "off"):
        return False, {"source": "SAM3_ASSUME_NVIDIA"}

    smi = which("nvidia-smi")
    if smi:
        runner = runner or CommandRunner()
        result = runner.run([smi, "-L"], timeout=15.0)
        text = "\n".join(result.lines)
        if result.returncode == 0 and "GPU" in text:
            names = [ln.strip() for ln in result.lines if ln.strip().startswith("GPU ")]
            detail: Dict[str, Any] = {"source": "nvidia-smi", "gpus": names}
            driver = nvidia_driver_version(runner=runner, smi=smi)
            if driver:
                detail["driver_version"] = driver
            return True, detail
        return False, {"source": "nvidia-smi", "returncode": result.returncode}

    kind = platform_key(platform_name)
    if kind == "windows":
        system_root = env.get("SystemRoot") or env.get("SYSTEMROOT") or "C:\\Windows"
        candidate = os.path.join(system_root, "System32", "nvidia-smi.exe")
        if exists(candidate):
            return True, {"source": "system32", "path": candidate}
    elif kind == "linux":
        for candidate in ("/proc/driver/nvidia/version", "/dev/nvidiactl"):
            if exists(candidate):
                return True, {"source": candidate}
    return False, {"source": "none"}


def nvidia_driver_version(*, runner: Optional["CommandRunner"] = None,
                          smi: str = "nvidia-smi") -> Optional[str]:
    """``"576.02"`` from ``nvidia-smi --query-gpu=driver_version``, or ``None``."""
    runner = runner or CommandRunner()
    try:
        result = runner.run(
            [smi, "--query-gpu=driver_version", "--format=csv,noheader"], timeout=15.0)
    except Exception:  # noqa: BLE001
        return None
    if result.returncode != 0:
        return None
    for line in result.lines:
        candidate = line.strip().split(",")[0].strip()
        if re.match(r"^\d+(\.\d+)*$", candidate):
            return candidate
    return None


#: Oldest NVIDIA driver the cu128 wheels run on (CUDA 12.x minor-version
#: compatibility: 525.60.13 on Linux, 528.33 on Windows).
CUDA_MIN_DRIVER = {"windows": (528, 33), "linux": (525, 60)}


def cuda_driver_warning(accel: "Accelerator") -> Optional[str]:
    """A sentence for the Setup dialog when the driver predates CUDA 12.8, else
    ``None``.  A warning, not a blocker: the install itself succeeds, it is the
    first model load that would fail -- after a 3 GB download."""
    if accel.kind != "cuda":
        return None
    driver = str(accel.detail.get("driver_version") or "")
    platform = str(accel.detail.get("platform") or platform_key())
    floor = CUDA_MIN_DRIVER.get(platform)
    if not driver or floor is None:
        return None
    try:
        parts = tuple(int(p) for p in driver.split(".")[:2])
    except ValueError:
        return None
    if len(parts) == 1:
        parts = (parts[0], 0)
    if parts >= floor:
        return None
    return ("NVIDIA driver %s is older than the %d.%02d these CUDA 12.8 wheels need. "
            "Update the driver before installing, or the model will fail to load "
            "with a 'CUDA driver version is insufficient' error."
            % (driver, floor[0], floor[1]))


def detect_rocm(
    *,
    which: Optional[Callable[[str], Optional[str]]] = None,
    exists: Optional[Callable[[str], bool]] = None,
    env: Optional[Dict[str, str]] = None,
    platform_name: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Is there an AMD GPU with the ROCm stack installed?  Linux only.

    PyTorch publishes ROCm wheels for Linux x86_64 and nothing else, so on any
    other platform the answer is simply no.  Evidence, cheapest first:

    1. ``SAM3_ASSUME_ROCM`` -- explicit override.
    2. ``/dev/kfd`` (the amdgpu compute device) together with either
       ``/opt/rocm`` or a ``rocminfo`` / ``rocm-smi`` binary.  ``/dev/kfd``
       alone is not enough: it exists on any box with an amdgpu-driven display,
       including ones with no ROCm runtime at all.

    Nothing is spawned; this runs in the same places :func:`detect_nvidia` does.
    """
    env = os.environ if env is None else env
    which = shutil.which if which is None else which
    exists = os.path.exists if exists is None else exists

    override = (env.get("SAM3_ASSUME_ROCM") or "").strip().lower()
    if override in ("1", "true", "yes", "on"):
        return True, {"source": "SAM3_ASSUME_ROCM"}
    if override in ("0", "false", "no", "off"):
        return False, {"source": "SAM3_ASSUME_ROCM"}

    if platform_key(platform_name) != "linux":
        return False, {"source": "none"}
    if not exists("/dev/kfd"):
        return False, {"source": "none"}
    if exists("/opt/rocm"):
        return True, {"source": "/dev/kfd + /opt/rocm"}
    for tool in ("rocminfo", "rocm-smi"):
        if which(tool):
            return True, {"source": "/dev/kfd + %s" % tool}
    return False, {"source": "/dev/kfd without a ROCm runtime"}


def platform_blocker(platform_name: Optional[str] = None,
                     machine: Optional[str] = None) -> Optional[str]:
    """Why the managed torch install cannot succeed on this host, or ``None``.

    The one host that is common enough to name is the Intel Mac: PyTorch
    stopped shipping macOS x86_64 wheels after 2.2, and the pinned 2.9 line
    simply does not exist there.  32-bit and armv7 have never had torch.
    """
    kind = platform_key(platform_name)
    mach = machine_key(machine)
    if kind == "macos" and mach == "x86_64":
        return ("PyTorch no longer publishes wheels for Intel Macs (macOS x86_64); "
                "the last was 2.2, and SAM 3 needs a current torch. Setup cannot "
                "build an environment here. Options: run sam3gimpd on another "
                "machine and point SAM3D_COMMAND at it, or an Apple Silicon Mac.")
    if mach in ("i686", "armv7"):
        return ("PyTorch has no wheels for 32-bit %s systems, so Setup cannot "
                "build an environment here." % mach)
    return None


def detect_accelerator(
    *,
    platform_name: Optional[str] = None,
    machine: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    runner: Optional["CommandRunner"] = None,
    which: Optional[Callable[[str], Optional[str]]] = None,
    exists: Optional[Callable[[str], bool]] = None,
) -> Accelerator:
    """Decide which torch flavour to install.

    ``SAM3_FORCE_DEVICE`` short-circuits everything -- useful when a user has a
    GPU we mis-detect, and the difference between a 2.5 GB CUDA wheel and a
    200 MB CPU wheel is worth an escape hatch.
    """
    env = os.environ if env is None else env
    kind = platform_key(platform_name)
    mach = machine_key(machine)

    forced = (env.get("SAM3_FORCE_DEVICE") or "").strip().lower()
    if forced in ACCELERATOR_KINDS:
        return Accelerator(forced, "forced by SAM3_FORCE_DEVICE", {"platform": kind, "machine": mach})

    if kind == "macos":
        if mach == "aarch64":
            return Accelerator("mps", "Apple Silicon: Metal Performance Shaders", {"machine": mach})
        return Accelerator("cpu", "Intel Mac: no supported accelerator", {"machine": mach})

    has_nv, detail = detect_nvidia(
        runner=runner, which=which, env=env, exists=exists, platform_name=kind
    )
    detail = dict(detail, platform=kind, machine=mach)
    if has_nv:
        return Accelerator("cuda", "NVIDIA driver detected", detail)

    has_amd, amd_detail = detect_rocm(which=which, exists=exists, env=env, platform_name=kind)
    if has_amd and mach == "x86_64":
        return Accelerator("rocm", "AMD ROCm detected", dict(amd_detail, platform=kind, machine=mach))
    return Accelerator("cpu", "no NVIDIA driver or ROCm runtime detected", detail)


def torch_extra_for(accel: "Accelerator | str") -> str:
    """Map an accelerator to the ``sam3gimpd`` extra defined in ``_daemon/pyproject.toml``."""
    kind = accel.kind if isinstance(accel, Accelerator) else str(accel)
    if kind in ACCELERATOR_KINDS:
        return kind
    return "cpu"


def torch_index_url(extra: str) -> Optional[str]:
    """The PyTorch index for a wheel flavour, or ``None`` when PyPI serves it."""
    return TORCH_INDEX_URLS.get(extra)


# --------------------------------------------------------------------------- #
# paths -- a stdlib mirror of _daemon/sam3gimpd/paths.py
# --------------------------------------------------------------------------- #
# The plug-in must not import sam3gimpd (API.md §16.10: different Python, zero
# third-party imports), so this is duplicated by design.  The layout is
# deliberately trivial precisely so that mirroring it is a dozen lines.
def _home() -> str:
    return os.path.expanduser("~")


def _default_base_dir() -> str:
    kind = platform_key()
    if kind == "windows":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return _spelled(os.path.join(local, APP_NAME))
        appdata = os.environ.get("APPDATA")
        if appdata:
            return _spelled(os.path.join(appdata, APP_NAME))
        return _spelled(os.path.join(_home(), "AppData", "Local", APP_NAME))
    if kind == "macos":
        return _spelled(os.path.join(_home(), "Library", "Application Support", APP_NAME))
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg and os.path.isabs(xdg):
        return _spelled(os.path.join(xdg, APP_NAME))
    return _spelled(os.path.join(_home(), ".local", "share", APP_NAME))


def _spelled(path: str) -> str:
    """``path`` spelled the way ``sam3gimpd.paths`` spells it.

    ``pathlib`` normalises separators and doubled slashes (``/tmp//x/`` is
    ``/tmp/x``, and ``\\tmp\\x`` on Windows).  Building every path the same
    way on both halves keeps Doctor, the logs and any comparison of the two in
    agreement about which file they mean.
    """
    return str(pathlib.Path(path))


def _override_path(value: str) -> str:
    """An environment override with ``~`` expanded, spelled as :func:`_spelled`.

    A ``~user`` that does not exist, or a ``~`` with no home directory to
    find, is kept as written -- what ``os.path.expanduser`` does -- rather than
    failing every caller.  ``pathlib`` reports those as ``RuntimeError`` on
    Python 3.10+ and as ``KeyError`` on 3.8 and 3.9.
    """
    try:
        return str(pathlib.Path(value).expanduser())
    except (RuntimeError, KeyError):
        return _spelled(value)


def base_dir() -> str:
    """Root of everything this project owns on disk.  Never cached -- tests flip
    ``SAM3_GIMP_HOME`` between cases and expect the change to be seen."""
    override = os.environ.get(ENV_HOME)
    if override:
        return _override_path(override)
    return _default_base_dir()


def runtime_file() -> str:
    override = os.environ.get(ENV_RUNTIME_FILE)
    if override:
        return _override_path(override)
    return os.path.join(base_dir(), "runtime.json")


def lock_file() -> str:
    return os.path.join(base_dir(), "sam3gimpd.lock")


def log_dir() -> str:
    return os.path.join(base_dir(), "logs")


def server_log() -> str:
    return os.path.join(log_dir(), "sam3gimpd.log")


def crash_log() -> str:
    return os.path.join(log_dir(), "crash.log")


def install_log() -> str:
    """Transcript of the last bootstrap run -- the Doctor panel shows its tail."""
    return os.path.join(log_dir(), "install.log")


def hf_home() -> str:
    return os.path.join(base_dir(), "hf")


def models_dir() -> str:
    return os.path.join(base_dir(), "models")


def venv_root() -> str:
    return os.path.join(base_dir(), "venv")


def venv_bin_dir() -> str:
    return os.path.join(venv_root(), "Scripts" if platform_key() == "windows" else "bin")


def venv_python() -> str:
    return os.path.join(venv_bin_dir(), "python.exe" if platform_key() == "windows" else "python")


def venv_pythonw() -> str:
    """Windowless interpreter.  On Windows the launcher MUST use this to spawn
    the daemon; ``python.exe`` flashes a console window (DESIGN.md §4)."""
    return os.path.join(venv_bin_dir(), "pythonw.exe" if platform_key() == "windows" else "python")


def sam3d_executable() -> str:
    return os.path.join(venv_bin_dir(), "sam3gimpd.exe" if platform_key() == "windows" else "sam3gimpd")


def tools_dir() -> str:
    return os.path.join(base_dir(), "tools")


def uv_binary() -> str:
    return os.path.join(tools_dir(), "uv.exe" if platform_key() == "windows" else "uv")


def cache_dir() -> str:
    return os.path.join(base_dir(), "cache")


def state_file() -> str:
    """Install journal.  Its existence is what makes the install resumable."""
    return os.path.join(base_dir(), "install-state.json")


def weights_config_file() -> str:
    """Records a user-supplied local checkpoint directory (the gated escape hatch)."""
    return os.path.join(base_dir(), "weights.json")


def ui_settings_file() -> str:
    """Every main-dialog setting, including the ones that are not procedure
    arguments (GIMP's ``Gimp.ProcedureConfig`` only remembers those)."""
    return os.path.join(base_dir(), "ui-settings.json")


def describe_paths() -> Dict[str, str]:
    """Flat ``{name: path}`` map for the Doctor panel.  Same keys as
    ``sam3gimpd.paths.describe`` plus the bootstrap-only entries."""
    return {
        "platform": platform_key(),
        "machine": machine_key(),
        "base": base_dir(),
        "runtime_file": runtime_file(),
        "lock_file": lock_file(),
        "log_dir": log_dir(),
        "server_log": server_log(),
        "crash_log": crash_log(),
        "install_log": install_log(),
        "hf_home": hf_home(),
        "models_dir": models_dir(),
        "venv_root": venv_root(),
        "venv_python": venv_python(),
        "sam3d_executable": sam3d_executable(),
        "uv_binary": uv_binary(),
        "cache_dir": cache_dir(),
        "state_file": state_file(),
    }


def _private_dir(path: str, *, tighten: bool = True) -> str:
    """``mkdir -p`` a directory only this user may enter (0700 on POSIX).

    ``tighten`` also narrows one that already exists, for directories that are
    unambiguously ours.  ``hf/`` is the one that matters most: huggingface_hub
    keeps the login token there.  Windows has no mode bits to set; the
    per-user profile directories already are private.
    """
    existed = os.path.isdir(path)
    os.makedirs(path, mode=0o700, exist_ok=True)
    if os.name != "nt" and (tighten or not existed):
        try:
            os.chmod(path, 0o700)
        except OSError:
            pass
    return path


def ensure_layout() -> Dict[str, str]:
    """``mkdir -p`` every directory the installer writes into, private to the
    user.  The base directory is only narrowed when this call creates it or it
    is the platform default: ``SAM3_GIMP_HOME`` may point somewhere the user
    manages themselves."""
    base = base_dir()
    _private_dir(base, tighten=(base == _default_base_dir()))
    layout = {
        "base": base,
        "logs": _private_dir(log_dir()),
        "hf": _private_dir(hf_home()),
        "models": _private_dir(models_dir()),
        "tools": _private_dir(tools_dir()),
        "cache": _private_dir(cache_dir()),
    }
    return layout


def _open_private_append(path: str):
    """``open(path, "a")`` for a text file created 0600, and narrowed to 0600
    if an older run left it wider."""
    # O_BINARY: the text layer above does the newline translation; a
    # text-mode descriptor on Windows would do it a second time.
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o600)
    if os.name != "nt":
        try:
            os.fchmod(fd, 0o600)
        except (OSError, AttributeError):
            pass
    return os.fdopen(fd, "a", encoding="utf-8")


def _atomic_write_bytes(path: str, data: bytes, mode: int = 0o600) -> str:
    """Temp file in the same directory, fsync, ``os.replace``: a reader sees
    the old file or the new one, never a partial one."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.chmod(tmp, mode)
        except OSError:
            pass
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return path


def _atomic_write_json(path: str, obj: Any, mode: int = 0o600) -> str:
    """Same discipline as ``sam3gimpd.paths.atomic_write_json``: temp file in the
    same directory, fsync, ``os.replace``.  A reader never sees a partial file."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.chmod(tmp, mode)
        except OSError:
            pass
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return path


def _read_json(path: str) -> Optional[Any]:
    # utf-8-sig: Windows PowerShell 5 writes a byte-order mark, and
    # tools/install.ps1 writes one of the files read here.
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            text = fh.read()
    except OSError:
        return None
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def tail_file(path: str, lines: int = 40, max_bytes: int = 262144) -> List[str]:
    """Last ``lines`` lines of a log, cheaply and without loading a huge file.

    Used for the "last crash log" pane and for the spawn-timeout diagnostic in
    ``API.md`` §3.3 step 8.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
                fh.readline()  # discard a partial line
            data = fh.read()
    except OSError:
        return []
    text = data.decode("utf-8", "replace")
    out = text.splitlines()
    return out[-lines:] if lines > 0 else out


# --------------------------------------------------------------------------- #
# uv release assets
# --------------------------------------------------------------------------- #
_UV_TRIPLES = {
    ("windows", "x86_64"): "x86_64-pc-windows-msvc",
    ("windows", "aarch64"): "aarch64-pc-windows-msvc",
    ("windows", "i686"): "i686-pc-windows-msvc",
    ("macos", "x86_64"): "x86_64-apple-darwin",
    ("macos", "aarch64"): "aarch64-apple-darwin",
    ("linux", "x86_64"): "x86_64-unknown-linux-gnu",
    ("linux", "aarch64"): "aarch64-unknown-linux-gnu",
    ("linux", "i686"): "i686-unknown-linux-gnu",
    ("linux", "armv7"): "armv7-unknown-linux-gnueabihf",
}


def uv_asset_name(platform_name: Optional[str] = None, machine: Optional[str] = None) -> str:
    """Release asset filename for this host, e.g. ``uv-x86_64-pc-windows-msvc.zip``.

    Windows assets are ``.zip``; everything else is ``.tar.gz``.  Raises
    :class:`BootstrapError` for a host uv does not publish for, rather than
    guessing a URL that would 404 halfway through the install.
    """
    kind = platform_key(platform_name)
    mach = machine_key(machine)
    triple = _UV_TRIPLES.get((kind, mach))
    if triple is None:
        raise BootstrapError(
            "no uv release for %s/%s -- install sam3gimpd manually into %s"
            % (kind, mach, venv_root())
        )
    suffix = ".zip" if kind == "windows" else ".tar.gz"
    return "uv-%s%s" % (triple, suffix)


def uv_download_url(
    version: str = UV_VERSION,
    platform_name: Optional[str] = None,
    machine: Optional[str] = None,
) -> str:
    return "%s/%s/%s" % (UV_RELEASE_BASE, version, uv_asset_name(platform_name, machine))


def uv_checksum_url(
    version: str = UV_VERSION,
    platform_name: Optional[str] = None,
    machine: Optional[str] = None,
) -> str:
    """Where uv publishes an asset's checksum.  Informational: the hash that
    is actually checked is the pinned one in :data:`UV_SHA256`."""
    return uv_download_url(version, platform_name, machine) + ".sha256"


def uv_expected_sha256(url: str) -> Optional[str]:
    """The pinned SHA-256 for a uv release URL, or ``None`` when this plug-in
    pins nothing for it (another version, an unknown asset) -- which the
    caller must treat as a refusal, not as "skip the check"."""
    parts = url.rstrip("/").split("/")
    if len(parts) < 2 or parts[-2] != UV_VERSION:
        return None
    return UV_SHA256.get(parts[-1])


# --------------------------------------------------------------------------- #
# command construction  (the part the tests pin down)
# --------------------------------------------------------------------------- #
def uv_venv_command(
    uv: str,
    venv: str,
    python_version: str = PINNED_PYTHON,
    *,
    python_preference: str = "managed",
) -> List[str]:
    """``uv venv --clear --python 3.11 --python-preference managed <venv>``

    ``--python-preference managed`` lets uv *download* the pinned interpreter
    when the host has none, which is the whole point: GIMP's embedded Python is
    unusable for torch and the user may have no system Python at all.

    ``--clear`` because this step only runs when the venv's interpreter is
    missing, i.e. when whatever is at ``<venv>`` is broken or half-created --
    and uv refuses to create a venv over a directory that has no
    ``pyvenv.cfg``, so Repair could never rebuild one without it.
    """
    return [uv, "venv", "--clear", "--python", python_version,
            "--python-preference", python_preference, venv]


def torch_install_command(
    uv: str,
    python: str,
    *,
    index_url: Optional[str],
    requirements: Sequence[str] = TORCH_REQUIREMENTS,
    reinstall: bool = True,
) -> List[str]:
    """``uv pip install --python <py> --index-url <pytorch index> torch==... torchvision==...``

    torch is installed on its own, from the PyTorch index **alone**, before the
    daemon.  The alternative -- one command with PyPI plus
    ``--extra-index-url <pytorch> --index-strategy unsafe-best-match`` -- lets
    the resolver take any package from whichever index offers the best
    version, which uv documents as open to dependency confusion.  The PyTorch
    index serves every dependency of torch and torchvision itself, for each
    accelerator and platform this installer targets, so it needs no PyPI
    fallback.  Apple Silicon (``index_url`` ``None``) uses PyPI, uv's default.

    ``reinstall`` makes uv replace an installed torch even when its version
    already satisfies the pin: ``2.9.0+cpu`` satisfies ``torch==2.9.0``, so
    without it switching a venv from CPU to CUDA kept the CPU build.
    """
    argv = [uv, "pip", "install", "--python", python]
    if index_url:
        argv += ["--index-url", index_url]
    if reinstall:
        for requirement in requirements:
            argv += ["--reinstall-package", re.split(r"[<>=!~;\[ ]", requirement, maxsplit=1)[0]]
    argv += list(requirements)
    return argv


def uv_install_command(
    uv: str,
    python: str,
    *,
    extra: str = "runtime",
    source: Optional[str] = None,
    upgrade: bool = False,
    offline: bool = False,
    pre: bool = False,
    reinstall: Sequence[str] = (),
) -> List[str]:
    """``uv pip install --python <venv-python> "<daemon>[runtime]"``

    Notes that are load-bearing:

    * ``--python`` targets the venv explicitly instead of relying on an
      activated environment -- there is no shell here, only ``subprocess``.
    * No index options: the daemon comes from its local path and its
      ``[runtime]`` dependencies from PyPI, uv's default.  torch is not among
      them; :func:`torch_install_command` has already installed it from the
      PyTorch index, and nothing here asks for it again.
    * ``source`` is the bundled daemon directory unless given (``SAM3D_SOURCE``
      points the dev loop at a working tree).
    * ``reinstall`` names distributions uv must reinstall even at an unchanged
      version -- the daemon itself when updating it, because uv decides a
      local path is unchanged from its ``pyproject.toml`` alone.
    """
    # ``None`` means "the bundled daemon".  Never a bare requirement name: see
    # BUNDLED_DAEMON_DIRNAME for why that once installed a stranger's package.
    source = daemon_source() if source is None else source
    requirement = "%s[%s]" % (source, extra) if extra else source
    argv = [uv, "pip", "install", "--python", python]
    if upgrade:
        argv.append("--upgrade")
    if offline:
        argv.append("--offline")
    if pre:
        argv.append("--prerelease=allow")
    for name in reinstall:
        argv += ["--reinstall-package", name]
    argv.append(requirement)
    return argv


#: Verification runs through the venv's *interpreter*, not the ``sam3gimpd`` console
#: script, so it works even if the entry point failed to be written.  It fails
#: (exit 1) when torch or transformers cannot be imported: either one missing
#: means the daemon cannot load SAM 3, and an install that passes verify and
#: then fails every segmentation is worse than one that stops here.
VERIFY_SNIPPET = (
    "import json,sys\n"
    "import sam3gimpd\n"
    "d={'sam3d_version':getattr(sam3gimpd,'__version__','?'),"
    "'api_version':getattr(sam3gimpd,'__api_version__','?'),"
    "'python':sys.version.split()[0],'executable':sys.executable}\n"
    "missing=[]\n"
    "try:\n"
    " import torch\n"
    " d['torch']=torch.__version__\n"
    " d['cuda']=bool(torch.cuda.is_available())\n"
    "except Exception as e:\n"
    " d['torch']=None\n"
    " d['torch_error']=str(e)\n"
    " missing.append('torch (%s)'%e)\n"
    "try:\n"
    " import transformers\n"
    " d['transformers']=transformers.__version__\n"
    "except Exception as e:\n"
    " d['transformers']=None\n"
    " d['transformers_error']=str(e)\n"
    " missing.append('transformers (%s)'%e)\n"
    "print('SAM3D_VERIFY '+json.dumps(d))\n"
    "if missing:\n"
    " print('The environment cannot load SAM 3; these do not import: '+'; '.join(missing))\n"
    " raise SystemExit(1)\n"
)


def verify_command(python: str) -> List[str]:
    """Import ``sam3gimpd``, torch and transformers inside the new venv and
    print a one-line JSON report; exits non-zero when torch or transformers
    is missing."""
    return [python, "-c", VERIFY_SNIPPET]


def doctor_command(sam3gimpd: str) -> List[str]:
    """``sam3gimpd doctor`` (API.md §13), which prints its report as JSON.

    There is no ``--json`` flag: the CLI's only output is JSON, and argparse
    rejects the flag outright.  :func:`parse_doctor_output` still copes with
    output that merely *contains* a JSON object.
    """
    return [sam3gimpd, "doctor"]


#: The token is passed through the **environment**, never argv: argv is visible
#: to every process on the box via ``ps`` / Task Manager.  ``huggingface_hub``
#: then owns storage from that point on -- we never write the token ourselves
#: (DESIGN.md §8.3).
HF_LOGIN_SNIPPET = (
    "import os;"
    "from huggingface_hub import login;"
    "login(token=os.environ['HF_TOKEN'],add_to_git_credential=False);"
    "print('SAM3D_LOGIN ok')"
)


def hf_login_command(python: str) -> List[str]:
    return [python, "-c", HF_LOGIN_SNIPPET]


def weights_download_command(python: str, *, repo_id: str = SAM3_REPO_ID) -> List[str]:
    """``<python> -m sam3gimpd download --repo-id facebook/sam3`` (API.md §13).

    Through the interpreter rather than the managed venv's console script, so
    it works in an environment the user chose, which has no such script where
    Setup would have put one.
    """
    return [python, "-m", "sam3gimpd", "download", "--repo-id", repo_id]


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
class BootstrapError(RuntimeError):
    """Anything the installer can explain to a user in one sentence."""


class CancelledError(BootstrapError):
    """The user pressed Cancel.  Not a failure; the journal is kept for resume."""


class UnsupportedPlatform(BootstrapError):
    """No torch wheel exists for this host; the managed install cannot work.

    Raised by :func:`build_install_plan` instead of letting ``uv`` discover it
    twenty minutes in with "no solution found".  The Setup dialog shows the
    message and leaves the *existing environment* route open.
    """


# --------------------------------------------------------------------------- #
# install plan
# --------------------------------------------------------------------------- #
@dataclass
class InstallStep:
    """One resumable unit of work.

    ``kind`` is ``"download"`` (fetch + unpack ``uv``) or ``"command"`` (run
    ``argv``).  ``satisfied`` is a filesystem probe, and it alone decides
    whether the step is skipped: a step whose probe holds is skipped even if
    the journal was lost, which is what makes a partial install self-heal, and
    one whose probe fails runs even if the journal says it was done, because
    the files it made can be gone (a deleted venv, antivirus).  A step with no
    probe (verify) always runs.  ``after`` runs once the step has succeeded.
    """

    key: str
    title: str
    kind: str
    argv: List[str] = field(default_factory=list)
    url: str = ""
    dest: str = ""
    weight: float = 1.0
    detail: str = ""
    satisfied: Optional[Callable[[], bool]] = None
    env: Dict[str, str] = field(default_factory=dict)
    after: Optional[Callable[[], None]] = None

    def is_satisfied(self) -> bool:
        if self.satisfied is None:
            return False
        try:
            return bool(self.satisfied())
        except Exception:
            return False

    def describe(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "kind": self.kind,
            "argv": list(self.argv),
            "url": self.url,
            "dest": self.dest,
            "weight": self.weight,
            "detail": self.detail,
        }


@dataclass
class InstallPlan:
    """An ordered list of steps plus the decisions that produced them."""

    steps: List[InstallStep]
    accelerator: Accelerator
    extra: str
    index_url: Optional[str]
    python_version: str
    uv_version: str
    source: str
    paths: Dict[str, str] = field(default_factory=dict)
    #: Advisory sentences to show above the Install button (an old NVIDIA
    #: driver, say).  They never stop the install.
    warnings: List[str] = field(default_factory=list)

    def step(self, key: str) -> Optional[InstallStep]:
        for s in self.steps:
            if s.key == key:
                return s
        return None

    @property
    def keys(self) -> List[str]:
        return [s.key for s in self.steps]

    def pending(self, state: Optional["InstallState"] = None, *, probe: bool = True) -> List[InstallStep]:
        """Steps an install would have to run now.  Called on every dialog
        open, so a finished install reports "ready".

        The disk decides, not the journal: a step with a probe is pending
        exactly while its probe fails.  A probe-less check (verify) is pending
        once anything before it is, or while it has never been journalled as
        passing.  ``probe=False`` ignores what is on disk and returns every
        step.
        """
        if not probe:
            return list(self.steps)
        done = set(state.completed) if state else set()
        out: List[InstallStep] = []
        for s in self.steps:
            if s.satisfied is not None:
                if s.is_satisfied():
                    continue
            elif s.key in done and not out:
                continue
            out.append(s)
        return out

    def total_weight(self) -> float:
        return sum(max(0.0, s.weight) for s in self.steps) or 1.0

    def describe(self) -> Dict[str, Any]:
        return {
            "accelerator": self.accelerator.kind,
            "accelerator_reason": self.accelerator.reason,
            "extra": self.extra,
            "index_url": self.index_url,
            "python_version": self.python_version,
            "uv_version": self.uv_version,
            "source": self.source,
            "fingerprint": plan_fingerprint(self),
            "warnings": list(self.warnings),
            "steps": [s.describe() for s in self.steps],
        }


def plan_fingerprint(plan: InstallPlan) -> str:
    """Identity of the *decisions* in a plan.

    If the user's GPU appears (cpu -> cuda), or we bump uv/Python/the requirement
    source, the journal from the previous plan is meaningless and is discarded
    rather than causing a half-CPU/half-CUDA venv.
    """
    import hashlib

    payload = "|".join(
        [
            str(STATE_VERSION),
            plan.extra,
            plan.index_url or "-",
            plan.python_version,
            plan.uv_version,
            plan.source,
            ",".join(plan.keys),
        ]
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def build_install_plan(
    *,
    accelerator: Optional[Accelerator] = None,
    with_weights: bool = False,
    source: Optional[str] = None,
    uv_version: str = UV_VERSION,
    python_version: str = PINNED_PYTHON,
    upgrade: bool = False,
    env: Optional[Dict[str, str]] = None,
    platform_name: Optional[str] = None,
    machine: Optional[str] = None,
    runner: Optional["CommandRunner"] = None,
) -> InstallPlan:
    """Assemble the whole install as data, before anything is executed.

    Keeping the plan a pure value means the setup dialog can *show* it ("we will
    download uv 0.9.7, create a Python 3.11 venv, install torch from
    download.pytorch.org/whl/cu128, then sam3gimpd"), the tests can assert on
    it, and the installer becomes a dumb loop.

    ``upgrade`` (Reinstall) removes the probes from the torch and daemon steps
    so they run again, and makes uv reinstall the daemon even at an unchanged
    version.
    """
    env = os.environ if env is None else env
    blocker = platform_blocker(platform_name, machine)
    if blocker:
        raise UnsupportedPlatform(blocker)
    accel = accelerator or detect_accelerator(
        platform_name=platform_name, machine=machine, env=env, runner=runner
    )
    extra = torch_extra_for(accel)
    index_url = torch_index_url(extra)
    src = source or daemon_source(env)

    uv_path = uv_binary()
    py = venv_python()

    steps: List[InstallStep] = [
        InstallStep(
            key="uv",
            title="Download uv",
            kind="download",
            url=uv_download_url(uv_version, platform_name, machine),
            dest=uv_path,
            weight=1.0,
            detail="uv %s (%s)" % (uv_version, uv_asset_name(platform_name, machine)),
            satisfied=lambda: os.path.exists(uv_binary()),
        ),
        InstallStep(
            key="venv",
            title="Create Python %s environment" % python_version,
            kind="command",
            argv=uv_venv_command(uv_path, venv_root(), python_version),
            weight=2.0,
            detail=venv_root(),
            satisfied=lambda: os.path.exists(venv_python()),
        ),
        InstallStep(
            key="torch",
            title="Install PyTorch (%s)" % extra,
            kind="command",
            argv=torch_install_command(uv_path, py, index_url=index_url),
            weight=11.0,  # multi-GB torch download: this is the long pole
            detail=index_url or "PyPI",
            # Satisfied only by torch installed *for this accelerator*: the
            # daemon's console script existing said nothing about which torch
            # sat beside it, so switching CPU -> CUDA kept the CPU build.
            satisfied=(None if upgrade else (lambda: installed_torch_extra() == extra)),
            after=lambda: _write_torch_marker(extra, index_url),
        ),
        InstallStep(
            key="sam3gimpd",
            title="Install sam3gimpd",
            kind="command",
            argv=uv_install_command(
                uv_path, py, extra="runtime", source=src, upgrade=upgrade,
                reinstall=(DAEMON_DIST_NAME,) if upgrade else (),
            ),
            weight=2.0,
            detail="the bundled daemon; its dependencies from PyPI",
            satisfied=(None if upgrade else (lambda: os.path.exists(sam3d_executable()))),
        ),
        InstallStep(
            key="verify",
            title="Verify the installation",
            kind="command",
            argv=verify_command(py),
            weight=0.5,
            detail="import sam3gimpd, torch, transformers",
            satisfied=None,  # always cheap; always re-run
        ),
    ]

    if with_weights:
        steps.append(_weights_step(py))

    return InstallPlan(
        steps=steps,
        accelerator=accel,
        extra=extra,
        index_url=index_url,
        python_version=python_version,
        uv_version=uv_version,
        source=src,
        paths=describe_paths(),
        warnings=[w for w in (cuda_driver_warning(accel),) if w],
    )


def _weights_step(python: str) -> InstallStep:
    return InstallStep(
        key="weights",
        title="Download SAM 3 weights",
        kind="command",
        argv=weights_download_command(python),
        weight=8.0,
        detail=SAM3_REPO_ID,
        satisfied=weights_present,
    )


def build_weights_plan(python: str, *, accelerator: Optional[Accelerator] = None) -> InstallPlan:
    """Sign in to HuggingFace and download the gated checkpoint, using
    ``python`` -- whichever environment is in use (:func:`daemon_interpreter`),
    not necessarily the managed venv, which a user of their own environment
    does not have.

    The token reaches both steps through the environment the caller gives the
    :class:`Installer`, never through argv.  Run it with ``journal=False``:
    it is a per-run flow and must not replace the install's journal.
    """
    login = InstallStep(
        key="hf_login",
        title="Sign in to HuggingFace",
        kind="command",
        argv=hf_login_command(python),
        weight=0.2,
        detail="huggingface_hub stores the token",
    )
    accel = accelerator or Accelerator("unknown", "not needed for a download")
    return InstallPlan(
        steps=[login, _weights_step(python)],
        accelerator=accel,
        extra="",
        index_url=None,
        python_version="",
        uv_version="",
        source="",
        paths=describe_paths(),
    )


# --------------------------------------------------------------------------- #
# which torch the managed venv holds
# --------------------------------------------------------------------------- #
def torch_marker_file() -> str:
    """Records the accelerator the managed venv's torch was installed for.

    Inside the venv on purpose: deleting or rebuilding the venv takes the
    record with it, so it can never vouch for a torch that is gone.
    """
    return os.path.join(venv_root(), "sam3-gimp-torch.json")


def _write_torch_marker(extra: str, index_url: Optional[str]) -> None:
    _atomic_write_json(torch_marker_file(), {
        "extra": extra,
        "index_url": index_url,
        "requirements": list(TORCH_REQUIREMENTS),
        "installed_at": time.time(),
    })


def _legacy_torch_extra() -> Optional[str]:
    """The accelerator of a venv built before the marker existed, when torch
    came in with the daemon's own extra: the journal recorded that extra, and
    the daemon step completing means torch arrived with it."""
    st = InstallState.from_json(_read_json(state_file()))
    if (st.version == STATE_VERSION and st.extra in ACCELERATOR_KINDS
            and "sam3gimpd" in st.completed and os.path.exists(sam3d_executable())
            and os.path.exists(venv_python())):
        return st.extra
    return None


def installed_torch_extra() -> Optional[str]:
    """The accelerator whose torch the managed venv holds (``"cuda"``,
    ``"cpu"`` ...), or ``None`` when that is unknown or the pins it was
    installed with are not today's :data:`TORCH_REQUIREMENTS` -- either way
    the torch step has to run."""
    if not os.path.exists(venv_python()):
        return None
    marker = _read_json(torch_marker_file())
    if isinstance(marker, dict):
        if list(marker.get("requirements") or []) != list(TORCH_REQUIREMENTS):
            return None
        extra = marker.get("extra")
        return str(extra) if extra else None
    return _legacy_torch_extra()


# --------------------------------------------------------------------------- #
# install journal (resumability)
# --------------------------------------------------------------------------- #
@dataclass
class InstallState:
    """``<base>/install-state.json`` -- what survives GIMP being closed mid-install."""

    version: int = STATE_VERSION
    fingerprint: str = ""
    completed: List[str] = field(default_factory=list)
    started_at: float = 0.0
    updated_at: float = 0.0
    last_error: Optional[str] = None
    last_step: Optional[str] = None
    accelerator: str = ""
    extra: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "fingerprint": self.fingerprint,
            "completed": list(self.completed),
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "last_error": self.last_error,
            "last_step": self.last_step,
            "accelerator": self.accelerator,
            "extra": self.extra,
        }

    @classmethod
    def from_json(cls, obj: Any) -> "InstallState":
        if not isinstance(obj, dict):
            return cls()
        completed = obj.get("completed")
        if not isinstance(completed, list):
            completed = []
        return cls(
            version=int(obj.get("version") or 0),
            fingerprint=str(obj.get("fingerprint") or ""),
            completed=[str(c) for c in completed if isinstance(c, (str, bytes))],
            started_at=float(obj.get("started_at") or 0.0),
            updated_at=float(obj.get("updated_at") or 0.0),
            last_error=obj.get("last_error"),
            last_step=obj.get("last_step"),
            accelerator=str(obj.get("accelerator") or ""),
            extra=str(obj.get("extra") or ""),
        )

    def mark(self, key: str) -> "InstallState":
        if key not in self.completed:
            self.completed.append(key)
        self.last_step = key
        self.updated_at = time.time()
        return self


def load_state(plan: Optional[InstallPlan] = None, path: Optional[str] = None) -> InstallState:
    """Read the journal, discarding it when it does not match ``plan``.

    A state file from a different plan (or a future format version) is worse
    than no state file: it would skip steps that were never run for *this*
    configuration.  Silently starting over is the safe failure.
    """
    st = InstallState.from_json(_read_json(path or state_file()))
    if st.version != STATE_VERSION:
        return InstallState()
    if plan is not None and st.fingerprint and st.fingerprint != plan_fingerprint(plan):
        return InstallState()
    return st


def save_state(state: InstallState, path: Optional[str] = None) -> str:
    state.updated_at = time.time()
    return _atomic_write_json(path or state_file(), state.to_json())


def clear_state(path: Optional[str] = None) -> None:
    try:
        os.remove(path or state_file())
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# running commands
# --------------------------------------------------------------------------- #
@dataclass
class CommandResult:
    argv: List[str]
    returncode: int
    lines: List[str] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def tail(self, n: int = 12) -> str:
        return "\n".join(self.lines[-n:])


class CommandRunner:
    """The single choke point for every subprocess the bootstrap runs.

    Streams merged stdout+stderr line by line to ``on_line`` so the setup
    dialog's log view fills in live (uv's resolver output is the only feedback a
    user gets during a 2.5 GB torch download).  Tests replace this object
    wholesale and assert on ``argv``.

    ``timeout`` and ``cancel`` are honoured within a tenth of a second whether
    or not the child prints anything: output is read on a helper thread while
    this one polls.  They were once checked only when a line arrived, so a
    silent ``uv`` download or a hung ``nvidia-smi`` could hold the caller -- the
    GTK thread, for ``nvidia-smi`` -- for as long as the child liked.
    Stopping a child ends its whole process tree, since uv does its work in
    children of its own.
    """

    #: How often the wait loop wakes to look at ``cancel`` and the clock.
    POLL_S = 0.1

    def __init__(self, *, encoding: str = "utf-8") -> None:
        # Explicit, never the locale's: uv and Python print UTF-8, and decoding
        # that as cp1252 on Windows turned every non-ASCII character to mojibake.
        self.encoding = encoding

    @staticmethod
    def _popen_kwargs() -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if os.name == "nt":
            # Never flash a console window out of a GIMP plug-in.
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if flags:
                kwargs["creationflags"] = flags
            startupinfo = getattr(subprocess, "STARTUPINFO", None)
            if startupinfo is not None:
                si = startupinfo()
                si.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
                kwargs["startupinfo"] = si
        else:
            # Its own process group, so the whole tree can be signalled.
            kwargs["start_new_session"] = True
        return kwargs

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        on_line: Optional[Callable[[str], None]] = None,
        timeout: Optional[float] = None,
        cancel: Optional[threading.Event] = None,
        input_text: Optional[str] = None,
    ) -> CommandResult:
        """``input_text`` is written to the child's stdin and the pipe closed.

        Used to hand a program to ``python -`` without it existing as a file,
        which removes an entire failure class: a script written to disk can be
        gone by the time the interpreter opens it.
        """
        argv = [str(a) for a in argv]
        started = time.monotonic()
        full_env = dict(os.environ)
        if env:
            full_env.update({str(k): str(v) for k, v in env.items()})
        lines: List[str] = []
        try:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=(subprocess.PIPE if input_text is not None
                       else subprocess.DEVNULL),
                cwd=cwd,
                env=full_env,
                bufsize=1,
                encoding=self.encoding,
                errors="replace",
                close_fds=True,
                **self._popen_kwargs()
            )
        except OSError as exc:
            msg = "%s: %s" % (argv[0], exc)
            if on_line:
                on_line(msg)
            return CommandResult(argv, 127, [msg], time.monotonic() - started)

        # ``None`` marks the end of the output.
        output: "queue.Queue[Optional[str]]" = queue.Queue()

        def _read() -> None:
            try:
                assert proc.stdout is not None
                for raw in proc.stdout:
                    output.put(raw)
            except Exception:  # noqa: BLE001 -- a closed pipe ends the output
                pass
            finally:
                output.put(None)

        reader = threading.Thread(target=_read, name="sam3-runner-output", daemon=True)
        reader.start()

        if input_text is not None:
            # The reader is already draining stdout, so a child that prints
            # before it has read all of its program cannot deadlock us.
            try:
                assert proc.stdin is not None
                proc.stdin.write(input_text)
                proc.stdin.close()
            except Exception:  # pragma: no cover - child died instantly
                pass

        def _emit(raw: str) -> None:
            line = raw.rstrip("\r\n")
            lines.append(line)
            if on_line:
                on_line(line)

        ended = False
        exited_at: Optional[float] = None
        try:
            while True:
                try:
                    raw = output.get(timeout=self.POLL_S)
                except queue.Empty:
                    raw = ""
                if raw is None:
                    ended = True
                elif raw:
                    _emit(raw)
                if cancel is not None and cancel.is_set():
                    self._terminate(proc)
                    break
                if timeout is not None and (time.monotonic() - started) > timeout:
                    self._terminate(proc)
                    lines.append("timed out after %.0fs" % timeout)
                    break
                if proc.poll() is not None:
                    if ended:
                        break
                    # Exited, but something it started still holds the pipe
                    # open: give the tail a moment, then stop waiting for it.
                    exited_at = exited_at or time.monotonic()
                    if time.monotonic() - exited_at > 2.0:
                        break
        except Exception as exc:  # pragma: no cover - defensive
            self._terminate(proc)
            lines.append("runner error: %s" % exc)

        reader.join(timeout=1.0)
        while True:
            try:
                raw = output.get_nowait()
            except queue.Empty:
                break
            if raw is None:
                break
            _emit(raw)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - _terminate kills
            self._terminate(proc)
        if not reader.is_alive():
            # Closing while the reader is still blocked in read() would block
            # here too; a pipe some grandchild holds is left to that process.
            try:
                if proc.stdout is not None:
                    proc.stdout.close()
            except Exception:
                pass
        rc = proc.returncode if proc.returncode is not None else -1
        return CommandResult(argv, rc, lines, time.monotonic() - started)

    @staticmethod
    def _terminate(proc: "subprocess.Popen") -> None:
        """End ``proc`` and everything it started: politely, then by force."""
        if proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                # taskkill /T is the tree; there is no gentler signal to send.
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, timeout=15,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            else:
                os.killpg(proc.pid, signal.SIGTERM)
        except Exception:  # noqa: BLE001 -- fall back to the child alone
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            if os.name == "nt":
                proc.kill()
            else:
                os.killpg(proc.pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=5)
        except Exception:  # pragma: no cover
            pass


# --------------------------------------------------------------------------- #
# downloading
# --------------------------------------------------------------------------- #
ProgressCb = Callable[[float, int, int], None]  # (fraction, done_bytes, total_bytes)


class Downloader:
    """Resumable HTTP download with progress, stdlib only.

    Writes to ``<dest>.part`` and only ``os.replace``s into place on success, so
    an interrupted download never leaves a truncated ``uv.exe`` that would fail
    mysteriously on the next run.  A partial ``.part`` is resumed with a
    ``Range`` header when the server allows it (GitHub releases do).
    """

    def __init__(self, opener: Optional[Callable[..., Any]] = None, *, chunk: int = 65536) -> None:
        self._opener = opener
        self.chunk = chunk

    def _open(self, url: str, headers: Dict[str, str], timeout: float):
        if self._opener is not None:
            return self._opener(url, headers=headers, timeout=timeout)
        import urllib.request  # lazy: keeps plug-in import cost down

        req = urllib.request.Request(url, headers=headers)
        return urllib.request.urlopen(req, timeout=timeout)

    def download(
        self,
        url: str,
        dest: str,
        *,
        on_progress: Optional[ProgressCb] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 60.0,
        resume: bool = True,
        cancel: Optional[threading.Event] = None,
    ) -> str:
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        part = dest + ".part"
        have = os.path.getsize(part) if (resume and os.path.exists(part)) else 0
        req_headers = {"User-Agent": "sam3-gimp-bootstrap/1"}
        if headers:
            req_headers.update(headers)
        if have:
            req_headers["Range"] = "bytes=%d-" % have

        resp = self._open(url, req_headers, timeout)
        try:
            status = getattr(resp, "status", None) or getattr(resp, "code", 200)
            if status == 200 and have:
                # Server ignored Range -- start over rather than concatenating.
                have = 0
            elif status not in (200, 206):
                raise BootstrapError("download failed (HTTP %s): %s" % (status, url))
            total = self._content_length(resp) + have
            flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_BINARY", 0)
            flags |= os.O_APPEND if have else os.O_TRUNC
            done = have
            with os.fdopen(os.open(part, flags, 0o600), "ab" if have else "wb") as fh:
                while True:
                    if cancel is not None and cancel.is_set():
                        raise CancelledError("download cancelled")
                    block = resp.read(self.chunk)
                    if not block:
                        break
                    fh.write(block)
                    done += len(block)
                    if on_progress:
                        frac = (done / total) if total > 0 else 0.0
                        on_progress(min(1.0, max(0.0, frac)), done, total)
        finally:
            try:
                resp.close()
            except Exception:
                pass
        os.replace(part, dest)
        return dest

    @staticmethod
    def _content_length(resp: Any) -> int:
        headers = getattr(resp, "headers", None)
        value = None
        if headers is not None:
            get = getattr(headers, "get", None)
            if callable(get):
                value = get("Content-Length")
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0


def extract_uv_archive(archive: str, dest: str) -> str:
    """Pull the single ``uv`` binary out of a release archive, executable by
    this user only.

    uv ships ``uv-<triple>/uv`` inside a ``.tar.gz`` (or ``uv.exe`` inside a
    ``.zip``); we do not care about the rest of the archive, so this extracts
    exactly one member and refuses anything with a traversal-looking name.

    The binary is written beside ``dest`` under a temporary name and renamed
    into place only once complete, because ``dest`` existing is what marks the
    uv step done: an extraction interrupted halfway must not leave a truncated
    ``uv`` that every later run would skip past and then fail to execute.
    """
    directory = os.path.dirname(dest) or "."
    os.makedirs(directory, exist_ok=True)

    def _acceptable(name: str) -> bool:
        base = os.path.basename(name.replace("\\", "/"))
        if base.lower() not in ("uv", "uv.exe"):
            return False
        norm = os.path.normpath(name)
        return not (norm.startswith("..") or os.path.isabs(norm))

    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(dest) + ".", suffix=".part", dir=directory)
    try:
        with os.fdopen(fd, "wb") as out:
            if archive.lower().endswith(".zip"):
                import zipfile

                with zipfile.ZipFile(archive) as zf:
                    member = next((n for n in zf.namelist() if _acceptable(n)), None)
                    if member is None:
                        raise BootstrapError("no uv binary inside %s" % os.path.basename(archive))
                    with zf.open(member) as src:
                        shutil.copyfileobj(src, out)
            else:
                import tarfile

                with tarfile.open(archive, "r:*") as tf:
                    member = next((m for m in tf.getmembers()
                                   if m.isfile() and _acceptable(m.name)), None)
                    if member is None:
                        raise BootstrapError("no uv binary inside %s" % os.path.basename(archive))
                    src = tf.extractfile(member)
                    if src is None:
                        raise BootstrapError("could not read uv from %s" % os.path.basename(archive))
                    with src:
                        shutil.copyfileobj(src, out)
            out.flush()
            os.fsync(out.fileno())
        if os.name != "nt":
            os.chmod(tmp, stat.S_IRWXU)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return dest


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_uv(
    url: str,
    dest: str,
    *,
    downloader: Optional[Downloader] = None,
    on_progress: Optional[ProgressCb] = None,
    on_log: Optional[Callable[[str], None]] = None,
    cancel: Optional[threading.Event] = None,
) -> str:
    """Download the uv release archive at ``url``, check it against the
    SHA-256 pinned in :data:`UV_SHA256`, and install the binary at ``dest``.

    Fails closed: no pinned hash, or a mismatch, and nothing is installed.
    """
    log = on_log or (lambda _line: None)
    expected = uv_expected_sha256(url)
    if not expected:
        raise BootstrapError(
            "This plug-in pins no checksum for %s, so it will not install it. "
            "Setup only installs uv %s." % (url, UV_VERSION))
    tmpdir = tempfile.mkdtemp(prefix="uv-", dir=_safe_tmp_dir())
    archive = os.path.join(tmpdir, os.path.basename(url) or "uv-archive")
    try:
        (downloader or Downloader()).download(url, archive, on_progress=on_progress, cancel=cancel)
        actual = _sha256_file(archive)
        if actual != expected:
            raise BootstrapError(
                "The uv download does not match the checksum this plug-in pins "
                "(expected %s..., got %s...). The file was corrupted or replaced "
                "in transit; nothing was installed. Try again, or check your "
                "proxy." % (expected[:12], actual[:12]))
        log("       sha256 verified")
        extract_uv_archive(archive, dest)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return dest


def find_uv() -> Optional[str]:
    """A uv to install with, without downloading one: Setup's own, else one
    on ``PATH`` (a user whose environment uv made usually has it there)."""
    managed = uv_binary()
    if os.path.isfile(managed):
        return managed
    return shutil.which("uv")


def ensure_uv(
    *,
    downloader: Optional[Downloader] = None,
    on_log: Optional[Callable[[str], None]] = None,
    cancel: Optional[threading.Event] = None,
) -> str:
    """Setup's own uv, fetched (pinned and verified) if it is not there yet."""
    dest = uv_binary()
    if os.path.isfile(dest):
        return dest
    ensure_layout()
    url = uv_download_url()
    if on_log:
        on_log("fetching uv %s from %s" % (UV_VERSION, url))
    return fetch_uv(url, dest, downloader=downloader, on_log=on_log, cancel=cancel)


# --------------------------------------------------------------------------- #
# weights: the gated-repo flow and its escape hatch
# --------------------------------------------------------------------------- #
_HF_WHOAMI = HF_API_BASE + "/whoami-v2"


def _http_json(
    url: str,
    *,
    token: Optional[str] = None,
    opener: Optional[Callable[..., Any]] = None,
    timeout: float = 15.0,
) -> Tuple[int, Any]:
    """GET a JSON document, returning ``(status, parsed_or_text)``.

    ``opener(url, headers=..., timeout=...)`` is injectable so the token flow is
    unit-testable with no network.  Never raises for an HTTP error status: the
    caller wants to *explain* 401 vs 403 to the user, not see a traceback.
    """
    headers = {"Accept": "application/json", "User-Agent": "sam3-gimp-bootstrap/1"}
    if token:
        headers["Authorization"] = "Bearer " + token
    try:
        if opener is not None:
            resp = opener(url, headers=headers, timeout=timeout)
        else:
            import urllib.request  # lazy

            resp = urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=timeout
            )
    except Exception as exc:  # urllib.error.HTTPError included
        status = getattr(exc, "code", None) or getattr(exc, "status", None)
        if status is None:
            return 0, str(exc)
        try:
            body = exc.read().decode("utf-8", "replace")  # type: ignore[attr-defined]
        except Exception:
            body = str(exc)
        try:
            return int(status), json.loads(body)
        except ValueError:
            return int(status), body
    try:
        status = int(getattr(resp, "status", None) or getattr(resp, "code", 200))
        raw = resp.read()
    finally:
        try:
            resp.close()
        except Exception:
            pass
    text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
    try:
        return status, json.loads(text)
    except ValueError:
        return status, text


_TOKEN_RE = re.compile(r"^hf_[A-Za-z0-9]{20,}$")


def validate_hf_token(
    token: str, *, opener: Optional[Callable[..., Any]] = None, timeout: float = 15.0
) -> Dict[str, Any]:
    """Check a HuggingFace token *before* starting a 3.6 GB download.

    Returns ``{ok, name, message, status}``.  Shape-checks locally first so an
    obvious paste error ("hf_..." truncated, or the whole URL pasted) is caught
    without a round trip.
    """
    token = (token or "").strip()
    if not token:
        return {"ok": False, "status": 0, "message": "Paste a token first.", "name": None}
    if not _TOKEN_RE.match(token):
        return {
            "ok": False,
            "status": 0,
            "name": None,
            "message": "That does not look like a HuggingFace token (they start with 'hf_').",
        }
    status, body = _http_json(_HF_WHOAMI, token=token, opener=opener, timeout=timeout)
    if status == 200 and isinstance(body, dict):
        return {
            "ok": True,
            "status": 200,
            "name": body.get("name") or body.get("fullname"),
            "message": "Token accepted for %s." % (body.get("name") or "your account"),
            "raw": body,
        }
    if status in (401, 403):
        return {"ok": False, "status": status, "name": None, "message": "HuggingFace rejected that token."}
    if status == 0:
        return {"ok": False, "status": 0, "name": None, "message": "Could not reach huggingface.co: %s" % body}
    return {"ok": False, "status": status, "name": None, "message": "Unexpected response (HTTP %s)." % status}


def _no_redirect_opener() -> Any:
    """A urllib opener that hands a 3xx back (as an ``HTTPError``) instead of
    following it.  urllib would turn a redirected HEAD into a GET and carry
    the ``Authorization`` header to whatever host the redirect names."""
    import urllib.request  # noqa: PLC0415

    class _Keep3xx(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args: Any, **kwargs: Any) -> None:
            return None

    return urllib.request.build_opener(_Keep3xx())


def _is_hf_host(url: str) -> bool:
    import urllib.parse  # noqa: PLC0415

    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    return host in ("huggingface.co", "hf.co") or host.endswith((".huggingface.co", ".hf.co"))


def _http_head(
    url: str,
    *,
    token: Optional[str] = None,
    opener: Optional[Callable[..., Any]] = None,
    timeout: float = 15.0,
    max_redirects: int = 5,
) -> Tuple[int, str]:
    """``HEAD url`` following redirects, returning ``(status, detail)``.

    ``detail`` is HuggingFace's ``X-Error-Code`` or the error text; status
    ``0`` means the host could not be reached.  Redirects are followed here,
    as HEAD, and the token only goes to huggingface.co hosts.  ``opener(url,
    headers=..., timeout=..., method="HEAD")`` is injectable for tests; it may
    return or raise a 3xx carrying a ``Location`` header.
    """
    import urllib.parse  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    def _open(target: str, headers: Dict[str, str]) -> Any:
        if opener is not None:
            return opener(target, headers=headers, timeout=timeout, method="HEAD")
        return _no_redirect_opener().open(
            urllib.request.Request(target, headers=headers, method="HEAD"), timeout=timeout)

    for _hop in range(max_redirects + 1):
        headers = {"User-Agent": "sam3-gimp-bootstrap/1"}
        if token and _is_hf_host(url):
            headers["Authorization"] = "Bearer " + token
        try:
            resp = _open(url, headers)
        except Exception as exc:  # urllib.error.HTTPError included
            resp = exc
            status = getattr(exc, "code", None) or getattr(exc, "status", None)
            if status is None:
                return 0, str(exc)
        status = int(getattr(resp, "status", None) or getattr(resp, "code", 0) or 0)
        resp_headers = getattr(resp, "headers", None) or {}
        try:
            close = getattr(resp, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
        get = getattr(resp_headers, "get", None)
        location = get("Location") if callable(get) else None
        if status in (301, 302, 303, 307, 308) and location:
            url = urllib.parse.urljoin(url, location)
            continue
        detail = (get("X-Error-Code") if callable(get) else None) or ""
        return status, str(detail)
    return 0, "too many redirects"


def check_gated_access(
    token: str,
    *,
    repo_id: str = SAM3_REPO_ID,
    opener: Optional[Callable[..., Any]] = None,
    timeout: float = 15.0,
) -> Dict[str, Any]:
    """Has this account accepted Meta's terms for the gated checkpoint?

    This is *the* friction point of the whole install, so it is checked
    explicitly and reported in words rather than surfacing as a 403 buried in a
    download traceback twenty minutes later.

    It asks for a file of the gated repo (``HEAD .../resolve/main/config.json``),
    which is the only question that has the right answer: the repo's metadata
    at ``/api/models/<repo>`` is public, and answers 200 to no token, a bogus
    token and an unaccepted licence alike.  The file answers 401 for a missing
    or rejected token, 403 for a valid token whose account has not accepted
    the licence, and 200 once it has.
    """
    url = "https://huggingface.co/%s/resolve/main/config.json" % repo_id
    status, detail = _http_head(url, token=token, opener=opener, timeout=timeout)
    if status == 200:
        return {"ok": True, "status": 200, "message": "Access to %s granted." % repo_id}
    if status == 401:
        return {
            "ok": False,
            "status": 401,
            "message": ("HuggingFace did not accept that token for %s. Check that it is "
                        "a current read token, then try again." % repo_id),
            "url": HF_TOKENS_URL,
        }
    if status == 403:
        return {
            "ok": False,
            "status": 403,
            "message": (
                "Your account has not accepted the licence for %s yet (or the request "
                "is still awaiting approval). Open the model page, accept the terms, "
                "then try again." % repo_id
            ),
            "url": SAM3_MODEL_URL,
        }
    if status == 404:
        return {"ok": False, "status": 404, "message": "%s not found on huggingface.co." % repo_id}
    if status == 0:
        return {"ok": False, "status": 0, "message": "Could not reach huggingface.co: %s" % detail}
    return {"ok": False, "status": status, "message": "Unexpected response (HTTP %s)." % status}


#: Files that make a directory look like a usable SAM 3 checkpoint.
_WEIGHT_MARKERS = ("config.json", "preprocessor_config.json")


def inspect_local_weights(path: str) -> Dict[str, Any]:
    """Validate a user-supplied checkpoint directory.

    The second exit from the gated flow (DESIGN.md §7): a user who already
    downloaded the weights points us at the folder.  We check it looks right
    *now* rather than letting the daemon fail at first inference.
    """
    result: Dict[str, Any] = {"ok": False, "path": path, "message": "", "files": []}
    if not path:
        result["message"] = "No path given."
        return result
    path = os.path.expanduser(path)
    result["path"] = path
    if not os.path.isdir(path):
        result["message"] = "Not a directory: %s" % path
        return result
    try:
        entries = sorted(os.listdir(path))
    except OSError as exc:
        result["message"] = "Cannot read %s: %s" % (path, exc)
        return result
    result["files"] = entries
    weights = [e for e in entries if e.endswith((".safetensors", ".bin", ".pth"))]
    has_config = any(m in entries for m in _WEIGHT_MARKERS)
    if not weights:
        # An original Meta checkpoint is a perfectly good thing to be holding;
        # it just needs converting first.  Say so, and let the caller offer it,
        # rather than reporting "no weights here" about a folder full of them.
        original = find_original_checkpoint(path)
        if original:
            result["original_checkpoint"] = original
            result["convertible"] = True
            result["message"] = (
                "That is an original SAM 3 checkpoint (%s), not the HuggingFace "
                "layout this loads. It can be converted -- no HuggingFace "
                "account needed." % os.path.basename(original)
            )
            return result
        result["message"] = "No .safetensors / .bin weight file in that folder."
        return result
    if not has_config:
        result["message"] = (
            "Found weights but no config.json -- point at the folder that contains "
            "the whole checkpoint, not just the weight file."
        )
        return result
    result["ok"] = True
    result["message"] = "Found %d weight file(s) and a config.json." % len(weights)
    return result


def set_local_weights(path: str) -> Dict[str, Any]:
    """Record a validated local checkpoint so the launcher can export
    ``SAM3_WEIGHTS_DIR`` to the daemon."""
    report = inspect_local_weights(path)
    if not report["ok"]:
        raise BootstrapError(report["message"])
    _atomic_write_json(
        weights_config_file(),
        {"local_path": report["path"], "set_at": time.time(), "source": "local"},
    )
    return report


def clear_local_weights() -> None:
    try:
        os.remove(weights_config_file())
    except OSError:
        pass


def local_weights_path() -> Optional[str]:
    """The recorded local checkpoint, or ``None``.  ``SAM3_WEIGHTS_DIR`` in the
    environment wins over the config file."""
    env_path = os.environ.get(ENV_WEIGHTS_DIR)
    if env_path and os.path.isdir(os.path.expanduser(env_path)):
        return os.path.expanduser(env_path)
    obj = _read_json(weights_config_file())
    if isinstance(obj, dict):
        p = obj.get("local_path")
        if isinstance(p, str) and os.path.isdir(p):
            return p
    return None


def weights_environ() -> Dict[str, str]:
    """Environment overlay the launcher should add when spawning the daemon.

    Complements ``sam3gimpd.paths.daemon_environ`` (which sets ``HF_HOME``); this
    only appears when the user took the local-path exit.
    """
    local = local_weights_path()
    return {ENV_WEIGHTS_DIR: local} if local else {}


def weights_present() -> bool:
    """Are the SAM 3 weights already on disk?

    Heuristic and offline by design -- same rule ``tests/conftest.py`` uses, so
    the two agree.  A HuggingFace snapshot lives at
    ``<HF_HOME>/hub/models--facebook--sam3``.
    """
    if local_weights_path():
        return True
    slug = "models--" + SAM3_REPO_ID.replace("/", "--")
    roots = [hf_home(), models_dir()]
    env_hf = os.environ.get("HF_HOME")
    if env_hf:
        roots.append(env_hf)
    roots.append(os.path.join(_home(), ".cache", "huggingface"))
    for root in roots:
        for candidate in (os.path.join(root, slug), os.path.join(root, "hub", slug)):
            if _snapshot_has_weights(candidate):
                return True
    return False


def _snapshot_has_weights(cache_dir: str) -> bool:
    """Does a HuggingFace cache directory hold loadable weights?

    Testing that the directory *exists* was not enough and reported the wrong
    thing to a real user: an aborted or partial download leaves the tree behind,
    and ``hf download facebook/sam3 sam3.pt`` creates a snapshot containing only
    the original checkpoint -- which ``Sam3Model.from_pretrained`` cannot load.
    Both cases claimed "SAM 3 weights are available".

    So look for what actually gets loaded: a ``.safetensors``/``.bin`` beside a
    ``config.json``, in any snapshot revision.  Entries are usually symlinks
    into ``blobs/``; ``os.path.isfile`` follows them, so a broken link is
    correctly treated as absent.
    """
    snapshots = os.path.join(cache_dir, "snapshots")
    if not os.path.isdir(snapshots):
        # Some layouts (and our own models_dir) hold the files directly.
        return inspect_local_weights(cache_dir).get("ok", False)
    try:
        revisions = sorted(os.listdir(snapshots))
    except OSError:
        return False
    for revision in revisions:
        if inspect_local_weights(os.path.join(snapshots, revision)).get("ok"):
            return True
    return False



# --------------------------------------------------------------------------- #
# original Meta checkpoints (sam3.pt) -- the third exit from the gated flow
# --------------------------------------------------------------------------- #

#: The original checkpoint Meta ships, also present in the gated HF repo beside
#: the converted files.  It is *not* loadable by ``Sam3Model.from_pretrained``:
#: the tensor names differ from the HuggingFace layout, and there is no
#: ``config.json`` or tokenizer next to it.
ORIGINAL_CHECKPOINT_NAMES = ("sam3.pt", "sam3.1.pt", "sam3_base.pt")

#: ``transformers``' own converter, run in the daemon's environment.
#:
#: Worth doing rather than sending the user back to the gated download: the
#: converter needs *only* the .pt.  It builds the config itself and takes the
#: tokenizer from ``openai/clip-vit-base-patch32``, which is public.  So a user
#: holding sam3.pt can get a working checkpoint without a HuggingFace account,
#: without accepting Meta's terms again, and without a 3.6 GB download.
#:
#: It has to be fetched: transformers' release process strips conversion
#: scripts, so ``convert_sam3_to_hf.py`` is in no released wheel -- not even
#: at the ``v5.16.1`` tag, whose tree lacks it.  It is fetched from one fixed
#: commit, the last to change the file, which is an ancestor of ``v5.16.1``;
#: that file imports nothing beyond argparse/gc/os/regex/torch and transformers
#: symbols 5.16.1 has, and runs standalone.  It is Apache-2.0.
#:
#: The commit pins *where*; :data:`CONVERT_SCRIPT_SHA256` pins *what*.  The
#: bytes are checked against it after every download and on every read of the
#: cached copy, and nothing that fails the check is ever run: this program
#: runs as the user, in the environment that holds their torch.
CONVERT_SCRIPT_COMMIT = "22278df3198c4219f033f2a4b0931da3e5c21af4"
CONVERT_SCRIPT_URL = (
    "https://raw.githubusercontent.com/huggingface/transformers/%s/"
    "src/transformers/models/sam3/convert_sam3_to_hf.py" % CONVERT_SCRIPT_COMMIT
)
CONVERT_SCRIPT_SHA256 = "d6bf4a6e3703fb677b16e8f6020556ba1d9aff74a6019a67ecc028ea5e3e813e"

#: The converter calls ``torch.load(path, map_location="cpu")`` without
#: ``weights_only``.  From torch 2.6 that defaults to True -- tensors only; on
#: older torch it unpickles arbitrary objects, so a doctored ``.pt`` would run
#: code as the user.  Conversion therefore refuses an older torch.
CONVERT_MIN_TORCH = (2, 6)


def convert_script_cache() -> str:
    """Local copy of the fetched converter."""
    return os.path.join(base_dir(), "tools", "convert_sam3_to_hf.py")


def _converter_verified(body: bytes) -> bool:
    return hashlib.sha256(body).hexdigest() == CONVERT_SCRIPT_SHA256


def _fetch_converter(opener: Optional[Callable[..., Any]] = None) -> bytes:
    if opener is not None:
        return opener(CONVERT_SCRIPT_URL)
    import urllib.request  # noqa: PLC0415

    request = urllib.request.Request(
        CONVERT_SCRIPT_URL, headers={"User-Agent": "sam3-gimp-bootstrap/1"})
    with urllib.request.urlopen(request, timeout=60.0) as resp:
        return resp.read()


def convert_script_source(
    *,
    on_line: Optional[Callable[[str], Any]] = None,
    opener: Optional[Callable[[str], bytes]] = None,
) -> Dict[str, Any]:
    """The pinned converter's **source text**, verified, from cache or the network.

    Returns ``{ok, source, message}``.  ``source`` is only ever text whose
    SHA-256 is :data:`CONVERT_SCRIPT_SHA256`.  A cached copy that does not match
    is fetched again, once; a download that does not match is refused and
    nothing is run.

    Deliberately not a path: handing a subprocess a filename meant the
    interpreter could fail to open a file we had just verified -- reported
    twice, with antivirus the likely culprit.  The text is piped to
    ``python -`` instead, so nothing on disk has to survive between checking it
    and running it.  The cache is a convenience for working offline, and a
    failure to write it is not fatal.

    ``opener(url) -> bytes`` replaces the network in tests.
    """
    cache = convert_script_cache()
    stale = False
    try:
        with open(cache, "rb") as fh:
            cached = fh.read()
    except OSError:
        cached = None
    if cached is not None:
        if _converter_verified(cached):
            return {"ok": True, "source": cached.decode("utf-8"),
                    "message": "using the cached converter (sha256 verified)"}
        stale = True
        if on_line:
            on_line("the cached conversion script does not match its pinned "
                    "checksum; fetching it again")

    if on_line:
        on_line("fetching the SAM 3 conversion script (transformers %s)"
                % CONVERT_SCRIPT_COMMIT[:12])
    try:
        body = _fetch_converter(opener)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "source": None,
                "message": "Could not fetch the conversion script:\n%s: %s\n\n"
                           "Tried: %s" % (type(exc).__name__, exc, CONVERT_SCRIPT_URL)}
    if not _converter_verified(body):
        if stale:
            try:
                os.remove(cache)
            except OSError:
                pass
        return {"ok": False, "source": None,
                "message": "The conversion script that was downloaded does not match "
                           "the checksum this plug-in pins (got %d bytes, sha256 %s...), "
                           "so it was not run. Nothing was changed. Try again later, or "
                           "use the HuggingFace token route above.\n\nFrom: %s"
                           % (len(body), hashlib.sha256(body).hexdigest()[:12],
                              CONVERT_SCRIPT_URL)}
    try:  # best effort only -- the conversion does not depend on it
        _private_dir(os.path.dirname(cache))
        _atomic_write_bytes(cache, body)
    except OSError:
        pass
    return {"ok": True, "source": body.decode("utf-8"),
            "message": "fetched the converter (%d bytes, sha256 verified)" % len(body)}


def convert_checkpoint_command(python: str, checkpoint: str, out_dir: str,
                               script: str = "-") -> List[str]:
    """``argv`` that converts an original ``sam3.pt`` into HuggingFace format.

    ``script`` ``"-"`` (the default) reads the program from stdin, so nothing
    has to exist on disk when python starts; otherwise it is a file to run.
    """
    return [python, script, "--checkpoint_path", checkpoint, "--output_path", out_dir]


def daemon_interpreter() -> Optional[str]:
    """The interpreter the daemon runs in: the user's choice, else the managed
    venv's.  The same order as ``launcher.build_command``.

    Anything that needs torch, transformers or the daemon's own CLI -- the
    conversion, the weights download, Doctor -- runs here and never in GIMP's
    embedded Python.
    """
    chosen, _ = _stored_python()
    if chosen:
        return chosen
    for candidate in (venv_python(), venv_pythonw()):
        if os.path.isfile(candidate):
            return candidate
    return None


def converted_weights_dir() -> str:
    """Where a converted checkpoint is written."""
    return os.path.join(models_dir(), "sam3-converted")


def find_original_checkpoint(path: str) -> Optional[str]:
    """A ``.pt`` at ``path``, or inside it when a directory was given."""
    if not path:
        return None
    path = os.path.expanduser(path)
    if os.path.isfile(path) and path.endswith(".pt"):
        return path
    if os.path.isdir(path):
        try:
            entries = sorted(os.listdir(path))
        except OSError:
            return None
        for name in ORIGINAL_CHECKPOINT_NAMES:
            if name in entries:
                return os.path.join(path, name)
        pts = [e for e in entries if e.endswith(".pt")]
        if len(pts) == 1:
            return os.path.join(path, pts[0])
    return None


def _version_tuple(text: Any) -> Optional[Tuple[int, int]]:
    match = re.match(r"^\s*(\d+)\.(\d+)", str(text or ""))
    return (int(match.group(1)), int(match.group(2))) if match else None


def convert_original_checkpoint(
    checkpoint: str,
    *,
    out_dir: Optional[str] = None,
    python: Optional[str] = None,
    runner: Optional[CommandRunner] = None,
    on_line: Optional[Callable[[str], Any]] = None,
    timeout: float = 1800.0,
    opener: Optional[Callable[[str], bytes]] = None,
    cancel: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """Convert ``sam3.pt`` to HuggingFace format and record the result.

    Returns ``{ok, path, message}``.  Never raises for an ordinary failure --
    a missing converter or an unreadable checkpoint is a message, not a crash.
    ``cancel`` stops the conversion (Setup's Cancel and Close).
    """
    out: Dict[str, Any] = {"ok": False, "path": None, "message": ""}
    checkpoint = os.path.expanduser(checkpoint or "")
    if not os.path.isfile(checkpoint):
        out["message"] = "No such checkpoint: %s" % checkpoint
        return out

    # If the .pt sits in a folder that is *already* a HuggingFace checkpoint,
    # there is nothing to convert.  This is the normal shape of a downloaded
    # snapshot -- huggingface/hub/models--facebook--sam3/snapshots/<rev>/ holds
    # sam3.pt beside config.json, model.safetensors and the tokenizer -- so the
    # user who points at the .pt in their cache is one click from done, not
    # thirty minutes of conversion they never needed.
    folder = os.path.dirname(checkpoint)
    already = inspect_local_weights(folder)
    if already.get("ok"):
        set_local_weights(folder)
        out.update(ok=True, path=folder,
                   message="No conversion needed - that folder is already a "
                           "HuggingFace checkpoint. Now using it.")
        return out

    python = python or daemon_interpreter()
    if not python:
        out["message"] = (
            "No Python environment with torch yet. Install the environment "
            "first, or choose an existing one, then convert."
        )
        return out

    # Check before spending the user's time, and before running anything: the
    # converter needs transformers, and it must not unpickle the checkpoint
    # with a torch that would execute whatever the file contains.
    runner = runner or CommandRunner()
    probe = probe_interpreter(python, runner=runner)
    if not probe.get("ok"):
        out["message"] = ("Could not inspect that environment, so nothing was run:\n%s"
                          % (probe.get("error") or "no answer"))
        return out
    if not probe.get("torch"):
        out["message"] = "There is no torch in that environment, and conversion needs it."
        return out
    torch_version = _version_tuple(probe.get("torch"))
    if torch_version is None or torch_version < CONVERT_MIN_TORCH:
        out["message"] = (
            "That environment has torch %s. The converter loads sam3.pt with "
            "torch.load, which before torch %d.%d unpickles arbitrary objects by "
            "default -- a checkpoint from anywhere but Meta could run code as you. "
            "Upgrade torch to %d.%d or newer (Setup's own environment uses 2.9), "
            "then convert." % ((probe.get("torch"),) + CONVERT_MIN_TORCH + CONVERT_MIN_TORCH)
        )
        return out
    if not probe.get("transformers"):
        out["message"] = (
            "transformers is not installed in that environment, and the "
            "converter is part of transformers.\n\n"
            "Press 'Install missing packages here' on the Install tab "
            "first -- it leaves your torch alone -- then convert."
        )
        return out

    fetched = convert_script_source(on_line=on_line, opener=opener)
    if not fetched["ok"]:
        out["message"] = fetched["message"]
        return out

    out_dir = out_dir or converted_weights_dir()
    os.makedirs(os.path.dirname(out_dir) or ".", exist_ok=True)
    argv = convert_checkpoint_command(python, checkpoint, out_dir)
    result = runner.run(argv, on_line=on_line, timeout=timeout,
                        input_text=fetched["source"], cancel=cancel)
    if cancel is not None and cancel.is_set():
        out["message"] = "Conversion cancelled."
        return out
    if not result.ok:
        out["message"] = (
            "Conversion failed (exit %s).\n%s" % (result.returncode, result.tail(20))
        )
        return out

    report = inspect_local_weights(out_dir)
    if not report["ok"]:
        out["message"] = ("Conversion reported success but %s does not look like "
                          "a checkpoint: %s" % (out_dir, report["message"]))
        return out
    set_local_weights(out_dir)
    out.update(ok=True, path=out_dir,
               message="Converted to %s and now in use." % out_dir)
    return out


# --------------------------------------------------------------------------- #
# environment report (what the Doctor panel and the main dialog both need)
# --------------------------------------------------------------------------- #
def pid_alive(pid: int) -> bool:
    """API.md §3.3 step 2.  ``ctypes`` is stdlib, so the plug-in may use it.

    Same answers as ``launcher.pid_alive``: a process that exists but belongs
    to someone else is alive, and on Windows an exited process that something
    still holds a handle to is not.
    """
    if not pid or pid <= 0:
        return False
    if platform_key() == "windows":
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            ERROR_INVALID_PARAMETER = 87  # what OpenProcess reports for "no such process"
            STILL_ACTIVE = 259
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not handle:
                return ctypes.get_last_error() != ERROR_INVALID_PARAMETER  # type: ignore[attr-defined]
            try:
                code = wintypes.DWORD()
                if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return code.value == STILL_ACTIVE
                return True
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # alive, owned by someone else
    except OSError:
        return False


def read_runtime() -> Optional[Dict[str, Any]]:
    """``runtime.json`` if it has the five required fields (API.md §3.2).

    Unknown keys are ignored, as the contract demands; a file missing any
    required field is treated as absent.
    """
    obj = _read_json(runtime_file())
    if not isinstance(obj, dict):
        return None
    required = ("port", "token", "pid", "version", "started_at")
    if not all(k in obj for k in required):
        return None
    try:
        obj["port"] = int(obj["port"])
        obj["pid"] = int(obj["pid"])
        obj["started_at"] = float(obj["started_at"])
    except (TypeError, ValueError):
        return None
    if not isinstance(obj.get("token"), str) or not obj["token"]:
        return None
    return obj


@dataclass
class EnvironmentReport:
    """Everything the setup dialog needs to decide what to show first."""

    platform: str
    machine: str
    base: str
    accelerator: Accelerator
    uv_present: bool
    venv_present: bool
    sam3d_present: bool
    weights_present: bool
    weights_source: Optional[str]
    daemon_running: bool
    daemon_pid: Optional[int]
    daemon_port: Optional[int]
    runtime_present: bool
    crash_log_present: bool
    state: InstallState
    paths: Dict[str, str] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    #: An interpreter the user chose in Setup, if it still exists on disk.
    external_python: Optional[str] = None
    #: Whether that interpreter had the daemon installed when it was accepted.
    external_has_daemon: bool = False

    @property
    def external_ready(self) -> bool:
        """The user pointed us at their own environment and it had the daemon.

        Recorded when they accepted it, rather than re-probed here: this is read
        during procedure *registration*, and spawning a subprocess to answer it
        would delay every GIMP start.  A stale answer costs a fast, explicit
        spawn failure, which is a far better trade than a slow menu.
        """
        return bool(self.external_python) and self.external_has_daemon

    @property
    def managed_ready(self) -> bool:
        """The environment Setup builds for itself is complete."""
        return self.uv_present and self.venv_present and self.sam3d_present

    @property
    def env_ready(self) -> bool:
        """True when a daemon could be spawned at all.

        Either half counts.  Only checking the managed venv was a real bug: a
        user who chose their own PyTorch environment was told to run Setup
        forever, because the venv Setup would have built did not exist and never
        would.  Weights are a separate axis -- ``--stub`` is useful without them.
        """
        return self.managed_ready or self.external_ready

    @property
    def fully_ready(self) -> bool:
        return self.env_ready and self.weights_present

    def summary(self) -> str:
        # Name *which* environment is in use.  "Ready" while pointing at an
        # interpreter the user chose, with no hint that the managed venv is not
        # involved, is how a working setup still looks broken.
        where = " (your own environment)" if self.external_ready and not self.managed_ready else ""
        if self.fully_ready:
            return "Ready: sam3gimpd installed (%s)%s, weights present." % (
                self.accelerator.kind, where)
        if self.env_ready:
            return "sam3gimpd installed (%s)%s; SAM 3 weights are missing." % (
                self.accelerator.kind, where)
        if self.external_python and not self.external_has_daemon:
            return ("Your chosen environment does not have sam3gimpd installed "
                    "-- use 'Install sam3gimpd here'.")
        if self.venv_present:
            return "Environment half-built -- resume the install."
        return "Not installed yet."

    def to_json(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "machine": self.machine,
            "base": self.base,
            "accelerator": self.accelerator.kind,
            "accelerator_reason": self.accelerator.reason,
            "uv_present": self.uv_present,
            "venv_present": self.venv_present,
            "sam3d_present": self.sam3d_present,
            "weights_present": self.weights_present,
            "weights_source": self.weights_source,
            "daemon_running": self.daemon_running,
            "daemon_pid": self.daemon_pid,
            "daemon_port": self.daemon_port,
            "runtime_present": self.runtime_present,
            "crash_log_present": self.crash_log_present,
            "env_ready": self.env_ready,
            "fully_ready": self.fully_ready,
            "missing": list(self.missing),
            "completed": list(self.state.completed),
            "summary": self.summary(),
            "paths": dict(self.paths),
        }


def inspect_environment(
    *,
    accelerator: Optional[Accelerator] = None,
    runner: Optional[CommandRunner] = None,
) -> EnvironmentReport:
    """Read-only probe of the whole install.  Touches no network and spawns
    nothing except (possibly) ``nvidia-smi -L``."""
    accel = accelerator or detect_accelerator(runner=runner)
    external, external_daemon = _stored_python()
    uv_ok = os.path.exists(uv_binary())
    venv_ok = os.path.exists(venv_python())
    sam3d_ok = os.path.exists(sam3d_executable())
    local = local_weights_path()
    weights_ok = weights_present()
    runtime = read_runtime()
    daemon_pid = int(runtime["pid"]) if runtime else None
    daemon_running = bool(runtime) and pid_alive(daemon_pid or 0)

    if external and external_daemon:
        # Nothing about the managed venv is missing, because it is not the
        # environment being used.
        missing = [] if weights_ok else ["weights"]
    else:
        missing = [
            key
            for key, ok in (("uv", uv_ok), ("venv", venv_ok),
                            ("sam3gimpd", sam3d_ok), ("weights", weights_ok))
            if not ok
        ]
    return EnvironmentReport(
        platform=platform_key(),
        machine=machine_key(),
        base=base_dir(),
        accelerator=accel,
        external_python=external,
        external_has_daemon=external_daemon,
        uv_present=uv_ok,
        venv_present=venv_ok,
        sam3d_present=sam3d_ok,
        weights_present=weights_ok,
        weights_source=("local:%s" % local) if local else ("hf" if weights_ok else None),
        daemon_running=daemon_running,
        daemon_pid=daemon_pid,
        daemon_port=int(runtime["port"]) if runtime else None,
        runtime_present=runtime is not None,
        crash_log_present=os.path.exists(crash_log()),
        state=load_state(),
        paths=describe_paths(),
        missing=missing,
    )


# --------------------------------------------------------------------------- #
# the installer
# --------------------------------------------------------------------------- #
@dataclass
class InstallOutcome:
    ok: bool
    completed: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    failed_step: Optional[str] = None
    error: Optional[str] = None
    cancelled: bool = False
    duration_s: float = 0.0
    verify: Optional[Dict[str, Any]] = None


class Installer:
    """Executes an :class:`InstallPlan`, journalling as it goes.

    Runs on a **worker thread** (the GTK main loop must never block --
    DESIGN.md §4); the callbacks are invoked from that thread and the dialog
    marshals them back with ``GLib.idle_add``.

    ``cancel`` is the Event this run watches -- between steps, inside every
    subprocess and inside the uv download.  The caller keeps a reference to
    the same object to press Cancel with.  ``journal=False`` runs without
    reading or writing ``install-state.json``, for flows (the weights
    download) that are not part of the install.
    """

    def __init__(
        self,
        plan: InstallPlan,
        *,
        runner: Optional[CommandRunner] = None,
        downloader: Optional[Downloader] = None,
        on_log: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[float, str], None]] = None,
        on_step: Optional[Callable[[str, str], None]] = None,
        cancel: Optional[threading.Event] = None,
        state: Optional[InstallState] = None,
        env: Optional[Dict[str, str]] = None,
        log_path: Optional[str] = None,
        journal: bool = True,
    ) -> None:
        self.plan = plan
        self.runner = runner or CommandRunner()
        self.downloader = downloader or Downloader()
        self.on_log = on_log or (lambda _line: None)
        self.on_progress = on_progress or (lambda _frac, _msg: None)
        self.on_step = on_step or (lambda _key, _state: None)
        self.cancel = cancel if cancel is not None else threading.Event()
        self.journal = journal
        if state is not None:
            self.state = state
        else:
            self.state = load_state(plan) if journal else InstallState()
        self.env = dict(env or {})
        self.log_path = log_path or install_log()
        self._log_handle = None

    # -- logging ---------------------------------------------------------- #
    def log(self, line: str) -> None:
        self.on_log(line)
        if self._log_handle is not None:
            try:
                self._log_handle.write(line + "\n")
                self._log_handle.flush()
            except Exception:
                pass

    def _save(self) -> None:
        if self.journal:
            save_state(self.state)

    # -- main loop -------------------------------------------------------- #
    def run(self) -> InstallOutcome:
        started = time.time()
        ensure_layout()
        if self.journal and self.plan.step("torch") is not None:
            # A venv built before the torch marker existed: record what the
            # old journal knew before this run overwrites it.
            if _read_json(torch_marker_file()) is None:
                legacy = _legacy_torch_extra()
                if legacy:
                    _write_torch_marker(legacy, torch_index_url(legacy))
        self.state.fingerprint = plan_fingerprint(self.plan)
        self.state.accelerator = self.plan.accelerator.kind
        self.state.extra = self.plan.extra
        if not self.state.started_at:
            self.state.started_at = started
        completed: List[str] = []
        skipped: List[str] = []
        verify_info: Optional[Dict[str, Any]] = None

        try:
            self._log_handle = _open_private_append(self.log_path)
        except OSError:
            self._log_handle = None

        try:
            self.log("=" * 68)
            self.log("sam3-gimp bootstrap  %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
            self.log(
                "platform=%s machine=%s accelerator=%s (%s) extra=%s"
                % (
                    platform_key(),
                    machine_key(),
                    self.plan.accelerator.kind,
                    self.plan.accelerator.reason,
                    self.plan.extra,
                )
            )
            self.log("base=%s" % base_dir())
            if self.plan.index_url:
                self.log("torch index=%s" % self.plan.index_url)

            done_weight = 0.0
            total = self.plan.total_weight()

            for step in self.plan.steps:
                if self.cancel.is_set():
                    raise CancelledError("cancelled before %s" % step.key)

                # The probe alone decides (see InstallStep): the journal can
                # say "done" about files that are gone, and verify has no
                # probe, so it always runs.
                if step.satisfied is not None and step.is_satisfied():
                    skipped.append(step.key)
                    self.on_step(step.key, "skipped")
                    self.log("[skip] %s (already done)" % step.title)
                    done_weight += step.weight
                    self.on_progress(min(1.0, done_weight / total), step.title)
                    self.state.mark(step.key)
                    self._save()
                    continue

                self.on_step(step.key, "running")
                self.log("[run ] %s" % step.title)
                base_frac = done_weight / total
                span = step.weight / total

                if step.kind == "download":
                    self._do_download(step, base_frac, span)
                else:
                    result = self._do_command(step, base_frac, span)
                    if step.key == "verify":
                        verify_info = _parse_verify(result.text)
                if step.after is not None:
                    step.after()

                done_weight += step.weight
                completed.append(step.key)
                self.state.mark(step.key)
                self.state.last_error = None
                self._save()
                self.on_step(step.key, "done")
                self.on_progress(min(1.0, done_weight / total), step.title)

            self.on_progress(1.0, "Done")
            if self.plan.step("venv") is not None:
                self.log("[ok  ] environment ready at %s" % venv_root())
            else:
                self.log("[ok  ] done")
            return InstallOutcome(
                True, completed, skipped, None, None, False, time.time() - started, verify_info
            )

        except CancelledError as exc:
            self.state.last_error = str(exc)
            self._save()
            self.log("[stop] %s" % exc)
            return InstallOutcome(
                False, completed, skipped, self.state.last_step, str(exc), True, time.time() - started
            )
        except BootstrapError as exc:
            self.state.last_error = str(exc)
            self._save()
            self.log("[fail] %s" % exc)
            failed = getattr(exc, "step_key", None) or self.state.last_step
            return InstallOutcome(
                False, completed, skipped, failed, str(exc), False, time.time() - started
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.state.last_error = repr(exc)
            self._save()
            self.log("[fail] unexpected: %r" % exc)
            return InstallOutcome(
                False, completed, skipped, self.state.last_step, repr(exc), False, time.time() - started
            )
        finally:
            if self._log_handle is not None:
                try:
                    self._log_handle.close()
                except Exception:
                    pass
                self._log_handle = None

    # -- step kinds ------------------------------------------------------- #
    def _do_download(self, step: InstallStep, base_frac: float, span: float) -> None:
        """Download uv, check it against the pinned hash, install the binary."""
        self.log("       %s" % step.url)

        def _progress(frac: float, done: int, total_bytes: int) -> None:
            self.on_progress(min(1.0, base_frac + span * frac), "%s (%s)" % (step.title, _human(done)))

        try:
            fetch_uv(step.url, step.dest, downloader=self.downloader, on_progress=_progress,
                     on_log=self.log, cancel=self.cancel)
        except CancelledError:
            raise
        except BootstrapError as exc:
            setattr(exc, "step_key", step.key)
            raise
        self.log("       -> %s" % step.dest)

    def _do_command(self, step: InstallStep, base_frac: float, span: float) -> CommandResult:
        env = dict(self.env)
        env.update(step.env)
        # Keep the HF cache inside our base dir, so the Doctor panel can show
        # and clear one location (DESIGN.md §7) -- unless the user set HF_HOME
        # themselves, which the launcher honours for the daemon too: weights
        # downloaded anywhere else would be invisible to it.
        if not os.environ.get("HF_HOME"):
            env.setdefault("HF_HOME", hf_home())
        env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault(ENV_HOME, base_dir())
        env.setdefault("UV_CACHE_DIR", os.path.join(cache_dir(), "uv"))

        self.log("       $ %s" % _redact(step.argv))
        pulse = _Pulse(self.on_progress, base_frac, span, step.title)

        def _line(text: str) -> None:
            self.log("       " + text)
            pulse.tick()

        result = self.runner.run(step.argv, env=env, on_line=_line, cancel=self.cancel)
        if self.cancel.is_set():
            raise CancelledError("cancelled during %s" % step.key)
        if not result.ok:
            err = BootstrapError(
                "%s failed (exit %d).\n%s" % (step.title, result.returncode, result.tail(12))
            )
            setattr(err, "step_key", step.key)
            raise err
        return result


class _Pulse:
    """Fake-but-honest progress for a step whose subprocess gives no percentage.

    Approaches (but never reaches) the end of its slice as output arrives, so
    the bar always moves during a long ``uv pip install`` without ever lying
    about being finished.
    """

    def __init__(self, on_progress, base: float, span: float, title: str) -> None:
        self._on_progress = on_progress
        self._base = base
        self._span = span
        self._title = title
        self._n = 0
        self._last = 0.0

    def tick(self) -> None:
        self._n += 1
        frac = 1.0 - (1.0 / (1.0 + self._n / 40.0))  # 0 -> ~1, asymptotic
        now = time.time()
        if now - self._last < 0.1:
            return
        self._last = now
        self._on_progress(self._base + self._span * frac * 0.95, self._title)


def _safe_tmp_dir() -> str:
    try:
        os.makedirs(cache_dir(), exist_ok=True)
        return cache_dir()
    except OSError:
        return tempfile.gettempdir()


def _human(nbytes: int) -> str:
    value = float(nbytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return "%.0f %s" % (value, unit) if unit == "B" else "%.1f %s" % (value, unit)
        value /= 1024.0
    return "%.1f GB" % value


_SECRET_RE = re.compile(r"(hf_[A-Za-z0-9]{6})[A-Za-z0-9]+")


def _redact(argv: Sequence[str]) -> str:
    """Never let a token reach a log file, on any code path."""
    return " ".join(_SECRET_RE.sub(r"\1...", str(a)) for a in argv)


def _parse_verify(text: str) -> Optional[Dict[str, Any]]:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("SAM3D_VERIFY "):
            try:
                return json.loads(line[len("SAM3D_VERIFY ") :])
            except ValueError:
                return None
    return None


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #
def parse_doctor_output(text: str) -> Optional[Dict[str, Any]]:
    """Extract the JSON object from ``sam3gimpd doctor`` output.

    Tolerant on purpose: warnings a library prints around the report (torch
    and transformers both do, on import) must not hide it, so output that
    merely *contains* a JSON object is still understood, and anything else
    yields ``None`` rather than an error.
    """
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except ValueError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except ValueError:
            return None
    return None


def doctor_view(report: Dict[str, Any]) -> Dict[str, Any]:
    """One flat dict for the Doctor panel from ``sam3gimpd doctor``'s report.

    The report nests what the installed engine could do under ``engine`` and a
    running daemon's ``/status`` under ``daemon.status`` (API.md §6.7); the
    panel reads one level.  The running daemon's own answer wins over the
    offline probe.  A report that is already flat passes through unchanged.
    """
    if not isinstance(report, dict):
        return {}
    engine = report.get("engine")
    running = report.get("daemon")
    if not isinstance(engine, dict) and not isinstance(running, dict):
        return dict(report)
    view = {k: v for k, v in report.items() if k not in ("engine", "daemon", "paths")}
    if isinstance(engine, dict):
        view.update(engine)
    status = None
    if isinstance(running, dict) and running.get("reachable"):
        status = running.get("status")
    if isinstance(status, dict):
        view.update(status)
    view["daemon_reachable"] = isinstance(status, dict)
    return view



# --------------------------------------------------------------------------- #
# "use the Python I already have" -- probing a user-chosen interpreter
# --------------------------------------------------------------------------- #

#: Emitted by :func:`probe_interpreter` inside the *target* interpreter.  It has
#: to survive an interpreter with no torch, no sam3gimpd and no network, so every
#: lookup is individually guarded and the result is a single JSON line.
_PROBE_SCRIPT = (
    "import json,sys\n"
    "d={'python_version':'.'.join(str(v) for v in sys.version_info[:3]),"
    "'executable':sys.executable}\n"
    "try:\n"
    " import torch\n"
    " d['torch']=getattr(torch,'__version__',None)\n"
    " try:\n"
    "  d['cuda']=bool(torch.cuda.is_available())\n"
    "  if d['cuda']:\n"
    "   d['device_name']=torch.cuda.get_device_name(0)\n"
    "   c=torch.cuda.get_device_capability(0); d['capability']=[c[0],c[1]]\n"
    " except Exception as e:\n"
    "  d['cuda_error']=str(e)\n"
    "except Exception as e:\n"
    " d['torch']=None; d['torch_error']=str(e)\n"
    "try:\n"
    " import sam3gimpd\n"
    " d['sam3gimpd']=getattr(sam3gimpd,'__version__','unknown')\n"
    "except Exception:\n"
    " d['sam3gimpd']=None\n"
    "try:\n"
    " import transformers\n"
    " d['transformers']=getattr(transformers,'__version__',None)\n"
    "except Exception:\n"
    " d['transformers']=None\n"
    "sys.stdout.write('SAM3PROBE'+json.dumps(d))\n"
)


def probe_interpreter(
    python_path: str, *, runner: Optional[CommandRunner] = None, timeout: float = 60.0
) -> Dict[str, Any]:
    """Ask an interpreter what it has: Python version, torch, CUDA, sam3gimpd.

    Used by **Setup ▸ Install ▸ Use an existing Python environment** so the user
    finds out *before* committing that the interpreter they picked has no torch,
    or has a CPU-only build, or is missing ``sam3gimpd``.

    Never raises: an unusable interpreter is a normal answer here, reported as
    ``ok: False`` with a human-readable ``error``.
    """
    out: Dict[str, Any] = {"ok": False, "path": python_path, "error": None}
    if not python_path:
        out["error"] = "No interpreter selected."
        return out
    if not os.path.isfile(python_path):
        out["error"] = "No such file: %s" % python_path
        return out

    runner = runner or CommandRunner()
    try:
        result = runner.run([python_path, "-c", _PROBE_SCRIPT], timeout=timeout)
    except Exception as exc:  # noqa: BLE001 -- a broken exe is a normal answer
        out["error"] = "Could not run it: %s: %s" % (type(exc).__name__, exc)
        return out

    text = result.text or ""
    marker = text.rfind("SAM3PROBE")
    if marker < 0:
        out["error"] = (
            "That does not look like a Python interpreter "
            "(it produced no probe output).\n%s" % text.strip()[-400:]
        )
        return out
    try:
        out.update(json.loads(text[marker + len("SAM3PROBE"):].strip()))
    except Exception as exc:  # noqa: BLE001
        out["error"] = "Unreadable probe output: %s" % exc
        return out
    out["ok"] = True
    return out


def describe_interpreter(probe: Dict[str, Any]) -> str:
    """One-line human summary of :func:`probe_interpreter`, for the dialog."""
    if not probe.get("ok"):
        return probe.get("error") or "Unusable."
    bits = ["Python %s" % probe.get("python_version", "?")]
    torch_version = probe.get("torch")
    if torch_version:
        if probe.get("cuda"):
            name = probe.get("device_name") or "CUDA device"
            cap = probe.get("capability")
            suffix = " (sm_%d%d)" % (cap[0], cap[1]) if cap else ""
            bits.append("torch %s on %s%s" % (torch_version, name, suffix))
        else:
            bits.append("torch %s, CPU only" % torch_version)
    else:
        bits.append("no torch")
    bits.append("transformers %s" % probe["transformers"]
                if probe.get("transformers") else "no transformers")
    bits.append("sam3gimpd %s" % probe["sam3gimpd"]
                if probe.get("sam3gimpd") else "sam3gimpd not installed")
    return " - ".join(bits)


def interpreter_problems(probe: Dict[str, Any]) -> List[str]:
    """Blocking and advisory problems with a chosen interpreter, worst first.

    Separated from :func:`describe_interpreter` so the dialog can refuse to save
    a hopeless choice while still merely warning about a workable-but-slow one.
    """
    problems: List[str] = []
    if not probe.get("ok"):
        return [probe.get("error") or "Unusable interpreter."]
    version = str(probe.get("python_version") or "0.0")
    try:
        major, minor = (int(p) for p in version.split(".")[:2])
    except Exception:  # noqa: BLE001
        major, minor = 0, 0
    if (major, minor) < MIN_INTERPRETER:
        problems.append("Python %s is too old; the daemon needs %d.%d or newer."
                        % ((version,) + MIN_INTERPRETER))
    if not probe.get("torch"):
        problems.append(
            "No torch in this environment. Install PyTorch there first, or let "
            "Setup build its own environment instead."
        )
    elif not probe.get("cuda"):
        problems.append(
            "torch is present but reports no CUDA device, so segmentation will "
            "run on the CPU and be slow."
        )
    if not probe.get("sam3gimpd"):
        problems.append(
            "The sam3gimpd daemon is not installed in this environment yet - use "
            "'Install sam3gimpd here', which is a small pure-Python install and "
            "does not touch torch."
        )
    elif not probe.get("transformers"):
        # Reachable on an environment set up before the [runtime] extra existed:
        # the daemon is there, imports fine, sees CUDA, and then fails every
        # encode with "transformers is not installed".
        problems.append(
            "transformers is missing, so the model cannot be loaded. Press "
            "'Install sam3gimpd here' again - it now pulls transformers too, and "
            "still leaves torch alone."
        )
    return problems


def interpreter_can_serve(probe: Dict[str, Any]) -> bool:
    """Could a daemon in this environment actually load SAM 3?

    Both halves are required.  Recording only "is sam3gimpd importable" let an
    environment with no transformers count as set up, so the plug-in opened its
    segmentation dialog and only failed once the user had typed a prompt.
    """
    return bool(probe.get("ok") and probe.get("sam3gimpd") and probe.get("transformers"))



def install_button_state(probe: Dict[str, Any]) -> Tuple[bool, str]:
    """``(enabled, label)`` for the install button on the existing-env row.

    Enabled whenever torch is present -- including when the environment can
    already serve.  It was disabled in that state, which left no way to *update*
    the daemon after pulling a new version: the user did every step and the
    old daemon kept running, visibly, with the old idle TTL in its log.
    """
    if not probe.get("ok") or not probe.get("torch"):
        return False, "Install sam3gimpd here"
    if not probe.get("sam3gimpd"):
        return True, "Install sam3gimpd here"
    if not interpreter_can_serve(probe):
        return True, "Install missing packages here"
    return True, "Reinstall / update sam3gimpd here"

def install_sam3d_command(python_path: str, source: Optional[str] = None, *,
                          uv: Optional[str] = None) -> List[str]:
    """``argv`` that installs (or updates) the daemon into an interpreter the
    user chose.

    ``uv pip install --python <python>`` whenever a uv is available (``uv``,
    else :func:`find_uv`), because an environment uv created has no pip at all
    -- ``python -m pip`` there fails with "No module named pip".  Otherwise
    ``python -m pip``; :func:`resolve_daemon_install_command` makes sure pip is
    really there before settling for it.

    The ``[runtime]`` extra is the whole point.  It pulls transformers and the
    rest, but **not** torch or torchvision, so a working CUDA install is left
    exactly as it is while the daemon still gets what it needs to import
    ``Sam3Model``.  Installing with no extra at all -- which this used to do,
    precisely to avoid touching torch -- skipped transformers too, and the
    daemon then started, saw CUDA, and failed every encode with
    ``engine_unavailable: transformers is not installed``.

    With uv, ``--reinstall-package`` rather than ``--upgrade``: the daemon is
    reinstalled even at an unchanged version, and nothing else in the user's
    environment is upgraded along the way.  pip's ``--upgrade`` already only
    touches what it must.
    """
    src = source or daemon_source()
    requirement = "%s[runtime]" % src
    uv = uv or find_uv()
    if uv:
        return [uv, "pip", "install", "--python", python_path,
                "--reinstall-package", DAEMON_DIST_NAME, requirement]
    return [python_path, "-m", "pip", "install", "--upgrade", requirement]


def interpreter_has_pip(python_path: str, *, runner: Optional[CommandRunner] = None,
                        timeout: float = 60.0) -> bool:
    """Can ``python -m pip`` run in this interpreter?"""
    try:
        result = (runner or CommandRunner()).run(
            [python_path, "-m", "pip", "--version"], timeout=timeout)
    except Exception:  # noqa: BLE001
        return False
    return result.ok


def resolve_daemon_install_command(
    python_path: str,
    *,
    source: Optional[str] = None,
    runner: Optional[CommandRunner] = None,
    downloader: Optional[Downloader] = None,
    on_log: Optional[Callable[[str], None]] = None,
    cancel: Optional[threading.Event] = None,
) -> List[str]:
    """:func:`install_sam3d_command`, after making sure its tool exists.

    A uv already here wins; with none, the environment's own pip if it has
    one; with neither, Setup's pinned uv is downloaded and verified first.
    May spawn a probe and download, so it belongs on a worker thread.
    """
    uv = find_uv()
    if uv is None and not interpreter_has_pip(python_path, runner=runner):
        if on_log:
            on_log("that environment has no pip and there is no uv yet; fetching uv %s"
                   % UV_VERSION)
        uv = ensure_uv(downloader=downloader, on_log=on_log, cancel=cancel)
    return install_sam3d_command(python_path, source, uv=uv)


#: The daemon ships *inside* the plug-in directory, at ``sam3_gimp/_daemon``,
#: and is installed from that path.
#:
#: Never by bare name: this project is not published, and ``sam3d`` on PyPI
#: belongs to an unrelated project ("Unified Python interface for Meta SAM 3D
#: Body and SAM 3D Objects"), so ``pip install sam3d`` silently fetches a
#: stranger's package -- into the user's *own* environment, in the "use an
#: existing environment" flow.
#:
#: It lives inside the plug-in rather than beside it, and is committed there
#: rather than copied in by a build step, so that the directory GIMP loads is
#: always complete.  When the daemon sat at the top level of the repository,
#: copying ``plugin/sam3_gimp`` by hand -- the obvious thing to do -- produced an
#: install with nothing to install from, and the only way out was a terminal.
BUNDLED_DAEMON_DIRNAME = "_daemon"


class DaemonSourceMissing(RuntimeError):
    """The bundled daemon source is not next to the plug-in."""


def daemon_source(env: Optional[Dict[str, str]] = None) -> str:
    """Resolve where to install the daemon *from*.

    Order: ``$SAM3D_SOURCE`` (a developer override), then ``<plug-in>/_daemon``
    as shipped, then a path recorded in Setup.  The bundled copy outranks the
    recorded one: the record exists only to rescue a plug-in tree that arrived
    without ``_daemon``, and once a complete release has been extracted over
    it, a remembered old checkout must not keep feeding "Update daemon" and
    the build-hash check stale code.  There is deliberately no fall-through to
    a bare requirement name -- see :data:`BUNDLED_DAEMON_DIRNAME`.
    """
    env = os.environ if env is None else env
    override = env.get("SAM3D_SOURCE")
    if override:
        return override

    bundled = bundled_daemon_dir()
    if os.path.isfile(os.path.join(bundled, "pyproject.toml")):
        return bundled

    # A path the user pointed us at in Setup, for a plug-in tree that was copied
    # by hand and so never got the bundled _daemon/ directory.
    stored = _stored_daemon_source()
    if stored:
        return stored

    raise DaemonSourceMissing(
        "This copy of the plug-in is incomplete: its bundled daemon is "
        "missing.\n\nExpected it at\n  %s\n\n"
        "The _daemon folder ships inside the plug-in, so this normally means "
        "only part of the plug-in was copied. Re-extract the release zip (or "
        "re-copy the whole sam3_gimp folder) into GIMP's plug-ins directory "
        "and restart GIMP." % bundled
    )



def validate_daemon_source(path: str) -> Dict[str, Any]:
    """Does ``path`` look like this project's ``_daemon`` directory?

    Checked rather than assumed, because the alternative is a chooser that
    accepts any folder and fails much later inside pip with something opaque.
    """
    out: Dict[str, Any] = {"ok": False, "path": path, "error": None}
    if not path:
        out["error"] = "No folder selected."
        return out
    if not os.path.isdir(path):
        out["error"] = "Not a folder: %s" % path
        return out
    if not os.path.isfile(os.path.join(path, "pyproject.toml")):
        out["error"] = ("No pyproject.toml in that folder.\nPick the '_daemon' "
                        "folder itself (inside sam3_gimp in the release zip or "
                        "the repository).")
        return out
    pkg = os.path.join(path, "sam3gimpd", "__init__.py")
    if not os.path.isfile(pkg):
        out["error"] = ("That folder has a pyproject.toml but no sam3gimpd "
                        "package, so it is not this project's daemon.")
        return out
    out["ok"] = True
    return out


def bundled_daemon_dir() -> str:
    """Where the daemon source lives once bundled beside the plug-in."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        BUNDLED_DAEMON_DIRNAME)


#: The daemon's distribution name, for ``uv pip install --reinstall-package``.
DAEMON_DIST_NAME = "sam3-gimp-daemon"


def bundled_daemon_version() -> str:
    """``version = "..."`` from the bundled ``pyproject.toml``, or ``""``."""
    try:
        source = daemon_source()
        with open(os.path.join(source, "pyproject.toml"), "r", encoding="utf-8") as fh:
            match = re.search(r'^version\s*=\s*"([^"]+)"', fh.read(), re.M)
        return match.group(1) if match else ""
    except Exception:  # noqa: BLE001
        return ""


def _hash_py_tree(root: str) -> str:
    """Same walk as ``sam3gimpd.build_hash``; the two must agree byte for byte."""
    digest = hashlib.sha1()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), root).replace(os.sep, "/")
            digest.update(rel.encode("utf-8"))
            try:
                with open(os.path.join(dirpath, name), "rb") as fh:
                    digest.update(fh.read())
            except OSError:
                digest.update(b"?")
    return digest.hexdigest()[:8]


def bundled_daemon_build() -> str:
    """Build hash of the daemon source the plug-in ships, computed exactly as
    the daemon computes its own for ``/hello``.  Equal hashes mean the running
    daemon *is* this code; different ones mean "update the daemon".  ``""``
    when there is no bundled source to hash."""
    try:
        pkg = os.path.join(daemon_source(), "sam3gimpd")
        if not os.path.isdir(pkg):
            return ""
        return _hash_py_tree(pkg)
    except Exception:  # noqa: BLE001
        return ""


def daemon_update_command(
    *,
    source: Optional[str] = None,
    runner: Optional[CommandRunner] = None,
    downloader: Optional[Downloader] = None,
    on_log: Optional[Callable[[str], None]] = None,
    cancel: Optional[threading.Event] = None,
) -> Optional[List[str]]:
    """``argv`` that reinstalls the daemon into whichever environment is in use.

    The interpreter the user chose in Setup wins
    (:func:`resolve_daemon_install_command`: uv, else its own pip, else a
    freshly fetched uv); otherwise the managed venv via its uv.  Either way it
    is the daemon's ``[runtime]`` extra, which never names torch, with
    ``--reinstall-package`` so an unchanged version number still gets the new
    source.  ``None`` when nothing is installed yet.  May probe and download,
    so call it on a worker thread.
    """
    python, _had = _stored_python()
    if python:
        return resolve_daemon_install_command(
            python, source=source, runner=runner, downloader=downloader,
            on_log=on_log, cancel=cancel)
    if not (os.path.exists(venv_python()) and os.path.exists(uv_binary())):
        return None
    return uv_install_command(uv_binary(), venv_python(), extra="runtime", source=source,
                              reinstall=(DAEMON_DIST_NAME,))


def install_daemon_source(path: str) -> Dict[str, Any]:
    """Copy a validated daemon source into ``<plug-in>/_daemon``.

    Copying rather than merely remembering the path is deliberate: a remembered
    path breaks the moment the user moves or deletes their checkout, and the
    whole point of bundling is that the installed plug-in is self-contained.
    If the plug-in directory is not writable we fall back to recording the
    path in ``settings.json``, which still works and is better than refusing.
    """
    check = validate_daemon_source(path)
    if not check["ok"]:
        return check

    dest = bundled_daemon_dir()
    try:
        if os.path.abspath(path) == os.path.abspath(dest):
            return {"ok": True, "path": dest, "copied": False, "error": None}
        import shutil  # noqa: PLC0415

        if os.path.isdir(dest):
            shutil.rmtree(dest)
        shutil.copytree(
            path, dest,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo",
                                          ".git", ".pytest_cache", "*.egg-info"),
        )
        return {"ok": True, "path": dest, "copied": True, "error": None}
    except Exception as exc:  # noqa: BLE001
        # Read-only plug-in directory (a packaged or shared install): remember
        # the path instead, which daemon_source() honours.
        try:
            _remember_daemon_source(path)
            return {"ok": True, "path": path, "copied": False,
                    "error": None, "remembered": True}
        except Exception as inner:  # noqa: BLE001
            return {"ok": False, "path": path,
                    "error": "Could not copy it (%s) and could not remember "
                             "it either (%s)." % (exc, inner)}


def _remember_daemon_source(path: str) -> None:
    """Record a daemon source in ``settings.json`` (atomically, 0600: the
    file also holds a remote daemon's token)."""
    data = _read_settings()
    data["daemon_source"] = os.path.abspath(path)
    _atomic_write_json(os.path.join(base_dir(), "settings.json"), data)


def daemon_source_status() -> Dict[str, Any]:
    """What the Setup dialog needs to render the daemon-source row."""
    try:
        return {"ok": True, "path": daemon_source(), "error": None}
    except DaemonSourceMissing as exc:
        return {"ok": False, "path": None, "error": str(exc)}


def _read_settings() -> Dict[str, Any]:
    """``settings.json``, or an empty dict.  Never raises: a corrupt settings
    file must not stop the plug-in from loading."""
    try:
        with open(os.path.join(base_dir(), "settings.json"), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _stored_python() -> Tuple[Optional[str], bool]:
    """``(interpreter, had_the_daemon)`` as recorded when the user accepted it.

    The path is re-checked for existence -- an environment that has been deleted
    should fall back to the managed one rather than pinning the plug-in to a
    file that is gone.
    """
    data = _read_settings()
    path = data.get("python")
    if not isinstance(path, str) or not path or not os.path.isfile(path):
        return (None, False)
    return (path, bool(data.get("python_has_daemon")))

def _stored_daemon_source() -> Optional[str]:
    """A daemon source path recorded in ``settings.json``, if it still exists.

    Read directly rather than through ``launcher``: this module is imported by
    the Setup dialog on a broken install, where importing more than necessary is
    how a repair tool becomes another thing that fails.
    """
    value = _read_settings().get("daemon_source")
    if isinstance(value, str) and value and os.path.isfile(
            os.path.join(value, "pyproject.toml")):
        return value
    return None


def doctor_argv() -> Optional[List[str]]:
    """How to run ``sam3gimpd doctor`` in the environment the daemon will
    actually be spawned from.

    The same order as ``launcher.build_command``: the interpreter the user
    chose in Setup, then the managed venv (its console script, else its
    interpreter).  Preferring the managed venv here while the launcher
    preferred the chosen interpreter meant that anyone with both examined one
    environment and ran the other.  (``$SAM3D_COMMAND``, a developer override
    of the launcher, is not consulted.)
    """
    chosen, _had = _stored_python()
    if chosen:
        return [chosen, "-m", "sam3gimpd", "doctor"]
    exe = sam3d_executable()
    if os.path.exists(exe):
        return doctor_command(exe)
    python = daemon_interpreter()
    if python:
        return [python, "-m", "sam3gimpd", "doctor"]
    return None

def run_doctor(
    *, runner: Optional[CommandRunner] = None, timeout: float = 60.0
) -> Dict[str, Any]:
    """Drive the Doctor panel from ``sam3gimpd doctor``, falling back to a local
    report when the environment is not installed yet.

    The returned dict always has ``local`` (what the plug-in itself can see) and
    optionally ``daemon`` (:func:`doctor_view` of the report, which includes a
    running daemon's ``/status``) plus ``crash_log``.  The panel renders
    whatever is present, so a broken install still produces a useful screen
    rather than an empty one.
    """
    report = inspect_environment()
    out: Dict[str, Any] = {
        "local": report.to_json(),
        "daemon": None,
        "doctor_raw": "",
        "doctor_ok": False,
        "crash_log": tail_file(crash_log(), 40),
        "server_log": tail_file(server_log(), 40),
        "install_log": tail_file(install_log(), 40),
    }
    argv = doctor_argv()
    if argv is None:
        out["message"] = "sam3gimpd is not installed yet -- run Setup."
        return out
    out["doctor_argv"] = list(argv)
    # The same environment the launcher gives the daemon: our HF cache unless
    # the user chose their own HF_HOME, which the daemon then honours too.
    env = {ENV_HOME: base_dir()}
    if not os.environ.get("HF_HOME"):
        env["HF_HOME"] = hf_home()
    runner = runner or CommandRunner()
    result = runner.run(argv, env=env, timeout=timeout)
    out["doctor_raw"] = result.text
    out["doctor_ok"] = result.ok
    parsed = parse_doctor_output(result.text)
    if parsed is not None:
        out["daemon"] = doctor_view(parsed)
    elif not result.ok:
        out["message"] = "sam3gimpd doctor exited %d" % result.returncode
    return out
