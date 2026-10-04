"""Platform-correct filesystem locations for the sam3-gimp stack.

Pure standard library.  Importable on every platform, with no torch, no GIMP and
no network.  Both halves of the project agree on these locations:

* ``sam3gimpd`` (the daemon) uses them to write ``runtime.json``, logs and the
  lockfile, and to point ``HF_HOME`` at a directory the Doctor panel can show.
* The GIMP plug-in *mirrors* this logic in its own ``launcher.py`` (it may not
  import this module -- the plug-in is stdlib-only and lives in a different
  Python).  The layout is deliberately trivial so that mirroring it is a dozen
  lines: one base directory, everything else directly underneath it.

Layout::

    <base>/
      runtime.json          handshake file: {port, token, pid, version, started_at}
      sam3gimpd.lock        single-instance lockfile
      logs/
        sam3gimpd.log       daemon stdout/stderr + logging (rotated)
        crash.log           last unhandled traceback
      weights.json          checkpoint location recorded by the plug-in's Setup
      hf/                   HF_HOME (weights cache, ~3.6 GB)
      models/               optional user-supplied local checkpoints
      venv/                 uv-managed virtualenv holding torch + sam3gimpd
      tools/                downloaded helper binaries (uv)
      cache/                scratch space

Everything here is private to the user: on POSIX the directories are created
``0700`` (an existing base directory is tightened to it) and the files the
daemon writes are ``0600``.  ``runtime.json`` carries the bearer token, and the
logs name every image and path the daemon has seen.

Base directory per platform:

============  ==========================================================
Windows       ``%LOCALAPPDATA%\\sam3-gimp``
macOS         ``~/Library/Application Support/sam3-gimp``
Linux/other   ``$XDG_DATA_HOME/sam3-gimp`` (default ``~/.local/share/sam3-gimp``)
============  ==========================================================

Overrides (both honoured by the daemon and by the plug-in launcher):

``SAM3_GIMP_HOME``
    Replaces the base directory wholesale.  Used by the test-suite and by
    portable installs.
``SAM3D_RUNTIME_FILE``
    Replaces the path of ``runtime.json`` only.  Used when a client talks to a
    daemon whose data directory lives somewhere else (remote-GPU scenario).
"""

from __future__ import annotations

import errno
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = [
    "APP_NAME",
    "ENV_HOME",
    "ENV_RUNTIME_FILE",
    "platform_key",
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
    "ensure_dir",
    "ensure_layout",
    "daemon_environ",
    "describe",
    "atomic_write_json",
    "read_json",
    "remove_quiet",
]

APP_NAME = "sam3-gimp"
ENV_HOME = "SAM3_GIMP_HOME"
ENV_RUNTIME_FILE = "SAM3D_RUNTIME_FILE"

#: Mode for every directory this project creates (POSIX; Windows ignores it).
DIR_MODE = 0o700
#: ``os.replace`` onto a file another process has open fails on Windows with a
#: sharing violation; the reader lets go within milliseconds, so try again.
REPLACE_ATTEMPTS = 20
REPLACE_RETRY_S = 0.025


# --------------------------------------------------------------------------- #
# platform
# --------------------------------------------------------------------------- #
def platform_key() -> str:
    """Return ``"windows"``, ``"macos"`` or ``"linux"``.

    Anything that is not Windows or Darwin is treated as Linux/XDG; that is the
    correct fallback for the BSDs too.
    """
    if os.name == "nt" or sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def _home() -> Path:
    return Path(os.path.expanduser("~"))


def _expanded(value: str) -> Path:
    """An environment override with ``~`` expanded.  A ``~user`` that does not
    exist is kept as written, as ``os.path.expanduser`` would, instead of
    raising -- the plug-in's launcher does the same, so both name one file."""
    try:
        return Path(value).expanduser()
    except RuntimeError:
        return Path(value)


def _default_base_dir() -> Path:
    kind = platform_key()
    if kind == "windows":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return Path(local) / APP_NAME
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / APP_NAME
        return _home() / "AppData" / "Local" / APP_NAME
    if kind == "macos":
        return _home() / "Library" / "Application Support" / APP_NAME
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg and os.path.isabs(xdg):
        return Path(xdg) / APP_NAME
    return _home() / ".local" / "share" / APP_NAME


def base_dir() -> Path:
    """Root of everything this project owns on disk.

    ``SAM3_GIMP_HOME`` overrides it.  The value is *not* cached: tests flip the
    environment variable between cases and expect the change to be seen.
    """
    override = os.environ.get(ENV_HOME)
    if override:
        return _expanded(override)
    return _default_base_dir()


# --------------------------------------------------------------------------- #
# individual locations
# --------------------------------------------------------------------------- #
def runtime_file() -> Path:
    """Path of ``runtime.json``, the find-or-spawn handshake file."""
    override = os.environ.get(ENV_RUNTIME_FILE)
    if override:
        return _expanded(override)
    return base_dir() / "runtime.json"


def lock_file() -> Path:
    """Single-instance lockfile held by the running daemon."""
    return base_dir() / "sam3gimpd.lock"


def log_dir() -> Path:
    return base_dir() / "logs"


def server_log() -> Path:
    """Daemon log; also the target for the detached child's stdout/stderr."""
    return log_dir() / "sam3gimpd.log"


def crash_log() -> Path:
    """Last unhandled traceback, surfaced by the Doctor panel."""
    return log_dir() / "crash.log"


def hf_home() -> Path:
    """Value to export as ``HF_HOME`` for the daemon process."""
    return base_dir() / "hf"


def models_dir() -> Path:
    """Optional local checkpoint directory for users who downloaded weights
    manually (the gated-repo escape hatch)."""
    return base_dir() / "models"


def venv_root() -> Path:
    return base_dir() / "venv"


def venv_bin_dir() -> Path:
    return venv_root() / ("Scripts" if platform_key() == "windows" else "bin")


def venv_python() -> Path:
    """Console python inside the managed venv."""
    return venv_bin_dir() / ("python.exe" if platform_key() == "windows" else "python")


def venv_pythonw() -> Path:
    """Windowless python.

    On Windows the launcher MUST spawn the daemon with ``pythonw.exe``; using
    ``python.exe`` flashes a console window.  Elsewhere this is just ``python``.
    """
    return venv_bin_dir() / ("pythonw.exe" if platform_key() == "windows" else "python")


def sam3d_executable() -> Path:
    """The ``sam3gimpd`` console script installed into the managed venv."""
    return venv_bin_dir() / ("sam3gimpd.exe" if platform_key() == "windows" else "sam3gimpd")


def tools_dir() -> Path:
    return base_dir() / "tools"


def uv_binary() -> Path:
    """Downloaded ``uv`` binary used to build the venv."""
    return tools_dir() / ("uv.exe" if platform_key() == "windows" else "uv")


def cache_dir() -> Path:
    return base_dir() / "cache"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def ensure_dir(path: os.PathLike | str) -> Path:
    """``mkdir -p`` that returns the path.  Idempotent, never raises on exists.

    Every directory it has to create is made :data:`DIR_MODE` (``0700``),
    parents included -- ``Path.mkdir(parents=True)`` would give the parents the
    default, usually world-readable, mode.  A directory that already exists is
    left alone here; :func:`ensure_layout` tightens the base directory.
    """
    p = Path(path)
    missing = []
    probe = p
    while not probe.exists():
        missing.append(probe)
        if probe.parent == probe:
            break
        probe = probe.parent
    for directory in reversed(missing):
        try:
            directory.mkdir(mode=DIR_MODE)
        except FileExistsError:
            if not directory.is_dir():
                raise
            continue
        _chmod_private(directory)
    return p


def _chmod_private(directory: os.PathLike | str) -> None:
    """Make a directory user-only on POSIX (``mkdir``'s mode is subject to umask)."""
    if os.name == "nt":
        return
    try:
        if os.stat(directory).st_mode & 0o777 != DIR_MODE:
            os.chmod(directory, DIR_MODE)
    except OSError:
        pass


def ensure_layout() -> Dict[str, Path]:
    """Create every directory the daemon needs and return the layout map.

    Files (``runtime.json``, ``sam3gimpd.lock``, the logs) are *not* created, only
    their parent directories.  The base directory is made user-only even if it
    already existed: it holds the bearer token and the logs.
    """
    layout = {
        "base": base_dir(),
        "logs": log_dir(),
        "hf": hf_home(),
        "models": models_dir(),
        "tools": tools_dir(),
        "cache": cache_dir(),
    }
    for key in ("base", "logs", "hf", "models", "tools", "cache"):
        ensure_dir(layout[key])
    _chmod_private(layout["base"])
    ensure_dir(runtime_file().parent)
    return layout


def daemon_environ(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Environment overlay for the daemon process.

    Pins the HuggingFace cache inside our base directory so the Doctor panel can
    display and clear it, and disables the HF progress-bar spam in a detached
    process whose stderr is a log file.  An existing ``HF_HOME`` set by the user
    is respected.
    """
    env = dict(os.environ if base is None else base)
    env.setdefault("HF_HOME", str(hf_home()))
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env[ENV_HOME] = str(base_dir())
    return env


def describe() -> Dict[str, Any]:
    """Flat ``{name: str(path)}`` map for ``/status``, the Doctor panel and logs."""
    return {
        "platform": platform_key(),
        "base": str(base_dir()),
        "runtime_file": str(runtime_file()),
        "lock_file": str(lock_file()),
        "log_dir": str(log_dir()),
        "server_log": str(server_log()),
        "crash_log": str(crash_log()),
        "hf_home": str(hf_home()),
        "models_dir": str(models_dir()),
        "venv_root": str(venv_root()),
        "venv_python": str(venv_python()),
        "sam3d_executable": str(sam3d_executable()),
        "uv_binary": str(uv_binary()),
        "cache_dir": str(cache_dir()),
    }


def atomic_write_json(path: os.PathLike | str, obj: Any, mode: int = 0o600) -> Path:
    """Write JSON atomically: temp file in the same directory, then ``os.replace``.

    A client polling ``runtime.json`` must never observe a half-written file, so
    every writer in this project goes through here.  ``mode`` is applied on POSIX
    (0600 -- the file carries the bearer token); on Windows ``chmod`` is a no-op
    and the file is already user-scoped by living under ``%LOCALAPPDATA%``.

    On Windows the replace fails while a reader has the target open; that is
    retried for up to half a second (:data:`REPLACE_ATTEMPTS`) rather than
    leaving the old file -- a stale port and token -- in place.
    """
    target = Path(path)
    ensure_dir(target.parent)
    fd, tmp = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
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
        for attempt in range(REPLACE_ATTEMPTS):
            try:
                os.replace(tmp, target)
                break
            except PermissionError:
                if attempt + 1 >= REPLACE_ATTEMPTS:
                    raise
                time.sleep(REPLACE_RETRY_S)
    except BaseException:
        remove_quiet(tmp)
        raise
    return target


def read_json(path: os.PathLike | str) -> Optional[Any]:
    """Read JSON, returning ``None`` if the file is missing, empty, unreadable or
    malformed.  Callers treat every ``None`` the same way: the state is stale."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return None
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def remove_quiet(path: os.PathLike | str) -> bool:
    """``os.remove`` that swallows "missing" and Windows sharing violations.

    Returns True when the file is gone afterwards.
    """
    try:
        os.remove(path)
        return True
    except OSError as exc:
        if exc.errno in (errno.ENOENT,):
            return True
        return not os.path.exists(path)
