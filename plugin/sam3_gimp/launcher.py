"""Find-or-spawn the ``sam3gimpd`` daemon from inside a GIMP plug-in process.

**Zero third-party dependencies** -- standard library only (``ctypes`` counts;
it is how a pid is probed on Windows).  This module mirrors ``sam3gimpd.paths`` by
hand: the plug-in runs in GIMP's embedded Python and can never import the
daemon's package (``API.md`` §16.10).

Why this module exists (``DESIGN.md`` Constraint C)
---------------------------------------------------
GIMP spawns a **fresh, short-lived plug-in process per invocation** and it exits
the moment ``run()`` returns.  There is therefore no long-lived plug-in process
that could supervise the daemon.  Consequences, all of which this module
implements:

* the daemon is spawned **detached** so it outlives the plug-in that started it;
* it is passed ``--parent-pid <GIMP's pid>`` -- *not* the plug-in's -- so it
  self-terminates when GIMP quits;
* there is no restart-on-crash supervisor: a crash simply means the *next*
  invocation respawns, which is why the handshake below is built to cope with a
  stale ``runtime.json``.

Windows is the P0 target and its spawn details are load-bearing:

* ``pythonw.exe``, never ``python.exe`` -- the latter flashes a console window
  every time the user presses a button;
* ``creationflags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`` so the child
  has no console and does not die with the plug-in's process group;
* stdout/stderr redirected to ``<base>/logs/sam3gimpd.log`` -- a detached process
  with inherited-but-dead handles would raise on its first ``print``;
* ``close_fds=True`` so no GIMP handle is leaked into a process that outlives it.

The handshake is ``API.md`` §3.3, implemented step for step in
:func:`find_or_spawn`, with two rules layered on top because the plug-in
shares a machine with other processes and other users:

* **a daemon is accepted only if it proves it holds the token** from
  ``runtime.json`` (the ``/hello`` nonce proof, see ``client.hello``).  A pid
  that is alive is not evidence: after a crash the pid can belong to anything,
  and anything -- another user's program included -- can listen on the port the
  dead daemon had;
* **nothing is ever signalled on a guess.**  A process is ended only when it is
  verifiably a ``sam3gimpd serve`` of this user that took the instance lock
  (its command line, its owner, and a creation time no later than the
  lockfile's), *and* it has held that lock longer than any cold start takes
  without serving.  A daemon that is merely slow to start -- ``import torch``
  and the CUDA probe run *after* it takes the lock and *before* it publishes
  ``runtime.json`` -- is waited for, never killed.
"""

from __future__ import annotations

import errno
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "APP_NAME",
    "ENV_HOME",
    "ENV_RUNTIME_FILE",
    "ENV_COMMAND",
    "split_command",
    "API_VERSION",
    "DETACHED_PROCESS",
    "CREATE_NEW_PROCESS_GROUP",
    "LauncherError",
    "SpawnError",
    "DaemonStartTimeout",
    "IncompatibleDaemon",
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
    "configured_remote",
    "parse_remote_url",
    "remote_url_error",
    "set_configured_remote",
    "configured_idle_ttl",
    "set_configured_idle_ttl",
    "sam3d_executable",
    "settings_file",
    "read_settings",
    "write_settings",
    "configured_python",
    "set_configured_python",
    "windowless_variant",
    "ensure_layout",
    "daemon_environ",
    "describe",
    "read_runtime_info",
    "delete_runtime_file",
    "pid_alive",
    "pid_owned",
    "gimp_pid",
    "lock_holder_pid",
    "plugin_build",
    "terminate_zombie_daemon",
    "tail_log",
    "build_command",
    "spawn_daemon",
    "find_or_spawn",
    "LaunchResult",
    "shutdown_daemon",
]

APP_NAME = "sam3-gimp"

#: Replaces the base directory wholesale (honoured by the daemon too).
ENV_HOME = "SAM3_GIMP_HOME"
#: Replaces the full path of ``runtime.json`` only.
ENV_RUNTIME_FILE = "SAM3D_RUNTIME_FILE"
#: Development / test escape hatch: an explicit argv prefix for the daemon.
ENV_COMMAND = "SAM3D_COMMAND"

#: API version this plug-in speaks; must match ``client.API_VERSION``.  Only
#: the major takes part in the compatibility check.
API_VERSION = "1.1"

# Windows CreateProcess flags (defined here so the module imports on POSIX,
# where subprocess does not expose them at all).
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

#: Steps of the §3.3 handshake, for logs and tests.
HANDSHAKE_POLL_INTERVAL = 0.1
#: How long :func:`find_or_spawn` waits for a daemon it had to start (or for
#: one that is running but not answering) unless the caller says otherwise.
#: Sized for the worst ordinary case -- a cold ``import torch`` plus the CUDA
#: probe on Windows, which has taken well over 20 s -- because giving up early
#: does not make the daemon start any faster; it only makes the user retry.
HANDSHAKE_COLD_TIMEOUT = 60.0
#: The client's connect timeout for requests after the handshake.  The /hello
#: probe itself passes HELLO_READ_TIMEOUT, which then bounds the connect too:
#: Windows takes about 2 s to refuse a closed local port, and a shorter connect
#: timeout would turn that refusal into a timeout (API.md section 3.3, step 3).
HELLO_CONNECT_TIMEOUT = 1.5
HELLO_READ_TIMEOUT = 3.0
MAX_SPAWN_ATTEMPTS = 2

#: A lock holder is only ever treated as stuck after holding the instance lock
#: this long without serving: twice the longest cold start we wait for.
COLD_START_BUDGET = 2 * HANDSHAKE_COLD_TIMEOUT
#: After a spawned child exits because the lock is taken, how long the
#: lockfile may name nobody (or a process that is not a daemon) before we
#: conclude the lock is free again.  The real holder writes its pid
#: immediately after taking the lock, so this is generous.
LOCK_SETTLE_S = 1.0
#: How long a verified-stuck holder (see :data:`COLD_START_BUDGET`) is given
#: to publish or exit on its own before it is ended -- enough for a daemon
#: that is in the middle of an orderly shutdown to finish it.
ZOMBIE_GRACE_S = 5.0
#: How long a daemon asked to exit with ``POST /shutdown`` gets to do so on
#: its own before it is signalled.
RETIRE_WAIT_S = 6.0
#: Tolerance when comparing a process's creation time with a file's mtime:
#: coarse clocks (``/proc/stat`` btime is whole seconds, FAT mtimes are 2 s)
#: and a wall-clock step while the daemon ran.
START_TIME_SLACK_S = 30.0

#: Popen objects for daemons we started.  We never ``wait()`` on them -- the
#: child must outlive us -- but holding the reference keeps CPython from
#: emitting a "subprocess is still running" ResourceWarning when the object is
#: collected.  The plug-in process exits seconds later, at which point the OS
#: reparents the daemon to init.
_SPAWNED: List[Any] = []
_SPAWNED_LOCK = threading.Lock()


def _reap_spawned() -> None:
    """Reap daemons we spawned that have already exited.

    We never ``wait()`` on a live daemon -- it must outlive us -- but a child
    that *has* exited stays a zombie until someone collects it, and a zombie
    still answers ``os.kill(pid, 0)``.  Without this, :func:`pid_alive` reports
    a daemon that shut down cleanly as running, and the handshake has to fall
    all the way through to the ``/hello`` probe to notice.  ``poll()`` is
    non-blocking and works on Windows too.
    """
    with _SPAWNED_LOCK:
        alive: List[Any] = []
        for proc in _SPAWNED:
            try:
                if proc.poll() is None:
                    alive.append(proc)
            except Exception:  # pragma: no cover -- a torn-down Popen
                alive.append(proc)   # unknown state: keep the reference, never drop it
        _SPAWNED[:] = alive


def _spawned_proc(pid: int) -> Any:
    """The Popen we hold for ``pid``, if we spawned it and it is not reaped.

    Looked up by pid rather than taken as "the last one": two threads of the
    same plug-in process can spawn at the same time.
    """
    with _SPAWNED_LOCK:
        for proc in reversed(_SPAWNED):
            if getattr(proc, "pid", None) == pid:
                return proc
    return None


# --------------------------------------------------------------------------- #
# exceptions
# --------------------------------------------------------------------------- #
class LauncherError(Exception):
    """Base class for launcher failures."""


class SpawnError(LauncherError):
    """The daemon process could not be started at all.

    ``command`` is the argv that failed; the Doctor panel shows it verbatim
    because "no such file" here almost always means the venv is missing and the
    bootstrap flow needs to run.
    """

    def __init__(self, message: str, command: Optional[Sequence[str]] = None) -> None:
        self.command = list(command or [])
        LauncherError.__init__(self, message)


class DaemonStartTimeout(LauncherError):
    """The daemon was spawned but never published a usable ``runtime.json``.

    ``log_tail`` holds the last lines of ``logs/sam3gimpd.log`` -- the only evidence
    available for a detached process whose pipes we deliberately never read
    (``API.md`` §3.3 step 8).
    """

    def __init__(self, message: str, log_tail: str = "", command: Optional[Sequence[str]] = None) -> None:
        self.log_tail = log_tail
        self.command = list(command or [])
        LauncherError.__init__(self, message + (("\n--- sam3gimpd.log ---\n" + log_tail) if log_tail else ""))


class IncompatibleDaemon(LauncherError):
    """A daemon is running but speaks a different API major, and respawning it
    did not help (usually: an old daemon is pinned by its lockfile)."""


# --------------------------------------------------------------------------- #
# platform layout -- hand mirror of sam3gimpd.paths
# --------------------------------------------------------------------------- #
def platform_key() -> str:
    """``"windows"``, ``"macos"`` or ``"linux"`` (BSDs fall back to XDG)."""
    if os.name == "nt" or sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def _home() -> str:
    return os.path.expanduser("~")


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
    """Root of everything the project owns on disk (``API.md`` §3.1).

    Never cached: the test-suite flips ``SAM3_GIMP_HOME`` between cases.
    """
    override = os.environ.get(ENV_HOME)
    if override:
        return _override_path(override)
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


def hf_home() -> str:
    return os.path.join(base_dir(), "hf")


def models_dir() -> str:
    return os.path.join(base_dir(), "models")


def venv_root() -> str:
    return os.path.join(base_dir(), "venv")


def venv_bin_dir() -> str:
    return os.path.join(venv_root(), "Scripts" if platform_key() == "windows" else "bin")


def venv_python() -> str:
    """Console python inside the managed venv."""
    return os.path.join(venv_bin_dir(), "python.exe" if platform_key() == "windows" else "python")


def venv_pythonw() -> str:
    """Windowless python.

    On Windows the daemon MUST be spawned with this; ``python.exe`` flashes a
    console window on every plug-in invocation.
    """
    return os.path.join(venv_bin_dir(), "pythonw.exe" if platform_key() == "windows" else "python")


def sam3d_executable() -> str:
    """The ``sam3gimpd`` console script installed into the managed venv."""
    return os.path.join(venv_bin_dir(), "sam3gimpd.exe" if platform_key() == "windows" else "sam3gimpd")



# --------------------------------------------------------------------------- #
# user settings -- the "use my own Python" choice, made in the Setup dialog
# --------------------------------------------------------------------------- #
def settings_file() -> str:
    """Where the plug-in stores choices the user made in the Setup dialog."""
    return os.path.join(base_dir(), "settings.json")


def read_settings() -> Dict[str, Any]:
    """Load ``settings.json``.  A missing or corrupt file is an empty dict --
    a bad settings file must never stop the plug-in from loading."""
    try:
        with open(settings_file(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_settings(data: Dict[str, Any]) -> None:
    """Persist ``settings.json`` atomically, readable by this user only.

    It can hold a remote daemon's bearer token, so it is written ``0600``
    (the base directory itself is ``0700``, see :func:`ensure_layout`).
    """
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    _write_private_file(settings_file(), text.encode("utf-8"))


def _private_dir(path: str) -> None:
    """``makedirs`` with mode ``0700`` on POSIX; a no-op if it exists.

    ``makedirs`` applies the mode only to the leaf and the umask still
    applies, so the leaf is ``chmod``-ed after creation.  Directories that
    already existed are left as they are: :func:`ensure_layout` tightens the
    one that matters, the base directory.
    """
    if os.path.isdir(path):
        return
    os.makedirs(path, mode=0o700, exist_ok=True)
    if os.name != "nt":
        try:
            os.chmod(path, 0o700)
        except OSError:
            pass


def _write_private_file(path: str, data: bytes) -> None:
    """Write ``path`` atomically with mode ``0600``: a unique temp file in the
    same directory (``mkstemp`` creates it ``0600``), fsync, ``os.replace``.

    A unique name rather than a fixed ``.tmp`` because two plug-in processes
    can save settings at the same moment.  ``os.replace`` onto a file another
    process has open fails transiently on Windows, so it is retried briefly.
    """
    directory = os.path.dirname(path) or "."
    _private_dir(directory)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix="." + os.path.basename(path) + ".",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        if os.name != "nt":
            os.chmod(tmp, 0o600)
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.05)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def configured_python() -> Optional[str]:
    """The interpreter the user picked in Setup, if it still exists.

    Returning ``None`` for a path that has been deleted or moved is deliberate:
    the resolution chain then falls through to the managed venv instead of
    failing, so uninstalling the chosen environment degrades rather than breaks.
    """
    value = read_settings().get("python")
    if isinstance(value, str) and value and os.path.isfile(value):
        return value
    return None


def set_configured_python(path: Optional[str],
                          has_daemon: Optional[bool] = None) -> None:
    """Record (or clear, with ``None``) the user's chosen interpreter.

    ``has_daemon`` records whether that interpreter had ``sam3gimpd`` installed
    at the moment it was accepted.  It is stored rather than re-probed because
    ``inspect_environment`` -- which decides whether the plug-in considers
    itself set up -- runs during procedure registration, where spawning a
    subprocess would delay every GIMP start.
    """
    data = read_settings()
    if path:
        data["python"] = os.path.abspath(path)
        if has_daemon is not None:
            data["python_has_daemon"] = bool(has_daemon)
    else:
        data.pop("python", None)
        data.pop("python_has_daemon", None)
    write_settings(data)


# --------------------------------------------------------------------------- #
# settings: a remote daemon, and the idle timeout
# --------------------------------------------------------------------------- #
def configured_remote() -> Optional[Tuple[str, int, str]]:
    """``(host, port, token)`` of a daemon the user runs elsewhere, or ``None``.

    Stored in ``settings.json`` as ``remote_url`` (``http://host:port``) and
    ``remote_token``.  ``sam3gimpd serve --host 0.0.0.0`` on the GPU box has
    existed since the first release; this is the half that lets the plug-in
    reach it.  It wins over a local spawn; ``SAM3D_COMMAND`` and an explicit
    ``command`` still win over it.
    """
    data = read_settings()
    url = data.get("remote_url")
    token = data.get("remote_token")
    if not isinstance(url, str) or not url.strip() or not isinstance(token, str):
        return None
    parsed = parse_remote_url(url)
    if parsed is None:
        return None
    return (parsed[0], parsed[1], token)


def _split_remote_url(url: Optional[str]) -> Tuple[Optional[Tuple[str, int]], str]:
    """``(host, port)`` and ``""``, or ``None`` and the reason, in words."""
    from urllib.parse import urlsplit  # noqa: PLC0415

    text = (url or "").strip()
    if not text:
        return None, "enter the daemon's address as http://host:port"
    if "://" not in text:
        text = "http://" + text
    try:
        parts = urlsplit(text)
    except ValueError:
        return None, "%r is not an address; it must look like http://host:port" % (url,)
    scheme = parts.scheme.lower()
    if scheme == "https":
        return None, ("the daemon speaks plain HTTP (there is no TLS to connect to); "
                      "use http://host:port, and use an SSH tunnel or a VPN if the "
                      "network between the machines is not trusted")
    if scheme != "http":
        return None, ("%r addresses are not supported; the daemon address must look "
                      "like http://host:port" % (parts.scheme + "://"))
    if parts.username is not None or parts.password is not None:
        return None, ("the address must not contain a user name or password; "
                      "the token goes in the Token field")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        return None, "the daemon address is just http://host:port, with no path or query"
    try:
        host = parts.hostname
        port = parts.port
    except ValueError:
        return None, "the port must be a number from 1 to 65535"
    if not host:
        return None, "the address has no host name; it must look like http://host:port"
    if not port:
        return None, ("a port is required: http://host:port (the daemon prints it, "
                      "and it is in that machine's runtime.json)")
    return (host, int(port)), ""


def parse_remote_url(url: str) -> Optional[Tuple[str, int]]:
    """``http://host:port`` -> ``(host, port)``; ``None`` for anything else.

    Only plain HTTP: the daemon speaks HTTP/1.1 with a bearer token and no
    TLS (API.md §1), so an ``https`` URL is a mistake, not a feature.
    :func:`remote_url_error` says *why* an address was refused."""
    return _split_remote_url(url)[0]


def remote_url_error(url: Optional[str]) -> Optional[str]:
    """Why ``url`` is not a usable daemon address, or ``None`` if it is."""
    parsed, reason = _split_remote_url(url)
    return None if parsed is not None else reason


def set_configured_remote(url: Optional[str], token: Optional[str] = None) -> None:
    """Record (or with ``None`` forget) the remote daemon.

    Raises :class:`ValueError` carrying :func:`remote_url_error`'s reason for
    an address that cannot work.  The token is stored, never logged.
    """
    data = read_settings()
    if not url:
        data.pop("remote_url", None)
        data.pop("remote_token", None)
    else:
        problem = remote_url_error(url)
        if problem:
            raise ValueError(problem)
        data["remote_url"] = url.strip()
        data["remote_token"] = (token or "").strip()
    write_settings(data)


#: The daemon's own default, in seconds (``sam3gimpd serve --idle-ttl``).
DEFAULT_IDLE_TTL_S = 1800.0


def configured_idle_ttl() -> Optional[float]:
    """Idle timeout in seconds from ``settings.json`` (``idle_ttl_minutes``),
    ``0`` for never, or ``None`` when the user has not set one."""
    value = read_settings().get("idle_ttl_minutes")
    if value is None:
        return None
    try:
        minutes = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, minutes) * 60.0


def set_configured_idle_ttl(minutes: Optional[float]) -> None:
    data = read_settings()
    if minutes is None:
        data.pop("idle_ttl_minutes", None)
    else:
        data["idle_ttl_minutes"] = max(0.0, float(minutes))
    write_settings(data)


def windowless_variant(python_path: str) -> str:
    """``pythonw.exe`` beside a given ``python.exe``, when it exists.

    Windows only, and the reason is user-visible: spawning ``python.exe``
    flashes a console window on every single prompt.  Falls back to the path as
    given, since a venv without ``pythonw.exe`` is unusual but not fatal.
    """
    if platform_key() != "windows":
        return python_path
    directory, name = os.path.split(python_path)
    if name.lower() == "python.exe":
        candidate = os.path.join(directory, "pythonw.exe")
        if os.path.isfile(candidate):
            return candidate
    return python_path

def ensure_layout() -> Dict[str, str]:
    """Create the directories the daemon needs; return the layout map.

    POSIX: everything is created ``0700``, and an existing base directory
    this user owns is tightened to ``0700`` -- it holds ``runtime.json`` (the
    daemon's bearer token), ``settings.json`` (a remote daemon's token) and
    the logs, none of which another local user has any business reading.
    """
    layout = {
        "base": base_dir(),
        "logs": log_dir(),
        "hf": hf_home(),
        "models": models_dir(),
    }
    for path in layout.values():
        try:
            _private_dir(path)
        except OSError:
            pass
    if os.name != "nt":
        base = layout["base"]
        try:
            st = os.stat(base)
            if st.st_uid == os.getuid() and (st.st_mode & 0o077):
                os.chmod(base, 0o700)
        except (OSError, AttributeError):
            pass
    try:
        _private_dir(os.path.dirname(runtime_file()))
    except OSError:
        pass
    return layout


def daemon_environ(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Environment for the detached child (``API.md`` §13).

    Pins the HuggingFace cache inside our base directory so the Doctor panel can
    show and clear it, silences progress bars in a process whose stderr is a log
    file, and passes ``SAM3_GIMP_HOME`` through so the child agrees with us
    about where ``runtime.json`` goes.
    """
    env = dict(os.environ if base is None else base)
    env.setdefault("HF_HOME", hf_home())
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env["PYTHONUNBUFFERED"] = "1"
    env[ENV_HOME] = base_dir()
    return env


_BUILD_FILES = ("sam3_gimp.py", "client.py", "launcher.py", "bootstrap.py", "contours.py",
                "outputs.py", "gimpbridge.py", "ui/main_dialog.py", "ui/canvas.py",
                "ui/setup_dialog.py")
_build_cache: Optional[str] = None


def plugin_build() -> str:
    """Eight hex digits identifying the plug-in files actually on disk.

    A content hash rather than a version string, because the failure mode it
    exists for is "the folder was not really re-copied": it is written to
    ``plugin.log`` at start-up and shown in the window's device line, so a
    report can be matched to the code that produced it.
    """
    global _build_cache
    if _build_cache is None:
        import hashlib  # noqa: PLC0415

        root = os.path.dirname(os.path.abspath(__file__))
        digest = hashlib.sha1()
        for name in _BUILD_FILES:
            try:
                with open(os.path.join(root, name), "rb") as fh:
                    digest.update(name.encode("utf-8"))
                    digest.update(fh.read())
            except OSError:
                digest.update(("missing:" + name).encode("utf-8"))
        _build_cache = digest.hexdigest()[:8]
    return _build_cache


def describe() -> Dict[str, str]:
    """Flat path map for the Doctor panel and for logs."""
    return {
        "platform": platform_key(),
        "base": base_dir(),
        "runtime_file": runtime_file(),
        "lock_file": lock_file(),
        "log_dir": log_dir(),
        "server_log": server_log(),
        "crash_log": crash_log(),
        "hf_home": hf_home(),
        "models_dir": models_dir(),
        "venv_root": venv_root(),
        "venv_python": venv_python(),
        "venv_pythonw": venv_pythonw(),
        "sam3d_executable": sam3d_executable(),
    }


# --------------------------------------------------------------------------- #
# runtime.json
# --------------------------------------------------------------------------- #
def read_runtime_info(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Read and validate ``runtime.json``; ``None`` means "stale, respawn".

    Delegates to ``client.read_runtime_json`` so there is exactly one validator.
    """
    return _client_module().read_runtime_json(path or runtime_file())


def delete_runtime_file(path: Optional[str] = None,
                        expect: Optional[Dict[str, Any]] = None) -> bool:
    """Remove a stale handshake file.  True when it is gone afterwards.

    With ``expect`` -- the ``runtime.json`` contents the caller judged stale --
    the file is removed only if it still describes that same daemon (same pid
    and token).  Between reading the file and deciding, a new daemon may have
    published its own; deleting *that* would strand a healthy daemon that no
    one could find again.  False when the file was left in place for that
    reason.

    Swallows Windows sharing violations: another plug-in invocation racing us
    may hold the file open, and the loop in :func:`find_or_spawn` copes.
    """
    target = path or runtime_file()
    if expect is not None:
        current = read_runtime_info(target)
        if current is not None and (
                current.get("pid") != expect.get("pid")
                or current.get("token") != expect.get("token")):
            return False
    try:
        os.remove(target)
        return True
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            return True
        return not os.path.exists(target)


def tail_log(path: Optional[str] = None, lines: int = 40, max_bytes: int = 65536) -> str:
    """Last ``lines`` lines of the daemon log, for error dialogs (§3.3 step 8)."""
    target = path or server_log()
    try:
        size = os.path.getsize(target)
        with open(target, "rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
            data = fh.read()
    except OSError:
        return ""
    text = data.decode("utf-8", "replace")
    return "\n".join(text.splitlines()[-int(lines):])


# --------------------------------------------------------------------------- #
# pid liveness (API.md section 3.3 step 2)
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# which process is GIMP?
# --------------------------------------------------------------------------- #
def _find_gimp_ancestor(table: Dict[int, Tuple[int, str]], start: int,
                        max_depth: int = 8) -> Optional[int]:
    """Nearest ancestor whose executable name starts with ``gimp``.

    ``table`` maps pid -> (parent pid, executable name).  Pure, so it can be
    tested with a hand-built tree.  Cycle- and depth-bounded: a corrupt table
    must not spin.
    """
    cur = start
    seen = set()
    for _ in range(max_depth):
        entry = table.get(cur)
        if entry is None:
            return None
        ppid, _name = entry
        if not ppid or ppid == cur or ppid in seen:
            return None
        seen.add(cur)
        parent = table.get(ppid)
        if parent is not None:
            # Split on either separator: a Windows table can carry full paths
            # with backslashes, and this must read the same on every platform.
            pname = re.split(r"[\\/]", parent[1] or "")[-1].lower()
            if pname.startswith("gimp"):
                return int(ppid)
        cur = ppid
    return None


def _windows_process_table() -> Dict[int, Tuple[int, str]]:
    """pid -> (ppid, exe) for every process, via Toolhelp32 (stdlib ctypes)."""
    try:
        import ctypes  # noqa: PLC0415
        import ctypes.wintypes as wt  # noqa: PLC0415

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", wt.DWORD), ("cntUsage", wt.DWORD),
                ("th32ProcessID", wt.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", wt.DWORD), ("cntThreads", wt.DWORD),
                ("th32ParentProcessID", wt.DWORD), ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", wt.DWORD), ("szExeFile", ctypes.c_wchar * 260),
            ]

        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        k32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
        k32.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESSENTRY32W)]
        k32.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESSENTRY32W)]
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        snap = k32.CreateToolhelp32Snapshot(0x2, 0)
        if not snap or snap == ctypes.c_void_p(-1).value:
            return {}
        table: Dict[int, Tuple[int, str]] = {}
        try:
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            ok = k32.Process32FirstW(snap, ctypes.byref(entry))
            while ok:
                table[int(entry.th32ProcessID)] = (int(entry.th32ParentProcessID), entry.szExeFile)
                ok = k32.Process32NextW(snap, ctypes.byref(entry))
        finally:
            k32.CloseHandle(snap)
        return table
    except Exception:  # noqa: BLE001 -- any failure means "unknown", never a crash
        return {}


def _posix_ancestor_table(start: int, max_depth: int = 8) -> Dict[int, Tuple[int, str]]:
    """The ancestor chain of ``start`` from ``/proc`` (Linux); empty elsewhere."""
    table: Dict[int, Tuple[int, str]] = {}
    cur = start
    for _ in range(max_depth):
        try:
            with open("/proc/%d/stat" % cur, "rb") as fh:
                raw = fh.read()
            name = raw[raw.index(b"(") + 1:raw.rindex(b")")].decode("utf-8", "replace")
            ppid = int(raw[raw.rindex(b")") + 1:].split()[1])
        except (OSError, ValueError, IndexError):
            break
        table[cur] = (ppid, name)
        if ppid <= 1:
            break
        cur = ppid
    return table


def gimp_pid() -> Optional[int]:
    """The pid the daemon should outlive everything *except*: GIMP's.

    ``os.getppid()`` was wrong on Windows.  GIMP 3 starts a Python plug-in
    through an intermediate process that lives exactly one invocation, so the
    daemon logged "parent pid ... is gone; exiting" about a minute after every
    start, and every use began with a cold model load.  Walk up the tree for
    an ancestor actually named ``gimp*`` instead.  When none can be found, tie
    the daemon to nothing (``None``): the idle TTL still reaps it, and a daemon
    that lives too long is far cheaper than one that dies every minute.
    """
    me = os.getpid()
    if platform_key() == "windows":
        return _find_gimp_ancestor(_windows_process_table(), me)
    found = _find_gimp_ancestor(_posix_ancestor_table(me), me)
    if found is not None:
        return found
    try:
        ppid = int(os.getppid())
    except (AttributeError, OSError):
        return None
    return ppid if ppid > 1 else None

def _plausible_pid(pid: Any) -> Optional[int]:
    try:
        value = int(pid)
    except (TypeError, ValueError, OverflowError):
        return None
    # A pid outside the OS range would make os.kill raise OverflowError; treat
    # anything implausible as "not running" rather than letting it escape.
    if value <= 0 or value > 2 ** 31:
        return None
    return value


def pid_alive(pid: Optional[int]) -> bool:
    """Does a process with this pid exist?  *Liveness* only, not ownership.

    POSIX: ``os.kill(pid, 0)``; ``EPERM`` counts as **alive** (the process
    exists, it just is not ours).
    Windows: ``OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)`` via ``ctypes``
    -- stdlib, so the plug-in may use it.  ``ERROR_ACCESS_DENIED`` likewise
    means alive (another user's, or a protected process); only
    ``ERROR_INVALID_PARAMETER`` -- no such process -- or an exit code other
    than ``STILL_ACTIVE`` means dead.

    Both platforms answer the same question, which is the one the callers
    that wait for a process to exit need.  Whether a live process may be
    *trusted* -- is it this user's? -- is :func:`pid_owned`; the handshake asks
    both, and treats a process owned by someone else as a stale entry, never
    as our daemon.
    """
    pid = _plausible_pid(pid)
    if pid is None:
        return False

    # Collect any daemon of ours that has already exited, so a clean shutdown
    # is not reported as "still running" by the zombie it briefly leaves.
    _reap_spawned()

    # Linux: a defunct process -- exited, not yet reaped by its parent -- still
    # answers kill(pid, 0), but it is dead for every purpose here: it holds no
    # lock, serves nothing, and will never answer /hello.  Reading it as alive
    # made the zombie-daemon handling wait out its whole grace period for
    # nothing.  /proc is absent elsewhere, which the except covers.
    try:
        with open("/proc/%d/stat" % pid, "rb") as fh:
            state = fh.read().rsplit(b")", 1)[1].split()[0]
        if state == b"Z":
            return False
    except (OSError, IndexError):
        pass

    if platform_key() == "windows":
        try:
            w = _win32()
            handle = w.k32.OpenProcess(w.PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                # Only "no such process" is dead.  ACCESS_DENIED is a process
                # that exists but is not ours to open.
                return _win_last_error() != w.ERROR_INVALID_PARAMETER
            try:
                code = w.DWORD()
                if w.k32.GetExitCodeProcess(handle, w.byref(code)):
                    # A zombie handle can outlive the process; STILL_ACTIVE is
                    # the only value that means "really running".
                    return code.value == w.STILL_ACTIVE
                return True
            finally:
                w.k32.CloseHandle(handle)
        except Exception:
            # ctypes unavailable or an unexpected Win32 failure: assume alive
            # rather than deleting a healthy daemon's runtime.json.  Nothing
            # is accepted on liveness alone anyway: /hello must prove itself.
            return True

    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except OverflowError:
        return False
    except OSError as exc:
        return exc.errno == errno.EPERM


def pid_owned(pid: Optional[int]) -> Optional[bool]:
    """Is ``pid`` a running process of **this** user?

    ``True`` / ``False``, or ``None`` when the platform will not say.  A pid
    that belongs to someone else is never our daemon, whatever answers on the
    port it once had: :func:`find_or_spawn` treats such a ``runtime.json`` as
    stale without even connecting, so the token in it is not handed to
    another user's process.

    POSIX: the owner of ``/proc/<pid>`` where there is one, else whether
    ``kill(pid, 0)`` is permitted.  Windows: the user SID of the process's
    token compared with ours; ``ERROR_ACCESS_DENIED`` on either handle means
    "not ours".
    """
    pid = _plausible_pid(pid)
    if pid is None or not pid_alive(pid):
        return False
    if platform_key() == "windows":
        try:
            return _win_pid_owned(pid)
        except Exception:  # noqa: BLE001 -- unknown, never a crash
            return None
    if os.path.isdir("/proc/%d" % pid):
        try:
            return os.stat("/proc/%d" % pid).st_uid == os.geteuid()
        except OSError:
            pass
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError as exc:
        return False if exc.errno in (errno.EPERM, errno.ESRCH) else None


# --------------------------------------------------------------------------- #
# process identity: is that pid really a sam3gimpd daemon of ours?
# --------------------------------------------------------------------------- #
class _Win32(object):
    """``kernel32`` / ``advapi32`` / ``ntdll`` entry points with their
    signatures declared.  Private ``WinDLL`` instances, so the ``argtypes``
    set here cannot collide with anyone else's use of ``ctypes.windll``."""

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    ERROR_ACCESS_DENIED = 5
    ERROR_INVALID_PARAMETER = 87
    TOKEN_QUERY = 0x0008
    TOKEN_USER_CLASS = 1
    PROCESS_COMMAND_LINE_INFORMATION = 60

    def __init__(self) -> None:
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        self.ctypes = ctypes
        self.byref = ctypes.byref
        self.DWORD = wintypes.DWORD
        self.ULONG = wintypes.ULONG
        self.HANDLE = wintypes.HANDLE
        self.FILETIME = wintypes.FILETIME

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        k32.CloseHandle.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = (wintypes.HANDLE,)
        k32.GetExitCodeProcess.restype = wintypes.BOOL
        k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.GetCurrentProcess.argtypes = ()
        k32.GetProcessTimes.restype = wintypes.BOOL
        k32.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
        k32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        k32.QueryFullProcessImageNameW.argtypes = (
            wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD))
        self.k32 = k32

        adv = ctypes.WinDLL("advapi32", use_last_error=True)
        adv.OpenProcessToken.restype = wintypes.BOOL
        adv.OpenProcessToken.argtypes = (wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE))
        adv.GetTokenInformation.restype = wintypes.BOOL
        adv.GetTokenInformation.argtypes = (
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD))
        adv.GetLengthSid.restype = wintypes.DWORD
        adv.GetLengthSid.argtypes = (ctypes.c_void_p,)
        adv.IsValidSid.restype = wintypes.BOOL
        adv.IsValidSid.argtypes = (ctypes.c_void_p,)
        self.adv = adv

        nt = ctypes.WinDLL("ntdll")
        nt.NtQueryInformationProcess.restype = ctypes.c_long
        nt.NtQueryInformationProcess.argtypes = (
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG))
        self.nt = nt

        class UNICODE_STRING(ctypes.Structure):
            _fields_ = [("Length", wintypes.USHORT), ("MaximumLength", wintypes.USHORT),
                        ("Buffer", ctypes.c_void_p)]

        self.UNICODE_STRING = UNICODE_STRING

    def open_process(self, pid: int) -> Any:
        return self.k32.OpenProcess(self.PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))

    def token_user_sid(self, process: Any) -> Optional[bytes]:
        """The user SID of ``process``'s token, as bytes; ``None`` if denied."""
        ctypes = self.ctypes
        token = self.HANDLE()
        if not self.adv.OpenProcessToken(process, self.TOKEN_QUERY, ctypes.byref(token)):
            return None
        try:
            size = self.DWORD(0)
            self.adv.GetTokenInformation(token, self.TOKEN_USER_CLASS, None, 0, ctypes.byref(size))
            if not size.value or size.value > 65536:
                return None
            buf = ctypes.create_string_buffer(size.value)
            if not self.adv.GetTokenInformation(token, self.TOKEN_USER_CLASS, buf, size,
                                                ctypes.byref(size)):
                return None
            # TOKEN_USER starts with SID_AND_ATTRIBUTES, whose first member is
            # the PSID; the SID itself lives further on in the same buffer.
            sid = ctypes.cast(buf, ctypes.POINTER(ctypes.c_void_p))[0]
            if not sid or not self.adv.IsValidSid(sid):
                return None
            return ctypes.string_at(sid, self.adv.GetLengthSid(sid))
        finally:
            self.k32.CloseHandle(token)


_WIN32: Optional[_Win32] = None


def _win32() -> _Win32:
    global _WIN32
    if _WIN32 is None:
        _WIN32 = _Win32()
    return _WIN32


def _win_last_error() -> int:
    import ctypes  # noqa: PLC0415

    return int(ctypes.get_last_error())


def _win_pid_owned(pid: int) -> Optional[bool]:
    w = _win32()
    handle = w.open_process(pid)
    if not handle:
        err = _win_last_error()
        if err in (w.ERROR_ACCESS_DENIED, w.ERROR_INVALID_PARAMETER):
            return False
        return None
    try:
        theirs = w.token_user_sid(handle)
        if theirs is None:
            return False if _win_last_error() == w.ERROR_ACCESS_DENIED else None
        mine = w.token_user_sid(w.k32.GetCurrentProcess())
        if mine is None:
            return None
        return theirs == mine
    finally:
        w.k32.CloseHandle(handle)


def _win_process_details(pid: int) -> Tuple[Optional[List[str]], Optional[str], Optional[float]]:
    """``(argv, image path, creation time)`` of a Windows process; ``None``
    for whatever could not be read.

    The command line comes from ``NtQueryInformationProcess`` class 60
    (``ProcessCommandLineInformation``, Windows 8.1+), which needs only
    ``PROCESS_QUERY_LIMITED_INFORMATION`` -- no reading another process's
    memory.
    """
    w = _win32()
    ctypes = w.ctypes
    handle = w.open_process(pid)
    if not handle:
        return None, None, None
    argv: Optional[List[str]] = None
    image: Optional[str] = None
    created: Optional[float] = None
    try:
        try:
            size = w.ULONG(0)
            w.nt.NtQueryInformationProcess(handle, w.PROCESS_COMMAND_LINE_INFORMATION, None, 0,
                                           ctypes.byref(size))
            if 0 < size.value <= 1 << 20:
                buf = ctypes.create_string_buffer(size.value)
                status = w.nt.NtQueryInformationProcess(
                    handle, w.PROCESS_COMMAND_LINE_INFORMATION, buf, size.value, ctypes.byref(size))
                if status >= 0:
                    us = w.UNICODE_STRING.from_buffer(buf)
                    if us.Buffer and us.Length:
                        argv = _split_windows_command_line(ctypes.wstring_at(us.Buffer, us.Length // 2))
        except Exception:  # noqa: BLE001 -- older Windows, or a denied query
            argv = None
        try:
            name = ctypes.create_unicode_buffer(32768)
            length = w.DWORD(len(name))
            if w.k32.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(length)):
                image = name.value
        except Exception:  # noqa: BLE001
            image = None
        try:
            times = [w.FILETIME() for _ in range(4)]
            if w.k32.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
                ticks = (int(times[0].dwHighDateTime) << 32) | int(times[0].dwLowDateTime)
                if ticks:
                    # FILETIME: 100 ns units since 1601-01-01 UTC.
                    created = ticks / 1e7 - 11644473600.0
        except Exception:  # noqa: BLE001
            created = None
    finally:
        w.k32.CloseHandle(handle)
    return argv, image, created


def _split_windows_command_line(text: str) -> List[str]:
    """Good enough to find our own arguments in a Windows command line."""
    try:
        parts = shlex.split(text, posix=False)
    except ValueError:
        parts = text.split()
    out: List[str] = []
    for part in parts:
        if len(part) >= 2 and part[0] == part[-1] and part[0] in ('"', "'"):
            part = part[1:-1]
        out.append(part)
    return out


def _posix_process_details(pid: int) -> Tuple[Optional[List[str]], Optional[float]]:
    """``(argv, creation time)`` from ``/proc`` (Linux) or ``ps`` (macOS, BSD)."""
    proc = "/proc/%d" % pid
    if os.path.isdir(proc):
        argv: Optional[List[str]] = None
        created: Optional[float] = None
        try:
            with open(proc + "/cmdline", "rb") as fh:
                raw = fh.read()
            argv = [a.decode("utf-8", "replace") for a in raw.split(b"\0")]
            while argv and argv[-1] == "":
                argv.pop()
        except OSError:
            argv = None
        try:
            with open(proc + "/stat", "rb") as fh:
                fields = fh.read().rsplit(b")", 1)[1].split()
            ticks = int(fields[19])     # field 22, starttime, clock ticks after boot
            boot = None
            with open("/proc/stat", "rb") as fh:
                for line in fh:
                    if line.startswith(b"btime "):
                        boot = int(line.split()[1])
                        break
            hz = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
            if boot is not None and hz:
                created = boot + ticks / float(hz)
        except (OSError, ValueError, IndexError):
            created = None
        return argv, created
    try:
        out = subprocess.run(
            ["/bin/ps" if os.path.exists("/bin/ps") else "ps", "-ww", "-o", "etime=,command=",
             "-p", str(int(pid))],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5.0,
            env=dict(os.environ, LC_ALL="C"),
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    text = out.stdout.decode("utf-8", "replace").strip()
    if out.returncode != 0 or not text:
        return None, None
    head, _, command = text.partition(" ")
    elapsed = _parse_ps_etime(head)
    return command.split(), (None if elapsed is None else time.time() - elapsed)


def _parse_ps_etime(text: str) -> Optional[float]:
    """``ps -o etime`` -- ``[[dd-]hh:]mm:ss`` -- in seconds."""
    try:
        days = 0
        if "-" in text:
            d, text = text.split("-", 1)
            days = int(d)
        parts = [int(p) for p in text.split(":")]
    except ValueError:
        return None
    if not 1 <= len(parts) <= 3:
        return None
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts
    return float(((days * 24 + hours) * 60 + minutes) * 60 + seconds)


def _is_daemon_argv(argv: Sequence[str]) -> bool:
    """Does this command line run ``sam3gimpd serve``?

    Recognises the three ways it is started: ``<python> -m sam3gimpd serve``,
    the ``sam3gimpd`` console script (``sam3gimpd.exe`` on Windows), and any
    ``SAM3D_COMMAND`` prefix the launcher itself extended -- which always ends
    in ``serve ... --no-stderr-log`` (:func:`build_command`).  A process that
    merely mentions the name -- ``tail -f sam3gimpd.log``, ``sam3gimpd
    doctor`` -- is not a daemon.
    """
    args = [str(a) for a in argv]
    if "serve" not in args:
        return False
    for i, arg in enumerate(args):
        name = re.split(r"[\\/]", arg)[-1].lower()
        if name in ("sam3gimpd", "sam3gimpd.exe", "sam3gimpd-script.py"):
            return True
        if arg == "-m" and i + 1 < len(args) and (
                args[i + 1] == "sam3gimpd" or args[i + 1].startswith("sam3gimpd.")):
            return True
    return "--no-stderr-log" in args


def _spawn_interpreters() -> List[str]:
    """Interpreters :func:`build_command` could have started a daemon with,
    lower-cased, for matching a Windows image path when no command line can
    be read."""
    out: List[str] = []
    chosen = configured_python()
    for path in (chosen and windowless_variant(chosen), chosen, venv_pythonw(), venv_python(),
                 sam3d_executable()):
        if path:
            out.append(os.path.normcase(os.path.abspath(path)))
    return out


def _daemon_process(pid: Optional[int], not_after: Optional[float]) -> Optional[bool]:
    """Is ``pid`` a ``sam3gimpd serve`` process of this user that already
    existed at ``not_after`` (an mtime: of the lockfile, or of runtime.json)?

    ``True`` only when every part is established; ``False`` when any part is
    refuted -- a dead pid, another user's, another program, or a process
    created after the file was written (so the pid in that file was reused);
    ``None`` when the platform cannot tell.  Only ``True`` ever leads to a
    process being signalled.
    """
    pid = _plausible_pid(pid)
    if pid is None or not pid_alive(pid):
        return False
    owned = pid_owned(pid)
    if owned is False:
        return False
    if platform_key() == "windows":
        try:
            argv, image, created = _win_process_details(pid)
        except Exception:  # noqa: BLE001
            return None
        if argv is not None:
            verdict: Optional[bool] = _is_daemon_argv(argv)
        elif image:
            # No command line (older Windows): the image being an interpreter
            # we would have spawned is the best evidence there is.
            verdict = True if os.path.normcase(image) in _spawn_interpreters() else None
        else:
            verdict = None
    else:
        argv, created = _posix_process_details(pid)
        verdict = None if argv is None else _is_daemon_argv(argv)
    if verdict is not True:
        return verdict
    if not_after is not None:
        if created is None:
            return None
        if created > not_after + START_TIME_SLACK_S:
            return False
    return True if owned else None


def _mtime(path: str) -> Optional[float]:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# building the spawn command (API.md section 13)
# --------------------------------------------------------------------------- #
def _client_module():
    """Import the sibling ``client`` module, package-relative or top-level.

    Inside GIMP the plug-in directory is on ``sys.path`` and these are top-level
    modules; under pytest they may be imported either way.  Done lazily so the
    two modules can reference each other without an import cycle.
    """
    try:
        from . import client as mod  # type: ignore[attr-defined]
    except ImportError:
        import client as mod  # type: ignore[no-redef]
    return mod


def this_interpreter_can_run_sam3d() -> bool:
    """Could *this* interpreter serve as the daemon?

    Asks the only question that actually matters -- is ``sam3gimpd`` importable
    here -- instead of trying to infer it from the environment.  An earlier
    version of this guard sniffed ``sys.modules`` for ``gi`` to detect "are we
    inside GIMP", which was wrong twice over: the test-suite's fake-GIMP stubs
    synthesise a ``gi`` module, and ``tools/canvas_harness.py`` imports ``gi``
    for GTK on a perfectly ordinary Linux box.  Both would have been refused.
    """
    try:
        import importlib.util  # noqa: PLC0415

        return importlib.util.find_spec("sam3gimpd") is not None
    except Exception:  # noqa: BLE001 -- a broken import system is a "no"
        return False


def _python_for_spawn() -> Optional[str]:
    """The interpreter to run the daemon with, or ``None`` if none is usable.

    Order: the managed venv's windowless python, then its console python, then
    -- outside GIMP only -- the interpreter running *this* code.

    That last fallback is right for the test-suite and ``tools/canvas_harness.py``
    and *catastrophically* wrong inside GIMP, where ``sys.executable`` is GIMP's
    own embedded Python.  Using it there spawns
    ``...\\WindowsApps\\GIMP...\\pythonw.exe -m sam3gimpd``, which exits instantly with
    "No module named sam3gimpd" and leaves the user reading a confusing error about
    an interpreter they never chose -- after waiting out the whole timeout.  So
    the fallback is taken only when ``sam3gimpd`` is actually importable here; when
    it is not, the caller says the true thing instead: nothing is installed yet.
    """
    windows = platform_key() == "windows"
    for candidate in (venv_pythonw(), venv_python()):
        if os.path.isfile(candidate):
            return candidate
    if not this_interpreter_can_run_sam3d():
        return None
    exe = sys.executable
    if not exe:
        return None
    if windows:
        # Never spawn python.exe on Windows: it flashes a console.  Prefer the
        # pythonw.exe sitting next to it.
        head, tail = os.path.split(exe)
        if tail.lower() == "python.exe":
            pythonw = os.path.join(head, "pythonw.exe")
            if os.path.isfile(pythonw):
                return pythonw
    return exe



def split_command(text: str) -> List[str]:
    """Split a ``SAM3D_COMMAND`` string into argv, correctly on both platforms.

    ``shlex`` alone gets Windows wrong in both directions.  In POSIX mode it
    eats the backslashes, turning ``C:\\Python\\python.exe`` into
    ``C:Pythonpython.exe``.  In non-POSIX mode it keeps the backslashes but also
    keeps the *quotes* as part of the token, so the quoting a user must write
    for ``C:\\Program Files\\...`` produces an argv[0] containing literal ``"``
    characters and CreateProcess fails to find the file.

    So: split without POSIX escaping on Windows to protect the backslashes,
    then strip the surrounding quotes the splitter left behind.  A path with
    spaces still has to be quoted -- unquoted, it is genuinely ambiguous -- but
    quoting it now works, which is what every Windows user will try first.
    """
    if os.name == "nt":
        parts = shlex.split(text, posix=False)
        out: List[str] = []
        for part in parts:
            if len(part) >= 2 and part[0] == part[-1] and part[0] in ('"', "'"):
                part = part[1:-1]
            out.append(part)
        return out
    return shlex.split(text)


def lock_holder_pid() -> Optional[int]:
    """The pid written into the daemon's lockfile, if any.

    ``None`` for a missing, empty or truncated file: a daemon clears its pid
    when it releases the lock, and one that has only just taken the lock may
    not have written it yet.  The pid of a daemon that crashed stays behind,
    so a value here names the holder *only* once :func:`_daemon_process` has
    confirmed it.
    """
    try:
        with open(lock_file(), "r", encoding="ascii", errors="ignore") as fh:
            return _plausible_pid(fh.read().strip().splitlines()[0])
    except (OSError, ValueError, IndexError):
        return None


def _lock_age() -> Optional[float]:
    """Seconds since the lockfile was last written -- i.e. since its holder
    took the lock (the daemon rewrites its pid there on acquiring it)."""
    mtime = _mtime(lock_file())
    return None if mtime is None else max(0.0, time.time() - mtime)


def _zombie_disqualifier(pid: int, min_lock_age: float) -> str:
    """Why ``pid`` must *not* be ended as a stuck lock holder; ``""`` if it may."""
    if lock_holder_pid() != pid:
        return "the lockfile does not name it"
    if not pid_alive(pid):
        return "it is not running"
    lock_mtime = _mtime(lock_file())
    if lock_mtime is None:
        return "there is no lockfile"
    verdict = _daemon_process(pid, not_after=lock_mtime)
    if verdict is None:
        return "it cannot be confirmed to be a sam3gimpd daemon of this user"
    if verdict is False:
        return "it is not a sam3gimpd daemon of this user that took the lock"
    age = max(0.0, time.time() - lock_mtime)
    if age < min_lock_age:
        return "it took the lock %.0f s ago and may still be starting" % age
    return ""


def _end_process(pid: int, note: Callable[[str, str], None], wait: float) -> bool:
    """SIGTERM (``TerminateProcess`` on Windows), then SIGKILL on POSIX if it
    ignores that.  True once the process is gone.  Callers establish first
    that ``pid`` is a daemon of ours; this only does the ending."""
    import signal  # noqa: PLC0415

    try:
        os.kill(pid, getattr(signal, "SIGTERM", 15))
    except ProcessLookupError:
        return True
    except OSError as exc:
        note("zombie", "could not terminate pid %d: %s" % (pid, exc))
        return False
    deadline = time.time() + wait
    while time.time() < deadline and pid_alive(pid):
        time.sleep(0.1)
    if pid_alive(pid) and platform_key() != "windows" and hasattr(signal, "SIGKILL"):
        note("zombie", "pid %d ignored SIGTERM for %.0f s; killing it" % (pid, wait))
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        deadline = time.time() + 3.0
        while time.time() < deadline and pid_alive(pid):
            time.sleep(0.1)
    if pid_alive(pid):
        note("zombie", "pid %d is still alive after %.0f s" % (pid, wait))
        return False
    return True


def terminate_zombie_daemon(pid: int, note: Callable[[str, str], None],
                            wait: float = 6.0, runtime_path: Optional[str] = None,
                            min_lock_age: Optional[float] = None) -> bool:
    """End a daemon that holds the lock but no longer serves.  True if it did.

    Seen in the field: a daemon logged "idle ... exiting", removed its
    runtime.json, and then hung in GPU teardown for minutes -- still holding
    the OS lock.  Every fresh spawn exited with "another instance holds the
    lock" and the plug-in stalled.

    What "holds the lock but is not serving" must *not* be mistaken for is a
    daemon that is starting (it takes the lock before importing torch), one
    that is briefly unresponsive, or an unrelated process that inherited a
    pid a crashed daemon left in the lockfile.  So ``pid`` is left alone
    unless all of these hold: the lockfile names it; it is a ``sam3gimpd
    serve`` process of this user created no later than the lockfile was
    written; it has held the lock for at least ``min_lock_age`` seconds
    (default :data:`COLD_START_BUDGET`); and it does not answer an
    identity-checked ``/hello`` via ``runtime_path``.  Its ``runtime.json`` is
    removed afterwards only if it still describes that daemon.
    """
    rt_path = runtime_path or runtime_file()
    budget = COLD_START_BUDGET if min_lock_age is None else float(min_lock_age)
    pid = _plausible_pid(pid) or 0
    reason = _zombie_disqualifier(pid, budget) if pid else "no pid"
    if reason:
        note("zombie", "leaving pid %s alone: %s" % (pid, reason))
        return False
    info = read_runtime_info(rt_path)
    if info and int(info.get("pid") or 0) == pid:
        client_mod = _client_module()
        c = None
        try:
            c = client_mod.Sam3Client.from_runtime_info(
                info, connect_timeout=HELLO_CONNECT_TIMEOUT, read_timeout=HELLO_READ_TIMEOUT)
            # Only an answer carrying a valid identity proof counts as serving.
            c.hello(check=False, timeout=HELLO_READ_TIMEOUT)
            note("zombie", "pid %d holds the lock and answers /hello; not a zombie" % pid)
            return False
        except client_mod.DaemonIdentityError as exc:
            if not exc.predates_proof:
                # A wrong proof: something else answers on its port.  That
                # is not evidence about this process, so it is left alone.
                note("zombie", "leaving pid %d alone: its port answers with an invalid "
                               "identity proof" % pid)
                return False
        except Exception:
            pass
        finally:
            if c is not None:
                c.close()
    age = _lock_age() or 0.0
    note("zombie", "pid %d has held the instance lock for %.0f s without serving; ending it"
         % (pid, age))
    if not _end_process(pid, note, wait):
        return False
    if info and int(info.get("pid") or 0) == pid:
        delete_runtime_file(rt_path, expect=info)
    return True

def build_command(
    *,
    stub: bool = False,
    parent_pid: Optional[int] = None,
    host: str = "127.0.0.1",
    port: int = 0,
    idle_ttl: Optional[float] = None,
    cache_size: Optional[int] = None,
    device: Optional[str] = None,
    log_level: Optional[str] = None,
    runtime_path: Optional[str] = None,
    command: Optional[Sequence[str]] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> List[str]:
    """Assemble the full argv for ``sam3gimpd serve`` (``API.md`` §13).

    The argv *prefix* (what actually runs Python) is resolved in this order:

    1. the explicit ``command`` argument;
    2. ``$SAM3D_COMMAND`` (shell-quoted), the dev/test escape hatch;
    3. the interpreter the user chose in **Setup ▸ Install ▸ Use an existing
       Python environment**, persisted in ``settings.json``;
    4. ``<venv>/bin/sam3gimpd`` (``Scripts\\sam3gimpd.exe``) if that console script exists;
    5. ``<python> -m sam3gimpd`` where ``<python>`` is ``pythonw.exe`` on Windows.

    Step 3 exists because "point the plug-in at the PyTorch I already have" is a
    UI choice, not a thing anyone should have to set an environment variable
    for.  It sits *below* ``$SAM3D_COMMAND`` so a developer override still wins.

    Everything after the prefix is the documented flag set, so a caller can log
    the result and paste it into a terminal unchanged.
    """
    prefix: List[str]
    if command:
        prefix = [str(c) for c in command]
    else:
        env_cmd = os.environ.get(ENV_COMMAND)
        chosen = configured_python()
        if env_cmd:
            prefix = split_command(env_cmd)
        elif chosen:
            prefix = [windowless_variant(chosen), "-m", "sam3gimpd"]
        else:
            script = sam3d_executable()
            if os.path.isfile(script) and platform_key() != "windows":
                prefix = [script]
            else:
                python = _python_for_spawn()
                if not python:
                    raise SpawnError(
                        "The SAM 3 environment has not been installed yet.\n\n"
                        "Nothing exists at\n  %s\n\n"
                        "Open Setup and either press Install to build it, or use "
                        "\"Use an existing Python environment\" to point at a "
                        "PyTorch install you already have."
                        % venv_python()
                    )
                prefix = [python, "-m", "sam3gimpd"]

    argv = list(prefix)
    if "serve" not in argv:
        argv.append("serve")
    argv += ["--host", str(host), "--port", str(int(port))]
    if stub:
        argv.append("--stub")
    if parent_pid:
        # GIMP's pid, not ours: plug-in processes are short-lived (Constraint C).
        argv += ["--parent-pid", str(int(parent_pid))]
    if idle_ttl is not None:
        argv += ["--idle-ttl", str(int(idle_ttl))]
    if cache_size is not None:
        argv += ["--cache-size", str(int(cache_size))]
    if device:
        argv += ["--device", str(device)]
    if log_level:
        argv += ["--log-level", str(log_level)]
    if runtime_path:
        argv += ["--runtime-file", str(runtime_path)]
    if extra_args:
        argv += [str(a) for a in extra_args]
    # The child's stderr is redirected into sam3gimpd.log by spawn_daemon and
    # the daemon adds its own file handler for that same file; without this
    # every line appeared twice.  Stray prints and tracebacks still arrive via
    # the redirect.
    if "--no-stderr-log" not in argv:
        argv.append("--no-stderr-log")
    return argv


def spawn_daemon(
    argv: Sequence[str],
    *,
    log_path: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
) -> int:
    """Start the daemon **detached** and return its pid.  Never waits on it.

    Windows (P0): ``DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`` -- no console
    is created and the child is not in the plug-in's process group, so closing
    GIMP or the plug-in cannot take it down; combined with ``pythonw.exe`` from
    :func:`build_command`, nothing flashes on screen.

    POSIX: ``start_new_session=True`` (``setsid``), which detaches from the
    controlling terminal and the plug-in's process group.

    Both: stdin from ``os.devnull``, stdout+stderr appended to
    ``logs/sam3gimpd.log`` (a detached process must never inherit handles that may
    close under it; created ``0600`` on POSIX), and ``close_fds=True`` so no
    GIMP file descriptor leaks into a process that outlives GIMP.
    """
    argv = [str(a) for a in argv]
    ensure_layout()
    target_log = log_path or server_log()
    try:
        _private_dir(os.path.dirname(target_log))
    except OSError:
        pass

    kwargs: Dict[str, Any] = {
        "cwd": cwd or base_dir(),
        "env": daemon_environ() if env is None else env,
        "close_fds": True,
    }
    if platform_key() == "windows":
        kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    log_fh = None
    devnull = None
    try:
        log_fh = open(target_log, "ab", buffering=0,
                      opener=lambda path, flags: os.open(path, flags, 0o600))
        devnull = open(os.devnull, "rb")
        log_fh.write(
            ("\n=== sam3gimpd spawn %s: %s ===\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(argv)))
            .encode("utf-8", "replace")
        )
        proc = subprocess.Popen(
            argv,
            stdin=devnull,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            **kwargs
        )
        # Reap the old ones *before* recording this one, so that a child that
        # exits at once still has its Popen (and exit status) for the caller.
        _reap_spawned()
        with _SPAWNED_LOCK:
            _SPAWNED.append(proc)
    except OSError as exc:
        raise SpawnError("could not start the sam3gimpd daemon (%s): %s" % (argv[0], exc), argv)
    finally:
        # The parent must not hold these open: it is about to exit, and on
        # Windows an open handle would keep the log file locked.
        for fh in (log_fh, devnull):
            if fh is not None:
                try:
                    fh.close()
                except OSError:
                    pass
    return proc.pid


def shutdown_daemon(info: Dict[str, Any], grace_ms: int = 0) -> bool:
    """Best-effort ``POST /shutdown`` against a daemon we intend to replace."""
    client_mod = _client_module()
    try:
        c = client_mod.Sam3Client.from_runtime_info(
            info, connect_timeout=HELLO_CONNECT_TIMEOUT, read_timeout=HELLO_READ_TIMEOUT
        )
    except Exception:
        return False
    try:
        c.shutdown_daemon(grace_ms=grace_ms, timeout=HELLO_READ_TIMEOUT)
        return True
    except Exception:
        return False
    finally:
        try:
            c.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# the handshake (API.md section 3.3)
# --------------------------------------------------------------------------- #
@dataclass
class LaunchResult:
    """What :func:`find_or_spawn` returns.

    ``client`` is ready to use; the caller owns it and should ``close()`` it.
    ``info`` holds the bearer token, so it is kept out of ``repr()``: a
    LaunchResult that ends up in a log line or an error message must not
    carry the token with it.
    """

    client: Any
    info: Dict[str, Any] = field(repr=False)
    hello: Any
    spawned: bool
    attempts: int = 1
    command: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def port(self) -> int:
        return int(self.info["port"])

    @property
    def pid(self) -> int:
        return int(self.info["pid"])

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


#: What :func:`_probe_runtime` concluded about an existing ``runtime.json``.
_ACCEPT = "accept"            # proved itself, speaks our major: use it
_INCOMPATIBLE = "incompatible"  # proved itself, another API major
_STALE = "stale"              # describes nothing usable; safe to delete (if unchanged)
_OUTDATED = "outdated"        # an older build of our daemon, which has no proof to give
_UNPROVEN = "unproven"        # a wrong proof, or none, from a pid that may be ours: hands off
_BUSY = "busy"                # alive, not answering yet: wait, never delete


def _probe_runtime(
    info: Dict[str, Any],
    rt_path: str,
    client_mod: Any,
    connect_timeout: float,
    read_timeout: float,
    note: Callable[[str, str], None],
) -> Tuple[str, Any]:
    """Steps 2-4 of ``API.md`` §3.3 for one ``runtime.json``: what is behind it?

    Returns ``(verdict, detail)``; ``detail`` is ``(client, hello)`` for
    :data:`_ACCEPT` and the hello (or ``None``) otherwise.  Liveness alone
    never accepts anything -- the daemon must answer the ``/hello`` nonce with
    the proof only the holder of the token can make -- and a failure to answer
    only condemns the file when the pid is dead, another user's, or verifiably
    not a daemon of ours.  A daemon of ours that is slow to answer keeps its
    ``runtime.json``: without it nothing could ever reach that daemon again.

    Of the answers without a valid proof, only one is acted on: *no* proof
    from a hello claiming API 1.0, from a pid that is (or may be) a sam3gimpd
    -- an older build of our daemon (:data:`_OUTDATED`).  A *wrong* proof means
    something accepted a token it does not hold; it is never sent anything
    more and never signalled.
    """
    pid = info.get("pid")
    if not pid_alive(pid):
        note("stale", "pid %s is not running" % pid)
        return _STALE, None
    if pid_owned(pid) is False:
        note("stale", "pid %s belongs to another user; it is not our daemon" % pid)
        return _STALE, None
    c = client_mod.Sam3Client.from_runtime_info(
        info, connect_timeout=connect_timeout, read_timeout=read_timeout)
    try:
        hello = c.hello(check=False, timeout=read_timeout)
    except client_mod.DaemonIdentityError as exc:
        c.close()
        ours = _daemon_process(pid, not_after=_mtime(rt_path))
        if ours is False:
            note("stale", "what answers on port %s is not the daemon that wrote "
                          "runtime.json: %s" % (info.get("port"), exc))
            return _STALE, None
        if getattr(exc, "predates_proof", False):
            # No proof at all, from a hello claiming API 1.0: what an older
            # build of our own daemon sends -- and the pid is, or may be, a
            # sam3gimpd.  It is asked to exit (and ended only if verified).
            note("stale", "pid %s answers like a sam3gimpd older than API %s, which "
                          "cannot prove its identity" % (pid, API_VERSION))
            return _OUTDATED, exc.hello
        # A wrong proof -- or none from something claiming a newer API -- is
        # not an older build of ours.  Nothing is sent to it and nothing is
        # signalled.
        note("stale", "port %s answered without a valid identity proof (%s)"
             % (info.get("port"), getattr(exc, "reason", "?")))
        return _UNPROVEN, exc.hello
    except (client_mod.RequestTimeout, client_mod.ConnectTimeout, client_mod.DaemonDied) as exc:
        # A connect that times out is caught here too, not as "unavailable":
        # the pid is alive and ours, and a stalled accept loop drops
        # connections just as a missing listener does.  _daemon_process decides.
        c.close()
        if _daemon_process(pid, not_after=_mtime(rt_path)) is False:
            note("stale", "GET /hello failed (%s) and pid %s is not a sam3gimpd daemon" % (exc, pid))
            return _STALE, None
        note("wait", "daemon pid %s did not answer /hello yet (%s)" % (pid, exc))
        return _BUSY, None
    except client_mod.Sam3ClientError as exc:
        c.close()
        note("stale", "GET /hello failed: %s" % exc)
        return _STALE, None
    if client_mod.versions_compatible(API_VERSION, hello.api_version):
        return _ACCEPT, (c, hello)
    c.close()
    note("stale", "daemon speaks API %s, plug-in speaks %s" % (hello.api_version, API_VERSION))
    return _INCOMPATIBLE, hello


def _retire_daemon(info: Dict[str, Any], rt_path: str, note: Callable[[str, str], None]) -> None:
    """Make a verified daemon of ours that we cannot use exit, then forget it.

    ``POST /shutdown`` first -- every build of the daemon honours it, and it
    lets the daemon release its lock and GPU in order -- and a signal only if
    it is still there afterwards and still verifiably the same daemon.
    """
    pid = int(info.get("pid") or 0)
    written = _mtime(rt_path)
    note("stale", "asking pid %d to exit" % pid)
    shutdown_daemon(info, grace_ms=0)
    deadline = time.time() + RETIRE_WAIT_S
    while time.time() < deadline and pid_alive(pid):
        time.sleep(0.1)
    if pid_alive(pid) and _daemon_process(pid, not_after=written) is True:
        _end_process(pid, note, 3.0)
    delete_runtime_file(rt_path, expect=info)


_OUTDATED_ADVICE = (
    "The sam3gimpd daemon is older than this plug-in: it cannot answer the "
    "identity check that API %s added, so the plug-in will not send it images. "
    "Open Setup and press \"Update daemon\" (or \"Reinstall / update\") to "
    "install the daemon that ships with this plug-in." % API_VERSION
)


def find_or_spawn(
    *,
    parent_pid: Optional[int] = None,
    stub: bool = False,
    runtime_path: Optional[str] = None,
    command: Optional[Sequence[str]] = None,
    host: str = "127.0.0.1",
    port: int = 0,
    idle_ttl: Optional[float] = None,
    cache_size: Optional[int] = None,
    device: Optional[str] = None,
    log_level: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
    spawn: bool = True,
    timeout: Optional[float] = None,
    poll_interval: float = HANDSHAKE_POLL_INTERVAL,
    max_attempts: int = MAX_SPAWN_ATTEMPTS,
    connect_timeout: float = HELLO_CONNECT_TIMEOUT,
    read_timeout: float = HELLO_READ_TIMEOUT,
    client_read_timeout: float = 30.0,
    on_event: Optional[Callable[[str, str], Any]] = None,
) -> LaunchResult:
    """Return a connected client, spawning the daemon if necessary.

    Implements ``API.md`` §3.3::

        1  read runtime.json ....................... missing/invalid -> SPAWN
        2  pid_alive(info.pid) ..................... dead            -> delete, SPAWN
           pid_owned(info.pid) ..................... another user's -> delete, SPAWN
        3  GET /hello + nonce (1.5s connect, 3.0s read)
             no answer yet, pid is a daemon of ours  -> wait (never delete), retry
             refused / 401 / not a daemon of ours    -> delete, SPAWN
             cannot prove it holds the token         -> never used; an older
                                                        sam3gimpd of ours is asked
                                                        to exit, then SPAWN
        4  api major mismatch ...................... POST /shutdown, delete, SPAWN
        5  ACCEPTED
        6  t0 = time()
        7  launch detached, never wait()
        8  poll every 100 ms for a fresh runtime.json, up to `timeout`
             child exits 0 ("another instance holds the lock", §3.4)
                                                  -> keep polling for the holder's
                                                     runtime.json until `timeout`
        9  back to step 3; at most `max_attempts` spawns

    Every "delete" removes ``runtime.json`` only if it still names the daemon
    that was judged stale.  ``timeout`` defaults to
    :data:`HANDSHAKE_COLD_TIMEOUT`, and also bounds how long a running daemon
    that does not answer is waited for.  Nothing is terminated except a
    verifiably-ours daemon that cannot be used (see
    :func:`terminate_zombie_daemon` and the module docstring).

    ``spawn=False`` turns this into a pure probe: it raises
    :class:`client.DaemonUnavailable` instead of starting or waiting for
    anything.

    ``on_event(kind, message)`` receives ``"remote"``, ``"stale"``,
    ``"spawn"``, ``"wait"``, ``"zombie"``, ``"accepted"`` -- enough for the
    setup dialog's live log without dragging a logging framework into GIMP's
    interpreter.  No message ever contains the token.
    """
    client_mod = _client_module()
    rt_path = runtime_path or runtime_file()
    wait = HANDSHAKE_COLD_TIMEOUT if timeout is None else max(0.0, float(timeout))
    notes: List[str] = []
    argv: List[str] = []

    def _note(kind: str, message: str) -> None:
        notes.append("%s: %s" % (kind, message))
        if on_event is not None:
            try:
                on_event(kind, message)
            except Exception:
                pass

    attempts = 0
    spawned = False
    last_child: Optional[int] = None
    last_error: Optional[BaseException] = None
    busy_until: Optional[float] = None
    zombie_ended = False

    # ---- step 0: a daemon the user runs elsewhere ------------------------ #
    remote = configured_remote() if not command and not os.environ.get(ENV_COMMAND) else None
    if remote is not None:
        r_host, r_port, r_token = remote
        c = client_mod.Sam3Client(r_host, r_port, r_token,
                                  connect_timeout=connect_timeout,
                                  read_timeout=client_read_timeout)
        try:
            hello = c.hello(check=True, timeout=read_timeout)
        except Exception as exc:
            try:
                c.close()
            except Exception:
                pass
            raise SpawnError(
                "The remote daemon at %s did not answer: %s\n\n"
                "Check that sam3gimpd is serving there with --host 0.0.0.0, "
                "that the port is reachable, that the token in Setup matches "
                "its runtime.json, and that it is the version that ships with "
                "this plug-in. Clear the remote address in Setup to go back "
                "to a local daemon." % (c.base_url, exc))
        _note("remote", "using the daemon at %s" % c.base_url)
        return LaunchResult(
            client=c,
            info={"host": r_host, "port": r_port, "token": r_token, "remote": True,
                  "pid": getattr(hello, "pid", 0), "version": getattr(hello, "sam3d_version", "")},
            hello=hello, spawned=False, notes=notes,
        )

    if idle_ttl is None:
        idle_ttl = configured_idle_ttl()

    while True:
        # ---- steps 1-5: is there a usable daemon already? ----------------- #
        info = client_mod.read_runtime_json(rt_path)
        if info is not None:
            verdict, detail = _probe_runtime(info, rt_path, client_mod, connect_timeout,
                                             read_timeout, _note)
            if verdict == _ACCEPT:
                c, hello = detail
                c.read_timeout = float(client_read_timeout)
                _note(
                    "accepted",
                    "daemon pid %s on port %s, api %s, engine %s"
                    % (info.get("pid"), info.get("port"), hello.api_version, hello.engine_mode),
                )
                return LaunchResult(
                    client=c,
                    info=info,
                    hello=hello,
                    spawned=spawned,
                    attempts=max(1, attempts),
                    command=argv,
                    notes=notes,
                )
            if verdict == _BUSY:
                pid = int(info.get("pid") or 0)
                if not spawn:
                    raise client_mod.DaemonUnavailable(
                        "sam3gimpd (pid %d) is running but did not answer /hello" % pid)
                if busy_until is None:
                    busy_until = time.time() + wait
                if time.time() < busy_until:
                    time.sleep(max(poll_interval, 0.5))
                    continue
                busy_until = None
                if not zombie_ended and terminate_zombie_daemon(pid, _note, runtime_path=rt_path):
                    zombie_ended = True
                    continue
                raise DaemonStartTimeout(
                    "sam3gimpd (pid %d) is running but has not answered for %.0f s. It may "
                    "be loading the model on a busy machine; if it stays like this, stop it "
                    "from Setup > Doctor > Stop daemon." % (pid, wait),
                    tail_log(), argv,
                )
            if verdict == _UNPROVEN:
                raise IncompatibleDaemon(
                    "Whatever answers on port %s did not prove it is the daemon recorded in "
                    "%s (pid %s), so the plug-in will not use it or touch it. If that pid is "
                    "a sam3gimpd, stop it from Setup > Doctor > Stop daemon and try again; "
                    "if the problem persists, reinstall the daemon from Setup."
                    % (info.get("port"), rt_path, info.get("pid")))
            if verdict == _OUTDATED:
                if not spawn:
                    raise IncompatibleDaemon(_OUTDATED_ADVICE)
                fresh = spawned and int(info.get("pid") or 0) == last_child
                _retire_daemon(info, rt_path, _note)
                if fresh:
                    # The daemon this call just started is the old one: the
                    # installed package predates the plug-in.  Spawning again
                    # would only start another.
                    raise IncompatibleDaemon(_OUTDATED_ADVICE)
            elif verdict == _INCOMPATIBLE:
                # Step 4: incompatible major -> ask it to go away, then respawn.
                shutdown_daemon(info, grace_ms=0)
                delete_runtime_file(rt_path, expect=info)
                if not spawn:
                    raise IncompatibleDaemon(
                        "running daemon speaks API %s, this plug-in speaks %s"
                        % (getattr(detail, "api_version", "?"), API_VERSION)
                    )
            elif verdict == _STALE:
                delete_runtime_file(rt_path, expect=info)

        # ---- steps 6-9: spawn ---------------------------------------------- #
        if not spawn:
            raise client_mod.DaemonUnavailable(
                "no running sam3gimpd daemon (no usable %s)" % rt_path
            )
        if attempts >= max_attempts:
            raise DaemonStartTimeout(
                "the sam3gimpd daemon did not come up after %d attempt(s)%s"
                % (attempts, "" if last_error is None else " (last error: %s)" % last_error),
                tail_log(),
                argv,
            )
        attempts += 1
        spawned = True

        ensure_layout()
        argv = build_command(
            stub=stub,
            parent_pid=parent_pid,
            host=host,
            port=port,
            idle_ttl=idle_ttl,
            cache_size=cache_size,
            device=device,
            log_level=log_level,
            runtime_path=runtime_path,
            command=command,
            extra_args=extra_args,
        )
        t0 = time.time()
        _note("spawn", " ".join(argv))
        child_pid = spawn_daemon(argv)
        last_child = child_pid
        _note("spawn", "child pid %d, waiting for %s" % (child_pid, rt_path))
        proc = _spawned_proc(child_pid)

        # Step 8: poll for a runtime.json newer than t0.  The 1 s slack absorbs
        # coarse filesystem timestamps (FAT/exFAT on Windows has 2 s
        # granularity for mtime, so this is deliberately generous).
        if _wait_for_runtime(rt_path, t0 - 1.0, wait, poll_interval, _note, proc=proc):
            continue    # loop back to step 3
        code = None
        if proc is not None:
            try:
                code = proc.poll()
            except Exception:  # pragma: no cover
                code = None
        if code is None:
            # Still running: it holds the lock and is still starting.  It is
            # left alone -- the next attempt will find it and wait for it.
            raise DaemonStartTimeout(
                "sam3gimpd did not publish %s within %.0fs" % (rt_path, wait), tail_log(), argv
            )
        if code != 0:
            raise DaemonStartTimeout(
                "sam3gimpd exited immediately (status %s) without starting up" % code,
                tail_log(), argv,
            )
        # A clean, immediate exit is §3.4: "another instance holds the lock".
        # That instance is usually a daemon still starting -- another dialog's,
        # or our own previous attempt's -- so wait for *its* runtime.json.
        state, holder = _await_lock_holder(rt_path, t0 - 1.0, t0 + wait, poll_interval,
                                           _note, child_pid)
        if state == "published":
            continue
        if state == "released":
            last_error = LauncherError(
                "sam3gimpd could not take its instance lock %s%s"
                % (lock_file(), "" if holder is None else
                   " (it names pid %d, which is not a running sam3gimpd)" % holder))
            continue
        if not zombie_ended and holder is not None and terminate_zombie_daemon(
                holder, _note, runtime_path=rt_path):
            zombie_ended = True
            attempts -= 1   # ending a stuck holder is not this attempt's failure
            continue
        raise DaemonStartTimeout(
            "another sam3gimpd (pid %s) holds the instance lock but has not published %s "
            "after %.0f s. It is most likely still starting -- the first start after an "
            "install or a reboot can take minutes -- so try again shortly."
            % (holder, rt_path, max(0.0, time.time() - t0)),
            tail_log(), argv,
        )


def _wait_for_runtime(
    path: str,
    min_mtime: float,
    timeout: float,
    poll_interval: float,
    note: Callable[[str, str], None],
    proc: Any = None,
) -> bool:
    """Poll for a parseable ``runtime.json`` at least as new as ``min_mtime``.

    ``proc`` is the child we just spawned, when we have it.  A daemon that dies
    on startup -- the wrong interpreter, a missing module, an import error --
    will never publish anything, so waiting the full timeout just makes the user
    stare at a frozen dialog for a minute before being told nothing useful.  One
    extra grace poll after the exit is deliberate: the child may have written
    ``runtime.json`` and exited in the same tick.
    """
    deadline = time.time() + float(timeout)
    reported = False
    client_mod = _client_module()
    grace = True
    while time.time() < deadline:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = None
        if mtime is not None and mtime >= min_mtime:
            if client_mod.read_runtime_json(path) is not None:
                return True
        if proc is not None:
            try:
                status = proc.poll()
            except Exception:  # pragma: no cover
                status = None
            if status is not None:
                if not grace:
                    note("spawn", "child exited with status %s" % status)
                    return False
                grace = False
        if not reported and time.time() - (deadline - timeout) > 2.0:
            note("wait", "still waiting for %s" % path)
            reported = True
        time.sleep(poll_interval)
    # One last look: the daemon may have written the file in the final tick.
    return client_mod.read_runtime_json(path) is not None and (
        os.path.exists(path) and os.path.getmtime(path) >= min_mtime
    )


def _await_lock_holder(
    rt_path: str,
    min_mtime: float,
    deadline: float,
    poll_interval: float,
    note: Callable[[str, str], None],
    child_pid: Optional[int],
) -> Tuple[str, Optional[int]]:
    """Our child exited 0 because another process holds the instance lock.

    Wait, until ``deadline``, for that holder to publish.  Returns
    ``(state, holder_pid)``:

    ``"published"``
        a ``runtime.json`` appeared -- the holder's, or any newer than
        ``min_mtime``; go back to step 3 and judge it there.
    ``"released"``
        for :data:`LOCK_SETTLE_S` the lockfile named nobody alive, or only a
        process that is verifiably not a sam3gimpd daemon (a crashed daemon's
        pid, since reused): the lock is free, or held for reasons other than
        a daemon, and a new spawn is the way to find out.
    ``"stuck"``
        the holder is still there at the deadline -- or, when it is a verified
        daemon of ours that has held the lock past :data:`COLD_START_BUDGET`,
        after :data:`ZOMBIE_GRACE_S`.  Whether it is then ended is
        :func:`terminate_zombie_daemon`'s decision, not this function's.
    """
    client_mod = _client_module()
    verdicts: Dict[int, Optional[bool]] = {}
    unclaimed_since: Optional[float] = None
    stuck_since: Optional[float] = None
    announced: Optional[int] = None
    while True:
        now = time.time()
        holder = lock_holder_pid()
        info = client_mod.read_runtime_json(rt_path)
        if info is not None and (info.get("pid") == holder or (_mtime(rt_path) or 0.0) >= min_mtime):
            return "published", holder
        claimed = False
        if holder is not None and holder != child_pid and pid_alive(holder):
            if holder not in verdicts:
                verdicts[holder] = _daemon_process(holder, not_after=_mtime(lock_file()))
            claimed = verdicts[holder] is not False
        if not claimed:
            stuck_since = None
            if unclaimed_since is None:
                unclaimed_since = now
            elif now - unclaimed_since >= LOCK_SETTLE_S:
                return "released", holder
        else:
            unclaimed_since = None
            if announced != holder:
                note("wait", "sam3gimpd pid %d holds the instance lock; waiting for it to "
                             "publish %s" % (holder, rt_path))
                announced = holder
            age = _lock_age()
            if verdicts.get(holder) is True and age is not None and age >= COLD_START_BUDGET:
                if stuck_since is None:
                    stuck_since = now
                elif now - stuck_since >= ZOMBIE_GRACE_S:
                    return "stuck", holder
        if now >= deadline:
            return ("stuck" if claimed else "released"), holder
        time.sleep(poll_interval)
