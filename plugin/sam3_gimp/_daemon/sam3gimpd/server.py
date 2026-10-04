"""The sam3gimpd HTTP server: every endpoint in ``API.md``, standard library only.

Design constraints this module exists to satisfy:

* **Standard library only.**  ``ThreadingHTTPServer`` and ``json``; no flask, no
  fastapi.  The base install has zero runtime dependencies so that
  ``sam3gimpd serve --stub`` works on a machine with no GPU, no torch and no
  weights.
* **The engine is injected.**  This module never imports torch, transformers or
  a concrete engine.  It is constructed with an object satisfying
  :class:`sam3gimpd.engines.base.BaseEngine`, and ``tests/daemon/test_server.py``
  drives the whole HTTP surface with a fake one.
* **Concurrent connections, serial inference** (``API.md`` §1 and §10).  The
  client long-polls ``GET /jobs/{id}`` on one connection while POSTing prompts
  on another; a single-threaded server would deadlock its UI.  Inference runs on
  the one worker thread owned by :class:`sam3gimpd.jobs.JobManager`.
* **Request paths never wait on the engine.**  A cold model load holds the
  engine for a minute; ``/hello``, ``/status`` and every POST still answer in
  milliseconds, because they work from a snapshot of the engine's description
  and wait only briefly for a fresh one (:meth:`Sam3dServer.engine_info`).
* **Bounded exposure.**  Loopback only by default, a ``Host`` guard against DNS
  rebinding, a bearer token on every request, a cap on concurrent connections
  and short timeouts for a request that has not arrived yet (``API.md`` §2).
* **Self-supervision** (``DESIGN.md`` constraint C).  GIMP plug-in processes are
  short-lived, so nothing supervises the daemon.  It watches ``--parent-pid``
  and an idle TTL itself and exits when either says so.

Division of labour with the modules either side of it:

``session.py``  owns the LRU of encoded images and their lifecycle state.
``engines/``    owns everything about how a mask is produced.
``jobs.py``     owns the queue, supersession and progress.
``server.py``   owns HTTP, validation, auth, ``runtime.json`` and the lifecycle.
"""

from __future__ import annotations

import hmac
import inspect
import ipaddress
import json
import logging
import math
import os
import re
import secrets
import socket
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlsplit

from . import paths
from .engines.base import (
    EngineInfo,
    ImageData,
    PromptResult,
    default_canvas_for,
)
from .jobs import EngineResult, Job, JobManager, JobResult, Stage
from .session import ImageSession, SessionCache, SessionState, compute_image_id
from .types import (
    API_VERSION,
    DEFAULT_SCORE_THRESHOLD,
    NONCE_HEADER,
    ApiError,
    CanvasTransform,
    Engine as EngineKind,
    ErrorCode,
    HelloResponse,
    ImageAccepted,
    JobAccepted,
    JobState,
    Limits,
    Point,
    PointPrompt,
    PromptKind,
    ResultHeader,
    RuntimeInfo,
    Size,
    StatusResponse,
    TextPrompt,
    is_valid_nonce,
    nonce_proof,
    pack_result,
)

__all__ = [
    "Sam3dServer",
    "Sam3dHandler",
    "InstanceLock",
    "pid_alive",
    "host_header_is_loopback",
    "parse_text_prompt",
    "parse_point_prompt",
    "DEFAULT_IDLE_TTL",
    "DEFAULT_CACHE_SIZE",
    "MAX_CONNECTIONS",
    "CONTENT_TYPE_JSON",
    "CONTENT_TYPE_OCTET",
    "CONTENT_TYPE_RESULT",
]


def _build_hash() -> str:
    """Source hash for ``/hello`` (``sam3gimpd.build_hash``), imported lazily."""
    from . import build_hash  # noqa: PLC0415
    return build_hash()


LOG = logging.getLogger("sam3gimpd.server")

CONTENT_TYPE_JSON = "application/json; charset=utf-8"
CONTENT_TYPE_OCTET = "application/octet-stream"
CONTENT_TYPE_RESULT = "application/vnd.sam3.result+binary"

#: 30 minutes, the CLI's default too (``API.md`` §13).
DEFAULT_IDLE_TTL = 1800.0
DEFAULT_CACHE_SIZE = 3
DEFAULT_MAX_QUEUE = 16
#: How often the self-supervision thread checks the parent pid and the idle TTL.
MONITOR_TICK_S = 1.0
#: Connections served at once.  The plug-in holds a handful; beyond this a new
#: connection is closed on accept rather than given a thread (``API.md`` §1).
MAX_CONNECTIONS = 32
#: A new connection must start its first request within this long.
FIRST_REQUEST_TIMEOUT_S = 10.0
#: Once a request has begun, its request line and headers must all arrive
#: within this long in total -- not per read, which a peer trickling a byte at
#: a time would never trip.
REQUEST_HEAD_DEADLINE_S = 10.0
#: A keep-alive connection with no request for this long is closed.  The
#: plug-in's client reconnects transparently when that happens.
KEEPALIVE_IDLE_TIMEOUT_S = 120.0
#: Per-read and per-write budget once a request has begun: request line,
#: headers, body and response.
REQUEST_IO_TIMEOUT_S = 15.0
#: A request body the handler did not need is read and discarded, up to this
#: size, so the connection stays usable; anything larger closes it instead.
DRAIN_LIMIT_BYTES = 64 * 1024
#: How long ``/hello`` and ``/status`` wait for a fresh engine description
#: before answering from the last one.
ENGINE_INFO_WAIT_S = 0.25
#: Smallest box side a point prompt may carry: the PVS engine rejects anything
#: under a pixel as degenerate, and that belongs in a 400, not a failed job.
MIN_PROMPT_BOX_SIDE = 1.0

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,%d}$" % Limits.MAX_REQUEST_ID_CHARS)
_PORT_RE = re.compile(r"^[0-9]{1,5}$")


# --------------------------------------------------------------------------- #
# process helpers
# --------------------------------------------------------------------------- #
def pid_alive(pid: Optional[int]) -> bool:
    """Is that process still running?

    POSIX: ``kill(pid, 0)``, with ``EPERM`` meaning "alive but not ours".
    Windows: ``OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)`` then
    ``GetExitCodeProcess`` via ctypes, which is stdlib, so the plug-in can
    mirror it exactly (``API.md`` §3.3).
    """
    if not pid or int(pid) <= 0:
        return False
    pid = int(pid)
    if os.name == "nt":
        return _pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _pid_alive_windows(pid: int) -> bool:
    """``OpenProcess`` is necessary but not sufficient.

    It succeeds for a process that has already exited as long as anything
    still holds a handle to it -- the launcher's own ``Popen`` object does --
    so the exit code must also still read ``STILL_ACTIVE``.  (A process that
    exits *with* 259 reads as alive; nothing of ours does.)
    """
    try:
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE,
                                                ctypes.POINTER(wintypes.DWORD))
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    except Exception:  # noqa: BLE001
        return True


def _rss_bytes() -> Optional[int]:
    """Resident set size: :func:`modelmgr._rss_bytes`, which knows every
    platform (Windows included), so ``/status`` and the engine agree."""
    try:
        from .modelmgr import _rss_bytes as rss  # noqa: PLC0415  (stdlib-only import)

        return rss()
    except Exception:  # noqa: BLE001
        return None


class InstanceLock:
    """Single-instance lockfile (``API.md`` §3.4).

    ``fcntl.flock(LOCK_EX | LOCK_NB)`` on POSIX, ``msvcrt.locking`` on Windows.
    A second daemon that cannot take the lock exits 0 quietly and leaves the
    holder's ``runtime.json`` authoritative -- which is why two plug-in
    invocations racing to spawn converge on one daemon instead of fighting.

    The file holds the holder's pid while the lock is held and nothing after
    a clean release, so a pid read from it names a live holder (or, after a
    crash, a dead one -- never a process that exited cleanly and whose pid
    has since been reused).
    """

    #: Byte locked on Windows -- past anything the file ever contains, so the
    #: pid at offset 0 stays readable by other processes.
    LOCK_OFFSET = 1 << 20

    def __init__(self, path: Optional[os.PathLike] = None) -> None:
        self.path = str(path if path is not None else paths.lock_file())
        self._fd: Optional[int] = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> bool:
        paths.ensure_dir(os.path.dirname(self.path) or ".")
        try:
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError as exc:
            LOG.warning("cannot open lockfile %s: %s", self.path, exc)
            return False
        try:
            if os.name == "nt":
                import msvcrt  # noqa: PLC0415

                # Lock a byte far past the pid, not byte 0.  msvcrt locks are
                # *mandatory*: another process cannot read a locked range, and
                # the launcher must read the pid written at offset 0 to tell a
                # zombie holder from a live one.  LK_NBLCK works beyond EOF.
                os.lseek(fd, self.LOCK_OFFSET, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl  # noqa: PLC0415

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        try:
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, ("%d\n" % os.getpid()).encode("ascii"))
        except OSError:
            pass
        self._fd = fd
        return True

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        # Empty the file while the lock is still ours, so nobody reads our pid
        # after we are gone -- the launcher would take it for a live holder, or
        # signal whatever unrelated process inherits the number.
        try:
            os.ftruncate(fd, 0)
        except OSError:
            pass
        try:
            if os.name == "nt":
                import msvcrt  # noqa: PLC0415

                os.lseek(fd, self.LOCK_OFFSET, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl  # noqa: PLC0415

                fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass

    def __enter__(self) -> "InstanceLock":
        if not self.acquire():
            raise RuntimeError("another sam3gimpd instance holds %s" % self.path)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


# --------------------------------------------------------------------------- #
# engine plumbing
# --------------------------------------------------------------------------- #
def _accepts_progress(fn: Any) -> bool:
    """Does this engine method take a ``progress`` argument?"""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True
    if "progress" in params:
        return True
    return any(p.kind == p.VAR_KEYWORD for p in params.values())


def _dtype_name(value: Any) -> str:
    if value is None:
        return "none"
    text = str(value)
    return text[6:] if text.startswith("torch.") else text


def _as_size(value: Any, default: Tuple[int, int] = (1008, 1008)) -> Size:
    if isinstance(value, Size):
        return value
    if isinstance(value, dict):
        return Size(int(value["width"]), int(value["height"]))
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return Size(int(value[0]), int(value[1]))
    return Size(default[0], default[1])


def describe_engine(engine: Any) -> EngineInfo:
    """``EngineInfo`` for any engine, real or duck-typed.

    ``BaseEngine.describe()`` is the contract; the attribute fallback exists so
    that a three-line fake in a test does not need to implement it.
    """
    fn = getattr(engine, "describe", None)
    if callable(fn):
        info = fn()
        if isinstance(info, EngineInfo):
            return info
        if isinstance(info, dict):
            return EngineInfo(
                mode=str(info.get("engine_mode", info.get("mode", "stub"))),
                device=str(info.get("device", "cpu")),
                dtype=_dtype_name(info.get("dtype", "float32")),
                capabilities=list(info.get("capabilities", [])),
                torch_available=bool(info.get("torch_available", False)),
                weights_available=bool(info.get("weights_available", False)),
                model_canvas=_as_size(info.get("model_canvas")),
                models_loaded=list(info.get("models_loaded", [])),
                detail=dict(info.get("detail", {})),
            )
    return EngineInfo(
        mode=str(getattr(engine, "MODE", None) or getattr(engine, "mode", "stub")),
        device=str(getattr(engine, "device", "cpu")),
        dtype=_dtype_name(getattr(engine, "dtype", "float32")),
        capabilities=list(getattr(engine, "capabilities", ("pcs", "pvs"))),
        torch_available=bool(getattr(engine, "torch_available", False)),
        weights_available=bool(getattr(engine, "weights_available", False)),
        model_canvas=_as_size(getattr(engine, "model_canvas", None)),
        models_loaded=list(getattr(engine, "models_loaded", ()) or ()),
    )


# --------------------------------------------------------------------------- #
# the Host guard
# --------------------------------------------------------------------------- #
def _is_loopback_name(name: str) -> bool:
    """``localhost`` or a loopback IP literal -- brackets and port already gone."""
    name = (name or "").strip().lower()
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def host_header_is_loopback(value: Optional[str]) -> bool:
    """Does a ``Host`` header name the local loopback interface?

    Exactly ``localhost`` or a loopback IP literal (``127.0.0.0/8``, ``::1``),
    with an optional numeric port, IPv6 in brackets.  Nothing is resolved and
    nothing is matched by prefix: ``127.attacker.example`` resolves wherever
    its owner likes, which is the whole of a DNS-rebinding attack.
    """
    host = (value or "").strip()
    if host.startswith("["):
        end = host.find("]")
        if end < 0:
            return False
        rest = host[end + 1:]
        if rest and not (rest.startswith(":") and _PORT_RE.match(rest[1:])):
            return False
        try:
            return ipaddress.IPv6Address(host[1:end]).is_loopback
        except ValueError:
            return False
    if host.count(":") == 1:
        host, port = host.split(":")
        if not _PORT_RE.match(port):
            return False
    # More than one colon and no brackets can only be a bare IPv6 literal.
    return _is_loopback_name(host)


# --------------------------------------------------------------------------- #
# the socket layer
# --------------------------------------------------------------------------- #
def _abort_connection(conn: socket.socket) -> None:
    """Make a read blocked on ``conn`` in a handler thread return now.

    ``shutdown()`` does that on Linux and macOS.  Windows leaves a pending read
    waiting after a shutdown, and a plain ``close()`` is deferred while the
    handler still reads through ``conn.makefile()``; closing the OS handle
    itself is what aborts the call there.  The socket object is detached first,
    so the handler's own cleanup fails cleanly instead of reaching a handle
    number Windows may already have handed to another socket.
    """
    try:
        conn.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    if os.name != "nt":
        return
    try:
        handle = conn.detach()
    except OSError:
        return
    if handle is not None and handle != -1:
        try:
            socket.close(handle)
        except OSError:
            pass


def _configure_listen_socket(sock: Any, windows: bool) -> None:
    """Address-reuse policy for the listening socket, before ``bind``.

    POSIX: ``SO_REUSEADDR`` so a restart can rebind while old connections sit
    in TIME_WAIT; it never lets a second process share an active listener.
    Windows: ``SO_REUSEADDR`` means the opposite -- another process may bind
    the same port and take connections meant for us -- so it gets
    ``SO_EXCLUSIVEADDRUSE`` instead.
    """
    if windows:
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive is not None:
            sock.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        return
    reuse = getattr(socket, "SO_REUSEADDR", None)
    if reuse is not None:
        sock.setsockopt(socket.SOL_SOCKET, reuse, 1)


class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True          # never join handler threads on close
    # Decided per platform in server_bind() instead; see _configure_listen_socket.
    allow_reuse_address = False
    request_queue_size = 64        # >= 8 simultaneous connections (API.md §1)

    def __init__(self, addr, handler, app: "Sam3dServer",
                 max_connections: int = MAX_CONNECTIONS,
                 family: int = socket.AF_INET) -> None:
        self.app = app
        self.address_family = family
        self.max_connections = max(1, int(max_connections))
        self._slots_lock = threading.Lock()
        self._active = 0
        self._refused = 0
        #: Connections part-way through a request head, and when it must be in.
        self._heads: Dict[Any, float] = {}
        super().__init__(addr, handler)

    @property
    def active_connections(self) -> int:
        with self._slots_lock:
            return self._active

    # -- request-head deadlines -------------------------------------------- #
    def begin_head(self, connection: Any, deadline: float) -> None:
        with self._slots_lock:
            self._heads[connection] = deadline

    def end_head(self, connection: Any) -> bool:
        """Stop the clock.  False if the deadline had already cut it off."""
        with self._slots_lock:
            return self._heads.pop(connection, None) is not None

    def service_actions(self) -> None:  # noqa: D102
        # Runs on the accept loop every poll interval: cut off any connection
        # whose request head is overdue.  The blocked read then ends and the
        # handler thread with it, freeing its slot -- no timer thread per request.
        now = time.monotonic()
        with self._slots_lock:
            late = [conn for conn, deadline in self._heads.items() if deadline <= now]
            for conn in late:
                del self._heads[conn]
        for conn in late:
            _abort_connection(conn)

    def server_bind(self) -> None:  # noqa: D102
        _configure_listen_socket(self.socket, windows=(os.name == "nt"))
        super().server_bind()

    def process_request(self, request, client_address) -> None:  # noqa: D102
        # One thread per connection, but only so many.  A local process that
        # opens connections and never speaks would otherwise cost a thread
        # each for as long as it liked.
        with self._slots_lock:
            admit = self._active < self.max_connections
            if admit:
                self._active += 1
            else:
                self._refused += 1
                refused = self._refused
        if not admit:
            if refused == 1 or refused % 100 == 0:
                LOG.warning("connection limit (%d) reached; refused %d connection(s) so far",
                            self.max_connections, refused)
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._release_slot()
            raise

    def process_request_thread(self, request, client_address) -> None:  # noqa: D102
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._release_slot()

    def _release_slot(self) -> None:
        with self._slots_lock:
            self._active = max(0, self._active - 1)

    def handle_error(self, request, client_address) -> None:  # noqa: D102
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, socket.timeout)):
            return
        LOG.debug("connection error from %s", client_address, exc_info=True)


class Sam3dServer:
    """Owns the socket, the token, the job queue, the image cache and the clock.

    Typical use::

        server = Sam3dServer(engine=my_engine).start()
        ...                                  # serving on server.port
        server.request_shutdown(grace_ms=0)
        server.wait_closed()

    :meth:`run` does the whole lifecycle and blocks; that is what ``cli.py``
    calls.
    """

    def __init__(self,
                 engine: Any,
                 host: str = "127.0.0.1",
                 port: int = 0,
                 token: Optional[str] = None,
                 parent_pid: Optional[int] = None,
                 idle_ttl: float = DEFAULT_IDLE_TTL,
                 cache_size: int = DEFAULT_CACHE_SIZE,
                 runtime_file: Optional[os.PathLike] = None,
                 write_runtime: bool = True,
                 lock: Optional[InstanceLock] = None,
                 hard_exit_after_s: Optional[float] = None,
                 max_queue: int = DEFAULT_MAX_QUEUE,
                 cache: Optional[SessionCache] = None,
                 log_path: Optional[str] = None,
                 version: Optional[str] = None,
                 max_connections: int = MAX_CONNECTIONS) -> None:
        self.engine = engine
        self.host = host
        self.requested_port = int(port)
        self.token = token or secrets.token_urlsafe(32)
        self.parent_pid = int(parent_pid) if parent_pid else None
        self.idle_ttl = float(idle_ttl)
        self.write_runtime = bool(write_runtime)
        self.runtime_file = str(runtime_file) if runtime_file else str(paths.runtime_file())
        self.log_path = log_path
        self.lock = lock
        self.max_connections = int(max_connections)
        #: When set, a shutdown request arms a watchdog that ends the *process*
        #: by force if teardown stalls.  Only the real daemon process (cli.py)
        #: sets it: an embedder -- the test-suite, a harness -- that calls
        #: request_shutdown() in its own process must never be exited from under.
        self.hard_exit_after_s = hard_exit_after_s
        self.cache = cache if cache is not None else SessionCache(
            capacity=cache_size, release=self._release_session)
        self.started_at = time.time()
        self.last_request_at = self.started_at
        self.last_error: Optional[Dict[str, Any]] = None
        self.exit_reason: Optional[str] = None

        from . import __version__ as _pkg_version  # local: keeps the import graph flat
        self.version = version or _pkg_version
        #: The source hash ``/hello`` reports, taken now.  Hashed lazily, a
        #: file replaced on disk before the first ``/hello`` would be reported
        #: as the running build when it is not.
        self.build = _build_hash()

        #: Only a non-loopback bind (the P1 remote-GPU case) skips the Host guard.
        self._host_check = self._is_loopback_literal(host)
        #: Reserved for capabilities the daemon gains later.  Contour tracing
        #: is deliberately *not* one: masks travel as soft uint8 and the client
        #: re-thresholds them locally, so polygons are traced plug-in side.
        self._extra_capabilities: List[str] = []
        #: Serialises every step that pairs a cache entry with queue work --
        #: upload, prompt, delete -- so each sees the other's result whole: an
        #: upload never answers ``cached`` before its encode job exists, a
        #: prompt is never queued between a failed image's reset and its retry,
        #: and a DELETE never cancels the encode of a re-upload racing it.
        #: Held only for bookkeeping; nothing under it calls the engine.
        self._admit_lock = threading.Lock()

        #: The engine's last self-description (see :meth:`engine_info`).
        self._info: Optional[EngineInfo] = None
        self._memory: Dict[str, Any] = {}
        self._info_lock = threading.Lock()
        self._info_pending: Optional[threading.Event] = None

        self.jobs = JobManager(max_queue=max_queue, on_finished=self._on_job_finished)
        self._httpd: Optional[_HTTPServer] = None
        self._serve_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._closed = threading.Event()
        self._state_lock = threading.RLock()
        self._shutting_down = False
        self._close_started = False
        self._runtime_written = False

    # -- lifecycle ---------------------------------------------------------- #
    @property
    def port(self) -> int:
        if self._httpd is None:
            return self.requested_port
        return int(self._httpd.server_address[1])

    @property
    def shutting_down(self) -> bool:
        with self._state_lock:
            return self._shutting_down

    def start(self) -> "Sam3dServer":
        """Bind, start the worker and the monitor, publish ``runtime.json``.

        The socket is listening *before* ``runtime.json`` appears, so a client
        that sees the file can always connect (``API.md`` §3.2).
        """
        # The first description is taken here, before anything is served and
        # before any model can be loading, so it is the one call that may
        # safely ask the engine directly.
        info = self.engine_info()
        family = socket.AF_INET6 if ":" in (self.host or "") else socket.AF_INET
        self._httpd = _HTTPServer((self.host, self.requested_port), Sam3dHandler, self,
                                  max_connections=self.max_connections, family=family)
        self.jobs.start()
        self._serve_thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.2},
            name="sam3gimpd-http", daemon=True)
        self._serve_thread.start()
        self._monitor_thread = threading.Thread(
            target=self._monitor, name="sam3gimpd-monitor", daemon=True)
        self._monitor_thread.start()
        if self.write_runtime:
            self._write_runtime()
        LOG.info("sam3gimpd %s (build %s) listening on %s:%d (engine=%s, idle_ttl=%.0fs, "
                 "parent=%s)", self.version, self.build, self.host, self.port, info.mode,
                 self.idle_ttl, self.parent_pid)
        return self

    def run(self) -> int:
        """Serve until shutdown; returns the process exit code."""
        self.start()
        try:
            self.wait_closed()
        except KeyboardInterrupt:
            self.request_shutdown(grace_ms=0, reason="keyboard-interrupt")
            self.wait_closed(timeout=5.0)
        return 0

    def wait_closed(self, timeout: Optional[float] = None) -> bool:
        return self._closed.wait(timeout)

    def request_shutdown(self, grace_ms: float = 500.0,
                         reason: str = "requested") -> bool:
        """Begin an orderly shutdown.  False if one is already under way.

        Every request after this returns ``503 shutting_down`` (§6.8).
        """
        with self._state_lock:
            if self._shutting_down:
                return False
            self._shutting_down = True
            self.exit_reason = reason
        LOG.info("shutdown requested (%s), grace %.0f ms", reason, grace_ms)
        threading.Thread(target=self._do_shutdown, args=(grace_ms,),
                         name="sam3gimpd-shutdown", daemon=True).start()
        if self.hard_exit_after_s:
            self._arm_exit_watchdog(self.hard_exit_after_s)
        return True

    def _do_shutdown(self, grace_ms: float) -> None:
        # §6.8: nothing queued starts from here on -- stop() cancels it all at
        # once -- and the running job may finish, or is abandoned after
        # grace_ms.  It is never interrupted; the process stops waiting for it.
        self.jobs.stop(timeout=0.0)
        deadline = time.monotonic() + max(0.0, float(grace_ms)) / 1000.0
        while time.monotonic() < deadline and self.jobs.running_job() is not None:
            time.sleep(0.02)
        self.close()

    #: How long a second :meth:`close` waits for the first one to finish.
    CLOSE_WAIT_S = 5.0

    def close(self) -> None:
        """Tear everything down.  Idempotent, safe from any thread.

        Only the first call tears down; a later one waits (briefly) for it to
        finish, so the engine is closed exactly once.
        """
        with self._state_lock:
            self._shutting_down = True
            first = not self._close_started
            self._close_started = True
        if not first:
            self._closed.wait(self.CLOSE_WAIT_S)
            return
        if not self.jobs.stop(timeout=2.0):
            running = self.jobs.running_job()
            LOG.warning("job %s is still running at shutdown; not waiting for it",
                        running.job_id if running is not None else "?")
        httpd, self._httpd = self._httpd, None
        if httpd is not None:
            for step in (httpd.shutdown, httpd.server_close):
                try:
                    step()
                except Exception:  # noqa: BLE001
                    pass
        # Hand the instance back *now*, before anything that can stall.
        # Freeing the GPU cache and closing a torch engine has hung on Windows
        # for minutes at exit; with this after them, the process sat holding
        # the lock with no runtime.json, every fresh spawn exited with
        # "another instance holds the lock", and the plug-in stalled.  Nothing
        # is served any more at this point, so nothing can race us.
        self._remove_runtime()
        if self.lock is not None:
            self.lock.release()
        try:
            self.cache.clear()
        except Exception:  # noqa: BLE001
            LOG.debug("cache clear failed during shutdown", exc_info=True)
        closer = getattr(self.engine, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:  # noqa: BLE001
                LOG.debug("engine close failed", exc_info=True)
        self._closed.set()

    #: Seconds after a shutdown request before the process is ended by force.
    HARD_EXIT_AFTER_S = 15.0

    def _arm_exit_watchdog(self, delay: Optional[float] = None,
                           exit_fn: Optional[Callable[[int], Any]] = None) -> threading.Thread:
        """Guarantee the process actually goes away after a shutdown request.

        Even after :meth:`close` has run, interpreter finalisation with CUDA
        loaded can block indefinitely on Windows -- a process that has already
        released everything but never exits.  The watchdog is a daemon thread;
        a normal exit takes it down, and only a stuck one ever reaches
        ``os._exit``.  Logs are flushed first so the reason is on record.
        """
        wait = float(self.HARD_EXIT_AFTER_S if delay is None else delay)
        do_exit = exit_fn or os._exit

        def _watch() -> None:
            time.sleep(wait)
            LOG.warning("teardown did not complete %.0f s after shutdown; exiting hard", wait)
            try:
                logging.shutdown()
            except Exception:  # noqa: BLE001
                pass
            do_exit(0)

        t = threading.Thread(target=_watch, name="sam3gimpd-exit-watchdog", daemon=True)
        t.start()
        return t

    # -- runtime.json ------------------------------------------------------- #
    def runtime_info(self) -> RuntimeInfo:
        return RuntimeInfo(
            port=self.port,
            token=self.token,
            pid=os.getpid(),
            version=self.version,
            started_at=self.started_at,
            api_version=API_VERSION,
            host=self.host,
            log_path=self.log_path,
        )

    def _write_runtime(self) -> None:
        try:
            paths.atomic_write_json(self.runtime_file, self.runtime_info().to_dict(),
                                    mode=0o600)
            self._runtime_written = True
            LOG.info("wrote %s", self.runtime_file)
        except OSError as exc:
            LOG.error("cannot write %s: %s", self.runtime_file, exc)

    def _remove_runtime(self) -> None:
        """Delete ``runtime.json`` -- but only while it is still ours.

        A racing daemon that took the file over must not have it deleted by our
        shutdown; the pid inside is the ownership test.
        """
        if not self._runtime_written:
            return
        self._runtime_written = False
        data = paths.read_json(self.runtime_file)
        if isinstance(data, dict) and int(data.get("pid", -1)) not in (os.getpid(), -1):
            LOG.info("leaving %s alone: owned by pid %s", self.runtime_file,
                     data.get("pid"))
            return
        paths.remove_quiet(self.runtime_file)

    # -- self-supervision --------------------------------------------------- #
    def _monitor(self) -> None:
        """Exit when the parent goes away or the daemon has been idle too long.

        Cheap on purpose: one ``kill(pid, 0)`` and one clock read per second.
        There is no supervisor to restart us (``DESIGN.md`` §4) -- exiting is
        the whole strategy, and the next plug-in invocation respawns.
        """
        while not self._closed.is_set():
            time.sleep(MONITOR_TICK_S)
            if self.shutting_down:
                return
            if self.parent_pid and not pid_alive(self.parent_pid):
                LOG.info("parent pid %d is gone; exiting", self.parent_pid)
                self.request_shutdown(grace_ms=0, reason="parent-exited")
                return
            if self.idle_ttl and self.idle_ttl > 0:
                counts = self.jobs.counts()
                busy = counts["queued"] or counts["running"]
                idle = time.time() - self.last_request_at
                if not busy and idle >= self.idle_ttl:
                    LOG.info("idle for %.0f s (ttl %.0f s); exiting", idle, self.idle_ttl)
                    self.request_shutdown(grace_ms=0, reason="idle-timeout")
                    return

    def touch(self) -> None:
        self.last_request_at = time.time()

    def _on_job_finished(self, job: Job) -> None:
        if job.error is not None:
            self.last_error = dict(job.error.to_dict(), at=time.time(), job_id=job.job_id)
        # One line per job, so the log tells the story of a session.  Before
        # this the only INFO between "listening" and "exiting" was cache
        # eviction, and a 60 s cold model load was indistinguishable from a
        # hang.  Every field is read defensively: logging must never fail a job.
        try:
            result = getattr(job, "result", None)
            count = len(getattr(getattr(result, "header", None), "instances", None) or ())
            elapsed = job.elapsed_ms()
            LOG.info("job %s (%s) %s in %.0f ms%s",
                     getattr(job, "job_id", "?"), getattr(job, "engine", "?"),
                     getattr(job, "state", "?"), elapsed,
                     (", %d instance(s)" % count) if count else "")
        except Exception:  # noqa: BLE001
            pass
        # A job is when an engine learns things -- a model loaded, a capability
        # probed -- so refresh the snapshot, without waiting for it.
        self._refresh_engine_info(0.0)

    def _release_session(self, session: ImageSession) -> None:
        """Called once per entry leaving the cache: drop the engine payload."""
        embedding = session.embedding
        dropper = getattr(embedding, "drop", None)
        if callable(dropper):
            try:
                dropper()
            except Exception:  # noqa: BLE001
                LOG.debug("embedding drop failed for %s", session.image_id, exc_info=True)
        session.embedding = None
        if session.encode_job_id:
            # This session's own encode job may now age out of the retention
            # window.  Taken from the session, never looked up by image id: the
            # deferred release of a deleted session must not unpin the encode
            # job of a newer upload of the same pixels.
            self.jobs.unpin(session.encode_job_id)
        LOG.info("released cached image %s", session.image_id)

    def _unpin_image(self, job: Job) -> None:
        """Job cleanup: the image the job pinned when it was accepted."""
        self.cache.unpin(job.image_id)

    # -- description -------------------------------------------------------- #
    def engine_info(self, fresh: bool = False) -> EngineInfo:
        """The engine's self-description, without ever waiting on a model load.

        ``describe()`` is meant to be cheap, but a real engine answers it from
        state its model loader guards, and a cold load holds that for a minute
        or more.  So the server keeps the last answer: ``fresh=True`` asks the
        engine again on a helper thread and waits at most
        :data:`ENGINE_INFO_WAIT_S` for it before settling for the last one.
        Nothing on a request path calls ``describe()`` directly.
        """
        if fresh:
            self._refresh_engine_info(ENGINE_INFO_WAIT_S)
        with self._info_lock:
            info = self._info
        if info is None:
            # Only before start(): nothing can be loading yet.
            info = describe_engine(self.engine)
            with self._info_lock:
                if self._info is None:
                    self._info = info
        return info

    def _refresh_engine_info(self, wait: float) -> None:
        """Ask the engine to describe itself on a helper thread, one at a time."""
        with self._info_lock:
            pending = self._info_pending
            if pending is None:
                pending = self._info_pending = threading.Event()
                try:
                    threading.Thread(target=self._describe_into, args=(pending,),
                                     name="sam3gimpd-describe", daemon=True).start()
                except RuntimeError:
                    self._info_pending = None
                    pending.set()
        if wait > 0:
            pending.wait(wait)

    def _describe_into(self, done: threading.Event) -> None:
        try:
            info = describe_engine(self.engine)
            memory = self._engine_memory()
        except Exception:  # noqa: BLE001 -- keep the last good description
            LOG.debug("engine describe failed", exc_info=True)
        else:
            with self._info_lock:
                self._info = info
                self._memory = memory
        finally:
            with self._info_lock:
                if self._info_pending is done:
                    self._info_pending = None
            done.set()

    def _engine_memory(self) -> Dict[str, Any]:
        """``memory()`` of the engine or of its model manager, if either has one."""
        for owner in (self.engine, getattr(self.engine, "mgr", None)):
            fn = getattr(owner, "memory", None)
            if callable(fn):
                got = fn()
                return dict(got) if isinstance(got, dict) else {}
        return {}

    def canvas_for(self, image: Size) -> Tuple[Size, CanvasTransform]:
        """Canvas geometry for an image, asked of the engine.

        Never assumed here: the client inverts whatever it is told (``API.md``
        §5), so an engine whose processor reports different offsets only has
        to say so.
        """
        fn = getattr(self.engine, "canvas_for", None)
        if callable(fn):
            got = fn(image)
            if isinstance(got, tuple) and len(got) == 2:
                canvas = _as_size(got[0])
                transform = got[1]
                if isinstance(transform, dict):
                    transform = CanvasTransform.from_dict(transform)
                if isinstance(transform, CanvasTransform):
                    return canvas, transform
        info = self.engine_info()
        return default_canvas_for(image, int(info.model_canvas.width))

    def capabilities(self, info: Optional[EngineInfo] = None) -> List[str]:
        """Capability strings, from the snapshot unless ``info`` is given."""
        caps = list((info if info is not None else self.engine_info()).capabilities)
        for extra in self._extra_capabilities:
            if extra not in caps:
                caps.append(extra)
        return caps

    def limits_dict(self) -> Dict[str, Any]:
        return {
            "max_image_side": Limits.MAX_IMAGE_SIDE,
            "min_image_side": Limits.MIN_IMAGE_SIDE,
            "max_upload_bytes": Limits.MAX_UPLOAD_BYTES,
            "max_json_bytes": Limits.MAX_JSON_BYTES,
            "max_text_chars": Limits.MAX_TEXT_CHARS,
            "max_points": Limits.MAX_POINTS,
            "max_boxes": Limits.MAX_BOXES,
            "max_instances": Limits.MAX_INSTANCES,
            "max_long_poll_seconds": Limits.MAX_LONG_POLL_SECONDS,
            "max_queue_depth": self.jobs.max_queue,
        }

    def hello(self, nonce: Optional[str] = None,
              info: Optional[EngineInfo] = None) -> HelloResponse:
        """``GET /hello``.  ``nonce`` is the request's ``X-Sam3-Nonce``, if any.

        A well-formed nonce is answered with ``nonce_proof`` (``API.md`` §2): a
        MAC under the token, which only this process can compute, so a client
        can tell this daemon from anything else that has since taken the port
        a stale ``runtime.json`` names.  A malformed one is ignored.
        """
        info = info if info is not None else self.engine_info(fresh=True)
        return HelloResponse(
            api_version=API_VERSION,
            sam3d_version=self.version,
            engine_mode=info.mode,
            device=info.device,
            dtype=_dtype_name(info.dtype),
            capabilities=self.capabilities(info),
            torch_available=bool(info.torch_available),
            weights_available=bool(info.weights_available),
            model_canvas=info.model_canvas,
            pid=os.getpid(),
            started_at=self.started_at,
            uptime_s=time.time() - self.started_at,
            # Additive: lets a client tell a stale daemon from a current one
            # without a version bump (see sam3gimpd.build_hash).
            extra={"build": self.build},
            limits=self.limits_dict(),
            nonce_proof=nonce_proof(self.token, nonce) if is_valid_nonce(nonce) else None,
        )

    def status(self) -> StatusResponse:
        info = self.engine_info(fresh=True)
        with self._info_lock:
            engine_memory = dict(self._memory)
        counts = self.jobs.counts()
        memory: Dict[str, Any] = {"rss_bytes": _rss_bytes(),
                                  "vram_total": None, "vram_free": None}
        for key in ("vram_total", "vram_free", "vram_reserved"):
            if key in info.detail:
                memory[key] = info.detail[key]
        for key, value in engine_memory.items():
            if value is not None or key not in memory:
                memory[key] = value
        memory["cache_bytes_estimate"] = self.cache.stats().get("bytes_estimate", 0)
        return StatusResponse(
            hello=self.hello(info=info),
            images=self.cache.entries(),
            jobs_running=counts["running"],
            jobs_queued=counts["queued"],
            jobs_total=counts["total"],
            cache_limit=self.cache.capacity,
            idle_ttl_s=self.idle_ttl,
            idle_seconds=time.time() - self.last_request_at,
            parent_pid=self.parent_pid,
            models_loaded=list(info.models_loaded),
            last_error=self.last_error,
            paths={
                "base": str(paths.base_dir()),
                "runtime_file": self.runtime_file,
                "server_log": self.log_path or str(paths.server_log()),
                "crash_log": str(paths.crash_log()),
                "hf_home": str(paths.hf_home()),
                "lock_file": self.lock.path if self.lock is not None else str(paths.lock_file()),
            },
            memory=memory,
        )

    def _require_capability(self, name: str) -> None:
        caps = self.capabilities()
        if name not in caps:
            raise ApiError(ErrorCode.ENGINE_UNAVAILABLE,
                           "this daemon has no %s engine" % name,
                           {"capabilities": caps})

    @staticmethod
    def _is_loopback_literal(host: str) -> bool:
        """Is this bind address loopback?  An empty one keeps the guard on."""
        h = (host or "").strip()
        if h.startswith("[") and h.endswith("]"):
            h = h[1:-1]
        return h == "" or _is_loopback_name(h)

    # -- POST /images ------------------------------------------------------- #
    def accept_image(self, pixels: bytes, width: int, height: int,
                     source_size: Optional[Tuple[int, int]] = None) -> ImageAccepted:
        """Hash, cache and enqueue the encode pass.  Returns immediately (§6.2).

        A new entry is pinned from the moment it exists until its encode job
        settles, so an image whose encode is still queued is never evicted
        (which would leave the encode to run for nothing and prompts to 404).
        """
        image = Size(int(width), int(height))
        canvas, transform = self.canvas_for(image)
        image_id = compute_image_id(pixels, image.width, image.height)
        with self._admit_lock:
            existing = self.cache.peek(image_id)
            if existing is None or existing.is_failed:
                # An encode will be needed.  Refuse now if the queue would
                # refuse it: inserting first could evict an innocent image to
                # make room for an upload that is then turned away.
                self.jobs.check_capacity()
            session, created = self.cache.get_or_create(
                pixels, image.width, image.height,
                model_canvas=canvas, canvas_from_image=transform,
                image_id=image_id, pin=True)
            if source_size:
                session.extra["source_size"] = [int(source_size[0]), int(source_size[1])]

            if not created:
                return ImageAccepted(
                    image_id=session.image_id, job_id=session.encode_job_id or "",
                    cached=True, image=session.image, model_canvas=session.model_canvas,
                    canvas_from_image=session.canvas_from_image, state=session.state)

            try:
                job, _ = self.jobs.submit(
                    engine=EngineKind.ENCODE, image_id=session.image_id,
                    fn=self._encode_fn(session, pixels),
                    request_id="",
                    supersedable=False,   # §10: prompts depend on the encode job
                    pinned=True,          # stays fetchable while the image is cached
                    cleanup=self._unpin_image,
                )
            except ApiError:
                # The queue refused the encode job (or we are shutting down).  Drop
                # the half-built entry rather than leaving an image in the cache
                # that can never become ready.
                self.cache.unpin(session.image_id)
                self.cache.delete(session.image_id)
                raise
            previous, session.encode_job_id = session.encode_job_id, job.job_id
        if previous and previous != job.job_id:
            # A retried encode: the failed attempt may age out of retention now.
            self.jobs.unpin(previous)
        return ImageAccepted(
            image_id=session.image_id, job_id=job.job_id, cached=False,
            image=session.image, model_canvas=session.model_canvas,
            canvas_from_image=session.canvas_from_image, state=job.state)

    def _encode_fn(self, session: ImageSession,
                   pixels: bytes) -> Callable[[Job, Callable[..., None]], None]:
        def _encode(job: Job, progress: Callable[..., None]) -> None:
            data = ImageData(image_id=session.image_id, width=session.image.width,
                             height=session.image.height,
                             pixels=session.pixels or pixels)
            session.mark_running()
            try:
                fn = self.engine.encode_image
                encoded = (fn(data, progress) if _accepts_progress(fn) else fn(data))
                session.mark_ready(encoded, getattr(encoded, "bytes_estimate", 0))
            except ApiError as exc:
                session.mark_failed(exc.info)
                raise
            except BaseException as exc:  # noqa: BLE001 -- record, then let jobs.py wrap it
                session.mark_failed(exc)
                raise
            progress(1.0, Stage.DONE)
            return None
        return _encode

    def delete_image(self, image_id: str) -> Dict[str, Any]:
        """``DELETE /images/{id}``: free now, cancel queued, never interrupt (§6.6)."""
        with self._admit_lock:
            if not self.cache.delete(image_id):
                raise ApiError(ErrorCode.IMAGE_NOT_FOUND, "no cached image with that id",
                               {"image_id": image_id})
            cancelled = self.jobs.cancel_queued_for_image(image_id)
        return {"image_id": image_id, "deleted": True, "cancelled_job_ids": cancelled}

    # -- prompts ------------------------------------------------------------ #
    def _session_for_prompt(self, image_id: str) -> ImageSession:
        """The cached session, pinned for the prompt about to be queued."""
        session = self.cache.get(image_id, pin=True)
        if session is None:
            raise ApiError(ErrorCode.IMAGE_NOT_FOUND,
                           "no cached image with that id; re-upload it",
                           {"image_id": image_id})
        if session.is_failed:
            self.cache.unpin(image_id)
            raise ApiError(ErrorCode.IMAGE_NOT_READY,
                           "the encode pass for this image failed",
                           {"image_id": image_id,
                            "error": session.error.to_dict() if session.error else None})
        return session

    def _submit_prompt(self, image_id: str, prompt: Any, engine_kind: str,
                       method: str, meta: Dict[str, Any]) -> JobAccepted:
        """Queue a prompt job; its image stays pinned until the job settles."""
        with self._admit_lock:
            session = self._session_for_prompt(image_id)

            def _run(job: Job, progress: Callable[..., None]) -> JobResult:
                return self._run_prompt(job, progress, session, prompt, engine_kind,
                                        method, meta, prompt.max_instances)

            try:
                job, superseded = self.jobs.submit(engine=engine_kind, image_id=image_id,
                                                   fn=_run, request_id=prompt.request_id,
                                                   cleanup=self._unpin_image)
            except ApiError:
                self.cache.unpin(image_id)
                raise
        return JobAccepted(job_id=job.job_id, request_id=prompt.request_id,
                           image_id=image_id, engine=engine_kind,
                           state=job.state, superseded_job_ids=superseded)

    def submit_text(self, image_id: str, prompt: TextPrompt) -> JobAccepted:
        self._require_capability("pcs")
        ignored = [] if "exemplar_boxes" in self.capabilities() or not prompt.boxes \
            else ["boxes"]
        meta: Dict[str, Any] = {
            "kind": PromptKind.TEXT,
            "text": prompt.text,
            "score_threshold": float(prompt.score_threshold),
            "max_instances": int(prompt.max_instances),
        }
        if prompt.boxes:
            meta["boxes"] = list(prompt.boxes)
        if ignored:
            meta["ignored"] = ignored
        return self._submit_prompt(image_id, prompt, EngineKind.PCS, "prompt_text", meta)

    def submit_points(self, image_id: str, prompt: PointPrompt) -> JobAccepted:
        self._require_capability("pvs")
        meta: Dict[str, Any] = {
            "kind": PromptKind.POINTS,
            "points": [p.to_dict() for p in prompt.points],
            "multimask": bool(prompt.multimask),
            "max_instances": int(prompt.max_instances),
        }
        if prompt.box is not None:
            meta["box"] = [float(v) for v in prompt.box]
        return self._submit_prompt(image_id, prompt, EngineKind.PVS, "prompt_points", meta)

    def _run_prompt(self, job: Job, progress: Callable[..., None],
                    session: ImageSession, prompt: Any, engine_kind: str,
                    method: str, meta: Dict[str, Any],
                    max_instances: int) -> JobResult:
        """Body of every prompt job: call the engine, then build the frame.

        Runs on the single worker thread, after this image's encode job, so the
        embedding is resident by construction (§10's ordering guarantee), and
        the image has been pinned since the prompt was accepted.
        """
        started = time.time()
        state = session.state
        if state != SessionState.DONE:
            # Never wait here: only this thread can settle an encode, so it
            # would be waiting on itself.  A prompt reaches the worker ahead of
            # a settled encode only when the encode it was queued behind failed
            # and a retry was queued after it -- which is image_not_ready.
            err = session.error
            raise ApiError(ErrorCode.IMAGE_NOT_READY,
                           "the image is not encoded (state %s)" % state,
                           {"image_id": session.image_id, "state": state,
                            "error": err.to_dict() if err else None})
        encoded = session.embedding
        if encoded is None:
            raise ApiError(ErrorCode.IMAGE_NOT_FOUND,
                           "the image left the cache before this prompt ran; re-upload it",
                           {"image_id": session.image_id})
        fn = getattr(self.engine, method, None)
        if not callable(fn):
            raise ApiError(ErrorCode.ENGINE_UNAVAILABLE,
                           "engine does not implement %s" % method,
                           {"engine": engine_kind})

        progress(0.60, Stage.PROMPTING)
        raw = (fn(encoded, prompt, progress) if _accepts_progress(fn)
               else fn(encoded, prompt))
        session.touch()

        progress(0.95, Stage.PACKING)
        elapsed = (time.time() - started) * 1000.0
        if isinstance(raw, PromptResult):
            raw.validate()
            if not raw.prompt:
                raw.prompt = meta
            raw.elapsed_ms = elapsed
            header, blobs = raw.header_and_blobs(job.job_id, job.request_id,
                                                 session.image_id)
        else:
            header, blobs = self._header_from_loose_result(
                raw, job, session, engine_kind, meta, max_instances, elapsed)
        frame = pack_result(header, blobs)
        progress(1.0, Stage.DONE)
        return JobResult(frame=frame, header=header)

    def _header_from_loose_result(self, raw: Any, job: Job, session: ImageSession,
                                  engine_kind: str, meta: Dict[str, Any],
                                  max_instances: int,
                                  elapsed_ms: float) -> Tuple[ResultHeader, List[bytes]]:
        """Adapter for an engine that returns something other than ``PromptResult``.

        ``EngineResult.coerce`` understands tuples, dicts and plain sequences of
        per-instance mappings.  The ordering and clipping the contract promises
        (§8.2) are re-established here, so a loosely written engine still
        produces a conforming frame.
        """
        result = EngineResult.coerce(raw)
        order = sorted(range(len(result.instances)),
                       key=lambda i: -float(result.instances[i].score))
        keep = order[:max(1, int(max_instances))]
        truncated = bool(result.truncated) or len(keep) < len(order)
        instances = []
        blobs = []
        for new_id, idx in enumerate(keep):
            inst = result.instances[idx]
            inst.instance_id = new_id
            instances.append(inst)
            blobs.append(result.blobs[idx])
        header = ResultHeader(
            job_id=job.job_id,
            request_id=job.request_id,
            image_id=session.image_id,
            engine=engine_kind,
            image=session.image,
            model_canvas=session.model_canvas,
            canvas_from_image=session.canvas_from_image,
            instances=instances,
            prompt=meta,
            elapsed_ms=elapsed_ms,
            state=JobState.DONE,
            truncated=truncated,
        )
        return header, blobs


# --------------------------------------------------------------------------- #
# request validation
# --------------------------------------------------------------------------- #
def _require_request_id(body: Dict[str, Any]) -> str:
    value = body.get("request_id")
    if not isinstance(value, str) or not _REQUEST_ID_RE.match(value):
        raise ApiError(ErrorCode.BAD_REQUEST,
                       "request_id must be 1-64 chars of [A-Za-z0-9._:-]",
                       {"field": "request_id"})
    return value


def _as_float(value: Any, field: str) -> float:
    """A finite number.  JSON parses ``NaN``, ``Infinity`` and ``1e400`` (to
    inf) happily, and a 400-digit integer overflows ``float()``; every one of
    them is a bad request here rather than a crash inside a job."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiError(ErrorCode.BAD_REQUEST, "%s must be a number" % field,
                       {"field": field})
    try:
        number = float(value)
    except OverflowError:
        raise ApiError(ErrorCode.BAD_REQUEST, "%s is out of range" % field,
                       {"field": field})
    if not math.isfinite(number):
        raise ApiError(ErrorCode.BAD_REQUEST, "%s must be a finite number" % field,
                       {"field": field})
    return number


def _validate_box(box: Any, field: str, min_side: float = 0.0) -> List[float]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ApiError(ErrorCode.BAD_REQUEST, "%s must be [x0, y0, x1, y1]" % field,
                       {"field": field})
    values = [_as_float(v, field) for v in box]
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ApiError(ErrorCode.BAD_REQUEST,
                       "%s must satisfy x1 > x0 and y1 > y0" % field,
                       {"field": field, "got": values})
    if values[2] - values[0] < min_side or values[3] - values[1] < min_side:
        raise ApiError(ErrorCode.BAD_REQUEST,
                       "%s must be at least %g px on each side" % (field, min_side),
                       {"field": field, "got": values, "min_side": min_side})
    return values


def _as_label(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ApiError(ErrorCode.BAD_REQUEST, "%s label must be 0 or 1" % field,
                       {"field": field})
    return int(value)


def _as_bounded_int(body: Dict[str, Any], field: str, default: int,
                    low: int, high: int) -> int:
    value = body.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ApiError(ErrorCode.BAD_REQUEST, "%s must be an integer" % field,
                       {"field": field})
    if not low <= value <= high:
        raise ApiError(ErrorCode.BAD_REQUEST,
                       "%s must be in [%d, %d]" % (field, low, high),
                       {"field": field, "got": value if abs(value) < 10 ** 12 else None})
    return int(value)


def parse_text_prompt(body: Dict[str, Any]) -> TextPrompt:
    """Validate ``POST /images/{id}/text`` (``API.md`` §6.3 and §15)."""
    request_id = _require_request_id(body)
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ApiError(ErrorCode.BAD_REQUEST, "text is required and must be non-empty",
                       {"field": "text"})
    if len(text) > Limits.MAX_TEXT_CHARS:
        raise ApiError(ErrorCode.BAD_REQUEST,
                       "text exceeds %d characters" % Limits.MAX_TEXT_CHARS,
                       {"field": "text", "limit": Limits.MAX_TEXT_CHARS,
                        "got": len(text)})
    threshold = _as_float(body.get("score_threshold", DEFAULT_SCORE_THRESHOLD),
                          "score_threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ApiError(ErrorCode.BAD_REQUEST, "score_threshold must be in [0, 1]",
                       {"field": "score_threshold", "got": threshold})
    max_instances = _as_bounded_int(body, "max_instances", 64, 1, Limits.MAX_INSTANCES)
    raw_boxes = body.get("boxes", []) or []
    if not isinstance(raw_boxes, list):
        raise ApiError(ErrorCode.BAD_REQUEST, "boxes must be an array",
                       {"field": "boxes"})
    if len(raw_boxes) > Limits.MAX_BOXES:
        raise ApiError(ErrorCode.BAD_REQUEST, "at most %d boxes" % Limits.MAX_BOXES,
                       {"field": "boxes", "limit": Limits.MAX_BOXES})
    boxes: List[Dict[str, Any]] = []
    for entry in raw_boxes:
        if not isinstance(entry, dict) or "box" not in entry:
            raise ApiError(ErrorCode.BAD_REQUEST,
                           "each box is {box: [x0,y0,x1,y1], label: 0|1}",
                           {"field": "boxes"})
        # The engine sees only what was validated: finite floats and a 0/1 label.
        boxes.append({"box": _validate_box(entry["box"], "boxes"),
                      "label": _as_label(entry.get("label", 1), "boxes")})
    return TextPrompt(request_id=request_id, text=text, score_threshold=threshold,
                      max_instances=max_instances, boxes=boxes)


def parse_point_prompt(body: Dict[str, Any]) -> PointPrompt:
    """Validate ``POST /images/{id}/points`` (``API.md`` §6.4 and §15)."""
    request_id = _require_request_id(body)
    raw_points = body.get("points", []) or []
    if not isinstance(raw_points, list):
        raise ApiError(ErrorCode.BAD_REQUEST, "points must be an array",
                       {"field": "points"})
    if len(raw_points) > Limits.MAX_POINTS:
        raise ApiError(ErrorCode.BAD_REQUEST, "at most %d points" % Limits.MAX_POINTS,
                       {"field": "points", "limit": Limits.MAX_POINTS,
                        "got": len(raw_points)})
    points: List[Point] = []
    for item in raw_points:
        if not isinstance(item, dict) or "x" not in item or "y" not in item:
            raise ApiError(ErrorCode.BAD_REQUEST, "each point is {x, y, label}",
                           {"field": "points"})
        label = _as_label(item.get("label", 1), "points")
        points.append(Point(_as_float(item["x"], "points.x"),
                            _as_float(item["y"], "points.y"), label))
    box = body.get("box")
    box_values = (_validate_box(box, "box", min_side=MIN_PROMPT_BOX_SIDE)
                  if box is not None else None)
    if not points and box_values is None:
        raise ApiError(ErrorCode.BAD_REQUEST,
                       "at least one of points or box must be provided",
                       {"fields": ["points", "box"]})
    multimask = body.get("multimask", True)
    if not isinstance(multimask, bool):
        raise ApiError(ErrorCode.BAD_REQUEST, "multimask must be a boolean",
                       {"field": "multimask"})
    max_instances = _as_bounded_int(body, "max_instances", 3, 1, 8)
    return PointPrompt(request_id=request_id, points=points, box=box_values,
                       multimask=bool(multimask), max_instances=max_instances)


# --------------------------------------------------------------------------- #
# the request handler
# --------------------------------------------------------------------------- #
_ROUTES = (
    (re.compile(r"^/hello$"), ("GET",)),
    (re.compile(r"^/status$"), ("GET",)),
    (re.compile(r"^/shutdown$"), ("POST",)),
    (re.compile(r"^/images$"), ("POST",)),
    (re.compile(r"^/images/([^/]+)$"), ("DELETE",)),
    (re.compile(r"^/images/([^/]+)/text$"), ("POST",)),
    (re.compile(r"^/images/([^/]+)/points$"), ("POST",)),
    (re.compile(r"^/jobs/([^/]+)$"), ("GET",)),
)


class Sam3dHandler(BaseHTTPRequestHandler):
    """One HTTP connection.  Runs on its own thread; inference does not.

    Every response carries ``Content-Length`` and nothing is ever chunked, in
    either direction (``API.md`` §1) -- which is what lets the plug-in read a
    binary frame with one sized read.
    """

    protocol_version = "HTTP/1.1"
    server_version = "sam3gimpd"
    sys_version = ""
    #: Applied to the socket by setup(); :meth:`handle_one_request` swaps in
    #: the idle budgets while it waits for each request to begin.
    timeout = REQUEST_IO_TIMEOUT_S

    # -- plumbing ----------------------------------------------------------- #
    @property
    def app(self) -> "Sam3dServer":
        return self.server.app  # type: ignore[attr-defined]

    def setup(self) -> None:  # noqa: D102
        super().setup()
        self._served = 0
        self._body_read = True
        # A response goes out as two writes, head then body.  With Nagle's
        # algorithm on, the second waits for the peer to ACK the first, and a
        # keep-alive peer delays that ACK: 40 ms added to every request on a
        # reused connection -- every poll and every click.
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except (OSError, AttributeError):
            pass

    def handle_one_request(self) -> None:
        """Wait for a request under the idle budget, then read it under the I/O one.

        A connection that never starts a request is dropped after
        :data:`FIRST_REQUEST_TIMEOUT_S`, and one that goes quiet between
        requests after :data:`KEEPALIVE_IDLE_TIMEOUT_S`, instead of holding one
        of the :data:`MAX_CONNECTIONS` slots.  Once a request has begun, its
        head must be complete within :data:`REQUEST_HEAD_DEADLINE_S` in total,
        and every read and write gets :data:`REQUEST_IO_TIMEOUT_S`.
        """
        try:
            self.connection.settimeout(KEEPALIVE_IDLE_TIMEOUT_S if self._served
                                       else FIRST_REQUEST_TIMEOUT_S)
            if not self.rfile.peek(1):
                self.close_connection = True
                return
            self.connection.settimeout(REQUEST_IO_TIMEOUT_S)
        except (OSError, ValueError):     # a timeout is an OSError too
            self.close_connection = True
            return
        self._served += 1
        self.server.begin_head(self.connection,  # type: ignore[attr-defined]
                               time.monotonic() + REQUEST_HEAD_DEADLINE_S)
        try:
            super().handle_one_request()
        finally:
            self.server.end_head(self.connection)  # type: ignore[attr-defined]

    def parse_request(self) -> bool:  # noqa: D102
        ok = super().parse_request()
        if not self.server.end_head(self.connection):  # type: ignore[attr-defined]
            # The deadline shut the socket while the headers were arriving, so
            # what was parsed is a fragment: never act on it.
            self.close_connection = True
            return False
        return ok

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        LOG.debug("%s %s", self.address_string(), fmt % args)

    def log_error(self, fmt: str, *args: Any) -> None:
        LOG.debug("%s %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")

    def do_PUT(self) -> None:  # noqa: N802
        self._dispatch("PUT")

    def do_PATCH(self) -> None:  # noqa: N802
        self._dispatch("PATCH")

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch("HEAD")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._dispatch("OPTIONS")

    # -- dispatch ----------------------------------------------------------- #
    def _dispatch(self, method: str) -> None:
        app = self.app
        split = urlsplit(self.path)
        path = unquote(split.path or "/")
        self._query = parse_qs(split.query or "")
        self._body_read = False
        try:
            self._check_host()
            self._check_auth()
            # Only an authenticated request is activity.  A stranger knocking
            # on the port must not keep the daemon -- and its VRAM -- alive.
            app.touch()
            self._check_api_version()
            allowed = self._route_methods(path)
            if allowed is None:
                raise ApiError(ErrorCode.NOT_FOUND, "no such endpoint", {"path": path})
            if method not in allowed:
                raise ApiError(ErrorCode.METHOD_NOT_ALLOWED,
                               "%s is not allowed here" % method,
                               {"allow": list(allowed)})
            if app.shutting_down:
                # §6.8: once a shutdown is accepted the daemon takes no work of
                # any kind, including a second POST /shutdown.
                raise ApiError(ErrorCode.SHUTTING_DOWN, "the daemon is shutting down")
            self._handle(method, path)
        except ApiError as exc:
            extra = {}
            if exc.code in (ErrorCode.UNAUTHORIZED, ErrorCode.FORBIDDEN_HOST):
                # Expected noise from anything probing the port: one DEBUG line,
                # never a traceback, and never last_error.  The connection is
                # closed too -- a real client has the token, and one that does
                # not must not keep a connection slot alive with keep-alive.
                LOG.debug("refused %s %s from %s: %s", method, path[:100],
                          self.client_address[0] if self.client_address else "?",
                          exc.code)
                self.close_connection = True
            if exc.code == ErrorCode.METHOD_NOT_ALLOWED:
                extra["Allow"] = ", ".join(exc.detail.get("allow", []))
            self._send_error_envelope(exc, extra)
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True
        except Exception as exc:  # noqa: BLE001
            trace_id = secrets.token_hex(6)
            LOG.exception("unhandled error [trace %s] on %s %s", trace_id, method, path)
            app.last_error = {"code": ErrorCode.INTERNAL_ERROR, "message": str(exc),
                              "detail": {"trace_id": trace_id}, "at": time.time()}
            self._send_error_envelope(ApiError(ErrorCode.INTERNAL_ERROR,
                                               "unhandled server error",
                                               {"trace_id": trace_id}))

    @staticmethod
    def _route_methods(path: str) -> Optional[Tuple[str, ...]]:
        for pattern, methods in _ROUTES:
            if pattern.match(path):
                return methods
        return None

    def _handle(self, method: str, path: str) -> None:
        if path == "/hello":
            return self._get_hello()
        if path == "/status":
            return self._get_status()
        if path == "/shutdown":
            return self._post_shutdown()
        if path == "/images":
            return self._post_image()
        match = re.match(r"^/images/([^/]+)/text$", path)
        if match:
            return self._post_text(match.group(1))
        match = re.match(r"^/images/([^/]+)/points$", path)
        if match:
            return self._post_points(match.group(1))
        match = re.match(r"^/images/([^/]+)$", path)
        if match:
            return self._delete_image(match.group(1))
        match = re.match(r"^/jobs/([^/]+)$", path)
        if match:
            return self._get_job(match.group(1))
        raise ApiError(ErrorCode.NOT_FOUND, "no such endpoint", {"path": path})

    # -- guards ------------------------------------------------------------- #
    def _check_host(self) -> None:
        """DNS-rebinding guard (``API.md`` §2).

        A web page in the user's browser must not be able to drive the daemon.
        Skipped when the daemon was deliberately started on a non-loopback
        ``--host`` for the P1 remote-GPU scenario.
        """
        if not self.app._host_check:
            return
        host = self.headers.get("Host")
        if host is None or not host.strip():
            return
        if host_header_is_loopback(host):
            return
        raise ApiError(ErrorCode.FORBIDDEN_HOST,
                       "Host header is not a loopback address", {"host": host[:200]})

    def _check_auth(self) -> None:
        """``Authorization: Bearer <token>``, scheme case-insensitive (RFC 7235).

        Compared as bytes: ``hmac.compare_digest`` raises ``TypeError`` for a
        ``str`` holding anything but ASCII, and a header holds whatever the
        peer sent -- that must be a 401, not an unhandled error.
        """
        header = (self.headers.get("Authorization") or "").strip()
        scheme, _, credentials = header.partition(" ")
        supplied = credentials.strip().encode("utf-8", "surrogateescape")
        expected = self.app.token.encode("utf-8")
        if (scheme.lower() != "bearer" or not supplied
                or not hmac.compare_digest(supplied, expected)):
            raise ApiError(ErrorCode.UNAUTHORIZED, "missing or invalid bearer token")

    def _check_api_version(self) -> None:
        sent = self.headers.get("X-Sam3-Api")
        if not sent:
            return
        try:
            major = int(str(sent).split(".", 1)[0])
        except (TypeError, ValueError):
            raise ApiError(ErrorCode.VERSION_MISMATCH, "malformed X-Sam3-Api header",
                           {"got": sent[:32]})
        if major != int(API_VERSION.split(".", 1)[0]):
            raise ApiError(ErrorCode.VERSION_MISMATCH,
                           "client API major %d != daemon %s" % (major, API_VERSION),
                           {"client": str(sent)[:32], "daemon": API_VERSION})

    # -- body reading ------------------------------------------------------- #
    def _content_length(self, required: bool = True) -> int:
        if self.headers.get("Transfer-Encoding"):
            self.close_connection = True
            raise ApiError(ErrorCode.BAD_REQUEST,
                           "chunked request bodies are not accepted; send Content-Length")
        raw = self.headers.get("Content-Length")
        if raw is None:
            if required:
                raise ApiError(ErrorCode.MISSING_HEADER, "Content-Length is required",
                               {"header": "Content-Length"})
            return 0
        try:
            length = int(raw)
        except (TypeError, ValueError):
            raise ApiError(ErrorCode.BAD_REQUEST, "malformed Content-Length",
                           {"got": raw[:32]})
        if length < 0:
            raise ApiError(ErrorCode.BAD_REQUEST, "negative Content-Length", {"got": raw})
        return length

    def _read_exactly(self, length: int) -> bytes:
        self._body_read = True
        chunks = []
        remaining = length
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 1 << 20))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        body = b"".join(chunks)
        if len(body) != length:
            self.close_connection = True
            raise ApiError(ErrorCode.BAD_REQUEST,
                           "request body ended early: %d of %d bytes"
                           % (len(body), length))
        return body

    def _read_json(self, required: bool = False) -> Dict[str, Any]:
        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if ctype and ctype != "application/json":
            self.close_connection = True
            raise ApiError(ErrorCode.UNSUPPORTED_MEDIA_TYPE,
                           "this endpoint takes application/json", {"got": ctype[:64]})
        length = self._content_length(required=False)
        if length > Limits.MAX_JSON_BYTES:
            self.close_connection = True
            raise ApiError(ErrorCode.PAYLOAD_TOO_LARGE,
                           "JSON body exceeds %d bytes" % Limits.MAX_JSON_BYTES,
                           {"limit": Limits.MAX_JSON_BYTES, "got": length})
        raw = self._read_exactly(length) if length else b""
        if not raw.strip():
            if required:
                raise ApiError(ErrorCode.INVALID_JSON, "a JSON object body is required")
            return {}
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ApiError(ErrorCode.INVALID_JSON, "body is not valid JSON",
                           {"reason": str(exc)[:200]})
        if not isinstance(body, dict):
            raise ApiError(ErrorCode.INVALID_JSON, "body must be a JSON object",
                           {"got": type(body).__name__})
        return body

    def _settle_request_body(self) -> None:
        """Never leave request bytes behind for the next request to trip on.

        On a keep-alive connection a body nobody read would be parsed as the
        next request line.  A small one is read and discarded -- a ``GET`` or
        ``DELETE`` that sent one still gets a usable connection -- and
        anything else (large, chunked, malformed, or rejected before it was
        read) closes the connection after this response.
        """
        if self._body_read:
            return
        self._body_read = True
        if self.headers.get("Transfer-Encoding"):
            self.close_connection = True
            return
        raw = self.headers.get("Content-Length")
        if raw is None:
            return
        try:
            length = int(raw)
        except (TypeError, ValueError):
            self.close_connection = True
            return
        if length == 0:
            return
        if length < 0 or length > DRAIN_LIMIT_BYTES:
            self.close_connection = True
            return
        try:
            remaining = length
            while remaining > 0:
                chunk = self.rfile.read(remaining)
                if not chunk:
                    break
                remaining -= len(chunk)
        except OSError:
            remaining = -1
        if remaining:
            self.close_connection = True

    # -- responses ---------------------------------------------------------- #
    def _send(self, status: int, body: bytes, content_type: str,
              headers: Optional[Dict[str, str]] = None) -> None:
        self._settle_request_body()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Sam3-Api", API_VERSION)
        self.send_header("X-Sam3-Version", self.app.version)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD" and body:
            self.wfile.write(body)

    def _send_json(self, status: int, payload: Any,
                   headers: Optional[Dict[str, str]] = None) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._send(status, body, CONTENT_TYPE_JSON, headers)

    def _send_error_envelope(self, exc: ApiError,
                             headers: Optional[Dict[str, str]] = None) -> None:
        try:
            self._send_json(exc.status, exc.to_envelope(), headers)
        except (ConnectionResetError, BrokenPipeError, OSError):
            self.close_connection = True

    # -- endpoints ---------------------------------------------------------- #
    def _get_hello(self) -> None:
        self._send_json(200, self.app.hello(nonce=self.headers.get(NONCE_HEADER)).to_dict())

    def _get_status(self) -> None:
        self._send_json(200, self.app.status().to_dict())

    def _post_shutdown(self) -> None:
        body = self._read_json(required=False)
        grace = _as_float(body.get("grace_ms", 500), "grace_ms")
        grace = max(0.0, min(10000.0, grace))
        self.close_connection = True
        self._send_json(202, {"ok": True, "pid": os.getpid(), "grace_ms": grace})
        try:
            self.wfile.flush()
        except OSError:
            pass
        self.app.request_shutdown(grace_ms=grace, reason="http-shutdown")

    def _post_image(self) -> None:
        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if ctype and ctype != CONTENT_TYPE_OCTET:
            self.close_connection = True
            raise ApiError(ErrorCode.UNSUPPORTED_MEDIA_TYPE,
                           "POST /images takes application/octet-stream",
                           {"got": ctype[:64]})
        width = self._pixel_dimension("X-Width")
        height = self._pixel_dimension("X-Height")
        length = self._content_length(required=True)
        if length > Limits.MAX_UPLOAD_BYTES:
            self.close_connection = True
            raise ApiError(ErrorCode.PAYLOAD_TOO_LARGE,
                           "upload exceeds %d bytes" % Limits.MAX_UPLOAD_BYTES,
                           {"limit": Limits.MAX_UPLOAD_BYTES, "got": length})
        expected = width * height * 3
        if length != expected:
            self.close_connection = True
            raise ApiError(ErrorCode.PAYLOAD_SIZE_MISMATCH,
                           "body length must be width * height * 3",
                           {"expected": expected, "got": length})
        pixels = self._read_exactly(length)
        source = None
        try:
            sw = self.headers.get("X-Source-Width")
            sh = self.headers.get("X-Source-Height")
            if sw and sh:
                source = (int(sw), int(sh))
        except (TypeError, ValueError):
            source = None
        self._send_json(202, self.app.accept_image(pixels, width, height,
                                                   source).to_dict())

    def _pixel_dimension(self, header: str) -> int:
        raw = self.headers.get(header)
        if raw is None:
            raise ApiError(ErrorCode.MISSING_HEADER, "%s is required" % header,
                           {"header": header})
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError):
            raise ApiError(ErrorCode.BAD_DIMENSIONS, "%s must be an integer" % header,
                           {"header": header, "got": raw[:32]})
        if not Limits.MIN_IMAGE_SIDE <= value <= Limits.MAX_IMAGE_SIDE:
            raise ApiError(ErrorCode.BAD_DIMENSIONS,
                           "%s must be in [%d, %d]"
                           % (header, Limits.MIN_IMAGE_SIDE, Limits.MAX_IMAGE_SIDE),
                           {"header": header, "got": value if abs(value) < 10 ** 12 else None,
                            "min": Limits.MIN_IMAGE_SIDE,
                            "max": Limits.MAX_IMAGE_SIDE})
        return value

    def _post_text(self, image_id: str) -> None:
        prompt = parse_text_prompt(self._read_json(required=True))
        self._send_json(202, self.app.submit_text(image_id, prompt).to_dict())

    def _post_points(self, image_id: str) -> None:
        prompt = parse_point_prompt(self._read_json(required=True))
        self._send_json(202, self.app.submit_points(image_id, prompt).to_dict())

    def _delete_image(self, image_id: str) -> None:
        self._send_json(200, self.app.delete_image(image_id))

    def _get_job(self, job_id: str) -> None:
        wait = max(0.0, min(self._query_float("wait", 0.0),
                            Limits.MAX_LONG_POLL_SECONDS))
        meta = self._query_bool("meta", False)
        status = self.app.jobs.wait(job_id, wait, include_result=meta)
        if status is None:
            raise ApiError(ErrorCode.JOB_NOT_FOUND,
                           "unknown job id, or its retention window expired",
                           {"job_id": job_id})
        if status.state == JobState.DONE and not meta:
            job = self.app.jobs.get(job_id)
            result = job.result if job is not None else None
            if result is None:
                # A finished encode job has no frame: answer with the JSON
                # status rather than inventing an empty result (§6.5).
                self._send_json(200, status.to_dict(),
                                {"X-Sam3-Job-State": status.state,
                                 "X-Sam3-Request-Id": status.request_id})
                return
            head_len = struct.unpack_from("<I", result.frame, 8)[0]
            self._send(200, result.frame, CONTENT_TYPE_RESULT, {
                "X-Sam3-Job-State": status.state,
                "X-Sam3-Header-Length": str(head_len),
                "X-Sam3-Request-Id": status.request_id,
            })
            return
        self._send_json(200, status.to_dict(),
                        {"X-Sam3-Job-State": status.state,
                         "X-Sam3-Request-Id": status.request_id})

    # -- query params ------------------------------------------------------- #
    def _query_float(self, name: str, default: float) -> float:
        values = self._query.get(name)
        if not values:
            return default
        try:
            value = float(values[-1])
        except (TypeError, ValueError):
            value = float("nan")
        # An infinite wait is clamped like any other long one (§15); NaN is
        # not a duration at all.
        if math.isnan(value):
            raise ApiError(ErrorCode.BAD_REQUEST, "%s must be a number" % name,
                           {"param": name, "got": values[-1][:32]})
        return value

    def _query_bool(self, name: str, default: bool) -> bool:
        values = self._query.get(name)
        if not values:
            return default
        value = str(values[-1]).strip().lower()
        if value in ("1", "true", "yes", "on"):
            return True
        if value in ("0", "false", "no", "off", ""):
            return False
        raise ApiError(ErrorCode.BAD_REQUEST, "%s must be 0 or 1" % name,
                       {"param": name, "got": values[-1][:32]})
