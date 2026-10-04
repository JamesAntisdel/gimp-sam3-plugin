"""``sam3gimpd`` command line: ``serve``, ``doctor``, ``download``.

Three subcommands, and the flags are a contract of their own -- the GIMP
plug-in's launcher spawns this process and ``API.md`` §13 fixes the spelling::

    sam3gimpd serve [--host 127.0.0.1] [--port 0] [--stub]
                [--parent-pid PID] [--idle-ttl 1800] [--cache-size 3]
                [--runtime-file PATH] [--log-level info] [--device auto]

``--bind`` is accepted as an alias for ``--host`` and ``--idle-timeout`` for
``--idle-ttl``; both spellings are in use.

Nothing here imports torch at module scope, and ``--stub`` never imports it at
all (``API.md`` §14).  The stub engine is a **product feature**, not a fixture:
it is how the plug-in is developed on a machine with no GPU and no weights.
``doctor`` and ``download`` import torch and ``huggingface_hub`` lazily inside
their handlers, so ``sam3gimpd --help`` stays instant in an environment that has
neither.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import logging.handlers
import os
import shutil
import signal
import sys
import time
import traceback
from typing import Any, Dict, Optional, Sequence

from . import paths

__all__ = ["main", "build_parser", "resolve_engine", "setup_logging"]

LOG = logging.getLogger("sam3gimpd")

DEFAULT_REPO_ID = "facebook/sam3"
GATED_URL = "https://huggingface.co/facebook/sam3"

#: What to tell someone whose environment has no torch engine.  The daemon is
#: installed from its own directory, never by name: the names it goes by are
#: not ours on PyPI, and ``pip install <name>`` would fetch a stranger's code.
INSTALL_HINT = (
    "  Install torch from the PyTorch index, then the daemon from its directory\n"
    "  (plugin/sam3_gimp/_daemon) with its runtime extra, e.g.:\n"
    "      pip install torch==2.9.0 torchvision==0.24.0 \\\n"
    "          --index-url https://download.pytorch.org/whl/cpu   (or cu128 / rocm6.4)\n"
    "      pip install './plugin/sam3_gimp/_daemon[runtime]'\n"
    "  The plug-in's Setup dialog does both for you.\n")

#: The daemon log rotates at this size, keeping this many old copies.
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def _open_private(path: str, mode: str, encoding: Optional[str] = None,
                  errors: Optional[str] = None):
    """``open()``, but a file this creates is user-only (0600) on POSIX.

    ``open()`` creates files ``0666 & ~umask`` -- usually world-readable -- and
    the log names every image id, path and error the daemon saw.  An existing
    file is tightened too.
    """
    flags = os.O_WRONLY | os.O_CREAT
    flags |= os.O_APPEND if "a" in mode else os.O_TRUNC
    fd = os.open(path, flags | getattr(os, "O_BINARY", 0), 0o600)
    if os.name != "nt":
        try:
            os.fchmod(fd, 0o600)
        except OSError:
            pass
    if "b" in mode:
        return open(fd, mode)
    return open(fd, mode, encoding=encoding or "utf-8", errors=errors)


class PrivateRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """A size-capped, user-only log that shares its file with our own stderr.

    The launcher points the detached daemon's stdout/stderr at the same
    ``sam3gimpd.log``, so rotation *copies* the file aside and truncates it in
    place rather than renaming it: a rename would leave that inherited handle
    writing into ``.1`` forever -- and on Windows, where an open file cannot
    be renamed, fail on every record.  Both writers append, so both carry on
    at the new end.
    """

    def _open(self):  # noqa: D102
        return _open_private(self.baseFilename, self.mode, self.encoding,
                             getattr(self, "errors", None))

    def doRollover(self) -> None:  # noqa: N802,D102
        if self.stream is None:
            self.stream = self._open()
        self.stream.flush()
        base = self.baseFilename
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                older, newer = "%s.%d" % (base, i), "%s.%d" % (base, i + 1)
                if os.path.exists(older):
                    os.replace(older, newer)
            with open(base, "rb") as src, _open_private(base + ".1", "wb") as dst:
                shutil.copyfileobj(src, dst)
        self.stream.seek(0)
        self.stream.truncate(0)


def setup_logging(level: str = "info", log_file: Optional[str] = None,
                  stderr: bool = True) -> Optional[str]:
    """Log to ``<base>/logs/sam3gimpd.log`` and (optionally) stderr.

    The detached daemon's stdout/stderr are already redirected to that file by
    the launcher; the explicit file handler means the log is complete even when
    the daemon is started by hand.  The file is user-only and rotates at
    :data:`LOG_MAX_BYTES`.  The bearer token is never logged (§2).
    """
    root = logging.getLogger("sam3gimpd")
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # noqa: BLE001
            pass
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    if stderr:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(fmt)
        root.addHandler(stream)
    resolved = None
    try:
        target = log_file or str(paths.server_log())
        paths.ensure_dir(os.path.dirname(target) or ".")
        file_handler = PrivateRotatingFileHandler(target, maxBytes=LOG_MAX_BYTES,
                                                  backupCount=LOG_BACKUP_COUNT,
                                                  encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
        resolved = target
    except OSError as exc:  # a read-only home must not stop the daemon
        root.warning("cannot open log file: %s", exc)
    return resolved


def write_crash_log(exc: BaseException) -> Optional[str]:
    """Persist the last traceback for the Doctor panel (``DESIGN.md`` §4)."""
    try:
        target = paths.crash_log()
        paths.ensure_dir(target.parent)
        with _open_private(str(target), "w", encoding="utf-8") as fh:
            fh.write("sam3gimpd crash at %s\n\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
            fh.write("".join(traceback.format_exception(type(exc), exc,
                                                        exc.__traceback__)))
        return str(target)
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# engine resolution
# --------------------------------------------------------------------------- #
def _looks_like_engine(obj: Any) -> bool:
    """Duck-check: does this object implement the engine contract?"""
    return all(callable(getattr(obj, name, None))
               for name in ("describe", "encode_image", "prompt_text", "prompt_points"))


def _try_factories(module_name: str, candidates: Sequence[str],
                   *args: Any, **kwargs: Any) -> Optional[Any]:
    """Import ``module_name`` and try each named factory until one yields an engine.

    Tolerant on purpose: the engines package is developed alongside this file,
    so a factory that is absent, renamed or takes a different signature degrades
    to the next candidate instead of crashing the daemon.
    """
    try:
        module = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 -- a missing or broken engine module is not fatal here
        LOG.debug("cannot import %s", module_name, exc_info=True)
        return None
    for name in candidates:
        factory = getattr(module, name, None)
        if factory is None:
            continue
        if not callable(factory):
            engine = factory
        else:
            try:
                engine = factory(*args, **kwargs)
            except TypeError:
                try:
                    engine = factory()
                except Exception:  # noqa: BLE001
                    continue
            except Exception as exc:  # noqa: BLE001
                LOG.warning("%s.%s() failed: %s", module_name, name, exc)
                continue
        if _looks_like_engine(engine):
            LOG.info("using engine %s.%s", module_name, name)
            return engine
    return None


def resolve_engine(stub: bool, device: str = "auto") -> Any:
    """Return the engine object to inject into the server.

    ``server.py`` never imports an engine; this is the only place that decides
    which one exists.  ``--stub`` resolves to ``sam3gimpd.engines.stub.StubEngine``
    -- a fully conforming fake (``API.md`` §14) that imports no torch at all,
    and the reason the plug-in is developable on a machine with no GPU.
    """
    if stub:
        engine = (_try_factories("sam3gimpd.engines.stub",
                                 ("make_engine", "create_engine", "StubEngine"))
                  or _try_factories("sam3gimpd.engines",
                                    ("make_stub_engine", "stub_engine", "StubEngine"))
                  or _try_factories("sam3gimpd.engines", ("make_engine", "create_engine"),
                                    "stub"))
        if engine is not None:
            return engine
        raise SystemExit(
            "sam3gimpd: --stub was requested but no stub engine could be constructed "
            "from sam3gimpd.engines.stub.  The package install is incomplete.")

    engine = (_try_factories("sam3gimpd.engines", ("make_engine", "create_engine"),
                             "torch", device=device)
              or _try_factories("sam3gimpd.engines", ("TorchEngine",), device=device)
              or _try_factories("sam3gimpd.engines.torch_engine",
                                ("make_engine", "TorchEngine"), device=device))
    if engine is not None:
        return engine
    raise SystemExit(
        "sam3gimpd: no torch engine is available in this environment.\n"
        + INSTALL_HINT +
        "  Fetch the gated weights:  sam3gimpd download --token <hf_token>\n"
        "  Or run the fake engine:   sam3gimpd serve --stub")


# --------------------------------------------------------------------------- #
# serve
# --------------------------------------------------------------------------- #
def cmd_serve(args: argparse.Namespace) -> int:
    from .server import InstanceLock, Sam3dServer   # local: keeps `doctor` cheap

    if args.runtime_file:
        os.environ[paths.ENV_RUNTIME_FILE] = str(args.runtime_file)
    paths.ensure_layout()
    log_path = setup_logging(args.log_level, args.log_file,
                             stderr=not args.no_stderr_log)

    lock = InstanceLock()
    if not lock.acquire():
        # §3.4: the holder's runtime.json stays authoritative; exit 0 quietly so
        # two plug-in invocations racing to spawn converge on one daemon.
        LOG.info("another sam3gimpd instance holds %s; exiting", lock.path)
        sys.stderr.write("sam3gimpd: another instance is already running\n")
        return 0

    engine = resolve_engine(stub=args.stub, device=args.device)
    server = Sam3dServer(
        engine=engine,
        host=args.host,
        port=args.port,
        parent_pid=args.parent_pid,
        idle_ttl=args.idle_ttl,
        cache_size=args.cache_size,
        runtime_file=args.runtime_file or None,
        lock=lock,
        log_path=log_path,
        # This process *is* the daemon: if teardown stalls after a shutdown
        # request (CUDA at exit has), end it by force so the lock is freed.
        hard_exit_after_s=Sam3dServer.HARD_EXIT_AFTER_S,
    )

    def _signal(signum: int, _frame: Any) -> None:
        LOG.info("signal %s received", signum)
        server.request_shutdown(grace_ms=0, reason="signal-%s" % signum)

    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, _signal)
            except (ValueError, OSError):
                pass  # not the main thread, or unsupported on this platform

    try:
        code = server.run()
    except BaseException as exc:  # noqa: BLE001 -- record, then re-raise nothing
        where = write_crash_log(exc)
        LOG.critical("fatal error (crash log: %s)", where, exc_info=True)
        try:
            server.close()
        except Exception:  # noqa: BLE001
            pass
        return 1
    if args.print_runtime:
        sys.stdout.write("sam3gimpd exited (%s)\n" % (server.exit_reason or "closed"))
    return code


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #
def _probe_engine() -> Dict[str, Any]:
    """Device, dtype, versions -- via ``modelmgr``, which owns that policy.

    ``modelmgr`` is torch-free to import and probes without ever raising, so
    this works identically with no torch at all and on a CUDA workstation.  The
    hand-rolled fallback below covers a stripped install where ``modelmgr`` is
    missing.
    """
    try:
        from . import modelmgr  # noqa: PLC0415

        probe = modelmgr.probe_torch()
        device = modelmgr.select_device("auto", probe)
        report = dict(probe.to_dict())
        report.update({
            "device": device,
            "dtype": modelmgr.select_dtype(device, "auto", probe),
            "model_id": modelmgr.DEFAULT_MODEL_ID,
            "local_checkpoint": modelmgr.local_checkpoint(),
            "weights_available": modelmgr.weights_available(),
        })
        return report
    except Exception as exc:  # noqa: BLE001
        LOG.debug("modelmgr probe failed", exc_info=True)
        return _probe_torch_fallback(str(exc))


def _probe_torch_fallback(reason: str) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "torch_available": False,
        "torch_version": None,
        "transformers_version": None,
        "device": "cpu",
        "dtype": "float32",
        "cuda_available": False,
        "mps_available": False,
        "probe_error": reason,
    }
    try:
        import importlib.util  # noqa: PLC0415

        if importlib.util.find_spec("torch") is None:
            return report
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        report["error"] = str(exc)
        return report
    report["torch_available"] = True
    report["torch_version"] = getattr(torch, "__version__", None)
    try:
        report["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        pass
    try:
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        report["mps_available"] = bool(mps and mps.is_available())
    except Exception:  # noqa: BLE001
        pass
    # DESIGN.md section 7: cuda -> mps -> cpu; half precision on GPU only, and
    # bf16 only where the hardware has it natively (Ampere, SM 8.0, and up) --
    # modelmgr.select_dtype is the authority and this mirrors it.
    if report["cuda_available"]:
        dtype = "float16"
        try:
            hip = getattr(getattr(torch, "version", None), "hip", None)
            if not hip and torch.cuda.get_device_capability(0)[0] >= 8:
                dtype = "bfloat16"
        except Exception:  # noqa: BLE001
            pass
        report["device"], report["dtype"] = "cuda", dtype
    elif report["mps_available"]:
        report["device"], report["dtype"] = "mps", "float16"
    return report


def _weights_present() -> bool:
    """Is the gated checkpoint on disk?  Offline and cheap, never a network probe."""
    try:
        from . import modelmgr  # noqa: PLC0415

        return bool(modelmgr.weights_available())
    except Exception:  # noqa: BLE001
        pass
    roots = [paths.hf_home(), paths.models_dir()]
    env = os.environ.get("HF_HOME")
    if env:
        roots.append(paths.Path(env))
    roots.append(paths.Path(os.path.expanduser("~")) / ".cache" / "huggingface")
    for root in roots:
        for pattern in ("models--facebook--sam3", "hub/models--facebook--sam3"):
            try:
                if (root / pattern).exists():
                    return True
            except OSError:
                continue
    return False


def _query_running_daemon(timeout: float = 3.0) -> Optional[Dict[str, Any]]:
    """``GET /status`` on the daemon named by ``runtime.json``, if it is alive."""
    import http.client  # noqa: PLC0415

    from .server import pid_alive  # noqa: PLC0415

    info = paths.read_json(paths.runtime_file())
    if not isinstance(info, dict):
        return None
    for key in ("port", "token", "pid"):
        if key not in info:
            return None
    if not pid_alive(int(info["pid"])):
        return {"reachable": False, "reason": "pid %s is not running" % info["pid"],
                "runtime": {k: v for k, v in info.items() if k != "token"}}
    host = str(info.get("host") or "127.0.0.1")
    # A wildcard bind is where the daemon listens, not an address to dial.
    host = {"0.0.0.0": "127.0.0.1", "::": "::1", "[::]": "::1"}.get(host, host)
    conn = http.client.HTTPConnection(host, int(info["port"]), timeout=timeout)
    try:
        conn.request("GET", "/status",
                     headers={"Authorization": "Bearer " + str(info["token"])})
        response = conn.getresponse()
        body = response.read()
        if response.status != 200:
            return {"reachable": False, "reason": "HTTP %d" % response.status}
        return {"reachable": True, "status": json.loads(body.decode("utf-8"))}
    except Exception as exc:  # noqa: BLE001
        return {"reachable": False, "reason": "%s: %s" % (type(exc).__name__, exc)}
    finally:
        conn.close()


def cmd_doctor(args: argparse.Namespace) -> int:
    """Environment report as JSON: device, dtype, versions, paths, daemon."""
    report: Dict[str, Any] = {
        "sam3d_version": _package_version(),
        "api_version": _api_version(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": sys.platform,
        "paths": paths.describe(),
        "weights_available": _weights_present(),
        "engine": _probe_engine(),
        "daemon": _query_running_daemon() if not args.no_daemon else None,
    }
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def cmd_download(args: argparse.Namespace) -> int:
    """Fetch the gated SAM 3 checkpoint with ``huggingface_hub``, lazily imported.

    The gated repo is the main onboarding failure (``DESIGN.md`` §8), so the
    403 path prints instructions rather than a traceback.
    """
    os.environ.setdefault("HF_HOME", str(paths.hf_home()))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    paths.ensure_dir(paths.hf_home())
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
    except ImportError:
        sys.stderr.write(
            "sam3gimpd: huggingface_hub is not installed in this environment.\n"
            + INSTALL_HINT +
            "  or just:  pip install huggingface_hub\n")
        return 2

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN")
    target = args.local_dir or None
    sys.stderr.write("sam3gimpd: downloading %s into %s ...\n"
                     % (args.repo_id, target or os.environ["HF_HOME"]))
    try:
        where = snapshot_download(repo_id=args.repo_id, token=token,
                                  local_dir=target, revision=args.revision)
    except Exception as exc:  # noqa: BLE001 -- the hub raises many types
        name = type(exc).__name__
        text = str(exc)
        gated = ("Gated" in name or "401" in text or "403" in text
                 or "awaiting a review" in text or "access to model" in text.lower())
        if gated:
            sys.stderr.write(
                "\nsam3gimpd: the SAM 3 weights are GATED.\n"
                "  1. Sign in and accept Meta's terms at\n"
                "       %s\n"
                "  2. Create a read token at https://huggingface.co/settings/tokens\n"
                "  3. Re-run:  sam3gimpd download --token hf_xxx\n"
                "  Already downloaded them by hand?  Point the daemon at them with\n"
                "       %s\n\n"
                "  (underlying error: %s: %s)\n"
                % (GATED_URL, paths.models_dir(), name, text))
            return 3
        sys.stderr.write("sam3gimpd: download failed: %s: %s\n" % (name, text))
        return 1
    sys.stdout.write(json.dumps({"repo_id": args.repo_id, "path": str(where),
                                 "hf_home": os.environ["HF_HOME"]}, indent=2) + "\n")
    return 0


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #
def _package_version() -> str:
    from . import __version__  # noqa: PLC0415

    return __version__


def _api_version() -> str:
    from . import __api_version__  # noqa: PLC0415

    return __api_version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sam3gimpd",
        description="Standalone SAM 3 segmentation daemon "
                    "(see plugin/sam3_gimp/_daemon/API.md).")
    parser.add_argument("--version", action="version",
                        version="sam3gimpd %s (API %s)" % (_package_version(), _api_version()))
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="run the HTTP daemon")
    serve.add_argument("--host", "--bind", dest="host", default="127.0.0.1",
                       help="bind address (default 127.0.0.1).  For a remote GPU keep "
                            "the default and use an SSH tunnel; 0.0.0.0 sends the token "
                            "and images over plain HTTP, so only on a trusted network")
    serve.add_argument("--port", type=int, default=0,
                       help="TCP port; 0 asks the OS and reports it in runtime.json")
    serve.add_argument("--stub", action="store_true",
                       help="fake engine: deterministic synthetic masks, no torch")
    serve.add_argument("--parent-pid", type=int, default=None,
                       help="exit when this pid disappears (the plug-in passes GIMP's)")
    serve.add_argument("--idle-ttl", "--idle-timeout", dest="idle_ttl", type=float,
                       # 30 minutes.  Every idle exit costs the next use a cold
                       # start -- a 3.6 GB model load plus CUDA init -- and at
                       # 10 minutes a real editing session paid that repeatedly.
                       default=1800.0,
                       help="exit after this many idle seconds; 0 disables")
    serve.add_argument("--cache-size", type=int, default=3,
                       help="LRU capacity of the embedding cache")
    serve.add_argument("--runtime-file", default=None,
                       help="override the path of runtime.json")
    serve.add_argument("--log-level", default="info",
                       choices=["debug", "info", "warning", "error", "critical"])
    serve.add_argument("--log-file", default=None, help="override the log file path")
    serve.add_argument("--no-stderr-log", action="store_true",
                       help="log only to the file (the detached daemon's default host)")
    serve.add_argument("--device", default="auto",
                       help="auto | cpu | cuda | cuda:0 | mps")
    serve.add_argument("--print-runtime", action="store_true",
                       help="print a line on exit (handy when run by hand)")
    serve.set_defaults(func=cmd_serve)

    doctor = sub.add_parser("doctor", help="print an environment report as JSON")
    doctor.add_argument("--no-daemon", action="store_true",
                        help="skip contacting a running daemon")
    doctor.set_defaults(func=cmd_doctor)

    download = sub.add_parser("download", help="fetch the gated SAM 3 checkpoint")
    download.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    download.add_argument("--revision", default=None)
    download.add_argument("--token", default=None,
                          help="HuggingFace read token (or set HF_TOKEN)")
    download.add_argument("--local-dir", default=None,
                          help="download into this directory instead of HF_HOME")
    download.set_defaults(func=cmd_download)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
