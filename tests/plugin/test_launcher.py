"""Tests for ``plugin/sam3_gimp/launcher.py`` -- find-or-spawn.

The centrepiece is :func:`test_spawn_then_full_round_trip`: it really spawns a
**detached** daemon process through ``launcher.find_or_spawn``, waits for the
real ``runtime.json`` handshake, and then drives that process with
``client.Sam3Client`` all the way to decoded soft masks.  Everything between the
two halves of the project -- the spawn flags, the handshake file, the bearer
token, the job queue, the binary frame -- is exercised in one test, without a
GPU, torch or GIMP.

Two daemons are used:

* a **script daemon** (:data:`FAKE_DAEMON_SOURCE`) that reuses the
  contract-conforming server from ``test_client.py`` in a separate process.  It
  is always available, so the spawn path is covered even before the ``sam3gimpd``
  package exists;
* the **real** ``sam3gimpd serve --stub``, when it is importable.

Windows is the main target, but these tests run on Linux and macOS too, so the
Windows spawn details (``pythonw.exe``, ``DETACHED_PROCESS |
CREATE_NEW_PROCESS_GROUP``) and pid semantics are asserted against a patched
``platform_key`` and captured or faked OS calls.  Those assertions verify
*what we ask the OS for*; only Windows can verify what it then does.

Every process a test signals -- directly, or through the launcher's
stuck-daemon handling -- is one the test started and still owns.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import client as C
import launcher as L


TESTS_PLUGIN_DIR = str(Path(__file__).resolve().parent)
PLUGIN_PKG_DIR = str(Path(__file__).resolve().parents[2] / "plugin" / "sam3_gimp")
DAEMON_DIR = str(Path(__file__).resolve().parents[2] / "plugin" / "sam3_gimp" / "_daemon")

# pytest imports test modules in importlib mode, so `tests/plugin` is not on
# sys.path and `import test_client` would fail.  Load the sibling module by
# path: its FakeDaemon is the contract-conforming server both suites use, and
# the stand-in daemon script below imports it the same way in its own process.
_spec = importlib.util.spec_from_file_location(
    "sam3_fake_daemon_server", str(Path(__file__).with_name("test_client.py"))
)
T = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = T
_spec.loader.exec_module(T)


# --------------------------------------------------------------------------- #
# a stand-in daemon process
# --------------------------------------------------------------------------- #
FAKE_DAEMON_SOURCE = '''\
"""A stand-in for `sam3gimpd serve`, used by the launcher tests.

It accepts the exact CLI of API.md section 13, reuses the contract-conforming
HTTP server from test_client.py, publishes runtime.json atomically after the
socket is listening, honours --parent-pid, and deletes runtime.json on a clean
shutdown.  With FAKE_DAEMON_LOCK it also keeps the real daemon's single-instance
rule (section 3.4): take the lock first, exit 0 quietly if it is taken, and only
then start up (FAKE_DAEMON_START_DELAY models the torch import that sits
between the two).  Other environment knobs simulate failure modes.
"""
import json, os, secrets, sys, tempfile, threading, time

sys.path.insert(0, os.environ["FAKE_DAEMON_PATH_TESTS"])
sys.path.insert(0, os.environ["FAKE_DAEMON_PATH_PLUGIN"])
import test_client  # noqa: E402  (the fake server lives with the client tests)
# The plug-in's own liveness check.  os.kill(pid, 0) is only a probe on POSIX:
# on Windows 0 is CTRL_C_EVENT, which a detached process cannot deliver, and
# CPython then falls back to TerminateProcess -- it ends the process.
import launcher  # noqa: E402


def parse_args(argv):
    opts = {"host": "127.0.0.1", "port": 0, "stub": False, "parent_pid": None,
            "idle_ttl": None, "cache_size": None, "runtime_file": None,
            "log_level": None, "device": None, "subcommand": None}
    i = 0
    while i < len(argv):
        a = argv[i]
        if not a.startswith("--"):
            opts["subcommand"] = a
        elif a == "--stub":
            opts["stub"] = True
        elif a == "--no-stderr-log":
            # a boolean flag on the real CLI; the launcher always passes it
            opts["no_stderr_log"] = True
        else:
            key = a[2:].replace("-", "_")
            i += 1
            opts[key] = argv[i]
        i += 1
    return opts


def main():
    opts = parse_args(sys.argv[1:])
    with open(os.environ["FAKE_DAEMON_ARGV_FILE"], "w") as fh:
        json.dump({"argv": sys.argv[1:], "opts": opts,
                   "env": {k: os.environ.get(k) for k in
                           ("SAM3_GIMP_HOME", "HF_HOME", "PYTHONUNBUFFERED",
                            "HF_HUB_DISABLE_PROGRESS_BARS")},
                   "cwd": os.getcwd(), "pid": os.getpid()}, fh)

    print("fake sam3gimpd starting: %s" % (sys.argv[1:],), flush=True)

    if os.environ.get("FAKE_DAEMON_DIE_AT_START"):
        print("simulated startup failure", flush=True)
        raise SystemExit(3)

    if os.environ.get("FAKE_DAEMON_LOCK"):
        import fcntl
        fd = os.open(os.path.join(os.environ["SAM3_GIMP_HOME"], "sam3gimpd.lock"),
                     os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("another instance holds the lock; exiting", flush=True)
            raise SystemExit(0)
        os.ftruncate(fd, 0)
        os.write(fd, ("%d\\n" % os.getpid()).encode("ascii"))
    time.sleep(float(os.environ.get("FAKE_DAEMON_START_DELAY", "0")))

    token = secrets.token_urlsafe(32)
    server = test_client.FakeDaemon(
        api_version=os.environ.get("FAKE_DAEMON_API_VERSION", test_client.C.API_VERSION),
        token=token,
        stage_delay=0.01,
        prove_identity=not os.environ.get("FAKE_DAEMON_NO_PROOF"),
    )

    if os.environ.get("FAKE_DAEMON_NEVER_PUBLISH"):
        # Listening but never publishing runtime.json: the launcher must time
        # out and surface the log tail.
        time.sleep(float(os.environ.get("FAKE_DAEMON_LINGER", "30")))
        raise SystemExit(0)

    runtime_path = opts["runtime_file"] or os.path.join(
        os.environ["SAM3_GIMP_HOME"], "runtime.json")
    info = {"port": server.port, "token": token, "pid": os.getpid(),
            "version": "0.0.0-fake", "started_at": time.time(),
            "api_version": server.api_version, "host": opts["host"],
            "log_path": os.path.join(os.environ["SAM3_GIMP_HOME"], "logs", "sam3gimpd.log")}
    os.makedirs(os.path.dirname(runtime_path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(runtime_path) or ".")
    with os.fdopen(fd, "w") as fh:
        json.dump(info, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, runtime_path)
    print("published %s port=%d" % (runtime_path, server.port), flush=True)

    parent = opts["parent_pid"]
    parent = int(parent) if parent else None
    deadline = time.time() + float(os.environ.get("FAKE_DAEMON_MAX_LIFETIME", "120"))
    try:
        while time.time() < deadline:
            if server.shutting_down:
                break
            if parent is not None and not launcher.pid_alive(parent):
                print("parent %d is gone, exiting" % parent, flush=True)
                break
            time.sleep(0.1)
    finally:
        try:
            os.remove(runtime_path)
        except OSError:
            pass
        server.close()
    raise SystemExit(0)


main()
'''


@pytest.fixture(autouse=True)
def _no_stray_daemons():
    """Guarantee no detached test daemon outlives the suite.

    The launcher deliberately never ``wait()``s on what it spawns -- the child
    must outlive the plug-in -- so the test harness has to clean up after
    itself.  ``launcher._SPAWNED`` holds every Popen we created.
    """
    # Identity, not an index: the launcher reaps exited children out of
    # ``_SPAWNED`` as it goes (that is what keeps ``pid_alive`` honest about a
    # daemon that shut down cleanly), so a saved *length* would slide.
    before = {id(proc) for proc in L._SPAWNED}
    try:
        yield
    finally:
        mine = [proc for proc in L._SPAWNED if id(proc) not in before]
        for proc in mine:
            try:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
        L._SPAWNED[:] = [proc for proc in L._SPAWNED if id(proc) in before]


@pytest.fixture
def fake_daemon_script(tmp_path, sam3_home, monkeypatch):
    """Write the stand-in daemon and point the launcher at it.

    ``sam3_home`` gives this test its own ``SAM3_GIMP_HOME``, so nothing here
    can touch a real installation.
    """
    script = tmp_path / "fake_sam3d.py"
    script.write_text(FAKE_DAEMON_SOURCE, encoding="utf-8")
    argv_file = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_DAEMON_PATH_TESTS", TESTS_PLUGIN_DIR)
    monkeypatch.setenv("FAKE_DAEMON_PATH_PLUGIN", PLUGIN_PKG_DIR)
    monkeypatch.setenv("FAKE_DAEMON_ARGV_FILE", str(argv_file))
    monkeypatch.setenv("FAKE_DAEMON_MAX_LIFETIME", "90")
    monkeypatch.delenv("SAM3D_COMMAND", raising=False)
    # daemon_environ() respects a user-set HF_HOME; clear any leaked from
    # another test so the spawn tests see the computed <base>/hf.
    monkeypatch.delenv("HF_HOME", raising=False)
    return SimpleNamespace(path=script, argv_file=argv_file)


def _spawn_argv(script):
    return [sys.executable, str(script.path)]


def _process_finished(pid: int) -> bool:
    """Has this process really ended?

    The daemon is spawned with ``start_new_session=True``, which detaches it
    from our terminal and process group but does **not** reparent it: under
    pytest it stays our child, so when it exits it lingers as a zombie that
    ``os.kill(pid, 0)`` still reports as alive.  In production the plug-in
    process exits moments after spawning, the daemon is reparented to init and
    reaped there -- so this Linux-only zombie check exists purely for the test
    harness, not for the product.
    """
    if not L.pid_alive(pid):
        return True
    try:
        with open("/proc/%d/stat" % pid, "r") as fh:
            return fh.read().rsplit(") ", 1)[1].split()[0] == "Z"
    except OSError:
        return True


def _reap(result):
    """Ask a spawned daemon to exit and wait for its runtime.json to vanish."""
    if result is None:
        return
    try:
        result.client.shutdown_daemon(grace_ms=0)
    except Exception:
        pass
    finally:
        result.close()
    deadline = time.time() + 10.0
    pid = None
    try:
        pid = int(result.info["pid"])
    except Exception:
        pass
    while pid and time.time() < deadline and L.pid_alive(pid):
        time.sleep(0.05)


# =========================================================================== #
# module hygiene
# =========================================================================== #
def test_launcher_imports_only_stdlib():
    src = Path(L.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            top_level.add(node.module.split(".")[0])
    for name in top_level:
        assert name in sys.stdlib_module_names, "%s is not in the standard library" % name


def test_api_versions_agree():
    """The launcher's compatibility check must use the client's API version."""
    assert L.API_VERSION == C.API_VERSION


# =========================================================================== #
# platform layout (API.md section 3.1)
# =========================================================================== #
def test_sam3_gimp_home_overrides_everything(sam3_home):
    assert L.base_dir() == str(sam3_home)
    assert L.runtime_file() == str(sam3_home / "runtime.json")
    assert L.lock_file() == str(sam3_home / "sam3gimpd.lock")
    assert L.server_log() == str(sam3_home / "logs" / "sam3gimpd.log")
    assert L.crash_log() == str(sam3_home / "logs" / "crash.log")
    assert L.hf_home() == str(sam3_home / "hf")


def test_runtime_file_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv(L.ENV_RUNTIME_FILE, str(tmp_path / "elsewhere.json"))
    assert L.runtime_file() == str(tmp_path / "elsewhere.json")


def test_windows_layout(monkeypatch, tmp_path):
    monkeypatch.delenv(L.ENV_HOME, raising=False)
    monkeypatch.delenv(L.ENV_RUNTIME_FILE, raising=False)
    monkeypatch.setattr(L, "platform_key", lambda: "windows")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    assert L.base_dir() == os.path.join(str(tmp_path / "Local"), "sam3-gimp")
    assert L.venv_bin_dir().endswith(os.path.join("venv", "Scripts"))
    assert os.path.basename(L.venv_pythonw()) == "pythonw.exe"
    assert os.path.basename(L.venv_python()) == "python.exe"
    assert os.path.basename(L.sam3d_executable()) == "sam3gimpd.exe"


def test_windows_layout_falls_back_to_appdata(monkeypatch, tmp_path):
    monkeypatch.delenv(L.ENV_HOME, raising=False)
    monkeypatch.setattr(L, "platform_key", lambda: "windows")
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert L.base_dir() == os.path.join(str(tmp_path / "Roaming"), "sam3-gimp")


def test_macos_and_linux_layouts(monkeypatch, tmp_path):
    monkeypatch.delenv(L.ENV_HOME, raising=False)
    monkeypatch.setattr(L, "platform_key", lambda: "macos")
    assert L.base_dir().endswith(os.path.join("Library", "Application Support", "sam3-gimp"))
    monkeypatch.setattr(L, "platform_key", lambda: "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert L.base_dir() == os.path.join(str(tmp_path / "xdg"), "sam3-gimp")
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert L.base_dir().endswith(os.path.join(".local", "share", "sam3-gimp"))


def test_ensure_layout_creates_the_directories(sam3_home):
    layout = L.ensure_layout()
    for path in layout.values():
        assert os.path.isdir(path)


def test_daemon_environ(sam3_home, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    env = L.daemon_environ()
    assert env["HF_HOME"] == L.hf_home()
    assert env[L.ENV_HOME] == str(sam3_home)
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"


def test_daemon_environ_respects_a_user_hf_home(sam3_home, monkeypatch):
    monkeypatch.setenv("HF_HOME", "/somewhere/else")
    assert L.daemon_environ()["HF_HOME"] == "/somewhere/else"


def test_describe_is_flat_strings(sam3_home):
    d = L.describe()
    assert set(d) >= {"base", "runtime_file", "server_log", "venv_pythonw"}
    assert all(isinstance(v, str) for v in d.values())


# =========================================================================== #
# pid liveness (API.md section 3.3 step 2)
# =========================================================================== #
def test_pid_alive_for_this_process():
    assert L.pid_alive(os.getpid()) is True


def test_pid_alive_for_a_dead_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    # A reaped pid is dead; the value is not recycled that fast in practice.
    assert L.pid_alive(proc.pid) is False


@pytest.mark.parametrize("bad", [None, 0, -1, "x", "", 1.5e300])
def test_pid_alive_rejects_nonsense(bad):
    assert L.pid_alive(bad) is False


@pytest.mark.skipif(os.name == "nt", reason="POSIX EPERM semantics")
def test_pid_alive_treats_eperm_as_alive(monkeypatch):
    def _kill(pid, sig):
        raise PermissionError("not yours")

    monkeypatch.setattr(os, "kill", _kill)
    assert L.pid_alive(4711) is True


# =========================================================================== #
# runtime.json handling
# =========================================================================== #
def test_read_runtime_info_and_delete(sam3_home, auth_token):
    path = Path(L.runtime_file())
    path.write_text(json.dumps({
        "port": 41573, "token": auth_token, "pid": os.getpid(),
        "version": "0.1.0", "started_at": time.time(),
    }))
    info = L.read_runtime_info()
    assert info["port"] == 41573
    assert L.delete_runtime_file() is True
    assert L.read_runtime_info() is None
    assert L.delete_runtime_file() is True  # idempotent


def test_tail_log(sam3_home):
    L.ensure_layout()
    Path(L.server_log()).write_text("\n".join("line %d" % i for i in range(200)))
    tail = L.tail_log(lines=5)
    assert tail.splitlines() == ["line %d" % i for i in range(195, 200)]
    assert L.tail_log(str(sam3_home / "nope.log")) == ""


# =========================================================================== #
# building the command (API.md section 13)
# =========================================================================== #
def test_build_command_flags():
    argv = L.build_command(
        command=["/usr/bin/python3", "-m", "sam3gimpd"],
        stub=True, parent_pid=4711, idle_ttl=600, cache_size=3,
        device="cuda", log_level="debug", runtime_path="/tmp/rt.json",
    )
    assert argv[:4] == ["/usr/bin/python3", "-m", "sam3gimpd", "serve"]
    assert "--stub" in argv
    assert argv[argv.index("--parent-pid") + 1] == "4711"
    assert argv[argv.index("--idle-ttl") + 1] == "600"
    assert argv[argv.index("--cache-size") + 1] == "3"
    assert argv[argv.index("--device") + 1] == "cuda"
    assert argv[argv.index("--log-level") + 1] == "debug"
    assert argv[argv.index("--runtime-file") + 1] == "/tmp/rt.json"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "0"


def test_build_command_omits_absent_options():
    argv = L.build_command(command=["python", "-m", "sam3gimpd"])
    for flag in ("--stub", "--parent-pid", "--idle-ttl", "--cache-size",
                 "--device", "--log-level", "--runtime-file"):
        assert flag not in argv


def test_build_command_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv(L.ENV_COMMAND, "%s -m sam3gimpd" % sys.executable)
    argv = L.build_command()
    assert argv[:3] == [sys.executable, "-m", "sam3gimpd"]


def test_build_command_prefers_the_venv_console_script(sam3_home, monkeypatch):
    monkeypatch.delenv(L.ENV_COMMAND, raising=False)
    monkeypatch.setattr(L, "platform_key", lambda: "linux")
    binp = Path(L.venv_bin_dir())
    binp.mkdir(parents=True)
    (binp / "sam3gimpd").write_text("#!/bin/sh\n")
    assert L.build_command()[0] == str(binp / "sam3gimpd")


def test_build_command_on_windows_uses_pythonw_never_python(sam3_home, monkeypatch):
    """P0 detail: ``python.exe`` flashes a console window on every invocation."""
    monkeypatch.delenv(L.ENV_COMMAND, raising=False)
    monkeypatch.setattr(L, "platform_key", lambda: "windows")
    scripts = Path(L.venv_bin_dir())
    scripts.mkdir(parents=True)
    (scripts / "sam3gimpd.exe").write_bytes(b"MZ")
    (scripts / "python.exe").write_bytes(b"MZ")
    (scripts / "pythonw.exe").write_bytes(b"MZ")
    argv = L.build_command(stub=True)
    assert argv[0] == str(scripts / "pythonw.exe")
    assert argv[1:3] == ["-m", "sam3gimpd"]
    assert "python.exe" not in argv[0]


def test_build_command_without_an_interpreter_raises(sam3_home, monkeypatch):
    monkeypatch.delenv(L.ENV_COMMAND, raising=False)
    monkeypatch.setattr(L, "_python_for_spawn", lambda: None)
    monkeypatch.setattr(L, "platform_key", lambda: "windows")
    with pytest.raises(L.SpawnError):
        L.build_command()


# =========================================================================== #
# detached spawn (DESIGN.md section 4 / API.md section 13)
# =========================================================================== #
class _FakePopen:
    captured = None

    def __init__(self, argv, **kwargs):
        _FakePopen.captured = (argv, kwargs)
        self.pid = 31337


def test_spawn_uses_posix_session_detachment(sam3_home, monkeypatch):
    monkeypatch.setattr(L, "platform_key", lambda: "linux")
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    pid = L.spawn_daemon(["/bin/true", "serve"])
    argv, kwargs = _FakePopen.captured
    assert pid == 31337
    assert kwargs["start_new_session"] is True
    assert "creationflags" not in kwargs
    assert kwargs["close_fds"] is True
    assert kwargs["stderr"] is subprocess.STDOUT
    assert kwargs["stdout"].name == L.server_log()
    assert kwargs["cwd"] == L.base_dir()
    assert kwargs["env"][L.ENV_HOME] == L.base_dir()


def test_spawn_uses_windows_detached_flags(sam3_home, monkeypatch):
    """``DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP``: no console, and the child
    survives the short-lived plug-in process (DESIGN.md Constraint C)."""
    monkeypatch.setattr(L, "platform_key", lambda: "windows")
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    L.spawn_daemon(["pythonw.exe", "-m", "sam3gimpd", "serve"])
    _argv, kwargs = _FakePopen.captured
    assert kwargs["creationflags"] == L.DETACHED_PROCESS | L.CREATE_NEW_PROCESS_GROUP
    assert kwargs["creationflags"] == 0x00000008 | 0x00000200
    assert "start_new_session" not in kwargs
    assert kwargs["close_fds"] is True
    # stdio must go to a log file: a detached process has no console to inherit.
    assert kwargs["stdout"].name == L.server_log()
    assert kwargs["stderr"] is subprocess.STDOUT
    assert kwargs["stdin"].name == os.devnull


def test_spawn_closes_the_parent_handles(sam3_home, monkeypatch):
    """The plug-in process exits immediately; it must not hold the log open."""
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    L.spawn_daemon(["/bin/true"])
    _argv, kwargs = _FakePopen.captured
    assert kwargs["stdout"].closed
    assert kwargs["stdin"].closed


def test_spawn_records_the_command_in_the_log(sam3_home, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    L.spawn_daemon(["/bin/true", "serve", "--stub"])
    assert "serve --stub" in Path(L.server_log()).read_text(encoding="utf-8")


def test_spawn_failure_is_a_spawn_error(sam3_home):
    with pytest.raises(L.SpawnError) as exc:
        L.spawn_daemon([str(Path(L.base_dir()) / "no-such-binary"), "serve"])
    assert exc.value.command[0].endswith("no-such-binary")


# =========================================================================== #
# find_or_spawn: the handshake (API.md section 3.3)
# =========================================================================== #
def test_find_without_spawn_raises_when_nothing_is_running(sam3_home):
    with pytest.raises(C.DaemonUnavailable):
        L.find_or_spawn(spawn=False)


def test_find_accepts_a_running_daemon_without_spawning(sam3_home, monkeypatch):
    """Steps 1-5: a healthy runtime.json short-circuits the whole spawn path."""
    d = T.FakeDaemon()
    try:
        Path(L.runtime_file()).write_text(json.dumps(d.runtime_info()))
        monkeypatch.setattr(
            L, "spawn_daemon",
            lambda *a, **k: pytest.fail("must not spawn when a daemon is healthy"),
        )
        result = L.find_or_spawn()
        assert result.spawned is False
        assert result.port == d.port
        assert result.hello.engine_mode == "stub"
        assert result.client.hello().api_version == C.API_VERSION
        result.close()
    finally:
        d.close()


def test_stale_runtime_file_with_a_dead_pid_is_deleted(sam3_home, monkeypatch, auth_token):
    Path(L.runtime_file()).write_text(json.dumps({
        "port": 1, "token": auth_token, "pid": 999999999,
        "version": "0.1.0", "started_at": time.time(),
    }))
    monkeypatch.setattr(L, "spawn_daemon", lambda *a, **k: 1)
    monkeypatch.setattr(L, "_wait_for_runtime", lambda *a, **k: False)
    with pytest.raises(L.DaemonStartTimeout):
        L.find_or_spawn(command=["/bin/true"], timeout=0.1)
    assert not os.path.exists(L.runtime_file()), "a stale handshake file must be removed"


def test_unreachable_port_is_treated_as_stale(sam3_home, monkeypatch, auth_token, free_port):
    """Step 3: the pid may be alive (ours is) yet nothing answers /hello."""
    Path(L.runtime_file()).write_text(json.dumps({
        "port": free_port, "token": auth_token, "pid": os.getpid(),
        "version": "0.1.0", "started_at": time.time(),
    }))
    calls = []
    monkeypatch.setattr(L, "spawn_daemon", lambda argv, **k: calls.append(argv) or 1)
    monkeypatch.setattr(L, "_wait_for_runtime", lambda *a, **k: False)
    with pytest.raises(L.DaemonStartTimeout):
        L.find_or_spawn(command=["/bin/true"], timeout=0.1)
    assert calls, "an unreachable daemon must trigger a respawn"
    assert not os.path.exists(L.runtime_file())


def test_incompatible_major_is_shut_down_and_replaced(sam3_home, monkeypatch):
    """Step 4: POST /shutdown, delete the file, respawn."""
    d = T.FakeDaemon(api_version="2.0")
    try:
        Path(L.runtime_file()).write_text(json.dumps(d.runtime_info()))
        monkeypatch.setattr(L, "spawn_daemon", lambda *a, **k: 1)
        monkeypatch.setattr(L, "_wait_for_runtime", lambda *a, **k: False)
        with pytest.raises(L.DaemonStartTimeout):
            L.find_or_spawn(command=["/bin/true"], timeout=0.1)
        assert d.shutting_down is True, "the old daemon must be asked to exit"
        assert not os.path.exists(L.runtime_file())
    finally:
        d.close()


def test_incompatible_major_without_spawn_raises(sam3_home):
    d = T.FakeDaemon(api_version="2.0")
    try:
        Path(L.runtime_file()).write_text(json.dumps(d.runtime_info()))
        with pytest.raises(L.IncompatibleDaemon):
            L.find_or_spawn(spawn=False)
    finally:
        d.close()


def test_spawn_attempts_are_bounded(sam3_home, monkeypatch):
    """Step 9: at most ``max_attempts`` spawns per plug-in invocation.

    Here the daemon *publishes* a file each time but it is never usable, so the
    loop returns to step 3 and would otherwise spin forever.
    """
    spawns = []
    monkeypatch.setattr(L, "spawn_daemon", lambda argv, **k: spawns.append(argv) or 1)
    monkeypatch.setattr(L, "_wait_for_runtime", lambda *a, **k: True)
    with pytest.raises(L.DaemonStartTimeout):
        L.find_or_spawn(command=["/bin/true"], timeout=0.05, max_attempts=2)
    assert len(spawns) == 2


def test_publish_timeout_fails_fast_with_the_log(sam3_home, monkeypatch):
    """A daemon that never publishes is not retried: the log is more useful."""
    spawns = []
    monkeypatch.setattr(L, "spawn_daemon", lambda argv, **k: spawns.append(argv) or 1)
    monkeypatch.setattr(L, "_wait_for_runtime", lambda *a, **k: False)
    with pytest.raises(L.DaemonStartTimeout) as exc:
        L.find_or_spawn(command=["/bin/true"], timeout=0.05, max_attempts=2)
    assert len(spawns) == 1
    assert "did not publish" in str(exc.value)


def test_start_timeout_surfaces_the_log_tail(sam3_home, monkeypatch, fake_daemon_script):
    """§3.3 step 8: on timeout, show the user the last lines of sam3gimpd.log."""
    monkeypatch.setenv("FAKE_DAEMON_NEVER_PUBLISH", "1")
    monkeypatch.setenv("FAKE_DAEMON_LINGER", "3")
    with pytest.raises(L.DaemonStartTimeout) as exc:
        L.find_or_spawn(
            command=_spawn_argv(fake_daemon_script),
            timeout=1.5, max_attempts=1, stub=True,
        )
    assert "fake sam3gimpd starting" in exc.value.log_tail
    assert "--stub" in " ".join(exc.value.command)


def test_start_failure_surfaces_the_traceback(sam3_home, monkeypatch, fake_daemon_script):
    monkeypatch.setenv("FAKE_DAEMON_DIE_AT_START", "1")
    with pytest.raises(L.DaemonStartTimeout) as exc:
        L.find_or_spawn(
            command=_spawn_argv(fake_daemon_script), timeout=1.5, max_attempts=1
        )
    assert "simulated startup failure" in exc.value.log_tail


# =========================================================================== #
# the integration test: spawn a detached daemon and drive it
# =========================================================================== #
@pytest.mark.needs_daemon
@pytest.mark.slow
def test_spawn_then_full_round_trip(sam3_home, fake_daemon_script, rgb_image):
    """Spawn detached -> handshake -> upload -> PCS prompt -> decoded soft masks.

    This is the whole plug-in/daemon contract in one test, with no GPU, no
    torch and no GIMP.
    """
    result = None
    try:
        result = L.find_or_spawn(
            command=_spawn_argv(fake_daemon_script),
            stub=True,
            parent_pid=os.getpid(),
            idle_ttl=600,
            cache_size=3,
            timeout=25.0,
        )
        assert result.spawned is True
        assert result.hello.api_version == C.API_VERSION
        assert result.hello.engine_mode == "stub"
        assert L.pid_alive(result.pid)

        # The daemon really received the documented CLI and environment.
        spawned = json.loads(Path(fake_daemon_script.argv_file).read_text(encoding="utf-8"))
        assert spawned["opts"]["subcommand"] == "serve"
        assert spawned["opts"]["stub"] is True
        assert spawned["opts"]["parent_pid"] == str(os.getpid())
        assert spawned["opts"]["port"] == "0"
        assert spawned["env"]["SAM3_GIMP_HOME"] == str(sam3_home)
        assert spawned["env"]["HF_HOME"] == L.hf_home()
        assert spawned["env"]["PYTHONUNBUFFERED"] == "1"

        # It is a different process, and it is not our child's child: it was
        # detached, so it has its own session/process group.
        assert spawned["pid"] != os.getpid()

        client = result.client
        width, height, pixels = rgb_image
        accepted = client.upload_image(
            pixels, width, height, source_width=3000, source_height=2000
        )
        assert client.wait_for_image(accepted, timeout=30.0).state == C.JobState.DONE

        stages = []
        masks = client.run_text(
            accepted.image_id, "yellow school bus", timeout=30.0,
            on_progress=lambda s: stages.append(s.stage),
        )
        assert masks is not None
        assert len(masks) >= 1
        assert stages
        for inst in masks:
            assert len(inst.mask) == inst.mask_width * inst.mask_height
            assert inst.is_soft()
            placement = masks.place(inst, 3000, 2000)
            assert placement.width >= 1 and placement.height >= 1

        # A second find_or_spawn finds the SAME daemon instead of starting one.
        again = L.find_or_spawn(command=_spawn_argv(fake_daemon_script))
        try:
            assert again.spawned is False
            assert again.port == result.port
        finally:
            again.close()

        assert client.delete_image(accepted.image_id)["deleted"] is True
    finally:
        _reap(result)
    assert not os.path.exists(L.runtime_file()), "runtime.json is deleted on clean shutdown"


@pytest.mark.needs_daemon
@pytest.mark.slow
@pytest.mark.skipif(os.name == "nt", reason="uses POSIX signals to kill the fake parent")
def test_daemon_exits_when_its_parent_pid_disappears(sam3_home, fake_daemon_script):
    """``--parent-pid`` carries **GIMP's** pid: plug-in processes are short-lived
    (DESIGN.md Constraint C), so the daemon watches GIMP, not its spawner."""
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    result = None
    try:
        result = L.find_or_spawn(
            command=_spawn_argv(fake_daemon_script),
            stub=True, parent_pid=parent.pid, timeout=25.0,
        )
        daemon_pid = result.pid
        assert L.pid_alive(daemon_pid)
        parent.send_signal(signal.SIGKILL)
        parent.wait(10)

        deadline = time.time() + 20.0
        while time.time() < deadline and not _process_finished(daemon_pid):
            time.sleep(0.1)
        assert _process_finished(daemon_pid), "the daemon must exit when GIMP does"
        assert not os.path.exists(L.runtime_file()), \
            "runtime.json must be removed when the daemon exits"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(5)
        if result is not None:
            result.close()


@pytest.mark.needs_daemon
@pytest.mark.slow
def test_spawn_recovers_from_a_stale_runtime_file(sam3_home, fake_daemon_script, auth_token):
    """The normal post-crash state: a runtime.json whose pid is long gone."""
    Path(L.runtime_file()).write_text(json.dumps({
        "port": 65000, "token": auth_token, "pid": 999999999,
        "version": "0.0.1", "started_at": time.time() - 10_000,
    }))
    result = None
    try:
        result = L.find_or_spawn(
            command=_spawn_argv(fake_daemon_script), stub=True, timeout=25.0
        )
        assert result.spawned is True
        assert result.port != 65000
        assert result.client.hello().engine_mode == "stub"
        assert any("stale" in n for n in result.notes)
    finally:
        _reap(result)


@pytest.mark.needs_daemon
@pytest.mark.slow
def test_on_event_reports_progress(sam3_home, fake_daemon_script):
    events = []
    result = None
    try:
        result = L.find_or_spawn(
            command=_spawn_argv(fake_daemon_script), stub=True, timeout=25.0,
            on_event=lambda kind, msg: events.append(kind),
        )
        assert "spawn" in events and "accepted" in events
    finally:
        _reap(result)


# =========================================================================== #
# the real sam3gimpd daemon, when it is installed
# =========================================================================== #
def _real_sam3d_available() -> bool:
    """True when ``python -m sam3gimpd serve --stub`` can actually run here;
    the tests that need the real daemon skip otherwise."""
    if importlib.util.find_spec("sam3gimpd") is None:
        return False
    # The probe runs in a *subprocess*, which does not inherit pytest's
    # ``pythonpath`` ini setting -- only this process's sys.path was patched.
    # Hand it the daemon's directory explicitly or the probe reports "no CLI"
    # on a repo that has a perfectly good one.
    env = dict(os.environ)
    env["PYTHONPATH"] = DAEMON_DIR + os.pathsep + env.get("PYTHONPATH", "")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "sam3gimpd", "serve", "--help"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60, env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and b"--stub" in proc.stdout


@pytest.mark.needs_daemon
@pytest.mark.slow
def test_real_sam3d_stub_round_trip(sam3_home, rgb_image, monkeypatch):
    """The most valuable test in the repo: the *actual* daemon, spawned by the
    *actual* launcher, driven by the *actual* client -- with no torch.  It
    also proves the real daemon answers the /hello identity check."""
    if not _real_sam3d_available():
        pytest.skip("the sam3gimpd package (or its CLI) is not available yet")
    monkeypatch.setenv("PYTHONPATH", DAEMON_DIR + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.delenv(L.ENV_COMMAND, raising=False)

    result = None
    try:
        result = L.find_or_spawn(
            command=[sys.executable, "-m", "sam3gimpd"],
            stub=True,
            parent_pid=os.getpid(),
            idle_ttl=120,
            timeout=60.0,
        )
        assert result.hello.engine_mode == "stub"
        assert result.hello.torch_available is False
        assert "pcs" in result.hello.capabilities

        client = result.client
        width, height, pixels = rgb_image
        accepted = client.upload_image(
            pixels, width, height, source_width=3000, source_height=2000
        )
        client.wait_for_image(accepted, timeout=60.0)

        masks = client.run_text(accepted.image_id, "yellow school bus", timeout=60.0)
        assert masks is not None and len(masks) >= 1
        for inst in masks:
            assert inst.mask_width == inst.bbox.width
            assert inst.mask_height == inst.bbox.height
            assert len(inst.mask) == inst.mask_width * inst.mask_height
            assert inst.is_soft(), "the stub must return genuine soft gradients"

        # Determinism (API.md section 14): the same prompt, the same bytes.
        again = client.run_text(accepted.image_id, "yellow school bus", timeout=60.0)
        assert [i.mask for i in again] == [i.mask for i in masks]

        points = client.run_points(accepted.image_id, [(10.0, 10.0, 1)], timeout=60.0)
        assert points is not None and len(points) >= 1
    finally:
        _reap(result)


# --------------------------------------------------------------------------- #
# SAM3D_COMMAND parsing -- regression for a Windows-only bug
# --------------------------------------------------------------------------- #
class TestSplitCommand:
    """``shlex`` alone gets Windows wrong in both directions.

    POSIX mode eats the backslashes (``C:\\Python\\python.exe`` becomes
    ``C:Pythonpython.exe``).  Non-POSIX mode keeps them but also keeps the
    *quotes* inside the token, so the quoting a user must write for a path under
    ``C:\\Program Files\\`` yields an argv[0] with literal ``"`` characters and
    CreateProcess cannot find the file.  Since ``C:\\Program Files\\`` and
    usernames with spaces are both entirely ordinary, this made the documented
    "reuse my existing torch environment" route unusable.
    """

    @staticmethod
    def _nt(monkeypatch, text):
        monkeypatch.setattr(L.os, "name", "nt", raising=False)
        return L.split_command(text)

    def test_windows_path_keeps_its_backslashes(self, monkeypatch):
        got = self._nt(monkeypatch, r"C:\Users\someone\venv\Scripts\python.exe -m sam3gimpd")
        assert got == [r"C:\Users\someone\venv\Scripts\python.exe", "-m", "sam3gimpd"]

    def test_quoted_program_files_path_loses_the_quotes(self, monkeypatch):
        got = self._nt(monkeypatch,
                       r'"C:\Program Files\Python312\python.exe" -m sam3gimpd')
        assert got == [r"C:\Program Files\Python312\python.exe", "-m", "sam3gimpd"]
        assert '"' not in got[0]

    def test_username_with_a_space(self, monkeypatch):
        got = self._nt(monkeypatch,
                       r'"C:\Users\Jane Doe\venv\Scripts\pythonw.exe" -m sam3gimpd')
        assert got[0] == r"C:\Users\Jane Doe\venv\Scripts\pythonw.exe"
        assert got[1:] == ["-m", "sam3gimpd"]

    def test_single_quotes_are_stripped_too(self, monkeypatch):
        got = self._nt(monkeypatch, r"'C:\Program Files\py\python.exe' -m sam3gimpd")
        assert got[0] == r"C:\Program Files\py\python.exe"

    def test_posix_is_unchanged(self, monkeypatch):
        monkeypatch.setattr(L.os, "name", "posix", raising=False)
        assert L.split_command("/usr/bin/python3 -m sam3gimpd") == [
            "/usr/bin/python3", "-m", "sam3gimpd"]

    def test_posix_still_honours_escaping(self, monkeypatch):
        monkeypatch.setattr(L.os, "name", "posix", raising=False)
        assert L.split_command(r"'/opt/my python/bin/python3' -m sam3gimpd") == [
            "/opt/my python/bin/python3", "-m", "sam3gimpd"]

    def test_build_command_uses_it(self, monkeypatch):
        monkeypatch.setattr(L.os, "name", "nt", raising=False)
        monkeypatch.setenv(L.ENV_COMMAND,
                           r'"C:\Program Files\Python312\python.exe" -m sam3gimpd')
        argv = L.build_command()
        assert argv[0] == r"C:\Program Files\Python312\python.exe"
        assert "serve" in argv


# --------------------------------------------------------------------------- #
# "use the Python I already have" -- the Setup dialog's persisted choice
# --------------------------------------------------------------------------- #
class TestConfiguredPython:
    """Pointing the plug-in at an existing interpreter is a UI choice.

    It is persisted in ``settings.json`` and sits in ``build_command``'s
    resolution chain below ``$SAM3D_COMMAND`` (so a developer override still
    wins) and above the managed venv.
    """

    def test_absent_by_default(self, sam3_home):
        assert L.configured_python() is None

    def test_round_trips(self, sam3_home, tmp_path):
        exe = tmp_path / "python"
        exe.write_text("#!/bin/sh\n")
        L.set_configured_python(str(exe))
        assert L.configured_python() == str(exe)
        L.set_configured_python(None)
        assert L.configured_python() is None

    def test_stored_as_an_absolute_path(self, sam3_home, tmp_path, monkeypatch):
        exe = tmp_path / "python"
        exe.write_text("#!/bin/sh\n")
        monkeypatch.chdir(tmp_path)
        L.set_configured_python("python")
        assert os.path.isabs(L.read_settings()["python"])

    def test_a_deleted_interpreter_is_ignored(self, sam3_home, tmp_path):
        """Uninstalling the chosen environment must degrade, not break.

        Returning the stale path would make every spawn fail with 'no such
        file'; returning None lets the chain fall through to the managed venv.
        """
        exe = tmp_path / "python"
        exe.write_text("#!/bin/sh\n")
        L.set_configured_python(str(exe))
        exe.unlink()
        assert L.configured_python() is None

    def test_corrupt_settings_file_is_survivable(self, sam3_home):
        os.makedirs(os.path.dirname(L.settings_file()), exist_ok=True)
        with open(L.settings_file(), "w", encoding="utf-8") as fh:
            fh.write("{not json at all")
        assert L.read_settings() == {}
        assert L.configured_python() is None

    def test_build_command_uses_it(self, sam3_home, tmp_path):
        exe = tmp_path / "python"
        exe.write_text("#!/bin/sh\n")
        L.set_configured_python(str(exe))
        argv = L.build_command()
        assert argv[0] == str(exe)
        assert argv[1:3] == ["-m", "sam3gimpd"]
        assert "serve" in argv

    def test_env_command_still_wins(self, sam3_home, tmp_path, monkeypatch):
        exe = tmp_path / "python"
        exe.write_text("#!/bin/sh\n")
        L.set_configured_python(str(exe))
        monkeypatch.setenv(L.ENV_COMMAND, "/opt/other/python -m sam3gimpd")
        assert L.build_command()[0] == "/opt/other/python"

    def test_settings_are_written_atomically(self, sam3_home, tmp_path):
        exe = tmp_path / "python"
        exe.write_text("#!/bin/sh\n")
        L.set_configured_python(str(exe))
        # no temp file left behind next to the settings
        leftovers = [n for n in os.listdir(os.path.dirname(L.settings_file())) if n.endswith(".tmp")]
        assert leftovers == []

    def test_unrelated_settings_survive_the_write(self, sam3_home, tmp_path):
        L.write_settings({"something_else": 42})
        exe = tmp_path / "python"
        exe.write_text("#!/bin/sh\n")
        L.set_configured_python(str(exe))
        assert L.read_settings()["something_else"] == 42


class TestWindowlessVariant:
    """On Windows the daemon must be spawned with pythonw.exe; python.exe
    flashes a console window on every prompt."""

    def test_posix_is_a_passthrough(self, monkeypatch):
        monkeypatch.setattr(L.os, "name", "posix", raising=False)
        monkeypatch.setattr(L.sys, "platform", "linux", raising=False)
        assert L.windowless_variant("/usr/bin/python3") == "/usr/bin/python3"

    def test_windows_prefers_pythonw_when_present(self, monkeypatch, tmp_path):
        monkeypatch.setattr(L.os, "name", "nt", raising=False)
        (tmp_path / "python.exe").write_text("")
        (tmp_path / "pythonw.exe").write_text("")
        got = L.windowless_variant(str(tmp_path / "python.exe"))
        assert got.endswith("pythonw.exe")

    def test_windows_falls_back_when_pythonw_is_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(L.os, "name", "nt", raising=False)
        (tmp_path / "python.exe").write_text("")
        got = L.windowless_variant(str(tmp_path / "python.exe"))
        assert got.endswith("python.exe")


class TestInterpreterFallback:
    """``sys.executable`` inside GIMP is GIMP's own embedded Python.

    Spawning it runs ``...WindowsApps\\GIMP...\\pythonw.exe -m sam3gimpd``, which exits
    instantly with "No module named sam3gimpd"; the user then reads a confusing
    error about an interpreter they never chose, after a 60 s wait.  The guard
    asks the only question that matters -- is ``sam3gimpd`` importable here -- rather
    than trying to infer "am I inside GIMP", which misfires on the fake-GIMP
    stubs and on the GTK canvas harness.
    """

    def test_sam3d_is_importable_under_pytest(self):
        assert L.this_interpreter_can_run_sam3d() is True

    def test_outside_gimp_still_falls_back_to_this_interpreter(self, sam3_home):
        """The harnesses and the test-suite depend on this fallback."""
        assert L._python_for_spawn() is not None

    def test_refuses_when_sam3d_is_not_importable(self, sam3_home, monkeypatch):
        monkeypatch.setattr(L, "this_interpreter_can_run_sam3d", lambda: False)
        assert L._python_for_spawn() is None

    def test_the_error_names_the_real_problem(self, sam3_home, monkeypatch):
        monkeypatch.setattr(L, "this_interpreter_can_run_sam3d", lambda: False)
        with pytest.raises(L.SpawnError) as excinfo:
            L.build_command()
        text = str(excinfo.value)
        assert "has not been installed yet" in text
        assert "Use an existing Python environment" in text

    def test_a_configured_interpreter_still_wins(self, sam3_home, tmp_path, monkeypatch):
        """Having chosen an environment must survive the new guard."""
        monkeypatch.setattr(L, "this_interpreter_can_run_sam3d", lambda: False)
        exe = tmp_path / "python"
        exe.write_text("#!/bin/sh\n")
        L.set_configured_python(str(exe))
        assert L.build_command()[0] == str(exe)

    def test_the_guard_is_not_fooled_by_a_fake_gi(self, sam3_home, monkeypatch):
        """Regression: the fake-GIMP stubs and the GTK harness both import gi."""
        monkeypatch.setitem(sys.modules, "gi.repository.Gimp", object())
        assert L._python_for_spawn() is not None


class TestEarlyChildExit:
    """A daemon that dies on startup never publishes runtime.json, so waiting
    out the whole timeout tells the user nothing and costs them a minute."""

    def test_a_dead_child_fails_fast_and_says_why(self, sam3_home):
        start = time.time()
        with pytest.raises(L.DaemonStartTimeout) as excinfo:
            L.find_or_spawn(
                command=[sys.executable, "-c", "import sys; sys.exit(3)"],
                timeout=30.0,
            )
        elapsed = time.time() - start
        assert elapsed < 10.0, "took %.1fs; should not wait out the timeout" % elapsed
        assert "exited immediately" in str(excinfo.value)
        assert "status 3" in str(excinfo.value)

    def test_the_missing_module_case_the_user_actually_hit(self, sam3_home):
        with pytest.raises(L.DaemonStartTimeout) as excinfo:
            L.find_or_spawn(command=[sys.executable, "-m", "sam3d_not_installed"],
                            timeout=30.0)
        assert "exited immediately" in str(excinfo.value)



class TestZombieLockHolder:
    """A daemon that holds the lock but no longer serves is ended and replaced --
    and nothing else ever is.  Every pid signalled here is a child this test
    started and still owns."""

    def test_lock_holder_pid_is_read_from_the_lockfile(self, sam3_home):
        os.makedirs(os.path.dirname(L.lock_file()), exist_ok=True)
        with open(L.lock_file(), "w") as fh:
            fh.write("4242\n")
        assert L.lock_holder_pid() == 4242

    @pytest.mark.parametrize("content", ["not a pid", "", "\n", "0\n", "-5\n"])
    def test_garbage_empty_or_released_lockfile_is_none(self, sam3_home, content):
        """The daemon clears its pid when it releases the lock."""
        assert L.lock_holder_pid() is None
        os.makedirs(os.path.dirname(L.lock_file()), exist_ok=True)
        with open(L.lock_file(), "w") as fh:
            fh.write(content)
        assert L.lock_holder_pid() is None

    def test_a_stuck_daemon_holding_the_lock_is_ended(self, sam3_home):
        """A verified daemon of ours, holding the lock past the budget: ended."""
        with _pseudo_daemon() as proc:
            _claim_lock(proc.pid)
            notes = []
            ok = L.terminate_zombie_daemon(proc.pid, lambda k, m: notes.append(m), min_lock_age=0)
            assert ok is True, notes
            assert proc.wait(timeout=5) is not None
        assert any("ending it" in m for m in notes)

    def test_its_runtime_file_goes_only_if_it_still_names_it(self, sam3_home, auth_token):
        with _pseudo_daemon() as proc:
            _claim_lock(proc.pid)
            Path(L.runtime_file()).write_text(json.dumps({
                "port": 1, "token": auth_token, "pid": proc.pid,
                "version": "0", "started_at": time.time()}))
            assert L.terminate_zombie_daemon(proc.pid, lambda k, m: None, min_lock_age=0)
        assert not os.path.exists(L.runtime_file())

    def test_the_runtime_path_it_is_given_is_the_one_it_uses(self, sam3_home, tmp_path, auth_token):
        """find_or_spawn(runtime_path=...) must not have the default file judged."""
        custom = tmp_path / "elsewhere" / "rt.json"
        custom.parent.mkdir()
        default_text = json.dumps({"port": 1, "token": auth_token, "pid": 999999999,
                                   "version": "0", "started_at": time.time()})
        Path(L.runtime_file()).write_text(default_text)
        with _pseudo_daemon() as proc:
            _claim_lock(proc.pid)
            custom.write_text(json.dumps({"port": 1, "token": auth_token, "pid": proc.pid,
                                          "version": "0", "started_at": time.time()}))
            assert L.terminate_zombie_daemon(proc.pid, lambda k, m: None,
                                             runtime_path=str(custom), min_lock_age=0)
        assert not custom.exists()
        assert Path(L.runtime_file()).read_text(encoding="utf-8") == default_text

    def test_a_young_lock_holder_is_left_alone(self, sam3_home):
        """It takes the lock *before* importing torch: it may simply be starting."""
        with _pseudo_daemon() as proc:
            _claim_lock(proc.pid)
            notes = []
            assert L.terminate_zombie_daemon(proc.pid, lambda k, m: notes.append(m)) is False
            assert proc.poll() is None
        assert any("may still be starting" in m for m in notes)

    def test_a_process_that_is_not_a_daemon_is_never_signalled(self, sam3_home):
        """The lockfile of a crashed daemon names a pid that now belongs to
        something else entirely."""
        with _sleeper() as proc:
            _claim_lock(proc.pid)
            notes = []
            assert L.terminate_zombie_daemon(proc.pid, lambda k, m: notes.append(m),
                                             min_lock_age=0) is False
            time.sleep(0.2)
            assert proc.poll() is None, "an unrelated process was signalled"
        assert any("not a sam3gimpd" in m for m in notes)

    def test_a_process_newer_than_the_lockfile_is_never_signalled(self, sam3_home):
        """Even a daemon-looking process did not write a lockfile older than itself."""
        with _pseudo_daemon() as proc:
            _claim_lock(proc.pid)
            old = time.time() - 10 * L.START_TIME_SLACK_S
            os.utime(L.lock_file(), (old, old))
            assert L.terminate_zombie_daemon(proc.pid, lambda k, m: None, min_lock_age=0) is False
            assert proc.poll() is None

    def test_a_pid_the_lockfile_does_not_name_is_left_alone(self, sam3_home):
        with _pseudo_daemon() as proc:
            assert L.terminate_zombie_daemon(proc.pid, lambda k, m: None, min_lock_age=0) is False
            assert proc.poll() is None

    def test_an_exited_process_is_not_a_success(self, sam3_home):
        """Nothing to end; the caller must not think it fixed anything.

        The child has exited but is not reaped yet, so its pid cannot have
        been handed to any other process while this runs.
        """
        proc = subprocess.Popen([sys.executable, "-c", "pass", "serve", "--no-stderr-log"])
        try:
            deadline = time.time() + 10
            while time.time() < deadline and L.pid_alive(proc.pid):
                time.sleep(0.05)
            _claim_lock(proc.pid)
            assert L.terminate_zombie_daemon(proc.pid, lambda k, m: None, min_lock_age=0) is False
        finally:
            proc.wait(timeout=10)


# --------------------------------------------------------------------------- #
# helpers for the lock tests: processes this test owns
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _pseudo_daemon():
    """A child whose command line is a launcher-spawned ``... serve --no-stderr-log``."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)",
                             "serve", "--no-stderr-log"])
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)


@contextlib.contextmanager
def _sleeper():
    """A child that is plainly not a daemon."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)


def _claim_lock(pid):
    """Write ``pid`` into the lockfile, as a daemon does on taking the lock."""
    os.makedirs(os.path.dirname(L.lock_file()), exist_ok=True)
    with open(L.lock_file(), "w") as fh:
        fh.write("%d\n" % pid)


@pytest.mark.skipif(not os.path.isdir("/proc"), reason="needs Linux /proc")
def test_a_defunct_process_is_not_alive():
    """An exited-but-unreaped child still answers kill(pid, 0); it is dead."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    deadline = time.time() + 5.0
    # wait for it to exit *without* reaping it, so it sits as a zombie
    while time.time() < deadline:
        with open("/proc/%d/stat" % proc.pid, "rb") as fh:
            if fh.read().rsplit(b")", 1)[1].split()[0] == b"Z":
                break
        time.sleep(0.02)
    try:
        assert L.pid_alive(proc.pid) is False
    finally:
        proc.wait()



class TestGimpPid:
    """The daemon must be tied to GIMP, not to the process that lives one
    invocation -- on Windows that is not the plug-in's direct parent."""

    TABLE = {
        10: (1, "systemd"),
        20: (10, "explorer.exe"),
        30: (20, "gimp-3.2.exe"),
        40: (30, "python.exe"),         # GIMP's plug-in launcher / interpreter
        50: (40, "pythonw.exe"),        # us
    }

    def test_finds_gimp_two_levels_up(self):
        assert L._find_gimp_ancestor(self.TABLE, 50) == 30

    def test_finds_gimp_when_it_is_the_direct_parent(self):
        assert L._find_gimp_ancestor(self.TABLE, 40) == 30

    def test_none_when_no_gimp_in_the_chain(self):
        table = {1: (0, "init"), 2: (1, "bash"), 3: (2, "python")}
        assert L._find_gimp_ancestor(table, 3) is None

    def test_a_cycle_cannot_spin(self):
        table = {1: (2, "a"), 2: (1, "b")}
        assert L._find_gimp_ancestor(table, 1) is None

    def test_full_paths_and_case_are_handled(self):
        table = {1: (0, "init"), 2: (1, r"C:\\Program Files\\GIMP 3\\bin\\GIMP-3.2.EXE"), 3: (2, "py")}
        assert L._find_gimp_ancestor(table, 3) == 2

    def test_gimp_pid_is_an_int_or_none_and_never_ourselves(self):
        pid = L.gimp_pid()
        assert pid is None or isinstance(pid, int)
        assert pid != os.getpid()

    @pytest.mark.skipif(not os.path.isdir("/proc"), reason="needs Linux /proc")
    def test_posix_table_reads_our_own_chain(self):
        table = L._posix_ancestor_table(os.getpid())
        assert os.getpid() in table
        ppid, name = table[os.getpid()]
        assert ppid == os.getppid() and name



def test_build_command_silences_the_duplicate_stderr_log(sam3_home):
    argv = L.build_command(command=["/py", "-m", "sam3gimpd"], stub=True)
    assert argv.count("--no-stderr-log") == 1


def test_plugin_build_is_a_stable_short_content_hash():
    import launcher as L
    a = L.plugin_build()
    assert len(a) == 8 and int(a, 16) >= 0
    assert L.plugin_build() == a


# --------------------------------------------------------------------------- #
# a daemon on another machine, and the idle timeout setting
# --------------------------------------------------------------------------- #
def test_remote_url_parsing():
    assert L.parse_remote_url("http://gpu-box:41573") == ("gpu-box", 41573)
    assert L.parse_remote_url("gpu-box:41573") == ("gpu-box", 41573)
    assert L.parse_remote_url("http://10.0.0.5:8000/") == ("10.0.0.5", 8000)
    assert L.parse_remote_url("https://gpu-box:41573") is None, "the daemon speaks plain HTTP"
    assert L.parse_remote_url("http://gpu-box") is None, "a port is required"
    assert L.parse_remote_url("") is None


def test_remote_and_idle_settings_round_trip(sam3_home):
    assert L.configured_remote() is None
    assert L.configured_idle_ttl() is None
    L.set_configured_remote("http://gpu-box:41573", "t" * 43)
    assert L.configured_remote() == ("gpu-box", 41573, "t" * 43)
    L.set_configured_idle_ttl(5)
    assert L.configured_idle_ttl() == 300.0
    L.set_configured_idle_ttl(0)
    assert L.configured_idle_ttl() == 0.0
    L.set_configured_remote(None)
    L.set_configured_idle_ttl(None)
    assert L.configured_remote() is None and L.configured_idle_ttl() is None
    with pytest.raises(ValueError):
        L.set_configured_remote("https://nope:1", "t")


def test_a_configured_remote_daemon_is_used_instead_of_spawning(sam3_home, monkeypatch):
    """The remote is a real stub daemon on loopback; runtime.json is removed
    so a local find could not succeed, and spawn=False so a local spawn
    cannot either.  Only the remote path can produce a client."""
    if not _real_sam3d_available():
        pytest.skip("the sam3gimpd package (or its CLI) is not available yet")
    monkeypatch.setenv("PYTHONPATH", DAEMON_DIR + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.delenv(L.ENV_COMMAND, raising=False)
    local = L.find_or_spawn(command=[sys.executable, "-m", "sam3gimpd"], stub=True,
                            parent_pid=os.getpid(), idle_ttl=120, timeout=60.0)
    try:
        info = dict(local.info)
        local.close()
        L.set_configured_remote("http://127.0.0.1:%d" % int(info["port"]), str(info["token"]))
        L.delete_runtime_file()
        result = L.find_or_spawn(spawn=False, parent_pid=os.getpid())
        try:
            assert result.spawned is False
            assert result.info["remote"] is True and result.info["port"] == int(info["port"])
            assert result.hello.engine_mode == "stub"
            assert any(n.startswith("remote:") for n in result.notes)
        finally:
            result.close()
        # A wrong token is a clear failure, not a silent fallback to a local spawn.
        L.set_configured_remote("http://127.0.0.1:%d" % int(info["port"]), "wrong-token")
        with pytest.raises(L.SpawnError) as exc:
            L.find_or_spawn(spawn=False, parent_pid=os.getpid())
        assert "remote daemon" in str(exc.value)
        # An explicit command still wins over the remote setting.
        L.set_configured_remote("http://127.0.0.1:1", "x")
        with pytest.raises(Exception):
            L.find_or_spawn(command=[sys.executable, "-c", "import sys; sys.exit(3)"],
                            spawn=True, timeout=2.0, max_attempts=1)
    finally:
        L.set_configured_remote(None)
        try:
            L.shutdown_daemon(info, grace_ms=0)
        except Exception:
            pass
        _reap(local)


def test_the_configured_idle_ttl_reaches_the_spawn_command(sam3_home, monkeypatch):
    L.set_configured_idle_ttl(7)
    seen = {}

    def fake_spawn(argv, **kw):
        seen["argv"] = list(argv)
        raise L.SpawnError("stop here")

    monkeypatch.setattr(L, "spawn_daemon", fake_spawn)
    monkeypatch.setattr(L, "_python_for_spawn", lambda: sys.executable)
    monkeypatch.delenv(L.ENV_COMMAND, raising=False)
    with pytest.raises(L.SpawnError):
        L.find_or_spawn(parent_pid=os.getpid(), timeout=1.0, max_attempts=1)
    assert "--idle-ttl" in seen["argv"]
    assert seen["argv"][seen["argv"].index("--idle-ttl") + 1] == "420"


# --------------------------------------------------------------------------- #
# liveness versus ownership
# --------------------------------------------------------------------------- #
def test_pid_owned_for_this_process_and_a_missing_one():
    assert L.pid_owned(os.getpid()) is True
    assert L.pid_owned(999999999) is False
    assert L.pid_owned(None) is False


@pytest.mark.skipif(not os.path.isdir("/proc/1") or os.geteuid() == 0
                    or os.stat("/proc/1").st_uid == os.geteuid(),
                    reason="needs a pid owned by another user")
def test_another_users_process_is_alive_but_not_owned():
    assert L.pid_alive(1) is True
    assert L.pid_owned(1) is False


@pytest.mark.skipif(os.name == "nt", reason="POSIX EPERM semantics")
def test_eperm_without_proc_means_not_ours(monkeypatch):
    def _kill(pid, sig):
        raise PermissionError("not yours")

    monkeypatch.setattr(os, "kill", _kill)
    monkeypatch.setattr(L.os.path, "isdir", lambda path: False)
    assert L.pid_alive(4711) is True
    assert L.pid_owned(4711) is False


def _fake_win32(monkeypatch, handle, last_error=0, exit_code=259):
    """Just enough of kernel32 for pid_alive / pid_owned's decisions."""

    class DWORD:
        def __init__(self, value=0):
            self.value = value

    def get_exit_code(_handle, ref):
        ref.value = exit_code
        return 1

    k32 = SimpleNamespace(OpenProcess=lambda access, inherit, pid: handle,
                          GetExitCodeProcess=get_exit_code, CloseHandle=lambda h: 1,
                          GetCurrentProcess=lambda: -1)
    w = SimpleNamespace(k32=k32, DWORD=DWORD, byref=lambda o: o, open_process=lambda pid: handle,
                        token_user_sid=lambda h: b"S-1-5-21-same",
                        PROCESS_QUERY_LIMITED_INFORMATION=0x1000, STILL_ACTIVE=259,
                        ERROR_ACCESS_DENIED=5, ERROR_INVALID_PARAMETER=87)
    monkeypatch.setattr(L, "_win32", lambda: w)
    monkeypatch.setattr(L, "_win_last_error", lambda: last_error)
    monkeypatch.setattr(L, "platform_key", lambda: "windows")
    return w


class TestWindowsPidSemantics:
    """``ERROR_ACCESS_DENIED`` means the process exists -- and is not ours."""

    PID = 999999999   # no /proc entry, so only the faked Win32 calls decide

    def test_access_denied_is_alive_but_not_owned(self, monkeypatch):
        _fake_win32(monkeypatch, handle=None, last_error=5)
        assert L.pid_alive(self.PID) is True
        assert L.pid_owned(self.PID) is False

    def test_no_such_process_is_dead(self, monkeypatch):
        _fake_win32(monkeypatch, handle=None, last_error=87)
        assert L.pid_alive(self.PID) is False
        assert L.pid_owned(self.PID) is False

    def test_an_exit_code_other_than_still_active_is_dead(self, monkeypatch):
        _fake_win32(monkeypatch, handle=1234, exit_code=0)
        assert L.pid_alive(self.PID) is False

    def test_a_running_process_of_ours(self, monkeypatch):
        _fake_win32(monkeypatch, handle=1234)
        assert L.pid_alive(self.PID) is True
        assert L.pid_owned(self.PID) is True

    def test_a_running_process_of_another_user(self, monkeypatch):
        w = _fake_win32(monkeypatch, handle=1234)
        sids = iter([b"S-1-5-21-theirs", b"S-1-5-21-mine"])
        w.token_user_sid = lambda h: next(sids)
        assert L.pid_owned(self.PID) is False


# --------------------------------------------------------------------------- #
# what counts as a daemon process
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("argv,ok", [
    (["/venv/bin/python", "-m", "sam3gimpd", "serve", "--port", "0"], True),
    (["/venv/bin/python", "/venv/bin/sam3gimpd", "serve"], True),
    ([r"C:\Users\Jane Doe\venv\Scripts\sam3gimpd.exe", "serve", "--no-stderr-log"], True),
    ([r"C:\Program Files\Py\pythonw.exe", "-m", "sam3gimpd", "serve"], True),
    (["python3", "/opt/wrapper.py", "serve", "--host", "127.0.0.1", "--no-stderr-log"], True),
    (["tail", "-f", "/home/you/.local/share/sam3-gimp/logs/sam3gimpd.log"], False),
    (["python", "-m", "sam3gimpd", "doctor"], False),
    (["sleep", "1000"], False),
    ([], False),
])
def test_is_daemon_argv(argv, ok):
    assert L._is_daemon_argv(argv) is ok


@pytest.mark.parametrize("text,seconds", [
    ("05:07", 307.0), ("1:02:03", 3723.0), ("2-01:00:00", 176400.0), ("nonsense", None),
])
def test_parse_ps_etime(text, seconds):
    assert L._parse_ps_etime(text) == seconds


@pytest.mark.skipif(os.name == "nt", reason="reads /proc or ps")
def test_daemon_process_identity():
    with _pseudo_daemon() as proc:
        time.sleep(0.2)
        assert L._daemon_process(proc.pid, not_after=time.time()) is True
        # Created long after the file it supposedly wrote: a reused pid.
        assert L._daemon_process(proc.pid, not_after=time.time() - 1000) is False
    with _sleeper() as proc:
        time.sleep(0.2)
        assert L._daemon_process(proc.pid, not_after=time.time()) is False
    assert L._daemon_process(os.getpid(), not_after=None) is False
    assert L._daemon_process(999999999, not_after=None) is False


# --------------------------------------------------------------------------- #
# runtime.json is only ever removed if it still says what was judged stale
# --------------------------------------------------------------------------- #
def test_delete_runtime_file_compares_before_it_deletes(sam3_home, auth_token):
    judged = {"port": 1, "token": auth_token, "pid": 111, "version": "0", "started_at": 1.0}
    fresh = dict(judged, pid=222, token="u" * 43)
    Path(L.runtime_file()).write_text(json.dumps(fresh))
    assert L.delete_runtime_file(expect=judged) is False
    assert json.loads(Path(L.runtime_file()).read_text(encoding="utf-8"))["pid"] == 222
    Path(L.runtime_file()).write_text(json.dumps(judged))
    assert L.delete_runtime_file(expect=judged) is True
    assert not os.path.exists(L.runtime_file())


def test_launch_result_repr_hides_the_token(sam3_home):
    d = T.FakeDaemon(token="s3cr3t-" + "x" * 36)
    try:
        Path(L.runtime_file()).write_text(json.dumps(d.runtime_info()))
        result = L.find_or_spawn(spawn=False)
        try:
            assert d.token not in repr(result)
            assert all(d.token not in n for n in result.notes)
        finally:
            result.close()
    finally:
        d.close()


# --------------------------------------------------------------------------- #
# a spawn waits for a cold start unless told otherwise
# --------------------------------------------------------------------------- #
def test_a_spawn_waits_the_cold_start_budget_by_default(sam3_home, monkeypatch):
    import inspect

    assert inspect.signature(L.find_or_spawn).parameters["timeout"].default is None
    seen = []
    monkeypatch.setattr(L, "spawn_daemon", lambda argv, **k: 1)
    monkeypatch.setattr(L, "_wait_for_runtime",
                        lambda path, min_mtime, timeout, *a, **k: seen.append(timeout) or False)
    with pytest.raises(L.DaemonStartTimeout):
        L.find_or_spawn(command=["/bin/true"])
    assert seen == [L.HANDSHAKE_COLD_TIMEOUT] and L.HANDSHAKE_COLD_TIMEOUT >= 60.0


# --------------------------------------------------------------------------- #
# a daemon bound to every interface is dialled on loopback
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("recorded", ["0.0.0.0", "127.0.0.1"])
def test_an_unspecified_bind_address_is_reached_on_loopback(sam3_home, recorded):
    d = T.FakeDaemon()
    try:
        Path(L.runtime_file()).write_text(json.dumps(d.runtime_info(host=recorded)))
        result = L.find_or_spawn(spawn=False)
        try:
            assert result.client.host == "127.0.0.1"
            assert result.port == d.port
        finally:
            result.close()
    finally:
        d.close()


# --------------------------------------------------------------------------- #
# what the launcher writes is private
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
class TestPrivateFiles:
    def test_settings_file_is_0600_and_the_directories_0700(self, sam3_home):
        os.chmod(str(sam3_home), 0o755)
        L.set_configured_remote("http://gpu-box:41573", "t" * 43)
        assert os.stat(L.settings_file()).st_mode & 0o777 == 0o600
        L.ensure_layout()
        assert os.stat(L.base_dir()).st_mode & 0o777 == 0o700
        for path in (L.log_dir(), L.hf_home(), L.models_dir()):
            assert os.stat(path).st_mode & 0o777 == 0o700, path

    def test_a_fresh_base_directory_is_created_private(self, tmp_path, monkeypatch):
        monkeypatch.setenv(L.ENV_HOME, str(tmp_path / "new" / "home"))
        L.write_settings({"a": 1})
        assert os.stat(L.base_dir()).st_mode & 0o777 == 0o700
        assert os.stat(L.settings_file()).st_mode & 0o777 == 0o600

    def test_the_daemon_log_is_created_private(self, sam3_home, monkeypatch):
        monkeypatch.setattr(subprocess, "Popen", _FakePopen)
        L.spawn_daemon(["/bin/true"])
        assert os.stat(L.server_log()).st_mode & 0o777 == 0o600

    def test_concurrent_writers_never_see_a_torn_file(self, sam3_home):
        import threading

        errors = []

        def writer(n):
            try:
                for i in range(20):
                    L.write_settings({"writer": n, "i": i})
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert errors == []
        assert set(L.read_settings()) == {"writer", "i"}
        assert [n for n in os.listdir(L.base_dir()) if n.endswith(".tmp")] == []


# --------------------------------------------------------------------------- #
# remote addresses: refused with a reason
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("url,words", [
    ("https://gpu-box:41573", "plain HTTP"),
    ("ftp://gpu-box:41573", "not supported"),
    ("file:///etc/passwd", "not supported"),
    ("http://user:secret@gpu-box:41573", "Token field"),
    ("http://gpu-box:41573/api", "no path"),
    ("http://gpu-box:41573/?token=abc", "no path"),
    ("http://gpu-box", "port is required"),
    ("http://gpu-box:99999", "port must be"),
    ("", "http://host:port"),
])
def test_bad_remote_addresses_are_refused_with_a_reason(sam3_home, url, words):
    assert L.parse_remote_url(url) is None
    assert words in L.remote_url_error(url)
    if url:
        with pytest.raises(ValueError) as exc:
            L.set_configured_remote(url, "t" * 43)
        assert words in str(exc.value)
    assert L.configured_remote() is None


def test_ipv6_remote_addresses_parse():
    assert L.parse_remote_url("http://[::1]:41573") == ("::1", 41573)
    assert L.remote_url_error("http://[fd00::5]:8000/") is None


# --------------------------------------------------------------------------- #
# a stale runtime.json: whatever answers on its port is never trusted
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.isdir("/proc/1") or os.geteuid() == 0
                    or os.stat("/proc/1").st_uid == os.geteuid(),
                    reason="needs a pid owned by another user")
def test_another_users_pid_is_stale_and_its_port_is_never_contacted(sam3_home):
    srv = T.RawServer(T.unproven_hello)
    try:
        Path(L.runtime_file()).write_text(json.dumps(srv.runtime_info(pid=1)))
        with pytest.raises(C.DaemonUnavailable):
            L.find_or_spawn(spawn=False)
        assert srv.requests == [], "the token was offered to another user's process"
        assert not os.path.exists(L.runtime_file())
    finally:
        srv.close()


def test_an_impostor_on_the_port_is_never_sent_an_image(sam3_home):
    """A stale runtime.json whose pid is some other live process of ours, and
    something else listening on the port: refused, file cleared, nothing
    signalled."""
    srv = T.RawServer(T.unproven_hello)
    try:
        with _sleeper() as proc:
            time.sleep(0.1)
            Path(L.runtime_file()).write_text(json.dumps(srv.runtime_info(pid=proc.pid)))
            with pytest.raises(C.DaemonUnavailable):
                L.find_or_spawn(spawn=False)
            assert proc.poll() is None
        assert [r[0] for r in srv.requests] == ["GET"]
        assert not os.path.exists(L.runtime_file())
    finally:
        srv.close()


@pytest.mark.needs_daemon
def test_an_impostor_is_replaced_by_a_daemon_we_start(sam3_home, fake_daemon_script):
    srv = T.RawServer(T.unproven_hello)
    result = None
    try:
        with _sleeper() as proc:
            time.sleep(0.1)
            Path(L.runtime_file()).write_text(json.dumps(srv.runtime_info(pid=proc.pid)))
            result = L.find_or_spawn(command=_spawn_argv(fake_daemon_script), timeout=25.0)
            assert result.spawned is True and result.port != srv.port
            assert proc.poll() is None
        assert all(r[0] == "GET" for r in srv.requests)
    finally:
        srv.close()
        _reap(result)


def test_an_older_daemon_of_ours_is_never_used_without_spawn(sam3_home):
    """A daemon that cannot prove itself is not used -- and a probe does not
    end anything."""
    d = T.FakeDaemon(prove_identity=False, api_version="1.0")
    try:
        with _pseudo_daemon() as proc:
            time.sleep(0.1)
            Path(L.runtime_file()).write_text(json.dumps(d.runtime_info(pid=proc.pid)))
            with pytest.raises(L.IncompatibleDaemon) as exc:
                L.find_or_spawn(spawn=False)
            assert "Update daemon" in str(exc.value)
            assert proc.poll() is None and d.shutting_down is False
            assert os.path.exists(L.runtime_file())
    finally:
        d.close()


@pytest.mark.needs_daemon
def test_an_older_daemon_of_ours_is_asked_to_exit_and_replaced(sam3_home, fake_daemon_script,
                                                                monkeypatch):
    """No proof, API 1.0, and the pid is verifiably a sam3gimpd of ours: an
    older build.  Asked to exit first, ended if it does not, then replaced."""
    monkeypatch.setattr(L, "RETIRE_WAIT_S", 0.5)
    d = T.FakeDaemon(prove_identity=False, api_version="1.0")
    result = None
    try:
        with _pseudo_daemon() as proc:
            time.sleep(0.1)
            Path(L.runtime_file()).write_text(json.dumps(d.runtime_info(pid=proc.pid)))
            result = L.find_or_spawn(command=_spawn_argv(fake_daemon_script), timeout=25.0)
            assert d.shutting_down is True, "it was asked to exit first"
            assert proc.wait(timeout=10) is not None, "and ended when it did not"
            assert result.spawned is True and result.port != d.port
            assert any("older than API" in n for n in result.notes)
    finally:
        d.close()
        _reap(result)


@pytest.mark.needs_daemon
def test_an_older_daemon_that_cannot_be_identified_is_asked_but_never_signalled(
        sam3_home, fake_daemon_script, monkeypatch):
    """Where the platform cannot say what the pid is, the older daemon still
    gets its POST /shutdown -- it holds the lock, so a new one could not
    start otherwise -- but no signal."""
    monkeypatch.setattr(L, "RETIRE_WAIT_S", 0.5)
    monkeypatch.setattr(L, "_daemon_process", lambda pid, not_after=None: None)
    d = T.FakeDaemon(prove_identity=False, api_version="1.0")
    result = None
    try:
        with _pseudo_daemon() as proc:
            time.sleep(0.1)
            Path(L.runtime_file()).write_text(json.dumps(d.runtime_info(pid=proc.pid)))
            result = L.find_or_spawn(command=_spawn_argv(fake_daemon_script), timeout=25.0)
            assert d.shutting_down is True
            assert proc.poll() is None, "an unidentified process was signalled"
            assert result.spawned is True and result.port != d.port
    finally:
        d.close()
        _reap(result)


@pytest.mark.parametrize("respond", [
    lambda m, p, h: T.unproven_hello(m, p, h, proof="ab" * 32),   # a wrong proof
    lambda m, p, h: T.unproven_hello(m, p, h),                    # none, yet claims API 1.1
], ids=["wrong-proof", "no-proof-new-api"])
def test_an_answer_with_an_invalid_proof_is_never_shut_down_or_signalled(sam3_home, respond,
                                                                         monkeypatch):
    """Even when the pid is verifiably a daemon of ours, a responder that
    fails the proof in any way other than "older build" is left alone."""
    monkeypatch.setattr(L, "spawn_daemon",
                        lambda *a, **k: pytest.fail("nothing may be started over it"))
    srv = T.RawServer(respond)
    try:
        with _pseudo_daemon() as proc:
            time.sleep(0.1)
            Path(L.runtime_file()).write_text(json.dumps(srv.runtime_info(pid=proc.pid)))
            with pytest.raises(L.IncompatibleDaemon):
                L.find_or_spawn(command=["/bin/true"])
            assert [r[0] for r in srv.requests] == ["GET"], "it was sent more than /hello"
            assert proc.poll() is None
            assert os.path.exists(L.runtime_file())
    finally:
        srv.close()


def test_a_zombie_check_never_acts_on_a_wrong_proof(sam3_home):
    srv = T.RawServer(lambda m, p, h: T.unproven_hello(m, p, h, proof="ab" * 32))
    try:
        with _pseudo_daemon() as proc:
            _claim_lock(proc.pid)
            Path(L.runtime_file()).write_text(json.dumps(srv.runtime_info(pid=proc.pid)))
            notes = []
            assert L.terminate_zombie_daemon(proc.pid, lambda k, m: notes.append(m),
                                             min_lock_age=0) is False
            assert proc.poll() is None
        assert any("invalid identity proof" in m for m in notes)
    finally:
        srv.close()


@pytest.mark.needs_daemon
def test_a_freshly_started_old_daemon_means_the_install_is_outdated(sam3_home, fake_daemon_script,
                                                                     monkeypatch):
    monkeypatch.setenv("FAKE_DAEMON_NO_PROOF", "1")
    monkeypatch.setenv("FAKE_DAEMON_API_VERSION", "1.0")
    before = {id(p) for p in L._SPAWNED}
    with pytest.raises(L.IncompatibleDaemon) as exc:
        L.find_or_spawn(command=_spawn_argv(fake_daemon_script), timeout=25.0)
    assert "Update daemon" in str(exc.value)
    for proc in [p for p in L._SPAWNED if id(p) not in before]:
        proc.wait(timeout=15)   # it was told to exit, and did
    assert L.read_runtime_info() is None


# --------------------------------------------------------------------------- #
# slow, busy or racing daemons are waited for, never killed
# --------------------------------------------------------------------------- #
def test_a_daemon_that_does_not_answer_yet_keeps_its_runtime_file(sam3_home):
    d = T.FakeDaemon()
    d.hello_gate.clear()
    try:
        with _pseudo_daemon() as proc:
            time.sleep(0.1)
            Path(L.runtime_file()).write_text(json.dumps(d.runtime_info(pid=proc.pid)))
            with pytest.raises(C.DaemonUnavailable):
                L.find_or_spawn(spawn=False, read_timeout=0.3)
            assert os.path.exists(L.runtime_file()), "a busy daemon's file was deleted"
            assert proc.poll() is None
    finally:
        d.close()


def test_a_daemon_that_does_not_accept_connections_keeps_its_runtime_file(sam3_home, monkeypatch):
    """A stalled accept loop (a frozen process, a full listen queue) drops
    connections, so connecting times out.  The pid is alive and verifiably
    ours, so that reads as busy, as a slow answer does -- not as gone, which
    would delete the file of a daemon that may recover a moment later and
    leave the next spawn fighting it for the lock."""
    import socket

    def _timeout(*_args, **_kwargs):
        raise socket.timeout("timed out")

    d = T.FakeDaemon()
    try:
        with _pseudo_daemon() as proc:
            time.sleep(0.1)
            Path(L.runtime_file()).write_text(json.dumps(d.runtime_info(pid=proc.pid)))
            monkeypatch.setattr(socket, "create_connection", _timeout)
            with pytest.raises(C.DaemonUnavailable):
                L.find_or_spawn(spawn=False, read_timeout=0.3)
            assert os.path.exists(L.runtime_file()), "a stalled daemon's file was deleted"
            assert proc.poll() is None
    finally:
        d.close()


def test_a_briefly_stalled_daemon_is_waited_for(sam3_home, monkeypatch):
    """Paused for a few seconds (swap, a GIL-bound load, a virus scan): the
    next dialog waits for it rather than deleting its file and replacing it."""
    import threading

    d = T.FakeDaemon()
    d.hello_gate.clear()
    threading.Timer(2.0, d.hello_gate.set).start()
    monkeypatch.setattr(L, "spawn_daemon",
                        lambda *a, **k: pytest.fail("a stalled daemon must not be replaced"))
    try:
        with _pseudo_daemon() as proc:
            time.sleep(0.1)
            Path(L.runtime_file()).write_text(json.dumps(d.runtime_info(pid=proc.pid)))
            result = L.find_or_spawn(command=["/bin/true"], read_timeout=0.5, timeout=15.0)
            try:
                assert result.port == d.port and result.spawned is False
                assert any("did not answer" in n for n in result.notes)
            finally:
                result.close()
            assert proc.poll() is None
    finally:
        d.close()


def test_a_daemon_that_never_answers_is_given_up_on_without_a_kill(sam3_home, monkeypatch):
    d = T.FakeDaemon()
    d.hello_gate.clear()
    monkeypatch.setattr(L, "spawn_daemon",
                        lambda *a, **k: pytest.fail("a live daemon of ours must not be replaced"))
    try:
        with _pseudo_daemon() as proc:
            time.sleep(0.1)
            Path(L.runtime_file()).write_text(json.dumps(d.runtime_info(pid=proc.pid)))
            with pytest.raises(L.DaemonStartTimeout) as exc:
                L.find_or_spawn(command=["/bin/true"], read_timeout=0.3, timeout=1.5)
            assert "has not answered" in str(exc.value)
            assert proc.poll() is None
            assert os.path.exists(L.runtime_file())
    finally:
        d.close()


def test_a_stale_pid_in_the_lockfile_is_never_signalled(sam3_home):
    """The lockfile a crashed daemon left behind names a pid that now belongs
    to an unrelated process.  A child that exits 0 ("lock busy") must not get
    that process killed."""
    with _sleeper() as victim:
        time.sleep(0.1)
        _claim_lock(victim.pid)
        start = time.time()
        with pytest.raises(L.DaemonStartTimeout) as exc:
            L.find_or_spawn(command=[sys.executable, "-c", "import sys; sys.exit(0)"],
                            timeout=20.0)
        assert time.time() - start < 15.0
        assert "instance lock" in str(exc.value)
        time.sleep(0.2)
        assert victim.poll() is None, "an unrelated process was signalled"


_needs_flock = pytest.mark.skipif(os.name == "nt", reason="the stand-in daemon locks with fcntl")


@_needs_flock
@pytest.mark.needs_daemon
@pytest.mark.slow
def test_two_launchers_starting_at_once_share_one_daemon(sam3_home, fake_daemon_script,
                                                          monkeypatch):
    """Two dialogs opened together: one daemon starts, the other launcher's
    child finds the lock taken, and that launcher waits for the first daemon
    instead of killing it."""
    import threading

    monkeypatch.setenv("FAKE_DAEMON_LOCK", "1")
    monkeypatch.setenv("FAKE_DAEMON_START_DELAY", "1.5")
    results, errors, notes = [], [], []

    def launch(delay):
        time.sleep(delay)
        try:
            results.append(L.find_or_spawn(command=_spawn_argv(fake_daemon_script), timeout=20.0,
                                           on_event=lambda k, m: notes.append((k, m))))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=launch, args=(d,)) for d in (0.0, 0.3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(40)
    try:
        assert errors == [], errors
        assert len(results) == 2 and results[0].pid == results[1].pid
        assert L.pid_alive(results[0].pid)
        assert not any(k == "zombie" for k, _m in notes), notes
    finally:
        for r in results[1:]:
            r.close()
        _reap(results[0] if results else None)


@_needs_flock
@pytest.mark.needs_daemon
@pytest.mark.slow
def test_a_retry_after_a_slow_start_waits_for_the_first_daemon(sam3_home, fake_daemon_script,
                                                                monkeypatch):
    """The first attempt gives up while its daemon is still importing torch;
    the retry must converge on that daemon, not kill it."""
    monkeypatch.setenv("FAKE_DAEMON_LOCK", "1")
    monkeypatch.setenv("FAKE_DAEMON_START_DELAY", "4")
    with pytest.raises(L.DaemonStartTimeout):
        L.find_or_spawn(command=_spawn_argv(fake_daemon_script), timeout=1.0)
    deadline = time.time() + 10
    while time.time() < deadline and L.lock_holder_pid() is None:
        time.sleep(0.05)
    first = L.lock_holder_pid()
    assert first is not None and L.pid_alive(first)
    assert L.read_runtime_info() is None, "the first daemon must still be starting"
    result = L.find_or_spawn(command=_spawn_argv(fake_daemon_script), timeout=20.0)
    try:
        assert result.pid == first
        assert not any(n.startswith("zombie") for n in result.notes), result.notes
    finally:
        _reap(result)


@_needs_flock
@pytest.mark.needs_daemon
@pytest.mark.slow
def test_a_stuck_lock_holder_is_ended_and_replaced(sam3_home, fake_daemon_script, monkeypatch):
    """The field case: a daemon of ours holds the lock long past any start-up
    and never publishes.  It -- and only it -- is ended."""
    env = dict(os.environ, FAKE_DAEMON_LOCK="1", FAKE_DAEMON_NEVER_PUBLISH="1",
               FAKE_DAEMON_LINGER="60")
    holder = subprocess.Popen([sys.executable, str(fake_daemon_script.path), "serve",
                               "--no-stderr-log"], env=env)
    try:
        deadline = time.time() + 20
        while time.time() < deadline and L.lock_holder_pid() != holder.pid:
            time.sleep(0.05)
        assert L.lock_holder_pid() == holder.pid
        monkeypatch.setattr(L, "COLD_START_BUDGET", 0.5)
        monkeypatch.setattr(L, "ZOMBIE_GRACE_S", 0.3)
        time.sleep(0.6)
        monkeypatch.setenv("FAKE_DAEMON_LOCK", "1")
        result = L.find_or_spawn(command=_spawn_argv(fake_daemon_script), timeout=20.0)
        try:
            assert holder.wait(timeout=10) is not None
            assert result.pid != holder.pid
            assert any("ending it" in n for n in result.notes), result.notes
        finally:
            _reap(result)
    finally:
        if holder.poll() is None:
            holder.kill()
        holder.wait(timeout=10)
