"""Headless tests for ``plugin/sam3_gimp/bootstrap.py``.

Everything here runs with no GPU, no torch, no GIMP and **no network** (a few
dialog smoke tests at the end need a display).  That is possible because
``bootstrap`` puts every side effect behind an injection point:

* subprocesses go through a ``CommandRunner`` -- :class:`FakeRunner` records
  argv and returns canned output;
* downloads go through a ``Downloader`` -- :class:`FakeDownloader` writes bytes
  from memory;
* the HuggingFace calls and the converter fetch take an ``opener`` --
  :func:`fake_opener` answers without touching the network.

The commands themselves are asserted *exactly*, because an install is the one
thing a user cannot debug: torch read from the wrong index silently produces a
CPU build on a 4090, and a wrong ``uv`` asset name 404s halfway through
onboarding.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import sys
import threading
import time
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

import bootstrap as bs


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakeRunner(bs.CommandRunner):
    """A ``CommandRunner`` that runs nothing and records everything."""

    def __init__(self, results=None, default=(0, ("ok",))):
        super().__init__()
        self.calls = []           # list of (argv, env)
        self.results = dict(results or {})   # argv[0] basename or full-argv key -> (rc, lines)
        self.default = default

    def run(self, argv, *, env=None, cwd=None, on_line=None, timeout=None, cancel=None,
            input_text=None):
        argv = [str(a) for a in argv]
        self.calls.append((argv, dict(env or {})))
        rc, lines = self._lookup(argv)
        for line in lines:
            if on_line:
                on_line(line)
        return bs.CommandResult(argv, rc, list(lines), 0.0)

    def _lookup(self, argv):
        joined = " ".join(argv)
        if joined in self.results:
            return self.results[joined]
        # Match on the program name, tolerating Windows-style paths on POSIX.
        program = argv[0].replace("\\", "/").rsplit("/", 1)[-1]
        for key, value in self.results.items():
            if key == program or program.startswith(key):
                return value
        return self.default

    @property
    def argvs(self):
        return [argv for argv, _env in self.calls]


class FakeDownloader(bs.Downloader):
    """Serves bytes from a dict instead of the network."""

    def __init__(self, payloads):
        super().__init__()
        self.payloads = dict(payloads)
        self.urls = []

    def download(self, url, dest, *, on_progress=None, headers=None, timeout=60.0,
                 resume=True, cancel=None):
        self.urls.append(url)
        if url not in self.payloads:
            raise bs.BootstrapError("unexpected URL: %s" % url)
        data = self.payloads[url]
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(data)
        if on_progress:
            on_progress(1.0, len(data), len(data))
        return dest


class FakeResponse:
    def __init__(self, status, body, headers=None):
        self.status = status
        self.code = status
        self._body = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.headers = headers or {}
        self._pos = 0

    def read(self, n=-1):
        if n is None or n < 0:
            chunk, self._pos = self._body[self._pos :], len(self._body)
            return chunk
        chunk = self._body[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk

    def close(self):
        pass


def fake_opener(status, body, record=None):
    def _open(url, headers=None, timeout=None):
        if record is not None:
            record.append((url, dict(headers or {})))
        if status >= 400:
            err = OSError("HTTP %s" % status)
            err.code = status
            err.read = lambda: json.dumps(body).encode()
            raise err
        return FakeResponse(status, body)

    return _open


def make_uv_tar(tmp_path: Path, member="uv-x86_64-unknown-linux-gnu/uv", payload=b"#!/bin/sh\n") -> bytes:
    buf = BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(member)
        info.size = len(payload)
        tf.addfile(info, BytesIO(payload))
    return buf.getvalue()


def make_uv_zip(member="uv-x86_64-pc-windows-msvc/uv.exe", payload=b"MZ") -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        # A fixed timestamp: the archive's hash is pinned, so it must not
        # change from one build to the next.
        zf.writestr(zipfile.ZipInfo(member, date_time=(2026, 1, 1, 0, 0, 0)), payload)
    return buf.getvalue()


def make_host_uv_archive(payload=None) -> bytes:
    """A fake of the uv release asset this host downloads: a ``.zip`` on
    Windows, a ``.tar.gz`` everywhere else, as ``bootstrap.uv_asset_name`` says."""
    if bs.uv_asset_name().endswith(".zip"):
        return make_uv_zip(payload=payload or b"MZ")
    return make_uv_tar(None, payload=payload or b"#!/bin/sh\n")


#: One fake uv archive for the whole module: gzip stamps the time into its
#: header, so building it twice gives two different SHA-256s.
FAKE_UV_ARCHIVE = make_host_uv_archive()


@pytest.fixture
def pinned_fake_uv(monkeypatch):
    """Pin :data:`FAKE_UV_ARCHIVE`'s hash as this host's uv asset, the way the
    real archives are pinned in ``bootstrap.UV_SHA256``."""
    monkeypatch.setitem(bs.UV_SHA256, bs.uv_asset_name(),
                        hashlib.sha256(FAKE_UV_ARCHIVE).hexdigest())
    return FAKE_UV_ARCHIVE


#: Stands in for the pinned transformers converter.
FAKE_CONVERTER = b"# convert_sam3_to_hf.py stand-in\nimport sys\nprint('converted', sys.argv[1:])\n"


@pytest.fixture
def pinned_converter(monkeypatch):
    """Pin :data:`FAKE_CONVERTER` as the converter's SHA-256 and return an
    opener that serves it; ``opener.calls`` counts the fetches."""
    monkeypatch.setattr(bs, "CONVERT_SCRIPT_SHA256", hashlib.sha256(FAKE_CONVERTER).hexdigest())

    def opener(url):
        opener.calls.append(url)
        return FAKE_CONVERTER

    opener.calls = []
    return opener


def fake_python(tmp_path):
    """An interpreter path that exists (probe_interpreter checks) but that a
    fake runner answers for."""
    py = tmp_path / "python"
    if not py.exists():
        py.write_text("")
    return str(py)


class ConversionRunner:
    """Answers ``probe_interpreter``'s probe, and records the program each
    conversion was fed on stdin."""

    def __init__(self, *, torch="2.9.0", transformers="5.16.1", probe_ok=True,
                 convert=(0, ["done"])):
        self.torch, self.transformers, self.probe_ok = torch, transformers, probe_ok
        self.convert = convert
        self.converted = []

    def run(self, argv, *, input_text=None, **_kw):
        argv = [str(a) for a in argv]
        if argv[1] == "-c":
            if not self.probe_ok:
                return bs.CommandResult(argv, 1, ["Traceback: it is not python"])
            report = {"python_version": "3.12.10", "executable": argv[0], "torch": self.torch,
                      "cuda": True, "transformers": self.transformers, "sam3gimpd": "0.1.1"}
            return bs.CommandResult(argv, 0, ["SAM3PROBE" + json.dumps(report)])
        assert argv[1] == "-", argv
        self.converted.append(input_text)
        rc, lines = self.convert
        return bs.CommandResult(argv, rc, list(lines))


class MakesFilesRunner(FakeRunner):
    """A FakeRunner that leaves behind the files each install command would,
    so the steps' filesystem probes see a finished install afterwards."""

    def run(self, argv, **kw):
        result = super().run(argv, **kw)
        if result.ok:
            joined = " ".join(str(a) for a in argv)
            if " venv " in " %s " % joined:
                os.makedirs(bs.venv_bin_dir(), exist_ok=True)
                Path(bs.venv_python()).write_text("")
            elif "[runtime]" in joined:
                os.makedirs(bs.venv_bin_dir(), exist_ok=True)
                Path(bs.sam3d_executable()).write_text("")
        return result


# --------------------------------------------------------------------------- #
# platform / accelerator detection
# --------------------------------------------------------------------------- #
class TestPlatformDetection:
    @pytest.mark.parametrize(
        "name,expected",
        [("Windows", "windows"), ("win32", "windows"), ("Darwin", "macos"),
         ("macos", "macos"), ("Linux", "linux"), ("FreeBSD", "linux")],
    )
    def test_platform_key(self, name, expected):
        assert bs.platform_key(name) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [("AMD64", "x86_64"), ("x86_64", "x86_64"), ("arm64", "aarch64"),
         ("aarch64", "aarch64"), ("i686", "i686"), ("x86", "i686")],
    )
    def test_machine_key(self, raw, expected):
        assert bs.machine_key(raw) == expected

    def test_machine_key_defaults_when_empty(self):
        assert bs.machine_key("") == "x86_64"


class TestAcceleratorDetection:
    def test_apple_silicon_is_mps(self):
        accel = bs.detect_accelerator(platform_name="Darwin", machine="arm64", env={})
        assert accel.kind == "mps" and accel.is_gpu

    def test_intel_mac_is_cpu(self):
        accel = bs.detect_accelerator(platform_name="Darwin", machine="x86_64", env={})
        assert accel.kind == "cpu"

    def test_windows_with_nvidia_smi_is_cuda(self):
        runner = FakeRunner({"nvidia-smi": (0, ["GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-abc)"])})
        accel = bs.detect_accelerator(
            platform_name="Windows", machine="AMD64", env={},
            runner=runner, which=lambda _n: "C:\\Windows\\System32\\nvidia-smi.exe",
        )
        assert accel.kind == "cuda"
        assert accel.detail["gpus"]

    def test_nvidia_smi_present_but_failing_is_cpu(self):
        """A driver-less machine with a leftover nvidia-smi must not get CUDA wheels."""
        runner = FakeRunner({"nvidia-smi": (9, ["NVIDIA-SMI has failed because it couldn't "
                                                "communicate with the NVIDIA driver."])})
        accel = bs.detect_accelerator(
            platform_name="Linux", machine="x86_64", env={},
            runner=runner, which=lambda _n: "/usr/bin/nvidia-smi",
        )
        assert accel.kind == "cpu"

    def test_linux_without_nvidia_is_cpu(self):
        accel = bs.detect_accelerator(
            platform_name="Linux", machine="x86_64", env={},
            which=lambda _n: None, exists=lambda _p: False,
        )
        assert accel.kind == "cpu"

    def test_linux_proc_driver_fallback(self):
        accel = bs.detect_accelerator(
            platform_name="Linux", machine="x86_64", env={},
            which=lambda _n: None,
            exists=lambda p: p == "/proc/driver/nvidia/version",
        )
        assert accel.kind == "cuda"
        assert accel.detail["source"] == "/proc/driver/nvidia/version"

    def test_windows_system32_fallback(self):
        env = {"SystemRoot": "C:\\Windows"}
        accel = bs.detect_accelerator(
            platform_name="Windows", machine="AMD64", env=env,
            which=lambda _n: None,
            exists=lambda p: p.endswith("nvidia-smi.exe"),
        )
        assert accel.kind == "cuda"

    def test_force_device_overrides_everything(self):
        accel = bs.detect_accelerator(
            platform_name="Windows", machine="AMD64", env={"SAM3_FORCE_DEVICE": "cpu"},
            which=lambda _n: "nvidia-smi",
        )
        assert accel.kind == "cpu"
        assert "SAM3_FORCE_DEVICE" in accel.reason

    def test_assume_nvidia_override(self):
        ok, detail = bs.detect_nvidia(env={"SAM3_ASSUME_NVIDIA": "1"}, which=lambda _n: None)
        assert ok and detail["source"] == "SAM3_ASSUME_NVIDIA"
        ok, _ = bs.detect_nvidia(env={"SAM3_ASSUME_NVIDIA": "0"}, which=lambda _n: "nvidia-smi")
        assert not ok

    @pytest.mark.parametrize("kind,extra", [("cuda", "cuda"), ("mps", "mps"), ("cpu", "cpu")])
    def test_extra_mapping(self, kind, extra):
        assert bs.torch_extra_for(bs.Accelerator(kind, "test")) == extra

    def test_extra_mapping_is_defensive(self):
        assert bs.torch_extra_for("xpu") == "cpu"
        assert bs.torch_extra_for("rocm") == "rocm"

    def test_index_urls_match_pyproject_comments(self):
        assert bs.torch_index_url("cuda") == "https://download.pytorch.org/whl/cu128"
        assert bs.torch_index_url("cpu") == "https://download.pytorch.org/whl/cpu"
        assert bs.torch_index_url("mps") is None
        assert bs.torch_index_url("rocm") == "https://download.pytorch.org/whl/rocm6.4"

    def test_every_accelerator_kind_is_an_extra_in_pyproject(self):
        """The Advanced combo offers ACCELERATOR_KINDS; each must resolve to a
        real extra or the install fails at uv with 'no such extra'."""
        from pathlib import Path

        text = (Path(bs.daemon_source()) / "pyproject.toml").read_text(encoding="utf-8")
        for kind in bs.ACCELERATOR_KINDS:
            assert ("\n%s = [" % kind) in text, kind
            assert kind in bs.TORCH_INDEX_URLS, kind


class TestRocmDetection:
    """AMD via ROCm: Linux x86_64 only, and /dev/kfd alone is not evidence."""

    def test_kfd_plus_opt_rocm_is_rocm(self):
        accel = bs.detect_accelerator(
            platform_name="Linux", machine="x86_64", env={},
            which=lambda _n: None,
            exists=lambda p: p in ("/dev/kfd", "/opt/rocm"),
        )
        assert accel.kind == "rocm" and accel.is_gpu
        assert bs.torch_extra_for(accel) == "rocm"

    def test_kfd_plus_rocminfo_on_path_is_rocm(self):
        accel = bs.detect_accelerator(
            platform_name="Linux", machine="x86_64", env={},
            which=lambda n: "/usr/bin/rocminfo" if n == "rocminfo" else None,
            exists=lambda p: p == "/dev/kfd",
        )
        assert accel.kind == "rocm"
        assert "rocminfo" in accel.detail["source"]

    def test_kfd_alone_is_cpu(self):
        """Any amdgpu-driven display has /dev/kfd; that is not a ROCm runtime."""
        accel = bs.detect_accelerator(
            platform_name="Linux", machine="x86_64", env={},
            which=lambda _n: None, exists=lambda p: p == "/dev/kfd",
        )
        assert accel.kind == "cpu"

    def test_nvidia_wins_over_rocm(self):
        accel = bs.detect_accelerator(
            platform_name="Linux", machine="x86_64", env={},
            which=lambda _n: None,
            exists=lambda p: p in ("/dev/kfd", "/opt/rocm", "/proc/driver/nvidia/version"),
        )
        assert accel.kind == "cuda"

    def test_rocm_is_never_chosen_off_linux(self):
        ok, _ = bs.detect_rocm(platform_name="Windows", env={},
                               which=lambda _n: "rocminfo", exists=lambda _p: True)
        assert not ok

    def test_rocm_is_never_chosen_on_aarch64(self):
        """PyTorch ships ROCm wheels for x86_64 only."""
        accel = bs.detect_accelerator(
            platform_name="Linux", machine="aarch64", env={},
            which=lambda _n: None, exists=lambda p: p in ("/dev/kfd", "/opt/rocm"),
        )
        assert accel.kind == "cpu"

    def test_force_device_accepts_rocm(self):
        accel = bs.detect_accelerator(
            platform_name="Linux", machine="x86_64", env={"SAM3_FORCE_DEVICE": "rocm"},
            which=lambda _n: None, exists=lambda _p: False,
        )
        assert accel.kind == "rocm"

    def test_assume_rocm_override(self):
        ok, detail = bs.detect_rocm(env={"SAM3_ASSUME_ROCM": "1"}, platform_name="Windows")
        assert ok and detail["source"] == "SAM3_ASSUME_ROCM"

    def test_rocm_plan_uses_the_rocm_index(self, sam3_home):
        plan = bs.build_install_plan(
            accelerator=bs.Accelerator("rocm", "test"), platform_name="Linux", machine="x86_64",
        )
        assert plan.extra == "rocm"
        assert plan.index_url == "https://download.pytorch.org/whl/rocm6.4"
        torch = plan.step("torch").argv
        assert torch[torch.index("--index-url") + 1] == "https://download.pytorch.org/whl/rocm6.4"
        assert "--extra-index-url" not in torch and "--index-strategy" not in torch
        assert any(a.endswith("[runtime]") for a in plan.step("sam3gimpd").argv)


class TestPlatformBlockers:
    """Hosts with no torch wheel are refused up front, not twenty minutes in."""

    def test_intel_mac_is_blocked(self):
        msg = bs.platform_blocker("Darwin", "x86_64")
        assert msg and "Intel" in msg

    def test_apple_silicon_and_pcs_are_not_blocked(self):
        assert bs.platform_blocker("Darwin", "arm64") is None
        assert bs.platform_blocker("Windows", "AMD64") is None
        assert bs.platform_blocker("Linux", "x86_64") is None
        assert bs.platform_blocker("Linux", "aarch64") is None

    def test_32_bit_is_blocked(self):
        assert bs.platform_blocker("Windows", "x86")
        assert bs.platform_blocker("Linux", "armv7l")

    def test_build_install_plan_refuses_a_blocked_host(self, sam3_home):
        with pytest.raises(bs.UnsupportedPlatform) as info:
            bs.build_install_plan(platform_name="Darwin", machine="x86_64",
                                  accelerator=bs.Accelerator("cpu", "test"))
        assert isinstance(info.value, bs.BootstrapError)
        assert "Intel Mac" in str(info.value)


class TestDriverWarning:
    """cu128 wheels need a 2023-or-newer driver; say so before the 3 GB download."""

    def test_driver_version_is_read_from_nvidia_smi(self):
        runner = FakeRunner({"nvidia-smi": (0, ["576.02"])})
        assert bs.nvidia_driver_version(runner=runner, smi="nvidia-smi") == "576.02"

    def test_a_failing_query_is_none(self):
        runner = FakeRunner({"nvidia-smi": (1, ["nope"])})
        assert bs.nvidia_driver_version(runner=runner, smi="nvidia-smi") is None

    def test_detect_nvidia_records_the_driver(self):
        class Runner:
            def run(self, argv, timeout=None):
                if "-L" in argv:
                    return bs.CommandResult(argv, 0, ["GPU 0: NVIDIA GeForce RTX 2080 Ti"], 0.1)
                return bs.CommandResult(argv, 0, ["456.71"], 0.1)
        ok, detail = bs.detect_nvidia(runner=Runner(), which=lambda _n: "nvidia-smi", env={})
        assert ok and detail["driver_version"] == "456.71"

    @pytest.mark.parametrize("platform,driver,warns", [
        ("windows", "456.71", True), ("windows", "528.33", False), ("windows", "576.02", False),
        ("linux", "470.199", True), ("linux", "525.60", False), ("linux", "550.54.14", False),
    ])
    def test_old_drivers_warn(self, platform, driver, warns):
        accel = bs.Accelerator("cuda", "t", {"driver_version": driver, "platform": platform})
        msg = bs.cuda_driver_warning(accel)
        assert bool(msg) == warns, (platform, driver, msg)
        if warns:
            assert driver in msg

    def test_no_driver_info_means_no_warning(self):
        assert bs.cuda_driver_warning(bs.Accelerator("cuda", "t", {})) is None
        assert bs.cuda_driver_warning(bs.Accelerator("cpu", "t", {"driver_version": "1.0"})) is None

    def test_the_plan_carries_the_warning(self, sam3_home):
        accel = bs.Accelerator("cuda", "t", {"driver_version": "456.71", "platform": "windows"})
        plan = bs.build_install_plan(accelerator=accel, platform_name="Windows", machine="AMD64")
        assert plan.warnings and "456.71" in plan.warnings[0]
        assert plan.describe()["warnings"] == plan.warnings


class TestDaemonIdentity:
    """The plug-in knows exactly which daemon source it ships."""

    def test_bundled_version_matches_the_package(self):
        import sam3gimpd
        assert bs.bundled_daemon_version() == sam3gimpd.__version__

    def test_bundled_build_matches_the_daemons_own_hash(self):
        """Both sides must hash the same files the same way, or the dialog
        would report a stale daemon for ever."""
        import sam3gimpd
        assert bs.bundled_daemon_build() == sam3gimpd.build_hash()
        assert len(bs.bundled_daemon_build()) == 8

    def test_update_command_prefers_the_users_interpreter(self, sam3_home, tmp_path, monkeypatch):
        python = tmp_path / "python.exe"
        python.write_text("")
        import launcher as L
        L.set_configured_python(str(python), has_daemon=True)
        monkeypatch.setattr(bs, "find_uv", lambda: None)
        monkeypatch.setattr(bs, "interpreter_has_pip", lambda *a, **k: True)
        argv = bs.daemon_update_command()
        assert argv[:4] == [str(python), "-m", "pip", "install"]
        assert any(a.endswith("[runtime]") for a in argv), "torch must be left alone"

    def test_update_command_uses_uv_for_the_users_interpreter_when_there_is_one(
            self, sam3_home, tmp_path, monkeypatch):
        """A venv uv created has no pip: `python -m pip` there fails outright."""
        python = tmp_path / "python.exe"
        python.write_text("")
        import launcher as L
        L.set_configured_python(str(python), has_daemon=True)
        monkeypatch.setattr(bs, "find_uv", lambda: "/tools/uv")
        argv = bs.daemon_update_command()
        assert argv[:5] == ["/tools/uv", "pip", "install", "--python", str(python)]
        assert "--reinstall-package" in argv and bs.DAEMON_DIST_NAME in argv
        assert "--upgrade" not in argv, "nothing else in their environment is upgraded"
        assert argv[-1].endswith("[runtime]")

    def test_update_command_uses_uv_for_the_managed_venv(self, sam3_home):
        os.makedirs(bs.venv_bin_dir(), exist_ok=True)
        os.makedirs(bs.tools_dir(), exist_ok=True)
        Path(bs.venv_python()).write_text("")
        Path(bs.uv_binary()).write_text("")
        bs.save_state(bs.InstallState(extra="cuda"))
        argv = bs.daemon_update_command()
        assert argv[:3] == [bs.uv_binary(), "pip", "install"]
        assert "--reinstall-package" in argv and bs.DAEMON_DIST_NAME in argv
        assert "--upgrade" not in argv, "torch must be left alone"
        # torch is its own step; the daemon never names it or its index.
        assert argv[-1].endswith("[runtime]")
        assert not any("download.pytorch.org" in a for a in argv)
        assert "--extra-index-url" not in argv and "--index-strategy" not in argv

    def test_update_command_is_none_with_nothing_installed(self, sam3_home):
        assert bs.daemon_update_command() is None


# --------------------------------------------------------------------------- #
# paths -- must mirror _daemon/sam3gimpd/paths.py exactly
# --------------------------------------------------------------------------- #
class TestPaths:
    def test_home_override_wins(self, sam3_home):
        assert bs.base_dir() == str(sam3_home)
        assert bs.runtime_file() == os.path.join(str(sam3_home), "runtime.json")
        assert bs.venv_root() == os.path.join(str(sam3_home), "venv")
        assert bs.uv_binary().startswith(os.path.join(str(sam3_home), "tools"))

    def test_runtime_file_override(self, sam3_home, monkeypatch, tmp_path):
        target = tmp_path / "elsewhere.json"
        monkeypatch.setenv(bs.ENV_RUNTIME_FILE, str(target))
        assert bs.runtime_file() == str(target)

    def test_windows_layout(self, monkeypatch, tmp_path):
        monkeypatch.delenv(bs.ENV_HOME, raising=False)
        monkeypatch.setattr(bs, "platform_key", lambda name=None: "windows")
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
        assert bs.base_dir() == str(tmp_path / "Local" / "sam3-gimp")
        assert bs.venv_bin_dir().endswith("Scripts")
        assert bs.venv_python().endswith("python.exe")
        assert bs.venv_pythonw().endswith("pythonw.exe")
        assert bs.sam3d_executable().endswith("sam3gimpd.exe")
        assert bs.uv_binary().endswith("uv.exe")

    def test_posix_layout(self, sam3_home, monkeypatch):
        monkeypatch.setattr(bs, "platform_key", lambda name=None: "linux")
        assert bs.venv_bin_dir().endswith("bin")
        assert bs.venv_python().endswith(os.path.join("bin", "python"))
        # DESIGN.md §4: pythonw only exists on Windows; elsewhere it is python.
        assert bs.venv_pythonw() == bs.venv_python()

    def test_xdg_default(self, monkeypatch, tmp_path):
        monkeypatch.delenv(bs.ENV_HOME, raising=False)
        monkeypatch.setattr(bs, "platform_key", lambda name=None: "linux")
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
        assert bs.base_dir() == str(tmp_path / "share" / "sam3-gimp")

    def test_describe_paths_covers_every_location(self, sam3_home):
        d = bs.describe_paths()
        for key in ("base", "runtime_file", "venv_python", "sam3d_executable",
                    "uv_binary", "hf_home", "crash_log", "state_file"):
            assert key in d and d[key]

    def test_ensure_layout_is_idempotent(self, sam3_home):
        first = bs.ensure_layout()
        second = bs.ensure_layout()
        assert first == second
        for path in first.values():
            assert os.path.isdir(path)

    @pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
    def test_layout_is_private_to_the_user(self, sam3_home):
        """hf/ holds the HuggingFace login token; nothing here is anyone
        else's business.  A directory an older version left 0755 is narrowed."""
        os.makedirs(bs.hf_home(), mode=0o755, exist_ok=True)
        os.chmod(bs.hf_home(), 0o755)
        for name, path in bs.ensure_layout().items():
            if name == "base":
                continue  # an existing SAM3_GIMP_HOME is the user's to manage
            assert stat.S_IMODE(os.stat(path).st_mode) == 0o700, name

    @pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
    def test_a_new_base_directory_is_created_private(self, tmp_path, monkeypatch):
        home = tmp_path / "fresh" / "sam3-gimp"
        monkeypatch.setenv("SAM3_GIMP_HOME", str(home))
        bs.ensure_layout()
        assert stat.S_IMODE(os.stat(str(home)).st_mode) == 0o700

    @pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
    def test_files_written_under_base_are_0600(self, sam3_home, tmp_path):
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "config.json").write_text("{}")
        (ckpt / "model.safetensors").write_bytes(b"")
        bs.save_state(bs.InstallState(fingerprint="x"))
        bs.set_local_weights(str(ckpt))
        bs._remember_daemon_source(str(tmp_path))
        for path in (bs.state_file(), bs.weights_config_file(),
                     os.path.join(bs.base_dir(), "settings.json")):
            assert stat.S_IMODE(os.stat(path).st_mode) == 0o600, path

    def test_paths_agree_with_sam3d_paths(self, sam3_home):
        """The plug-in mirrors ``sam3gimpd.paths`` by hand (API.md §16.10).  If the
        daemon package happens to be importable here, prove the mirror is exact."""
        sam3d_paths = pytest.importorskip("sam3gimpd.paths")
        assert bs.base_dir() == str(sam3d_paths.base_dir())
        assert bs.runtime_file() == str(sam3d_paths.runtime_file())
        assert bs.lock_file() == str(sam3d_paths.lock_file())
        assert bs.server_log() == str(sam3d_paths.server_log())
        assert bs.crash_log() == str(sam3d_paths.crash_log())
        assert bs.hf_home() == str(sam3d_paths.hf_home())
        assert bs.models_dir() == str(sam3d_paths.models_dir())
        assert bs.venv_root() == str(sam3d_paths.venv_root())
        assert bs.venv_python() == str(sam3d_paths.venv_python())
        assert bs.venv_pythonw() == str(sam3d_paths.venv_pythonw())
        assert bs.sam3d_executable() == str(sam3d_paths.sam3d_executable())
        assert bs.uv_binary() == str(sam3d_paths.uv_binary())
        assert bs.cache_dir() == str(sam3d_paths.cache_dir())
        assert bs.tools_dir() == str(sam3d_paths.tools_dir())


# --------------------------------------------------------------------------- #
# uv release assets
# --------------------------------------------------------------------------- #
class TestUvAssets:
    @pytest.mark.parametrize(
        "plat,mach,asset",
        [
            ("Windows", "AMD64", "uv-x86_64-pc-windows-msvc.zip"),
            ("Windows", "arm64", "uv-aarch64-pc-windows-msvc.zip"),
            ("Linux", "x86_64", "uv-x86_64-unknown-linux-gnu.tar.gz"),
            ("Linux", "aarch64", "uv-aarch64-unknown-linux-gnu.tar.gz"),
            ("Darwin", "arm64", "uv-aarch64-apple-darwin.tar.gz"),
            ("Darwin", "x86_64", "uv-x86_64-apple-darwin.tar.gz"),
        ],
    )
    def test_asset_names(self, plat, mach, asset):
        assert bs.uv_asset_name(plat, mach) == asset

    def test_download_url_is_pinned(self):
        url = bs.uv_download_url("0.9.7", "Windows", "AMD64")
        assert url == (
            "https://github.com/astral-sh/uv/releases/download/0.9.7/"
            "uv-x86_64-pc-windows-msvc.zip"
        )
        assert bs.uv_checksum_url("0.9.7", "Windows", "AMD64") == url + ".sha256"

    def test_unsupported_host_fails_loudly(self, sam3_home):
        with pytest.raises(bs.BootstrapError) as exc:
            bs.uv_asset_name("Linux", "sparc64")
        assert "no uv release" in str(exc.value)

    def test_every_asset_the_code_can_name_has_a_pinned_hash(self):
        """The expected hash must come from here, never from the release that
        serves the archive -- that could only ever catch corruption."""
        for (plat, mach) in bs._UV_TRIPLES:
            asset = bs.uv_asset_name(plat, mach)
            digest = bs.UV_SHA256.get(asset)
            assert digest and len(digest) == 64 and int(digest, 16) >= 0, asset
            assert bs.uv_expected_sha256(bs.uv_download_url(bs.UV_VERSION, plat, mach)) == digest

    def test_another_version_has_no_hash_to_check_against(self):
        assert bs.uv_expected_sha256(bs.uv_download_url("0.1.0", "Linux", "x86_64")) is None
        assert bs.uv_expected_sha256("https://example.invalid/uv.tar.gz") is None

    def test_install_script_pins_the_same_windows_hashes(self):
        """tools/install.ps1 downloads the same two assets and must check them
        against the same values."""
        script = (Path(__file__).resolve().parents[2] / "tools" / "install.ps1").read_text(encoding="utf-8")
        for mach in ("x86_64", "aarch64"):
            asset = bs.uv_asset_name("Windows", mach)
            assert "'%s' = '%s'" % (asset, bs.UV_SHA256[asset]) in script.replace("  =", " =")
        assert "Get-FileHash -LiteralPath $zip -Algorithm SHA256" in script


# --------------------------------------------------------------------------- #
# command construction
# --------------------------------------------------------------------------- #
def _daemon_cli_parser():
    """The daemon's real argparse parser.  Commands built here are checked
    against it, not against what we believe its flags are: ``doctor --json``
    was an assumption, argparse rejected it, and Doctor never once worked."""
    from sam3gimpd import cli
    return cli.build_parser()


def _daemon_args(argv):
    """argv after the program: ``<exe> <sub>...`` or ``<py> -m sam3gimpd <sub>...``."""
    argv = list(argv)
    if argv[1:3] == ["-m", "sam3gimpd"]:
        return argv[3:]
    return argv[1:]


class TestCommandConstruction:
    def test_venv_command_pins_python_and_allows_download(self):
        argv = bs.uv_venv_command("/t/uv", "/b/venv", "3.11")
        assert argv == ["/t/uv", "venv", "--clear", "--python", "3.11",
                        "--python-preference", "managed", "/b/venv"]

    def test_venv_command_replaces_a_broken_venv(self):
        """uv refuses to create a venv over a directory with no pyvenv.cfg --
        exactly what a half-created or antivirus-damaged venv is -- so Repair
        could never rebuild one without --clear."""
        assert "--clear" in bs.uv_venv_command("/t/uv", "/b/venv")

    def test_torch_command_uses_the_pytorch_index_alone(self):
        argv = bs.torch_install_command("/t/uv", "/b/venv/bin/python",
                                        index_url=bs.torch_index_url("cuda"))
        assert argv == [
            "/t/uv", "pip", "install", "--python", "/b/venv/bin/python",
            "--index-url", "https://download.pytorch.org/whl/cu128",
            "--reinstall-package", "torch", "--reinstall-package", "torchvision",
            "torch==2.9.0", "torchvision==0.24.0",
        ]

    @pytest.mark.parametrize("extra", ["cuda", "rocm", "cpu", "mps"])
    def test_no_command_mixes_indexes(self, sam3_home, extra):
        """``--extra-index-url`` with ``unsafe-best-match`` lets any package
        come from whichever index has the best version -- uv documents that
        as open to dependency confusion.  torch comes from the PyTorch index
        alone; the daemon's dependencies from PyPI alone."""
        plan = bs.build_install_plan(accelerator=bs.Accelerator(extra, "t"),
                                     platform_name="Linux", machine="x86_64", env={})
        for step in plan.steps:
            assert "--extra-index-url" not in step.argv, step.key
            assert "--index-strategy" not in step.argv, step.key
            assert not any("unsafe-best-match" in a for a in step.argv), step.key
        torch = plan.step("torch").argv
        if extra == "mps":
            assert "--index-url" not in torch, "Apple Silicon wheels are on PyPI"
        else:
            assert torch[torch.index("--index-url") + 1] == bs.torch_index_url(extra)
        daemon = plan.step("sam3gimpd").argv
        assert "--index-url" not in daemon
        assert not any(a.startswith("torch") for a in daemon), "the daemon step installs no torch"

    def test_torch_pins_match_every_accelerator_extra(self):
        """The torch step installs what the extras would have: same pins."""
        import re as _re

        text = (Path(bs.daemon_source()) / "pyproject.toml").read_text(encoding="utf-8")
        for kind in bs.ACCELERATOR_KINDS:
            block = text.split("\n%s = [" % kind)[1].split("]")[0]
            pins = sorted(_re.findall(r'"(torch(?:vision)?==[^"]+)"', block))
            assert pins == sorted(bs.TORCH_REQUIREMENTS), kind

    def test_install_script_pins_the_same_torch(self):
        script = (Path(__file__).resolve().parents[2] / "tools" / "install.ps1").read_text(encoding="utf-8")
        assert "$TorchRequirements = @(%s)" % ", ".join(
            "'%s'" % r for r in bs.TORCH_REQUIREMENTS) in script

    def test_install_script_never_mixes_indexes_in_a_command(self):
        script = (Path(__file__).resolve().parents[2] / "tools" / "install.ps1").read_text(encoding="utf-8")
        code = [ln for ln in script.splitlines() if not ln.lstrip().startswith("#")]
        assert not any("unsafe-best-match" in ln or "--extra-index-url" in ln for ln in code)
        assert "--index-url $index" in script
        assert "$DaemonDir[runtime]" in script

    def test_install_command_is_the_runtime_extra_from_a_path(self):
        argv = bs.uv_install_command("/t/uv", "/b/venv/bin/python", source="/src/daemon")
        assert argv == ["/t/uv", "pip", "install", "--python", "/b/venv/bin/python",
                        "/src/daemon[runtime]"]

    def test_install_command_defaults_to_the_bundled_daemon_path(self):
        """Never a bare requirement name -- ``sam3d`` on PyPI is someone else's
        project, and installing it silently was a real, shipped bug."""
        argv = bs.uv_install_command("/t/uv", "/p")
        requirement = [a for a in argv if a.endswith("[runtime]")][0]
        source = requirement[: -len("[runtime]")]
        assert os.path.isabs(source), source
        assert os.path.isfile(os.path.join(source, "pyproject.toml"))

    def test_install_command_targets_the_venv_not_the_shell(self):
        argv = bs.uv_install_command("/t/uv", "/b/venv/bin/python")
        assert argv[argv.index("--python") + 1] == "/b/venv/bin/python"

    def test_install_command_local_source_for_dev_loop(self):
        argv = bs.uv_install_command("/t/uv", "/p", source="/repo/daemon")
        assert "/repo/daemon[runtime]" in argv

    def test_install_command_flags(self):
        argv = bs.uv_install_command("/t/uv", "/p", upgrade=True, offline=True,
                                     reinstall=[bs.DAEMON_DIST_NAME])
        assert "--upgrade" in argv and "--offline" in argv
        assert argv[argv.index("--reinstall-package") + 1] == bs.DAEMON_DIST_NAME

    def test_verify_command_uses_the_venv_interpreter(self):
        argv = bs.verify_command("/b/venv/bin/python")
        assert argv[0] == "/b/venv/bin/python" and argv[1] == "-c"
        assert "import sam3gimpd" in argv[2]

    def test_verify_snippet_actually_runs(self):
        """The snippet is a string shipped to another interpreter -- compile it
        here so a syntax error cannot reach a user's first install."""
        compile(bs.VERIFY_SNIPPET, "<verify>", "exec")

    def test_verify_fails_without_torch_or_transformers(self, tmp_path):
        """It passed with neither: the ImportError was caught and it exited 0,
        so "Environment ready." was shown over an environment that could not
        load SAM 3.  Run it for real in an interpreter that has sam3gimpd
        but (here) no torch and no transformers."""
        daemon_root = Path(bs.daemon_source())
        env = dict(os.environ)
        blocker = tmp_path / "block"
        blocker.mkdir()
        # Shadow whatever this interpreter has, so the test means the same
        # thing on a machine that does have torch installed.
        for name in ("torch", "transformers"):
            (blocker / (name + ".py")).write_text("raise ImportError('no %s here')\n" % name)
        env["PYTHONPATH"] = os.pathsep.join([str(blocker), str(daemon_root)])
        proc = subprocess.run([sys.executable, "-c", bs.VERIFY_SNIPPET], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True, timeout=60)
        assert proc.returncode == 1, proc.stdout
        assert "torch" in proc.stdout and "transformers" in proc.stdout
        report = bs._parse_verify(proc.stdout)
        assert report["torch"] is None and report["transformers"] is None

    def test_hf_login_snippet_compiles_and_reads_the_env(self):
        compile(bs.HF_LOGIN_SNIPPET, "<login>", "exec")
        argv = bs.hf_login_command("/b/venv/bin/python")
        # The token must never appear in argv: `ps` is world-readable.
        assert not any("hf_" in a for a in argv)
        assert "HF_TOKEN" in argv[2]

    def test_doctor_command(self):
        assert bs.doctor_command("/b/venv/bin/sam3gimpd") == ["/b/venv/bin/sam3gimpd", "doctor"]

    def test_doctor_command_is_accepted_by_the_real_cli(self):
        args = _daemon_cli_parser().parse_args(_daemon_args(bs.doctor_command("/x")))
        assert args.command == "doctor"

    def test_weights_download_command(self):
        assert bs.weights_download_command("/py") == [
            "/py", "-m", "sam3gimpd", "download", "--repo-id", "facebook/sam3"]

    def test_weights_download_command_is_accepted_by_the_real_cli(self):
        args = _daemon_cli_parser().parse_args(_daemon_args(bs.weights_download_command("/py")))
        assert args.command == "download" and args.repo_id == "facebook/sam3"


# --------------------------------------------------------------------------- #
# the install plan
# --------------------------------------------------------------------------- #
class TestInstallPlan:
    def test_plan_order_and_keys(self, sam3_home):
        plan = bs.build_install_plan(accelerator=bs.Accelerator("cpu", "test"), env={})
        assert plan.keys == ["uv", "venv", "torch", "sam3gimpd", "verify"]

    def test_plan_with_weights(self, sam3_home):
        plan = bs.build_install_plan(
            accelerator=bs.Accelerator("cuda", "test"), with_weights=True, env={}
        )
        assert plan.keys == ["uv", "venv", "torch", "sam3gimpd", "verify", "weights"]
        assert plan.step("weights").argv[-1] == bs.SAM3_REPO_ID
        assert plan.step("weights").argv[:3] == [bs.venv_python(), "-m", "sam3gimpd"]

    def test_plan_uses_the_detected_accelerator(self, sam3_home):
        plan = bs.build_install_plan(accelerator=bs.Accelerator("cuda", "rtx"), env={})
        assert plan.extra == "cuda"
        assert plan.index_url == "https://download.pytorch.org/whl/cu128"
        assert "https://download.pytorch.org/whl/cu128" in plan.step("torch").argv
        assert bs.daemon_source() + "[runtime]" in plan.step("sam3gimpd").argv

    def test_plan_points_every_command_at_our_base_dir(self, sam3_home):
        plan = bs.build_install_plan(accelerator=bs.Accelerator("cpu", "t"), env={})
        assert plan.step("uv").dest == bs.uv_binary()
        assert bs.venv_root() in plan.step("venv").argv
        assert bs.venv_python() in plan.step("torch").argv
        assert bs.venv_python() in plan.step("sam3gimpd").argv
        assert plan.step("verify").argv[0] == bs.venv_python()

    def test_plan_source_from_env(self, sam3_home):
        plan = bs.build_install_plan(
            accelerator=bs.Accelerator("cpu", "t"), env={"SAM3D_SOURCE": "/repo/daemon"}
        )
        assert "/repo/daemon[runtime]" in plan.step("sam3gimpd").argv

    def test_reinstall_reruns_torch_and_the_daemon(self, sam3_home):
        plan = bs.build_install_plan(accelerator=bs.Accelerator("cpu", "t"), upgrade=True, env={})
        assert plan.step("torch").satisfied is None
        assert plan.step("sam3gimpd").satisfied is None
        daemon = plan.step("sam3gimpd").argv
        assert daemon[daemon.index("--reinstall-package") + 1] == bs.DAEMON_DIST_NAME

    def test_fingerprint_changes_with_the_decisions(self, sam3_home):
        cpu = bs.build_install_plan(accelerator=bs.Accelerator("cpu", "t"), env={})
        cuda = bs.build_install_plan(accelerator=bs.Accelerator("cuda", "t"), env={})
        assert bs.plan_fingerprint(cpu) != bs.plan_fingerprint(cuda)
        again = bs.build_install_plan(accelerator=bs.Accelerator("cpu", "t"), env={})
        assert bs.plan_fingerprint(cpu) == bs.plan_fingerprint(again)

    def test_describe_is_json_serialisable_for_the_dialog(self, sam3_home):
        plan = bs.build_install_plan(accelerator=bs.Accelerator("cuda", "t"), env={})
        json.dumps(plan.describe())


# --------------------------------------------------------------------------- #
# idempotency and resumability
# --------------------------------------------------------------------------- #
class TestResumability:
    def _plan(self):
        return bs.build_install_plan(accelerator=bs.Accelerator("cpu", "t"), env={})

    @staticmethod
    def _touch(*paths):
        for p in paths:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            Path(p).write_text("")

    def test_everything_pending_on_a_clean_box(self, sam3_home):
        plan = self._plan()
        assert [s.key for s in plan.pending(bs.load_state(plan))] == [
            "uv", "venv", "torch", "sam3gimpd", "verify"]

    def test_filesystem_probe_skips_finished_steps_without_a_journal(self, sam3_home):
        plan = self._plan()
        os.makedirs(os.path.dirname(bs.uv_binary()), exist_ok=True)
        Path(bs.uv_binary()).write_text("#!/bin/sh\n")
        assert [s.key for s in plan.pending(bs.load_state(plan))] == [
            "venv", "torch", "sam3gimpd", "verify"]

    def test_journal_and_files_together_resume_after_gimp_was_closed(self, sam3_home):
        plan = self._plan()
        self._touch(bs.uv_binary(), bs.venv_python())
        state = bs.load_state(plan)
        state.fingerprint = bs.plan_fingerprint(plan)
        state.mark("uv")
        state.mark("venv")
        bs.save_state(state)

        reloaded = bs.load_state(plan)
        assert reloaded.completed == ["uv", "venv"]
        assert [s.key for s in plan.pending(reloaded)] == ["torch", "sam3gimpd", "verify"]

    def test_the_journal_cannot_vouch_for_files_that_are_gone(self, sam3_home):
        """A journal said "done" and the step was skipped although antivirus
        (or the user) had removed the venv: Setup showed "Not installed" next
        to a greyed-out button, and Install returned ok after running
        nothing.  The disk decides."""
        plan = self._plan()
        state = bs.InstallState(fingerprint=bs.plan_fingerprint(plan),
                                completed=["uv", "venv", "torch", "sam3gimpd", "verify"])
        bs.save_state(state)
        self._touch(bs.uv_binary())  # uv survived; the venv did not

        assert [s.key for s in plan.pending(bs.load_state(plan))] == [
            "venv", "torch", "sam3gimpd", "verify"]

        runner = MakesFilesRunner()
        outcome = bs.Installer(plan, runner=runner, downloader=FakeDownloader({})).run()
        assert outcome.ok, outcome.error
        assert outcome.completed == ["venv", "torch", "sam3gimpd", "verify"]
        assert runner.argvs[0][:2] == [bs.uv_binary(), "venv"]

    def test_journal_from_a_different_plan_is_discarded(self, sam3_home):
        """A cpu journal must not let a cuda install skip the sam3gimpd step -- that
        would leave a half-CPU/half-CUDA venv."""
        cpu = self._plan()
        state = bs.load_state(cpu)
        state.fingerprint = bs.plan_fingerprint(cpu)
        state.mark("uv")
        state.mark("venv")
        state.mark("sam3gimpd")
        bs.save_state(state)

        cuda = bs.build_install_plan(accelerator=bs.Accelerator("cuda", "t"), env={})
        assert bs.load_state(cuda).completed == []

    def test_switching_cpu_to_cuda_reinstalls_torch(self, sam3_home):
        """The journal was discarded -- and then every step was skipped anyway,
        because the only probe asked whether the daemon's console script
        existed, which says nothing about which torch sits beside it.  The CPU
        build stayed on a machine with a GPU."""
        cpu = self._plan()
        runner = MakesFilesRunner()
        self._touch(bs.uv_binary())
        assert bs.Installer(cpu, runner=runner, downloader=FakeDownloader({})).run().ok
        assert bs.installed_torch_extra() == "cpu"

        cuda = bs.build_install_plan(accelerator=bs.Accelerator("cuda", "t"), env={})
        assert [s.key for s in cuda.pending(bs.load_state(cuda))] == ["torch", "verify"]

        runner2 = MakesFilesRunner()
        outcome = bs.Installer(cuda, runner=runner2, downloader=FakeDownloader({})).run()
        assert outcome.ok and outcome.completed == ["torch", "verify"]
        torch = runner2.argvs[0]
        assert torch[torch.index("--index-url") + 1] == "https://download.pytorch.org/whl/cu128"
        # 2.9.0+cpu satisfies torch==2.9.0: without this uv would keep it.
        assert torch[torch.index("--reinstall-package") + 1] == "torch"
        assert bs.installed_torch_extra() == "cuda"

    def test_a_venv_from_before_the_torch_record_is_recognised(self, sam3_home):
        """A venv built when torch came in with the daemon's own extra has no
        torch record; its journal says which extra it was, and that is kept
        rather than re-downloading 2.5 GB."""
        self._touch(bs.uv_binary(), bs.venv_python(), bs.sam3d_executable())
        bs.save_state(bs.InstallState(fingerprint="an-old-plan", extra="cpu",
                                      completed=["uv", "venv", "sam3gimpd", "verify"]))
        plan = self._plan()
        assert bs.installed_torch_extra() == "cpu"
        runner = MakesFilesRunner()
        outcome = bs.Installer(plan, runner=runner, downloader=FakeDownloader({})).run()
        assert outcome.ok and "torch" in outcome.skipped
        assert bs._read_json(bs.torch_marker_file())["extra"] == "cpu"

    def test_a_pin_bump_reinstalls_torch(self, sam3_home, monkeypatch):
        self._touch(bs.venv_python())
        bs._write_torch_marker("cpu", bs.torch_index_url("cpu"))
        assert bs.installed_torch_extra() == "cpu"
        monkeypatch.setattr(bs, "TORCH_REQUIREMENTS", ("torch==9.9.9", "torchvision==9.9.9"))
        assert bs.installed_torch_extra() is None

    def test_journal_with_a_future_format_version_is_discarded(self, sam3_home):
        plan = self._plan()
        state = bs.load_state(plan)
        state.fingerprint = bs.plan_fingerprint(plan)
        state.version = bs.STATE_VERSION + 99
        state.mark("uv")
        bs.save_state(state)
        assert bs.load_state(plan).completed == []

    def test_corrupt_journal_is_survivable(self, sam3_home):
        Path(bs.state_file()).write_text("{not json")
        assert bs.load_state().completed == []

    def test_verify_step_always_reruns(self, sam3_home):
        """Checking only is_satisfied() missed that the Installer skipped any
        journalled step -- verify included, so it never ran twice."""
        plan = self._plan()
        assert plan.step("verify").is_satisfied() is False
        self._touch(bs.uv_binary())
        assert bs.Installer(plan, runner=MakesFilesRunner(), downloader=FakeDownloader({})).run().ok
        assert "verify" in bs.load_state(plan).completed

        runner = MakesFilesRunner()
        outcome = bs.Installer(plan, runner=runner, downloader=FakeDownloader({})).run()
        assert outcome.ok and outcome.completed == ["verify"]
        assert runner.argvs == [bs.verify_command(bs.venv_python())]

    def test_clear_state(self, sam3_home):
        bs.save_state(bs.InstallState(fingerprint="x", completed=["uv"]))
        assert os.path.exists(bs.state_file())
        bs.clear_state()
        assert not os.path.exists(bs.state_file())
        bs.clear_state()  # idempotent


# --------------------------------------------------------------------------- #
# the installer, driven with fakes
# --------------------------------------------------------------------------- #
class TestInstaller:
    @pytest.fixture(autouse=True)
    def _pinned(self, pinned_fake_uv):
        return pinned_fake_uv

    def _payloads(self, plan):
        return {plan.step("uv").url: FAKE_UV_ARCHIVE}

    def test_full_run_issues_exactly_the_expected_commands(self, sam3_home):
        plan = bs.build_install_plan(accelerator=bs.Accelerator("cuda", "t"), env={})
        runner = FakeRunner({
            " ".join(bs.verify_command(bs.venv_python())):
                (0, ['SAM3D_VERIFY {"sam3d_version": "0.1.0", "torch": "2.9.0"}']),
        })
        # The venv/sam3gimpd steps' filesystem probes must not skip them.
        installer = bs.Installer(
            plan, runner=runner, downloader=FakeDownloader(self._payloads(plan))
        )
        outcome = installer.run()

        assert outcome.ok, outcome.error
        assert outcome.completed == ["uv", "venv", "torch", "sam3gimpd", "verify"]
        assert runner.argvs[0] == bs.uv_venv_command(bs.uv_binary(), bs.venv_root(), "3.11")
        assert runner.argvs[1] == bs.torch_install_command(
            bs.uv_binary(), bs.venv_python(), index_url="https://download.pytorch.org/whl/cu128")
        assert runner.argvs[2] == bs.uv_install_command(bs.uv_binary(), bs.venv_python())
        assert runner.argvs[3] == bs.verify_command(bs.venv_python())
        assert outcome.verify == {"sam3d_version": "0.1.0", "torch": "2.9.0"}
        # The torch step records which accelerator it installed for.
        assert bs._read_json(bs.torch_marker_file())["extra"] == "cuda"

    def test_uv_is_downloaded_unpacked_and_made_executable(self, sam3_home):
        plan = bs.build_install_plan(accelerator=bs.Accelerator("cpu", "t"), env={})
        dl = FakeDownloader(self._payloads(plan))
        bs.Installer(plan, runner=FakeRunner(), downloader=dl).run()
        # Only the archive: its hash is pinned here, not fetched beside it.
        assert dl.urls == [bs.uv_download_url()]
        assert os.path.exists(bs.uv_binary())
        assert os.access(bs.uv_binary(), os.X_OK)

    def test_uv_download_is_verified_against_the_pinned_checksum(self, sam3_home):
        """uv is an unsigned binary fetched from GitHub and then executed."""
        plan = bs.build_install_plan(accelerator=bs.Accelerator("cpu", "t"), env={})
        url = plan.step("uv").url
        lines = []
        outcome = bs.Installer(plan, runner=FakeRunner(), downloader=FakeDownloader(
            {url: FAKE_UV_ARCHIVE}), on_log=lines.append).run()
        assert outcome.ok, outcome.error
        assert any("sha256 verified" in l for l in lines)

        bs.clear_state()
        os.remove(bs.uv_binary())
        substituted = make_host_uv_archive(payload=b"#!/bin/sh\necho substituted\n")
        # A sidecar that agrees with the substitute changes nothing: the hash
        # that counts is the pinned one.
        dl = FakeDownloader({url: substituted, url + ".sha256": (
            "%s  uv.tar.gz\n" % hashlib.sha256(substituted).hexdigest()).encode()})
        outcome = bs.Installer(plan, runner=FakeRunner(), downloader=dl).run()
        assert not outcome.ok and outcome.failed_step == "uv"
        assert "checksum" in outcome.error
        assert not os.path.exists(bs.uv_binary()), "a bad download must not be installed"

    def test_an_asset_with_no_pinned_checksum_is_refused(self, sam3_home, monkeypatch):
        """It was logged and skipped when the checksum could not be fetched --
        a proxy that blocked the sidecar turned verification off."""
        plan = bs.build_install_plan(accelerator=bs.Accelerator("cpu", "t"), env={})
        monkeypatch.delitem(bs.UV_SHA256, bs.uv_asset_name())
        runner = FakeRunner()
        outcome = bs.Installer(plan, runner=runner,
                               downloader=FakeDownloader(self._payloads(plan))).run()
        assert not outcome.ok and outcome.failed_step == "uv"
        assert "checksum" in outcome.error
        assert not os.path.exists(bs.uv_binary())
        assert runner.argvs == []

    def test_every_command_gets_our_hf_home(self, sam3_home, monkeypatch):
        monkeypatch.delenv("HF_HOME", raising=False)
        plan = bs.build_install_plan(accelerator=bs.Accelerator("cpu", "t"), env={})
        runner = FakeRunner()
        bs.Installer(plan, runner=runner, downloader=FakeDownloader(self._payloads(plan))).run()
        for _argv, env in runner.calls:
            assert env["HF_HOME"] == bs.hf_home()
            assert env[bs.ENV_HOME] == bs.base_dir()

    def test_a_users_own_hf_home_is_left_alone(self, sam3_home, monkeypatch, tmp_path):
        """The launcher gives the daemon the user's HF_HOME when they set one;
        weights downloaded anywhere else would be invisible to it."""
        monkeypatch.setenv("HF_HOME", str(tmp_path / "their-hf"))
        plan = bs.build_weights_plan("/py")
        runner = FakeRunner()
        bs.Installer(plan, runner=runner, journal=False).run()
        assert runner.calls and all("HF_HOME" not in env for _argv, env in runner.calls)

    def test_failure_stops_and_journals_the_failed_step(self, sam3_home):
        plan = bs.build_install_plan(accelerator=bs.Accelerator("cpu", "t"), env={})
        runner = FakeRunner(default=(0, ["ok"]))
        runner.results[" ".join(bs.uv_install_command(bs.uv_binary(), bs.venv_python()))] = (
            1, ["error: no solution found", "  transformers==5.16.1 is unavailable"]
        )
        outcome = bs.Installer(
            plan, runner=runner, downloader=FakeDownloader(self._payloads(plan))
        ).run()

        assert not outcome.ok
        assert outcome.failed_step == "sam3gimpd"
        assert "no solution found" in outcome.error
        # The earlier steps stay journalled, so the retry resumes.
        assert bs.load_state(plan).completed == ["uv", "venv", "torch"]

    def test_rerun_after_success_only_verifies(self, sam3_home):
        plan = bs.build_install_plan(accelerator=bs.Accelerator("cpu", "t"), env={})
        payloads = self._payloads(plan)
        bs.Installer(plan, runner=MakesFilesRunner(), downloader=FakeDownloader(payloads)).run()

        runner2 = MakesFilesRunner()
        dl2 = FakeDownloader(payloads)
        plan2 = bs.build_install_plan(accelerator=bs.Accelerator("cpu", "t"), env={})
        assert plan2.pending(bs.load_state(plan2)) == [], "Setup shows 'Everything is installed'"
        outcome = bs.Installer(plan2, runner=runner2, downloader=dl2).run()

        assert outcome.ok
        assert outcome.completed == ["verify"]
        assert set(outcome.skipped) == {"uv", "venv", "torch", "sam3gimpd"}
        assert dl2.urls == []
        assert runner2.argvs == [bs.verify_command(bs.venv_python())]

    def test_a_failed_verify_fails_the_install(self, sam3_home):
        plan = bs.build_install_plan(accelerator=bs.Accelerator("cpu", "t"), env={})
        runner = FakeRunner({" ".join(bs.verify_command(bs.venv_python())): (1, [
            "The environment cannot load SAM 3; these do not import: transformers (x)"])})
        outcome = bs.Installer(plan, runner=runner,
                               downloader=FakeDownloader(self._payloads(plan))).run()
        assert not outcome.ok and outcome.failed_step == "verify"
        assert "transformers" in outcome.error

    def test_the_weights_flow_leaves_the_install_journal_alone(self, sam3_home):
        """The weights download wrote its own two-step plan's fingerprint over
        the install journal, although it said it was never journalled."""
        plan = bs.build_install_plan(accelerator=bs.Accelerator("cpu", "t"), env={})
        bs.Installer(plan, runner=MakesFilesRunner(),
                     downloader=FakeDownloader(self._payloads(plan))).run()
        before = Path(bs.state_file()).read_text(encoding="utf-8")

        weights = bs.build_weights_plan(bs.venv_python())
        outcome = bs.Installer(weights, runner=FakeRunner(), journal=False,
                               env={"HF_TOKEN": "hf_x"}).run()
        assert outcome.ok and outcome.completed == ["hf_login", "weights"]
        assert Path(bs.state_file()).read_text(encoding="utf-8") == before

    @pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
    def test_the_install_log_is_private(self, sam3_home):
        os.makedirs(bs.log_dir(), exist_ok=True)
        Path(bs.install_log()).write_text("from an older version\n")
        os.chmod(bs.install_log(), 0o644)
        plan = bs.build_install_plan(accelerator=bs.Accelerator("cpu", "t"), env={})
        bs.Installer(plan, runner=FakeRunner(), downloader=FakeDownloader(self._payloads(plan))).run()
        assert stat.S_IMODE(os.stat(bs.install_log()).st_mode) == 0o600
        assert stat.S_IMODE(os.stat(bs.state_file()).st_mode) == 0o600

    def test_cancel_stops_before_the_next_step(self, sam3_home):
        plan = bs.build_install_plan(accelerator=bs.Accelerator("cpu", "t"), env={})
        cancel = threading.Event()
        cancel.set()
        outcome = bs.Installer(
            plan, runner=FakeRunner(), downloader=FakeDownloader(self._payloads(plan)),
            cancel=cancel,
        ).run()
        assert not outcome.ok and outcome.cancelled
        assert outcome.completed == []

    def test_progress_is_monotonic_and_reaches_one(self, sam3_home):
        plan = bs.build_install_plan(accelerator=bs.Accelerator("cpu", "t"), env={})
        seen = []
        bs.Installer(
            plan, runner=FakeRunner(), downloader=FakeDownloader(self._payloads(plan)),
            on_progress=lambda frac, msg: seen.append(frac),
        ).run()
        assert seen and seen[-1] == pytest.approx(1.0)
        assert all(b >= a - 1e-9 for a, b in zip(seen, seen[1:]))
        assert all(0.0 <= f <= 1.0 for f in seen)

    def test_log_is_written_to_disk_for_the_doctor_panel(self, sam3_home):
        plan = bs.build_install_plan(accelerator=bs.Accelerator("cpu", "t"), env={})
        bs.Installer(plan, runner=FakeRunner(), downloader=FakeDownloader(self._payloads(plan))).run()
        text = Path(bs.install_log()).read_text(encoding="utf-8")
        assert "sam3-gimp bootstrap" in text
        # The log names the binary it ran: tools/uv, or tools\uv.exe on Windows.
        assert re.search(r"\buv(\.exe)? pip install", text), text

    def test_tokens_never_reach_the_log(self, sam3_home):
        assert "hf_abcdefghijklmnopqrstuvwxyz" not in bs._redact(
            ["sam3gimpd", "download", "--token", "hf_abcdefghijklmnopqrstuvwxyz"]
        )
        assert "hf_abcdef..." in bs._redact(["x", "hf_abcdefghijklmnopqrstuvwxyz"])


# --------------------------------------------------------------------------- #
# uv archive extraction
# --------------------------------------------------------------------------- #
class TestExtraction:
    def test_tar_gz(self, tmp_path):
        archive = tmp_path / "uv.tar.gz"
        archive.write_bytes(make_uv_tar(tmp_path, payload=b"BINARY"))
        dest = tmp_path / "uv"
        bs.extract_uv_archive(str(archive), str(dest))
        assert dest.read_bytes() == b"BINARY"
        assert os.access(str(dest), os.X_OK)

    def test_zip(self, tmp_path):
        archive = tmp_path / "uv.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("uv-x86_64-pc-windows-msvc/uv.exe", b"WINBIN")
            zf.writestr("uv-x86_64-pc-windows-msvc/README.md", b"noise")
        dest = tmp_path / "uv.exe"
        bs.extract_uv_archive(str(archive), str(dest))
        assert dest.read_bytes() == b"WINBIN"

    def test_archive_without_uv_is_rejected(self, tmp_path):
        archive = tmp_path / "empty.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("README.md", b"nothing here")
        with pytest.raises(bs.BootstrapError):
            bs.extract_uv_archive(str(archive), str(tmp_path / "uv"))

    def test_traversal_member_is_ignored(self, tmp_path):
        archive = tmp_path / "evil.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("../../uv", b"pwned")
        with pytest.raises(bs.BootstrapError):
            bs.extract_uv_archive(str(archive), str(tmp_path / "uv"))

    def test_an_interrupted_extraction_leaves_no_binary(self, tmp_path, monkeypatch):
        """``uv`` existing is what marks the step done, so a half-written one
        would be skipped past on every later run and then fail to execute."""
        archive = tmp_path / "uv.tar.gz"
        archive.write_bytes(make_uv_tar(tmp_path, payload=b"B" * 100000))
        dest = tmp_path / "tools" / "uv"

        def dies_halfway(src, out, *a, **k):
            out.write(src.read(1000))
            raise OSError("disk full")

        monkeypatch.setattr(bs.shutil, "copyfileobj", dies_halfway)
        with pytest.raises(OSError):
            bs.extract_uv_archive(str(archive), str(dest))
        assert not dest.exists()
        assert os.listdir(str(dest.parent)) == [], "no temporary file is left behind"

    @pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
    def test_the_binary_is_executable_by_its_owner_only(self, tmp_path):
        archive = tmp_path / "uv.tar.gz"
        archive.write_bytes(make_uv_tar(tmp_path))
        dest = tmp_path / "uv"
        bs.extract_uv_archive(str(archive), str(dest))
        assert stat.S_IMODE(os.stat(str(dest)).st_mode) == 0o700


# --------------------------------------------------------------------------- #
# downloader (resume behaviour, no network)
# --------------------------------------------------------------------------- #
class TestDownloader:
    def test_writes_via_a_part_file(self, tmp_path):
        calls = []

        def opener(url, headers=None, timeout=None):
            calls.append(dict(headers or {}))
            return FakeResponse(200, b"0123456789", headers={"Content-Length": "10"})

        dest = tmp_path / "thing.bin"
        bs.Downloader(opener=opener, chunk=4).download("http://x/thing.bin", str(dest))
        assert dest.read_bytes() == b"0123456789"
        assert not (tmp_path / "thing.bin.part").exists()
        assert "Range" not in calls[0]
        if os.name != "nt":
            assert stat.S_IMODE(os.stat(str(dest)).st_mode) == 0o600

    def test_resumes_from_a_partial_download(self, tmp_path):
        (tmp_path / "thing.bin.part").write_bytes(b"0123")
        calls = []

        def opener(url, headers=None, timeout=None):
            calls.append(dict(headers or {}))
            return FakeResponse(206, b"456789", headers={"Content-Length": "6"})

        dest = tmp_path / "thing.bin"
        bs.Downloader(opener=opener).download("http://x/thing.bin", str(dest))
        assert calls[0]["Range"] == "bytes=4-"
        assert dest.read_bytes() == b"0123456789"

    def test_server_ignoring_range_restarts_cleanly(self, tmp_path):
        """A 200 in reply to a Range request means "here is the whole file"; we
        must overwrite rather than append, or the binary is corrupt."""
        (tmp_path / "thing.bin.part").write_bytes(b"GARBAGE")

        def opener(url, headers=None, timeout=None):
            return FakeResponse(200, b"WHOLE", headers={"Content-Length": "5"})

        dest = tmp_path / "thing.bin"
        bs.Downloader(opener=opener).download("http://x/thing.bin", str(dest))
        assert dest.read_bytes() == b"WHOLE"

    def test_progress_fractions(self, tmp_path):
        seen = []

        def opener(url, headers=None, timeout=None):
            return FakeResponse(200, b"x" * 100, headers={"Content-Length": "100"})

        bs.Downloader(opener=opener, chunk=25).download(
            "http://x/f", str(tmp_path / "f"), on_progress=lambda f, d, t: seen.append((f, d, t))
        )
        assert seen[-1] == (1.0, 100, 100)
        assert all(0.0 <= f <= 1.0 for f, _d, _t in seen)

    def test_http_error_status_is_reported(self, tmp_path):
        def opener(url, headers=None, timeout=None):
            return FakeResponse(500, b"", headers={})

        with pytest.raises(bs.BootstrapError):
            bs.Downloader(opener=opener).download("http://x/f", str(tmp_path / "f"))

    def test_cancel_mid_download(self, tmp_path):
        cancel = threading.Event()
        cancel.set()

        def opener(url, headers=None, timeout=None):
            return FakeResponse(200, b"x" * 100, headers={"Content-Length": "100"})

        with pytest.raises(bs.CancelledError):
            bs.Downloader(opener=opener, chunk=10).download(
                "http://x/f", str(tmp_path / "f"), cancel=cancel
            )


# --------------------------------------------------------------------------- #
# the gated-weights flow
# --------------------------------------------------------------------------- #
class TestHuggingFaceFlow:
    def test_empty_token_is_rejected_locally(self):
        assert bs.validate_hf_token("")["ok"] is False

    def test_obviously_wrong_token_never_hits_the_network(self):
        def boom(*a, **k):
            raise AssertionError("should not have called out")

        r = bs.validate_hf_token("not-a-token", opener=boom)
        assert not r["ok"] and "hf_" in r["message"]

    def test_valid_token(self):
        record = []
        opener = fake_opener(200, {"name": "someuser", "type": "user"}, record)
        r = bs.validate_hf_token("hf_" + "a" * 30, opener=opener)
        assert r["ok"] and r["name"] == "someuser"
        assert record[0][0] == "https://huggingface.co/api/whoami-v2"
        assert record[0][1]["Authorization"].startswith("Bearer hf_")

    def test_rejected_token(self):
        r = bs.validate_hf_token("hf_" + "b" * 30, opener=fake_opener(401, {"error": "Invalid"}))
        assert not r["ok"] and r["status"] == 401

    def test_offline_is_explained_not_crashed(self):
        def opener(*a, **k):
            raise OSError("Name or service not known")

        r = bs.validate_hf_token("hf_" + "c" * 30, opener=opener)
        assert not r["ok"] and "huggingface.co" in r["message"]

    GATED_FILE = "https://huggingface.co/facebook/sam3/resolve/main/config.json"

    @staticmethod
    def head_opener(answers, record=None):
        """HEAD-only fake of huggingface.co: ``answers`` maps a URL to
        ``(status, headers)``; anything else is a 404."""
        def _open(url, headers=None, timeout=None, method=None):
            assert method == "HEAD", "the gate check must not download anything"
            if record is not None:
                record.append((url, dict(headers or {})))
            status, hdrs = answers.get(url, (404, {}))
            if status >= 300:
                err = OSError("HTTP %s" % status)
                err.code = status
                err.headers = hdrs
                raise err
            return FakeResponse(status, b"", headers=hdrs)
        return _open

    def test_gated_access_granted(self):
        record = []
        r = bs.check_gated_access("hf_" + "d" * 30, opener=self.head_opener(
            {self.GATED_FILE: (200, {})}, record))
        assert r["ok"] and r["status"] == 200
        # It asks for a file of the gated repo, with the token.
        assert [url for url, _h in record] == [self.GATED_FILE]
        assert record[0][1]["Authorization"] == "Bearer hf_" + "d" * 30

    def test_the_public_model_api_is_not_the_question(self):
        """``/api/models/facebook/sam3`` is public: it answers 200 to no token,
        a bogus token and an unaccepted licence alike (checked live), so every
        token used to come back "Access granted"."""
        seen = []
        bs.check_gated_access("hf_" + "d" * 30, opener=self.head_opener({}, seen))
        assert all("/api/" not in url for url, _h in seen)

    def test_gated_access_denied_points_at_the_model_page(self):
        """403 from the file: the token is valid, the licence is not accepted."""
        r = bs.check_gated_access("hf_" + "e" * 30, opener=self.head_opener(
            {self.GATED_FILE: (403, {"X-Error-Code": "GatedRepo"})}))
        assert not r["ok"] and r["status"] == 403
        assert r["url"] == bs.SAM3_MODEL_URL
        assert "accept" in r["message"].lower()

    def test_a_rejected_or_missing_token_is_not_called_a_licence_problem(self):
        """401 is what huggingface.co answers with no token or a bogus one."""
        r = bs.check_gated_access("hf_" + "f" * 30, opener=self.head_opener(
            {self.GATED_FILE: (401, {"X-Error-Code": "GatedRepo"})}))
        assert not r["ok"] and r["status"] == 401
        assert "token" in r["message"].lower()
        assert r["url"] == bs.HF_TOKENS_URL

    def test_redirects_are_followed_as_head_and_keep_the_token_on_huggingface(self):
        record = []
        moved = "https://huggingface.co/meta/sam3-renamed/resolve/main/config.json"
        r = bs.check_gated_access("hf_" + "g" * 30, opener=self.head_opener({
            self.GATED_FILE: (307, {"Location": "/meta/sam3-renamed/resolve/main/config.json"}),
            moved: (200, {}),
        }, record))
        assert r["ok"]
        assert [u for u, _h in record] == [self.GATED_FILE, moved]
        assert record[1][1]["Authorization"].startswith("Bearer hf_")

    def test_the_token_never_follows_a_redirect_off_huggingface(self):
        record = []
        cdn = "https://cdn.example.net/blob/config.json"
        bs.check_gated_access("hf_" + "h" * 30, opener=self.head_opener({
            self.GATED_FILE: (302, {"Location": cdn}), cdn: (200, {})}, record))
        assert [u for u, _h in record] == [self.GATED_FILE, cdn]
        assert "Authorization" not in record[1][1]

    def test_offline_gate_check_is_explained(self):
        def opener(*a, **k):
            raise OSError("Name or service not known")

        r = bs.check_gated_access("hf_" + "i" * 30, opener=opener)
        assert not r["ok"] and r["status"] == 0 and "huggingface.co" in r["message"]


class TestLocalWeights:
    def _checkpoint(self, tmp_path: Path) -> Path:
        d = tmp_path / "sam3"
        d.mkdir()
        (d / "config.json").write_text("{}")
        (d / "model.safetensors").write_bytes(b"\x00" * 16)
        return d

    def test_valid_checkpoint_dir(self, tmp_path):
        r = bs.inspect_local_weights(str(self._checkpoint(tmp_path)))
        assert r["ok"], r["message"]

    def test_missing_dir(self, tmp_path):
        assert not bs.inspect_local_weights(str(tmp_path / "nope"))["ok"]

    def test_empty_path(self):
        assert not bs.inspect_local_weights("")["ok"]

    def test_weights_without_config_is_explained(self, tmp_path):
        d = tmp_path / "half"
        d.mkdir()
        (d / "model.safetensors").write_bytes(b"\x00")
        r = bs.inspect_local_weights(str(d))
        assert not r["ok"] and "config.json" in r["message"]

    def test_config_without_weights_is_explained(self, tmp_path):
        d = tmp_path / "half2"
        d.mkdir()
        (d / "config.json").write_text("{}")
        r = bs.inspect_local_weights(str(d))
        assert not r["ok"] and "weight file" in r["message"]

    def test_set_and_clear_round_trip(self, sam3_home, tmp_path):
        ckpt = self._checkpoint(tmp_path)
        bs.set_local_weights(str(ckpt))
        assert bs.local_weights_path() == str(ckpt)
        assert bs.weights_present()
        # The launcher exports this to the daemon (conftest probes the same name).
        assert bs.weights_environ() == {bs.ENV_WEIGHTS_DIR: str(ckpt)}
        bs.clear_local_weights()
        assert bs.local_weights_path() is None
        assert bs.weights_environ() == {}

    def test_setting_a_bad_path_raises(self, sam3_home, tmp_path):
        with pytest.raises(bs.BootstrapError):
            bs.set_local_weights(str(tmp_path / "missing"))

    def test_hf_snapshot_layout_counts_as_present(self, sam3_home, monkeypatch):
        """A populated snapshot counts; a bare directory does not.

        This asserted that creating the directory was enough, which is how a
        user with only sam3.pt was told "SAM 3 weights are available".
        """
        monkeypatch.delenv("HF_HOME", raising=False)
        assert not bs.weights_present()
        snap = Path(bs.hf_home()) / "hub" / "models--facebook--sam3"
        snap.mkdir(parents=True)
        assert not bs.weights_present(), "an empty cache tree is not weights"

        revision = snap / "snapshots" / "rev1"
        revision.mkdir(parents=True)
        (revision / "sam3.pt").write_bytes(b"")
        assert not bs.weights_present(), "the original checkpoint is not loadable"

        (revision / "model.safetensors").write_bytes(b"")
        (revision / "config.json").write_text("{}")
        assert bs.weights_present()


# --------------------------------------------------------------------------- #
# environment report / doctor
# --------------------------------------------------------------------------- #
class TestEnvironmentReport:
    def test_clean_box(self, sam3_home, monkeypatch):
        monkeypatch.delenv("HF_HOME", raising=False)
        report = bs.inspect_environment(accelerator=bs.Accelerator("cpu", "t"))
        assert not report.env_ready and not report.fully_ready
        assert report.missing == ["uv", "venv", "sam3gimpd", "weights"]
        assert report.summary() == "Not installed yet."
        json.dumps(report.to_json())

    def test_half_built(self, sam3_home):
        os.makedirs(bs.venv_bin_dir(), exist_ok=True)
        Path(bs.venv_python()).write_text("")
        report = bs.inspect_environment(accelerator=bs.Accelerator("cpu", "t"))
        assert not report.env_ready
        assert "resume" in report.summary()

    def test_env_ready_without_weights(self, sam3_home, monkeypatch):
        monkeypatch.delenv("HF_HOME", raising=False)
        os.makedirs(bs.venv_bin_dir(), exist_ok=True)
        os.makedirs(bs.tools_dir(), exist_ok=True)
        for p in (bs.uv_binary(), bs.venv_python(), bs.sam3d_executable()):
            Path(p).write_text("")
        report = bs.inspect_environment(accelerator=bs.Accelerator("cuda", "t"))
        assert report.env_ready and not report.fully_ready
        assert "weights are missing" in report.summary()

    def test_runtime_json_must_have_the_five_required_fields(self, sam3_home):
        Path(bs.runtime_file()).write_text(json.dumps({"port": 1, "token": "t"}))
        assert bs.read_runtime() is None

        Path(bs.runtime_file()).write_text(json.dumps({
            "port": 41573, "token": "t" * 43, "pid": os.getpid(),
            "version": "0.1.0", "started_at": 1.0, "unknown_key": "ignored",
        }))
        info = bs.read_runtime()
        assert info["port"] == 41573 and info["pid"] == os.getpid()

        report = bs.inspect_environment(accelerator=bs.Accelerator("cpu", "t"))
        assert report.runtime_present and report.daemon_running
        assert report.daemon_port == 41573

    def test_stale_runtime_json_is_not_a_running_daemon(self, sam3_home):
        Path(bs.runtime_file()).write_text(json.dumps({
            "port": 41573, "token": "t" * 43, "pid": 2 ** 31 - 1,
            "version": "0.1.0", "started_at": 1.0,
        }))
        report = bs.inspect_environment(accelerator=bs.Accelerator("cpu", "t"))
        assert report.runtime_present and not report.daemon_running

    def test_pid_alive(self):
        assert bs.pid_alive(os.getpid())
        assert not bs.pid_alive(0)
        assert not bs.pid_alive(2 ** 31 - 1)


class TestDoctor:
    def test_reports_missing_install(self, sam3_home):
        out = bs.run_doctor(runner=FakeRunner())
        assert out["daemon"] is None
        assert "Setup" in out["message"]
        assert out["local"]["base"] == str(sam3_home)

    def test_drives_the_panel_from_sam3d_doctor(self, sam3_home):
        os.makedirs(bs.venv_bin_dir(), exist_ok=True)
        Path(bs.sam3d_executable()).write_text("")
        payload = {"device": "cuda:0", "dtype": "bfloat16", "sam3d_version": "0.1.0",
                   "models_loaded": [], "memory": {"vram_total": 25757220864}}
        runner = FakeRunner({"sam3gimpd": (0, [json.dumps(payload)])})
        out = bs.run_doctor(runner=runner)
        assert out["doctor_ok"]
        assert out["daemon"]["device"] == "cuda:0"
        assert runner.argvs[0] == bs.doctor_command(bs.sam3d_executable())
        assert out["doctor_argv"] == runner.argvs[0]

    def test_reads_the_report_the_real_cli_prints(self, sam3_home, monkeypatch):
        """``sam3gimpd doctor``'s own output, flattened for the panel: the
        engine probe, overridden by a running daemon's /status."""
        import contextlib
        import io
        from sam3gimpd import cli

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert cli.main(["doctor", "--no-daemon"]) == 0
        report = json.loads(buf.getvalue())
        report["daemon"] = {"reachable": True, "status": {
            "device": "cuda", "dtype": "float16", "engine_mode": "real",
            "models_loaded": ["pcs"], "memory": {"vram_total": 11811160064}}}
        os.makedirs(bs.venv_bin_dir(), exist_ok=True)
        Path(bs.sam3d_executable()).write_text("")
        out = bs.run_doctor(runner=FakeRunner(
            {"sam3gimpd": (0, json.dumps(report, indent=2).splitlines())}))
        view = out["daemon"]
        assert view["sam3d_version"] == report["sam3d_version"]
        assert "torch_available" in view, "the engine probe is lifted to the top"
        assert view["device"] == "cuda" and view["engine_mode"] == "real"
        assert view["daemon_reachable"] is True

    def test_uses_the_users_hf_home_when_they_set_one(self, sam3_home, monkeypatch, tmp_path):
        """The launcher gives the daemon the user's HF_HOME; Doctor forced ours,
        so it reported on a cache the daemon never reads."""
        os.makedirs(bs.venv_bin_dir(), exist_ok=True)
        Path(bs.sam3d_executable()).write_text("")
        monkeypatch.setenv("HF_HOME", str(tmp_path / "their-hf"))
        runner = FakeRunner({"sam3gimpd": (0, ["{}"])})
        bs.run_doctor(runner=runner)
        assert "HF_HOME" not in runner.calls[0][1]
        monkeypatch.delenv("HF_HOME")
        bs.run_doctor(runner=runner)
        assert runner.calls[1][1]["HF_HOME"] == bs.hf_home()

    def test_tolerates_a_human_readable_dump(self, sam3_home):
        os.makedirs(bs.venv_bin_dir(), exist_ok=True)
        Path(bs.sam3d_executable()).write_text("")
        runner = FakeRunner({"sam3gimpd": (0, ["sam3gimpd doctor", '{"device": "cpu"}', "bye"])})
        out = bs.run_doctor(runner=runner)
        assert out["daemon"] == {"device": "cpu"}

    def test_failing_doctor_still_returns_local_facts(self, sam3_home):
        os.makedirs(bs.venv_bin_dir(), exist_ok=True)
        Path(bs.sam3d_executable()).write_text("")
        runner = FakeRunner({"sam3gimpd": (1, ["ImportError: torch"])})
        out = bs.run_doctor(runner=runner)
        assert not out["doctor_ok"]
        assert out["local"]["venv_present"] is False
        assert "exited 1" in out["message"]

    def test_crash_log_tail_is_surfaced(self, sam3_home):
        os.makedirs(bs.log_dir(), exist_ok=True)
        Path(bs.crash_log()).write_text("\n".join("line %d" % i for i in range(200)))
        out = bs.run_doctor(runner=FakeRunner())
        assert out["crash_log"][-1] == "line 199"
        assert len(out["crash_log"]) == 40

    @pytest.mark.parametrize("text,expected", [
        ("", None),
        ("not json at all", None),
        ('{"a": 1}', {"a": 1}),
        ('noise {"a": 1} noise', {"a": 1}),
        ("[1,2]", None),
    ])
    def test_parse_doctor_output(self, text, expected):
        assert bs.parse_doctor_output(text) == expected


class TestCommandRunner:
    """Timeout and cancel hold whether or not the child prints anything.

    They were checked only when a line of output arrived: ``timeout=1`` on a
    silent ``sleep 6`` took 6 s, a Cancel pressed during a quiet uv download
    did nothing until uv next spoke, and a hung ``nvidia-smi -L`` could hold
    the GTK thread for as long as it liked.
    """

    SILENT = [sys.executable, "-c", "import time; time.sleep(8)"]

    def test_timeout_on_a_silent_child(self):
        started = time.monotonic()
        result = bs.CommandRunner().run(self.SILENT, timeout=1.0)
        assert time.monotonic() - started < 4.0
        assert not result.ok
        assert "timed out" in result.text

    def test_cancel_on_a_silent_child(self):
        cancel = threading.Event()
        threading.Timer(0.5, cancel.set).start()
        started = time.monotonic()
        result = bs.CommandRunner().run(self.SILENT, cancel=cancel)
        assert time.monotonic() - started < 4.0
        assert not result.ok

    @pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
    def test_cancel_ends_the_whole_process_tree(self, tmp_path):
        """uv does its work in children; stopping only uv left them running
        (and holding the output pipe open)."""
        script = ("import subprocess,sys,time\n"
                  "c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
                  "print(c.pid, flush=True)\n"
                  "time.sleep(60)\n")
        cancel = threading.Event()
        pids = []

        def on_line(line):
            if line.strip().isdigit():
                pids.append(int(line))
                cancel.set()

        started = time.monotonic()
        bs.CommandRunner().run([sys.executable, "-c", script], on_line=on_line, cancel=cancel)
        assert time.monotonic() - started < 10.0
        assert pids, "the grandchild reported its pid"
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and bs.pid_alive(pids[0]):
            time.sleep(0.05)
        assert not bs.pid_alive(pids[0]), "the grandchild outlived the cancel"

    def test_output_still_streams_and_the_exit_code_is_kept(self):
        seen = []
        result = bs.CommandRunner().run(
            [sys.executable, "-c", "print('a'); print('b'); raise SystemExit(3)"],
            on_line=seen.append)
        assert seen == ["a", "b"] and result.lines == ["a", "b"]
        assert result.returncode == 3

    def test_output_is_decoded_as_utf8_not_the_locale(self, monkeypatch):
        """uv prints UTF-8; decoding it with the Windows locale (cp1252) turned
        every non-ASCII character to mojibake."""
        seen = {}
        real = subprocess.Popen

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(bs.subprocess, "Popen", spy)
        env = {"PYTHONIOENCODING": "utf-8"}
        result = bs.CommandRunner().run(
            [sys.executable, "-c", "print('\\u2713 caf\\u00e9')"], env=env)
        assert seen.get("encoding") == "utf-8" and seen.get("errors") == "replace"
        assert result.lines == ["✓ café"]


class TestWeightsWithTheUsersEnvironment:
    """Download weights with "Use an existing Python environment" failed at
    hf_login with "No such file": it ran the managed venv, which that user
    does not have."""

    def test_both_steps_run_in_the_given_interpreter(self):
        plan = bs.build_weights_plan("/their/env/python")
        assert plan.keys == ["hf_login", "weights"]
        assert plan.step("hf_login").argv[0] == "/their/env/python"
        assert plan.step("weights").argv[:3] == ["/their/env/python", "-m", "sam3gimpd"]

    def test_the_token_is_not_in_argv(self):
        for step in bs.build_weights_plan("/py").steps:
            assert not any("hf_" in a for a in step.argv)


class TestTailFile:
    def test_missing_file(self, tmp_path):
        assert bs.tail_file(str(tmp_path / "nope")) == []

    def test_last_n_lines(self, tmp_path):
        p = tmp_path / "log"
        p.write_text("\n".join(str(i) for i in range(100)))
        assert bs.tail_file(str(p), 5) == ["95", "96", "97", "98", "99"]

    def test_huge_file_is_not_fully_read(self, tmp_path):
        p = tmp_path / "big"
        p.write_text("\n".join("x" * 100 for _ in range(20000)))
        lines = bs.tail_file(str(p), 10, max_bytes=4096)
        assert len(lines) == 10


# --------------------------------------------------------------------------- #
# the two dialogs, as far as they can be exercised headlessly
# --------------------------------------------------------------------------- #
# These are smoke tests, not UI tests: they prove that the plug-in's GTK widget
# trees actually build against the installed PyGObject, and that the setup
# dialog reads the same environment the functions above return.  They are the
# only defence against a typo in a GTK constructor that would otherwise first
# appear on a user's Windows machine.
@pytest.mark.needs_gtk
class TestDialogsBuild:
    def test_setup_dialog_builds_and_reflects_the_environment(self, sam3_home, monkeypatch):
        monkeypatch.delenv("HF_HOME", raising=False)
        from ui import setup_dialog

        assert setup_dialog.needs_setup() is True
        dialog = setup_dialog.SetupDialog(None)
        try:
            assert dialog._report is not None
            assert dialog._report.base == str(sam3_home)
            assert dialog._plan.keys == ["uv", "venv", "torch", "sam3gimpd", "verify"]
            # A clean box offers Install, not Resume.
            assert dialog._install_button.get_label() == "Install"
            assert dialog._install_button.get_sensitive()
        finally:
            dialog.destroy()

    def test_setup_dialog_offers_resume_after_a_partial_install(self, sam3_home):
        from ui import setup_dialog

        plan = bs.build_install_plan(accelerator=bs.Accelerator("cpu", "t"), env={})
        state = bs.load_state(plan)
        state.fingerprint = bs.plan_fingerprint(plan)
        state.mark("uv")
        bs.save_state(state)

        dialog = setup_dialog.SetupDialog(None)
        try:
            assert dialog._install_button.get_label() == "Resume install"
        finally:
            dialog.destroy()

    def test_a_journal_over_a_deleted_venv_still_offers_the_install(self, sam3_home, monkeypatch):
        """Everything journalled, the venv gone: the button read "Everything is
        installed" and was greyed out, beside a summary saying "Not installed
        yet"."""
        monkeypatch.setenv("SAM3_FORCE_DEVICE", "cpu")
        from ui import setup_dialog

        plan = bs.build_install_plan(accelerator=bs.Accelerator("cpu", "t"), env={})
        bs.save_state(bs.InstallState(fingerprint=bs.plan_fingerprint(plan),
                                      completed=list(plan.keys)))
        dialog = setup_dialog.SetupDialog(None)
        try:
            assert dialog._install_button.get_sensitive()
            assert dialog._install_button.get_label() == "Resume install"
        finally:
            dialog.destroy()

    # -- the main dialog, driven with a real result frame ------------------ #
    @staticmethod
    def _frame():
        """A conforming §8 frame: two instances, genuinely soft masks."""
        from ui import canvas

        header = {
            "api_version": "1.0", "job_id": "j-1", "request_id": "r-000001",
            "image_id": "abc", "engine": "pcs", "state": "done",
            "prompt": {"kind": "text", "text": "red car"},
            "image": {"width": 64, "height": 48},
            "model_canvas": {"width": 1008, "height": 1008},
            "canvas_from_image": {"scale_x": 15.75, "scale_y": 21.0,
                                  "offset_x": 0.0, "offset_y": 0.0},
            "mask_encoding": "u8_soft", "elapsed_ms": 12.0, "truncated": False,
            "instances": [
                {"instance_id": 0, "score": 0.93, "label": "red car",
                 "bbox": [10, 20, 14, 23], "mask_width": 4, "mask_height": 3,
                 "blob_offset": 0, "blob_length": 12},
                {"instance_id": 1, "score": 0.41, "label": "red car",
                 "bbox": [0, 0, 2, 2], "mask_width": 2, "mask_height": 2,
                 "blob_offset": 0, "blob_length": 4},
            ],
        }
        blobs = [bytes(range(20, 32)), b"\xff\x80\x40\x10"]
        return canvas.encode_frame(header, blobs)

    def test_main_dialog_adopts_a_result_frame(self, sam3_home):
        import client
        from ui import main_dialog

        result = client.parse_result_frame(self._frame())
        dialog = main_dialog.MainDialog(None)
        try:
            assert dialog._canvas is not None, "ui/canvas.py should provide Sam3Canvas here"
            dialog._latest_request_id = "r-000001"
            dialog._on_prompt_done(result, None)

            assert len(dialog._instances) == 2
            # The blob region is reconstructed losslessly: §8.1 packs the masks
            # tightly, in instance order, with no padding.
            assert dialog._blob == bytes(range(20, 32)) + b"\xff\x80\x40\x10"
            assert dialog._canvas.request_id == "r-000001"

            # Score slider filters locally, with no round trip.
            dialog._score_scale.set_value(0.5)
            assert [i.instance_id for i in dialog.selected_instances()] == [0]
            dialog._score_scale.set_value(0.1)
            assert [i.instance_id for i in dialog.selected_instances()] == [0, 1]

            # Unticking a row is a manual override that the slider respects.
            dialog._on_row_toggled(None, "1")
            assert [i.instance_id for i in dialog.selected_instances()] == [0]
            assert dialog._canvas.get_instance(1).visible is False
        finally:
            dialog.destroy()

    def test_ticking_a_row_does_not_rebuild_the_list(self, sam3_home):
        """The user ticked rows after None and the list jumped: it was cleared
        and re-appended on every tick, so the scroll went back to the top,
        the row selection vanished, and the next click landed elsewhere.
        Rows must be updated in place and keep their order and selection."""
        import client
        from gi.repository import Gtk
        from ui import main_dialog

        result = client.parse_result_frame(self._frame())
        dialog = main_dialog.MainDialog(None)
        try:
            dialog._latest_request_id = "r-000001"
            dialog._on_prompt_done(result, None)
            ids_before = [int(row[3]) for row in dialog._store]
            paths_before = [row.path.to_string() for row in dialog._store]
            assert len(ids_before) == 2

            dialog._tree.get_selection().select_path(Gtk.TreePath.new_from_string("1"))
            # None, then tick the first row back on -- the reported sequence.
            dialog._set_all_visible(False)
            assert [bool(row[0]) for row in dialog._store] == [False, False]
            dialog._on_row_toggled(None, "0")
            assert [bool(row[0]) for row in dialog._store] == [True, False]
            assert [int(row[3]) for row in dialog._store] == ids_before, "order must not change"
            assert [row.path.to_string() for row in dialog._store] == paths_before
            model, it = dialog._tree.get_selection().get_selected()
            assert it is not None and model.get_path(it).to_string() == "1", (
                "the row selection must survive a tick on another row")
            assert [i.instance_id for i in dialog.selected_instances()] == [ids_before[0]]

            # The score slider only re-flags rows; it does not rebuild either.
            dialog._score_scale.set_value(0.5)
            assert [int(row[3]) for row in dialog._store] == ids_before
            assert [row.path.to_string() for row in dialog._store] == paths_before
        finally:
            dialog.destroy()

    def test_the_canvas_click_mode_is_switchable_and_follows_the_job(self, sam3_home):
        """Segment-by-clicking (PVS) was unreachable: the canvas has a
        "select" and a "points" mode, the dialog never switched it out of
        "select", and a click on empty canvas did nothing.  Now: points until
        there is something to pick, "select" once a text prompt returns
        objects, back to points when the user says so or refines by clicking."""
        import client
        from ui import main_dialog

        dialog = main_dialog.MainDialog(None)
        try:
            assert dialog.canvas_mode() == "points"
            assert dialog._canvas.interaction_mode == "points"
            assert not dialog._mode_pick.get_active()

            result = client.parse_result_frame(self._frame())
            dialog._latest_request_id = "r-000001"
            dialog._job_kind = "text"
            dialog._on_prompt_done(result, None)
            assert dialog.canvas_mode() == "select"
            assert dialog._canvas.interaction_mode == "select"

            dialog._mode_points.set_active(True)
            assert dialog._canvas.interaction_mode == "points"
            assert "places a point" in dialog._status.get_text()

            dialog._mode_pick.set_active(True)
            assert dialog._canvas.interaction_mode == "select"

            # A click-refine result keeps the conversation with the canvas going.
            dialog._latest_request_id = "r-000002"
            dialog._job_kind = "points"
            result2 = client.parse_result_frame(self._frame())
            result2.request_id = "r-000002"
            dialog._on_prompt_done(result2, None)
            assert dialog.canvas_mode() == "points"
        finally:
            dialog.destroy()

    def test_clicking_an_instance_on_the_canvas_selects_its_row(self, sam3_home):
        """The list followed the canvas in one direction only."""
        import client
        from ui import main_dialog

        result = client.parse_result_frame(self._frame())
        dialog = main_dialog.MainDialog(None)
        try:
            dialog._latest_request_id = "r-000001"
            dialog._job_kind = "text"
            dialog._on_prompt_done(result, None)
            second = int(dialog._store[1][3])
            dialog._canvas.set_active_instance(second)
            model, it = dialog._tree.get_selection().get_selected()
            assert it is not None and int(model[it][3]) == second
            dialog._canvas.set_active_instance(-1)
            _model, it = dialog._tree.get_selection().get_selected()
            assert it is None
        finally:
            dialog.destroy()

    def test_a_point_prompt_ticks_only_the_best_candidate(self, sam3_home):
        """PVS candidates are alternatives for one object; ticking all of
        them applied whole-plus-parts and read as three hits."""
        import client
        from ui import main_dialog

        result = client.parse_result_frame(self._frame())
        dialog = main_dialog.MainDialog(None)
        try:
            dialog._latest_request_id = "r-000001"
            dialog._job_kind = "points"
            dialog._on_prompt_done(result, None)
            ids = [int(i.instance_id) for i in dialog._instances]
            assert len(ids) == 2
            assert dialog._is_visible(ids[0]) and not dialog._is_visible(ids[1])
            assert [i.instance_id for i in dialog.selected_instances()] == [ids[0]]
            # a text result keeps every instance ticked
            dialog._latest_request_id = "r-000002"
            dialog._job_kind = "text"
            result2 = client.parse_result_frame(self._frame())
            result2.request_id = "r-000002"
            dialog._on_prompt_done(result2, None)
            assert all(dialog._is_visible(i) for i in ids)
        finally:
            dialog.destroy()

    def test_main_dialog_drops_stale_results(self, sam3_home):
        """API.md §10: anything not from the latest request_id is discarded."""
        import client
        from ui import main_dialog

        result = client.parse_result_frame(self._frame())
        dialog = main_dialog.MainDialog(None)
        try:
            dialog._latest_request_id = "r-000009"
            dialog._on_prompt_done(result, None)
            assert dialog._instances == []
            assert dialog._result is None
        finally:
            dialog.destroy()

    def test_main_dialog_treats_none_as_superseded(self, sam3_home):
        """``run_text`` returns None when a newer prompt displaced this one."""
        from ui import main_dialog

        dialog = main_dialog.MainDialog(None)
        try:
            dialog._on_prompt_done(None, None)
            assert dialog._instances == []
        finally:
            dialog.destroy()

    def test_main_dialog_builds_output_options_outputs_understands(self, sam3_home):
        import outputs
        from ui import main_dialog

        dialog = main_dialog.MainDialog(None)
        try:
            # 128 is a raw logit of 0 -- the model's own edge (API.md §8.3).
            assert main_dialog.DEFAULT_MASK_THRESHOLD == 128
            dialog._mode_combo.set_active_id("layer-groups")
            dialog._feather.set_value(2.5)
            dialog._grow.set_value(3)
            dialog._fill_holes.set_active(True)

            options = dialog.output_options().validated()
            assert options.mode in outputs.OutputMode.ALL
            assert options.mode == outputs.OutputMode.LAYER_GROUPS
            assert options.selection_op in outputs.SelectionOp.ALL
            assert options.post.threshold == 128
            assert options.post.feather == 2.5
            assert options.post.grow == 3
            assert options.post.hole_fill is True

            # Every mode id in the combo is one outputs.py accepts.
            for key, _text in main_dialog.OUTPUT_MODES:
                assert key in outputs.OutputMode.ALL
            for key, _text in main_dialog.SELECTION_OPS:
                assert key in outputs.SelectionOp.ALL
        finally:
            dialog.destroy()

    def test_main_dialog_maps_a_frame_into_outputs_mask_result(self, sam3_home):
        """The canvas -> uploaded -> original chain of API.md §9, end to end."""
        import client
        from ui import main_dialog

        result = client.parse_result_frame(self._frame())
        dialog = main_dialog.MainDialog(None)
        try:
            dialog._latest_request_id = "r-000001"
            dialog._on_prompt_done(result, None)
            dialog._geometry = _FakeGeometry(source_width=640, source_height=480,
                                             width=64, height=48)
            mask_result = dialog._mask_result()
            assert mask_result.source == (640, 480)
            assert mask_result.uploaded == (64, 48)
            # bbox [10,20,14,23] on a 1008 canvas from a 64x48 upload of a
            # 640x480 image: canvas -> uploaded -> original.
            x, y, w, h = mask_result.rect_for(mask_result.instances[0])
            assert (x, y) == (6, 10)
            assert w >= 1 and h >= 1
        finally:
            dialog.destroy()

    def test_main_dialog_settings_round_trip_without_gimp(self, sam3_home):
        """No ``Gimp.ProcedureConfig`` outside GIMP, so the JSON fallback runs."""
        from ui import main_dialog

        first = main_dialog.MainDialog(None)
        try:
            first._prompt_entry.set_text("yellow school bus")
            first._mode_combo.set_active_id("channels")
            first._mask_scale.set_value(200)
            first._min_area.set_value(512)
            first._save_settings()
        finally:
            first.destroy()

        assert os.path.exists(bs.ui_settings_file())
        second = main_dialog.MainDialog(None)
        try:
            assert second._prompt_entry.get_text() == "yellow school bus"
            assert second._mode_combo.get_active_id() == "channels"
            assert second.post_ops().threshold == 200
            assert second.post_ops().min_area == 512
        finally:
            second.destroy()


class _FakeGeometry:
    """Stands in for ``gimpbridge.UploadGeometry`` (which needs GIMP to build)."""

    def __init__(self, source_width, source_height, width, height):
        self.source_width = source_width
        self.source_height = source_height
        self.width = width
        self.height = height


# --------------------------------------------------------------------------- #
# probing an interpreter the user chose in Setup
# --------------------------------------------------------------------------- #
class TestProbeInterpreter:
    """Setup lets the user point at their own PyTorch environment; this is the
    check that tells them, before they commit, what is actually in it."""

    def test_probes_this_very_interpreter(self):
        probe = bs.probe_interpreter(sys.executable)
        assert probe["ok"] is True
        assert probe["python_version"].startswith("3.")
        # torch/sam3gimpd may or may not be here; the keys must exist either way
        assert "torch" in probe and "sam3gimpd" in probe

    def test_a_missing_file_is_a_normal_answer_not_a_raise(self):
        probe = bs.probe_interpreter("/definitely/not/here/python")
        assert probe["ok"] is False
        assert "No such file" in probe["error"]

    def test_an_empty_path_is_handled(self):
        assert bs.probe_interpreter("")["ok"] is False

    def test_a_non_python_executable_is_rejected_clearly(self, tmp_path):
        """Picking, say, notepad.exe must say so rather than crash."""
        fake = tmp_path / "notpython.sh"
        fake.write_text("#!/bin/sh\necho hello\n")
        fake.chmod(0o755)
        probe = bs.probe_interpreter(str(fake))
        assert probe["ok"] is False
        assert "does not look like a Python interpreter" in probe["error"]

    def test_describe_is_a_single_readable_line(self):
        probe = bs.probe_interpreter(sys.executable)
        text = bs.describe_interpreter(probe)
        assert "\n" not in text and "Python 3." in text

    def test_describe_reports_a_cuda_device(self):
        text = bs.describe_interpreter({
            "ok": True, "python_version": "3.11.9", "torch": "2.9.0",
            "cuda": True, "device_name": "NVIDIA GeForce RTX 2080 Ti",
            "capability": [7, 5], "sam3gimpd": "0.1.0"})
        assert "RTX 2080 Ti" in text and "sm_75" in text and "sam3gimpd 0.1.0" in text

    def test_no_torch_is_a_blocking_problem(self):
        problems = bs.interpreter_problems({
            "ok": True, "python_version": "3.11.9", "torch": None, "sam3gimpd": None})
        assert any("No torch" in p for p in problems)

    def test_cpu_only_torch_warns_but_does_not_block(self):
        problems = bs.interpreter_problems({
            "ok": True, "python_version": "3.11.9", "torch": "2.9.0",
            "cuda": False, "sam3gimpd": "0.1.0"})
        assert any("no CUDA device" in p for p in problems)
        assert not any("No torch" in p for p in problems)

    def test_an_old_python_is_flagged(self):
        problems = bs.interpreter_problems({
            "ok": True, "python_version": "3.8.10", "torch": "2.9.0",
            "cuda": True, "sam3gimpd": "0.1.0"})
        assert any("too old" in p for p in problems)

    def test_python_39_is_flagged_because_the_daemon_will_not_install_there(self):
        """The gate said 3.9 while ``requires-python`` said 3.10.

        A 3.9 environment passed Check, and the failure surfaced two clicks
        later as pip's "requires a different Python" under *Install sam3gimpd
        here* -- our dialog having just told the user it was fine.
        """
        problems = bs.interpreter_problems({
            "ok": True, "python_version": "3.9.18", "torch": "2.9.0",
            "cuda": True, "sam3gimpd": "0.1.0"})
        assert any("too old" in p for p in problems), problems

    def test_the_floor_itself_is_accepted(self):
        assert bs.interpreter_problems({
            "ok": True, "python_version": "%d.%d.0" % bs.MIN_INTERPRETER,
            "torch": "2.9.0", "cuda": True, "transformers": "5.16.1",
            "sam3gimpd": "0.1.0"}) == []

    def test_a_python_newer_than_anything_we_pin_is_not_flagged(self):
        """No upper bound: the environment a user already has is usually the
        newest one they installed, and the daemon is pure standard library."""
        assert bs.interpreter_problems({
            "ok": True, "python_version": "3.14.5", "torch": "2.9.0",
            "cuda": True, "transformers": "5.16.1", "sam3gimpd": "0.1.0"}) == []

    def test_a_fully_ready_environment_has_no_problems(self):
        assert bs.interpreter_problems({
            "ok": True, "python_version": "3.11.9", "torch": "2.9.0",
            "cuda": True, "device_name": "RTX 4090",
            "transformers": "5.0.0", "sam3gimpd": "0.1.0"}) == []

    def test_install_command_pulls_runtime_but_not_torch(self, monkeypatch):
        """It must pull transformers while leaving an existing torch alone.

        This asserted "no extras at all" until an environment with torch and no
        transformers reached a user: the daemon started, saw CUDA, and failed
        every encode with engine_unavailable.
        """
        monkeypatch.setattr(bs, "find_uv", lambda: None)
        argv = bs.install_sam3d_command("/usr/bin/python3")
        assert argv[:5] == ["/usr/bin/python3", "-m", "pip", "install", "--upgrade"]
        assert argv[-1].endswith("[runtime]")
        assert "cuda" not in argv[-1] and "cpu" not in argv[-1]

    def test_install_command_uses_uv_when_there_is_one(self, monkeypatch):
        """``uv venv`` makes environments without pip: "No module named pip"."""
        monkeypatch.setattr(bs, "find_uv", lambda: "/tools/uv")
        argv = bs.install_sam3d_command("/env/bin/python")
        assert argv[:5] == ["/tools/uv", "pip", "install", "--python", "/env/bin/python"]
        assert argv[argv.index("--reinstall-package") + 1] == bs.DAEMON_DIST_NAME
        assert "--upgrade" not in argv
        assert argv[-1].endswith("[runtime]")

    def test_pip_is_used_only_when_there_is_no_uv_and_pip_exists(self, sam3_home, monkeypatch):
        monkeypatch.setattr(bs, "find_uv", lambda: None)
        runner = FakeRunner(default=(0, ["pip 24.0"]))
        argv = bs.resolve_daemon_install_command("/env/bin/python", runner=runner)
        assert runner.argvs == [["/env/bin/python", "-m", "pip", "--version"]]
        assert argv[:4] == ["/env/bin/python", "-m", "pip", "install"]

    def test_no_uv_and_no_pip_fetches_the_pinned_uv(self, sam3_home, monkeypatch, pinned_fake_uv):
        monkeypatch.setattr(bs, "find_uv", lambda: None)
        runner = FakeRunner(default=(1, ["No module named pip"]))
        dl = FakeDownloader({bs.uv_download_url(): pinned_fake_uv})
        argv = bs.resolve_daemon_install_command("/env/bin/python", runner=runner, downloader=dl)
        assert dl.urls == [bs.uv_download_url()]
        assert argv[:5] == [bs.uv_binary(), "pip", "install", "--python", "/env/bin/python"]
        assert os.path.isfile(bs.uv_binary())

    def test_a_real_uv_venv_really_has_no_pip(self, tmp_path):
        """The premise, checked whenever uv is on PATH."""
        uv = shutil.which("uv")
        if not uv:
            pytest.skip("uv is not installed here")
        venv = tmp_path / "v"
        made = subprocess.run([uv, "venv", "--python", sys.executable, str(venv)],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        if made.returncode != 0:
            pytest.skip("uv could not make a venv here")
        py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        assert not bs.interpreter_has_pip(str(py))

    def test_install_command_uses_a_path_never_a_bare_name(self):
        """Regression, and a nasty one.

        ``sam3d`` on PyPI is an unrelated project ("Unified Python interface for
        Meta SAM 3D Body and SAM 3D Objects").  Installing by bare name fetched
        *that* -- and in this flow it goes into the user's own PyTorch
        environment.  The failure surfaced only as ``ModuleNotFoundError: No
        module named 'pydantic'`` raised from a stranger's ``service.py``.
        """
        requirement = bs.install_sam3d_command("/usr/bin/python3")[-1]
        source = requirement.split("[")[0]
        assert os.path.isabs(source), source
        assert os.path.isfile(os.path.join(source, "pyproject.toml"))

    def test_install_command_honours_a_local_source(self, monkeypatch):
        monkeypatch.setenv("SAM3D_SOURCE", "/src/daemon")
        assert bs.install_sam3d_command("/usr/bin/python3")[-1] == "/src/daemon[runtime]"


class TestDaemonSourceRepair:
    """A plug-in tree copied by hand has no bundled ``_daemon/``.

    That used to be a dead end you could only escape from a terminal; Setup can
    now repair it with a folder chooser.
    """

    def test_validates_the_real_daemon_directory(self):
        from pathlib import Path

        repo = Path(__file__).resolve().parents[2]
        assert bs.validate_daemon_source(
            str(repo / "plugin" / "sam3_gimp" / "_daemon"))["ok"]

    def test_rejects_a_folder_with_no_pyproject(self, tmp_path):
        result = bs.validate_daemon_source(str(tmp_path))
        assert not result["ok"] and "pyproject.toml" in result["error"]

    def test_rejects_a_pyproject_that_is_not_ours(self, tmp_path):
        """A random Python project must not be accepted as the daemon."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname='something'\n")
        result = bs.validate_daemon_source(str(tmp_path))
        assert not result["ok"] and "sam3gimpd" in result["error"]

    def test_rejects_a_missing_folder(self):
        assert not bs.validate_daemon_source("/no/such/folder")["ok"]

    def test_rejects_an_empty_selection(self):
        assert not bs.validate_daemon_source("")["ok"]

    def test_install_copies_the_tree_and_skips_junk(self, tmp_path, monkeypatch):
        from pathlib import Path

        repo = Path(__file__).resolve().parents[2]
        dest = tmp_path / "_daemon"
        monkeypatch.setattr(bs, "bundled_daemon_dir", lambda: str(dest))
        result = bs.install_daemon_source(
            str(repo / "plugin" / "sam3_gimp" / "_daemon"))
        assert result["ok"] and result["copied"]
        assert (dest / "pyproject.toml").is_file()
        assert (dest / "sam3gimpd" / "__init__.py").is_file()
        assert not list(dest.rglob("__pycache__"))

    def test_install_refuses_a_folder_that_is_not_the_daemon(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bs, "bundled_daemon_dir", lambda: str(tmp_path / "_daemon"))
        assert not bs.install_daemon_source(str(tmp_path))["ok"]

    def test_a_repaired_tree_then_resolves(self, tmp_path, monkeypatch):
        from pathlib import Path

        repo = Path(__file__).resolve().parents[2]
        dest = tmp_path / "_daemon"
        monkeypatch.setattr(bs, "bundled_daemon_dir", lambda: str(dest))
        bs.install_daemon_source(str(repo / "plugin" / "sam3_gimp" / "_daemon"))
        # daemon_source() looks beside bootstrap.py, so assert the copy directly
        assert bs.validate_daemon_source(str(dest))["ok"]

    def test_status_reports_ok_in_this_checkout(self):
        assert bs.daemon_source_status()["ok"]

    def test_the_bundled_daemon_outranks_a_remembered_path(self, sam3_home, tmp_path, monkeypatch):
        """The remembered path rescues a tree with no _daemon.  Once a proper
        release is extracted, an old checkout it still names must not keep
        feeding Update daemon and the build-hash check."""
        monkeypatch.delenv("SAM3D_SOURCE", raising=False)
        old = tmp_path / "old-checkout" / "_daemon"
        (old / "sam3gimpd").mkdir(parents=True)
        (old / "pyproject.toml").write_text("[project]\nname='sam3-gimp-daemon'\n")
        (old / "sam3gimpd" / "__init__.py").write_text("")
        bs._remember_daemon_source(str(old))
        assert bs.daemon_source() == bs.bundled_daemon_dir()

        # ...and it is still used when there is nothing bundled.
        monkeypatch.setattr(bs, "bundled_daemon_dir", lambda: str(tmp_path / "missing"))
        assert bs.daemon_source() == str(old)


class TestExternalEnvironmentCountsAsReady:
    """Choosing your own PyTorch environment must satisfy the readiness check.

    It did not: ``env_ready`` only looked at the managed venv, so a user who
    pointed the plug-in at their own interpreter was told to run Setup for ever
    -- the venv Setup would have built did not exist and never would. Setup
    reopened every time they tried to segment.
    """

    def _report(self, **kw):
        base = dict(
            platform="windows", machine="x86_64", base="/b",
            accelerator=bs.Accelerator("cuda", "t"),
            uv_present=False, venv_present=False, sam3d_present=False,
            weights_present=True, weights_source=None, daemon_running=False,
            daemon_pid=None, daemon_port=None, runtime_present=False,
            crash_log_present=False, state=bs.InstallState(),
        )
        base.update(kw)
        return bs.EnvironmentReport(**base)

    def test_managed_venv_alone_is_ready(self):
        r = self._report(uv_present=True, venv_present=True, sam3d_present=True)
        assert r.managed_ready and r.env_ready

    def test_external_interpreter_alone_is_ready(self):
        r = self._report(external_python="/py", external_has_daemon=True)
        assert not r.managed_ready
        assert r.external_ready and r.env_ready

    def test_external_without_the_daemon_is_not_ready(self):
        r = self._report(external_python="/py", external_has_daemon=False)
        assert not r.external_ready and not r.env_ready

    def test_neither_is_not_ready(self):
        assert not self._report().env_ready

    def test_summary_says_which_environment_is_in_use(self):
        r = self._report(external_python="/py", external_has_daemon=True)
        assert "your own environment" in r.summary()

    def test_summary_points_at_the_install_button_when_the_daemon_is_absent(self):
        r = self._report(external_python="/py", external_has_daemon=False)
        assert "Install sam3gimpd here" in r.summary()

    def test_missing_does_not_list_the_managed_venv_when_external(self):
        """Reporting 'uv, venv' missing while using another environment is noise."""
        r = self._report(external_python="/py", external_has_daemon=True)
        assert "uv" not in r.missing and "venv" not in r.missing

    def test_a_deleted_interpreter_falls_back(self, sam3_home):
        import launcher as L

        L.set_configured_python("/definitely/not/here/python", has_daemon=True)
        assert bs._stored_python() == (None, False)
        assert not bs.inspect_environment().external_ready

    def test_the_flag_round_trips_through_settings(self, sam3_home, tmp_path):
        import launcher as L

        exe = tmp_path / "python"
        exe.write_text("#!/bin/sh\n")
        L.set_configured_python(str(exe), has_daemon=True)
        assert bs._stored_python() == (str(exe), True)
        L.set_configured_python(str(exe), has_daemon=False)
        assert bs._stored_python() == (str(exe), False)

    def test_clearing_removes_both_keys(self, sam3_home, tmp_path):
        import launcher as L

        exe = tmp_path / "python"
        exe.write_text("#!/bin/sh\n")
        L.set_configured_python(str(exe), has_daemon=True)
        L.set_configured_python(None)
        assert "python_has_daemon" not in L.read_settings()


class TestOriginalCheckpointConversion:
    """A third exit from the gated flow, beside the token and the HF folder.

    ``sam3.pt`` is what Meta ships (and sits in the gated repo beside the
    converted files). It is not loadable by ``Sam3Model.from_pretrained``: the
    tensor names differ and there is no config or tokenizer next to it. But
    ``transformers``' converter needs *only* the .pt -- it builds the config and
    takes the tokenizer from a public CLIP repo -- so a user holding one needs
    no HuggingFace account at all.
    """

    def test_finds_a_checkpoint_given_the_file(self, tmp_path):
        pt = tmp_path / "sam3.pt"
        pt.write_bytes(b"")
        assert bs.find_original_checkpoint(str(pt)) == str(pt)

    def test_finds_a_checkpoint_given_its_folder(self, tmp_path):
        (tmp_path / "sam3.pt").write_bytes(b"")
        assert bs.find_original_checkpoint(str(tmp_path)).endswith("sam3.pt")

    def test_finds_a_single_differently_named_pt(self, tmp_path):
        (tmp_path / "my-checkpoint.pt").write_bytes(b"")
        assert bs.find_original_checkpoint(str(tmp_path)).endswith("my-checkpoint.pt")

    def test_refuses_to_guess_between_several(self, tmp_path):
        (tmp_path / "a.pt").write_bytes(b"")
        (tmp_path / "b.pt").write_bytes(b"")
        assert bs.find_original_checkpoint(str(tmp_path)) is None

    def test_no_checkpoint_is_none(self, tmp_path):
        assert bs.find_original_checkpoint(str(tmp_path)) is None
        assert bs.find_original_checkpoint("") is None

    def test_validator_reports_a_pt_as_convertible_not_as_junk(self, tmp_path):
        """It used to say "no weights here" about a folder full of weights."""
        (tmp_path / "sam3.pt").write_bytes(b"")
        report = bs.inspect_local_weights(str(tmp_path))
        assert not report["ok"]
        assert report.get("convertible") is True
        assert "can be converted" in report["message"]
        assert "no HuggingFace account" in report["message"]

    def test_a_real_hf_folder_is_still_accepted(self, tmp_path):
        (tmp_path / "model.safetensors").write_bytes(b"")
        (tmp_path / "config.json").write_text("{}")
        assert bs.inspect_local_weights(str(tmp_path))["ok"]

    def test_conversion_command_shape(self):
        argv = bs.convert_checkpoint_command("/py", "/w/sam3.pt", "/out")
        assert argv == ["/py", "-", "--checkpoint_path", "/w/sam3.pt", "--output_path", "/out"]

    def test_missing_checkpoint_is_a_message_not_a_raise(self):
        result = bs.convert_original_checkpoint("/definitely/not/here.pt")
        assert not result["ok"] and "No such checkpoint" in result["message"]

    def test_no_torch_environment_is_a_message_not_a_raise(self, sam3_home, tmp_path):
        pt = tmp_path / "sam3.pt"
        pt.write_bytes(b"")
        result = bs.convert_original_checkpoint(str(pt), python=None)
        assert not result["ok"]
        assert "environment" in result["message"]

    def test_a_failed_conversion_reports_the_tail(self, sam3_home, tmp_path, pinned_converter):
        pt = tmp_path / "sam3.pt"
        pt.write_bytes(b"")
        runner = ConversionRunner(convert=(3, ["boom", "it broke"]))
        result = bs.convert_original_checkpoint(
            str(pt), python=fake_python(tmp_path), runner=runner, out_dir=str(tmp_path / "out"),
            opener=pinned_converter)
        assert not result["ok"]
        assert "exit 3" in result["message"] and "it broke" in result["message"]
        # What ran is exactly the verified source, from stdin.
        assert runner.converted == [FAKE_CONVERTER.decode("utf-8")]

    def test_an_old_torch_is_refused_before_anything_runs(self, sam3_home, tmp_path, pinned_converter):
        """The converter's torch.load has no weights_only; before torch 2.6
        that unpickles arbitrary objects, so a doctored .pt runs code."""
        pt = tmp_path / "sam3.pt"
        pt.write_bytes(b"")
        runner = ConversionRunner(torch="2.5.1+cu121")
        result = bs.convert_original_checkpoint(str(pt), python=fake_python(tmp_path), runner=runner,
                                                opener=pinned_converter)
        assert not result["ok"] and "2.5.1" in result["message"] and "2.6" in result["message"]
        assert runner.converted == []

    def test_a_current_torch_is_accepted(self, sam3_home, tmp_path, pinned_converter):
        pt = tmp_path / "sam3.pt"
        pt.write_bytes(b"")
        runner = ConversionRunner(torch="2.10.0+cu128")
        bs.convert_original_checkpoint(str(pt), python=fake_python(tmp_path), runner=runner,
                                       out_dir=str(tmp_path / "out"), opener=pinned_converter)
        assert len(runner.converted) == 1

    def test_an_interpreter_that_cannot_be_inspected_is_refused(self, sam3_home, tmp_path,
                                                                 pinned_converter):
        pt = tmp_path / "sam3.pt"
        pt.write_bytes(b"")
        runner = ConversionRunner(probe_ok=False)
        result = bs.convert_original_checkpoint(str(pt), python=fake_python(tmp_path), runner=runner,
                                                opener=pinned_converter)
        assert not result["ok"] and "nothing was run" in result["message"]
        assert runner.converted == []

    def test_daemon_interpreter_prefers_the_users_choice(self, sam3_home, tmp_path):
        import launcher as L

        exe = tmp_path / "python"
        exe.write_text("#!/bin/sh\n")
        L.set_configured_python(str(exe), has_daemon=True)
        assert bs.daemon_interpreter() == str(exe)


class TestTransformersIsLoadBearing:
    """An environment with torch but no transformers is not usable.

    The "install into your own environment" flow used no extra at all, to leave
    a working CUDA torch alone -- and skipped transformers with it. The daemon
    then started, reported CUDA and a 2080 Ti, and failed every encode with
    ``engine_unavailable: transformers is not installed``. Worse, the plug-in
    counted that environment as set up, so the failure only surfaced after the
    user had typed a prompt.
    """

    #: Exactly what the affected machine reported.
    TORCH_NO_TRANSFORMERS = {
        "ok": True, "path": "/py", "python_version": "3.12.10",
        "torch": "2.10.0+cu128", "cuda": True,
        "device_name": "NVIDIA GeForce RTX 2080 Ti", "capability": [7, 5],
        "transformers": None, "sam3gimpd": "0.1.0",
    }

    def test_that_environment_cannot_serve(self):
        assert not bs.interpreter_can_serve(self.TORCH_NO_TRANSFORMERS)

    def test_the_same_environment_with_transformers_can(self):
        ok = dict(self.TORCH_NO_TRANSFORMERS, transformers="5.0.0")
        assert bs.interpreter_can_serve(ok)

    def test_no_daemon_cannot_serve(self):
        assert not bs.interpreter_can_serve(
            dict(self.TORCH_NO_TRANSFORMERS, transformers="5.0.0", sam3gimpd=None))

    def test_the_problem_names_the_button_that_fixes_it(self):
        problems = bs.interpreter_problems(self.TORCH_NO_TRANSFORMERS)
        assert any("transformers is missing" in p for p in problems)
        assert any("Install sam3gimpd here" in p for p in problems)

    def test_the_summary_says_transformers_is_absent(self):
        text = bs.describe_interpreter(self.TORCH_NO_TRANSFORMERS)
        assert "no transformers" in text
        assert "2080 Ti" in text and "sm_75" in text

    def test_install_pulls_the_runtime_extra(self):
        """Without it transformers is silently skipped."""
        argv = bs.install_sam3d_command("/py")
        assert argv[-1].endswith("[runtime]")

    def test_the_runtime_extra_excludes_torch(self):
        """The whole point: never disturb an existing CUDA install."""
        from pathlib import Path

        text = (Path(bs.daemon_source()) / "pyproject.toml").read_text(encoding="utf-8")
        block = text.split("runtime = [")[1].split("]")[0]
        assert "transformers" in block
        assert "torch" not in block and "torchvision" not in block

    def test_a_probe_that_failed_outright_cannot_serve(self):
        assert not bs.interpreter_can_serve({"ok": False, "error": "nope"})


class TestInstallButtonIsReachable:
    """The row must never present a diagnosis with no pressable action.

    An environment with the daemon but no transformers displayed "press
    'Install sam3gimpd here' again" above that very button, greyed out --
    because the button keyed on "sam3gimpd missing" -- while "Use this
    environment" was also disabled. Nothing on the row could be pressed.
    """

    BASE = {"ok": True, "path": "/py", "python_version": "3.12.10",
            "torch": "2.10.0+cu128", "cuda": True,
            "device_name": "RTX 2080 Ti", "capability": [7, 5]}

    @staticmethod
    def _install_enabled(probe):
        # Mirrors ui/setup_dialog._on_python_checked.
        return (bool(probe.get("ok")) and bool(probe.get("torch"))
                and not bs.interpreter_can_serve(probe))

    def test_enabled_when_transformers_is_missing(self):
        probe = dict(self.BASE, transformers=None, sam3gimpd="0.1.0")
        assert self._install_enabled(probe)

    def test_enabled_when_the_daemon_is_missing(self):
        probe = dict(self.BASE, transformers="5.0.0", sam3gimpd=None)
        assert self._install_enabled(probe)

    def test_disabled_once_the_environment_can_serve(self):
        probe = dict(self.BASE, transformers="5.0.0", sam3gimpd="0.1.0")
        assert not self._install_enabled(probe)

    def test_disabled_without_torch_which_we_cannot_supply(self):
        probe = dict(self.BASE, torch=None, transformers=None, sam3gimpd=None)
        assert not self._install_enabled(probe)

    def test_every_unusable_state_offers_some_action(self):
        """No dead ends: if it cannot serve, either install or torch is at fault
        and the problems list says which."""
        for transformers in (None, "5.0.0"):
            for daemon in (None, "0.1.0"):
                probe = dict(self.BASE, transformers=transformers, sam3gimpd=daemon)
                if bs.interpreter_can_serve(probe):
                    continue
                assert self._install_enabled(probe), probe
                assert bs.interpreter_problems(probe)


class TestConverterIsFetched:
    """``convert_sam3_to_hf.py`` is in no released transformers.

    The release process strips conversion scripts -- the file is absent even
    from the ``v5.16.1`` tag's tree -- so no installed wheel contains it and it
    has to be fetched.  What is fetched is one pinned commit's file, and what
    runs is only ever bytes with the pinned SHA-256.
    """

    def test_the_url_is_pinned_to_a_commit_not_a_branch(self):
        """It was ``main``: whatever was on main when the user pressed
        Convert ran as them, and was then cached for good."""
        assert bs.CONVERT_SCRIPT_URL == (
            "https://raw.githubusercontent.com/huggingface/transformers/%s/"
            "src/transformers/models/sam3/convert_sam3_to_hf.py" % bs.CONVERT_SCRIPT_COMMIT)
        assert len(bs.CONVERT_SCRIPT_COMMIT) == 40 and int(bs.CONVERT_SCRIPT_COMMIT, 16) >= 0
        assert "/main/" not in bs.CONVERT_SCRIPT_URL
        assert len(bs.CONVERT_SCRIPT_SHA256) == 64 and int(bs.CONVERT_SCRIPT_SHA256, 16) >= 0

    def test_command_runs_an_explicit_script_when_given_one(self):
        argv = bs.convert_checkpoint_command("/py", "/w/sam3.pt", "/out", "/t/conv.py")
        assert argv[:2] == ["/py", "/t/conv.py"]
        assert argv[-4:] == ["--checkpoint_path", "/w/sam3.pt", "--output_path", "/out"]

    def test_a_verified_cache_is_reused_without_network(self, sam3_home, pinned_converter):
        os.makedirs(os.path.dirname(bs.convert_script_cache()), exist_ok=True)
        Path(bs.convert_script_cache()).write_bytes(FAKE_CONVERTER)
        result = bs.convert_script_source(opener=pinned_converter)
        assert result["ok"] and result["source"] == FAKE_CONVERTER.decode()
        assert pinned_converter.calls == []

    def test_a_tampered_cache_is_fetched_again(self, sam3_home, pinned_converter):
        """The cache was trusted on size alone, for ever."""
        os.makedirs(os.path.dirname(bs.convert_script_cache()), exist_ok=True)
        Path(bs.convert_script_cache()).write_bytes(b"import os\n" + b"#" * 4096)
        result = bs.convert_script_source(opener=pinned_converter)
        assert result["ok"] and result["source"] == FAKE_CONVERTER.decode()
        assert len(pinned_converter.calls) == 1
        assert Path(bs.convert_script_cache()).read_bytes() == FAKE_CONVERTER

    def test_a_tampered_cache_and_a_bad_download_fail_closed(self, sam3_home, pinned_converter):
        os.makedirs(os.path.dirname(bs.convert_script_cache()), exist_ok=True)
        Path(bs.convert_script_cache()).write_bytes(b"import os\n" + b"#" * 4096)
        result = bs.convert_script_source(opener=lambda url: b"print('not it')\n" * 200)
        assert not result["ok"] and result["source"] is None
        assert "checksum" in result["message"]
        assert not os.path.exists(bs.convert_script_cache()), "the bad cache is dropped"

    def test_a_failed_fetch_is_a_message_not_a_raise(self, sam3_home):
        def boom(url):
            raise OSError("no network")

        result = bs.convert_script_source(opener=boom)
        assert not result["ok"] and "no network" in result["message"]
        assert bs.CONVERT_SCRIPT_URL in result["message"]


class TestConverterFetchIsVerified:
    """Nothing reaches ``python -`` unless its SHA-256 is the pinned one."""

    def test_a_substituted_script_is_never_run(self, sam3_home, tmp_path, monkeypatch):
        """The reviewer's proof of concept: raw.githubusercontent.com serving a
        changed file.  It ran as the user and was cached; now it is refused
        and nothing is cached."""
        payload = (b"import os\nopen(%r, 'w').write('pwned')\n" % str(tmp_path / "PWNED")
                   + b"#" * 2000 + b"\n")
        pt = tmp_path / "sam3.pt"
        pt.write_bytes(b"not a checkpoint")
        runner = ConversionRunner()
        result = bs.convert_original_checkpoint(
            str(pt), python=fake_python(tmp_path), runner=runner, opener=lambda url: payload)
        assert not result["ok"] and "checksum" in result["message"]
        assert runner.converted == []
        assert not (tmp_path / "PWNED").exists()
        assert not os.path.exists(bs.convert_script_cache())

    def test_the_real_pinned_file_is_what_the_hash_names(self, sam3_home):
        """A truncated or empty body fails the hash as surely as a hostile one."""
        for body in (b"", b"x", FAKE_CONVERTER[:-1]):
            assert not bs.convert_script_source(opener=lambda url, b=body: b)["ok"]

    def test_a_successful_fetch_lands_in_the_cache(self, sam3_home, pinned_converter):
        result = bs.convert_script_source(opener=pinned_converter)
        assert result["ok"]
        assert Path(bs.convert_script_cache()).read_bytes() == FAKE_CONVERTER
        if os.name != "nt":
            assert stat.S_IMODE(os.stat(bs.convert_script_cache()).st_mode) == 0o600
        # No temporary file beside it.
        assert os.listdir(os.path.dirname(bs.convert_script_cache())) == [
            os.path.basename(bs.convert_script_cache())]

    def test_every_transformers_group_also_pulls_regex(self):
        """The converter does `import regex as re`, and transformers 5 does not
        always bring it -- the failure after the path bug was fixed."""
        from pathlib import Path

        text = (Path(bs.daemon_source()) / "pyproject.toml").read_text(encoding="utf-8")
        for group in ("cuda", "cpu", "mps", "runtime"):
            block = text.split(group + " = [")[1].split("]")[0]
            assert "transformers" in block
            assert "regex" in block, group


class TestConversionShortCircuits:
    """A .pt inside a downloaded snapshot needs no conversion at all.

    ``huggingface/hub/models--facebook--sam3/snapshots/<rev>/`` holds sam3.pt
    beside config.json, model.safetensors and the tokenizer, so a user who
    points at the .pt in their own cache already has everything. Converting it
    would spend half an hour rebuilding files sitting in the same folder.
    """

    @staticmethod
    def _snapshot(tmp_path):
        snap = tmp_path / "hub" / "models--facebook--sam3" / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        for name in ("config.json", "processor_config.json", "tokenizer.json"):
            (snap / name).write_text("{}")
        (snap / "model.safetensors").write_bytes(b"")
        (snap / "sam3.pt").write_bytes(b"")
        return snap

    def test_a_pt_inside_a_snapshot_is_adopted_not_converted(self, sam3_home, tmp_path):
        snap = self._snapshot(tmp_path)

        class Boom:
            def run(self, *a, **k):
                raise AssertionError("must not run the converter")

        result = bs.convert_original_checkpoint(str(snap / "sam3.pt"), runner=Boom())
        assert result["ok"]
        assert "No conversion needed" in result["message"]
        assert result["path"] == str(snap)

    def test_it_becomes_the_recorded_checkpoint(self, sam3_home, tmp_path):
        snap = self._snapshot(tmp_path)
        bs.convert_original_checkpoint(str(snap / "sam3.pt"))
        assert bs.local_weights_path() == str(snap)
        assert bs.weights_present()

    def test_a_lone_pt_still_converts(self, sam3_home, tmp_path):
        """The short circuit must not swallow the case it was built for."""
        pt = tmp_path / "sam3.pt"
        pt.write_bytes(b"")
        result = bs.convert_original_checkpoint(str(pt), python=None)
        assert not result["ok"]
        assert "No conversion needed" not in result["message"]


class TestConverterRunsFromStdin:
    """The program is piped, not opened from a path.

    Handing a filename meant the interpreter could fail to open a file that had
    just been written and verified -- "can't open file ... [Errno 2]", reported
    twice. Nothing on disk now has to survive between checking and running.
    """

    def test_command_reads_the_program_from_stdin(self):
        argv = bs.convert_checkpoint_command("/py", "/w/sam3.pt", "/out", "-")
        assert argv[:2] == ["/py", "-"]
        assert argv[-4:] == ["--checkpoint_path", "/w/sam3.pt", "--output_path", "/out"]

    def test_runner_can_feed_a_program_to_stdin(self):
        result = bs.CommandRunner().run(
            [sys.executable, "-", "x"],
            input_text="import sys; print('got', sys.argv[1])")
        assert result.returncode == 0 and "got x" in result.text

    def test_source_is_returned_not_a_path(self, sam3_home, pinned_converter):
        result = bs.convert_script_source(opener=pinned_converter)
        assert result["ok"] and result["source"].startswith("# convert_sam3_to_hf.py")

    def test_a_deleted_cache_does_not_break_a_run(self, sam3_home, pinned_converter):
        """The exact hazard: the file is gone by the time python starts."""
        source = bs.convert_script_source(opener=pinned_converter)["source"]
        os.remove(bs.convert_script_cache())
        result = bs.CommandRunner().run([sys.executable, "-", "a"], input_text=source)
        assert result.returncode == 0 and "converted ['a']" in result.text


class TestWeightsPresenceIsNotJustADirectory:
    """"SAM 3 weights are available (hf)" was shown to a user who had only
    sam3.pt -- the original checkpoint, which Sam3Model cannot load.

    The check tested that the cache *directory* existed. An aborted download
    leaves that behind, and `hf download facebook/sam3 sam3.pt` creates a
    snapshot holding nothing else.
    """

    @staticmethod
    def _hub(tmp_path, monkeypatch, files):
        snap = tmp_path / "hub" / "models--facebook--sam3" / "snapshots" / "rev1"
        snap.mkdir(parents=True)
        for name, body in files.items():
            (snap / name).write_text(body) if name.endswith(".json") \
                else (snap / name).write_bytes(b"")
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        return snap

    def test_only_the_original_checkpoint_is_not_enough(self, sam3_home, tmp_path, monkeypatch):
        self._hub(tmp_path, monkeypatch, {"sam3.pt": ""})
        assert not bs.weights_present()

    def test_an_aborted_download_is_not_enough(self, sam3_home, tmp_path, monkeypatch):
        self._hub(tmp_path, monkeypatch, {"sam3.pt.incomplete": ""})
        assert not bs.weights_present()

    def test_an_empty_snapshot_tree_is_not_enough(self, sam3_home, tmp_path, monkeypatch):
        self._hub(tmp_path, monkeypatch, {})
        assert not bs.weights_present()

    def test_a_real_checkpoint_counts(self, sam3_home, tmp_path, monkeypatch):
        self._hub(tmp_path, monkeypatch,
                  {"model.safetensors": "", "config.json": "{}"})
        assert bs.weights_present()

    def test_any_revision_counts(self, sam3_home, tmp_path, monkeypatch):
        base = tmp_path / "hub" / "models--facebook--sam3" / "snapshots"
        (base / "old").mkdir(parents=True)
        (base / "old" / "sam3.pt").write_bytes(b"")
        (base / "new").mkdir(parents=True)
        (base / "new" / "model.safetensors").write_bytes(b"")
        (base / "new" / "config.json").write_text("{}")
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        assert bs.weights_present()

    def test_a_recorded_local_path_still_wins(self, sam3_home, tmp_path):
        folder = tmp_path / "ckpt"
        folder.mkdir()
        (folder / "model.safetensors").write_bytes(b"")
        (folder / "config.json").write_text("{}")
        bs.set_local_weights(str(folder))
        assert bs.weights_present()


class TestTransformersFloor:
    """The converter imports Sam3ImageProcessor, which 5.0.0 does not have."""

    def test_pinned_above_the_image_processor_floor(self):
        from pathlib import Path

        text = (Path(bs.daemon_source()) / "pyproject.toml").read_text(encoding="utf-8")
        assert "transformers==5.0.0" not in text, "5.0.0 lacks Sam3ImageProcessor"
        import re
        pins = set(re.findall(r'"transformers==([\d.]+)"', text))
        assert pins, "no transformers pin found"
        for pin in pins:
            major, minor = (int(p) for p in pin.split(".")[:2])
            assert (major, minor) >= (5, 8), pin



class TestInstallButtonState:
    """After an update there must be a way to reinstall the daemon."""

    BASE = {"ok": True, "python_version": "3.12.10", "torch": "2.10.0+cu128", "cuda": True}

    def test_serving_environment_still_offers_reinstall(self):
        enabled, label = bs.install_button_state(dict(self.BASE, transformers="5.16.1", sam3gimpd="0.1.0"))
        assert enabled and "Reinstall" in label

    def test_missing_transformers_offers_missing_packages(self):
        enabled, label = bs.install_button_state(dict(self.BASE, transformers=None, sam3gimpd="0.1.0"))
        assert enabled and label == "Install missing packages here"

    def test_no_daemon_offers_install(self):
        enabled, label = bs.install_button_state(dict(self.BASE, transformers="5.16.1", sam3gimpd=None))
        assert enabled and label == "Install sam3gimpd here"

    def test_no_torch_is_disabled(self):
        enabled, _ = bs.install_button_state(dict(self.BASE, torch=None))
        assert not enabled

    def test_failed_probe_is_disabled(self):
        enabled, _ = bs.install_button_state({"ok": False})
        assert not enabled


def test_reinstall_restarts_the_running_daemon():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "plugin" / "sam3_gimp" / "ui"
           / "setup_dialog.py").read_text(encoding="utf-8")
    body = src.split("def _on_install_sam3d_here(")[1].split("\n    def ")[0]
    assert "_restart_daemon_for_new_code()" in body
    helper = src.split("def _restart_daemon_for_new_code(")[1].split("\nclass ")[0]
    assert "shutdown_daemon(info, grace_ms=0)" in helper and "delete_runtime_file()" in helper



class TestDoctorFindsTheRealDaemon:
    """Doctor answered "not installed yet" to every external-environment user."""

    def test_uses_the_managed_console_script_when_nothing_was_chosen(self, sam3_home, monkeypatch):
        exe = os.path.join(str(sam3_home), "fake-sam3gimpd")
        with open(exe, "w") as fh:
            fh.write("")
        monkeypatch.setattr(bs, "sam3d_executable", lambda: exe)
        assert bs.doctor_argv() == [exe, "doctor"]

    def test_the_chosen_interpreter_wins_as_it_does_for_the_launcher(
            self, sam3_home, monkeypatch, tmp_path):
        """With both present, Doctor examined the managed venv while the
        launcher ran the interpreter the user chose."""
        import launcher as L

        exe = os.path.join(str(sam3_home), "fake-sam3gimpd")
        with open(exe, "w") as fh:
            fh.write("")
        monkeypatch.setattr(bs, "sam3d_executable", lambda: exe)
        monkeypatch.setattr(L, "sam3d_executable", lambda: exe)
        py = tmp_path / "python"; py.write_text("")
        L.set_configured_python(str(py), has_daemon=True)
        argv = bs.doctor_argv()
        assert argv == [str(py), "-m", "sam3gimpd", "doctor"]
        # The launcher's own resolution names the same interpreter.
        monkeypatch.delenv(L.ENV_COMMAND, raising=False)
        spawn = L.build_command()
        assert spawn[0] in (str(py), L.windowless_variant(str(py)))
        assert spawn[1:3] == ["-m", "sam3gimpd"]

    def test_falls_back_to_the_chosen_interpreter(self, sam3_home, monkeypatch, tmp_path):
        import launcher as L
        monkeypatch.setattr(bs, "sam3d_executable", lambda: str(tmp_path / "absent"))
        py = tmp_path / "python"; py.write_text("")
        L.set_configured_python(str(py), has_daemon=True)
        assert bs.doctor_argv() == [str(py), "-m", "sam3gimpd", "doctor"]

    def test_every_form_is_accepted_by_the_real_cli(self, sam3_home, monkeypatch, tmp_path):
        """``--json`` made argparse exit 2 before Doctor did anything."""
        import launcher as L

        exe = str(tmp_path / "sam3gimpd")
        Path(exe).write_text("")
        monkeypatch.setattr(bs, "sam3d_executable", lambda: exe)
        forms = [bs.doctor_argv()]
        py = tmp_path / "python"; py.write_text("")
        L.set_configured_python(str(py), has_daemon=True)
        forms.append(bs.doctor_argv())
        parser = _daemon_cli_parser()
        for argv in forms:
            assert parser.parse_args(_daemon_args(argv)).command == "doctor", argv

    def test_none_when_nothing_is_installed(self, sam3_home, monkeypatch, tmp_path):
        monkeypatch.setattr(bs, "sam3d_executable", lambda: str(tmp_path / "absent"))
        monkeypatch.setattr(bs, "daemon_interpreter", lambda: None)
        assert bs.doctor_argv() is None

    def test_run_doctor_uses_the_external_interpreter(self, sam3_home, monkeypatch, tmp_path):
        import launcher as L
        monkeypatch.setattr(bs, "sam3d_executable", lambda: str(tmp_path / "absent"))
        py = tmp_path / "python"; py.write_text("")
        L.set_configured_python(str(py), has_daemon=True)
        seen = []

        class Runner:
            def run(self, argv, **kw):
                seen.append(list(argv))
                return bs.CommandResult(argv=argv, returncode=0, lines=['{"device": "cuda"}'])

        report = bs.run_doctor(runner=Runner())
        assert seen and seen[0][:4] == [str(py), "-m", "sam3gimpd", "doctor"]
        assert "not installed" not in (report.get("message") or "")
