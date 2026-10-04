"""Shared pytest configuration for the whole repository.

Goals, in order:

1. ``python3 -m pytest`` from the repo root passes on a machine with **no GPU,
   no torch, no GIMP and no SAM 3 weights**, on any Python from 3.10 up.
2. Anything that needs hardware or software we do not have is marked and
   **skips** rather than failing or erroring at import time.
3. Tests never touch the user's real ``~/.local/share/sam3-gimp`` (or
   ``%LOCALAPPDATA%\\sam3-gimp``): the whole session runs against a temporary
   ``SAM3_GIMP_HOME``.

Markers (declared in ``pytest.ini``, auto-skipped here):

``needs_torch``   torch importable
``needs_gpu``     a real CUDA or MPS device (implies torch)
``needs_weights`` the gated ``facebook/sam3`` checkpoint present on disk
``needs_gimp``    ``gi.repository.Gimp`` importable (only true inside GIMP)
``needs_gtk``     GTK3 + PyGObject + a display
``needs_daemon``  spawns or talks to a real ``sam3gimpd`` process
``slow``          multi-second test

Import layout, mirrored from ``pytest.ini`` so that ``pytest tests/...`` works
even when invoked from a different rootdir:

* ``plugin/sam3_gimp/_daemon/`` -> ``import sam3gimpd``
* ``plugin/sam3_gimp/`` -> ``import client``, ``import launcher``, ``from ui import canvas``
* ``plugin/``           -> ``from sam3_gimp import ...``
* ``tests/``            -> shared helpers such as ``fake_gimp``
"""

from __future__ import annotations

import importlib.util
import os
import socket
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DAEMON_DIR = REPO_ROOT / "plugin" / "sam3_gimp" / "_daemon"
PLUGIN_DIR = REPO_ROOT / "plugin"
PLUGIN_PKG_DIR = PLUGIN_DIR / "sam3_gimp"
TESTS_DIR = REPO_ROOT / "tests"

for _p in (DAEMON_DIR, PLUGIN_PKG_DIR, PLUGIN_DIR, TESTS_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


# --------------------------------------------------------------------------- #
# capability probes (cached; each is answered exactly once per session)
# --------------------------------------------------------------------------- #
def _module_available(name: str) -> bool:
    """True if ``name`` can be imported without actually importing it.

    ``find_spec`` keeps a base install from paying torch's multi-second import
    just to decide that a test should skip.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


_CACHE: dict = {}


def _probe(key, fn):
    if key not in _CACHE:
        try:
            _CACHE[key] = bool(fn())
        except Exception:
            _CACHE[key] = False
    return _CACHE[key]


def have_torch() -> bool:
    return _probe("torch", lambda: _module_available("torch"))


def have_transformers() -> bool:
    return _probe("transformers", lambda: _module_available("transformers"))


def have_gpu() -> bool:
    def _check() -> bool:
        if not have_torch():
            return False
        import torch  # noqa: PLC0415  (deliberately lazy)

        if torch.cuda.is_available():
            return True
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        return bool(mps and mps.is_available())

    return _probe("gpu", _check)


def have_weights() -> bool:
    """True when the gated SAM 3 checkpoint appears to be cached locally.

    Cheap and heuristic on purpose: it looks for a ``models--facebook--sam3``
    directory under HF_HOME / the sam3-gimp cache, and never hits the network.
    """

    def _check() -> bool:
        roots = []
        for env in ("SAM3_WEIGHTS_DIR", "HF_HOME", "HUGGINGFACE_HUB_CACHE"):
            v = os.environ.get(env)
            if v:
                roots.append(Path(v))
        roots.append(Path(os.path.expanduser("~")) / ".cache" / "huggingface")
        for root in roots:
            if not root.exists():
                continue
            for pattern in ("models--facebook--sam3", "hub/models--facebook--sam3"):
                if (root / pattern).exists():
                    return True
        return False

    return _probe("weights", _check)


def have_gimp() -> bool:
    def _check() -> bool:
        if not _module_available("gi"):
            return False
        import gi  # noqa: PLC0415

        try:
            gi.require_version("Gimp", "3.0")
        except (ValueError, AttributeError):
            return False
        try:
            from gi.repository import Gimp  # noqa: F401,PLC0415
        except Exception:
            return False
        return True

    return _probe("gimp", _check)


def have_gtk() -> bool:
    """GTK3 + PyGObject + a usable display.

    True on any Linux desktop, and in CI under Xvfb, which is why the canvas
    can be developed and tested without GIMP or Windows.
    """

    def _check() -> bool:
        if not _module_available("gi"):
            return False
        if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            return False
        import gi  # noqa: PLC0415

        try:
            gi.require_version("Gtk", "3.0")
            from gi.repository import Gtk  # noqa: PLC0415
        except Exception:
            return False
        return bool(Gtk.init_check()[0]) if hasattr(Gtk, "init_check") else True

    return _probe("gtk", _check)


_MARKER_GUARDS = (
    ("needs_torch", have_torch, "torch is not installed"),
    ("needs_gpu", have_gpu, "no CUDA/MPS device available"),
    ("needs_weights", have_weights, "SAM 3 weights (gated) are not present"),
    ("needs_gimp", have_gimp, "gi.repository.Gimp unavailable (not running inside GIMP)"),
    ("needs_gtk", have_gtk, "GTK3/PyGObject or a display is unavailable"),
)


def pytest_configure(config: "pytest.Config") -> None:
    # Also registered in pytest.ini; repeated here so the suite still works when
    # pytest is invoked with a different rootdir or --strict-markers elsewhere.
    for name, help_text in (
        ("needs_torch", "requires torch"),
        ("needs_gpu", "requires a CUDA/MPS device"),
        ("needs_weights", "requires the gated facebook/sam3 checkpoint"),
        ("needs_gimp", "requires gi.repository.Gimp"),
        ("needs_gtk", "requires GTK3 + PyGObject + a display"),
        ("needs_daemon", "spawns or talks to a real sam3gimpd process"),
        ("slow", "takes more than a couple of seconds"),
    ):
        config.addinivalue_line("markers", "%s: %s" % (name, help_text))


def pytest_runtest_setup(item: "pytest.Item") -> None:
    for name, probe, reason in _MARKER_GUARDS:
        if item.get_closest_marker(name) is not None and not probe():
            pytest.skip(reason)


# --------------------------------------------------------------------------- #
# isolation: never write into the user's real data directory
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session", autouse=True)
def _isolated_sam3_home(tmp_path_factory: "pytest.TempPathFactory"):
    """Point ``SAM3_GIMP_HOME`` at a session-scoped temp dir for every test.

    A test that wants the *real* platform defaults (e.g. one asserting the
    Windows/XDG layout) should ``monkeypatch.delenv("SAM3_GIMP_HOME",
    raising=False)`` itself.
    """
    home = tmp_path_factory.mktemp("sam3-gimp-home")
    mp = pytest.MonkeyPatch()
    mp.setenv("SAM3_GIMP_HOME", str(home))
    mp.delenv("SAM3D_RUNTIME_FILE", raising=False)
    mp.setenv("HF_HUB_OFFLINE", os.environ.get("HF_HUB_OFFLINE", "1"))
    try:
        yield home
    finally:
        mp.undo()


# --------------------------------------------------------------------------- #
# path fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def daemon_dir() -> Path:
    return DAEMON_DIR


@pytest.fixture(scope="session")
def plugin_dir() -> Path:
    return PLUGIN_PKG_DIR


@pytest.fixture
def sam3_home(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> Path:
    """A fresh, empty ``SAM3_GIMP_HOME`` for one test."""
    home = tmp_path / "sam3-gimp"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SAM3_GIMP_HOME", str(home))
    monkeypatch.delenv("SAM3D_RUNTIME_FILE", raising=False)
    return home


# --------------------------------------------------------------------------- #
# capability booleans as fixtures (for skipif-style branching inside a test)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def torch_available() -> bool:
    return have_torch()


@pytest.fixture(scope="session")
def gpu_available() -> bool:
    return have_gpu()


@pytest.fixture(scope="session")
def gimp_available() -> bool:
    return have_gimp()


@pytest.fixture(scope="session")
def gtk_available() -> bool:
    return have_gtk()


# --------------------------------------------------------------------------- #
# small helpers every suite needs
# --------------------------------------------------------------------------- #
@pytest.fixture
def free_port() -> int:
    """An ephemeral port that was free a moment ago.

    Inherently racy; use it only where binding to port 0 is impossible.  The
    daemon itself always binds port 0 and reports the result in runtime.json.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


@pytest.fixture(scope="session")
def make_rgb():
    """Factory for deterministic raw RGB uint8 payloads.

    ``make_rgb(w, h, seed=0) -> bytes`` of length ``w * h * 3``, row-major,
    top-to-bottom, no padding -- exactly the ``POST /images`` body layout.
    The pattern is a smooth gradient plus a seed-dependent block so different
    seeds hash differently (the embedding cache is keyed by pixel hash).
    """

    def _make(width: int, height: int, seed: int = 0) -> bytes:
        buf = bytearray(width * height * 3)
        i = 0
        for y in range(height):
            for x in range(width):
                buf[i] = (x * 255) // max(1, width - 1) if width > 1 else 0
                buf[i + 1] = (y * 255) // max(1, height - 1) if height > 1 else 0
                buf[i + 2] = (x + y + seed * 37) & 0xFF
                i += 3
        return bytes(buf)

    return _make


@pytest.fixture
def rgb_image(make_rgb):
    """A small default upload: ``(width, height, pixels)`` = 64x48 RGB."""
    width, height = 64, 48
    return width, height, make_rgb(width, height)


@pytest.fixture
def auth_token() -> str:
    """A syntactically valid bearer token for tests that fake runtime.json."""
    return "t" * 43
