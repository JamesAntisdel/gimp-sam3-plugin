"""``fake_gimp`` -- run GIMP-facing plug-in code with no GIMP installed.

This package installs in-memory stand-ins for ``gi.repository.Gimp``,
``gi.repository.Gegl`` and ``gi.repository.Babl`` so that
``plugin/sam3_gimp/gimpbridge.py`` and ``plugin/sam3_gimp/outputs.py`` can be
imported and unit-tested without GIMP and without the Gimp/Gegl typelibs.
The real ``gi`` package is left alone -- ``GLib``, ``GObject`` and ``Gtk`` keep
working, so a GTK canvas test may run in the same session.

How to use it in a test module
------------------------------
Import the fixtures directly; pytest picks up fixture functions that are
present in a test module's namespace::

    from fake_gimp import fake_gimp, fake_image      # noqa: F401  (fixtures)

    def test_something(fake_gimp, fake_image):
        import gimpbridge
        Gimp = fake_gimp.Gimp                        # the stub module
        assert gimpbridge.gimp_available()
        ...

``fake_gimp`` (the fixture) yields the :mod:`fake_gimp.gimp_stubs` module, so
``fake_gimp.Gimp``, ``fake_gimp.Gegl``, ``fake_gimp.Rectangle``,
``fake_gimp.live_images()`` and ``fake_gimp.make_rgb_bytes()`` are all reachable
from the one fixture value.

``fake_image`` yields a :class:`~fake_gimp.gimp_stubs.Image`:
a 24x16 RGB image with two layers --

* ``"backdrop"``  24x16 at (0, 0), opaque, filled with mid gray (64, 64, 64)
* ``"patch"``      6x4  at (4, 3), opaque red (255, 0, 0), RGBA

so the projection differs from either layer alone and layer offsets matter.

Rebinding client modules
------------------------
``gimpbridge`` (and, by the same convention, ``outputs``) binds ``Gimp`` and
``Gegl`` as module globals at import time.  ``install()`` therefore calls
``module.rebind_gi()`` on every already-imported client module that exposes it,
and falls back to ``importlib.reload``.  Modules imported *after* ``install()``
pick up the stubs on their own.  Write client code that reads ``Gimp`` /
``Gegl`` as **module globals** (``gimpbridge.Gimp``), never as
``from gimpbridge import Gimp``, and this is automatic.

What the stubs are and are not
------------------------------
See the module docstring of :mod:`fake_gimp.gimp_stubs`.  The one rule worth
repeating: **the stub scaler is nearest-neighbour**, so tests may assert on
constant or blocky mask content but never on interpolated values.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import types

from . import gimp_stubs
from .gimp_stubs import (  # noqa: F401  (re-exported for convenience)
    Babl,
    Gegl,
    Gimp,
    MODULES,
    format_components,
    live_images,
    make_rgb_bytes,
    reset,
)

Rectangle = gimp_stubs.Rectangle
Color = gimp_stubs.Color
Buffer = gimp_stubs.Buffer

#: Client modules whose ``Gimp`` / ``Gegl`` globals are re-resolved on install.
DEFAULT_CLIENT_MODULES = ("gimpbridge", "outputs")

__all__ = [
    "gimp_stubs",
    "Gimp",
    "Gegl",
    "Babl",
    "Rectangle",
    "Color",
    "Buffer",
    "MODULES",
    "DEFAULT_CLIENT_MODULES",
    "install",
    "installed",
    "is_installed",
    "make_rgb_bytes",
    "live_images",
    "reset",
    "fake_gimp",
    "fake_image",
]


# --------------------------------------------------------------------------- #
# sys.modules surgery
# --------------------------------------------------------------------------- #
def is_installed() -> bool:
    return getattr(sys.modules.get("gi.repository.Gimp"), "__fake_gimp__", False)


def _rebind(module_names):
    """Point already-imported client modules at the freshly installed stubs."""
    for name in module_names:
        module = sys.modules.get(name)
        if module is None:
            continue
        rebind = getattr(module, "rebind_gi", None)
        if callable(rebind):
            rebind()
            continue
        try:
            importlib.reload(module)
        except Exception:
            # A half-reloaded module is worse than none: drop it so the next
            # `import` starts clean, then let the failure surface there.
            sys.modules.pop(name, None)
            raise


def install(client_modules=DEFAULT_CLIENT_MODULES):
    """Install the stubs into ``sys.modules``; returns an undo callable.

    Prefer the :func:`fake_gimp` fixture.  Use this directly only outside
    pytest (for example from ``tools/`` scripts).
    """
    saved = {name: sys.modules.get(name) for name in MODULES}
    saved_require = None
    synthesised = []
    try:
        import gi as gi_module
    except ImportError:
        # No PyGObject at all -- a bare CI runner, or a base `pip install`.
        # `gimpbridge`/`outputs` resolve their bindings with `import gi;
        # gi.require_version(...); from gi.repository import Gimp`, so a stub
        # dropped into `gi.repository` is invisible unless `gi` itself exists.
        # Synthesise the two shell modules; `DESIGN.md` §12 promises this suite
        # runs anywhere, and PyGObject is not part of what it is testing.
        gi_module = types.ModuleType("gi")
        gi_module.__fake_gimp__ = True
        gi_module.require_version = lambda namespace, version: None
        repository_module = types.ModuleType("gi.repository")
        repository_module.__fake_gimp__ = True
        gi_module.repository = repository_module
        saved["gi"] = sys.modules.get("gi")
        saved["gi.repository"] = sys.modules.get("gi.repository")
        sys.modules["gi"] = gi_module
        sys.modules["gi.repository"] = repository_module
        synthesised = ["gi", "gi.repository"]

    for name, module in MODULES.items():
        sys.modules[name] = module
    repository = sys.modules.get("gi.repository")
    if repository is not None:
        for name, module in MODULES.items():
            setattr(repository, name.rsplit(".", 1)[1], module)

    if gi_module is not None:
        saved_require = gi_module.require_version

        def require_version(namespace, version, _real=saved_require):
            if namespace in ("Gimp", "Gegl", "Babl"):
                return None
            return _real(namespace, version)

        gi_module.require_version = require_version

    gimp_stubs.reset()
    _rebind(client_modules)

    def undo():
        for name in synthesised:
            sys.modules.pop(name, None)
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
                if repository is not None and name.startswith("gi.repository."):
                    try:
                        delattr(repository, name.rsplit(".", 1)[1])
                    except AttributeError:
                        pass
            else:
                sys.modules[name] = module
                if repository is not None and name.startswith("gi.repository."):
                    setattr(repository, name.rsplit(".", 1)[1], module)
        if gi_module is not None and saved_require is not None:
            gi_module.require_version = saved_require
        _rebind(client_modules)

    return undo


@contextlib.contextmanager
def installed(client_modules=DEFAULT_CLIENT_MODULES):
    """Context manager form of :func:`install`, yielding :mod:`fake_gimp.gimp_stubs`."""
    undo = install(client_modules)
    try:
        yield gimp_stubs
    finally:
        undo()


# --------------------------------------------------------------------------- #
# pytest fixtures  (import them into your test module to use them)
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - pytest is always present in this repo's test runs
    import pytest
except ImportError:  # pragma: no cover
    pytest = None


if pytest is not None:

    @pytest.fixture
    def fake_gimp():
        """Install the GIMP/GEGL stubs for one test; yields ``fake_gimp.gimp_stubs``."""
        with installed() as stubs:
            yield stubs

    @pytest.fixture
    def fake_image(fake_gimp):
        """A 24x16 RGB image: gray ``backdrop`` plus a red ``patch`` at (4, 3).

        24x16 is the smallest size that needs no upload rescaling: both sides are
        >= the daemon's 16 px minimum and <= its 1008 px maximum.
        """
        stubs = fake_gimp
        image = stubs.Image.new(24, 16, stubs.ImageBaseType.RGB)

        backdrop = stubs.Layer.new(
            image, "backdrop", 24, 16, stubs.ImageType.RGB_IMAGE, 100.0,
            stubs.LayerMode.NORMAL,
        )
        backdrop.fill_bytes(bytes((64, 64, 64)) * (24 * 16))
        image.insert_layer(backdrop, None, 0)

        patch = stubs.Layer.new(
            image, "patch", 6, 4, stubs.ImageType.RGBA_IMAGE, 100.0,
            stubs.LayerMode.NORMAL,
        )
        patch.fill_bytes(bytes((255, 0, 0, 255)) * (6 * 4))
        patch.set_offsets(4, 3)
        image.insert_layer(patch, None, 0)

        image.set_selected_layers([patch])
        return image
