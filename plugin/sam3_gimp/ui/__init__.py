"""GTK3 user-interface widgets for the sam3-gimp plug-in.

Everything under ``plugin/`` runs inside GIMP's *embedded* Python interpreter,
so this package obeys the project's zero-dependency rule: **standard library
and ``gi`` only**.  No numpy, no requests, no pillow -- ever.

``cairo`` (pycairo) is the one apparent exception and is not really one: a GTK3
``draw`` handler is *handed* a ``cairo.Context``, and PyGObject cannot marshal
that struct at all unless pycairo is installed.  pycairo therefore ships as
part of the PyGObject stack that GIMP bundles; drawing anything in a Python
GTK3 plug-in requires it by construction.

Submodules
----------
``canvas``
    :class:`~ui.canvas.Sam3Canvas` -- the interactive preview widget: an image
    with translucent instance overlays, per-instance hover/selection,
    positive/negative point prompts, zoom/pan, client-side re-thresholding of
    the soft uint8 masks, and a marching-ants outline for the active mask.

The submodule is imported lazily so that merely importing ``ui`` does not pull
in GTK.
"""

from __future__ import annotations

__all__ = ["canvas", "Sam3Canvas"]


def __getattr__(name):  # PEP 562 lazy re-export
    # ``importlib.import_module`` rather than ``from . import canvas``: the
    # latter probes the package with ``hasattr``, which lands straight back in
    # this function and recurses.
    if name in ("canvas", "Sam3Canvas"):
        import importlib
        import sys

        module = importlib.import_module(__name__ + ".canvas")
        setattr(sys.modules[__name__], "canvas", module)
        return module if name == "canvas" else module.Sam3Canvas
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(__all__)
