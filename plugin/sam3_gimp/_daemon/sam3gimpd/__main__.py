"""``python -m sam3gimpd`` -- the same entry point as the ``sam3gimpd`` console script.

The plug-in's launcher prefers the installed console script but falls back to
``<python> -m sam3gimpd`` (``API.md`` §13, ``launcher.build_command``), and
``tools/canvas_harness.py`` uses that spelling directly.  Both need this module
to exist; without it the fallback dies with "No module named sam3gimpd.__main__".

Nothing heavy is imported here: ``cli`` itself imports torch lazily, so
``python -m sam3gimpd serve --stub`` runs in an environment with no torch at all.
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":  # pragma: no cover -- exercised as a subprocess
    sys.exit(main())
