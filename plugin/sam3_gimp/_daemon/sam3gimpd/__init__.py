"""sam3gimpd -- a standalone SAM 3 segmentation daemon.

``sam3gimpd`` is a pip-installable package that knows nothing about GIMP.  It serves
a small HTTP API (see ``API.md`` beside this package) over loopback, caches
image embeddings so that prompting is interactive, and runs one inference at a
time.

Importing this package MUST stay cheap and dependency-free: ``torch`` and
``transformers`` are imported lazily, inside the functions that need them, so
that the package imports on a machine with no GPU, no torch and no weights.
That is what makes ``--stub`` mode -- a first-class fake engine returning
structurally valid synthetic masks -- possible.

Only wire types and paths are re-exported here; server internals are imported
from their own modules to keep this module's import graph flat.
"""

from __future__ import annotations

from . import paths, types
from .types import (
    API_VERSION,
    ApiError,
    BBox,
    CanvasTransform,
    Engine,
    ErrorCode,
    ErrorInfo,
    JobState,
    JobStatus,
    Limits,
    MaskInstance,
    Point,
    PointPrompt,
    ResultHeader,
    RuntimeInfo,
    Size,
    TextPrompt,
    pack_result,
    unpack_result,
)

#: Package version.  Keep in sync with ``pyproject.toml`` beside this package.
__version__ = "0.1.1"

_BUILD_HASH = None


def build_hash() -> str:
    """Eight hex digits identifying the daemon *source files* actually running.

    ``__version__`` only changes when someone bumps it; during development the
    code changes far more often than that.  The plug-in ships a copy of this
    package and hashes it the same way (``bootstrap.bundled_daemon_build``), so
    a running daemon whose ``/hello`` build differs from the bundled copy is
    known to be stale and the dialog can say "update the daemon".  Only ``.py``
    files count, so a fresh ``__pycache__`` does not change the answer.

    Computed once and cached; the server takes it at startup, so it describes
    the files that were there when the daemon started, not later replacements.
    """
    global _BUILD_HASH
    if _BUILD_HASH is None:
        import hashlib  # noqa: PLC0415
        import os  # noqa: PLC0415

        root = os.path.dirname(os.path.abspath(__file__))
        digest = hashlib.sha1()
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for name in sorted(filenames):
                if not name.endswith(".py"):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, name), root).replace(os.sep, "/")
                digest.update(rel.encode("utf-8"))
                try:
                    with open(os.path.join(dirpath, name), "rb") as fh:
                        digest.update(fh.read())
                except OSError:
                    digest.update(b"?")
        _BUILD_HASH = digest.hexdigest()[:8]
    return _BUILD_HASH

#: Version of the HTTP contract; clients compare the major component.
__api_version__ = API_VERSION

__all__ = [
    "__version__",
    "__api_version__",
    "build_hash",
    "API_VERSION",
    "paths",
    "types",
    "ApiError",
    "BBox",
    "CanvasTransform",
    "Engine",
    "ErrorCode",
    "ErrorInfo",
    "JobState",
    "JobStatus",
    "Limits",
    "MaskInstance",
    "Point",
    "PointPrompt",
    "ResultHeader",
    "RuntimeInfo",
    "Size",
    "TextPrompt",
    "pack_result",
    "unpack_result",
]
