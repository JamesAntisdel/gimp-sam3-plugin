"""Package marker and *declared surface* of the GIMP plug-in.

This file exists for the repository's benefit, not for GIMP's.  GIMP loads
``sam3_gimp/sam3_gimp.py`` as a standalone script; it never imports this
package.  Tooling (``tools/dev_sync.py``), the test-suite, and any future
packaging step do import it, so it must obey two rules:

1. **Standard library only.**  No ``gi``, no ``Gimp``, nothing third-party.
   It has to import on a machine with no GIMP at all.
2. **No side effects.**  Importing it must not touch the filesystem, spawn
   anything, or read the environment.

What it holds is the small amount of metadata that both the entry point and the
tooling need to agree on: the PDB procedure names and the output-mode nicks.
``sam3_gimp.py`` re-declares the same constants (it cannot import this module --
inside GIMP only the plug-in *directory* is on ``sys.path``, so
``import sam3_gimp`` there would find the script, not the package).  The
duplication is deliberate and is asserted to be consistent by
``tests/plugin/test_entry.py``.
"""

#: Version of the plug-in half of the project.  Tracks ``sam3gimpd.__version__``
#: loosely; compatibility with the daemon is decided by ``GET /hello``, never
#: by this string.
__version__ = "0.1.0"

#: Major.minor of the HTTP contract in ``_daemon/API.md`` that this plug-in
#: speaks.  The launcher compares only the MAJOR component against the
#: daemon's ``api_version``.
API_VERSION = "1.0"

# --------------------------------------------------------------------------- #
# PDB procedure names
# --------------------------------------------------------------------------- #
# GIMP requires a PDB name to be lowercase, to use '-' as the separator, and to
# contain at least one '-'.  The ``plug-in-`` prefix is the convention for
# procedures installed by a plug-in.

#: Interactive canvas flow: the main dialog with preview, prompt box and
#: click-to-refine.  Registered as an image procedure.
PROC_SEGMENT = "plug-in-sam3-segment"

#: Scriptable PCS entry point: a noun phrase -> every matching instance.
PROC_SEGMENT_TEXT = "plug-in-sam3-segment-by-text"

#: Scriptable PVS entry point: points and/or a box -> one instance.
PROC_SEGMENT_POINTS = "plug-in-sam3-segment-by-points"

#: Setup / Doctor: environment install, HF token flow, diagnostics.  Registered
#: as a plain procedure so it can run with no image open.
PROC_SETUP = "plug-in-sam3-setup"

#: Every procedure ``do_query_procedures`` returns, in menu order.
#: Registration order is menu order, and Setup leads deliberately: on a fresh
#: install it is the only entry that does anything.  Kept identical to the
#: tuple in ``sam3_gimp.py``; ``tests/plugin/test_entry.py`` asserts they agree.
PROCEDURES = (
    PROC_SETUP,
    PROC_SEGMENT,
    PROC_SEGMENT_TEXT,
    PROC_SEGMENT_POINTS,
)

#: Where the procedures land in GIMP's menus.
MENU_PATH = "<Image>/Filters/AI Segmentation"

# --------------------------------------------------------------------------- #
# output modes
# --------------------------------------------------------------------------- #
#: Nicks of the ``output-mode`` choice argument, in the order they are offered.
#: ``outputs.py`` is the module that actually implements them; this tuple is the
#: agreed vocabulary between it and the entry point.
OUTPUT_MODES = (
    "selection-replace",
    "selection-add",
    "selection-subtract",
    "selection-intersect",
    "channels",
    "layer-masks",
    "layer-groups",
    "paths",
)

#: PDB nick -> ``(outputs.OutputMode, outputs.SelectionOp)``.  ``outputs.py``
#: keeps "what to build" and "how to combine it with the existing selection" as
#: two orthogonal values; a PDB choice argument is one value, so the four
#: selection variants are flattened here and expanded again by the entry point.
OUTPUT_MODE_MAP = {
    "selection-replace": ("selection", "replace"),
    "selection-add": ("selection", "add"),
    "selection-subtract": ("selection", "subtract"),
    "selection-intersect": ("selection", "intersect"),
    "channels": ("channels", "replace"),
    "layer-masks": ("layer-masks", "replace"),
    "layer-groups": ("layer-groups", "replace"),
    "paths": ("paths", "replace"),
}

#: Default when the user has expressed no preference.
DEFAULT_OUTPUT_MODE = "selection-replace"

#: Default client-side threshold applied to the soft uint8 masks the daemon
#: returns.  128 == logit 0 (``_daemon/API.md`` §8).
DEFAULT_MASK_THRESHOLD = 128

#: Default client-side score filter.  The daemon is always asked for everything
#: above 0.1 so that moving this filter costs no round trip.
#:
#: 0.30, matching ``ui/main_dialog``.  This was 0.5, which silently discarded
#: most of what PCS returns: SAM 3 routinely describes one object as several
#: partial instances scoring 0.2-0.4, so a prompt like "guitar" applied a
#: single fragment while "person" -- one confident instance -- looked fine.
DEFAULT_SCORE_THRESHOLD = 0.30

#: What the daemon is asked for, regardless of the client-side filter.
#: The floor *requested from the daemon*, not the user-facing filter.
#: post_process_instance_segmentation drops everything below this before
#: the client ever sees it, so it is a hard ceiling on what the score
#: slider can reveal.  At 0.1 the slider's bottom tenth was dead and weak
#: matches were unreachable: a prompt like "guitar" kept only the
#: headstock, and "guitar strap" -- whose instances all score lower --
#: returned nothing at all and looked like a model failure.
DAEMON_SCORE_THRESHOLD = 0.02

#: Longest side, in pixels, of the image uploaded to the daemon.  The model
#: works at 1008 px; sending more buys nothing (``DESIGN.md`` §1).
MAX_UPLOAD_SIDE = 1008

__all__ = [
    "__version__",
    "API_VERSION",
    "PROC_SEGMENT",
    "PROC_SEGMENT_TEXT",
    "PROC_SEGMENT_POINTS",
    "PROC_SETUP",
    "PROCEDURES",
    "MENU_PATH",
    "OUTPUT_MODES",
    "OUTPUT_MODE_MAP",
    "DEFAULT_OUTPUT_MODE",
    "DEFAULT_MASK_THRESHOLD",
    "DEFAULT_SCORE_THRESHOLD",
    "DAEMON_SCORE_THRESHOLD",
    "MAX_UPLOAD_SIDE",
]
