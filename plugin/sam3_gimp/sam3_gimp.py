#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sam3-gimp -- GIMP 3.x entry point for SAM 3 segmentation.

GIMP discovers a Python plug-in by looking for ``<plug-ins>/<name>/<name>.py``:
the file must be named exactly like its containing directory, and on POSIX it
must be executable.  This file is that file.  It does as little as possible --
register procedures, sort out run modes, and hand off -- because every line here
runs inside GIMP's embedded Python, where nothing can be tested on a machine
without GIMP.

Hard rules this file obeys (``DESIGN.md`` sec. 3, ``_daemon/API.md`` sec. 16.10):

* **Standard library and ``gi`` only.**  No numpy, no requests, no pillow, ever.
  torch lives in the daemon's separate virtualenv and is never importable here.
* Nothing blocking on the GTK main loop: the interactive dialog does its HTTP on
  a worker thread; the non-interactive path has no main loop to block, so it may
  call synchronously.
* Every write to the image is one undo step, so a single Ctrl+Z reverts an
  Apply.  ``outputs.apply_result`` owns that undo group; this file must not add
  a second one around it.


Registered procedures
---------------------

``plug-in-sam3-segment``            interactive canvas dialog (the real UI)
``plug-in-sam3-segment-by-text``    scriptable PCS: noun phrase -> N instances
``plug-in-sam3-segment-by-points``  scriptable PVS: points/box -> 1 instance
``plug-in-sam3-setup``              Setup / Doctor, runs with no image open

The two ``segment-by-*`` procedures are the scriptable surface: they run
headless from Script-Fu, from other plug-ins, and from ``gimp -i -b``, and in
INTERACTIVE mode they put up a plain ``GimpUi.ProcedureDialog`` over their
arguments rather than the custom canvas.

``plug-in-sam3-segment`` declares the *same* arguments as ``segment-by-text``.
That is deliberate: ``Gimp.ProcedureConfig`` then persists the last prompt,
thresholds and output mode for the canvas dialog (``DESIGN.md`` sec. 6), and it
gives the canvas a usable fallback -- if ``ui/main_dialog.py`` is not present in
the installed tree, the procedure degrades to the argument dialog instead of
failing outright.


Interface used from the sibling plug-in modules
-----------------------------------------------

Every one is imported **lazily**, so a missing or broken sibling produces an
actionable error dialog rather than GIMP silently dropping the plug-in during
procedure query::

    launcher.find_or_spawn(parent_pid=int) -> LaunchResult
        .client  a connected client.Sam3Client
        .close()

    client.Sam3Client
        .upload_image(pixels, w, h, source_width=, source_height=) -> ImageAccepted
        .run_text(image_id, text, request_id=, score_threshold=, max_instances=,
                  on_progress=) -> Result | None
        .run_points(image_id, points, box=, request_id=, multimask=,
                    max_instances=, on_progress=) -> Result | None
        (no wall-clock ``timeout=``: the client gives up on a stalled job,
        not on a slow one, and a CPU-only daemon can take minutes)
        Result.header / .instances[].mask  (API.md sec. 8)

    gimpbridge.read_upload_pixels(image, source=, drawable=, max_side=)
                                                          -> UploadedImage
        .pixels .width .height .geometry.{source_width,source_height,to_upload}
    gimpbridge.source_layer(image, drawable) -> the layer a drawable stands for

    outputs.MaskResult.from_frame(header, blob, source, prompt_text) -> MaskResult
    outputs.OutputOptions(mode=, selection_op=, post=PostOps(threshold=), ...)
    outputs.apply_result(image, result, options=, layer=, selected=) -> AppliedResult

    ui.setup_dialog.run_setup(parent=None, page="install") -> bool
    ui.setup_dialog.needs_setup() -> bool
    ui.main_dialog.run_main_dialog(parent=None, image=, drawable=, procedure=,
                                   config=) -> int (a Gtk.ResponseType)
        Optional: if the module is absent the interactive procedure degrades to
        the argument dialog instead of failing.  The dialog owns its own
        Apply -- it calls ``outputs.apply_result`` itself -- so this file only
        translates its response into a PDB status.


What cannot be verified without GIMP
------------------------------------

The unit tests run this file under a fake ``gi``, so no real ``Gimp.*`` call
below executes there.  The shapes used are GIMP 3.0's: ``Gimp.ImageProcedure``
run functions take
``(procedure, run_mode, image, drawables, config, run_data)`` and plain
``Gimp.Procedure`` run functions take ``(procedure, config, run_data)``; the
latter is accepted through ``*args`` so a 3.0.x signature change cannot lock the
user out of the repair panel.  ``tests/plugin/test_entry.py`` checks everything
checkable without GIMP: that the file parses, that it registers the names it
claims, that every run callback exists, and that its declared surface still
matches ``sam3_gimp/__init__.py``.
"""

import os
import pathlib
import re
import sys
import time
import traceback

# ---------------------------------------------------------------------------
# sys.path: make the sibling modules importable.
#
# GIMP executes this file as a script, so its directory is normally already on
# sys.path -- but not on every platform and not when GIMP runs from a bundle.
# Inserting it explicitly is cheap and removes a whole class of "works on my
# machine".  Done before any sibling import and before gi.
# ---------------------------------------------------------------------------
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)

import gi  # noqa: E402

gi.require_version("Gimp", "3.0")
gi.require_version("GimpUi", "3.0")
gi.require_version("Gtk", "3.0")
from gi.repository import Gimp  # noqa: E402
from gi.repository import GimpUi  # noqa: E402
from gi.repository import GObject  # noqa: E402
from gi.repository import GLib  # noqa: E402
from gi.repository import Gtk  # noqa: E402


# =========================================================================== #
# Declared surface.  Kept identical to sam3_gimp/__init__.py, which this file
# deliberately does not import: inside GIMP only the plug-in *directory* is on
# sys.path, so ``import sam3_gimp`` there resolves to this script, not to the
# package.  tests/plugin/test_entry.py asserts the two agree.
# =========================================================================== #
__version__ = "0.1.0"
API_VERSION = "1.0"

PROC_SEGMENT = "plug-in-sam3-segment"
PROC_SEGMENT_TEXT = "plug-in-sam3-segment-by-text"
PROC_SEGMENT_POINTS = "plug-in-sam3-segment-by-points"
PROC_SETUP = "plug-in-sam3-setup"

#: Registration order *is* menu order, and Setup comes first on purpose: on a
#: fresh install it is the only entry that does anything, and burying it under
#: three segmentation commands is how a user ends up pressing Segment to find
#: out nothing is installed.
PROCEDURES = (
    PROC_SETUP,
    PROC_SEGMENT,
    PROC_SEGMENT_TEXT,
    PROC_SEGMENT_POINTS,
)

MENU_PATH = "<Image>/Filters/AI Segmentation"

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

#: PDB nick -> ``(outputs.OutputMode, outputs.SelectionOp)``.
#:
#: ``outputs.py`` models "what to build" and "how to combine it with the
#: existing selection" as two orthogonal values.  A PDB choice argument is one
#: value, and four separate selection entries are far friendlier to a Script-Fu
#: author than a second enum they must remember to set, so the four selection
#: nicks are flattened here and expanded again on the way through.
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

DEFAULT_OUTPUT_MODE = "selection-replace"
DEFAULT_MASK_THRESHOLD = 128
#: Matches ui/main_dialog.  The scriptable path used 0.5, which quietly discards
#: most of what PCS returns: a prompt like "guitar" often comes back as several
#: partial instances scoring 0.2-0.4, and at 0.5 only a fragment survives, so
#: the user sees a sliver of the object and no hint that anything was dropped.
DEFAULT_SCORE_THRESHOLD = 0.30
#: The floor *requested from the daemon*, not the user-facing filter.
#: post_process_instance_segmentation drops everything below this before
#: the client ever sees it, so it is a hard ceiling on what the score
#: slider can reveal.  At 0.1 the slider's bottom tenth was dead and weak
#: matches were unreachable: a prompt like "guitar" kept only the
#: headstock, and "guitar strap" -- whose instances all score lower --
#: returned nothing at all and looked like a model failure.
DAEMON_SCORE_THRESHOLD = 0.02
MAX_UPLOAD_SIDE = 1008

#: ``(nick, label, tooltip)`` for the ``output-mode`` choice.  The nicks are
#: OUTPUT_MODES in the same order; the test asserts that.
OUTPUT_MODE_CHOICES = (
    ("selection-replace", "Selection (replace)", "Replace the selection with the masks"),
    ("selection-add", "Selection (add)", "Add the masks to the current selection"),
    ("selection-subtract", "Selection (subtract)", "Subtract the masks from the current selection"),
    ("selection-intersect", "Selection (intersect)", "Intersect the masks with the current selection"),
    ("channels", "Channels", "One channel per instance, named from the prompt and score"),
    ("layer-masks", "Layer mask", "Apply as a mask on a duplicate of the source layer"),
    ("layer-groups", "Layer group", "One masked layer per instance, inside a group"),
    ("paths", "Paths", "Trace the masks into GIMP paths"),
)

#: Authors / copyright / date used by ``set_attribution`` on every procedure.
AUTHORS = "sam3-gimp contributors"
COPYRIGHT = "sam3-gimp contributors"
YEAR = "2026"

#: Accepted by the scriptable ``points`` argument, e.g. "512,300,1;640,410,0".
#: Each coordinate is one decimal number: "12", "12.5", ".5" -- never "1.2.3".
_POINT_RE = re.compile(
    r"^\s*(-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))\s*,"
    r"\s*(-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))\s*(?:,\s*([01])\s*)?$"
)


class Sam3Error(Exception):
    """A failure worth showing the user verbatim.

    Carries an optional ``hint`` (what to do about it) and ``offer_setup``
    (whether the error dialog should offer to open the Setup/Doctor panel).
    """

    def __init__(self, message, hint=None, offer_setup=False):
        Exception.__init__(self, message)
        self.message = message
        self.hint = hint
        self.offer_setup = offer_setup


# =========================================================================== #
# Lazy sibling-module loading
# =========================================================================== #
def _import_sibling(name, optional=False):
    """Import a sibling plug-in module, turning ImportError into a Sam3Error.

    Deliberately not done at module scope: an ImportError while GIMP is querying
    procedures makes the whole plug-in vanish from the menus with nothing but a
    line in the console.  Failing here instead gets the user a dialog that names
    the module and points at Setup.  ``optional=True`` returns None instead.
    """
    try:
        module = __import__(name)
    except ImportError as exc:
        if optional:
            _log("optional module %s is unavailable: %s" % (name, exc))
            return None
        raise Sam3Error(
            "The SAM 3 plug-in is incomplete: could not import '%s'." % (name,),
            hint=(
                "Expected it next to this file, in\n%s\n\n"
                "Re-run the installer (tools/install.ps1) or tools/dev_sync.py "
                "to copy the full plug-in directory.\n\nPython said: %s"
                % (PLUGIN_DIR, exc)
            ),
            offer_setup=True,
        )
    for part in name.split(".")[1:]:
        try:
            module = getattr(module, part)
        except AttributeError as exc:
            if optional:
                return None
            raise Sam3Error(
                "The SAM 3 plug-in is incomplete: '%s' is missing." % (name,),
                hint=str(exc),
                offer_setup=True,
            )
    return module


def _need_attr(module, attr, modname):
    """Fetch ``module.attr`` or raise a Sam3Error naming what is missing."""
    value = getattr(module, attr, None)
    if value is None:
        raise Sam3Error(
            "The SAM 3 plug-in is out of date: %s.%s is missing." % (modname, attr),
            hint=(
                "The plug-in files in\n%s\ndo not all come from the same "
                "release. Re-copy the whole sam3_gimp directory." % (PLUGIN_DIR,)
            ),
            offer_setup=False,
        )
    return value


# =========================================================================== #
# Small GIMP/GTK helpers
# =========================================================================== #
def _gimp_pid():
    """The pid the daemon should watch, i.e. GIMP's -- not this process's.

    Plug-in processes are short-lived (``DESIGN.md`` Constraint C), so passing
    our own pid as ``--parent-pid`` would kill the daemon the moment ``run()``
    returns.  Our parent is the GIMP process that spawned us.
    """
    launcher = _import_sibling("launcher", optional=True)
    finder = getattr(launcher, "gimp_pid", None) if launcher is not None else None
    if callable(finder):
        try:
            return finder()
        except Exception:
            pass
    # Fallback: the parent, or *nothing*.  Never our own pid -- that killed the
    # daemon the moment run() returned, every single time.
    try:
        ppid = int(os.getppid())
    except (AttributeError, OSError):
        return None
    return ppid if ppid > 1 else None


def _ui_init(name):
    """``GimpUi.init`` guarded, so a headless GIMP cannot crash us."""
    try:
        GimpUi.init(name)
        return True
    except Exception:
        return False


#: ``plugin.log`` is cut back to its last LOG_KEEP_BYTES once it passes
#: LOG_LIMIT_BYTES: it is for the last few sessions, not an archive.
LOG_LIMIT_BYTES = 2 * 1024 * 1024
LOG_KEEP_BYTES = 256 * 1024


def _owner_only(path, flags):
    """``open(..., opener=)`` hook: create the file readable by its owner only."""
    return os.open(path, flags, 0o600)


def _open_log(path):
    """``plugin.log`` for appending, created 0600 (it names paths, prompts and
    tracebacks), and tightened to 0600 if an older run left it wider."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fh = open(path, "a", encoding="utf-8", errors="replace", opener=_owner_only)
    if os.name != "nt":
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    return fh


def _trim_log(path):
    """Cut ``plugin.log`` back to its tail once it passes LOG_LIMIT_BYTES.

    Done in place, through the file's own bytes.  Everything that writes
    here -- this process's faulthandler (``_CRASH_LOG``, the file the crash
    and hang dumps go to), the dialog's own lines, other plug-in processes
    -- holds the file open for *appending*, so after the cut each of them
    carries on writing at the new end of the same file.  Deleting or
    renaming it would strand them: on POSIX the dumps would go to an
    unlinked file, and on Windows a file another handle has open cannot be
    removed at all, so the size cap would never apply.
    """
    try:
        if os.path.getsize(path) <= LOG_LIMIT_BYTES:
            return
        with open(path, "r+b") as fh:
            fh.seek(-LOG_KEEP_BYTES, os.SEEK_END)
            tail = fh.read()
            newline = tail.find(b"\n")
            if newline >= 0:
                tail = tail[newline + 1:]
            marker = ("%s [sam3-gimp] plugin.log trimmed to its last %d KiB by pid %d "
                      "(plug-in %s)\n" % (time.strftime("%Y-%m-%d %H:%M:%S"),
                                          LOG_KEEP_BYTES // 1024, os.getpid(), __version__))
            fh.seek(0)
            fh.write(marker.encode("utf-8") + tail)
            fh.truncate()
    except OSError:
        pass


def _plugin_log_path():
    """``<base>/logs/plugin.log``, resolved without importing the launcher.

    Deliberately standalone: this has to work even when the failure being logged
    *is* an import failure, so it mirrors ``launcher.base_dir()`` by hand rather
    than depending on it.
    """
    override = os.environ.get("SAM3_GIMP_HOME")
    if override:
        try:
            base = str(pathlib.Path(override).expanduser())
        except (RuntimeError, KeyError):  # no such ~user, or no home: kept as written
            base = override
    elif os.name == "nt" or sys.platform.startswith("win"):
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        base = (os.path.join(root, "sam3-gimp") if root else
                os.path.join(os.path.expanduser("~"), "AppData", "Local", "sam3-gimp"))
    elif sys.platform == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library",
                            "Application Support", "sam3-gimp")
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        root = xdg if xdg and os.path.isabs(xdg) else os.path.join(
            os.path.expanduser("~"), ".local", "share")
        base = os.path.join(root, "sam3-gimp")
    # Spelled the way pathlib spells it, as launcher.base_dir() does.
    return str(pathlib.Path(base, "logs", "plugin.log"))



def _daemon_log_path():
    """``<base>/logs/sam3gimpd.log``, resolved without importing the launcher."""
    try:
        launcher = _import_sibling("launcher", optional=True)
        if launcher is not None:
            return launcher.server_log()
    except Exception:
        pass
    return os.path.join(os.path.dirname(_plugin_log_path()), "sam3gimpd.log")

def _install_crash_diagnostics():
    """Make a dying plug-in process leave evidence.

    GIMP reports "Plug-in crashed" when this process exits without answering
    -- a segfault or abort in cairo/GDK, or GTK touched from the wrong thread
    -- and then there is nothing to read.  ``faulthandler`` writes the Python
    stack of every thread to ``plugin.log`` on a fatal signal, and the two
    excepthooks catch what an ordinary ``except`` never sees: an exception
    escaping the main thread or a worker thread.  Never raises: diagnostics
    that can fail registration are worse than none.
    """
    try:
        import faulthandler  # noqa: PLC0415

        path = _plugin_log_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Kept open for the life of the process; faulthandler needs the fd.
        # Appending, so trimming the file (_trim_log) never strands it.
        global _CRASH_LOG
        _trim_log(path)
        _CRASH_LOG = _open_log(path)
        # Which interpreter GIMP handed us.  It is not ours to choose -- the
        # Windows bundle embeds its own, a Linux build uses the distro's -- and
        # it is the first thing worth knowing about "the plug-in does not load
        # on my GIMP", so it goes in the log before anything can fail.
        _CRASH_LOG.write("%s [sam3-gimp] pid %d started on Python %s (%s)\n"
                         % (time.strftime("%Y-%m-%d %H:%M:%S"), os.getpid(),
                            "%d.%d.%d" % sys.version_info[:3],
                            sys.executable or "no sys.executable"))
        _CRASH_LOG.flush()
        faulthandler.enable(file=_CRASH_LOG, all_threads=True)
    except Exception:
        pass
    try:
        # Which files are actually running: the first question after any
        # report that "the update did not help".
        build = _import_sibling("launcher").plugin_build()
        _CRASH_LOG.write("%s [sam3-gimp] plug-in %s build %s\n"
                         % (time.strftime("%Y-%m-%d %H:%M:%S"), __version__, build))
        _CRASH_LOG.flush()
    except Exception:
        pass

    def _hook(exc_type, exc, tb):
        try:
            _log("UNHANDLED in main thread:\n%s"
                 % "".join(traceback.format_exception(exc_type, exc, tb)))
        finally:
            sys.__excepthook__(exc_type, exc, tb)

    def _thread_hook(args):
        _log("UNHANDLED in thread %r:\n%s" % (
            getattr(args.thread, "name", "?"),
            "".join(traceback.format_exception(args.exc_type, args.exc_value,
                                               args.exc_traceback))))

    try:
        sys.excepthook = _hook
        import threading  # noqa: PLC0415

        if hasattr(threading, "excepthook"):
            threading.excepthook = _thread_hook
    except Exception:
        pass


_CRASH_LOG = None


def _log(message):
    """One line to stderr *and* to ``<base>/logs/plugin.log``.

    stderr alone is not enough on the primary target.  GIMP started from the
    Start Menu -- which is how the Microsoft Store build is always started --
    has no console attached, so anything written to stderr is discarded and a
    plug-in failure looks to the user like the menu item simply doing nothing.
    A file means "nothing happened" always leaves evidence behind.

    Never raises: a logger that can throw would turn a reportable failure into
    an unreportable one.
    """
    line = "[sam3-gimp] %s" % (message,)
    try:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    except Exception:
        pass
    try:
        path = _plugin_log_path()
        _trim_log(path)
        with _open_log(path) as fh:
            fh.write("%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), line))
    except Exception:
        pass



def _env_ready():
    """Is the daemon environment installed?  Never raises, never blocks.

    Called during procedure *registration*, where an exception would make the
    whole plug-in disappear from the menus with only a console line to say why,
    and where anything slow would delay GIMP's startup.  So: filesystem checks
    only, and every failure answers "not ready" rather than propagating.
    """
    try:
        bootstrap = _import_sibling("bootstrap", optional=True)
        if bootstrap is None:
            return False
        # Hand it an accelerator so it does not go looking for one: the default
        # path runs `nvidia-smi -L` with a 15 second timeout, and this is called
        # while GIMP is building its procedure database.  A subprocess there
        # delays every GIMP start and, on Windows, can flash a console window.
        # Readiness does not depend on which accelerator is present.
        stub = bootstrap.Accelerator("unknown", "not probed during registration")
        return bool(bootstrap.inspect_environment(accelerator=stub).env_ready)
    except Exception:
        return False


def _require_setup(procedure, run_mode):
    """Send the user to Setup instead of letting them discover it the hard way.

    Returns a PDB return value when the environment is not ready and the caller
    should stop, or ``None`` to carry on.  Interactively this offers to open
    Setup directly; non-interactively it fails with a message a script author
    can act on.
    """
    if _env_ready():
        return None
    message = "The SAM 3 environment is not installed yet."
    hint = ("Open Filters > AI Segmentation > SAM 3 Setup / Doctor. If you "
            "already have PyTorch with CUDA, use \"Use an existing Python "
            "environment\" there rather than installing a second copy.")
    if run_mode == Gimp.RunMode.NONINTERACTIVE:
        return _failure(procedure, "%s %s" % (message, hint))
    _log("environment is not ready; offering Setup")
    _show_error("SAM 3", message, hint, offer_setup=True)
    return _cancel(procedure)

_install_crash_diagnostics()



#: How long ``run()`` waits for the dialog's threads before returning to GIMP.
#: Covers one daemon long-poll (the client waits up to 10 s per status poll)
#: with room to spare; it only costs anything when a thread is still busy.
QUIESCE_S = 12.0


def _quiesce_threads(timeout=QUIESCE_S):
    """Wait for the plug-in's own background threads before returning to GIMP.

    When ``run()`` returns, libgimp closes the pipe and calls ``exit()``, and
    Python finalisation begins with any daemon thread still running.  A thread
    that re-enters the interpreter at that point -- a long-poll returning, a
    ``close()`` finishing -- aborts the process, and GIMP reports "Plug-in
    crashed".  It is timing-dependent, which is why it looked like it depended
    on the word: a short session closed quickly is the worst case.

    Only threads this plug-in named ``sam3-*`` are waited for, with one shared
    deadline; anything still alive is logged rather than blocked on forever.
    """
    import threading  # noqa: PLC0415

    deadline = time.time() + float(timeout)
    for thread in threading.enumerate():
        name = getattr(thread, "name", "") or ""
        if thread is threading.current_thread() or not name.startswith("sam3-"):
            continue
        remaining = deadline - time.time()
        if remaining > 0:
            thread.join(remaining)
        if thread.is_alive():
            _log("thread %r still running at exit; returning to GIMP anyway" % name)

def _open_setup(parent=None):
    """Open the Setup/Doctor panel, directly if possible, else through the PDB."""
    setup_dialog = _import_sibling("ui.setup_dialog", optional=True)
    if setup_dialog is not None and hasattr(setup_dialog, "run_setup"):
        try:
            return bool(setup_dialog.run_setup(parent))
        except Exception:
            _log(traceback.format_exc())
    try:
        pdb = Gimp.get_pdb()
        proc = pdb.lookup_procedure(PROC_SETUP)
        if proc is None:
            return False
        proc.run(proc.create_config())
        return True
    except Exception:
        return False


def _show_error(title, message, hint=None, offer_setup=False):
    """Actionable modal error.  Falls back to ``Gimp.message`` with no display.

    Returns True when the user chose to open Setup and Setup reported success.
    """
    text = message if not hint else "%s\n\n%s" % (message, hint)
    if not _ui_init("sam3-gimp-error"):
        try:
            Gimp.message(text)
        except Exception:
            pass
        return False

    try:
        dialog = Gtk.MessageDialog(
            transient_for=None,
            modal=True,
            destroy_with_parent=True,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.NONE,
            text=message,
        )
        dialog.set_title(title)
        if hint:
            dialog.format_secondary_text(hint)
        dialog.add_button("_Close", Gtk.ResponseType.CLOSE)
        if offer_setup:
            dialog.add_button("Open _Setup...", Gtk.ResponseType.ACCEPT)
            dialog.set_default_response(Gtk.ResponseType.ACCEPT)
        else:
            dialog.set_default_response(Gtk.ResponseType.CLOSE)
        response = dialog.run()
        dialog.destroy()
        # Let GTK tear the window down before control goes back to GIMP.
        while Gtk.events_pending():
            Gtk.main_iteration_do(False)
    except Exception:
        try:
            Gimp.message(text)
        except Exception:
            pass
        return False

    if offer_setup and response == Gtk.ResponseType.ACCEPT:
        return _open_setup()
    return False


# --------------------------------------------------------------------------- #
# return values
# --------------------------------------------------------------------------- #

#: A run that has not returned after this long gets every thread's Python
#: stack written to plugin.log, once.  It turns "GIMP is frozen" into a line
#: number without waiting for a crash that may never come.
HANG_DUMP_AFTER_S = 45.0
#: Once the GTK loop is running, a heartbeat re-arms a shorter dump every
#: HEARTBEAT_S; if the loop stalls for STALL_DUMP_AFTER_S the C-level
#: watchdog (which needs no GIL) writes every thread's stack to plugin.log.
HEARTBEAT_S = 5
STALL_DUMP_AFTER_S = 15.0
_HEARTBEAT_SOURCE = 0


def _arm_hang_dump():
    """A stack dump if this process stops making progress.

    Before the dialog's loop runs (or for the scriptable procedures, which
    have no loop) the only signal is elapsed time, so a single dump is armed
    at HANG_DUMP_AFTER_S.  Once the loop ticks, the heartbeat keeps re-arming
    a STALL_DUMP_AFTER_S dump: a healthy loop cancels it every HEARTBEAT_S,
    a frozen one lets it fire, and faulthandler's watchdog thread writes the
    stacks without needing the GIL the frozen thread is holding.
    """
    global _HEARTBEAT_SOURCE
    try:
        import faulthandler  # noqa: PLC0415

        if _CRASH_LOG is not None:
            faulthandler.dump_traceback_later(HANG_DUMP_AFTER_S, repeat=False,
                                              file=_CRASH_LOG)

            def _heartbeat():
                try:
                    faulthandler.dump_traceback_later(STALL_DUMP_AFTER_S, repeat=False,
                                                      file=_CRASH_LOG)
                except Exception:
                    pass
                return True

            if not _HEARTBEAT_SOURCE:
                _HEARTBEAT_SOURCE = GLib.timeout_add_seconds(HEARTBEAT_S, _heartbeat)
    except Exception:
        pass


def _disarm_hang_dump():
    global _HEARTBEAT_SOURCE
    try:
        import faulthandler  # noqa: PLC0415

        faulthandler.cancel_dump_traceback_later()
    except Exception:
        pass
    source, _HEARTBEAT_SOURCE = _HEARTBEAT_SOURCE, 0
    if source:
        try:
            GLib.source_remove(source)
        except Exception:
            pass

def _success(procedure, num_instances=None):
    _disarm_hang_dump()
    retval = procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())
    if num_instances is not None:
        _set_int_return(retval, 1, num_instances)
    return retval


def _cancel(procedure):
    _disarm_hang_dump()
    return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())


def _failure(procedure, message, status=None):
    """A PDB error carrying ``message``.

    ``GLib.Error`` is what GIMP surfaces to the caller (and to Script-Fu), so
    the message must be complete on its own.
    """
    _disarm_hang_dump()
    if status is None:
        status = Gimp.PDBStatusType.EXECUTION_ERROR
    _log(message)
    return procedure.new_return_values(status, GLib.Error(message))


def _set_int_return(retval, index, value):
    """Overwrite return value ``index`` of a ``Gimp.ValueArray`` with an int.

    ``new_return_values`` pre-fills declared return values with defaults, so
    remove-then-insert is the way to set one from Python; ``GimpValueArray``
    exposes no setter.  Wrapped defensively: failing to report a count must
    never turn a successful segmentation into an error.
    """
    try:
        retval.remove(index)
    except Exception:
        pass
    try:
        retval.insert(index, GObject.Value(GObject.TYPE_INT, int(value)))
    except Exception:
        _log("could not set return value %d" % (index,))
    return retval


# --------------------------------------------------------------------------- #
# config access
# --------------------------------------------------------------------------- #
def _get(config, name, default=None):
    """``config.get_property`` that never raises.

    ``GimpProcedureConfig`` raises on an unknown property; a plug-in that dies
    because an argument was renamed between releases is not acceptable.
    """
    if config is None:
        return default
    try:
        value = config.get_property(name)
    except Exception:
        return default
    return default if value is None else value


def _progress(fraction, text=None):
    try:
        if text is not None:
            Gimp.progress_set_text(text)
        Gimp.progress_update(float(max(0.0, min(1.0, fraction))))
    except Exception:
        pass


# =========================================================================== #
# Prompt parsing (scriptable surface)
# =========================================================================== #
def parse_points(spec):
    """Parse ``"x,y[,label];x,y[,label];..."`` into the wire shape.

    Coordinates are **original-image** pixels -- what a Script-Fu author has in
    hand.  They are converted to uploaded-image space later, once the upload
    size is known (``_daemon/API.md`` sec. 5).  ``label`` defaults to 1
    (include); 0 means exclude.  Semicolons, newlines and whitespace all
    separate points, so a heredoc of one point per line works too.
    """
    points = []
    if not spec:
        return points
    for chunk in re.split(r"[;\n]+", spec):
        if not chunk.strip():
            continue
        match = _POINT_RE.match(chunk)
        try:
            if match is None:
                raise ValueError(chunk)
            x, y, label = float(match.group(1)), float(match.group(2)), match.group(3)
        except ValueError:
            raise Sam3Error(
                "Could not parse the point %r." % (chunk.strip(),),
                hint='Points look like "x,y" or "x,y,label", separated by ";" '
                '- for example "512,300,1;640,410,0". label 1 includes, 0 excludes.',
            )
        points.append({"x": x, "y": y, "label": 1 if label is None else int(label)})
    if len(points) > 64:
        raise Sam3Error(
            "Too many points: %d (the daemon accepts at most 64)." % (len(points),)
        )
    return points


def parse_box(spec):
    """Parse ``"x0,y0,x1,y1"`` into a list of four floats, or return None."""
    if not spec or not spec.strip():
        return None
    parts = [p for p in re.split(r"[,\s]+", spec.strip()) if p]
    if len(parts) != 4:
        raise Sam3Error(
            "Could not parse the box %r." % (spec,),
            hint='A box is four numbers: "x0,y0,x1,y1" in image pixels.',
        )
    try:
        box = [float(p) for p in parts]
    except ValueError:
        raise Sam3Error(
            "The box %r contains something that is not a number." % (spec,),
            hint='A box is four numbers: "x0,y0,x1,y1" in image pixels.',
        )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise Sam3Error(
            "The box %r is empty: x1 must exceed x0 and y1 must exceed y0." % (spec,)
        )
    return box


def scale_to_upload(value, source_size, upload_size):
    """Map one original-image coordinate into uploaded-image space.

    The client owns this half of the coordinate chain: the daemon never learns
    the original size (``_daemon/API.md`` sec. 5), so nothing but us can do it.
    Used when ``gimpbridge`` hands back geometry without its own converter;
    normally ``UploadGeometry.to_upload`` does the same arithmetic.
    """
    if not source_size or not upload_size:
        return float(value)
    return float(value) * (float(upload_size) / float(source_size))


# =========================================================================== #
# The non-interactive segmentation flow
# =========================================================================== #
def _source_layer(image, drawables):
    """The layer this run's drawables stand for, resolved once per run.

    GIMP passes the *selected drawables*, which are the mask while a layer's
    mask is being edited and the channels while a channel is selected; see
    ``gimpbridge.source_layer``.  Without the bridge (or outside GIMP) the
    first drawable is passed through unchanged.
    """
    drawable = drawables[0] if drawables else None
    if image is None:
        return drawable
    gimpbridge = _import_sibling("gimpbridge", optional=True)
    resolve = getattr(gimpbridge, "source_layer", None) if gimpbridge is not None else None
    if not callable(resolve):
        return drawable
    try:
        return resolve(image, drawable)
    except Exception as exc:
        _log("could not resolve the source layer (%s); using the drawable as given" % exc)
        return drawable


def _read_upload(image, layer, use_projection):
    """Pixels for the daemon: the projection by default, else the source layer."""
    gimpbridge = _import_sibling("gimpbridge")
    read = _need_attr(gimpbridge, "read_upload_pixels", "gimpbridge")
    if use_projection or layer is None:
        source = getattr(gimpbridge, "SOURCE_PROJECTION", "projection")
        drawable = None
    else:
        source = getattr(gimpbridge, "SOURCE_LAYER", "layer")
        drawable = layer
    try:
        return read(image, source=source, drawable=drawable, max_side=MAX_UPLOAD_SIDE)
    except Sam3Error:
        raise
    except Exception as exc:
        raise Sam3Error(
            "Could not read the image pixels.",
            hint="%s: %s" % (type(exc).__name__, exc),
        )


def _to_upload(geometry, x, y):
    """original-image -> uploaded-image, via the bridge's geometry if it has one."""
    converter = getattr(geometry, "to_upload", None)
    if callable(converter):
        return converter(x, y)
    source_width = int(getattr(geometry, "source_width", 0) or 0)
    source_height = int(getattr(geometry, "source_height", 0) or 0)
    width = int(getattr(geometry, "width", 0) or 0)
    height = int(getattr(geometry, "height", 0) or 0)
    return (
        scale_to_upload(x, source_width, width),
        scale_to_upload(y, source_height, height),
    )


def _launch():
    """Find-or-spawn the daemon and return the launcher's ``LaunchResult``.

    Everything that can go wrong here -- no venv, no weights, a daemon that will
    not start -- is the Setup panel's business, so failures are raised with
    ``offer_setup=True``.
    """
    launcher = _import_sibling("launcher")
    find_or_spawn = _need_attr(launcher, "find_or_spawn", "launcher")
    try:
        return find_or_spawn(parent_pid=_gimp_pid())
    except Sam3Error:
        raise
    except Exception as exc:
        hint = "%s" % (exc,)
        tail = getattr(exc, "log_tail", "")
        if tail:
            hint = "%s\n\nLast lines of the daemon log:\n%s" % (hint, tail)
        raise Sam3Error(
            "The SAM 3 daemon could not be started.",
            hint=hint + "\n\nOpen Setup to install or repair the segmentation environment.",
            offer_setup=True,
        )


def _mask_result(outputs, result, upload, prompt_text):
    """``client.Result`` -> ``outputs.MaskResult``.

    The frame's blob region is rebuilt by concatenating the per-instance mask
    crops.  ``API.md`` sec. 8.1 guarantees they are tightly packed, in array
    order, with ``instances[0].blob_offset == 0``, so the concatenation is
    byte-identical to what came off the wire and every ``blob_offset`` in the
    header still indexes it correctly.
    """
    blob = b"".join(bytes(inst.mask) for inst in result.instances)
    geometry = getattr(upload, "geometry", None)
    source = (
        int(getattr(geometry, "source_width", 0) or 0),
        int(getattr(geometry, "source_height", 0) or 0),
    )
    from_frame = _need_attr(outputs.MaskResult, "from_frame", "outputs.MaskResult")
    return from_frame(result.header, blob, source, prompt_text)


def _contours_for_paths(mask_result, threshold):
    """Trace every instance mask into beziers in **model-canvas** coordinates.

    Uses the ``contours`` module rather than ``ui.canvas``: the canvas tracer
    exists to draw marching ants, so it emits simplified polylines with no hole
    winding, while this one does marching squares with nesting, Douglas-Peucker
    and a Schneider bezier fit -- which is what a vector path actually wants.
    It is also pure stdlib, so the scriptable (non-interactive) Paths mode no
    longer drags GTK in behind it.
    """
    contours_mod = _import_sibling("contours")
    strokes_for = _need_attr(contours_mod, "instance_strokes", "contours")
    contours = {}
    for inst in mask_result.instances:
        strokes = strokes_for(inst.mask, inst.mask_width, inst.mask_height,
                              inst.bbox, threshold=threshold)
        if strokes:
            contours[int(inst.instance_id)] = strokes
    return contours



def _report_selection(instances, selected, score_threshold, prompt_text):
    """Log, and where useful surface, how the score threshold filtered matches.

    SAM 3 routinely returns one object as several partial instances.  Applying
    only the top-scoring one then yields a fragment, which reads as "the model
    got it wrong" rather than "the threshold is too high".
    """
    scores = sorted((float(i.score) for i in instances), reverse=True)
    if not scores:
        return
    kept = len(selected)
    summary = "%r: %d match(es), %d above %.2f (scores %.2f-%.2f)" % (
        prompt_text, len(scores), kept, score_threshold, scores[-1], scores[0])
    _log(summary)

    dropped = len(scores) - kept
    if kept and dropped:
        # Only worth interrupting for when something was actually discarded.
        best_dropped = max(s for s in scores if s < score_threshold)
        try:
            Gimp.message(
                "SAM 3: applied %d of %d matches for %r.\n"
                "%d more scored up to %.2f. If the selection is missing part of "
                "the object, lower the score threshold to about %.2f and run "
                "again."
                % (kept, len(scores), prompt_text, dropped, best_dropped,
                   max(DAEMON_SCORE_THRESHOLD, best_dropped - 0.05))
            )
        except Exception:
            pass

def _segment(image, drawables, config, engine):
    """Upload, prompt, wait, apply.  Returns the number of instances applied.

    Synchronous on purpose: this path has no GTK main loop to keep responsive.
    The interactive canvas uses a worker thread instead (``DESIGN.md`` sec. 4).
    """
    use_projection = bool(_get(config, "use-projection", True))
    mode_nick = str(_get(config, "output-mode", DEFAULT_OUTPUT_MODE))
    mask_threshold = int(_get(config, "mask-threshold", DEFAULT_MASK_THRESHOLD))
    score_threshold = float(_get(config, "score-threshold", DEFAULT_SCORE_THRESHOLD))
    max_instances = int(_get(config, "max-instances", 64 if engine == "text" else 1))

    if mode_nick not in OUTPUT_MODE_MAP:
        raise Sam3Error(
            "Unknown output mode %r." % (mode_nick,),
            hint="Valid modes: %s." % (", ".join(OUTPUT_MODES),),
        )
    output_mode, selection_op = OUTPUT_MODE_MAP[mode_nick]

    if engine == "text":
        text = str(_get(config, "text", "") or "").strip()
        if not text:
            raise Sam3Error(
                "No prompt was given.",
                hint='SAM 3 wants a simple noun phrase such as "red car" - not '
                'a relational description like "the car on the left". Use '
                "negative points or boxes to exclude things.",
            )
        if len(text) > 512:
            raise Sam3Error("The prompt is longer than the 512-character limit.")
        points, box = [], None
        prompt_text = text
    else:
        points = parse_points(_get(config, "points", "") or "")
        box = parse_box(_get(config, "box", "") or "")
        if not points and box is None:
            raise Sam3Error(
                "No point or box prompt was given.",
                hint='Pass points as "x,y,label;x,y,label" and/or a box as '
                '"x0,y0,x1,y1", in image pixels.',
            )
        text = None
        prompt_text = "points"

    _progress(0.02, "Reading pixels")
    layer = _source_layer(image, drawables)
    upload = _read_upload(image, layer, use_projection)
    geometry = getattr(upload, "geometry", None)

    # original-image -> uploaded-image, the only conversion we owe the daemon.
    for point in points:
        point["x"], point["y"] = _to_upload(geometry, point["x"], point["y"])
    if box is not None:
        bx0, by0 = _to_upload(geometry, box[0], box[1])
        bx1, by1 = _to_upload(geometry, box[2], box[3])
        box = [bx0, by0, bx1, by1]

    def on_progress(status):
        # Job progress covers the 0.2 - 0.9 slice of our own bar.
        try:
            _progress(0.20 + 0.70 * float(status.progress), status.stage)
        except Exception:
            pass

    launch = _launch()
    try:
        client = launch.client
        _progress(0.10, "Uploading image")
        accepted = client.upload_image(
            upload.pixels,
            int(upload.width),
            int(upload.height),
            source_width=int(getattr(geometry, "source_width", 0) or image.get_width()),
            source_height=int(getattr(geometry, "source_height", 0) or image.get_height()),
        )
        image_id = accepted.image_id

        request_id = "pdb-%d-%d" % (os.getpid(), int(time.time() * 1000) % 1000000)
        _progress(0.20, "Segmenting")
        if engine == "text":
            result = client.run_text(
                image_id,
                text,
                request_id=request_id,
                score_threshold=DAEMON_SCORE_THRESHOLD,
                max_instances=max(1, min(256, max_instances)),
                on_progress=on_progress,
            )
        else:
            result = client.run_points(
                image_id,
                points,
                box=box,
                request_id=request_id,
                multimask=max_instances > 1,
                max_instances=max(1, min(8, max_instances)),
                on_progress=on_progress,
            )
    except Sam3Error:
        raise
    except Exception as exc:
        # An encode or inference failure is a *consequence*; the reason lives in
        # the daemon's log, and the message alone ("the encode pass failed") is
        # not something a user can act on.  Name the file.
        hint = "%s: %s" % (type(exc).__name__, exc)
        code = getattr(exc, "code", "")
        # The outer code is often a consequence: "image_not_ready" wrapping an
        # inner "engine_unavailable" is exactly the case where Setup is the
        # answer, and checking only the outer code offered the user a lone
        # Close button on the one error Setup could have fixed.
        detail = getattr(exc, "detail", None)
        inner = ""
        if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
            inner = detail["error"].get("code") or ""
        if code in ("image_not_ready", "inference_failed", "model_load_failed"):
            hint += (
                "\n\nThe daemon's own log has the full traceback:\n  %s\n\n"
                "Common causes: the GPU ran out of memory (try a smaller image, "
                "or close other GPU applications), or the checkpoint could not "
                "be loaded." % _daemon_log_path()
            )
        setup_codes = ("engine_unavailable", "model_load_failed", "weights_missing")
        raise Sam3Error(
            "The segmentation request failed.",
            hint=hint,
            offer_setup=code in setup_codes or inner in setup_codes,
        )
    finally:
        try:
            launch.close()
        except Exception:
            pass

    if result is None:
        # Superseded: only possible if something else prompted the same image
        # on this daemon. Nothing to apply, and nothing has gone wrong.
        _log("result was superseded; nothing applied")
        return 0

    outputs = _import_sibling("outputs")
    _progress(0.92, "Applying")
    mask_result = _mask_result(outputs, result, upload, prompt_text)

    # The score slider filters locally: the daemon was asked for everything
    # above 0.1 precisely so this costs no round trip (API.md sec. 6.3).
    selected = [
        int(inst.instance_id)
        for inst in mask_result.instances
        if float(inst.score) >= score_threshold
    ]
    # Say what was found versus what survived.  Without this the two sliders are
    # unguessable: a prompt that returns seven partial matches and applies one
    # looks identical to a prompt that only ever matched once, and the user has
    # no way to tell that lowering the score threshold is the answer.
    _report_selection(mask_result.instances, selected, score_threshold, prompt_text)

    if not selected:
        best = max((float(i.score) for i in mask_result.instances), default=0.0)
        if mask_result.instances:
            # The daemon already dropped everything under DAEMON_SCORE_THRESHOLD,
            # so a suggestion below that floor cannot be acted on, and one
            # above the best score is useless.
            suggest = max(DAEMON_SCORE_THRESHOLD, best - 0.05)
            hint = ("The best candidate scored %.2f; the threshold is %.2f. "
                    "Lower the score threshold to about %.2f."
                    % (best, score_threshold, suggest))
            if best <= DAEMON_SCORE_THRESHOLD + 0.01:
                hint = ("The best candidate scored only %.2f, at the floor of what "
                        "the daemon reports; no score threshold will select it. "
                        "Try a plainer noun, or Segment by points." % best)
        else:
            hint = (
                "The model recognised nothing for that phrase.\n\n"
                "SAM 3 wants a simple noun phrase it knows -- 'guitar', 'dog', "
                "'red car'. Compound or unusual things ('guitar strap') often "
                "return nothing at all, and no threshold will help.\n\n"
                "For those, use Segment by points or box instead: click the "
                "object itself. That path does not depend on the model knowing "
                "a name for it."
            )
        raise Sam3Error("Nothing matched the prompt at this score threshold.",
                        hint=hint)

    contours = None
    if output_mode == "paths":
        contours = _contours_for_paths(mask_result, mask_threshold)

    options = outputs.OutputOptions(
        mode=output_mode,
        selection_op=selection_op,
        post=outputs.PostOps(threshold=mask_threshold),
        name_prefix="",
        group_name=prompt_text if engine == "text" else "",
    )

    try:
        # apply_result opens and closes the undo group itself, so one Ctrl+Z
        # reverts the whole Apply; do not nest another group around it.
        applied = outputs.apply_result(
            image,
            mask_result,
            options=options,
            layer=layer,
            selected=selected,
            contours=contours,
        )
    except Sam3Error:
        raise
    except Exception as exc:
        raise Sam3Error(
            "The masks could not be applied to the image.",
            hint="%s: %s" % (type(exc).__name__, exc),
        )

    _progress(1.0, "Done")
    names = getattr(applied, "names", None)
    if names:
        return len(names)
    return len(selected)


# =========================================================================== #
# run functions
# =========================================================================== #
def run_segment(procedure, run_mode, image, drawables, config, run_data):
    """Interactive canvas flow (``plug-in-sam3-segment``).

    There is no non-interactive meaning for "open the canvas", so NONINTERACTIVE
    is refused with a message naming the two procedures that do work from a
    script.  When ``ui/main_dialog.py`` is not present the procedure degrades to
    the plain argument dialog rather than failing.
    """
    _arm_hang_dump()
    if run_mode == Gimp.RunMode.NONINTERACTIVE:
        return _failure(
            procedure,
            "%s opens an interactive dialog and cannot run non-interactively. "
            "Use %s or %s from a script."
            % (PROC_SEGMENT, PROC_SEGMENT_TEXT, PROC_SEGMENT_POINTS),
            status=Gimp.PDBStatusType.CALLING_ERROR,
        )

    blocked = _require_setup(procedure, run_mode)
    if blocked is not None:
        return blocked

    main_dialog = _import_sibling("ui.main_dialog", optional=True)
    entry = None
    if main_dialog is not None:
        entry = getattr(main_dialog, "run_main_dialog", None) or getattr(
            main_dialog, "run_dialog", None
        )
    if entry is None:
        _log("ui.main_dialog is unavailable; falling back to the argument dialog")
        return _run_scriptable(
            procedure, run_mode, image, drawables, config, "text", "Segment with SAM 3"
        )

    try:
        _ui_init("sam3-gimp")
        response = entry(
            None,
            image=image,
            drawable=_source_layer(image, drawables),
            procedure=procedure,
            config=config,
        )
    except Sam3Error as exc:
        _quiesce_threads()
        _show_error("SAM 3", exc.message, exc.hint, exc.offer_setup)
        return _failure(procedure, exc.message)
    except Exception as exc:
        _log(traceback.format_exc())
        _quiesce_threads()
        _show_error(
            "SAM 3",
            "The segmentation dialog failed to open.",
            "%s: %s" % (type(exc).__name__, exc),
            offer_setup=True,
        )
        return _failure(procedure, str(exc))

    # The dialog's worker and release threads must be done before this
    # returns: libgimp calls exit() as soon as it does, and a daemon thread
    # re-entering Python during finalisation is a hard crash.
    _quiesce_threads()
    if not _dialog_finished(response):
        return _cancel(procedure)
    Gimp.displays_flush()
    return _success(procedure)


def _dialog_finished(response):
    """Did the canvas session end normally, or was it abandoned?

    The dialog applies to the image itself, so its response is not "did
    anything happen" -- it is only "how did the window close".  Closing it
    after a session is a SUCCESS, whether or not the user pressed Apply, and
    only on SUCCESS does GIMP store the procedure's last-used values.  The
    dialog reports its own window X / Escape after a live session as CLOSE
    for exactly that reason, so what still arrives here as DELETE_EVENT or
    CANCEL is a window that never got that far.  A boolean is accepted too,
    in case the dialog's entry point ever returns one.
    """
    if isinstance(response, bool):
        return response
    try:
        value = int(response)
    except (TypeError, ValueError):
        return bool(response)
    for name in ("DELETE_EVENT", "CANCEL", "NONE", "REJECT"):
        member = getattr(Gtk.ResponseType, name, None)
        if member is not None and value == int(member):
            return False
    return True



#: Response id of the Setup button on an argument dialog.  Any positive value
#: GTK does not reserve will do; it only has to be recognisable in the handler.
SETUP_RESPONSE = 100


def _add_setup_button(dialog):
    """Put "Setup / Doctor..." on a ``GimpUi.ProcedureDialog``.

    Returns a callable that reports whether it was pressed.  ``ProcedureDialog.run``
    answers only OK-or-not, so a Setup press comes back as "not OK", exactly
    like Cancel; watching the ``response`` signal is what tells them apart.
    Never raises -- a dialog without the button is still a working dialog.
    """
    state = {"setup": False}
    try:
        dialog.add_button("_Setup / Doctor\u2026", SETUP_RESPONSE)

        def on_response(_dialog, response_id):
            if response_id == SETUP_RESPONSE:
                state["setup"] = True

        dialog.connect("response", on_response)
    except Exception as exc:
        _log("could not add the Setup button: %s" % exc)
    return lambda: state["setup"]

def _run_scriptable(procedure, run_mode, image, drawables, config, engine, title):
    """Shared body of the two scriptable procedures.

    INTERACTIVE puts a ``GimpUi.ProcedureDialog`` over the declared arguments --
    the plain, non-canvas path.  WITH_LAST_VALS runs straight away on the values
    GIMP restored into ``config``.  NONINTERACTIVE runs on what the caller
    passed and never opens a window.
    """
    _arm_hang_dump()
    if run_mode == Gimp.RunMode.INTERACTIVE:
        _ui_init("sam3-gimp")
        dialog = GimpUi.ProcedureDialog(procedure=procedure, config=config)
        try:
            dialog.set_title(title)
        except Exception:
            pass
        # A button beside OK / Reset / Cancel, not a checkbox among the
        # arguments: Setup is an action, and this dialog had no other way to it.
        wants_setup = _add_setup_button(dialog)
        dialog.fill(None)
        ok = dialog.run()
        dialog.destroy()
        if wants_setup():
            _log("%s: Setup requested from the argument dialog" % engine)
            _open_setup(None)
            return _cancel(procedure)
        if not ok:
            return _cancel(procedure)

    blocked = _require_setup(procedure, run_mode)
    if blocked is not None:
        return blocked

    interactive = run_mode != Gimp.RunMode.NONINTERACTIVE
    if interactive:
        try:
            Gimp.progress_init(title)
        except Exception:
            pass
    # Log every invocation, not just the failures.  "I pressed it and nothing
    # happened" is only diagnosable if the log distinguishes never-ran from
    # ran-and-found-nothing, and on Windows this file is the only witness.
    _log("%s: starting (run_mode=%s, output=%s)"
         % (engine, run_mode, _get(config, "output-mode", "?")))
    try:
        count = _segment(image, drawables, config, engine)
    except Sam3Error as exc:
        _log("%s: failed: %s%s"
             % (engine, exc.message, (" | " + exc.hint) if exc.hint else ""))
        if interactive:
            _show_error("SAM 3", exc.message, exc.hint, exc.offer_setup)
        message = exc.message if not exc.hint else "%s %s" % (exc.message, exc.hint)
        return _failure(procedure, message)
    except Exception as exc:
        _log(traceback.format_exc())
        message = "SAM 3 segmentation failed: %s: %s" % (type(exc).__name__, exc)
        if interactive:
            _show_error("SAM 3", "Segmentation failed.", "%s: %s" % (type(exc).__name__, exc))
        return _failure(procedure, message)
    finally:
        if interactive:
            try:
                Gimp.progress_end()
            except Exception:
                pass

    _log("%s: applied %s instance(s)" % (engine, count))
    if interactive and not count:
        # Succeeding with nothing to show is indistinguishable from a no-op, so
        # say so rather than leaving the user staring at an unchanged image.
        _show_error(
            "SAM 3",
            "No objects matched that prompt.",
            "SAM 3 wants a simple noun phrase - \"red car\", \"person\", \"leaf\" - "
            "rather than a description like \"the car on the left\". Try a plainer "
            "word, or lower the score threshold.",
        )
    if interactive:
        Gimp.displays_flush()
    return _success(procedure, num_instances=count)


def run_segment_by_text(procedure, run_mode, image, drawables, config, run_data):
    return _run_scriptable(
        procedure, run_mode, image, drawables, config, "text", "Segment by text (SAM 3)"
    )


def run_segment_by_points(procedure, run_mode, image, drawables, config, run_data):
    return _run_scriptable(
        procedure, run_mode, image, drawables, config, "points", "Segment by points (SAM 3)"
    )


def _refresh_menu_label(ready_before):
    """Make GIMP re-read our menu labels if Setup changed what they should say.

    The Setup entry's label reports state ("first-time setup required" versus
    "Setup / Doctor"), but GIMP decides it once, at *query* time, and caches
    the answer in ``pluginrc`` until the plug-in file's mtime changes.  So
    after a successful install the warning label stayed in the menu for ever
    -- or until the next plug-in update.  Touching our own entry file is the
    documented way to invalidate that cache; GIMP re-queries at its next
    start.  Best effort: a read-only plug-in directory is not an error.
    """
    try:
        if bool(ready_before) == bool(_env_ready()):
            return
        os.utime(os.path.abspath(__file__), None)
        _log("environment readiness changed; touched %s so GIMP re-queries the menu"
             % os.path.basename(__file__))
    except Exception as exc:
        _log("could not touch the entry file to refresh the menu label: %s" % exc)


def run_setup(procedure, *args):
    """Setup / Doctor (``plug-in-sam3-setup``).

    A plain ``Gimp.Procedure``, so it can run with no image open.  GIMP 3.0
    passes ``(procedure, config, run_data)``; the signature is absorbed with
    ``*args`` and the config picked out by type, so a 3.0.x change to the
    run-func prototype cannot stop the user reaching the repair panel -- which
    is precisely the moment they need it most.
    """
    _arm_hang_dump()
    config = None
    for arg in args:
        if isinstance(arg, Gimp.ProcedureConfig):
            config = arg
            break
    _ = config  # the setup dialog keeps its own state; nothing to read yet.
    ready_before = _env_ready()

    try:
        _ui_init("sam3-gimp-setup")
        setup_dialog = _import_sibling("ui.setup_dialog")
        run = _need_attr(setup_dialog, "run_setup", "ui.setup_dialog")
        ok = run(None)
    except Sam3Error as exc:
        _show_error("SAM 3 Setup", exc.message, exc.hint, offer_setup=False)
        return _failure(procedure, exc.message)
    except Exception as exc:
        _log(traceback.format_exc())
        _show_error(
            "SAM 3 Setup",
            "The Setup panel failed to open.",
            "%s: %s\n\nThe plug-in directory is:\n%s"
            % (type(exc).__name__, exc, PLUGIN_DIR),
        )
        return _failure(procedure, str(exc))

    # A user who closes Setup without finishing the install has not failed at
    # anything; SUCCESS with nothing done is the honest status.
    if not ok:
        _log("setup closed without a ready environment")
    _refresh_menu_label(ready_before)
    return _success(procedure)


# =========================================================================== #
# argument declaration
# =========================================================================== #
def _output_mode_choice():
    choice = Gimp.Choice.new()
    for index, (nick, label, blurb) in enumerate(OUTPUT_MODE_CHOICES):
        choice.add(nick, index, label, blurb)
    return choice


def _add_common_arguments(procedure, max_instances_default, max_instances_max):
    procedure.add_boolean_argument(
        "use-projection",
        "Use visible _projection",
        "ON: segment what you see, all layers combined. "
        "OFF: segment only the active layer, ignoring everything above it.",
        True,
        GObject.ParamFlags.READWRITE,
    )
    procedure.add_choice_argument(
        "output-mode",
        "_Output",
        "What to build from the result. Selection = marching ants you can act on. "
        "Channels = one saved mask per object. Layer mask = cut the layer out. "
        "Layer group = one masked copy per object. Paths = editable vector outlines.",
        _output_mode_choice(),
        DEFAULT_OUTPUT_MODE,
        GObject.ParamFlags.READWRITE,
    )
    procedure.add_int_argument(
        "mask-threshold",
        "Mask _threshold",
        "How much of each mask to keep (0-255). LOWER IT (try 90) if the "
        "selection is too tight or has holes; RAISE IT (try 170) if it bleeds "
        "into the background. 128 is the model's own boundary.",
        1,
        255,
        DEFAULT_MASK_THRESHOLD,
        GObject.ParamFlags.READWRITE,
    )
    procedure.add_double_argument(
        "score-threshold",
        "_Score threshold",
        "How sure the model must be to keep a match. LOWER IT (try 0.15) if "
        "part of the object is missing or fewer objects were found than you "
        "expected; RAISE IT if unrelated things get selected. One object is "
        "often returned as several partial matches, so a high value here is "
        "the usual reason you get only a fragment.",
        0.0,
        1.0,
        DEFAULT_SCORE_THRESHOLD,
        GObject.ParamFlags.READWRITE,
    )
    procedure.add_int_argument(
        "max-instances",
        "_Maximum instances",
        "Upper bound on how many matches to apply, best first. Raise it for "
        "crowded scenes ('every car'); it does not affect a single object.",
        1,
        max_instances_max,
        max_instances_default,
        GObject.ParamFlags.READWRITE,
    )
    procedure.add_int_return_value(
        "num-instances",
        "Instances applied",
        "How many instances were written to the image",
        0,
        256,
        0,
        GObject.ParamFlags.READWRITE,
    )


def _add_text_argument(procedure):
    procedure.add_string_argument(
        "text",
        "_Prompt",
        'A simple noun phrase, e.g. "red car" (1-512 characters)',
        "",
        GObject.ParamFlags.READWRITE,
    )



def _always_sensitive():
    """``Gimp.ProcedureSensitivityMask.ALWAYS``, built defensively.

    Composed from whichever flags this GIMP exposes so that a missing name can
    only make the entry *more* available, never make registration fail --
    a raise here would remove the whole plug-in from the menus.
    """
    mask_type = Gimp.ProcedureSensitivityMask
    always = getattr(mask_type, "ALWAYS", None)
    if always is not None:
        return always
    value = 0
    for name in ("DRAWABLE", "DRAWABLES", "NO_DRAWABLES", "NO_IMAGE"):
        flag = getattr(mask_type, name, None)
        if flag is not None:
            value |= int(flag)
    return mask_type(value) if value else mask_type(0)

def _configure_image_procedure(procedure, menu_label, blurb, help_text, menu=True):
    procedure.set_image_types("RGB*, GRAY*, INDEXED*")
    procedure.set_sensitivity_mask(
        Gimp.ProcedureSensitivityMask.DRAWABLE
        | Gimp.ProcedureSensitivityMask.DRAWABLES
        | Gimp.ProcedureSensitivityMask.NO_DRAWABLES
    )
    procedure.set_menu_label(menu_label)
    if menu:
        procedure.add_menu_path(MENU_PATH)
    procedure.set_documentation(blurb, help_text, None)
    procedure.set_attribution(AUTHORS, COPYRIGHT, YEAR)


# =========================================================================== #
# the plug-in
# =========================================================================== #
class Sam3Plugin(Gimp.PlugIn):
    """Registration for every procedure this plug-in installs."""

    # ----------------------------------------------------------------- i18n
    def do_set_i18n(self, procname):
        """No translation catalogue is shipped yet.

        Returning False tells GIMP not to look for one; without it GIMP logs a
        warning about a missing domain for every procedure.
        """
        return False

    # ----------------------------------------------------------- procedures
    def do_query_procedures(self):
        return list(PROCEDURES)

    def do_create_procedure(self, name):
        if name == PROC_SEGMENT:
            return self._create_segment(name)
        if name == PROC_SEGMENT_TEXT:
            return self._create_segment_by_text(name)
        if name == PROC_SEGMENT_POINTS:
            return self._create_segment_by_points(name)
        if name == PROC_SETUP:
            return self._create_setup(name)
        return None

    # ------------------------------------------------------------- builders
    def _create_segment(self, name):
        procedure = Gimp.ImageProcedure.new(
            self, name, Gimp.PDBProcType.PLUGIN, run_segment, None
        )
        _configure_image_procedure(
            procedure,
            "_Segment interactively (canvas)...",
            "Open the SAM 3 canvas: preview masks, click to refine, then apply",
            "Opens the SAM 3 canvas: type a noun phrase to select every matching "
            "object at once (PCS), or click to select one object and refine it "
            "(PVS). Inference runs in a separate daemon process, so GIMP's "
            "embedded Python never needs torch.",
        )
        # Same arguments as segment-by-text: Gimp.ProcedureConfig then remembers
        # the last prompt, thresholds and output mode for the canvas dialog, and
        # the procedure has something to fall back to if the canvas is missing.
        _add_text_argument(procedure)
        _add_common_arguments(procedure, 64, 256)
        return procedure

    def _create_segment_by_text(self, name):
        procedure = Gimp.ImageProcedure.new(
            self, name, Gimp.PDBProcType.PLUGIN, run_segment_by_text, None
        )
        _configure_image_procedure(
            procedure,
            "Segment by _text prompt (all matching objects)...",
            "Type a noun phrase, select every matching object at once (scriptable)",
            "Promptable Concept Segmentation: one noun phrase returns every "
            "matching instance in the image. SAM 3 wants simple noun phrases "
            '("yellow school bus"), not relational descriptions ("the bus on '
            'the left"). Scriptable: callable non-interactively from Script-Fu.',
            menu=False,  # PDB / Script-Fu only; the canvas is the menu entry
        )
        _add_text_argument(procedure)
        _add_common_arguments(procedure, 64, 256)
        return procedure

    def _create_segment_by_points(self, name):
        procedure = Gimp.ImageProcedure.new(
            self, name, Gimp.PDBProcType.PLUGIN, run_segment_by_points, None
        )
        _configure_image_procedure(
            procedure,
            "Segment by _points or box (one object)...",
            "Click points or give a box to select a single object (scriptable)",
            "Promptable Visual Segmentation: positive and negative points, and "
            "optionally a bounding box, select a single object. Coordinates are "
            "in image pixels. Scriptable: callable non-interactively from "
            "Script-Fu.",
            menu=False,  # PDB / Script-Fu only; the canvas is the menu entry
        )
        procedure.add_string_argument(
            "points",
            "_Points",
            'Semicolon-separated "x,y,label" in image pixels; label 1 includes, '
            '0 excludes. Example: "512,300,1;640,410,0"',
            "",
            GObject.ParamFlags.READWRITE,
        )
        procedure.add_string_argument(
            "box",
            "_Box",
            'Optional bounding box "x0,y0,x1,y1" in image pixels',
            "",
            GObject.ParamFlags.READWRITE,
        )
        _add_common_arguments(procedure, 1, 8)
        return procedure

    def _create_setup(self, name):
        # A plain Gimp.Procedure, not an ImageProcedure: Setup must be reachable
        # with no image open, which is exactly the state of a first run.
        procedure = Gimp.Procedure.new(self, name, Gimp.PDBProcType.PLUGIN, run_setup, None)
        # Being a plain Procedure is not enough to be reachable with no image:
        # GIMP's default sensitivity for *any* procedure is "an image with one
        # or more drawables selected", so without this the menu entry sat
        # greyed out on an empty GIMP -- exactly the state of a first run, and
        # exactly when the user is told to "open Setup".
        procedure.set_sensitivity_mask(_always_sensitive())
        # The label reports state, so "not installed yet" is visible in the menu
        # rather than something you discover by pressing Segment.  Guarded: an
        # exception during registration would remove the plug-in entirely.
        try:
            ready = _env_ready()
        except Exception:
            ready = True
        procedure.set_menu_label(
            "SAM 3 Set_up / Doctor..." if ready
            else "\u26a0 SAM 3: first-time set_up required..."
        )
        procedure.add_menu_path(MENU_PATH)
        procedure.set_documentation(
            "Start here: install the SAM 3 environment, get the weights, diagnose problems",
            "Creates the sam3gimpd virtualenv, downloads the SAM 3 weights (which "
            "are gated - the panel walks through accepting the terms and "
            "supplying a HuggingFace token), and reports device, VRAM, dtype, "
            "daemon status and the last crash log.",
            None,
        )
        procedure.set_attribution(AUTHORS, COPYRIGHT, YEAR)
        return procedure


Gimp.main(Sam3Plugin.__gtype__, sys.argv)
