"""Execute the real entry point's registration under a minimal fake GIMP.

``tests/fake_gimp`` models images, layers and buffers for ``gimpbridge`` and
``outputs``; it knows nothing about procedure registration.  This file fakes
exactly the registration surface (``Gimp.PlugIn``, ``Gimp.Procedure``, the
sensitivity mask, run modes, ``GimpUi.ProcedureDialog``) and *runs*
``sam3_gimp.py``, so the assertions are about behaviour -- what the plug-in
actually asks GIMP for -- not about strings in the source.

Why this exists: Setup has to be reachable on an empty GIMP, the state of a
first run.  A procedure that never sets a sensitivity mask gets GIMP's default
-- "an image with one or more drawables selected" -- and is greyed out there.
"""

from __future__ import annotations

import enum
import faulthandler
import runpy
import sys
import types
from pathlib import Path

import pytest

ENTRY = Path(__file__).resolve().parents[2] / "plugin" / "sam3_gimp" / "sam3_gimp.py"


def _pygobject_available() -> bool:
    """PyGObject with the GTK 3 typelib -- bindings, not a display.

    The fake below covers ``Gimp`` and ``GimpUi``, but it hands the entry module
    the *real* ``GLib``, ``GObject`` and ``Gtk``: the plug-in declares GObject
    param specs, returns ``GLib.Error`` values and reads ``Gtk.ResponseType``,
    and a stand-in for those would be a reimplementation of PyGObject.  No
    window is ever realised here, so unlike the canvas tests this file is happy
    headless -- which is why it guards on the bindings rather than carrying
    ``needs_gtk``.  Without PyGObject at all (a base install, and the stub-only
    CI job) there is nothing to run and the file skips.
    """
    try:
        import gi

        gi.require_version("Gtk", "3.0")
        from gi.repository import GLib, GObject, Gtk  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _pygobject_available(),
    reason="PyGObject with the GTK 3 typelib is unavailable",
)


# --------------------------------------------------------------------------- #
# the fake registration surface
# --------------------------------------------------------------------------- #
class SensitivityMask(enum.IntFlag):
    DRAWABLE = 1
    DRAWABLES = 2
    NO_DRAWABLES = 4
    NO_IMAGE = 8
    ALWAYS = 15


class PDBProcType(enum.Enum):
    PLUGIN = "plugin"


class RunMode(enum.Enum):
    INTERACTIVE = 0
    NONINTERACTIVE = 1
    WITH_LAST_VALS = 2


class PDBStatusType(enum.Enum):
    SUCCESS = "success"
    CANCEL = "cancel"
    CALLING_ERROR = "calling-error"
    EXECUTION_ERROR = "execution-error"


class Choice:
    def __init__(self):
        self.entries = []

    @classmethod
    def new(cls):
        return cls()

    def add(self, nick, ident, label, help_text):
        self.entries.append((nick, ident, label, help_text))


class ProcedureConfig:
    """Records every property set, so a test can prove a flag was reset."""

    def __init__(self, **values):
        self.values = dict(values)
        self.sets = []

    def get_property(self, name):
        return self.values.get(name)

    def set_property(self, name, value):
        self.sets.append((name, value))
        self.values[name] = value


class Procedure:
    """Records what the plug-in asks GIMP for."""

    def __init__(self, plugin, name, proc_type, run_func, run_data):
        self.plugin, self.name, self.proc_type = plugin, name, proc_type
        self.run_func, self.run_data = run_func, run_data
        self.menu_label = None
        self.menu_paths = []
        self.sensitivity = None          # None == "never set" == GIMP default
        self.arguments = []              # (kind, name)
        self.image_types = None
        self.returns = []

    @classmethod
    def new(cls, plugin, name, proc_type, run_func, run_data):
        return cls(plugin, name, proc_type, run_func, run_data)

    def set_menu_label(self, label): self.menu_label = label
    def add_menu_path(self, path): self.menu_paths.append(path)
    def set_documentation(self, *a): pass
    def set_attribution(self, *a): pass
    def set_image_types(self, kinds): self.image_types = kinds
    def set_sensitivity_mask(self, mask): self.sensitivity = mask

    def _arg(self, kind, name):
        self.arguments.append((kind, name))

    def add_boolean_argument(self, name, *a): self._arg("bool", name)
    def add_int_argument(self, name, *a): self._arg("int", name)
    def add_double_argument(self, name, *a): self._arg("double", name)
    def add_string_argument(self, name, *a): self._arg("string", name)
    def add_choice_argument(self, name, *a): self._arg("choice", name)
    def add_int_return_value(self, name, *a): self.returns.append(name)

    def new_return_values(self, status, error):
        return ("return-values", status, error)


class ImageProcedure(Procedure):
    pass


class PlugIn:
    __gtype__ = "fake-gtype"


class ProcedureDialog:
    """A dialog the user either confirms or answers with the Setup button.

    ``press`` is what the simulated user does: ``None`` presses OK,
    ``"setup"`` presses the Setup button (a non-OK response, like Cancel).
    """

    instances = []
    press = None

    def __init__(self, procedure=None, config=None):
        self.procedure, self.config = procedure, config
        self.filled = False
        self.buttons = []
        self.handlers = []
        ProcedureDialog.instances.append(self)

    def set_title(self, t): self.title = t
    def fill(self, names): self.filled = True
    def add_button(self, text, response_id): self.buttons.append((text, response_id))
    def connect(self, signal, cb): self.handlers.append((signal, cb))

    def run(self):
        if ProcedureDialog.press == "setup":
            rid = next(r for _t, r in self.buttons if "Setup" in _t)
            for sig, cb in self.handlers:
                if sig == "response":
                    cb(self, rid)
            return False
        return True

    def destroy(self): pass


def _install_fake_gi(monkeypatch):
    import gi as real_gi
    real_gi.require_version("Gtk", "3.0")
    from gi.repository import GLib, GObject, Gtk  # noqa: E402

    gimp = types.ModuleType("gi.repository.Gimp")
    gimp.PlugIn = PlugIn
    gimp.Procedure = Procedure
    gimp.ImageProcedure = ImageProcedure
    gimp.ProcedureSensitivityMask = SensitivityMask
    gimp.PDBProcType = PDBProcType
    gimp.RunMode = RunMode
    gimp.PDBStatusType = PDBStatusType
    gimp.ProcedureConfig = ProcedureConfig
    gimp.Choice = Choice
    gimp.main = lambda *a, **k: None
    gimp.message = lambda *a, **k: None
    gimp.progress_init = gimp.progress_end = gimp.displays_flush = lambda *a, **k: None
    gimp.progress_update = lambda *a, **k: None

    gimpui = types.ModuleType("gi.repository.GimpUi")
    gimpui.init = lambda *a, **k: None
    gimpui.ProcedureDialog = ProcedureDialog

    repository = types.ModuleType("gi.repository")
    repository.Gimp, repository.GimpUi = gimp, gimpui
    repository.GObject, repository.GLib, repository.Gtk = GObject, GLib, Gtk

    fake_gi = types.ModuleType("gi")
    fake_gi.require_version = lambda *a, **k: None
    fake_gi.repository = repository

    for name, module in (("gi", fake_gi), ("gi.repository", repository),
                         ("gi.repository.Gimp", gimp), ("gi.repository.GimpUi", gimpui)):
        monkeypatch.setitem(sys.modules, name, module)
    return gimp


@pytest.fixture
def entry(monkeypatch, sam3_home):
    """The executed entry module, with Setup/launch side effects intercepted."""
    _install_fake_gi(monkeypatch)
    ns = runpy.run_path(str(ENTRY), run_name="sam3_gimp_under_test")
    yield ns
    # The module keeps plugin.log open for faulthandler.  Point faulthandler
    # back at stderr before closing it, or a later crash in the suite would be
    # written into whatever reuses the descriptor.
    crash_log = ns.get("_CRASH_LOG")
    if crash_log is not None:
        faulthandler.cancel_dump_traceback_later()
        faulthandler.enable(file=sys.__stderr__, all_threads=True)
        crash_log.close()


# --------------------------------------------------------------------------- #
# what the plug-in registers
# --------------------------------------------------------------------------- #
def test_setup_is_registered_first_and_always_sensitive(entry):
    plugin = entry["Sam3Plugin"]()
    names = plugin.do_query_procedures()
    assert names[0] == "plug-in-sam3-setup"

    proc = plugin.do_create_procedure("plug-in-sam3-setup")
    assert isinstance(proc, Procedure) and not isinstance(proc, ImageProcedure)
    assert proc.menu_paths == [entry["MENU_PATH"]]
    assert proc.menu_label and "et_up" in proc.menu_label
    # The whole bug: unset means "needs an image with a selected drawable".
    assert proc.sensitivity is not None, "Setup left at GIMP's default sensitivity"
    assert proc.sensitivity == SensitivityMask.ALWAYS


def test_setup_label_reports_an_uninstalled_environment(entry):
    """With nothing installed the entry says so in the menu itself."""
    proc = entry["Sam3Plugin"]().do_create_procedure("plug-in-sam3-setup")
    assert "first-time" in proc.menu_label.lower()


def test_the_menu_holds_only_the_canvas_and_setup(entry):
    """The scriptable procedures stay in the PDB for Script-Fu, but as menu
    entries they were strictly worse than the canvas and only confused."""
    plugin = entry["Sam3Plugin"]()
    in_menu = {name for name in plugin.do_query_procedures()
               if plugin.do_create_procedure(name).menu_paths}
    assert in_menu == {"plug-in-sam3-setup", "plug-in-sam3-segment"}


def test_scriptable_procedures_are_still_registered(entry):
    plugin = entry["Sam3Plugin"]()
    for name in ("plug-in-sam3-segment-by-text", "plug-in-sam3-segment-by-points"):
        proc = plugin.do_create_procedure(name)
        assert proc is not None and proc.menu_paths == []
        assert any(n == "text" or n == "points" for _k, n in proc.arguments)


def test_no_segment_procedure_carries_the_old_setup_checkbox(entry):
    plugin = entry["Sam3Plugin"]()
    for name in plugin.do_query_procedures():
        proc = plugin.do_create_procedure(name)
        assert all(n != "open-setup" for _k, n in proc.arguments), name


# --------------------------------------------------------------------------- #
# what happens when the user reaches for Setup
# --------------------------------------------------------------------------- #
def test_menu_entry_runs_the_setup_dialog(entry, monkeypatch):
    """The exact path the menu takes, with the dialog itself intercepted."""
    opened = []
    import ui.setup_dialog as setup_dialog
    monkeypatch.setattr(setup_dialog, "run_setup", lambda parent=None, **k: opened.append(1) or True)
    plugin = entry["Sam3Plugin"]()
    proc = plugin.do_create_procedure("plug-in-sam3-setup")
    result = proc.run_func(proc, ProcedureConfig(), None)
    assert opened == [1]
    assert result[1] == PDBStatusType.SUCCESS


def test_the_argument_dialog_has_a_setup_button(entry, monkeypatch):
    ProcedureDialog.instances.clear(); ProcedureDialog.press = None
    entry["_run_scriptable"].__globals__["_require_setup"] = (
        lambda proc, mode: proc.new_return_values(PDBStatusType.CALLING_ERROR, None))
    plugin = entry["Sam3Plugin"]()
    proc = plugin.do_create_procedure("plug-in-sam3-segment-by-text")
    proc.run_func(proc, RunMode.INTERACTIVE, None, [], ProcedureConfig(text="guitar"), None)
    dialog = ProcedureDialog.instances[-1]
    assert any("Setup" in text for text, _r in dialog.buttons), dialog.buttons
    assert any(sig == "response" for sig, _cb in dialog.handlers)


def test_pressing_setup_on_the_argument_dialog_opens_setup_before_anything_else(entry, monkeypatch):
    opened = []
    entry["_run_scriptable"].__globals__["_open_setup"] = lambda parent=None: opened.append(1) or True

    def must_not_gate(*a, **k):
        raise AssertionError("readiness gate ran before the Setup button was honoured")
    entry["_run_scriptable"].__globals__["_require_setup"] = must_not_gate

    ProcedureDialog.press = "setup"
    try:
        plugin = entry["Sam3Plugin"]()
        proc = plugin.do_create_procedure("plug-in-sam3-segment-by-text")
        result = proc.run_func(proc, RunMode.INTERACTIVE, None, [],
                               ProcedureConfig(text="guitar"), None)
    finally:
        ProcedureDialog.press = None
    assert opened == [1], "Setup was not opened"
    assert result[1] == PDBStatusType.CANCEL


def test_pressing_ok_proceeds_to_segmentation(entry, monkeypatch):
    opened, gated = [], []
    entry["_run_scriptable"].__globals__["_open_setup"] = lambda p=None: opened.append(1)
    entry["_run_scriptable"].__globals__["_require_setup"] = (
        lambda proc, mode: gated.append(1) or proc.new_return_values(PDBStatusType.CALLING_ERROR, None))
    ProcedureDialog.press = None
    plugin = entry["Sam3Plugin"]()
    proc = plugin.do_create_procedure("plug-in-sam3-segment-by-text")
    proc.run_func(proc, RunMode.INTERACTIVE, None, [], ProcedureConfig(text="guitar"), None)
    assert opened == [] and gated == [1]



def test_run_segment_waits_for_its_threads_before_returning(entry, monkeypatch):
    """A background thread the dialog left running is joined before run()
    hands control back to GIMP -- the moment libgimp calls exit()."""
    import threading, time
    import ui.main_dialog as main_dialog
    from gi.repository import Gtk

    started = []

    def fake_dialog(parent=None, **kwargs):
        t = threading.Thread(target=lambda: time.sleep(0.4), name="sam3-release", daemon=True)
        t.start(); started.append(t)
        return int(Gtk.ResponseType.CLOSE)

    monkeypatch.setattr(main_dialog, "run_main_dialog", fake_dialog)
    entry["run_segment"].__globals__["_require_setup"] = lambda proc, mode: None

    plugin = entry["Sam3Plugin"]()
    proc = plugin.do_create_procedure("plug-in-sam3-segment")
    result = proc.run_func(proc, RunMode.INTERACTIVE, None, [], ProcedureConfig(), None)

    assert started and not started[0].is_alive(), "returned to GIMP with a thread still running"
    assert result[1] == PDBStatusType.SUCCESS


def test_quiesce_gives_up_on_a_stuck_thread_instead_of_hanging(entry):
    import threading, time
    stop = threading.Event()
    t = threading.Thread(target=stop.wait, name="sam3-stuck", daemon=True)
    t.start()
    try:
        t0 = time.time()
        entry["_quiesce_threads"](timeout=0.3)
        assert time.time() - t0 < 2.0
        assert t.is_alive()          # not killed, just no longer waited for
    finally:
        stop.set()


# --------------------------------------------------------------------------- #
# plugin.log
# --------------------------------------------------------------------------- #
def test_trimming_the_log_keeps_the_crash_log_attached(entry):
    """Trimming plugin.log must leave the faulthandler file (_CRASH_LOG)
    attached to it.  Deleting the file instead sends every later crash or
    hang dump to an unlinked file on POSIX, and fails on Windows, where a
    file that is open cannot be removed -- so the cap would never apply."""
    import os
    path = entry["_plugin_log_path"]()
    crash = entry["_CRASH_LOG"]
    assert crash is not None and os.path.samefile(crash.name, path)
    with open(path, "a") as fh:
        fh.write("filler line\n" * (entry["LOG_LIMIT_BYTES"] // 12 + 10))
    entry["_log"]("this line triggers the trim")
    assert os.path.getsize(path) <= entry["LOG_KEEP_BYTES"] + 4096
    assert os.fstat(crash.fileno()).st_nlink == 1, "the crash log lost its file"
    crash.write("FAULTHANDLER DUMP WOULD GO HERE\n")
    crash.flush()
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    assert "FAULTHANDLER DUMP WOULD GO HERE" in text
    assert "this line triggers the trim" in text
    assert text.splitlines()[0].endswith("(plug-in %s)" % entry["__version__"])
    assert "trimmed" in text.splitlines()[0]


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX permission bits")
def test_plugin_log_is_owner_only(entry):
    import os
    import stat
    path = entry["_plugin_log_path"]()
    os.chmod(path, 0o644)       # as an older version left it
    entry["_log"]("hello")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


# --------------------------------------------------------------------------- #
# which drawable the dialog is given
# --------------------------------------------------------------------------- #
def test_run_segment_hands_the_dialog_a_layer_not_a_mask(entry, monkeypatch):
    """GIMP passes the selected *drawables*: after a Layer mask Apply that is
    the mask, whose pixels are grey and whose copy is a channel.  The entry
    point resolves the layer once, through gimpbridge.source_layer."""
    import gimpbridge
    import ui.main_dialog as main_dialog
    from gi.repository import Gtk

    mask, layer, image = object(), object(), object()
    monkeypatch.setattr(gimpbridge, "source_layer",
                        lambda img, drawable: layer if drawable is mask else drawable)
    seen = {}

    def fake_dialog(parent=None, **kwargs):
        seen.update(kwargs)
        return int(Gtk.ResponseType.CLOSE)

    monkeypatch.setattr(main_dialog, "run_main_dialog", fake_dialog)
    entry["run_segment"].__globals__["_require_setup"] = lambda proc, mode: None
    plugin = entry["Sam3Plugin"]()
    proc = plugin.do_create_procedure("plug-in-sam3-segment")
    proc.run_func(proc, RunMode.INTERACTIVE, image, [mask], ProcedureConfig(), None)
    assert seen["drawable"] is layer
