"""The onboarding window: install the environment, get the weights, run Doctor.

``DESIGN.md`` §8 calls the bootstrap flow a design pillar, and this is its face.
The plug-in this one succeeds asked users to clone a repo, pip-install SAM,
download ``.pth`` files and hand-configure paths; most installs died there.  So
this dialog has exactly three jobs and shows them as three tabs:

============  ==========================================================
**Install**   one button.  Download ``uv``, build a pinned-Python venv,
              install torch from the PyTorch index for the GPU, then
              ``sam3gimpd[runtime]``.  A live log, a real progress bar and a
              Cancel that works, because a silent ten-minute freeze inside
              GIMP is indistinguishable from a crash.
**Weights**   the SAM 3 checkpoint is **gated** on HuggingFace.  Explained
              in plain words, with a link to accept Meta's terms, a token
              box that is validated *before* a 3.6 GB download starts --
              and a second exit for people who already have the files on
              disk.
**Doctor**    device, VRAM, dtype, versions, daemon status, last crash log,
              Repair.  Driven by ``sam3gimpd doctor``.
============  ==========================================================

All logic lives in ``bootstrap.py``; this module is the view.  Two rules it
never breaks:

* **Nothing blocks the GTK main loop** (DESIGN.md §4).  Every install step,
  subprocess and HTTP call happens on a worker thread and comes back through
  ``GLib.idle_add``.
* **Only stdlib and ``gi``** are imported -- this runs inside GIMP's embedded
  Python, which has nothing else.

It also runs standalone on any desktop with GTK 3 (``python3 ui/setup_dialog.py``),
which is how it is developed.
"""

from __future__ import annotations

import ipaddress
import os
import sys
import threading
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, GObject, Gtk, Pango  # noqa: E402

# ``bootstrap`` is a sibling module.  Inside GIMP the plug-in directory is on
# ``sys.path`` (sam3_gimp.py puts it there), so a flat import works; the
# fallback covers being imported as ``sam3_gimp.ui.setup_dialog``.
try:
    import bootstrap
except ImportError:  # pragma: no cover - packaging fallback
    from .. import bootstrap  # type: ignore

# ``launcher`` owns settings.json, where the "use my own Python" choice lives.
# Optional on purpose: a missing launcher must degrade the row to read-only
# rather than stop the whole Setup dialog from opening.
try:
    import launcher as launcher_mod
except ImportError:  # pragma: no cover - packaging fallback
    try:
        from .. import launcher as launcher_mod  # type: ignore
    except ImportError:
        launcher_mod = None  # type: ignore

__all__ = ["SetupDialog", "needs_setup", "run_setup"]

_MAX_LOG_LINES = 4000


# --------------------------------------------------------------------------- #
# small GTK helpers
# --------------------------------------------------------------------------- #
#: A wrapping ``Gtk.Label`` with no width bound reports its natural height for
#: an arbitrarily narrow width, and a dialog built from several of them ends up
#: thousands of pixels tall.  Every wrapping label here is capped.
_WRAP_CHARS = 78


def _wrappable(lab: Gtk.Label) -> Gtk.Label:
    lab.set_line_wrap(True)
    lab.set_line_wrap_mode(Pango.WrapMode.WORD_CHAR)
    lab.set_max_width_chars(_WRAP_CHARS)
    lab.set_width_chars(24)
    return lab


def _label(text: str, *, bold: bool = False, wrap: bool = True, xalign: float = 0.0) -> Gtk.Label:
    lab = Gtk.Label()
    if bold:
        lab.set_markup("<b>%s</b>" % GLib.markup_escape_text(text))
    else:
        lab.set_text(text)
    lab.set_xalign(xalign)
    if wrap:
        _wrappable(lab)
    return lab


def _tab(text: str) -> Gtk.Label:
    """Notebook tab titles must never wrap -- "Doctor" turning into "Doc-tor"."""
    lab = Gtk.Label(label=text)
    lab.set_single_line_mode(True)
    return lab


def _dim(text: str) -> Gtk.Label:
    lab = Gtk.Label()
    lab.set_markup('<span alpha="70%%">%s</span>' % GLib.markup_escape_text(text))
    lab.set_xalign(0.0)
    return _wrappable(lab)


def _scrolled(child: Gtk.Widget) -> Gtk.ScrolledWindow:
    """Each notebook page scrolls, so the dialog can open smaller than its
    content instead of forcing a window taller than the screen."""
    sw = Gtk.ScrolledWindow()
    sw.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
    sw.set_propagate_natural_height(False)
    sw.add(child)
    return sw


def _frame(title: str, child: Gtk.Widget) -> Gtk.Frame:
    frame = Gtk.Frame()
    frame.set_label_widget(_label(title, bold=True))
    frame.set_shadow_type(Gtk.ShadowType.NONE)
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    box.set_margin_start(12)
    box.set_margin_top(6)
    box.set_margin_bottom(6)
    box.pack_start(child, True, True, 0)
    frame.add(box)
    return frame


def _vbox(spacing: int = 8, margin: int = 12) -> Gtk.Box:
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=spacing)
    for setter in ("set_margin_start", "set_margin_end", "set_margin_top", "set_margin_bottom"):
        getattr(box, setter)(margin)
    return box


def _hbox(spacing: int = 6) -> Gtk.Box:
    return Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=spacing)




def _restart_daemon_for_new_code() -> None:
    """Stop a running daemon after its package was reinstalled.

    The launcher's handshake only checks the API major, so an old daemon keeps
    answering /hello -- and keeps its old code -- until it idles out.  Ending
    it here is what makes "Reinstall / update" actually take effect.
    """
    if launcher_mod is None:
        return
    try:
        info = launcher_mod.read_runtime_info()
        if info:
            launcher_mod.shutdown_daemon(info, grace_ms=0)
            launcher_mod.delete_runtime_file()
    except Exception:  # noqa: BLE001 -- best effort; the idle TTL is the backstop
        pass


def remote_is_off_machine(url: str) -> bool:
    """Does a remote-daemon address point anywhere but this computer?

    The remote route is plain HTTP, so anything that leaves the machine
    carries the bearer token and every uploaded image in the clear.  Loopback
    addresses (an SSH tunnel's local end) are the safe case.  An address that
    cannot be parsed counts as off the machine.
    """
    import urllib.parse  # noqa: PLC0415

    text = (url or "").strip()
    if not text:
        return False
    if "://" not in text:
        text = "http://" + text
    try:
        host = (urllib.parse.urlsplit(text).hostname or "").strip().lower()
    except ValueError:
        return True
    if not host:
        return True
    if host == "localhost" or host.endswith(".localhost"):
        return False
    try:
        return not ipaddress.ip_address(host).is_loopback
    except ValueError:
        return True


class PathPicker(Gtk.Box):
    """An editable path entry plus a Browse button.

    Replaces ``Gtk.FileChooserButton``, which is deprecated and, more to the
    point, unreliable here: on at least one Windows GIMP the button simply did
    not open its dialog, leaving no way to choose a file at all.  This opens a
    ``Gtk.FileChooserDialog`` explicitly, with a transient parent so it cannot
    appear behind the Setup window, and -- the part that matters when a native
    dialog misbehaves -- the entry can always be typed or pasted into.
    """

    def __init__(self, title: str, action: Gtk.FileChooserAction,
                 placeholder: str = "") -> None:
        Gtk.Box.__init__(self, orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self._title = title
        self._action = action
        self.entry = Gtk.Entry()
        self.entry.set_hexpand(True)
        if placeholder:
            self.entry.set_placeholder_text(placeholder)
        self.pack_start(self.entry, True, True, 0)
        self._browse = Gtk.Button(label="Browse…")
        self._browse.connect("clicked", self._on_browse)
        self.pack_start(self._browse, False, False, 0)

    def get_filename(self) -> str:
        return (self.entry.get_text() or "").strip().strip('"')

    def set_filename(self, path: str) -> None:
        self.entry.set_text(path or "")

    def _on_browse(self, *_args: Any) -> None:
        parent = self.get_toplevel()
        if not isinstance(parent, Gtk.Window):
            parent = None
        dialog = Gtk.FileChooserDialog(
            title=self._title, transient_for=parent, action=self._action
        )
        dialog.add_buttons("_Cancel", Gtk.ResponseType.CANCEL,
                           "_Open", Gtk.ResponseType.ACCEPT)
        current = self.get_filename()
        if current:
            try:
                if os.path.isdir(current):
                    dialog.set_current_folder(current)
                elif os.path.isfile(current):
                    dialog.set_filename(current)
            except Exception:  # pragma: no cover
                pass
        try:
            if dialog.run() == Gtk.ResponseType.ACCEPT:
                chosen = dialog.get_filename()
                if chosen:
                    self.set_filename(chosen)
                    self.emit("changed-path")
        finally:
            dialog.destroy()


GObject.type_register(PathPicker)
GObject.signal_new("changed-path", PathPicker,
                   GObject.SignalFlags.RUN_FIRST, None, ())


class LogView(Gtk.ScrolledWindow):
    """A monospace, auto-scrolling, bounded transcript.

    Bounded matters: ``uv pip install torch`` emits thousands of lines and an
    unbounded ``Gtk.TextBuffer`` inside GIMP is a memory leak with a UI.
    """

    def __init__(self, height: int = 220) -> None:
        super().__init__()
        self.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.set_shadow_type(Gtk.ShadowType.IN)
        self.set_min_content_height(height)
        self._view = Gtk.TextView()
        self._view.set_editable(False)
        self._view.set_cursor_visible(False)
        self._view.set_monospace(True)
        self._view.set_wrap_mode(Gtk.WrapMode.CHAR)
        self._buffer = self._view.get_buffer()
        self.add(self._view)
        self._pinned = True
        adj = self.get_vadjustment()
        if adj is not None:
            adj.connect("value-changed", self._on_scrolled)

    def _on_scrolled(self, adj: Gtk.Adjustment) -> None:
        # Stop auto-scrolling as soon as the user scrolls up to read something.
        at_end = adj.get_value() + adj.get_page_size() >= adj.get_upper() - 24.0
        self._pinned = bool(at_end)

    def append(self, text: str) -> None:
        end = self._buffer.get_end_iter()
        self._buffer.insert(end, text.rstrip("\n") + "\n")
        if self._buffer.get_line_count() > _MAX_LOG_LINES:
            start = self._buffer.get_start_iter()
            cut = self._buffer.get_iter_at_line(self._buffer.get_line_count() - _MAX_LOG_LINES)
            self._buffer.delete(start, cut)
        if self._pinned:
            mark = self._buffer.create_mark(None, self._buffer.get_end_iter(), False)
            self._view.scroll_mark_onscreen(mark)
            self._buffer.delete_mark(mark)

    def set_text(self, text: str) -> None:
        self._buffer.set_text(text)

    def clear(self) -> None:
        self._buffer.set_text("")

    @property
    def text(self) -> str:
        start, end = self._buffer.get_bounds()
        return self._buffer.get_text(start, end, False)


# --------------------------------------------------------------------------- #
# worker-thread plumbing
# --------------------------------------------------------------------------- #
class _Worker:
    """One background job at a time, with results marshalled onto the GTK loop.

    Deliberately single-slot: the install, the token check and Doctor must never
    run concurrently -- they all touch the same venv.  A job that arrives while
    another runs can be queued rather than dropped (:meth:`queue`).

    ``alive`` says whether the dialog still exists; once it does not, ``done``
    callbacks are skipped and nothing queued starts.
    """

    def __init__(self, alive: Callable[[], bool] = lambda: True) -> None:
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._alive = alive
        self._queued: List[Tuple[Callable[[], Any], Callable[[Any, Optional[BaseException]], Any]]] = []
        self.cancel = threading.Event()

    @property
    def busy(self) -> bool:
        return self._running

    def start(self, fn: Callable[[], Any], done: Callable[[Any, Optional[BaseException]], Any],
              *, cancel: Optional[threading.Event] = None) -> bool:
        """Run ``fn`` on a thread and ``done(result, error)`` on the GTK loop.

        ``cancel`` is the Event :meth:`request_cancel` sets.  A job that
        watches an Event of its own -- the Installer, built before this call --
        must pass that same object here, or Cancel and Close never reach it.
        """
        if self._running:
            return False
        self.cancel = cancel if cancel is not None else threading.Event()
        self._running = True

        def _run() -> None:
            result: Any = None
            error: Optional[BaseException] = None
            try:
                result = fn()
            except BaseException as exc:  # noqa: BLE001 - reported to the user
                error = exc
                traceback.print_exc()
            GLib.idle_add(self._finish, done, result, error, priority=GLib.PRIORITY_DEFAULT)

        self._thread = threading.Thread(target=_run, name="sam3-setup", daemon=True)
        self._thread.start()
        return True

    def _finish(self, done: Callable[[Any, Optional[BaseException]], Any],
                result: Any, error: Optional[BaseException]) -> bool:
        self._running = False
        if self._alive():
            try:
                done(result, error)
            except Exception:  # noqa: BLE001 -- a broken callback must not stall the queue
                traceback.print_exc()
            if self._queued and not self._running:
                fn, then = self._queued.pop(0)
                self.start(fn, then)
        return False

    def queue(self, fn: Callable[[], Any], done: Callable[[Any, Optional[BaseException]], Any]) -> bool:
        """Start now, or as soon as the running job has finished.  Returns
        True when it started immediately."""
        if self.start(fn, done):
            return True
        if (fn, done) not in self._queued:
            self._queued.append((fn, done))
        return False

    def request_cancel(self) -> None:
        self.cancel.set()

    def shutdown(self) -> None:
        """The dialog is going away: drop queued work and cancel the job."""
        self._queued = []
        self.cancel.set()

    def join(self, timeout: float) -> None:
        """Wait (at most ``timeout`` s) for the running thread to end."""
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)


# --------------------------------------------------------------------------- #
# the dialog
# --------------------------------------------------------------------------- #
class SetupDialog(Gtk.Dialog):
    """Install / weights / doctor.  Modal, resizable, safe to reopen any time."""

    def __init__(self, parent: Optional[Gtk.Window] = None, *, page: str = "install") -> None:
        super().__init__(title="SAM 3 — Setup", transient_for=parent, modal=parent is not None)
        self.set_default_size(820, 700)

        # Set on "destroy".  Worker threads keep running briefly after Close
        # (until their child processes are stopped), and everything they
        # schedule on the GTK loop checks this first: writing into destroyed
        # widgets is a stream of Gtk-CRITICALs at best.
        self._closed = False
        self._worker = _Worker(alive=lambda: not self._closed)
        self._report: Optional[bootstrap.EnvironmentReport] = None
        self._plan: Optional[bootstrap.InstallPlan] = None
        self._token_ok = False
        # Detected once per dialog.  ``inspect_environment`` with no accelerator
        # runs ``nvidia-smi`` (about a second on Windows; each query is cut off
        # at 15 s) on the GTK thread, and refresh() is called on every checkbox
        # and combo change -- the dialog visibly hitched each time.  A GPU does
        # not appear or vanish while a window is open.
        self._detected_accel: Optional[bootstrap.Accelerator] = None

        self._notebook = Gtk.Notebook()
        content = self.get_content_area()
        content.set_spacing(0)
        content.pack_start(self._notebook, True, True, 0)

        # The Install page scrolls its *explanatory* upper half only; the
        # progress bar, the log and the Install button row stay pinned at the
        # bottom.  Inside one big scroller the Install button sat below the
        # log, off-screen at the default window size -- a Setup dialog whose
        # Setup button had to be hunted for.
        self._notebook.append_page(self._build_install_page(), _tab("Install"))
        self._notebook.append_page(_scrolled(self._build_weights_page()), _tab("SAM 3 weights"))
        self._notebook.append_page(_scrolled(self._build_doctor_page()), _tab("Doctor"))

        self._status = _label("")
        self._status.set_xalign(0.0)
        self._status.set_margin_start(12)
        self._status.set_margin_end(12)
        self._status.set_margin_bottom(6)
        content.pack_start(self._status, False, False, 0)

        self.add_button("Close", Gtk.ResponseType.CLOSE)
        self.connect("response", self._on_response)
        self.connect("delete-event", self._on_delete)
        self.connect("destroy", self._on_destroy)

        self.show_all()
        # After show_all(): a Gtk.Notebook silently ignores set_current_page()
        # while its pages are still unrealised, so opening straight on the
        # Doctor tab (what the main dialog's Doctor button does) would land on
        # Install instead.
        pages = {"install": 0, "weights": 1, "doctor": 2}
        index = pages.get(page, 0)
        self._notebook.set_current_page(index)
        self.refresh()
        if index == 2:
            # Queued, not dropped: the existing-environment row may already
            # have the worker busy probing its interpreter.
            self._idle(self._run_doctor)

    # ------------------------------------------------------------------ #
    # GTK-loop plumbing
    # ------------------------------------------------------------------ #
    def _idle(self, fn: Callable[..., Any], *args: Any) -> None:
        """``GLib.idle_add(fn, *args)`` that does nothing once the dialog is
        gone.  Every callback from a worker thread comes through here."""
        def _guarded() -> bool:
            if not self._closed:
                fn(*args)
            return False
        GLib.idle_add(_guarded)

    def _log_line(self, line: str) -> None:
        """Thread-safe: append to the log view from any thread."""
        self._idle(self._log.append, line)

    # ------------------------------------------------------------------ #
    # page 1: install
    # ------------------------------------------------------------------ #
    def _build_install_page(self) -> Gtk.Widget:
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        page = _vbox()
        outer.pack_start(_scrolled(page), True, True, 0)

        page.pack_start(
            _dim(
                "SAM 3 needs PyTorch, which cannot live inside GIMP's own Python. "
                "This builds a small private environment beside it — nothing is "
                "installed system-wide and nothing outside the folder below is touched. "
                "There is nothing to clone or compile: SAM 3 runs through the "
                "transformers library installed here, and the model weights come "
                "from the next tab."
            ),
            False, False, 0,
        )

        self._env_summary = _label("Checking…", bold=True, wrap=False)
        self._env_summary.set_ellipsize(Pango.EllipsizeMode.MIDDLE)
        page.pack_start(self._env_summary, False, False, 0)

        self._plan_view = Gtk.Grid(column_spacing=12, row_spacing=2)
        page.pack_start(_frame("What will happen", self._plan_view), False, False, 0)
        # Advisory sentences from the plan (an NVIDIA driver too old for the
        # CUDA 12.8 wheels, say).  Hidden when there are none.
        self._plan_warnings = _label("", wrap=True)
        self._plan_warnings.set_no_show_all(True)
        page.pack_start(self._plan_warnings, False, False, 0)

        page.pack_start(self._build_daemon_source_frame(), False, False, 0)
        page.pack_start(self._build_existing_env_frame(), False, False, 0)

        # -- advanced ---------------------------------------------------- #
        adv = Gtk.Expander(label="Advanced")
        adv_box = _vbox(6, 6)
        row = _hbox()
        row.pack_start(_label("Accelerator:", wrap=False), False, False, 0)
        self._device_combo = Gtk.ComboBoxText()
        for key, text in (
            ("auto", "Detect automatically"),
            ("cuda", "NVIDIA CUDA"),
            ("rocm", "AMD ROCm (Linux only)"),
            ("mps", "Apple Silicon (MPS)"),
            ("cpu", "CPU only"),
        ):
            self._device_combo.append(key, text)
        self._device_combo.set_active_id("auto")
        self._device_combo.connect("changed", lambda *_a: self.refresh())
        row.pack_start(self._device_combo, False, False, 0)
        adv_box.pack_start(row, False, False, 0)

        self._force_check = Gtk.CheckButton(label="Reinstall from scratch (ignore what is already there)")
        self._force_check.connect("toggled", lambda *_a: self.refresh())
        adv_box.pack_start(self._force_check, False, False, 0)

        self._weights_check = Gtk.CheckButton(label="Also download the SAM 3 weights (needs a HuggingFace token)")
        adv_box.pack_start(self._weights_check, False, False, 0)

        # The daemon's idle exit was fixed at 30 minutes in code.  Small
        # cards want it shorter, all-day users want it off.
        idle_row = _hbox()
        idle_row.pack_start(_label("Daemon exits after idle minutes:", wrap=False), False, False, 0)
        self._idle_spin = Gtk.SpinButton.new_with_range(0, 24 * 60, 5)
        self._idle_spin.set_tooltip_text(
            "The daemon frees the GPU by exiting after this long without a request. "
            "0 keeps it running until GIMP closes. Takes effect the next time it starts."
        )
        minutes = 30.0
        if launcher_mod is not None:
            try:
                seconds = launcher_mod.configured_idle_ttl()
                if seconds is not None:
                    minutes = seconds / 60.0
            except Exception:  # pragma: no cover
                pass
        self._idle_spin.set_value(minutes)
        self._idle_spin.connect("value-changed", self._on_idle_changed)
        idle_row.pack_start(self._idle_spin, False, False, 0)
        adv_box.pack_start(idle_row, False, False, 0)

        # A daemon on another machine, reached through an SSH tunnel: the
        # protocol is plain HTTP, so the tunnel is what keeps it private.
        remote_box = _vbox(4, 0)
        remote_box.pack_start(
            _dim("Use a daemon running on another machine. The connection is plain, "
                 "unencrypted HTTP, so reach it through an SSH tunnel: on that machine "
                 "run 'sam3gimpd serve --port 8765', here run "
                 "'ssh -L 8765:127.0.0.1:8765 <gpu-host>', and use the address "
                 "http://127.0.0.1:8765. The token is in that machine's runtime.json."),
            False, False, 0)
        url_row = _hbox()
        url_row.pack_start(_label("Address:", wrap=False), False, False, 0)
        self._remote_url = Gtk.Entry()
        self._remote_url.set_placeholder_text("http://127.0.0.1:8765")
        self._remote_url.set_hexpand(True)
        self._remote_url.connect("changed", lambda *_a: self._update_remote_warning())
        url_row.pack_start(self._remote_url, True, True, 0)
        url_row.pack_start(_label("Token:", wrap=False), False, False, 0)
        self._remote_token = Gtk.Entry()
        self._remote_token.set_visibility(False)
        self._remote_token.set_placeholder_text("from runtime.json")
        url_row.pack_start(self._remote_token, True, True, 0)
        remote_box.pack_start(url_row, False, False, 0)
        btn_row = _hbox()
        self._remote_save = Gtk.Button(label="Use this daemon")
        self._remote_save.connect("clicked", self._on_remote_save)
        btn_row.pack_start(self._remote_save, False, False, 0)
        self._remote_clear = Gtk.Button(label="Back to a local daemon")
        self._remote_clear.connect("clicked", self._on_remote_clear)
        btn_row.pack_start(self._remote_clear, False, False, 0)
        remote_box.pack_start(btn_row, False, False, 0)
        # Shown whenever the address leaves this computer.
        self._remote_warning = _label("", wrap=True)
        self._remote_warning.set_no_show_all(True)
        remote_box.pack_start(self._remote_warning, False, False, 0)
        self._remote_status = _label("", wrap=True)
        remote_box.pack_start(self._remote_status, False, False, 0)
        adv_box.pack_start(_frame("Remote daemon (optional)", remote_box), False, False, 0)
        self._refresh_remote_row()

        adv.add(adv_box)
        page.pack_start(adv, False, False, 0)

        # -- progress + log + buttons: pinned below the scrolling half ---- #
        bottom = _vbox(6, 12)
        bottom.set_margin_top(0)
        outer.pack_start(bottom, False, False, 0)

        self._progress = Gtk.ProgressBar()
        self._progress.set_show_text(True)
        self._progress.set_text("Idle")
        bottom.pack_start(self._progress, False, False, 0)

        self._log = LogView(height=130)
        bottom.pack_start(self._log, False, False, 0)

        buttons = _hbox()
        self._install_button = Gtk.Button(label="Install")
        self._install_button.get_style_context().add_class("suggested-action")
        self._install_button.connect("clicked", self._on_install_clicked)
        buttons.pack_start(self._install_button, False, False, 0)

        self._cancel_button = Gtk.Button(label="Cancel")
        self._cancel_button.set_sensitive(False)
        self._cancel_button.connect("clicked", self._on_cancel_clicked)
        buttons.pack_start(self._cancel_button, False, False, 0)

        # Re-copying the plug-in folder does not update the daemon inside the
        # Python environment; this does, for whichever environment is in use,
        # and stops the running daemon so the next use starts the new code.
        self._update_button = Gtk.Button(label="Update daemon")
        self._update_button.set_sensitive(False)
        self._update_button.connect("clicked", self._on_update_daemon_clicked)
        buttons.pack_start(self._update_button, False, False, 0)

        recheck = Gtk.Button(label="Re-check")
        recheck.connect("clicked", lambda *_a: self.refresh())
        buttons.pack_end(recheck, False, False, 0)
        bottom.pack_start(buttons, False, False, 0)

        return outer

    # ------------------------------------------------------------------ #
    # daemon source repair
    # ------------------------------------------------------------------ #
    def _build_daemon_source_frame(self) -> Gtk.Widget:
        """Repair a plug-in tree that arrived without its bundled daemon.

        The daemon is installed from a path, never by name, so a tree copied by
        hand (rather than by the installer) has nothing to install *from*.  That
        used to be a dead end reachable only from a terminal; this makes it a
        file chooser.
        """
        box = _vbox(6, 6)
        self._src_status = _label("", wrap=True)
        box.pack_start(self._src_status, False, False, 0)

        row = _hbox()
        row.pack_start(_label("Daemon source:", wrap=False), False, False, 0)
        self._src_chooser = PathPicker(
            "The plug-in's _daemon folder", Gtk.FileChooserAction.SELECT_FOLDER,
            placeholder="…\\sam3_gimp\\_daemon")
        row.pack_start(self._src_chooser, True, True, 0)
        self._src_button = Gtk.Button(label="Use this folder")
        self._src_button.connect("clicked", self._on_use_daemon_source)
        row.pack_start(self._src_button, False, False, 0)
        box.pack_start(row, False, False, 0)

        self._src_frame = _frame("Daemon source", box)
        return self._src_frame

    def _refresh_daemon_source(self) -> None:
        status = bootstrap.daemon_source_status()
        if status["ok"]:
            # Nothing to repair: keep the row out of the way entirely.
            self._src_frame.hide()
            return
        self._src_frame.show_all()
        self._src_status.set_markup(
            "<b>The daemon source is missing, so nothing can be installed.</b>\n"
            "This plug-in folder was copied without its bundled <tt>_daemon</tt> "
            "directory. Point at a <tt>_daemon</tt> folder (inside <tt>sam3_gimp</tt> "
            "in the release zip or the repository) and it will be copied in "
            "permanently."
        )

    def _on_use_daemon_source(self, *_args: Any) -> None:
        path = ""
        try:
            path = self._src_chooser.get_filename() or ""
        except Exception:  # pragma: no cover
            pass
        if not path:
            self._src_status.set_text("Choose a '_daemon' folder first.")
            return
        result = bootstrap.install_daemon_source(path)
        if not result.get("ok"):
            self._src_status.set_markup(
                '<span alpha="80%%">%s</span>'
                % GLib.markup_escape_text(result.get("error") or "Could not use that folder.")
            )
            return
        where = "Copied into the plug-in" if result.get("copied") else "Recorded"
        self._src_status.set_markup(
            "<b>%s.</b> The daemon can now be installed."
            % GLib.markup_escape_text(where)
        )
        self.refresh()

    # ------------------------------------------------------------------ #
    # "use the Python I already have"
    # ------------------------------------------------------------------ #
    def _build_existing_env_frame(self) -> Gtk.Widget:
        """Point the plug-in at an interpreter the user already has.

        Someone with a working PyTorch + CUDA install should not be made to
        download another 2.5 GB of it, and should not have to set an environment
        variable to say so.  ``SAM3D_COMMAND`` still exists for developers; this
        is the same capability with a file chooser in front of it.
        """
        box = _vbox(6, 6)
        box.pack_start(
            _dim(
                "Already have PyTorch with CUDA working? Point at its python.exe "
                "and Setup will use it instead of building its own environment."
            ),
            False, False, 0,
        )

        row = _hbox()
        row.pack_start(_label("Interpreter:", wrap=False), False, False, 0)
        self._python_chooser = PathPicker(
            "Python interpreter", Gtk.FileChooserAction.OPEN,
            placeholder="…\\.venv\\Scripts\\python.exe")
        try:
            existing = launcher_mod.configured_python() if launcher_mod else None
            if existing:
                self._python_chooser.set_filename(existing)
        except Exception:  # pragma: no cover -- a chooser is never worth crashing over
            pass
        # A probe describes one interpreter.  Editing the path invalidates it,
        # and choosing a file with Browse checks the new one straight away.
        self._python_chooser.entry.connect("changed", self._on_python_path_edited)
        self._python_chooser.connect("changed-path", lambda *_a: self._on_check_python())
        row.pack_start(self._python_chooser, True, True, 0)

        self._python_check_button = Gtk.Button(label="Check")
        self._python_check_button.connect("clicked", self._on_check_python)
        row.pack_start(self._python_check_button, False, False, 0)
        box.pack_start(row, False, False, 0)

        self._python_status = _label("", wrap=True)
        box.pack_start(self._python_status, False, False, 0)

        buttons = _hbox()
        self._python_install_button = Gtk.Button(label="Install sam3gimpd here")
        self._python_install_button.set_sensitive(False)
        self._python_install_button.set_tooltip_text(
            "Installs only the small pure-Python daemon. Does not touch torch."
        )
        self._python_install_button.connect("clicked", self._on_install_sam3d_here)
        buttons.pack_start(self._python_install_button, False, False, 0)

        self._python_use_button = Gtk.Button(label="Use this environment")
        self._python_use_button.set_sensitive(False)
        self._python_use_button.connect("clicked", self._on_use_python)
        buttons.pack_start(self._python_use_button, False, False, 0)

        self._python_clear_button = Gtk.Button(label="Stop using it")
        self._python_clear_button.connect("clicked", self._on_clear_python)
        buttons.pack_end(self._python_clear_button, False, False, 0)
        box.pack_start(buttons, False, False, 0)

        self._python_probe: Optional[Dict[str, Any]] = None
        frame = _frame("Use an existing Python environment (optional)", box)
        self._refresh_python_row()
        return frame

    def _chosen_python(self) -> str:
        try:
            return self._python_chooser.get_filename() or ""
        except Exception:  # pragma: no cover
            return ""

    def _refresh_python_row(self) -> None:
        """Reflect the persisted choice, whether or not one has been probed."""
        if launcher_mod is None:
            return
        try:
            active = launcher_mod.configured_python()
        except Exception:  # pragma: no cover
            active = None
        self._python_clear_button.set_sensitive(bool(active))
        if active and self._python_probe is None:
            self._python_status.set_markup(
                "In use: <tt>%s</tt>" % GLib.markup_escape_text(active)
            )
            # Probe the chosen interpreter without being asked.  Until a probe
            # has run every action on this row is disabled, so a user who opened
            # Setup to press "Reinstall / update" found it greyed out and, quite
            # reasonably, concluded there was nothing to press.
            worker = getattr(self, "_worker", None)
            if not getattr(self, "_python_autochecked", False) and worker is not None:
                self._python_autochecked = True
                self._idle(self._on_check_python)

    def _on_python_path_edited(self, *_args: Any) -> None:
        probe = self._python_probe
        if probe is None or probe.get("path") == self._chosen_python():
            return
        # The buttons' state describes the interpreter that was checked, not
        # this one: "Use" would otherwise record the old one's verdict against
        # the new path.
        self._python_use_button.set_sensitive(False)
        self._python_install_button.set_sensitive(False)
        self._python_status.set_text("Press Check to inspect this interpreter.")

    def _on_check_python(self, *_args: Any) -> None:
        path = self._chosen_python()
        if not path:
            self._python_status.set_text("Choose a python.exe first.")
            return
        self._python_status.set_markup('<span alpha="70%">Checking…</span>')
        self._python_check_button.set_sensitive(False)
        self._worker.queue(lambda: bootstrap.probe_interpreter(path),
                           self._on_python_checked)

    def _on_python_checked(self, result: Any, error: Optional[BaseException]) -> bool:
        self._python_check_button.set_sensitive(True)
        if error is not None or not isinstance(result, dict):
            self._python_probe = None
            self._python_status.set_text("Could not check it: %s" % error)
            self._python_use_button.set_sensitive(False)
            self._python_install_button.set_sensitive(False)
            return False

        self._python_probe = result
        summary = bootstrap.describe_interpreter(result)
        problems = bootstrap.interpreter_problems(result)
        # torch is the one thing we cannot install for them; everything else is
        # either fixable in place or merely a warning.
        # transformers is as fatal as torch: without it every encode fails with
        # engine_unavailable, which the user only discovers after prompting.
        fatal = (not result.get("ok") or not result.get("torch")
                 or not result.get("transformers"))
        text = "<b>%s</b>" % GLib.markup_escape_text(summary)
        if problems:
            text += "\n" + "\n".join(
                "• " + GLib.markup_escape_text(p) for p in problems)
        self._python_status.set_markup(text)

        # Enabled whenever the environment cannot serve -- not merely when the
        # daemon is absent.  Keying it on "sam3gimpd missing" produced a dead
        # end: an environment with the daemon but no transformers showed the
        # message "press 'Install sam3gimpd here' again" above a button that was
        # greyed out, and disabled "Use this environment" too, so nothing on the
        # row could be pressed at all.
        enabled, label = bootstrap.install_button_state(result)
        self._python_install_button.set_sensitive(enabled)
        self._python_install_button.set_label(label)
        self._python_use_button.set_sensitive(not fatal)
        return False

    def _daemon_source_problem(self) -> Optional[str]:
        """The reason nothing can be installed right now, or ``None``."""
        try:
            bootstrap.daemon_source()
        except bootstrap.DaemonSourceMissing as exc:
            return str(exc)
        return None

    def _on_install_sam3d_here(self, *_args: Any) -> None:
        if self._worker.busy:
            return
        path = self._chosen_python()
        if not path:
            return
        problem = self._daemon_source_problem()
        if problem:
            self._python_status.set_text(problem)
            return
        self._python_status.set_markup('<span alpha="70%">Installing sam3gimpd…</span>')
        self._python_install_button.set_sensitive(False)
        self._cancel_button.set_sensitive(True)
        cancel = threading.Event()

        def _work() -> Any:
            runner = bootstrap.CommandRunner()
            # Resolved here, off the GTK thread: it may probe the interpreter
            # for pip and download uv.
            argv = bootstrap.resolve_daemon_install_command(
                path, runner=runner, on_log=self._log_line, cancel=cancel)
            self._log_line("$ " + " ".join(argv))
            result = runner.run(argv, on_line=self._log_line, cancel=cancel)
            if cancel.is_set():
                raise RuntimeError("cancelled")
            if not result.ok:
                raise RuntimeError("the installer exited %d" % result.returncode)
            # The daemon that is already running still has the old code loaded;
            # stop it so the next use spawns the version just installed.
            _restart_daemon_for_new_code()
            return bootstrap.probe_interpreter(path)

        def _done(result: Any, error: Optional[BaseException]) -> bool:
            self._cancel_button.set_sensitive(False)
            self._on_python_checked(result, error)
            # If this is the interpreter already in use, what was recorded
            # about it when it was chosen ("no daemon yet") is now wrong, and
            # it is the record that decides whether the plug-in is set up.
            if error is None and isinstance(result, dict) and launcher_mod is not None:
                try:
                    configured = launcher_mod.configured_python()
                    if configured and os.path.abspath(configured) == os.path.abspath(path):
                        launcher_mod.set_configured_python(
                            path, has_daemon=bootstrap.interpreter_can_serve(result))
                except Exception as exc:  # noqa: BLE001
                    self._python_status.set_text("Could not save that choice: %s" % exc)
            self.refresh()
            return False

        self._worker.start(_work, _done, cancel=cancel)

    def _on_use_python(self, *_args: Any) -> None:
        path = self._chosen_python()
        if not path or launcher_mod is None:
            return
        probe = self._python_probe or {}
        if probe.get("path") != path:
            # The last probe was of another interpreter; saving its verdict
            # against this path is how an environment without the daemon got
            # recorded as ready.  Check this one, then save.
            self._python_status.set_markup('<span alpha="70%">Checking…</span>')

            def _checked(result: Any, error: Optional[BaseException]) -> bool:
                self._on_python_checked(result, error)
                if (error is None and isinstance(result, dict)
                        and self._python_use_button.get_sensitive()):
                    self._save_python_choice(path, result)
                return False

            self._worker.queue(lambda: bootstrap.probe_interpreter(path), _checked)
            return
        self._save_python_choice(path, probe)

    def _save_python_choice(self, path: str, probe: Dict[str, Any]) -> None:
        # Carry the probe forward: it is what makes the plug-in consider itself
        # set up without re-running the interpreter on every GIMP start.
        try:
            launcher_mod.set_configured_python(
                path, has_daemon=bootstrap.interpreter_can_serve(probe))
        except Exception as exc:  # noqa: BLE001
            self._python_status.set_text("Could not save that choice: %s" % exc)
            return
        self._python_status.set_markup(
            "Saved. The daemon will start from <tt>%s</tt>.\n"
            "Restart GIMP if a daemon is already running."
            % GLib.markup_escape_text(path)
        )
        self._refresh_python_row()
        self.refresh()

    def _on_clear_python(self, *_args: Any) -> None:
        if launcher_mod is None:
            return
        try:
            launcher_mod.set_configured_python(None)
        except Exception as exc:  # noqa: BLE001
            self._python_status.set_text("Could not clear it: %s" % exc)
            return
        self._python_probe = None
        self._python_status.set_text(
            "Cleared. Setup will use its own environment again."
        )
        self._refresh_python_row()
        self.refresh()

    def _render_plan(self) -> None:
        for child in list(self._plan_view.get_children()):
            self._plan_view.remove(child)
        plan = self._plan
        if plan is None:
            return
        pending = {s.key for s in self._pending_steps()}
        for row, step in enumerate(plan.steps):
            done = step.key not in pending
            mark = "✓" if done else "•"
            title = _label("%s  %s" % (mark, step.title), wrap=False)
            if done:
                title.set_markup(
                    '<span alpha="60%%">✓  %s</span>' % GLib.markup_escape_text(step.title)
                )
            self._plan_view.attach(title, 0, row, 1, 1)
            detail = _dim(step.detail)
            detail.set_ellipsize(Pango.EllipsizeMode.MIDDLE)
            detail.set_line_wrap(False)
            detail.set_hexpand(True)
            self._plan_view.attach(detail, 1, row, 1, 1)
        self._plan_view.show_all()

    # ------------------------------------------------------------------ #
    # page 2: weights
    # ------------------------------------------------------------------ #
    def _build_weights_page(self) -> Gtk.Widget:
        page = _vbox()

        page.pack_start(
            _label("The SAM 3 checkpoint is gated", bold=True), False, False, 0
        )
        page.pack_start(
            _dim(
                "Meta distributes the SAM 3 weights through HuggingFace under a licence you "
                "must accept once, with your own account. This plug-in ships no weights and "
                "cannot accept the licence for you.\n\n"
                "Two ways forward: sign in below, or point us at a copy you already have."
            ),
            False, False, 0,
        )

        # -- route A: token --------------------------------------------- #
        token_box = _vbox(6, 0)
        steps = _label(
            "1.  Open the model page and click “Agree and access repository”.\n"
            "2.  Create a read token on your HuggingFace settings page.\n"
            "3.  Paste it here. The token is handed to huggingface_hub, which stores it — "
            "this plug-in never writes it to disk."
        )
        token_box.pack_start(steps, False, False, 0)

        links = _hbox()
        links.pack_start(
            Gtk.LinkButton.new_with_label(bootstrap.SAM3_MODEL_URL, "Open the SAM 3 model page"),
            False, False, 0,
        )
        links.pack_start(
            Gtk.LinkButton.new_with_label(bootstrap.HF_TOKENS_URL, "Create a token"),
            False, False, 0,
        )
        token_box.pack_start(links, False, False, 0)

        entry_row = _hbox()
        self._token_entry = Gtk.Entry()
        self._token_entry.set_placeholder_text("hf_…")
        self._token_entry.set_visibility(False)
        self._token_entry.set_input_purpose(Gtk.InputPurpose.PASSWORD)
        self._token_entry.set_hexpand(True)
        self._token_entry.connect("activate", self._on_check_token)
        entry_row.pack_start(self._token_entry, True, True, 0)
        check = Gtk.Button(label="Check token")
        check.connect("clicked", self._on_check_token)
        entry_row.pack_start(check, False, False, 0)
        token_box.pack_start(entry_row, False, False, 0)

        self._token_status = _dim("")
        token_box.pack_start(self._token_status, False, False, 0)

        self._download_button = Gtk.Button(label="Download weights (~3.6 GB)")
        self._download_button.set_sensitive(False)
        self._download_button.set_halign(Gtk.Align.START)
        self._download_button.connect("clicked", self._on_download_weights)
        token_box.pack_start(self._download_button, False, False, 0)

        page.pack_start(_frame("Sign in to HuggingFace", token_box), False, False, 0)

        # -- route B: local path ---------------------------------------- #
        local_box = _vbox(6, 0)
        local_box.pack_start(
            _dim(
                "If you already downloaded facebook/sam3 (from the website, another machine, "
                "or a colleague), choose the folder that contains config.json and the "
                ".safetensors file. Nothing is copied; the daemon reads it in place."
            ),
            False, False, 0,
        )
        chooser_row = _hbox()
        self._local_chooser = PathPicker(
            "SAM 3 checkpoint folder", Gtk.FileChooserAction.SELECT_FOLDER,
            placeholder="folder containing config.json and model.safetensors")
        chooser_row.pack_start(self._local_chooser, True, True, 0)
        use_local = Gtk.Button(label="Use this folder")
        use_local.connect("clicked", self._on_use_local_weights)
        chooser_row.pack_start(use_local, False, False, 0)
        local_box.pack_start(chooser_row, False, False, 0)

        self._local_status = _dim("")
        local_box.pack_start(self._local_status, False, False, 0)

        # Third exit: an original Meta checkpoint (sam3.pt).  Kept beside the
        # token flow rather than replacing it -- both are useful, and this one
        # needs no HuggingFace account at all, because the converter builds the
        # config itself and takes the tokenizer from a public CLIP repo.
        pt_row = _hbox()
        pt_row.pack_start(_label("Original sam3.pt:", wrap=False), False, False, 0)
        self._pt_chooser = PathPicker(
            "Original SAM 3 checkpoint (.pt)", Gtk.FileChooserAction.OPEN,
            placeholder="…\\sam3.pt")
        pt_row.pack_start(self._pt_chooser, True, True, 0)
        self._pt_button = Gtk.Button(label="Convert and use")
        self._pt_button.connect("clicked", self._on_convert_checkpoint)
        pt_row.pack_start(self._pt_button, False, False, 0)
        local_box.pack_start(pt_row, False, False, 0)

        local_box.pack_start(
            _dim("Have the .pt Meta ships? Convert it here -- no HuggingFace "
                 "account, no terms to accept, no 3.6 GB download. Needs the "
                 "environment installed first, since conversion runs in it. "
                 "The converter is in no released transformers, so it is fetched "
                 "on first use from one fixed transformers commit and run only if it "
                 "matches the checksum this plug-in pins; if it fails, the token "
                 "route above is the dependable one."),
            False, False, 0,
        )
        self._pt_status = _dim("")
        local_box.pack_start(self._pt_status, False, False, 0)

        forget = Gtk.Button(label="Forget the local folder")
        forget.set_halign(Gtk.Align.START)
        forget.connect("clicked", self._on_forget_local_weights)
        local_box.pack_start(forget, False, False, 0)

        page.pack_start(_frame("Or use weights you already have", local_box), False, False, 0)

        self._weights_summary = _label("")
        page.pack_start(self._weights_summary, False, False, 0)
        return page

    # ------------------------------------------------------------------ #
    # page 3: doctor
    # ------------------------------------------------------------------ #
    def _build_doctor_page(self) -> Gtk.Widget:
        page = _vbox()
        page.pack_start(
            _dim(
                "What the daemon reports about this computer. Paste this into a bug report — "
                "it contains no tokens."
            ),
            False, False, 0,
        )

        self._doctor_grid = Gtk.Grid(column_spacing=16, row_spacing=3)
        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroller.set_min_content_height(200)
        scroller.set_shadow_type(Gtk.ShadowType.IN)
        scroller.add(self._doctor_grid)
        page.pack_start(scroller, True, True, 0)

        self._crash_expander = Gtk.Expander(label="Last crash log")
        self._crash_log = LogView(height=140)
        self._crash_expander.add(self._crash_log)
        page.pack_start(self._crash_expander, False, False, 0)

        buttons = _hbox()
        refresh = Gtk.Button(label="Refresh")
        refresh.connect("clicked", lambda *_a: self._run_doctor())
        buttons.pack_start(refresh, False, False, 0)

        repair = Gtk.Button(label="Repair")
        repair.set_tooltip_text(
            "Forget the install journal and re-run every step. Downloads already on "
            "disk are reused; a broken venv is rebuilt."
        )
        repair.connect("clicked", self._on_repair_clicked)
        buttons.pack_start(repair, False, False, 0)

        self._stop_button = Gtk.Button(label="Stop daemon")
        self._stop_button.set_tooltip_text(
            "Ask the running sam3gimpd to exit now. Frees the GPU memory; the "
            "next segmentation starts a fresh daemon."
        )
        self._stop_button.connect("clicked", self._on_stop_daemon_clicked)
        buttons.pack_start(self._stop_button, False, False, 0)

        copy = Gtk.Button(label="Copy report")
        copy.connect("clicked", self._on_copy_report)
        buttons.pack_end(copy, False, False, 0)
        page.pack_start(buttons, False, False, 0)

        self._doctor_payload: Dict[str, Any] = {}
        return page

    def _render_doctor(self, payload: Dict[str, Any]) -> None:
        self._doctor_payload = payload
        for child in list(self._doctor_grid.get_children()):
            self._doctor_grid.remove(child)

        local = payload.get("local") or {}
        daemon = payload.get("daemon") or {}
        memory = daemon.get("memory") or {}

        def _fmt_bytes(value: Any) -> str:
            try:
                return "%.1f GB" % (float(value) / (1024 ** 3))
            except (TypeError, ValueError):
                return "—"

        rows: List[tuple] = [
            ("Summary", local.get("summary", "—")),
            ("Platform", "%s / %s" % (local.get("platform", "?"), local.get("machine", "?"))),
            ("Accelerator (planned)", "%s — %s" % (local.get("accelerator", "?"),
                                                   local.get("accelerator_reason", ""))),
            ("Device (daemon)", daemon.get("device", "— daemon not reachable")),
            ("Dtype", daemon.get("dtype", "—")),
            ("Engine mode", daemon.get("engine_mode", "—")),
            ("VRAM total", _fmt_bytes(memory.get("vram_total")) if memory else "—"),
            ("VRAM free", _fmt_bytes(memory.get("vram_free")) if memory else "—"),
            ("RSS", _fmt_bytes(memory.get("rss_bytes")) if memory else "—"),
            ("sam3gimpd version", daemon.get("sam3d_version", "—")),
            ("API version", daemon.get("api_version", "—")),
            ("torch", str(daemon.get("torch_version") or daemon.get("torch_available", "—"))),
            ("transformers", str(daemon.get("transformers_version") or "—")),
            ("Checked with", " ".join(payload.get("doctor_argv") or []) or "—"),
            ("Weights present", str(local.get("weights_present"))),
            ("Weights source", str(local.get("weights_source") or "—")),
            ("Models loaded", ", ".join(daemon.get("models_loaded") or []) or "none"),
            ("Daemon", ("running, pid %s on port %s" % (local.get("daemon_pid"), local.get("daemon_port")))
                       if local.get("daemon_running") else "not running"),
            ("Last daemon error", _describe_error(daemon.get("last_error"))),
            ("Install steps done", ", ".join(local.get("completed") or []) or "none"),
        ]
        for key, value in sorted((local.get("paths") or {}).items()):
            rows.append(("path: %s" % key, value))

        for row, (key, value) in enumerate(rows):
            self._doctor_grid.attach(_label(key, wrap=False), 0, row, 1, 1)
            val = Gtk.Label(label=str(value))
            val.set_xalign(0.0)
            val.set_selectable(True)
            val.set_ellipsize(Pango.EllipsizeMode.MIDDLE)
            val.set_hexpand(True)
            val.set_tooltip_text(str(value))
            self._doctor_grid.attach(val, 1, row, 1, 1)
        self._doctor_grid.show_all()

        crash = payload.get("crash_log") or []
        self._crash_expander.set_visible(bool(crash))
        self._crash_log.set_text("\n".join(crash) if crash else "")
        if payload.get("message"):
            self._set_status(str(payload["message"]))

    # ------------------------------------------------------------------ #
    # state refresh
    # ------------------------------------------------------------------ #
    def refresh(self) -> None:
        """Re-probe the filesystem and redraw.  Cheap; safe to call often."""
        forced = self._device_combo.get_active_id() if hasattr(self, "_device_combo") else "auto"
        if forced and forced != "auto":
            accel = bootstrap.Accelerator(forced, "chosen in Advanced")
        else:
            if self._detected_accel is None:
                try:
                    self._detected_accel = bootstrap.detect_accelerator()
                except Exception as exc:  # pragma: no cover - defensive
                    self._detected_accel = bootstrap.Accelerator(
                        "cpu", "detection failed: %s" % exc)
            accel = self._detected_accel
        try:
            report = bootstrap.inspect_environment(accelerator=accel)
        except Exception as exc:  # pragma: no cover - defensive
            self._set_status("Could not inspect the environment: %s" % exc)
            return
        self._report = report
        # Shown (and only shown) when there is nothing to install *from*.
        self._refresh_daemon_source()

        try:
            self._plan = bootstrap.build_install_plan(
                accelerator=report.accelerator,
                with_weights=self._weights_check.get_active() if hasattr(self, "_weights_check") else False,
                upgrade=self._force_check.get_active() if hasattr(self, "_force_check") else False,
            )
        except bootstrap.DaemonSourceMissing:
            # A repairable state, not a broken dialog: the row above says what to
            # do, so leave the rest of the panel usable and stop here.
            self._plan = None
            self._env_summary.set_markup(
                "<b>Cannot install yet: the daemon source is missing.</b>"
            )
            self._install_button.set_sensitive(False)
            return
        except bootstrap.UnsupportedPlatform as exc:
            # No torch wheels for this host (an Intel Mac, a 32-bit box).  Say so
            # up front instead of after a long, doomed resolve; the existing-
            # environment row below stays usable for a daemon running elsewhere.
            self._plan = None
            self._env_summary.set_markup(
                "<b>Setup cannot build an environment on this computer.</b>\n"
                "<span alpha='70%%'>%s</span>" % GLib.markup_escape_text(str(exc))
            )
            self._install_button.set_label("Not available here")
            self._install_button.set_sensitive(False)
            return

        self._env_summary.set_markup(
            "<b>%s</b>\n<span alpha='70%%'>%s</span>"
            % (
                GLib.markup_escape_text(report.summary()),
                GLib.markup_escape_text(report.base),
            )
        )
        self._render_plan()

        pending = self._pending_steps()
        if self._force_check.get_active():
            self._install_button.set_label("Reinstall")
        elif not pending:
            self._install_button.set_label("Everything is installed")
        elif report.state.completed:
            self._install_button.set_label("Resume install")
        else:
            self._install_button.set_label("Install")
        self._install_button.set_sensitive(bool(pending) and not self._worker.busy)

        warnings = list(getattr(self._plan, "warnings", []) or [])
        if warnings:
            self._plan_warnings.set_markup(
                "\n".join("\u26a0 " + GLib.markup_escape_text(w) for w in warnings))
            self._plan_warnings.show()
        else:
            self._plan_warnings.hide()

        can_update = report.env_ready and not self._worker.busy
        self._update_button.set_sensitive(can_update)
        self._update_button.set_tooltip_text(
            "Reinstall the daemon from the copy this plug-in ships (version %s, build %s) "
            "into the environment in use, leaving torch alone, then stop the running "
            "daemon so the next use starts the new code."
            % (bootstrap.bundled_daemon_version() or "?", bootstrap.bundled_daemon_build() or "?")
        )

        # weights page
        local = bootstrap.local_weights_path()
        if local:
            self._local_status.set_markup(
                '<span alpha="70%%">Using: %s</span>' % GLib.markup_escape_text(local)
            )
        else:
            self._local_status.set_markup('<span alpha="70%">No local folder configured.</span>')
        if report.weights_present:
            self._weights_summary.set_markup(
                "<b>SAM 3 weights are available (%s).</b>"
                % GLib.markup_escape_text(str(report.weights_source))
            )
        else:
            self._weights_summary.set_markup(
                "<b>SAM 3 weights are not on this computer yet.</b>\n"
                "<span alpha='70%'>The daemon still runs in stub mode without them, "
                "which is useful for testing the UI but returns synthetic masks.</span>"
            )
        self._download_button.set_sensitive(
            self._token_ok and report.env_ready and not self._worker.busy
        )

    def _pending_steps(self) -> List[Any]:
        """What pressing Install would run: exactly what the Installer will
        do.  Reinstall ignores the journal (its plan has no probes on the
        torch and daemon steps); otherwise the journal only matters for
        verify."""
        if self._plan is None:
            return []
        if self._force_check.get_active():
            return self._plan.pending(None)
        return self._plan.pending(bootstrap.load_state(self._plan))

    def _set_status(self, text: str) -> None:
        self._status.set_markup('<span alpha="80%%">%s</span>' % GLib.markup_escape_text(text))

    # ------------------------------------------------------------------ #
    # install
    # ------------------------------------------------------------------ #
    def _on_install_clicked(self, _button: Gtk.Button) -> None:
        if self._worker.busy or self._plan is None:
            return
        if self._force_check.get_active():
            bootstrap.clear_state()
        plan = self._plan
        state = bootstrap.load_state(plan)
        if self._force_check.get_active():
            state = bootstrap.InstallState()

        self._log.clear()
        self._progress.set_fraction(0.0)
        self._progress.set_text("Starting…")
        self._install_button.set_sensitive(False)
        self._cancel_button.set_sensitive(True)
        self._set_status("Installing. You can keep using GIMP; this window stays responsive.")

        # The Installer watches this Event; the worker is handed the same one,
        # so Cancel and Close reach the running step.
        cancel = threading.Event()
        installer = bootstrap.Installer(
            plan,
            on_log=self._log_line,
            on_progress=lambda frac, msg: self._idle(self._set_progress, frac, msg),
            cancel=cancel,
            state=state,
        )
        self._worker.start(installer.run, self._on_install_done, cancel=cancel)

    def _on_cancel_clicked(self, *_args: Any) -> None:
        self._worker.request_cancel()
        self._cancel_button.set_sensitive(False)
        self._set_status("Cancelling…")

    def _set_progress(self, fraction: float, message: str) -> bool:
        self._progress.set_fraction(max(0.0, min(1.0, float(fraction))))
        self._progress.set_text("%d%%  %s" % (round(fraction * 100), message))
        return False

    def _on_install_done(self, outcome: Any, error: Optional[BaseException],
                         *, ready: str = "Environment ready.") -> bool:
        self._cancel_button.set_sensitive(False)
        if error is not None:
            self._log.append("internal error: %r" % error)
            self._set_status("Install failed: %r" % error)
        elif outcome is None:
            self._set_status("Install produced no result.")
        elif outcome.ok:
            self._progress.set_fraction(1.0)
            self._progress.set_text("100%  Done")
            self._set_status(ready)
            if not (self._report and self._report.weights_present):
                self._notebook.set_current_page(1)
        elif outcome.cancelled:
            self._set_status("Cancelled. Press Resume install to carry on where it stopped.")
        else:
            self._set_status("Install failed at “%s”. See the log." % (outcome.failed_step or "?"))
            self._show_error("Install failed", outcome.error or "Unknown failure.")
        self.refresh()
        return False

    def _on_idle_changed(self, spin: Gtk.SpinButton) -> None:
        if launcher_mod is None:
            return
        try:
            launcher_mod.set_configured_idle_ttl(float(spin.get_value()))
        except Exception as exc:  # noqa: BLE001
            self._set_status("Could not save the idle timeout: %s" % exc)
            return
        value = int(round(spin.get_value()))
        self._set_status("Idle timeout saved: %s. Applies when the daemon next starts."
                         % ("never exit" if value == 0 else "%d min" % value))

    def _refresh_remote_row(self) -> None:
        if launcher_mod is None:
            self._remote_status.set_text("")
            return
        try:
            data = launcher_mod.read_settings()
        except Exception:  # pragma: no cover
            data = {}
        url = data.get("remote_url") if isinstance(data, dict) else None
        if url:
            self._remote_url.set_text(str(url))
            self._remote_token.set_text(str(data.get("remote_token") or ""))
            self._remote_status.set_markup(
                "In use: <tt>%s</tt>. No local daemon is started while this is set."
                % GLib.markup_escape_text(str(url)))
        else:
            self._remote_status.set_markup('<span alpha="70%">Not set: a local daemon is used.</span>')
        self._remote_clear.set_sensitive(bool(url))
        self._update_remote_warning()

    def _update_remote_warning(self) -> None:
        """Say plainly, whenever the address leaves this computer, that the
        token and the images would cross the network unencrypted."""
        if not remote_is_off_machine(self._remote_url.get_text()):
            self._remote_warning.hide()
            return
        self._remote_warning.set_markup(
            "\u26a0 <b>Not encrypted.</b> This address is not on this computer, so the "
            "daemon's token and every image you segment would cross the network as "
            "plain HTTP that anyone on the way can read or copy. Use an SSH tunnel "
            "instead: on the GPU machine run <tt>sam3gimpd serve --port 8765</tt>, here "
            "run <tt>ssh -L 8765:127.0.0.1:8765 &lt;gpu-host&gt;</tt>, and enter "
            "<tt>http://127.0.0.1:8765</tt> as the address.")
        self._remote_warning.show()

    def _on_remote_save(self, _button: Gtk.Button) -> None:
        if launcher_mod is None:
            return
        url = self._remote_url.get_text().strip()
        token = self._remote_token.get_text().strip()
        if not url:
            self._remote_status.set_text("Enter the daemon's address first.")
            return
        if not token:
            self._remote_status.set_text("Enter the token from that machine's runtime.json.")
            return
        try:
            launcher_mod.set_configured_remote(url, token)
        except Exception as exc:  # noqa: BLE001
            self._remote_status.set_text(str(exc))
            return
        self._refresh_remote_row()
        _restart_daemon_for_new_code()   # a local daemon would otherwise sit idle
        self._set_status("Remote daemon saved. The next segmentation uses it.")

    def _on_remote_clear(self, _button: Gtk.Button) -> None:
        if launcher_mod is None:
            return
        try:
            launcher_mod.set_configured_remote(None)
        except Exception as exc:  # noqa: BLE001
            self._remote_status.set_text(str(exc))
            return
        self._remote_url.set_text("")
        self._remote_token.set_text("")
        self._refresh_remote_row()
        self._set_status("Back to a local daemon.")

    def _on_update_daemon_clicked(self, _button: Gtk.Button) -> None:
        if self._worker.busy:
            return
        problem = self._daemon_source_problem()
        if problem:
            self._set_status("Cannot update: the daemon source is missing.")
            self._show_error("Cannot update the daemon", problem)
            return
        self._update_button.set_sensitive(False)
        self._install_button.set_sensitive(False)
        self._cancel_button.set_sensitive(True)
        self._set_status("Updating the daemon…")
        cancel = threading.Event()

        def _work() -> Any:
            runner = bootstrap.CommandRunner()
            # Resolved off the GTK thread: for an environment the user chose it
            # may probe for pip and download uv.
            argv = bootstrap.daemon_update_command(
                runner=runner, on_log=self._log_line, cancel=cancel)
            if argv is None:
                return None
            self._log_line("$ " + " ".join(argv))
            result = runner.run(argv, on_line=self._log_line, cancel=cancel)
            if cancel.is_set():
                raise RuntimeError("cancelled")
            if not result.ok:
                raise RuntimeError("the installer exited %d" % result.returncode)
            _restart_daemon_for_new_code()
            return True

        def _done(result: Any, error: Optional[BaseException]) -> bool:
            self._cancel_button.set_sensitive(False)
            if error is not None and cancel.is_set():
                self._set_status("Update cancelled.")
            elif error is not None:
                self._set_status("Update failed: %s. See the log." % error)
                self._show_error("Update failed", str(error))
            elif result is None:
                self._set_status("Nothing to update: no environment is installed yet.")
            else:
                self._set_status(
                    "Daemon updated to build %s. It restarts on the next use."
                    % (bootstrap.bundled_daemon_build() or "?"))
            self.refresh()
            return False

        self._worker.start(_work, _done, cancel=cancel)

    def _on_stop_daemon_clicked(self, _button: Gtk.Button) -> None:
        if self._worker.busy or launcher_mod is None:
            return
        self._set_status("Stopping the daemon…")

        def _work() -> str:
            info = launcher_mod.read_runtime_info()
            if not info:
                return "No daemon is running."
            pid = int(info.get("pid", 0) or 0)
            if not launcher_mod.shutdown_daemon(info, grace_ms=0):
                if pid and launcher_mod.pid_alive(pid):
                    return "The daemon (pid %d) did not answer the shutdown request." % pid
                launcher_mod.delete_runtime_file()
                return "No daemon was running; cleared a stale runtime.json."
            launcher_mod.delete_runtime_file()
            return "Daemon (pid %d) asked to exit." % pid

        def _done(message: Any, error: Optional[BaseException]) -> bool:
            self._set_status(str(error) if error is not None else str(message))
            self._run_doctor()
            return False

        self._worker.start(_work, _done)

    def _on_repair_clicked(self, _button: Gtk.Button) -> None:
        bootstrap.clear_state()
        self._force_check.set_active(True)
        self._notebook.set_current_page(0)
        self.refresh()
        self._set_status("Journal cleared. Press Reinstall to rebuild the environment.")

    # ------------------------------------------------------------------ #
    # weights
    # ------------------------------------------------------------------ #
    def _on_check_token(self, *_args: Any) -> None:
        token = self._token_entry.get_text().strip()
        self._token_status.set_markup('<span alpha="70%">Checking…</span>')

        def _work() -> Dict[str, Any]:
            result = bootstrap.validate_hf_token(token)
            if result.get("ok"):
                result["gate"] = bootstrap.check_gated_access(token)
            return result

        self._worker.queue(_work, self._on_token_checked)

    def _on_token_checked(self, result: Any, error: Optional[BaseException]) -> bool:
        if error is not None or not isinstance(result, dict):
            self._token_ok = False
            self._token_status.set_markup(
                '<span alpha="70%%">Could not check the token: %s</span>'
                % GLib.markup_escape_text(str(error))
            )
            return False
        if not result.get("ok"):
            self._token_ok = False
            self._token_status.set_markup(
                "<span alpha='70%%'>%s</span>" % GLib.markup_escape_text(str(result.get("message")))
            )
            self.refresh()
            return False
        gate = result.get("gate") or {}
        if not gate.get("ok", True):
            self._token_ok = False
            self._token_status.set_markup(
                "<span alpha='70%%'>%s</span>" % GLib.markup_escape_text(str(gate.get("message")))
            )
        else:
            self._token_ok = True
            self._token_status.set_markup(
                "<span alpha='70%%'>%s %s</span>"
                % (
                    GLib.markup_escape_text(str(result.get("message"))),
                    GLib.markup_escape_text(str(gate.get("message", ""))),
                )
            )
        self.refresh()
        return False

    def _on_download_weights(self, _button: Gtk.Button) -> None:
        if self._worker.busy or self._report is None:
            return
        token = self._token_entry.get_text().strip()
        if not token:
            return
        # The environment the daemon runs in -- the user's own, when they
        # chose one -- not the managed venv, which they may not have at all.
        python = bootstrap.daemon_interpreter()
        if python is None:
            self._token_status.set_markup(
                '<span alpha="70%">Install the environment first (or choose an '
                'existing one): the download runs in it.</span>')
            return
        self._notebook.set_current_page(0)
        self._log.clear()
        self._progress.set_fraction(0.0)
        self._cancel_button.set_sensitive(True)
        self._set_status("Downloading the SAM 3 checkpoint (~3.6 GB).")

        plan = bootstrap.build_weights_plan(python, accelerator=self._report.accelerator)
        cancel = threading.Event()
        installer = bootstrap.Installer(
            plan,
            on_log=self._log_line,
            on_progress=lambda frac, msg: self._idle(self._set_progress, frac, msg),
            cancel=cancel,
            journal=False,  # the token flow is per-run; the install's journal stays as it is
            env={"HF_TOKEN": token, "HUGGING_FACE_HUB_TOKEN": token},
        )
        self._worker.start(
            installer.run,
            lambda outcome, error: self._on_install_done(
                outcome, error, ready="SAM 3 weights downloaded."),
            cancel=cancel)

    def _on_use_local_weights(self, _button: Gtk.Button) -> None:
        path = self._local_chooser.get_filename()
        if not path:
            self._local_status.set_markup('<span alpha="70%">Choose a folder first.</span>')
            return
        report = bootstrap.inspect_local_weights(path)
        if not report["ok"]:
            self._local_status.set_markup(
                '<span alpha="70%%">%s</span>' % GLib.markup_escape_text(report["message"])
            )
            return
        try:
            bootstrap.set_local_weights(path)
        except bootstrap.BootstrapError as exc:
            self._local_status.set_markup(
                '<span alpha="70%%">%s</span>' % GLib.markup_escape_text(str(exc))
            )
            return
        self._set_status("Using local weights at %s" % path)
        self.refresh()

    def _on_convert_checkpoint(self, *_args: Any) -> None:
        """Convert an original ``sam3.pt`` into the HuggingFace layout."""
        if self._worker.busy:
            return
        path = ""
        try:
            path = self._pt_chooser.get_filename() or ""
        except Exception:  # pragma: no cover
            pass
        found = bootstrap.find_original_checkpoint(path)
        if not found:
            self._pt_status.set_markup(
                '<span alpha="80%">Choose the original checkpoint file '
                '(usually sam3.pt).</span>'
            )
            return
        if bootstrap.daemon_interpreter() is None:
            self._pt_status.set_markup(
                '<span alpha="80%">Install the environment first (or choose an '
                'existing one) -- conversion needs torch, so it runs there.</span>'
            )
            return

        self._pt_status.set_markup('<span alpha="70%">Converting…</span>')
        self._pt_button.set_sensitive(False)
        self._cancel_button.set_sensitive(True)
        self._log.clear()
        self._log.append("$ converting %s" % found)
        cancel = threading.Event()

        def _work() -> Any:
            return bootstrap.convert_original_checkpoint(
                found, on_line=self._log_line, cancel=cancel)

        self._worker.start(_work, self._on_converted, cancel=cancel)

    def _on_converted(self, result: Any, error: Optional[BaseException]) -> bool:
        self._pt_button.set_sensitive(True)
        self._cancel_button.set_sensitive(False)
        if error is not None or not isinstance(result, dict):
            self._pt_status.set_text("Conversion failed: %s" % error)
            return False
        if not result.get("ok"):
            self._pt_status.set_markup(
                '<span alpha="80%%">%s</span>'
                % GLib.markup_escape_text(result.get("message") or "Conversion failed.")
            )
            return False
        self._pt_status.set_markup(
            "<b>%s</b>" % GLib.markup_escape_text(result["message"])
        )
        self.refresh()
        return False

    def _on_forget_local_weights(self, _button: Gtk.Button) -> None:
        bootstrap.clear_local_weights()
        self._set_status("Local weights folder forgotten.")
        self.refresh()

    # ------------------------------------------------------------------ #
    # doctor
    # ------------------------------------------------------------------ #
    def _run_doctor(self) -> None:
        """Run Doctor now, or right after whatever the worker is doing."""
        if self._worker.queue(bootstrap.run_doctor, self._on_doctor_done):
            self._set_status("Running sam3gimpd doctor…")
        else:
            self._set_status("Doctor will run when the current task finishes…")

    def _on_doctor_done(self, payload: Any, error: Optional[BaseException]) -> bool:
        if error is not None:
            self._set_status("Doctor failed: %r" % error)
            return False
        if isinstance(payload, dict):
            self._render_doctor(payload)
            self._set_status("Doctor finished.")
        return False

    def _on_copy_report(self, _button: Gtk.Button) -> None:
        import json

        try:
            text = json.dumps(self._doctor_payload, indent=2, sort_keys=True, default=str)
        except Exception:  # pragma: no cover - defensive
            text = str(self._doctor_payload)
        clipboard = Gtk.Clipboard.get_default(self.get_display())
        clipboard.set_text(text, -1)
        self._set_status("Report copied to the clipboard.")

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def _on_response(self, _dialog: Gtk.Dialog, response: int) -> None:
        if response in (Gtk.ResponseType.CLOSE, Gtk.ResponseType.DELETE_EVENT):
            self._worker.request_cancel()

    def _on_delete(self, *_args: Any) -> bool:
        self._worker.request_cancel()
        return False

    def _on_destroy(self, *_args: Any) -> None:
        self._closed = True
        self._worker.shutdown()

    def wait_for_worker(self, timeout: float = 15.0) -> None:
        """After Close: wait for a cancelled job to actually stop.

        Cancelling stops the running installer's child processes within about
        a second, but the thread still has to unwind.  Returning before it
        has would let the caller -- the main dialog -- inspect the
        environment and spawn a daemon from a venv an install was still
        writing into.
        """
        self._worker.join(timeout)

    def _show_error(self, title: str, message: str) -> None:
        dlg = Gtk.MessageDialog(
            transient_for=self,
            modal=True,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.CLOSE,
            text=title,
        )
        dlg.format_secondary_text(message[:2000])
        dlg.run()
        dlg.destroy()

    # ------------------------------------------------------------------ #
    def switch_to_doctor(self) -> None:
        self._notebook.set_current_page(2)
        self._run_doctor()


def _describe_error(err: Any) -> str:
    if not isinstance(err, dict):
        return "—"
    return "%s: %s" % (err.get("code", "?"), err.get("message", ""))


# --------------------------------------------------------------------------- #
# entry points used by sam3_gimp.py
# --------------------------------------------------------------------------- #
#: Readiness is a filesystem question; handing ``inspect_environment`` an
#: accelerator stops it from running ``nvidia-smi`` just to answer it.
_UNPROBED = bootstrap.Accelerator("unknown", "not probed for a readiness check")


def needs_setup() -> bool:
    """True when the daemon environment is not usable yet.

    Weights are *not* part of this test: the daemon runs in ``--stub`` mode
    without them (API.md §14), which is a legitimate way to try the plug-in.
    """
    try:
        return not bootstrap.inspect_environment(accelerator=_UNPROBED).env_ready
    except Exception:  # pragma: no cover - defensive
        return True


def run_setup(parent: Optional[Gtk.Window] = None, *, page: str = "install") -> bool:
    """Show the setup dialog modally; return True if the environment is ready."""
    dialog = SetupDialog(parent, page=page)
    try:
        dialog.run()
    finally:
        dialog.destroy()
        dialog.wait_for_worker()
    try:
        return bootstrap.inspect_environment(accelerator=_UNPROBED).env_ready
    except Exception:  # pragma: no cover - defensive
        return False


if __name__ == "__main__":  # pragma: no cover - manual harness
    dlg = SetupDialog(None)
    dlg.connect("destroy", Gtk.main_quit)
    dlg.connect("response", lambda *_a: Gtk.main_quit())
    Gtk.main()
    sys.exit(0)
