"""The main segmentation window.

This is the plug-in's working surface: type a noun phrase, get every matching
instance, tick the ones you want, choose what GIMP should make of them, Apply.

::

    +-------------------------------------------+---------------------------+
    |  prompt entry            [ Segment ]      |  Instances                |
    |  hint: simple noun phrases                |   [x] red car      0.93   |
    |                                           |   [ ] red car      0.41   |
    |  +-------------------------------------+  |  ------------------------ |
    |  |                                     |  |  Score threshold  ----o-- |
    |  |        ui/canvas.py                 |  |  Mask threshold   ---o--- |
    |  |   (image + live mask overlays)      |  |  Overlay opacity  -----o- |
    |  |                                     |  |  ------------------------ |
    |  +-------------------------------------+  |  Output: Selection    v   |
    |  [progress]  cuda · bfloat16 · sam3gimpd 0.1  |  Post-ops, then [ Apply ]  |
    +-------------------------------------------+---------------------------+

Rules this module does not break
--------------------------------
* **The GTK main loop never blocks.**  Every daemon call runs on a worker
  thread and comes back through ``GLib.idle_add`` (DESIGN.md §4).  A frozen
  dialog inside GIMP is indistinguishable from a crashed one.
* **Both sliders filter locally.**  Masks arrive as cropped *soft* uint8
  (API.md §8): the score slider filters the instance list and the mask-threshold
  slider re-binarises bytes already in memory.  Neither ever makes an HTTP
  request.  That is the whole reason the wire format looks the way it does.
* **Stale results are dropped.**  Every prompt carries a monotonic
  ``request_id``; anything that returns for an older one is discarded
  (API.md §10), so typing over a running inference feels instant.
* **Only stdlib and ``gi``.**  No numpy, no requests, no pillow.

Collaborating modules (this file is a client of them):

``ui/canvas.py``   ``Sam3Canvas``: the preview.  It owns per-instance
                   ``visible`` flags, the mask/score thresholds and the click
                   points, and emits ``point-added``, ``instance-toggled``,
                   ``instance-visibility-changed`` and ``threshold-changed``.
                   Where the canvas is present it is the single source of truth
                   for visibility; the list on the right mirrors it.
``client.py``      ``Sam3Client``: typed access to every endpoint, plus
                   ``run_text`` / ``run_points`` which prompt and long-poll in
                   one call and return ``None`` for a superseded job.
``launcher.py``    ``find_or_spawn()``: API.md §3.3, returns a connected client.
``gimpbridge.py``  ``read_upload_pixels()``: the projection as raw ``R'G'B' u8``,
                   already downscaled to ≤1008 px, plus the ``UploadGeometry``
                   that records the downscale factor; ``source_layer()``: the
                   layer a drawable stands for.  Every GIMP call runs on the
                   GTK thread -- libgimp's pipe to GIMP is not thread-safe --
                   and only the HTTP upload goes to the worker.
``outputs.py``     ``apply_result()``: masks -> selection / channels / layer
                   masks / layer group / paths, inside one undo group.

Every one of those is imported defensively: a missing module degrades to a
readable message in the status line instead of taking the dialog down, which is
what makes this file runnable standalone, without GIMP.
"""

from __future__ import annotations

import json
import os
import threading
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, GObject, Gtk, Pango  # noqa: E402


def _optional(name: str) -> Any:
    """Import a sibling plug-in module, or ``None``.

    Inside GIMP the plug-in directory is on ``sys.path`` so the flat name works;
    the dotted fallback covers being imported as ``sam3_gimp.ui.main_dialog``.
    """
    try:
        module = __import__(name)
    except ImportError:
        try:
            module = __import__("sam3_gimp." + name, fromlist=["_"])
        except ImportError:
            return None
    for part in name.split(".")[1:]:
        module = getattr(module, part, None)
        if module is None:
            return None
    return module


bootstrap = _optional("bootstrap")
client_mod = _optional("client")
launcher_mod = _optional("launcher")
gimpbridge = _optional("gimpbridge")
outputs_mod = _optional("outputs")
canvas_mod = _optional("ui.canvas")
contours_mod = _optional("contours")

__all__ = ["MainDialog", "OUTPUT_MODES", "SELECTION_OPS", "run_main_dialog"]


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
#: API.md §8.3: mask byte 128 == logit 0 == the model's own binarisation point.
#: Not a taste decision, which is why it is not simply "half of 255".
DEFAULT_MASK_THRESHOLD = 128

#: PCS returns every candidate scoring >= 0.1 (API.md §6.3) precisely so this
#: slider can filter with no round trip.  0.30 is a sane starting point.
DEFAULT_SCORE_THRESHOLD = 0.30

#: What the daemon is *asked* for.  Deliberately the contract floor rather than
#: the slider value: fetch everything once, filter locally for ever after.
#: The floor *requested from the daemon*, not the user-facing filter.
#: post_process_instance_segmentation drops everything below this before
#: the client ever sees it, so it is a hard ceiling on what the score
#: slider can reveal.  At 0.1 the slider's bottom tenth was dead and weak
#: matches were unreachable: a prompt like "guitar" kept only the
#: headstock, and "guitar strap" -- whose instances all score lower --
#: returned nothing at all and looked like a model failure.
PROMPT_SCORE_FLOOR = 0.02
PROMPT_MAX_INSTANCES = 64
#: The daemon's hard ceiling per prompt (API.md §15 ``max_instances``); the
#: model itself has 200 object queries, so nothing above this can exist.
MAX_INSTANCES_LIMIT = 256
#: Example boxes a text prompt may carry (API.md §15 ``max_boxes``).
MAX_EXEMPLAR_BOXES = 16
#: Recent phrases kept in the prompt drop-down.
PROMPT_HISTORY_MAX = 20

SELECTION_BOX_TIP = (
    "Make a selection in GIMP's own window first (rectangle select, lasso, "
    "anything), then press this. Its bounds become a box: with the click set "
    "to place a point, SAM 3 segments inside it; with a phrase typed and the "
    "click set to pick, it is an example box for that phrase."
)

#: Longest side we upload.  The model resizes to 1008 anyway (DESIGN.md §1) and
#: the daemon rejects more (API.md §15) rather than silently downscaling.
MAX_UPLOAD_SIDE = 1008

#: Mode ids come from ``outputs.OutputMode`` (note the hyphens) so the combo's
#: active id can be handed straight to ``OutputOptions``.
OUTPUT_MODES = (
    ("selection", "Selection"),
    ("channels", "Channels (one per instance)"),
    ("layer-masks", "Layer mask"),
    ("layer-groups", "Layer group (one masked layer per instance)"),
    ("paths", "Paths (vectors)"),
)

SELECTION_OPS = (
    ("replace", "Replace"),
    ("add", "Add"),
    ("subtract", "Subtract"),
    ("intersect", "Intersect"),
)

#: Settings that are also arguments of ``plug-in-sam3-segment`` (see
#: ``sam3_gimp._add_text_argument`` / ``_add_common_arguments``), so GIMP's
#: ``Gimp.ProcedureConfig`` stores them too.  Every other setting lives only
#: in ``bootstrap.ui_settings_file()``, which gets all of them.
PROCEDURE_SETTINGS = (
    "text", "use-projection", "output-mode", "mask-threshold",
    "score-threshold", "max-instances",
)


def mode_nick(mode: str, op: str) -> str:
    """The ``output-mode`` choice nick for a mode and selection operation.

    The procedure's choice folds the selection operation into the mode
    (``selection-add``, ...), because one choice is friendlier to a Script-Fu
    author than two; GIMP rejects any other string, including a bare
    ``"selection"``, and keeps its default instead.
    """
    return "selection-%s" % (op or "replace") if mode == "selection" else mode


def split_mode_nick(nick: str) -> Tuple[str, Optional[str]]:
    """``(mode, op)`` from an ``output-mode`` nick; ``op`` is ``None`` for
    the modes that do not carry one."""
    nick = str(nick or "")
    if nick == "selection" or nick.startswith("selection-"):
        op = nick[len("selection-"):] or None
        return ("selection", op)
    return (nick, None)


PROMPT_HINT = (
    "SAM 3 wants a <b>simple noun phrase</b> — “red car”, “yellow school bus”, “dog”. "
    "It is not a chat box: relational descriptions such as “the car on the left” do not work. "
    "Or segment by clicking: set the canvas click to <b>place a point</b>, left-click an "
    "object to select it, right-click (or Shift-click) a spot to exclude it, Ctrl-drag a box."
)

#: What the status line says when the canvas click mode changes.
MODE_HINTS = {
    "select": ("Canvas click picks objects: left-click an object to tick or untick it. "
               "Switch to “places a point” to segment by clicking."),
    "points": ("Canvas click places a point: left-click an object to segment it, "
               "right-click or Shift-click to exclude a spot, Ctrl-drag a box. Each point "
               "refines the result; Backspace removes the last point, Esc clears them."),
}


# --------------------------------------------------------------------------- #
# worker thread
# --------------------------------------------------------------------------- #
#: While the window is open, GET /hello every this-many seconds so the
#: daemon's idle clock never runs out under a user who is merely thinking.
#: Must stay well under the daemon's idle TTL (1800 s).
KEEPALIVE_SECONDS = 240


class _Worker:
    """Background threads for daemon calls, results delivered on the GTK loop.

    A call may be submitted while another is in flight.  That is what the
    wire protocol is built for: the daemon runs one inference at a time and a
    newer prompt on the same image *supersedes* a queued one (API.md §10),
    and the dialog drops any result whose request id is not the latest.  The
    worker used to refuse a second submission, so typing over a slow prompt
    got "Still working on the previous prompt" instead of winning.
    ``busy`` is still reported, for the callers that must not overlap
    (session start, keepalive, Apply).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._threads: List[threading.Thread] = []

    @property
    def busy(self) -> bool:
        with self._lock:
            self._reap()
            return any(not getattr(t, "_sam3_finished", False) for t in self._threads)

    def _reap(self) -> None:
        self._threads = [t for t in self._threads if t.is_alive()]

    def submit(
        self,
        fn: Callable[[], Any],
        done: Callable[[Any, Optional[BaseException]], None],
    ) -> bool:
        with self._lock:
            self._reap()

            def _run() -> None:
                result: Any = None
                error: Optional[BaseException] = None
                try:
                    result = fn()
                except BaseException as exc:  # noqa: BLE001 - surfaced in the UI
                    error = exc
                    traceback.print_exc()
                threading.current_thread()._sam3_finished = True  # type: ignore[attr-defined]
                GLib.idle_add(done, result, error, priority=GLib.PRIORITY_DEFAULT)

            thread = threading.Thread(target=_run, name="sam3-daemon", daemon=True)
            self._threads.append(thread)
            thread.start()
            return True


# --------------------------------------------------------------------------- #
# the dialog
# --------------------------------------------------------------------------- #
#: Response id of the Setup / Doctor button in the dialog's action area.  Any
#: positive value GTK does not reserve; it only has to be recognisable in
#: ``_on_response``, which swallows it so the window stays open.
RESPONSE_SETUP = 100


class MainDialog(Gtk.Dialog):
    """Segmentation window.  Nothing touches the GIMP image until **Apply**."""

    def __init__(
        self,
        parent: Optional[Gtk.Window] = None,
        *,
        image: Any = None,
        drawable: Any = None,
        procedure: Any = None,
        config: Any = None,
        client: Any = None,
        launch: Any = None,
        stub: bool = False,
    ) -> None:
        super().__init__(title="SAM 3 — Segment", transient_for=parent, modal=parent is not None)
        self.set_default_size(1200, 780)

        self._image = image
        # Resolved once: the drawable GIMP passes can be a layer mask or a
        # channel, and everything here (the layer-only upload, the layer
        # modes of Apply) wants the layer it stands for.
        self._drawable = _source_layer(image, drawable)
        self._procedure = procedure
        self._config = config
        self._stub = bool(stub)

        self._worker = _Worker()
        self._launch = launch
        self._client = client
        self._hello: Any = None
        self._accepted: Any = None          # client.ImageAccepted
        self._geometry: Any = None          # gimpbridge.UploadGeometry
        self._result: Any = None            # client.Result
        self._blob: bytes = b""             # the frame's blob region, for outputs/canvas
        self._instances: List[Any] = []     # canvas.Instance when the canvas is up
        self._visible: Dict[int, bool] = {}  # fallback bookkeeping with no canvas
        self._request_counter = 0
        self._latest_request_id = ""
        self._suppress_toggle = False
        self._last_action: Optional[Callable[[], None]] = None  # replayed after a reconnect
        self._recovering = False     # a session restart or a re-upload for a replay is in flight
        self._retrying = False       # the running job is the replay of _last_action
        self._upload_serial = 0      # bumped per upload; only the newest one is adopted
        self._uploading = False      # an upload is in flight, so there is no image to prompt
        self._upload_projection: Optional[bool] = None  # what the daemon's copy was read from
        self._pvs_box: Optional[Tuple[float, float, float, float]] = None  # box of the point prompt
        self._point_count = 0
        self._closed_normally = False   # Close / X / Escape, settings saved
        self._closed = False
        self._daemon_stale = False      # the running daemon's build != the bundled source
        self._bundled_build: Optional[str] = None
        self._keepalive_source = GLib.timeout_add_seconds(KEEPALIVE_SECONDS, self._keepalive)
        # A dialog destroyed without going through Close (the entry point's
        # ``destroy()`` after ``run()``, a test, a crash elsewhere) still had its
        # idle "start session" and keepalive sources armed, and they fired into
        # a dead window the next time *anything* ran the main loop -- spawning
        # daemons on behalf of a widget that no longer existed.
        self.connect("destroy", self._on_destroy)

        content = self.get_content_area()
        content.set_spacing(0)
        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        paned.pack1(self._build_left(), True, False)
        paned.pack2(self._build_right(), False, False)
        paned.set_position(820)
        content.pack_start(paned, True, True, 0)

        # Setup / Doctor lives in the button row, not at the foot of the
        # scrolling side panel where it used to be: there it sat below the
        # instance list, three slider frames and the output frame, and on a
        # 780 px window it was scrolled clean out of view.  The user who most
        # needs it -- staring at "Not connected" -- could not find it.  A
        # *secondary* action-area child is GTK's convention for exactly this
        # (the Help button): always visible, kept apart from Close / Apply.
        self._setup_button = self.add_button("Setup / _Doctor…", RESPONSE_SETUP)
        self._setup_button.set_tooltip_text(
            "Install or repair the segmentation environment, download the "
            "weights, or run the diagnostics."
        )
        try:
            # The button's parent is the action area's Gtk.ButtonBox; asking it
            # directly avoids the deprecated Gtk.Dialog.get_action_area().
            self._setup_button.get_parent().set_child_secondary(self._setup_button, True)
        except Exception:  # pragma: no cover - a theme without a button box
            pass
        self.add_button("Close", Gtk.ResponseType.CLOSE)
        self._apply_button = self.add_button("Apply", Gtk.ResponseType.APPLY)
        self._apply_button.get_style_context().add_class("suggested-action")
        self._apply_button.set_sensitive(False)
        self.connect("response", self._on_response)

        self._job_kind = ""            # "text" | "points": what the running job is
        self._exemplar_boxes: List[Dict[str, Any]] = []   # PCS example boxes, uploaded space
        self._load_settings()
        self.show_all()
        self._sync_output_sensitivity()
        self._set_canvas_mode("points", announce=False)   # nothing to pick yet
        GLib.idle_add(self._start_session)

    # ------------------------------------------------------------------ #
    # left: prompt, canvas, progress
    # ------------------------------------------------------------------ #
    def _build_left(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.set_margin_start(12)
        box.set_margin_end(6)
        box.set_margin_top(12)
        box.set_margin_bottom(12)

        prompt_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        # An entry with a drop-down of recent phrases: people iterate on the
        # wording constantly, and the config remembered only the last one.
        self._prompt_combo = Gtk.ComboBoxText.new_with_entry()
        self._prompt_entry = self._prompt_combo.get_child()
        self._prompt_entry.set_placeholder_text("red car")
        self._prompt_combo.set_hexpand(True)
        self._prompt_entry.connect("activate", lambda *_a: self._segment_text())
        for phrase in self._load_prompt_history():
            self._prompt_combo.append_text(phrase)
        prompt_row.pack_start(self._prompt_combo, True, True, 0)

        self._segment_button = Gtk.Button(label="Segment")
        self._segment_button.connect("clicked", lambda *_a: self._segment_text())
        prompt_row.pack_start(self._segment_button, False, False, 0)

        self._selection_box_button = Gtk.Button(label="GIMP selection \u2192 box")
        self._selection_box_button.set_tooltip_text(SELECTION_BOX_TIP)
        # GIMP stays usable while this window is open, so the user can draw a
        # selection there and come back; re-check it whenever they do.
        self.connect("focus-in-event", lambda *_a: self._refresh_selection_button() or False)
        self._selection_box_button.connect("clicked", lambda *_a: self._use_selection_as_box())
        prompt_row.pack_start(self._selection_box_button, False, False, 0)

        self._clear_points_button = Gtk.Button(label="Clear points")
        self._clear_points_button.set_tooltip_text(
            "Forget the click points and boxes; the next Segment is a pure text prompt again."
        )
        self._clear_points_button.connect("clicked", lambda *_a: self._clear_points())
        prompt_row.pack_start(self._clear_points_button, False, False, 0)
        box.pack_start(prompt_row, False, False, 0)

        # The canvas has two click modes and the two gestures collide: "tick
        # this object" and "segment the thing under the pointer" are the same
        # click.  The canvas expected the dialog to switch modes and the
        # dialog never did, so it sat in pick mode for ever and clicking to
        # segment (PVS) was unreachable.  A visible switch, not a modifier.
        mode_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        mode_row.pack_start(_hint("Canvas click:"), False, False, 0)
        self._mode_pick = Gtk.RadioButton.new_with_label(None, "picks an object")
        self._mode_pick.set_tooltip_text(
            "Left-click an object in the preview to tick or untick it in the list.")
        self._mode_points = Gtk.RadioButton.new_with_label_from_widget(
            self._mode_pick, "places a point (segment by clicking)")
        self._mode_points.set_tooltip_text(
            "Left-click an object to segment it from that point; right-click or "
            "Shift-click to exclude a spot; Ctrl-drag a box. Each click refines.")
        self._suppress_mode = False
        for button in (self._mode_pick, self._mode_points):
            button.connect("toggled", self._on_mode_toggled)
            mode_row.pack_start(button, False, False, 0)
        box.pack_start(mode_row, False, False, 0)

        hint = Gtk.Label()
        hint.set_markup('<span size="small" alpha="75%%">%s</span>' % PROMPT_HINT)
        hint.set_xalign(0.0)
        hint.set_line_wrap(True)
        hint.set_line_wrap_mode(Pango.WrapMode.WORD_CHAR)
        hint.set_max_width_chars(80)
        hint.set_width_chars(30)
        box.pack_start(hint, False, False, 0)

        box.pack_start(self._build_canvas_area(), True, True, 0)

        self._progress = Gtk.ProgressBar()
        self._progress.set_show_text(True)
        self._progress.set_text("Connecting…")
        box.pack_start(self._progress, False, False, 0)

        self._status = Gtk.Label()
        self._status.set_xalign(0.0)
        self._status.set_ellipsize(Pango.EllipsizeMode.END)
        box.pack_start(self._status, False, False, 0)
        self._set_status("Connecting to the daemon…")
        return box

    def _build_canvas_area(self) -> Gtk.Widget:
        frame = Gtk.Frame()
        frame.set_shadow_type(Gtk.ShadowType.IN)
        self._canvas = self._make_canvas()
        if self._canvas is not None:
            frame.add(self._canvas)
        else:
            placeholder = Gtk.Label()
            placeholder.set_markup(
                '<span alpha="60%">The preview canvas (ui/canvas.py) is unavailable.\n'
                "Text prompts and Apply still work; click-to-refine needs the canvas.</span>"
            )
            placeholder.set_justify(Gtk.Justification.CENTER)
            frame.add(placeholder)
        if self._canvas is None:
            return frame

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        box.pack_start(frame, True, True, 0)
        # Zoom was scroll-wheel and keyboard only, and neither was written
        # down anywhere a user would see it.
        tools = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self._zoom_buttons = {}
        for label, tip, fn in (
            ("Fit", "Fit the image to the window (F or 0)", lambda: self._canvas.zoom_fit()),
            ("1:1", "One image pixel per screen pixel (1)", lambda: self._canvas.zoom_to(1.0)),
            ("+", "Zoom in (+)", lambda: self._canvas.zoom_by(1.25)),
            ("\u2212", "Zoom out (\u2212)", lambda: self._canvas.zoom_by(1.0 / 1.25)),
        ):
            btn = Gtk.Button(label=label)
            btn.set_tooltip_text(tip)
            btn.connect("clicked", lambda _b, f=fn: _quiet(f))
            tools.pack_start(btn, False, False, 0)
            self._zoom_buttons[label] = btn
        keys = Gtk.Label()
        keys.set_markup(
            '<span size="small" alpha="65%">Scroll zooms, middle-drag pans, '
            "[ and ] nudge the mask threshold, A toggles the ants, "
            "Backspace removes the last point, Esc clears them.</span>"
        )
        keys.set_xalign(1.0)
        keys.set_ellipsize(Pango.EllipsizeMode.END)
        tools.pack_end(keys, True, True, 0)
        box.pack_start(tools, False, False, 0)
        return box

    def _make_canvas(self) -> Optional[Gtk.Widget]:
        ctor = getattr(canvas_mod, "Sam3Canvas", None) if canvas_mod is not None else None
        if ctor is None:
            return None
        try:
            widget = ctor(
                mask_threshold=DEFAULT_MASK_THRESHOLD,
                score_threshold=DEFAULT_SCORE_THRESHOLD,
            )
        except Exception:  # pragma: no cover - a canvas that fails to build
            traceback.print_exc()
            return None
        widget.set_hexpand(True)
        widget.set_vexpand(True)
        for name, handler in (
            ("point-added", self._on_canvas_point),
            ("points-changed", self._on_points_changed),
            ("instance-toggled", self._on_canvas_instance_toggled),
            ("instance-visibility-changed", self._on_canvas_visibility),
            ("instance-activated", self._on_canvas_activated),
            ("threshold-changed", self._on_canvas_threshold),
            ("box-drawn", self._on_canvas_box),
        ):
            if _has_signal(widget, name):
                try:
                    widget.connect(name, handler)
                except Exception:  # pragma: no cover
                    traceback.print_exc()
        return widget

    # ------------------------------------------------------------------ #
    # right: instances, filters, output
    # ------------------------------------------------------------------ #
    def _build_right(self) -> Gtk.Widget:
        outer = Gtk.ScrolledWindow()
        outer.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        outer.set_min_content_width(360)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_margin_start(6)
        box.set_margin_end(12)
        box.set_margin_top(12)
        box.set_margin_bottom(12)

        box.pack_start(self._build_instance_list(), True, True, 0)
        box.pack_start(self._build_filters(), False, False, 0)
        box.pack_start(self._build_output(), False, False, 0)

        outer.add(box)
        return outer

    def _build_instance_list(self) -> Gtk.Widget:
        frame = _section("Instances")
        inner = frame.get_child()

        # columns: visible, label, score text, instance_id, score, dimmed
        self._store = Gtk.ListStore(bool, str, str, int, float, bool)
        self._tree = Gtk.TreeView(model=self._store)

        toggle = Gtk.CellRendererToggle()
        toggle.connect("toggled", self._on_row_toggled)
        self._tree.append_column(Gtk.TreeViewColumn("", toggle, active=0))

        text = Gtk.CellRendererText()
        text.set_property("ellipsize", Pango.EllipsizeMode.END)
        col = Gtk.TreeViewColumn("Instance", text, text=1)
        col.add_attribute(text, "strikethrough", 5)
        col.set_expand(True)
        self._tree.append_column(col)

        score_cell = Gtk.CellRendererText()
        score_cell.set_property("xalign", 1.0)
        self._tree.append_column(Gtk.TreeViewColumn("Score", score_cell, text=2))

        selection = self._tree.get_selection()
        selection.set_mode(Gtk.SelectionMode.SINGLE)
        selection.connect("changed", self._on_row_selected)

        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroller.set_min_content_height(170)
        scroller.set_shadow_type(Gtk.ShadowType.IN)
        scroller.add(self._tree)
        inner.pack_start(scroller, True, True, 0)

        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        for label, fn in (
            ("All", lambda: self._set_all_visible(True)),
            ("None", lambda: self._set_all_visible(False)),
            ("Invert", self._invert_visible),
        ):
            btn = Gtk.Button(label=label)
            btn.connect("clicked", lambda _b, f=fn: f())
            row.pack_start(btn, False, False, 0)
        self._count_label = Gtk.Label()
        self._count_label.set_xalign(1.0)
        row.pack_end(self._count_label, False, False, 0)
        inner.pack_start(row, False, False, 0)
        return frame

    def _build_filters(self) -> Gtk.Widget:
        frame = _section("Filters — local, no round trip")
        inner = frame.get_child()

        self._score_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.0, 1.0, 0.01)
        self._score_scale.set_value(DEFAULT_SCORE_THRESHOLD)
        self._score_scale.set_digits(2)
        self._score_scale.set_value_pos(Gtk.PositionType.RIGHT)
        self._score_scale.connect("value-changed", lambda *_a: self._on_score_changed())
        self._score_scale.set_tooltip_text(
            "How sure the model must be to keep a match.\n\n"
            "Lower it (try 0.15) when part of the object is missing, or fewer "
            "objects were found than you expected. SAM 3 often returns one "
            "object as several partial matches, so a high value here is the "
            "usual reason you get only a fragment.\n\n"
            "Raise it when unrelated things get selected."
        )
        inner.pack_start(_labelled("Score threshold", self._score_scale), False, False, 0)
        inner.pack_start(
            _hint("Lower if the object is only partly found; raise if extra "
                  "things are selected."),
            False, False, 0)

        self._max_instances = _spin(PROMPT_MAX_INSTANCES, 1, MAX_INSTANCES_LIMIT, 1)
        self._max_instances.set_tooltip_text(
            "How many matches a text prompt may return, best first. The daemon "
            "keeps the highest-scoring N and says so when it had to drop some. "
            "Applies to the next Segment; the model itself never finds more "
            "than about 200."
        )
        inner.pack_start(_labelled("Max instances", self._max_instances), False, False, 0)
        inner.pack_start(
            _hint("Raise for crowded scenes (\u201cevery window\u201d); "
                  "takes effect on the next Segment."),
            False, False, 0)

        self._mask_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 1.0, 255.0, 1.0)
        self._mask_scale.set_value(DEFAULT_MASK_THRESHOLD)
        self._mask_scale.set_digits(0)
        self._mask_scale.set_value_pos(Gtk.PositionType.RIGHT)
        self._mask_scale.add_mark(DEFAULT_MASK_THRESHOLD, Gtk.PositionType.BOTTOM, None)
        self._mask_scale.connect("value-changed", lambda *_a: self._on_mask_threshold_changed())
        self._mask_scale.set_tooltip_text(
            "How much of each mask to keep.\n\n"
            "Lower it (try 90) when the selection is too tight, ragged, or has "
            "holes in it. Raise it (try 170) when it bleeds into the "
            "background.\n\n"
            "128 is the model's own boundary; the tick marks it."
        )
        inner.pack_start(_labelled("Mask threshold", self._mask_scale), False, False, 0)
        inner.pack_start(
            _hint("Lower for a fatter selection; raise for a tighter one. "
                  "128 = the model's own edge."),
            False, False, 0)

        self._opacity_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.0, 1.0, 0.01)
        self._opacity_scale.set_value(0.55)
        self._opacity_scale.set_digits(2)
        self._opacity_scale.set_value_pos(Gtk.PositionType.RIGHT)
        self._opacity_scale.connect("value-changed", lambda *_a: self._on_opacity_changed())
        self._opacity_scale.set_tooltip_text(
            "How strongly the coloured mask preview is painted over the image. "
            "Affects only this preview, never the result."
        )
        inner.pack_start(_labelled("Overlay opacity", self._opacity_scale), False, False, 0)

        # This used to repeat the mask-threshold explanation under the opacity
        # slider, which made the two read as one control.
        inner.pack_start(
            _hint("Preview only: how strongly masks are painted over the image. "
                  "It never changes the result."),
            False, False, 0)
        return frame

    def _build_output(self) -> Gtk.Widget:
        frame = _section("Output")
        inner = frame.get_child()

        self._mode_combo = Gtk.ComboBoxText()
        for key, text in OUTPUT_MODES:
            self._mode_combo.append(key, text)
        self._mode_combo.set_active_id("selection")
        self._mode_combo.connect("changed", lambda *_a: self._sync_output_sensitivity())
        inner.pack_start(_labelled("Mode", self._mode_combo), False, False, 0)

        self._op_combo = Gtk.ComboBoxText()
        for key, text in SELECTION_OPS:
            self._op_combo.append(key, text)
        self._op_combo.set_active_id("replace")
        self._op_row = _labelled("Selection", self._op_combo)
        inner.pack_start(self._op_row, False, False, 0)

        grid = Gtk.Grid(column_spacing=8, row_spacing=4)
        self._feather = _spin(0.0, 0.0, 250.0, 0.5, digits=1)
        self._grow = _spin(0, 0, 200, 1)
        self._shrink = _spin(0, 0, 200, 1)
        self._smooth = _spin(0.0, 0.0, 100.0, 0.5, digits=1)
        self._min_area = _spin(0, 0, 4000000, 64)
        self._softness = _spin(32, 0, 128, 1)
        rows = (
            ("Feather (px)", self._feather),
            ("Grow (px)", self._grow),
            ("Shrink (px)", self._shrink),
            ("Smooth (px)", self._smooth),
            ("Min area (px²)", self._min_area),
            ("Edge softness", self._softness),
        )
        for row, (label, widget) in enumerate(rows):
            lab = Gtk.Label(label=label)
            lab.set_xalign(0.0)
            grid.attach(lab, 0, row, 1, 1)
            grid.attach(widget, 1, row, 1, 1)
        inner.pack_start(grid, False, False, 0)

        self._fill_holes = Gtk.CheckButton(label="Fill holes")
        inner.pack_start(self._fill_holes, False, False, 0)
        # The scriptable procedures had this switch; the canvas dialog always
        # segmented the composite, so "segment this layer alone" needed a
        # Script-Fu call.
        self._use_projection = Gtk.CheckButton(label="Segment the visible image (all layers)")
        self._use_projection.set_active(True)
        self._use_projection.set_tooltip_text(
            "On: segment what you see, every visible layer composited. "
            "Off: segment the active layer alone, ignoring the others. "
            "Changing this re-uploads the image."
        )
        self._use_projection.connect("toggled", lambda *_a: self._reupload())
        inner.pack_start(self._use_projection, False, False, 0)
        self._duplicate_layer = Gtk.CheckButton(label="Work on a duplicate layer")
        self._duplicate_layer.set_active(True)
        inner.pack_start(self._duplicate_layer, False, False, 0)

        note = Gtk.Label()
        note.set_markup(
            '<span size="small" alpha="65%">Apply is one undo step: '
            "a single Ctrl+Z reverts the whole thing.</span>"
        )
        note.set_xalign(0.0)
        note.set_line_wrap(True)
        note.set_max_width_chars(44)
        note.set_width_chars(24)
        inner.pack_start(note, False, False, 0)
        return frame

    def _sync_output_sensitivity(self) -> None:
        mode = self._mode_combo.get_active_id() or "selection"
        self._op_row.set_sensitive(mode == "selection")
        self._duplicate_layer.set_sensitive(mode in ("layer-masks", "layer-groups"))
        # Paths are traced from the binarised mask, so the pixel-level selection
        # ops have nothing to act on.
        for widget in (self._feather, self._grow, self._shrink, self._smooth):
            widget.set_sensitive(mode != "paths")

    # ------------------------------------------------------------------ #
    # session: connect, hello, upload
    # ------------------------------------------------------------------ #
    def _on_destroy(self, *_args: Any) -> None:
        self._closed = True
        source, self._keepalive_source = self._keepalive_source, None
        if source:
            _quiet(lambda: GLib.source_remove(source))

    def _start_session(self) -> bool:
        """Connect (find or spawn the daemon) and upload the image.

        The pixels are read here, on the GTK thread, before the worker
        starts: libgimp talks to GIMP over one pipe that is not
        thread-safe, and the GTK thread makes GIMP calls of its own (Apply,
        the selection probe on focus-in) at any moment.  Only the network
        half runs on the worker.
        """
        if self._closed:
            return False
        stub = self._stub
        serial = self._begin_upload()
        uploaded, read_error = self._read_for_upload()
        projection = self._projection_wanted()
        existing = self._client

        def _work() -> Dict[str, Any]:
            payload: Dict[str, Any] = {"projection": projection}
            client = existing
            if client is None:
                if launcher_mod is None:
                    raise RuntimeError("launcher.py is unavailable; cannot reach the daemon.")
                launch = launcher_mod.find_or_spawn(
                    parent_pid=_gimp_pid(),
                    stub=stub,
                    on_event=lambda kind, msg: GLib.idle_add(
                        self._set_status, "%s: %s" % (kind, msg)
                    ),
                )
                payload["launch"] = launch
                client = launch.client
                payload["hello"] = launch.hello
            payload["client"] = client
            if "hello" not in payload:
                payload["hello"] = client.hello()
            if read_error is not None:
                payload["read_error"] = read_error
            elif uploaded is not None:
                payload["uploaded"] = uploaded
                payload["accepted"] = self._upload(client, uploaded)
            return payload

        self._worker.submit(
            _work, lambda payload, error, s=serial: self._on_session_ready(payload, error, s))
        return False

    def _projection_wanted(self) -> bool:
        try:
            return bool(self._use_projection.get_active())
        except Exception:  # pragma: no cover - widget not built yet
            return True

    def _read_pixels(self) -> Any:
        """Projection (or the source layer alone) -> raw RGB, on the GTK thread.

        ``gimpbridge`` owns the downscale and the flatten-onto-white step; this
        just asks for the result, which the worker then POSTs to ``/images``.
        """
        if gimpbridge is None or self._image is None:
            return None
        if not getattr(gimpbridge, "gimp_available", lambda: False)():
            return None
        source = (getattr(gimpbridge, "SOURCE_PROJECTION", "projection")
                  if self._projection_wanted() or self._drawable is None
                  else getattr(gimpbridge, "SOURCE_LAYER", "layer"))
        return gimpbridge.read_upload_pixels(
            self._image,
            source=source,
            drawable=self._drawable,
            max_side=MAX_UPLOAD_SIDE,
        )

    def _read_for_upload(self) -> Tuple[Any, Optional[BaseException]]:
        """:meth:`_read_pixels`, with a failure returned rather than raised."""
        try:
            return (self._read_pixels(), None)
        except Exception as exc:  # noqa: BLE001 - shown in the status line
            traceback.print_exc()
            return (None, exc)

    @staticmethod
    def _upload(client: Any, uploaded: Any) -> Any:
        """``POST /images`` -- worker thread only."""
        return client.upload_image(
            uploaded.pixels,
            uploaded.geometry.width,
            uploaded.geometry.height,
            source_width=uploaded.geometry.source_width,
            source_height=uploaded.geometry.source_height,
        )

    def _begin_upload(self) -> int:
        """Start an upload: from here until it lands there is no image to prompt.

        The request id moves on as well, so anything still running against
        the old image comes back stale and is dropped, and the serial makes
        a later upload win over an earlier one still in flight.
        """
        self._upload_serial += 1
        self._uploading = True
        self._accepted = None
        self._next_request_id()
        if self._canvas is not None:
            _quiet(lambda: self._canvas.set_busy(False))
        return self._upload_serial

    def _reupload(self) -> None:
        """Upload the pixels again after the projection switch changed.

        The daemon caches embeddings per upload, so a different source means a
        new image id.  A running prompt is no reason to wait: it is for the old
        pixels and its result is dropped.  With no session yet, the upload
        that is setting one up compares the switch when it lands and comes
        back here if it moved.
        """
        if self._client is None or self._recovering:
            return
        self._last_action = None    # a prompt still running was for the old pixels
        self._upload_again("Re-uploading the %s…" % (
            "visible image" if self._projection_wanted() else "active layer alone"))

    def _upload_again(self, message: str) -> bool:
        """Read the pixels (GTK thread) and upload them with the current client."""
        client, hello = self._client, self._hello
        if client is None:
            return False
        serial = self._begin_upload()
        uploaded, read_error = self._read_for_upload()
        if read_error is not None:
            self._uploading = False
            self._recovering = False
            self._last_action = None
            self._set_status("Could not read the image from GIMP: %s" % _explain(read_error))
            return True
        projection = self._projection_wanted()

        def _work() -> Dict[str, Any]:
            payload: Dict[str, Any] = {"client": client, "hello": hello,
                                       "projection": projection}
            if uploaded is not None:
                payload["uploaded"] = uploaded
                payload["accepted"] = self._upload(client, uploaded)
            return payload

        self._set_status(message)
        self._worker.submit(
            _work,
            lambda payload, error, s=serial: self._on_session_ready(
                payload, error, s, restart_on_error=True))
        return True

    def _on_session_ready(self, payload: Any, error: Optional[BaseException],
                          serial: Optional[int] = None, restart_on_error: bool = False) -> bool:
        if getattr(self, "_closed", False):
            # A late idle callback after the window closed: nothing to update,
            # and touching destroyed widgets is how a dialog takes GIMP down.
            return False
        if serial is not None and serial != self._upload_serial:
            return False  # a newer upload has started since; it decides
        self._uploading = False
        if error is not None:
            if restart_on_error and _is_transport_error(error):
                # The daemon went away under the upload: restart it.  A
                # replay the upload was for stays pending in _last_action.
                self._recovering = False
                if self._restart_daemon(error):
                    return False
            # The button is in the row below the status line now; say so.
            hint = ("  Open Setup / Doctor (bottom left) to install or repair the environment."
                    if not self._recovering else
                    "  Open Setup / Doctor (bottom left) if this keeps happening.")
            self._recovering = False
            self._last_action = None
            self._set_status(_explain(error) + hint)
            self._progress.set_text("Not connected")
            return False
        if not isinstance(payload, dict):
            return False

        self._launch = payload.get("launch") or self._launch
        self._client = payload.get("client")
        self._hello = payload.get("hello")
        self._update_device_line()

        uploaded = payload.get("uploaded")
        accepted = payload.get("accepted")
        read_error = payload.get("read_error")
        if uploaded is not None and accepted is not None:
            self._adopt_upload(uploaded, accepted, payload.get("projection"))
            self._set_status("Ready. Type a noun phrase and press Segment." + self._stale_hint())
            self._refresh_selection_button()
        elif read_error is not None:
            self._recovering = False
            self._last_action = None
            self._set_status("Connected, but the image could not be read from GIMP: %s"
                             % _explain(read_error))
            return False
        else:
            self._set_status(
                "Connected, but no image pixels are available (running outside GIMP)."
                + self._stale_hint()
            )
        if self._recovering:
            self._recovering = False
            action, self._last_action = self._last_action, None
            if action is not None:
                self._set_status("Reconnected; repeating the last prompt…")
                self._retrying = True
                action()
        if accepted is not None and payload.get("projection") != self._projection_wanted():
            # The switch moved while these pixels were on their way.
            self._reupload()
        return False

    def _adopt_upload(self, uploaded: Any, accepted: Any, projection: Optional[bool]) -> None:
        """Make a finished upload the image the dialog works on.

        The canvas starts again from the new pixels, and the dialog's own
        result goes with it -- list, ``_result``, Apply -- unless the daemon
        says these are the very pixels the result was computed on (the same
        image id at the same size: a reconnect, or an evicted image uploaded
        again), in which case it is shown again, ticks and all.
        """
        geometry = uploaded.geometry
        previous = self._geometry
        same_size = (previous is not None
                     and (previous.width, previous.height) == (geometry.width, geometry.height)
                     and (previous.source_width, previous.source_height)
                     == (geometry.source_width, geometry.source_height))
        result = self._result
        keep = (result is not None and same_size
                and getattr(result, "image_id", None) == accepted.image_id)
        self._geometry = geometry
        self._accepted = accepted
        self._upload_projection = projection
        visible = {int(i.instance_id): self._is_visible(i.instance_id)
                   for i in self._instances} if keep else {}
        if self._canvas is not None:
            try:
                self._canvas.set_image(uploaded.pixels, geometry.width, geometry.height)
            except Exception:  # pragma: no cover
                traceback.print_exc()
        if keep:
            self._adopt_result(result)
            for instance_id, shown in visible.items():
                self._set_visible(instance_id, shown)
            self._refresh_instance_list()
        else:
            self._clear_result(forget_boxes=not same_size)

    def _clear_result(self, forget_boxes: bool = False) -> None:
        """Drop the result, and the point prompt that produced it.

        ``forget_boxes`` drops the text prompt's example boxes as well, for
        pixels of a different size, where their coordinates mean nothing.
        """
        self._result = None
        self._blob = b""
        self._instances = []
        self._visible = {}
        self._pvs_box = None
        if forget_boxes:
            self._exemplar_boxes = []
        if self._canvas is not None:
            _quiet(lambda: self._canvas.set_result({}, b"", keep_view=True))
            self._point_count = 0     # so the canvas's points-changed is not a "Backspace"
            _quiet(self._canvas.clear_points)
            self._sync_canvas_boxes()
        self._refresh_instance_list(rebuild=True)

    # ------------------------------------------------------------------ #
    # keeping the daemon: keepalive + reconnect
    # ------------------------------------------------------------------ #
    def _recover_from(self, error: BaseException) -> bool:
        """Put the session back after a prompt failed for want of a daemon.

        Two cases qualify, each retried once by replaying the action that hit
        the gap (kept in ``_last_action``) when the session is back:

        * a transport error: the daemon went away.  :meth:`_restart_daemon`
          finds or spawns one again and uploads the image to it;
        * ``image_not_found``: the daemon is alive but no longer holds this
          image -- its cache keeps a few and evicts the oldest, and closing
          another window on the same pixels deletes them (image ids are
          content hashes, shared by every window).  The image is uploaded
          again (``API.md`` §6.2: re-``POST /images`` and retry once).

        Any other ``ApiError`` means the daemon is alive and disagrees with
        us; neither restarting it nor uploading again would change that.
        """
        if self._recovering:
            return False
        if _is_transport_error(error):
            return self._restart_daemon(error)
        if _is_image_gone(error) and self._client is not None:
            self._recovering = True
            return self._upload_again(
                "The daemon no longer holds this image; uploading it again…")
        return False

    def _restart_daemon(self, error: BaseException) -> bool:
        """Find or spawn the daemon again (the launcher deletes the stale
        ``runtime.json`` and starts a fresh process) and upload the image to
        it; ``_on_session_ready`` then replays ``_last_action`` once."""
        if not _is_transport_error(error) or self._recovering:
            return False
        old_client, old_launch = self._client, self._launch
        self._client = None
        self._launch = None
        self._accepted = None
        self._hello = None
        self._recovering = True

        def _drop() -> None:
            if old_launch is not None:
                _quiet(old_launch.close)
            elif old_client is not None:
                _quiet(old_client.close)

        threading.Thread(target=_drop, name="sam3-drop", daemon=True).start()
        reason = str(error).split(" -- ", 1)[0] or type(error).__name__
        self._set_status("The daemon went away (%s). Restarting it…" % reason)
        self._progress.set_text("Reconnecting")
        self._start_session()
        return True

    def _keepalive(self) -> bool:
        """GET /hello on a timer.  Every request touches the daemon's idle
        clock, so an open window keeps the model warm; and a daemon that has
        died is noticed here, before the user's next click has to fail."""
        if getattr(self, "_closed", False):
            return False
        client = self._client
        if client is None or self._recovering or self._worker.busy:
            return True
        self._worker.submit(lambda: client.hello(check=False), self._on_keepalive)
        return True

    def _on_keepalive(self, hello: Any, error: Optional[BaseException]) -> bool:
        if getattr(self, "_closed", False):
            return False
        if error is None:
            return self._on_hello(hello, None)
        self._last_action = None       # nothing to replay: the user did not ask
        if not self._recover_from(error):
            self._set_status(_explain(error))
        return False

    def _device_pending(self) -> bool:
        """The daemon has not probed torch yet: until the first prompt loads
        the model, ``/hello`` reports its device and dtype as ``"auto"``."""
        return str(getattr(self._hello, "device", "") or "").lower() == "auto"

    def _refresh_hello(self) -> None:
        """Ask ``/hello`` again, for the device the first prompt settled."""
        client = self._client
        if client is None or self._closed:
            return
        self._worker.submit(lambda: client.hello(check=False), self._on_hello)

    def _on_hello(self, hello: Any, error: Optional[BaseException]) -> bool:
        """Keep the newest ``/hello`` and redraw the device line from it --
        unless a job is running, whose progress owns that line; it is
        redrawn from the stored answer when the job ends.  A failure is the
        keepalive's business, not this one's."""
        if getattr(self, "_closed", False) or error is not None or hello is None:
            return False
        self._hello = hello
        if not self._worker.busy:
            self._update_device_line()
        return False

    def _check_daemon_build(self, hello: Any) -> None:
        """Compare the running daemon's source hash with the copy we ship.

        Re-copying the plug-in updates the plug-in; the daemon in the venv
        keeps its old code until it is reinstalled, and the two disagreeing
        is how "I updated but nothing changed" happens.  Both sides hash the
        same files the same way (``sam3gimpd.build_hash`` and
        ``bootstrap.bundled_daemon_build``), so a mismatch is a fact, not a
        guess.  A daemon too old to report a build is simply not checked.
        """
        if self._bundled_build is None:
            build = ""
            if bootstrap is not None:
                try:
                    build = bootstrap.bundled_daemon_build()
                except Exception:  # noqa: BLE001
                    build = ""
            self._bundled_build = build
        running = str(getattr(hello, "build", "") or "")
        info = getattr(self._launch, "info", None) or {}
        remote = bool(isinstance(info, dict) and info.get("remote"))
        # A daemon on another machine is not ours to update from here.
        self._daemon_stale = bool(not remote and self._bundled_build and running
                                  and running != self._bundled_build)

    def _stale_hint(self) -> str:
        return ("  The daemon is running older code than the plug-in ships: "
                "open Setup / Doctor and press Update daemon."
                if self._daemon_stale else "")

    def _update_device_line(self) -> None:
        hello = self._hello
        if hello is None:
            return
        self._check_daemon_build(hello)
        device = str(getattr(hello, "device", "?"))
        dtype = str(getattr(hello, "dtype", "?"))
        if device.lower() == "auto":
            # The daemon picks its device when the first prompt loads the
            # model; until then there is nothing to report but that.
            fields = ["device: detecting on first prompt"]
        else:
            fields = [device] + ([] if dtype.lower() == "auto" else [dtype])
        text = " · ".join(fields + [
            "sam3gimpd %s" % getattr(hello, "sam3d_version", "?"),
            "api %s" % getattr(hello, "api_version", "?"),
        ] + (["plug-in build %s" % _plugin_build()] if _plugin_build() else []))
        if getattr(hello, "engine_mode", "") == "stub":
            text += "   —   STUB ENGINE: synthetic masks, not real segmentation"
        elif str(getattr(hello, "device", "")).lower().startswith("cpu"):
            # A CUDA install that silently failed looks like this: the same
            # window, a hundred times slower, and nothing saying why.
            text += "   —   CPU ONLY: expect minutes per prompt; check Setup > Doctor"
        if self._daemon_stale:
            text += "   —   DAEMON OUT OF DATE (build %s, plug-in ships %s)" % (
                getattr(hello, "build", "?"), self._bundled_build)
        self._progress.set_text(text)
        self._progress.set_fraction(0.0)

    # ------------------------------------------------------------------ #
    # prompting
    # ------------------------------------------------------------------ #
    def _next_request_id(self) -> str:
        self._request_counter += 1
        rid = "r-%06d" % self._request_counter
        self._latest_request_id = rid
        return rid

    def _ready_to_prompt(self) -> bool:
        """Connected with an image uploaded.  A running prompt is no bar: the
        newer request supersedes it on the daemon (API.md §10) and its late
        result is dropped here by request id.  An upload in flight is: until
        it lands there is no image id the prompt could name."""
        if self._recovering:
            self._set_status("Reconnecting to the daemon…")
            return False
        if self._client is not None and self._accepted is None and self._uploading:
            self._set_status("Uploading the image to the daemon; one moment…")
            return False
        if self._client is None or self._accepted is None:
            self._set_status("Not connected to the daemon yet.")
            return False
        return True

    def _submit_prompt(self, rid: str, work: Callable[[], Any]) -> None:
        """Run a prompt on the worker; its answer is handled only while ``rid``
        is still the latest request."""
        self._worker.submit(
            work, lambda result, error, rid=rid: self._on_prompt_done(result, error, rid))

    def _segment_text(self) -> None:
        text = self._prompt_entry.get_text().strip()
        if not text:
            self._set_status("Type a noun phrase first — “red car”, not “the car on the left”.")
            return
        if not self._ready_to_prompt():
            return
        client = self._client
        image_id = self._accepted.image_id
        rid = self._next_request_id()
        progress = self._progress_callback(rid)
        limit = self._max_instances_value()
        boxes = list(self._exemplar_boxes) if self._daemon_takes_boxes() else None
        # The canvas shows exactly the boxes this prompt carries: a point
        # prompt's box is not part of a text prompt.
        self._pvs_box = None
        self._sync_canvas_boxes()

        def _work() -> Any:
            return client.run_text(
                image_id,
                text,
                request_id=rid,
                score_threshold=PROMPT_SCORE_FLOOR,
                max_instances=limit,
                boxes=boxes,
                on_progress=progress,
            )

        self._last_action = self._segment_text
        self._job_kind = "text"
        self._remember_prompt(text)
        self._begin_job("Segmenting “%s”%s…" % (
            text, (" with %d example box(es)" % len(boxes)) if boxes else ""))
        self._submit_prompt(rid, _work)

    def _current_pvs_box(self) -> Optional[List[float]]:
        """The point prompt's box, while the canvas still draws it.

        Kept here because the canvas only draws boxes; it is sent with every
        point prompt until the canvas stops showing it (Clear points, a text
        prompt's example boxes, new pixels).
        """
        box = self._pvs_box
        if box is None:
            return None
        if self._canvas is not None:
            try:
                drawn = [tuple(float(v) for v in b[:4]) for b in self._canvas.boxes]
            except Exception:  # pragma: no cover
                drawn = [box]
            if tuple(box) not in drawn:
                self._pvs_box = None
                return None
        return [float(v) for v in box]

    def _segment_points(self) -> None:
        points = self._current_points()
        box = self._current_pvs_box()
        if not points and box is None:
            return
        if not self._ready_to_prompt():
            return
        client = self._client
        image_id = self._accepted.image_id
        rid = self._next_request_id()
        progress = self._progress_callback(rid)
        # SAM's three alternative candidates exist to resolve the ambiguity
        # of ONE click (whole object, part, sub-part).  Once a second point
        # -- or a box -- says which of those was meant, asking for three
        # again returns guesses the extra prompt was supposed to remove,
        # which is what "add too many points and it gets lost" looked like.
        multimask = len(points) == 1 and box is None

        def _work() -> Any:
            return client.run_points(
                image_id,
                points,
                box=box,
                request_id=rid,
                multimask=multimask,
                max_instances=3 if multimask else 1,
                on_progress=progress,
            )

        self._last_action = self._segment_points
        self._job_kind = "points"
        if box is not None and not points:
            self._begin_job("Refining from a box…")
        else:
            self._begin_job("Refining from %d point(s)%s…" % (
                len(points), " and a box" if box is not None else ""))
        self._submit_prompt(rid, _work)

    # ------------------------------------------------------------------ #
    # canvas click mode
    # ------------------------------------------------------------------ #
    def canvas_mode(self) -> str:
        return "points" if self._mode_points.get_active() else "select"

    def _set_canvas_mode(self, mode: str, announce: bool = True) -> None:
        button = self._mode_points if mode == "points" else self._mode_pick
        if not button.get_active():
            self._suppress_mode = True
            try:
                button.set_active(True)
            finally:
                self._suppress_mode = False
        self._apply_canvas_mode(announce)

    def _on_mode_toggled(self, button: Gtk.RadioButton) -> None:
        if self._suppress_mode or not button.get_active():
            return
        self._apply_canvas_mode(announce=True)

    def _apply_canvas_mode(self, announce: bool) -> None:
        mode = self.canvas_mode()
        if self._canvas is not None:
            try:
                self._canvas.set_interaction_mode(mode)
            except Exception:  # pragma: no cover
                traceback.print_exc()
        if announce:
            self._set_status(MODE_HINTS[mode])

    def _max_instances_value(self) -> int:
        try:
            return max(1, min(MAX_INSTANCES_LIMIT, int(self._max_instances.get_value())))
        except Exception:  # pragma: no cover - a widget that is not there
            return PROMPT_MAX_INSTANCES

    def _current_points(self) -> List[Dict[str, Any]]:
        """Click points in **uploaded-image** pixels (API.md §5).

        The canvas already works in uploaded-image space, so no conversion
        happens here -- which is exactly why the canvas was built that way.
        """
        if self._canvas is None:
            return []
        try:
            raw = self._canvas.points
        except Exception:  # pragma: no cover
            return []
        out = []
        for point in raw or []:
            if isinstance(point, dict):
                out.append({"x": float(point["x"]), "y": float(point["y"]),
                            "label": int(point.get("label", 1))})
            else:
                out.append({"x": float(point[0]), "y": float(point[1]),
                            "label": int(point[2]) if len(point) > 2 else 1})
        return out

    def _clear_points(self) -> None:
        self._exemplar_boxes = []
        self._pvs_box = None
        if self._canvas is not None:
            try:
                self._canvas.clear_points()
                self._canvas.clear_boxes()
            except Exception:  # pragma: no cover
                traceback.print_exc()
        self._set_status("Points and boxes cleared.")

    def _daemon_takes_boxes(self) -> bool:
        hello = self._hello
        fn = getattr(hello, "has_capability", None)
        if callable(fn):
            try:
                return bool(fn("exemplar_boxes"))
            except Exception:  # noqa: BLE001
                return False
        return "exemplar_boxes" in list(getattr(hello, "capabilities", []) or [])

    def _sync_canvas_boxes(self) -> None:
        if self._canvas is None:
            return
        _quiet(lambda: self._canvas.set_boxes(
            [tuple(b["box"]) + (int(b.get("label", 1)),) for b in self._exemplar_boxes]))

    def _refresh_selection_button(self) -> None:
        """Enable the button only when GIMP has a selection to read."""
        button = getattr(self, "_selection_box_button", None)
        if button is None:
            return
        if gimpbridge is None or self._image is None or self._geometry is None:
            button.set_sensitive(False)
            button.set_tooltip_text(SELECTION_BOX_TIP + "\n\n(No GIMP image is open here.)")
            return
        try:
            has = gimpbridge.selection_box(self._image, self._geometry) is not None
        except Exception:  # noqa: BLE001
            has = True   # do not hide the button over a probe failure
        button.set_sensitive(has)
        button.set_tooltip_text(
            SELECTION_BOX_TIP if has else
            SELECTION_BOX_TIP + "\n\n(GIMP has no selection right now: draw one there, "
            "then come back.)")

    def _use_selection_as_box(self) -> None:
        """GIMP's own selection bounds as the box: rectangle-select with the
        tool you know, then segment inside it."""
        if gimpbridge is None or self._image is None or self._geometry is None:
            self._set_status("No GIMP image to read a selection from.")
            return
        try:
            box = gimpbridge.selection_box(self._image, self._geometry)
        except Exception as exc:  # noqa: BLE001
            self._set_status("Could not read the selection: %s" % exc)
            return
        if box is None:
            self._set_status("Nothing is selected in GIMP: draw a selection there with GIMP's "
                             "own tools, then press this again.")
            self._refresh_selection_button()
            return
        _plugin_log("selection -> box %r (%s)" % (box, self.canvas_mode()))
        self._on_canvas_box(None, *box)

    # -- recent prompts --------------------------------------------------- #
    def _prompt_history_path(self) -> Optional[str]:
        if bootstrap is None:
            return None
        try:
            return os.path.join(bootstrap.base_dir(), "prompt-history.json")
        except Exception:  # noqa: BLE001
            return None

    def _load_prompt_history(self) -> List[str]:
        path = self._prompt_history_path()
        data = _read_json_quiet(path) if path else None
        if not isinstance(data, list):
            return []
        return [str(p) for p in data if isinstance(p, str) and p.strip()][:PROMPT_HISTORY_MAX]

    def _remember_prompt(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        history = [p for p in self._load_prompt_history() if p != text]
        history.insert(0, text)
        history = history[:PROMPT_HISTORY_MAX]
        path = self._prompt_history_path()
        if path and bootstrap is not None:
            _quiet(lambda: bootstrap._atomic_write_json(path, history))
        combo = getattr(self, "_prompt_combo", None)
        if combo is not None:
            _quiet(combo.remove_all)
            for phrase in history:
                combo.append_text(phrase)

    def _progress_callback(self, request_id: str) -> Callable[[Any], None]:
        """``client.run_*`` calls this from the worker thread on every long-poll
        step; hop back onto the GTK loop before touching a widget."""

        def _on_progress(status: Any) -> None:
            GLib.idle_add(
                self._on_progress,
                float(getattr(status, "progress", 0.0) or 0.0),
                str(getattr(status, "stage", "") or getattr(status, "state", "")),
                request_id,
            )

        return _on_progress

    def _begin_job(self, message: str) -> None:
        self._progress.set_fraction(0.0)
        self._set_status(message)
        # Segment stays enabled: pressing it again supersedes the running job.
        if self._canvas is not None:
            try:
                self._canvas.set_busy(True, "queued", 0.0)
            except Exception:  # pragma: no cover
                pass

    def _on_progress(self, fraction: float, stage: str, request_id: str) -> bool:
        if getattr(self, "_closed", False):
            # A late idle callback after the window closed: nothing to update,
            # and touching destroyed widgets is how a dialog takes GIMP down.
            return False
        if request_id != self._latest_request_id:
            return False  # stale (API.md §10)
        self._progress.set_fraction(max(0.0, min(1.0, fraction)))
        if str(stage).startswith("loading"):
            # A cold start: the daemon is reading a 3.6 GB checkpoint and
            # initialising CUDA.  Named, so it does not read as a freeze.
            self._progress.set_text("Loading model (first use after start; can take a minute)…")
        else:
            self._progress.set_text("%d%%  %s" % (round(fraction * 100), stage))
        if self._canvas is not None:
            try:
                self._canvas.set_progress(fraction, stage)
            except Exception:  # pragma: no cover
                pass
        return False

    def _on_prompt_done(self, result: Any, error: Optional[BaseException],
                        request_id: Optional[str] = None) -> bool:
        if getattr(self, "_closed", False):
            # A late idle callback after the window closed: nothing to update,
            # and touching destroyed widgets is how a dialog takes GIMP down.
            return False
        # Anything for a request that is no longer the latest -- its result,
        # its "superseded" None, and its *error* alike -- must not touch the
        # busy state, _last_action or the retry flag: the newer job owns them,
        # and a transport error on the newer job must still be replayable.
        if request_id is not None:
            if request_id != self._latest_request_id:
                return False
        else:
            stale = (result is not None
                     and getattr(result, "request_id", "") != self._latest_request_id)
            if stale or (result is None and error is None and self._worker.busy):
                return False
        self._segment_button.set_sensitive(True)
        if self._canvas is not None:
            try:
                self._canvas.set_busy(False)
            except Exception:  # pragma: no cover
                pass
        self._update_device_line()

        retried, self._retrying = self._retrying, False
        if error is not None:
            if not retried and self._recover_from(error):
                return False
            self._set_status(_explain(error))
            self._last_action = None
            return False
        self._last_action = None
        if result is None:
            # ``run_text``/``run_points`` return None for a superseded job -- a
            # newer prompt won, which is the design, not a failure.
            self._set_status("Superseded by a newer prompt.")
            return False
        if getattr(result, "request_id", "") != self._latest_request_id:
            # API.md §10, the one-line rule: drop anything that is not the
            # latest request.  This is what keeps stale masks off the canvas.
            return False
        accepted = self._accepted
        if self._uploading or (accepted is not None and getattr(
                result, "image_id", accepted.image_id) != accepted.image_id):
            return False  # computed on pixels the dialog has since replaced

        self._result = result
        self._blob = b"".join(inst.mask for inst in result.instances)
        self._adopt_result(result)
        if self._device_pending():
            self._refresh_hello()       # the prompt has made the daemon choose
        # A text prompt returns a crowd to browse, so clicks should pick; a
        # point prompt is a conversation with the canvas, so clicks keep
        # placing points.
        if self._job_kind == "text" and result.instances:
            self._set_canvas_mode("select", announce=False)
        elif self._job_kind == "points":
            self._set_canvas_mode("points", announce=False)
            # The candidates are alternatives for the same object, best
            # first.  Ticking all of them applied their union -- the whole
            # object plus its parts, so the parts were invisible -- and the
            # list read as three hits.  Tick the best; the rest are there to
            # switch to.
            for index, inst in enumerate(self._instances):
                if index > 0:
                    self._set_visible(inst.instance_id, False)
            self._refresh_instance_list()

        elapsed = getattr(result, "elapsed_ms", 0.0)
        truncated = (" (more matched than the limit of %d; raise Max instances)"
                     % self._max_instances_value()
                     if getattr(result, "truncated", False) else "")
        self._set_status(
            "%d instance(s)%s%s"
            % (len(result.instances), truncated,
               (" in %d ms" % round(float(elapsed))) if elapsed else "")
        )
        return False

    def _adopt_result(self, result: Any) -> None:
        """Hand the frame to the canvas and rebuild the instance list.

        The canvas is fed the *header + blob region*, which is exactly what a
        frame carries; reconstructing the blob from the instances is lossless
        because §8.1 packs them tightly, in order, with no padding.
        """
        if self._canvas is not None:
            try:
                self._canvas.set_result(result.header, self._blob, keep_view=True)
                self._canvas.set_score_threshold(self._score_scale.get_value())
                self._canvas.set_mask_threshold(int(round(self._mask_scale.get_value())))
                self._instances = list(self._canvas.instances)
            except Exception:  # pragma: no cover
                traceback.print_exc()
                self._instances = list(result.instances)
        else:
            self._instances = list(result.instances)
        self._visible = {int(i.instance_id): True for i in self._instances}
        self._refresh_instance_list(rebuild=True)

    # ------------------------------------------------------------------ #
    # local filtering (the point of the soft-mask wire format)
    # ------------------------------------------------------------------ #
    def _is_visible(self, instance_id: int) -> bool:
        if self._canvas is not None:
            inst = self._canvas.get_instance(int(instance_id))
            if inst is not None:
                return bool(inst.visible)
        return bool(self._visible.get(int(instance_id), True))

    def _set_visible(self, instance_id: int, visible: bool) -> None:
        if self._canvas is not None:
            try:
                self._canvas.set_instance_visible(int(instance_id), bool(visible), notify=False)
            except Exception:  # pragma: no cover
                traceback.print_exc()
        self._visible[int(instance_id)] = bool(visible)

    def _on_score_changed(self) -> None:
        value = self._score_scale.get_value()
        if self._canvas is not None:
            try:
                self._canvas.set_score_threshold(value)
            except Exception:  # pragma: no cover
                traceback.print_exc()
        self._refresh_instance_list()

    def _on_mask_threshold_changed(self) -> None:
        if self._canvas is not None:
            try:
                self._canvas.set_mask_threshold(int(round(self._mask_scale.get_value())))
            except Exception:  # pragma: no cover
                traceback.print_exc()

    def _on_opacity_changed(self) -> None:
        if self._canvas is not None:
            try:
                self._canvas.set_overlay_opacity(self._opacity_scale.get_value())
            except Exception:  # pragma: no cover
                traceback.print_exc()

    def selected_instances(self) -> List[Any]:
        """Instances the user actually wants: visible *and* above the score slider."""
        threshold = self._score_scale.get_value()
        return [
            inst for inst in self._instances
            if self._is_visible(inst.instance_id) and float(inst.score) >= threshold
        ]

    def _refresh_instance_list(self, rebuild: bool = False) -> None:
        """Bring the list in step with the instances' tick and score state.

        Rows are updated **in place** whenever the set of instances is the
        one already shown.  The list used to be cleared and re-appended on
        every tick, which threw the scroll position back to the top and
        dropped the row selection each time: the user ticked one row, the
        list jumped, and the next click landed on whichever row had scrolled
        under the pointer -- "randomly selected or not".  Only a new result
        (``rebuild=True``) replaces the rows.
        """
        threshold = self._score_scale.get_value()
        ids_shown = [int(row[3]) for row in self._store]
        ids_now = [int(inst.instance_id) for inst in self._instances]
        self._suppress_toggle = True
        try:
            if rebuild or ids_shown != ids_now:
                self._store.clear()
                for inst in self._instances:
                    label = inst.label or "instance %d" % inst.instance_id
                    below = float(inst.score) < threshold
                    self._store.append([
                        self._is_visible(inst.instance_id) and not below,
                        label,
                        "%.3f" % float(inst.score),
                        int(inst.instance_id),
                        float(inst.score),
                        below,
                    ])
            else:
                for row, inst in zip(self._store, self._instances):
                    below = float(inst.score) < threshold
                    ticked = self._is_visible(inst.instance_id) and not below
                    if bool(row[0]) != ticked:
                        row[0] = ticked
                    if bool(row[5]) != below:
                        row[5] = below
        finally:
            self._suppress_toggle = False
        self._refresh_count()

    def _refresh_count(self) -> None:
        count = len(self.selected_instances())
        self._count_label.set_markup(
            '<span alpha="70%%">%d of %d selected</span>' % (count, len(self._instances))
        )
        self._apply_button.set_sensitive(count > 0)

    def _on_row_toggled(self, _cell: Gtk.CellRendererToggle, path: str) -> None:
        if self._suppress_toggle:
            return
        row = self._store[path]
        instance_id = int(row[3])
        visible = not self._is_visible(instance_id)
        self._set_visible(instance_id, visible)
        # Just this row: no rebuild, no scroll jump, the selection stays put.
        row[0] = visible and not bool(row[5])
        self._refresh_count()

    def _on_row_selected(self, selection: Gtk.TreeSelection) -> None:
        model, tree_iter = selection.get_selected()
        if tree_iter is None or self._canvas is None:
            return
        if getattr(self, "_suppress_activate", False):
            return
        self._suppress_activate = True
        try:
            self._canvas.set_active_instance(int(model[tree_iter][3]))
        except Exception:  # pragma: no cover
            traceback.print_exc()
        finally:
            self._suppress_activate = False

    def _set_all_visible(self, visible: bool) -> None:
        if self._canvas is not None:
            try:
                self._canvas.set_all_visible(bool(visible))
            except Exception:  # pragma: no cover
                traceback.print_exc()
        for inst in self._instances:
            self._visible[int(inst.instance_id)] = bool(visible)
        self._refresh_instance_list()

    def _invert_visible(self) -> None:
        for inst in self._instances:
            self._set_visible(inst.instance_id, not self._is_visible(inst.instance_id))
        self._refresh_instance_list()

    # ------------------------------------------------------------------ #
    # canvas signals
    # ------------------------------------------------------------------ #
    def _on_canvas_point(self, _widget: Gtk.Widget, x: float, y: float, label: int) -> None:
        """A click on the canvas: refine with PVS, reusing the cached embedding."""
        self._point_count = len(self._current_points())
        self._segment_points()

    def _on_points_changed(self, _widget: Gtk.Widget) -> None:
        """Backspace or Esc took points away: the result must follow, or the
        canvas shows a mask that the remaining points no longer describe.
        A point being *added* is handled by ``point-added`` just before this
        fires, so only a shrinking list acts here.  A box still drawn is
        still part of the prompt, so it alone is enough to run again."""
        count = len(self._current_points())
        previous = self._point_count
        self._point_count = count
        if count >= previous:
            return
        if count > 0 or self._current_pvs_box() is not None:
            self._segment_points()
        else:
            self._set_status("Points cleared. The last result is kept; click to start again.")

    def _on_canvas_box(self, _widget: Gtk.Widget, x0: float, y0: float,
                       x1: float, y1: float) -> None:
        if not self._ready_to_prompt():
            return
        text = self._prompt_entry.get_text().strip()
        if self.canvas_mode() == "select" and text:
            # "Like this one": SAM 3's concept prompt takes example boxes
            # alongside the phrase (API.md §6.3).  The daemon advertised it
            # and the dialog never sent one.
            if not self._daemon_takes_boxes():
                self._set_status("This daemon does not take example boxes for text prompts.")
                return
            if len(self._exemplar_boxes) >= MAX_EXEMPLAR_BOXES:
                self._set_status("At most %d example boxes." % MAX_EXEMPLAR_BOXES)
                return
            self._exemplar_boxes.append(
                {"box": [float(x0), float(y0), float(x1), float(y1)], "label": 1})
            self._sync_canvas_boxes()
            self._segment_text()
            return
        # A box for the point prompt: it stays part of the prompt -- every
        # later click is refined inside it -- for as long as it is drawn.
        box = (min(float(x0), float(x1)), min(float(y0), float(y1)),
               max(float(x0), float(x1)), max(float(y0), float(y1)))
        self._pvs_box = box
        if self._canvas is not None:
            _quiet(lambda: self._canvas.set_boxes([box + (1,)]))
        self._segment_points()

    def _on_canvas_instance_toggled(self, _widget: Gtk.Widget, instance_id: int,
                                    selected: bool) -> None:
        self._refresh_instance_list()

    def _on_canvas_activated(self, _widget: Gtk.Widget, instance_id: int) -> None:
        """A click on the preview made an instance active: show its row.

        The list used to be one-way -- selecting a row activated the canvas
        instance, but clicking an object on the canvas left the list where it
        was, so relating a mask to its score meant scanning sixty rows."""
        if getattr(self, "_suppress_activate", False):
            return
        selection = self._tree.get_selection()
        if int(instance_id) < 0:
            selection.unselect_all()
            return
        for row in self._store:
            if int(row[3]) == int(instance_id):
                self._suppress_activate = True
                try:
                    selection.select_path(row.path)
                    self._tree.scroll_to_cell(row.path, None, False, 0.0, 0.0)
                finally:
                    self._suppress_activate = False
                return

    def _on_canvas_visibility(self, _widget: Gtk.Widget, instance_id: int,
                              visible: bool) -> None:
        self._visible[int(instance_id)] = bool(visible)
        self._refresh_instance_list()

    def _on_canvas_threshold(self, _widget: Gtk.Widget, mask_threshold: int,
                             score_threshold: float) -> None:
        # Keep the sliders in step with keyboard/scroll changes made on the
        # canvas, without echoing the change straight back at it.
        if abs(self._mask_scale.get_value() - float(mask_threshold)) > 0.5:
            self._mask_scale.set_value(float(mask_threshold))
        if abs(self._score_scale.get_value() - float(score_threshold)) > 0.005:
            self._score_scale.set_value(float(score_threshold))

    # ------------------------------------------------------------------ #
    # apply
    # ------------------------------------------------------------------ #
    def post_ops(self) -> Any:
        """The mask post-processing block, as ``outputs.PostOps``."""
        return outputs_mod.PostOps(
            threshold=int(round(self._mask_scale.get_value())),
            edge_softness=int(self._softness.get_value()),
            hole_fill=bool(self._fill_holes.get_active()),
            min_area=float(self._min_area.get_value()),
            grow=int(self._grow.get_value()),
            shrink=int(self._shrink.get_value()),
            smooth=float(self._smooth.get_value()),
            feather=float(self._feather.get_value()),
        )

    def output_options(self) -> Any:
        """Everything ``outputs.apply_result`` needs, as ``outputs.OutputOptions``."""
        return outputs_mod.OutputOptions(
            mode=self._mode_combo.get_active_id() or "selection",
            selection_op=self._op_combo.get_active_id() or "replace",
            post=self.post_ops(),
            duplicate_layer=bool(self._duplicate_layer.get_active()),
            group_name=self._prompt_entry.get_text().strip(),
            path_space=outputs_mod.Space.CANVAS,
        )

    def _mask_result(self) -> Any:
        """Bridge the decoded frame into ``outputs.MaskResult``.

        ``source`` is the user's original image size -- the daemon never sees it
        (API.md §5), so the client supplies it here and ``MaskResult`` does the
        canvas -> uploaded -> original mapping of §9.
        """
        geometry = self._geometry
        source = (
            (int(geometry.source_width), int(geometry.source_height))
            if geometry is not None
            else (int(self._result.image.width), int(self._result.image.height))
        )
        return outputs_mod.MaskResult.from_frame(
            self._result.header,
            self._blob,
            source=source,
            prompt_text=self._prompt_entry.get_text().strip() or None,
        )

    def _contours(self, instances: Sequence[Any]) -> Optional[Dict[int, Any]]:
        """Trace beziers for the Paths mode, in **model-canvas** coordinates.

        ``contours.instance_strokes`` works in mask-local pixels and applies the
        instance's bbox origin itself -- which is what makes
        ``path_space=Space.CANVAS`` correct for ``build_path``.  It returns the
        explicit ``{"control_points": ..., "closed": ...}`` form, so
        ``outputs.normalise_stroke`` has nothing to guess at.
        """
        if contours_mod is None:
            return None
        threshold = int(round(self._mask_scale.get_value()))
        out: Dict[int, Any] = {}
        for inst in instances:
            try:
                strokes = contours_mod.instance_strokes(
                    inst.mask, inst.mask_width, inst.mask_height,
                    # Passed whole, not subscripted: ``self._instances`` holds
                    # canvas instances (tuple bbox) normally but client ones
                    # (``BBox`` dataclass) when the canvas is unavailable, and
                    # ``instance_strokes`` normalises both.
                    inst.bbox,
                    threshold=threshold,
                )
            except Exception:  # pragma: no cover
                traceback.print_exc()
                continue
            if strokes:
                out[int(inst.instance_id)] = strokes
        return out

    def _on_apply(self) -> None:
        instances = self.selected_instances()
        if not instances:
            self._set_status("Nothing selected.")
            return
        if outputs_mod is None or self._image is None or self._result is None:
            self._set_status("Nothing to apply (outputs.py or the GIMP image is unavailable).")
            return
        try:
            options = self.output_options().validated()
            mask_result = self._mask_result()
            selected = [int(i.instance_id) for i in instances]
            contours = self._contours(instances) if options.mode == "paths" else None
            _plugin_log("apply: mode=%s op=%s instances=%s threshold=%d score>=%.2f"
                        % (options.mode, options.selection_op, selected,
                           options.post.threshold, self._score_scale.get_value()))
            applied = outputs_mod.apply_result(
                self._image,
                mask_result,
                options,
                layer=self._drawable,
                selected=selected,
                contours=contours,
                on_log=_plugin_log,
            )
        except Exception as exc:
            traceback.print_exc()
            _plugin_log("apply failed: %s" % traceback.format_exc())
            self._set_status("Apply failed: %s" % exc)
            return
        self._save_settings()
        bounds = ""
        try:
            bounds = outputs_mod._selection_bounds_text(self._image)
            _plugin_log("apply: returned; selection %s" % bounds)
        except Exception:  # noqa: BLE001
            pass
        dropped = getattr(applied, "dropped", 0)
        note = getattr(applied, "note", "")
        self._set_status(_describe_applied(applied, options, len(selected), bounds, note))

    # ------------------------------------------------------------------ #
    # settings persistence
    # ------------------------------------------------------------------ #
    def _settings_map(self) -> List[Tuple[str, Callable[[], Any], Callable[[Any], None]]]:
        """``(setting, getter, setter)`` for everything the dialog remembers.

        One table, so the names in :data:`PROCEDURE_SETTINGS` can be checked
        against the arguments ``plug-in-sam3-segment`` registers, and so a
        setting that cannot be restored is skipped on its own rather than
        losing the whole set.  ``output-mode`` is the procedure's choice
        nick (``selection-add``, ``channels``...); ``selection-op`` is kept
        separately as well, for the modes whose nick does not carry it.
        """
        return [
            ("text",
             lambda: self._prompt_entry.get_text(),
             lambda v: self._prompt_entry.set_text(str(v or ""))),
            ("score-threshold",
             lambda: float(self._score_scale.get_value()),
             lambda v: self._score_scale.set_value(float(v))),
            ("max-instances",
             lambda: self._max_instances_value(),
             lambda v: self._max_instances.set_value(float(v))),
            ("use-projection",
             lambda: bool(self._use_projection.get_active()),
             lambda v: self._use_projection.set_active(bool(v))),
            ("mask-threshold",
             lambda: int(round(self._mask_scale.get_value())),
             lambda v: self._mask_scale.set_value(float(v))),
            ("overlay-opacity",
             lambda: float(self._opacity_scale.get_value()),
             lambda v: self._opacity_scale.set_value(float(v))),
            ("selection-op",
             lambda: self._op_combo.get_active_id() or "replace",
             lambda v: self._op_combo.set_active_id(str(v))),
            ("output-mode",
             lambda: mode_nick(self._mode_combo.get_active_id() or "selection",
                               self._op_combo.get_active_id() or "replace"),
             self._set_output_mode),
            ("feather", lambda: float(self._feather.get_value()),
             lambda v: self._feather.set_value(float(v))),
            ("grow", lambda: int(self._grow.get_value()),
             lambda v: self._grow.set_value(float(v))),
            ("shrink", lambda: int(self._shrink.get_value()),
             lambda v: self._shrink.set_value(float(v))),
            ("smooth", lambda: float(self._smooth.get_value()),
             lambda v: self._smooth.set_value(float(v))),
            ("min-area", lambda: int(self._min_area.get_value()),
             lambda v: self._min_area.set_value(float(v))),
            ("edge-softness", lambda: int(self._softness.get_value()),
             lambda v: self._softness.set_value(float(v))),
            ("fill-holes", lambda: bool(self._fill_holes.get_active()),
             lambda v: self._fill_holes.set_active(bool(v))),
            ("duplicate-layer", lambda: bool(self._duplicate_layer.get_active()),
             lambda v: self._duplicate_layer.set_active(bool(v))),
        ]

    def _set_output_mode(self, nick: Any) -> None:
        mode, op = split_mode_nick(str(nick))
        if mode and self._mode_combo.set_active_id(mode) is False:
            return
        if op:
            self._op_combo.set_active_id(op)

    def _load_settings(self) -> None:
        """Restore the last session's choices.

        ``bootstrap.ui_settings_file()`` holds every setting (it is written
        whatever else happens); for the ones that are also procedure
        arguments, GIMP's ``Gimp.ProcedureConfig`` -- the canonical store
        (DESIGN.md §6), which also remembers them per image -- wins when it
        has a value.  ``output-mode`` is applied last, so the selection
        operation its nick carries wins over the stored ``selection-op``.
        """
        stored: Dict[str, Any] = {}
        if bootstrap is not None:
            data = _read_json_quiet(bootstrap.ui_settings_file())
            if isinstance(data, dict):
                stored.update(data)
                if "text" not in stored and "prompt" in stored:
                    stored["text"] = stored["prompt"]   # the key an older version used
        if self._config is not None:
            for name in PROCEDURE_SETTINGS:
                try:
                    value = self._config.get_property(name)
                except Exception:
                    continue
                if value is None or (name == "text" and not str(value).strip()):
                    continue
                stored[name] = value
        table = self._settings_map()
        table.sort(key=lambda row: row[0] == "output-mode")
        for name, _get, setter in table:
            if stored.get(name) is None:
                continue
            try:
                setter(stored[name])
            except Exception:
                continue

    def _save_settings(self) -> None:
        """Write every setting to ``ui-settings.json`` (0600, like everything
        under the plug-in's base directory) and the procedure's own arguments
        to ``Gimp.ProcedureConfig``; GIMP stores those when the run ends in
        SUCCESS (see :func:`run_main_dialog`)."""
        values: Dict[str, Any] = {}
        for name, getter, _set in self._settings_map():
            try:
                values[name] = getter()
            except Exception:
                continue
        if bootstrap is not None:
            _quiet(lambda: bootstrap._atomic_write_json(bootstrap.ui_settings_file(), values))
        if self._config is not None:
            for name in PROCEDURE_SETTINGS:
                if name not in values:
                    continue
                try:
                    self._config.set_property(name, values[name])
                except Exception:
                    continue

    # ------------------------------------------------------------------ #
    # misc
    # ------------------------------------------------------------------ #
    def _set_status(self, text: str) -> bool:
        self._status.set_markup('<span alpha="80%%">%s</span>' % GLib.markup_escape_text(text))
        return False

    def _open_setup(self, page: Optional[str] = None) -> None:
        """Open Setup on ``page``; by default Doctor when the session is down,
        Install otherwise -- the tab that matches what the user is looking at."""
        module = _optional("ui.setup_dialog")
        if module is None:
            self._set_status("Setup dialog unavailable.")
            return
        if page is None:
            page = "install" if (self._client is not None or self._daemon_stale) else "doctor"
        dialog = module.SetupDialog(self, page=page)
        try:
            dialog.run()
        finally:
            dialog.destroy()
        # Setup was most likely opened because the daemon would not start.  If
        # there is still no session, try again now rather than making the user
        # close this window and reopen it to find out whether the repair took.
        if self._client is None and not self._recovering and not self._worker.busy:
            self._set_status("Connecting to the daemon…")
            self._progress.set_text("Connecting…")
            self._start_session()

    def _on_response(self, _dialog: Gtk.Dialog, response: int) -> None:
        if response == RESPONSE_SETUP:
            self.stop_emission_by_name("response")  # a tool, not a verdict
            self._open_setup()
            return
        if response == Gtk.ResponseType.APPLY:
            self._on_apply()
            self.stop_emission_by_name("response")  # Apply must not close the dialog
            return
        if response in (Gtk.ResponseType.CLOSE, Gtk.ResponseType.DELETE_EVENT):
            self._save_settings()
            self._closed_normally = True
            self._release()

    @property
    def closed_normally(self) -> bool:
        """The user closed a live session -- Close, the window's X or Escape
        -- and its settings were saved on the way out."""
        return self._closed_normally

    def _release(self) -> None:
        """Drop the cached embedding now instead of waiting for LRU eviction
        (API.md §6.6), then close the connection.  Fire and forget: a dying
        dialog must never block on the network."""
        client = self._client
        accepted = self._accepted
        launch = self._launch
        self._accepted = None

        def _work() -> None:
            if client is not None and accepted is not None:
                _quiet(lambda: client.delete_image(accepted.image_id))
            if launch is not None:
                _quiet(launch.close)
            elif client is not None:
                _quiet(client.close)

        # Started as a thread so a slow daemon cannot freeze the closing
        # window, but *joined* with a short bound: this used to be fire and
        # forget, which left an HTTP call in flight while the plug-in process
        # was already heading for exit() -- and a daemon thread re-entering
        # Python during interpreter finalisation aborts the process ("Plug-in
        # crashed").  The entry point quiesces any remainder before returning.
        self._closed = True
        source, self._keepalive_source = self._keepalive_source, None
        if source:
            _quiet(lambda: GLib.source_remove(source))
        thread = threading.Thread(target=_work, name="sam3-release", daemon=True)
        thread.start()
        thread.join(1.5)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _section(title: str) -> Gtk.Frame:
    frame = Gtk.Frame()
    label = Gtk.Label()
    label.set_markup("<b>%s</b>" % GLib.markup_escape_text(title))
    frame.set_label_widget(label)
    frame.set_shadow_type(Gtk.ShadowType.NONE)
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
    box.set_margin_start(10)
    box.set_margin_top(6)
    box.set_margin_bottom(4)
    frame.add(box)
    return frame



def _hint(text: str) -> Gtk.Label:
    """A small dimmed line under a control.

    Tooltips are not enough here: the two thresholds are the whole tuning story
    and a user who does not know to hover sees only a number.
    """
    label = Gtk.Label()
    label.set_markup('<span size="small" alpha="65%%">%s</span>'
                     % GLib.markup_escape_text(text))
    label.set_xalign(0.0)
    label.set_line_wrap(True)
    label.set_max_width_chars(46)
    return label

def _labelled(text: str, widget: Gtk.Widget) -> Gtk.Box:
    box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    label = Gtk.Label(label=text)
    label.set_xalign(0.0)
    label.set_size_request(130, -1)
    box.pack_start(label, False, False, 0)
    widget.set_hexpand(True)
    box.pack_start(widget, True, True, 0)
    return box


def _spin(value: float, lower: float, upper: float, step: float, digits: int = 0) -> Gtk.SpinButton:
    adj = Gtk.Adjustment(value=value, lower=lower, upper=upper, step_increment=step,
                         page_increment=step * 10, page_size=0)
    spin = Gtk.SpinButton(adjustment=adj, climb_rate=step, digits=digits)
    spin.set_numeric(True)
    return spin


def _has_signal(widget: Any, name: str) -> bool:
    try:
        return bool(GObject.signal_lookup(name, type(widget)))
    except Exception:
        return False


def _gimp_pid() -> Optional[int]:
    """GIMP's pid, for ``--parent-pid``.

    DESIGN.md Constraint C: plug-in processes are short-lived, so the daemon
    must watch **GIMP**, not us.  Inside GIMP the plug-in's parent is GIMP;
    outside it this returns None and the daemon simply has no parent to watch.
    """
    if launcher_mod is not None:
        fn = getattr(launcher_mod, "gimp_pid", None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    try:
        import os

        return os.getppid()
    except Exception:  # pragma: no cover
        return None


def _plugin_build() -> str:
    fn = getattr(launcher_mod, "plugin_build", None)
    if fn is None:
        return ""
    try:
        return str(fn() or "")
    except Exception:
        return ""


def _source_layer(image: Any, drawable: Any) -> Any:
    """``gimpbridge.source_layer`` when there is a GIMP to ask, else ``drawable``."""
    resolve = getattr(gimpbridge, "source_layer", None) if gimpbridge is not None else None
    if image is None or not callable(resolve):
        return drawable
    try:
        return resolve(image, drawable)
    except Exception:  # noqa: BLE001 - no GIMP bindings: nothing to resolve against
        return drawable


def _is_image_gone(error: BaseException) -> bool:
    """The daemon no longer holds the image (``404 image_not_found``, or a
    queued job that failed for the same reason).  ``API.md`` §6.2: upload
    it again and retry once."""
    return str(getattr(error, "code", "") or "") == "image_not_found"


def _is_transport_error(error: BaseException) -> bool:
    """``DaemonUnavailable`` / ``DaemonDied`` and their subclasses (such as
    the identity check's error), by class when the client module is
    importable and by name otherwise (the plug-in can be imported flat or
    dotted, which yields two distinct copies of the same classes)."""
    names = ("DaemonUnavailable", "DaemonDied")
    if client_mod is not None:
        classes = tuple(c for c in (getattr(client_mod, n, None) for n in names) if c is not None)
        if classes and isinstance(error, classes):
            return True
    return any(cls.__name__ in names for cls in type(error).__mro__)


def _explain(error: BaseException) -> str:
    name = type(error).__name__
    text = str(error) or name
    return text if text.startswith(name) else "%s: %s" % (name, text)


def _read_json_quiet(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _describe_applied(applied: Any, options: Any, requested: int, bounds: str, note: str) -> str:
    """One sentence saying what Apply made and *where GIMP put it*.

    Channels and paths live in dock dialogs that are closed by default and
    change nothing on the canvas, so "Applied 12 instances as channels"
    read as "nothing happened".  The selection bounds stay in the selection
    line on purpose: they separate "empty" from "not drawn".
    """
    mode = options.mode
    dropped = len(getattr(applied, "dropped", []) or [])
    tail = (" %d dropped by min-area." % dropped) if dropped else ""
    if note:
        tail = "; " + note + "." + tail
    if mode == "selection":
        text = "Applied %d instance(s) to the selection (%s)%s" % (
            requested, options.selection_op, tail or ".")
        if bounds:
            text += " Selection now: %s." % bounds
        return text
    if mode == "channels":
        n = len(getattr(applied, "channels", []) or [])
        return ("Saved %d channel(s)%s Open Windows > Dockable Dialogs > Channels: click "
                "a channel's eye to preview it, or right-click it > Channel to Selection."
                % (n, tail or "."))
    if mode == "paths":
        n = len(getattr(applied, "paths", []) or [])
        if n == 0:
            return ("No path was traced (%d instance(s) requested): the masks were too "
                    "small or too thin at this mask threshold." % requested)
        return ("Added %d path(s)%s They are drawn on the canvas and listed in Windows > "
                "Dockable Dialogs > Paths; Edit > Stroke Path paints them, Select > "
                "From Path turns one into a selection." % (n, tail or "."))
    if mode == "layer-masks":
        n = len(getattr(applied, "layers", []) or [])
        return "Added a mask to %d layer(s)%s See the Layers dialog." % (n, tail or ".")
    if mode == "layer-groups":
        n = len(getattr(applied, "layers", []) or [])
        return ("Made a group of %d masked layer(s)%s See the Layers dialog."
                % (n, tail or "."))
    return "Applied %d instance(s) as %s%s" % (requested, mode, tail or ".")


def _owner_only(path: str, flags: int) -> int:
    """``open(..., opener=)`` hook: create the file readable by its owner only."""
    return os.open(path, flags, 0o600)


def _plugin_log(message: str) -> None:
    """Append one timestamped line to ``<base>/logs/plugin.log`` -- the file
    the entry point writes, so an Apply can be read back next to the launch
    that preceded it.  Appending (never truncating or replacing) is what lets
    the entry point trim the file in place while this and its crash log keep
    writing to it; created 0600 like everything under the base directory.
    Best effort; a dialog never fails over a log line."""
    try:
        import time  # noqa: PLC0415
        if bootstrap is None:
            return
        path = os.path.join(bootstrap.log_dir(), "plugin.log")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8", errors="replace", opener=_owner_only) as fh:
            fh.write("%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), message))
    except Exception:  # noqa: BLE001
        pass


def _quiet(fn: Callable[[], Any]) -> None:
    try:
        fn()
    except Exception:
        pass


def run_main_dialog(parent: Optional[Gtk.Window] = None, **kwargs: Any) -> int:
    """Show the dialog modally and return its final response.

    The window's X and Escape answer ``DELETE_EVENT``, which the entry point
    reports to GIMP as CANCEL -- and GIMP stores a procedure's last-used
    values only after SUCCESS.  This dialog has no Cancel (Apply has already
    done its work by then), so closing a live session that way is reported
    as CLOSE, exactly like the Close button.
    """
    dialog = MainDialog(parent, **kwargs)
    try:
        response = int(dialog.run())
        if response == int(Gtk.ResponseType.DELETE_EVENT) and dialog.closed_normally:
            response = int(Gtk.ResponseType.CLOSE)
        return response
    finally:
        dialog.destroy()


if __name__ == "__main__":  # pragma: no cover - manual harness
    win = MainDialog(None, stub=True)
    win.connect("destroy", Gtk.main_quit)
    win.connect(
        "response",
        lambda _d, r: Gtk.main_quit()
        if r in (Gtk.ResponseType.CLOSE, Gtk.ResponseType.DELETE_EVENT)
        else None,
    )
    Gtk.main()
