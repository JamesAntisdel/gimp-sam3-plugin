"""Real-GTK checks of the segmentation window's daemon lifecycle.

Skipped without PyGObject and a display.  The launcher and the daemon are
replaced by fakes; what is under test is the dialog's reaction to a daemon that
goes away while the window is open -- the "connection refused ... the daemon is
not listening" screen after an idle timeout -- or that no longer holds the
image, to uploads racing prompts, and to its settings surviving a restart.
"""

from __future__ import annotations

import os
import stat
import threading
import time
import types

import pytest

pytestmark = pytest.mark.needs_gtk


def _modules():
    import gi
    gi.require_version("Gtk", "3.0")
    import client as C
    import ui.main_dialog as md
    return C, md


def _pump(seconds):
    from gi.repository import Gtk
    deadline = time.time() + seconds
    while time.time() < deadline:
        while Gtk.events_pending():
            Gtk.main_iteration_do(False)
        time.sleep(0.02)


class _Hello:
    """Permissive stand-in for ``client.Hello``: any attribute reads as ''."""

    api_version = "1.0"
    engine_mode = "stub"
    capabilities = ["pcs", "pvs", "exemplar_boxes"]

    def has_capability(self, name):
        return name in self.capabilities

    def __getattr__(self, name):
        return ""


class _Accepted:
    def __init__(self, image_id):
        self.image_id = image_id


class _Result:
    def __init__(self, request_id, image_id=""):
        self.request_id = request_id
        self.image_id = image_id
        self.instances = []
        self.header = {}
        self.elapsed_ms = 12.0
        self.truncated = False


def _frame(request_id, image_id, w=8, h=8):
    """A real decoded result frame with one 4x4 instance: something for the
    list, the canvas and Apply to disagree about."""
    import client as C
    from ui import canvas as cv
    header = {
        "api_version": "1.0", "job_id": "j-" + request_id, "request_id": request_id,
        "image_id": image_id, "engine": "pcs", "state": "done",
        "prompt": {"kind": "text", "text": "x"},
        "image": {"width": w, "height": h},
        "model_canvas": {"width": w, "height": h},
        "canvas_from_image": {"scale_x": 1.0, "scale_y": 1.0, "offset_x": 0.0, "offset_y": 0.0},
        "mask_encoding": "u8_soft", "elapsed_ms": 1.0, "truncated": False,
        "instances": [
            {"instance_id": 0, "score": 0.9, "label": "x", "bbox": [0, 0, 4, 4],
             "mask_width": 4, "mask_height": 4, "blob_offset": 0, "blob_length": 16},
        ],
    }
    return C.parse_result_frame(cv.encode_frame(header, [b"\xff" * 16]))


class _FakeClient:
    """Alive until ``kill()``; afterwards every call is a refused connection."""

    def __init__(self, C, log):
        self._C = C
        self._log = log
        self.alive = True
        self.pid = 4242
        self.image_id = None          # a fixed id: the same pixels uploaded again
        self.frames = False           # answer prompts with real result frames
        self.upload_gate = None       # an Event an upload waits on
        self.prompt_gates = {}        # text -> Event the prompt waits on
        self.prompt_errors = {}       # text -> [exception to raise, once each]
        self.hello_answer = None      # what GET /hello answers, when not _Hello()

    def _check(self, what):
        self._log.append(what)
        if not self.alive:
            raise self._C.DaemonUnavailable(
                "connection refused at http://127.0.0.1:1 -- the daemon is not listening"
            )

    def hello(self, check=True, timeout=None):
        self._check("hello")
        return self.hello_answer or _Hello()

    def upload_image(self, pixels, width, height, **kw):
        if self.upload_gate is not None:
            self.upload_gate.wait(5.0)
        self._check("upload")
        return _Accepted(self.image_id or "img-%d" % len(self._log))

    def run_text(self, image_id, text, request_id="", **kw):
        self._check("run_text:%s" % text)
        self._log.append("max_instances:%s" % kw.get("max_instances"))
        self._log.append("boxes:%d" % len(kw.get("boxes") or []))
        self._log.append("image:%s" % image_id)
        gate = self.prompt_gates.pop(text, None)
        if gate is not None:
            gate.wait(5.0)
        errors = self.prompt_errors.get(text)
        if errors:
            raise errors.pop(0)
        return _frame(request_id, image_id) if self.frames else _Result(request_id, image_id)

    def run_points(self, image_id, points, box=None, request_id="", **kw):
        self._check("run_points")
        self._log.append("box:%s" % (list(box) if box else None))
        self._log.append("points:%d multimask:%s max:%s" % (
            len(list(points or [])), kw.get("multimask"), kw.get("max_instances")))
        return _frame(request_id, image_id) if self.frames else _Result(request_id, image_id)

    def delete_image(self, image_id):
        self._log.append("delete")

    def close(self):
        self._log.append("close")

    def kill(self):
        self.alive = False


def _uploaded():
    geometry = types.SimpleNamespace(width=8, height=8, source_width=8, source_height=8)
    return types.SimpleNamespace(pixels=bytes(8 * 8 * 3), geometry=geometry)


@pytest.fixture
def harness(sam3_home, monkeypatch):
    C, md = _modules()
    log = []
    clients = []

    def fake_find_or_spawn(**kw):
        client = _FakeClient(C, log)
        clients.append(client)
        return types.SimpleNamespace(client=client, hello=_Hello(), info={"pid": 4242, "port": 1},
                                     spawned=True, close=client.close)

    monkeypatch.setattr(md.launcher_mod, "find_or_spawn", fake_find_or_spawn)
    monkeypatch.setattr(md.MainDialog, "_read_pixels", lambda self: _uploaded())
    dlg = md.MainDialog(None, stub=True)
    dlg.show_all()
    _pump(0.5)
    if not clients:
        dlg._start_session()
    deadline = time.time() + 5.0
    while time.time() < deadline and dlg._accepted is None:
        _pump(0.1)
    assert dlg._accepted is not None, "the first session never came up"
    yield md, dlg, clients, log
    dlg._release()
    dlg.destroy()


def _wait_for_status(dlg, needle, seconds=8.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        _pump(0.1)
        if needle in dlg._status.get_text():
            return True
    return False


def _wait(predicate, seconds=8.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        _pump(0.05)
        if predicate():
            return True
    return False


def test_a_prompt_against_a_dead_daemon_respawns_reuploads_and_repeats(harness):
    md, dlg, clients, log = harness
    assert len(clients) == 1
    clients[0].kill()

    dlg._prompt_entry.set_text("guitar")
    dlg._segment_text()

    assert _wait_for_status(dlg, "instance(s)"), dlg._status.get_text()
    assert len(clients) == 2, "the launcher was not asked for a fresh daemon"
    assert log.count("upload") == 2, "the new daemon never received the image"
    assert log.count("run_text:guitar") == 2, "the prompt was not repeated"
    assert "0 instance(s)" in dlg._status.get_text()
    assert dlg._client is clients[1]
    assert not dlg._recovering and not dlg._retrying and dlg._last_action is None


def test_a_second_failure_is_reported_not_retried_forever(harness):
    md, dlg, clients, log = harness
    clients[0].kill()
    # Every daemon the launcher hands back is already dead.
    orig = md.launcher_mod.find_or_spawn

    def dead_find_or_spawn(**kw):
        launch = orig(**kw)
        launch.client.kill()
        return launch

    md.launcher_mod.find_or_spawn = dead_find_or_spawn

    dlg._prompt_entry.set_text("guitar")
    dlg._segment_text()

    assert _wait_for_status(dlg, "Open Setup / Doctor"), dlg._status.get_text()
    assert len(clients) == 2   # exactly one respawn, then the error is shown
    assert not dlg._recovering


def test_keepalive_notices_a_dead_daemon_before_the_next_click(harness):
    md, dlg, clients, log = harness
    clients[0].kill()

    assert dlg._keepalive() is True          # the timer stays armed
    assert _wait_for_status(dlg, "Ready"), dlg._status.get_text()
    assert len(clients) == 2
    assert log.count("upload") == 2
    assert log.count("run_text:guitar") == 0  # nothing to replay: the user asked for nothing
    assert dlg._client is clients[1]


def test_keepalive_touches_a_healthy_daemon(harness):
    md, dlg, clients, log = harness
    before = log.count("hello")
    dlg._keepalive()
    _pump(0.5)
    assert log.count("hello") == before + 1
    assert len(clients) == 1


def test_setup_is_a_secondary_button_in_the_action_area(harness):
    """Setup / Doctor used to sit at the foot of the scrolling side panel,
    below the instance list and three frames of sliders -- out of view on the
    default window size, and hardest to find exactly when the session had
    failed.  It belongs in the button row, as a secondary (left-aligned)
    action-area child the way GTK places Help, and pressing it must not close
    the window."""
    from gi.repository import Gtk
    md, dlg, clients, log = harness
    button = dlg.get_widget_for_response(md.RESPONSE_SETUP)
    assert button is not None and button.get_visible()
    assert "Doctor" in button.get_label()
    box = button.get_parent()
    assert isinstance(box, Gtk.ButtonBox)
    assert box.get_child_secondary(button), "must be kept apart from Close / Apply"
    # No Setup / Doctor buttons buried in the side panel any more.
    labels = []

    def walk(widget):
        if isinstance(widget, Gtk.ButtonBox):
            return  # the action area itself, where the button now lives
        if isinstance(widget, Gtk.Button) and widget.get_label():
            labels.append(widget.get_label())
        if isinstance(widget, Gtk.Container):
            widget.foreach(walk)
    dlg.get_content_area().foreach(walk)
    assert not [l for l in labels if "Setup" in l or "Doctor" in l], labels

    # The response is swallowed: the window stays open and Setup is opened
    # instead (replaced here so no modal window appears).
    opened = []
    dlg._open_setup = lambda page=None: opened.append(page)
    closed = []
    dlg.connect("close", lambda *_a: closed.append(True))
    dlg.response(md.RESPONSE_SETUP)
    _pump(0.2)
    assert opened == [None]
    assert not closed and dlg.get_visible()


def test_setup_reconnects_when_the_session_is_down(harness, monkeypatch):
    """Setup is most often opened because the daemon would not start.  When it
    closes and there is still no session, the dialog tries again itself
    instead of making the user close and reopen the window."""
    md, dlg, clients, log = harness

    class _NoSetup:
        def __init__(self, *_a, **_k):
            pass

        def run(self):
            return 0

        def destroy(self):
            pass

    fake_module = types.SimpleNamespace(SetupDialog=_NoSetup)
    monkeypatch.setattr(md, "_optional", lambda name: fake_module)
    dlg._client = None
    before = len(clients)
    dlg._open_setup()
    deadline = time.time() + 5.0
    while time.time() < deadline and len(clients) == before:
        _pump(0.1)
    assert len(clients) == before + 1, "no new session was started after Setup closed"
    # With a live session, Setup opens on Install and nothing is restarted.
    _pump(0.3)
    assert dlg._client is not None
    dlg._open_setup()
    _pump(0.3)
    assert len(clients) == before + 1


def test_max_instances_is_a_setting_that_reaches_the_daemon(harness):
    """The canvas used to hard-code 64 per text prompt; the daemon allows 256
    and crowded scenes need it.  The spin button's value must be what the
    prompt carries, and it must survive a save/load of the settings."""
    md, dlg, clients, log = harness
    assert dlg._max_instances_value() == md.PROMPT_MAX_INSTANCES
    dlg._max_instances.set_value(200)
    dlg._prompt_entry.set_text("window")
    dlg._segment_text()
    assert _wait_for_status(dlg, "instance(s)"), dlg._status.get_text()
    assert "max_instances:200" in log, log
    dlg._max_instances.set_value(md.MAX_INSTANCES_LIMIT + 50)
    assert dlg._max_instances_value() == md.MAX_INSTANCES_LIMIT, "clamped to the API ceiling"
    names = [name for name, _g, _s in dlg._settings_map()]
    assert "max-instances" in names, "must persist like the other prompt settings"


def test_a_stale_daemon_is_called_out(harness, monkeypatch):
    """A daemon whose /hello build differs from the bundled source is older
    code; the device line and the status line must say so and Setup must open
    on the Install tab, where Update daemon lives."""
    md, dlg, clients, log = harness
    monkeypatch.setattr(md.bootstrap, "bundled_daemon_build", lambda: "0badf00d")
    dlg._bundled_build = None

    class _Fresh(_Hello):
        build = "0badf00d"

    class _Old(_Hello):
        build = "deadbeef"

    dlg._hello = _Fresh()
    dlg._update_device_line()
    assert not dlg._daemon_stale
    assert "OUT OF DATE" not in dlg._progress.get_text()

    dlg._hello = _Old()
    dlg._update_device_line()
    assert dlg._daemon_stale
    assert "OUT OF DATE" in dlg._progress.get_text()
    assert "Update daemon" in dlg._stale_hint()

    opened = []
    fake_module = types.SimpleNamespace(
        SetupDialog=lambda parent, page: opened.append(page) or types.SimpleNamespace(
            run=lambda: 0, destroy=lambda: None))
    monkeypatch.setattr(md, "_optional", lambda name: fake_module)
    dlg._open_setup()
    assert opened == ["install"]

    # A daemon too old to report a build is not accused of anything.
    dlg._hello = _Hello()
    dlg._update_device_line()
    assert not dlg._daemon_stale


def test_a_new_prompt_supersedes_a_running_one(harness):
    """The daemon supersedes queued prompts and the dialog drops stale
    results; the worker used to refuse a second submission anyway."""
    md, dlg, clients, log = harness
    import threading
    gate = threading.Event()
    dlg._worker.submit(lambda: gate.wait(5.0), lambda *_a: False)
    assert dlg._worker.busy
    assert dlg._ready_to_prompt(), "a running job must not block a new prompt"
    assert dlg._segment_button.get_sensitive()
    dlg._prompt_entry.set_text("apple")
    dlg._segment_text()
    gate.set()
    assert _wait_for_status(dlg, "instance(s)"), dlg._status.get_text()
    assert log.count("run_text:apple") == 1


def test_a_stale_result_does_not_reset_the_busy_state(harness):
    md, dlg, clients, log = harness
    import threading
    gate = threading.Event()
    dlg._worker.submit(lambda: gate.wait(5.0), lambda *_a: False)
    dlg._latest_request_id = "r-newer"
    if dlg._canvas is not None:
        dlg._canvas.set_busy(True, "queued", 0.0)
    dlg._on_prompt_done(_Result("r-older"), None)          # stale: ignored
    assert dlg._canvas is None or dlg._canvas._busy
    dlg._on_prompt_done(None, None)                        # superseded while busy
    assert dlg._canvas is None or dlg._canvas._busy
    gate.set()


def test_the_projection_switch_reuploads_and_clears_results(harness):
    md, dlg, clients, log = harness
    assert dlg._use_projection.get_active()
    assert "use-projection" in [n for n, _g, _s in dlg._settings_map()]
    uploads_before = log.count("upload")
    dlg._use_projection.set_active(False)
    deadline = time.time() + 5.0
    while time.time() < deadline and log.count("upload") == uploads_before:
        _pump(0.1)
    assert log.count("upload") == uploads_before + 1, "the image must be uploaded again"
    assert dlg._result is None and dlg._instances == []


def test_zoom_buttons_and_the_key_hint_exist(harness):
    md, dlg, clients, log = harness
    if dlg._canvas is None:
        pytest.skip("no canvas")
    assert set(dlg._zoom_buttons) == {"Fit", "1:1", "+", "−"}
    before = dlg._canvas.view.zoom
    dlg._zoom_buttons["+"].clicked()
    assert dlg._canvas.view.zoom > before


def test_a_cpu_daemon_is_called_out(harness):
    md, dlg, clients, log = harness

    class _Cpu(_Hello):
        device = "cpu"
        engine_mode = "torch"

    dlg._hello = _Cpu()
    dlg._update_device_line()
    assert "CPU ONLY" in dlg._progress.get_text()


def test_a_box_drawn_in_pick_mode_is_an_example_for_the_phrase(harness):
    """The daemon advertised exemplar boxes and the dialog never sent one."""
    md, dlg, clients, log = harness
    dlg._mode_pick.set_active(True)
    dlg._prompt_entry.set_text("apple")
    dlg._on_canvas_box(None, 10.0, 20.0, 30.0, 40.0)
    assert _wait_for_status(dlg, "instance(s)"), dlg._status.get_text()
    assert "boxes:1" in log and "run_points" not in log
    assert dlg._exemplar_boxes == [{"box": [10.0, 20.0, 30.0, 40.0], "label": 1}]
    if dlg._canvas is not None:
        assert dlg._canvas.boxes == [(10.0, 20.0, 30.0, 40.0, 1)]
    dlg._on_canvas_box(None, 50.0, 20.0, 70.0, 40.0)
    assert _wait_for_status(dlg, "instance(s)")
    assert log[-2:] != [] and "boxes:2" in log
    dlg._clear_points()
    assert dlg._exemplar_boxes == []
    if dlg._canvas is not None:
        assert dlg._canvas.boxes == []


def test_a_box_drawn_in_point_mode_is_a_visual_prompt(harness):
    md, dlg, clients, log = harness
    dlg._mode_points.set_active(True)
    dlg._prompt_entry.set_text("apple")   # a phrase must not turn it into an exemplar
    dlg._on_canvas_box(None, 10.0, 20.0, 30.0, 40.0)
    assert _wait_for_status(dlg, "instance(s)"), dlg._status.get_text()
    assert "run_points" in log and "box:[10.0, 20.0, 30.0, 40.0]" in log
    assert dlg._exemplar_boxes == []


def test_gimps_selection_becomes_a_box(harness, monkeypatch):
    md, dlg, clients, log = harness
    fake_bridge = types.SimpleNamespace(
        selection_box=lambda image, geometry: (1.0, 2.0, 3.0, 4.0),
        gimp_available=lambda: True,
    )
    monkeypatch.setattr(md, "gimpbridge", fake_bridge)
    dlg._image = object()
    dlg._mode_points.set_active(True)
    dlg._use_selection_as_box()
    assert _wait_for_status(dlg, "instance(s)"), dlg._status.get_text()
    assert "box:[1.0, 2.0, 3.0, 4.0]" in log
    fake_bridge.selection_box = lambda image, geometry: None
    dlg._use_selection_as_box()
    assert "Nothing is selected" in dlg._status.get_text()


def test_recent_prompts_are_remembered(harness):
    md, dlg, clients, log = harness
    dlg._prompt_entry.set_text("green apple")
    dlg._segment_text()
    assert _wait_for_status(dlg, "instance(s)")
    dlg._prompt_entry.set_text("pear")
    dlg._segment_text()
    assert _wait_for_status(dlg, "instance(s)")
    assert dlg._load_prompt_history()[:2] == ["pear", "green apple"]
    model = dlg._prompt_combo.get_model()
    assert [row[0] for row in model][:2] == ["pear", "green apple"]
    # A fresh dialog on the same home sees the history.
    other = md.MainDialog(None, stub=True)
    try:
        assert [row[0] for row in other._prompt_combo.get_model()][:2] == ["pear", "green apple"]
    finally:
        other.destroy()


def test_only_a_lone_click_asks_for_alternative_candidates(harness):
    """Three candidates resolve ONE ambiguous click.  With a second point the
    ambiguity is gone; asking for three again returned guesses the extra
    point was meant to remove."""
    md, dlg, clients, log = harness
    if dlg._canvas is None:
        pytest.skip("no canvas")
    dlg._mode_points.set_active(True)
    dlg._canvas.add_point(10.0, 10.0, 1)
    assert _wait_for_status(dlg, "instance(s)"), dlg._status.get_text()
    assert "points:1 multimask:True max:3" in log
    dlg._canvas.add_point(20.0, 20.0, 1)
    assert _wait_for_status(dlg, "instance(s)")
    assert "points:2 multimask:False max:1" in log
    assert log.count("run_points") == 2, "a click must run exactly one prompt"
    # Backspace re-runs with what is left; Esc keeps the result.
    dlg._canvas.remove_last_point()
    assert _wait_for_status(dlg, "instance(s)")
    assert log.count("run_points") == 3 and log[-1] == "points:1 multimask:True max:3"
    dlg._canvas.clear_points()
    _pump(0.2)
    assert log.count("run_points") == 3
    assert "Points cleared" in dlg._status.get_text()


def test_a_box_prompt_asks_for_one_mask(harness):
    md, dlg, clients, log = harness
    dlg._mode_points.set_active(True)
    dlg._on_canvas_box(None, 1.0, 2.0, 3.0, 4.0)
    assert _wait_for_status(dlg, "instance(s)")
    assert "points:0 multimask:False max:1" in log


# --------------------------------------------------------------------------- #
# uploads and prompts in step
# --------------------------------------------------------------------------- #
def test_no_prompt_is_sent_while_an_upload_is_in_flight(harness):
    """A prompt sent during a re-upload named the *old* image id, and its
    result was then shown over the new pixels with Apply enabled."""
    md, dlg, clients, log = harness
    gate = threading.Event()
    clients[0].upload_gate = gate
    old = dlg._accepted.image_id
    dlg._use_projection.set_active(False)
    _pump(0.2)
    assert not dlg._ready_to_prompt()
    assert "Uploading" in dlg._status.get_text()
    dlg._prompt_entry.set_text("apple")
    dlg._segment_text()
    _pump(0.2)
    assert "run_text:apple" not in log
    gate.set()
    assert _wait(lambda: dlg._accepted is not None and dlg._accepted.image_id != old)
    dlg._segment_text()
    assert _wait_for_status(dlg, "instance(s)")
    assert log[-1] == "image:%s" % dlg._accepted.image_id


def test_a_prompt_running_when_the_pixels_change_is_dropped(harness):
    md, dlg, clients, log = harness
    clients[0].frames = True
    gate = threading.Event()
    clients[0].prompt_gates["apple"] = gate
    dlg._prompt_entry.set_text("apple")
    dlg._segment_text()
    _pump(0.2)
    dlg._use_projection.set_active(False)          # re-upload while "apple" runs
    assert _wait(lambda: dlg._accepted is not None and log.count("upload") == 2)
    gate.set()
    _pump(0.5)
    assert dlg._result is None, "a result for the old pixels was adopted"
    assert not dlg._apply_button.get_sensitive()
    if dlg._canvas is not None:
        assert not dlg._canvas._busy, "nothing is running any more"


def test_the_projection_switch_reuploads_even_while_the_worker_is_busy(harness):
    """The switch used to be ignored while anything ran, leaving the checkbox
    saying one thing and the daemon holding the other."""
    md, dlg, clients, log = harness
    gate = threading.Event()
    dlg._worker.submit(lambda: gate.wait(5.0), lambda *_a: False)
    uploads = log.count("upload")
    dlg._use_projection.set_active(False)
    assert _wait(lambda: log.count("upload") == uploads + 1)
    gate.set()
    assert _wait(lambda: dlg._accepted is not None)
    assert dlg._upload_projection is False


def test_a_switch_flipped_while_the_session_starts_is_caught_up(harness, monkeypatch):
    md, dlg, clients, log = harness
    dlg._client = None                  # as if the session were still coming up
    gate = threading.Event()
    clients[0].upload_gate = gate
    orig = md.launcher_mod.find_or_spawn
    dlg._start_session()                # reads the pixels with the switch ON
    dlg._use_projection.set_active(False)
    gate.set()
    clients[-1].upload_gate = None
    assert _wait(lambda: dlg._upload_projection is False and dlg._accepted is not None), (
        dlg._upload_projection, log)
    assert orig is md.launcher_mod.find_or_spawn


def test_pixels_are_read_on_the_gtk_thread(harness, monkeypatch):
    """libgimp's pipe to GIMP is not thread-safe, and the GTK thread makes
    GIMP calls of its own; the worker only uploads."""
    md, dlg, clients, log = harness
    threads = []

    def read(self):
        threads.append(threading.current_thread() is threading.main_thread())
        return _uploaded()

    monkeypatch.setattr(md.MainDialog, "_read_pixels", read)
    dlg._use_projection.set_active(False)
    assert _wait(lambda: dlg._accepted is not None and threads)
    clients[0].kill()
    dlg._keepalive()
    assert _wait(lambda: len(clients) == 2 and dlg._accepted is not None)
    assert threads == [True, True], threads


def test_list_canvas_and_apply_agree_after_new_pixels(harness):
    """New pixels clear the canvas's result; the list, ``_result`` and the
    Apply button used to keep the old one."""
    md, dlg, clients, log = harness
    clients[0].frames = True
    dlg._prompt_entry.set_text("apple")
    dlg._segment_text()
    assert _wait(lambda: dlg._result is not None)
    assert len(dlg._store) == 1 and dlg._apply_button.get_sensitive()
    clients[0].kill()
    dlg._keepalive()                          # restart: a new image id
    assert _wait(lambda: len(clients) == 2 and dlg._accepted is not None
                 and not dlg._recovering)
    assert dlg._result is None and len(dlg._store) == 0
    assert not dlg._apply_button.get_sensitive()
    if dlg._canvas is not None:
        assert dlg._canvas.instances == []


def test_same_pixels_keep_the_result_everywhere(harness):
    md, dlg, clients, log = harness
    clients[0].frames = True
    clients[0].image_id = "same-pixels"
    dlg._use_projection.set_active(False)
    assert _wait(lambda: dlg._accepted is not None and dlg._accepted.image_id == "same-pixels")
    dlg._prompt_entry.set_text("apple")
    dlg._segment_text()
    assert _wait(lambda: dlg._result is not None)
    orig = md.launcher_mod.find_or_spawn

    def same(**kw):
        launch = orig(**kw)
        launch.client.image_id, launch.client.frames = "same-pixels", True
        return launch

    md.launcher_mod.find_or_spawn = same
    clients[0].kill()
    dlg._keepalive()
    assert _wait(lambda: len(clients) == 2 and dlg._accepted is not None
                 and not dlg._recovering)
    assert dlg._result is not None and len(dlg._store) == 1
    assert dlg._apply_button.get_sensitive()
    if dlg._canvas is not None:
        assert len(dlg._canvas.instances) == 1


# --------------------------------------------------------------------------- #
# the daemon no longer holds the image
# --------------------------------------------------------------------------- #
def test_an_evicted_image_is_uploaded_again_and_the_prompt_repeated(harness):
    """image ids are content hashes shared by every window, the daemon keeps
    only a few images, and closing another window on the same pixels deletes
    them: ``image_not_found`` must lead to one re-upload and one retry."""
    md, dlg, clients, log = harness
    import client as C
    clients[0].prompt_errors["apple"] = [
        C.ApiError("image_not_found", "no such image", status=404)]
    uploads = log.count("upload")
    dlg._prompt_entry.set_text("apple")
    dlg._segment_text()
    assert _wait_for_status(dlg, "instance(s)"), dlg._status.get_text()
    assert log.count("upload") == uploads + 1
    assert log.count("run_text:apple") == 2
    assert len(clients) == 1, "the daemon was alive; nothing to restart"
    assert not dlg._recovering and dlg._last_action is None


def test_an_image_that_stays_gone_is_reported_once(harness):
    md, dlg, clients, log = harness
    import client as C
    clients[0].prompt_errors["apple"] = [
        C.ApiError("image_not_found", "no such image", status=404) for _ in range(3)]
    dlg._prompt_entry.set_text("apple")
    dlg._segment_text()
    assert _wait_for_status(dlg, "image_not_found"), dlg._status.get_text()
    _pump(0.3)
    assert log.count("run_text:apple") == 2, "retried exactly once"


# --------------------------------------------------------------------------- #
# stale answers
# --------------------------------------------------------------------------- #
def test_an_older_requests_error_does_not_touch_the_newer_job(harness):
    md, dlg, clients, log = harness
    import client as C
    first, second = threading.Event(), threading.Event()
    clients[0].prompt_gates["first"] = first
    clients[0].prompt_errors["first"] = [C.JobFailed("j-1", "inference_failed", "old")]
    clients[0].prompt_gates["second"] = second
    dlg._prompt_entry.set_text("first")
    dlg._segment_text()
    _pump(0.2)
    dlg._prompt_entry.set_text("second")
    dlg._segment_text()
    _pump(0.2)
    action = dlg._last_action
    first.set()                          # the older request fails now
    _pump(0.5)
    assert "old" not in dlg._status.get_text(), dlg._status.get_text()
    assert dlg._last_action == action, "the newer job's replay was forgotten"
    if dlg._canvas is not None:
        assert dlg._canvas._busy, "the newer job is still running"
    second.set()
    assert _wait_for_status(dlg, "instance(s)")


# --------------------------------------------------------------------------- #
# the point prompt's box
# --------------------------------------------------------------------------- #
def test_a_box_stays_in_the_prompt_for_the_clicks_that_refine_it(harness):
    """The canvas kept drawing the box, but the first refining click sent
    the points alone -- and the result jumped out of the box."""
    md, dlg, clients, log = harness
    if dlg._canvas is None:
        pytest.skip("no canvas")
    dlg._mode_points.set_active(True)
    dlg._on_canvas_box(None, 1.0, 1.0, 7.0, 7.0)
    assert _wait_for_status(dlg, "instance(s)")
    dlg._canvas.add_point(3.0, 3.0, 1)
    assert _wait(lambda: log.count("run_points") == 2)
    _pump(0.2)
    assert log[-2:] == ["box:[1.0, 1.0, 7.0, 7.0]", "points:1 multimask:False max:1"], log[-2:]
    # Esc clears the points; the box is still drawn, so it still prompts.
    dlg._canvas.clear_points()
    assert _wait(lambda: log.count("run_points") == 3)
    _pump(0.2)
    assert log[-2:] == ["box:[1.0, 1.0, 7.0, 7.0]", "points:0 multimask:False max:1"]
    # Clear points takes the box off the canvas, and out of the prompt.
    dlg._clear_points()
    dlg._canvas.add_point(4.0, 4.0, 1)
    assert _wait(lambda: log.count("run_points") == 4)
    _pump(0.2)
    assert log[-2:] == ["box:None", "points:1 multimask:True max:3"]


def test_a_text_prompt_takes_the_point_box_off_the_canvas(harness):
    md, dlg, clients, log = harness
    if dlg._canvas is None:
        pytest.skip("no canvas")
    dlg._mode_points.set_active(True)
    dlg._on_canvas_box(None, 1.0, 1.0, 7.0, 7.0)
    assert _wait_for_status(dlg, "instance(s)")
    dlg._prompt_entry.set_text("apple")
    dlg._segment_text()
    assert _wait_for_status(dlg, "instance(s)")
    assert dlg._canvas.boxes == []
    dlg._mode_points.set_active(True)
    dlg._canvas.add_point(3.0, 3.0, 1)
    assert _wait(lambda: log.count("run_points") == 2)
    _pump(0.2)
    assert "box:None" in log[-2:]


# --------------------------------------------------------------------------- #
# the drawable
# --------------------------------------------------------------------------- #
def test_the_dialog_works_on_the_layer_a_mask_stands_for(harness, monkeypatch):
    md, dlg, clients, log = harness
    mask, layer, image = object(), object(), object()
    bridge = types.SimpleNamespace(
        source_layer=lambda img, drawable: layer if drawable is mask else drawable,
        gimp_available=lambda: False,
    )
    monkeypatch.setattr(md, "gimpbridge", bridge)
    other = md.MainDialog(None, stub=True, image=image, drawable=mask)
    try:
        assert other._drawable is layer
    finally:
        other._release()
        other.destroy()


# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #
def _config_class():
    """A GObject shaped like plug-in-sam3-segment's Gimp.ProcedureConfig: the
    arguments sam3_gimp.py registers, and nothing else -- set_property on
    anything more raises TypeError, as a real config does."""
    from gi.repository import GObject

    nicks = ("selection-replace", "selection-add", "selection-subtract",
             "selection-intersect", "channels", "layer-masks", "layer-groups", "paths")

    class Cfg(GObject.Object):
        text = GObject.Property(type=str, default="")
        use_projection = GObject.Property(type=bool, default=True, nick="use-projection")
        output_mode = GObject.Property(type=str, default="selection-replace",
                                       nick="output-mode")
        mask_threshold = GObject.Property(type=int, default=128, minimum=1, maximum=255,
                                          nick="mask-threshold")
        score_threshold = GObject.Property(type=float, default=0.3, minimum=0.0,
                                           maximum=1.0, nick="score-threshold")
        max_instances = GObject.Property(type=int, default=64, minimum=1, maximum=256,
                                         nick="max-instances")

        def set_property(self, name, value):
            # gimp_param_choice_validate: an unknown nick falls back to the default
            if name == "output-mode" and value not in nicks:
                value = "selection-replace"
            GObject.Object.set_property(self, name, value)

    return Cfg


def test_every_setting_survives_a_restart_through_the_procedure_config(harness):
    """Only the procedure's own arguments reach Gimp.ProcedureConfig; "prompt"
    was not one (the argument is "text"), "selection" is not an output-mode
    nick, and the rest were silently dropped whenever a config existed."""
    md, dlg, clients, log = harness
    Cfg = _config_class()
    cfg = Cfg()
    first = md.MainDialog(None, stub=True, config=cfg)
    try:
        first._prompt_entry.set_text("yellow school bus")
        first._mode_combo.set_active_id("selection")
        first._op_combo.set_active_id("subtract")
        first._feather.set_value(4.0)
        first._grow.set_value(3)
        first._opacity_scale.set_value(0.8)
        first._fill_holes.set_active(True)
        first._duplicate_layer.set_active(False)
        first._mask_scale.set_value(90)
        first._save_settings()
    finally:
        first._release()
        first.destroy()
    assert cfg.get_property("text") == "yellow school bus"
    assert cfg.get_property("output-mode") == "selection-subtract"
    assert cfg.get_property("mask-threshold") == 90

    second = md.MainDialog(None, stub=True, config=cfg)
    try:
        assert second._prompt_entry.get_text() == "yellow school bus"
        assert second._mode_combo.get_active_id() == "selection"
        assert second._op_combo.get_active_id() == "subtract"
        assert second._feather.get_value() == 4.0 and second._grow.get_value() == 3
        assert abs(second._opacity_scale.get_value() - 0.8) < 1e-6
        assert second._fill_holes.get_active() and not second._duplicate_layer.get_active()
        assert second._mask_scale.get_value() == 90
    finally:
        second._release()
        second.destroy()


def test_a_non_selection_mode_keeps_the_selection_operation(harness):
    md, dlg, clients, log = harness
    cfg = _config_class()()
    first = md.MainDialog(None, stub=True, config=cfg)
    try:
        first._op_combo.set_active_id("intersect")
        first._mode_combo.set_active_id("paths")
        first._save_settings()
    finally:
        first._release()
        first.destroy()
    assert cfg.get_property("output-mode") == "paths"
    second = md.MainDialog(None, stub=True, config=cfg)
    try:
        assert second._mode_combo.get_active_id() == "paths"
        assert second._op_combo.get_active_id() == "intersect"
    finally:
        second._release()
        second.destroy()


def test_the_procedure_settings_are_the_registered_arguments(harness):
    """Every name the dialog hands Gimp.ProcedureConfig is an argument of
    plug-in-sam3-segment, and every output-mode it writes is a valid nick."""
    import ast
    from pathlib import Path
    md, dlg, clients, log = harness
    entry = Path(md.__file__).resolve().parents[1] / "sam3_gimp.py"
    tree = ast.parse(entry.read_text(encoding="utf-8"))
    declared = set()
    nicks = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "").startswith("add_")
                and getattr(node.func, "attr", "").endswith("_argument") and node.args
                and isinstance(node.args[0], ast.Constant)):
            declared.add(node.args[0].value)
        if (isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "OUTPUT_MODES"
                                                  for t in node.targets)):
            nicks = ast.literal_eval(node.value)
    assert set(md.PROCEDURE_SETTINGS) <= declared, set(md.PROCEDURE_SETTINGS) - declared
    names = [n for n, _g, _s in dlg._settings_map()]
    assert set(md.PROCEDURE_SETTINGS) <= set(names)
    for mode, _label in md.OUTPUT_MODES:
        for op, _l in md.SELECTION_OPS:
            nick = md.mode_nick(mode, op)
            assert nick in nicks, nick
            assert md.split_mode_nick(nick)[0] == mode


def test_the_window_x_after_a_session_counts_as_close(harness, monkeypatch):
    """GIMP stores last-used values only when the run succeeds, and the X /
    Escape answer DELETE_EVENT, which the entry point reports as CANCEL."""
    from gi.repository import GLib, Gtk
    md, dlg, clients, log = harness

    base = md.MainDialog

    class Closing(base):
        def __init__(self, *a, **k):
            base.__init__(self, *a, **k)
            GLib.timeout_add(300, lambda: self.response(Gtk.ResponseType.DELETE_EVENT) or False)

    monkeypatch.setattr(md, "MainDialog", Closing)
    assert md.run_main_dialog(None, stub=True) == int(Gtk.ResponseType.CLOSE)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_what_the_dialog_writes_is_owner_only(harness):
    md, dlg, clients, log = harness
    dlg._remember_prompt("pear")
    dlg._save_settings()
    md._plugin_log("a line")
    for path in (dlg._prompt_history_path(), md.bootstrap.ui_settings_file(),
                 os.path.join(md.bootstrap.log_dir(), "plugin.log")):
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600, path


# --------------------------------------------------------------------------- #
# the device line
# --------------------------------------------------------------------------- #
class _UndecidedHello(_Hello):
    """What /hello answers before the first prompt has loaded the model."""

    device = "auto"
    dtype = "auto"


class _CudaHello(_Hello):
    device = "cuda:0"
    dtype = "bfloat16"
    engine_mode = "torch"


def test_an_undecided_device_reads_as_such(harness):
    md, dlg, clients, log = harness
    dlg._hello = _UndecidedHello()
    dlg._update_device_line()
    text = dlg._progress.get_text()
    assert "detecting on first prompt" in text
    assert "auto" not in text


def test_the_first_prompt_brings_the_device_line_up_to_date(harness):
    """The daemon chooses its device when the first prompt loads the model,
    so the launch-time /hello said "auto"; the line must not keep saying so."""
    md, dlg, clients, log = harness
    dlg._hello = _UndecidedHello()
    dlg._update_device_line()
    clients[0].hello_answer = _CudaHello()
    dlg._prompt_entry.set_text("apple")
    dlg._segment_text()
    assert _wait(lambda: "cuda:0" in dlg._progress.get_text()), dlg._progress.get_text()
    assert "bfloat16" in dlg._progress.get_text()
    hellos = log.count("hello")
    dlg._prompt_entry.set_text("pear")
    dlg._segment_text()
    assert _wait_for_status(dlg, "instance(s)")
    _pump(0.3)
    assert log.count("hello") == hellos, "once the device is known, no more asking"


def test_the_keepalive_brings_the_device_line_up_to_date(harness):
    md, dlg, clients, log = harness
    dlg._hello = _UndecidedHello()
    dlg._update_device_line()
    clients[0].hello_answer = _CudaHello()
    dlg._keepalive()
    assert _wait(lambda: "cuda:0" in dlg._progress.get_text()), dlg._progress.get_text()
    assert isinstance(dlg._hello, _CudaHello)
