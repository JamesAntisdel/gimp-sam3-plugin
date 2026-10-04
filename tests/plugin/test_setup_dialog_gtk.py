"""Real-GTK checks of the Setup dialog: the existing-environment row, Cancel
and Close, the weights download, Doctor, and the remote-daemon warning.

Skipped without PyGObject and a display.  Probes and installers are replaced
with canned answers (or a harmless silent child process) so the tests exercise
the dialog's plumbing, not a real environment.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

pytestmark = pytest.mark.needs_gtk


def _dialog_module():
    import gi
    gi.require_version("Gtk", "3.0")
    import ui.setup_dialog as sd
    return sd


SERVING = {"ok": True, "python_version": "3.12.10", "torch": "2.10.0+cu128",
           "cuda": True, "device_name": "RTX 2080 Ti", "capability": [7, 5],
           "transformers": "5.16.1", "sam3gimpd": "0.1.1"}


def _pump(seconds):
    from gi.repository import Gtk
    deadline = time.time() + seconds
    while time.time() < deadline:
        while Gtk.events_pending():
            Gtk.main_iteration_do(False)
        time.sleep(0.02)


def test_row_probes_itself_and_offers_reinstall(sam3_home, monkeypatch, tmp_path):
    import launcher as L
    sd = _dialog_module()
    py = tmp_path / "python"; py.write_text("")
    L.set_configured_python(str(py), has_daemon=True)
    probe = dict(SERVING, path=str(py))
    monkeypatch.setattr(sd.bootstrap, "probe_interpreter", lambda path, **kw: probe)

    dlg = sd.SetupDialog()
    dlg.show_all()
    try:
        deadline = time.time() + 5.0
        while time.time() < deadline and not dlg._python_install_button.get_sensitive():
            _pump(0.1)
        assert dlg._python_install_button.get_sensitive(), "auto-check never ran"
        assert dlg._python_install_button.get_label() == "Reinstall / update sam3gimpd here"
        assert dlg._python_use_button.get_sensitive()
        assert "transformers 5.16.1" in dlg._python_status.get_text()
    finally:
        dlg.destroy()


def test_no_configured_interpreter_means_no_auto_probe(sam3_home, monkeypatch):
    sd = _dialog_module()
    calls = []
    monkeypatch.setattr(sd.bootstrap, "probe_interpreter", lambda path, **kw: calls.append(path) or {"ok": False, "error": "x"})
    dlg = sd.SetupDialog()
    dlg.show_all()
    try:
        _pump(0.5)
        assert calls == []
        assert not dlg._python_install_button.get_sensitive()
    finally:
        dlg.destroy()


def test_idle_timeout_and_remote_daemon_controls_persist(sam3_home):
    import launcher as L
    sd = _dialog_module()
    dlg = sd.SetupDialog()
    try:
        _pump(0.3)
        assert dlg._idle_spin.get_value() == 30.0
        dlg._idle_spin.set_value(10)
        assert L.configured_idle_ttl() == 600.0
        assert "10 min" in dlg._status.get_text()
        dlg._idle_spin.set_value(0)
        assert L.configured_idle_ttl() == 0.0

        assert not dlg._remote_clear.get_sensitive()
        dlg._remote_url.set_text("http://gpu-box:41573")
        dlg._remote_save.clicked()
        assert "token" in dlg._remote_status.get_text().lower()
        assert L.configured_remote() is None
        dlg._remote_token.set_text("abc")
        dlg._remote_save.clicked()
        assert L.configured_remote() == ("gpu-box", 41573, "abc")
        assert dlg._remote_clear.get_sensitive()
        dlg._remote_clear.clicked()
        assert L.configured_remote() is None
        assert dlg._remote_url.get_text() == ""
    finally:
        dlg.destroy()
    # a new dialog sees the saved idle value
    L.set_configured_idle_ttl(45)
    dlg = sd.SetupDialog()
    try:
        assert dlg._idle_spin.get_value() == 45.0
    finally:
        dlg.destroy()


# --------------------------------------------------------------------------- #
# Cancel and Close reach the running job
# --------------------------------------------------------------------------- #
SILENT_S = 8.0


def _silent_runner_class(bs, seen):
    """Every command becomes a child that sleeps without printing -- a uv
    download, as far as the runner can tell -- and each Event the runner is
    handed is recorded."""
    class SilentRunner(bs.CommandRunner):
        def run(self, argv, **kw):
            seen.append(kw.get("cancel"))
            return super().run(
                [sys.executable, "-c", "import time; time.sleep(%s)" % SILENT_S], **kw)
    return SilentRunner


def _managed_install_ready_to_run(bs, monkeypatch):
    monkeypatch.setenv("SAM3_FORCE_DEVICE", "cpu")
    os.makedirs(os.path.dirname(bs.uv_binary()), exist_ok=True)
    open(bs.uv_binary(), "w").close()     # no download: straight to the venv step


def _wait_idle(dlg, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline and dlg._worker.busy:
        _pump(0.05)


def test_cancel_reaches_the_running_install(sam3_home, monkeypatch):
    """The worker replaced its Event after the Installer had been built with
    the old one: Cancel set an Event nobody watched, and a cancelled install
    ran to the end and said "Environment ready."."""
    sd = _dialog_module()
    bs = sd.bootstrap
    _managed_install_ready_to_run(bs, monkeypatch)
    seen = []
    monkeypatch.setattr(bs, "CommandRunner", _silent_runner_class(bs, seen))
    dlg = sd.SetupDialog()
    try:
        _pump(0.2)
        dlg._install_button.clicked()
        _pump(0.5)
        assert dlg._cancel_button.get_sensitive()
        started = time.time()
        dlg._cancel_button.clicked()
        assert seen and seen[0] is dlg._worker.cancel and seen[0].is_set()
        _wait_idle(dlg, SILENT_S + 5)
        assert time.time() - started < SILENT_S / 2, "the silent child was not stopped"
        assert "Cancelled" in dlg._status.get_text()
        assert "ready" not in dlg._status.get_text().lower()
    finally:
        dlg.destroy()


def test_close_stops_the_install_and_nothing_touches_the_dead_window(sam3_home, monkeypatch):
    """After Close the install kept going and its callbacks kept writing into
    destroyed widgets; run_setup then returned while a venv was being built,
    and the main dialog spawned a daemon from it."""
    sd = _dialog_module()
    bs = sd.bootstrap
    _managed_install_ready_to_run(bs, monkeypatch)
    monkeypatch.setattr(bs, "CommandRunner", _silent_runner_class(bs, []))
    late = []
    monkeypatch.setattr(sd.SetupDialog, "_on_install_done",
                        lambda self, *a, **k: late.append(a) or False)
    dlg = sd.SetupDialog()
    _pump(0.2)
    dlg._install_button.clicked()
    _pump(0.5)
    started = time.time()
    dlg.response(sd.Gtk.ResponseType.CLOSE)
    dlg.destroy()                        # what run_setup does on Close
    dlg.wait_for_worker(SILENT_S + 5)    # ...and then this
    assert time.time() - started < SILENT_S / 2, "run_setup would have waited out the step"
    assert not dlg._worker._thread.is_alive()
    _pump(0.5)
    assert late == [], "a callback ran against the destroyed dialog"


# --------------------------------------------------------------------------- #
# the weights download in the user's own environment
# --------------------------------------------------------------------------- #
def test_weights_download_runs_in_the_chosen_environment(sam3_home, monkeypatch, tmp_path):
    """It ran `<managed venv>/python`, which a user of their own environment
    does not have: "Install failed at hf_login ... No such file"."""
    import launcher as L
    sd = _dialog_module()
    bs = sd.bootstrap
    monkeypatch.setenv("SAM3_FORCE_DEVICE", "cpu")
    py = tmp_path / "python"
    py.write_text("")
    L.set_configured_python(str(py), has_daemon=True)
    monkeypatch.setattr(bs, "probe_interpreter", lambda p, **k: dict(SERVING, path=p))
    monkeypatch.setattr(bs, "validate_hf_token", lambda t, **k: {"ok": True, "message": "ok"})
    monkeypatch.setattr(bs, "check_gated_access", lambda t, **k: {"ok": True, "message": "ok"})
    ran = []

    class Recorder(bs.CommandRunner):
        def run(self, argv, **kw):
            ran.append((list(argv), dict(kw.get("env") or {})))
            return bs.CommandResult(list(argv), 0, ["ok"])

    monkeypatch.setattr(bs, "CommandRunner", Recorder)
    errors = []
    monkeypatch.setattr(sd.SetupDialog, "_show_error", lambda self, t, m: errors.append(m))
    bs.save_state(bs.InstallState(fingerprint="the-install", completed=["uv"]))
    with open(bs.state_file()) as fh:
        journal = fh.read()

    dlg = sd.SetupDialog(page="weights")
    try:
        _pump(0.5)
        dlg._token_entry.set_text("hf_" + "a" * 30)
        dlg._on_check_token()
        _wait_idle(dlg, 5)
        assert dlg._download_button.get_sensitive()
        dlg._download_button.clicked()
        _wait_idle(dlg, 5)
        _pump(0.2)
        assert errors == []
        assert [argv[0] for argv, _env in ran] == [str(py), str(py)]
        assert ran[1][0][1:4] == ["-m", "sam3gimpd", "download"]
        assert all(env.get("HF_TOKEN") == "hf_" + "a" * 30 for _argv, env in ran)
        assert "weights downloaded" in dlg._status.get_text()
        with open(bs.state_file()) as fh:
            assert fh.read() == journal, "the install journal was rewritten"
    finally:
        dlg.destroy()


# --------------------------------------------------------------------------- #
# the existing-environment row records the truth
# --------------------------------------------------------------------------- #
def test_install_here_updates_the_recorded_daemon_flag(sam3_home, monkeypatch, tmp_path):
    """Use (no daemon yet), then Install sam3gimpd here: the row said the
    daemon was there, but settings kept python_has_daemon False, so the
    plug-in went on asking for Setup."""
    import launcher as L
    sd = _dialog_module()
    bs = sd.bootstrap
    monkeypatch.setenv("SAM3_FORCE_DEVICE", "cpu")
    py = tmp_path / "python"
    py.write_text("")
    probe = {"p": dict(SERVING, path=str(py), sam3gimpd=None)}
    monkeypatch.setattr(bs, "probe_interpreter", lambda p, **k: probe["p"])
    monkeypatch.setattr(bs, "resolve_daemon_install_command",
                        lambda p, **k: ["uv", "pip", "install"])

    class Installs(bs.CommandRunner):
        def run(self, argv, **kw):
            probe["p"] = dict(SERVING, path=str(py))
            return bs.CommandResult(list(argv), 0, ["ok"])

    monkeypatch.setattr(bs, "CommandRunner", Installs)
    monkeypatch.setattr(sd, "_restart_daemon_for_new_code", lambda: None)
    dlg = sd.SetupDialog()
    try:
        dlg._python_chooser.set_filename(str(py))
        dlg._on_check_python()
        _wait_idle(dlg, 5)
        dlg._on_use_python()
        _pump(0.1)
        assert L.read_settings()["python_has_daemon"] is False
        assert sd.needs_setup()
        dlg._on_install_sam3d_here()
        _wait_idle(dlg, 5)
        _pump(0.1)
        assert L.read_settings()["python_has_daemon"] is True
        assert not sd.needs_setup()
    finally:
        dlg.destroy()


def test_use_on_an_edited_path_checks_that_interpreter(sam3_home, monkeypatch, tmp_path):
    """Env A (with the daemon) was probed; the user pasted env B (without)
    and pressed Use: B was saved with A's verdict, and every spawn failed."""
    import launcher as L
    sd = _dialog_module()
    bs = sd.bootstrap
    monkeypatch.setenv("SAM3_FORCE_DEVICE", "cpu")
    a = tmp_path / "envA-python"
    b = tmp_path / "envB-python"
    a.write_text("")
    b.write_text("")
    L.set_configured_python(str(a), has_daemon=True)
    monkeypatch.setattr(bs, "probe_interpreter", lambda p, **k: dict(
        SERVING, path=p, sam3gimpd=SERVING["sam3gimpd"] if p == str(a) else None))
    dlg = sd.SetupDialog()
    try:
        _pump(0.3)
        _wait_idle(dlg, 5)
        assert dlg._python_use_button.get_sensitive()
        dlg._python_chooser.set_filename(str(b))
        assert not dlg._python_use_button.get_sensitive(), "the verdict was about env A"
        dlg._on_use_python()           # pressed anyway (or via the keyboard)
        _wait_idle(dlg, 5)
        _pump(0.2)
        settings = L.read_settings()
        assert settings["python"] == str(b)
        assert settings["python_has_daemon"] is False
        assert not bs.inspect_environment(accelerator=sd._UNPROBED).env_ready
    finally:
        dlg.destroy()


def test_install_here_explains_a_missing_daemon_source(sam3_home, monkeypatch, tmp_path):
    import launcher as L
    sd = _dialog_module()
    bs = sd.bootstrap
    monkeypatch.setenv("SAM3_FORCE_DEVICE", "cpu")
    py = tmp_path / "python"
    py.write_text("")
    L.set_configured_python(str(py), has_daemon=True)
    monkeypatch.setattr(bs, "probe_interpreter", lambda p, **k: dict(SERVING, path=p))
    dlg = sd.SetupDialog()
    try:
        _pump(0.3)
        _wait_idle(dlg, 5)

        def missing(env=None):
            raise bs.DaemonSourceMissing("This copy of the plug-in is incomplete")

        monkeypatch.setattr(bs, "daemon_source", missing)
        dlg._on_install_sam3d_here()
        assert "incomplete" in dlg._python_status.get_text()
        assert not dlg._worker.busy
    finally:
        dlg.destroy()


# --------------------------------------------------------------------------- #
# Doctor
# --------------------------------------------------------------------------- #
def test_opening_on_the_doctor_tab_runs_doctor_after_the_row_probe(sam3_home, monkeypatch, tmp_path):
    """The row's automatic probe held the worker, and Doctor -- the reason the
    tab was opened -- was silently dropped."""
    import launcher as L
    sd = _dialog_module()
    bs = sd.bootstrap
    monkeypatch.setenv("SAM3_FORCE_DEVICE", "cpu")
    py = tmp_path / "python"
    py.write_text("")
    L.set_configured_python(str(py), has_daemon=True)

    def slow_probe(p, **k):
        time.sleep(0.5)
        return dict(SERVING, path=p)

    monkeypatch.setattr(bs, "probe_interpreter", slow_probe)
    ran = []
    monkeypatch.setattr(bs, "run_doctor", lambda **k: ran.append(1) or {"local": {"summary": "x"}})
    dlg = sd.SetupDialog(page="doctor")
    try:
        deadline = time.time() + 5
        while time.time() < deadline and not ran:
            _pump(0.05)
        _wait_idle(dlg, 5)
        _pump(0.1)
        assert ran == [1]
        assert "Doctor finished" in dlg._status.get_text()
    finally:
        dlg.destroy()


# --------------------------------------------------------------------------- #
# the remote daemon is plain HTTP
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("url,off", [
    ("http://gpu-box:41573", True),
    ("192.168.1.20:8765", True),
    ("http://127.0.0.1:8765", False),
    ("http://localhost:8765", False),
    ("http://[::1]:8765", False),
    ("", False),
])
def test_remote_is_off_machine(url, off):
    assert _dialog_module().remote_is_off_machine(url) is off


def test_a_remote_address_off_this_machine_is_warned_about(sam3_home):
    """The token and every image go over plain HTTP; say so, and say how to
    tunnel instead."""
    import launcher as L
    sd = _dialog_module()
    dlg = sd.SetupDialog()
    try:
        _pump(0.2)
        assert not dlg._remote_warning.get_visible()
        dlg._remote_url.set_text("http://gpu-box:41573")
        assert dlg._remote_warning.get_visible()
        text = dlg._remote_warning.get_text()
        assert "ssh -L 8765:127.0.0.1:8765" in text and "http://127.0.0.1:8765" in text
        dlg._remote_url.set_text("http://127.0.0.1:8765")
        assert not dlg._remote_warning.get_visible()
    finally:
        dlg.destroy()
    # A saved off-machine address is warned about as soon as Setup opens.
    L.set_configured_remote("http://gpu-box:41573", "tok")
    dlg = sd.SetupDialog()
    try:
        assert dlg._remote_warning.get_visible()
    finally:
        dlg.destroy()
