#!/usr/bin/env python3
"""Standalone GTK harness for the sam3-gimp preview canvas -- **no GIMP required**.

This is how :class:`ui.canvas.Sam3Canvas` is developed, demoed and regression-eyeballed
on a machine with GTK3, PyGObject and a display but **no GPU, no torch and no
SAM 3 weights** (``DESIGN.md`` §5 and §12).  It drives the *real* widget -- not a
mock -- through the *real* wire format of ``plugin/sam3_gimp/_daemon/API.md``.

Usage
-----
::

    tools/canvas_harness.py                       # synthetic demo image, offline masks
    tools/canvas_harness.py photo.png             # load a real image
    tools/canvas_harness.py photo.png --spawn     # spawn `sam3gimpd serve --stub` and use it
    tools/canvas_harness.py photo.png --offline   # never touch a daemon

Three back ends, chosen automatically unless forced:

``daemon``
    An already-running ``sam3gimpd`` found through ``runtime.json`` (API §3).
``spawn``
    Start ``sam3gimpd serve --stub`` ourselves, detached, and talk to it.
``offline``
    Synthesise conforming result frames in-process.  The frames are built with
    the same ``SAM3RES`` layout, the same soft uint8 mask semantics and the same
    coordinate spaces as the daemon's, so every client-side code path -- decode,
    affine inversion, thresholding, hit-testing, ants -- is exercised for real.
    This is what makes the canvas developable before ``sam3gimpd`` exists.

Only the standard library, ``gi`` and ``cairo`` are used, so the harness doubles
as a check that the canvas has not grown a forbidden dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import http.client

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN_PKG = os.path.join(REPO_ROOT, "plugin", "sam3_gimp")
DAEMON_SRC = os.path.join(PLUGIN_PKG, "_daemon")
if PLUGIN_PKG not in sys.path:
    sys.path.insert(0, PLUGIN_PKG)

import gi  # noqa: E402

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gdk, GdkPixbuf, GLib, Gtk  # noqa: E402

from ui import canvas as canvas_mod  # noqa: E402
from ui.canvas import Sam3Canvas  # noqa: E402

MAX_SIDE = 1008          # API §15: neither uploaded side may exceed this
MODEL_CANVAS = 1008      # the reference processor's square canvas
API_VERSION = "1.0"


# --------------------------------------------------------------------------- #
# image loading  (GdkPixbuf is part of gi, so this stays inside the rules)
# --------------------------------------------------------------------------- #
def pixbuf_to_rgb(pixbuf):
    """``GdkPixbuf`` -> packed RGB u8 with stride ``w*3`` (the API §7 layout).

    Alpha is composited onto mid grey rather than dropped, because the daemon
    takes no alpha and a straight drop would make cut-outs look like haloes.
    """
    width = pixbuf.get_width()
    height = pixbuf.get_height()
    stride = pixbuf.get_rowstride()
    channels = pixbuf.get_n_channels()
    raw = pixbuf.get_pixels()
    out = bytearray(width * height * 3)
    row_len = width * channels
    for y in range(height):
        row = raw[y * stride:y * stride + row_len]
        dst = y * width * 3
        if channels == 3:
            out[dst:dst + width * 3] = row
        else:
            rgb = bytearray(width * 3)
            rgb[0::3] = row[0::4]
            rgb[1::3] = row[1::4]
            rgb[2::3] = row[2::4]
            alpha = row[3::4]
            for x in range(width):
                a = alpha[x]
                if a == 255:
                    continue
                inv = 255 - a
                base = x * 3
                for c in range(3):
                    rgb[base + c] = (rgb[base + c] * a + 128 * inv) // 255
            out[dst:dst + width * 3] = rgb
    return bytes(out)


def load_image(path, max_side=MAX_SIDE):
    """Load any GdkPixbuf-supported file and downscale to the API's limit.

    The *client* owns this downscale (API §5): the daemon rejects oversize
    uploads rather than silently resizing, precisely so the client's own
    coordinate bookkeeping cannot desynchronise.
    """
    pixbuf = GdkPixbuf.Pixbuf.new_from_file(path)
    w, h = pixbuf.get_width(), pixbuf.get_height()
    scale = min(1.0, float(max_side) / max(w, h))
    if scale < 1.0:
        w = max(16, int(round(w * scale)))
        h = max(16, int(round(h * scale)))
        pixbuf = pixbuf.scale_simple(w, h, GdkPixbuf.InterpType.BILINEAR)
    return pixbuf_to_rgb(pixbuf), pixbuf.get_width(), pixbuf.get_height()


def synthetic_image(width=640, height=440):
    """A deterministic stand-in image so the harness runs with no arguments."""
    out = bytearray(width * height * 3)
    i = 0
    for y in range(height):
        for x in range(width):
            u = x / float(width - 1)
            v = y / float(height - 1)
            r = int(40 + 120 * u)
            g = int(50 + 90 * v)
            b = int(90 + 100 * (1.0 - u) * (1.0 - v))
            if ((x // 40) + (y // 40)) % 2 == 0:
                r = min(255, r + 18)
                g = min(255, g + 18)
                b = min(255, b + 18)
            for (cx, cy, rad, tint) in (
                (int(width * 0.28), int(height * 0.38), 78, (230, 96, 70)),
                (int(width * 0.68), int(height * 0.30), 54, (96, 210, 130)),
                (int(width * 0.52), int(height * 0.74), 66, (250, 210, 90)),
            ):
                d = math.hypot(x - cx, y - cy) / float(rad)
                if d < 1.0:
                    k = min(1.0, (1.0 - d) * 3.0)
                    r = int(r * (1 - k) + tint[0] * k)
                    g = int(g * (1 - k) + tint[1] * k)
                    b = int(b * (1 - k) + tint[2] * k)
            out[i] = r
            out[i + 1] = g
            out[i + 2] = b
            i += 3
    return bytes(out), width, height


# --------------------------------------------------------------------------- #
# offline engine -- conforming frames without a daemon
# --------------------------------------------------------------------------- #
def _soft_ellipse(canvas_w, canvas_h, cx, cy, rx, ry, feather=0.07):
    """A cropped soft uint8 mask for an ellipse, in **model-canvas** space.

    The falloff is built so that the value is exactly ``128`` on the ellipse
    boundary -- API §8.3 defines ``128`` as logit 0, the model's own binarisation
    point -- which means the default threshold reproduces the nominal shape and
    dragging the slider visibly grows or shrinks it.  A flat 255 rectangle would
    hide every client-side thresholding bug, so it is deliberately not that.
    """
    rx = max(2.0, float(rx))
    ry = max(2.0, float(ry))
    pad = 1.0 + feather
    x0 = max(0, int(math.floor(cx - rx * pad)))
    y0 = max(0, int(math.floor(cy - ry * pad)))
    x1 = min(canvas_w, int(math.ceil(cx + rx * pad)))
    y1 = min(canvas_h, int(math.ceil(cy + ry * pad)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    mw = x1 - x0
    mh = y1 - y0
    mask = bytearray(mw * mh)
    i = 0
    for y in range(y0, y1):
        dy = (y + 0.5 - cy) / ry
        dy2 = dy * dy
        for x in range(x0, x1):
            dx = (x + 0.5 - cx) / rx
            d = math.sqrt(dx * dx + dy2)
            t = (1.0 - d) / feather
            if t <= -1.0:
                v = 0
            elif t >= 1.0:
                v = 255
            else:
                v = int(round(127.5 + 127.5 * t))
            mask[i] = v
            i += 1
    return (x0, y0, x1, y1), bytes(mask)


class OfflineEngine:
    """Synthesises ``SAM3RES`` frames locally, mirroring the ``--stub`` guarantees.

    Deterministic (same prompt -> byte-identical frame), soft-edged, scored in
    ``[0.1, 1.0]`` descending -- see ``API.md`` §14.  It exists so the
    canvas is fully drivable without ``sam3gimpd`` installed at all.
    """

    def __init__(self, image_w, image_h, canvas_side=MODEL_CANVAS):
        self.image_w = image_w
        self.image_h = image_h
        self.canvas_w = canvas_side
        self.canvas_h = canvas_side
        self.affine = {
            "scale_x": canvas_side / float(image_w),
            "scale_y": canvas_side / float(image_h),
            "offset_x": 0.0,
            "offset_y": 0.0,
        }
        self._job = 0

    # -- helpers ----------------------------------------------------------- #
    def _base_header(self, engine, prompt, request_id):
        self._job += 1
        return {
            "api_version": API_VERSION,
            "job_id": "j-offline-%04d" % self._job,
            "request_id": request_id,
            "image_id": "offline",
            "engine": engine,
            "state": "done",
            "prompt": prompt,
            "image": {"width": self.image_w, "height": self.image_h},
            "model_canvas": {"width": self.canvas_w, "height": self.canvas_h},
            "canvas_from_image": dict(self.affine),
            "mask_encoding": "u8_soft",
            "elapsed_ms": 0.0,
            "truncated": False,
            "instances": [],
            "blob_length": 0,
        }

    def _pack(self, header, entries, blobs):
        header["instances"] = entries
        return canvas_mod.encode_frame(header, blobs)

    # -- PCS --------------------------------------------------------------- #
    def text(self, text, request_id, max_instances=8):
        digest = hashlib.blake2b(text.strip().lower().encode("utf-8"), digest_size=16).digest()
        count = 1 + digest[0] % min(5, max(1, max_instances))
        header = self._base_header(
            "pcs", {"kind": "text", "text": text, "score_threshold": 0.1}, request_id
        )
        entries, blobs = [], []
        for n in range(count):
            b = digest[(n * 3) % 13:(n * 3) % 13 + 3] or b"\x40\x40\x40"
            cx = (0.12 + 0.76 * (b[0] / 255.0)) * self.canvas_w
            cy = (0.12 + 0.76 * (b[1] / 255.0)) * self.canvas_h
            rx = (0.06 + 0.13 * (b[2] / 255.0)) * self.canvas_w
            ry = rx * (0.6 + 0.8 * ((b[0] ^ b[1]) / 255.0))
            made = _soft_ellipse(self.canvas_w, self.canvas_h, cx, cy, rx, ry)
            if made is None:
                continue
            bbox, mask = made
            entries.append(
                {
                    "instance_id": len(entries),
                    "score": round(0.95 - 0.11 * len(entries), 3),
                    "label": text,
                    "bbox": list(bbox),
                    "mask_width": bbox[2] - bbox[0],
                    "mask_height": bbox[3] - bbox[1],
                    "blob_offset": 0,
                    "blob_length": 0,
                }
            )
            blobs.append(mask)
        return self._pack(header, entries, blobs)

    # -- PVS --------------------------------------------------------------- #
    def points(self, points, request_id, box=None, multimask=True):
        """``points`` are ``(x, y, label)`` in **uploaded-image** space (API §5)."""
        header = self._base_header(
            "pvs",
            {"kind": "points", "points": len(points), "multimask": bool(multimask)},
            request_id,
        )
        positives = [p for p in points if p[2]] or list(points)
        if not positives:
            return self._pack(header, [], [])
        sx = self.affine["scale_x"]
        sy = self.affine["scale_y"]
        cxs = [p[0] * sx for p in positives]
        cys = [p[1] * sy for p in positives]
        cx = sum(cxs) / len(cxs)
        cy = sum(cys) / len(cys)
        spread_x = max(24.0, (max(cxs) - min(cxs)) * 0.8 + 60.0)
        spread_y = max(24.0, (max(cys) - min(cys)) * 0.8 + 60.0)
        # Negative points push the blob away from themselves, so a right-click
        # visibly changes the result -- the point of exercising PVS at all.
        for (px, py, label) in points:
            if label:
                continue
            dx = cx - px * sx
            dy = cy - py * sy
            dist = math.hypot(dx, dy) or 1.0
            if dist < max(spread_x, spread_y):
                spread_x = max(16.0, min(spread_x, dist * 0.85))
                spread_y = max(16.0, min(spread_y, dist * 0.85))
        entries, blobs = [], []
        variants = ((1.0, 0.9), (0.72, 0.62), (1.35, 1.2)) if multimask else ((1.0, 0.9),)
        for n, (fx, fy) in enumerate(variants):
            made = _soft_ellipse(
                self.canvas_w, self.canvas_h, cx, cy, spread_x * fx, spread_y * fy
            )
            if made is None:
                continue
            bbox, mask = made
            entries.append(
                {
                    "instance_id": len(entries),
                    "score": round(0.92 - 0.2 * n, 3),
                    "label": "",
                    "bbox": list(bbox),
                    "mask_width": bbox[2] - bbox[0],
                    "mask_height": bbox[3] - bbox[1],
                    "blob_offset": 0,
                    "blob_length": 0,
                }
            )
            blobs.append(mask)
        return self._pack(header, entries, blobs)


# --------------------------------------------------------------------------- #
# daemon client -- stdlib HTTP, exactly what plugin/client.py will do
# --------------------------------------------------------------------------- #
def runtime_file_path():
    """``runtime.json`` per API §3.1, honouring both documented overrides."""
    override = os.environ.get("SAM3D_RUNTIME_FILE")
    if override:
        return override
    base = os.environ.get("SAM3_GIMP_HOME")
    if not base:
        if sys.platform == "win32":
            root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
            base = os.path.join(root, "sam3-gimp")
        elif sys.platform == "darwin":
            base = os.path.expanduser("~/Library/Application Support/sam3-gimp")
        else:
            root = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
            base = os.path.join(root, "sam3-gimp")
    return os.path.join(base, "runtime.json")


class DaemonError(Exception):
    pass


class DaemonClient:
    """Minimal, synchronous ``sam3gimpd`` client for the harness worker thread."""

    def __init__(self, host, port, token, timeout=30.0):
        self.host = host
        self.port = int(port)
        self.token = token
        self.timeout = timeout

    # -- discovery --------------------------------------------------------- #
    @classmethod
    def from_runtime_file(cls, path=None):
        path = path or runtime_file_path()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                info = json.load(fh)
        except Exception:
            return None
        if not all(k in info for k in ("port", "token", "pid", "version", "started_at")):
            return None
        return cls(info.get("host", "127.0.0.1"), info["port"], info["token"])

    # -- transport --------------------------------------------------------- #
    def _request(self, method, path, body=None, headers=None, timeout=None):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout or self.timeout)
        try:
            hdrs = {"Authorization": "Bearer " + self.token}
            if headers:
                hdrs.update(headers)
            if body is not None:
                hdrs.setdefault("Content-Length", str(len(body)))
            conn.request(method, path, body=body, headers=hdrs)
            resp = conn.getresponse()
            payload = resp.read()
            ctype = resp.getheader("Content-Type", "")
            if resp.status >= 300:
                try:
                    err = json.loads(payload.decode("utf-8"))["error"]
                    raise DaemonError("%s: %s" % (err.get("code"), err.get("message")))
                except (ValueError, KeyError, TypeError):
                    raise DaemonError("HTTP %d on %s" % (resp.status, path))
            return resp.status, ctype, payload
        finally:
            conn.close()

    def _json(self, method, path, obj=None, timeout=None):
        body = None
        headers = {}
        if obj is not None:
            body = json.dumps(obj).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        _status, _ctype, payload = self._request(method, path, body, headers, timeout)
        return json.loads(payload.decode("utf-8")) if payload else {}

    # -- endpoints --------------------------------------------------------- #
    def hello(self):
        return self._json("GET", "/hello", timeout=3.0)

    def upload(self, rgb, width, height):
        _s, _c, payload = self._request(
            "POST",
            "/images",
            rgb,
            {
                "Content-Type": "application/octet-stream",
                "X-Width": str(width),
                "X-Height": str(height),
            },
            timeout=60.0,
        )
        return json.loads(payload.decode("utf-8"))

    def prompt_text(self, image_id, request_id, text, max_instances=32):
        return self._json(
            "POST",
            "/images/%s/text" % image_id,
            {
                "request_id": request_id,
                "text": text,
                "score_threshold": 0.1,
                "max_instances": max_instances,
            },
        )

    def prompt_points(self, image_id, request_id, points, multimask=True):
        return self._json(
            "POST",
            "/images/%s/points" % image_id,
            {
                "request_id": request_id,
                "points": [{"x": p[0], "y": p[1], "label": p[2]} for p in points],
                "multimask": multimask,
                "max_instances": 3,
            },
        )

    def job(self, job_id, wait=5.0):
        """Long-poll one job.  Returns ``("json", dict)`` or ``("frame", bytes)``."""
        _s, ctype, payload = self._request(
            "GET", "/jobs/%s?wait=%.1f" % (job_id, wait), timeout=wait + 15.0
        )
        if ctype.startswith("application/vnd.sam3.result"):
            return "frame", payload
        return "json", json.loads(payload.decode("utf-8"))


def spawn_daemon(extra_args=(), timeout=25.0):
    """Start ``sam3gimpd serve --stub`` detached and wait for ``runtime.json`` (API §13)."""
    path = runtime_file_path()
    t0 = time.time()
    try:
        os.remove(path)
    except OSError:
        pass
    log_dir = os.path.join(os.path.dirname(path), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "sam3gimpd-harness.log")
    # Three ways to reach the daemon, in decreasing order of "properly installed":
    # the console script, ``-m sam3gimpd``, and finally its documented CLI entry
    # point.  The checkout's ``plugin/sam3_gimp/_daemon`` is put on the child's
    # PYTHONPATH, so the last two work straight from a source tree with nothing
    # installed.  A candidate that dies immediately (missing package, no
    # ``__main__``) is skipped rather than waited on for the full timeout.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (DAEMON_SRC, env.get("PYTHONPATH", "")) if p
    )
    candidates = (
        ["sam3gimpd"],
        [sys.executable, "-m", "sam3gimpd"],
        [sys.executable, "-c", "from sam3gimpd.cli import main; raise SystemExit(main())"],
    )
    started = None
    for candidate in candidates:
        argv = candidate + ["serve", "--stub", "--parent-pid", str(os.getpid())] + list(extra_args)
        try:
            with open(log_path, "ab") as log:
                proc = subprocess.Popen(
                    argv,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    close_fds=True,
                    env=env,
                )
        except (OSError, ValueError):
            continue
        time.sleep(0.5)
        if proc.poll() not in (None, 0):
            continue
        started = argv
        break
    if started is None:
        raise DaemonError(
            "could not start sam3gimpd (tried `sam3gimpd`, `python -m sam3gimpd`, `sam3gimpd.cli:main`) - see %s"
            % log_path
        )
    deadline = time.time() + timeout
    while time.time() < deadline:
        client = DaemonClient.from_runtime_file(path)
        if client is not None and os.path.getmtime(path) >= t0 - 1.0:
            try:
                client.hello()
                return client
            except Exception:
                pass
        time.sleep(0.1)
    raise DaemonError("sam3gimpd did not publish %s within %.0fs (see %s)" % (path, timeout, log_path))


# --------------------------------------------------------------------------- #
# the application window
# --------------------------------------------------------------------------- #
class HarnessWindow(Gtk.Window):
    """Canvas on the left, the controls a real dialog would have on the right.

    The threading discipline is the one ``DESIGN.md`` §4 mandates for the actual
    plug-in and is worth rehearsing here: **the GTK main loop never blocks on
    HTTP**.  Every daemon call runs on a worker thread and comes back through
    ``GLib.idle_add``.
    """

    def __init__(self, rgb, width, height, backend, client=None, title="sam3 canvas harness"):
        Gtk.Window.__init__(self, title=title)
        self.set_default_size(1180, 760)
        self.connect("destroy", self._on_destroy)

        self._rgb = rgb
        self._image_w = width
        self._image_h = height
        self._backend = backend           # "offline" | "daemon"
        self._client = client
        self._offline = OfflineEngine(width, height)
        self._image_id = None
        self._request_seq = 0
        self._latest_request_id = None
        self._stopping = False

        self._work_q = queue.Queue()
        self._worker = threading.Thread(target=self._worker_loop, name="sam3-harness", daemon=True)
        self._worker.start()

        self.canvas = Sam3Canvas()
        self.canvas.set_image(rgb, width, height)
        self.canvas.set_status_text("%dx%d  -  backend: %s" % (width, height, backend))
        self.canvas.connect("point-added", self._on_point_added)
        self.canvas.connect("points-changed", self._on_points_changed)
        self.canvas.connect("instance-toggled", self._on_instance_toggled)
        self.canvas.connect("instance-visibility-changed", self._on_instance_visibility)
        self.canvas.connect("instance-activated", self._on_instance_activated)
        self.canvas.connect("instance-hovered", self._on_instance_hovered)
        self.canvas.connect("view-changed", self._on_view_changed)
        self.canvas.connect("box-drawn", self._on_box_drawn)
        self.canvas.connect("threshold-changed", self._on_threshold_changed)

        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        frame = Gtk.Frame()
        frame.add(self.canvas)
        paned.pack1(frame, True, False)
        paned.pack2(self._build_sidebar(), False, False)
        paned.set_position(860)
        self.add(paned)

        if backend == "daemon":
            self._submit(self._task_upload)

    # ------------------------------------------------------------------ #
    # sidebar
    # ------------------------------------------------------------------ #
    def _build_sidebar(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.set_border_width(10)
        box.set_size_request(300, -1)

        # -- mode --
        box.pack_start(self._heading("Interaction"), False, False, 0)
        modes = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self._rb_select = Gtk.RadioButton.new_with_label_from_widget(None, "Select instances")
        self._rb_points = Gtk.RadioButton.new_with_label_from_widget(self._rb_select, "Place points")
        self._rb_select.connect("toggled", self._on_mode_toggled, "select")
        self._rb_points.connect("toggled", self._on_mode_toggled, "points")
        modes.pack_start(self._rb_select, False, False, 0)
        modes.pack_start(self._rb_points, False, False, 0)
        box.pack_start(modes, False, False, 0)

        # -- PCS --
        box.pack_start(self._heading("Text prompt (PCS)"), False, False, 0)
        hint = Gtk.Label(label="A simple noun phrase - “red car” - not\na relational description.")
        hint.set_xalign(0.0)
        hint.get_style_context().add_class("dim-label")
        box.pack_start(hint, False, False, 0)
        prompt_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self._entry = Gtk.Entry()
        self._entry.set_placeholder_text("yellow school bus")
        self._entry.set_activates_default(False)
        self._entry.connect("activate", lambda *_a: self._run_text())
        btn = Gtk.Button(label="Segment")
        btn.connect("clicked", lambda *_a: self._run_text())
        prompt_row.pack_start(self._entry, True, True, 0)
        prompt_row.pack_start(btn, False, False, 0)
        box.pack_start(prompt_row, False, False, 0)

        # -- PVS --
        box.pack_start(self._heading("Point prompt (PVS)"), False, False, 0)
        pvs = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        b_refine = Gtk.Button(label="Refine")
        b_refine.connect("clicked", lambda *_a: self._run_points())
        b_clear = Gtk.Button(label="Clear points")
        b_clear.connect("clicked", lambda *_a: self.canvas.clear_points())
        pvs.pack_start(b_refine, True, True, 0)
        pvs.pack_start(b_clear, True, True, 0)
        box.pack_start(pvs, False, False, 0)
        self._points_label = Gtk.Label(label="0 points")
        self._points_label.set_xalign(0.0)
        box.pack_start(self._points_label, False, False, 0)

        # -- local filters --
        box.pack_start(self._heading("Local filters (no round trip)"), False, False, 0)
        self._s_mask = self._slider(box, "Mask threshold", 1, 255, 1,
                                    canvas_mod.DEFAULT_MASK_THRESHOLD, self._on_mask_slider)
        self._s_score = self._slider(box, "Score threshold", 0.0, 1.0, 0.01,
                                     0.1, self._on_score_slider, digits=2)
        self._s_opacity = self._slider(box, "Overlay opacity", 0.0, 1.0, 0.01,
                                       0.55, self._on_opacity_slider, digits=2)

        # -- view --
        view_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        for (label, cb) in (
            ("Fit", lambda *_a: self.canvas.zoom_fit()),
            ("100%", lambda *_a: self.canvas.zoom_to(1.0)),
        ):
            b = Gtk.Button(label=label)
            b.connect("clicked", cb)
            view_row.pack_start(b, True, True, 0)
        ants = Gtk.CheckButton(label="Ants")
        ants.set_active(True)
        ants.connect("toggled", lambda w: self.canvas.set_show_ants(w.get_active()))
        view_row.pack_start(ants, False, False, 0)
        box.pack_start(view_row, False, False, 0)

        # -- instances --
        box.pack_start(self._heading("Instances"), False, False, 0)
        self._list = Gtk.ListBox()
        self._list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self._list.connect("row-selected", self._on_row_selected)
        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroller.add(self._list)
        box.pack_start(scroller, True, True, 0)

        self._status = Gtk.Label(label="ready")
        self._status.set_xalign(0.0)
        self._status.set_line_wrap(True)
        box.pack_start(self._status, False, False, 0)
        return box

    @staticmethod
    def _heading(text):
        label = Gtk.Label()
        label.set_markup("<b>%s</b>" % GLib.markup_escape_text(text))
        label.set_xalign(0.0)
        label.set_margin_top(6)
        return label

    @staticmethod
    def _slider(box, caption, lo, hi, step, value, handler, digits=0):
        label = Gtk.Label(label=caption)
        label.set_xalign(0.0)
        box.pack_start(label, False, False, 0)
        scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, lo, hi, step)
        scale.set_digits(digits)
        scale.set_value(value)
        scale.set_draw_value(True)
        scale.connect("value-changed", handler)
        box.pack_start(scale, False, False, 0)
        return scale

    # ------------------------------------------------------------------ #
    # worker thread
    # ------------------------------------------------------------------ #
    def _submit(self, fn, *args):
        self._work_q.put((fn, args))

    def _worker_loop(self):
        while True:
            item = self._work_q.get()
            if item is None:
                return
            fn, args = item
            try:
                fn(*args)
            except Exception as exc:  # keep the UI alive whatever the daemon does
                GLib.idle_add(self._set_status, "error: %s" % (exc,))

    def _task_upload(self):
        info = self._client.upload(self._rgb, self._image_w, self._image_h)
        self._image_id = info["image_id"]
        GLib.idle_add(
            self._set_status,
            "uploaded %s (cached=%s)" % (info["image_id"][:12], info.get("cached")),
        )

    def _task_prompt(self, kind, request_id, payload):
        if self._image_id is None:
            self._task_upload()
        if kind == "text":
            accepted = self._client.prompt_text(self._image_id, request_id, payload)
        else:
            accepted = self._client.prompt_points(self._image_id, request_id, payload)
        job_id = accepted["job_id"]
        deadline = time.time() + 120.0
        while time.time() < deadline and not self._stopping:
            kind_, body = self._client.job(job_id, wait=5.0)
            if kind_ == "frame":
                GLib.idle_add(self._deliver_frame, request_id, body)
                return
            state = body.get("state")
            if state in ("failed", "superseded", "cancelled"):
                GLib.idle_add(self._set_status, "job %s: %s" % (job_id, state))
                return
            GLib.idle_add(
                self._set_progress, body.get("progress", 0.0), body.get("stage", state or "")
            )
        GLib.idle_add(self._set_status, "job %s timed out" % job_id)

    def _task_offline(self, kind, request_id, payload):
        time.sleep(0.12)  # a visible, honest hint that inference is not free
        if kind == "text":
            frame = self._offline.text(payload, request_id)
        else:
            frame = self._offline.points(payload, request_id)
        GLib.idle_add(self._deliver_frame, request_id, frame)

    # ------------------------------------------------------------------ #
    # prompting
    # ------------------------------------------------------------------ #
    def _next_request_id(self):
        self._request_seq += 1
        self._latest_request_id = "r-%06d" % self._request_seq
        return self._latest_request_id

    def _run_text(self):
        text = self._entry.get_text().strip()
        if not text:
            self._set_status("type a noun phrase first")
            return
        rid = self._next_request_id()
        self.canvas.set_busy(True, "prompting", 0.1)
        if self._backend == "daemon":
            self._submit(self._task_prompt, "text", rid, text)
        else:
            self._submit(self._task_offline, "text", rid, text)

    def _run_points(self):
        points = self.canvas.points
        if not points:
            self._set_status("click the image to place points first")
            return
        rid = self._next_request_id()
        self.canvas.set_busy(True, "prompting", 0.1)
        if self._backend == "daemon":
            self._submit(self._task_prompt, "points", rid, points)
        else:
            self._submit(self._task_offline, "points", rid, points)

    # ------------------------------------------------------------------ #
    # main-thread callbacks
    # ------------------------------------------------------------------ #
    def _deliver_frame(self, request_id, payload):
        # API §10: drop any result whose request_id is not the latest issued.
        if request_id != self._latest_request_id:
            self._set_status("dropped stale result %s" % request_id)
            return False
        self.canvas.set_busy(False)
        try:
            header = self.canvas.set_result_frame(payload)
        except ValueError as exc:
            self._set_status("bad frame: %s" % exc)
            return False
        self._rebuild_list()
        self._set_status(
            "%d instance(s), engine=%s, %.0f ms"
            % (len(header.get("instances", [])), header.get("engine"), header.get("elapsed_ms", 0.0))
        )
        return False

    def _set_status(self, text):
        self._status.set_text(text)
        return False

    def _set_progress(self, progress, stage):
        self.canvas.set_busy(True, stage, progress)
        return False

    def _rebuild_list(self):
        for child in self._list.get_children():
            self._list.remove(child)
        for inst in self.canvas.instances:
            self._list.add(self._instance_row(inst))
        self._list.show_all()

    def _instance_row(self, inst):
        row = Gtk.ListBoxRow()
        row.instance_id = inst.instance_id
        line = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        line.set_border_width(4)

        check = Gtk.CheckButton()
        check.set_active(inst.visible)
        check.connect("toggled", self._on_visible_check, inst.instance_id)
        line.pack_start(check, False, False, 0)

        swatch = Gtk.DrawingArea()
        swatch.set_size_request(14, 14)
        swatch.connect("draw", self._draw_swatch, inst.color)
        line.pack_start(swatch, False, False, 0)

        label = Gtk.Label(label="#%d  %.3f  %s" % (inst.instance_id, inst.score, inst.label or "-"))
        label.set_xalign(0.0)
        label.set_ellipsize(3)  # Pango.EllipsizeMode.END
        line.pack_start(label, True, True, 0)
        row.add(line)
        return row

    @staticmethod
    def _draw_swatch(widget, cr, color):
        alloc = widget.get_allocation()
        cr.set_source_rgb(*color)
        cr.rectangle(0, 0, alloc.width, alloc.height)
        cr.fill()
        cr.set_source_rgba(0, 0, 0, 0.5)
        cr.set_line_width(1)
        cr.rectangle(0.5, 0.5, alloc.width - 1, alloc.height - 1)
        cr.stroke()
        return False

    # ------------------------------------------------------------------ #
    # widget signal handlers
    # ------------------------------------------------------------------ #
    def _on_mode_toggled(self, button, mode):
        if button.get_active():
            self.canvas.set_interaction_mode(mode)

    def _on_mask_slider(self, scale):
        self.canvas.set_mask_threshold(int(scale.get_value()))

    def _on_score_slider(self, scale):
        self.canvas.set_score_threshold(scale.get_value())

    def _on_opacity_slider(self, scale):
        self.canvas.set_overlay_opacity(scale.get_value())

    def _on_threshold_changed(self, _canvas, mask_threshold, score_threshold):
        self._s_mask.set_value(mask_threshold)
        self._s_score.set_value(score_threshold)

    def _on_visible_check(self, check, instance_id):
        self.canvas.set_instance_visible(instance_id, check.get_active(), notify=False)

    def _on_row_selected(self, _list, row):
        if row is not None:
            self.canvas.set_active_instance(row.instance_id)

    def _on_point_added(self, _canvas, x, y, label):
        self._set_status("point %s at (%.1f, %.1f) [uploaded-image px]"
                         % ("+" if label else "-", x, y))

    def _on_points_changed(self, _canvas):
        self._points_label.set_text("%d point(s)" % len(self.canvas.points))

    def _on_instance_toggled(self, _canvas, instance_id, selected):
        self._set_status("instance %d %s" % (instance_id, "included" if selected else "excluded"))

    def _on_instance_visibility(self, _canvas, instance_id, visible):
        for row in self._list.get_children():
            if getattr(row, "instance_id", None) == instance_id:
                check = row.get_child().get_children()[0]
                check.set_active(visible)

    def _on_instance_activated(self, _canvas, instance_id):
        for row in self._list.get_children():
            if getattr(row, "instance_id", None) == instance_id:
                if self._list.get_selected_row() is not row:
                    self._list.select_row(row)
                return

    def _on_instance_hovered(self, _canvas, instance_id):
        if instance_id >= 0:
            inst = self.canvas.get_instance(instance_id)
            if inst is not None:
                self.canvas.set_status_text("#%d  score %.3f  %s"
                                            % (inst.instance_id, inst.score, inst.label or ""))
                return
        self.canvas.set_status_text("%dx%d  -  backend: %s"
                                    % (self._image_w, self._image_h, self._backend))

    def _on_view_changed(self, _canvas, zoom, _ox, _oy):
        self._set_status("zoom %.0f%%" % (zoom * 100.0))

    def _on_box_drawn(self, _canvas, x0, y0, x1, y1):
        self._set_status("box [%.0f, %.0f, %.0f, %.0f] (uploaded-image px)" % (x0, y0, x1, y1))

    def _on_destroy(self, *_args):
        self._stopping = True
        self._work_q.put(None)
        if Gtk.main_level() > 0:
            Gtk.main_quit()


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def choose_backend(mode):
    """Resolve ``--backend`` into ``(name, client)``, falling back gracefully."""
    if mode == "offline":
        return "offline", None
    client = DaemonClient.from_runtime_file()
    if client is not None:
        try:
            hello = client.hello()
            print("connected to sam3gimpd %s (%s, %s)"
                  % (hello.get("sam3d_version"), hello.get("engine_mode"), hello.get("device")))
            return "daemon", client
        except Exception as exc:
            print("runtime.json present but unusable (%s)" % exc)
    if mode in ("auto", "spawn"):
        try:
            client = spawn_daemon()
            hello = client.hello()
            print("spawned sam3gimpd %s (%s)" % (hello.get("sam3d_version"), hello.get("engine_mode")))
            return "daemon", client
        except Exception as exc:
            if mode == "spawn":
                raise SystemExit("could not spawn sam3gimpd: %s" % exc)
            print("no daemon (%s) - falling back to the offline engine" % exc)
    return "offline", None


def run_selftest(window, out_png, width=900, height=640, prompt="yellow school bus"):
    """Headless render check: draw the widget onto an ImageSurface and prove it painted.

    This is the fallback for a machine with no usable window manager, and it is also
    what the test suite does.  It exercises the *real* ``do_draw`` path, not a
    stub.
    """
    import cairo

    canvas = window.canvas
    # GTK3 ignores size_allocate on an invisible widget, so mark it visible
    # first.  No window, no display and no realization are needed beyond that:
    # ``do_draw`` only reads the allocation.
    canvas.show()
    alloc = Gdk.Rectangle()
    alloc.x = alloc.y = 0
    alloc.width, alloc.height = width, height
    canvas.size_allocate(alloc)

    frame = window._offline.text(prompt, "r-selftest")
    header = canvas.set_result_frame(frame)
    canvas.set_status_text("selftest: %s" % prompt)

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    cr = cairo.Context(surface)
    canvas.do_draw(cr)
    surface.flush()
    data = bytes(surface.get_data())
    non_empty = sum(1 for b in data[::997] if b)
    print("selftest: %d instances, %d bytes of surface, %d/%d sampled bytes non-zero"
          % (len(header["instances"]), len(data), non_empty, len(data[::997])))
    if out_png:
        surface.write_to_png(out_png)
        print("selftest: wrote %s" % out_png)
    return non_empty > 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", nargs="?", help="PNG/JPEG/PPM/... to load (default: synthetic)")
    parser.add_argument("--backend", choices=("auto", "daemon", "spawn", "offline"),
                        default="auto", help="where masks come from (default: auto)")
    parser.add_argument("--offline", action="store_true", help="shorthand for --backend offline")
    parser.add_argument("--spawn", action="store_true", help="shorthand for --backend spawn")
    parser.add_argument("--max-side", type=int, default=MAX_SIDE,
                        help="downscale the long side to this before upload (API limit: 1008)")
    parser.add_argument("--selftest", action="store_true",
                        help="render once offscreen, report, and exit (no window needed)")
    parser.add_argument("--screenshot", metavar="PNG", help="with --selftest, write the render here")
    args = parser.parse_args(argv)

    if args.offline:
        args.backend = "offline"
    elif args.spawn:
        args.backend = "spawn"

    if args.image:
        rgb, width, height = load_image(args.image, min(MAX_SIDE, args.max_side))
        title = "sam3 canvas harness - %s" % os.path.basename(args.image)
    else:
        rgb, width, height = synthetic_image()
        title = "sam3 canvas harness - synthetic"

    backend, client = ("offline", None) if args.selftest else choose_backend(args.backend)
    window = HarnessWindow(rgb, width, height, backend, client, title=title)

    if args.selftest:
        ok = run_selftest(window, args.screenshot)
        return 0 if ok else 1

    window.show_all()
    window.canvas.grab_focus()
    Gtk.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
