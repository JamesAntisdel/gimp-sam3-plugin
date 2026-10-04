"""Tests for ``plugin/sam3_gimp/client.py``.

Everything here runs without torch, GIMP, a GPU or model weights.

The client is exercised against a real HTTP server -- a compact but
*contract-conforming* implementation of ``_daemon/API.md`` living in this file
(:class:`FakeDaemon`).  Testing against a real socket rather than a mocked
``http.client`` is deliberate: keep-alive reuse, long-polling on one connection
while POSTing on another, ``Content-Length`` framing and the binary result frame
are exactly the things that break, and none of them are visible to a mock.

``FakeDaemon`` is *not* the ``sam3gimpd`` stub engine (that lives in the daemon
package and is tested in ``tests/daemon/``).  It exists so this suite does not
depend on the daemon package at all; ``tests/plugin/test_launcher.py`` covers
the real ``sam3gimpd serve --stub`` when it is importable.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import socket
import struct
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import client as C


# =========================================================================== #
# a contract-conforming fake daemon
# =========================================================================== #
MAGIC = b"SAM3RES\x00"


def pack_frame(header: dict, blobs) -> bytes:
    """Build an ``API.md`` §8.1 frame: magic | uint32 header length | JSON | blobs."""
    blobs = list(blobs)
    header = dict(header)
    header["blob_length"] = sum(len(b) for b in blobs)
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return MAGIC + struct.pack("<I", len(raw)) + raw + b"".join(blobs)


def soft_blob(width: int, height: int) -> bytes:
    """An anti-aliased elliptical blob: genuine soft uint8, never 0/255 only.

    ``API.md`` §14 requires this of the stub engine because a constant-255
    rectangle would hide client threshold bugs.
    """
    out = bytearray(width * height)
    cx, cy = width / 2.0, height / 2.0
    rx, ry = max(1.0, width / 2.0), max(1.0, height / 2.0)
    for y in range(height):
        for x in range(width):
            d = ((x + 0.5 - cx) / rx) ** 2 + ((y + 0.5 - cy) / ry) ** 2
            v = 255.0 / (1.0 + math.exp(-6.0 * (1.0 - d)))
            out[y * width + x] = max(0, min(255, int(round(v))))
    return bytes(out)


def _error(code: str, message: str, detail=None) -> bytes:
    return json.dumps(
        {"error": {"code": code, "message": message, "detail": detail or {}}}
    ).encode("utf-8")


_STATUS_FOR = {
    "bad_request": 400,
    "invalid_json": 400,
    "missing_header": 400,
    "bad_dimensions": 400,
    "payload_size_mismatch": 400,
    "version_mismatch": 400,
    "unauthorized": 401,
    "forbidden_host": 403,
    "not_found": 404,
    "image_not_found": 404,
    "job_not_found": 404,
    "method_not_allowed": 405,
    "image_not_ready": 409,
    "payload_too_large": 413,
    "unsupported_media_type": 415,
    "internal_error": 500,
    "inference_failed": 500,
    "shutting_down": 503,
    "queue_full": 503,
}


class FakeDaemon:
    """A threaded HTTP server implementing enough of ``API.md`` to drive a client.

    Deliberately faithful on the parts that matter to the plug-in:

    * bearer auth + the loopback ``Host`` guard (§2), and the ``/hello``
      identity proof (§6.1);
    * ``202`` + job ids for uploads and prompts, one serial worker (§10);
    * supersession of *queued* jobs for the same image (§10);
    * long-poll ``?wait=`` that returns on a terminal state, a progress delta of
      0.01, a stage change, or the timeout (§11);
    * the binary result frame with tightly-packed masks (§8).

    ``gate`` lets a test freeze the worker so supersession is deterministic
    rather than a race; ``hello_gate`` does the same to ``GET /hello``, which
    is how a stalled daemon looks from outside.  ``prove_identity=False``
    models a daemon that predates the proof (or an impostor that cannot make
    it), ``host`` a daemon bound to some other address.
    """

    def __init__(self, api_version: str = C.API_VERSION, token: str = "t" * 43,
                 stage_delay: float = 0.02, fail_prompts: bool = False,
                 prove_identity: bool = True, host: str = "127.0.0.1") -> None:
        self.api_version = api_version
        self.token = token
        self.stage_delay = stage_delay
        self.fail_prompts = fail_prompts
        self.prove_identity = prove_identity
        self.host = host
        self.hello_gate = threading.Event()
        self.hello_gate.set()
        self.accepted_connections = 0
        self.seen_hosts = []      # Host headers, for assertions

        self.images = {}          # image_id -> dict
        self.jobs = {}            # job_id -> dict
        self.queue = []           # job ids awaiting the worker
        self.lock = threading.Condition()
        self.gate = threading.Event()
        self.gate.set()
        self.shutting_down = False
        self.request_log = []     # (method, path) for assertions
        self.connections = set()  # live sockets, so close() can really kill them
        self._counter = 0

        daemon = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = "fake-sam3gimpd/1.0"

            def log_message(self, *args):  # silence stderr noise
                pass

            def setup(self):
                BaseHTTPRequestHandler.setup(self)
                daemon.connections.add(self.connection)
                daemon.accepted_connections += 1

            def finish(self):
                daemon.connections.discard(self.connection)
                BaseHTTPRequestHandler.finish(self)

            # -- plumbing --------------------------------------------------- #
            def _send(self, status, body, ctype="application/json; charset=utf-8", extra=None):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Sam3-Api", daemon.api_version)
                self.send_header("X-Sam3-Version", "0.0.0-fake")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def _json(self, status, obj, extra=None):
                self._send(status, json.dumps(obj).encode("utf-8"), extra=extra)

            def _fail(self, code, message="", detail=None):
                self._send(_STATUS_FOR.get(code, 500), _error(code, message or code, detail))

            def _authorised(self):
                raw = self.headers.get("Host") or ""
                daemon.seen_hosts.append(raw)
                # §2: a loopback literal with an optional port; IPv6 in brackets.
                match = re.match(r"^(\[[^\]]*\]|[^:]*)(?::\d+)?$", raw)
                host = match.group(1).strip("[]") if match else raw
                if host not in ("127.0.0.1", "localhost", "::1"):
                    self._fail("forbidden_host", "Host %r is not loopback" % host)
                    return False
                auth = self.headers.get("Authorization") or ""
                if auth != "Bearer " + daemon.token:
                    self._fail("unauthorized", "missing or wrong bearer token")
                    return False
                if daemon.shutting_down and self.path != "/shutdown":
                    self._fail("shutting_down", "the daemon is shutting down")
                    return False
                return True

            def _body(self):
                n = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(n) if n else b""

            # -- routing ---------------------------------------------------- #
            def do_GET(self):
                if not self._authorised():
                    return
                path, _, query = self.path.partition("?")
                daemon.request_log.append(("GET", path))
                params = {}
                for part in query.split("&"):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        params[k] = v
                if path == "/hello":
                    daemon.hello_gate.wait(30.0)
                    return self._json(200, daemon.hello_payload(
                        self.headers.get("X-Sam3-Nonce")))
                if path == "/status":
                    payload = daemon.hello_payload()
                    payload.update({"jobs_total": len(daemon.jobs), "images": [], "last_error": None})
                    return self._json(200, payload)
                if path.startswith("/jobs/"):
                    return self._job(path[len("/jobs/"):], params)
                return self._fail("not_found", "no such path")

            def do_POST(self):
                if not self._authorised():
                    return
                path = self.path.partition("?")[0]
                daemon.request_log.append(("POST", path))
                body = self._body()
                if path == "/images":
                    return self._upload(body)
                if path == "/shutdown":
                    daemon.shutting_down = True
                    return self._json(202, {"ok": True, "pid": 4242, "grace_ms": 0})
                parts = path.strip("/").split("/")
                if len(parts) == 3 and parts[0] == "images" and parts[2] in ("text", "points"):
                    return self._prompt(parts[1], parts[2], body)
                return self._fail("not_found", "no such path")

            def do_DELETE(self):
                if not self._authorised():
                    return
                path = self.path.partition("?")[0]
                daemon.request_log.append(("DELETE", path))
                parts = path.strip("/").split("/")
                if len(parts) == 2 and parts[0] == "images":
                    return daemon.delete_image(self, parts[1])
                return self._fail("not_found", "no such path")

            # -- handlers ---------------------------------------------------- #
            def _upload(self, body):
                if (self.headers.get("Content-Type") or "") != "application/octet-stream":
                    return self._fail("unsupported_media_type", "want application/octet-stream")
                try:
                    width = int(self.headers["X-Width"])
                    height = int(self.headers["X-Height"])
                except (KeyError, TypeError, ValueError):
                    return self._fail("missing_header", "X-Width and X-Height are required")
                if not (16 <= width <= 1008 and 16 <= height <= 1008):
                    return self._fail("bad_dimensions", "sides must be within [16, 1008]")
                if len(body) != width * height * 3:
                    return self._fail(
                        "payload_size_mismatch", "bad body length",
                        {"expected": width * height * 3, "got": len(body)},
                    )
                payload = daemon.accept_image(body, width, height)
                return self._json(202, payload)

            def _prompt(self, image_id, kind, body):
                try:
                    req = json.loads(body.decode("utf-8"))
                except ValueError:
                    return self._fail("invalid_json", "body is not JSON")
                if image_id not in daemon.images:
                    return self._fail("image_not_found", "unknown image", {"image_id": image_id})
                if not req.get("request_id"):
                    return self._fail("bad_request", "request_id is required")
                if kind == "text" and not str(req.get("text", "")).strip():
                    return self._fail("bad_request", "text is required")
                payload = daemon.accept_prompt(image_id, kind, req)
                return self._json(202, payload)

            def _job(self, job_id, params):
                try:
                    wait = float(params.get("wait", 0.0))
                except ValueError:
                    wait = 0.0
                meta = params.get("meta", "0") == "1"
                job = daemon.wait_job(job_id, min(max(wait, 0.0), 30.0))
                if job is None:
                    return self._fail("job_not_found", "unknown job", {"job_id": job_id})
                if job["state"] == "done" and not meta:
                    frame = job["frame"]
                    hlen = struct.unpack_from("<I", frame, 8)[0]
                    return self._send(
                        200, frame, ctype=C.RESULT_CONTENT_TYPE,
                        extra={
                            "X-Sam3-Job-State": "done",
                            "X-Sam3-Header-Length": str(hlen),
                            "X-Sam3-Request-Id": job["request_id"],
                        },
                    )
                return self._json(200, daemon.job_status_payload(job, meta))

        class Server(ThreadingHTTPServer):
            address_family = socket.AF_INET6 if ":" in host else socket.AF_INET

            def handle_error(self, request, client_address):
                # A client that hung up mid-response (a closed client, a
                # timed-out probe) is expected here, not a bug in the fake.
                if isinstance(sys.exc_info()[1], (ConnectionError, socket.timeout)):
                    return
                ThreadingHTTPServer.handle_error(self, request, client_address)

        self.httpd = Server((host, 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self._serve_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._serve_thread.start()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_stop = False
        self._worker.start()

    # -- lifecycle ---------------------------------------------------------- #
    def close(self):
        """Stop serving *and* drop live sockets, so a client sees a real death."""
        self._worker_stop = True
        self.gate.set()
        self.hello_gate.set()
        with self.lock:
            self.lock.notify_all()
        self.httpd.shutdown()
        self.httpd.server_close()
        for sock in list(self.connections):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        self.connections.clear()

    def runtime_info(self, pid=None, host=None):
        return {
            "port": self.port,
            "token": self.token,
            "pid": int(os.getpid() if pid is None else pid),
            "version": "0.0.0-fake",
            "started_at": time.time(),
            "api_version": self.api_version,
            "host": self.host if host is None else host,
            "unknown_future_key": "ignored by readers",
        }

    def client(self, **kwargs):
        kwargs.setdefault("read_timeout", 10.0)
        return C.Sam3Client.from_runtime_info(self.runtime_info(), **kwargs)

    # -- payloads ------------------------------------------------------------ #
    def hello_payload(self, nonce=None):
        payload = {
            "api_version": self.api_version,
            "sam3d_version": "0.0.0-fake",
            "engine_mode": "stub",
            "device": "stub",
            "dtype": "float32",
            "capabilities": ["pcs", "pvs"],
            "torch_available": False,
            "weights_available": False,
            "model_canvas": {"width": 1008, "height": 1008},
            "pid": 4242,
            "started_at": 1772395551.812,
            "uptime_s": 1.0,
            "limits": {"max_image_side": 1008, "max_upload_bytes": 1008 * 1008 * 3},
        }
        # Written out rather than calling client.nonce_proof, so a mistake in
        # the client's version cannot cancel itself out here.
        if self.prove_identity and nonce and re.match(r"^[A-Za-z0-9_-]{16,128}$", nonce):
            payload["nonce_proof"] = hmac.new(
                self.token.encode("utf-8"), b"sam3gimpd-hello:" + nonce.encode("ascii"),
                hashlib.sha256).hexdigest()
        return payload

    def _next_id(self, prefix):
        self._counter += 1
        return "%s-%06d" % (prefix, self._counter)

    def accept_image(self, body, width, height):
        import hashlib

        image_id = hashlib.blake2b(
            body + b"|%d|%d" % (width, height), digest_size=16
        ).hexdigest()
        canvas = {"width": 1008, "height": 1008}
        transform = {
            "scale_x": canvas["width"] / float(width),
            "scale_y": canvas["height"] / float(height),
            "offset_x": 0.0,
            "offset_y": 0.0,
        }
        with self.lock:
            cached = image_id in self.images
            if cached:
                job_id = self.images[image_id]["job_id"]
                state = self.jobs[job_id]["state"]
            else:
                job_id = self._next_id("j")
                self.images[image_id] = {
                    "width": width, "height": height, "job_id": job_id,
                    "canvas": canvas, "transform": transform,
                }
                self.jobs[job_id] = self._new_job(job_id, "encode", image_id, "")
                self.queue.append(job_id)
                state = "queued"
                self.lock.notify_all()
        return {
            "image_id": image_id,
            "job_id": job_id,
            "cached": cached,
            "state": state,
            "image": {"width": width, "height": height},
            "model_canvas": canvas,
            "canvas_from_image": transform,
        }

    def accept_prompt(self, image_id, kind, req):
        engine = "pcs" if kind == "text" else "pvs"
        with self.lock:
            job_id = self._next_id("j")
            superseded = []
            # API.md section 10: queued-not-started jobs for the SAME image are
            # superseded; encode jobs are exempt.
            for other in list(self.queue):
                job = self.jobs[other]
                if job["image_id"] == image_id and job["engine"] != "encode":
                    job["state"] = "superseded"
                    job["superseded_by"] = job_id
                    self.queue.remove(other)
                    superseded.append(other)
            job = self._new_job(job_id, engine, image_id, req["request_id"])
            job["prompt"] = req
            self.jobs[job_id] = job
            self.queue.append(job_id)
            self.lock.notify_all()
        return {
            "job_id": job_id,
            "request_id": req["request_id"],
            "image_id": image_id,
            "engine": engine,
            "state": "queued",
            "superseded_job_ids": superseded,
        }

    def delete_image(self, handler, image_id):
        with self.lock:
            if image_id not in self.images:
                handler._fail("image_not_found", "unknown image", {"image_id": image_id})
                return
            del self.images[image_id]
            cancelled = []
            for other in list(self.queue):
                job = self.jobs[other]
                if job["image_id"] == image_id:
                    job["state"] = "cancelled"
                    self.queue.remove(other)
                    cancelled.append(other)
            self.lock.notify_all()
        handler._json(200, {"image_id": image_id, "deleted": True, "cancelled_job_ids": cancelled})

    def _new_job(self, job_id, engine, image_id, request_id):
        return {
            "job_id": job_id, "engine": engine, "image_id": image_id,
            "request_id": request_id, "state": "queued", "progress": 0.0,
            "stage": "queued", "created_at": time.time(), "started_at": None,
            "finished_at": None, "error": None, "frame": None,
            "superseded_by": None, "prompt": None, "header": None,
        }

    def job_status_payload(self, job, meta=False):
        payload = {
            "job_id": job["job_id"],
            "state": job["state"],
            "engine": job["engine"],
            "image_id": job["image_id"],
            "request_id": job["request_id"],
            "progress": job["progress"],
            "stage": job["stage"],
            "created_at": job["created_at"],
            "started_at": job["started_at"],
            "finished_at": job["finished_at"],
            "elapsed_ms": 1.0,
            "queue_position": 0 if job["state"] == "queued" else None,
            "superseded_by": job["superseded_by"],
            "masks_available": job["state"] == "done" and job["frame"] is not None,
            "error": job["error"],
        }
        if meta and job["state"] == "done":
            payload["result"] = job["header"]
        return payload

    # -- long poll ----------------------------------------------------------- #
    def wait_job(self, job_id, wait):
        deadline = time.time() + wait
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            base = (job["state"], job["stage"], job["progress"])
            while True:
                cur = (job["state"], job["stage"], job["progress"])
                terminal = cur[0] in ("done", "failed", "superseded", "cancelled")
                material = cur[0] != base[0] or cur[1] != base[1] or (cur[2] - base[2]) >= 0.01
                if terminal or material:
                    return dict(job)
                remaining = deadline - time.time()
                if remaining <= 0:
                    return dict(job)
                self.lock.wait(remaining)

    # -- the single serial worker -------------------------------------------- #
    def _worker_loop(self):
        while not self._worker_stop:
            # The gate is checked BEFORE dequeuing: a held gate leaves jobs in
            # the `queued` state, which is what supersession acts on (a running
            # job is never cancelled).
            if not self.gate.wait(0.05):
                continue
            with self.lock:
                if not self.queue:
                    self.lock.wait(0.05)
                    continue
                job_id = self.queue.pop(0)
                job = self.jobs[job_id]
                job["state"] = "running"
                job["started_at"] = time.time()
                self.lock.notify_all()
            self._run_job(job)

    def _set(self, job, stage, progress):
        with self.lock:
            job["stage"] = stage
            job["progress"] = progress
            self.lock.notify_all()
        time.sleep(self.stage_delay)

    def _run_job(self, job):
        try:
            if job["engine"] == "encode":
                self._set(job, "encoding", 0.05)
                self._set(job, "encoding", 0.60)
                with self.lock:
                    job["state"] = "done"
                    job["stage"] = "done"
                    job["progress"] = 1.0
                    job["finished_at"] = time.time()
                    job["frame"] = pack_frame(self._header(job, []), [])
                    job["header"] = json.loads(
                        job["frame"][12:12 + struct.unpack_from("<I", job["frame"], 8)[0]]
                    )
                    self.lock.notify_all()
                return
            if self.fail_prompts:
                with self.lock:
                    job["state"] = "failed"
                    job["stage"] = "failed"
                    job["progress"] = 1.0
                    job["finished_at"] = time.time()
                    job["error"] = {
                        "code": "inference_failed",
                        "message": "the forward pass raised",
                        "detail": {"trace_id": "abc123"},
                    }
                    self.lock.notify_all()
                return
            self._set(job, "prompting", 0.60)
            self._set(job, "decoding", 0.80)
            self._set(job, "packing", 0.95)
            header, blobs = self._masks(job)
            frame = pack_frame(header, blobs)
            with self.lock:
                job["state"] = "done"
                job["stage"] = "done"
                job["progress"] = 1.0
                job["finished_at"] = time.time()
                job["frame"] = frame
                job["header"] = json.loads(
                    frame[12:12 + struct.unpack_from("<I", frame, 8)[0]]
                )
                self.lock.notify_all()
        except Exception as exc:  # pragma: no cover - a bug in the fake
            with self.lock:
                job["state"] = "failed"
                job["error"] = {"code": "internal_error", "message": str(exc), "detail": {}}
                self.lock.notify_all()

    def _header(self, job, instances):
        img = self.images.get(job["image_id"]) or {
            "width": 64, "height": 48,
            "canvas": {"width": 1008, "height": 1008},
            "transform": {"scale_x": 1.0, "scale_y": 1.0, "offset_x": 0.0, "offset_y": 0.0},
        }
        prompt = job.get("prompt") or {}
        return {
            "api_version": self.api_version,
            "job_id": job["job_id"],
            "request_id": job["request_id"],
            "image_id": job["image_id"],
            "engine": job["engine"],
            "state": "done",
            "prompt": {"kind": "text" if job["engine"] == "pcs" else "points",
                       "text": prompt.get("text", "")},
            "image": {"width": img["width"], "height": img["height"]},
            "model_canvas": img["canvas"],
            "canvas_from_image": img["transform"],
            "mask_encoding": "u8_soft",
            "elapsed_ms": 12.5,
            "truncated": False,
            "instances": instances,
        }

    def _masks(self, job):
        """Deterministic instances derived from the prompt, as §14 requires."""
        prompt = job.get("prompt") or {}
        # Seeded by the prompt CONTENT only -- never the request_id -- so the
        # same prompt yields byte-identical frames (API.md section 14).
        material = json.dumps(
            [prompt.get("text", ""), prompt.get("points", []), prompt.get("box")],
            sort_keys=True,
        ).encode("utf-8")
        seed = int.from_bytes(__import__("hashlib").blake2b(material, digest_size=4).digest(), "big") % 7919
        count = 1 + (seed % 3)
        instances, blobs, offset = [], [], 0
        for i in range(count):
            w = 20 + ((seed >> (i * 3)) % 17)
            h = 16 + ((seed >> (i * 5)) % 13)
            x0 = 10 + ((seed >> (i * 2)) % 200)
            y0 = 20 + ((seed >> (i * 4)) % 200)
            blob = soft_blob(w, h)
            instances.append({
                "instance_id": i,
                "score": round(0.95 - 0.1 * i, 3),
                "label": prompt.get("text", ""),
                "bbox": [x0, y0, x0 + w, y0 + h],
                "mask_width": w,
                "mask_height": h,
                "blob_offset": offset,
                "blob_length": w * h,
            })
            blobs.append(blob)
            offset += len(blob)
        return self._header(job, instances), blobs


class RawServer:
    """Something listening on a port that is *not* a conforming daemon.

    ``respond(method, path, headers)`` returns the raw bytes to send back, so a
    test can produce what no well-behaved server would: a /hello without the
    identity proof, a body far larger than its purpose, a chunked or unsized
    response.  Every request is recorded as ``(method, path, headers)``.  The
    connection is closed after each response.
    """

    def __init__(self, respond, host: str = "127.0.0.1") -> None:
        import socketserver

        self.requests = []
        server = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                line = self.rfile.readline(65537).decode("latin-1").strip()
                if not line:
                    return
                method, path = (line.split(" ") + ["", ""])[:2]
                headers = {}
                while True:
                    h = self.rfile.readline(65537).decode("latin-1")
                    if h in ("\r\n", "\n", ""):
                        break
                    k, _, v = h.partition(":")
                    headers[k.strip().lower()] = v.strip()
                n = int(headers.get("content-length") or 0)
                if n:
                    self.rfile.read(n)
                server.requests.append((method, path, headers))
                try:
                    self.wfile.write(respond(method, path, headers))
                except OSError:
                    pass

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self.httpd = Server((host, 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def runtime_info(self, pid, token="t" * 43):
        return {"port": self.port, "token": token, "pid": int(pid), "version": "0.1.1",
                "started_at": time.time(), "host": "127.0.0.1"}


def http_response(body: bytes, status: str = "200 OK", headers=None) -> bytes:
    head = ["HTTP/1.1 " + status, "Content-Type: application/json", "Connection: close"]
    head += list(headers if headers is not None else ["Content-Length: %d" % len(body)])
    return ("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + body


def unproven_hello(method, path, headers, proof=None):
    """A plausible /hello from something that does not hold the token."""
    payload = {"api_version": C.API_VERSION, "sam3d_version": "0.1.1", "engine_mode": "torch",
               "device": "cuda", "capabilities": ["pcs", "pvs"], "pid": 1}
    if proof is not None:
        payload["nonce_proof"] = proof
    if path.startswith("/hello"):
        return http_response(json.dumps(payload).encode("utf-8"))
    return http_response(json.dumps({"image_id": "x", "job_id": "y"}).encode("utf-8"),
                         "202 Accepted")


# =========================================================================== #
# fixtures
# =========================================================================== #
@pytest.fixture
def daemon():
    d = FakeDaemon()
    try:
        yield d
    finally:
        d.close()


@pytest.fixture
def cl(daemon):
    c = daemon.client()
    try:
        yield c
    finally:
        c.close()


@pytest.fixture
def uploaded(cl, rgb_image):
    width, height, pixels = rgb_image
    accepted = cl.upload_image(pixels, width, height, source_width=3000, source_height=2000)
    cl.wait_for_image(accepted, timeout=10.0)
    return accepted


# =========================================================================== #
# module hygiene
# =========================================================================== #
def test_client_imports_only_stdlib_and_gi():
    """The zero-dependency rule: nothing but stdlib may be imported at module
    top level, and ``gi`` only lazily inside a function."""
    import ast
    import pathlib
    import sys

    src = pathlib.Path(C.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    top_level = set()
    for node in tree.body:  # only module level, not function bodies
        if isinstance(node, ast.Import):
            top_level.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            top_level.add(node.module.split(".")[0])
    assert "gi" not in top_level, "gi must be imported lazily, not at module level"
    for name in top_level:
        assert name in sys.stdlib_module_names, "%s is not in the standard library" % name


def test_gi_is_optional():
    """``glib_idle_add`` answers without raising whether or not gi is present."""
    C._reset_glib_cache()
    fn = C.glib_idle_add()
    assert fn is None or callable(fn)


# =========================================================================== #
# handshake, auth, transport errors
# =========================================================================== #
def test_hello_carries_the_daemon_build(cl):
    hello = cl.hello()
    assert hello.build == hello.raw.get("build", "")


def test_hello_build_defaults_to_empty_for_an_old_daemon():
    import client as C
    hello = C.Hello.from_dict({"api_version": "1.0", "sam3d_version": "0.1.0"})
    assert hello.build == ""
    assert C.Hello.from_dict({"api_version": "1.0", "build": "0c5d7059"}).build == "0c5d7059"


def test_hello_roundtrip(cl):
    hello = cl.hello()
    assert hello.api_version == C.API_VERSION
    assert hello.engine_mode == "stub"
    assert hello.is_stub
    assert hello.has_capability("pcs") and hello.has_capability("pvs")
    assert not hello.has_capability("exemplar_boxes")
    assert hello.model_canvas.width == 1008
    assert hello.limits["max_image_side"] == 1008
    assert hello.raw["sam3d_version"] == "0.0.0-fake"


def test_status_is_a_superset_of_hello(cl):
    st = cl.status()
    assert st["api_version"] == C.API_VERSION
    assert "jobs_total" in st


def test_hello_rejects_a_different_api_major():
    d = FakeDaemon(api_version="2.3")
    try:
        c = d.client()
        with pytest.raises(C.VersionMismatch) as exc:
            c.hello()
        assert exc.value.server_api == "2.3"
        assert exc.value.client_api == C.API_VERSION
        # ...but check=False still returns the payload, which is what the
        # launcher needs in order to POST /shutdown to it.
        assert c.hello(check=False).api_version == "2.3"
        c.close()
    finally:
        d.close()


def test_a_newer_minor_is_compatible():
    d = FakeDaemon(api_version="1.7")
    try:
        c = d.client()
        assert c.hello().api_version == "1.7"
        c.close()
    finally:
        d.close()


@pytest.mark.parametrize(
    "client_api,server_api,ok",
    [("1.0", "1.0", True), ("1.0", "1.9", True), ("1.0", "2.0", False),
     ("1.0", "", False), ("1.0", "banana", False)],
)
def test_versions_compatible(client_api, server_api, ok):
    assert C.versions_compatible(client_api, server_api) is ok


def test_bad_token_raises_auth_error(daemon):
    info = daemon.runtime_info()
    info["token"] = "wrong" * 8
    c = C.Sam3Client.from_runtime_info(info)
    with pytest.raises(C.AuthError) as exc:
        c.hello()
    assert exc.value.status == 401
    assert exc.value.code == C.ErrorCode.UNAUTHORIZED
    c.close()


def test_connection_refused_is_daemon_unavailable(free_port):
    c = C.Sam3Client("127.0.0.1", free_port, "t" * 43, connect_timeout=1.0)
    with pytest.raises(C.DaemonUnavailable):
        c.hello()
    c.close()


def test_a_connect_timeout_is_daemon_unavailable(monkeypatch, free_port):
    """Windows retries a connection to a closed local port for about two
    seconds instead of refusing it, so there an absent daemon arrives as a
    timeout while connecting.  It is reported as ConnectTimeout, a kind of
    DaemonUnavailable -- never as RequestTimeout, which means a daemon that
    took the connection and then did not answer."""
    def _timeout(*_args, **_kwargs):
        raise socket.timeout("timed out")

    monkeypatch.setattr(C.socket, "create_connection", _timeout)
    c = C.Sam3Client("127.0.0.1", free_port, "t" * 43, connect_timeout=1.0)
    with pytest.raises(C.ConnectTimeout) as info:
        c.hello()
    assert isinstance(info.value, C.DaemonUnavailable)
    assert not isinstance(info.value, C.RequestTimeout)
    assert "nothing accepted a connection" in str(info.value)
    c.close()


class _RecordingSocket:
    def __init__(self):
        self.calls = []

    def shutdown(self, how):
        self.calls.append(("shutdown", how))

    def detach(self):
        self.calls.append(("detach",))
        return 1234


@pytest.mark.parametrize("os_name,closes_handle", [("posix", False), ("nt", True)])
def test_abort_socket_closes_the_os_handle_on_windows(monkeypatch, os_name, closes_handle):
    """shutdown() wakes a blocked read on POSIX; Windows needs the handle itself
    closed, after the socket object has let go of it."""
    closed = []
    monkeypatch.setattr(C, "os", types.SimpleNamespace(name=os_name))
    monkeypatch.setattr(C.socket, "close", closed.append)
    sock = _RecordingSocket()
    C._abort_socket(sock)
    assert sock.calls[0] == ("shutdown", socket.SHUT_RDWR)
    if closes_handle:
        assert sock.calls[1:] == [("detach",)] and closed == [1234]
    else:
        assert sock.calls[1:] == [] and closed == []


def test_daemon_death_mid_session_is_reported(daemon):
    c = daemon.client()
    assert c.hello().api_version == C.API_VERSION
    daemon.close()
    with pytest.raises(C.TransportError):
        c.hello()
    c.close()


def test_error_envelope_becomes_a_typed_api_error(cl):
    with pytest.raises(C.ApiError) as exc:
        cl.prompt_text("nosuchimage", "cat")
    assert exc.value.code == C.ErrorCode.IMAGE_NOT_FOUND
    assert exc.value.status == 404
    assert exc.value.detail["image_id"] == "nosuchimage"


def test_shutdown_then_everything_is_503(cl):
    assert cl.shutdown_daemon(grace_ms=0)["ok"] is True
    with pytest.raises(C.ApiError) as exc:
        cl.hello()
    assert exc.value.code == C.ErrorCode.SHUTTING_DOWN


# =========================================================================== #
# runtime.json
# =========================================================================== #
def test_read_runtime_json_accepts_valid_and_ignores_unknown_keys(tmp_path, auth_token):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps({
        "port": 41573, "token": auth_token, "pid": 1, "version": "0.1.0",
        "started_at": 1772395551.812, "surprise": [1, 2, 3],
    }))
    info = C.read_runtime_json(str(path))
    assert info["port"] == 41573
    assert info["surprise"] == [1, 2, 3]


@pytest.mark.parametrize("payload", [
    "",
    "   ",
    "not json",
    "[1,2,3]",
    json.dumps({"port": 1, "token": "x", "pid": 1, "version": "0.1.0"}),      # no started_at
    json.dumps({"port": 0, "token": "x", "pid": 1, "version": "0.1.0", "started_at": 1.0}),
    json.dumps({"port": "nope", "token": "x", "pid": 1, "version": "0", "started_at": 1.0}),
    json.dumps({"port": 1, "token": "", "pid": 1, "version": "0", "started_at": 1.0}),
])
def test_read_runtime_json_rejects_junk(tmp_path, payload):
    path = tmp_path / "runtime.json"
    path.write_text(payload)
    assert C.read_runtime_json(str(path)) is None


def test_read_runtime_json_missing_file(tmp_path):
    assert C.read_runtime_json(str(tmp_path / "absent.json")) is None


def test_from_runtime_file(tmp_path, daemon):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(daemon.runtime_info()))
    c = C.Sam3Client.from_runtime_file(str(path))
    assert c.hello().engine_mode == "stub"
    c.close()


def test_from_runtime_file_missing_is_daemon_unavailable(tmp_path):
    with pytest.raises(C.DaemonUnavailable):
        C.Sam3Client.from_runtime_file(str(tmp_path / "absent.json"))


def test_default_runtime_file_follows_sam3_gimp_home(sam3_home):
    assert C.default_runtime_file() == str(sam3_home / "runtime.json")


# =========================================================================== #
# upload
# =========================================================================== #
def test_upload_roundtrip(cl, rgb_image):
    width, height, pixels = rgb_image
    acc = cl.upload_image(pixels, width, height, source_width=3000, source_height=2000)
    assert len(acc.image_id) == 32
    assert acc.cached is False
    assert acc.image.width == width and acc.image.height == height
    assert acc.model_canvas.width == 1008
    # The transform is *reported*, never assumed (API.md section 5).
    assert acc.canvas_from_image.scale_x == pytest.approx(1008.0 / width)
    assert acc.canvas_from_image.offset_x == 0.0
    status = cl.wait_for_image(acc, timeout=10.0)
    assert status.state == C.JobState.DONE


def test_identical_pixels_hit_the_cache(cl, rgb_image):
    width, height, pixels = rgb_image
    first = cl.upload_image(pixels, width, height)
    second = cl.upload_image(pixels, width, height)
    assert second.image_id == first.image_id
    assert second.cached is True


def test_different_pixels_get_different_ids(cl, make_rgb):
    a = cl.upload_image(make_rgb(32, 32, seed=1), 32, 32)
    b = cl.upload_image(make_rgb(32, 32, seed=2), 32, 32)
    assert a.image_id != b.image_id


@pytest.mark.parametrize("w,h", [(8, 64), (64, 8), (1009, 64), (64, 1009)])
def test_upload_rejects_illegal_dimensions_client_side(cl, w, h):
    """Checked locally so a 3 MB body is never sent just to be refused."""
    with pytest.raises(ValueError):
        cl.upload_image(b"\x00" * (w * h * 3), w, h)


def test_upload_rejects_a_mismatched_buffer(cl):
    with pytest.raises(ValueError) as exc:
        cl.upload_image(b"\x00" * 10, 32, 32)
    assert "3072" in str(exc.value)


def test_upload_sends_the_documented_headers(cl, rgb_image, daemon):
    width, height, pixels = rgb_image
    cl.upload_image(pixels, width, height)
    assert ("POST", "/images") in daemon.request_log


# =========================================================================== #
# prompting and results
# =========================================================================== #
def test_text_prompt_end_to_end(cl, uploaded):
    result = cl.run_text(uploaded.image_id, "yellow school bus", timeout=15.0)
    assert result is not None
    assert result.engine == "pcs"
    assert result.state == "done"
    assert result.mask_encoding == C.MASK_ENCODING_U8_SOFT
    assert 1 <= len(result) <= 8
    assert result.prompt["text"] == "yellow school bus"
    for inst in result:
        assert inst.mask_width == inst.bbox.width
        assert inst.mask_height == inst.bbox.height
        assert len(inst.mask) == inst.blob_length == inst.mask_width * inst.mask_height
        assert 0.0 <= inst.score <= 1.0
        assert inst.is_soft(), "masks must be genuine soft gradients"
    scores = [i.score for i in result]
    assert scores == sorted(scores, reverse=True)


def test_text_prompt_is_deterministic(cl, uploaded):
    a = cl.run_text(uploaded.image_id, "red car", timeout=15.0)
    b = cl.run_text(uploaded.image_id, "red car", timeout=15.0)
    assert [i.bbox.to_list() for i in a] == [i.bbox.to_list() for i in b]
    assert [i.mask for i in a] == [i.mask for i in b]


def test_points_prompt_end_to_end(cl, uploaded):
    result = cl.run_points(
        uploaded.image_id,
        [C.Point(12.0, 9.0, 1), {"x": 20.0, "y": 30.0, "label": 0}, (5.0, 5.0)],
        box=[1.0, 1.0, 40.0, 40.0],
        timeout=15.0,
    )
    assert result is not None
    assert result.engine == "pvs"
    assert len(result) >= 1


def test_prompt_argument_validation(cl, uploaded):
    with pytest.raises(ValueError):
        cl.prompt_text(uploaded.image_id, "   ")
    with pytest.raises(ValueError):
        cl.prompt_text(uploaded.image_id, "x" * 513)
    with pytest.raises(ValueError):
        cl.prompt_points(uploaded.image_id, [])
    with pytest.raises(ValueError):
        cl.prompt_points(uploaded.image_id, [(1.0, 2.0, 7)])          # bad label
    with pytest.raises(ValueError):
        cl.prompt_points(uploaded.image_id, box=[10.0, 10.0, 5.0, 20.0])  # x1 <= x0
    with pytest.raises(ValueError):
        cl.prompt_points(uploaded.image_id, [(1.0, 1.0)] * 65)        # over the 64 limit
    with pytest.raises(ValueError):
        cl.prompt_text(uploaded.image_id, "car", request_id="bad id!")


def test_exemplar_boxes_are_normalised(cl, uploaded):
    acc = cl.prompt_text(
        uploaded.image_id, "bus",
        boxes=[[1.0, 2.0, 3.0, 4.0], {"box": [5, 6, 7, 8], "label": 0}],
    )
    assert acc.engine == "pcs"


def test_failed_job_raises_job_failed(rgb_image):
    d = FakeDaemon(fail_prompts=True)
    try:
        c = d.client()
        width, height, pixels = rgb_image
        acc = c.upload_image(pixels, width, height)
        c.wait_for_image(acc, timeout=10.0)
        with pytest.raises(C.JobFailed) as exc:
            c.run_text(acc.image_id, "cat", timeout=15.0)
        assert exc.value.code == C.ErrorCode.INFERENCE_FAILED
        assert exc.value.detail["trace_id"] == "abc123"
        # A failed job is HTTP 200 with state=failed, not an HTTP error.
        assert c.job_status(exc.value.job_id).state == C.JobState.FAILED
        c.close()
    finally:
        d.close()


def test_job_result_on_an_unfinished_job_is_a_protocol_error(cl, uploaded, daemon):
    daemon.gate.clear()
    try:
        acc = cl.prompt_text(uploaded.image_id, "cat")
        with pytest.raises(C.ProtocolError):
            cl.job_result(acc.job_id)
    finally:
        daemon.gate.set()


def test_unknown_job_id(cl):
    with pytest.raises(C.ApiError) as exc:
        cl.job_status("j-nope")
    assert exc.value.code == C.ErrorCode.JOB_NOT_FOUND


def test_delete_image(cl, uploaded):
    out = cl.delete_image(uploaded.image_id)
    assert out["deleted"] is True
    with pytest.raises(C.ApiError) as exc:
        cl.prompt_text(uploaded.image_id, "cat")
    assert exc.value.code == C.ErrorCode.IMAGE_NOT_FOUND


def test_progress_callback_sees_monotonic_progress(cl, uploaded):
    seen = []
    result = cl.run_text(uploaded.image_id, "cat", timeout=15.0, poll_wait=5.0,
                         on_progress=seen.append)
    assert result is not None
    assert seen, "the progress callback must fire at least once"
    values = [s.progress for s in seen]
    assert values == sorted(values), "progress must be non-decreasing"
    assert seen[-1].state == C.JobState.DONE
    assert {s.stage for s in seen} & {"prompting", "decoding", "packing", "done"}


def test_job_timeout_is_raised(cl, uploaded, daemon):
    daemon.gate.clear()
    try:
        acc = cl.prompt_text(uploaded.image_id, "cat")
        with pytest.raises(C.JobTimeout):
            cl.wait_for_job(acc.job_id, timeout=0.3, poll_wait=0.1)
    finally:
        daemon.gate.set()


# =========================================================================== #
# request ids and supersession (API.md section 10)
# =========================================================================== #
def test_request_ids_are_unique_and_tracked(cl):
    a = cl.next_request_id("img-a")
    b = cl.next_request_id("img-a")
    c3 = cl.next_request_id("img-b")
    assert a != b != c3
    assert cl.latest_request_id("img-a") == b
    assert cl.latest_request_id("img-b") == c3
    assert cl.is_latest("img-a", b)
    assert cl.is_stale("img-a", a)
    # Supersession is per image: img-b's prompt does not stale img-a's.
    assert cl.is_latest("img-b", c3)


def test_stale_results_are_discarded(cl, uploaded):
    """The one-line §10 rule: drop any result whose request_id is not the latest."""
    rid_old = cl.next_request_id(uploaded.image_id)
    accepted = cl.prompt_text(uploaded.image_id, "cat", request_id=rid_old)
    cl.next_request_id(uploaded.image_id)  # the user typed a new prompt
    result = cl.wait_for_job(
        accepted.job_id, image_id=uploaded.image_id, request_id=rid_old, timeout=15.0
    )
    assert result is None, "a stale result must never reach the canvas"
    # The job itself still completed; we simply ignored it.
    assert cl.job_status(accepted.job_id).state == C.JobState.DONE


def test_server_side_supersession_yields_none(cl, uploaded, daemon):
    daemon.gate.clear()
    first = cl.prompt_text(uploaded.image_id, "cat")
    second = cl.prompt_text(uploaded.image_id, "dog")
    daemon.gate.set()
    assert second.superseded_job_ids == [first.job_id]
    assert cl.wait_for_job(first.job_id, timeout=15.0) is None
    assert cl.job_status(first.job_id).state == C.JobState.SUPERSEDED
    assert cl.wait_for_job(second.job_id, timeout=15.0) is not None


def test_encode_jobs_are_never_superseded(cl, rgb_image, daemon):
    daemon.gate.clear()
    width, height, pixels = rgb_image
    acc = cl.upload_image(pixels, width, height)
    prompt = cl.prompt_text(acc.image_id, "cat")
    assert prompt.superseded_job_ids == []
    daemon.gate.set()
    assert cl.wait_for_image(acc, timeout=15.0).state == C.JobState.DONE


def test_forget_image_clears_the_latest_id(cl):
    rid = cl.next_request_id("img")
    cl.forget_image("img")
    assert cl.latest_request_id("img") is None
    assert cl.is_latest("img", rid)  # nothing newer is known


# =========================================================================== #
# the binary frame (API.md section 8)
# =========================================================================== #
def _frame(instances=None, blobs=None, **header_overrides):
    header = {
        "api_version": "1.0", "job_id": "j-1", "request_id": "r-1",
        "image_id": "abc", "engine": "pcs", "state": "done",
        "prompt": {"kind": "text", "text": "bus"},
        "image": {"width": 1008, "height": 672},
        "model_canvas": {"width": 1008, "height": 1008},
        "canvas_from_image": {"scale_x": 1.0, "scale_y": 1.5, "offset_x": 0.0, "offset_y": 0.0},
        "mask_encoding": "u8_soft", "elapsed_ms": 1.0, "truncated": False,
        "instances": instances if instances is not None else [],
    }
    header.update(header_overrides)
    return pack_frame(header, blobs or [])


def test_parse_frame_zero_instances():
    result = C.parse_result_frame(_frame())
    assert len(result) == 0
    assert result.image.width == 1008
    assert result.canvas_from_image.scale_y == 1.5


def test_parse_frame_two_instances_are_tightly_packed():
    a, b = bytes(range(256)) * 2, bytes([7]) * 30
    instances = [
        {"instance_id": 0, "score": 0.9, "label": "bus", "bbox": [0, 0, 32, 16],
         "mask_width": 32, "mask_height": 16, "blob_offset": 0, "blob_length": 512},
        {"instance_id": 1, "score": 0.5, "label": "bus", "bbox": [10, 10, 16, 15],
         "mask_width": 6, "mask_height": 5, "blob_offset": 512, "blob_length": 30},
    ]
    result = C.parse_result_frame(_frame(instances, [a, b]))
    assert [i.instance_id for i in result] == [0, 1]
    assert result.instances[0].mask == a
    assert result.instances[1].mask == b
    assert result.instances[1].value_at(0, 0) == 7
    assert result.instances[1].value_at(99, 0) == 0     # outside the crop is 0
    assert result.instances[1].row(2) == bytes([7]) * 6
    assert result.instances[1].count_above(7) == 30
    assert result.instances[1].count_above(8) == 0


@pytest.mark.parametrize("mutate,message", [
    (lambda f: b"NOTSAM3\x00" + f[8:], "magic"),
    (lambda f: f[:4], "prefix"),
    (lambda f: f[:8] + struct.pack("<I", 10 ** 6) + f[12:], "overruns"),
    (lambda f: f + b"\x00" * 4, "blob region"),
])
def test_parse_frame_rejects_corruption(mutate, message):
    frame = _frame(
        [{"instance_id": 0, "score": 0.9, "label": "", "bbox": [0, 0, 4, 4],
          "mask_width": 4, "mask_height": 4, "blob_offset": 0, "blob_length": 16}],
        [bytes(16)],
    )
    with pytest.raises(C.ProtocolError) as exc:
        C.parse_result_frame(mutate(frame))
    assert message in str(exc.value)


def test_parse_frame_enforces_mask_equals_bbox():
    bad = [{"instance_id": 0, "score": 0.9, "label": "", "bbox": [0, 0, 5, 4],
            "mask_width": 4, "mask_height": 4, "blob_offset": 0, "blob_length": 16}]
    with pytest.raises(C.ProtocolError) as exc:
        C.parse_result_frame(_frame(bad, [bytes(16)]))
    assert "bbox" in str(exc.value)


def test_parse_frame_enforces_blob_length():
    bad = [{"instance_id": 0, "score": 0.9, "label": "", "bbox": [0, 0, 4, 4],
            "mask_width": 4, "mask_height": 4, "blob_offset": 0, "blob_length": 15}]
    with pytest.raises(C.ProtocolError) as exc:
        C.parse_result_frame(_frame(bad, [bytes(15)]))
    assert "blob_length" in str(exc.value)


def test_parse_frame_enforces_offsets_are_relative_to_the_blob_region():
    """The classic integration bug: an offset measured from the body start."""
    bad = [{"instance_id": 0, "score": 0.9, "label": "", "bbox": [0, 0, 4, 4],
            "mask_width": 4, "mask_height": 4, "blob_offset": 512, "blob_length": 16}]
    with pytest.raises(C.ProtocolError) as exc:
        C.parse_result_frame(_frame(bad, [bytes(16)]))
    assert "blob_offset" in str(exc.value)


def test_parse_frame_rejects_an_unknown_encoding():
    with pytest.raises(C.ProtocolError):
        C.parse_result_frame(_frame(mask_encoding="rle"))


def test_parse_frame_rejects_a_bad_declared_blob_length():
    frame = _frame()
    hlen = struct.unpack_from("<I", frame, 8)[0]
    header = json.loads(frame[12:12 + hlen])
    header["blob_length"] = 99
    raw = json.dumps(header, separators=(",", ":")).encode()
    tampered = MAGIC + struct.pack("<I", len(raw)) + raw
    with pytest.raises(C.ProtocolError) as exc:
        C.parse_result_frame(tampered)
    assert "blob_length" in str(exc.value)


# =========================================================================== #
# placing a mask on the original image (API.md section 9 / 12.6)
# =========================================================================== #
def test_placement_matches_the_worked_example():
    """The exact numbers from ``API.md`` §12.6."""
    instances = [{"instance_id": 0, "score": 0.934, "label": "yellow school bus",
                  "bbox": [412, 300, 700, 480], "mask_width": 288, "mask_height": 180,
                  "blob_offset": 0, "blob_length": 288 * 180}]
    result = C.parse_result_frame(_frame(instances, [bytes(288 * 180)]))
    placement = result.place(result.instances[0], 3000, 2000)
    assert placement.to_tuple() == (1226, 595, 857, 357)


def test_placement_uses_the_reported_transform_not_a_hardcoded_one():
    """A letterboxing processor sets non-zero offsets; the client must follow."""
    instances = [{"instance_id": 0, "score": 0.5, "label": "", "bbox": [100, 200, 200, 300],
                  "mask_width": 100, "mask_height": 100, "blob_offset": 0,
                  "blob_length": 100 * 100}]
    frame = _frame(
        instances, [bytes(100 * 100)],
        image={"width": 500, "height": 500},
        canvas_from_image={"scale_x": 2.0, "scale_y": 2.0, "offset_x": 4.0, "offset_y": 8.0},
    )
    result = C.parse_result_frame(frame)
    p = result.place(result.instances[0], 1000, 1000)
    # canvas->uploaded: (100-4)/2 = 48, (200-8)/2 = 96 ; uploaded->original: x2
    assert p.to_tuple() == (96, 192, 100, 100)


def test_placement_edges_are_rounded_independently_so_instances_tile():
    """§9 step 3: round each edge, *then* derive the size -- no gaps."""
    t = C.CanvasTransform(1.0, 1.0)
    up = C.Size(100, 100)
    src = C.Size(301, 301)

    def make(x0, x1):
        return C.MaskInstance(0, 1.0, "", C.BBox(x0, 0, x1, 10), x1 - x0, 10, 0, (x1 - x0) * 10)

    left = C.place_instance(make(0, 33), t, up, src)
    right = C.place_instance(make(33, 66), t, up, src)
    assert left.x + left.width == right.x


def test_transform_roundtrip():
    t = C.CanvasTransform(1.0, 1.5, 3.0, -2.0)
    cx, cy = t.image_to_canvas(10.0, 20.0)
    assert (cx, cy) == (13.0, 28.0)
    assert t.canvas_to_image(cx, cy) == (10.0, 20.0)


# =========================================================================== #
# the worker-thread API (DESIGN.md section 4)
# =========================================================================== #
def test_dispatcher_runs_off_the_calling_thread_and_marshals_back():
    d = C.Dispatcher(marshal="direct")
    try:
        seen = {}
        done = threading.Event()

        def work():
            seen["worker_thread"] = threading.current_thread().name
            return 42

        def on_done(value):
            seen["value"] = value
            done.set()

        call = d.submit(work, on_done=on_done)
        assert done.wait(5.0)
        assert seen["value"] == 42
        assert seen["worker_thread"] != threading.current_thread().name
        assert call.result(1.0) == 42
        assert call.state == C.Call.DONE
    finally:
        d.shutdown()


def test_dispatcher_reports_errors_to_on_error():
    d = C.Dispatcher(marshal="direct")
    try:
        got = {}
        done = threading.Event()

        def boom():
            raise C.DaemonUnavailable("nope")

        d.submit(boom, on_done=lambda v: got.setdefault("done", v),
                 on_error=lambda e: (got.setdefault("error", e), done.set()))
        assert done.wait(5.0)
        assert isinstance(got["error"], C.DaemonUnavailable)
        assert "done" not in got
    finally:
        d.shutdown()


def test_call_result_reraises_in_the_caller():
    d = C.Dispatcher(marshal="direct")
    try:
        call = d.submit(lambda: (_ for _ in ()).throw(C.ProtocolError("bad frame")))
        with pytest.raises(C.ProtocolError):
            call.result(5.0)
    finally:
        d.shutdown()


def test_cancel_suppresses_a_pending_call():
    d = C.Dispatcher(marshal="direct")
    try:
        gate = threading.Event()
        started = threading.Event()
        calls = []
        d.submit(lambda: (started.set(), gate.wait(5.0)))
        assert started.wait(5.0)
        second = d.submit(lambda: calls.append("ran"), on_done=lambda v: calls.append("cb"))
        assert second.cancel() is True
        gate.set()
        time.sleep(0.2)
        assert calls == []
        assert second.cancelled
    finally:
        d.shutdown()


def test_marshal_glib_requires_gi():
    """``marshal='glib'`` is explicit about its dependency; ``'auto'`` never is."""
    d = C.Dispatcher(marshal="glib")
    try:
        if C.glib_idle_add() is None:
            with pytest.raises(RuntimeError):
                d.marshal(lambda: None)
        else:
            d.marshal(lambda: None)  # queued on the (idle) GLib main loop
    finally:
        d.shutdown()


@pytest.mark.needs_gtk
def test_callbacks_really_land_on_the_glib_main_loop():
    """The production path: a worker thread's result arrives on the main loop.

    With ``gi`` present this exercises ``GLib.idle_add`` for real -- the
    callback must run on the thread spinning the main loop, never on the
    worker.
    """
    from gi.repository import GLib

    d = C.Dispatcher(marshal="glib")
    loop = GLib.MainLoop()
    seen = {}
    try:
        def work():
            seen["worker"] = threading.get_ident()
            return "masks"

        def on_done(value):
            seen["value"] = value
            seen["callback"] = threading.get_ident()
            loop.quit()

        GLib.timeout_add(8000, lambda: (loop.quit(), False)[1])  # never hang CI
        d.submit(work, on_done=on_done)
        loop.run()
    finally:
        d.shutdown()
    assert seen.get("value") == "masks"
    assert seen["callback"] == threading.get_ident(), "callback must run on the main loop"
    assert seen["worker"] != threading.get_ident(), "work must not run on the main loop"


def test_wrap_marshals_progress_callbacks():
    d = C.Dispatcher(marshal="direct")
    try:
        seen = []
        report = d.wrap(seen.append)
        report("encoding")
        assert seen == ["encoding"]
    finally:
        d.shutdown()


def test_dispatcher_rejects_bad_configuration():
    with pytest.raises(ValueError):
        C.Dispatcher(workers=0)
    with pytest.raises(ValueError):
        C.Dispatcher(marshal="carrier-pigeon")


def test_submit_a_real_call_through_the_client(cl, uploaded):
    """The shape the GTK dialog uses: submit a prompt, get masks on a callback."""
    d = C.Dispatcher(marshal="direct")
    c = C.Sam3Client("127.0.0.1", cl.port, cl.token, dispatcher=d)
    try:
        outcome = {}
        done = threading.Event()
        progress = d.wrap(lambda status: outcome.setdefault("stages", []).append(status.stage))
        c.submit(
            lambda: c.run_text(uploaded.image_id, "bus", timeout=15.0, on_progress=progress),
            on_done=lambda r: (outcome.update(result=r), done.set()),
            on_error=lambda e: (outcome.update(error=e), done.set()),
        )
        assert done.wait(20.0)
        assert "error" not in outcome, outcome.get("error")
        assert len(outcome["result"]) >= 1
        assert outcome["stages"]
    finally:
        c.close()
        d.shutdown()


def test_long_poll_and_post_run_on_separate_connections(cl, uploaded, daemon):
    """§1: the client long-polls on one connection while POSTing on another.

    This is the deadlock a single-threaded daemon would cause; asserting it here
    keeps the plug-in honest about needing per-thread connections.
    """
    daemon.gate.clear()
    first = cl.prompt_text(uploaded.image_id, "cat")
    outcome = {}
    waiter = threading.Thread(
        target=lambda: outcome.update(res=cl.wait_for_job(first.job_id, timeout=15.0))
    )
    waiter.start()
    time.sleep(0.2)
    second = cl.prompt_text(uploaded.image_id, "dog")   # a different connection
    daemon.gate.set()
    waiter.join(20.0)
    assert not waiter.is_alive()
    assert outcome["res"] is None                        # first was superseded
    assert cl.wait_for_job(second.job_id, timeout=15.0) is not None


def test_client_is_usable_as_a_context_manager(daemon):
    with C.Sam3Client.from_runtime_info(daemon.runtime_info()) as c:
        assert c.hello().api_version == C.API_VERSION


def test_keep_alive_connection_is_reused(cl, daemon):
    cl.hello()
    cl.hello()
    cl.status()
    assert daemon.accepted_connections == 1, "the keep-alive connection must be reused"


def test_short_lived_threads_leave_no_connections_behind(cl, daemon):
    """The dialog runs every daemon call on a new thread.

    A connection per *thread* outlived each one: a socket here and a parked
    handler thread in the daemon per call, for minutes.  Connections now
    belong to requests and go back to a small pool.
    """
    fds_before = len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else None
    for _ in range(60):
        t = threading.Thread(target=lambda: cl.hello(check=False))
        t.start()
        t.join()
    assert cl.open_connections() <= C.Sam3Client.MAX_IDLE_CONNECTIONS
    assert daemon.accepted_connections <= C.Sam3Client.MAX_IDLE_CONNECTIONS
    assert len(daemon.connections) <= C.Sam3Client.MAX_IDLE_CONNECTIONS
    if fds_before is not None:
        assert len(os.listdir("/proc/self/fd")) <= fds_before + C.Sam3Client.MAX_IDLE_CONNECTIONS


def test_concurrent_requests_still_get_their_own_connections(cl, daemon):
    """Checkout is per request, so parallel calls never share a socket."""
    errors = []

    def work():
        try:
            for _ in range(5):
                cl.hello()
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert errors == []
    assert cl.open_connections() <= C.Sam3Client.MAX_IDLE_CONNECTIONS


def test_close_releases_every_connection_and_is_final(daemon):
    c = daemon.client()
    c.hello()
    assert c.open_connections() == 1
    c.close()
    assert c.open_connections() == 0 and c.closed
    before = len(daemon.request_log)
    with pytest.raises(C.DaemonUnavailable) as exc:
        c.hello()
    assert "client closed" in str(exc.value)
    assert len(daemon.request_log) == before, "a closed client reconnected"
    c.close()   # idempotent


def test_close_stops_a_long_poll_on_another_thread(cl, uploaded, daemon):
    """A worker blocked in a long-poll must stop when its client is closed --
    not carry on reconnecting to a daemon the dialog has let go of."""
    daemon.gate.clear()      # the job never moves: the worker sits in ?wait=
    try:
        acc = cl.prompt_text(uploaded.image_id, "cat")
        outcome = {}

        def worker():
            try:
                outcome["result"] = cl.wait_for_job(acc.job_id, poll_wait=10.0)
            except Exception as exc:  # noqa: BLE001 - asserted below
                outcome["error"] = exc

        t = threading.Thread(target=worker)
        t.start()
        time.sleep(0.5)
        before = len(daemon.request_log)
        start = time.monotonic()
        cl.close()
        t.join(5.0)
        assert not t.is_alive(), "the long-poll outlived close()"
        assert time.monotonic() - start < 3.0
        assert isinstance(outcome.get("error"), C.DaemonUnavailable), outcome
        time.sleep(0.5)
        assert len(daemon.request_log) == before, "requests were sent after close()"
    finally:
        daemon.gate.set()


def test_helpers():
    assert C._quote("a b/c") == "a%20b%2Fc"
    assert C._fmt_float(10.0) == "10"
    assert C._fmt_float(0.5) == "0.5"
    assert C.api_major("1.4") == 1
    assert C.api_major("junk") is None


class TestApiErrorShowsTheCause:
    """Several API.md §4 errors are consequences, not causes.

    ``image_not_ready`` means the encode pass failed; *why* it failed -- out of
    memory, an unloadable checkpoint, no torch -- arrives in
    ``detail["error"]``. Rendering only the outer message showed the user "the
    encode pass for this image failed" and left the real reason unread in the
    payload.
    """

    def test_nested_cause_is_rendered(self):
        exc = C.ApiError(
            "image_not_ready", "the encode pass for this image failed",
            {"image_id": "img-1",
             "error": {"code": "inference_failed",
                       "message": "CUDA out of memory. Tried to allocate 2.50 GiB"}},
            409,
        )
        text = str(exc)
        assert "the encode pass for this image failed" in text
        assert "CUDA out of memory" in text
        assert "inference_failed" in text

    def test_cause_detail_is_summarised(self):
        exc = C.ApiError(
            "image_not_ready", "failed",
            {"error": {"code": "c", "message": "m",
                       "detail": {"device": "cuda", "half": "pcs"}}}, 409)
        assert "device=cuda" in str(exc) and "half=pcs" in str(exc)

    def test_the_redundant_image_id_is_not_repeated(self):
        exc = C.ApiError(
            "image_not_ready", "failed",
            {"error": {"code": "c", "message": "m",
                       "detail": {"image_id": "img-1", "device": "cpu"}}}, 409)
        assert "img-1" not in str(exc) and "device=cpu" in str(exc)

    def test_no_cause_renders_as_before(self):
        exc = C.ApiError("bad_request", "nope", {"image_id": "x"}, 400)
        assert str(exc) == "bad_request (HTTP 400): nope"

    def test_a_malformed_cause_does_not_break_rendering(self):
        for bad in ("not a dict", 42, None, {}, {"detail": "flat"}):
            exc = C.ApiError("image_not_ready", "failed", {"error": bad}, 409)
            assert "the request" or True
            assert str(exc).startswith("image_not_ready (HTTP 409): failed")

    def test_fields_are_still_available_programmatically(self):
        exc = C.ApiError("image_not_ready", "failed",
                              {"error": {"code": "x", "message": "y"}}, 409)
        assert exc.code == "image_not_ready" and exc.status == 409
        assert exc.detail["error"]["code"] == "x"


# =========================================================================== #
# the /hello identity proof (API.md section 6.1)
# =========================================================================== #
def test_the_proof_is_the_documented_hmac():
    token, nonce = "k" * 43, "n" * 32
    expected = hmac.new(token.encode("utf-8"), b"sam3gimpd-hello:" + nonce.encode("ascii"),
                        hashlib.sha256).hexdigest()
    assert C.nonce_proof(token, nonce) == expected
    assert C.NONCE_HEADER == "X-Sam3-Nonce"
    assert C.NONCE_PROOF_KEY == "nonce_proof"
    assert C.NONCE_PROOF_PREFIX == b"sam3gimpd-hello:"
    assert "DaemonIdentityError" in C.__all__
    assert issubclass(C.DaemonIdentityError, C.DaemonUnavailable)


def test_every_hello_sends_a_fresh_well_formed_nonce(daemon, cl):
    seen = []
    original = daemon.hello_payload

    def spy(nonce=None):
        seen.append(nonce)
        return original(nonce)

    daemon.hello_payload = spy
    cl.hello()
    cl.hello()
    assert len(seen) == 2 and seen[0] != seen[1]
    for nonce in seen:
        assert re.match(r"^[A-Za-z0-9_-]{16,128}$", nonce)


@pytest.mark.parametrize("api,older", [("1.0", True), (C.API_VERSION, False)])
def test_a_daemon_without_the_proof_is_refused(api, older):
    """What an older daemon -- or anything else on the port -- looks like.
    Only "no proof, API 1.0" reads as an older build of our own daemon."""
    d = FakeDaemon(prove_identity=False, api_version=api)
    try:
        c = d.client()
        with pytest.raises(C.DaemonIdentityError) as exc:
            c.hello()
        assert exc.value.hello is not None and exc.value.hello.api_version == api
        assert exc.value.reason == C.DaemonIdentityError.MISSING
        assert exc.value.predates_proof is older
        with pytest.raises(C.DaemonIdentityError):
            c.hello(check=False)
        assert d.token not in str(exc.value)
        c.close()
    finally:
        d.close()


def test_a_wrong_proof_is_never_mistaken_for_an_older_daemon():
    token = "t" * 43
    srv = RawServer(lambda m, p, h: unproven_hello(m, p, h, proof="ab" * 32))
    try:
        c = C.Sam3Client.from_runtime_info(srv.runtime_info(pid=1, token=token))
        with pytest.raises(C.DaemonIdentityError) as exc:
            c.hello()
        assert exc.value.reason == C.DaemonIdentityError.WRONG
        assert exc.value.predates_proof is False
        c.close()
    finally:
        srv.close()


@pytest.mark.parametrize("proof", [None, "", "0" * 64, "not hex", "é" * 64])
def test_an_impostor_without_the_token_cannot_answer(proof):
    """A squatter on the port: accepts anything, cannot compute the HMAC."""
    srv = RawServer(lambda m, p, h: unproven_hello(m, p, h, proof=proof))
    try:
        c = C.Sam3Client.from_runtime_info(srv.runtime_info(pid=1))
        with pytest.raises(C.DaemonIdentityError):
            c.hello()
        c.close()
        assert [r[0] for r in srv.requests] == ["GET"]
        assert re.match(r"^[A-Za-z0-9_-]{16,128}$", srv.requests[0][2]["x-sam3-nonce"])
    finally:
        srv.close()


def test_a_replayed_proof_does_not_verify():
    """A proof captured for one nonce is useless for the next."""
    token = "t" * 43
    replay = C.nonce_proof(token, "A" * 32)
    srv = RawServer(lambda m, p, h: unproven_hello(m, p, h, proof=replay))
    try:
        c = C.Sam3Client.from_runtime_info(srv.runtime_info(pid=1, token=token))
        with pytest.raises(C.DaemonIdentityError):
            c.hello()
        c.close()
    finally:
        srv.close()


# =========================================================================== #
# nothing the other end sends can make the plug-in buffer without limit
# =========================================================================== #
def _respond_with(headers, body=b"{}"):
    return lambda m, p, h: http_response(body, headers=headers)


@pytest.mark.parametrize("headers,message", [
    (["Content-Length: 99999999999"], "exceeds"),
    (["Transfer-Encoding: chunked"], "chunked"),
    ([], "Content-Length"),
])
def test_oversized_chunked_or_unsized_responses_are_refused(headers, message):
    body = b"2\r\n{}\r\n0\r\n\r\n" if "chunked" in message else b"{}"
    srv = RawServer(_respond_with(headers, body))
    try:
        c = C.Sam3Client.from_runtime_info(srv.runtime_info(pid=1), read_timeout=5.0)
        start = time.monotonic()
        with pytest.raises(C.ProtocolError) as exc:
            c.status()
        assert message in str(exc.value)
        assert time.monotonic() - start < 5.0, "the body must not be read at all"
        c.close()
    finally:
        srv.close()


def test_a_result_frame_larger_than_any_legal_frame_is_refused():
    headers = ["Content-Length: %d" % (C.MAX_RESULT_FRAME_BYTES + 1)]
    srv = RawServer(lambda m, p, h: http_response(b"", headers=headers))
    try:
        c = C.Sam3Client.from_runtime_info(srv.runtime_info(pid=1))
        with pytest.raises(C.ProtocolError):
            c.job_result_frame("j-1")
        c.close()
    finally:
        srv.close()
    # ... while the largest legal frame fits.
    assert C.MAX_RESULT_FRAME_BYTES >= (C.Limits.MAX_INSTANCES * C.Limits.MAX_IMAGE_SIDE ** 2
                                        + C.RESULT_PREFIX_SIZE)


def test_json_responses_have_their_own_smaller_ceiling():
    assert C.MAX_JSON_RESPONSE_BYTES < C.MAX_RESULT_FRAME_BYTES
    headers = ["Content-Length: %d" % (C.MAX_JSON_RESPONSE_BYTES + 1)]
    srv = RawServer(lambda m, p, h: http_response(b"", headers=headers))
    try:
        c = C.Sam3Client.from_runtime_info(srv.runtime_info(pid=1))
        with pytest.raises(C.ProtocolError):
            c.job_status("j-1")
        c.close()
    finally:
        srv.close()


def _one_instance(bbox, **overrides):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    inst = {"instance_id": 0, "score": 0.9, "label": "", "bbox": list(bbox),
            "mask_width": w, "mask_height": h, "blob_offset": 0, "blob_length": w * h}
    return _frame([inst], [bytes(max(0, w * h))], **overrides)


@pytest.mark.parametrize("bbox,overrides,message", [
    ([1000, 0, 1010, 10], {}, "model canvas"),               # past the canvas edge
    ([-5, 0, 5, 10], {}, "model canvas"),                     # negative
    ([0, 0, 10, 10], {"canvas_from_image": {"scale_x": 1e-6, "scale_y": 1.5}}, "off the"),
    ([0, 0, 10, 10], {"canvas_from_image": {"scale_x": 0, "scale_y": 1.5}}, "finite positive"),
    ([0, 0, 10, 10], {"canvas_from_image": {"scale_x": float("nan"), "scale_y": 1.5}},
     "finite positive"),
    ([0, 0, 10, 10], {"image": {"width": 50000, "height": 672}}, "result image"),
    ([0, 0, 10, 10], {"model_canvas": {"width": 10 ** 6, "height": 1008}}, "model canvas"),
])
def test_result_geometry_is_bounded(bbox, overrides, message):
    """A bbox or transform that would place a mask far off the image -- and so
    make GIMP allocate a layer of any size the daemon likes -- is refused."""
    with pytest.raises(C.ProtocolError) as exc:
        C.parse_result_frame(_one_instance(bbox, **overrides))
    assert message in str(exc.value)


def test_a_bbox_a_pixel_or_two_past_the_image_is_tolerated():
    """§8.3 lets the daemon pad the crop; the default frame is 1008x672 in a
    1008x1008 canvas with scale_y 1.5."""
    result = C.parse_result_frame(_one_instance([0, 1000, 10, 1008]))
    assert len(result) == 1


def test_too_many_instances_or_too_big_a_header_are_refused():
    many = [{"instance_id": i, "score": 0.5, "label": "", "bbox": [0, 0, 1, 1],
             "mask_width": 1, "mask_height": 1, "blob_offset": i, "blob_length": 1}
            for i in range(C.Limits.MAX_INSTANCES + 1)]
    with pytest.raises(C.ProtocolError):
        C.parse_result_frame(_frame(many, [b"\x80"] * len(many)))
    huge = MAGIC + struct.pack("<I", C.MAX_RESULT_HEADER_BYTES + 1) + b"{}"
    with pytest.raises(C.ProtocolError) as exc:
        C.parse_result_frame(huge)
    assert "exceeds" in str(exc.value)


def test_a_result_must_describe_the_image_that_was_uploaded(cl, uploaded, daemon):
    original = daemon._header

    def lying_header(job, instances):
        header = original(job, instances)
        header["image"] = {"width": 1008, "height": 1008}
        return header

    daemon._header = lying_header
    with pytest.raises(C.ProtocolError) as exc:
        cl.run_text(uploaded.image_id, "bus")
    assert "uploaded" in str(exc.value)


# =========================================================================== #
# waits fail on a stall, not on a wall clock
# =========================================================================== #
def test_prompt_waits_have_no_wall_clock_limit_by_default():
    import inspect

    for name in ("run_text", "run_points", "wait_for_job", "wait_for_image"):
        params = inspect.signature(getattr(C.Sam3Client, name)).parameters
        assert params["timeout"].default is None, name
        assert params["stall_timeout"].default == C.DEFAULT_STALL_TIMEOUT == 600.0, name


def test_a_slow_job_that_keeps_moving_is_waited_for(rgb_image):
    """Every stage takes longer than a poll, the whole job far longer than the
    stall window -- but something always moves, so it is never abandoned."""
    d = FakeDaemon(stage_delay=0.25)
    try:
        c = d.client()
        width, height, pixels = rgb_image
        acc = c.upload_image(pixels, width, height)
        c.wait_for_image(acc, stall_timeout=2.0)
        result = c.run_text(acc.image_id, "bus", stall_timeout=0.6, poll_wait=0.1)
        assert result is not None and len(result) >= 1
        c.close()
    finally:
        d.close()


def test_a_job_that_stops_moving_times_out_as_a_stall(cl, uploaded, daemon):
    daemon.gate.clear()
    try:
        acc = cl.prompt_text(uploaded.image_id, "cat")
        start = time.monotonic()
        with pytest.raises(C.JobTimeout) as exc:
            cl.wait_for_job(acc.job_id, stall_timeout=0.5, poll_wait=0.1)
        assert exc.value.stalled is True
        assert exc.value.job_id == acc.job_id
        assert "no progress" in str(exc.value)
        assert time.monotonic() - start < 5.0
    finally:
        daemon.gate.set()
    # The job was not lost: waiting again picks the finished result up.
    assert cl.wait_for_job(acc.job_id) is not None


def test_an_explicit_overall_timeout_still_applies(cl, uploaded, daemon):
    daemon.gate.clear()
    try:
        acc = cl.prompt_text(uploaded.image_id, "cat")
        with pytest.raises(C.JobTimeout) as exc:
            cl.wait_for_job(acc.job_id, timeout=0.3, stall_timeout=60.0, poll_wait=0.1)
        assert exc.value.stalled is False
    finally:
        daemon.gate.set()


def _cancel(daemon, job_id, error=None):
    with daemon.lock:
        job = daemon.jobs[job_id]
        daemon.queue.remove(job_id)
        job.update(state="cancelled", stage="cancelled", error=error, finished_at=time.time())
        daemon.lock.notify_all()


def test_a_job_cancelled_because_the_daemon_is_leaving_is_a_transport_error(cl, uploaded, daemon):
    """Not "superseded, show nothing": the caller has to reconnect."""
    daemon.gate.clear()
    try:
        acc = cl.prompt_text(uploaded.image_id, "cat")
        _cancel(daemon, acc.job_id, {"code": "shutting_down", "message": "bye", "detail": {}})
        with pytest.raises(C.DaemonUnavailable) as exc:
            cl.wait_for_job(acc.job_id)
        assert "shutting down" in str(exc.value)
    finally:
        daemon.gate.set()


def test_a_job_cancelled_for_any_other_reason_is_still_just_none(cl, uploaded, daemon):
    daemon.gate.clear()
    try:
        acc = cl.prompt_text(uploaded.image_id, "cat")
        _cancel(daemon, acc.job_id)
        assert cl.wait_for_job(acc.job_id) is None
    finally:
        daemon.gate.set()


# =========================================================================== #
# request ids the caller chooses
# =========================================================================== #
def test_an_explicit_request_id_becomes_the_latest(cl, uploaded):
    cl.next_request_id(uploaded.image_id)
    cl.prompt_text(uploaded.image_id, "cat", request_id="dialog-7")
    assert cl.latest_request_id(uploaded.image_id) == "dialog-7"
    cl.prompt_points(uploaded.image_id, [(5.0, 5.0)], request_id="dialog-8")
    assert cl.latest_request_id(uploaded.image_id) == "dialog-8"


def test_results_for_an_older_explicit_id_are_dropped(cl, uploaded, daemon):
    daemon.gate.clear()
    first = cl.prompt_text(uploaded.image_id, "cat", request_id="mine-1")
    cl.prompt_text(uploaded.image_id, "dog", request_id="mine-2")
    daemon.gate.set()
    assert cl.wait_for_job(first.job_id, image_id=uploaded.image_id, request_id="mine-1") is None
    result = cl.run_text(uploaded.image_id, "bird", request_id="mine-3")
    assert result is not None and result.request_id == "mine-3"


def test_an_invalid_explicit_id_is_refused_before_anything_is_sent(cl, uploaded, daemon):
    before = len(daemon.request_log)
    with pytest.raises(ValueError):
        cl.run_points(uploaded.image_id, [(1.0, 1.0)], request_id="has spaces")
    assert len(daemon.request_log) == before
    assert cl.latest_request_id(uploaded.image_id) != "has spaces"


# =========================================================================== #
# addresses: unspecified binds and IPv6
# =========================================================================== #
@pytest.mark.parametrize("recorded,dialled", [
    ("0.0.0.0", "127.0.0.1"), ("::", "::1"), ("[::]", "::1"), ("", "127.0.0.1"),
    ("[::1]", "::1"), ("127.0.0.1", "127.0.0.1"), ("gpu-box", "gpu-box"),
])
def test_the_address_dialled_for_a_recorded_host(recorded, dialled):
    assert C.Sam3Client(recorded, 1, "t").host == dialled


def test_a_daemon_that_recorded_an_unspecified_bind_is_reached_on_loopback(daemon):
    info = daemon.runtime_info(host="0.0.0.0")
    c = C.Sam3Client.from_runtime_info(info)
    assert c.hello().api_version == C.API_VERSION
    c.close()


def _ipv6_loopback() -> bool:
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        try:
            s.bind(("::1", 0))
        finally:
            s.close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _ipv6_loopback(), reason="no IPv6 loopback here")
def test_ipv6_host_header_is_bracketed():
    d = FakeDaemon(host="::1")
    try:
        c = d.client()
        assert c.hello().api_version == C.API_VERSION     # the Host guard let it through
        assert d.seen_hosts[-1] == "[::1]:%d" % d.port
        assert c.base_url == "http://[::1]:%d" % d.port
        c.close()
    finally:
        d.close()


# =========================================================================== #
# the token stays out of messages
# =========================================================================== #
def test_an_echoed_token_is_not_repeated_in_errors():
    token = "s3cr3t-" + "x" * 36

    def echo(method, path, headers):
        body = json.dumps({"error": {"code": "bad_request",
                                     "message": "you sent " + headers.get("authorization", ""),
                                     "detail": {}}}).encode("utf-8")
        return http_response(body, "400 Bad Request")

    srv = RawServer(echo)
    try:
        c = C.Sam3Client.from_runtime_info(srv.runtime_info(pid=1, token=token))
        with pytest.raises(C.ApiError) as exc:
            c.status()
        assert token not in str(exc.value) and token not in exc.value.message
        c.close()
    finally:
        srv.close()
