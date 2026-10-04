"""HTTP-level tests for the sam3gimpd daemon.

Every test drives a **real server on a real ephemeral port** with
``http.client`` -- the same stdlib client the GIMP plug-in is restricted to --
because the thing under test is a wire contract, not a set of Python calls.

Two engines are used:

``FakeEngine``  a controllable :class:`~sam3gimpd.engines.base.BaseEngine` with no
                sleeps, plus gates for the slow-encode and failure paths.
``StubEngine``  the real ``--stub`` engine at zero latency, so the happy path is
                an integration test rather than a mock conversation.

Nothing here needs torch, weights, a GPU or GIMP.
"""

from __future__ import annotations

import gc
import hashlib
import hmac
import http.client
import json
import logging
import os
import secrets
import socket
import struct
import threading
import time
import types
import weakref

import pytest

import sam3gimpd
import sam3gimpd.server as server_mod
from sam3gimpd import paths
from sam3gimpd.engines.base import (
    BaseEngine,
    EngineInfo,
    ImageData,
    EncodedImage,
    PromptResult,
    RawInstance,
)
from sam3gimpd.engines.stub import StubEngine
from sam3gimpd.server import (
    CONTENT_TYPE_RESULT,
    DRAIN_LIMIT_BYTES,
    InstanceLock,
    Sam3dServer,
    host_header_is_loopback,
    parse_text_prompt,
    pid_alive,
)
from sam3gimpd.types import (
    API_VERSION,
    DEFAULT_SCORE_THRESHOLD,
    ApiError,
    BBox,
    ErrorCode,
    JobState,
    Limits,
    MaskInstance,
    RESULT_MAGIC,
    RESULT_PREFIX_SIZE,
    Size,
    unpack_result,
)


# --------------------------------------------------------------------------- #
# engines under our control
# --------------------------------------------------------------------------- #
def _soft_blob(width: int, height: int) -> bytes:
    """A gradient, so a client's threshold slider has something to bite on."""
    return bytes(((x * 255) // max(1, width - 1)) for _ in range(height)
                 for x in range(width))


class FakeEngine(BaseEngine):
    """Deterministic, instant, and rigged for the paths a real engine cannot be."""

    MODE = "stub"

    def __init__(self, canvas_side: int = 256, capabilities=("pcs", "pvs")) -> None:
        self._canvas_side = canvas_side
        self._capabilities = list(capabilities)
        self.encode_gate: "threading.Event | None" = None
        self.encode_error: "BaseException | None" = None
        self.prompt_error: "BaseException | None" = None
        self.encoded_ids = []
        self.closed = False

    @property
    def canvas_side(self) -> int:
        return self._canvas_side

    def describe(self) -> EngineInfo:
        return EngineInfo(mode=self.MODE, device="stub", dtype="float32",
                          capabilities=list(self._capabilities),
                          torch_available=False, weights_available=False,
                          model_canvas=Size(self._canvas_side, self._canvas_side),
                          models_loaded=["pcs"])

    def encode_image(self, image: ImageData, progress=None) -> EncodedImage:
        if self.encode_gate is not None:
            self.encode_gate.wait(10.0)
        if self.encode_error is not None:
            raise self.encode_error
        canvas, transform = self.canvas_for(image.size)
        self.encoded_ids.append(image.image_id)
        if progress is not None:
            progress(0.30, "encoding")
        return EncodedImage(image_id=image.image_id, image=image.size, canvas=canvas,
                            transform=transform, pixels=image.pixels,
                            parts={"pcs": True}, bytes_estimate=1234)

    def _result(self, encoded, engine, label, count, prompt_meta):
        """Instances placed *within* the encoded canvas.

        These coordinates used to be hardcoded (10,12)-(30,36) and fitted the
        old square 256x256 canvas. Now the canvas is the uploaded image, a
        64x32 upload made bbox (20,24,41,40) overflow, and the frame packer
        rejected it -- a fixture that only worked because the canvas was
        always big and always square.
        """
        cw, ch = int(encoded.canvas.width), int(encoded.canvas.height)
        instances = []
        for i in range(count):
            w = max(2, min(20 + i, max(2, cw // 3)))
            h = max(2, min(15 + i, max(2, ch // 3)))
            x0 = min((i + 1) * max(1, cw // (count + 2)), max(0, cw - w))
            y0 = min((i + 1) * max(1, ch // (count + 2)), max(0, ch - h))
            instances.append(RawInstance(score=0.9 - 0.1 * i,
                                         bbox=BBox(x0, y0, x0 + w, y0 + h),
                                         mask=_soft_blob(w, h), label=label))
        return PromptResult(engine=engine, instances=instances, image=encoded.image,
                            canvas=encoded.canvas, transform=encoded.transform,
                            prompt=prompt_meta, truncated=False, elapsed_ms=1.0)

    def prompt_text(self, encoded, prompt, progress=None) -> PromptResult:
        if self.prompt_error is not None:
            raise self.prompt_error
        if progress is not None:
            progress(0.85, "decoding")
        count = min(3, int(prompt.max_instances))
        return self._result(encoded, "pcs", prompt.text, count,
                            self.text_prompt_dict(prompt))

    def prompt_points(self, encoded, prompt, progress=None) -> PromptResult:
        if self.prompt_error is not None:
            raise self.prompt_error
        return self._result(encoded, "pvs", "", min(2, int(prompt.max_instances)),
                            self.point_prompt_dict(prompt))

    def close(self) -> None:
        self.closed = True


class LooseEngine(FakeEngine):
    """Returns ``(instances, blobs)`` instead of a ``PromptResult``.

    The server's adapter must still produce a conforming frame -- sorted by
    score, ids renumbered -- which is what keeps a loosely written third-party
    engine from emitting something the plug-in cannot parse.
    """

    def prompt_text(self, encoded, prompt, progress=None):
        # Sized from the canvas: the second box used to be a fixed (40,40,12,10)
        # that only fitted while the canvas was a 256 square.
        cw, ch = int(encoded.canvas.width), int(encoded.canvas.height)
        a = (1, 1, max(2, cw // 6), max(2, ch // 6))
        b = (max(2, cw // 2), max(2, ch // 2), max(2, cw // 5), max(2, ch // 5))
        out = []
        for score, (x0, y0, w, h) in ((0.20, a), (0.95, b)):
            out.append((MaskInstance(instance_id=99, score=score,
                                     bbox=BBox(x0, y0, x0 + w, y0 + h),
                                     mask_width=w, mask_height=h,
                                     blob_offset=0, blob_length=0,
                                     label=prompt.text),
                        _soft_blob(w, h)))
        return [i for i, _ in out], [b for _, b in out]


# --------------------------------------------------------------------------- #
# client plumbing
# --------------------------------------------------------------------------- #
class Client:
    """A thin, explicit http.client wrapper -- one connection, keep-alive."""

    def __init__(self, server: Sam3dServer, token: str = None) -> None:
        self.server = server
        self.token = server.token if token is None else token
        self.conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=20)

    def raw(self, method, path, body=None, headers=None, auth=True):
        head = dict(headers or {})
        if auth and "Authorization" not in head:
            head["Authorization"] = "Bearer " + self.token
        if body is not None and "Content-Length" not in head:
            head["Content-Length"] = str(len(body))
        self.conn.request(method, path, body=body, headers=head)
        response = self.conn.getresponse()
        payload = response.read()
        return response.status, dict(response.getheaders()), payload

    def json(self, method, path, body=None, headers=None, auth=True):
        head = dict(headers or {})
        raw_body = None
        if body is not None:
            raw_body = json.dumps(body).encode("utf-8")
            head.setdefault("Content-Type", "application/json")
        status, headers_out, payload = self.raw(method, path, raw_body, head, auth)
        return status, json.loads(payload.decode("utf-8")) if payload else None

    def upload(self, width, height, seed=0, extra=None):
        pixels = bytes(((x * 7 + seed * 31) % 256) for x in range(width * height * 3))
        head = {"Content-Type": "application/octet-stream",
                "X-Width": str(width), "X-Height": str(height)}
        head.update(extra or {})
        status, _, payload = self.raw("POST", "/images", pixels, head)
        return status, json.loads(payload.decode("utf-8")), pixels

    def poll(self, job_id, timeout=15.0, wait=2.0):
        """Long-poll until terminal; returns ``(content_type, payload)``."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status, headers, payload = self.raw("GET", "/jobs/%s?wait=%s"
                                                % (job_id, wait))
            assert status == 200, payload
            if headers.get("Content-Type", "").startswith(CONTENT_TYPE_RESULT):
                return CONTENT_TYPE_RESULT, payload
            body = json.loads(payload.decode("utf-8"))
            if body["state"] in (JobState.DONE, JobState.FAILED,
                                 JobState.SUPERSEDED, JobState.CANCELLED):
                return "json", body
        raise AssertionError("job %s never reached a terminal state" % job_id)

    def close(self):
        self.conn.close()


@pytest.fixture
def serve(tmp_path):
    """Factory starting a real server on 127.0.0.1:0, torn down afterwards."""
    started = []

    def _make(engine=None, **kwargs):
        kwargs.setdefault("runtime_file", str(tmp_path / "runtime.json"))
        kwargs.setdefault("idle_ttl", 0.0)
        kwargs.setdefault("host", "127.0.0.1")
        server = Sam3dServer(engine=engine or FakeEngine(), port=0, **kwargs).start()
        started.append(server)
        return server

    yield _make
    for server in started:
        server.close()


@pytest.fixture
def server(serve):
    return serve()


def _eventually(predicate, timeout=5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def client(server):
    c = Client(server)
    yield c
    c.close()


# --------------------------------------------------------------------------- #
# transport and auth (API.md sections 1 and 2)
# --------------------------------------------------------------------------- #
def test_server_binds_an_ephemeral_loopback_port(server):
    assert server.port > 0
    assert server.host == "127.0.0.1"
    sock = socket.create_connection(("127.0.0.1", server.port), timeout=2)
    sock.close()


def test_every_endpoint_requires_the_bearer_token(client):
    for method, path in (("GET", "/hello"), ("GET", "/status"), ("POST", "/images"),
                         ("GET", "/jobs/j-1"), ("DELETE", "/images/x"),
                         ("POST", "/shutdown")):
        status, _, payload = client.raw(method, path, auth=False)
        assert status == 401, (method, path)
        body = json.loads(payload.decode("utf-8"))
        assert body["error"]["code"] == ErrorCode.UNAUTHORIZED


def test_a_wrong_or_malformed_token_is_rejected(server):
    bad = Client(server, token="not-the-token")
    try:
        assert bad.json("GET", "/hello")[0] == 401
        status, _, _ = bad.raw("GET", "/hello", headers={"Authorization": "Basic x"},
                               auth=False)
        assert status == 401
        status, _, _ = bad.raw("GET", "/hello", headers={"Authorization": "Bearer "},
                               auth=False)
        assert status == 401
    finally:
        bad.close()


def test_the_token_never_appears_in_a_response(client):
    status, headers, payload = client.raw("GET", "/status")
    assert status == 200
    assert client.token not in payload.decode("utf-8")
    assert client.token not in json.dumps(headers)


def test_non_loopback_host_headers_are_refused(client):
    status, body = client.json("GET", "/hello", headers={"Host": "evil.example.com"})
    assert status == 403
    assert body["error"]["code"] == ErrorCode.FORBIDDEN_HOST
    # ...but the loopback spellings are all fine.
    for host in ("127.0.0.1:%d" % client.server.port, "localhost", "[::1]:8080"):
        assert client.json("GET", "/hello", headers={"Host": host})[0] == 200


def test_a_non_loopback_bind_skips_the_host_guard(serve):
    server = serve(host="0.0.0.0")
    client = Client(server)
    client.conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    try:
        assert client.json("GET", "/hello", headers={"Host": "example.com"})[0] == 200
    finally:
        client.close()


def test_responses_carry_version_headers_and_a_content_length(client):
    status, headers, payload = client.raw("GET", "/hello")
    assert status == 200
    assert headers["X-Sam3-Api"] == API_VERSION
    assert headers["X-Sam3-Version"]
    assert int(headers["Content-Length"]) == len(payload)
    assert "Transfer-Encoding" not in headers
    assert headers["Content-Type"].startswith("application/json")


def test_keep_alive_reuses_one_connection(client):
    for _ in range(5):
        assert client.json("GET", "/hello")[0] == 200


def test_chunked_request_bodies_are_refused(client):
    status, body = client.json("POST", "/shutdown", headers={
        "Transfer-Encoding": "chunked", "Content-Length": "0"})
    assert status == 400
    assert body["error"]["code"] == ErrorCode.BAD_REQUEST


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
def test_unknown_paths_are_404_and_wrong_verbs_are_405(client):
    status, body = client.json("GET", "/nope")
    assert status == 404 and body["error"]["code"] == ErrorCode.NOT_FOUND

    status, headers, payload = client.raw("DELETE", "/hello")
    assert status == 405
    assert headers["Allow"] == "GET"
    assert json.loads(payload)["error"]["code"] == ErrorCode.METHOD_NOT_ALLOWED


def test_every_error_body_is_the_documented_envelope(client):
    for method, path in (("GET", "/nope"), ("DELETE", "/hello"),
                         ("GET", "/jobs/j-missing"), ("DELETE", "/images/missing")):
        status, body = client.json(method, path)
        assert status >= 400
        assert set(body) == {"error"}
        assert set(body["error"]) == {"code", "message", "detail"}
        assert isinstance(body["error"]["detail"], dict)


# --------------------------------------------------------------------------- #
# GET /hello and GET /status (sections 6.1, 6.7)
# --------------------------------------------------------------------------- #
def test_hello_reports_the_engine_and_the_limits(client):
    status, body = client.json("GET", "/hello")
    assert status == 200
    assert body["api_version"] == API_VERSION
    assert body["engine_mode"] == "stub"
    assert body["device"] == "stub"
    assert body["torch_available"] is False
    assert body["weights_available"] is False
    assert "pcs" in body["capabilities"] and "pvs" in body["capabilities"]
    # /hello reports the engine's nominal working resolution, which stays
    # square; the canvas that matters is per-image (see the upload test).
    assert body["model_canvas"] == {"width": 256, "height": 256}
    assert body["pid"] == os.getpid()
    assert body["uptime_s"] >= 0.0
    limits = body["limits"]
    assert limits["max_image_side"] == Limits.MAX_IMAGE_SIDE
    assert limits["max_upload_bytes"] == Limits.MAX_UPLOAD_BYTES
    assert limits["max_text_chars"] == Limits.MAX_TEXT_CHARS
    assert limits["max_points"] == Limits.MAX_POINTS
    assert limits["max_long_poll_seconds"] == Limits.MAX_LONG_POLL_SECONDS


def test_an_incompatible_client_api_major_is_rejected(client):
    status, body = client.json("GET", "/hello", headers={"X-Sam3-Api": "2.0"})
    assert status == 400
    assert body["error"]["code"] == ErrorCode.VERSION_MISMATCH
    # A matching major with a different minor is fine: minors are additive.
    assert client.json("GET", "/hello", headers={"X-Sam3-Api": "1.7"})[0] == 200


def test_hello_carries_a_source_build_hash(client):
    """Eight hex digits hashing the package's .py files, so a client shipping a
    copy of the daemon can tell a stale one without a version bump."""
    import sam3gimpd
    status, body = client.json("GET", "/hello")
    assert status == 200
    assert body["build"] == sam3gimpd.build_hash()
    assert len(body["build"]) == 8 and int(body["build"], 16) >= 0


def test_status_is_a_superset_of_hello(client):
    _, hello = client.json("GET", "/hello")
    status, body = client.json("GET", "/status")
    assert status == 200
    for key in hello:
        assert key in body
    assert body["images"] == []
    assert body["jobs_queued"] == 0
    assert body["cache_limit"] == 3
    assert body["last_error"] is None
    assert body["models_loaded"] == ["pcs"]
    assert set(body["paths"]) >= {"base", "runtime_file", "server_log", "crash_log",
                                  "hf_home"}
    assert "rss_bytes" in body["memory"]


def test_status_lists_cached_images(client):
    status, accepted, _ = client.upload(32, 24)
    assert status == 202
    client.poll(accepted["job_id"])
    _, body = client.json("GET", "/status")
    assert [entry["image_id"] for entry in body["images"]] == [accepted["image_id"]]
    entry = body["images"][0]
    assert entry["width"] == 32 and entry["height"] == 24
    assert entry["state"] == JobState.DONE
    assert entry["bytes_estimate"] > 0


# --------------------------------------------------------------------------- #
# POST /images (sections 6.2, 7)
# --------------------------------------------------------------------------- #
def test_upload_accepts_pixels_and_reports_the_geometry(client):
    status, body, _ = client.upload(64, 32)
    assert status == 202
    assert len(body["image_id"]) == 32 and int(body["image_id"], 16) >= 0
    assert body["cached"] is False
    assert body["image"] == {"width": 64, "height": 32}
    # The canvas is the uploaded image now, not a square.
    assert body["model_canvas"] == {"width": 64, "height": 32}
    # Identity: masks are post-processed straight to the uploaded image, so
    # there is no squash to invert. This asserted (4.0, 8.0) -- unequal scales,
    # which silently distorted every mask on a non-square image.
    assert body["canvas_from_image"] == {"scale_x": 1.0, "scale_y": 1.0,
                                         "offset_x": 0.0, "offset_y": 0.0}
    assert body["state"] in (JobState.QUEUED, JobState.RUNNING, JobState.DONE)


def test_identical_pixels_hit_the_cache_and_keep_their_job_id(client):
    _, first, _ = client.upload(32, 24, seed=1)
    client.poll(first["job_id"])
    _, second, _ = client.upload(32, 24, seed=1)
    assert second["image_id"] == first["image_id"]
    assert second["cached"] is True
    assert second["job_id"] == first["job_id"]
    assert second["state"] == JobState.DONE
    # ...and the job it names is still fetchable.
    assert client.raw("GET", "/jobs/%s?meta=1" % second["job_id"])[0] == 200


def test_different_pixels_get_different_ids(client):
    _, a, _ = client.upload(32, 24, seed=1)
    _, b, _ = client.upload(32, 24, seed=2)
    assert a["image_id"] != b["image_id"]


def test_upload_rejects_missing_headers_bad_dimensions_and_size_mismatch(client):
    body = bytes(16 * 16 * 3)
    common = {"Content-Type": "application/octet-stream"}

    status, _, payload = client.raw("POST", "/images", body,
                                    dict(common, **{"X-Height": "16"}))
    assert status == 400
    assert json.loads(payload)["error"]["code"] == ErrorCode.MISSING_HEADER

    for width in ("8", "2000", "abc"):
        status, _, payload = client.raw("POST", "/images", body,
                                        dict(common, **{"X-Width": width,
                                                        "X-Height": "16"}))
        assert status == 400
        assert json.loads(payload)["error"]["code"] == ErrorCode.BAD_DIMENSIONS
        client.conn.close()

    status, _, payload = client.raw("POST", "/images", b"short",
                                    dict(common, **{"X-Width": "16", "X-Height": "16"}))
    assert status == 400
    error = json.loads(payload)["error"]
    assert error["code"] == ErrorCode.PAYLOAD_SIZE_MISMATCH
    assert error["detail"] == {"expected": 768, "got": 5}


def test_upload_rejects_the_wrong_media_type(client):
    status, _, payload = client.raw("POST", "/images", bytes(768),
                                    {"Content-Type": "text/plain",
                                     "X-Width": "16", "X-Height": "16"})
    assert status == 415
    assert json.loads(payload)["error"]["code"] == ErrorCode.UNSUPPORTED_MEDIA_TYPE


def test_an_oversized_body_is_rejected_without_being_read(client):
    """The limit is checked from Content-Length, so no 3 MB is transferred."""
    status, _, payload = client.raw(
        "POST", "/images", None,
        {"Content-Type": "application/octet-stream",
         "X-Width": "1008", "X-Height": "1008",
         "Content-Length": str(Limits.MAX_UPLOAD_BYTES + 1)})
    assert status == 413
    assert json.loads(payload)["error"]["code"] == ErrorCode.PAYLOAD_TOO_LARGE


# --------------------------------------------------------------------------- #
# prompting and the binary frame (sections 6.3, 6.4, 8)
# --------------------------------------------------------------------------- #
def _prompt_text(client, image_id, request_id="r-1", **extra):
    body = dict({"request_id": request_id, "text": "yellow school bus"}, **extra)
    return client.json("POST", "/images/%s/text" % image_id, body)


def test_a_text_prompt_returns_a_conforming_binary_frame(client):
    _, accepted, _ = client.upload(64, 32)
    status, job = _prompt_text(client, accepted["image_id"], "r-000007")
    assert status == 202
    assert job["engine"] == "pcs"
    assert job["request_id"] == "r-000007"
    assert job["superseded_job_ids"] == []

    kind, payload = client.poll(job["job_id"])
    assert kind == CONTENT_TYPE_RESULT
    assert payload[:8] == RESULT_MAGIC

    (head_len,) = struct.unpack_from("<I", payload, 8)
    header = json.loads(payload[RESULT_PREFIX_SIZE:RESULT_PREFIX_SIZE + head_len])
    assert header["job_id"] == job["job_id"]
    assert header["request_id"] == "r-000007"
    assert header["image_id"] == accepted["image_id"]
    assert header["engine"] == "pcs"
    assert header["mask_encoding"] == "u8_soft"
    assert header["canvas_from_image"] == accepted["canvas_from_image"]
    assert header["prompt"]["kind"] == "text"
    assert header["truncated"] is False

    # section 16: masks tightly packed in instance order, offsets relative to
    # the blob region, dimensions agreeing with the bbox.
    parsed, blob = unpack_result(payload)
    assert len(blob) == parsed.blob_length
    offset = 0
    scores = []
    for index, inst in enumerate(parsed.instances):
        assert inst.instance_id == index
        assert inst.blob_offset == offset
        assert inst.blob_length == inst.mask_width * inst.mask_height
        assert inst.mask_width == inst.bbox.width
        assert inst.mask_height == inst.bbox.height
        assert 0 <= inst.bbox.x0 < inst.bbox.x1 <= parsed.model_canvas.width
        assert len(inst.mask_bytes(blob)) == inst.blob_length
        offset += inst.blob_length
        scores.append(inst.score)
    assert scores == sorted(scores, reverse=True)
    assert parsed.blob_length == offset
    assert len(payload) == RESULT_PREFIX_SIZE + head_len + parsed.blob_length


def test_the_binary_response_carries_the_documented_headers(client):
    _, accepted, _ = client.upload(32, 24)
    _, job = _prompt_text(client, accepted["image_id"], "r-42")
    client.poll(job["job_id"])
    status, headers, payload = client.raw("GET", "/jobs/%s" % job["job_id"])
    assert status == 200
    assert headers["Content-Type"] == CONTENT_TYPE_RESULT
    assert headers["X-Sam3-Job-State"] == "done"
    assert headers["X-Sam3-Request-Id"] == "r-42"
    assert int(headers["X-Sam3-Header-Length"]) == struct.unpack_from("<I", payload, 8)[0]
    assert int(headers["Content-Length"]) == len(payload)


def test_meta_1_returns_the_json_form_of_a_finished_job(client):
    _, accepted, _ = client.upload(32, 24)
    _, job = _prompt_text(client, accepted["image_id"])
    client.poll(job["job_id"])
    status, body = client.json("GET", "/jobs/%s?meta=1" % job["job_id"])
    assert status == 200
    assert body["state"] == JobState.DONE
    assert body["masks_available"] is True
    assert body["progress"] == 1.0
    assert body["error"] is None
    assert body["result"]["instances"]
    assert body["elapsed_ms"] >= 0.0


def test_a_point_prompt_runs_the_pvs_engine(client):
    _, accepted, _ = client.upload(32, 24)
    status, job = client.json("POST", "/images/%s/points" % accepted["image_id"], {
        "request_id": "r-8",
        "points": [{"x": 10.0, "y": 12.0, "label": 1}, {"x": 3.0, "y": 4.0, "label": 0}],
        "box": [1.0, 2.0, 20.0, 18.0], "multimask": True, "max_instances": 2})
    assert status == 202
    assert job["engine"] == "pvs"
    kind, payload = client.poll(job["job_id"])
    assert kind == CONTENT_TYPE_RESULT
    header, _ = unpack_result(payload)
    assert header.engine == "pvs"
    assert header.prompt["kind"] == "points"
    assert len(header.instances) == 2


def test_prompt_validation_matches_the_contract(client):
    _, accepted, _ = client.upload(32, 24)
    image_id = accepted["image_id"]
    cases = [
        ({"text": "x"}, ErrorCode.BAD_REQUEST),                       # no request_id
        ({"request_id": "bad id!", "text": "x"}, ErrorCode.BAD_REQUEST),
        ({"request_id": "r", "text": "   "}, ErrorCode.BAD_REQUEST),
        ({"request_id": "r", "text": "x" * 513}, ErrorCode.BAD_REQUEST),
        ({"request_id": "r", "text": "x", "score_threshold": 2.0}, ErrorCode.BAD_REQUEST),
        ({"request_id": "r", "text": "x", "max_instances": 0}, ErrorCode.BAD_REQUEST),
        ({"request_id": "r", "text": "x", "max_instances": 999}, ErrorCode.BAD_REQUEST),
        ({"request_id": "r", "text": "x", "boxes": [{"box": [1, 1, 0, 0]}]},
         ErrorCode.BAD_REQUEST),
    ]
    for body, code in cases:
        status, response = client.json("POST", "/images/%s/text" % image_id, body)
        assert status == 400, body
        assert response["error"]["code"] == code, body

    for body in ({"request_id": "r"},
                 {"request_id": "r", "points": [{"x": 1.0}]},
                 {"request_id": "r", "points": [{"x": 1.0, "y": 1.0, "label": 7}]},
                 {"request_id": "r", "points": [{"x": 0.0, "y": 0.0}] * 65},
                 {"request_id": "r", "box": [5.0, 5.0, 1.0, 1.0]},
                 {"request_id": "r", "points": [{"x": 1.0, "y": 1.0}], "max_instances": 9}):
        status, response = client.json("POST", "/images/%s/points" % image_id, body)
        assert status == 400, body
        assert response["error"]["code"] == ErrorCode.BAD_REQUEST


def test_a_malformed_json_body_is_invalid_json(client):
    _, accepted, _ = client.upload(32, 24)
    status, _, payload = client.raw("POST", "/images/%s/text" % accepted["image_id"],
                                    b"{not json", {"Content-Type": "application/json"})
    assert status == 400
    assert json.loads(payload)["error"]["code"] == ErrorCode.INVALID_JSON

    status, _, payload = client.raw("POST", "/images/%s/text" % accepted["image_id"],
                                    b"[1,2]", {"Content-Type": "application/json"})
    assert status == 400
    assert json.loads(payload)["error"]["code"] == ErrorCode.INVALID_JSON


def test_prompting_an_unknown_image_is_404(client):
    status, body = _prompt_text(client, "0" * 32)
    assert status == 404
    assert body["error"]["code"] == ErrorCode.IMAGE_NOT_FOUND


def test_max_instances_clips_and_flags_truncation(serve):
    """The server enforces the clip even when the engine does not."""
    client = Client(serve(LooseEngine()))
    try:
        _, accepted, _ = client.upload(32, 24)
        _, job = _prompt_text(client, accepted["image_id"], max_instances=1)
        kind, payload = client.poll(job["job_id"])
        header, _ = unpack_result(payload)
        assert header.truncated is True
        assert len(header.instances) == 1
        assert header.instances[0].score == pytest.approx(0.95)   # the best one
        assert header.instances[0].instance_id == 0               # renumbered
    finally:
        client.close()


def test_a_loosely_typed_engine_result_is_still_a_conforming_frame(serve):
    client = Client(serve(LooseEngine()))
    try:
        _, accepted, _ = client.upload(32, 24)
        _, job = _prompt_text(client, accepted["image_id"])
        kind, payload = client.poll(job["job_id"])
        header, blob = unpack_result(payload)          # validates every invariant
        assert [i.instance_id for i in header.instances] == [0, 1]
        assert [round(i.score, 2) for i in header.instances] == [0.95, 0.20]
        assert header.prompt["kind"] == "text"
    finally:
        client.close()


# --------------------------------------------------------------------------- #
# jobs, failures and supersession (sections 6.5, 10, 11)
# --------------------------------------------------------------------------- #
def test_an_engine_failure_is_a_failed_job_not_an_http_error(serve):
    engine = FakeEngine()
    engine.prompt_error = ApiError(ErrorCode.INFERENCE_FAILED, "forward pass blew up",
                                   {"why": "test"})
    client = Client(serve(engine))
    try:
        _, accepted, _ = client.upload(32, 24)
        status, job = _prompt_text(client, accepted["image_id"])
        assert status == 202                       # accepting the job still succeeded
        kind, body = client.poll(job["job_id"])
        assert kind == "json"
        assert body["state"] == JobState.FAILED
        assert body["error"]["code"] == ErrorCode.INFERENCE_FAILED
        assert body["error"]["detail"]["why"] == "test"
        assert body["masks_available"] is False
        _, status_body = client.json("GET", "/status")
        assert status_body["last_error"]["code"] == ErrorCode.INFERENCE_FAILED
    finally:
        client.close()


def test_a_failed_encode_makes_prompts_409(serve):
    engine = FakeEngine()
    engine.encode_error = ApiError(ErrorCode.MODEL_LOAD_FAILED, "no checkpoint")
    client = Client(serve(engine))
    try:
        _, accepted, _ = client.upload(32, 24)
        kind, body = client.poll(accepted["job_id"])
        assert kind == "json" and body["state"] == JobState.FAILED

        status, response = _prompt_text(client, accepted["image_id"])
        assert status == 409
        assert response["error"]["code"] == ErrorCode.IMAGE_NOT_READY
        assert response["error"]["detail"]["error"]["code"] == ErrorCode.MODEL_LOAD_FAILED
    finally:
        client.close()


def test_a_newer_prompt_supersedes_the_queued_one(serve):
    engine = FakeEngine()
    engine.encode_gate = threading.Event()          # hold the worker on the encode
    client = Client(serve(engine))
    try:
        _, accepted, _ = client.upload(32, 24)
        image_id = accepted["image_id"]
        _, first = _prompt_text(client, image_id, "r-1")
        _, second = _prompt_text(client, image_id, "r-2")
        assert second["superseded_job_ids"] == [first["job_id"]]

        _, body = client.json("GET", "/jobs/%s" % first["job_id"])
        assert body["state"] == JobState.SUPERSEDED
        assert body["superseded_by"] == second["job_id"]
        assert body["request_id"] == "r-1"

        engine.encode_gate.set()
        kind, payload = client.poll(second["job_id"])
        assert kind == CONTENT_TYPE_RESULT
        header, _ = unpack_result(payload)
        assert header.request_id == "r-2"
    finally:
        engine.encode_gate.set()
        client.close()


def test_deleting_an_image_cancels_queued_jobs_and_frees_the_cache(serve):
    engine = FakeEngine()
    engine.encode_gate = threading.Event()
    client = Client(serve(engine))
    try:
        _, accepted, _ = client.upload(32, 24)
        _, job = _prompt_text(client, accepted["image_id"])
        status, body = client.json("DELETE", "/images/%s" % accepted["image_id"])
        assert status == 200
        assert body == {"image_id": accepted["image_id"], "deleted": True,
                        "cancelled_job_ids": [job["job_id"]]}

        _, job_body = client.json("GET", "/jobs/%s" % job["job_id"])
        assert job_body["state"] == JobState.CANCELLED

        status, body = client.json("DELETE", "/images/%s" % accepted["image_id"])
        assert status == 404 and body["error"]["code"] == ErrorCode.IMAGE_NOT_FOUND

        _, status_body = client.json("GET", "/status")
        assert status_body["images"] == []
    finally:
        engine.encode_gate.set()
        client.close()


def test_queue_position_and_progress_are_reported_while_waiting(serve):
    engine = FakeEngine()
    engine.encode_gate = threading.Event()
    client = Client(serve(engine, cache_size=8))
    try:
        _, first, _ = client.upload(32, 24, seed=1)
        _, second, _ = client.upload(32, 24, seed=2)
        _, body = client.json("GET", "/jobs/%s" % second["job_id"])
        assert body["state"] == JobState.QUEUED
        assert body["queue_position"] == 0        # next to run
        assert body["progress"] == 0.0
        assert body["stage"] == "queued"
        assert body["masks_available"] is False
    finally:
        engine.encode_gate.set()
        client.close()


def test_a_long_poll_times_out_with_200_and_the_unchanged_status(serve):
    engine = FakeEngine()
    engine.encode_gate = threading.Event()
    client = Client(serve(engine))
    try:
        _, accepted, _ = client.upload(32, 24)
        _, job = _prompt_text(client, accepted["image_id"])
        started = time.monotonic()
        status, body = client.json("GET", "/jobs/%s?wait=0.5" % job["job_id"])
        elapsed = time.monotonic() - started
        assert status == 200
        assert 0.4 <= elapsed < 5.0
        assert body["state"] == JobState.QUEUED
    finally:
        engine.encode_gate.set()
        client.close()


def test_a_long_poll_does_not_block_other_connections(serve):
    """API.md section 1: the client polls on one connection and posts on another."""
    engine = FakeEngine()
    engine.encode_gate = threading.Event()
    poller = Client(serve(engine))
    other = Client(poller.server)
    try:
        _, accepted, _ = poller.upload(32, 24)
        _, job = _prompt_text(poller, accepted["image_id"])
        done = threading.Event()

        def long_poll():
            poller.raw("GET", "/jobs/%s?wait=3" % job["job_id"])
            done.set()

        thread = threading.Thread(target=long_poll)
        thread.start()
        time.sleep(0.15)
        started = time.monotonic()
        assert other.json("GET", "/hello")[0] == 200
        assert time.monotonic() - started < 1.0        # not serialised behind the poll
        assert not done.is_set()
        engine.encode_gate.set()
        thread.join(10.0)
        assert done.is_set()
    finally:
        engine.encode_gate.set()
        other.close()
        poller.close()


def test_an_unknown_job_id_is_404(client):
    status, body = client.json("GET", "/jobs/j-000999-dead")
    assert status == 404
    assert body["error"]["code"] == ErrorCode.JOB_NOT_FOUND


def test_a_bad_wait_or_meta_parameter_is_a_bad_request(client):
    _, accepted, _ = client.upload(32, 24)
    status, body = client.json("GET", "/jobs/%s?wait=soon" % accepted["job_id"])
    assert status == 400 and body["error"]["code"] == ErrorCode.BAD_REQUEST
    status, body = client.json("GET", "/jobs/%s?meta=maybe" % accepted["job_id"])
    assert status == 400 and body["error"]["code"] == ErrorCode.BAD_REQUEST


def test_the_queue_refuses_work_when_it_is_full(serve):
    engine = FakeEngine()
    engine.encode_gate = threading.Event()
    client = Client(serve(engine, max_queue=2, cache_size=8))
    try:
        for seed in (1, 2, 3):                 # 1 runs, 2 and 3 queue
            status, _, _ = client.upload(32, 24, seed=seed)
            assert status == 202
        status, body, _ = client.upload(32, 24, seed=4)
        assert status == 503
        assert body["error"]["code"] == ErrorCode.QUEUE_FULL
    finally:
        engine.encode_gate.set()
        client.close()


# --------------------------------------------------------------------------- #
# the real stub engine, end to end (section 14)
# --------------------------------------------------------------------------- #
def test_the_stub_engine_serves_a_full_session(serve):
    client = Client(serve(StubEngine(latency_scale=0.0)))
    try:
        _, hello = client.json("GET", "/hello")
        assert hello["engine_mode"] == "stub"
        assert hello["model_canvas"] == {"width": 1008, "height": 1008}

        _, accepted, _ = client.upload(64, 48)
        _, job = _prompt_text(client, accepted["image_id"], "r-1")
        kind, payload = client.poll(job["job_id"])
        assert kind == CONTENT_TYPE_RESULT
        header, blob = unpack_result(payload)
        assert 1 <= len(header.instances) <= 8
        assert all(0.1 <= i.score <= 1.0 for i in header.instances)

        # Soft masks, not binary: a threshold slider must have something to do.
        mask = header.instances[0].mask_bytes(blob)
        assert len(set(mask)) > 2
        assert min(mask) < 128 <= max(mask)

        # Determinism: the same prompt on the same image is byte-identical.
        _, again = _prompt_text(client, accepted["image_id"], "r-2")
        _, payload2 = client.poll(again["job_id"])
        header2, blob2 = unpack_result(payload2)
        assert blob2 == blob
        assert [i.bbox.to_list() for i in header2.instances] == \
               [i.bbox.to_list() for i in header.instances]

        # ...and a different prompt is visibly different.
        _, other = client.json("POST", "/images/%s/text" % accepted["image_id"],
                               {"request_id": "r-3", "text": "a red bicycle"})
        _, payload3 = client.poll(other["job_id"])
        header3, blob3 = unpack_result(payload3)
        assert blob3 != blob
    finally:
        client.close()


# --------------------------------------------------------------------------- #
# lifecycle: runtime.json, shutdown, the lockfile (sections 3, 6.8)
# --------------------------------------------------------------------------- #
def test_runtime_json_is_written_after_the_socket_is_listening(server, tmp_path):
    path = tmp_path / "runtime.json"
    assert path.exists()
    info = json.loads(path.read_text(encoding="utf-8"))
    for key in ("port", "token", "pid", "version", "started_at"):
        assert key in info
    assert info["port"] == server.port
    assert info["token"] == server.token
    assert info["pid"] == os.getpid()
    assert len(info["token"]) == 43
    assert info["api_version"] == API_VERSION
    # The file exists only because the port is already connectable.
    socket.create_connection(("127.0.0.1", info["port"]), timeout=2).close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes only")
def test_runtime_json_is_user_only(server, tmp_path):
    mode = os.stat(tmp_path / "runtime.json").st_mode & 0o777
    assert mode == 0o600


def test_shutdown_returns_202_then_refuses_everything(serve, tmp_path):
    server = serve()
    client = Client(server)
    try:
        status, body = client.json("POST", "/shutdown", {"grace_ms": 0})
        assert status == 202
        assert body["ok"] is True and body["pid"] == os.getpid()
        assert server.wait_closed(timeout=10.0)
        assert not (tmp_path / "runtime.json").exists()
    finally:
        client.close()


def test_requests_during_shutdown_are_503(serve):
    server = serve()
    client = Client(server)
    try:
        # Take the shutdown flag without letting the server actually close, so
        # the socket is still up when the next request arrives.
        server._shutting_down = True
        status, body = client.json("GET", "/hello")
        assert status == 503
        assert body["error"]["code"] == ErrorCode.SHUTTING_DOWN
    finally:
        client.close()


def test_closing_the_server_closes_the_engine(serve):
    engine = FakeEngine()
    server = serve(engine)
    server.close()
    assert engine.closed is True
    assert server.wait_closed(timeout=1.0)


def test_the_instance_lock_is_exclusive(tmp_path):
    """flock and msvcrt locks belong to the open file, not the process, so a
    second InstanceLock is refused even here; test_the_lock_blocks_a_second_process
    covers the real case of another daemon."""
    first = InstanceLock(tmp_path / "sam3gimpd.lock")
    second = InstanceLock(tmp_path / "sam3gimpd.lock")
    assert first.acquire() is True
    try:
        assert second.acquire() is False
        assert second.held is False
        # A refused acquire must not have clobbered the holder's pid.
        assert (tmp_path / "sam3gimpd.lock").read_text(encoding="utf-8").strip() == str(os.getpid())
    finally:
        first.release()
    assert second.acquire() is True
    second.release()


def test_releasing_the_lock_leaves_no_pid_behind(tmp_path):
    """A pid left in the file after a clean exit names a process that holds
    nothing -- or, once the number is reused, a stranger the launcher might
    signal as a "zombie holder"."""
    lock = InstanceLock(tmp_path / "sam3gimpd.lock")
    assert lock.acquire() is True
    assert (tmp_path / "sam3gimpd.lock").read_text(encoding="utf-8").strip() == str(os.getpid())
    lock.release()
    assert (tmp_path / "sam3gimpd.lock").read_text(encoding="utf-8") == ""


def test_the_lock_blocks_a_second_process(tmp_path):
    import subprocess
    import sys
    import textwrap

    lock = InstanceLock(tmp_path / "sam3gimpd.lock")
    assert lock.acquire() is True
    try:
        code = textwrap.dedent("""
            import sys
            sys.path.insert(0, %r)
            from sam3gimpd.server import InstanceLock
            sys.exit(0 if InstanceLock(%r).acquire() else 3)
        """ % (str(__import__("pathlib").Path(__file__).resolve().parents[2] / "plugin" / "sam3_gimp" / "_daemon"),
               str(tmp_path / "sam3gimpd.lock")))
        result = subprocess.run([sys.executable, "-c", code], timeout=30)
        assert result.returncode == 3
    finally:
        lock.release()


def test_pid_alive_knows_this_process_and_not_a_bogus_one():
    assert pid_alive(os.getpid()) is True
    assert pid_alive(None) is False
    assert pid_alive(0) is False
    assert pid_alive(-1) is False
    # 0x7FFFFFFF is above every plausible pid_max.
    assert pid_alive(0x7FFFFFFF) is False


def test_the_daemon_exits_when_its_parent_disappears(serve):
    """DESIGN.md constraint C: nothing supervises the daemon, so it supervises
    the process it was told to follow."""
    import subprocess
    import sys

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    server = serve(parent_pid=child.pid)
    try:
        assert not server.wait_closed(timeout=0.5)
        child.terminate()
        child.wait(timeout=10)
        assert server.wait_closed(timeout=10.0)
        assert server.exit_reason == "parent-exited"
    finally:
        if child.poll() is None:
            child.kill()


def test_the_daemon_exits_after_the_idle_ttl(serve):
    server = serve(idle_ttl=1.0)
    assert server.wait_closed(timeout=10.0)
    assert server.exit_reason == "idle-timeout"


def test_a_request_resets_the_idle_clock(serve):
    server = serve(idle_ttl=2.0)
    client = Client(server)
    try:
        for _ in range(4):
            time.sleep(0.6)
            assert client.json("GET", "/hello")[0] == 200
        assert not server.shutting_down
    finally:
        client.close()


# --------------------------------------------------------------------------- #
# cache eviction (section 15)
# --------------------------------------------------------------------------- #
def test_the_lru_cache_evicts_and_the_evicted_id_is_gone(serve):
    client = Client(serve(cache_size=2))
    try:
        ids = []
        for seed in (1, 2, 3):
            _, accepted, _ = client.upload(32, 24, seed=seed)
            client.poll(accepted["job_id"])
            ids.append(accepted["image_id"])
        status, body = _prompt_text(client, ids[0])
        assert status == 404
        assert body["error"]["code"] == ErrorCode.IMAGE_NOT_FOUND
        assert _prompt_text(client, ids[2])[0] == 202
    finally:
        client.close()



class TestShutdownReleasesTheInstanceFirst:
    """runtime.json is removed and the lock released *before* GPU teardown.

    Field log: a daemon logged "idle ... exiting", removed nothing else in time,
    and hung in engine/cache teardown for minutes holding the OS lock. Every
    spawn exited with "another instance holds the lock" and the plug-in
    stalled.
    """

    def test_engine_close_runs_after_the_instance_is_released(self, tmp_path):
        seen = []

        class Recording(FakeEngine):
            def close(self_inner):
                # What another process could observe at this moment: is the
                # lock free, and is the stale runtime.json gone?
                probe = InstanceLock(str(tmp_path / "test.lock"))
                free = probe.acquire()
                if free:
                    probe.release()
                seen.append({"runtime_exists": os.path.exists(server.runtime_file),
                             "lock_held": server.lock.held,
                             "lock_free_to_others": free})
                FakeEngine.close(self_inner)

        engine = Recording()
        lock = InstanceLock(str(tmp_path / "test.lock"))
        assert lock.acquire()
        server = Sam3dServer(engine=engine, host="127.0.0.1", port=0, lock=lock,
                             runtime_file=str(tmp_path / "runtime.json"))
        server.start()
        try:
            assert os.path.exists(server.runtime_file)
            server.request_shutdown(grace_ms=0, reason="test")
            assert server.wait_closed(10.0)
        finally:
            server.close()
        assert engine.closed
        # Exactly one close, and it saw the instance already handed back.
        assert seen == [{"runtime_exists": False, "lock_held": False,
                         "lock_free_to_others": True}], seen

    def test_exit_watchdog_fires_only_if_teardown_stalls(self, tmp_path):
        server = Sam3dServer(engine=FakeEngine(), host="127.0.0.1", port=0,
                             runtime_file=str(tmp_path / "runtime.json"))
        fired = []
        t = server._arm_exit_watchdog(delay=0.15, exit_fn=fired.append)
        t.join(1.0)
        assert fired == [0], "a stalled teardown must be ended by force"

    def test_every_finished_job_is_logged(self, client, caplog):
        import logging
        caplog.set_level(logging.INFO, logger="sam3gimpd.server")
        _, accepted, _ = client.upload(32, 24)
        status, job = _prompt_text(client, accepted["image_id"], "r-log")
        assert status == 202
        client.poll(job["job_id"])
        lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("job ")]
        mine = [ln for ln in lines if job["job_id"] in ln]
        assert mine and "done" in mine[0], lines
        # elapsed_ms is a method; read as an attribute it was never a number
        # and the timing silently vanished from every line.
        assert " ms" in mine[0] and "instance(s)" in mine[0], mine



def test_shutdown_never_arms_a_hard_exit_unless_asked(tmp_path, monkeypatch):
    """The suite runs servers in-process; an os._exit bomb here killed pytest
    silently once. Only cli.py, the real daemon process, opts in."""
    armed = []
    monkeypatch.setattr(Sam3dServer, "_arm_exit_watchdog",
                        lambda self, *a, **k: armed.append(a))
    server = Sam3dServer(engine=FakeEngine(), host="127.0.0.1", port=0,
                         runtime_file=str(tmp_path / "runtime.json"))
    server.request_shutdown(grace_ms=0, reason="test")
    server.close()
    assert armed == []
    server2 = Sam3dServer(engine=FakeEngine(), host="127.0.0.1", port=0,
                          runtime_file=str(tmp_path / "runtime2.json"),
                          hard_exit_after_s=15.0)
    server2.request_shutdown(grace_ms=0, reason="test")
    server2.close()
    assert armed == [(15.0,)]



def test_lock_holder_pid_is_readable_while_held(tmp_path):
    """The launcher reads the holder's pid from the lockfile while it is held.

    On Windows msvcrt locks are mandatory, so the locked byte must not be the
    byte the pid lives in; the lock sits at LOCK_OFFSET, far past EOF.
    """
    lock = InstanceLock(str(tmp_path / "x.lock"))
    assert lock.acquire()
    try:
        assert InstanceLock.LOCK_OFFSET >= 4096
        with open(lock.path, "r", encoding="ascii") as fh:
            assert int(fh.read().strip()) == os.getpid()
    finally:
        lock.release()


def test_a_second_close_does_not_close_the_engine_again(serve):
    closes = []

    class Counting(FakeEngine):
        def close(self_inner):
            closes.append(time.monotonic())

    server = serve(Counting())
    server.close()
    server.close()
    assert len(closes) == 1


def test_hello_reports_the_build_taken_at_startup(serve, monkeypatch):
    """Files replaced on disk after the daemon started are not what it runs."""
    server = serve()
    taken = server.build
    monkeypatch.setattr(sam3gimpd, "_BUILD_HASH", "0badf00d")
    client = Client(server)
    try:
        assert client.json("GET", "/hello")[1]["build"] == taken != "0badf00d"
    finally:
        client.close()


def test_shutdown_cancels_queued_work_before_the_engine_closes(serve):
    """§6.8: queued jobs never start once a shutdown is accepted.  They used to
    run on after it -- one of them against an engine already closed."""
    engine = FakeEngine()
    engine.encode_gate = threading.Event()
    server = serve(engine, cache_size=8)
    client = Client(server)
    try:
        accepted = [client.upload(32, 24, seed=s)[1] for s in (1, 2, 3)]
        assert _eventually(lambda: server.jobs.status(accepted[0]["job_id"]).state
                           == JobState.RUNNING)
        server.request_shutdown(grace_ms=0, reason="test")
        for queued in accepted[1:]:
            assert _eventually(lambda: server.jobs.status(queued["job_id"]).state
                               == JobState.CANCELLED)
            status = server.jobs.status(queued["job_id"])
            assert status.error.code == ErrorCode.SHUTTING_DOWN
        engine.encode_gate.set()
        assert server.wait_closed(10.0)
        assert engine.encoded_ids == [accepted[0]["image_id"]]
        assert engine.closed
    finally:
        engine.encode_gate.set()
        client.close()


# --------------------------------------------------------------------------- #
# request paths never wait on the engine
# --------------------------------------------------------------------------- #
class BusyEngine(FakeEngine):
    """``describe()`` blocks while ``busy`` is set, as a real engine's does
    while its model manager holds the lock a cold load takes for a minute."""

    def __init__(self) -> None:
        super().__init__()
        self.busy = threading.Event()
        self.release = threading.Event()

    def describe(self) -> EngineInfo:
        if self.busy.is_set():
            self.release.wait(10.0)
        return super().describe()


def test_hello_status_and_prompts_answer_while_the_engine_is_busy(serve):
    engine = BusyEngine()
    client = Client(serve(engine))
    try:
        _, accepted, _ = client.upload(32, 24)
        client.poll(accepted["job_id"])
        engine.busy.set()

        def timed(fn):
            started = time.monotonic()
            result = fn()
            return result, time.monotonic() - started

        (status, hello), took = timed(lambda: client.json("GET", "/hello"))
        assert status == 200 and took < 2.0
        assert "pcs" in hello["capabilities"]          # from the last description
        (status, _), took = timed(lambda: client.json("GET", "/status"))
        assert status == 200 and took < 2.0
        (status, _, _), took = timed(lambda: client.upload(32, 24, seed=9))
        assert status == 202 and took < 2.0
        (status, _), took = timed(lambda: _prompt_text(client, accepted["image_id"]))
        assert status == 202 and took < 2.0
        (status, _), took = timed(lambda: client.json(
            "POST", "/images/%s/points" % accepted["image_id"],
            {"request_id": "r-p", "points": [{"x": 3.0, "y": 4.0, "label": 1}]}))
        assert status == 202 and took < 2.0
    finally:
        engine.release.set()
        client.close()


def test_status_reports_the_model_managers_memory(serve):
    """``ModelManager.memory()`` is where VRAM figures live; /status used to
    report them as permanently null because nothing asked it."""

    class Managed(FakeEngine):
        class _Manager:
            def memory(self):
                return {"rss_bytes": None, "vram_total": 8 << 30,
                        "vram_free": 6 << 30, "vram_reserved": 1 << 30}

        def __init__(self):
            super().__init__()
            self.mgr = self._Manager()

    client = Client(serve(Managed()))
    try:
        status, body = client.json("GET", "/status")
        assert status == 200
        memory = body["memory"]
        assert memory["vram_total"] == 8 << 30
        assert memory["vram_free"] == 6 << 30
        assert memory["vram_reserved"] == 1 << 30
        assert memory["rss_bytes"]                    # ours, not the None above
    finally:
        client.close()


# --------------------------------------------------------------------------- #
# the cache and the queue agree (API.md sections 6.2, 6.6, 10)
# --------------------------------------------------------------------------- #
def test_an_image_whose_encode_is_queued_is_not_evicted(serve):
    """Evicting it would leave its encode to run for nothing -- holding an
    embedding nobody could reach -- and every prompt for it to 404."""
    engine = FakeEngine()
    engine.encode_gate = threading.Event()
    server = serve(engine, cache_size=2)
    client = Client(server)
    try:
        uploads = [client.upload(32, 24, seed=s)[1] for s in (1, 2, 3)]
        sessions = [server.cache.peek(u["image_id"]) for u in uploads]
        a, b, c = uploads
        # A is encoding, B and C are queued: nothing may go yet.
        assert all(u["image_id"] in server.cache for u in uploads)
        status, job = _prompt_text(client, b["image_id"])
        assert status == 202
        engine.encode_gate.set()
        kind, _ = client.poll(job["job_id"])
        assert kind == CONTENT_TYPE_RESULT
        # Once the work settles the cache is back within capacity, and no
        # session that left it is still holding an embedding.
        assert _eventually(lambda: server.jobs.counts()["queued"] == 0
                           and len(server.cache) <= 2)
        for session in sessions:
            if session.image_id not in server.cache:
                assert session.embedding is None
    finally:
        engine.encode_gate.set()
        client.close()


def test_a_prompt_queued_behind_a_failed_encode_fails_at_once(serve):
    """Only the worker can settle an encode, so it must never wait for one.

    A's encode fails; the client re-uploads A, whose retry queues *behind* the
    prompt it had already sent.  That prompt used to park the worker for 30 s
    waiting on an encode only the worker itself could run.
    """

    class Flaky(FakeEngine):
        def __init__(self):
            super().__init__()
            self.fail_gate = threading.Event()   # holds A's first, failing encode
            self.hold_b = threading.Event()      # holds B's encode
            self.failures = 1

        def encode_image(self, image, progress=None):
            if image.width == 16 and self.failures:
                self.failures -= 1
                self.fail_gate.wait(10.0)
                raise ApiError(ErrorCode.MODEL_LOAD_FAILED, "transient")
            if image.width == 40:
                self.hold_b.wait(10.0)
            return super().encode_image(image, progress)

    engine = Flaky()
    server = serve(engine)
    client = Client(server)
    try:
        _, a1, _ = client.upload(16, 16, seed=1)
        _, b, _ = client.upload(40, 40, seed=2)
        status, prompt = _prompt_text(client, a1["image_id"], "r-a")
        assert status == 202
        engine.fail_gate.set()
        assert _eventually(lambda: server.jobs.status(a1["job_id"]).state == JobState.FAILED)
        _, a2, _ = client.upload(16, 16, seed=1)        # the retry, queued after the prompt
        assert a2["cached"] is False and a2["job_id"] != a1["job_id"]

        started = time.monotonic()
        engine.hold_b.set()
        kind, body = client.poll(prompt["job_id"], timeout=10.0)
        assert kind == "json" and body["state"] == JobState.FAILED
        assert body["error"]["code"] == ErrorCode.IMAGE_NOT_READY
        kind, body = client.poll(a2["job_id"], timeout=10.0)
        assert body["state"] == JobState.DONE
        assert time.monotonic() - started < 5.0
    finally:
        engine.fail_gate.set()
        engine.hold_b.set()
        client.close()


def test_settled_jobs_do_not_keep_evicted_images_alive(serve):
    """Every retained job's closure used to hold its session and pixels: thirty
    1008px uploads with a cache of three kept thirty images alive."""
    server = serve(cache_size=1)
    client = Client(server)
    try:
        refs = []
        for seed in range(4):
            _, accepted, _ = client.upload(32, 24, seed=seed)
            refs.append(weakref.ref(server.cache.peek(accepted["image_id"])))
            _, job = _prompt_text(client, accepted["image_id"], "r-%d" % seed)
            client.poll(job["job_id"])
        assert _eventually(lambda: server.jobs.counts()["running"] == 0)
        gc.collect()
        alive = [ref() for ref in refs if ref() is not None]
        assert len(alive) == 1, alive
        assert alive[0].image_id in server.cache
    finally:
        client.close()


def test_a_retried_encode_does_not_leave_the_failed_attempt_pinned(serve):
    """Pinned jobs are never pruned; each retry used to leave one behind."""
    engine = FakeEngine()
    engine.encode_error = ApiError(ErrorCode.MODEL_LOAD_FAILED, "no checkpoint")
    server = serve(engine, cache_size=1)
    client = Client(server)
    try:
        attempts = []
        for _ in range(3):
            _, accepted, _ = client.upload(32, 24, seed=1)
            client.poll(accepted["job_id"])
            attempts.append(accepted["job_id"])
        assert len(set(attempts)) == 3
        assert [server.jobs.get(j).pinned for j in attempts] == [False, False, True]
        engine.encode_error = None
        client.upload(32, 24, seed=2)                    # evicts the failed image
        assert not any(server.jobs.get(j).pinned for j in attempts)
    finally:
        client.close()


def test_the_deferred_release_of_a_deleted_image_spares_its_re_upload(serve):
    """Delete an image mid-encode and upload the same pixels again: when the
    first encode finishes and the deleted session is finally released, that
    release must not touch the new upload's encode job."""
    engine = FakeEngine()
    engine.encode_gate = threading.Event()
    server = serve(engine)
    client = Client(server)
    try:
        _, first, _ = client.upload(32, 24, seed=5)
        assert _eventually(lambda: server.jobs.status(first["job_id"]).state
                           == JobState.RUNNING)
        assert client.json("DELETE", "/images/%s" % first["image_id"])[0] == 200
        _, second, _ = client.upload(32, 24, seed=5)
        assert second["cached"] is False and second["job_id"] != first["job_id"]
        engine.encode_gate.set()
        client.poll(second["job_id"])
        # The deleted session is released once nothing pins the id...
        assert _eventually(lambda: not server.jobs.get(first["job_id"]).pinned)
        # ...and the new upload still names its own, still-pinned encode job.
        assert server.jobs.get(second["job_id"]).pinned
        _, third, _ = client.upload(32, 24, seed=5)
        assert third["cached"] is True and third["job_id"] == second["job_id"]
    finally:
        engine.encode_gate.set()
        client.close()


def test_concurrent_identical_uploads_share_one_encode_job(serve, monkeypatch):
    """The second of two racing uploads used to answer ``cached: true`` with
    ``job_id: ""`` -- the first had not yet recorded its job."""
    server = serve()
    real_submit = server.jobs.submit

    def slow_submit(*args, **kwargs):
        time.sleep(0.2)                         # widen the window
        return real_submit(*args, **kwargs)

    monkeypatch.setattr(server.jobs, "submit", slow_submit)
    pixels = bytes([7]) * (16 * 16 * 3)
    first = []
    thread = threading.Thread(target=lambda: first.append(server.accept_image(pixels, 16, 16)))
    thread.start()
    time.sleep(0.05)
    second = server.accept_image(pixels, 16, 16)
    thread.join(5.0)
    assert first[0].cached is False
    assert second.cached is True
    assert second.job_id == first[0].job_id != ""


def test_a_full_queue_refuses_an_upload_without_evicting_anything(serve):
    """The upload is refused before it touches the cache; it used to evict an
    innocent image first and then be turned away anyway."""
    engine = FakeEngine()
    server = serve(engine, max_queue=1, cache_size=3)
    client = Client(server)
    try:
        _, done, _ = client.upload(32, 24, seed=1)
        client.poll(done["job_id"])                         # cached, idle, evictable
        engine.encode_gate = threading.Event()
        _, running, _ = client.upload(32, 24, seed=2)
        assert _eventually(lambda: server.jobs.status(running["job_id"]).state
                           == JobState.RUNNING)
        _, queued, _ = client.upload(32, 24, seed=3)        # the queue is now full
        status, body, _ = client.upload(32, 24, seed=4)
        assert status == 503 and body["error"]["code"] == ErrorCode.QUEUE_FULL
        for kept in (done, running, queued):
            assert kept["image_id"] in server.cache
    finally:
        if engine.encode_gate is not None:
            engine.encode_gate.set()
        client.close()


# --------------------------------------------------------------------------- #
# validation: numbers must be numbers (API.md sections 6.3, 6.4)
# --------------------------------------------------------------------------- #
def test_non_finite_and_out_of_range_numbers_are_bad_requests(client):
    """JSON parses ``NaN``, ``Infinity`` and ``1e400`` (as inf) without
    complaint, and a 400-digit integer overflows ``float()``.  Each used to be
    accepted and fail inside the job -- or answer 500."""
    _, accepted, _ = client.upload(32, 24)
    image_id = accepted["image_id"]
    huge = "9" * 400
    cases = [
        ("points", '{"request_id":"r","points":[{"x":NaN,"y":3,"label":1}]}'),
        ("points", '{"request_id":"r","points":[{"x":Infinity,"y":3,"label":1}]}'),
        ("points", '{"request_id":"r","points":[{"x":1e400,"y":3,"label":1}]}'),
        ("points", '{"request_id":"r","points":[{"x":%s,"y":3,"label":1}]}' % huge),
        ("points", '{"request_id":"r","points":[{"x":1,"y":3,"label":true}]}'),
        ("points", '{"request_id":"r","box":[NaN,0,10,10]}'),
        ("points", '{"request_id":"r","box":[0,0,10,-Infinity]}'),
        ("points", '{"request_id":"r","box":[1.0,1.0,1.5,9.0]}'),   # under a pixel
        ("text", '{"request_id":"r","text":"x","score_threshold":NaN}'),
        ("text", '{"request_id":"r","text":"x","boxes":[{"box":[0,0,NaN,5],"label":1}]}'),
        ("text", '{"request_id":"r","text":"x","boxes":[{"box":[0,0,%s,5]}]}' % huge),
        ("text", '{"request_id":"r","text":"x","boxes":[{"box":[0,0,5,5],"label":"yes"}]}'),
        ("text", '{"request_id":"r","text":"x","boxes":[{"box":[0,0,5,5],"label":2}]}'),
        ("text", '{"request_id":"r","text":"x","max_instances":%s}' % huge),
    ]
    for kind, body in cases:
        status, _, payload = client.raw("POST", "/images/%s/%s" % (image_id, kind),
                                        body.encode("ascii"),
                                        {"Content-Type": "application/json"})
        assert status == 400, body
        assert json.loads(payload)["error"]["code"] == ErrorCode.BAD_REQUEST, body

    status, _, payload = client.raw("POST", "/shutdown", b'{"grace_ms":NaN}',
                                    {"Content-Type": "application/json"})
    assert status == 400
    status, _, payload = client.raw("POST", "/shutdown", b'{"grace_ms":%s}' % huge.encode(),
                                    {"Content-Type": "application/json"})
    assert status == 400
    assert not client.server.shutting_down
    assert client.server.last_error is None


def test_valid_exemplar_boxes_reach_the_engine_normalised():
    prompt = parse_text_prompt({"request_id": "r", "text": "x",
                                "boxes": [{"box": [0, 0, 5, 5]},
                                          {"box": [1, 2, 3.5, 4], "label": 0}]})
    assert prompt.boxes == [{"box": [0.0, 0.0, 5.0, 5.0], "label": 1},
                            {"box": [1.0, 2.0, 3.5, 4.0], "label": 0}]


def test_an_omitted_score_threshold_is_the_contract_default():
    """API.md §6.3: 0.02 -- a hard floor on what the client's slider can show."""
    prompt = parse_text_prompt({"request_id": "r", "text": "guitar strap"})
    assert prompt.score_threshold == DEFAULT_SCORE_THRESHOLD == 0.02


def test_the_server_clamps_the_long_poll_budget(client, monkeypatch):
    seen = []
    real_wait = client.server.jobs.wait

    def spy(job_id, timeout=0.0, include_result=False):
        seen.append(timeout)
        return real_wait(job_id, 0.0, include_result)

    monkeypatch.setattr(client.server.jobs, "wait", spy)
    _, accepted, _ = client.upload(32, 24)
    for value in ("99", "inf"):
        assert client.json("GET", "/jobs/%s?wait=%s" % (accepted["job_id"], value))[0] == 200
    assert seen == [Limits.MAX_LONG_POLL_SECONDS] * 2
    status, body = client.json("GET", "/jobs/%s?wait=nan" % accepted["job_id"])
    assert status == 400 and body["error"]["code"] == ErrorCode.BAD_REQUEST


# --------------------------------------------------------------------------- #
# the Host guard, auth and the identity proof (API.md section 2)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("host,loopback", [
    ("127.0.0.1", True), ("127.0.0.1:8080", True), ("127.9.8.7:1", True),
    ("localhost", True), ("LOCALHOST:80", True),
    ("[::1]", True), ("[::1]:8080", True), ("::1", True),
    ("127.attacker.example", False), ("127.0.0.1.evil.example", False),
    ("127.0.0.1.evil.example:80", False), ("localhost.evil.example", False),
    ("[::1]x", False), ("[::1]:80x", False), ("[::1", False), ("[127.0.0.1]", False),
    ("127.0.0.1:", False), ("127.0.0.1:http", False), ("127.0.0.1:8080:1", False),
    ("evil.example", False), ("0.0.0.0", False), ("10.0.0.1", False), ("", False),
])
def test_only_exact_loopback_hosts_pass_the_guard(host, loopback):
    assert host_header_is_loopback(host) is loopback


def test_rebinding_hostnames_are_refused_over_http(server):
    for host in ("127.attacker.example", "127.0.0.1.rebind.evil.example:%d" % server.port,
                 "[::1]x"):
        conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
        try:
            conn.putrequest("GET", "/hello", skip_host=True)
            conn.putheader("Host", host)
            conn.putheader("Authorization", "Bearer " + server.token)
            conn.endheaders()
            response = conn.getresponse()
            body = json.loads(response.read())
            assert response.status == 403, host
            assert body["error"]["code"] == ErrorCode.FORBIDDEN_HOST
        finally:
            conn.close()


def test_only_an_authenticated_request_resets_the_idle_clock(server):
    """A process knocking on the port must not keep the daemon -- and its
    VRAM -- alive past the idle TTL."""
    client = Client(server)
    try:
        before = server.last_request_at
        time.sleep(0.05)
        assert client.raw("GET", "/hello", auth=False)[0] == 401
        assert client.raw("GET", "/hello", headers={"Host": "evil.example"})[0] == 403
        assert server.last_request_at == before
        assert client.json("GET", "/hello")[0] == 200
        assert server.last_request_at > before
    finally:
        client.close()


def test_a_non_ascii_token_is_a_401_not_an_internal_error(client, caplog):
    """``hmac.compare_digest`` raises on a non-ASCII str; that used to escape as
    a 500 with a traceback in the log and in ``last_error`` on every probe."""
    caplog.set_level(logging.DEBUG, logger="sam3gimpd")
    status, _, payload = client.raw("GET", "/hello",
                                    headers={"Authorization": "Bearer tökén"},
                                    auth=False)
    assert status == 401
    assert json.loads(payload)["error"]["code"] == ErrorCode.UNAUTHORIZED
    assert client.server.last_error is None
    assert not [r for r in caplog.records if r.exc_info]
    assert not [r for r in caplog.records if r.levelno > logging.DEBUG
                and "refused" in r.getMessage()]


def test_the_bearer_scheme_is_case_insensitive(client):
    for scheme in ("Bearer", "bearer", "BEARER"):
        status, _ = client.json("GET", "/hello",
                                headers={"Authorization": "%s %s" % (scheme, client.token)})
        assert status == 200, scheme
    status, _ = client.json("GET", "/hello",
                            headers={"Authorization": "Token %s" % client.token})
    assert status == 401


def _expected_proof(token, nonce):
    return hmac.new(token.encode("utf-8"), b"sam3gimpd-hello:" + nonce.encode("ascii"),
                    hashlib.sha256).hexdigest()


def test_hello_proves_it_holds_the_token(client):
    """A stale runtime.json names a port that anything may hold by now; only
    the process that minted the token can answer a fresh nonce with it."""
    assert API_VERSION == "1.1"
    for _ in range(2):
        nonce = secrets.token_urlsafe(24)
        status, body = client.json("GET", "/hello", headers={"X-Sam3-Nonce": nonce})
        assert status == 200
        assert body["nonce_proof"] == _expected_proof(client.token, nonce)


def test_a_missing_or_malformed_nonce_gets_no_proof(client):
    for headers in ({}, {"X-Sam3-Nonce": "too-short"}, {"X-Sam3-Nonce": "n" * 129},
                    {"X-Sam3-Nonce": "has spaces in it, 0123456789"},
                    {"X-Sam3-Nonce": "slash/and+plus/0123456789"}):
        status, body = client.json("GET", "/hello", headers=headers)
        assert status == 200
        assert "nonce_proof" not in body, headers
    # The boundaries are inclusive.
    for nonce in ("a" * 16, "Z" * 128):
        assert "nonce_proof" in client.json("GET", "/hello",
                                            headers={"X-Sam3-Nonce": nonce})[1]


def test_the_proof_is_only_for_hello_and_only_with_the_token(client):
    nonce = secrets.token_urlsafe(24)
    status, body = client.json("GET", "/status", headers={"X-Sam3-Nonce": nonce})
    assert status == 200 and "nonce_proof" not in body
    status, _, payload = client.raw("GET", "/hello", headers={"X-Sam3-Nonce": nonce},
                                    auth=False)
    assert status == 401 and b"nonce_proof" not in payload


# --------------------------------------------------------------------------- #
# connections: bodies, caps and timeouts (API.md section 1)
# --------------------------------------------------------------------------- #
def test_a_small_unread_body_does_not_desync_a_keep_alive_connection(client):
    """A GET or DELETE body nobody reads used to be parsed as the next request
    line, and the next request on the connection got a 501 HTML page."""
    _, accepted, _ = client.upload(32, 24)
    status, headers, _ = client.raw("DELETE", "/images/%s" % accepted["image_id"],
                                    b'{"reason":"done"}',
                                    {"Content-Type": "application/json"})
    assert status == 200 and headers.get("Connection") != "close"
    status, headers, payload = client.raw("GET", "/hello", b"x" * 100)
    assert status == 200 and headers.get("Connection") != "close"
    status, headers, payload = client.raw("GET", "/hello")
    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    json.loads(payload)


def test_a_large_unread_body_closes_the_connection(server):
    sock = socket.create_connection(("127.0.0.1", server.port), timeout=5)
    try:
        sock.sendall(("GET /hello HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                      "Authorization: Bearer %s\r\nContent-Length: %d\r\n\r\n"
                      % (server.token, DRAIN_LIMIT_BYTES + 1)).encode("ascii"))
        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break                               # the server closed it
            data += chunk
        head = data.split(b"\r\n\r\n", 1)[0].decode("latin-1").lower()
        assert head.startswith("http/1.1 200")
        assert "connection: close" in head
    finally:
        sock.close()


def test_connections_beyond_the_cap_are_closed_on_accept(serve):
    server = serve(max_connections=3)
    idle = [socket.create_connection(("127.0.0.1", server.port), timeout=5)
            for _ in range(3)]
    try:
        assert _eventually(lambda: server._httpd.active_connections == 3)
        extra = socket.create_connection(("127.0.0.1", server.port), timeout=5)
        try:
            extra.sendall(b"GET /hello HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
            data = extra.recv(1024)
        except OSError:
            data = b""
        finally:
            extra.close()
        assert data == b""                           # refused without an answer
        assert server._httpd.active_connections == 3
        idle.pop().close()
        assert _eventually(lambda: server._httpd.active_connections == 2)
        client = Client(server)
        try:
            assert client.json("GET", "/hello")[0] == 200
        finally:
            client.close()
    finally:
        for sock in idle:
            sock.close()


def test_a_connection_that_never_speaks_is_dropped(serve, monkeypatch):
    monkeypatch.setattr(server_mod, "FIRST_REQUEST_TIMEOUT_S", 0.3)
    server = serve()
    sock = socket.create_connection(("127.0.0.1", server.port), timeout=5)
    try:
        started = time.monotonic()
        assert sock.recv(1) == b""
        assert time.monotonic() - started < 3.0
        assert _eventually(lambda: server._httpd.active_connections == 0)
    finally:
        sock.close()


def test_a_request_head_that_trickles_in_is_cut_off(serve, monkeypatch):
    """The head has a total deadline.  A per-read timeout alone never fires
    for a peer that sends one byte at a time, so it could hold a connection
    slot -- and with enough of them, the whole daemon -- indefinitely."""
    monkeypatch.setattr(server_mod, "REQUEST_HEAD_DEADLINE_S", 0.5)
    server = serve()
    sock = socket.create_connection(("127.0.0.1", server.port), timeout=5)
    try:
        sock.sendall(b"GET /hello HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer ")
        sock.settimeout(0.2)
        started = time.monotonic()
        closed = False
        while not closed and time.monotonic() - started < 4.0:
            try:
                sock.sendall(b"x")                  # never finishing the head
                closed = sock.recv(1024) == b""
            except socket.timeout:
                continue
            except OSError:
                closed = True
        assert closed, "the server never cut the trickling head off"
        assert time.monotonic() - started < 3.0
        assert _eventually(lambda: server._httpd.active_connections == 0)
    finally:
        sock.close()


def test_a_refused_request_closes_its_connection(server):
    """A client with the token never sees a 401; one without it must not keep
    a connection slot alive through keep-alive."""
    stranger = Client(server, token="not-the-token")
    try:
        status, headers, _ = stranger.raw("GET", "/hello")
        assert status == 401 and headers.get("Connection") == "close"
        status, headers, _ = stranger.raw("GET", "/hello", headers={"Host": "evil.example"})
        assert status == 403 and headers.get("Connection") == "close"
    finally:
        stranger.close()


def test_keep_alive_requests_do_not_stall(client):
    """A response is two writes, head then body.  With Nagle's algorithm on
    the second waited for the peer's delayed ACK: ~40 ms on every request on a
    reused connection, which is every poll and every click."""
    for _ in range(3):
        assert client.raw("GET", "/hello")[0] == 200
    started = time.monotonic()
    for _ in range(50):
        assert client.raw("GET", "/hello")[0] == 200
    assert time.monotonic() - started < 1.0


def test_an_idle_keep_alive_connection_is_dropped(serve, monkeypatch):
    monkeypatch.setattr(server_mod, "KEEPALIVE_IDLE_TIMEOUT_S", 0.3)
    server = serve()
    client = Client(server)
    try:
        assert client.json("GET", "/hello")[0] == 200
        assert _eventually(lambda: server._httpd.active_connections == 0, timeout=3.0)
        assert client.conn.sock.recv(1) == b""        # the server closed its end
        # The plug-in's client retries once on a fresh socket when this happens.
        client.conn.close()
        assert client.json("GET", "/hello")[0] == 200
    finally:
        client.close()


class _RecordingSocket:
    def __init__(self):
        self.options = []

    def setsockopt(self, level, name, value):
        self.options.append((level, name, value))


def test_the_listener_is_exclusive_on_windows(monkeypatch):
    """On Windows SO_REUSEADDR lets *another* process bind our port and take
    connections meant for us; there it is SO_EXCLUSIVEADDRUSE instead."""
    monkeypatch.setattr(socket, "SO_EXCLUSIVEADDRUSE", -5, raising=False)
    windows = _RecordingSocket()
    server_mod._configure_listen_socket(windows, windows=True)
    assert windows.options == [(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)]
    posix = _RecordingSocket()
    server_mod._configure_listen_socket(posix, windows=False)
    assert posix.options == [(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)]
    # socketserver must not add SO_REUSEADDR behind our back.
    assert server_mod._HTTPServer.allow_reuse_address is False


# --------------------------------------------------------------------------- #
# private files (runtime.json, logs, the data directory)
# --------------------------------------------------------------------------- #
def _mode(path):
    return os.stat(path).st_mode & 0o777


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes only")
def test_the_data_directory_is_private(tmp_path, monkeypatch):
    base = tmp_path / "fresh" / "sam3-gimp"
    monkeypatch.setenv("SAM3_GIMP_HOME", str(base))
    monkeypatch.delenv("SAM3D_RUNTIME_FILE", raising=False)
    paths.ensure_layout()
    for directory in (base.parent, base, base / "logs", base / "hf", base / "tools",
                      base / "models", base / "cache"):
        assert _mode(directory) == 0o700, directory
    os.chmod(base, 0o755)                       # an existing, wider base...
    paths.ensure_layout()
    assert _mode(base) == 0o700                 # ...is tightened


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes only")
def test_the_daemon_log_is_private_and_rotates(tmp_path, monkeypatch):
    from sam3gimpd import cli

    monkeypatch.setattr(cli, "LOG_MAX_BYTES", 2000)
    root = logging.getLogger("sam3gimpd")
    saved_level, saved_handlers = root.level, list(root.handlers)
    target = tmp_path / "logs" / "sam3gimpd.log"
    try:
        assert cli.setup_logging("info", str(target), stderr=False) == str(target)
        log = logging.getLogger("sam3gimpd.rotation-test")
        for i in range(200):
            log.info("line %03d %s", i, "x" * 40)
        names = sorted(p.name for p in target.parent.iterdir())
        assert names == ["sam3gimpd.log", "sam3gimpd.log.1", "sam3gimpd.log.2",
                         "sam3gimpd.log.3"]
        for path in target.parent.iterdir():
            assert _mode(path) == 0o600, path.name
        assert target.stat().st_size < 2000
        assert "line 199" in target.read_text(encoding="utf-8")
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes only")
def test_the_crash_log_is_private(tmp_path, monkeypatch):
    from sam3gimpd import cli

    monkeypatch.setenv("SAM3_GIMP_HOME", str(tmp_path / "home"))
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        where = cli.write_crash_log(exc)
    with open(where, encoding="utf-8") as fh:
        assert "boom" in fh.read()
    assert _mode(where) == 0o600


def test_runtime_json_replace_is_retried_while_a_reader_holds_it(tmp_path, monkeypatch):
    """Windows refuses to replace a file another process has open; the reader
    lets go in milliseconds, so the write retries instead of leaving a stale
    port and token behind."""
    calls = []
    real_replace = os.replace

    def flaky(src, dst):
        calls.append(dst)
        if len(calls) < 3:
            raise PermissionError(13, "sharing violation")
        return real_replace(src, dst)

    monkeypatch.setattr(paths.os, "replace", flaky)
    monkeypatch.setattr(paths, "REPLACE_RETRY_S", 0.0)
    target = tmp_path / "runtime.json"
    paths.atomic_write_json(target, {"port": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"port": 1}
    assert len(calls) == 3
    assert [p.name for p in tmp_path.iterdir()] == ["runtime.json"]


@pytest.mark.parametrize("os_name,closes_handle", [("posix", False), ("nt", True)])
def test_abort_connection_closes_the_os_handle_on_windows(monkeypatch, os_name, closes_handle):
    """The request-head deadline cuts a slow client off with shutdown() on
    POSIX; Windows leaves the handler's read waiting unless the handle itself
    is closed, after the socket object has let go of it."""
    calls, closed = [], []

    class _Conn:
        def shutdown(self, how):
            calls.append(("shutdown", how))

        def detach(self):
            calls.append(("detach",))
            return 4321

    monkeypatch.setattr(server_mod, "os", types.SimpleNamespace(name=os_name))
    monkeypatch.setattr(server_mod.socket, "close", closed.append)
    server_mod._abort_connection(_Conn())
    assert calls[0] == ("shutdown", socket.SHUT_RDWR)
    if closes_handle:
        assert calls[1:] == [("detach",)] and closed == [4321]
    else:
        assert calls[1:] == [] and closed == []


def test_status_memory_uses_the_engine_layers_rss_reader(monkeypatch):
    """One RSS reader for the daemon: the server's /status figure and the model
    manager's must not disagree, as they did on Windows when only one knew it."""
    from sam3gimpd import modelmgr

    monkeypatch.setattr(modelmgr, "_rss_bytes", lambda: 424242)
    assert server_mod._rss_bytes() == 424242
