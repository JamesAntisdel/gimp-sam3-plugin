"""The whole stack, running for real, on a box with no GPU and no torch.

Every other test file exercises one half against a fake of the other.  This one
puts the *actual* pieces together and drives them the way GIMP will:

    launcher.find_or_spawn      spawns a detached `python -m sam3gimpd serve --stub`
        -> client.Sam3Client    stdlib http.client, the plug-in's real client
        -> POST /images         a synthetic RGB buffer at API.md section 12's size
        -> POST .../text        a PCS prompt, long-polled to done
        -> the binary frame     decoded by BOTH client.parse_result_frame and
                                ui/canvas.decode_frame, which must agree
        -> outputs.MaskResult   the bridge the GIMP entry point uses at Apply
        -> Placement            model-canvas -> uploaded -> original (section 9)

plus supersession from both sides (section 10) and a clean shutdown that removes
`runtime.json` (section 3.2).

This is possible *only* because `--stub` is a first-class engine (API.md
section 14).  It imports no torch on any path, so this file is the CI proof that
the contract between the two halves actually holds -- not that each half agrees
with its own mock.

The daemon is spawned once for the module (it is a real process and a real
handshake) and every test shares it; `test_shutdown_*` deliberately start their
own so they can kill it without stranding the rest.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DAEMON_DIR = str(REPO_ROOT / "plugin" / "sam3_gimp" / "_daemon")
PLUGIN_PKG_DIR = str(REPO_ROOT / "plugin" / "sam3_gimp")

for _p in (DAEMON_DIR, PLUGIN_PKG_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import client as C  # noqa: E402
import launcher as L  # noqa: E402
import outputs as O  # noqa: E402
from ui import canvas as CV  # noqa: E402

pytestmark = [pytest.mark.needs_daemon, pytest.mark.slow]

#: The worked example in API.md section 12: a 3000x2000 photo downscaled to
#: 1008x672 before upload.  Using the documented numbers means the geometry
#: assertions below are checking the contract, not an arbitrary choice.
SOURCE_W, SOURCE_H = 3000, 2000
UPLOAD_W, UPLOAD_H = 1008, 672

#: `python -m sam3gimpd` rather than the console script: this repo is not pip
#: installed in CI, and the launcher's own fallback uses exactly this spelling.
DAEMON_COMMAND = [sys.executable, "-m", "sam3gimpd"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _synthetic_rgb(width: int, height: int, seed: int = 0) -> bytes:
    """`width * height * 3` bytes, exactly the section 7 upload layout.

    Cheap to build (one bytearray pass, no per-pixel Python arithmetic beyond a
    running counter) because a 1008x672 buffer is two million bytes and this
    runs on every CI job.
    """
    row = bytearray(width * 3)
    buf = bytearray()
    for y in range(height):
        base = (y * 7 + seed * 31) & 0xFF
        for x in range(width):
            i = x * 3
            row[i] = (x + base) & 0xFF
            row[i + 1] = (y * 3) & 0xFF
            row[i + 2] = (x ^ y ^ seed) & 0xFF
        buf += row
    return bytes(buf)


def _child_env() -> dict:
    """Environment for a daemon spawned out of the source tree.

    pytest's `pythonpath` ini setting patches *this* interpreter's `sys.path`;
    a subprocess inherits nothing from it.  The daemon is not installed in CI,
    so `daemon/` has to be handed over explicitly.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = DAEMON_DIR + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _spawn(home: Path, **kwargs):
    """Start a real detached daemon under `home` and return the LaunchResult."""
    os.environ["SAM3_GIMP_HOME"] = str(home)
    os.environ.pop("SAM3D_RUNTIME_FILE", None)
    os.environ["PYTHONPATH"] = DAEMON_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")
    params = dict(
        command=DAEMON_COMMAND,
        stub=True,
        parent_pid=os.getpid(),
        idle_ttl=300,
        timeout=60.0,
    )
    params.update(kwargs)
    return L.find_or_spawn(**params)


def _stop(launch) -> None:
    """Ask a daemon to exit and wait for its pid to go away."""
    if launch is None:
        return
    try:
        launch.client.shutdown_daemon(grace_ms=0)
    except Exception:
        pass
    finally:
        launch.close()
    deadline = time.time() + 15.0
    while time.time() < deadline and L.pid_alive(launch.pid):
        time.sleep(0.05)


# --------------------------------------------------------------------------- #
# module-scoped live daemon
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def home(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("e2e-home")


@pytest.fixture(scope="module")
def daemon(home):
    """One detached `sam3gimpd serve --stub`, spawned by the real launcher."""
    previous = {k: os.environ.get(k) for k in ("SAM3_GIMP_HOME", "SAM3D_RUNTIME_FILE", "PYTHONPATH")}
    launch = None
    try:
        launch = _spawn(home)
        yield launch
    finally:
        _stop(launch)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture(scope="module")
def uploaded(daemon):
    """`(client, ImageAccepted)` for one encoded 1008x672 image."""
    cl = daemon.client
    pixels = _synthetic_rgb(UPLOAD_W, UPLOAD_H, seed=1)
    accepted = cl.upload_image(
        pixels, UPLOAD_W, UPLOAD_H, source_width=SOURCE_W, source_height=SOURCE_H
    )
    status = cl.wait_for_image(accepted, timeout=60.0)
    assert status.state == "done", status
    return cl, accepted


@pytest.fixture(scope="module")
def pcs_result(uploaded):
    """A real PCS frame, decoded by the plug-in's own client."""
    cl, accepted = uploaded
    result = cl.run_text(accepted.image_id, "yellow school bus", timeout=60.0)
    assert result is not None, "the fresh prompt must not be reported stale"
    return result


# =========================================================================== #
# 1. the daemon starts, with no torch anywhere
# =========================================================================== #
def test_daemon_starts_and_reports_stub_mode(daemon):
    hello = daemon.hello
    assert hello.api_version.split(".")[0] == C.API_VERSION.split(".")[0]
    assert hello.engine_mode == "stub"
    assert hello.device == "stub"
    assert hello.torch_available is False
    assert {"pcs", "pvs"} <= set(hello.capabilities)
    assert daemon.spawned is True, "nothing else should have been listening"


def test_runtime_json_has_the_five_required_fields(daemon, home):
    """API.md section 3.2: exactly these are required; readers ignore the rest."""
    info = json.loads((home / "runtime.json").read_text(encoding="utf-8"))
    for field in ("port", "token", "pid", "version", "started_at"):
        assert field in info, field
    assert len(info["token"]) == 43
    assert info["pid"] == daemon.pid
    if os.name != "nt":
        mode = (home / "runtime.json").stat().st_mode & 0o777
        assert mode == 0o600, "the token file must not be world-readable"


def test_no_daemon_module_imports_torch_at_module_scope():
    """API.md section 16.9: `--stub` never imports torch.

    Checked in a *subprocess* so the assertion is about a clean interpreter that
    has imported the whole daemon, not about whatever pytest already loaded.
    """
    code = (
        "import sys;"
        "import sam3gimpd, sam3gimpd.server, sam3gimpd.jobs, sam3gimpd.session, sam3gimpd.masks,"
        " sam3gimpd.cli, sam3gimpd.modelmgr, sam3gimpd.engines,"
        " sam3gimpd.engines.stub;"
        "sam3gimpd.engines.stub.StubEngine();"
        "bad=[m for m in sys.modules if m.split('.')[0] in "
        "('torch','transformers','numpy','PIL','huggingface_hub')];"
        "print(','.join(sorted(bad)))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=_child_env(), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", "heavy modules were imported: " + proc.stdout


def test_python_dash_m_sam3d_runs_the_cli():
    """``API.md`` §13's invocation contract, on the branch Windows always takes.

    ``launcher.build_command`` only uses the ``sam3gimpd`` console script on POSIX --
    on Windows it must be ``pythonw.exe -m sam3gimpd``, because ``sam3gimpd.exe`` flashes
    a console window (``DESIGN.md`` §4).  So a missing ``sam3gimpd/__main__.py``
    would break the *primary target platform* while leaving Linux fine, which is
    exactly the kind of gap that only shows up on a user's machine.
    """
    proc = subprocess.run(
        DAEMON_COMMAND + ["serve", "--help"],
        env=_child_env(), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    for flag in ("--stub", "--parent-pid", "--idle-ttl", "--cache-size",
                 "--runtime-file", "--host", "--port", "--device", "--log-level"):
        assert flag in proc.stdout, "%s is missing from `sam3gimpd serve --help`" % flag


def test_the_launcher_builds_the_documented_argv():
    """Whatever prefix is resolved, the flags after it are §13's, verbatim."""
    argv = L.build_command(
        command=DAEMON_COMMAND, stub=True, parent_pid=4711, idle_ttl=600,
        cache_size=3, host="127.0.0.1", port=0,
    )
    assert argv[:4] == DAEMON_COMMAND + ["serve"]
    assert "--stub" in argv
    assert argv[argv.index("--parent-pid") + 1] == "4711"
    assert argv[argv.index("--idle-ttl") + 1] == "600"
    assert argv[argv.index("--cache-size") + 1] == "3"


# =========================================================================== #
# 2. upload -> prompt -> frame
# =========================================================================== #
def test_upload_reports_the_documented_geometry(uploaded):
    _cl, accepted = uploaded
    assert len(accepted.image_id) == 32
    assert accepted.image.width == UPLOAD_W and accepted.image.height == UPLOAD_H
    # The canvas is the uploaded image: masks are post-processed straight to
    # it, so there is no squash to invert.  This asserted a 1008x1008 canvas
    # with scale (1.0, 1.5), an anisotropy the processor never applies -- every
    # mask on a non-square image came back stretched.
    assert accepted.model_canvas.width == UPLOAD_W
    assert accepted.model_canvas.height == UPLOAD_H
    t = accepted.canvas_from_image
    assert t.scale_x == pytest.approx(1.0)
    assert t.scale_y == pytest.approx(1.0)
    assert (t.offset_x, t.offset_y) == (0.0, 0.0)


def test_uploading_identical_pixels_is_cached(uploaded):
    """Section 6.2: the id is a digest of the bytes, so a re-upload is a hit."""
    cl, accepted = uploaded
    again = cl.upload_image(
        _synthetic_rgb(UPLOAD_W, UPLOAD_H, seed=1), UPLOAD_W, UPLOAD_H
    )
    assert again.image_id == accepted.image_id
    assert again.cached is True


def test_frame_satisfies_every_section_16_invariant(pcs_result):
    r = pcs_result
    assert r.mask_encoding == C.MASK_ENCODING_U8_SOFT
    assert r.engine == "pcs"
    assert r.state == "done"
    assert len(r.instances) >= 1
    offset = 0
    previous_score = 2.0
    for inst in r.instances:
        assert inst.mask_width == inst.bbox.width      # section 16.4
        assert inst.mask_height == inst.bbox.height
        assert inst.blob_length == inst.mask_width * inst.mask_height
        assert len(inst.mask) == inst.blob_length
        assert inst.blob_offset == offset              # section 16.2, tightly packed
        offset += inst.blob_length
        assert 0 <= inst.bbox.x0 < inst.bbox.x1 <= r.model_canvas.width
        assert 0 <= inst.bbox.y0 < inst.bbox.y1 <= r.model_canvas.height
        assert 0.0 <= inst.score <= 1.0
        assert inst.score <= previous_score            # sorted descending
        previous_score = inst.score
        assert inst.is_soft(), "section 14: the stub must return real gradients"


def test_threshold_is_a_local_byte_comparison(pcs_result):
    """Section 8.3: moving the slider must change the mask with no round trip."""
    inst = pcs_result.instances[0]
    loose = inst.count_above(64)
    default = inst.count_above(128)
    tight = inst.count_above(200)
    assert loose > default > tight > 0


def test_the_two_frame_decoders_agree(uploaded, pcs_result):
    """`client.parse_result_frame` and `ui/canvas.decode_frame` are two
    independent implementations of section 8, written separately.  On
    real wire bytes they must agree -- otherwise the canvas and the Apply path
    would disagree about what the user is looking at.
    """
    cl, _accepted = uploaded
    raw = cl.job_result_frame(pcs_result.job_id)
    assert raw[:8] == C.RESULT_MAGIC

    header, blob = CV.decode_frame(raw)
    assert header["request_id"] == pcs_result.request_id
    assert header["blob_length"] == len(blob)
    assert len(header["instances"]) == len(pcs_result.instances)

    for mine, entry in zip(pcs_result.instances, header["instances"]):
        assert list(mine.bbox.to_list()) == list(entry["bbox"])
        # section 16.2: blob_offset indexes the blob *region*, not the body.
        start = int(entry["blob_offset"])
        assert bytes(mine.mask) == blob[start:start + int(entry["blob_length"])]

    # The blob a client rebuilds by concatenating crops -- what main_dialog and
    # the GIMP entry point both do -- must be the wire blob, byte for byte.
    assert b"".join(bytes(i.mask) for i in pcs_result.instances) == blob


# =========================================================================== #
# 3. geometry: model-canvas -> uploaded -> original
# =========================================================================== #
def test_instances_map_back_onto_the_original_image(pcs_result):
    """Section 9, applied to real daemon output.

    Every placement must land inside the 3000x2000 original, and re-deriving the
    canvas bbox from the placement must return where it started (to within the
    per-edge rounding section 9 step 3 mandates).
    """
    r = pcs_result
    t = r.canvas_from_image
    u = SOURCE_W / float(r.image.width)
    v = SOURCE_H / float(r.image.height)
    for inst in r.instances:
        place = r.place(inst, SOURCE_W, SOURCE_H)
        assert 0 <= place.x < SOURCE_W
        assert 0 <= place.y < SOURCE_H
        assert place.width >= 1 and place.height >= 1
        assert place.x + place.width <= SOURCE_W + 1
        assert place.y + place.height <= SOURCE_H + 1

        # Round-trip: original -> uploaded -> canvas is the bbox we were given.
        cx0, cy0 = t.image_to_canvas(place.x / u, place.y / v)
        cx1, cy1 = t.image_to_canvas((place.x + place.width) / u,
                                     (place.y + place.height) / v)
        assert abs(cx0 - inst.bbox.x0) <= 1.0
        assert abs(cy0 - inst.bbox.y0) <= 1.0
        assert abs(cx1 - inst.bbox.x1) <= 1.0
        assert abs(cy1 - inst.bbox.y1) <= 1.0


def test_placement_reproduces_the_worked_example(pcs_result):
    """API.md section 12.6 is a golden number; check it against the *reported*
    transform of a live daemon rather than a hand-built fixture."""
    t = pcs_result.canvas_from_image
    assert (t.scale_x, t.scale_y, t.offset_x, t.offset_y) == (1.0, 1.0, 0.0, 0.0)
    fake = C.MaskInstance(
        instance_id=0, score=0.934, label="yellow school bus",
        bbox=C.BBox(412, 300, 700, 480),
        mask_width=288, mask_height=180, blob_offset=0, blob_length=288 * 180,
        mask=b"\x00" * (288 * 180),
    )
    place = C.place_instance(
        fake, t, uploaded=C.Size(UPLOAD_W, UPLOAD_H), source=C.Size(SOURCE_W, SOURCE_H)
    )
    # Recomputed for the identity transform.  The y numbers moved because the
    # old ones encoded a 1.5x vertical stretch that was never really applied.
    assert place.to_tuple() == (1226, 893, 857, 536)


# =========================================================================== #
# 4. the bridge into outputs.py (what the entry point does at Apply)
# =========================================================================== #
def test_result_bridges_into_outputs_mask_result(pcs_result):
    """`sam3_gimp._mask_result` rebuilds the blob by concatenating the crops --
    legal only because section 8.1 packs them tightly in array order.  Verify
    that `outputs.MaskResult` then agrees with the client about placement."""
    r = pcs_result
    blob = b"".join(bytes(i.mask) for i in r.instances)
    mask_result = O.MaskResult.from_frame(
        r.header, blob, (SOURCE_W, SOURCE_H), prompt_text="yellow school bus"
    )
    assert len(mask_result.instances) == len(r.instances)
    for mine, theirs in zip(r.instances, mask_result.instances):
        assert bytes(theirs.mask) == bytes(mine.mask)
        assert (theirs.mask_width, theirs.mask_height) == (mine.mask_width, mine.mask_height)
        rect = mask_result.rect_for(theirs)
        assert tuple(rect)[:4] == r.place(mine, SOURCE_W, SOURCE_H).to_tuple()


def test_masks_trace_into_polygons_for_the_paths_output(pcs_result):
    """The Paths mode routes real mask bytes through `canvas.contour_polygons`."""
    inst = max(pcs_result.instances, key=lambda i: i.blob_length)
    polys = CV.contour_polygons(inst.mask, inst.mask_width, inst.mask_height, 128)
    assert polys, "a soft blob at threshold 128 must produce at least one contour"
    for poly in polys:
        assert len(poly) >= 3
        for (x, y) in poly:
            assert -1.0 <= x <= inst.mask_width + 1.0
            assert -1.0 <= y <= inst.mask_height + 1.0
    stroke = {"polyline": [(x, y) for (x, y) in polys[0]], "closed": True}
    points, closed = O.normalise_stroke(stroke)
    assert closed is True
    assert len(points) == len(polys[0]) * 6, "one anchor plus two handles per vertex"


# =========================================================================== #
# 4b. all the way into (stubbed) GIMP -- the Apply button's real path
# =========================================================================== #
@pytest.fixture
def gimp_stubs():
    """`tests/fake_gimp` installed for one test: real pixels, no GIMP."""
    import fake_gimp

    with fake_gimp.installed() as stubs:
        yield stubs


def _blank_image(stubs, width, height):
    Gimp = stubs.Gimp
    image = Gimp.Image.new(width, height, Gimp.ImageBaseType.RGB)
    layer = Gimp.Layer.new(
        image, "Background", width, height, Gimp.ImageType.RGB_IMAGE,
        100.0, Gimp.LayerMode.NORMAL,
    )
    image.insert_layer(layer, None, 0)
    return image, layer


@pytest.mark.parametrize(
    "mode", ["selection", "channels", "layer-masks", "layer-groups"]
)
def test_a_real_frame_applies_in_every_output_mode(pcs_result, gimp_stubs, mode):
    """Daemon -> client -> outputs -> GIMP, on masks the daemon actually produced.

    `tests/plugin/test_outputs.py` drives the same code with synthesised frames.
    The point here is that a frame nobody hand-wrote -- the daemon's own bytes,
    its own bbox arithmetic, its own header -- survives the whole journey.
    """
    blob = b"".join(bytes(i.mask) for i in pcs_result.instances)
    result = O.MaskResult.from_frame(
        pcs_result.header, blob, (SOURCE_W, SOURCE_H), prompt_text="yellow school bus"
    )
    image, layer = _blank_image(gimp_stubs, SOURCE_W, SOURCE_H)
    applied = O.apply_result(image, result, O.OutputOptions(mode=mode), layer=layer)

    assert applied.mode == mode
    assert applied.names, "every applied instance should be named from the prompt"
    if mode == "selection":
        assert applied.selection_changed is True
    elif mode == "channels":
        assert len(applied.channels) == len(result.instances)
    elif mode == "layer-masks":
        assert len(applied.layers) == 1
        assert applied.layers[0].get_mask() is not None
    elif mode == "layer-groups":
        assert applied.group is not None
        assert len(applied.layers) == len(result.instances)


def test_the_applied_selection_lands_inside_the_computed_rectangle(pcs_result, gimp_stubs):
    """The geometry claim, checked against pixels rather than arithmetic.

    §9 says where an instance goes; `outputs` scales the soft crop into that
    rectangle and thresholds it.  Whatever ends up selected must therefore lie
    inside the union of those rectangles -- if the affine were inverted the wrong
    way, or the client's own downscale were applied twice, the selection would
    land somewhere else entirely and this is what would notice.
    """
    blob = b"".join(bytes(i.mask) for i in pcs_result.instances)
    result = O.MaskResult.from_frame(
        pcs_result.header, blob, (SOURCE_W, SOURCE_H), prompt_text="yellow school bus"
    )
    image, layer = _blank_image(gimp_stubs, SOURCE_W, SOURCE_H)
    O.apply_result(image, result, O.OutputOptions(mode="selection"), layer=layer)

    _ok, non_empty, sx0, sy0, sx1, sy1 = gimp_stubs.Gimp.Selection.bounds(image)
    assert non_empty, "a soft blob thresholded at 128 must select something"

    rects = [result.rect_for(inst) for inst in result.instances]
    ux0 = min(r[0] for r in rects)
    uy0 = min(r[1] for r in rects)
    ux1 = max(r[0] + r[2] for r in rects)
    uy1 = max(r[1] + r[3] for r in rects)
    assert ux0 <= sx0 and sx1 <= ux1, (sx0, sx1, ux0, ux1)
    assert uy0 <= sy0 and sy1 <= uy1, (sy0, sy1, uy0, uy1)

    # And the same rectangles the *client* computed, independently, from §9.
    for inst, rect in zip(pcs_result.instances, rects):
        assert pcs_result.place(inst, SOURCE_W, SOURCE_H).to_tuple() == tuple(rect)[:4]


def test_apply_is_one_undo_step(pcs_result, gimp_stubs):
    """DESIGN.md §6: one Ctrl+Z reverts the whole Apply, however many instances."""
    blob = b"".join(bytes(i.mask) for i in pcs_result.instances)
    result = O.MaskResult.from_frame(pcs_result.header, blob, (SOURCE_W, SOURCE_H))
    image, layer = _blank_image(gimp_stubs, SOURCE_W, SOURCE_H)
    O.apply_result(image, result, O.OutputOptions(mode="channels"), layer=layer)

    assert image.undo_groups == ["start", "end"], image.undo_groups
    assert image.undo_depth == 0, "the group must be closed even on the happy path"


# =========================================================================== #
# 5. supersession (API.md section 10)
# =========================================================================== #
def test_queued_prompts_are_superseded_server_side(daemon):
    """A prompt accepted while the worker is busy displaces the queued one.

    The encode job created by `POST /images` is never superseded and always runs
    first (section 10), so uploading a fresh image and immediately firing two
    prompts puts both of them in the queue behind it -- which is exactly the
    situation supersession exists for.
    """
    cl = daemon.client

    # Retried, not because the rule is probabilistic, but because the *setup* is
    # a race we do not control: the two prompts have to reach the daemon inside
    # the ~250 ms the stub's encode pass takes, and a badly loaded CI runner can
    # stall a loopback POST for longer than that.  Each attempt is a full,
    # honest exercise of the contract; a first attempt that loses the race
    # leaves `first` already running, which §10 says is *not* superseded.
    first = second = accepted = None
    for attempt in range(4):
        accepted = cl.upload_image(_synthetic_rgb(320, 240, seed=7 + attempt), 320, 240)
        first = cl.prompt_text(accepted.image_id, "first prompt")
        second = cl.prompt_text(accepted.image_id, "second prompt")
        if first.job_id in second.superseded_job_ids:
            break
        cl.wait_for_job(second.job_id, timeout=60.0)
    else:
        raise AssertionError(
            "no queued prompt was ever superseded; the newer prompt reported %r"
            % (second.superseded_job_ids,)
        )
    status = cl.job_status(first.job_id)
    assert status.state == "superseded"
    assert status.superseded_by == second.job_id

    # And the client must not surface it.
    assert cl.wait_for_job(first.job_id, timeout=30.0) is None
    fresh = cl.wait_for_job(second.job_id, timeout=60.0, request_id=second.request_id)
    assert fresh is not None and len(fresh.instances) >= 1
    assert fresh.prompt.get("text") == "second prompt"


def test_stale_results_are_dropped_client_side(daemon):
    """Section 10's one-line rule: drop any result whose request_id is not the
    latest issued for that image -- even when the daemon ran it to completion."""
    cl = daemon.client
    accepted = cl.upload_image(_synthetic_rgb(160, 128, seed=9), 160, 128)
    cl.wait_for_image(accepted, timeout=60.0)

    stale = cl.prompt_text(accepted.image_id, "stale prompt")
    latest = cl.prompt_text(accepted.image_id, "latest prompt")
    assert cl.latest_request_id(accepted.image_id) == latest.request_id
    assert cl.is_stale(accepted.image_id, stale.request_id) is True

    # Whatever the daemon did with it -- ran it or superseded it -- the client
    # returns nothing, so a stale mask can never reach the canvas.
    assert cl.wait_for_job(
        stale.job_id, timeout=60.0, image_id=accepted.image_id,
        request_id=stale.request_id,
    ) is None
    assert cl.wait_for_job(
        latest.job_id, timeout=60.0, image_id=accepted.image_id,
        request_id=latest.request_id,
    ) is not None


# =========================================================================== #
# 6. PVS, determinism, deletion
# =========================================================================== #
def test_points_prompt_returns_an_instance(uploaded):
    cl, accepted = uploaded
    result = cl.run_points(
        accepted.image_id, [(504.0, 336.0, 1), (100.0, 100.0, 0)], timeout=60.0
    )
    assert result is not None
    assert result.engine == "pvs"
    assert len(result.instances) >= 1
    for inst in result.instances:
        assert inst.blob_length == inst.mask_width * inst.mask_height
        assert inst.mask_width == inst.bbox.width


def test_the_stub_is_deterministic(uploaded):
    """Section 14: the same (image, prompt) yields byte-identical masks."""
    cl, accepted = uploaded
    a = cl.run_text(accepted.image_id, "a repeatable prompt", timeout=60.0)
    b = cl.run_text(accepted.image_id, "a repeatable prompt", timeout=60.0)
    assert a is not None and b is not None
    assert [i.bbox.to_list() for i in a] == [i.bbox.to_list() for i in b]
    assert [bytes(i.mask) for i in a] == [bytes(i.mask) for i in b]


def test_deleting_an_image_makes_later_prompts_404(daemon):
    cl = daemon.client
    accepted = cl.upload_image(_synthetic_rgb(128, 96, seed=11), 128, 96)
    cl.wait_for_image(accepted, timeout=60.0)
    body = cl.delete_image(accepted.image_id)
    assert body["deleted"] is True
    with pytest.raises(C.ApiError) as excinfo:
        cl.prompt_text(accepted.image_id, "gone")
    assert excinfo.value.code == "image_not_found"


# =========================================================================== #
# 7. lifecycle: single instance, clean shutdown
# =========================================================================== #
def test_a_second_daemon_exits_on_the_lockfile(daemon, home):
    """Section 3.4: the loser exits 0 and leaves runtime.json alone."""
    before = (home / "runtime.json").read_text(encoding="utf-8")
    proc = subprocess.run(
        DAEMON_COMMAND + ["serve", "--stub", "--idle-ttl", "5"],
        cwd=str(home), env=_child_env(), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert (home / "runtime.json").read_text(encoding="utf-8") == before
    assert L.pid_alive(daemon.pid), "the holder must be untouched"


def test_shutdown_removes_runtime_json(tmp_path):
    """Section 6.8: stop serving, delete runtime.json, release the lock, exit 0.

    Its own daemon, so killing it cannot stranded the module-scoped one.
    """
    previous = {k: os.environ.get(k) for k in ("SAM3_GIMP_HOME", "PYTHONPATH")}
    launch = None
    try:
        launch = _spawn(tmp_path / "shutdown-home")
        runtime = Path(launch.client.runtime_path) if getattr(
            launch.client, "runtime_path", None
        ) else (tmp_path / "shutdown-home" / "runtime.json")
        assert runtime.exists()
        pid = launch.pid

        body = launch.client.shutdown_daemon(grace_ms=0)
        assert body["ok"] is True
        assert body["pid"] == pid

        deadline = time.time() + 15.0
        while time.time() < deadline and (runtime.exists() or L.pid_alive(pid)):
            time.sleep(0.05)
        assert not runtime.exists(), "runtime.json must be deleted on clean exit"
        assert not L.pid_alive(pid), "the daemon must actually be gone"
        launch.close()
        launch = None

        # And the handshake now correctly reports "nothing there" rather than
        # trusting the file it just removed.
        with pytest.raises(Exception):
            L.find_or_spawn(spawn=False, runtime_path=str(runtime))
    finally:
        _stop(launch)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
