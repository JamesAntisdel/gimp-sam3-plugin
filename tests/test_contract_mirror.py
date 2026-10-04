"""The plug-in mirrors ``sam3gimpd/types.py`` by hand -- keep the copies honest.

``API.md`` §16.10 forbids the plug-in from importing the daemon's types module:
different Python, zero third-party imports.  The cost of that rule is two
hand-maintained copies of the same constants, in two files owned by two
different people, that nothing links together.  This file is the link.

Nothing here starts a daemon or takes a second to run, so a drift introduced by
an edit to either half is caught by the fastest test in the suite rather than by
a user whose masks land in the wrong place.

The daemon side is the authority (``API.md``: "``types.Limits`` is the source of
truth"); a mismatch means the *plug-in* is wrong unless the change was a
deliberate contract revision, in which case ``API.md`` moves too.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT / "plugin" / "sam3_gimp" / "_daemon"),
           str(REPO_ROOT / "plugin" / "sam3_gimp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import client as C  # noqa: E402
import launcher as L  # noqa: E402
from sam3gimpd import paths as P  # noqa: E402
from sam3gimpd import types as T  # noqa: E402


def _constants(obj) -> dict:
    """The UPPER_CASE attributes of a constants holder, minus its own indexes."""
    skip = {"ALL", "STATUS"}
    return {k: getattr(obj, k) for k in dir(obj) if k.isupper() and k not in skip}


# =========================================================================== #
# wire constants
# =========================================================================== #
def test_api_version_matches():
    assert C.API_VERSION == L.API_VERSION == T.API_VERSION


def test_the_hello_identity_proof_matches():
    """The plug-in checks the daemon's proof itself; both halves must compute
    the same HMAC over the same bytes, under the same header and field names."""
    import secrets

    assert C.NONCE_HEADER == T.NONCE_HEADER
    assert C.NONCE_PROOF_PREFIX == T.NONCE_PROOF_PREFIX
    token, nonce = secrets.token_urlsafe(32), secrets.token_urlsafe(24)
    assert T.is_valid_nonce(nonce), "the client's nonce shape is one the daemon answers"
    assert C.nonce_proof(token, nonce) == T.nonce_proof(token, nonce)
    hello = T.HelloResponse(api_version=T.API_VERSION, sam3d_version="0", engine_mode="stub",
                            device="stub", dtype="none",
                            nonce_proof=T.nonce_proof(token, nonce))
    assert hello.to_dict()[C.NONCE_PROOF_KEY] == C.nonce_proof(token, nonce)


def test_the_default_score_threshold_matches():
    assert C.DEFAULT_SCORE_THRESHOLD == T.DEFAULT_SCORE_THRESHOLD


def test_the_client_accepts_the_largest_frame_the_daemon_can_send():
    most = T.Limits.MAX_INSTANCES * T.Limits.MAX_IMAGE_SIDE ** 2
    assert C.MAX_RESULT_FRAME_BYTES > most


def test_frame_constants_match():
    assert C.RESULT_MAGIC == T.RESULT_MAGIC
    assert C.RESULT_PREFIX_SIZE == T.RESULT_PREFIX_SIZE
    assert C.RESULT_PREFIX_SIZE == len(T.RESULT_MAGIC) + 4


def test_result_content_type_matches_the_server():
    from sam3gimpd import server  # imported here: it is heavier than types

    assert C.RESULT_CONTENT_TYPE == server.CONTENT_TYPE_RESULT


def test_mask_encoding_matches():
    assert C.MASK_ENCODING_U8_SOFT == T.MASK_ENCODING_U8_SOFT


def test_default_threshold_is_the_models_own_binarisation_point():
    """§8.3: 128 == round(255 * sigmoid(0)).  Not "half of 255" by taste."""
    from sam3gimpd import masks

    assert C.DEFAULT_MASK_THRESHOLD == 128
    assert round(255 * masks.sigmoid(0.0)) == C.DEFAULT_MASK_THRESHOLD


# =========================================================================== #
# limits (§15)
# =========================================================================== #
def test_limits_are_identical():
    assert _constants(C.Limits) == _constants(T.Limits)


def test_limits_match_the_documented_numbers():
    assert T.Limits.MAX_IMAGE_SIDE == 1008
    assert T.Limits.MIN_IMAGE_SIDE == 16
    assert T.Limits.MAX_UPLOAD_BYTES == 1008 * 1008 * 3
    assert T.Limits.MAX_JSON_BYTES == 256 * 1024
    assert T.Limits.MAX_TEXT_CHARS == 512
    assert T.Limits.MAX_POINTS == 64
    assert T.Limits.MAX_BOXES == 16
    assert T.Limits.MAX_INSTANCES == 256
    assert T.Limits.MAX_REQUEST_ID_CHARS == 64
    assert T.Limits.MAX_LONG_POLL_SECONDS == 30.0


# =========================================================================== #
# closed vocabularies (§4, §11)
# =========================================================================== #
@pytest.mark.parametrize("name", ["ErrorCode", "JobState", "Engine"])
def test_vocabularies_are_identical(name):
    assert _constants(getattr(C, name)) == _constants(getattr(T, name))


def test_error_codes_cover_the_documented_list():
    documented = {
        "bad_request", "invalid_json", "missing_header", "bad_dimensions",
        "payload_size_mismatch", "version_mismatch", "unauthorized",
        "forbidden_host", "not_found", "image_not_found", "job_not_found",
        "method_not_allowed", "image_not_ready", "payload_too_large",
        "unsupported_media_type", "model_load_failed", "inference_failed",
        "internal_error", "engine_unavailable", "queue_full", "shutting_down",
    }
    assert set(T.ErrorCode.ALL) == documented
    assert set(_constants(C.ErrorCode).values()) == documented


def test_job_states_cover_the_documented_list():
    assert set(T.JobState.ALL) == {
        "queued", "running", "done", "failed", "superseded", "cancelled"
    }


# =========================================================================== #
# the on-disk layout (§3.1) -- two implementations, one answer
# =========================================================================== #
_LAYOUT = (
    ("base_dir", "base_dir"),
    ("runtime_file", "runtime_file"),
    ("lock_file", "lock_file"),
    ("log_dir", "log_dir"),
    ("server_log", "server_log"),
    ("crash_log", "crash_log"),
    ("hf_home", "hf_home"),
)


@pytest.mark.parametrize("daemon_fn,plugin_fn", _LAYOUT)
@pytest.mark.parametrize(
    "env",
    [
        {"SAM3_GIMP_HOME": "/tmp/sam3-mirror-home"},
        {"SAM3_GIMP_HOME": "/tmp//sam3-mirror-home/"},
        {"SAM3_GIMP_HOME": "~no-such-user-sam3/home"},
        {"XDG_DATA_HOME": "/tmp/sam3-mirror-xdg"},
        {"XDG_DATA_HOME": "/tmp//sam3-mirror-xdg/"},
        {},  # platform default
    ],
    ids=["sam3_gimp_home", "sam3_gimp_home_untidy", "sam3_gimp_home_unknown_user",
         "xdg_data_home", "xdg_data_home_untidy", "platform_default"],
)
def test_paths_agree(daemon_fn, plugin_fn, env, monkeypatch):
    """``sam3gimpd.paths`` is canonical; ``launcher`` mirrors it in stdlib (§3.1).

    A disagreement here means the plug-in writes or reads ``runtime.json``
    somewhere the daemon never looks, and find-or-spawn loops forever.
    """
    for key in ("SAM3_GIMP_HOME", "SAM3D_RUNTIME_FILE", "XDG_DATA_HOME"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert str(getattr(P, daemon_fn)()) == str(getattr(L, plugin_fn)())


def test_runtime_file_override_is_honoured_by_both(monkeypatch, tmp_path):
    """``SAM3D_RUNTIME_FILE`` replaces the full path, not just the directory."""
    target = tmp_path / "elsewhere" / "rt.json"
    monkeypatch.setenv("SAM3D_RUNTIME_FILE", str(target))
    assert str(P.runtime_file()) == str(target)
    assert str(L.runtime_file()) == str(target)


def test_daemon_environ_carries_the_documented_variables(monkeypatch, tmp_path):
    """§13: the launcher hands the child HF_HOME, SAM3_GIMP_HOME and friends."""
    monkeypatch.setenv("SAM3_GIMP_HOME", str(tmp_path))
    monkeypatch.delenv("SAM3D_RUNTIME_FILE", raising=False)
    for key in ("HF_HOME", "HF_HUB_DISABLE_PROGRESS_BARS", "PYTHONUNBUFFERED"):
        monkeypatch.delenv(key, raising=False)

    env = L.daemon_environ()
    assert env["SAM3_GIMP_HOME"] == str(tmp_path)
    assert env["HF_HOME"] == str(L.hf_home())
    assert env["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert env["PYTHONUNBUFFERED"] == "1"

    daemon_side = P.daemon_environ()
    for key in ("HF_HOME", "SAM3_GIMP_HOME", "HF_HUB_DISABLE_PROGRESS_BARS",
                "PYTHONUNBUFFERED"):
        assert daemon_side[key] == env[key], key


def test_both_halves_respect_a_user_set_hf_home(monkeypatch, tmp_path):
    """DESIGN.md §7 wants the 3.6 GB cache under our base dir *by default*, but a
    user who already keeps one HuggingFace cache should not be made to download
    the checkpoint twice.  Both sides use ``setdefault``, and they must keep
    agreeing about that -- a launcher that forced its own value while the daemon
    honoured the user's would point the two at different caches."""
    monkeypatch.setenv("SAM3_GIMP_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "my-own-cache"))
    assert L.daemon_environ()["HF_HOME"] == str(tmp_path / "my-own-cache")
    assert P.daemon_environ()["HF_HOME"] == str(tmp_path / "my-own-cache")


# =========================================================================== #
# the zero-dependency rule (§16.10)
# =========================================================================== #
#: Everything GIMP's embedded Python actually imports.
#:
#: ``_daemon/`` is excluded deliberately, and the distinction is the whole
#: architecture: it sits inside the plug-in directory so that the folder GIMP
#: loads is self-contained, but it is a *separate package for a separate
#: interpreter*.  It is pip-installed into its own environment and imports
#: torch; GIMP never imports a line of it.  Scanning it here would either fail
#: this rule or, worse, tempt someone to relax the rule for the half of the tree
#: that genuinely must obey it.
PLUGIN_SOURCES = sorted(
    p for p in (REPO_ROOT / "plugin" / "sam3_gimp").rglob("*.py")
    if "_daemon" not in p.parts
)


#: The daemon half, which may import anything it declares as a dependency.
DAEMON_SOURCES = sorted(
    (REPO_ROOT / "plugin" / "sam3_gimp" / "_daemon").rglob("*.py")
)


def _imported_roots(path: Path) -> set:
    """Top-level package names a module imports, from its AST.

    The AST rather than a substring search, so that a *comment* saying "must not
    import sam3gimpd" -- which gimpbridge.py has, and rightly -- is not a violation.
    """
    import ast

    roots = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:      # a relative import stays inside the plug-in
                continue
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


def test_the_plugin_never_imports_the_daemons_types_module():
    """The rule that makes this whole file necessary."""
    assert PLUGIN_SOURCES, "the plug-in tree disappeared"
    for path in PLUGIN_SOURCES:
        assert "sam3gimpd" not in _imported_roots(path), path.name


def test_the_plugin_imports_only_the_standard_library_and_gi():
    """``DESIGN.md`` §3: GIMP's embedded Python has nothing else in it.

    Every third-party import in this tree is a plug-in that fails to load on a
    user's machine, which is the failure mode that killed the project this one
    succeeds.
    """
    allowed = set(sys.stdlib_module_names) | {
        "gi",                       # GObject Introspection: the one exception
        "cairo",                    # ships with PyGObject inside GIMP
        # sibling plug-in modules, imported flat because GIMP puts the plug-in
        # directory on sys.path
        "client", "launcher", "outputs", "gimpbridge", "bootstrap", "ui",
        "sam3_gimp",
    }
    for path in PLUGIN_SOURCES:
        extra = _imported_roots(path) - allowed
        assert not extra, "%s imports %s" % (path.name, sorted(extra))


def test_plugin_sources_are_seven_bit_ascii_where_gimp_needs_it():
    """``sam3_gimp.py`` is read by GIMP's own loader before any encoding
    declaration takes effect on some Windows builds, so the entry point stays
    pure ASCII.  The rest of the tree may use UTF-8 freely."""
    entry = REPO_ROOT / "plugin" / "sam3_gimp" / "sam3_gimp.py"
    raw = entry.read_bytes()
    offenders = [i for i, b in enumerate(raw) if b > 0x7F]
    assert not offenders, "non-ASCII byte at offset %s" % (offenders[:5],)
    assert os.access(entry, os.X_OK), "GIMP only loads an executable plug-in file"


# =========================================================================== #
# output vocabularies: three files, one set of strings
# =========================================================================== #
def _module_constants(path: Path, names) -> dict:
    """Literal module-level constants, read from the AST.

    ``sam3_gimp.py`` and ``ui/main_dialog.py`` cannot be imported without GIMP
    and GTK respectively, but their vocabularies still have to agree with
    ``outputs.py`` -- so read them rather than import them.
    """
    import ast

    wanted = set(names)
    found = {}
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                found[target.id] = ast.literal_eval(node.value)
    missing = wanted - set(found)
    assert not missing, "%s has no %s" % (path.name, sorted(missing))
    return found


ENTRY = REPO_ROOT / "plugin" / "sam3_gimp" / "sam3_gimp.py"
MAIN_DIALOG = REPO_ROOT / "plugin" / "sam3_gimp" / "ui" / "main_dialog.py"


def test_the_pdb_output_modes_expand_into_real_outputs_values():
    """The entry point flattens ``(mode, selection_op)`` into one PDB choice.

    Every pair it can produce has to be a pair ``outputs.apply_result`` accepts,
    or a Script-Fu caller gets a validation error from deep inside Apply.
    """
    import outputs

    consts = _module_constants(
        ENTRY, ("OUTPUT_MODE_MAP", "OUTPUT_MODE_CHOICES", "DEFAULT_OUTPUT_MODE")
    )
    for nick, (mode, op) in consts["OUTPUT_MODE_MAP"].items():
        assert mode in outputs.OutputMode.ALL, nick
        assert op in outputs.SelectionOp.ALL, nick
    # Every mode outputs.py can build must be reachable from the PDB.
    assert {m for m, _op in consts["OUTPUT_MODE_MAP"].values()} == set(outputs.OutputMode.ALL)
    # ... and every selection op, via the four flattened selection nicks.
    assert {op for m, op in consts["OUTPUT_MODE_MAP"].values() if m == "selection"} == \
        set(outputs.SelectionOp.ALL)

    nicks = [nick for nick, _label, _tip in consts["OUTPUT_MODE_CHOICES"]]
    assert nicks == list(consts["OUTPUT_MODE_MAP"]), "choice order must match the map"
    assert consts["DEFAULT_OUTPUT_MODE"] in consts["OUTPUT_MODE_MAP"]


def test_the_canvas_dialogs_combo_ids_are_outputs_values():
    """``MainDialog`` hands its combo's active id straight to ``OutputOptions``."""
    import outputs

    consts = _module_constants(MAIN_DIALOG, ("OUTPUT_MODES", "SELECTION_OPS"))
    assert [mode for mode, _label in consts["OUTPUT_MODES"]] == list(outputs.OutputMode.ALL)
    assert [op for op, _label in consts["SELECTION_OPS"]] == list(outputs.SelectionOp.ALL)


def test_every_half_agrees_on_the_default_thresholds():
    """128 is fixed by the contract; the score default is a UI taste decision and
    is allowed to differ between the scriptable path and the canvas -- but the
    value *sent to the daemon* must be the contract floor in both, or the score
    slider stops being a local, round-trip-free filter (§6.3)."""
    entry = _module_constants(
        ENTRY, ("DEFAULT_MASK_THRESHOLD", "DAEMON_SCORE_THRESHOLD", "MAX_UPLOAD_SIDE")
    )
    dialog = _module_constants(
        MAIN_DIALOG, ("DEFAULT_MASK_THRESHOLD", "PROMPT_SCORE_FLOOR", "MAX_UPLOAD_SIDE")
    )
    assert entry["DEFAULT_MASK_THRESHOLD"] == C.DEFAULT_MASK_THRESHOLD == 128
    assert dialog["DEFAULT_MASK_THRESHOLD"] == C.DEFAULT_MASK_THRESHOLD
    # 0.02, not 0.1.  This value is a hard ceiling on what the score slider can
    # reveal: post_process_instance_segmentation drops everything below it
    # server-side, so at 0.1 the slider's bottom tenth was dead and weak matches
    # were unreachable -- "guitar" kept only the headstock and "guitar strap"
    # returned nothing.  It must stay well under any user-facing default.
    assert entry["DAEMON_SCORE_THRESHOLD"] == C.DEFAULT_SCORE_THRESHOLD == 0.02
    assert entry["DAEMON_SCORE_THRESHOLD"] < 0.30, "the floor must sit under the UI default"
    assert dialog["PROMPT_SCORE_FLOOR"] == C.DEFAULT_SCORE_THRESHOLD
    assert entry["MAX_UPLOAD_SIDE"] == dialog["MAX_UPLOAD_SIDE"] == T.Limits.MAX_IMAGE_SIDE


def test_the_daemon_is_not_scanned_as_plug_in_code():
    """The two halves live in one directory but are not one program.

    ``_daemon/`` ships inside the plug-in so the installed folder is complete,
    but it runs in its own interpreter with torch. If it ever ended up in
    PLUGIN_SOURCES the zero-dependency rule above would start failing on code
    that is *supposed* to import torch.
    """
    assert DAEMON_SOURCES, "the daemon tree disappeared"
    assert not any("_daemon" in p.parts for p in PLUGIN_SOURCES)
    names = {p.name for p in DAEMON_SOURCES}
    assert {"server.py", "cli.py"} <= names


def test_the_daemon_ships_inside_the_plugin_directory():
    """Regression: it used to live at the repository top level and be copied in
    by a build step, so a hand-copied plug-in folder had nothing to install
    from and the only way out was a terminal."""
    bundled = REPO_ROOT / "plugin" / "sam3_gimp" / "_daemon"
    assert (bundled / "pyproject.toml").is_file()
    assert (bundled / "sam3gimpd" / "__init__.py").is_file()
    assert not (REPO_ROOT / "daemon").exists(), "the old top-level copy is back"


def test_the_interpreter_gate_matches_the_daemons_requires_python():
    """The dialog's floor and the package's floor are the same number.

    They are written in two files nobody edits together: ``interpreter_problems``
    in ``bootstrap.py`` decides whether Setup will accept an environment the user
    already has, and ``requires-python`` in ``_daemon/pyproject.toml`` decides
    whether the daemon can actually be installed into it.  When they disagreed
    -- the gate at 3.9, the package at 3.10 -- Setup approved an interpreter and
    then pip refused it, in pip's words rather than ours.

    Parsed by hand rather than with ``tomllib``: this suite runs on 3.10, where
    there is none.
    """
    import re

    import bootstrap as bs  # noqa: PLC0415

    text = (REPO_ROOT / "plugin" / "sam3_gimp" / "_daemon"
            / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^requires-python\s*=\s*"([^"]+)"', text, re.M)
    assert match, "requires-python disappeared from the daemon's pyproject"
    spec = match.group(1)

    floor = re.match(r">=\s*(\d+)\.(\d+)$", spec.strip())
    assert floor, (
        "requires-python is %r; a floor with no ceiling is deliberate -- the "
        "'use my own PyTorch' flow runs the daemon in whatever interpreter the "
        "user already has" % spec)
    assert bs.MIN_INTERPRETER == (int(floor.group(1)), int(floor.group(2))), (
        "bootstrap.MIN_INTERPRETER %r does not match requires-python %r"
        % (bs.MIN_INTERPRETER, spec))
