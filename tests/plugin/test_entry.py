# -*- coding: utf-8 -*-
"""Tests for the GIMP entry point and the dev/install tooling.

Nothing here can import ``sam3_gimp.py``: it does ``gi.require_version("Gimp",
"3.0")`` at module scope, which fails without GIMP.  So the entry point is
checked the only way it can be -- statically, through ``ast`` --
and the parts of it that are pure Python (prompt parsing, coordinate scaling)
are lifted out of the module's AST and executed in isolation.  That is not a
substitute for running it inside GIMP, but it does catch the failures that
actually happen in practice: a syntax error, a procedure that is queried but
never created, a run function whose name does not exist, a declared surface that
has drifted from ``sam3_gimp/__init__.py``.

``tools/dev_sync.py`` needs no such contortions -- it is plain standard library
and is tested for real, including its Windows and macOS path resolution (via
injected environments) and its whole copy/clean/watch behaviour in a temp dir.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import pathlib
import stat
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_PKG_DIR = REPO_ROOT / "plugin" / "sam3_gimp"
ENTRY_PATH = PLUGIN_PKG_DIR / "sam3_gimp.py"
INIT_PATH = PLUGIN_PKG_DIR / "__init__.py"
DEV_SYNC_PATH = REPO_ROOT / "tools" / "dev_sync.py"
INSTALL_PS1_PATH = REPO_ROOT / "tools" / "install.ps1"

POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics")


# --------------------------------------------------------------------------- #
# loading helpers
# --------------------------------------------------------------------------- #
def _load_module(name: str, path: Path):
    """Import a module from an explicit path, independent of ``sys.path``."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def entry_source() -> str:
    return ENTRY_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def entry_tree(entry_source: str) -> ast.Module:
    return ast.parse(entry_source, filename=str(ENTRY_PATH))


@pytest.fixture(scope="module")
def entry_constants(entry_tree: ast.Module) -> dict:
    """Module-level literal assignments from the entry point.

    ``literal_eval`` is the point: it reads the constants without importing
    ``gi``, and it silently ignores anything that is not a literal (the regex,
    the class, the functions).
    """
    values = {}
    for node in entry_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError, TypeError):
            # Not a bare literal.  It may still be built from constants defined
            # above it (PROCEDURES is a tuple of the PROC_* names), so retry
            # with the constants seen so far and no builtins at all.
            try:
                code = compile(ast.Expression(body=node.value), str(ENTRY_PATH), "eval")
                value = eval(code, {"__builtins__": {}}, dict(values))
            except Exception:
                continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                values[target.id] = value
    return values


@pytest.fixture(scope="module")
def plugin_pkg():
    return _load_module("sam3_gimp_pkg_under_test", INIT_PATH)


@pytest.fixture(scope="module")
def dev_sync():
    return _load_module("sam3_dev_sync_under_test", DEV_SYNC_PATH)


@pytest.fixture(scope="module")
def pure_entry(entry_tree: ast.Module):
    """Execute only the GIMP-free part of the entry point.

    ``Sam3Error``, ``parse_points``, ``parse_box`` and ``scale_to_upload`` touch
    nothing but ``re``, so lifting those nodes out of the AST and executing them
    gives genuine unit tests of the scriptable prompt syntax -- the part most
    likely to be got wrong and the part a Script-Fu user hits first.
    """
    wanted_funcs = {"parse_points", "parse_box", "scale_to_upload"}
    wanted_classes = {"Sam3Error"}
    body = []
    for node in entry_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_funcs:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in wanted_classes:
            body.append(node)
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(n.startswith("_POINT_RE") for n in names):
                body.append(node)

    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"re": __import__("re")}
    exec(compile(module, filename=str(ENTRY_PATH), mode="exec"), namespace)

    missing = (wanted_funcs | wanted_classes) - set(namespace)
    assert not missing, "entry point lost pure helpers: %s" % (sorted(missing),)
    return namespace


# =========================================================================== #
# the entry file itself
# =========================================================================== #
def test_entry_file_is_named_after_its_directory():
    # GIMP will not even look at the plug-in otherwise.
    assert ENTRY_PATH.is_file(), ENTRY_PATH
    assert ENTRY_PATH.stem == PLUGIN_PKG_DIR.name


def test_entry_parses(entry_tree):
    assert isinstance(entry_tree, ast.Module)
    assert entry_tree.body


def test_entry_has_a_shebang(entry_source):
    assert entry_source.startswith("#!/usr/bin/env python3\n")


@POSIX_ONLY
def test_entry_is_executable():
    # On Linux/macOS GIMP skips plug-in scripts without the executable bit,
    # with no error message at all.
    mode = ENTRY_PATH.stat().st_mode
    assert mode & stat.S_IXUSR, "chmod +x plugin/sam3_gimp/sam3_gimp.py"


def test_entry_is_ascii(entry_source):
    # GIMP's embedded Python has been known to mis-handle source encodings on
    # Windows; the entry point stays 7-bit so that can never be the problem.
    non_ascii = sorted({c for c in entry_source if ord(c) > 127})
    assert not non_ascii, "non-ASCII characters in the entry point: %r" % (non_ascii,)


# =========================================================================== #
# procedure registration
# =========================================================================== #
#: Order matters and is asserted: registration order is menu order, and Setup
#: leads so a fresh install does not hide its only useful entry behind three
#: segmentation commands that cannot work yet.
EXPECTED_PROCEDURES = (
    "plug-in-sam3-setup",
    "plug-in-sam3-segment",
    "plug-in-sam3-segment-by-text",
    "plug-in-sam3-segment-by-points",
)


def test_declares_the_expected_procedure_names(entry_constants):
    assert entry_constants["PROC_SEGMENT"] == "plug-in-sam3-segment"
    assert entry_constants["PROC_SEGMENT_TEXT"] == "plug-in-sam3-segment-by-text"
    assert entry_constants["PROC_SEGMENT_POINTS"] == "plug-in-sam3-segment-by-points"
    assert entry_constants["PROC_SETUP"] == "plug-in-sam3-setup"
    assert tuple(entry_constants["PROCEDURES"]) == EXPECTED_PROCEDURES


def test_procedure_names_are_valid_pdb_names(entry_constants):
    # GIMP requires lowercase, '-' separated, at least one '-'.
    for name in entry_constants["PROCEDURES"]:
        assert name == name.lower()
        assert "-" in name
        assert "_" not in name
        assert all(ch.isalnum() or ch == "-" for ch in name), name


def _class_def(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError("class %s not found" % (name,))


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError("method %s not found on %s" % (name, cls.name))


def test_plugin_class_subclasses_gimp_plugin(entry_tree):
    cls = _class_def(entry_tree, "Sam3Plugin")
    bases = [ast.unparse(b) for b in cls.bases]
    assert bases == ["Gimp.PlugIn"]


@pytest.mark.parametrize("method", ["do_query_procedures", "do_create_procedure", "do_set_i18n"])
def test_plugin_class_implements_the_registration_methods(entry_tree, method):
    _method(_class_def(entry_tree, "Sam3Plugin"), method)


def test_query_procedures_returns_the_declared_tuple(entry_tree):
    body = _method(_class_def(entry_tree, "Sam3Plugin"), "do_query_procedures").body
    returns = [n for n in ast.walk(ast.Module(body=body, type_ignores=[])) if isinstance(n, ast.Return)]
    assert len(returns) == 1
    assert "PROCEDURES" in ast.unparse(returns[0])


def test_create_procedure_handles_every_queried_procedure(entry_tree):
    source = ast.unparse(_method(_class_def(entry_tree, "Sam3Plugin"), "do_create_procedure"))
    for constant in ("PROC_SEGMENT", "PROC_SEGMENT_TEXT", "PROC_SEGMENT_POINTS", "PROC_SETUP"):
        assert constant in source, "do_create_procedure ignores %s" % (constant,)


def test_every_run_callback_referenced_by_a_builder_exists(entry_tree):
    defined = {n.name for n in entry_tree.body if isinstance(n, ast.FunctionDef)}
    referenced = set()
    for node in ast.walk(entry_tree):
        if not isinstance(node, ast.Call):
            continue
        callee = ast.unparse(node.func)
        if callee not in ("Gimp.ImageProcedure.new", "Gimp.Procedure.new"):
            continue
        # (plug_in, name, proc_type, run_func, run_data)
        assert len(node.args) == 5, ast.unparse(node)
        run_func = node.args[3]
        assert isinstance(run_func, ast.Name), ast.unparse(node)
        referenced.add(run_func.id)

    assert referenced, "no procedures are constructed at all"
    missing = referenced - defined
    assert not missing, "run callbacks referenced but not defined: %s" % (sorted(missing),)


def test_image_run_functions_use_the_gimp3_signature(entry_tree):
    # GIMP 3.0: (procedure, run_mode, image, drawables, config, run_data).
    expected = ["procedure", "run_mode", "image", "drawables", "config", "run_data"]
    for name in ("run_segment", "run_segment_by_text", "run_segment_by_points"):
        func = next(n for n in entry_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert [a.arg for a in func.args.args] == expected, name


def test_module_calls_gimp_main(entry_tree):
    calls = [
        node
        for node in entry_tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "Gimp.main"
    ]
    assert len(calls) == 1, "Gimp.main must be called exactly once, at module level"
    assert ast.unparse(calls[0].value) == "Gimp.main(Sam3Plugin.__gtype__, sys.argv)"
    assert entry_tree.body[-1] is calls[0], "Gimp.main must be the last statement"


# =========================================================================== #
# run modes and status codes
# =========================================================================== #
def _attribute_chains(tree: ast.Module, prefix: str) -> set:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            text = ast.unparse(node)
            if text.startswith(prefix):
                found.add(text)
    return found


def test_all_three_run_modes_are_handled(entry_tree):
    modes = _attribute_chains(entry_tree, "Gimp.RunMode.")
    assert "Gimp.RunMode.INTERACTIVE" in modes
    assert "Gimp.RunMode.NONINTERACTIVE" in modes


def test_cancel_and_error_statuses_are_used(entry_tree):
    statuses = _attribute_chains(entry_tree, "Gimp.PDBStatusType.")
    for wanted in ("SUCCESS", "CANCEL", "EXECUTION_ERROR", "CALLING_ERROR"):
        assert "Gimp.PDBStatusType." + wanted in statuses, wanted


def test_interactive_path_uses_a_procedure_dialog(entry_tree):
    source = ast.unparse(entry_tree)
    assert "GimpUi.ProcedureDialog" in source
    assert "GimpUi.init" in source


def test_apply_is_delegated_to_outputs(entry_tree):
    # outputs.apply_result opens and closes the undo group itself, so one
    # Ctrl+Z reverts a whole Apply.  The entry point must not nest a second
    # group around it, and must not reimplement the plumbing.
    source = ast.unparse(entry_tree)
    assert "outputs.apply_result" in source
    assert "outputs.OutputOptions" in source
    assert "undo_group_start" not in source
    assert "undo_group_end" not in source


def test_output_mode_map_covers_every_mode(entry_constants):
    mapping = entry_constants["OUTPUT_MODE_MAP"]
    assert set(mapping) == set(entry_constants["OUTPUT_MODES"])
    # The right-hand side is outputs.OutputMode.ALL / outputs.SelectionOp.ALL.
    modes = {"selection", "channels", "layer-masks", "layer-groups", "paths"}
    ops = {"replace", "add", "subtract", "intersect"}
    for nick, (mode, op) in mapping.items():
        assert mode in modes, nick
        assert op in ops, nick
    # All four selection ops are reachable from the PDB surface.
    assert {op for mode, op in mapping.values() if mode == "selection"} == ops


def test_interactive_procedure_falls_back_when_the_canvas_is_absent(entry_tree):
    # ui/main_dialog.py may not be installed yet; the plug-in must degrade to
    # the argument dialog rather than refusing to run.
    source = ast.unparse(
        next(n for n in entry_tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_segment")
    )
    assert "ui.main_dialog" in source
    assert "optional=True" in source
    assert "run_main_dialog" in source
    assert "_run_scriptable" in source


def test_daemon_is_told_to_watch_gimps_pid(entry_tree):
    # DESIGN.md Constraint C: passing our own pid would kill the daemon as soon
    # as this short-lived plug-in process exits.
    source = ast.unparse(
        next(n for n in entry_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_gimp_pid")
    )
    assert "os.getppid" in source
    launch = ast.unparse(
        next(n for n in entry_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_launch")
    )
    assert "parent_pid=_gimp_pid()" in launch


# =========================================================================== #
# the zero-dependency rule
# =========================================================================== #
ALLOWED_NON_STDLIB = {"gi"}


def _top_level_imports(tree: ast.Module) -> set:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import, inside the package
                continue
            if node.module:
                names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("path", [ENTRY_PATH, INIT_PATH])
def test_plugin_files_import_only_stdlib_and_gi(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    offenders = sorted(
        name
        for name in _top_level_imports(tree)
        if name not in ALLOWED_NON_STDLIB and stdlib and name not in stdlib
    )
    assert not offenders, "%s imports non-stdlib modules: %s" % (path.name, offenders)


def test_the_entry_point_never_imports_the_daemon(entry_source):
    # sam3gimpd lives in a different Python with torch in it; API.md 16.10.
    assert "import sam3gimpd" not in entry_source
    assert "from sam3gimpd" not in entry_source


# =========================================================================== #
# declared surface consistency with __init__.py
# =========================================================================== #
def test_package_metadata_matches_the_entry_point(plugin_pkg, entry_constants):
    for name in (
        "PROC_SEGMENT",
        "PROC_SEGMENT_TEXT",
        "PROC_SEGMENT_POINTS",
        "PROC_SETUP",
        "MENU_PATH",
        "DEFAULT_OUTPUT_MODE",
        "DEFAULT_MASK_THRESHOLD",
        "DEFAULT_SCORE_THRESHOLD",
        "DAEMON_SCORE_THRESHOLD",
        "MAX_UPLOAD_SIDE",
        "API_VERSION",
        "__version__",
    ):
        assert getattr(plugin_pkg, name) == entry_constants[name], name
    assert tuple(plugin_pkg.PROCEDURES) == tuple(entry_constants["PROCEDURES"])
    assert tuple(plugin_pkg.OUTPUT_MODES) == tuple(entry_constants["OUTPUT_MODES"])
    assert plugin_pkg.OUTPUT_MODE_MAP == entry_constants["OUTPUT_MODE_MAP"]


def test_package_imports_without_gi(plugin_pkg):
    # It is imported by tooling on machines with no GIMP at all.
    assert plugin_pkg.PROCEDURES
    assert "gi" not in {m.split(".")[0] for m in dir(plugin_pkg)}


def test_output_mode_choices_agree_with_output_modes(entry_constants):
    nicks = tuple(nick for nick, _label, _blurb in entry_constants["OUTPUT_MODE_CHOICES"])
    assert nicks == tuple(entry_constants["OUTPUT_MODES"])
    assert entry_constants["DEFAULT_OUTPUT_MODE"] in nicks
    for _nick, label, blurb in entry_constants["OUTPUT_MODE_CHOICES"]:
        assert label and blurb


def test_thresholds_match_the_http_contract(entry_constants):
    # API.md 8: 128 == logit 0 == the default client threshold.
    assert entry_constants["DEFAULT_MASK_THRESHOLD"] == 128
    # API.md 6.3: the daemon is always asked for everything above 0.1 so the
    # client-side score slider costs no round trip.
    assert entry_constants["DAEMON_SCORE_THRESHOLD"] == 0.02
    # The floor caps what the local slider can ever show, so it must stay
    # comfortably below the default the slider starts at.
    assert (entry_constants["DAEMON_SCORE_THRESHOLD"]
            < entry_constants["DEFAULT_SCORE_THRESHOLD"])
    # API.md 15 / DESIGN 1: never upload more than the model canvas.
    assert entry_constants["MAX_UPLOAD_SIDE"] == 1008
    assert entry_constants["API_VERSION"] == "1.0"


# =========================================================================== #
# the pure helpers, executed for real
# =========================================================================== #
def test_parse_points_defaults_to_positive(pure_entry):
    assert pure_entry["parse_points"]("512,300") == [{"x": 512.0, "y": 300.0, "label": 1}]


def test_parse_points_reads_labels_and_separators(pure_entry):
    parsed = pure_entry["parse_points"]("512,300,1; 640.5,410,0\n10,20,1")
    assert parsed == [
        {"x": 512.0, "y": 300.0, "label": 1},
        {"x": 640.5, "y": 410.0, "label": 0},
        {"x": 10.0, "y": 20.0, "label": 1},
    ]


def test_parse_points_empty_is_empty(pure_entry):
    assert pure_entry["parse_points"]("") == []
    assert pure_entry["parse_points"](None) == []
    assert pure_entry["parse_points"]("  ;  ") == []


def test_parse_points_rejects_rubbish(pure_entry):
    with pytest.raises(pure_entry["Sam3Error"]):
        pure_entry["parse_points"]("512")
    with pytest.raises(pure_entry["Sam3Error"]):
        pure_entry["parse_points"]("512,300,7")
    with pytest.raises(pure_entry["Sam3Error"]):
        pure_entry["parse_points"]("a,b")


@pytest.mark.parametrize("spec", ["1.2.3,4", "4,1.2.3", "..,5", "-,5", "5,.", "1e3,4"])
def test_parse_points_turns_a_malformed_number_into_a_sam3error(pure_entry, spec):
    """A malformed number gets the point-syntax hint, not a bare ValueError
    from float() escaping as a traceback."""
    with pytest.raises(pure_entry["Sam3Error"]):
        pure_entry["parse_points"](spec)


def test_parse_points_accepts_every_plain_decimal_spelling(pure_entry):
    assert pure_entry["parse_points"]("12.,.5;-3,4.25,0") == [
        {"x": 12.0, "y": 0.5, "label": 1},
        {"x": -3.0, "y": 4.25, "label": 0},
    ]


def test_parse_points_enforces_the_documented_limit(pure_entry):
    # API.md 15: at most 64 points.
    ok = ";".join("%d,%d,1" % (i, i) for i in range(64))
    assert len(pure_entry["parse_points"](ok)) == 64
    with pytest.raises(pure_entry["Sam3Error"]):
        pure_entry["parse_points"](ok + ";1,1,1")


def test_parse_box(pure_entry):
    assert pure_entry["parse_box"]("400,250,700,480") == [400.0, 250.0, 700.0, 480.0]
    assert pure_entry["parse_box"]("400 250 700 480") == [400.0, 250.0, 700.0, 480.0]
    assert pure_entry["parse_box"]("") is None
    assert pure_entry["parse_box"](None) is None


def test_parse_box_rejects_bad_input(pure_entry):
    for spec in ("400,250,700", "a,b,c,d", "700,250,400,480", "400,480,700,250"):
        with pytest.raises(pure_entry["Sam3Error"]):
            pure_entry["parse_box"](spec)


def test_scale_to_upload_matches_the_worked_example(pure_entry):
    # API.md 12.7: original (1524, 893) on a 3000x2000 image uploaded at
    # 1008x672 becomes (512.06, 300.05) in uploaded-image space.
    scale = pure_entry["scale_to_upload"]
    assert scale(1524, 3000, 1008) == pytest.approx(512.064, abs=1e-3)
    assert scale(893, 2000, 672) == pytest.approx(300.048, abs=1e-3)
    # Degenerate inputs must not divide by zero.
    assert scale(10, 0, 1008) == 10.0
    assert scale(10, 3000, 0) == 10.0


# =========================================================================== #
# tools/dev_sync.py -- path resolution
# =========================================================================== #
def test_dev_sync_default_source_is_the_plugin_dir(dev_sync):
    assert dev_sync.default_source_dir() == PLUGIN_PKG_DIR.resolve()
    assert dev_sync.PLUGIN_NAME == "sam3_gimp"


def test_windows_plugin_dir(dev_sync):
    # Windows separators cannot be parsed by PurePosixPath, so the assertion
    # compares against the same join rather than a literal string.
    env = {"APPDATA": r"C:\Users\you\AppData\Roaming", "USERPROFILE": r"C:\Users\you"}
    result = dev_sync.gimp_plugin_dir(system="windows", environ=env)
    assert result == Path(env["APPDATA"]) / "GIMP" / "3.0" / "plug-ins"
    assert result.parts[-3:] == ("GIMP", "3.0", "plug-ins")


def test_windows_plugin_dir_without_appdata(dev_sync):
    env = {"USERPROFILE": r"C:\Users\you"}
    result = dev_sync.gimp_plugin_dir(system="windows", environ=env)
    expected = Path(env["USERPROFILE"]) / "AppData" / "Roaming" / "GIMP" / "3.0" / "plug-ins"
    assert result == expected


def test_macos_plugin_dir(dev_sync):
    env = {"HOME": "/Users/you"}
    result = dev_sync.gimp_plugin_dir(system="macos", environ=env)
    assert result == Path("/Users/you/Library/Application Support/GIMP/3.0/plug-ins")


def test_linux_plugin_dir(dev_sync):
    env = {"HOME": "/home/you"}
    result = dev_sync.gimp_plugin_dir(system="linux", environ=env, flatpak=False)
    assert result == Path("/home/you/.config/GIMP/3.0/plug-ins")


def test_linux_plugin_dir_honours_xdg(dev_sync):
    env = {"HOME": "/home/you", "XDG_CONFIG_HOME": "/home/you/cfg"}
    result = dev_sync.gimp_plugin_dir(system="linux", environ=env, flatpak=False)
    assert result == Path("/home/you/cfg/GIMP/3.0/plug-ins")


def test_linux_plugin_dir_ignores_relative_xdg(dev_sync):
    # The XDG spec says a relative XDG_CONFIG_HOME must be ignored.
    env = {"HOME": "/home/you", "XDG_CONFIG_HOME": "relative/path"}
    result = dev_sync.gimp_plugin_dir(system="linux", environ=env, flatpak=False)
    assert result == Path("/home/you/.config/GIMP/3.0/plug-ins")


def test_flatpak_can_be_forced(dev_sync):
    env = {"HOME": "/home/you"}
    result = dev_sync.gimp_plugin_dir(system="linux", environ=env, flatpak=True)
    assert result == Path("/home/you/.var/app/org.gimp.GIMP/config/GIMP/3.0/plug-ins")


def test_flatpak_autodetected_only_when_it_is_the_only_one(dev_sync, tmp_path):
    env = {"HOME": str(tmp_path)}
    flat = tmp_path / ".var/app/org.gimp.GIMP/config/GIMP/3.0/plug-ins"
    native = tmp_path / ".config/GIMP/3.0/plug-ins"

    # Neither exists -> native (the thing to create).
    assert dev_sync.gimp_plugin_dir(system="linux", environ=env) == native

    flat.mkdir(parents=True)
    assert dev_sync.gimp_plugin_dir(system="linux", environ=env) == flat

    native.mkdir(parents=True)
    assert dev_sync.gimp_plugin_dir(system="linux", environ=env) == native


def test_environment_override_wins(dev_sync):
    env = {"HOME": "/home/you", "GIMP3_PLUGIN_DIR": "/somewhere/else"}
    assert dev_sync.gimp_plugin_dir(system="linux", environ=env) == Path("/somewhere/else")
    env = {"HOME": "/home/you", "GIMP_PLUGIN_DIR": "/older/name"}
    assert dev_sync.gimp_plugin_dir(system="linux", environ=env) == Path("/older/name")


def test_version_is_part_of_the_path(dev_sync):
    env = {"HOME": "/home/you"}
    result = dev_sync.gimp_plugin_dir(version="3.2", system="linux", environ=env, flatpak=False)
    assert result == Path("/home/you/.config/GIMP/3.2/plug-ins")


# =========================================================================== #
# tools/dev_sync.py -- copying
# =========================================================================== #
@pytest.fixture
def fake_plugin(tmp_path: Path) -> Path:
    """A miniature plug-in tree with litter in it that must not be copied."""
    src = tmp_path / "src" / "sam3_gimp"
    (src / "ui").mkdir(parents=True)
    (src / "__pycache__").mkdir()
    (src / "sam3_gimp.py").write_text("#!/usr/bin/env python3\nENTRY = 1\n", encoding="utf-8")
    (src / "__init__.py").write_text("VERSION = '0.1.0'\n", encoding="utf-8")
    (src / "client.py").write_text("CLIENT = 1\n", encoding="utf-8")
    (src / "ui" / "canvas.py").write_text("CANVAS = 1\n", encoding="utf-8")
    (src / "__pycache__" / "client.cpython-310.pyc").write_bytes(b"\x00\x01")
    (src / "client.py~").write_text("backup\n", encoding="utf-8")
    (src / ".DS_Store").write_bytes(b"\x00")
    return src


def test_iter_source_files_skips_litter(dev_sync, fake_plugin):
    rels = {p.as_posix() for p in dev_sync.iter_source_files(fake_plugin)}
    assert rels == {"__init__.py", "client.py", "sam3_gimp.py", "ui/canvas.py"}


def test_sync_creates_the_correctly_named_directory(dev_sync, fake_plugin, tmp_path):
    dest_root = tmp_path / "plug-ins"
    report = dev_sync.sync(fake_plugin, dest_root)

    target = dest_root / "sam3_gimp"
    assert target.is_dir()
    assert (target / "sam3_gimp.py").is_file()
    assert (target / "ui" / "canvas.py").read_text(encoding="utf-8") == "CANVAS = 1\n"
    assert not (target / "__pycache__").exists()
    assert not (target / "client.py~").exists()
    assert sorted(report.created) == ["__init__.py", "client.py", "sam3_gimp.py", "ui/canvas.py"]
    assert report.updated == []
    assert report.changed


@POSIX_ONLY
def test_sync_sets_the_executable_bit_on_the_entry_file(dev_sync, fake_plugin, tmp_path):
    dest_root = tmp_path / "plug-ins"
    dev_sync.sync(fake_plugin, dest_root)
    entry = dest_root / "sam3_gimp" / "sam3_gimp.py"
    assert entry.stat().st_mode & stat.S_IXUSR
    # ... and only on the entry file; the rest are plain modules.
    assert not ((dest_root / "sam3_gimp" / "client.py").stat().st_mode & stat.S_IXUSR)


@POSIX_ONLY
def test_sync_repairs_a_lost_executable_bit(dev_sync, fake_plugin, tmp_path):
    dest_root = tmp_path / "plug-ins"
    dev_sync.sync(fake_plugin, dest_root)
    entry = dest_root / "sam3_gimp" / "sam3_gimp.py"
    entry.chmod(0o644)

    report = dev_sync.sync(fake_plugin, dest_root)
    assert entry.stat().st_mode & stat.S_IXUSR
    assert "sam3_gimp.py" in report.chmodded
    # The content was identical, so nothing was copied -- only the mode fixed.
    assert report.created == [] and report.updated == []


def test_sync_is_idempotent(dev_sync, fake_plugin, tmp_path):
    dest_root = tmp_path / "plug-ins"
    dev_sync.sync(fake_plugin, dest_root)
    report = dev_sync.sync(fake_plugin, dest_root)
    assert report.created == []
    assert report.updated == []
    assert len(report.unchanged) == 4
    assert not report.changed


def test_sync_copies_only_what_changed(dev_sync, fake_plugin, tmp_path):
    dest_root = tmp_path / "plug-ins"
    dev_sync.sync(fake_plugin, dest_root)
    (fake_plugin / "client.py").write_text("CLIENT = 2\n", encoding="utf-8")

    report = dev_sync.sync(fake_plugin, dest_root)
    assert report.updated == ["client.py"]
    assert report.created == []
    assert (dest_root / "sam3_gimp" / "client.py").read_text(encoding="utf-8") == "CLIENT = 2\n"


def test_dry_run_changes_nothing(dev_sync, fake_plugin, tmp_path):
    dest_root = tmp_path / "plug-ins"
    report = dev_sync.sync(fake_plugin, dest_root, dry_run=True)
    assert report.dry_run
    assert len(report.created) == 4
    assert not (dest_root / "sam3_gimp").exists()
    assert "would " in report.summary()


def test_clean_removes_stale_files(dev_sync, fake_plugin, tmp_path):
    dest_root = tmp_path / "plug-ins"
    dev_sync.sync(fake_plugin, dest_root)
    target = dest_root / "sam3_gimp"
    (target / "obsolete.py").write_text("gone\n", encoding="utf-8")
    (target / "old").mkdir()
    (target / "old" / "stale.py").write_text("gone\n", encoding="utf-8")

    report = dev_sync.sync(fake_plugin, dest_root, clean=True)
    assert sorted(report.removed) == ["obsolete.py", "old/stale.py"]
    assert not (target / "obsolete.py").exists()
    assert not (target / "old").exists()
    assert (target / "sam3_gimp.py").is_file()


def test_clean_is_off_by_default(dev_sync, fake_plugin, tmp_path):
    dest_root = tmp_path / "plug-ins"
    dev_sync.sync(fake_plugin, dest_root)
    stray = dest_root / "sam3_gimp" / "obsolete.py"
    stray.write_text("kept\n", encoding="utf-8")
    report = dev_sync.sync(fake_plugin, dest_root)
    assert report.removed == []
    assert stray.exists()


def test_dry_run_clean_does_not_delete(dev_sync, fake_plugin, tmp_path):
    dest_root = tmp_path / "plug-ins"
    dev_sync.sync(fake_plugin, dest_root)
    stray = dest_root / "sam3_gimp" / "obsolete.py"
    stray.write_text("kept\n", encoding="utf-8")
    report = dev_sync.sync(fake_plugin, dest_root, dry_run=True, clean=True)
    assert report.removed == ["obsolete.py"]
    assert stray.exists()


def test_sync_rejects_a_directory_without_the_entry_file(dev_sync, tmp_path):
    src = tmp_path / "sam3_gimp"
    src.mkdir()
    (src / "client.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError) as excinfo:
        dev_sync.sync(src, tmp_path / "plug-ins")
    assert "sam3_gimp.py" in str(excinfo.value)


def test_sync_rejects_a_missing_source(dev_sync, tmp_path):
    with pytest.raises(FileNotFoundError):
        dev_sync.sync(tmp_path / "nope", tmp_path / "plug-ins")


def test_sync_of_the_real_plugin_tree(dev_sync, tmp_path):
    """The actual repository tree must be syncable, entry file and all."""
    dest_root = tmp_path / "plug-ins"
    report = dev_sync.sync(PLUGIN_PKG_DIR, dest_root)
    target = dest_root / "sam3_gimp"
    assert (target / "sam3_gimp.py").is_file()
    assert "sam3_gimp.py" in report.created
    if os.name != "nt":
        assert (target / "sam3_gimp.py").stat().st_mode & stat.S_IXUSR


# =========================================================================== #
# tools/dev_sync.py -- watching and the CLI
# =========================================================================== #
def test_signature_changes_when_a_file_changes(dev_sync, fake_plugin):
    before = dev_sync.signature(fake_plugin)
    assert set(before) == {"__init__.py", "client.py", "sam3_gimp.py", str(Path("ui/canvas.py"))}
    (fake_plugin / "client.py").write_text("CLIENT = 999\n", encoding="utf-8")
    assert dev_sync.signature(fake_plugin) != before


def test_watch_syncs_once_before_polling(dev_sync, fake_plugin, tmp_path):
    dest_root = tmp_path / "plug-ins"
    reports = []
    dev_sync.watch(
        fake_plugin,
        dest_root,
        interval=0.01,
        on_sync=reports.append,
        stop=lambda: True,
    )
    assert len(reports) == 1
    assert (dest_root / "sam3_gimp" / "sam3_gimp.py").is_file()


def test_watch_resyncs_after_a_change(dev_sync, fake_plugin, tmp_path):
    dest_root = tmp_path / "plug-ins"
    reports = []
    state = {"passes": 0}

    def on_sync(report):
        reports.append(report)

    def stop():
        state["passes"] += 1
        # Mutate the source once the first sync is behind us, so the very next
        # poll must notice it.
        if state["passes"] == 1:
            (fake_plugin / "client.py").write_text("CLIENT = 42\n", encoding="utf-8")
        # A bounded number of polls, so a bug here fails rather than hangs.
        return len(reports) >= 2 or state["passes"] > 40

    dev_sync.watch(fake_plugin, dest_root, interval=0.01, on_sync=on_sync, stop=stop)
    assert len(reports) == 2
    assert reports[1].updated == ["client.py"]
    assert (dest_root / "sam3_gimp" / "client.py").read_text(encoding="utf-8") == "CLIENT = 42\n"


def test_cli_print_dest(dev_sync, capsys, tmp_path):
    code = dev_sync.main(["--print-dest", "--dest", str(tmp_path)])
    assert code == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "sam3_gimp")


def test_cli_dry_run(dev_sync, fake_plugin, capsys, tmp_path):
    dest_root = tmp_path / "plug-ins"
    code = dev_sync.main(
        ["--source", str(fake_plugin), "--dest", str(dest_root), "--dry-run", "--verbose"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "would 4 new" in out
    assert "+ sam3_gimp.py" in out
    assert not dest_root.exists()


def test_cli_syncs_and_reports(dev_sync, fake_plugin, capsys, tmp_path):
    dest_root = tmp_path / "plug-ins"
    code = dev_sync.main(["--source", str(fake_plugin), "--dest", str(dest_root)])
    assert code == 0
    out = capsys.readouterr().out
    assert "4 new" in out
    assert "restart GIMP" in out
    assert (dest_root / "sam3_gimp" / "sam3_gimp.py").is_file()


def test_cli_quiet_says_nothing(dev_sync, fake_plugin, capsys, tmp_path):
    dest_root = tmp_path / "plug-ins"
    code = dev_sync.main(["--source", str(fake_plugin), "--dest", str(dest_root), "--quiet"])
    assert code == 0
    assert capsys.readouterr().out == ""


def test_cli_missing_source_is_an_error(dev_sync, capsys, tmp_path):
    code = dev_sync.main(["--source", str(tmp_path / "nope"), "--dest", str(tmp_path)])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_cli_rejects_both_flatpak_flags(dev_sync):
    with pytest.raises(SystemExit):
        dev_sync.main(["--flatpak", "--no-flatpak", "--print-dest"])


# =========================================================================== #
# tools/install.ps1
# =========================================================================== #
@pytest.fixture(scope="module")
def install_ps1() -> str:
    return INSTALL_PS1_PATH.read_text(encoding="utf-8")


def test_install_script_exists(install_ps1):
    assert len(install_ps1) > 1000


def test_install_script_has_comment_based_help(install_ps1):
    assert install_ps1.lstrip().startswith("<#")
    for section in (".SYNOPSIS", ".DESCRIPTION", ".PARAMETER", ".EXAMPLE"):
        assert section in install_ps1, section


def test_install_script_declares_its_parameters(install_ps1):
    assert "[CmdletBinding()]" in install_ps1
    assert "param(" in install_ps1
    for parameter in ("$PluginDir", "$GimpVersion", "$Source", "$SetupVenv", "$DryRun", "$Clean"):
        assert parameter in install_ps1, parameter


def test_install_script_targets_the_documented_locations(install_ps1):
    assert "APPDATA" in install_ps1
    assert "GIMP\\$Version\\plug-ins" in install_ps1 or 'GIMP\\$Version\\plug-ins' in install_ps1
    assert "LOCALAPPDATA" in install_ps1
    assert "sam3-gimp" in install_ps1
    assert "sam3_gimp" in install_ps1


def test_install_script_stops_on_error(install_ps1):
    assert "$ErrorActionPreference = 'Stop'" in install_ps1
    assert "Set-StrictMode" in install_ps1
    assert "catch" in install_ps1


def test_install_script_uses_lf_or_crlf_consistently(install_ps1):
    raw = INSTALL_PS1_PATH.read_bytes()
    crlf = raw.count(b"\r\n")
    lf = raw.count(b"\n")
    assert crlf == 0 or crlf == lf, "mixed line endings in install.ps1"


def test_install_script_balances_braces(install_ps1):
    # A crude but effective smoke test: PowerShell will not even parse an
    # unbalanced script, and nothing here can run PowerShell.
    assert install_ps1.count("{") == install_ps1.count("}")
    assert install_ps1.count("(") == install_ps1.count(")")


def test_install_script_installs_the_daemon_from_inside_the_plugin(install_ps1):
    """The daemon moved to sam3_gimp/_daemon; -SetupVenv still pointed at a
    top-level daemon/ that no longer exists and failed on every checkout."""
    assert "Join-Path $repoRoot 'daemon'" not in install_ps1
    assert "Join-Path $sourceDir '_daemon'" in install_ps1


def test_install_script_does_not_run_pip_inside_a_uv_venv(install_ps1):
    """`uv venv` creates no pip, so `python -m pip` there fails with
    'No module named pip'.  With uv present the install must go through
    `uv pip install --python <venv python>`."""
    body = install_ps1.split("function New-Sam3Venv {")[1].split("\nfunction ")[0]
    assert "uv pip install --python $venvPython" in body
    # pip is only the fallback for a venv made by `python -m venv`
    assert body.index("uv pip install") < body.index("-m pip install")


def test_install_script_installs_torch_from_its_own_index(install_ps1):
    """torch comes from the PyTorch index alone, then the daemon's [runtime]
    extra from PyPI: no command mixes the two indexes, which is what
    unsafe-best-match used to allow (dependency confusion)."""
    code = [l for l in install_ps1.splitlines() if not l.lstrip().startswith("#")]
    assert not any("unsafe-best-match" in l or "--extra-index-url" in l for l in code)
    assert "--index-url $index" in install_ps1 and "@TorchRequirements" in install_ps1
    assert "$DaemonDir[runtime]" in install_ps1
    assert "pip install --index-url $CudaIndex torch" not in install_ps1


def test_install_script_pins_the_same_python_as_setup(install_ps1):
    import bootstrap as bs

    assert "$PythonVersion = '%s'" % bs.PINNED_PYTHON in install_ps1


def test_install_script_names_the_real_menu_entry(install_ps1):
    assert "Segment interactively (canvas)" in install_ps1
    assert "Segment with SAM 3" not in install_ps1


def test_install_script_pins_the_same_uv_as_setup(install_ps1):
    """Both install routes must fetch the same uv release, or they drift."""
    import bootstrap as bs

    assert "$UvVersion  = '%s'" % bs.UV_VERSION in install_ps1
    assert "releases/download/$UvVersion/" in install_ps1
    assert "astral.sh/uv/install.ps1" not in install_ps1, "that fetches whatever is newest"


# =========================================================================== #
# integration: the symbols the entry point actually reaches for
# =========================================================================== #
# The entry point imports its siblings lazily and by name, so a rename in one of
# them is invisible until someone runs GIMP.  These checks close that gap
# without GIMP.  Modules that cannot be imported here (GTK, a display) are skipped
# rather than failed: the point is to catch drift, not to require a desktop.
SIBLING_SYMBOLS = {
    "launcher": ["find_or_spawn"],
    "client": ["Sam3Client"],
    "gimpbridge": ["read_upload_pixels", "SOURCE_PROJECTION", "SOURCE_LAYER",
                   "source_layer"],
    "outputs": ["apply_result", "OutputOptions", "PostOps", "MaskResult"],
}


def _import_sibling_for_test(name):
    path = PLUGIN_PKG_DIR / (name.replace(".", "/") + ".py")
    if not path.is_file():
        pytest.skip("%s has not been written yet" % (path.name,))
    sys.path.insert(0, str(PLUGIN_PKG_DIR))
    try:
        return _load_module("sam3_sibling_" + name.replace(".", "_"), path)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip("%s is not importable here: %s" % (name, exc))
    finally:
        try:
            sys.path.remove(str(PLUGIN_PKG_DIR))
        except ValueError:
            pass


@pytest.mark.parametrize("modname,symbols", sorted(SIBLING_SYMBOLS.items()))
def test_sibling_modules_expose_what_the_entry_point_calls(modname, symbols):
    module = _import_sibling_for_test(modname)
    missing = [s for s in symbols if not hasattr(module, s)]
    assert not missing, "%s is missing %s (sam3_gimp.py calls it)" % (modname, missing)


def test_client_exposes_the_synchronous_prompt_helpers():
    client = _import_sibling_for_test("client")
    for method in ("upload_image", "run_text", "run_points", "close"):
        assert callable(getattr(client.Sam3Client, method, None)), method


def test_output_mode_map_matches_outputs_module(entry_constants):
    outputs = _import_sibling_for_test("outputs")
    modes = set(getattr(outputs.OutputMode, "ALL", ()))
    ops = set(getattr(outputs.SelectionOp, "ALL", ()))
    assert modes and ops
    for nick, (mode, op) in entry_constants["OUTPUT_MODE_MAP"].items():
        assert mode in modes, "%s maps to unknown outputs mode %r" % (nick, mode)
        assert op in ops, "%s maps to unknown selection op %r" % (nick, op)


def test_main_dialog_entry_point_exists_if_the_module_does():
    path = PLUGIN_PKG_DIR / "ui" / "main_dialog.py"
    if not path.is_file():
        pytest.skip("ui/main_dialog.py has not been written yet")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert names & {"run_main_dialog", "run_dialog"}, (
        "sam3_gimp.py calls ui.main_dialog.run_main_dialog (or run_dialog)"
    )


def test_setup_dialog_entry_points_exist():
    path = PLUGIN_PKG_DIR / "ui" / "setup_dialog.py"
    if not path.is_file():
        pytest.skip("ui/setup_dialog.py has not been written yet")
    # Importing it needs GTK and a display; the names it promises are checked
    # statically instead, which is enough to catch a rename.
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "run_setup" in names, "sam3_gimp.py calls ui.setup_dialog.run_setup"


# --------------------------------------------------------------------------- #
# GIMP version detection -- regression for a silent wrong-directory sync
# --------------------------------------------------------------------------- #
class TestGimpVersionDetection:
    """Hardcoding "3.0" was wrong the moment GIMP 3.2 shipped.

    On a 3.2 machine the sync wrote into ``GIMP/3.0/plug-ins``, GIMP never
    looked there, the plug-in silently did not appear, and nothing said why.
    """

    @staticmethod
    def _win_env(tmp_path, versions):
        for v in versions:
            (tmp_path / "GIMP" / v / "plug-ins").mkdir(parents=True)
        return {"APPDATA": str(tmp_path)}

    def test_picks_the_newest_version_present(self, dev_sync, tmp_path):
        env = self._win_env(tmp_path, ["3.0", "3.2"])
        assert dev_sync.detect_gimp_version("windows", env) == "3.2"

    def test_double_digit_minors_sort_numerically_not_lexically(self, dev_sync, tmp_path):
        """"3.10" must beat "3.2"; a string sort would get this backwards."""
        env = self._win_env(tmp_path, ["3.2", "3.10"])
        assert dev_sync.detect_gimp_version("windows", env) == "3.10"

    def test_falls_back_when_nothing_is_installed(self, dev_sync, tmp_path):
        env = {"APPDATA": str(tmp_path)}
        assert dev_sync.detect_gimp_version("windows", env) == dev_sync.DEFAULT_GIMP_VERSION

    def test_ignores_unrelated_directories(self, dev_sync, tmp_path):
        (tmp_path / "GIMP" / "themes").mkdir(parents=True)
        (tmp_path / "GIMP" / "2.10").mkdir(parents=True)
        env = {"APPDATA": str(tmp_path)}
        assert dev_sync.detect_gimp_version("windows", env) == dev_sync.DEFAULT_GIMP_VERSION

    def test_plugin_dir_uses_the_detected_version(self, dev_sync, tmp_path):
        env = self._win_env(tmp_path, ["3.0", "3.2"])
        got = dev_sync.gimp_plugin_dir(system="windows", environ=env)
        assert got.parts[-2] == "3.2"

    def test_an_explicit_version_still_wins(self, dev_sync, tmp_path):
        env = self._win_env(tmp_path, ["3.0", "3.2"])
        got = dev_sync.gimp_plugin_dir("3.0", system="windows", environ=env)
        assert got.parts[-2] == "3.0"


class TestDaemonIsBundled:
    """The daemon is installed from a path, never by name, so its source has to
    travel with the plug-in."""

    def test_bundle_includes_the_daemon_package_and_its_pyproject(self, dev_sync):
        mapping = dev_sync.iter_bundle_files(REPO_ROOT / "plugin" / "sam3_gimp")
        rels = {str(k).replace("\\", "/") for k in mapping}
        assert "_daemon/pyproject.toml" in rels
        assert any(r.startswith("_daemon/sam3gimpd/") for r in rels)
        assert "sam3_gimp.py" in rels

    def test_every_bundled_source_actually_exists(self, dev_sync):
        mapping = dev_sync.iter_bundle_files(REPO_ROOT / "plugin" / "sam3_gimp")
        missing = [str(v) for v in mapping.values() if not v.is_file()]
        assert missing == []


class TestMenuIsSelfExplanatory:
    """Setup comes first and every entry says what it does.

    Registration order is menu order, and burying Setup under three
    segmentation commands is how a user presses Segment to discover that
    nothing is installed.
    """

    def test_setup_is_registered_first(self, entry_source):
        assert "PROC_SETUP," in entry_source
        block = entry_source.split("PROCEDURES = (")[1].split(")")[0]
        names = [line.strip().rstrip(",") for line in block.strip().splitlines()]
        assert names[0] == "PROC_SETUP", names

    def test_every_menu_label_names_its_prompt_or_job(self, entry_source):
        for label in ("Segment interactively (canvas)",
                      "Segment by _text prompt (all matching objects)",
                      "Segment by _points or box (one object)"):
            assert label in entry_source, label

    def test_the_setup_label_reports_an_uninstalled_environment(self, entry_source):
        assert "first-time set_up required" in entry_source
        assert "_env_ready()" in entry_source

    def test_every_segment_procedure_checks_readiness_first(self, entry_source):
        """Both run paths must gate, or one of them still fails late."""
        assert entry_source.count("_require_setup(procedure, run_mode)") >= 2

    def test_readiness_probe_cannot_raise(self, entry_source):
        """It runs during registration; an exception there hides the plug-in."""
        body = entry_source.split("def _env_ready():")[1].split("\ndef ")[0]
        assert "except Exception:" in body and "return False" in body

    def test_setup_refreshes_the_cached_menu_label(self, entry_source):
        """GIMP caches menu labels in pluginrc until the plug-in file's mtime
        changes, so a label that reports 'setup required' would say so for
        ever after a successful install unless Setup touches the entry file."""
        assert "def _refresh_menu_label(" in entry_source
        body = entry_source.split("def _refresh_menu_label(")[1].split("\ndef ")[0]
        assert "os.utime(" in body
        assert "except Exception" in body, "touching our own file is best effort"
        run_setup = entry_source.split("def run_setup(")[1].split("\n# ====")[0]
        assert "ready_before = _env_ready()" in run_setup
        assert "_refresh_menu_label(ready_before)" in run_setup


class TestRegistrationIsCheap:
    """`_env_ready` runs while GIMP builds its procedure database.

    It must not spawn anything there. The first version called
    ``inspect_environment()`` with no accelerator, which falls through to
    ``nvidia-smi -L`` with a 15 second timeout -- on every GIMP start, and on
    Windows with a console window flashing.
    """

    def test_env_ready_passes_a_stub_accelerator(self, entry_source):
        body = entry_source.split("def _env_ready():")[1].split("\ndef ")[0]
        assert "Accelerator(" in body, "must not let inspect_environment probe"
        assert "accelerator=stub" in body

    def test_env_ready_still_cannot_raise(self, entry_source):
        body = entry_source.split("def _env_ready():")[1].split("\ndef ")[0]
        assert "except Exception:" in body and "return False" in body

    def test_inspect_environment_skips_the_probe_when_given_one(self, monkeypatch):
        """The behaviour the entry point relies on."""
        import bootstrap as bs

        called = []
        monkeypatch.setattr(bs, "detect_accelerator",
                            lambda *a, **k: called.append(1) or bs.Accelerator("cpu", "x"))
        bs.inspect_environment(accelerator=bs.Accelerator("unknown", "stub"))
        assert called == []


class TestTuningIsExplained:
    """The two thresholds are the whole tuning story.

    Bare numbers in the argument dialog explain nothing, and a prompt like
    "guitar" returning a fragment while "person" works comes back to the
    score threshold: SAM 3 routinely returns one object as several partial
    matches, and a scriptable default of 0.5 discarded most of them silently.
    """

    def test_the_scriptable_default_matches_the_canvas(self, entry_constants):
        """Three different defaults across the code was itself a bug."""
        assert entry_constants["DEFAULT_SCORE_THRESHOLD"] == 0.30

    def test_every_argument_says_which_way_to_move_it(self, entry_source):
        block = entry_source.split("def _add_common_arguments(")[1].split("\ndef ")[0]
        for phrase in ("LOWER IT", "RAISE IT"):
            assert phrase in block, phrase
        assert "try 0.15" in block and "try 90" in block

    def test_the_score_help_names_the_fragment_symptom(self, entry_source):
        block = entry_source.split("def _add_common_arguments(")[1].split("\ndef ")[0]
        assert "fragment" in block
        assert "several partial matches" in block

    def test_output_modes_are_described_individually(self, entry_source):
        block = entry_source.split("def _add_common_arguments(")[1].split("\ndef ")[0]
        for mode in ("Selection", "Channels", "Layer mask", "Paths"):
            assert mode in block, mode

    def test_a_selection_report_exists(self, entry_source):
        assert "def _report_selection(" in entry_source
        assert "_report_selection(mask_result.instances" in entry_source

    def test_the_report_only_interrupts_when_matches_were_dropped(self, entry_source):
        body = entry_source.split("def _report_selection(")[1].split("\ndef ")[0]
        assert "if kept and dropped:" in body
        assert "Gimp.message" in body

    def test_the_report_suggests_a_concrete_new_threshold(self, entry_source):
        body = entry_source.split("def _report_selection(")[1].split("\ndef ")[0]
        assert "best_dropped" in body and "lower the score threshold" in body



class TestSetupIsAlwaysReachable:
    """Setup must open with no image and no drawable -- the state of a first run.

    GIMP's default sensitivity for any procedure is "an image with one or more
    drawables selected". A Setup entry that never set a mask was greyed out on
    an empty GIMP, the moment the user is told to open it.
    """

    def test_setup_sets_an_always_sensitivity_mask(self, entry_source):
        block = entry_source.split("def _create_setup(self, name):")[1].split("\n    def ")[0]
        assert "set_sensitivity_mask(_always_sensitive())" in block

    def test_the_mask_helper_cannot_break_registration(self, entry_source):
        body = entry_source.split("def _always_sensitive():")[1].split("\ndef ")[0]
        assert 'getattr(mask_type, "ALWAYS", None)' in body
        statements = [ln.strip() for ln in body.splitlines()]
        assert not any(ln.startswith("raise") for ln in statements)

    def test_the_argument_dialog_gets_a_real_setup_button(self, entry_source):
        """A button beside OK / Reset / Cancel -- not a checkbox among the arguments."""
        assert "def _add_setup_button(dialog):" in entry_source
        assert 'dialog.add_button("_Setup / Doctor' in entry_source
        assert '"open-setup"' not in entry_source, "the checkbox should be gone"

    def test_the_button_is_honoured_before_the_readiness_gate(self, entry_source):
        body = entry_source.split("def _run_scriptable(")[1].split("\ndef ")[0]
        assert body.index("if wants_setup():") < body.index("_require_setup(procedure, run_mode)")

    def test_only_the_canvas_and_setup_are_menu_entries(self, entry_source):
        for fn in ("_create_segment_by_text", "_create_segment_by_points"):
            block = entry_source.split("def %s(self, name):" % fn)[1].split("\n    def ")[0]
            assert "menu=False" in block, fn
        canvas = entry_source.split("def _create_segment(self, name):")[1].split("\n    def ")[0]
        assert "menu=False" not in canvas



class TestPlugInDoesNotDieOnExit:
    """The plug-in process must not die after the dialog closes.

    libgimp calls exit() the moment run() returns; a daemon thread still mid
    HTTP call re-enters Python during finalisation and aborts the process,
    which GIMP reports as "Plug-in crashed".  Timing-dependent, so it can
    look as if it depends on what was typed.
    """

    def test_crash_diagnostics_are_armed_at_import(self, entry_source):
        assert "faulthandler.enable(file=_CRASH_LOG, all_threads=True)" in entry_source
        assert "sys.excepthook = _hook" in entry_source
        assert "threading.excepthook = _thread_hook" in entry_source
        assert "\n_install_crash_diagnostics()\n" in entry_source

    def test_the_first_log_line_names_the_interpreter_gimp_handed_us(
            self, entry_tree, tmp_path, monkeypatch):
        """``plugin.log`` must open by saying which Python is running.

        GIMP chooses that interpreter, not us -- its own in the Windows and
        macOS bundles, the distro's on Linux -- so "it does not load on my
        GIMP" is unanswerable without it.  Lifted out of the AST rather than
        run under a fake ``gi``, so it is checked on every interpreter in the
        CI matrix and not only on the one job that has PyGObject.
        """
        import faulthandler
        import threading
        import time
        import traceback
        import types as _types

        wanted = {"_plugin_log_path", "_install_crash_diagnostics", "_trim_log",
                  "_open_log", "_owner_only"}
        body = [node for node in entry_tree.body
                if isinstance(node, ast.FunctionDef) and node.name in wanted]
        assert {node.name for node in body} == wanted, "entry point lost its log helpers"
        body += [node for node in entry_tree.body if isinstance(node, ast.Assign)
                 and any(getattr(t, "id", "").startswith("LOG_") for t in node.targets)]

        module = ast.Module(body=body, type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {
            "os": os, "pathlib": pathlib, "sys": sys, "time": time, "traceback": traceback,
            "__version__": "0.0.0-test",
            "_CRASH_LOG": None,
            "_log": lambda *a, **k: None,
            # The real one imports launcher, which imports gi.
            "_import_sibling": lambda *a, **k: _types.SimpleNamespace(
                plugin_build=lambda: "testbuild"),
        }
        exec(compile(module, filename=str(ENTRY_PATH), mode="exec"), namespace)

        # Leave the session's own diagnostics alone: monkeypatch records the
        # live values now and restores them once the entry point has overwritten
        # them.  Only ``enable`` is stubbed, never the whole module -- pytest's
        # faulthandler plugin calls into the real one the moment a test fails,
        # and a stand-in module turns that failure into an INTERNALERROR.
        monkeypatch.setattr(sys, "excepthook", sys.excepthook)
        monkeypatch.setattr(threading, "excepthook", threading.excepthook)
        monkeypatch.setattr(faulthandler, "enable", lambda *a, **k: None)
        monkeypatch.setenv("SAM3_GIMP_HOME", str(tmp_path))

        try:
            namespace["_install_crash_diagnostics"]()
        finally:
            if namespace["_CRASH_LOG"] is not None:
                namespace["_CRASH_LOG"].close()

        first = (tmp_path / "logs" / "plugin.log").read_text(
            encoding="utf-8").splitlines()[0]
        assert "pid %d" % os.getpid() in first, first
        assert "Python %d.%d.%d" % sys.version_info[:3] in first, first
        assert sys.executable in first, first
        if os.name != "nt":
            mode = stat.S_IMODE(os.stat(str(tmp_path / "logs" / "plugin.log")).st_mode)
            assert mode == 0o600, oct(mode)

    def test_run_segment_quiesces_threads_on_every_exit_path(self, entry_source):
        body = entry_source.split("def run_segment(")[1].split("\ndef ")[0]
        assert body.count("_quiesce_threads()") >= 3

    def test_quiesce_only_waits_for_our_own_threads_with_a_deadline(self, entry_source):
        body = entry_source.split("def _quiesce_threads(")[1].split("\ndef ")[0]
        assert 'name.startswith("sam3-")' in body
        assert "thread.join(remaining)" in body
        assert "still running at exit" in body



class TestHangDump:
    """A run that exceeds 45 s writes every thread's stack to plugin.log."""

    def test_armed_at_the_start_of_every_run_function(self, entry_source):
        for fn in ("def run_segment(", "def run_setup(", "def _run_scriptable("):
            body = entry_source.split(fn)[1].split("\ndef ")[0]
            assert "_arm_hang_dump()" in body, fn

    def test_disarmed_on_every_return_path(self, entry_source):
        for fn in ("def _success(", "def _cancel(", "def _failure("):
            body = entry_source.split(fn)[1].split("\ndef ")[0]
            assert "_disarm_hang_dump()" in body, fn

    def test_uses_faulthandler_dump_traceback_later(self, entry_source):
        assert "dump_traceback_later(HANG_DUMP_AFTER_S" in entry_source
        assert "cancel_dump_traceback_later()" in entry_source



def test_gimp_pid_never_falls_back_to_our_own_pid(entry_source):
    """Passing our own pid killed the daemon the moment run() returned."""
    body = entry_source.split("def _gimp_pid():")[1].split("\ndef ")[0]
    assert "os.getpid()" not in body
    assert 'getattr(launcher, "gimp_pid", None)' in body


def test_the_plugin_and_the_daemon_carry_the_mit_notice():
    """MIT requires the notice to travel with every copy.  The plug-in folder is
    what users copy into GIMP by hand, and the daemon folder is what pip builds
    a wheel from, so each holds the repository's LICENSE verbatim."""
    text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    for copy in (REPO_ROOT / "plugin" / "sam3_gimp" / "LICENSE",
                 REPO_ROOT / "plugin" / "sam3_gimp" / "_daemon" / "LICENSE"):
        assert copy.read_text(encoding="utf-8") == text, "%s differs from LICENSE" % copy
