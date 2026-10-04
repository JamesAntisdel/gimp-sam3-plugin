#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Copy ``plugin/sam3_gimp/`` into GIMP's plug-in directory.

This is the inner development loop.  GIMP will not load a plug-in from a
checkout: it wants ``<plug-ins>/<name>/<name>.py``, with the directory and the
entry file sharing a name, and on POSIX with the entry file executable.  Running
this script puts the tree there, correctly named and correctly permissioned, in
about a millisecond -- and ``--watch`` repeats that whenever a file changes, so
the edit/reload cycle is "save, then restart GIMP" instead of "save, then
remember six manual steps".

Usage::

    python3 tools/dev_sync.py                  # sync once to the default dir
    python3 tools/dev_sync.py --dry-run        # say what would change, change nothing
    python3 tools/dev_sync.py --watch          # sync now, then on every change
    python3 tools/dev_sync.py --clean          # also delete stale files in the target
    python3 tools/dev_sync.py --print-dest     # just print the resolved directory
    python3 tools/dev_sync.py --dest D:\\gimp\\plug-ins

Target directory, in precedence order:

1. ``--dest``
2. ``$GIMP3_PLUGIN_DIR`` / ``$GIMP_PLUGIN_DIR``
3. the platform default:

   =========  ==========================================================
   Windows    ``%APPDATA%\\GIMP\\<ver>\\plug-ins``
   macOS      ``~/Library/Application Support/GIMP/<ver>/plug-ins``
   Linux      ``$XDG_CONFIG_HOME/GIMP/<ver>/plug-ins`` (``~/.config`` default)
   =========  ==========================================================

   ``<ver>`` is ``--gimp-version`` if given, else the newest ``3.x``
   configuration directory that exists.  If there is none (GIMP has never been
   started), the command stops and asks for ``--gimp-version`` or ``--dest``
   rather than guessing: a copy into the wrong version's directory is never
   loaded, and nothing says why.

   On Linux, if the Flatpak configuration directory
   (``~/.var/app/org.gimp.GIMP/config/GIMP/<ver>/plug-ins``) exists and the
   plain one does not, it is used automatically; ``--flatpak`` / ``--no-flatpak``
   force the decision either way.

What is copied: in a git checkout, the files git tracks under
``plugin/sam3_gimp`` (with their working-tree content, so uncommitted edits are
synced), minus caches and editor litter.  Symlinks are never copied.  Untracked
files are left behind and listed, so a local ``.env``, a virtualenv or a log
cannot leak into GIMP's plug-in directory; ``git add`` a new module to include
it.  Outside a git checkout the directory is walked instead, with the same
filters.

Standard library only, Python 3.8+, no dependency on GIMP being installed --
which is what lets ``tests/plugin/test_entry.py`` exercise every path resolution
and the whole copy behaviour on a machine with no GIMP at all.
"""

from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

#: Name of the plug-in directory *and* of its entry file.  GIMP requires them to
#: match; this constant is the single place that fact is written down.
PLUGIN_NAME = "sam3_gimp"

#: GIMP major.minor used by :func:`gimp_plugin_dir` when asked for a path and
#: no configuration directory exists yet.  The command line never falls back to
#: it; see :func:`resolve_plugin_dir`.
DEFAULT_GIMP_VERSION = "3.0"

#: Flatpak application id for GIMP.
FLATPAK_APP_ID = "org.gimp.GIMP"

#: Environment overrides for the destination, highest precedence first.
DEST_ENV_VARS = ("GIMP3_PLUGIN_DIR", "GIMP_PLUGIN_DIR")

#: Directory names never copied.
SKIP_DIRS = frozenset({"__pycache__", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache"})

#: Filename suffixes never copied.
SKIP_SUFFIXES = ("~", ".pyc", ".pyo", ".orig", ".rej", ".swp")

#: Exact filenames never copied.
SKIP_NAMES = frozenset({".DS_Store", "Thumbs.db"})

#: Files that must be executable in the target on POSIX.  GIMP refuses to load
#: a plug-in whose entry script is not marked executable.
EXECUTABLE_NAMES = frozenset({PLUGIN_NAME + ".py"})

#: git index modes that are not regular files: a symlink and a submodule.
GIT_SYMLINK_MODE = "120000"
GIT_GITLINK_MODE = "160000"


# --------------------------------------------------------------------------- #
# path resolution
# --------------------------------------------------------------------------- #
def default_source_dir() -> Path:
    """``plugin/sam3_gimp`` relative to this file, without needing a checkout root."""
    return (Path(__file__).resolve().parent.parent / "plugin" / PLUGIN_NAME).resolve()


def platform_key(system: str = None) -> str:
    """``"windows"`` / ``"macos"`` / ``"linux"``.

    Mirrors ``sam3gimpd.paths.platform_key`` deliberately -- the two halves of the
    project must not disagree about what platform they are on.  ``system`` lets
    tests ask about a platform they are not running on.
    """
    if system:
        return system
    if os.name == "nt" or sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def _home(environ) -> Path:
    for var in ("HOME", "USERPROFILE"):
        value = environ.get(var)
        if value:
            return Path(value)
    profile = environ.get("HOMEDRIVE", "") + environ.get("HOMEPATH", "")
    if profile:
        return Path(profile)
    return Path(os.path.expanduser("~"))


def flatpak_plugin_dir(version: str = DEFAULT_GIMP_VERSION, environ=None) -> Path:
    """Where a Flatpak GIMP looks for user plug-ins."""
    environ = os.environ if environ is None else environ
    return (
        _home(environ)
        / ".var"
        / "app"
        / FLATPAK_APP_ID
        / "config"
        / "GIMP"
        / version
        / "plug-ins"
    )



def gimp_config_root(system: str = None, environ=None) -> Path:
    """The directory that holds GIMP's per-version config folders."""
    environ = os.environ if environ is None else environ
    kind = platform_key(system)
    home = _home(environ)
    if kind == "windows":
        appdata = environ.get("APPDATA")
        return (Path(appdata) if appdata else home / "AppData" / "Roaming") / "GIMP"
    if kind == "macos":
        return home / "Library" / "Application Support" / "GIMP"
    xdg = environ.get("XDG_CONFIG_HOME")
    return (Path(xdg) if xdg and os.path.isabs(xdg) else home / ".config") / "GIMP"


def gimp_config_roots(system: str = None, environ=None, flatpak=None):
    """Every directory that may hold GIMP's per-version config folders."""
    roots = [gimp_config_root(system, environ)]
    if flatpak is not False and platform_key(system) == "linux":
        environ_ = os.environ if environ is None else environ
        roots.append(_home(environ_) / ".var" / "app" / FLATPAK_APP_ID
                     / "config" / "GIMP")
    return roots


def detect_gimp_version(system: str = None, environ=None, flatpak=None,
                        default=DEFAULT_GIMP_VERSION):
    """The newest GIMP 3.x config directory that actually exists.

    Hardcoding "3.0" was wrong the moment GIMP 3.2 shipped: the sync silently
    wrote into ``GIMP/3.0/plug-ins`` on a machine running 3.2, the plug-in never
    appeared, and nothing said why.  So look at what is on disk and take the
    highest version.  With nothing on disk, return ``default``; the command
    line passes ``None`` so that it can refuse to guess.
    """
    roots = gimp_config_roots(system, environ, flatpak)
    found = []
    for root in roots:
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir():
                continue
            parts = entry.name.split(".")
            if len(parts) == 2 and all(p.isdigit() for p in parts) and parts[0] == "3":
                found.append((int(parts[0]), int(parts[1]), entry.name))
    if not found:
        return default
    found.sort()
    return found[-1][2]

def gimp_plugin_dir(
    version: str = None,
    system: str = None,
    environ=None,
    flatpak=None,
) -> Path:
    """Resolve GIMP's user plug-in directory.

    ``flatpak`` is tri-state: ``True`` forces the Flatpak location, ``False``
    forces the native one, ``None`` (the default) picks Flatpak only when its
    directory already exists and the native one does not -- which is exactly the
    situation on a machine where GIMP was installed from Flathub.

    The environment overrides are honoured on every platform; they are how a
    user with a portable or Snap install points the sync somewhere unusual.
    """
    environ = os.environ if environ is None else environ

    for var in DEST_ENV_VARS:
        value = environ.get(var)
        if value:
            return Path(value).expanduser()

    if version is None:
        version = detect_gimp_version(system, environ, flatpak)

    kind = platform_key(system)
    home = _home(environ)

    if kind == "windows":
        appdata = environ.get("APPDATA")
        base = Path(appdata) if appdata else home / "AppData" / "Roaming"
        return base / "GIMP" / version / "plug-ins"

    if kind == "macos":
        return home / "Library" / "Application Support" / "GIMP" / version / "plug-ins"

    xdg = environ.get("XDG_CONFIG_HOME")
    native = (Path(xdg) if xdg and os.path.isabs(xdg) else home / ".config") / "GIMP" / version / "plug-ins"
    flat = flatpak_plugin_dir(version, environ)

    if flatpak is True:
        return flat
    if flatpak is False:
        return native
    if flat.is_dir() and not native.is_dir():
        return flat
    return native


def resolve_plugin_dir(version: str = None, system: str = None, environ=None,
                       flatpak=None) -> Path:
    """:func:`gimp_plugin_dir`, but refuse to guess a GIMP version.

    With no ``version``, no environment override and no ``GIMP/3.x`` config
    directory on disk, there is nothing to go on: GIMP has not been started
    yet, or it keeps its configuration somewhere unusual.  Guessing "3.0" is
    how a sync ends up in a directory GIMP 3.2 never reads, so raise instead
    and say how to be explicit.
    """
    environ = os.environ if environ is None else environ
    if version is None and not any(environ.get(var) for var in DEST_ENV_VARS):
        version = detect_gimp_version(system, environ, flatpak, default=None)
        if version is None:
            roots = ", ".join(str(root) for root in gimp_config_roots(system, environ, flatpak))
            raise FileNotFoundError(
                "no GIMP 3.x configuration directory found (looked in %s). "
                "Start GIMP once so it creates one, or say where to put the "
                "plug-in with --gimp-version X.Y or --dest DIR" % roots
            )
    return gimp_plugin_dir(version=version, system=system, environ=environ, flatpak=flatpak)


# --------------------------------------------------------------------------- #
# the copy
# --------------------------------------------------------------------------- #
def should_skip(path: Path) -> bool:
    """True for build droppings and editor litter."""
    name = path.name
    if name in SKIP_NAMES:
        return True
    if path.is_dir():
        return name in SKIP_DIRS
    return any(name.endswith(suffix) for suffix in SKIP_SUFFIXES)


def is_litter(rel: Path) -> bool:
    """:func:`should_skip` for a relative path, judged by name alone.

    Used on git's file list, where a tracked ``__pycache__/x.pyc`` is still
    litter and nothing on disk needs to be consulted to say so.
    """
    rel = Path(rel)
    if any(part in SKIP_DIRS for part in rel.parts[:-1]):
        return True
    name = rel.name
    return name in SKIP_NAMES or any(name.endswith(suffix) for suffix in SKIP_SUFFIXES)


def _git(source: Path, *args):
    """Run ``git -C source args...``; ``None`` if git is missing or fails."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(source)] + list(args),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def git_tracked_files(source: Path):
    """``{relative Path: git mode}`` for every path git tracks under ``source``.

    Paths are relative to ``source``; the mode is git's octal string
    (``100644``, ``100755``, ``120000`` for a symlink, ``160000`` for a
    submodule).  ``None`` when ``source`` is not inside a git work tree or git
    is not installed.
    """
    out = _git(Path(source), "ls-files", "--stage", "-z", "--", ".")
    if out is None:
        return None
    tracked = {}
    for record in out.split(b"\0"):
        if not record:
            continue
        meta, _, path = record.partition(b"\t")
        tracked[Path(os.fsdecode(path))] = meta.split(b" ", 1)[0].decode("ascii", "replace")
    return tracked


def git_untracked_files(source: Path):
    """Untracked, not-ignored files under ``source`` (relative), or ``[]``."""
    out = _git(Path(source), "ls-files", "--others", "--exclude-standard", "-z", "--", ".")
    if not out:
        return []
    return sorted(Path(os.fsdecode(p)) for p in out.split(b"\0") if p)


def _walk_files(root: Path, skip_links: bool = True):
    """Every non-litter file under ``root``, relative and sorted.

    Directories are pruned rather than filtered afterwards so ``__pycache__``
    is never even walked.  ``os.walk`` does not descend into symlinked
    directories; ``skip_links`` also drops symlinked files.
    """
    root = Path(root)
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        dir_path = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if not should_skip(dir_path / d))
        for filename in sorted(filenames):
            candidate = dir_path / filename
            if should_skip(candidate):
                continue
            if skip_links and candidate.is_symlink():
                continue
            results.append(candidate.relative_to(root))
    results.sort()
    return results


def select_source_files(source: Path):
    """``(files, skipped)`` for ``source``: what to copy, and what was left out.

    In a git checkout, ``files`` is what git tracks (by name; the content is
    the working tree's), and ``skipped`` lists ``(relative path, reason)`` for
    tracked symlinks and submodules, tracked files deleted from the working
    tree, and untracked files.  Outside git -- or when the entry file is not
    tracked, which means ``source`` is not this project's checkout -- the
    directory is walked with the same filters and symlinks are dropped.
    """
    source = Path(source)
    tracked = git_tracked_files(source)
    if tracked is None or Path(PLUGIN_NAME + ".py") not in tracked:
        return _walk_files(source), []

    files, skipped = [], []
    for rel in sorted(tracked):
        mode = tracked[rel]
        if is_litter(rel):
            continue
        path = source / rel
        if mode == GIT_SYMLINK_MODE or path.is_symlink():
            skipped.append((rel, "symlink"))
        elif mode == GIT_GITLINK_MODE:
            skipped.append((rel, "submodule"))
        elif not path.is_file():
            skipped.append((rel, "tracked but missing from the working tree"))
        else:
            files.append(rel)
    for rel in git_untracked_files(source):
        if not is_litter(rel):
            skipped.append((rel, "untracked; git add it to include it"))
    return files, skipped


def iter_source_files(source: Path):
    """Every file to copy, as a path relative to ``source``, sorted.

    See :func:`select_source_files` for what is included.
    """
    return select_source_files(source)[0]


#: The daemon lives *inside* the plug-in directory (``sam3_gimp/_daemon``) and
#: is therefore picked up with everything else.  That way the repository's own
#: ``plugin/sam3_gimp`` is complete: copying that folder by hand produces an
#: install that has the daemon to install *from*.
BUNDLED_DAEMON_DIRNAME = "_daemon"


def iter_bundle_files(source: Path, daemon: Path = None):
    """``{relative destination: absolute source}`` for everything to install.

    Kept as a mapping (rather than a plain list) because callers depend on the
    shape.  ``daemon`` is accepted for compatibility and ignored: the daemon is
    part of ``source``.
    """
    source = Path(source)
    return {rel: source / rel for rel in iter_source_files(source)}


class SyncReport:
    """What one :func:`sync` call did (or, with ``dry_run``, would have done)."""

    def __init__(self, source: Path, target: Path, dry_run: bool = False):
        self.source = Path(source)
        self.target = Path(target)
        self.dry_run = bool(dry_run)
        self.created = []
        self.updated = []
        self.unchanged = []
        self.removed = []
        self.chmodded = []
        #: ``(relative path, reason)`` for files deliberately not copied.
        self.skipped = []

    @property
    def changed(self) -> bool:
        return bool(self.created or self.updated or self.removed or self.chmodded)

    def summary(self) -> str:
        prefix = "would " if self.dry_run else ""
        return "%s%d new, %d updated, %d unchanged, %d removed" % (
            prefix,
            len(self.created),
            len(self.updated),
            len(self.unchanged),
            len(self.removed),
        )

    def lines(self):
        verb = {"new": "+", "upd": "~", "del": "-", "chmod": "x"}
        out = []
        for rel in self.created:
            out.append("%s %s" % (verb["new"], rel))
        for rel in self.updated:
            out.append("%s %s" % (verb["upd"], rel))
        for rel in self.removed:
            out.append("%s %s" % (verb["del"], rel))
        for rel in self.chmodded:
            out.append("%s %s (+x)" % (verb["chmod"], rel))
        return out

    def skipped_lines(self):
        return ["not copied: %s (%s)" % (rel, reason) for rel, reason in self.skipped]

    def __repr__(self):
        return "<SyncReport %s -> %s: %s>" % (self.source, self.target, self.summary())


def _needs_copy(src: Path, dst: Path) -> bool:
    """Content-aware: compare size and bytes, not mtime.

    ``shallow=False`` costs nothing at this scale (a few dozen small files) and
    means a checkout whose mtimes were rewritten by git does not trigger a
    pointless full re-copy -- which matters for ``--watch``, where a spurious
    "changed" is a spurious "restart GIMP".
    """
    if not dst.exists():
        return True
    try:
        return not filecmp.cmp(str(src), str(dst), shallow=False)
    except OSError:
        return True


def _wants_exec(rel: Path) -> bool:
    return rel.name in EXECUTABLE_NAMES


def _is_executable(path: Path) -> bool:
    try:
        return bool(path.stat().st_mode & stat.S_IXUSR)
    except OSError:
        return False


def _make_executable(path: Path) -> None:
    """chmod +x for user/group/other, preserving the read/write bits."""
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def sync(
    source: Path = None,
    dest_root: Path = None,
    dry_run: bool = False,
    clean: bool = False,
) -> SyncReport:
    """Copy ``source`` into ``dest_root/sam3_gimp``.

    ``dest_root`` is GIMP's *plug-ins* directory; the correctly named
    sub-directory is created inside it.  Returns a :class:`SyncReport`.  With
    ``dry_run`` nothing on disk is touched, not even directory creation.

    ``clean`` deletes files under the target that the source no longer has --
    off by default, because a stray file in a plug-in directory is harmless and
    deleting things the user did not ask about is not.
    """
    source = default_source_dir() if source is None else Path(source)
    if not source.is_dir():
        raise FileNotFoundError("plug-in source directory not found: %s" % (source,))
    entry = source / (PLUGIN_NAME + ".py")
    if not entry.is_file():
        raise FileNotFoundError(
            "%s does not contain %s.py -- GIMP requires the entry file to be "
            "named after its directory" % (source, PLUGIN_NAME)
        )

    dest_root = resolve_plugin_dir() if dest_root is None else Path(dest_root)
    target = dest_root / PLUGIN_NAME
    report = SyncReport(source, target, dry_run=dry_run)

    files, skipped = select_source_files(source)
    report.skipped = [(Path(rel).as_posix(), reason) for rel, reason in skipped]
    bundle = {rel: source / rel for rel in files}
    wanted = sorted(bundle)
    wanted_set = set(wanted)

    for rel in wanted:
        src = bundle[rel]
        dst = target / rel
        if _needs_copy(src, dst):
            if dst.exists():
                report.updated.append(rel.as_posix())
            else:
                report.created.append(rel.as_posix())
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(str(src), str(dst))
                shutil.copymode(str(src), str(dst))
        else:
            report.unchanged.append(rel.as_posix())

        # The executable bit is checked every run, not only after a copy: a
        # target copied from a Windows share, or extracted from a zip, arrives
        # without it and GIMP then silently ignores the plug-in.
        if os.name != "nt" and _wants_exec(rel):
            if dry_run:
                if not dst.exists() or not _is_executable(dst):
                    report.chmodded.append(rel.as_posix())
            elif not _is_executable(dst):
                _make_executable(dst)
                report.chmodded.append(rel.as_posix())

    if clean and target.is_dir():
        # The target is walked, never listed from version control: it is
        # GIMP's directory, and whatever is in it is a candidate for removal,
        # links included.
        for rel in _walk_files(target, skip_links=False):
            if rel in wanted_set:
                continue
            report.removed.append(rel.as_posix())
            if not dry_run:
                try:
                    (target / rel).unlink()
                except OSError:
                    pass
        if not dry_run:
            _prune_empty_dirs(target)

    return report


def _prune_empty_dirs(root: Path) -> None:
    """Remove directories left empty by ``--clean`` (deepest first, root kept)."""
    for dirpath, dirnames, filenames in os.walk(str(root), topdown=False):
        if Path(dirpath) == root:
            continue
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# watching
# --------------------------------------------------------------------------- #
def signature(source: Path):
    """A cheap fingerprint of the source tree: ``{relpath: (mtime_ns, size)}``.

    Polling beats inotify/ReadDirectoryChangesW here: the tree is tiny, the
    watcher must work identically on Windows and Linux, and ``watchdog`` is a
    dependency this project refuses to take.
    """
    source = Path(source)
    result = {}
    for rel in iter_source_files(source):
        try:
            st = (source / rel).stat()
        except OSError:
            continue
        result[str(rel)] = (st.st_mtime_ns, st.st_size)
    return result


def watch(
    source: Path,
    dest_root: Path,
    interval: float = 1.0,
    clean: bool = False,
    on_sync=None,
    stop=None,
):
    """Sync once, then re-sync whenever the source fingerprint changes.

    ``stop`` is a zero-argument predicate polled between passes; ``on_sync`` is
    called with each :class:`SyncReport`.  Both exist so the loop is drivable
    from a test without a signal handler or a background thread.
    """
    # Fingerprint *before* the initial sync, so an edit made while the copy is
    # in flight is caught on the next pass instead of being swallowed.
    last = signature(source)
    report = sync(source, dest_root, clean=clean)
    if on_sync:
        on_sync(report)
    while True:
        if stop is not None and stop():
            return
        time.sleep(max(0.05, float(interval)))
        if stop is not None and stop():
            return
        current = signature(source)
        if current == last:
            continue
        last = current
        report = sync(source, dest_root, clean=clean)
        if on_sync:
            on_sync(report)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dev_sync.py",
        description="Copy plugin/sam3_gimp into GIMP's plug-in directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="GIMP must be restarted to pick up a newly added plug-in; an "
        "edit to an already-registered one also needs a restart, because GIMP "
        "runs each plug-in as a fresh process from the file on disk.",
    )
    parser.add_argument(
        "--source",
        metavar="DIR",
        default=None,
        help="plug-in source directory (default: plugin/sam3_gimp next to this script)",
    )
    parser.add_argument(
        "--dest",
        metavar="DIR",
        default=None,
        help="GIMP plug-ins directory (default: resolved per platform)",
    )
    parser.add_argument(
        "--gimp-version",
        metavar="VER",
        default=None,
        help="GIMP major.minor whose config dir to target "
             "(default: the newest 3.x config directory found on disk)",
    )
    flat = parser.add_mutually_exclusive_group()
    flat.add_argument(
        "--flatpak",
        dest="flatpak",
        action="store_true",
        default=None,
        help="force the Flatpak plug-in directory (Linux)",
    )
    flat.add_argument(
        "--no-flatpak",
        dest="flatpak",
        action="store_false",
        help="force the native plug-in directory (Linux)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="delete files in the target that the source no longer has",
    )
    parser.add_argument(
        "-n", "--dry-run", action="store_true", help="report changes without making them"
    )
    parser.add_argument(
        "-w", "--watch", action="store_true", help="stay running and re-sync on change"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        metavar="SEC",
        help="polling interval for --watch (default: 1.0)",
    )
    parser.add_argument(
        "--print-dest",
        action="store_true",
        help="print the resolved destination directory and exit",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="only report errors")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="list every file, not just a summary"
    )
    return parser


def _emit(report: SyncReport, quiet: bool, verbose: bool) -> None:
    if quiet:
        return
    if verbose:
        for line in report.lines():
            print("  " + line)
    for line in report.skipped_lines():
        print("note: " + line)
    stamp = time.strftime("%H:%M:%S")
    print("[%s] %s -> %s (%s)" % (stamp, report.source, report.target, report.summary()))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    source = Path(args.source).expanduser() if args.source else default_source_dir()
    if args.dest:
        dest_root = Path(args.dest).expanduser()
    else:
        try:
            dest_root = resolve_plugin_dir(version=args.gimp_version, flatpak=args.flatpak)
        except FileNotFoundError as exc:
            sys.stderr.write("error: %s\n" % (exc,))
            return 2

    if args.print_dest:
        print(dest_root / PLUGIN_NAME)
        return 0

    if not source.is_dir():
        sys.stderr.write("error: source directory not found: %s\n" % (source,))
        return 2

    if not args.dry_run and not dest_root.exists():
        # A config directory without plug-ins/ is normal until the first user
        # plug-in goes in.  Creating it is correct and harmless -- GIMP reads it
        # at startup -- but say so, because a silent success in the wrong place
        # is the worst outcome here.
        if not args.quiet:
            print("note: creating %s" % (dest_root,))
        try:
            dest_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            sys.stderr.write("error: cannot create %s: %s\n" % (dest_root, exc))
            return 2

    try:
        if args.watch:
            if not args.quiet:
                print("watching %s (Ctrl-C to stop)" % (source,))
            try:
                watch(
                    source,
                    dest_root,
                    interval=args.interval,
                    clean=args.clean,
                    on_sync=lambda r: _emit(r, args.quiet, args.verbose),
                )
            except KeyboardInterrupt:
                if not args.quiet:
                    print("\nstopped")
            return 0

        report = sync(source, dest_root, dry_run=args.dry_run, clean=args.clean)
    except FileNotFoundError as exc:
        sys.stderr.write("error: %s\n" % (exc,))
        return 2
    except OSError as exc:
        sys.stderr.write("error: %s\n" % (exc,))
        return 1

    _emit(report, args.quiet, args.verbose)
    if not args.quiet and report.changed and not args.dry_run:
        print("restart GIMP to load the changes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
