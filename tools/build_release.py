#!/usr/bin/env python3
"""Build the distributable plug-in zip.

The zip *is* the install path for people who are not developers: download it,
extract it into GIMP's plug-ins folder, restart GIMP.  No terminal, no git, no
Python on PATH.  Everything after that -- the environment, the weights, pointing
at an existing PyTorch install -- happens inside the Setup dialog.

The archive contains exactly one top-level directory, ``sam3_gimp/``, because
GIMP requires ``<plug-ins>/sam3_gimp/sam3_gimp.py`` and the commonest install
mistake is extracting a flat pile of files (or a doubled
``sam3_gimp/sam3_gimp/``) into the plug-ins folder.

The daemon ships inside the plug-in at ``sam3_gimp/_daemon/`` and is installed
from that path, never by name: the ``sam3d`` and ``sam3gimpd`` names on PyPI
are not this project, so a bare ``pip install`` fetches a stranger's package.  Because it lives inside the plug-in directory in the repository too,
the folder GIMP loads is complete however it got there -- zip, installer, or a
hand copy.

What goes in, and nothing else:

* the files git tracks under ``plugin/sam3_gimp`` -- never whatever else
  happens to be on disk there (a ``.env``, a virtualenv, ``build/``,
  ``*.egg-info``, logs, settings), and never ``__pycache__`` or ``*.pyc``;
* the repository's ``LICENSE``, as ``sam3_gimp/LICENSE``, because the MIT
  licence asks for its notice to travel with every copy;
* ``sam3_gimp/INSTALL.txt``, three steps for someone who opened the zip.

The build refuses to run outside a git checkout, refuses tracked symlinks and
submodules, and refuses uncommitted changes to the files it would ship unless
``--allow-dirty`` is given, so a release is always a commit.  Entries are
sorted, stamped with the commit's time (or ``SOURCE_DATE_EPOCH``), and carry
POSIX permissions -- ``0644``, or ``0755`` for the entry script and anything
git records as executable -- so the same commit gives the same zip, and a zip
built on Windows still extracts with GIMP's executable bit on Linux and macOS.

Usage::

    python tools/build_release.py                 # -> dist/sam3-gimp-<version>.zip
    python tools/build_release.py --out /tmp/x.zip
    python tools/build_release.py --allow-dirty   # a test build of the working tree
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dev_sync  # noqa: E402  (shares the file selection, so the zip cannot drift)

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_NAME = "sam3_gimp"
ENTRY = PLUGIN_NAME + ".py"

#: The earliest timestamp a zip entry can carry (1980-01-01T00:00:00Z).
ZIP_EPOCH = 315532800

FILE_MODE = 0o644
EXEC_MODE = 0o755
GIT_EXEC_MODE = "100755"


class BuildError(SystemExit):
    """A refusal, with a message that says what to do about it."""

    def __init__(self, message: str):
        super().__init__("build_release: " + message)


def plugin_version(source: Path) -> str:
    """Read ``__version__`` out of the entry script without importing gi."""
    text = (source / ENTRY).read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.M)
    return match.group(1) if match else "0.0.0"


def _git_text(cwd: Path, *args) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(cwd)] + list(args),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BuildError("git is needed to build a release and could not be run: %s" % exc)
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise BuildError(
            "%s is not inside a git checkout (%s). A release is built from the "
            "files git tracks; clone the repository and build from the clone."
            % (cwd, detail or "git %s failed" % args[0])
        )
    return proc.stdout.decode("utf-8", "replace")


def uncommitted_changes(source: Path):
    """Tracked files under ``source`` whose working copy or index differs from HEAD."""
    out = _git_text(source, "status", "--porcelain=v1", "--untracked-files=no", "--", ".")
    return [line for line in out.splitlines() if line.strip()]


def source_date_epoch(source: Path) -> int:
    """``SOURCE_DATE_EPOCH`` if set, else the commit time of ``HEAD``."""
    value = os.environ.get("SOURCE_DATE_EPOCH", "").strip()
    if value:
        try:
            return max(ZIP_EPOCH, int(value))
        except ValueError:
            raise BuildError("SOURCE_DATE_EPOCH is not an integer: %r" % value)
    return max(ZIP_EPOCH, int(_git_text(source, "log", "-1", "--format=%ct", "HEAD").strip()))


def release_files(source: Path):
    """``[(relative path, mode)]`` to ship from ``source``, sorted.

    Raises :class:`BuildError` for anything that must not go into a release: a
    tracked symlink (its target could be anything on the builder's disk), a
    submodule, or a tracked file missing from the working tree.
    """
    tracked = dev_sync.git_tracked_files(source)
    if tracked is None:
        _git_text(source, "rev-parse", "--show-toplevel")  # raises with git's reason
        raise BuildError("could not list the files git tracks under %s" % source)
    if Path(ENTRY) not in tracked:
        raise BuildError("%s is not tracked by git under %s" % (ENTRY, source))

    problems, files = [], []
    for rel in sorted(tracked, key=lambda p: p.as_posix()):
        git_mode = tracked[rel]
        if dev_sync.is_litter(rel):
            continue
        path = source / rel
        if git_mode == dev_sync.GIT_SYMLINK_MODE or path.is_symlink():
            problems.append("%s is a symlink" % rel.as_posix())
        elif git_mode == dev_sync.GIT_GITLINK_MODE:
            problems.append("%s is a submodule" % rel.as_posix())
        elif not path.is_file():
            problems.append("%s is tracked but missing from the working tree" % rel.as_posix())
        else:
            executable = git_mode == GIT_EXEC_MODE or rel.as_posix() == ENTRY
            files.append((rel, EXEC_MODE if executable else FILE_MODE))
    if problems:
        raise BuildError("refusing to package:\n  " + "\n  ".join(problems))
    return files


def _zip_info(arcname: str, mode: int, date_time) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(arcname, date_time=date_time)
    info.compress_type = zipfile.ZIP_DEFLATED
    # create_system 3 (Unix) is what makes unzip, bsdtar and GIMP's users'
    # file managers honour the permission bits; zipfile defaults to 0 (MS-DOS)
    # when it runs on Windows.
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | mode) << 16
    return info


def build(out: Path = None, source: Path = None, allow_dirty: bool = False) -> Path:
    source = (REPO_ROOT / "plugin" / PLUGIN_NAME) if source is None else Path(source)
    source = source.resolve()
    if not (source / ENTRY).is_file():
        raise BuildError("not a plug-in source directory: %s" % source)

    files = release_files(source)
    if not any(rel.as_posix() == "_daemon/pyproject.toml" for rel, _ in files):
        raise BuildError(
            "the daemon is missing from %s -- it ships inside the plug-in, at "
            "sam3_gimp/_daemon, and must be committed there" % source
        )

    dirty = uncommitted_changes(source)
    if dirty and not allow_dirty:
        raise BuildError(
            "uncommitted changes to files the release would ship:\n  %s\n"
            "Commit them, or pass --allow-dirty for a test build." % "\n  ".join(dirty)
        )

    shipped = {rel.as_posix() for rel, _ in files}
    license_file = REPO_ROOT / "LICENSE"
    if "LICENSE" not in shipped and not license_file.is_file():
        raise BuildError("%s is missing; the MIT notice must ship with the plug-in" % license_file)

    version = plugin_version(source)
    date_time = time.gmtime(source_date_epoch(source))[:6]
    out = (REPO_ROOT / "dist" / ("sam3-gimp-%s.zip" % version)) if out is None else Path(out)
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    entries = [("%s/%s" % (PLUGIN_NAME, rel.as_posix()), mode, (source / rel).read_bytes())
               for rel, mode in files]
    if "LICENSE" not in shipped:
        entries.append(("%s/LICENSE" % PLUGIN_NAME, FILE_MODE, license_file.read_bytes()))
    entries.append(("%s/INSTALL.txt" % PLUGIN_NAME, FILE_MODE,
                    _install_note(version).encode("utf-8")))
    entries.sort(key=lambda entry: entry[0])

    partial = out.with_name(out.name + ".partial")
    try:
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as zf:
            for arcname, mode, data in entries:
                zf.writestr(_zip_info(arcname, mode, date_time), data)
        os.replace(partial, out)
    finally:
        if partial.exists():
            partial.unlink()
    return out


def _install_note(version: str) -> str:
    return (
        "sam3-gimp %s\n"
        "=================\n\n"
        "1. Extract this archive into GIMP's plug-ins folder so that you end up\n"
        "   with:\n\n"
        "       <plug-ins>/sam3_gimp/sam3_gimp.py\n\n"
        "   On Windows the folder is usually:\n"
        "       %%APPDATA%%\\GIMP\\<version>\\plug-ins\\\n"
        "   Check Edit > Preferences > Folders > Plug-ins for the real path.\n\n"
        "2. Restart GIMP.\n\n"
        "3. Filters > AI Segmentation > SAM 3 Setup / Doctor.\n"
        "   Everything else -- the environment, the model weights, or pointing\n"
        "   at a PyTorch install you already have -- is done in that window.\n\n"
        "No terminal required.\n" % version
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the plug-in release zip.")
    parser.add_argument("--out", type=Path, default=None,
                        help="output .zip path (default: dist/sam3-gimp-<version>.zip)")
    parser.add_argument("--source", type=Path, default=None,
                        help="plug-in source dir inside a git checkout "
                             "(default: plugin/sam3_gimp)")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="package uncommitted changes to tracked files (test builds only)")
    args = parser.parse_args(argv)

    path = build(args.out, args.source, allow_dirty=args.allow_dirty)
    size = os.path.getsize(path)
    with zipfile.ZipFile(path) as zf:
        count = len(zf.namelist())
    print("%s  (%d files, %.1f KiB)" % (path, count, size / 1024.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
