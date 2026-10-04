"""Paste-into-GIMP diagnostic for "I pressed it and nothing happened".

Open **Filters > Development > Python-Fu > Console** in GIMP and paste the
contents of this file. It prints, in order: where the plug-in is, whether every
sibling module imports, where the logs live and what they say, whether an
environment is configured, and whether the daemon can actually be reached.

It never raises -- each check reports its own failure and moves on, because the
whole point is to run on a broken install.
"""

import os
import sys
import traceback

OUT = []


def say(label, value=""):
    line = "%-22s %s" % (label, value)
    OUT.append(line)
    print(line)


def rule(title):
    OUT.append("")
    OUT.append("--- %s " % title + "-" * max(0, 52 - len(title)))
    print(OUT[-1])


rule("plug-in")
here = None
for p in sys.path:
    cand = os.path.join(p, "sam3_gimp.py")
    if os.path.isfile(cand):
        here = p
        break
say("plug-in dir", here or "NOT ON sys.path (this is the problem)")
say("python", sys.version.split()[0])

if here and here not in sys.path:
    sys.path.insert(0, here)

rule("sibling modules")
mods = {}
for name in ("launcher", "client", "bootstrap", "gimpbridge",
             "outputs", "contours", "ui.canvas", "ui.main_dialog",
             "ui.setup_dialog"):
    try:
        mods[name] = __import__(name)
        say(name, "ok")
    except Exception as exc:
        say(name, "FAILED: %s: %s" % (type(exc).__name__, exc))

launcher = mods.get("launcher")

rule("paths")
if launcher is None:
    say("(launcher missing)", "cannot resolve paths")
else:
    for label, fn in (("base dir", "base_dir"), ("plugin log", None),
                      ("daemon log", "server_log"), ("crash log", "crash_log"),
                      ("runtime.json", "runtime_file"),
                      ("settings.json", "settings_file")):
        try:
            if fn is None:
                path = os.path.join(launcher.base_dir(), "logs", "plugin.log")
            else:
                path = getattr(launcher, fn)()
            mark = "" if os.path.exists(path) else "   (does not exist)"
            say(label, path + mark)
        except Exception as exc:
            say(label, "FAILED: %s" % exc)

rule("configuration")
if launcher is not None:
    try:
        say("chosen interpreter", launcher.configured_python() or "(none - using managed venv)")
        say("SAM3D_COMMAND", os.environ.get("SAM3D_COMMAND") or "(unset)")
        say("venv python", launcher.venv_python())
        say("  exists", str(os.path.isfile(launcher.venv_python())))
        say("spawn command", " ".join(launcher.build_command(stub=False)))
    except Exception:
        say("configuration", "FAILED")
        print(traceback.format_exc())

rule("daemon")
if launcher is not None:
    try:
        info = launcher.read_runtime_info()
        if not info:
            say("runtime.json", "absent - no daemon has published itself")
        else:
            say("port / pid", "%s / %s" % (info.get("port"), info.get("pid")))
            say("pid alive", str(launcher.pid_alive(info.get("pid"))))
    except Exception as exc:
        say("runtime.json", "FAILED: %s" % exc)
    try:
        res = launcher.find_or_spawn(timeout=30.0)
        say("find_or_spawn", "OK (spawned=%s)" % res.spawned)
        say("hello", str(res.hello))
    except Exception as exc:
        say("find_or_spawn", "FAILED: %s: %s" % (type(exc).__name__, exc))
        print(traceback.format_exc())

rule("recent log tails")
if launcher is not None:
    for label, path in (
        ("plugin.log", os.path.join(launcher.base_dir(), "logs", "plugin.log")),
        ("sam3gimpd.log", launcher.server_log()),
        ("crash.log", launcher.crash_log()),
    ):
        print("\n===== %s =====" % label)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                tail = fh.read()[-3000:]
            print(tail or "(empty)")
        except Exception as exc:
            print("(unreadable: %s)" % exc)

print("\n" + "=" * 60)
print("Copy everything above into the issue / chat.")
