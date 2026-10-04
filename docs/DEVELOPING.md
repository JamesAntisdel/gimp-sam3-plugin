# Developing sam3-gimp

Most of this project can be worked on without a GPU, without GIMP and without the model
weights. The daemon has a stub engine that implements the whole HTTP contract with no
torch, the GTK canvas and both windows run outside GIMP, and the GIMP-facing code is
tested against a hand-written stand-in for libgimp. This document explains how, and what
still needs the real thing.

You need Python 3.10 or newer and `pytest`. The GTK tests, the dialogs and the canvas
harness also need GTK 3, PyGObject, pycairo and a display; without them those tests skip.
Nothing else needs installing.

---

## Quick start

From the repository root:

```console
$ python3 -m pytest                       # the whole suite: no torch, no GIMP, no GPU
$ export SAM3_GIMP_HOME=/tmp/sam3-dev     # keep dev state out of your real data directory
$ PYTHONPATH=plugin/sam3_gimp/_daemon python3 -m sam3gimpd serve --stub &     # a conforming daemon
$ tools/canvas_harness.py --selftest      # renders the real canvas offscreen
```

---

## Why it is testable at all

Three rules, and they are load-bearing rather than stylistic:

1. **`torch` and `transformers` are imported lazily, inside functions, never at module
   scope.** So `import sam3gimpd.server`, `sam3gimpd.engines.pcs`, `sam3gimpd.modelmgr` —
   all of them — succeed on a machine that has never heard of torch. A base install of the
   daemon (`pip install plugin/sam3_gimp/_daemon`) has **zero** runtime dependencies.
   torch is installed separately, from the PyTorch wheel index, and `transformers` and the
   other runtime libraries come with the daemon's `[runtime]` extra.
2. **The plug-in imports only the standard library and `gi`.** No numpy, no requests, no
   pillow, ever — it runs inside GIMP's embedded Python where those do not exist. Every
   module under `plugin/sam3_gimp/` (except `_daemon/`, which is a separate package for a
   separate interpreter) is written so that its pure logic imports and tests with no GIMP
   present: `gi.repository.Gimp` is bound lazily behind a rebindable hook.
3. **The stub engine is production code.** See below.

---

## The stub engine

`sam3gimpd serve --stub` runs a **fully conforming implementation of the entire HTTP
contract** that imports no torch, no transformers and no numpy on any code path. It is
specified in [`plugin/sam3_gimp/_daemon/API.md`](../plugin/sam3_gimp/_daemon/API.md) §14
and it is a feature, not a fixture:

* **Deterministic.** The same `(image_id, prompt)` produces byte-identical frames, so tests
  can assert on bytes.
* **Structurally honest.** Real `SAM3RES` frames, real cropped **soft** uint8 masks with
  anti-aliased falloff (a constant-255 rectangle would hide client threshold bugs), real
  model-canvas bboxes, real scores in `[0.1, 1.0]` sorted descending, 1–8 PCS instances
  chosen from a hash of the prompt text.
* **Realistically staged.** Progress is reported through the same stages as the real engine
  and the total latency is a few hundred milliseconds, so progress bars, long-polling and
  supersession all get exercised.

If your client works against the stub, the only thing left to be wrong is the model.

---

## Running the daemon by hand

```console
$ PYTHONPATH=plugin/sam3_gimp/_daemon python3 -m sam3gimpd serve --stub
```

or, after `pip install -e plugin/sam3_gimp/_daemon`, simply `sam3gimpd serve --stub`.
Install it from that path, never by name: the `sam3d` and `sam3gimpd` names on PyPI are not
this project.

Useful flags (`sam3gimpd serve --help` lists them all):

| Flag | Default | |
|---|---|---|
| `--stub` | off | the fake engine; never imports torch |
| `--host` | `127.0.0.1` | the bind address; see [A daemon on another machine](INSTALL.md#a-daemon-on-another-machine) before changing it |
| `--port` | `0` | 0 = ask the OS; the real port lands in `runtime.json` |
| `--parent-pid PID` | none | exit when that pid disappears |
| `--idle-ttl` | `1800` | seconds of idleness before exiting; `0` disables |
| `--cache-size` | `3` | LRU capacity of the embedding cache |
| `--runtime-file PATH` | per platform | override the handshake file |
| `--device` | `auto` | `auto` = cuda → mps → cpu |
| `--log-level` | `info` | `debug` is verbose and useful |

`sam3gimpd doctor` prints a JSON report: device, dtype, versions and paths, and, if a daemon
is running, its whole `/status` payload (`--no-daemon` skips contacting it).
`HF_TOKEN=hf_… sam3gimpd download` fetches the gated checkpoint; the token also works as
`--token`, but a command-line argument is visible in process listings.

### Keeping dev state out of your real data directory

Everything the daemon and the plug-in own lives under one base directory, and
**`SAM3_GIMP_HOME` relocates all of it**:

```console
$ export SAM3_GIMP_HOME=/tmp/sam3-dev
```

The test suite does the same, and so should you, so that a stray dev daemon never fights
with a real install. `SAM3D_RUNTIME_FILE` overrides just the path of `runtime.json`.

### A full round trip with `curl`

The outputs below are from a stub daemon; ports, ids and timings differ from run to run.

```console
$ export SAM3_GIMP_HOME=/tmp/sam3-dev
$ RT=$SAM3_GIMP_HOME/runtime.json
$ PYTHONPATH=plugin/sam3_gimp/_daemon python3 -m sam3gimpd serve --stub &
$ until [ -f $RT ]; do sleep 0.1; done
$ PORT=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['port'])" $RT)
$ TOKEN=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['token'])" $RT)
$ AUTH="Authorization: Bearer $TOKEN"
$ URL=http://127.0.0.1:$PORT

$ curl -s -H "$AUTH" $URL/hello
{"api_version":"1.0","sam3d_version":"0.1.1","engine_mode":"stub","device":"stub",
 "dtype":"float32","capabilities":["pcs","pvs"],"torch_available":false, ...}

# Upload raw RGB: exactly W*H*3 bytes, no header, no container (API.md §7).
$ python3 -c "open('/tmp/px.bin','wb').write(bytes(64*64*3))"
$ curl -s -H "$AUTH" -H 'Content-Type: application/octet-stream' \
       -H 'X-Width: 64' -H 'X-Height: 64' \
       --data-binary @/tmp/px.bin $URL/images
{"image_id":"c36a9d4d09454f5107ef891ffd76a426","job_id":"j-000001-37a0","cached":false,
 "image":{"width":64,"height":64},"model_canvas":{"width":64,"height":64},
 "canvas_from_image":{"scale_x":1.0,"scale_y":1.0,"offset_x":0.0,"offset_y":0.0},
 "state":"queued"}

$ ID=c36a9d4d09454f5107ef891ffd76a426     # the image_id above
$ JOB=$(curl -s -H "$AUTH" -H 'Content-Type: application/json' \
       -d '{"request_id":"r-1","text":"red car"}' $URL/images/$ID/text \
       | python3 -c "import json,sys;print(json.load(sys.stdin)['job_id'])")

# ?wait= returns at every progress step (API.md §11), so poll until the job ends.
# ?meta=1 returns the JSON result header instead of the binary frame; handy in a shell.
$ until curl -s -H "$AUTH" "$URL/jobs/$JOB?wait=10&meta=1" | grep -qE '"state":"(done|failed)"'; do :; done
$ curl -s -H "$AUTH" "$URL/jobs/$JOB?meta=1"
{"job_id":"j-000002-a694","state":"done","engine":"pcs",...,"masks_available":true,
 "result":{...,"instances":[{"instance_id":0,"score":0.9277,"label":"red car",
 "bbox":[39,21,49,29],"mask_width":10,"mask_height":8,"blob_offset":0,"blob_length":80},
 ...],"blob_length":968}}

# Without ?meta=1 a finished job is the binary SAM3RES frame: 12-byte magic and header
# length, a JSON header, then the mask blobs, tightly packed in instance order (API.md §8).
$ curl -s -H "$AUTH" "$URL/jobs/$JOB" -o /tmp/frame.bin
$ curl -s -H "$AUTH" $URL/status | head -c 200; echo
$ curl -s -X POST -H "$AUTH" -H 'Content-Type: application/json' -d '{}' $URL/shutdown
{"ok":true,"pid":981555,"grace_ms":500.0}
```

Mind the coordinate spaces — mixing them up is the most likely integration bug, which is
why `API.md` §5 names all three. **Everything you send** (points, boxes) is in
**uploaded-image** space. **Everything you receive** (bboxes, and therefore mask crops) is in
**model-canvas** space. The daemon reports the transform between them as
`canvas_from_image`; it is the identity today, and clients must still use the reported
values and **never hardcode it**.

---

## The GTK canvas harness

`plugin/sam3_gimp/ui/canvas.py` is a plain `Gtk.DrawingArea` that knows nothing about GIMP
or HTTP, so it can be developed anywhere GTK 3 runs:

```console
$ tools/canvas_harness.py                       # synthetic image, best available back end
$ tools/canvas_harness.py photo.png             # a real image
$ tools/canvas_harness.py photo.png --spawn     # start `sam3gimpd serve --stub` and use it
$ tools/canvas_harness.py photo.png --offline   # never touch a daemon
$ tools/canvas_harness.py --selftest            # headless: render once, assert, exit
$ tools/canvas_harness.py --selftest --screenshot /tmp/canvas.png
```

Three back ends, tried in this order unless one is forced: an already-running daemon found
through `runtime.json`; a stub daemon the harness starts itself (from the checkout, with
nothing installed); or **offline**, where it synthesises frames in-process using the same
`SAM3RES` layout, the same soft-mask semantics and the same coordinate spaces as the daemon.
Offline mode needs no daemon at all and still exercises decode, affine inversion,
thresholding, hit-testing and marching ants for real. `--selftest` always uses it.

The harness imports only the standard library, `gi` and `cairo`, so it doubles as a check
that the canvas has not grown a forbidden dependency.

### The windows, standalone

Both GTK windows run outside GIMP on a machine with a display:

```console
$ SAM3_GIMP_HOME=/tmp/sam3-dev PYTHONPATH=plugin/sam3_gimp python3 plugin/sam3_gimp/ui/setup_dialog.py
$ PYTHONPATH=$PWD/plugin/sam3_gimp:$PWD/plugin/sam3_gimp/_daemon \
  SAM3_GIMP_HOME=/tmp/sam3-dev \
  SAM3D_COMMAND="python3 -m sam3gimpd serve --stub" \
  python3 plugin/sam3_gimp/ui/main_dialog.py
```

`SAM3D_COMMAND` is the launcher's escape hatch: it replaces the argv prefix used to start
the daemon, so you can point at a checkout instead of an installed environment. The
documented flags are appended to whatever you give it. The daemon inherits `PYTHONPATH`
but not the working directory, which is why the paths above are absolute.

That second command exercises the whole lifecycle without GIMP: the window calls
`launcher.find_or_spawn()`, which starts a **detached** daemon with
`--parent-pid <this process>`, waits for `runtime.json`, handshakes on `/hello`, uploads
pixels, prompts, long-polls, decodes the frame and renders it — and when you close the
window, the daemon notices its parent is gone and exits on its own. Only the `Gimp.*` calls
behind *Apply* are out of reach.

---

## The fake-GIMP stubs

`tests/fake_gimp/` is a hand-written stand-in for `gi.repository.Gimp` and `Gegl`: images,
layers, channels, selections, buffers with real byte storage, `ChannelOps`, undo groups.
`gimpbridge.py` and `outputs.py` reach GIMP only through a rebindable module reference
(`set_gimp_modules()` / `reset_gimp_modules()`), so the tests swap the stubs in and drive
the real code.

What that buys: the geometry, the placement maths, the threshold ramp, hole filling,
min-area rejection, instance naming, path construction and the undo grouping are all tested
for real. What it cannot buy: proof that libgimp behaves the same way. The stubs model GIMP
as far as its source has been read — channel transforms clip to the channel, `Drawable.fill`
on a channel uses the context colour — and a behaviour they do not model passes the tests
and fails in GIMP. Treat green tests here as "the logic is right", not "GIMP will accept
this".

---

## The test suite

```console
$ python3 -m pytest                        # everything
$ python3 -m pytest tests/daemon           # the sam3gimpd half
$ python3 -m pytest tests/plugin           # the GIMP half
$ python3 -m pytest -k contours -v
$ python3 -m pytest -m "not slow"
```

`pytest.ini` puts `plugin/sam3_gimp/_daemon`, `plugin`, `plugin/sam3_gimp` and `tests` on
`sys.path`, which is why the plug-in's modules import as **top-level** names
(`import client`, `from ui import canvas`) — exactly as they do inside GIMP once
`sam3_gimp.py` adds its own directory to `sys.path`.

Markers:

| Marker | Means |
|---|---|
| `needs_torch` | torch (and usually transformers) importable; skips otherwise |
| `needs_gpu` | a real CUDA/MPS device; skips otherwise |
| `needs_weights` | the gated `facebook/sam3` checkpoint on disk; skips otherwise |
| `needs_gimp` | running inside GIMP; skips otherwise |
| `needs_gtk` | GTK 3 + PyGObject + a display; skips otherwise |
| `needs_daemon` | spawns or talks to a real `sam3gimpd` over loopback |
| `slow` | more than a couple of seconds |

On a machine with no torch, GPU, weights or GIMP, the suite passes with only the tests
that need those skipped. If a change makes something require torch at import time, the
suite tells you immediately. CI runs it on every supported Python, once more with GTK
under Xvfb, and checks that a base install of the daemon pulls in nothing
([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)).

---

## Iterating against a real GIMP

GIMP will not load a plug-in from a checkout: it wants `<plug-ins>/<name>/<name>.py`, with
the directory and the entry file sharing a name and (on POSIX) the entry file executable.
`tools/dev_sync.py` puts it there:

```console
$ python3 tools/dev_sync.py --print-dest     # where would this go?
/home/you/.config/GIMP/3.2/plug-ins/sam3_gimp
$ python3 tools/dev_sync.py --dry-run        # say what would change
$ python3 tools/dev_sync.py                  # sync once
$ python3 tools/dev_sync.py --watch          # sync now, then on every change
$ python3 tools/dev_sync.py --clean          # also delete files the source no longer has
$ python3 tools/dev_sync.py --gimp-version 3.0
$ python3 tools/dev_sync.py --dest 'D:\path\to\plug-ins'
```

Destination, in precedence order: `--dest`, then `$GIMP3_PLUGIN_DIR` / `$GIMP_PLUGIN_DIR`,
then the platform's GIMP config directory (`%APPDATA%\GIMP\<ver>\plug-ins`,
`~/Library/Application Support/GIMP/<ver>/plug-ins`, or
`$XDG_CONFIG_HOME/GIMP/<ver>/plug-ins`), where `<ver>` is `--gimp-version` or the newest
`3.x` directory that exists. If there is none — GIMP has never been started — it stops and
asks for `--gimp-version` or `--dest` rather than guessing. On Linux it switches to the
Flatpak config directory automatically when only that one exists; `--flatpak` /
`--no-flatpak` force the decision.

It copies the files git tracks under `plugin/sam3_gimp` (their working-tree content, so
uncommitted edits are included), skips caches and symlinks, and lists any untracked file it
left behind: `git add` a new module to have it synced.

**GIMP must be restarted after every sync.** It runs each plug-in as a fresh process read
from disk, so there is no reload; `--watch` keeps the copy current so the only manual step
is the restart.

On Windows, `tools/install.ps1` does the same copy from a console and can also build the
daemon environment; see [INSTALL.md](INSTALL.md#option-c--the-installer-script).

### Building the release zip

```console
$ python3 tools/build_release.py              # dist/sam3-gimp-<version>.zip
$ python3 tools/build_release.py --allow-dirty --out /tmp/test.zip
```

The zip holds one top-level `sam3_gimp/` folder: the files git tracks under
`plugin/sam3_gimp`, the repository's `LICENSE`, and a short `INSTALL.txt`. The build
refuses to run outside a git checkout, refuses tracked symlinks, and refuses uncommitted
changes to the files it would ship unless `--allow-dirty` is given. Entries are sorted,
stamped with the commit time (or `SOURCE_DATE_EPOCH`) and carry POSIX permissions, so the
same commit gives the same zip on any platform and the entry script stays executable.

---

## The seam: `plugin/sam3_gimp/_daemon/API.md`

The two halves of this project agree on **nothing except** the contract in
[`plugin/sam3_gimp/_daemon/API.md`](../plugin/sam3_gimp/_daemon/API.md). That keeps
either half replaceable, and it is why each can be tested without the other.

Read §16 ("Non-negotiables") before changing anything on the wire. In brief:

* Coordinates: client → daemon is **uploaded-image**; daemon → client is **model-canvas**.
  Never mix, never hardcode the transform.
* Masks are **soft uint8, cropped to the bbox, at model resolution**; `128` is logit 0.
  There is no server-side threshold parameter — the client re-thresholds locally, which is
  the whole reason the slider is instant.
* `mask_width == bbox width`, `mask_height == bbox height`,
  `blob_length == mask_width * mask_height`. Always.
* One inference at a time; queued-not-started jobs for the same image are superseded;
  clients drop results whose `request_id` is stale.
* The HTTP server is multi-threaded, every response carries `Content-Length`, nothing is
  chunked.
* Every endpoint needs the bearer token; every non-2xx body is `{error: {code, message,
  detail}}`; a *failed job* is `200` with `state: "failed"`, not an HTTP error.
* `--stub` never imports torch; a base install of the daemon has zero runtime
  dependencies.
* The plug-in never imports `sam3gimpd.types` — it mirrors it by hand, and
  `tests/test_contract_mirror.py` checks that the two agree.

`plugin/sam3_gimp/_daemon/sam3gimpd/types.py` is the reference implementation of every JSON
shape and the source of truth for the limits.

The daemon reports a hash of its own source as `build` in `GET /hello`, and the plug-in
hashes its bundled `_daemon/sam3gimpd` the same way (`bootstrap.bundled_daemon_build`).
When they differ the window says **DAEMON OUT OF DATE**, so a dev-loop edit to the daemon
is noticed without a version bump. Bump `__version__` for releases all the same.

---

## Repository layout

```
gimp-sam3-plugin/
├── README.md
├── DESIGN.md                     architecture and rationale
├── LICENSE                       MIT
├── pytest.ini                    sys.path wiring + markers
├── plugin/sam3_gimp/             the unit of distribution; stdlib + gi only, except _daemon/
│   ├── sam3_gimp.py              entry point (name must match the folder)
│   ├── __init__.py               declared surface: procedure names, output-mode vocabulary
│   ├── client.py                 typed stdlib client for every endpoint + frame decoder
│   ├── launcher.py               find-or-spawn, platform layout, detached spawn
│   ├── bootstrap.py              uv/venv installer, HF token flow, doctor, journal
│   ├── gimpbridge.py             projection → R'G'B' u8, upload geometry, placement
│   ├── outputs.py                masks → selection / channels / layer masks / groups / paths
│   ├── contours.py               marching squares → Douglas–Peucker → Bézier (pure stdlib)
│   ├── ui/
│   │   ├── canvas.py             the GTK3 preview widget
│   │   ├── main_dialog.py        the segmentation window
│   │   └── setup_dialog.py       Install / SAM 3 weights / Doctor
│   └── _daemon/                  `sam3gimpd`, installed from here into its own environment
│       ├── API.md                the contract
│       ├── README.md
│       ├── pyproject.toml        zero base deps; transformers pinned in the [runtime] extra
│       └── sam3gimpd/
│           ├── types.py          every JSON shape + the frame codec (source of truth)
│           ├── paths.py          base dir, runtime.json, logs, hf home
│           ├── server.py         HTTP surface, auth, runtime.json, self-supervision
│           ├── jobs.py           job manager, one worker thread, supersession, progress
│           ├── session.py        LRU embedding cache keyed by pixel hash
│           ├── masks.py          soft-mask quantisation, tight bbox, frame encode/decode
│           ├── modelmgr.py       device/dtype policy, lazy load, idle unload, HF wiring
│           ├── cli.py            serve / doctor / download
│           ├── __main__.py       python -m sam3gimpd
│           └── engines/
│               ├── base.py       the engine ABC and shared data types
│               ├── stub.py       the fake engine; no heavy imports, ever
│               ├── pcs.py        Sam3Model + Sam3Processor (text → N instances)
│               └── pvs.py        Sam3TrackerModel + Sam3TrackerProcessor (points → 1)
├── tests/
│   ├── daemon/                   headless, CPU, synthetic images, stub engine
│   ├── plugin/                   client, launcher, bridge, outputs, canvas, bootstrap, entry
│   ├── fake_gimp/                the gi.repository.Gimp / Gegl stand-in
│   ├── test_contract_mirror.py   the plug-in's mirror of types.py agrees with the original
│   └── test_end_to_end.py        a detached stub daemon driven through the plug-in's client
├── tools/
│   ├── build_release.py          the release zip, from tracked files
│   ├── canvas_harness.py         standalone GTK app for the canvas
│   ├── dev_sync.py               checkout → GIMP's plug-ins directory
│   ├── diagnose.py               paste into GIMP's Python console to debug an install
│   └── install.ps1               Windows installer
└── docs/                         INSTALL.md, DEVELOPING.md
```

Rules of thumb for a change:

* Touching the wire? Change `plugin/sam3_gimp/_daemon/API.md` **first**, then `types.py`,
  then both sides.
* Adding a plug-in dependency? You cannot. Standard library and `gi`, full stop.
* Adding a daemon import of torch? It must be inside a function, and `--stub` must not
  reach it.
* Adding a GIMP call? Model it in `tests/fake_gimp/` as well, from what libgimp actually
  does (its source, not only its documentation). A fake that is kinder than GIMP hides
  bugs.

---

## Environment variables

| Variable | Read by | Meaning |
|---|---|---|
| `SAM3_GIMP_HOME` | both halves | Relocate the whole base directory. Use this for dev. |
| `SAM3D_RUNTIME_FILE` | both halves | Override only the path of `runtime.json`. |
| `SAM3D_COMMAND` | launcher | Replace the argv prefix used to start the daemon. Wins over the interpreter chosen in Setup, which is persisted in `<base>/settings.json` and read by `launcher.configured_python()`. |
| `SAM3_WEIGHTS_DIR` | daemon | A user-supplied checkpoint directory (the non-gated exit). |
| `HF_HOME` | daemon | Pinned to `<base>/hf` by the launcher so Doctor can show and clear it. |
| `SAM3D_DEVICE`, `SAM3D_DTYPE` | daemon | Force the device or precision at run time. |
| `SAM3_FORCE_DEVICE` | Setup | Override accelerator detection for the install. |
| `GIMP3_PLUGIN_DIR`, `GIMP_PLUGIN_DIR` | `dev_sync.py` | Override the sync destination. |

---

## What needs real GIMP, a GPU or Windows

CI and a plain checkout cover the logic. These need the real thing:

* **Every `Gimp.*` and `Gegl.*` call.** The GIMP-facing code is tested against
  `tests/fake_gimp/`; its real behaviour shows only inside GIMP, through
  `tools/dev_sync.py`. GIMP 3 from Flathub, driven headlessly with `gimp -i -b`, can
  exercise `gimpbridge` and `outputs` against real `Gegl.Buffer` and PDB calls on Linux.
* **Every CUDA, ROCm and MPS path**, every real `Sam3Model` / `Sam3TrackerModel` forward
  pass and every weight load. Those tests carry `needs_torch`, `needs_gpu` or
  `needs_weights`. Whether both engines fit in 8 GB of VRAM together is still unmeasured
  (`DESIGN.md` §7).
* **The Windows detached spawn** (`pythonw.exe`, `DETACHED_PROCESS |
  CREATE_NEW_PROCESS_GROUP`) and the Microsoft Store build's MSIX container. The POSIX
  equivalent (`start_new_session=True`) runs in every test pass.

The tested configuration is Microsoft Store GIMP 3.2 on Windows with an NVIDIA RTX 2080 Ti
and an existing PyTorch environment. Not yet run on real hardware: `tools/install.ps1`, the
Setup window's own *Install* button end to end, and inference on ROCm, MPS or CPU. Both
install routes are unit-tested by the commands they build.

### GIMP behaviour worth knowing

* **`Gimp.Channel` transforms clip.** `gimp_channel_get_clip()` returns
  `GIMP_TRANSFORM_RESIZE_CLIP` unconditionally and `gimp_channel_scale()` pins the offset at
  (0, 0). Never `transform_scale` a channel into place; scale in a scratch image
  (`outputs._scale_gray`) and write into an image-sized channel.
* **Never `Drawable.fill` a channel.** `FillType.TRANSPARENT` fills with the context's
  background colour and drops alpha only where there is alpha to drop; a channel has none,
  so it comes out white. New GEGL buffers start zeroed.
* **Marching ants need a selected drawable.** GIMP draws the selection boundary only while
  a layer or channel is selected, so an Apply that inserts and removes a scratch channel
  restores the previously selected items afterwards.
