# gimp-sam3-plugin — Design

A GIMP 3 plug-in for AI segmentation powered by **SAM 3**. It follows
[gimpsegany](https://github.com/Shriinivas/gimpsegany), which brought SAM 1 and SAM 2 to
GIMP, and is built around the capabilities SAM 3 adds.

**Status:** implemented. This document records the architecture and the reasoning behind
it. Where it and [`API.md`](plugin/sam3_gimp/_daemon/API.md) differ in detail, `API.md` wins.
**Primary target:** Windows with an NVIDIA GPU and GIMP 3 from the Microsoft Store; this is
the tested configuration.
**Also supported, not yet tested end to end:** CPU-only, Apple Silicon (MPS), AMD ROCm on
Linux, GIMP on Linux, and a daemon on another machine.

---

## 1. What SAM 3 changes

SAM 3 (Meta, Nov 2025) ships **two models under one checkpoint** (`facebook/sam3`, 0.9B params).
Both are natively supported in Hugging Face `transformers`, so we do not vendor
`facebookresearch/sam3` (which demands Python 3.12+, torch 2.7+, CUDA 12.6+).

| Engine | Classes | Prompt | Returns |
|---|---|---|---|
| **PCS** — Promptable *Concept* Segmentation | `Sam3Model` + `Sam3Processor` | text (`"yellow school bus"`), image exemplars, +/- boxes | **every** matching instance at once |
| **PVS** — Promptable *Visual* Segmentation | `Sam3TrackerModel` + `Sam3TrackerProcessor` | points, boxes, masks | one instance per prompt; SAM2 drop-in |

The headline: **PCS finds every instance of a concept from one text prompt**, which SAM 1 and
SAM 2 could not do. "Select every car" is one prompt returning N masks — not N clicks.

PVS preserves the click-to-segment workflow users already expect, so nothing is lost.

### The performance fact that dictates the architecture

Both engines expose embedding reuse:

- `Sam3Model.get_vision_features()` — encode the image once, then run many text prompts against it.
- `Sam3TrackerModel(..., image_embeddings=prev.image_embeddings)` — refine points without re-encoding.

Image encoding is the expensive step (0.9B params at 1008px native resolution). Everything after
it is cheap. This single fact is why the architecture below looks the way it does.

### Two facts about SAM's masks that shape the data path

1. **The model resizes every input to 1008px.** Sending a 6000px image buys nothing; the daemon
   would downscale it anyway.
2. **Masks are low-resolution logits upsampled.** Mask edge quality comes from upsampling the
   *soft* mask well, not from the input resolution.

Consequences in §4: we send ≤1008px, we return soft masks, and GIMP itself does the upsampling.

---

## 2. The two hard constraints

**Constraint A — GIMP 3 embeds its own Python and will not use the system Python.**
Third-party modules must be bundled into the plug-in directory. `torch` + CUDA is multiple GB and
platform-specific; vendoring it into `%APPDATA%\GIMP\<version>\plug-ins\` is not viable.
→ *Inference must live in a separate process with its own environment.* Not a preference; a constraint.

**Constraint B — model load is slow and prompts are fast.**
Loading a 0.9B checkpoint costs seconds. Encoding an image costs seconds. A text prompt against a
cached embedding costs milliseconds.

A design that starts a fresh process for each call pays model load and encode every time. That
is a reasonable trade for one-shot "segment this image", which is how gimpsegany works; for
the interactive loop SAM 3 makes possible, the cost is too high.

→ **A persistent daemon that caches embeddings is the core design decision of this project.**

**Constraint C — GIMP plug-in processes are themselves short-lived.**
GIMP spawns a fresh plug-in process per invocation; it exits when `run()` returns. There is no
long-lived plug-in process to supervise a daemon. So the daemon must be **self-supervising and
detached**, and "restart on crash" really means "respawn on next use." See §4.

---

## 3. Architecture

**The daemon is a product; the GIMP plug-in is one client of it.**

`sam3gimpd` is a standalone package with its own CLI (`sam3gimpd serve`, `sam3gimpd doctor`,
`sam3gimpd download`). It knows nothing about GIMP. It ships inside the plug-in folder at
`_daemon/`, and the plug-in's bootstrap installs it from there into a separate virtualenv and
talks to it over HTTP. The daemon is therefore testable and usable without GIMP, a daemon on
another machine is the same program reached over the network (§4), and other hosts (Krita, a
CLI, a web page) can reuse it.

```
┌────────────────────────────────────────────────────────┐
│ GIMP 3  —  embedded Python, ZERO third-party imports   │
│                                                        │
│  sam3_gimp.py     entry / procedure registration       │
│  ui/canvas.py     GTK3 preview + live mask overlays    │
│  client.py        stdlib http.client, worker thread    │
│  gimpbridge.py    projection -> RGB bytes (via babl)   │
│  outputs.py       masks -> selection/channel/layer/path│
│  contours.py      mask -> polygons/Beziers for paths   │
│  launcher.py      find-or-spawn daemon, runtime.json   │
│  bootstrap.py     env doctor, installer, HF token flow │
└───────────────────────┬────────────────────────────────┘
                        │  HTTP/1.1 on 127.0.0.1:<ephemeral>
                        │  bearer token; binary bodies for pixels/masks
┌───────────────────────┴────────────────────────────────┐
│ sam3gimpd  —  standalone package, own uv-managed venv  │
│                                                        │
│  server.py     HTTP; one inference worker thread       │
│  jobs.py       request ids, supersede, progress        │
│  session.py    LRU embedding cache, keyed by pixel hash│
│  engines/pcs.py   Sam3Model        (text -> instances) │
│  engines/pvs.py   Sam3TrackerModel (points -> instance)│
│  masks.py      soft-mask codec (the wire format)       │
│  modelmgr.py   lazy load, device/dtype, idle unload    │
│  cli.py        serve / doctor / download               │
└────────────────────────────────────────────────────────┘
```

### Why Python on both sides (and not Go or TypeScript)

- **The GIMP side has no choice.** Plug-ins bind through GObject Introspection.
- **The daemon side has no choice.** PyTorch and `transformers` are Python; a Go/TS daemon would
  still shell out to Python for inference.
- **A Go supervisor** would add a toolchain to do what ~200 lines of stdlib Python does. Rejected.
- **TypeScript** only earns its place behind a web UI; GTK is already in-process. Rejected.

The real isolation win is **two hermetically separate Python environments**. The plug-in imports
nothing but stdlib + `gi`. The daemon owns torch. `uv` (a single static binary) builds the venv.
The zero-dependency rule does not cover `_daemon/`: it is a separate package for a separate
interpreter and may import torch.

### Alternative considered: a persistent GIMP "extension" plug-in

GIMP supports long-running extension-type plug-ins (`Gimp.PDBProcType.EXTENSION`) that register
temporary procedures. One could own the daemon as a child and offer a non-modal dialog. Rejected:
Python extension plug-ins in GIMP 3 are thinly documented and add a second lifecycle to get
right, and the detached daemon achieves the same persistence with less surface. It remains the
fallback if a future GIMP build stops a plug-in from starting a detached process (§9).

---

## 4. Data path and IPC

### Transport: HTTP, not custom framing

An early draft specified a custom length-prefixed binary protocol, justified by 36 MB
full-resolution transfers. Two observations removed that justification:

- We send ≤1008px images (§1), so a transfer is ~3 MB, not 36 MB.
- With the daemon as a standalone product, interoperability and debuggability matter more than
  the last few milliseconds.

So: **HTTP/1.1 with binary bodies** (`application/octet-stream` for pixels and masks, JSON for
everything else). The plug-in uses stdlib `http.client`; the daemon can use any server library.
`curl` works against it. A browser can hit `/status`.

The daemon writes `%LOCALAPPDATA%\sam3-gimp\runtime.json` — `{port, token, pid, version,
started_at}` — and the plug-in sends the token as a bearer header. `%LOCALAPPDATA%` is
user-scoped; on Linux and macOS the data directory is created private (mode 0700, files 0600).

### Endpoints

| Endpoint | Body | Notes |
|---|---|---|
| `GET /hello` | — | version handshake, device, dtype, VRAM, capabilities, and a proof that this is the daemon `runtime.json` describes |
| `POST /images` | raw RGB, `X-Width/X-Height` headers | returns `image_id`; encodes + caches; returns immediately with a job id |
| `POST /images/{id}/text` | `{text, request_id}` | PCS → every candidate above a low score floor (0.02 by default) |
| `POST /images/{id}/points` | `{points, labels, box?, request_id}` | PVS → instance + multimask candidates |
| `GET /jobs/{id}` | — | progress / result; long-poll |
| `DELETE /images/{id}` | — | free cache entry |
| `GET /status` · `POST /shutdown` | — | doctor panel, lifecycle |

### Mask transfer: cropped soft masks, upsampled by GIMP

Masks return as **uint8 soft masks (0–255), cropped to their bounding box, at model resolution**.
Not RLE, not binary, not full-res. Reasons:

- **Threshold becomes client-side and instant.** Dragging the mask-threshold slider re-thresholds
  locally; no round trip. Same for the score slider, since the daemon returns every candidate
  above a low floor.
- **Edge quality is better.** Upsampling a soft mask then thresholding gives smooth edges;
  upsampling a binary mask gives stair-steps.
- **It's small.** A cropped 1008px-scale soft mask is tens of KB; twenty of them are ~1 MB.

The plug-in has no numpy and must not do pure-Python resampling. Instead it loads the soft mask
into a throwaway grayscale GIMP image at model resolution, lets **GIMP's own scaler** (GEGL, in
C, with proper interpolation) bring it to image resolution, and writes the result into an
image-sized channel. Scaling a channel in place does not work: GIMP clips a channel's transforms
to the channel's own bounds. Zero-dep preserved, quality high, fast.

### Request ids and supersession

Every prompt carries a `request_id`. The daemon runs **one inference at a time** (one user, one
GPU — no reason for more). Queued-but-not-started requests for the same image are dropped when a
newer one arrives. The plug-in ignores any result whose `request_id` is not the latest it issued.
Typing a new prompt mid-inference therefore feels instant and never shows stale masks.

### Embedding cache

Keyed by `blake2b(downscaled pixels)`. The plug-in does not hash; it uploads what it is about to
segment when the window opens, and again when the input changes (switching between the visible
projection and the active layer). An unchanged image hits the daemon cache, across sessions too.
LRU-bounded, default 3 images.

### Daemon lifecycle (shaped by Constraint C)

- **Find-or-spawn.** The plug-in reads `runtime.json`; if present, checks the pid is alive and
  `GET /hello` succeeds with a compatible version and a valid proof that this is the daemon
  `runtime.json` describes, so another process on the same port cannot pose as it. If any
  check fails it discards the file and spawns. If absent it spawns, then polls for the file
  with a timeout. When a remote daemon is configured (Setup ▸ Advanced), the plug-in connects
  to that instead and never spawns.
- **Detached spawn on Windows.** `pythonw.exe` (not `python.exe` — that flashes a console),
  `creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`, stdio redirected to log files,
  `close_fds=True`. On POSIX, `start_new_session=True`. The daemon must outlive the plug-in
  process that spawned it.
- **Self-supervision.** The daemon is passed GIMP's pid — found by walking up from the plug-in
  process, since GIMP starts Python plug-ins through an intermediate process — and exits when
  GIMP is gone. It also exits after an idle TTL (default 30 min after the last request, set in
  Setup ▸ Advanced) to free VRAM. Single instance via a lockfile.
- **No restart-on-crash supervisor exists** — there's nothing to host one. A crash means the next
  plug-in invocation respawns. The Doctor panel shows the last crash log.
- **Remote daemons.** The same program runs on a GPU machine. The link is plain HTTP, so the
  recommended setup keeps that daemon on its own loopback address and reaches it through an SSH
  tunnel; Setup warns when the configured address is not loopback. See `docs/INSTALL.md`.

### Plug-in threading

The GTK main loop must never block on HTTP. All daemon calls run on a **worker thread**; results
are marshalled to GTK with `GLib.idle_add`. Progress from `/jobs/{id}` drives a real progress bar.
This is the difference between a responsive dialog and a frozen one; it is specified here so it is
not discovered late.

---

## 5. Interactive UI

GIMP plug-ins **cannot receive canvas click events** — there is no such API. So we build our
**own GTK3 canvas inside the plug-in dialog**.

- `Gtk.DrawingArea` + cairo, rendering the image with mask overlays composited on top.
- Text prompt box → PCS → instance list with per-instance toggles, score display, hover highlight.
  Click an instance to include/exclude it. Score and mask-threshold sliders filter **locally**.
- Click on canvas to drop positive/negative points → PVS refine, reusing the cached embedding so
  a refinement never re-encodes the image.
- Marching-ants overlay, overlay opacity, zoom/pan.
- A hint that SAM 3 wants **simple noun phrases** (`"red car"`), not relational descriptions
  (`"the car on the left"`) — negative boxes/points are how you exclude.
- Only on **Apply** do we touch the GIMP image.

The canvas is developed and tested outside GIMP, on Linux, where GTK 3, PyGObject and a display
are all it needs: `tools/canvas_harness.py` loads a PNG and talks to a local daemon or
synthesises results offline. Only `Gimp.*` calls need stubbing.

A selection drawn in GIMP's own window can stand in for a box prompt (*GIMP selection → box*),
and the scriptable procedures take a phrase, or points and a box, as arguments, so the whole
feature is reachable from Script-Fu and batch mode as well.

---

## 6. What gets segmented, and what comes out

### Input: the projection, not the drawable

Users segment what they **see**. Default input is the image's flattened projection (what GIMP
displays), with an option for "active layer only." Pixels are read via `Gegl.Buffer.get()` with
format `"R'G'B' u8"` — babl converts grayscale, indexed, and alpha for free. Layer offsets are
handled by working in **image space** throughout; only layer masks are translated back to layer
space at apply time.

### Output plumbing

- **Selection** — replace / add / subtract / intersect
- **Channels** — one per instance, named from prompt + score
- **Layer masks** — applied to a duplicate of the source layer
- **Layer groups** — one masked layer per instance
- **Vector paths** — marching squares → Douglas–Peucker → Bézier fit → `Gimp.Path`
- **Mask post-ops** — feather, grow, shrink, smooth, hole-fill, min-area filter
- Every Apply wrapped in `image.undo_group_start()/end()` — **one Ctrl+Z reverts the whole thing.**
- Last prompt / thresholds / output mode persist via `Gimp.ProcedureConfig`.

All registered as PDB procedures so it is scriptable from Script-Fu and other plug-ins.

---

## 7. Model management

- **Device:** cuda → mps → cpu. A ROCm GPU is `cuda` to torch.
- **Dtype:** bf16 on Ampere+ (safest for the DETR-style decoder), fp16 on older CUDA (Turing
  only emulates bf16), on ROCm and on MPS, fp32 on CPU. On a numerical failure in half
  precision, fall back to fp32 and log it.
- **Lazy per-engine load.** `Sam3TrackerModel` is only loaded on the first click-refine.
- **Open VRAM question:** loading `Sam3Model` and `Sam3TrackerModel` separately may duplicate
  the vision backbone in VRAM. It has not been measured yet. If both loaded exceed ~8 GB, either
  share the backbone weights or swap engines on demand; the answer decides whether 8 GB cards
  can hold both engines at once.
- **Weights location:** set `HF_HOME` to `%LOCALAPPDATA%\sam3-gimp\hf` so the ~3.6 GB cache lives
  somewhere the Doctor panel can show and clear. Also accept a **local path** for users who
  downloaded weights another way, and convert Meta's original `sam3.pt` on request — the gated
  flow is the main friction point, so give it more than one exit.

---

## 8. Onboarding

Installing a SAM-based tool has usually meant cloning a repository, installing SAM with pip,
downloading checkpoints and configuring paths by hand, and every one of those steps is a place
for an install to fail. **The bootstrap flow is a design pillar.**

1. First run detects no environment at `%LOCALAPPDATA%\sam3-gimp\`.
2. The Setup dialog downloads `uv` (one exe, checked against its published SHA-256), creates a
   venv with a pinned Python, installs PyTorch from the PyTorch wheel index (`torch==2.9.0`,
   `torchvision==0.24.0`; CUDA 12.8, ROCm 6.4 or CPU builds, or PyPI on Apple Silicon), then
   installs the daemon from the plug-in's own `_daemon` folder with its `[runtime]` extra
   (`transformers` pinned exactly) — live log + progress bar inside GIMP. Idempotent and
   resumable if GIMP is closed mid-install. The daemon is never installed by name: the `sam3d`
   and `sam3gimpd` names on PyPI are not this project.
3. **SAM 3 weights are gated.** The dialog explains, links to the model page to accept Meta's
   terms, takes a token, validates it, downloads with progress. The token is stored by
   `huggingface_hub` itself, not by us.
4. **Doctor** panel: device, VRAM, dtype, versions, daemon status, last crash log, Repair button.
5. `tools/install.ps1` for users who would rather work from a console.

A user who already has PyTorch can point Setup at that interpreter instead; Setup then installs
only the daemon and its `[runtime]` extra there, never torch.

---

## 9. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| MSIX-packaged (Store) GIMP may restrict subprocess spawn or socket binding | Resolved | Phase 0 (below): both work under Store GIMP 3.2 |
| GIMP 3.0.0 on Windows had broken Python plug-ins (GTK/Pango runtime) | High | Require 3.0.4+; diagnostic reports the exact import failure |
| Two engines may duplicate the backbone in VRAM | High (for 8 GB cards) | Measure; share weights or swap engines |
| `transformers` v5 SAM3 API churn | Medium | Pin exact versions; HTTP API isolates plug-in from daemon internals |
| 0.9B F32 ≈ 3.6 GB download | Medium | bf16 at load; check for an fp16 safetensors variant; local-path option |
| Remote daemon traffic is plain HTTP | Medium | Loopback by default; SSH tunnel recommended; Setup warns on a non-loopback address |
| Another local process on the daemon's port | Low | Bearer token, loopback `Host` check, and an identity proof in `/hello` (API.md) |
| Checkpoint converter fetched at runtime | Low | Downloaded from a pinned `transformers` commit and checked by SHA-256 before it runs |
| Windows Firewall prompt on loopback bind | Low | 127.0.0.1 normally doesn't prompt; INSTALL.md says what to do if it does |

### Phase 0 result

The one assumption that could have invalidated the architecture was whether a plug-in inside
the MSIX-packaged Microsoft Store build of GIMP may start a detached child process that
outlives it, and talk to that child over a loopback socket. It can: under Store GIMP 3.2 on
Windows the plug-in starts the daemon detached, the daemon survives the plug-in process, and
the plug-in reaches it over HTTP on 127.0.0.1. The container does redirect the plug-in's own
writes under `%LOCALAPPDATA%` into the package's private cache, so the plug-in's log can land
there while the daemon's, written by a separate process, does not (INSTALL.md, Troubleshooting).

### Licensing

- The plug-in and the daemon are MIT.
- The plug-in uses GIMP's `libgimp` and `libgimpui` through GObject introspection. Those
  libraries are LGPL-3.0-or-later (GIMP's `LICENSE` file and the libraries' source headers).
  The repository contains no GIMP code, and the daemon is a separate program in its own process
  and Python environment that talks to the plug-in only over HTTP.
- No SAM 3 weights are shipped. Users download them from Hugging Face with their own account
  after accepting Meta's SAM License, which is linked from the README.
- The `sam3.pt` converter is `transformers` code (Apache-2.0), downloaded at runtime and not
  redistributed.

---

## 10. Build order

| Phase | Deliverable |
|---|---|
| **0 — Spike** | Prove GTK + detached subprocess + loopback HTTP + pixel I/O under Store GIMP. |
| **1 — `sam3gimpd`** | Standalone package: modelmgr, PCS + PVS engines, cache, jobs, HTTP API, CLI, tests. No GIMP needed. |
| **2 — Plug-in MVP** | Launcher, bootstrap/doctor, HTTP client on a worker thread, text prompt → selection/channels/layers via ProcedureDialog. First usable end-to-end. |
| **3 — Interactive canvas** | GTK3 preview built on Linux first, instance overlays/toggles, local thresholds, click-to-refine. |
| **4 — Output polish** | Paths/vectors, layer groups, mask post-ops, PDB surface, undo groups. |
| **5 — Packaging** | Installer, release zip, docs, CI, the other platforms (CPU / Mac / Linux / remote). |

Phases 0–4 are done. Of phase 5, the installer, the release zip, the docs and CI exist; the
other platforms are implemented but not yet tested end to end, and the VRAM measurement in §7
is still open.

---

## 11. Repository layout

```
gimp-sam3-plugin/
├── plugin/sam3_gimp/          # the unit of distribution; runs in GIMP's embedded Python
│   ├── sam3_gimp.py           # entry (filename must match folder name)
│   ├── client.py  launcher.py  bootstrap.py  gimpbridge.py  outputs.py  contours.py
│   ├── ui/  canvas.py  main_dialog.py  setup_dialog.py
│   └── _daemon/               # the daemon, shipped inside the plug-in
│       ├── API.md  pyproject.toml
│       └── sam3gimpd/  server.py  jobs.py  session.py  masks.py  modelmgr.py  cli.py
│           └── engines/  base.py  stub.py  pcs.py  pvs.py
├── tests/
│   ├── daemon/                # headless, CPU, synthetic images, stub engine
│   ├── plugin/                # client, launcher, bridge, outputs, canvas, dialogs, entry
│   └── fake_gimp/             # stubs gi.repository.Gimp for outputs/gimpbridge logic
├── tools/  build_release.py  canvas_harness.py  dev_sync.py  diagnose.py  install.ps1
└── docs/   INSTALL.md  DEVELOPING.md
```

The plug-in directory is complete on its own: the release zip, the installer and a hand copy
all produce the same tree, with the daemon inside it to install from. The HTTP API (documented
in `plugin/sam3_gimp/_daemon/API.md`) is the contract between the two halves.

---

## 12. Testing

The configuration that matters most — Windows, CUDA, GIMP from the Microsoft Store — is the
hardest to put in CI, so the design keeps as much as possible testable without it.

- **`sam3gimpd`** — fully testable headless, on CPU, with small synthetic images, through the
  stub engine. Tests that need torch, a GPU or the weights carry `needs_torch`, `needs_gpu` or
  `needs_weights` and skip without them. The CLI makes manual testing trivial:
  `sam3gimpd serve --stub`, then `curl`.
- **HTTP API** — protocol tests run against a real daemon process on the stub engine; no model
  needed. `tests/test_end_to_end.py` drives a detached stub daemon through the plug-in's own
  client.
- **The contract mirror** — the plug-in copies the daemon's constants by hand (it may not import
  them); `tests/test_contract_mirror.py` checks the two agree.
- **Canvas and windows** — run on Linux with GTK 3 and a display, and in CI under Xvfb, via
  `tools/canvas_harness.py` and the dialogs' standalone entry points.
- **Output plumbing / gimpbridge logic** — `tests/fake_gimp/` stands in for
  `gi.repository.Gimp` and `Gegl`, modelled on GIMP's source where GIMP's behaviour is
  surprising (channel transforms clip to the channel; filling a channel uses the context
  colour).
- **What needs the real thing** — every `Gimp.*` call against real libgimp, the Windows detached
  spawn and the MSIX container, every CUDA / ROCm / MPS path, and the VRAM question in §7. These
  are exercised by running the plug-in in GIMP, synced there with `tools/dev_sync.py`. GIMP 3
  from Flathub, driven headlessly with `gimp -i -b`, can cover the libgimp calls on Linux
  without Windows.

Nothing is described as working in GIMP or on a GPU until it has run there;
`docs/DEVELOPING.md` lists what has.

---

## Design history

- **HTTP and a standalone daemon.** The first draft had a custom binary protocol and a daemon
  private to the plug-in. The daemon became a product with its own CLI and HTTP API; ≤1008px
  uploads and cropped soft masks upsampled by GIMP replaced full-resolution RGB and RLE masks;
  thresholds moved to the client; request ids and supersession were added; Constraint C led to
  the self-supervising detached daemon.
- **Contour tracing moved to the plug-in.** Masks travel as soft uint8 and are re-thresholded
  client-side, so tracing belongs on the same side as the threshold slider. Paths mode emits
  fitted Béziers with hole winding.
- **The daemon moved inside the plug-in.** It lives at `plugin/sam3_gimp/_daemon/` rather than
  at the top of the repository, so `plugin/sam3_gimp/` is the unit of distribution and a hand
  copy of that folder has everything Setup needs.
- **Identity geometry.** Masks were once mapped through an anisotropic squash onto a 1008×1008
  canvas that the processor never performed; the model canvas is now the uploaded image itself
  (API.md §5).
