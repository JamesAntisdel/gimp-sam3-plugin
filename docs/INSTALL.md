# Installing sam3-gimp on Windows

This guide covers Windows 10/11 with GIMP 3 from the Microsoft Store and an NVIDIA GPU.

**Tested configuration:** GIMP 3.2 from the Microsoft Store with an NVIDIA RTX 2080 Ti
(11 GB), using an existing PyTorch environment
([2b](#2b-using-an-environment-you-already-have)). The Setup window's own *Install* button
and `tools/install.ps1` have not yet been run on Windows. If something goes wrong, start
with [Troubleshooting](#troubleshooting) and the plug-in log described there.

---

## 0. What you need

| | |
|---|---|
| **GIMP 3.0.4 or newer** | **Not 3.0.0.** That release shipped a broken Python/GTK runtime on Windows and Python plug-ins simply do not load. Check *Help ▸ About*. |
| Windows 10 or 11, 64-bit | |
| An NVIDIA GPU | CUDA 12.8 wheels are installed by default. CPU-only works but is slow and is not the tested path. |
| ~8 GB free disk | ~3.6 GB of model weights plus a torch virtualenv. |
| A HuggingFace account | The SAM 3 weights are **gated**; you must accept Meta's terms yourself. See [Tab 2](#tab-2--sam-3-weights). |
| Internet | For the one-time install. Afterwards everything is local. |

You do **not** need Python installed — the installer downloads its own pinned interpreter.
But if you *already* have a working PyTorch + CUDA environment, use it instead and skip the
3 GB download: see [2b](#2b-using-an-environment-you-already-have).

**You do not need to clone, download or build SAM 3 itself.** There is no Meta repository to
check out and nothing to compile. SAM 3 runs through the `transformers` library, which Setup
installs, and the model weights come from HuggingFace through the Setup dialog, or from a
folder you already have ([Tab 2](#tab-2--sam-3-weights)). The only code fetched separately is
the `transformers` conversion script, and only if you convert a raw `sam3.pt` yourself.

---

## 1. Put the plug-in where GIMP will find it

GIMP 3 discovers a Python plug-in only at

```
<plug-ins>\sam3_gimp\sam3_gimp.py
```

**The folder name and the file name must match exactly.** A file at
`<plug-ins>\sam3_gimp.py`, or a folder named `sam3-gimp` (hyphen), or `sam3_gimp\main.py`,
will be silently ignored — no error, the plug-in just never appears.

The user plug-ins directory is normally:

```
%APPDATA%\GIMP\<version>\plug-ins\
```

where `<version>` is GIMP's major.minor — `3.2` for a current install, `3.0` for the first
3.x releases. **It is not always `3.0`**: a plug-in copied into `GIMP\3.0\plug-ins` on a
GIMP 3.2 machine is never seen, with nothing to say why. The finished layout is:

```
%APPDATA%\GIMP\3.2\plug-ins\sam3_gimp\
    sam3_gimp.py          <- entry point; must match the folder name
    __init__.py
    bootstrap.py
    client.py
    contours.py
    gimpbridge.py
    launcher.py
    outputs.py
    ui\
        __init__.py
        canvas.py
        main_dialog.py
        setup_dialog.py
    _daemon\              <- the sam3gimpd package; Setup installs it from here
        pyproject.toml
        sam3gimpd\ ...
```

Copy the **whole** folder. Without `_daemon\` the Setup dialog has nothing to install the
daemon from and says so.

> **Confirm the real path in GIMP.** Open **Edit ▸ Preferences ▸ Folders ▸ Plug-ins** — the
> first writable entry in that list is the directory to use, whatever this document says. On
> the tested Store install it was the ordinary `%APPDATA%\GIMP\3.2\plug-ins`, not a path
> redirected into the MSIX package; if yours differs, use what GIMP tells you.

### Option A — the release zip (no terminal)

Download `sam3-gimp-<version>.zip`, then extract it **into** the plug-ins folder so you end up
with `<plug-ins>\sam3_gimp\sam3_gimp.py`. The archive has a single top-level `sam3_gimp`
folder, so extracting it directly into the plug-ins directory is correct — you should **not**
end up with `sam3_gimp\sam3_gimp\`.

That is the whole file-copying step. Everything after it — the environment, the model weights,
or pointing at a PyTorch install you already have — happens inside the Setup dialog, with no
command line.

(Building the zip yourself, from a checkout: `python tools/build_release.py`.)

### Option B — copy it by hand

Download or clone this repository, then copy the whole `plugin\sam3_gimp` folder into the
plug-ins directory. In PowerShell, from the repository root:

```powershell
$dest = "$env:APPDATA\GIMP\3.2\plug-ins"    # use the version Preferences shows you
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Copy-Item -Recurse -Force .\plugin\sam3_gimp $dest
```

### Option C — the installer script

```powershell
powershell -ExecutionPolicy Bypass -File tools\install.ps1
```

It picks the newest `GIMP\3.x` config directory that exists, copies the tree and reports
what it did. Useful flags:

| Flag | Effect |
|---|---|
| `-DryRun` | Say what would happen; change nothing. |
| `-PluginDir <path>` | Install somewhere other than the default (use what Preferences showed you). |
| `-GimpVersion 3.0` | Target a specific GIMP config directory instead of the newest one found. |
| `-Clean` | Delete the existing `sam3_gimp` folder first instead of merging into it. |
| `-SetupVenv` | Also build `%LOCALAPPDATA%\sam3-gimp\venv` from the console instead of in the GUI (stub-capable, no torch). |
| `-SetupVenv -Torch` | As above, and install PyTorch and the daemon's runtime libraries (`transformers` and friends) — the ~3 GB part — with the same pins Setup uses. |
| `-Accelerator cpu` | With `-Torch`, install the CPU wheels instead of CUDA. |

Nothing here needs administrator rights; every path written is user-scoped.

### Then restart GIMP

GIMP scans for plug-ins at startup only. It also runs each plug-in as a **fresh process read
from disk**, so after any later edit you also need a restart.

---

## 2. First run: the Setup dialog

Open **Filters ▸ AI Segmentation ▸ SAM 3 Setup / Doctor…** — it is enabled with **no image
open**. On a fresh install the same entry reads **⚠ SAM 3: first-time setup required…** (GIMP
caches menu labels until the plug-in file changes, so the warning can linger one restart after a
successful install). The canvas dialog has a **Setup / Doctor…** button in its bottom button row,
next to Close and Apply, and the plain argument dialog (used when the canvas is unavailable) has
the same button beside OK / Reset / Cancel.

*(It is a plain procedure rather than an image procedure precisely so it is reachable with no
image open — which is the state you are in on a first run.)*

If the menu is not there at all, jump to [Troubleshooting](#troubleshooting) — that is a
plug-in-loading problem, not a setup problem.

The dialog has three tabs.

> **Already have a working PyTorch + CUDA environment?** Don't press Install — it would build
> a second one and download another ~2.5 GB of torch. Open **Use an existing Python
> environment** on that same tab instead; see
> [2b](#2b-using-an-environment-you-already-have).

### Tab 1 — Install

One button (plus an **Use an existing Python environment** section, see
[2b](#2b-using-an-environment-you-already-have), if you would rather reuse a PyTorch install
you already have). The button performs, in order:

1. **Download `uv`** (a single ~15 MB static binary, version 0.9.7, pinned) into
   `%LOCALAPPDATA%\sam3-gimp\tools\`, and check it against the SHA-256 that `uv` publishes
   beside each release.
2. **Create a virtualenv** at `%LOCALAPPDATA%\sam3-gimp\venv` with a **pinned Python 3.11**
   that `uv` downloads itself. Your system Python, if any, is not touched or used.
3. **Install PyTorch** from PyTorch's own wheel index — the CUDA 12.8 build here, the CPU or
   ROCm build on machines without an NVIDIA card:
   `uv pip install --python <venv python> --index-url https://download.pytorch.org/whl/cu128 torch==2.9.0 torchvision==0.24.0`.
   This is the big step: torch and the CUDA runtime are around 3 GB.
4. **Install the daemon** from the plug-in folder, with the libraries it needs from PyPI
   (`transformers` and a few small ones):
   `uv pip install --python <venv python> "<plug-ins>\sam3_gimp\_daemon[runtime]"`.
   It is always installed from that folder, never by name: the `sam3d` and `sam3gimpd` names
   on PyPI are not this project.
5. **Verify** the venv can import what it needs and that `sam3gimpd --version` answers.

A live log and a real progress bar run throughout, and the window stays responsive: every
step runs on a worker thread, because a silent ten-minute freeze inside GIMP is
indistinguishable from a crash.

**It is resumable.** If you close GIMP mid-install, reopening the dialog offers *Resume
install* and skips the steps already on disk. A journal at
`%LOCALAPPDATA%\sam3-gimp\install-state.json` is fingerprinted by the choices that produced
it, so if the accelerator decision changes (CPU → CUDA) the stale journal is discarded
rather than leaving you with a half-and-half environment.

*Reinstall from scratch (ignore what is already there)* forces the whole sequence again.

### Tab 2 — SAM 3 weights

This is the step most likely to frustrate you, and it is not our doing: **Meta gates the
SAM 3 checkpoint.** We ship no weights and cannot ship any.

1. Click through to **<https://huggingface.co/facebook/sam3>**, sign in, and **accept the
   terms**. Access is usually immediate but can be queued for review.
2. Create a **read token** at **<https://huggingface.co/settings/tokens>**.
3. Paste it into the token box and press **Check token**. This validates the token *and*
   your access to the gated repo **before** starting a 3.6 GB download — so a permissions
   problem is a sentence of English, not a failed download an hour later.
4. Press **Download weights (~3.6 GB)**. Files land in `%LOCALAPPDATA%\sam3-gimp\hf`
   (`HF_HOME` is pinned there so the Doctor tab can show and clear it).

Your token is stored by `huggingface_hub` itself; this plug-in does not keep a copy, and it
is passed to subprocesses through the environment, never on a command line (`ps`-style
process listings are world-readable).

**Have the original `sam3.pt`?** Use the third row, *Original sam3.pt*. That file is what Meta
ships (and sits in the gated repo beside the converted files), but it is **not** the layout
`transformers` loads — the tensor names differ and there is no config or tokenizer next to it.
Setup can convert it: pick the `.pt`, press **Convert and use**, and it writes a proper
checkpoint into `%LOCALAPPDATA%\sam3-gimp\models\sam3-converted` and starts using it.

This route needs **no HuggingFace account, no terms to accept and no 3.6 GB download** — the
converter builds the config itself and takes the tokenizer from the public
`openai/clip-vit-base-patch32`. It does need the environment installed first, because the
conversion runs there (it needs torch), and it needs a network connection the first time, for
that tokenizer and for the converter itself. The converter is the `convert_sam3_to_hf.py`
script from Hugging Face `transformers` (Apache-2.0); Setup downloads it from a pinned
commit and checks its SHA-256 before running it.

**Already downloaded the weights another way?** Use the second exit at the bottom of the
tab: point *Use this folder* at the directory containing the checkpoint. It is validated on
the spot — it must hold at least one `.safetensors`/`.bin` file **and** a `config.json`
(point it at the whole checkpoint folder, not at a single weight file). The daemon is then
started with `SAM3_WEIGHTS_DIR` pointing there instead of downloading anything.

### Tab 3 — Doctor

The diagnostics panel: detected device and dtype, torch and `transformers` versions,
whether the weights are present and where they came from, the daemon's status if one is
running, the paths it uses, and the tail of the last crash log. **Refresh** re-runs it,
**Repair** clears the install journal and hands you back to the Install tab set to
*Reinstall* (downloads already on disk are reused; a broken venv is rebuilt),
**Stop daemon** asks a running `sam3gimpd` to exit now (freeing the GPU; the next
segmentation starts a fresh one), and **Copy report** puts the whole thing on the clipboard —
please paste that into any bug report.

### Updating the daemon

Re-copying the plug-in folder updates the plug-in, but the daemon is a separate package inside
your Python environment and keeps its old code until it is reinstalled. The segmentation
window notices: the daemon reports a hash of its source in `/hello`, the plug-in hashes the
copy it ships, and when they differ the device line reads **DAEMON OUT OF DATE** and the
status line tells you what to press. That is **Update daemon** on the Install tab: it
reinstalls the daemon from the bundled copy into whichever environment is in use (yours or
the managed one), never touches torch, and stops the running daemon so the next use starts
the new code.

---

## 2b. Using an environment you already have

If you already have a working PyTorch + CUDA install, don't let Setup build a second one.
**This is a normal option in the dialog — no command line, no environment variables.**

On the **Install** tab, open **Use an existing Python environment**:

1. **Interpreter** — browse to the `python.exe` of the environment you want to use
   (in a venv that is `...\Scripts\python.exe`; in a conda env it sits at the env root).
2. Press **Check**. Setup runs that interpreter and reports back what it found, for example:

   > **Python 3.11.9 - torch 2.9.0 on NVIDIA GeForce RTX 2080 Ti (sm_75) - sam3gimpd not installed**

   It will also list anything that would stop it working — no torch, a CPU-only torch build,
   a Python older than 3.10 (the oldest the daemon installs into), or a missing daemon.
   There is no upper limit: 3.14 is as welcome as 3.11, and the daemon is tested on both.
3. If it says **sam3gimpd not installed**, press **Install sam3gimpd here**. That installs the
   daemon from the plug-in folder together with its `[runtime]` libraries — `transformers`
   and a few small ones, **never torch** — so whatever PyTorch you already have is left
   exactly as it is.
4. Press **Use this environment**. The choice is saved to
   `%LOCALAPPDATA%\sam3-gimp\settings.json` and used from then on.

**Stop using it** returns to Setup's own managed environment at any time. If you later delete
or move the interpreter, the plug-in notices the path is gone and quietly falls back rather
than failing to start.

Setup refuses to accept an environment with **no torch at all** — that is the one thing it
cannot install for you. A CPU-only torch is accepted with a warning, since it does work, just
slowly.

### Installing the daemon by hand

What Setup does can be done from a terminal, which is how you would prepare a GPU machine
that has no GIMP. With [`uv`](https://docs.astral.sh/uv/) installed (it fetches Python 3.11
itself if needed) and a copy of the plug-in folder:

```
uv venv --python 3.11 sam3-env
uv pip install --python <sam3-env python> --index-url https://download.pytorch.org/whl/cu128 torch==2.9.0 torchvision==0.24.0
uv pip install --python <sam3-env python> "<path to>/sam3_gimp/_daemon[runtime]"
```

`<sam3-env python>` is `sam3-env\Scripts\python.exe` on Windows and `sam3-env/bin/python`
elsewhere. Use `whl/cpu` instead of `whl/cu128` for a CPU-only machine and `whl/rocm6.4` for
AMD on Linux; on Apple Silicon leave out `--index-url`, since PyTorch's macOS wheels come
from PyPI. Always install the daemon from the `_daemon` folder, never by name.

### A daemon on another machine

The GPU need not be in the computer running GIMP. The link between the plug-in and the
daemon is **plain HTTP**: the bearer token and your images cross the network unencrypted.
So keep the daemon listening on the GPU machine's loopback address, as it does by default,
and reach it through an SSH tunnel:

1. On the GPU machine, install the daemon ([by hand](#installing-the-daemon-by-hand), or
   with *Install sam3gimpd here* in a GIMP there) and start it on a fixed port:

   ```
   sam3gimpd serve --port 8765 --idle-ttl 0
   ```

   `--idle-ttl 0` keeps it running until you stop it; without it, it exits after 30 idle
   minutes. Its `runtime.json`, in that machine's `sam3-gimp` data directory
   (`~/.local/share/sam3-gimp` on Linux), holds the bearer token.
2. On the GIMP machine, open the tunnel and leave it open while you work (Windows 10 and 11
   include the `ssh` command):

   ```
   ssh -L 8765:127.0.0.1:8765 <gpu-host>
   ```

3. In GIMP, open *Setup ▸ Install ▸ Advanced ▸ Remote daemon*, enter
   `http://127.0.0.1:8765` and the token, and press **Use this daemon**.

From then on the plug-in connects there instead of starting a local daemon; **Back to a
local daemon** undoes it.

The daemon can also listen on the network directly (`sam3gimpd serve --host 0.0.0.0`), with
no tunnel. Then anyone who can see the traffic can read the token and the images, so do that
only on a network you trust completely. Setup warns when the remote address is not a
loopback address.

### The idle timeout

The daemon exits after 30 minutes without a request, which frees the GPU. *Setup ▸ Install ▸
Advanced* has **Daemon exits after idle minutes**: shorter for a small card, 0 to keep it
running until GIMP closes. It applies the next time the daemon starts.

### For scripting and development: `SAM3D_COMMAND`

The UI above is the supported route. If you are automating an install, or developing the
plug-in, the environment variable `SAM3D_COMMAND` overrides the command used to start the
daemon and **takes precedence over the dialog's setting**:

```powershell
setx SAM3D_COMMAND "C:\path\to\your\python.exe -m sam3gimpd"
```

(`setx` is the built-in Windows command for a *persistent* environment variable — unlike
`set`, which lasts only for the current window. It does not affect the window you type it in,
so restart GIMP afterwards.) **Quote the whole value**, which is what makes a path under
`C:\Program Files\` work.

### Related variables

| Variable | What it does |
|---|---|
| `SAM3D_COMMAND` | Override the command used to start the daemon. Wins over the dialog setting. |
| `SAM3D_DEVICE` | Force `cuda`, `cuda:1`, `cpu`, or `mps` (a ROCm GPU is `cuda` to torch). |
| `SAM3D_DTYPE` | Force `bf16`, `fp16`, or `fp32`. |
| `SAM3_GIMP_HOME` | Move the whole working directory (logs, `runtime.json`, settings, weights cache). |

> **On precision:** the automatic policy picks `fp16` on Turing cards (GTX 16-series,
> RTX 20-series) and `bf16` only on Ampere and newer. That is deliberate. Recent PyTorch
> reports bf16 as "supported" on Turing, but it is *emulated* there — slower, with no accuracy
> benefit. Forcing `SAM3D_DTYPE=bf16` on such a card defeats the guard and makes things worse.

## 3. Your first segmentation

1. Open an image.
2. **Filters ▸ AI Segmentation ▸ Segment interactively (canvas)…**
3. The first prompt of a session is the slow one: the daemon starts, loads the checkpoint
   and encodes the image. Later prompts on the same image reuse the cached embedding.
4. Type a **simple noun phrase** — `red car`, `yellow school bus`, `person`. SAM 3's concept
   head wants noun phrases, **not** relational descriptions: `the car on the left` will not
   work as you hope. Exclude things with negative points or boxes instead.
5. Every match appears as a separate instance with a score. Tick the ones you want. The
   *score* and *mask threshold* sliders filter instantly and locally — they never re-run
   inference.
6. Or segment by clicking (SAM 3's *visual* prompting, PVS). Above the preview, set
   **Canvas click** to *places a point*: left-click an object and SAM 3 segments the thing
   under the pointer; right-click (or Shift-click) a spot to exclude it; Ctrl-drag a box. Each
   click refines the previous result, Backspace removes the last point and re-runs, Esc clears
   them. A single click returns up to three candidates (the whole object, a part, a sub-part)
   with only the best ticked; pick another row if the model guessed wrong, or add a second
   point to settle it, after which one mask comes back. This route does not depend on the
   model knowing a *name* for the object, so it works for the strap, the cable, the
   odd-shaped offcut that no phrase will find. **GIMP selection → box** turns a selection you
   drew in GIMP's own window (rectangle, lasso, anything) into the box: GIMP stays usable
   while the plug-in window is open, so draw there, then press it. With *picks an object*,
   the other setting, a click on the preview ticks or unticks an object from the list instead.
   The dialog switches to *picks an object* by itself when a text prompt returns results, and
   back to points when you refine by clicking.
7. Choose an output — selection, channels, layer masks, a layer group, or paths — and press
   **Apply**. The whole thing is a single undo step.

### What each output produces, and where to look

Only two of the five change the canvas directly. The rest land in dockable dialogs that GIMP
keeps closed by default, so an Apply can look like it did nothing; the status line at the
bottom of the plug-in window says what was made and where.

| Output | What appears | Where |
|---|---|---|
| **Selection** | Marching ants around every ticked instance, combined with your existing selection by the *Selection* combo (Replace / Add / Subtract / Intersect). | On the canvas. |
| **Channels (one per instance)** | One saved channel per instance, named from the prompt and score (`green apple (0.93)`), hidden, at 50 % opacity. Nothing changes on the canvas. | *Windows ▸ Dockable Dialogs ▸ Channels*. Click a channel's eye to see it as an overlay; right-click ▸ *Channel to Selection* to use it. Same result as *Select ▸ Save to Channel* per object. |
| **Layer mask** | A copy of the source layer with the union of the instances as its mask (or the layer itself, with *Duplicate layer* off). | The Layers dialog; the canvas shows the masked copy. |
| **Layer group** | A group holding one masked copy of the source layer per instance. | The Layers dialog. |
| **Paths (vectors)** | One vector path per instance, traced from the mask edge (marching squares, simplified, Bézier-fitted), shown on the canvas in the path colour and selected. | *Windows ▸ Dockable Dialogs ▸ Paths*. *Edit ▸ Stroke Path…* draws it; *Select ▸ From Path* makes a selection; the Paths tool edits the nodes. Thin or tiny masks may trace to nothing, and the status line says so. |

There is also a scriptable, dialog-free surface for Script-Fu and batch work:
`plug-in-sam3-segment-by-text` and `plug-in-sam3-segment-by-points`.

---

## 3b. Tuning: what the settings actually do

Two sliders decide almost everything. Both filter **locally**, so in the canvas dialog they
re-render instantly with no round trip.

| Setting | What it does | Change it when |
|---|---|---|
| **Score threshold** (0–1) | How sure the model must be to keep a match. | **Lower it (0.15–0.25) when you only get part of the object.** SAM 3 often returns one object as several partial matches; a high threshold keeps just the best fragment. Raise it when unrelated things get selected. |
| **Mask threshold** (1–255) | How much of each mask to keep. 128 is the model's own edge. | Lower (≈90) for a fatter selection that fills holes; raise (≈170) when it bleeds into the background. |
| **Max instances** (1–256) | Upper bound on matches a text prompt returns, best first. Default 64. | Raise for crowded scenes ("every window"). The status line says when the daemon had to drop matches. The model itself never finds more than about 200. |
| **Use visible projection** | On: segment what you see, all layers combined. Off: the active layer only. | Turn off to segment one layer while others are visible. |

### "It found the person but only a sliver of the guitar"

That is the score threshold, not the model. A well-known object like a person comes back as one
confident instance; something smaller or more unusual comes back as several partial ones scoring
0.2–0.4, and a threshold much above that keeps only a fragment. The default is **0.30**.

After each run the plug-in reports what it did, for example *"applied 1 of 7 matches; 6 more scored
up to 0.42"*, and suggests a threshold to try. Lower it and run again.

### Why "nothing found" can be final

SAM 3 scores every match as `sigmoid(instance logit) × sigmoid(presence logit)`. The
**presence** term is a per-image verdict on whether the concept is there at all, and a confident
"no" scales *every* candidate down — often below even the 0.02 floor. When a phrase returns
nothing, that is usually the model saying "absent", not "found but weak", and no threshold will
recover it. Click the object instead (next section).

### When text prompting is the wrong tool

SAM 3's text path only finds concepts its text encoder knows. Plain nouns work well — `guitar`,
`dog`, `red car`. Compound, thin or unusual things often return **nothing at all**, and no
threshold will help: `guitar strap`, `shoelace`, `the guitar on the left`. It is not a chat box.

For those, use **Segment by points or box** instead and click the object. That path (SAM 3's
tracker) does not depend on the model knowing a name for the thing, so it handles the strap, the
cable, the odd-shaped offcut — anything you can point at. The canvas dialog lets you add positive
clicks to grow the selection and negative clicks to push it back.

A reasonable habit: try text first for whole, nameable objects; switch to clicking the moment a
prompt returns a fragment or nothing.

## 4. Where everything lives

| Path | What |
|---|---|
| `%APPDATA%\GIMP\<version>\plug-ins\sam3_gimp\` | The plug-in itself, including the daemon's source in `_daemon\` |
| `%LOCALAPPDATA%\sam3-gimp\` | Everything else this project owns |
| `%LOCALAPPDATA%\sam3-gimp\venv\` | The daemon's virtualenv: Python 3.11, torch, transformers |
| `%LOCALAPPDATA%\sam3-gimp\hf\` | HuggingFace cache — the ~3.6 GB of weights |
| `%LOCALAPPDATA%\sam3-gimp\tools\uv.exe` | The pinned installer binary |
| `%LOCALAPPDATA%\sam3-gimp\logs\sam3gimpd.log` | Daemon log — **the first place to look** |
| `%LOCALAPPDATA%\sam3-gimp\logs\install.log` | Transcript of the last install |
| `%LOCALAPPDATA%\sam3-gimp\logs\crash.log` | Last daemon traceback, shown by Doctor |
| `%LOCALAPPDATA%\sam3-gimp\runtime.json` | Live daemon handshake: port, bearer token, pid. Deleted on clean exit. |
| `%LOCALAPPDATA%\sam3-gimp\install-state.json` | Resumable-install journal |

Nothing is written outside your user profile, and nothing needs administrator rights. On
Linux and macOS the equivalent directory (`~/.local/share/sam3-gimp`,
`~/Library/Application Support/sam3-gimp`) is created readable by your user only.

### The daemon's lifecycle, so nothing surprises you

There is a background process, `sam3gimpd`, and it is deliberate: keeping the model loaded and
the image embedding cached is the entire reason prompts are fast. It is spawned **detached**
by the plug-in the first time you segment. It exits by itself when

* GIMP quits (it is given GIMP's pid and watches it), or
* it has been idle for 30 minutes (freeing your VRAM), or
* you press **Stop daemon** on the Doctor tab (or, from a Command Prompt,
  `curl -H "Authorization: Bearer <token from runtime.json>" -X POST http://127.0.0.1:<port>/shutdown`).

It listens on `127.0.0.1` on an OS-assigned port, requires a bearer token that is regenerated
every start, and refuses requests whose `Host` header is not loopback. It is not reachable
from your network. When the plug-in connects, the daemon's `/hello` reply has to carry a
proof that it is the daemon described in `runtime.json`, so another program that happens to
listen on the same port cannot pose as it.

---

## Troubleshooting

### The menu "AI Segmentation" does not appear

In rough order of likelihood:

1. **GIMP is 3.0.0.** Python plug-ins are broken in that release on Windows. Update to
   3.0.4 or newer. *Help ▸ About* shows the version.
2. **The folder/file names do not match.** It must be
   `<plug-ins>\sam3_gimp\sam3_gimp.py`. Not `sam3-gimp`, not a loose `.py`.
3. **Wrong plug-ins directory.** Check *Edit ▸ Preferences ▸ Folders ▸ Plug-ins* and use the
   path GIMP itself lists.
4. **GIMP was not restarted.**
5. **An import failed during registration.** Start GIMP from a console
   (`"C:\Program Files\GIMP 3\bin\gimp-3.0.exe" --verbose`) and look for a traceback
   mentioning `sam3_gimp`. Store builds are awkward to run this way; the *Filters ▸ Script-Fu
   ▸ Console* error log and the GIMP error console are the fallback.

### "Setup / Doctor" opens but the Install button fails

Open `%LOCALAPPDATA%\sam3-gimp\logs\install.log`. Common causes:

* **No network / a corporate proxy.** `uv` and PyPI must be reachable. Set `HTTPS_PROXY`
  in your user environment and restart GIMP.
* **Antivirus quarantined `uv.exe`.** It is an unsigned single-file binary downloaded from
  the official `astral-sh/uv` GitHub releases; some scanners dislike that. Whitelist
  `%LOCALAPPDATA%\sam3-gimp\tools\`.
* **Disk full.** The venv plus weights want ~8 GB.
* Press **Install** again — it resumes rather than restarting.

### The download says 401, 403, or "gated"

You have not accepted Meta's terms with the account that owns your token, or access is still
queued. Go to <https://huggingface.co/facebook/sam3> while signed in, accept, and confirm
the page no longer shows a gate. Then re-run **Check token**. A token created *before* you
accepted still works — the gate is on the account, not the token — but it must be a token
belonging to the account that accepted.

### "The SAM 3 daemon could not be started" / No module named sam3gimpd

If the error names GIMP's own interpreter — something under
`...\WindowsApps\GIMP...\bin\pythonw.exe` — then **the environment has not been installed
yet**. That path is GIMP's embedded Python, which has no `sam3gimpd` and never will.

Fix it in **Filters ▸ AI Segmentation ▸ SAM 3 Setup / Doctor…**, either by pressing **Install**,
or — if you already have PyTorch working — with **Use an existing Python environment**
([2b](#2b-using-an-environment-you-already-have)), which is much faster.

### "Plug-in crashed: sam3_gimp.py"

That is GIMP reporting that the plug-in *process* died rather than returning an error. The
plug-in writes the reason to `plugin.log` (below) as a `Fatal Python error` block with the
Python stack of every thread — send that block.

### "Frozen" right after opening the dialog, or "another instance holds the lock"

The first segmentation after the daemon starts loads a 3.6 GB checkpoint and initialises CUDA —
often a minute. The dialog says **Loading model (first use after start…)** during it, and
`sam3gimpd.log` records `loading pcs …` / `loaded pcs in N s`. The daemon stays up for 30 idle
minutes so a working session pays this once.

If `sam3gimpd.log` shows repeated *another sam3gimpd instance holds … exiting*, an earlier daemon
hung on its way out while still holding the lock. The plug-in recognises that (a holder that is
alive but not answering), terminates it and spawns again; the daemon also releases its lock before
GPU teardown and ends itself by force if teardown stalls. If GIMP appears frozen for more than 45 s,
`plugin.log` gets a stack of every thread — send that block.

### After updating the plug-in files

Re-copying the plug-in folder updates the *plug-in*; the daemon is a separate package inside your
Python environment and keeps its old code until reinstalled. The segmentation window says
**DAEMON OUT OF DATE** in its device line when that is the case. Open Setup and press
**Update daemon** on the Install tab (or **Reinstall / update sam3gimpd here** on the
existing-environment row; they do the same thing for that route). Either reinstalls the daemon
and stops the one that is running, so the next use starts the new version — `sam3gimpd.log`
shows it: `sam3gimpd 0.1.1 listening …`. Until then the old one keeps serving.

### Every instance got selected, then they all unselected at the end

That is the **Selection** combo in the Output section set to *Subtract* or *Intersect* while
nothing was selected beforehand. Taken literally, intersecting with an empty selection is
empty and subtracting from it is empty. The plug-in follows GIMP's convention that no
selection means the whole image instead: *Intersect* with nothing selected keeps
the result, and *Subtract* from nothing selected selects everything **except** the matches.
The status line says which happened. If you wanted the matches themselves, set the combo to
*Replace*. The dialog remembers the combo between sessions, which is how it can be set to
something you do not remember choosing.

### Apply produced an empty selection, a blank layer mask, or empty channels

After a selection Apply the status line reports the selection's bounds (*Selection now: …*),
and every Apply writes one line per stage to `plugin.log` with the selection bounds after
it, so the log shows at which step a mask went missing. Send those lines with a bug report.

### Where the logs are

| File | What is in it |
|---|---|
| `%LOCALAPPDATA%\sam3-gimp\logs\plugin.log` | The plug-in side: every invocation, every error. **Start here.** |
| `%LOCALAPPDATA%\sam3-gimp\logs\sam3gimpd.log` | The daemon's own stdout/stderr, including why it failed to start. |
| `%LOCALAPPDATA%\sam3-gimp\logs\crash.log` | The daemon's last crash, if any. |
| `%LOCALAPPDATA%\sam3-gimp\logs\install.log` | The Setup tab's install transcript. |

Paste `%LOCALAPPDATA%\sam3-gimp\logs` into the File Explorer address bar to open the folder.

> **Store GIMP and a missing `plugin.log`.** The Microsoft Store build runs in an MSIX
> container, and Windows can redirect a packaged app's writes to `%LOCALAPPDATA%` into the
> package's own cache. The daemon is a separate process and writes to the real folder, so
> `sam3gimpd.log` is where you expect, while the plug-in's `plugin.log` may instead be under
> `%LOCALAPPDATA%\Packages\<the GIMP package folder>\LocalCache\Local\sam3-gimp\logs\`.
> From a Command Prompt:
>
> ```
> dir /s /b %LOCALAPPDATA%\Packages\*GIMP*\LocalCache\Local\sam3-gimp\logs\plugin.log
> ```
>
> Or skip the hunt: paste [`tools/diagnose.py`](../tools/diagnose.py) into
> *Filters ▸ Development ▸ Python-Fu ▸ Console*; it prints the log from inside the container.

GIMP itself will not show you plug-in stderr when launched from the Start Menu, which is why the
plug-in writes `plugin.log` — if a menu item appears to do nothing at all, that file is the
first place to look.

For a full picture, open **Filters ▸ Development ▸ Python-Fu ▸ Console** and paste in the
contents of [`tools/diagnose.py`](../tools/diagnose.py). It reports where the plug-in is, which
modules import, what is configured, whether the daemon can be reached, and the tail of every
log — then prints it all in one block to copy into an issue.

### Segmenting hangs, or reports that the daemon would not start

Read `%LOCALAPPDATA%\sam3-gimp\logs\sam3gimpd.log`; the plug-in also surfaces its last lines in
the error dialog. Then, from a normal Command Prompt:

```
%LOCALAPPDATA%\sam3-gimp\venv\Scripts\sam3gimpd.exe doctor
```

That prints a JSON environment report — device, dtype, versions, paths, and whether a live
daemon answered. To see the daemon's own startup errors directly, run it in the foreground:

```
%LOCALAPPDATA%\sam3-gimp\venv\Scripts\sam3gimpd.exe serve --stub
```

`--stub` uses the fake engine: no torch, no weights, instant start. **If the stub serves but
the real engine does not, your problem is torch or the weights, not the plug-in.**

If a stale `runtime.json` is confusing the handshake, delete it — the next invocation will
spawn a fresh daemon. That file being absent is normal.

### Windows Firewall prompts when I first segment

Binding `127.0.0.1` should not prompt. If it does, allow it for **Private** networks only,
or deny it and check whether things still work — loopback traffic normally is not filtered.
Please report it either way.

### A console window flashes every time

That means the daemon was spawned with `python.exe` instead of `pythonw.exe`. It is
cosmetic but it is a bug — please report it with your `sam3gimpd.log`.

### Out of memory on the GPU

SAM 3 is 0.9B parameters, and the plug-in may hold both the concept and the tracker engine.
Whether both fit in 8 GB **is an open question that has not been measured yet** (DESIGN.md
§7). The Doctor tab's *Models loaded* line shows which engines are in memory. Workarounds
today: close other GPU applications; press **Stop daemon** on the Doctor tab, or shorten the
idle timeout, between sessions; or install the CPU build and accept the speed.

### Everything is broken; how do I start over?

Close GIMP, delete `%LOCALAPPDATA%\sam3-gimp\`, reopen GIMP, and run Setup again. That
removes the venv, the weights and all state; the plug-in files themselves are untouched.

### Uninstalling

1. Delete `%APPDATA%\GIMP\<version>\plug-ins\sam3_gimp\` (the plug-in).
2. Delete `%LOCALAPPDATA%\sam3-gimp\` (the venv, the weights, the logs).

Nothing is written to the registry, to Program Files, or anywhere else.

---

## Other platforms

Linux, macOS and CPU-only have not been tested end to end: the code paths exist and the
daemon runs on all of them, but the plug-in has only been run inside GIMP on Windows. If you
want to try, see [DEVELOPING.md](DEVELOPING.md) — `tools/dev_sync.py` resolves the correct
plug-ins directory on Linux (including Flatpak) and macOS.

Setup's install is the same two steps everywhere — PyTorch from the index below, then the
daemon from the plug-in folder with its `[runtime]` libraries from PyPI:

| Host | Where PyTorch comes from | Notes |
|---|---|---|
| Linux + NVIDIA | the cu128 index | Detected via `nvidia-smi` or `/proc/driver/nvidia` |
| Linux + AMD | the rocm6.4 index | Detected via `/dev/kfd` plus `/opt/rocm` or `rocminfo`; x86_64 only. The daemon runs fp16 on ROCm (`SAM3D_DTYPE=bf16` to override on CDNA/RDNA3) |
| Apple Silicon | PyPI | fp16 on MPS |
| Intel Mac | nothing | PyTorch has shipped no macOS x86_64 wheels since 2.2; Setup says so and disables Install. Run the daemon on another machine ([above](#a-daemon-on-another-machine)) |
| Anything else | the cpu index | Slow but correct |

`SAM3_FORCE_DEVICE=cuda|rocm|mps|cpu` overrides the detection for the *install*;
`SAM3D_DEVICE` / `SAM3D_DTYPE` steer the daemon at *run* time.
