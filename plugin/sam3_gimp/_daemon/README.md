# sam3gimpd

A standalone segmentation daemon for **SAM 3**, with a small HTTP API on
loopback. It knows nothing about GIMP.

`sam3gimpd` exists because [GIMP 3 embeds its own Python](../../../DESIGN.md) and will not
load `torch` — so inference has to live in a separate process with its own
environment. Making that process a proper pip-installable package rather than a
private subprocess costs nothing and buys a lot: it is testable and usable
without GIMP, a GPU on another machine can serve it (see below), and any other
host (Krita, a CLI, a web page) can drive it.

## Install

The distribution (`sam3-gimp-daemon`) is not on PyPI, so it is always installed
**from this directory, by path** -- `.` below is `plugin/sam3_gimp/_daemon`.
Never install it by name: the names `sam3d` and `sam3gimpd` on PyPI are not
ours, and `pip install <name>` would fetch and run whatever someone else
publishes under them.

```console
$ pip install .                        # base: zero dependencies, stub engine only
```

A real engine takes two steps, each confined to one index. First torch and
torchvision, from the PyTorch index only; then the daemon from this directory
with its `[runtime]` extra (transformers and friends), from PyPI only:

```console
$ pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128    # NVIDIA
$ pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/rocm6.4  # AMD, Linux x86_64
$ pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu      # CPU only
$ pip install torch==2.9.0 torchvision==0.24.0                                                        # Apple Silicon: PyPI wheels
$ pip install '.[runtime]'
```

One command with `--extra-index-url` would let pip take *any* requirement from
whichever index offers the higher version, which is how a dependency-confusion
attack gets in; two commands with one index each leave no such choice. The
plug-in's Setup dialog runs the same two steps with `uv`:

```console
$ uv pip install --python <py> --index-url https://download.pytorch.org/whl/cu128 torch==2.9.0 torchvision==0.24.0
$ uv pip install --python <py> "<path>/_daemon[runtime]"
```

(Apple Silicon takes torch from PyPI in the first step.) The `[cuda]`, `[rocm]`,
`[cpu]` and `[mps]` extras still exist and pin the same versions, for anyone
who manages indexes some other way.

The **base install has no runtime dependencies at all** and torch is imported
lazily, inside the functions that need it. That is what makes the next section
work on a laptop with no GPU.

## Run it

```console
$ sam3gimpd serve --stub                 # a fake engine; no torch is ever imported
$ sam3gimpd doctor                       # device, dtype, weights, paths, live daemon
$ sam3gimpd download --token hf_...      # fetch the gated facebook/sam3 checkpoint
```

`--stub` is a **product feature, not a test fixture**: it returns deterministic,
structurally valid soft masks with genuine anti-aliased edges, so a client can be
built and exercised end to end on a machine with no GPU and no weights. Every
protocol test in this repository runs against it.

The daemon binds `127.0.0.1` on an ephemeral port and publishes
`{port, token, pid, version, started_at}` in `runtime.json` under the
platform data directory (`$XDG_DATA_HOME/sam3-gimp` on Linux, `0700`; the file
itself is `0600`). It is single-instance via a lockfile, exits when its
`--parent-pid` disappears, and exits after 30 idle minutes to free VRAM.

```console
$ RT="${XDG_DATA_HOME:-$HOME/.local/share}/sam3-gimp/runtime.json"
$ TOKEN=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["token"])' "$RT")
$ PORT=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["port"])' "$RT")
$ curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:$PORT/hello
```

Every request needs that token, and a client should also send an
`X-Sam3-Nonce` on `/hello` and check the `nonce_proof` it gets back: it is how a
client knows the port still belongs to the daemon that wrote `runtime.json`
(`API.md` §2).

## A GPU on another machine

Keep the daemon on its default loopback bind on the GPU machine and reach it
through an SSH tunnel:

```console
gpu-host$ sam3gimpd serve --port 8765 --idle-ttl 0
gimp-pc$  ssh -N -L 8765:127.0.0.1:8765 <gpu-host>
```

Then point the plug-in (*Setup ▸ Install ▸ Advanced ▸ Remote daemon*) at
`http://127.0.0.1:8765`, with the token from the GPU machine's `runtime.json`.
The tunnel encrypts the traffic and the daemon keeps its loopback-only
`Host` guard.

`sam3gimpd serve --host 0.0.0.0` also works, and turns the `Host` guard off,
but then the bearer token and every image travel over the network as plain
HTTP: use it only on a network you trust.

## The API

**[`API.md`](API.md) is the contract** — endpoints, the binary result frame, the
three coordinate spaces, supersession, limits, and the guarantees `--stub` makes.
It is frozen for API major version 1.

The short version: `POST /images` with raw RGB (≤1008 px on the long side) gets an
`image_id`; `POST /images/{id}/text` runs PCS (a noun phrase → *every* matching
instance) and `POST /images/{id}/points` runs PVS (points/box → one instance);
`GET /jobs/{id}` long-polls and returns a binary frame of **cropped soft uint8
masks on the model canvas** (for the reference engines, the uploaded image's own
pixels), which the client thresholds locally with no round trip.

## Two engines, one checkpoint

| Engine | Model | Prompt | Returns |
|---|---|---|---|
| **PCS** | `Sam3Model` + `Sam3Processor` | text, exemplar boxes | every matching instance |
| **PVS** | `Sam3TrackerModel` + `Sam3TrackerProcessor` | points, box | one instance (+ candidates) |

Image encoding is the expensive step and prompting against a cached embedding is
cheap, which is why the daemon is persistent and keeps an LRU of embeddings
keyed by a hash of the pixels.

## Weights

We ship none. `facebook/sam3` is **gated**: accept Meta's terms on the model page,
then `sam3gimpd download --token ...`, or point the daemon at a checkpoint you
already have. The download cache lives under the data directory's `hf/` so it
can be inspected and cleared.

A checkpoint already on disk is found, in this order, at `$SAM3_WEIGHTS_DIR`;
at the directory the plug-in's Setup recorded in `<base>/weights.json`; and
under `<base>/models/`. A directory counts only if it holds a `config.json`, and
none of this ever touches the network.

## License

MIT for this code. The SAM 3 **weights** are covered by their own license; review
it before any commercial use.
