# sam3gimpd HTTP API — contract v1.1

**Status:** frozen for API major version 1. Additive minor changes are allowed
(new optional fields, new capability strings); anything that removes or
re-types a field is a major bump. 1.1 added the `/hello` identity proof
(`X-Sam3-Nonce` → `nonce_proof`, §2).

This document is the **contract between the two halves of the project**. The
GIMP plug-in (stdlib + `gi` only) and the `sam3gimpd` daemon (torch, lazily) are
developed independently and agree on nothing except what is written here. Where
this document and `DESIGN.md` differ in detail, this document wins for
wire-level questions.

Reference implementations of every JSON shape live in `sam3gimpd/types.py`,
beside this file (`plugin/sam3_gimp/_daemon/`). The plug-in **must not import
it** (zero third-party dependency rule, different Python), but it is the
authority on field names and types; mirror it.

---

## 0. Table of contents

1. [Transport](#1-transport)
2. [Authentication](#2-authentication)
3. [`runtime.json` and the find-or-spawn handshake](#3-runtimejson-and-the-find-or-spawn-handshake)
4. [Error envelope](#4-error-envelope)
5. [Coordinate spaces](#5-coordinate-spaces)
6. [Endpoints](#6-endpoints)
7. [The image upload binary layout](#7-the-image-upload-binary-layout)
8. [The mask return format](#8-the-mask-return-format)
9. [Mapping a mask back onto the original image](#9-mapping-a-mask-back-onto-the-original-image)
10. [Request ids, the queue and supersession](#10-request-ids-the-queue-and-supersession)
11. [Jobs and progress](#11-jobs-and-progress)
12. [Worked end-to-end example](#12-worked-end-to-end-example)
13. [Daemon invocation contract](#13-daemon-invocation-contract)
14. [Stub engine guarantees](#14-stub-engine-guarantees)
15. [Limits](#15-limits)
16. [Stable guarantees for client authors](#16-stable-guarantees-for-client-authors)

---

## 1. Transport

* **HTTP/1.1** over TCP, bound to **`127.0.0.1`** on an **ephemeral port**
  (the daemon binds port `0` and reports what the OS gave it). A daemon on a
  remote GPU is best left on its loopback bind and reached through an SSH
  tunnel; `sam3gimpd serve --host 0.0.0.0` exists for trusted networks only,
  because the token and every image then cross the network as plain HTTP.
* **The server MUST be concurrent** — `ThreadingHTTPServer` or equivalent, at
  least 8 simultaneous connections. A client long-polls `GET /jobs/{id}` on one
  connection while issuing `POST` prompts on another; a single-threaded server
  would deadlock the UI. This is a hard requirement, not an optimisation.
  Concurrency is at the *connection* level only — inference itself is strictly
  serialised (§10).
* **Connections are bounded.** At most **32** are served at once; one beyond
  that is closed as soon as it is accepted, with no response. A new connection
  must begin its first request within **10 s**, and a keep-alive connection
  with no request for **120 s** is closed. Once a request has begun, its
  request line and headers must all arrive within **10 s** in total, and every
  read and write has **15 s**. A request refused with `401` or `403` closes its
  connection. A local process that opens connections and never speaks, or
  speaks a byte at a time, therefore cannot hold a thread for as long as it
  likes.
* **Keep-alive** is supported and expected (`Connection: keep-alive`). The
  plug-in reuses one `http.client.HTTPConnection` per worker thread. Because the
  daemon closes idle connections, a client must retry a request **once** on a
  fresh socket when a reused one turns out to be closed.
* **Every response carries `Content-Length`.** No chunked transfer encoding, in
  either direction. `http.client` handles chunked fine, but fixed lengths make
  the binary payloads trivially sliceable and keep the plug-in simple.
* **Request bodies must carry `Content-Length`.** Chunked request bodies are
  rejected with `400 bad_request`. A body the endpoint does not use (on a
  `GET` or `DELETE`, or after an error) is read and discarded up to 64 KiB so
  the connection stays usable; anything larger closes the connection after the
  response (`Connection: close`).
* Media types:

  | Purpose | `Content-Type` |
  |---|---|
  | all JSON | `application/json; charset=utf-8` |
  | image upload (§7) | `application/octet-stream` |
  | binary result frame (§8) | `application/vnd.sam3.result+binary` |

* Response headers present on **every** response:

  | Header | Value |
  |---|---|
  | `X-Sam3-Api` | the API version, e.g. `1.1` |
  | `X-Sam3-Version` | the `sam3gimpd` package version, e.g. `0.1.1` |

* All text is UTF-8. All JSON numbers are IEEE doubles; fields documented as
  integers must be sent without a fractional part. Every number must be
  **finite**: `NaN`, `Infinity` and anything that overflows a double (`1e400`,
  a 400-digit integer) are `400 bad_request`.
* All multi-byte integers in binary payloads are **little-endian**.

---

## 2. Authentication

Every endpoint, including `GET /hello`, requires:

```
Authorization: Bearer <token>
```

* The token is generated once per daemon process with
  `secrets.token_urlsafe(32)` (43 URL-safe characters) and published in
  `runtime.json`.
* The scheme is case-insensitive (`bearer` works too). Comparison is
  constant-time (`hmac.compare_digest`) and done on bytes, so a header carrying
  non-ASCII is simply a wrong token.
* A missing, malformed or wrong token → **`401 unauthorized`** with the standard
  error envelope. No `WWW-Authenticate` challenge is sent (there is no
  interactive login). Refusals are logged at DEBUG only, never with a
  traceback, and never become `/status`'s `last_error`.
* **DNS-rebinding guard:** the daemon rejects any request whose `Host` header
  is not exactly `localhost` or a loopback IP literal (`127.0.0.0/8`, or `::1`
  in brackets), with an optional numeric `:port`, with **`403 forbidden_host`**
  — unless it was started with an explicit non-loopback `--host`, in which case
  the check is skipped. Nothing is resolved and nothing is matched by prefix:
  `127.attacker.example` and `127.0.0.1.evil.example` are names their owner can
  point anywhere, which is the whole of a rebinding attack. This costs nothing
  and stops a web page in the user's browser from driving the daemon.
* Only a request that passes both checks counts as activity for the idle TTL
  (§13): something knocking on the port cannot keep the daemon alive.
* The token is a *capability*, not a secret to be logged: the daemon must never
  write it to `sam3gimpd.log`, and clients must never put it in a URL.

**Identity proof (API 1.1).** `runtime.json` names a port and a token, but a
file left behind by a daemon that has since died names a port anything may
hold by now — including a process belonging to another local user, waiting for
a client to hand it prompts and images. The token alone cannot tell the
difference, because the client is the one sending it. So on `GET /hello` the
client sends a fresh nonce:

```
X-Sam3-Nonce: <secrets.token_urlsafe(24)>        # [A-Za-z0-9_-], 16-128 chars
```

and the daemon answers with a MAC under the token it minted, which no other
process can compute:

```
nonce_proof = hex(HMAC-SHA256(key = token (UTF-8),
                              msg = b"sam3gimpd-hello:" + nonce (ASCII)))
```

The client checks it with `hmac.compare_digest` and does not use a daemon whose
`/hello` lacks the field or carries a wrong one. A missing or malformed header
is not an error: the field is simply omitted. `/hello` still requires the
bearer token, and only `/hello` answers a nonce.

---

## 3. `runtime.json` and the find-or-spawn handshake

### 3.1 Location

Written to `<base>/runtime.json`, where `<base>` is:

| Platform | `<base>` |
|---|---|
| Windows | `%LOCALAPPDATA%\sam3-gimp` (fallback `%APPDATA%\sam3-gimp`, then `~\AppData\Local\sam3-gimp`) |
| macOS | `~/Library/Application Support/sam3-gimp` |
| Linux / other | `$XDG_DATA_HOME/sam3-gimp`, default `~/.local/share/sam3-gimp` |

Overrides, honoured identically by the daemon and by the plug-in launcher:

* `SAM3_GIMP_HOME` — replaces `<base>` entirely.
* `SAM3D_RUNTIME_FILE` — replaces the full path of `runtime.json` only.

The canonical implementation is `sam3gimpd.paths` (`base_dir()`,
`runtime_file()`); the plug-in's `launcher.py` mirrors it in stdlib.

### 3.2 Contents

Exactly these five fields are **required** and frozen for API 1.x:

| Field | Type | Meaning |
|---|---|---|
| `port` | int | TCP port on `127.0.0.1` |
| `token` | string | bearer token, 43 URL-safe chars |
| `pid` | int | the daemon's process id |
| `version` | string | the **`sam3gimpd` package** version (informational — compatibility is decided by `GET /hello`, never by this field) |
| `started_at` | float | Unix epoch seconds, when the daemon began serving |

Optional, documented, and safe to ignore: `api_version` (string), `host`
(string, default `"127.0.0.1"`), `log_path` (string). **Readers must ignore
unknown keys** rather than failing.

```json
{
  "api_version": "1.1",
  "host": "127.0.0.1",
  "log_path": "/home/you/.local/share/sam3-gimp/logs/sam3gimpd.log",
  "pid": 48211,
  "port": 41573,
  "started_at": 1772395551.812,
  "token": "Vv3nQ2c9m8UuI1Nn0pS7bYy4LxKkD6ZzRrE5TtWq2Ab",
  "version": "0.1.1"
}
```

A `host` of `0.0.0.0` or `::` (a remote-GPU bind) is where the daemon listens,
not an address to dial: a local client connects to `127.0.0.1` or `::1`.

Writing rules (daemon side):

* Written **atomically**: temp file in the same directory, `fsync`, then
  `os.replace`. A client polling the file must never see a partial write. On
  Windows the replace fails while a reader has the file open; the daemon
  retries for up to half a second rather than leave the old port and token.
* Mode `0600` on POSIX, in a `<base>` directory that is `0700` (the daemon
  creates its directories `0700`, tightens an existing `<base>` to it, and
  writes every file there -- logs included -- `0600`). On Windows the file is
  user-scoped by living under `%LOCALAPPDATA%`.
* Written **after** the socket is listening, so its existence means "connectable".
* Deleted on clean shutdown. A stale file after a crash is normal and the
  handshake below is designed for it.

### 3.3 Find-or-spawn algorithm (the client's exact behaviour)

```
1.  info = read_json(runtime_file())
    If missing, empty, unparseable, or lacking any required field  -> goto SPAWN.

2.  If not pid_alive(info.pid)                                     -> delete file, goto SPAWN.
      POSIX  : os.kill(pid, 0), treating EPERM as "alive".
      Windows: ctypes OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
               NULL with ERROR_INVALID_PARAMETER (no such process) -> dead,
               any other NULL (e.g. ACCESS_DENIED) -> alive;
               otherwise GetExitCodeProcess must still read STILL_ACTIVE (259) --
               OpenProcess succeeds for an exited process while anything holds
               a handle to it. (ctypes is stdlib, so the plug-in may use it.)
    If the pid belongs to another user                              -> delete file, goto SPAWN.
      A daemon this user spawned always runs as this user; the port of one
      that does not is never contacted.

3.  GET http://127.0.0.1:{info.port}/hello       (0.0.0.0 / :: are dialled as loopback)
      Authorization: Bearer {info.token}
      X-Sam3-Nonce: {a fresh nonce}                                 (§2)
      3.0 s for the connect and the answer together.  (Windows takes about 2 s to
      refuse a connection to a closed local port, so a shorter connect timeout
      would turn that refusal into a timeout.)
    No answer yet (a timeout connecting or reading), and the pid    -> wait and retry, up to the
      is verifiably a sam3gimpd of ours                                spawn timeout; never delete.
    ConnectionRefused / 401 / not a daemon of ours                  -> delete file, goto SPAWN.
    nonce_proof missing and api_version 1.0 (an older build)        -> POST /shutdown (best
                                                                       effort), delete, goto SPAWN.
    nonce_proof wrong, or missing from a 1.1+ daemon                -> never used, never shut
                                                                       down or signalled (§2).

4.  If hello.api_version major != client's API major               -> POST /shutdown (best effort),
                                                                      delete file, goto SPAWN.

5.  ACCEPTED. Cache (port, token) for the life of the plug-in process.

SPAWN:
6.  Take the spawn lock (see 3.4). Record t0 = time().
7.  Launch the daemon detached (see §13). Never wait on it, never read its pipes.
8.  Poll every 100 ms for runtime.json with mtime >= t0 - 1.0 and a parseable body,
    up to 60 s (a first start imports torch cold).
    If the child exits 0 ("another instance holds the lock", §3.4)  -> keep polling for the
                                                                       holder's runtime.json until
                                                                       the same deadline.
    On timeout -> surface the last 40 lines of logs/sam3gimpd.log to the user and fail.
9.  Go to step 3. Allow at most 2 SPAWN attempts per plug-in invocation.
```

Every "delete" is a compare-and-delete: `runtime.json` is removed only if it
still holds the pid and token that were judged stale, so a file a newer daemon
has just written is never lost.

A lock holder that never serves is ended (SIGTERM, TerminateProcess on Windows)
only when all of these hold: the lock file names its pid; the process is a
`sam3gimpd serve` running as this user (checked from its command line);
it was created no later than the lock file was written; it has held the lock
for longer than a cold start can take (120 s); and it does not answer `/hello`
with a valid proof. Anything less and the client waits or gives up instead --
a pid it cannot prove is its own daemon is never signalled.

### 3.4 Single instance

The daemon holds `<base>/sam3gimpd.lock` for its lifetime (`msvcrt.locking` on
Windows, `fcntl.flock(LOCK_EX | LOCK_NB)` on POSIX). A second daemon that
cannot take the lock **exits 0 quietly** without touching `runtime.json` — the
holder's file stays authoritative. Two plug-in invocations racing to spawn
therefore converge on one daemon, which is why step 9 loops back to step 3
rather than failing.

While the lock is held the file contains the holder's pid (at offset 0; on
Windows the locked byte is 1 MiB further on, so the pid stays readable). A
clean exit empties the file before releasing the lock, so a pid read from it
names a live holder or one that crashed -- never one that exited normally and
whose number may since have been reused.

---

## 4. Error envelope

Every non-2xx response — from every endpoint, including auth failures and
404s — has this body and no other:

```json
{
  "error": {
    "code": "image_not_found",
    "message": "no cached image with that id",
    "detail": { "image_id": "9f0c…" }
  }
}
```

* `code` — machine-readable, from the closed list below. Clients switch on this.
* `message` — one short human sentence. Never contains the token.
* `detail` — always an object, possibly `{}`. Free-form, additive.

**A failed *job* is not an HTTP error.** `GET /jobs/{id}` returns `200` with
`"state": "failed"` and the same `{code, message, detail}` object under
`"error"`. Non-2xx means the *request* failed; the job envelope means the
*inference* failed.

| `code` | HTTP | When |
|---|---|---|
| `bad_request` | 400 | malformed request the other codes do not cover, including any non-finite number |
| `invalid_json` | 400 | body is not valid JSON, or not a JSON object |
| `missing_header` | 400 | a required header (`X-Width`, `X-Height`, `Content-Length`) is absent |
| `bad_dimensions` | 400 | width/height not integers, or outside `[16, 1008]` |
| `payload_size_mismatch` | 400 | body length ≠ `width * height * 3` |
| `version_mismatch` | 400 | client sent `X-Sam3-Api` with an incompatible major |
| `unauthorized` | 401 | missing/wrong bearer token |
| `forbidden_host` | 403 | `Host` header is not loopback |
| `not_found` | 404 | unknown path |
| `image_not_found` | 404 | `{image_id}` is not in the cache (never uploaded, evicted, or deleted). Also a *job* error, should an image leave the cache before a prompt for it ran. Either way: re-upload and retry once. |
| `job_not_found` | 404 | `{job_id}` unknown or its retention window expired |
| `method_not_allowed` | 405 | wrong verb on a known path |
| `image_not_ready` | 409 | the image's encode job failed; `detail.error` carries the original failure. Also a *job* error for a prompt that was queued behind an encode that then failed. |
| `payload_too_large` | 413 | body exceeds the limit in §15 |
| `unsupported_media_type` | 415 | wrong `Content-Type` for the endpoint |
| `model_load_failed` | 500 | checkpoint present but would not load (also used as a *job* error) |
| `inference_failed` | 500 | job error only: the forward pass raised |
| `internal_error` | 500 | unhandled exception; `detail.trace_id` correlates with `sam3gimpd.log` |
| `engine_unavailable` | 503 | torch/transformers missing, or the gated weights are absent, and the daemon is not in `--stub` mode |
| `queue_full` | 503 | job queue at capacity (§15) |
| `shutting_down` | 503 | `POST /shutdown` has been accepted; the daemon takes no new work. Also the `error` of a job cancelled because the daemon shut down before it started. |

---

## 5. Coordinate spaces

Three spaces exist. Confusing them is the single most likely integration bug,
so they are named here and used with those names everywhere below.

| Space | Origin | Units | Who uses it |
|---|---|---|---|
| **original-image** | top-left | pixels of the user's GIMP image (e.g. 3000×2000) | the client, privately |
| **uploaded-image** | top-left | pixels of what was sent to `POST /images` (≤1008 on the long side) | **all client → daemon coordinates** |
| **model-canvas** | top-left | pixels of the space masks are returned in -- with the reference engines, the uploaded image itself | **all daemon → client mask geometry** |

Rules:

* **Every coordinate a client sends** — points, boxes — is in
  **uploaded-image** space, as a float, y down. Clients never compute canvas
  coordinates for input; the daemon converts.
* **Every coordinate a client receives** — instance bboxes — is in
  **model-canvas** space, as an integer, y down, half-open `[x0, x1) × [y0, y1)`.
* The daemon reports the exact affine transform between them:

  ```
  canvas_x = image_x * scale_x + offset_x
  canvas_y = image_y * scale_y + offset_y
  ```

  as `canvas_from_image: {scale_x, scale_y, offset_x, offset_y}`, on both
  `POST /images` and every result header.
* **Clients MUST use the reported transform and MUST NOT hardcode it.** The
  reference policy is the **identity**: masks are post-processed with
  `target_sizes` equal to the uploaded image (the reference example's
  `original_sizes`), so `Sam3Processor` undoes its own resize and padding and
  the canvas *is* the uploaded image.
  This replaced an "anisotropic squash onto 1008×1008" policy that the
  processor never actually performed — SAM 3 resizes preserving aspect ratio
  and pads — so on any non-square image every mask came back stretched and
  landed beside its object. A client that hardcodes either policy will
  misplace masks the moment it changes again.
* Mapping from original-image to uploaded-image is the **client's** business —
  the client chose the downscale factor. The daemon never sees the original size
  unless the client passes the optional `X-Source-Width` / `X-Source-Height`
  headers, which it only echoes back for diagnostics.

---

## 6. Endpoints

| Method | Path | One line |
|---|---|---|
| `GET` | `/hello` | Cheap version/capability handshake. Never loads a model. |
| `POST` | `/images` | Upload raw RGB pixels; returns an `image_id` and an encode job. |
| `POST` | `/images/{image_id}/text` | PCS: a noun phrase → every matching instance. |
| `POST` | `/images/{image_id}/points` | PVS: points/box → one instance (+ multimask candidates). |
| `GET` | `/jobs/{job_id}` | Poll or long-poll a job; returns the binary result frame when done. |
| `DELETE` | `/images/{image_id}` | Drop the cached embedding and cancel its queued jobs. |
| `GET` | `/status` | Everything the Doctor panel shows: devices, cache, queue, paths, last error. |
| `POST` | `/shutdown` | Ask the daemon to exit. |

Unknown paths → `404 not_found`. Known path, wrong verb → `405
method_not_allowed` with an `Allow` header.

---

### 6.1 `GET /hello`

Handshake. Must answer in **milliseconds** and must **never** trigger a model
load, a weights download or any network access — the plug-in calls it on the
critical path of every invocation. Nor does it wait for one: a cold model load
holds the engine for a minute or more, and if the engine cannot describe
itself within 250 ms the daemon answers from its last description.

**Request**

```
GET /hello HTTP/1.1
Host: 127.0.0.1:41573
Authorization: Bearer <token>
X-Sam3-Nonce: 3k5Jq0bW1xYf9ZrN2c8HdTg4uVpLm6Ae      (optional; §2)
```

**`200` response**

```json
{
  "api_version": "1.1",
  "sam3d_version": "0.1.1",
  "build": "5e1f0a9c",
  "engine_mode": "stub",
  "device": "cpu",
  "dtype": "float32",
  "capabilities": ["pcs", "pvs"],
  "torch_available": false,
  "weights_available": false,
  "model_canvas": { "width": 1008, "height": 1008 },
  "pid": 48211,
  "started_at": 1772395551.812,
  "uptime_s": 93.4,
  "limits": {
    "max_image_side": 1008,
    "max_upload_bytes": 3048192,
    "max_text_chars": 512,
    "max_points": 64,
    "max_instances": 256,
    "max_long_poll_seconds": 30.0
  },
  "nonce_proof": "9b0c3f…64 hex digits…e21d"
}
```

| Field | Type | Notes |
|---|---|---|
| `api_version` | string | `"MAJOR.MINOR"`. The **only** compatibility check clients perform: majors must be equal. |
| `sam3d_version` | string | package version |
| `build` | string | eight hex digits hashing the daemon's own `.py` files (`sam3gimpd.build_hash()`), taken when the daemon started. Optional and additive; a client that ships a copy of the package can tell a stale daemon from a current one without waiting for a version bump. |
| `nonce_proof` | string | 64 lowercase hex digits, present only when the request carried a well-formed `X-Sam3-Nonce`: `HMAC-SHA256(token, "sam3gimpd-hello:" + nonce)` (§2). |
| `engine_mode` | string | `"stub"` or `"torch"` |
| `device` | string | `"stub"`, `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"` -- or `"auto"` on a torch daemon whose first prompt has not yet probed torch (answering `/hello` never imports it), unless `--device` named one |
| `dtype` | string | `"float32"`, `"bfloat16"`, `"float16"`, `"none"` -- or `"auto"`, as for `device` |
| `capabilities` | string[] | subset of `"pcs"`, `"pvs"`, `"exemplar_boxes"`. Clients must feature-check rather than assume. (Contour tracing is client-side: masks travel as soft uint8 and are re-thresholded locally, so the daemon returns no polygons.) |
| `torch_available` | bool | torch importable in the daemon's environment |
| `weights_available` | bool | the checkpoint is present locally (no network probe) |
| `model_canvas` | object | the engine's nominal working resolution; the canvas masks are actually returned in is per image and comes from `POST /images` |
| `pid`, `started_at`, `uptime_s` | int/float/float | lifecycle |
| `limits` | object | mirrors §15 so a client can adapt without a version bump |

A client may send `X-Sam3-Api: 1.1`; a major mismatch is answered with
`400 version_mismatch` instead of the body above.

---

### 6.2 `POST /images`

Uploads pixels, hashes them, and enqueues the (slow) vision-encoder pass. The
response is immediate — encoding happens in the job queue.

**Request**

```
POST /images HTTP/1.1
Authorization: Bearer <token>
Content-Type: application/octet-stream
Content-Length: 2032128
X-Width: 1008
X-Height: 672
X-Source-Width: 3000        (optional, diagnostics only)
X-Source-Height: 2000       (optional, diagnostics only)

<raw RGB bytes, see §7>
```

* `X-Width` / `X-Height` are **required**; absent → `400 missing_header`.
* `Content-Length` **must** equal `X-Width * X-Height * 3`; otherwise
  `400 payload_size_mismatch` with `detail = {expected, got}`.
* Both sides must be in `[16, 1008]`; otherwise `400 bad_dimensions`.
* Wrong `Content-Type` → `415 unsupported_media_type`.

**`202` response**

```json
{
  "image_id": "5b1d3d3fd6a34c2f8f8d3c9a41b7e0d2",
  "job_id": "j-000001-a91f",
  "cached": false,
  "state": "queued",
  "image": { "width": 1008, "height": 672 },
  "model_canvas": { "width": 1008, "height": 672 },
  "canvas_from_image": { "scale_x": 1.0, "scale_y": 1.0, "offset_x": 0.0, "offset_y": 0.0 }
}
```

* `image_id` — 32 lowercase hex characters. It is the `blake2b` digest of the
  pixel bytes plus the dimensions, so **uploading identical pixels always yields
  the same id**. Clients must nonetheless treat it as opaque.
* `cached` — `true` when the image was already in the cache. No work is queued,
  and `job_id` names the image's existing encode job -- usually `done`, but
  still `queued` or `running` if the same pixels were uploaded moments ago;
  `state` says which.
* `model_canvas` and `canvas_from_image` are computed deterministically from the
  dimensions; they are available before the encode job runs, so a client can do
  its geometry setup immediately.
* `state` is `"queued"`, `"running"` or `"done"`.

Clients **may issue prompts immediately** without waiting for the encode job;
the queue guarantees ordering. If the encode job later fails, subsequent prompts
for that image return `409 image_not_ready`, and a prompt already queued behind
it fails with the same code as its job error. Re-uploading the same pixels
retries the encode (`cached: false`, a new `job_id`).

Cache eviction is LRU with a default capacity of 3 images. An image with work
pending -- its encode or a prompt queued or running -- is never evicted; the
cache runs over capacity while such work is outstanding and shrinks back as it
settles. A prompt against an evicted id returns `404 image_not_found`; the
correct client response is to re-`POST /images` and retry once.

When the job queue is full (§15) an upload that needs an encode is refused
with `503 queue_full` before it touches the cache, so it never evicts another
image on its way to being turned away. A cached upload needs no job and is
answered as usual.

---

### 6.3 `POST /images/{image_id}/text` — PCS

One noun phrase → **every** matching instance.

**Request**

```
POST /images/5b1d…/text HTTP/1.1
Authorization: Bearer <token>
Content-Type: application/json
```

```json
{
  "request_id": "r-000007",
  "text": "yellow school bus",
  "score_threshold": 0.02,
  "max_instances": 64,
  "boxes": [
    { "box": [120.0, 300.5, 400.0, 520.0], "label": 1 },
    { "box": [800.0, 10.0, 900.0, 90.0], "label": 0 }
  ]
}
```

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `request_id` | string | **yes** | — | client-generated, 1–64 chars of `[A-Za-z0-9._:-]`. Echoed everywhere. See §10. |
| `text` | string | **yes** | — | 1–512 chars. SAM 3 wants a *simple noun phrase* (`"red car"`), not a relational description (`"the car on the left"`). The daemon does not rewrite it. |
| `score_threshold` | float | no | `0.02` | `[0.0, 1.0]`. Deliberately low: the client filters locally with a slider, so a round trip per slider tick is unnecessary. This value is a **hard ceiling** on what that slider can reveal -- anything below it is dropped here and never reaches the client -- so it sits well under any sensible user-facing default. |
| `max_instances` | int | no | `64` | `[1, 256]`; the daemon keeps the highest-scoring N and sets `truncated: true` in the result header if it dropped any. |
| `boxes` | array | no | `[]` | Exemplar boxes in **uploaded-image** coordinates, `[x0, y0, x1, y1]` finite floats, half-open, `x1 > x0`, `y1 > y0`. `label` is the integer `1` (positive, the default) or `0` (negative); anything else is `400 bad_request`. Requires the `exemplar_boxes` capability; otherwise ignored and reported in `result.prompt.ignored` as `"boxes"`. A negative box is used only when the processor accepts box labels -- sent unlabelled it would be read as positive and select exactly what was excluded -- and is otherwise dropped and reported as `"boxes:label=0"`. The capability is confirmed on the first text prompt, so `/hello` may drop it after that if the processor takes no boxes. |

**`202` response**

```json
{
  "job_id": "j-000002-77c1",
  "request_id": "r-000007",
  "image_id": "5b1d3d3fd6a34c2f8f8d3c9a41b7e0d2",
  "engine": "pcs",
  "state": "queued",
  "superseded_job_ids": ["j-000001-b330"]
}
```

`superseded_job_ids` lists jobs this request displaced (§10). Errors: `404
image_not_found`, `409 image_not_ready`, `400 bad_request` (empty `text`,
missing `request_id`), `503 queue_full`, `503 engine_unavailable`.

---

### 6.4 `POST /images/{image_id}/points` — PVS

Points and/or a box → one instance, optionally with alternative candidates.

```json
{
  "request_id": "r-000008",
  "points": [
    { "x": 512.0, "y": 300.0, "label": 1 },
    { "x": 640.5, "y": 410.0, "label": 0 }
  ],
  "box": [400.0, 250.0, 700.0, 480.0],
  "multimask": true,
  "max_instances": 3
}
```

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `request_id` | string | **yes** | — | as above |
| `points` | array | no | `[]` | `{x, y, label}` in **uploaded-image** coordinates, finite. `label`: `1` include (the default), `0` exclude. Max 64. |
| `box` | `[x0,y0,x1,y1]` | no | `null` | uploaded-image finite floats, half-open, at least 1 px on each side (`x1 - x0 >= 1`, `y1 - y0 >= 1`); a smaller box is degenerate and `400 bad_request` |
| `multimask` | bool | no | `true` | when `true` the model's alternative candidates are returned, best first |
| `max_instances` | int | no | `3` | `[1, 8]` |

At least one of `points` or `box` must be non-empty, else `400 bad_request`.
Response is the same `202` shape as §6.3 with `"engine": "pvs"`.

Successive point prompts on the same `image_id` reuse the cached embedding, so
a refine costs milliseconds, not seconds. Clients send the **complete** point
set each time; the API is stateless per request.

---

### 6.5 `GET /jobs/{job_id}`

The one endpoint with two response media types.

**Query parameters**

| Param | Type | Default | Meaning |
|---|---|---|---|
| `wait` | float seconds, `[0, 30]` | `0` | long-poll budget (§11); outside the range it is clamped, `nan` is `400 bad_request` |
| `meta` | `0`/`1` | `0` | force the JSON form even when the job is `done` |

**Responses**

| Job state | `meta=0` (default) | `meta=1` |
|---|---|---|
| `queued`, `running` | `200` JSON `JobStatus` | same |
| `failed`, `superseded`, `cancelled` | `200` JSON `JobStatus` | same |
| `done` | `200` **`application/vnd.sam3.result+binary`** (§8) | `200` JSON `JobStatus` with `result` populated and `masks_available: true` |

An **encode** job (`engine: "encode"`, the `job_id` from `POST /images`) has no
frame: when `done` it answers with the JSON `JobStatus` even without `meta=1`,
with `masks_available: false` and no `result`. Clients poll it for progress
and state, never for masks.

Unknown or expired id → `404 job_not_found`.

The binary response additionally carries:

```
Content-Type: application/vnd.sam3.result+binary
X-Sam3-Job-State: done
X-Sam3-Header-Length: 792
X-Sam3-Request-Id: r-000007
```

`X-Sam3-Header-Length` duplicates the uint32 in the frame so a client can size
its read without parsing; the in-band value is authoritative.

**`JobStatus` JSON**

```json
{
  "job_id": "j-000002-77c1",
  "state": "running",
  "engine": "pcs",
  "image_id": "5b1d3d3fd6a34c2f8f8d3c9a41b7e0d2",
  "request_id": "r-000007",
  "progress": 0.45,
  "stage": "decoding",
  "created_at": 1772395601.114,
  "started_at": 1772395601.130,
  "finished_at": null,
  "elapsed_ms": 214.0,
  "queue_position": null,
  "superseded_by": null,
  "masks_available": false,
  "error": null
}
```

| Field | Type | Notes |
|---|---|---|
| `state` | string | `queued` \| `running` \| `done` \| `failed` \| `superseded` \| `cancelled` |
| `progress` | float | `[0.0, 1.0]`, monotonically non-decreasing within a job |
| `stage` | string | free-form but stable: `queued`, `encoding`, `prompting`, `decoding`, `packing`, `done`, `failed` |
| `queue_position` | int \| null | `0` = next to run; `null` unless `state == "queued"` |
| `superseded_by` | string \| null | the newer `job_id` that displaced this one |
| `masks_available` | bool | `true` only when `state == "done"` and the frame is still retained |
| `error` | object \| null | `{code, message, detail}` when `state == "failed"`; also on a job `cancelled` because the daemon shut down before it started (`code: "shutting_down"`) |
| `result` | object \| null | present only with `meta=1` and `state == "done"`; identical to the frame's JSON header (§8) |

Finished jobs, frame bytes included, are retained for **at least 120 s** after
completion, and the **64** most recent are kept whatever their age; past 512
retained jobs the oldest go regardless. After that the id yields
`404 job_not_found`, and a client that polls slower than that must re-prompt.
An image's encode job stays fetchable for as long as the image is cached, so a
`cached: true` upload always names a job that exists.

---

### 6.6 `DELETE /images/{image_id}`

Frees the cached embedding immediately rather than waiting for LRU eviction.

**`200` response**

```json
{
  "image_id": "5b1d3d3fd6a34c2f8f8d3c9a41b7e0d2",
  "deleted": true,
  "cancelled_job_ids": ["j-000009-1a2b"]
}
```

Queued jobs for that image move to `cancelled`. A **running** job is *not*
interrupted — it finishes and its result is still fetchable; the embedding it
is using is released when it finishes. The id stops resolving at once either
way, and uploading the same pixels again makes a fresh entry. Unknown id →
`404 image_not_found`.

---

### 6.7 `GET /status`

A superset of `/hello` (less `nonce_proof`), for the Doctor panel. Also cheap
and model-free, and like `/hello` it never waits on a model load.

```json
{
  "api_version": "1.1",
  "sam3d_version": "0.1.1",
  "build": "5e1f0a9c",
  "engine_mode": "stub",
  "device": "cpu",
  "dtype": "float32",
  "capabilities": ["pcs", "pvs"],
  "torch_available": false,
  "weights_available": false,
  "model_canvas": { "width": 1008, "height": 1008 },
  "pid": 48211,
  "started_at": 1772395551.812,
  "uptime_s": 412.9,
  "limits": { "max_image_side": 1008, "max_upload_bytes": 3048192 },

  "images": [
    {
      "image_id": "5b1d3d3fd6a34c2f8f8d3c9a41b7e0d2",
      "width": 1008, "height": 672,
      "created_at": 1772395601.0, "last_used_at": 1772395702.3,
      "state": "done", "bytes_estimate": 12582912
    }
  ],
  "jobs_running": 0,
  "jobs_queued": 0,
  "jobs_total": 12,
  "cache_limit": 3,
  "idle_ttl_s": 1800.0,
  "idle_seconds": 41.2,
  "parent_pid": 4711,
  "models_loaded": [],
  "last_error": null,
  "paths": {
    "base": "/home/you/.local/share/sam3-gimp",
    "runtime_file": "/home/you/.local/share/sam3-gimp/runtime.json",
    "server_log": "/home/you/.local/share/sam3-gimp/logs/sam3gimpd.log",
    "crash_log": "/home/you/.local/share/sam3-gimp/logs/crash.log",
    "hf_home": "/home/you/.local/share/sam3-gimp/hf",
    "lock_file": "/home/you/.local/share/sam3-gimp/sam3gimpd.lock"
  },
  "memory": { "rss_bytes": 84213760, "vram_total": null, "vram_free": null,
              "cache_bytes_estimate": 12582912 }
}
```

`last_error` is the most recent job or internal failure as a
`{code, message, detail, at}` object, or `null`. `models_loaded` lists engine
names currently resident (`"pcs"`, `"pvs"`) — useful for answering the VRAM
duplication question in `DESIGN.md` §7. `memory` carries the engine's own
figures when it has them: on CUDA `vram_total` and `vram_free` are numbers, and
`vram_reserved` / `vram_allocated` may appear; without CUDA they are `null`.

---

### 6.8 `POST /shutdown`

```json
{ "grace_ms": 500 }
```

`grace_ms` is optional, a finite number clamped to `[0, 10000]`, default `500`.
An empty body is legal.

**`202` response**

```json
{ "ok": true, "pid": 48211, "grace_ms": 500 }
```

The daemon then: stops accepting new work (every subsequent request →
`503 shutting_down`), cancels every queued job at once (state `cancelled`,
`error.code` `shutting_down`; none of them starts), lets the running job finish
or abandons it after `grace_ms`, deletes `runtime.json`, empties and releases
the lockfile, and only then tears the engine down and exits `0`.

The daemon also exits on its own when the parent pid passed via `--parent-pid`
disappears, or after the idle TTL (default 1800 s since the last authenticated
request, and never while a job is queued or running) — see §13.

---

## 7. The image upload binary layout

The body of `POST /images` is **raw pixels and nothing else**. No header, no
magic, no compression, no container.

```
byte 0                                        byte W*H*3 - 1
+-------------------------------------------------------------+
| R G B | R G B | R G B | ...                                  |
+-------------------------------------------------------------+
  px(0,0) px(1,0) px(2,0) ...
```

* **Element type:** `uint8`, one byte per channel, three channels per pixel.
  There is no byte-order question for a single byte; the *channel* order is
  fixed as **R, G, B**.
* **Row order:** **top-to-bottom**. Row `0` is the top row of the image, exactly
  as GIMP's `Gegl.Buffer.get()` yields it.
* **Column order:** left-to-right within a row.
* **Stride:** `width * 3` bytes. **No row padding, no alignment.** Total length
  is exactly `width * height * 3`.
* **No alpha.** The client composites onto its chosen background before sending.
  GIMP's `babl` does this for free by requesting the format `"R'G'B' u8"`.
* **Colour space:** whatever `"R'G'B' u8"` gives — sRGB-ish, non-linear
  ("primed"). The daemon does not colour-manage; it feeds the bytes to the
  processor as an 8-bit RGB image.
* **Size discipline:** the client downscales so the longest side is ≤ 1008 px
  *before* uploading. Sending more is rejected (`400 bad_dimensions`) rather
  than silently downscaled, because a silent downscale would desynchronise the
  client's own coordinate mapping.

Reference client snippet (stdlib only, no numpy):

```python
buf = drawable_or_projection_buffer          # gegl
pixels = buf.get(rect, 1.0, "R'G'B' u8", Gegl.AbyssPolicy.CLAMP)  # bytes
assert len(pixels) == width * height * 3
conn.request("POST", "/images", body=pixels, headers={
    "Authorization": "Bearer " + token,
    "Content-Type": "application/octet-stream",
    "Content-Length": str(len(pixels)),
    "X-Width": str(width),
    "X-Height": str(height),
})
```

---

## 8. The mask return format

This is the subtle part of the contract. Read it twice.

Masks come back as **uint8 soft masks (0–255), cropped to their bounding box, at
model-canvas resolution** — *not* binary, *not* RLE, *not* at image resolution.

Why, restated from `DESIGN.md` §4 because it drives client design:

* **Thresholding is client-side and instant.** Dragging the mask-threshold
  slider re-thresholds bytes already in memory. **No round trip.** Same for the
  score slider, which is why PCS returns every candidate ≥ 0.02.
* **Edge quality is better.** Upsample the *soft* mask, then threshold → smooth
  edges. Threshold first, then upsample → stair-steps.
* **It is small.** A cropped 1008-scale soft mask is tens of KB; twenty are ~1 MB.

### 8.1 Frame layout

```
 offset            size          contents
 ---------------------------------------------------------------------------
 0                 8             magic  b"SAM3RES\x00"
 8                 4             uint32 LE  H = length of the JSON header
 12                H             JSON header, UTF-8, compact separators
 12 + H            B             blob region: mask bytes, tightly packed
 ---------------------------------------------------------------------------
 total body length = 12 + H + B
```

* All integers little-endian. `12` is `len(magic) + 4`; the constant is
  `sam3gimpd.types.RESULT_PREFIX_SIZE`.
* **`blob_offset` is relative to the start of the blob region**, i.e. to
  absolute byte `12 + H`. It is *never* relative to the start of the body.
  Absolute position of instance *i*'s pixels:
  `12 + H + instances[i].blob_offset`.
* Masks appear in the blob in the **same order as the `instances` array**,
  **tightly packed**, with **no alignment padding** between them. Therefore
  `instances[0].blob_offset == 0` and
  `instances[i].blob_offset == sum(instances[:i].blob_length)`.
* `B == header.blob_length == sum of every instance's blob_length`.
* A frame with zero instances is legal: `instances: []`, `blob_length: 0`,
  body length `12 + H`.

### 8.2 JSON header

```json
{
  "api_version": "1.1",
  "job_id": "j-000002-77c1",
  "request_id": "r-000007",
  "image_id": "5b1d3d3fd6a34c2f8f8d3c9a41b7e0d2",
  "engine": "pcs",
  "state": "done",
  "prompt": { "kind": "text", "text": "yellow school bus", "score_threshold": 0.02 },
  "image": { "width": 1008, "height": 672 },
  "model_canvas": { "width": 1008, "height": 672 },
  "canvas_from_image": { "scale_x": 1.0, "scale_y": 1.0, "offset_x": 0.0, "offset_y": 0.0 },
  "mask_encoding": "u8_soft",
  "elapsed_ms": 812.4,
  "truncated": false,
  "instances": [
    {
      "instance_id": 0,
      "score": 0.934,
      "label": "yellow school bus",
      "bbox": [412, 300, 700, 480],
      "mask_width": 288,
      "mask_height": 180,
      "blob_offset": 0,
      "blob_length": 51840
    },
    {
      "instance_id": 1,
      "score": 0.612,
      "label": "yellow school bus",
      "bbox": [40, 330, 190, 452],
      "mask_width": 150,
      "mask_height": 122,
      "blob_offset": 51840,
      "blob_length": 18300
    }
  ],
  "blob_length": 70140
}
```

Per-instance fields:

| Field | Type | Meaning |
|---|---|---|
| `instance_id` | int | stable within this result only; `0..N-1` in returned order |
| `score` | float | `[0, 1]`, model confidence. Instances are sorted **descending**. |
| `label` | string | the text that produced it (PCS) or `""` (PVS) |
| `bbox` | `[x0, y0, x1, y1]` ints | **model-canvas** pixels, half-open. `0 ≤ x0 < x1 ≤ canvas.width`, likewise y. |
| `mask_width` | int | `== x1 - x0`, always |
| `mask_height` | int | `== y1 - y0`, always |
| `blob_offset` | int | bytes from the start of the blob region |
| `blob_length` | int | `== mask_width * mask_height`, always |

`truncated` is `true` when `max_instances` clipped the list.

### 8.3 Mask pixel semantics

* One **`uint8` per pixel**, no padding, **row-major, top-to-bottom,
  left-to-right**, `mask_width` bytes per row, `mask_height` rows.
* Value = `round(255 * sigmoid(logit))`. So:
  * `0` = definitely outside, `255` = definitely inside;
  * **`128` corresponds to a raw logit of 0 — the model's own binarisation
    point, and therefore the client's default threshold.**
* A client thresholds with a plain byte comparison: `inside = value >= T`,
  `T ∈ [1, 255]`, default `128`. **This is a local operation with no round
  trip**, and it is the entire justification for this format. Do not add a
  server-side threshold parameter.
* The crop is tight but not guaranteed *minimal*: the daemon may pad the bbox by
  a pixel or two to avoid clipping soft falloff. Values outside the crop are
  defined to be `0`.

---

## 9. Mapping a mask back onto the original image

Given, from the client's own bookkeeping:

* `W0 × H0` — original-image size (e.g. `3000 × 2000`)
* `W1 × H1` — uploaded-image size (e.g. `1008 × 672`)

and, from the result header:

* `canvas_from_image = {sx, sy, ox, oy}`
* `instance.bbox = [x0, y0, x1, y1]` in model-canvas pixels

**Step 1 — canvas → uploaded-image** (invert the reported affine):

```
ix0 = (x0 - ox) / sx        ix1 = (x1 - ox) / sx
iy0 = (y0 - oy) / sy        iy1 = (y1 - oy) / sy
```

**Step 2 — uploaded-image → original-image** (the client's own downscale):

```
u = W0 / W1                 v = H0 / H1
dx0 = ix0 * u               dx1 = ix1 * u
dy0 = iy0 * v               dy1 = iy1 * v
```

**Step 3 — snap to integers.** Round each edge independently, *then* derive the
size, so that adjacent instances tile without gaps:

```
X0 = round(dx0)   X1 = round(dx1)   DW = max(1, X1 - X0)
Y0 = round(dy0)   Y1 = round(dy1)   DH = max(1, Y1 - Y0)
```

**Step 4 — resample the soft crop, then threshold.** The plug-in has no numpy
and must not resample in pure Python. Instead:

1. Create a temporary grayscale drawable/channel of `mask_width × mask_height`
   and write the raw crop bytes into it with the babl format `"Y' u8"` — the
   byte layout in §8.3 is exactly what `Gegl.Buffer.set()` expects.
2. Scale it to `DW × DH` with **GIMP's own scaler** (GEGL, in C, cubic or
   NoHalo). Quality is high and the cost is negligible.
3. Composite it at `(X0, Y0)` into a full-size `W0 × H0` channel that is `0`
   everywhere else.
4. **Threshold at the user's `T`** — after scaling, never before.

Steps 2–4 are what "GIMP does the upsampling" means in `DESIGN.md`. A canvas
preview (which draws at its own zoom) does the same thing with cairo instead.

**Worth stating:** the client never needs the model's resize policy, the
processor's internals, or a padding convention. The reported affine plus its own
downscale factor is sufficient and exact.

---

## 10. Request ids, the queue and supersession

* **Every prompt carries a client-generated `request_id`** (1–64 chars,
  `[A-Za-z0-9._:-]`). The daemon treats it as opaque and echoes it on the job,
  on every `JobStatus`, in the result header and in the `X-Sam3-Request-Id`
  response header. Uniqueness is the client's responsibility; a monotonic
  counter per dialog session is the intended pattern.

* **The daemon runs exactly one inference at a time.** One user, one GPU — there
  is no reason for more, and serialising removes every VRAM race. HTTP handling
  stays concurrent (§1); only the worker is serial.

* **Supersession.** When a prompt for image *I* is accepted, every job that is
  **queued but not yet started** for the **same image id** — regardless of
  engine — transitions to `superseded`, with `superseded_by` set to the new job
  id. The new job's `202` response lists them in `superseded_job_ids`.
  * A **running** job is never cancelled. It finishes and its result is
    retained; the client simply ignores it.
  * Jobs for *other* images are untouched; they queue FIFO.
  * The encode job created by `POST /images` is **never** superseded — prompts
    depend on it.

* **Clients ignore stale results.** The rule is one line and must be implemented
  exactly: *drop any result whose `request_id` is not the latest the client
  issued for that image.* This is what makes typing a new prompt mid-inference
  feel instant and prevents stale masks from ever reaching the canvas.

* **Ordering guarantee.** Jobs start in the order they were accepted, minus the
  superseded ones. A prompt accepted after `POST /images` always runs after that
  image's encode job.

---

## 11. Jobs and progress

**States** (`queued` → `running` → one of `done` / `failed`; or `superseded` /
`cancelled` from `queued`):

| State | Terminal | Meaning |
|---|---|---|
| `queued` | no | accepted, waiting for the worker; `queue_position` is set |
| `running` | no | the worker is on it; `progress` and `stage` advance |
| `done` | yes | result available (binary frame or `meta=1` JSON) |
| `failed` | yes | `error` is set; `progress` stops where it stopped |
| `superseded` | yes | a newer prompt for the same image displaced it (§10) |
| `cancelled` | yes | its image was deleted, or the daemon shut down (`error.code` `shutting_down`), before it started |

**Progress** is a float in `[0, 1]`, non-decreasing within a job. Reference
milestones so a progress bar behaves sensibly across engines:

| Stage | Progress | Note |
|---|---|---|
| `queued` | `0.0` | |
| `encoding` | `0.05 → 0.60` | `POST /images` jobs spend nearly all their time here |
| `prompting` | `0.60 → 0.80` | cheap when the embedding is cached |
| `decoding` | `0.80 → 0.95` | |
| `packing` | `0.95 → 1.0` | crop, quantise to uint8, build the frame |
| `done` / `failed` | `1.0` | |

**Long-poll semantics for `GET /jobs/{id}?wait=N`:**

1. If the job is already terminal, return immediately.
2. Otherwise wait up to `N` seconds (clamped to `[0, 30]`) for **either** a
   terminal state **or** a material progress change — `progress` increased by
   ≥ 0.01, or `stage` changed.
3. Return the current status whichever happened, including on timeout. A
   timeout is **not** an error: `200` with the unchanged status.

A client loop of `wait=10` therefore gets prompt progress updates, sees the
terminal state the moment it happens, and issues roughly one request per
progress step rather than per 100 ms. Clients must still cap total waiting
themselves and must tolerate `404 job_not_found` after the retention window.

---

## 12. Worked end-to-end example

A 3000×2000 photo; the user types "yellow school bus". Tokens abbreviated.

### 12.1 Handshake

```
$ cat "$XDG_DATA_HOME/sam3-gimp/runtime.json"
{"api_version":"1.1","host":"127.0.0.1","pid":48211,"port":41573,
 "started_at":1772395551.812,"token":"Vv3n…2Ab","version":"0.1.1"}

$ curl -s -H "Authorization: Bearer Vv3n…2Ab" \
       -H "X-Sam3-Nonce: 3k5Jq0bW1xYf9ZrN2c8HdTg4uVpLm6Ae" http://127.0.0.1:41573/hello
{"api_version":"1.1","sam3d_version":"0.1.1","engine_mode":"stub","device":"cpu",
 "dtype":"float32","capabilities":["pcs","pvs"],"torch_available":false,
 "weights_available":false,"model_canvas":{"width":1008,"height":1008},
 "pid":48211,"started_at":1772395551.812,"uptime_s":93.4,"limits":{…},
 "nonce_proof":"9b0c3f…e21d","build":"5e1f0a9c"}
```

`nonce_proof` matches the HMAC the client computes for its nonce, and `1`
major matches `1` major → accepted.

### 12.2 Upload

The client picks the downscale itself: `3000 × 2000` → `1008 × 672`
(`1008/3000 = 0.336`; `2000 × 0.336 = 672`). Body length
`1008 × 672 × 3 = 2 032 128`.

```
POST /images HTTP/1.1
Authorization: Bearer Vv3n…2Ab
Content-Type: application/octet-stream
Content-Length: 2032128
X-Width: 1008
X-Height: 672
X-Source-Width: 3000
X-Source-Height: 2000

<2032128 raw RGB bytes>
```

```json
HTTP/1.1 202 Accepted

{ "image_id": "5b1d3d3fd6a34c2f8f8d3c9a41b7e0d2",
  "job_id": "j-000001-a91f",
  "cached": false,
  "state": "queued",
  "image": { "width": 1008, "height": 672 },
  "model_canvas": { "width": 1008, "height": 672 },
  "canvas_from_image": { "scale_x": 1.0, "scale_y": 1.0, "offset_x": 0.0, "offset_y": 0.0 } }
```

The canvas equals the upload and the transform is the identity. The client
still stores `W0=3000, H0=2000, W1=1008, H1=672` and the transform, because
§5 forbids assuming either.

### 12.3 Prompt

```
POST /images/5b1d3d3fd6a34c2f8f8d3c9a41b7e0d2/text HTTP/1.1
Authorization: Bearer Vv3n…2Ab
Content-Type: application/json
Content-Length: 74

{"request_id":"r-000007","text":"yellow school bus","score_threshold":0.02}
```

```json
HTTP/1.1 202 Accepted

{ "job_id": "j-000002-77c1", "request_id": "r-000007",
  "image_id": "5b1d3d3fd6a34c2f8f8d3c9a41b7e0d2",
  "engine": "pcs", "state": "queued", "superseded_job_ids": [] }
```

### 12.4 Poll

```
GET /jobs/j-000002-77c1?wait=10 HTTP/1.1
Authorization: Bearer Vv3n…2Ab
```

```json
HTTP/1.1 200 OK
Content-Type: application/json; charset=utf-8

{ "job_id": "j-000002-77c1", "state": "running", "engine": "pcs",
  "image_id": "5b1d…", "request_id": "r-000007",
  "progress": 0.6, "stage": "prompting",
  "created_at": 1772395601.114, "started_at": 1772395601.130,
  "finished_at": null, "elapsed_ms": 214.0,
  "queue_position": null, "superseded_by": null,
  "masks_available": false, "error": null }
```

The next poll returns the frame:

```
HTTP/1.1 200 OK
Content-Type: application/vnd.sam3.result+binary
Content-Length: 70944
X-Sam3-Job-State: done
X-Sam3-Header-Length: 792
X-Sam3-Request-Id: r-000007
```

Body: `12 + 792 + 70140 = 70944` bytes — the JSON header of §8.2 serialised with
compact separators is 792 bytes, and the two masks are `51840 + 18300 = 70140`.

### 12.5 Decode (client, stdlib only)

```python
import json, struct

MAGIC = b"SAM3RES\x00"
assert body[:8] == MAGIC
(hlen,) = struct.unpack_from("<I", body, 8)
header = json.loads(body[12:12 + hlen].decode("utf-8"))
blob_start = 12 + hlen

if header["request_id"] != latest_request_id:      # §10 — drop stale results
    return

for inst in header["instances"]:
    off = blob_start + inst["blob_offset"]
    mask = body[off:off + inst["blob_length"]]     # uint8, row-major
    assert len(mask) == inst["mask_width"] * inst["mask_height"]
```

### 12.6 Place instance 0 on the original image

Header values: `bbox = [412, 300, 700, 480]`, `mask 288 × 180`,
`sx = sy = 1.0, ox = oy = 0` -- the identity, because the reference engines
post-process masks back to the uploaded image (§5). The arithmetic is written
out in full anyway: a client inverts whatever transform it is given.

```
canvas -> uploaded:  ix0 = (412 - 0)/1.0 = 412.0    ix1 = (700 - 0)/1.0 = 700.0
                     iy0 = (300 - 0)/1.0 = 300.0    iy1 = (480 - 0)/1.0 = 480.0

uploaded -> original: u = 3000/1008 = 2.976190…  v = 2000/672 = 2.976190…
                     dx0 = 412 * u = 1226.19    dx1 = 700 * u = 2083.33
                     dy0 = 300 * v =  892.86    dy1 = 480 * v = 1428.57

snap:                X0 = 1226   X1 = 2083   DW = 857
                     Y0 =  893   Y1 = 1429   DH = 536
```

So: load 288×180 soft bytes into a temp channel, let GIMP scale it to 857×536,
composite at (1226, 893) into a 3000×2000 channel, threshold at 128. Moving the
threshold slider afterwards re-thresholds the same scaled channel — **no HTTP
at all**.

### 12.7 Refine with a click, then finish

```
POST /images/5b1d…/points     {"request_id":"r-000008",
                               "points":[{"x":512.0,"y":300.0,"label":1}],
                               "multimask":true}
→ 202 {"job_id":"j-000003-2c40","engine":"pvs",…,"superseded_job_ids":[]}

DELETE /images/5b1d…           → 200 {"deleted":true,"cancelled_job_ids":[]}
```

The point coordinates are **uploaded-image** pixels: a canvas click at original
`(1524, 893)` becomes `(1524 × 1008/3000, 893 × 672/2000) = (512.06, 300.05)`.

---

## 13. Daemon invocation contract

The plug-in's launcher and the daemon's CLI must agree on this. Flags:

```
sam3gimpd serve [--host 127.0.0.1] [--port 0] [--stub]
            [--parent-pid PID] [--idle-ttl 1800] [--cache-size 3]
            [--runtime-file PATH] [--log-level info] [--device auto]
```

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | bind address |
| `--port` | `0` | `0` = ephemeral, reported in `runtime.json` |
| `--stub` | off | fake engine, no torch import at all (§14) |
| `--parent-pid` | none | exit when that pid disappears (the plug-in passes **GIMP's** pid, not its own — plug-in processes are short-lived) |
| `--idle-ttl` | `1800` | seconds since the last authenticated request before exiting to free VRAM (never while a job is queued or running); `0` disables. Thirty minutes because every idle exit costs the next use a cold start. |
| `--cache-size` | `3` | LRU capacity of the embedding cache |
| `--runtime-file` | per §3.1 | override the handshake file |
| `--device` | `auto` | `auto` = cuda → mps → cpu |

Other subcommands: `sam3gimpd doctor` prints a JSON report -- always JSON;
there is no `--json` flag -- of versions, device, dtype, paths and weights,
with the running daemon's `/status` under `daemon` if one answers (`--no-daemon`
skips that). `sam3gimpd download` fetches the gated checkpoint given an HF
token.

**Logging.** The daemon logs to `<base>/logs/sam3gimpd.log` (`0600`), rotated
at 5 MB with three old copies kept. Rotation copies the file aside and truncates
it in place, because the launcher's stdout/stderr redirect (below) writes to
the same file and must keep landing in the live one.

**Detached spawn** (mirrors `DESIGN.md` §4):

* Windows: `pythonw.exe` (never `python.exe` — it flashes a console),
  `creationflags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`, stdout/stderr
  redirected to `<base>/logs/sam3gimpd.log`, `close_fds=True`.
* POSIX: `start_new_session=True`, same redirection.
* The child must outlive the plug-in process that spawned it. The launcher never
  calls `wait()`.
* There is no supervisor and no restart-on-crash: a crash means the next plug-in
  invocation respawns. `<base>/logs/crash.log` holds the last traceback for the
  Doctor panel.

**Environment** the launcher sets for the child: `HF_HOME=<base>/hf`,
`HF_HUB_DISABLE_PROGRESS_BARS=1`, `HF_HUB_DISABLE_TELEMETRY=1`,
`PYTHONUNBUFFERED=1`, and `SAM3_GIMP_HOME=<base>` — see
`sam3gimpd.paths.daemon_environ()`.

---

## 14. Stub engine guarantees

`--stub` is a **product feature**, not a test fixture: it is how the plug-in is
developed on a machine with no GPU and no weights. A stub daemon is a fully
conforming implementation of this document. Specifically:

* It imports **no torch, no transformers, no numpy** — ever, on any code path.
* `GET /hello` reports `engine_mode: "stub"`, `device: "stub"`, and the same
  `capabilities` list a real engine would.
* Results are **deterministic**: the same `(image_id, prompt)` yields byte-identical
  frames. Geometry is derived from a hash of the prompt text (PCS) or from the
  supplied points/box (PVS), so different prompts give visibly different blobs.
* Masks are genuine **soft** uint8 gradients — an anti-aliased falloff at the
  blob edge — so that a client's threshold slider visibly does something. A
  constant-255 rectangle would hide client bugs.
* Scores are in `[0.1, 1.0]`, sorted descending, and PCS returns 1–8 instances
  depending on the prompt hash.
* Timings are simulated in the same stage order as the real engine (§11) so
  progress bars and supersession can be exercised. Total latency is a few
  hundred milliseconds, not seconds.

Every protocol test in `tests/daemon/` runs against the stub.

---

## 15. Limits

| Limit | Value | Enforcement |
|---|---|---|
| image side | `[16, 1008]` px each | `400 bad_dimensions` |
| upload body | `1008 × 1008 × 3 = 3 048 192` bytes | `413 payload_too_large` |
| JSON body | 256 KiB | `413 payload_too_large` |
| `text` | 512 characters | `400 bad_request` |
| `points` | 64 | `400 bad_request` |
| `boxes` | 16 | `400 bad_request` |
| `request_id` | 64 chars, `[A-Za-z0-9._:-]` | `400 bad_request` |
| instances per result | 256 (`max_instances` ≤ this) | clipped, `truncated: true` |
| queue depth | 16 pending jobs | `503 queue_full` |
| `?wait=` | `[0, 30]` seconds | clamped, not an error |
| job result retention | ≥ 120 s (up to 512 jobs); the 64 most recent always | then `404 job_not_found` |
| embedding cache | 3 images (LRU, `--cache-size`); over it only while work is pending | eviction → `404 image_not_found` |
| concurrent connections | 32 | closed on accept, no response |
| connection idle | 10 s before the first request, 120 s between requests | closed |
| request head | request line + headers within 10 s in total | closed |
| request I/O | 15 s per read or write once a request has begun | closed |

The protocol numbers appear in `sam3gimpd.types.Limits` and in `/hello`'s
`limits` object. `types.Limits` is the source of truth; keep the other two in
sync. The connection limits are server policy (`sam3gimpd.server`) rather than
protocol: a conforming client never comes near them.

---

## 16. Stable guarantees for client authors

Every 1.x daemon keeps these; changing any of them is a major version bump, so
a client may build on them without defensive code:

1. **Client → daemon coordinates are uploaded-image space; daemon → client mask
   geometry is model-canvas space.** Never mix them. Never hardcode the
   transform between them.
2. **`blob_offset` is relative to the blob region** (`12 + header_length`), and
   masks are tightly packed in instance order with no padding.
3. **Soft uint8 masks, cropped, at model resolution. Threshold 128 = logit 0.**
   No server-side threshold parameter; the client re-thresholds locally.
4. **`mask_width == bbox width`, `mask_height == bbox height`,
   `blob_length == mask_width * mask_height`.** Always, no exceptions.
5. **One inference at a time; queued-not-started jobs for the same image are
   superseded; clients drop results whose `request_id` is stale.**
6. **The HTTP server is multi-threaded, responses always carry
   `Content-Length`, and no response is chunked.**
7. **Every endpoint requires the bearer token; every non-2xx body is the
   `{error: {code, message, detail}}` envelope; a failed job is `200` with
   `state: "failed"`.**
8. **`runtime.json` has exactly the five required fields, is written atomically
   after the socket is listening, and unknown keys are ignored by readers.**
9. **`--stub` never imports torch.** The base install (this directory, no
   extras) has zero runtime dependencies.
10. **The plug-in imports only the standard library and `gi`.** It mirrors
    `sam3gimpd/types.py` by hand; it never imports it.
11. **From 1.1, `/hello` answers a well-formed `X-Sam3-Nonce` with
    `nonce_proof`** (§2), and a client never uses a daemon that cannot prove it
    holds the token.
