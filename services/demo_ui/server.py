"""Demo-UI sidecar — serves the live SPA and an SSE event stream.

Reads events from the NDJSON file written by `eerful._emit.emit_event`
and broadcasts each new line to connected EventSource clients.

Also exposes POST /events so remote agents (refiner@louie) can push
events directly over the LAN — those POSTs get appended to the same
NDJSON file and fan out via SSE.

Run:
  uv sync --extra demo
  uv run uvicorn services.demo_ui.server:app --port 8088 --reload

Env:
  EERFUL_DEMO_UI_NDJSON  default: /tmp/eerful-events.ndjson
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

NDJSON_PATH = Path(os.environ.get("EERFUL_DEMO_UI_NDJSON", "/tmp/eerful-events.ndjson"))
HEARTBEAT_SEC = 15.0
TAIL_POLL_SEC = 0.1
HISTORY_CAP = 1000

# POST /events is reachable from any LAN host when the server is bound
# to 0.0.0.0 (required so refiner@louie can push events). Cap the body
# so a buggy or malicious client can't OOM the sidecar with a 10 MB
# payload — real events are a few hundred bytes each.
MAX_EVENT_BYTES = 64 * 1024
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

HERE = Path(__file__).parent
STATIC_DIR = HERE / "static"

app = FastAPI(title="eerful demo-ui")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def _validate_event(event: object) -> dict:
    """Light shape check for events arriving over the LAN. The SPA's
    renderer assumes specific keys and typed values; rejecting
    malformed input at the boundary keeps the SSE stream parseable
    and prevents `kind`/`source` from being non-strings the JS would
    try to interpolate as DOM attributes."""
    if not isinstance(event, dict):
        raise HTTPException(status_code=400, detail="event must be a JSON object")
    for field in ("source", "kind"):
        v = event.get(field)
        if not isinstance(v, str) or not v:
            raise HTTPException(
                status_code=400,
                detail=f"event.{field} must be a non-empty string",
            )
    payload = event.get("payload", {})
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="event.payload must be an object")
    return event


@app.post("/events")
async def post_event(request: Request) -> dict[str, str]:
    """Append a remote-pushed event to the NDJSON file. Refiner@louie
    uses this over LAN."""
    # `Content-Length` is advisory — clients can lie or omit it. Read
    # the raw body up to the cap and reject if it overflows; this
    # bounds memory regardless of the header.
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > MAX_EVENT_BYTES:
                raise HTTPException(status_code=413, detail="event too large")
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid Content-Length") from None
    body = await request.body()
    if len(body) > MAX_EVENT_BYTES:
        raise HTTPException(status_code=413, detail="event too large")
    try:
        raw = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="event must be valid JSON") from None
    event = _validate_event(raw)
    NDJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NDJSON_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, separators=(",", ":")) + "\n")
    return {"ok": "true"}


@app.post("/admin/clear")
async def clear_events(request: Request) -> dict[str, str]:
    """Truncate the NDJSON file so a fresh SSE connection replays an
    empty history. Used by the SPA's clear button between recording
    takes. Loopback-only — a LAN-reachable clear endpoint would let
    any homelab host disrupt a recording."""
    client = request.client.host if request.client else None
    if client not in _LOOPBACK_HOSTS:
        raise HTTPException(
            status_code=403,
            detail="/admin/clear is loopback-only",
        )
    NDJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    NDJSON_PATH.write_text("")
    return {"ok": "true"}


async def _tail_ndjson() -> AsyncIterator[str | None]:
    """Yield each new line as it's appended. Replays history first
    (capped to HISTORY_CAP), then tails. Yields ``None`` as a
    heartbeat sentinel after every ``HEARTBEAT_SEC`` of idle so the
    consumer can flush a `: heartbeat` comment — without this, an
    SSE client behind a NAT/proxy with an idle-connection timeout
    silently drops during narration pauses.

    Detects file truncation (e.g. /admin/clear) by comparing the
    current size on disk to the descriptor's read offset; on
    truncation it `seek(0)` so an existing tailer doesn't sit past
    the new EOF forever waiting for a write that already happened
    near the start of the file.
    """
    NDJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    NDJSON_PATH.touch(exist_ok=True)

    with NDJSON_PATH.open("r", encoding="utf-8") as f:
        history = f.readlines()[-HISTORY_CAP:]
        for line in history:
            line = line.strip()
            if line:
                yield line
        last_hb = asyncio.get_event_loop().time()
        # Track the last-observed file size so we can detect a shrink
        # event between polls. Comparing current size to current read
        # offset (the simpler check) misses the race where writes
        # *after* truncation re-grow the file past the old offset
        # before the next poll fires — at that point st_size would
        # match tell() and the truncation goes undetected.
        last_size = f.tell()
        while True:
            line = f.readline()
            if not line:
                try:
                    cur_size = NDJSON_PATH.stat().st_size
                    if cur_size < last_size:
                        f.seek(0)
                        last_size = 0
                except FileNotFoundError:
                    # File transiently missing (e.g. rotated); ignore
                    # and let the next poll resync.
                    pass
                await asyncio.sleep(TAIL_POLL_SEC)
                now = asyncio.get_event_loop().time()
                if now - last_hb >= HEARTBEAT_SEC:
                    yield None  # heartbeat sentinel
                    last_hb = now
                continue
            line = line.strip()
            if line:
                yield line
                last_size = f.tell()
                last_hb = asyncio.get_event_loop().time()


@app.get("/events")
async def sse_events(request: Request) -> StreamingResponse:
    async def stream() -> AsyncIterator[bytes]:
        async for item in _tail_ndjson():
            if await request.is_disconnected():
                break
            if item is None:
                yield b": heartbeat\n\n"
            else:
                yield f"data: {item}\n\n".encode()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# Serve any extra static assets (app.js, styles.css) under /static
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
