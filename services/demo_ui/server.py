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

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

NDJSON_PATH = Path(os.environ.get("EERFUL_DEMO_UI_NDJSON", "/tmp/eerful-events.ndjson"))
HEARTBEAT_SEC = 15.0
TAIL_POLL_SEC = 0.1
HISTORY_CAP = 1000

HERE = Path(__file__).parent
STATIC_DIR = HERE / "static"

app = FastAPI(title="eerful demo-ui")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/events")
async def post_event(request: Request) -> dict[str, str]:
    """Append a remote-pushed event to the NDJSON file. Refiner@louie
    uses this over LAN."""
    event = await request.json()
    NDJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NDJSON_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, separators=(",", ":")) + "\n")
    return {"ok": "true"}


async def _tail_ndjson() -> AsyncIterator[str]:
    """Yield each new line as it's appended. Replays history first
    (capped to HISTORY_CAP), then tails."""
    NDJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    NDJSON_PATH.touch(exist_ok=True)

    with NDJSON_PATH.open("r", encoding="utf-8") as f:
        history = f.readlines()[-HISTORY_CAP:]
        for line in history:
            line = line.strip()
            if line:
                yield line
        # tail
        while True:
            line = f.readline()
            if not line:
                await asyncio.sleep(TAIL_POLL_SEC)
                continue
            line = line.strip()
            if line:
                yield line


@app.get("/events")
async def sse_events(request: Request) -> StreamingResponse:
    async def stream() -> AsyncIterator[bytes]:
        last_heartbeat = asyncio.get_event_loop().time()
        async for line in _tail_ndjson():
            if await request.is_disconnected():
                break
            yield f"data: {line}\n\n".encode()
            now = asyncio.get_event_loop().time()
            if now - last_heartbeat > HEARTBEAT_SEC:
                yield b": heartbeat\n\n"
                last_heartbeat = now

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
