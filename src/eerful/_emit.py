"""Demo-UI event emission. Best-effort; never fails the caller.

Writes one NDJSON line per event to a local file (always), and POSTs
the same payload to a remote collector if `EERFUL_DEMO_UI_URL` is set.

Both sinks swallow errors — emitting an event MUST NOT break the
demo if the UI sidecar is down or the disk is full.

Env:
  EERFUL_DEMO_UI_NDJSON  default: /tmp/eerful-events.ndjson
  EERFUL_DEMO_UI_URL     if set, POST events to <url>/events
  EERFUL_DEMO_RUN_ID     run identifier; auto-generated per process if unset
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_NDJSON = "/tmp/eerful-events.ndjson"
_HTTP_TIMEOUT_SEC = 0.25


def _run_id() -> str:
    rid = os.environ.get("EERFUL_DEMO_RUN_ID")
    if not rid:
        rid = uuid.uuid4().hex[:12]
        os.environ["EERFUL_DEMO_RUN_ID"] = rid
    return rid


def emit_event(source: str, kind: str, **payload: Any) -> None:
    """Write one event to the local NDJSON file and (best-effort) POST
    to EERFUL_DEMO_UI_URL. Never raises."""
    event = {
        "ts": time.time(),
        "run_id": _run_id(),
        "source": source,
        "kind": kind,
        "payload": payload,
    }
    line = json.dumps(event, separators=(",", ":")) + "\n"

    path = Path(os.environ.get("EERFUL_DEMO_UI_NDJSON", _DEFAULT_NDJSON))
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError as e:
        log.debug("emit_event: file write failed: %s", e)

    url = os.environ.get("EERFUL_DEMO_UI_URL")
    if url:
        try:
            import httpx  # local import: keep _emit importable without httpx

            with httpx.Client(timeout=_HTTP_TIMEOUT_SEC) as client:
                client.post(f"{url.rstrip('/')}/events", json=event)
        except Exception as e:  # noqa: BLE001 — best-effort, must not raise
            log.debug("emit_event: POST failed: %s", e)
