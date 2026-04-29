"""Thin wrapper around AXL's local HTTP bridge — /send + /recv.

Each agent talks to its local AXL node at 127.0.0.1:9002. To send,
POST bytes with the destination peer ID in `X-Destination-Peer-Id`.
To receive, GET /recv (returns 204 if empty, 200 + body if a message
is queued). Source peer ID lands in `X-From-Peer-Id`.

For the demo we wrap a JSON envelope around the bytes, so explorer
and refiner exchange structured payloads (`{"kind": "STRATEGY_DRAFT",
...}` / `{"kind": "OPTIMIZATION_RESULT", ...}`) over a transport that
itself only cares about bytes."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

# httpx logs every request at INFO; mute so the demo logs stay readable.
logging.getLogger("httpx").setLevel(logging.WARNING)

AXL_BRIDGE_URL = "http://127.0.0.1:9002"
_POLL_INTERVAL_SEC = 0.5


def send_envelope(*, dest_peer_id: str, payload: dict[str, Any]) -> int:
    """Serialize a JSON envelope and ship to a peer. Returns bytes sent."""
    body = json.dumps(payload).encode()
    with httpx.Client(timeout=10.0) as client:
        r = client.post(
            f"{AXL_BRIDGE_URL}/send",
            content=body,
            headers={
                "X-Destination-Peer-Id": dest_peer_id,
                "Content-Type": "application/octet-stream",
            },
        )
    r.raise_for_status()
    return int(r.headers.get("X-Sent-Bytes", "0"))


def recv_envelope(*, timeout_sec: float = 60.0) -> tuple[str, dict[str, Any]] | None:
    """Block-poll the bridge for one envelope. Returns (from_peer_id,
    payload) or None on timeout. Polls every 0.5s — fine for demo
    cadence, fast enough that round-trip log lines feel snappy on
    camera."""
    deadline = time.time() + timeout_sec
    with httpx.Client(timeout=5.0) as client:
        while time.time() < deadline:
            r = client.get(f"{AXL_BRIDGE_URL}/recv")
            if r.status_code == 204:
                time.sleep(_POLL_INTERVAL_SEC)
                continue
            r.raise_for_status()
            from_peer = r.headers.get("X-From-Peer-Id", "<unknown>")
            try:
                payload = json.loads(r.content.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                # Garbage from a peer we don't recognize — log & skip.
                # Both decode errors and JSON errors land here so a
                # malformed envelope doesn't crash the polling loop.
                logging.getLogger(__name__).warning(
                    "dropping malformed envelope from %s: %s", from_peer[:16], e
                )
                continue
            return from_peer, payload
    return None
