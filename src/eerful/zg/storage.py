"""Storage adapter — content-addressed blob upload/download.

Two implementations satisfy the same `StorageClient` Protocol:

- `BridgeStorageClient` talks to the local zg-bridge service
  (`services/zg-bridge/`), which wraps `@0gfoundation/0g-ts-sdk` over HTTP.
  This is the production path for Galileo testnet. Like `ComputeClient`,
  the Python side stays thin; the Node bridge holds the wallet and
  drives the indexer.
- `MockStorageClient` is an in-process dict keyed by sha256. Suitable for
  tests and the local demo when no testnet access is needed.

Both speak content hashes (sha256, 0x-prefixed lowercase 64-hex). The
receipt format is content-addressed by sha256 (`evaluator_id`,
`attestation_report_hash`); the bridge maps sha256 → 0G rootHash
internally so callers never need to handle 0G-specific URIs.

Error contract (mirrors lockstep's locked storage-ts contract):

- 200: success.
- 400: programming bug on our side (malformed request). Raised as
  `RuntimeError` so retry loops don't catch it — the caller's input is
  fine; the marshalling is wrong.
- 404: the bridge's sha256 → rootHash index has no entry. Raised as
  `StorageError`. The bytes may exist on 0G but a different process
  uploaded them, or the bridge restarted since upload.
- 422: bytes returned by the indexer don't re-hash to the requested
  content hash. Raised as `TrustViolation`. Byzantine evidence — never
  retry.
- 5xx: SDK / indexer / RPC failure. Raised as `StorageError`. Transient.
- Transport (connect refused, timeout): `StorageError`.
"""

from __future__ import annotations

import hashlib
from typing import Any, NoReturn, Protocol, runtime_checkable

import httpx

from eerful.canonical import Bytes32Hex, is_bytes32_hex, to_lower_hex
from eerful.errors import StorageError, TrustViolation

__all__ = [
    "BridgeStorageClient",
    "MockStorageClient",
    "StorageClient",
]


@runtime_checkable
class StorageClient(Protocol):
    """Vendor-agnostic content-addressed storage."""

    def upload_blob(self, data: bytes) -> Bytes32Hex:
        """Upload `data`, return its sha256 content hash. Idempotent."""
        ...

    def download_blob(self, content_hash: Bytes32Hex) -> bytes:
        """Return bytes whose sha256 equals `content_hash`. Raises on miss."""
        ...


def _sha256_hex(data: bytes) -> Bytes32Hex:
    return "0x" + hashlib.sha256(data).hexdigest()


class BridgeStorageClient:
    """Synchronous httpx client for the local zg-bridge storage endpoints.

    Construction is cheap (no network); the underlying httpx.Client
    lazy-connects on first request and pools connections across calls.
    Use as a context manager or call `close()` when done.
    """

    def __init__(
        self,
        *,
        bridge_url: str,
        http: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._bridge_url = bridge_url.rstrip("/")
        self._owns_http = http is None
        self._http = http or httpx.Client(timeout=timeout)

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> BridgeStorageClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def upload_blob(self, data: bytes) -> Bytes32Hex:
        if not data:
            raise ValueError("upload_blob: data must be non-empty")
        try:
            r = self._http.post(
                f"{self._bridge_url}/storage/upload-blob",
                content=data,
                headers={"Content-Type": "application/octet-stream"},
            )
        except httpx.RequestError as exc:
            raise StorageError(
                f"transport error uploading to {self._bridge_url}: {exc}"
            ) from exc
        if not r.is_success:
            self._raise_for_status(r, "POST /storage/upload-blob")
        body = _json_or_raise(r)
        bridge_hash_raw = body.get("content_hash")
        if not isinstance(bridge_hash_raw, str):
            raise StorageError(
                f"upload-blob: bridge returned no content_hash (body={body!r})"
            )
        # to_lower_hex raises ValueError on non-hex input; wrap so the
        # adapter's documented error contract holds (only StorageError /
        # TrustViolation / RuntimeError leave this method).
        try:
            bridge_hash = to_lower_hex(bridge_hash_raw)
        except (TypeError, ValueError) as exc:
            raise StorageError(
                f"upload-blob: bridge returned non-hex content_hash {bridge_hash_raw!r}"
            ) from exc
        if not is_bytes32_hex(bridge_hash):
            raise StorageError(
                f"upload-blob: bridge returned non-bytes32 content_hash {bridge_hash!r}"
            )
        local_hash = _sha256_hex(data)
        if bridge_hash != local_hash:
            # Bridge claims a different sha256 for the same bytes. Either
            # the bridge is buggy or something corrupted the body in
            # transit; never trust the storage URI in that case.
            raise TrustViolation(
                f"upload-blob hash mismatch: bridge said {bridge_hash}, "
                f"local computed {local_hash}"
            )
        return local_hash

    def download_blob(self, content_hash: Bytes32Hex) -> bytes:
        canonical = to_lower_hex(content_hash)
        # Validate before sending: the bridge would 400 a malformed hash
        # and surface it as RuntimeError, but that's caller-input
        # validation dressed up as a programming bug. Catch it locally.
        if not is_bytes32_hex(canonical):
            raise ValueError(
                f"content_hash must be 0x-prefixed 64-char lowercase hex, got {canonical!r}"
            )
        try:
            r = self._http.get(
                f"{self._bridge_url}/storage/download-blob",
                params={"content_hash": canonical},
            )
        except httpx.RequestError as exc:
            raise StorageError(
                f"transport error downloading from {self._bridge_url}: {exc}"
            ) from exc
        if not r.is_success:
            self._raise_for_status(r, "GET /storage/download-blob")
        data = r.content
        # Defense in depth: the bridge already re-hashes on its side, but
        # the verifier's job is to trust no one. A successful response
        # whose body doesn't hash to the requested content_hash is a
        # TrustViolation regardless of which intermediary lied.
        actual = _sha256_hex(data)
        if actual != canonical:
            raise TrustViolation(
                f"download-blob hash mismatch: requested {canonical}, "
                f"received bytes hash to {actual}"
            )
        return data

    @staticmethod
    def _raise_for_status(r: httpx.Response, op: str) -> NoReturn:
        status = r.status_code
        try:
            payload: Any = r.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            error = payload.get("error", "unknown")
            detail = payload.get("detail", payload)
        elif payload is None:
            error = "unknown"
            detail = r.text[:500]
        else:
            # JSON list/scalar — possible if a future bridge change
            # returns an envelope-less error. Don't AttributeError.
            error = "unknown"
            detail = payload
        msg = f"{op} failed ({status} {error}): {detail}"
        if status == 422:
            raise TrustViolation(msg)
        if status == 400:
            raise RuntimeError(msg)
        raise StorageError(msg)


def _json_or_raise(r: httpx.Response) -> dict[str, Any]:
    try:
        body = r.json()
    except ValueError as exc:
        raise StorageError(f"bridge returned non-JSON body: {r.text[:500]}") from exc
    if not isinstance(body, dict):
        raise StorageError(f"bridge returned non-object body: {body!r}")
    return body


class MockStorageClient:
    """In-process content-addressed storage. For tests and offline demos.

    Stores bytes keyed by their sha256. `upload_blob` is idempotent;
    `download_blob` raises `StorageError` on miss and `TrustViolation`
    if the in-process map has been tampered with so a key no longer
    matches its bytes (catches test bugs that mutate the dict).
    """

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def upload_blob(self, data: bytes) -> Bytes32Hex:
        if not data:
            raise ValueError("upload_blob: data must be non-empty")
        h = _sha256_hex(data)
        self._store[h] = data
        return h

    def download_blob(self, content_hash: Bytes32Hex) -> bytes:
        canonical = to_lower_hex(content_hash)
        data = self._store.get(canonical)
        if data is None:
            raise StorageError(f"unknown content_hash: {canonical}")
        actual = _sha256_hex(data)
        if actual != canonical:
            raise TrustViolation(
                f"mock storage tampered: key {canonical} maps to bytes hashing to {actual}"
            )
        return data
