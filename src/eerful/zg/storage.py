"""Storage adapter — content-addressed blob upload/download.

Two implementations satisfy the same `StorageClient` Protocol:

- `BridgeStorageClient` talks to the local zg-bridge service
  (`services/zg-bridge/`), which wraps `@0gfoundation/0g-ts-sdk` over HTTP.
  This is the production path for Galileo testnet. Like `ComputeClient`,
  the Python side stays thin; the Node bridge holds the wallet and
  drives the indexer.
- `MockStorageClient` is an in-process dict keyed by sha256. Suitable for
  tests and the local demo when no testnet access is needed.

Both speak content hashes (sha256, 0x-prefixed lowercase 64-hex) AND a
backend storage_root (the indexer's lookup key — 0G's Merkle root for
the bridge backend, sha256 again for the mock). Receipts carry both as
a (integrity_hash, retrieval_locator) pair so any verifier with any
storage instance can fetch by storage_root and re-hash for integrity.

Trust asymmetry on storage_root: the upload caller cannot recompute the
0G Merkle root locally (Python has no @0gfoundation/0g-ts-sdk), so
`BridgeStorageClient.upload_blob` trusts the bridge's claimed root_hash
verbatim. The sha256 cross-check still catches byte tampering. A future
re-upload-and-compare check is deferred.

Error contract (mirrors lockstep's locked storage-ts contract):

- 200: success.
- 400: programming bug on our side (malformed request). Raised as
  `RuntimeError` so retry loops don't catch it — the caller's input is
  fine; the marshalling is wrong.
- 422: bytes returned by the indexer don't re-hash to the requested
  content hash. Raised as `TrustViolation`. Byzantine evidence — never
  retry.
- 5xx: SDK / indexer / RPC failure. Raised as `StorageError`. Transient.
- Transport (connect refused, timeout): `StorageError`.
"""

from __future__ import annotations

import hashlib
from typing import Any, NamedTuple, NoReturn, Protocol, runtime_checkable

import httpx

from eerful.canonical import Bytes32Hex, is_bytes32_hex, to_lower_hex
from eerful.errors import StorageError, TrustViolation

__all__ = [
    "BridgeStorageClient",
    "MockStorageClient",
    "StorageClient",
    "UploadResult",
]


class UploadResult(NamedTuple):
    """Result of `StorageClient.upload_blob`.

    `content_hash` is sha256 of the uploaded bytes (same value the
    receipt fields `evaluator_id` / `attestation_report_hash` carry).
    `storage_root` is the backend's retrieval locator — for the 0G
    bridge that's the Merkle rootHash; for the in-process mock it's
    sha256 again. Both are 0x-prefixed lowercase 32-byte hex.
    """

    content_hash: Bytes32Hex
    storage_root: Bytes32Hex


@runtime_checkable
class StorageClient(Protocol):
    """Vendor-agnostic content-addressed storage."""

    def upload_blob(self, data: bytes) -> UploadResult:
        """Upload `data`, return content_hash + storage_root. Idempotent."""
        ...

    def download_blob(
        self,
        content_hash: Bytes32Hex,
        storage_root: Bytes32Hex,
    ) -> bytes:
        """Fetch by `storage_root` (the indexer key); confirm bytes hash to
        `content_hash` for integrity. Both are required — `storage_root`
        without `content_hash` would skip the integrity check, and
        `content_hash` without `storage_root` would force a producer-side
        sha256→root index (Tier 2's whole point is to remove that)."""
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

    def upload_blob(self, data: bytes) -> UploadResult:
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
        bridge_hash = _read_bytes32_field(body, "content_hash")
        local_hash = _sha256_hex(data)
        if bridge_hash != local_hash:
            # Bridge claims a different sha256 for the same bytes. Either
            # the bridge is buggy or something corrupted the body in
            # transit; never trust the storage URI in that case.
            raise TrustViolation(
                f"upload-blob hash mismatch: bridge said {bridge_hash}, "
                f"local computed {local_hash}"
            )
        # No analogous local cross-check for storage_root: Python has no
        # 0G Merkle implementation. We trust the bridge's claimed
        # root_hash. Future hardening (re-upload through a second
        # bridge, compare roots) is out of scope for Tier 2.
        storage_root = _read_bytes32_field(body, "root_hash")
        return UploadResult(content_hash=local_hash, storage_root=storage_root)

    def download_blob(
        self,
        content_hash: Bytes32Hex,
        storage_root: Bytes32Hex,
    ) -> bytes:
        canonical_hash = to_lower_hex(content_hash)
        canonical_root = to_lower_hex(storage_root)
        # Validate before sending: the bridge would 400 a malformed hash
        # and surface it as RuntimeError, but that's caller-input
        # validation dressed up as a programming bug. Catch it locally.
        if not is_bytes32_hex(canonical_hash):
            raise ValueError(
                f"content_hash must be 0x-prefixed 64-char lowercase hex, got {canonical_hash!r}"
            )
        if not is_bytes32_hex(canonical_root):
            raise ValueError(
                f"storage_root must be 0x-prefixed 64-char lowercase hex, got {canonical_root!r}"
            )
        try:
            r = self._http.get(
                f"{self._bridge_url}/storage/download-blob",
                params={
                    "content_hash": canonical_hash,
                    "root_hash": canonical_root,
                },
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
        if actual != canonical_hash:
            raise TrustViolation(
                f"download-blob hash mismatch: requested {canonical_hash}, "
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


def _read_bytes32_field(body: dict[str, Any], field: str) -> Bytes32Hex:
    """Pull a Bytes32Hex out of an upload-blob response body, raising
    StorageError (the adapter's documented contract) on shape failures.

    Used for both `content_hash` and `root_hash`: each must be a
    0x-prefixed lowercase 64-hex string for downstream callers to
    canonicalize/compare without surprise.
    """
    raw = body.get(field)
    if not isinstance(raw, str):
        raise StorageError(
            f"upload-blob: bridge returned no {field} (body={body!r})"
        )
    try:
        canonical = to_lower_hex(raw)
    except (TypeError, ValueError) as exc:
        raise StorageError(
            f"upload-blob: bridge returned non-hex {field} {raw!r}"
        ) from exc
    if not is_bytes32_hex(canonical):
        raise StorageError(
            f"upload-blob: bridge returned non-bytes32 {field} {canonical!r}"
        )
    return canonical


class MockStorageClient:
    """In-process content-addressed storage. For tests and offline demos.

    Stores bytes keyed by their sha256. The mock IS its own backend, so
    `storage_root` equals `content_hash` — distinct values are not
    needed for the cross-instance property test (the test's whole
    point is that the verifier doesn't depend on producer-side state,
    not that the keys differ).

    `upload_blob` is idempotent; `download_blob` raises `StorageError`
    on miss and `TrustViolation` if the in-process map has been tampered
    with so a key no longer matches its bytes (catches test bugs that
    mutate the dict).
    """

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def upload_blob(self, data: bytes) -> UploadResult:
        if not data:
            raise ValueError("upload_blob: data must be non-empty")
        h = _sha256_hex(data)
        self._store[h] = data
        return UploadResult(content_hash=h, storage_root=h)

    def download_blob(
        self,
        content_hash: Bytes32Hex,
        storage_root: Bytes32Hex,
    ) -> bytes:
        canonical_hash = to_lower_hex(content_hash)
        canonical_root = to_lower_hex(storage_root)
        # Mock invariant: storage_root and content_hash must agree.
        # A test that passes mismatched values is asserting something
        # the mock can't model; surface it loudly rather than silently
        # ignoring the root.
        if canonical_root != canonical_hash:
            raise StorageError(
                f"MockStorageClient requires storage_root == content_hash; "
                f"got root={canonical_root}, hash={canonical_hash}"
            )
        data = self._store.get(canonical_hash)
        if data is None:
            raise StorageError(f"unknown content_hash: {canonical_hash}")
        actual = _sha256_hex(data)
        if actual != canonical_hash:
            raise TrustViolation(
                f"mock storage tampered: key {canonical_hash} maps to bytes hashing to {actual}"
            )
        return data
