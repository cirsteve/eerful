"""StorageClient tests — bridge contract + Mock parity.

Bridge tests use httpx.MockTransport so they're hermetic. Live testnet
exercise lives in `examples/smoke_testnet.py` (Day 3 add).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import pytest

from eerful.errors import StorageError, TrustViolation
from eerful.zg.storage import (
    BridgeStorageClient,
    MockStorageClient,
    StorageClient,
)


def _sha256_hex(data: bytes) -> str:
    return "0x" + hashlib.sha256(data).hexdigest()


@contextmanager
def _make_bridge(handler: httpx.MockTransport) -> Iterator[BridgeStorageClient]:
    http = httpx.Client(transport=handler)
    try:
        yield BridgeStorageClient(bridge_url="http://bridge.test", http=http)
    finally:
        http.close()


# ---------------- Protocol satisfaction ----------------


def test_both_clients_satisfy_protocol():
    assert isinstance(MockStorageClient(), StorageClient)
    with _make_bridge(httpx.MockTransport(lambda r: httpx.Response(200))) as c:
        assert isinstance(c, StorageClient)


# ---------------- BridgeStorageClient ----------------


def test_upload_blob_returns_local_hash():
    payload = b"hello eerful"
    expected = _sha256_hex(payload)

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/storage/upload-blob"
        assert request.headers["content-type"] == "application/octet-stream"
        assert request.content == payload
        return httpx.Response(
            200,
            json={
                "content_hash": expected,
                "storage_uri": "zg://0xabc",
                "root_hash": "0xabc",
                "tx_hash": "0xtx",
                "tx_seq": 7,
                "size_bytes": len(payload),
            },
        )

    with _make_bridge(httpx.MockTransport(handle)) as c:
        out = c.upload_blob(payload)
    assert out == expected


def test_upload_blob_rejects_empty():
    with _make_bridge(httpx.MockTransport(lambda r: httpx.Response(200))) as c:
        with pytest.raises(ValueError, match="non-empty"):
            c.upload_blob(b"")


def test_upload_blob_detects_bridge_hash_lie():
    """Bridge that returns a content_hash inconsistent with the bytes we
    sent is a TrustViolation — the storage URI it returns can't be
    trusted to map back to our bytes."""
    payload = b"truthy bytes"
    bogus = "0x" + "0" * 64

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content_hash": bogus,
                "storage_uri": "zg://0xdef",
                "root_hash": "0xdef",
                "tx_hash": "0xtx",
                "tx_seq": 1,
                "size_bytes": len(payload),
            },
        )

    with _make_bridge(httpx.MockTransport(handle)) as c:
        with pytest.raises(TrustViolation, match="hash mismatch"):
            c.upload_blob(payload)


def test_upload_blob_normalizes_uppercase_bridge_hash():
    """Bridge could regress to uppercase hex; adapter must canonicalize
    before comparing or it false-positives a mismatch."""
    payload = b"case test"
    expected = _sha256_hex(payload)
    upper = "0x" + expected[2:].upper()

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content_hash": upper,
                "storage_uri": "zg://0xabc",
                "root_hash": "0xabc",
                "tx_hash": "0xtx",
                "tx_seq": 3,
                "size_bytes": len(payload),
            },
        )

    with _make_bridge(httpx.MockTransport(handle)) as c:
        out = c.upload_blob(payload)
    assert out == expected


def test_upload_blob_5xx_raises_storage_error():
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, json={"error": "upload_failed", "detail": "indexer down"})

    with _make_bridge(httpx.MockTransport(handle)) as c:
        with pytest.raises(StorageError, match="indexer down"):
            c.upload_blob(b"anything")


def test_upload_blob_400_raises_runtime_error():
    """400 means the adapter sent a malformed request — programming bug,
    not a transient failure. RuntimeError is what propagates past
    SubstrateError-only retry loops."""
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": "empty_body", "detail": "POST body required"},
        )

    with _make_bridge(httpx.MockTransport(handle)) as c:
        with pytest.raises(RuntimeError, match="empty_body"):
            c.upload_blob(b"x")


def test_upload_blob_transport_error_wrapped():
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with _make_bridge(httpx.MockTransport(handle)) as c:
        with pytest.raises(StorageError, match="transport error"):
            c.upload_blob(b"anything")


def test_download_blob_round_trip():
    payload = b"download payload"
    h = _sha256_hex(payload)

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/storage/download-blob"
        assert request.url.params["content_hash"] == h
        return httpx.Response(
            200,
            content=payload,
            headers={"X-Content-Hash": h, "Content-Type": "application/octet-stream"},
        )

    with _make_bridge(httpx.MockTransport(handle)) as c:
        out = c.download_blob(h)
    assert out == payload


def test_download_blob_404_raises_storage_error():
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"error": "not_in_index", "detail": "different uploader"},
        )

    with _make_bridge(httpx.MockTransport(handle)) as c:
        with pytest.raises(StorageError, match="not_in_index"):
            c.download_blob("0x" + "0" * 64)


def test_download_blob_422_from_bridge_raises_trust_violation():
    """Bridge already does its own re-hash; if it returns 422 the
    adapter must surface that as TrustViolation rather than retry."""
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={"error": "content_hash_mismatch", "detail": "indexer swapped bytes"},
        )

    with _make_bridge(httpx.MockTransport(handle)) as c:
        with pytest.raises(TrustViolation, match="content_hash_mismatch"):
            c.download_blob("0x" + "1" * 64)


def test_download_blob_local_hash_check_catches_lying_bridge():
    """Defense in depth: even if the bridge claims success, the adapter
    re-hashes the body and rejects mismatches as TrustViolation."""
    requested = "0x" + "ab" * 32
    actual_payload = b"different bytes"

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=actual_payload,
            headers={"X-Content-Hash": requested},
        )

    with _make_bridge(httpx.MockTransport(handle)) as c:
        with pytest.raises(TrustViolation, match="hash mismatch"):
            c.download_blob(requested)


def test_download_blob_normalizes_input_hash():
    """Caller-supplied content_hash is canonicalized before being sent
    on the wire — uppercase or 0X-prefixed input must not break the
    request."""
    payload = b"normalize me"
    canonical = _sha256_hex(payload)
    upper = "0X" + canonical[2:].upper()
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params["content_hash"])
        return httpx.Response(200, content=payload)

    with _make_bridge(httpx.MockTransport(handle)) as c:
        out = c.download_blob(upper)
    assert out == payload
    assert seen == [canonical]


def test_bridge_client_context_manager_closes_owned_http():
    closed = {"flag": False}

    class _TrackedClient(httpx.Client):
        def close(self) -> None:
            closed["flag"] = True
            super().close()

    http = _TrackedClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    c = BridgeStorageClient(bridge_url="http://bridge.test", http=http)
    # Borrowed http (we passed it in) MUST NOT be closed by the adapter.
    c.close()
    assert closed["flag"] is False
    http.close()
    assert closed["flag"] is True


# ---------------- MockStorageClient ----------------


def test_mock_round_trip():
    m = MockStorageClient()
    payload = b"mock payload"
    h = m.upload_blob(payload)
    assert m.download_blob(h) == payload


def test_mock_upload_is_idempotent():
    m = MockStorageClient()
    payload = b"upload twice"
    h1 = m.upload_blob(payload)
    h2 = m.upload_blob(payload)
    assert h1 == h2
    assert m.download_blob(h1) == payload


def test_mock_download_miss_raises_storage_error():
    m = MockStorageClient()
    with pytest.raises(StorageError, match="unknown content_hash"):
        m.download_blob("0x" + "0" * 64)


def test_mock_download_normalizes_input():
    m = MockStorageClient()
    payload = b"case mock"
    h = m.upload_blob(payload)
    upper = "0X" + h[2:].upper()
    assert m.download_blob(upper) == payload


def test_mock_detects_in_process_tampering():
    """If a test mutates the underlying dict so a key no longer matches
    its bytes, downloads must raise TrustViolation. Catches mock-misuse
    bugs that would otherwise let bad bytes propagate to verify()."""
    m = MockStorageClient()
    payload = b"original"
    h = m.upload_blob(payload)
    m._store[h] = b"tampered"
    with pytest.raises(TrustViolation, match="tampered"):
        m.download_blob(h)


def test_mock_rejects_empty_upload():
    m = MockStorageClient()
    with pytest.raises(ValueError, match="non-empty"):
        m.upload_blob(b"")
