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
    root = "0x" + "ab" * 32

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/storage/upload-blob"
        assert request.headers["content-type"] == "application/octet-stream"
        assert request.content == payload
        return httpx.Response(
            200,
            json={
                "content_hash": expected,
                "storage_uri": f"zg://{root}",
                "root_hash": root,
                "tx_hash": "0xtx",
                "tx_seq": 7,
                "size_bytes": len(payload),
            },
        )

    with _make_bridge(httpx.MockTransport(handle)) as c:
        out = c.upload_blob(payload)
    assert out.content_hash == expected
    assert out.storage_root == root


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
    root = "0x" + "de" * 32

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content_hash": bogus,
                "storage_uri": f"zg://{root}",
                "root_hash": root,
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
    root = "0x" + "cd" * 32

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content_hash": upper,
                "storage_uri": f"zg://{root}",
                "root_hash": root.upper(),
                "tx_hash": "0xtx",
                "tx_seq": 3,
                "size_bytes": len(payload),
            },
        )

    with _make_bridge(httpx.MockTransport(handle)) as c:
        out = c.upload_blob(payload)
    assert out.content_hash == expected
    # storage_root canonicalized to lowercase even if bridge regresses.
    assert out.storage_root == root


def test_upload_blob_wraps_non_hex_bridge_response_as_storage_error():
    """to_lower_hex raises ValueError on garbage input; the adapter
    must wrap it so the documented error contract holds (only
    StorageError / TrustViolation / RuntimeError leave the method)."""
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content_hash": "not-a-hex-string",
                "storage_uri": "zg://0xabc",
                "root_hash": "0x" + "ab" * 32,
                "tx_hash": "0xtx",
                "tx_seq": 1,
                "size_bytes": 4,
            },
        )

    with _make_bridge(httpx.MockTransport(handle)) as c:
        with pytest.raises(StorageError, match="non-hex content_hash"):
            c.upload_blob(b"abcd")


def test_upload_blob_rejects_short_bridge_hash_as_storage_error():
    """A bridge that returns hex of the wrong length is buggy, not
    byzantine — surface as StorageError, not TrustViolation."""
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content_hash": "0xdead",  # well-formed hex but too short
                "storage_uri": "zg://0xabc",
                "root_hash": "0x" + "ab" * 32,
                "tx_hash": "0xtx",
                "tx_seq": 1,
                "size_bytes": 4,
            },
        )

    with _make_bridge(httpx.MockTransport(handle)) as c:
        with pytest.raises(StorageError, match="non-bytes32 content_hash"):
            c.upload_blob(b"abcd")


def test_upload_blob_rejects_short_bridge_root_as_storage_error():
    """Symmetric to content_hash: a wrong-length root_hash is a bridge
    bug. Receipts carry storage_root in their canonical signing payload,
    so a malformed value would silently break receipt_id derivation."""
    payload = b"short root"
    expected = _sha256_hex(payload)

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content_hash": expected,
                "storage_uri": "zg://0xdead",
                "root_hash": "0xdead",  # well-formed but too short
                "tx_hash": "0xtx",
                "tx_seq": 1,
                "size_bytes": len(payload),
            },
        )

    with _make_bridge(httpx.MockTransport(handle)) as c:
        with pytest.raises(StorageError, match="non-bytes32 root_hash"):
            c.upload_blob(payload)


def test_upload_blob_missing_root_hash_field_raises_storage_error():
    """Bridge regression that drops `root_hash` from the response: the
    adapter has nothing to put in `evaluator_storage_root` /
    `attestation_storage_root`, and silently returning a bogus value
    would corrupt downstream receipts. Loud StorageError instead."""
    payload = b"no root"
    expected = _sha256_hex(payload)

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content_hash": expected,
                "storage_uri": "zg://0xdead",
                "tx_hash": "0xtx",
                "tx_seq": 1,
                "size_bytes": len(payload),
            },
        )

    with _make_bridge(httpx.MockTransport(handle)) as c:
        with pytest.raises(StorageError, match="root_hash"):
            c.upload_blob(payload)


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
    root = "0x" + "11" * 32

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/storage/download-blob"
        assert request.url.params["content_hash"] == h
        assert request.url.params["root_hash"] == root
        return httpx.Response(
            200,
            content=payload,
            headers={
                "X-Content-Hash": h,
                "X-Root-Hash": root,
                "Content-Type": "application/octet-stream",
            },
        )

    with _make_bridge(httpx.MockTransport(handle)) as c:
        out = c.download_blob(h, root)
    assert out == payload


def test_download_blob_rejects_malformed_content_hash():
    """Malformed input is caller's bug; raise ValueError before hitting
    the wire (otherwise the bridge 400s and surfaces it as RuntimeError,
    confusing the boundary)."""
    sent: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(str(request.url))
        return httpx.Response(200, content=b"x")

    with _make_bridge(httpx.MockTransport(handle)) as c:
        with pytest.raises(ValueError, match="content_hash.*64-char lowercase hex"):
            c.download_blob("0xdead", "0x" + "0" * 64)  # content_hash too short
    assert sent == []  # request never went out


def test_download_blob_rejects_malformed_storage_root():
    """Symmetric to content_hash: a malformed storage_root is caller-
    side, raise ValueError before hitting the wire."""
    sent: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(str(request.url))
        return httpx.Response(200, content=b"x")

    with _make_bridge(httpx.MockTransport(handle)) as c:
        with pytest.raises(ValueError, match="storage_root.*64-char lowercase hex"):
            c.download_blob("0x" + "0" * 64, "0xdead")
    assert sent == []


def test_download_blob_5xx_raises_storage_error_when_body_is_list():
    """Bridge regression returning a non-dict JSON envelope on error
    must not AttributeError out of _raise_for_status."""
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json=["oops"])

    with _make_bridge(httpx.MockTransport(handle)) as c:
        with pytest.raises(StorageError):
            c.download_blob("0x" + "0" * 64, "0x" + "1" * 64)


def test_download_blob_502_root_unresolved_raises_storage_error():
    """Bridge can't resolve the rootHash through the indexer (e.g., the
    bytes were never persisted, or the indexer is wedged). Adapter
    surfaces as StorageError — transient, retry-eligible."""
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            502,
            json={"error": "download_failed", "detail": "indexer cannot resolve root"},
        )

    with _make_bridge(httpx.MockTransport(handle)) as c:
        with pytest.raises(StorageError, match="indexer cannot resolve"):
            c.download_blob("0x" + "0" * 64, "0x" + "1" * 64)


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
            c.download_blob("0x" + "1" * 64, "0x" + "2" * 64)


def test_download_blob_local_hash_check_catches_lying_bridge():
    """Defense in depth: even if the bridge claims success, the adapter
    re-hashes the body and rejects mismatches as TrustViolation."""
    requested = "0x" + "ab" * 32
    root = "0x" + "cd" * 32
    actual_payload = b"different bytes"

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=actual_payload,
            headers={"X-Content-Hash": requested, "X-Root-Hash": root},
        )

    with _make_bridge(httpx.MockTransport(handle)) as c:
        with pytest.raises(TrustViolation, match="hash mismatch"):
            c.download_blob(requested, root)


def test_download_blob_normalizes_input_hash():
    """Caller-supplied content_hash is canonicalized before being sent
    on the wire — uppercase or 0X-prefixed input must not break the
    request."""
    payload = b"normalize me"
    canonical = _sha256_hex(payload)
    upper = "0X" + canonical[2:].upper()
    root = "0x" + "11" * 32
    upper_root = "0X" + root[2:].upper()
    seen: list[tuple[str, str]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(
            (
                request.url.params["content_hash"],
                request.url.params["root_hash"],
            )
        )
        return httpx.Response(200, content=payload)

    with _make_bridge(httpx.MockTransport(handle)) as c:
        out = c.download_blob(upper, upper_root)
    assert out == payload
    assert seen == [(canonical, root)]


def test_bridge_client_does_not_close_borrowed_http():
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


def test_bridge_client_closes_owned_http():
    """When the adapter constructs its own httpx.Client, it owns the
    lifecycle and must close it on .close()."""
    c = BridgeStorageClient(bridge_url="http://bridge.test")
    assert c._http.is_closed is False
    c.close()
    assert c._http.is_closed is True


# ---------------- MockStorageClient ----------------


def test_mock_round_trip():
    m = MockStorageClient()
    payload = b"mock payload"
    upload = m.upload_blob(payload)
    # Mock invariant: storage_root == content_hash.
    assert upload.storage_root == upload.content_hash
    assert m.download_blob(upload.content_hash, upload.storage_root) == payload


def test_mock_upload_is_idempotent():
    m = MockStorageClient()
    payload = b"upload twice"
    u1 = m.upload_blob(payload)
    u2 = m.upload_blob(payload)
    assert u1 == u2
    assert m.download_blob(u1.content_hash, u1.storage_root) == payload


def test_mock_download_miss_raises_storage_error():
    m = MockStorageClient()
    h = "0x" + "0" * 64
    with pytest.raises(StorageError, match="unknown content_hash"):
        m.download_blob(h, h)


def test_mock_download_normalizes_input():
    m = MockStorageClient()
    payload = b"case mock"
    upload = m.upload_blob(payload)
    upper_hash = "0X" + upload.content_hash[2:].upper()
    upper_root = "0X" + upload.storage_root[2:].upper()
    assert m.download_blob(upper_hash, upper_root) == payload


def test_mock_detects_in_process_tampering():
    """If a test mutates the underlying dict so a key no longer matches
    its bytes, downloads must raise TrustViolation. Catches mock-misuse
    bugs that would otherwise let bad bytes propagate to verify()."""
    m = MockStorageClient()
    payload = b"original"
    upload = m.upload_blob(payload)
    m._store[upload.content_hash] = b"tampered"
    with pytest.raises(TrustViolation, match="tampered"):
        m.download_blob(upload.content_hash, upload.storage_root)


def test_mock_rejects_empty_upload():
    m = MockStorageClient()
    with pytest.raises(ValueError, match="non-empty"):
        m.upload_blob(b"")


def test_mock_rejects_mismatched_root_and_hash():
    """The mock IS its own backend (content_hash == storage_root). A
    test passing distinct values is asserting something the mock can't
    model — surface as StorageError so the test fails loudly instead
    of silently passing the integrity check."""
    m = MockStorageClient()
    payload = b"distinct values"
    upload = m.upload_blob(payload)
    other_root = "0x" + "f" * 64
    with pytest.raises(StorageError, match="storage_root == content_hash"):
        m.download_blob(upload.content_hash, other_root)
