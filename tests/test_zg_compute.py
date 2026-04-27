"""ComputeClient (eerful.zg.compute) — bridge contract + pubkey recovery.

Live testnet smoke lives in `examples/smoke_testnet.py`; these tests use
`httpx.MockTransport` so they're hermetic.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx
import pytest
from eth_keys import keys
from eth_utils import keccak
from pydantic import ValidationError

from eerful.zg.compute import (
    ComputeClient,
    ComputeError,
    ComputeResult,
    recover_pubkey_from_personal_sign,
)


# ---------------- recover_pubkey_from_personal_sign ----------------


def _sign_personal(text: str, privkey_bytes: bytes) -> tuple[str, bytes, str]:
    """EIP-191 personal_sign with v=27/28 (EVM convention).

    Returns (signature_hex, expected_pubkey_64_bytes, expected_address_hex).
    """
    text_bytes = text.encode("utf-8")
    prefix = b"\x19Ethereum Signed Message:\n" + str(len(text_bytes)).encode() + text_bytes
    msg_hash = keccak(prefix)
    pk = keys.PrivateKey(privkey_bytes)
    sig = pk.sign_msg_hash(msg_hash)
    raw = sig.to_bytes()  # 65 bytes, v in {0, 1}
    # EVM convention shifts v by 27.
    evm_v = raw[64] + 27
    sig_evm = raw[:64] + bytes([evm_v])
    return "0x" + sig_evm.hex(), pk.public_key.to_bytes(), pk.public_key.to_checksum_address()


def test_recover_pubkey_round_trips():
    text = "hello eerful"
    privkey = b"\x42" * 32
    sig_hex, expected_pub, expected_addr = _sign_personal(text, privkey)
    pub_hex, addr_hex = recover_pubkey_from_personal_sign(text, sig_hex)
    assert pub_hex == "0x" + expected_pub.hex()
    assert addr_hex.lower() == expected_addr.lower()


def test_recover_pubkey_handles_unicode_message():
    text = "ünïcödé — receipt #1"
    privkey = b"\x07" * 32
    sig_hex, _, expected_addr = _sign_personal(text, privkey)
    _, addr_hex = recover_pubkey_from_personal_sign(text, sig_hex)
    assert addr_hex.lower() == expected_addr.lower()


def test_recover_pubkey_handles_v0_signature():
    """eth-keys produces v in {0,1} natively. We accept that form too — only
    the EVM v=27/28 form needs normalization."""
    text = "raw v0/v1 form"
    text_bytes = text.encode("utf-8")
    msg_hash = keccak(b"\x19Ethereum Signed Message:\n" + str(len(text_bytes)).encode() + text_bytes)
    pk = keys.PrivateKey(b"\x09" * 32)
    sig = pk.sign_msg_hash(msg_hash)
    raw_hex = "0x" + sig.to_bytes().hex()  # v in {0, 1}, no shift
    _, addr_hex = recover_pubkey_from_personal_sign(text, raw_hex)
    assert addr_hex.lower() == pk.public_key.to_checksum_address().lower()


def test_recover_pubkey_rejects_wrong_length():
    with pytest.raises(ValueError, match="65 bytes"):
        recover_pubkey_from_personal_sign("hi", "0x" + "ab" * 64)


# ---------------- ComputeClient with MockTransport ----------------


@contextmanager
def _make_client(handler: httpx.MockTransport) -> Iterator[ComputeClient]:
    """Hand the test a ComputeClient backed by a MockTransport, then close
    the underlying httpx.Client on exit. ComputeClient.close() no-ops here
    because we're explicitly handing it a borrowed http (`_owns_http=False`)."""
    http = httpx.Client(transport=handler)
    try:
        yield ComputeClient(bridge_url="http://bridge.test", http=http)
    finally:
        http.close()


def test_healthz_ok():
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/healthz"
        return httpx.Response(200, json={"status": "ok", "wallet": "0xabc", "chain_id": 16602})

    with _make_client(httpx.MockTransport(handle)) as c:
        out = c.healthz()
    assert out["status"] == "ok"
    assert out["chain_id"] == 16602


def test_acknowledge_posts_provider_address():
    captured: dict[str, Any] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/admin/acknowledge"
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "provider_address": captured["body"]["provider_address"],
                "tee_signer_address": "0xdef",
                "already_acknowledged": False,
            },
        )

    with _make_client(httpx.MockTransport(handle)) as c:
        out = c.acknowledge("0x" + "a" * 40)
    assert captured["body"] == {"provider_address": "0x" + "a" * 40}
    assert out["tee_signer_address"] == "0xdef"


def test_infer_passes_optional_params_only_when_set():
    seen: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "chat_id": "chat-1",
                "response_content": "hi",
                "model_served": "qwen/qwen-2.5-7b-instruct",
                "provider_endpoint": "https://provider/v1",
            },
        )

    with _make_client(httpx.MockTransport(handle)) as c:
        c.infer(provider_address="0xabc", messages=[{"role": "user", "content": "hello"}])
        c.infer(
            provider_address="0xabc",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.2,
            max_tokens=64,
        )

    assert seen[0] == {"provider_address": "0xabc", "messages": [{"role": "user", "content": "hello"}]}
    assert seen[1]["temperature"] == 0.2
    assert seen[1]["max_tokens"] == 64


def test_fetch_signature_uses_query_param_and_parses():
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/compute/signature/chat-7"
        assert request.url.params["provider"] == "0x" + "b" * 40
        return httpx.Response(200, json={"signature_hex": "0xdead", "message_text": "signed body"})

    with _make_client(httpx.MockTransport(handle)) as c:
        out = c.fetch_signature(chat_id="chat-7", provider_address="0x" + "b" * 40)
    assert out == {"signature_hex": "0xdead", "message_text": "signed body"}


def test_fetch_attestation_returns_bytes_and_hash():
    raw = b'{"quote":"...","tcb_info":"..."}'
    expected_hash = "0x" + hashlib.sha256(raw).hexdigest()

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=raw,
            headers={"X-Report-Hash": expected_hash, "Content-Type": "application/octet-stream"},
        )

    with _make_client(httpx.MockTransport(handle)) as c:
        body, h = c.fetch_attestation("0x" + "c" * 40)
    assert body == raw
    assert h == expected_hash


def test_fetch_attestation_rejects_hash_mismatch():
    raw = b"actual report"

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=raw,
            headers={"X-Report-Hash": "0x" + "0" * 64},  # bogus
        )

    with _make_client(httpx.MockTransport(handle)) as c:
        with pytest.raises(ComputeError, match="hash mismatch"):
            c.fetch_attestation("0x" + "c" * 40)


def test_raise_for_status_wraps_bridge_error():
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "Account does not exist"})

    with _make_client(httpx.MockTransport(handle)) as c:
        with pytest.raises(ComputeError, match="Account does not exist"):
            c.acknowledge("0x" + "a" * 40)


def test_infer_full_end_to_end():
    """Stand up a fake bridge that handles all three endpoints, then check
    that infer_full assembles a ComputeResult with the recovered pubkey
    and a locally-computed attestation hash."""
    text = "canonical-signed-body"
    privkey = b"\x55" * 32
    sig_hex, expected_pub, expected_addr = _sign_personal(text, privkey)
    raw_report = b'{"quote":"abc","tcb_info":"xyz"}'
    expected_hash = "0x" + hashlib.sha256(raw_report).hexdigest()

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/compute/inference":
            return httpx.Response(
                200,
                json={
                    "chat_id": "chat-x",
                    "response_content": "Hi there!",  # display text — discarded for receipt
                    "model_served": "qwen/qwen-2.5-7b-instruct",
                    "provider_endpoint": "https://example/v1",
                },
            )
        if path == "/compute/signature/chat-x":
            return httpx.Response(200, json={"signature_hex": sig_hex, "message_text": text})
        if path.startswith("/compute/attestation/"):
            return httpx.Response(
                200,
                content=raw_report,
                headers={"X-Report-Hash": expected_hash},
            )
        return httpx.Response(404)

    with _make_client(httpx.MockTransport(handle)) as c:
        result = c.infer_full(
            provider_address="0x" + "a" * 40,
            messages=[{"role": "user", "content": "hi"}],
        )

    assert isinstance(result, ComputeResult)
    assert result.chat_id == "chat-x"
    assert result.response_content == text  # the SIGNED text, not "Hi there!"
    assert result.enclave_pubkey == "0x" + expected_pub.hex()
    assert result.enclave_signature == sig_hex
    assert result.signing_address.lower() == expected_addr.lower()
    assert result.attestation_report_bytes == raw_report
    assert result.attestation_report_hash == expected_hash
    assert result.model_served == "qwen/qwen-2.5-7b-instruct"


def test_compute_result_is_immutable():
    result = ComputeResult(
        chat_id="x",
        response_content="hi",
        model_served="m",
        provider_endpoint="https://p",
        enclave_pubkey="0x" + "1" * 128,
        enclave_signature="0x" + "2" * 130,
        signing_address="0x" + "3" * 40,
        attestation_report_bytes=b"r",
        attestation_report_hash="0x" + "4" * 64,
    )
    with pytest.raises(ValidationError, match="frozen"):
        result.chat_id = "y"
