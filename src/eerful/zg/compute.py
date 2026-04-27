"""Python adapter for the local zg-bridge (services/zg-bridge/).

The bridge wraps @0glabs/0g-serving-broker over HTTP so eerful (Python) can
drive TeeML inference without porting the SDK. This module is a thin httpx
client; per-call orchestration (`infer_full`) collects everything an
`EnhancedReceipt` needs:

- `chat_id` — the provider-side identifier (ZG-Res-Key header)
- `response_content` — the canonical signed text returned by the
  provider's signature endpoint (NOT the chat completion's assistant
  message; spec §6.2 requires `response_content` to be the body the
  signature is over)
- `enclave_pubkey` — recovered from the signature via EIP-191 personal_sign
  (uncompressed 64-byte X||Y, hex-encoded)
- `signing_address` — derived from the recovered pubkey; used at Step 5 to
  confirm against the address bound by the attestation report
- `enclave_signature` — 65-byte ECDSA signature, hex-encoded
- `attestation_report_bytes` + `attestation_report_hash` — for §6.1
  receipt construction; bridge precomputes the hash in `X-Report-Hash`
"""

from __future__ import annotations

import hashlib
from typing import Any

import httpx
from eth_keys import keys
from eth_utils import keccak
from pydantic import BaseModel, ConfigDict

from eerful.canonical import Address, Bytes32Hex, BytesHex, to_lower_hex
from eerful.errors import ComputeError

__all__ = [
    "ComputeClient",
    "ComputeError",
    "ComputeResult",
    "recover_pubkey_from_personal_sign",
]


def recover_pubkey_from_personal_sign(
    message_text: str,
    signature_hex: str,
) -> tuple[BytesHex, Address]:
    """Recover the secp256k1 pubkey + address that produced an EIP-191
    personal_sign signature over `message_text`.

    Mirrors `ethers.recoverAddress(ethers.hashMessage(text), signature)` —
    the SDK's verifier (verifier.js:616) hashes via `ethers.hashMessage`
    which is EIP-191: `keccak256("\\x19Ethereum Signed Message:\\n" + len + msg)`.

    Returns (pubkey_hex_64_bytes, address_hex). Pubkey is X||Y without the
    SEC1 0x04 prefix, matching ethers' `SigningKey.computePublicKey(..., false)`
    behavior minus the leading byte.
    """
    text_bytes = message_text.encode("utf-8")
    prefix = b"\x19Ethereum Signed Message:\n" + str(len(text_bytes)).encode() + text_bytes
    msg_hash = keccak(prefix)
    # Normalize through to_lower_hex so uppercase / mixed-case / 0X-prefixed
    # input from the bridge or provider doesn't trip bytes.fromhex (spec §6.4).
    canonical = to_lower_hex(signature_hex)
    sig_bytes = bytes.fromhex(canonical.removeprefix("0x"))
    if len(sig_bytes) != 65:
        raise ValueError(f"signature must be 65 bytes, got {len(sig_bytes)}")
    # EVM convention: v is 27 or 28; eth-keys expects 0 or 1 as the last byte.
    if sig_bytes[64] in (27, 28):
        sig_bytes = sig_bytes[:64] + bytes([sig_bytes[64] - 27])
    sig = keys.Signature(sig_bytes)
    pub = sig.recover_public_key_from_msg_hash(msg_hash)
    return to_lower_hex(pub.to_bytes()), to_lower_hex(pub.to_canonical_address())


class ComputeResult(BaseModel):
    """Bundle of artifacts a single TeeML call yields.

    All fields together are sufficient to construct an `EnhancedReceipt`:
    `enclave_pubkey` and `enclave_signature` go straight into the
    attestation block; `response_content` is what the signature is over;
    `attestation_report_hash` is the §6.1 receipt field; the raw
    `attestation_report_bytes` are uploaded to Storage on Day 3.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chat_id: str
    response_content: str
    model_served: str
    provider_endpoint: str
    enclave_pubkey: BytesHex
    enclave_signature: BytesHex
    signing_address: Address
    attestation_report_bytes: bytes
    attestation_report_hash: Bytes32Hex


class ComputeClient:
    """Synchronous httpx client for the local zg-bridge.

    The bridge holds the wallet and runs the broker SDK; Python stays thin.
    Async support is deferred — when jig integration (Track C) lands and
    needs concurrency, `ComputeClient` will gain an `AsyncComputeClient`
    variant or refactor to async-first.
    """

    def __init__(
        self,
        *,
        bridge_url: str,
        http: httpx.Client | None = None,
        timeout: float = 90.0,
    ) -> None:
        self._bridge_url = bridge_url.rstrip("/")
        self._owns_http = http is None
        self._http = http or httpx.Client(timeout=timeout)

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> ComputeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------- bridge endpoints ----------------

    def healthz(self) -> dict[str, Any]:
        r = self._http.get(f"{self._bridge_url}/healthz")
        self._raise_for_status(r, "GET /healthz")
        return r.json()  # type: ignore[no-any-return]

    def add_ledger(self, amount_0g: float) -> dict[str, Any]:
        """Create the ledger sub-account if absent (addLedger) or top it up
        (depositFund). Required once before any provider can be acknowledged
        or paid. Broker's MIN_LOCKED_BALANCE = 1 0G."""
        r = self._http.post(
            f"{self._bridge_url}/admin/add-ledger",
            json={"amount_0g": amount_0g},
        )
        self._raise_for_status(r, "POST /admin/add-ledger")
        return r.json()  # type: ignore[no-any-return]

    def acknowledge(self, provider_address: Address) -> dict[str, Any]:
        r = self._http.post(
            f"{self._bridge_url}/admin/acknowledge",
            json={"provider_address": provider_address},
        )
        self._raise_for_status(r, "POST /admin/acknowledge")
        return r.json()  # type: ignore[no-any-return]

    def infer(
        self,
        *,
        provider_address: Address,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"provider_address": provider_address, "messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        r = self._http.post(f"{self._bridge_url}/compute/inference", json=body)
        self._raise_for_status(r, "POST /compute/inference")
        return r.json()  # type: ignore[no-any-return]

    def fetch_signature(self, *, chat_id: str, provider_address: Address) -> dict[str, Any]:
        r = self._http.get(
            f"{self._bridge_url}/compute/signature/{chat_id}",
            params={"provider": provider_address},
        )
        self._raise_for_status(r, "GET /compute/signature")
        return r.json()  # type: ignore[no-any-return]

    def fetch_attestation(self, provider_address: Address) -> tuple[bytes, Bytes32Hex]:
        r = self._http.get(f"{self._bridge_url}/compute/attestation/{provider_address}")
        self._raise_for_status(r, "GET /compute/attestation")
        report_bytes = r.content
        bridge_hash_raw = r.headers.get("X-Report-Hash", "")
        local_hash = "0x" + hashlib.sha256(report_bytes).hexdigest()
        # Normalize the bridge-supplied hash before comparing — spec §6.4
        # mandates lowercase hex but a future bridge change could emit
        # uppercase or 0X-prefixed and false-positive a mismatch here.
        if bridge_hash_raw:
            bridge_hash = to_lower_hex(bridge_hash_raw)
            if bridge_hash != local_hash:
                raise ComputeError(
                    f"attestation hash mismatch: bridge said {bridge_hash}, "
                    f"local computed {local_hash}"
                )
        return report_bytes, local_hash

    # ---------------- orchestration ----------------

    def infer_full(
        self,
        *,
        provider_address: Address,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ComputeResult:
        """One-shot: call the model, fetch the signature + attestation,
        recover the enclave pubkey, return everything needed to build an
        `EnhancedReceipt`. Spec §6.2: `response_content` here is the
        canonical signed text from the signature endpoint, NOT the chat
        completion's assistant message — those may differ (the provider
        signs a normalized form), and the receipt's Step 6 verification
        replays the signature against this exact text."""
        inference = self.infer(
            provider_address=provider_address,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        signature = self.fetch_signature(
            chat_id=inference["chat_id"],
            provider_address=provider_address,
        )
        report_bytes, report_hash = self.fetch_attestation(provider_address)

        signed_text = signature["message_text"]
        signature_hex = signature["signature_hex"]
        pubkey_hex, signing_address = recover_pubkey_from_personal_sign(signed_text, signature_hex)

        return ComputeResult(
            chat_id=inference["chat_id"],
            response_content=signed_text,
            model_served=inference["model_served"],
            provider_endpoint=inference["provider_endpoint"],
            enclave_pubkey=pubkey_hex,
            enclave_signature=signature_hex,
            signing_address=signing_address,
            attestation_report_bytes=report_bytes,
            attestation_report_hash=report_hash,
        )

    # ---------------- internals ----------------

    @staticmethod
    def _raise_for_status(r: httpx.Response, op: str) -> None:
        if r.is_success:
            return
        try:
            payload = r.json()
            detail = payload.get("error") or payload
        except ValueError:  # JSONDecodeError subclasses ValueError
            detail = r.text[:500]
        raise ComputeError(f"{op} failed ({r.status_code}): {detail}")
